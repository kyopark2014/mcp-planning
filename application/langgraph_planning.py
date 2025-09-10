import logging
import sys
import mcp_config
import langgraph_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient
import chat

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger("chat")

async def plan_agent(query: str):
    tools = []
    app = langgraph_agent.buildChatAgentWithHistory(tools)

    system_prompt=(
        "For the given objective, come up with a simple step by step plan."
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps."
        "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."
        "생성된 계획은 <plan> 태그로 감싸서 반환합니다."
    )

    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": chat.user_id},
        "tools": tools,
        "system_prompt": system_prompt
    }
    
    inputs = {
        "messages": [HumanMessage(content=query)]
    }

    result = await app.ainvoke(inputs, config)
    logger.info(f"result: {result}")
    
    return result

async def run_langgraph_planning_agent(query: str, mcp_servers: list, containers: dict):    
    chat.index = 0

    image_url = []
    references = []

    result = await plan_agent(query)

    plan = result['messages'][-1].content

    plan = plan.replace("<plan>", "").replace("</plan>", "")
    logger.info(f"plan: {plan}")
    chat.add_notification(containers, plan)

    logger.info(f"=== Use Plan to Execute ===")

    mcp_json = mcp_config.load_selected_config(mcp_servers)
    logger.info(f"mcp_json: {mcp_json}")

    server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)
    logger.info(f"server_params: {server_params}")    

    try:
        client = MultiServerMCPClient(server_params)
        logger.info(f"MCP client created successfully")
        
        tools = await client.get_tools()
        logger.info(f"get_tools() returned: {tools}")
        
        if tools is None:
            logger.error("tools is None - MCP client failed to get tools")
            tools = []
        
        tool_list = [tool.name for tool in tools] if tools else []
        logger.info(f"tool_list: {tool_list}")
        
    except Exception as e:
        logger.error(f"Error creating MCP client or getting tools: {e}")
        pass
        
    # If no tools available, use general conversation
    if not tools:
        logger.warning("No tools available, using general conversation mode")
        result = "MCP 설정을 확인하세요."
        if containers is not None:
            containers['notification'][0].markdown(result)
        return result, image_url
    
    app = langgraph_agent.buildChatAgentWithHistory(tools)

    system_prompt=(
        "You are an executor who executes the plan."
        "주어진 질문에 답변하기 위하여 다음의 plan을 순차적으로 실행합니다."        
        f"<plan>{plan}</plan>"
        "tavily-search 도구를 사용하여 정보를 수집합니다."
    )

    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": chat.user_id},
        "tools": tools,
        "system_prompt": system_prompt
    }
    
    inputs = {
        "messages": [HumanMessage(content=query)]
    }
            
    result = ""
    tool_used = False  # Track if tool was used
    tool_name = toolUseId = ""
    
    async for output in app.astream(inputs, config, stream_mode="messages"):
        # logger.info(f"output: {output}")

        # Handle tuple output (message, metadata)
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], AIMessageChunk):
            message = output[0]    
            input = {}        
            if isinstance(message.content, list):
                for content_item in message.content:
                    if isinstance(content_item, dict):
                        if content_item.get('type') == 'text':
                            text_content = content_item.get('text', '')
                            # logger.info(f"text_content: {text_content}")
                            
                            # If tool was used, start fresh result
                            if tool_used:
                                result = text_content
                                tool_used = False
                            else:
                                result += text_content
                                
                            # logger.info(f"result: {result}")                
                            chat.update_streaming_result(containers, result)

                        elif content_item.get('type') == 'tool_use':
                            logger.info(f"content_item: {content_item}")      
                            if 'id' in content_item and 'name' in content_item:
                                toolUseId = content_item.get('id', '')
                                tool_name = content_item.get('name', '')
                                logger.info(f"tool_name: {tool_name}, toolUseId: {toolUseId}")
                                # chat.add_notification(containers, f"Tool: {tool_name}, Input: {input}")

                                chat.tool_info_list[toolUseId] = chat.index                     
                                chat.tool_name_list[toolUseId] = tool_name     
                                                                    
                            if 'partial_json' in content_item:
                                partial_json = content_item.get('partial_json', '')
                                logger.info(f"partial_json: {partial_json}")
                                
                                if toolUseId not in chat.tool_input_list:
                                    chat.tool_input_list[toolUseId] = ""                                
                                chat.tool_input_list[toolUseId] += partial_json
                                input = chat.tool_input_list[toolUseId]
                                logger.info(f"input: {input}")

                                logger.info(f"tool_name: {tool_name}, input: {input}, toolUseId: {toolUseId}")
                                # add_notification(containers, f"Tool: {tool_name}, Input: {input}")
                                index = chat.tool_info_list[toolUseId]
                                containers['notification'][index-1].info(f"Tool: {tool_name}, Input: {input}")
                        
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], ToolMessage):
            message = output[0]
            logger.info(f"ToolMessage: {message.name}, {message.content}")
            tool_name = message.name
            toolResult = message.content
            toolUseId = message.tool_call_id
            logger.info(f"toolResult: {toolResult}, toolUseId: {toolUseId}")
            chat.add_notification(containers, f"Tool Result: {toolResult}")
            tool_used = True
            
            content, urls, refs = chat.get_tool_info(tool_name, toolResult)
            if refs:
                for r in refs:
                    references.append(r)
                logger.info(f"refs: {refs}")
            if urls:
                for url in urls:
                    image_url.append(url)
                logger.info(f"urls: {urls}")

            if content:
                logger.info(f"content: {content}")        
    
    if not result:
        result = "답변을 찾지 못하였습니다."        
    logger.info(f"result: {result}")

    # if references:
    #     ref = "\n\n### Reference\n"
    #     for i, reference in enumerate(references):
    #         page_content = reference['content'][:100].replace("\n", "")
    #         ref += f"{i+1}. [{reference['title']}]({reference['url']}), {page_content}...\n"    
    #     result += ref
    
    if containers is not None:
        containers['notification'][index].markdown(result)
    
    return result, image_url
