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

async def plan_agent(query: str, containers: dict):
    tools = []
    app = langgraph_agent.buildChatAgent(tools)

    system_prompt=(
        "For the given objective, come up with a simple step by step plan."
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer." 
        "Do not add any superfluous steps."
        "The result of the final step should be the final answer. Make sure that each step has all the information needed."
        "The plan should be returned in <plan> tag."
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

    response = await app.ainvoke(inputs, config)
    logger.info(f"response: {response}")
    result = response['messages'][-1].content
    logger.info(f"result: {result}")
    # result = ""    
    # async for stream in app.astream(inputs, config, stream_mode="messages"):
    #     if isinstance(stream[0], AIMessageChunk):
    #         message = stream[0]    
    #         if isinstance(message.content, list):
    #             for content_item in message.content:
    #                 if isinstance(content_item, dict):
    #                     if content_item.get('type') == 'text':
    #                         text_content = content_item.get('text', '')
    #                         result += text_content                                
    #                         chat.update_streaming_result(containers, result, "info")

    logger.info(f"result: {result}")

    plan = result[result.find('<plan>')+6:result.find('</plan>')]
    logger.info(f"plan: {plan}")

    return plan

async def execute_agent(prompt: str, mcp_servers: list, containers: dict):
    image_url = []
    references = []
    
    mcp_json = mcp_config.load_selected_config(mcp_servers)
    logger.info(f"mcp_json: {mcp_json}")

    server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)
    logger.info(f"server_params: {server_params}")    

    client = MultiServerMCPClient(server_params)
    logger.info(f"MCP client created successfully")
    
    tools = await client.get_tools()
    logger.info(f"get_tools() returned: {tools}")
    
    if tools is None:
        logger.error("tools is None - MCP client failed to get tools")
        tools = []
    
    tool_list = [tool.name for tool in tools] if tools else []
    logger.info(f"tool_list: {tool_list}")
        
    app = langgraph_agent.buildChatAgent(tools)
    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": chat.user_id},
        "tools": tools
    }
    
    inputs = {
        "messages": [HumanMessage(content=prompt)]
    }
            
    result = ""
    tool_used = False  # Track if tool was used
    tool_name = toolUseId = ""
    
    async for stream in app.astream(inputs, config, stream_mode="messages"):
        if isinstance(stream[0], AIMessageChunk):
            message = stream[0]    
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
                            chat.update_streaming_result(containers, result, "markdown")

                        elif content_item.get('type') == 'tool_use':
                            logger.info(f"content_item: {content_item}")      
                            if 'id' in content_item and 'name' in content_item:
                                toolUseId = content_item.get('id', '')
                                tool_name = content_item.get('name', '')
                                logger.info(f"tool_name: {tool_name}, toolUseId: {toolUseId}")
                                chat.streaming_index = chat.index
                                chat.index += 1

                            if 'partial_json' in content_item:
                                partial_json = content_item.get('partial_json', '')
                                logger.info(f"partial_json: {partial_json}")
                                
                                if toolUseId not in chat.tool_input_list:
                                    chat.tool_input_list[toolUseId] = ""                                
                                chat.tool_input_list[toolUseId] += partial_json
                                input = chat.tool_input_list[toolUseId]
                                logger.info(f"input: {input}")

                                logger.info(f"tool_name: {tool_name}, input: {input}, toolUseId: {toolUseId}")
                                chat.update_streaming_result(containers, f"Tool: {tool_name}, Input: {input}", "info")
                        
        elif isinstance(stream[0], ToolMessage):
            message = stream[0]
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

    if containers is not None:
        containers['notification'][chat.index].markdown(result)
    
    return result, image_url

async def planning_agent(query: str, mcp_servers: list, containers: dict):    
    chat.index = 0

    chat.add_notification(containers, f"계획을 생성하는 중입니다...")    
    logger.info(f"=== Use Plan Agent ===")    
    plan = await plan_agent(query, containers)
    
    logger.info(f"plan: {plan}")
    chat.add_notification(containers, f"생성된 계획:\n{plan}")

    prompt = query + "\n 다음의 계획을 참고하여 답변하세요.\n" + plan
    logger.info(f"prompt: {prompt}")

    logger.info(f"=== Use Execute Agent ===")
    result, image_url = await execute_agent(prompt, mcp_servers, containers)

    logger.info(f"result: {result}")
    logger.info(f"image_url: {image_url}")

    return result, image_url
    