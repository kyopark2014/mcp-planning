
import logging
import sys
import mcp_config
import chat

from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from strands.multiagent import GraphBuilder
from strands import Agent

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("strands-agent")

def get_tool_list(tools):
    tool_list = []
    for tool in tools:
        if hasattr(tool, 'tool_name'):  # MCP tool
            tool_list.append(tool.tool_name)
        elif hasattr(tool, 'name'):  # MCP tool with name attribute
            tool_list.append(tool.name)
        elif hasattr(tool, '__name__'):  # Function or module
            tool_list.append(tool.__name__)
        elif str(tool).startswith("<module 'strands_tools."):   
            module_name = str(tool).split("'")[1].split('.')[-1]
            tool_list.append(module_name)
        else:
            # For MCP tools that might have different structure
            tool_str = str(tool)
            if 'MCPAgentTool' in tool_str:
                # Try to extract tool name from MCP tool
                try:
                    if hasattr(tool, 'tool'):
                        tool_list.append(tool.tool.name)
                    else:
                        tool_list.append(f"MCP_Tool_{len(tool_list)}")
                except:
                    tool_list.append(f"MCP_Tool_{len(tool_list)}")
            else:
                tool_list.append(str(tool))
    return tool_list

debug_mode = 'Enable'

async def planning_agent(query, containers):
    global status_msg
    status_msg = []

    chat.index = 0

    logger.info(f"=== Use Plan Agent ===")
    chat.add_notification(containers, f"계획을 생성하는 중입니다...")

    # Create specialized agents
    system_prompt=(
        "For the given objective, come up with a simple step by step plan."
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer." 
        "Do not add any superfluous steps."
        "The result of the final step should be the final answer. Make sure that each step has all the information needed."
        "The plan should be returned in <plan> tag."
    )

    planner = Agent(
        name="plan", 
        system_prompt=system_prompt
    )

    #agent_stream = await planner.stream_async(query)
    #result = await show_result(agent_stream, containers)
    response = planner(query)
    logger.info(f"planner result: {response}")

    result = str(response)

    plan = result[result.find('<plan>')+6:result.find('</plan>')]
    logger.info(f"plan: {plan}")

    chat.add_notification(containers, f"생성된 계획:\n{plan}")

    logger.info(f"=== Use Execute Agent ===")
    tool = "tavily-search"
    config = mcp_config.load_config(tool)
    mcp_servers = config["mcpServers"]
    logger.info(f"mcp_servers: {mcp_servers}")

    mcp_client = None
    for server_name, server_config in mcp_servers.items():
        logger.info(f"server_name: {server_name}")
        logger.info(f"server_config: {server_config}")
        env = server_config["env"] if "env" in server_config else None

        mcp_client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=server_config["command"], 
                args=server_config["args"], 
                env=env
            )
        ))
        break

    with mcp_client as client:
        mcp_tools = client.list_tools_sync()
        logger.info(f"mcp_tools: {mcp_tools}")
        
        tools = []
        tools.extend(mcp_tools)

        tool_list = get_tool_list(tools)
        logger.info(f"tools loaded: {tool_list}")
        
        executor = Agent(
            name="executor", 
            tools=tools
        )

        prompt = query + "\n 다음의 계획을 참고하여 답변하세요.\n" + plan
        logger.info(f"prompt: {prompt}")

        agent_stream = executor.stream_async(prompt)
        
        current = ""
        async for event in agent_stream:
            text = ""            
            if "data" in event:
                text = event["data"]
                logger.info(f"[data] {text}")
                current += text
                chat.update_streaming_result(containers, current, "markdown")

            elif "result" in event:
                final = event["result"]                
                message = final.message
                if message:
                    content = message.get("content", [])
                    result = content[0].get("text", "")
                    logger.info(f"[result] {result}")
                    final_result = result

            elif "current_tool_use" in event:
                current_tool_use = event["current_tool_use"]
                logger.info(f"current_tool_use: {current_tool_use}")
                name = current_tool_use.get("name", "")
                input = current_tool_use.get("input", "")
                toolUseId = current_tool_use.get("toolUseId", "")

                text = f"name: {name}, input: {input}"
                logger.info(f"[current_tool_use] {text}")

                if toolUseId not in chat.tool_info_list: # new tool info
                    chat.index += 1
                    current = ""
                    logger.info(f"new tool info: {toolUseId} -> {chat.index}")
                    chat.tool_info_list[toolUseId] = chat.index
                    chat.tool_name_list[toolUseId] = name
                    chat.add_notification(containers, f"Tool: {name}, Input: {input}")
                else: # overwrite tool info if already exists
                    logger.info(f"overwrite tool info: {toolUseId} -> {chat.tool_info_list[toolUseId]}")
                    containers['notification'][chat.tool_info_list[toolUseId]].info(f"Tool: {name}, Input: {input}")

            elif "message" in event:
                message = event["message"]
                logger.info(f"[message] {message}")

                if "content" in message:
                    content = message["content"]
                    logger.info(f"tool content: {content}")
                    if "toolResult" in content[0]:
                        toolResult = content[0]["toolResult"]
                        toolUseId = toolResult["toolUseId"]
                        toolContent = toolResult["content"]
                        toolResult = toolContent[0].get("text", "")
                        tool_name = chat.tool_name_list[toolUseId]
                        logger.info(f"[toolResult] {toolResult}, [toolUseId] {toolUseId}")
                        chat.add_notification(containers, f"Tool Result: {str(toolResult)}")

                        if content:
                            logger.info(f"content: {content}")                
                
            elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
                pass

            else:
                logger.info(f"event: {event}")
        
        if containers is not None:
            containers['notification'][chat.index].markdown(result)

    return result

async def show_result(graph_result, containers):
    """Batch processing for GraphResult object"""
    result = ""
    
    # Debug: Log the GraphResult object structure
    logger.info(f"GraphResult type: {type(graph_result)}")
    logger.info(f"GraphResult attributes: {[attr for attr in dir(graph_result) if not attr.startswith('_')]}")
    
    # Process execution order information
    if hasattr(graph_result, 'execution_order'):
        chat.add_notification(containers, "=== Execution Order ===")
        for node in graph_result.execution_order:
            chat.add_notification(containers, f"Executed: {node.node_id}")
    
    # Process performance metrics
    if hasattr(graph_result, 'total_nodes'):
        chat.add_notification(containers, f"Total nodes: {graph_result.total_nodes}")
    if hasattr(graph_result, 'completed_nodes'):
        chat.add_notification(containers, f"Completed nodes: {graph_result.completed_nodes}")
    if hasattr(graph_result, 'failed_nodes'):
        chat.add_notification(containers, f"Failed nodes: {graph_result.failed_nodes}")
    if hasattr(graph_result, 'execution_time'):
        chat.add_notification(containers, f"Execution time: {graph_result.execution_time}ms")
    if hasattr(graph_result, 'accumulated_usage'):
        chat.add_notification(containers, f"Token usage: {graph_result.accumulated_usage}")
    
    # Process specific node results and combine them
    if hasattr(graph_result, 'results'):
        chat.add_notification(containers, "=== Individual Node Results ===")
        node_results = []
        for node_id, node_result in graph_result.results.items():
            if hasattr(node_result, 'result'):
                node_content = f"{node_id}: {node_result.result}"
                chat.add_notification(containers, node_content)
                node_results.append(node_content)
        
        # Combine individual node results as the final result
        if node_results:
            result = "\n\n".join(node_results)
            logger.info(f"Combined result from individual nodes: {result}")
            chat.update_streaming_result(containers, result, "markdown")
    
    return result

# currently cyclic graph is not supported (Sep. 10 2025)
async def run_strands_agent_with_plan(query: str, containers: dict):
    global status_msg
    status_msg = []

    global index
    index = 0

    tool = "tavily-search"
    config = mcp_config.load_config(tool)
    mcp_servers = config["mcpServers"]
    logger.info(f"mcp_servers: {mcp_servers}")

    mcp_client = None
    for server_name, server_config in mcp_servers.items():
        logger.info(f"server_name: {server_name}")
        logger.info(f"server_config: {server_config}")
        env = server_config["env"] if "env" in server_config else None

        mcp_client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=server_config["command"], 
                args=server_config["args"], 
                env=env
            )
        ))
        break

    with mcp_client as client:
        mcp_tools = client.list_tools_sync()
        logger.info(f"mcp_tools: {mcp_tools}")
        
        tools = []
        tools.extend(mcp_tools)

        tool_list = get_tool_list(tools)
        logger.info(f"tools loaded: {tool_list}")

        # Create specialized agents
        system_prompt=(
            "For the given objective, come up with a simple step by step plan."
            "This plan should involve individual tasks, that if executed correctly will yield the correct answer." 
            "Do not add any superfluous steps."
            "The result of the final step should be the final answer. Make sure that each step has all the information needed."
            "The plan should be returned in <plan> tag."
        )

        planner = Agent(
            name="plan", 
            system_prompt=system_prompt
        )

        executor = Agent(
            name="executor", 
            system_prompt=(
                "You are an executor who executes the plan."
                "주어진 plan을 순차적으로 실행합니다."
                # "모든 task가 완료되었다면 'All tasks completed'라고 리턴합니다."
            ),
            tools=tools
        )
        
        # Build the graph
        builder = GraphBuilder()

        # Add nodes
        builder.add_node(planner, "planner")
        builder.add_node(executor, "executor")

        # Set entry points (optional - will be auto-detected if not specified)
        builder.set_entry_point("planner")

        # Add edges (dependencies)
        builder.add_edge("planner", "executor")
        
        # Build the graph
        graph = builder.build()

        result = await graph.invoke_async(query)
        final_result = await show_result(result, containers)
        logger.info(f"final_result: {final_result}")

        if containers is not None:
            containers['notification'][index].markdown(final_result)

    return final_result, []


