# MCP agent based on Planning Pattern

MCP agents can collect necessary information through multi-step reasoning, generally achieving good results. However, if you want to achieve more results depending on your objectives, you can enhance the agent's performance using planning. Below, I explain how to use multi-agent approaches and graph-based methods to implement planning. This allows you to create richer and more meaningful agentic AI applications.

## Planning using Multi Agent

Below, I implement a multi-agent pattern using plan agent and execute agent. The [Plan and Execute](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#plan-and-execute) pattern uses plan nodes and execute nodes in a cycle format to update and execute plans, achieving good results but increasing execution time by the number of cycles. Here, we execute plans without cycles through multi-step reasoning, achieving faster execution time while obtaining more output and improved results compared to single agents.

<img width="283" height="377" alt="image" src="https://github.com/user-attachments/assets/1e541e64-b959-407a-8791-0b4538f4a192" />

### Implementation using LangGraph

When using multi-agent approaches, you can utilize the method of executing the plan from plan_agent in execute_agent as shown below. The plan, which is the result of the plan agent, becomes the input for the execute agent, providing clear guidance.

```python
plan = await plan_agent(query)
prompt = query + "\n Please refer to the following plan when answering.\n" + plan
result, image_url = await execute_agent(prompt, mcp_servers)
```

Here, the plan_agent operates as follows. To extract only the plan from the agent's result, we attach the <plan> tag to the prompt as shown below and remove it during extraction.

```python
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
    "tools": tools,
    "system_prompt": system_prompt
}
inputs = {
    "messages": [HumanMessage(content=query)]
}

response = await app.ainvoke(inputs, config)
result = response['messages'][-1].content
plan = result[result.find('<plan>')+6:result.find('</plan>')]
```

LangGraph is configured as follows. 

```python
def buildChatAgent(tools):
    tool_node = ToolNode(tools)

    workflow = StateGraph(State)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")

    return workflow.compile() 
```

The execute agent queries MCP according to the plan to collect information and returns results as a stream as shown below.

```python
async def execute_agent(prompt: str, mcp_servers: list):
    image_url = []
    references = []
    
    mcp_json = mcp_config.load_selected_config(mcp_servers)
    server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)
    client = MultiServerMCPClient(server_params)
    tools = await client.get_tools()    
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
    async for stream in app.astream(inputs, config, stream_mode="messages"):
        if isinstance(stream[0], AIMessageChunk):
            message = stream[0]    
            input = {}        
            if isinstance(message.content, list):
                for content_item in message.content:
                    if isinstance(content_item, dict):
                        if content_item.get('type') == 'text':
                            text_content = content_item.get('text', '')                            
                            result += text_content
                                
    return result, image_url
```

### Implementation using Strands SDK

We generate a plan using the Strands SDK as shown below. We request step-by-step planning and extract only the plan using the <plan> tag. 

```python
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

response = planner(question)
result = str(response)
plan = result[result.find('<plan>')+6:result.find('</plan>')]
```

Now we collect information using MCP and generate answers according to the given plan as shown below.

```python
tool = "tavily-search"
config = mcp_config.load_config(tool)
mcp_servers = config["mcpServers"]
mcp_client = None
for server_name, server_config in mcp_servers.items():
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
    tools = []
    tools.extend(mcp_tools)

    executor = Agent(
        name="executor", 
        tools=tools
    )
    prompt = question + "\n Please refer to the following plan when answering.\n" + plan

    agent_stream = executor.stream_async(prompt)

    current = ""
    async for event in agent_stream:
        text = ""            
        if "data" in event:
            text = event["data"]
            current += text
```

## Planning using Graph

Below, we explain how to achieve more output and improved results by utilizing agents with plan nodes added in LangGraph. Agents are basically composed of agent nodes and action nodes. After generating a plan from user input, we can collect sufficient information to answer by utilizing MCP in agent and action forms.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d5a2b2d8-9946-47c2-add7-9fd0411c4274" />

We add a plan node to the agent as shown below.

```python
def buildChatAgentWithPlan(tools):
    tool_node = ToolNode(tools)

    workflow = StateGraph(State)

    workflow.add_node("plan", plan_node)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")

    return workflow.compile() 
```

The plan node creates a plan using prompts and returns the result as a HumanMessage as shown below, so that the agent recognizes it as user instructions.

```python
async def plan_node(state: State, config):
    system=(
        "For the given objective, come up with a simple step by step plan."
        "This plan should involve individual tasks, that if executed correctly will yield the correct answer." 
        "Do not add any superfluous steps."
        "The result of the final step should be the final answer. Make sure that each step has all the information needed"
        "The plan should be returned in <plan> tag."
    )
    chatModel = chat.get_chat(extended_thinking="Disable")    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | chatModel
        
    result = await chain.ainvoke(state["messages"])
    plan = result.content[result.content.find('<plan>')+6:result.content.find('</plan>')]
    plan = plan.strip()
    response = HumanMessage(content="Please refer to the following plan when answering.\n" + plan)

    return {"messages": [response]}
```

Then, we collect necessary information using MCP tool information and generate answers as shown below.

```python
mcp_json = mcp_config.load_selected_config(mcp_servers)
server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)
client = MultiServerMCPClient(server_params)
tools = await client.get_tools()
    
app = langgraph_agent.buildChatAgentWithPlan(tools)
config = {
    "recursion_limit": 50,
    "tools": tools,
}        
inputs = {
    "messages": [HumanMessage(content=query)]
}
        
result = ""
async for stream in app.astream(inputs, config, stream_mode="messages"):
    if isinstance(stream[0], AIMessageChunk):
        message = stream[0]    
        input = {}        
        if isinstance(message.content, list):
            for content_item in message.content:
                if isinstance(content_item, dict):
                    if content_item.get('type') == 'text':
                        text_content = content_item.get('text', '')
                        result += text_content                            
```

## Execution Results

When asking a question like "How to travel from Seoul to Jeju?" as shown below, we generate a plan to solve this question as shown below. Afterwards, the MCP agent performs searches to execute the plan, so it performs more searches than when using a single agent. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/1bafdfcb-69ef-47ac-8795-a917a065cd86" />

When sufficient information is collected to execute the plan, we can obtain results as shown below.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/30505964-93de-4ac6-97cf-213d2fd6ee5f" />


In this way, by utilizing planning, we can provide clear guidance to the MCP agent, achieving more output and improved results compared to single agents as shown below.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/6facb328-e345-496e-a762-97a2f879a55b" />


