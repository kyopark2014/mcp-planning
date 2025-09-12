# MCP를 이용한 Planning의 구현

MCP agent는 필요한 정보를 multi step resoning을 통해 수집할 수 있어서 일반적으로 좋은 결과를 얻습니다. 하지만 목적에 따라서 더 많은 결과를 얻기를 원한다면 planning을 이용해 agent의 성능을 향상시킬 수 있습니다. 아래에서는 planning을 구현하기 위하여 multi agent를 이용하는 방법과 graph를 이용하는 방법을 설명합니다. 이를 통해 좀더 풍부하고 의미있는 agentic AI 애플리케이션을 만들 수 있습니다.

## Multi Agent를 이용한 Planning

아래에서는 plan agent와 execute agent를 이용해 multi agent 패턴을 구현하고 있습니다. [Plan and Execute](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#plan-and-execute) 패턴은 plan node와 execute node를 이용하여 cycle 형태로 plan을 업데이트하고 실행하는 방법을 이용해 좋은 결과를 얻지만 cycle의 횟수만큼 실행 시간이 증가합니다. 여기에서는 multi-step reasoning을 통해 cycle 없이 plan을 실행함으로써 실행시간을 빠르게 하면서도 single agent 대비 많은 출력과 향상된 결과를 얻을 수 있습니다.

<img width="283" height="377" alt="image" src="https://github.com/user-attachments/assets/1e541e64-b959-407a-8791-0b4538f4a192" />

### LangGraph로 구현시

Multi agent를 이용할 경우에 아래와 같이 plan_agent의 plan을 execute_agent에서 실행하는 방법을 활용할 수 있습니다.

```python
plan = await plan_agent(query)

result, image_url = await execute_agent(query, plan, mcp_servers)
```

여기서 plan_agent는 아래와 같이 동작합니다. Agent의 결과에서 plan만을 추출하기 위하여 아래와 같이 <plan> tag를 prompt에 붙이도록 하고, 추출시 제거합니다.

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

Langgraph는 아래와 같이 구성합니다. 

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

Execute agent는 아래와 같이 plan에 따라 MCP를 조회하여 정보를 수집하고 결과를 stream으로 반환합니다.

```python
async def execute_agent(query: str, plan: str, mcp_servers: list):
    image_url = []
    references = []
    
    mcp_json = mcp_config.load_selected_config(mcp_servers)
    server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)
    client = MultiServerMCPClient(server_params)
    tools = await client.get_tools()    
    app = langgraph_agent.buildChatAgent(tools)

    system_prompt=(
        "You are an executor who executes the plan."
        "Make sure that each step has all the information needed."
        f"<plan>{plan}</plan>"
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

이때의 결과는 아래와 같습니다. Muli agent로 구현한 planning agent는 plan에 기반하여 충분한 정보를 수집한 후에 아래와 같이 single agent보다 더 많은 출력을 제공합니다.

![ezgif-699b13ca6aa10d](https://github.com/user-attachments/assets/6facb328-e345-496e-a762-97a2f879a55b)


## Graph를 이용한 Planning

아래에서는 LangGraph에서 plan node를 추가한 agent를 활용함으로써 많은 출력과 향상된 결과를 얻는 방법에 대해 설명합니다. Agent는 기본적으로 agent node와 action node로 구성됩니다. 사용자의 입력으로 plan을 생성한 후에 agent와 action형태로 MCP를 활용함으로써 충분한 정보를 수집하여 답변할 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d5a2b2d8-9946-47c2-add7-9fd0411c4274" />

Agent에 plan node를 아래와 같이 추가합니다.

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

Plan node는 prompt를 이용해 계획을 세우고, 결과는 아래와 같이 HumanMessage로 반환함으로써, agent가 사용자의 지시사항으로 인지하게 합니다.

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
    response = HumanMessage(content="다음의 plan을 참고하여 답변하세요.\n" + plan)

    return {"messages": [response]}
```

이후 아래와 같이 MCP tool에 대한 정보를 이용해 필요한 정보를 수집해 답변을 생성합니다.

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
