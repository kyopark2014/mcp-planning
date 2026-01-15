# MCP를 이용한 Planning Agent의 구현

MCP agent는 필요한 정보를 multi step reasoning을 통해 수집할 수 있어서 일반적으로 좋은 결과를 얻습니다. 하지만 목적에 따라서 더 많은 결과를 얻기를 원한다면 planning을 이용해 agent의 성능을 향상시킬 수 있습니다. 아래에서는 planning을 구현하기 위하여 multi agent를 이용하는 방법과 graph를 이용하는 방법을 설명합니다. 이를 통해 좀더 풍부하고 의미있는 agentic AI 애플리케이션을 만들 수 있습니다.

## Multi Agent를 이용한 Planning

아래에서는 plan agent와 execute agent를 이용해 multi agent 패턴을 구현하고 있습니다. [Plan and Execute](https://github.com/kyopark2014/langgraph-agent?tab=readme-ov-file#plan-and-execute) 패턴은 plan node와 execute node를 이용하여 cycle 형태로 plan을 업데이트하고 실행하는 방법을 이용해 좋은 결과를 얻지만 cycle의 횟수만큼 실행 시간이 증가합니다. 여기에서는 multi-step reasoning을 통해 cycle 없이 plan을 실행함으로써 실행시간을 빠르게 하면서도 single agent 대비 많은 출력과 향상된 결과를 얻을 수 있습니다.

<img width="283" height="377" alt="image" src="https://github.com/user-attachments/assets/1e541e64-b959-407a-8791-0b4538f4a192" />

### LangGraph로 구현시

Multi agent를 이용할 경우에 아래와 같이 plan_agent의 plan을 execute_agent에서 실행하는 방법을 활용할 수 있습니다. Plan agent의 결과인 plan은 execute의 agent의 입력이 되어서 명확한 가이드를 제공합니다.

```python
plan = await plan_agent(query)
prompt = query + "\n 다음의 계획을 참고하여 답변하세요.\n" + plan
result, image_url = await execute_agent(prompt, mcp_servers)
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

### Strands SDK로 구현시

아래와 같이 Strands SDK를 이용해 plan을 생성합니다. Step by step으로 게획을 작성하라고 요청하고, <plan> tag를 이용해 plan만을 추출하고 있습니다. 

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

이제 아래와 같이 주어진 plan에 따라 MCP를 이용해 정보를 수집하고 답변을 생성합니다.

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
    prompt = question + "\n 다음의 계획을 참고하여 답변하세요.\n" + plan

    agent_stream = executor.stream_async(prompt)

    current = ""
    async for event in agent_stream:
        text = ""            
        if "data" in event:
            text = event["data"]
            current += text
```

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

## 배포하기

### EC2로 배포하기

AWS console의 EC2로 접속하여 [Launch an instance](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)를 선택합니다. [Launch instance]를 선택한 후에 적당한 Name을 입력합니다. (예: es) key pair은 "Proceed without key pair"을 선택하고 넘어갑니다. 

<img width="700" alt="ec2이름입력" src="https://github.com/user-attachments/assets/c551f4f3-186d-4256-8a7e-55b1a0a71a01" />


Instance가 준비되면 [Connet] - [EC2 Instance Connect]를 선택하여 아래처럼 접속합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/e8a72859-4ac7-46af-b7ae-8546ea19e7a6" />

이후 아래와 같이 python, pip, git, boto3를 설치합니다.

```text
sudo yum install python3 python3-pip git docker -y
pip install boto3
```

Workshop의 경우에 아래 형태로 된 Credential을 복사하여 EC2 터미널에 입력합니다.

<img width="700" alt="credential" src="https://github.com/user-attachments/assets/261a24c4-8a02-46cb-892a-02fb4eec4551" />

아래와 같이 git source를 가져옵니다.

```python
git clone https://github.com/kyopark2014/es-us-project
```

아래와 같이 installer.py를 이용해 설치를 시작합니다.

```python
cd es-us-project && python3 installer.py
```

API 구현에 필요한 credential은 secret으로 관리합니다. 따라서 설치시 필요한 credential 입력이 필요한데 아래와 같은 방식을 활용하여 미리 credential을 준비합니다. 

- 일반 인터넷 검색: [Tavily Search](https://app.tavily.com/sign-in)에 접속하여 가입 후 API Key를 발급합니다. 이것은 tvly-로 시작합니다.  
- 날씨 검색: [openweathermap](https://home.openweathermap.org/api_keys)에 접속하여 API Key를 발급합니다. 이때 price plan은 "Free"를 선택합니다.

설치가 완료되면 아래와 같은 CloudFront로 접속하여 동작을 확인합니다. 

<img width="500" alt="cloudfront_address" src="https://github.com/user-attachments/assets/7ab1a699-eefb-4b55-b214-23cbeeeb7249" />

접속한 후 아래와 같이 Agent를 선택한 후에 적절한 MCP tool을 선택하여 원하는 작업을 수행합니다.

<img width="750" alt="image" src="https://github.com/user-attachments/assets/30ea945a-e896-438f-9f16-347f24c2f330" />

인프라가 더이상 필요없을 때에는 uninstaller.py를 이용해 제거합니다.

```text
python uninstaller.py
```


### 배포된 Application 업데이트 하기

AWS console의 EC2로 접속하여 [Launch an instance](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)를 선택하여 아래와 같이 아래와 같이 "app-for-es-us"라는 이름을 가지는 instance id를 선택합니다.

<img width="750" alt="image" src="https://github.com/user-attachments/assets/7d6d756a-03ba-4422-9413-9e4b6d3bc1da" />

[connect]를 선택한 후에 Session Manager를 선택하여 접속합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d1119cd6-08fb-4d3e-b1c2-77f2d7c1216a" />

이후 아래와 같이 업데이트한 후에 다시 브라우저에서 확인합니다.

```text
cd ~/es-us-project/ && sudo ./update.sh
```

### 실행 로그 확인

[EC2 console](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Instances:)에서 "app-for-es-us"라는 이름을 가지는 instance id를 선택 한 후에, EC2의 Session Manager를 이용해 접속합니다. 

먼저 아래와 같이 현재 docker container ID를 확인합니다.

```text
sudo docker ps
```

이후 아래와 같이 container ID를 이용해 로그를 확인합니다.

```text
sudo docker logs [container ID]
```

실제 실행시 결과는 아래와 같습니다.

<img width="600" src="https://github.com/user-attachments/assets/2ca72116-0077-48a0-94be-3ab15334e4dd" />

### Local에서 실행하기

AWS 환경을 잘 활용하기 위해서는 [AWS CLI를 설치](https://docs.aws.amazon.com/ko_kr/cli/v1/userguide/cli-chap-install.html)하여야 합니다. EC2에서 배포하는 경우에는 별도로 설치가 필요하지 않습니다. Local에 설치시는 아래 명령어를 참조합니다.

```text
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" 
unzip awscliv2.zip
sudo ./aws/install
```

AWS credential을 아래와 같이 AWS CLI를 이용해 등록합니다.

```text
aws configure
```

설치하다가 발생하는 각종 문제는 [Kiro-cli](https://aws.amazon.com/ko/blogs/korea/kiro-general-availability/)를 이용해 빠르게 수정합니다. 아래와 같이 설치할 수 있지만, Windows에서는 [Kiro 설치](https://kiro.dev/downloads/)에서 다운로드 설치합니다. 실행시는 셀에서 "kiro-cli"라고 입력합니다. 

```python
curl -fsSL https://cli.kiro.dev/install | bash
```

venv로 환경을 구성하면 편리하게 패키지를 관리합니다. 아래와 같이 환경을 설정합니다.

```text
python -m venv .venv
source .venv/bin/activate
```

이후 다운로드 받은 github 폴더로 이동한 후에 아래와 같이 필요한 패키지를 추가로 설치 합니다.

```text
pip install -r requirements.txt
```

이후 아래와 같은 명령어로 streamlit을 실행합니다. 

```text
streamlit run application/app.py
```

## 실행 결과

아래와 같이 "서울에서 제주 여행하는 방법은?"와 같은 질문을 하면, 이 질문을 해결하기 위한 plan을 아래와 같이 생성합니다. 이후 MCP agent는 plan을 수행하기 위한 검색을 수행하므로 sigle agent일 때보다 더 많은 많은 검색을 수행하게 됩니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/1bafdfcb-69ef-47ac-8795-a917a065cd86" />

Plan을 수행하기 위한 충분한 정보가 수집되면 아래와 같은 결과를 얻을 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/30505964-93de-4ac6-97cf-213d2fd6ee5f" />


이와 같이, planning을 활용하면, MCP agent에게 명확한 가이드를 제공할 수 있어서, 아래와 같이 single agent보다 더 많은 출력과 향상된 결과를 얻을 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/6facb328-e345-496e-a762-97a2f879a55b" />


