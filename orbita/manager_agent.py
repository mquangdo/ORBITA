from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from email_agent import email_agent  
from budget_agent import budget_agent
from opik import configure
from opik.integrations.langchain import OpikTracer
from langgraph.checkpoint.memory import InMemorySaver


configure()
load_dotenv()


class ManagerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str 


llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.0,
    max_new_tokens=128,
)

llm = ChatHuggingFace(llm=llm_endpoint)

manager_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant and a manager. 
1. If the user request is about email (sending/reading), respond ONLY with the word 'email'.
2. If the user request is about budget, respond ONLY with the word 'budget'.
3. For any other general questions, greetings, or normal conversation, answer the user directly in a friendly manner.
"""),
    ("human", "{input}")
])

def manager_router(state: ManagerState):
    user_input = state["messages"][-1].content

    response = llm.invoke(
        manager_prompt.format(input=user_input)
    )
    
    decision = response.content.strip().lower()

    if "email" in decision:
        return {"route": "email"}

    if "budget" in decision:
        return {"route": "budget"}
    
    return {
        "messages": [response], 
        "route": "end"
    }
    

def call_email_agent(state: ManagerState):
    result = email_agent.invoke({
        "messages": state["messages"]
    })

    final_message = result["messages"][-1]

    return {"messages": [final_message]}

def call_budget_agent(state: ManagerState):
    result = budget_agent.invoke({
        "messages": state["messages"]
    })

    final_message = result["messages"][-1]

    return {"messages": [final_message]}

def build_manager_agent():    
    graph = StateGraph(ManagerState)
    graph.add_node("router", manager_router)
    graph.add_node("email_agent", call_email_agent)
    graph.add_node("budget_agent", call_budget_agent)
    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "email": "email_agent",
            "budget": "budget_agent",   
            "end": END,
        }
    )

    graph.add_edge("email_agent", END)
    graph.add_edge("budget_agent", END)
    
    return graph.compile()

manager_agent = build_manager_agent()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    project_name = 'ManagerAgent'
    tracer = OpikTracer(graph=manager_agent.get_graph(xray=True), project_name=project_name) 
    inputs = {
        "messages": [
            HumanMessage(content="Hello my name is Quang")
        ]
    }
    
    
    result = manager_agent.invoke(
        inputs,
        config={
            "callbacks": [tracer],
            "configurable": {"thread_id": "1"} 
        },
    )
    print(result["messages"])
    
    inputs = {
        "messages": [
            HumanMessage(content="What is my name?")
        ]
    }
    
    
    result = manager_agent.invoke(
        inputs,
        config={
            "callbacks": [tracer],
            "configurable": {"thread_id": "1"} 
        },
    )
    print(result["messages"])
    
    # config = {"configurable": {"thread_id": "1"}}
    # while True:
    #     user_input = input("User: ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         break

    #     inputs = {
    #         "messages": [
    #             HumanMessage(content=user_input)
    #         ]
    #     }

    #     result = manager_agent.invoke(inputs, config=config)
    #     print("Manager Agent:", result['messages'][-1].content)



