from typing import Annotated, Sequence, TypedDict, Dict, List
from unittest import result
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolCall, ToolMessage # Message for providing instructions to the LLM
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import get_budget_tool
from pprint import pprint
from opik import configure
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv

configure()
load_dotenv()

class BudgetHobbyAgentState(TypedDict):
    budget: Dict
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools = [get_budget_tool]
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=llm_endpoint).bind_tools(tools)

def model_call(state: BudgetHobbyAgentState) -> BudgetHobbyAgentState:
    system_prompt = SystemMessage(content=
        "You are my budget and hobby planning assistant, help me manage my budget effectively."
    )

    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]} 


def should_continue(state: BudgetHobbyAgentState) -> BudgetHobbyAgentState: 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"


def build_budget_hobby_agent_graph():
    graph = StateGraph(BudgetHobbyAgentState)
    graph.add_node("llm", model_call)


    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    graph.add_edge("tools", "llm")

    return graph.compile()
    
budget_agent = build_budget_hobby_agent_graph()

if __name__ == "__main__":
    project_name = 'BudgetHobbyAgent'
    tracer = OpikTracer(graph=budget_agent.get_graph(xray=True), project_name=project_name) 
    inputs = {
        'messages': [
            HumanMessage(content="Get me the budget details for account number 3211555699.")
        ]
    }
    result = budget_agent.invoke(
        inputs,
        config={
            "callbacks": [tracer], 
        },
    )
    print(result["messages"][-1].content)
    