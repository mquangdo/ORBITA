from typing import Annotated, Sequence, TypedDict, Dict, List
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    AIMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import get_budget_tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pprint import pprint
from opik import configure
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

configure()
load_dotenv()


class BudgetHobbyAgentState(TypedDict):
    budget: Dict
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [get_budget_tool]
# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     temperature=0.2,
#     max_new_tokens=512,
# )

llm = ChatNVIDIA(model="openai/gpt-oss-120b").bind_tools(tools)


def model_call(state: BudgetHobbyAgentState) -> BudgetHobbyAgentState:
    system_prompt = SystemMessage(
        content="""You are the Budget Agent, a specialized sub-agent within the ORBITA multi-agent system.

Your role:
- You report to the manager agent named ORBITA
- You handle ALL budget-related tasks including budget tracking, financial queries, and budget management
- You work collaboratively with other specialized agents (Email Agent, etc.)
- You have access to budget tools: get_budget_tool

When processing requests:
1. Use the provided tools to retrieve and analyze budget information
2. Focus exclusively on budget tasks - let other agents handle their specialties
3. Provide clear, actionable responses about budget operations
4. Report back to ORBITA when tasks are complete

Remember: You are part of a larger system. Stay focused on budget management.
"""
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
    project_name = "BudgetHobbyAgent"
    tracer = OpikTracer(
        graph=budget_agent.get_graph(xray=True), project_name=project_name
    )
    inputs = {
        "messages": [
            HumanMessage(
                content="Get me the budget details for account number 3211555699."
            )
        ]
    }
    result = budget_agent.invoke(
        inputs,
        config={
            "callbacks": [tracer],
        },
    )
    print(result["messages"][-1].content)
