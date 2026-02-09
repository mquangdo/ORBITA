from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from email_agent import email_agent
from budget_agent import budget_agent
from opik import configure
from opik.integrations.langchain import OpikTracer
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableConfig
from manager_memory import (
    load_manager_memories,
    decide_what_to_update,
    update_profile_memory,
    update_preferences_memory,
    update_instructions_memory,
    store,
)
import os
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


class ManagerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str
    memory_context: str
    update_decision: dict


# Use same LLM as all agents
llm = ChatNVIDIA(model="openai/gpt-oss-120b", temperature=0.2)


# Runs at start of every conversation
def load_memory(state: ManagerState, config: RunnableConfig):
    """Load user memories to personalize responses"""
    memory_data = load_manager_memories(state, config, store)
    return {**memory_data, "messages": state["messages"]}


# Enhanced router that uses memory
def manager_router(state: ManagerState):
    """Route with memory-aware context"""

    user_input = state["messages"][-1].content
    memory_context = state.get("memory_context", "")

    enriched_prompt = f"""
    <user_profile>
    {memory_context}
    </user_profile>
    
    User asked: {user_input}
    """.strip()

    response = llm.invoke([HumanMessage(content=enriched_prompt)])

    # Check if user asked about themselves
    if "what is my name" in user_input.lower() or "who am i" in user_input.lower():
        # Use memory to answer
        if "name" in memory_context:
            # Extract name from memory and respond directly
            import json
            import re

            try:
                # Parse the memory context to find name
                # memory_context contains: {'name': 'Quang', ...}
                name_match = re.search(r"'name':\s*'([^']+)'", memory_context)
                if name_match:
                    name = name_match.group(1)
                    return {
                        "messages": [AIMessage(content=f"I remember you are {name}!")],
                        "route": "end",
                    }
            except:
                pass

    # Normal routing
    content = response.content.strip().lower()
    if "email" in content:
        return {"route": "email", "messages": state["messages"]}
    if "budget" in content:
        return {"route": "budget", "messages": state["messages"]}

    return {"messages": [response], "route": "end"}


# Runs after each sub-agent call
def update_memory(state: ManagerState, config: RunnableConfig):
    """Save what we learned from this interaction"""

    # Extract and save profile info
    update_profile_memory(state, config, store)

    # Extract and save preferences
    update_preferences_memory(state, config, store)

    # Extract and save system instructions
    update_instructions_memory(state, config, store)

    return {"messages": state["messages"]}


def build_manager_agent():
    graph = StateGraph(ManagerState)

    # Add all nodes
    graph.add_node("load_memory", load_memory)
    graph.add_node("router", manager_router)
    graph.add_node("email_agent", email_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("update_memory", update_memory)

    # Graph flow
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "router")

    # Route to appropriate sub-agent
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"email": "email_agent", "budget": "budget_agent", "end": "update_memory"},
    )

    # After sub-agent, always update memory
    graph.add_edge("email_agent", "update_memory")
    graph.add_edge("budget_agent", "update_memory")

    # After updating memory, end
    graph.add_edge("update_memory", END)

    return graph.compile()


# Compile once
manager_agent = build_manager_agent()


def safe_print(text):
    """Print safely with UTF-8 encoding"""
    print(text.encode("utf-8", errors="replace").decode("utf-8"))


if __name__ == "__main__":
    # from langchain_core.messages import HumanMessage
    # project_name = 'ManagerAgent'
    # tracer = OpikTracer(graph=manager_agent.get_graph(xray=True), project_name=project_name)
    # inputs = {
    #     "messages": [
    #         HumanMessage(content="Hello my name is Quang")
    #     ]
    # }

    # result = manager_agent.invoke(
    #     inputs,
    #     config={
    #         "callbacks": [tracer],
    #         "configurable": {"thread_id": "1"}
    #     },
    # )
    # print(result["messages"])

    # inputs = {
    #     "messages": [
    #         HumanMessage(content="What is my name?")
    #     ]
    # }

    # result = manager_agent.invoke(
    #     inputs,
    #     config={
    #         "callbacks": [tracer],
    #         "configurable": {"thread_id": "1"}
    #     },
    # )
    # print(result["messages"])

    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        result = manager_agent.invoke(inputs, config=config)
        print("Manager Agent:", result["messages"][-1].content)
