from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolCall,
    ToolMessage,
    HumanMessage,
)  # The foundational class for all message types in LangGraph
from langchain_core.messages import (
    ToolMessage,
)  # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import (
    SystemMessage,
)  # Message for providing instructions to the LLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import fetch_emails_tool, send_email_tool
from opik import configure
from opik.integrations.langchain import OpikTracer
from langchain_openai import ChatOpenAI


# configure()
# load_dotenv()

email_agent_prompt = """
You are the Email Agent, a specialized sub-agent within the ORBITA multi-agent system. 

Your role:
- You report to the manager agent named ORBITA
- You handle ALL email-related tasks including reading, composing, and sending emails
- You work collaboratively with other specialized agents (Budget Agent, etc.)
- You have access to email tools: fetch_emails_tool and send_email_tool

When processing requests:
1. Use the provided tools to perform email operations
2. Focus exclusively on email tasks - let other agents handle their specialties
3. Provide clear, actionable responses about email operations
4. Report back to ORBITA when tasks are complete

Remember: You are part of a larger system. Stay focused on email management.
"""


class EmailAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [fetch_emails_tool, send_email_tool]
# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-Small-24B-Instruct-v0.4",
#     temperature=0.2,
#     max_new_tokens=1024,
# )
llm = ChatNVIDIA(model="openai/gpt-oss-120b").bind_tools(tools)


def model_call(state: EmailAgentState) -> EmailAgentState:
    system_prompt = SystemMessage(content=email_agent_prompt)
    # dòng trên tương đương {"role": "system", "content": "..."}
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}  # returns the part being updated


def should_continue(state: EmailAgentState) -> EmailAgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if (
        not last_message.tool_calls
    ):  # kiểm tra xem có tín hiệu gọi tools từ message cuối cùng không
        return "end"
    else:
        return "continue"


def build_email_agent_graph() -> StateGraph:
    graph = StateGraph(EmailAgentState)
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

    return graph.compile(checkpointer=InMemorySaver())


email_agent = build_email_agent_graph()

if __name__ == "__main__":
    # from langchain_core.messages import HumanMessage
    # project_name = 'EmailAgent'
    # tracer = OpikTracer(graph=email_agent.get_graph(xray=True), project_name=project_name)
    # inputs = {
    #     "messages": [
    #         HumanMessage(content=" viết mail gửi lời chào đến địa chỉ dominhquang_t67@hus.edu.vn và đọc cho tôi mail mới nhất đến từ địa chỉ nguyenthuytrang1_t67@hus.edu.vn, sau đó kiểm tra thông tin về ngân sách trong tài khoản 3211555699")
    #     ]
    # }
    # result = email_agent.invoke(
    #     inputs,
    #     config={
    #         "callbacks": [tracer],
    #     },
    # )

    # print(result["messages"][-1].content)
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("You: ")
        inputs = {"messages": [HumanMessage(content=user_input)]}
        result = email_agent.invoke(inputs, config=config)
        print("Email Agent:", result["messages"][-1].content)
