from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, AIMessage, ToolCall, ToolMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import fetch_emails_tool, send_email_tool
from opik import configure 
from opik.integrations.langchain import OpikTracer 


configure()
load_dotenv()


class EmailAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [fetch_emails_tool, send_email_tool]
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=llm_endpoint).bind_tools(tools)

def model_call(state: EmailAgentState) -> EmailAgentState:
    system_prompt = SystemMessage(content=
        "You are my email assistant, help me manage my emails effectively."
    )
    #dòng trên tương đương {"role": "system", "content": "..."}
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]} #returns the part being updated


def should_continue(state: EmailAgentState) -> EmailAgentState: 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: #kiểm tra xem có tín hiệu gọi tools từ message cuối cùng không
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
    
    return graph.compile()

email_agent = build_email_agent_graph()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    project_name = 'EmailAgent'
    tracer = OpikTracer(graph=email_agent.get_graph(xray=True), project_name=project_name) 
    inputs = {
        "messages": [
            HumanMessage(content=" viết mail gửi lời chào đến địa chỉ dominhquang_t67@hus.edu.vn và đọc cho tôi mail mới nhất đến từ địa chỉ nguyenthuytrang1_t67@hus.edu.vn, sau đó kiểm tra thông tin về ngân sách trong tài khoản 3211555699")
        ]
    }
    result = email_agent.invoke(
        inputs,
        config={
            "callbacks": [tracer], 
        },
    )
    
    print(result["messages"][-1].content)
    
