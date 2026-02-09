"""
ORBITA Manager Agent - Streamlit UI
A modern, professional interface for the ORBITA Multi-Agent System
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import sys
import uuid

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")

# Page configuration
st.set_page_config(
    page_title="ORBITA - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for ChatGPT-like design
st.markdown(
    f"""
<style>
/* ==================== MAIN LAYOUT ==================== */

/* Chuyển nền chính sang màu xanh đen đậm #111827 */
.main, .stApp {{
    background-color: #111827 !important;
    color: #ffffff !important;
}}

/* Chỉnh màu nền Sidebar (vùng bên trái) */
[data-testid="stSidebar"] {{
    background-color: #0b0f1a !important;
    border-right: 1px solid #1f2937;
}}

/* ==================== CHAT CONTAINER ==================== */

.chat-container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 1rem;
    background-color: #111827; /* Đồng bộ với nền chính hoặc dùng #1f2937 nếu muốn tách khối */
    border-radius: 12px;
    margin-top: 1rem;
    margin-bottom: 2rem;
}}

/* ==================== CONVERSATION TEXT ==================== */

/* User messages - Giữ màu xanh nhạt #d1fae5 nhưng chữ đen để rõ ràng */
.user-message {{
    background-color: #d1fae5 !important;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    margin-left: 15%;
    margin-right: 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    max-width: 85%;
}}

.user-message, .user-message strong, .user-message div, .user-message p {{
    color: #064e3b !important; /* Xanh lá đậm cho chữ để dễ đọc trên nền xanh nhạt */
    font-weight: 500;
    font-size: 15px;
    line-height: 1.5;
}}

/* Assistant messages - Nền tối nhẹ #1f2937 để nổi bật trên nền đen chính */
.bot-message {{
    background-color: #1f2937 !important; 
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    margin-left: 0%;
    margin-right: 15%;
    max-width: 85%;
    border: 1px solid #374151;
}}

.bot-message, .bot-message strong, .bot-message div, .bot-message p {{
    color: #f3f4f6 !important; /* Chữ trắng xám nhẹ */
    font-weight: 500;
    font-size: 15px;
    line-height: 1.5;
}}

/* Nhãn tên You/ORBITA */
.user-message strong {{
    color: #059669 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
}}
.bot-message strong {{
    color: #9ca3af !important;
    font-size: 0.8rem;
    text-transform: uppercase;
}}

/* ==================== UI ELEMENTS ==================== */

/* Tiêu đề chính */
.header-title {{
    color: #ffffff !important;
    font-weight: 600;
    font-size: 2rem;
}}
.header-subtitle {{
    color: #9ca3af !important; /* Gray-400 */
    font-size: 0.95rem;
}}

/* Sidebar text */
.sidebar .stMarkdown, [data-testid="stSidebar"] p {{
    color: #9ca3af !important;
}}
.sidebar .stMarkdown strong {{
    color: #e5e7eb !important;
}}

/* Tips and captions */
.stCaption, .caption {{
    color: #6b7280 !important;
}}

/* ==================== INPUT FIELD ==================== */

/* Ô nhập liệu tối màu */
.stTextInput input {{
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border: 1px solid #374151 !important;
    border-radius: 10px;
    padding: 12px;
}}

.stTextInput input::placeholder {{
    color: #4b5563 !important;
}}

/* Đảm bảo label của input không bị mất trong nền tối */
.stTextInput label {{
    color: #9ca3af !important;
}}

/* ==================== BUTTONS ==================== */

.stButton>button {{
    background-color: #10b981 !important;
    color: #ffffff !important;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: all 0.2s;
}}

.stButton>button:hover {{
    background-color: #059669 !important;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
}}

/* ==================== SCROLLBAR ==================== */

::-webkit-scrollbar {{
    width: 8px;
}}
::-webkit-scrollbar-track {{
    background: #111827;
}}
::-webkit-scrollbar-thumb {{
    background: #374151;
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #4b5563;
}}
</style>
""",
    unsafe_allow_html=True,
)

# Import the agent at the top level so it's available to all sessions
try:
    from manager_agent import manager_agent

    AGENT_AVAILABLE = True
except Exception as e:
    AGENT_AVAILABLE = False
    st.error(f"❌ Failed to load manager agent: {str(e)}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "agent_status" not in st.session_state:
    st.session_state.agent_status = "idle"

# Header
st.markdown('<h1 class="header-title">🤖 ORBITA</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-subtitle">Your Intelligent Multi-Agent Assistant</p>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Agent status
    st.markdown("#### Agent Status")
    if AGENT_AVAILABLE:
        st.markdown(
            '<span class="status-indicator status-online"></span>Online',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-indicator status-offline"></span>Offline',
            unsafe_allow_html=True,
        )

    # Session info
    st.markdown("#### Session Info")
    st.caption(f"Session ID: `{st.session_state.user_id[:8]}...`")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    # About
    st.markdown("---")
    st.markdown("#### About ORBITA")
    st.caption(
        "A multi-agent system with long-term memory capabilities. Recognizes users and remembers conversations across sessions."
    )

# Main chat area
st.markdown("### 💬 Conversation")

# Display chat messages
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown(
            """
        <div style="text-align: center; padding: 3rem; color: #999;">
            <h3>👋 Welcome to ORBITA</h3>
            <p>Start a conversation to experience intelligent assistance</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="bot-message">
                    <strong>🤖 ORBITA:</strong> {message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )

# Input area
st.markdown("---")

# Use form to handle Enter key submission
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "💬 Type your message and press Enter to send",
        placeholder="Type here...",
        label_visibility="collapsed",
    )

    # Hidden submit button (Enter key will trigger form submission)
    submitted = st.form_submit_button("Send", use_container_width=False)

# Handle form submission (either Enter key or Send button)
if submitted:
    if user_input and user_input.strip():
        if not AGENT_AVAILABLE:
            st.error("Agent is not available. Please check the backend.")
        else:
            # Update UI
            st.session_state.agent_status = "thinking"

            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Show thinking indicator
            with st.spinner("Agent is thinking..."):
                try:
                    # Prepare config with thread ID
                    config = {
                        "configurable": {
                            "thread_id": st.session_state.user_id,
                            "user_id": st.session_state.user_id,
                        }
                    }

                    # Call the agent
                    result = manager_agent.invoke(
                        {"messages": [HumanMessage(content=user_input)]}, config=config
                    )

                    # Extract agent response
                    if result and "messages" in result and result["messages"]:
                        agent_response = result["messages"][-1].content

                        # Add to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": agent_response}
                        )
                    else:
                        st.error("No response received from agent.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

            # Reset status and rerun
            st.session_state.agent_status = "idle"
            st.rerun()
    else:
        st.warning("Please enter a message before sending.")

# Tips
c1, c2 = st.columns([1, 5])
with c2:
    st.caption("💡 Tip: Type your message and press Enter to send!")

# Keyboard shortcut handling (Enter key)
if st.session_state.get("keyboard_event"):
    if user_input:
        st.session_state.enter_pressed = True
        st.rerun()

# Footer
st.markdown("---")
st.caption(
    "Built with LangGraph | Powered by AI🤖 {st.session_state.get('agent_status', 'offline')}"
)

# Handle window refresh
if st.session_state.get("needs_rerun"):
    st.session_state.needs_rerun = False
    st.rerun()
