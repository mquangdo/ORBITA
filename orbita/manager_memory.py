"""
Long-term memory system for ORBITA Manager Agent
Based on module-5 memory_agent.ipynb pattern
"""

from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from trustcall import create_extractor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
import os
from datetime import datetime
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Use the same LLM model as all agents
model = ChatNVIDIA(model="openai/gpt-oss-120b")

# ============= SCHEMAS =============


class Profile(BaseModel):
    """User profile information collected from conversations"""

    name: Optional[str] = Field(description="User's name", default=None)
    preferred_tone: str = Field(
        description="User's preferred communication style (formal, casual, friendly)",
        default="friendly",
    )
    primary_interests: List[str] = Field(
        description="Topics user asks about most", default_factory=list
    )
    favorite_agents: List[str] = Field(
        description="Which agents user prefers (email_agent, budget_agent, etc)",
        default_factory=list,
    )
    usage_frequency: Dict[str, int] = Field(
        description="How often user interacts with each agent", default_factory=dict
    )
    last_seen: Optional[datetime] = Field(
        description="Last interaction time", default=None
    )


class UserPreference(BaseModel):
    """Specific preferences for how the assistant behaves"""

    preference_type: str = Field(
        description="Category of preference (response_style, verbosity, etc)"
    )
    preference_value: str = Field(description="Value for this preference")
    importance: int = Field(
        description="How important this preference is (1-10)", default=5
    )


class UserPreferences(BaseModel):
    """Collection of user preferences"""

    preferences: List[UserPreference] = Field(default_factory=list)


class SystemInstruction(BaseModel):
    """Instructions about how to manage memories and agents"""

    instruction_type: str = Field(
        description="Type of instruction (routing, tool_use, etc)"
    )
    instruction_text: str = Field(description="The actual instruction")
    created_date: datetime = Field(default_factory=datetime.now)
    active: bool = Field(default=True)


class SystemInstructions(BaseModel):
    """Collection of system instructions"""

    instructions: List[SystemInstruction] = Field(default_factory=list)


# ============= EXTRACTORS =============

# Profile extractor
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
    enable_inserts=True,
)

# Preferences extractor
preferences_extractor = create_extractor(
    model,
    tools=[UserPreferences],
    tool_choice="UserPreferences",
    enable_inserts=True,
)

# System instructions extractor
instructions_extractor = create_extractor(
    model,
    tools=[SystemInstructions],
    tool_choice="SystemInstructions",
    enable_inserts=True,
)

# ============= STORE INITIALIZATION =============

# Global store instance
store = InMemoryStore()


def get_user_id(config: RunnableConfig) -> str:
    """Extract user ID from config - use thread_id if user_id not available"""
    return config["configurable"].get(
        "user_id", config["configurable"].get("thread_id", "default")
    )


# ============= MEMORY NODES =============


def load_manager_memories(
    state: MessagesState, config: RunnableConfig, store: BaseStore
):
    """
    Load all memories for the manager agent to personalize responses
    """
    user_id = get_user_id(config)

    # Load profile
    namespace = ("manager", "profile", user_id)
    memories = store.search(namespace)
    profile = memories[0].value if memories else None

    # Load preferences
    namespace = ("manager", "preferences", user_id)
    memories = store.search(namespace)
    preferences = "\n".join(f"- {mem.value}" for mem in memories) if memories else ""

    # Load instructions
    namespace = ("manager", "instructions", user_id)
    memories = store.search(namespace)
    instructions = "\n".join(f"- {mem.value}" for mem in memories) if memories else ""

    # Create augmented system message with memories
    memory_context = f"""
    
    <user_profile>
    {profile}
    </user_profile>
    
    <user_preferences>
    {preferences}
    </user_preferences>
    
    <system_instructions>
    {instructions}
    </system_instructions>
    """

    return {"memory_context": memory_context}


def update_profile_memory(
    state: MessagesState, config: RunnableConfig, store: BaseStore
):
    """Update user profile based on conversation"""
    user_id = get_user_id(config)
    namespace = ("manager", "profile", user_id)

    # Get existing profile for context
    existing_items = store.search(namespace)
    tool_name = "Profile"
    existing_memories = (
        [(item.key, tool_name, item.value) for item in existing_items]
        if existing_items
        else None
    )

    # Extract update
    trustcall_result = profile_extractor.invoke(
        {
            "messages": state["messages"][-5:],  # Last 5 messages
            "existing": existing_memories,
        }
    )

    # Save to store
    if trustcall_result["responses"]:
        for i, memory in enumerate(trustcall_result["responses"]):
            store.put(namespace, str(i), memory.model_dump())

    return {"messages": [f"Profile updated with: {trustcall_result['responses']}"]}


def update_preferences_memory(
    state: MessagesState, config: RunnableConfig, store: BaseStore
):
    """Update user preferences based on feedback"""
    user_id = get_user_id(config)
    namespace = ("manager", "preferences", user_id)

    # Get existing preferences
    existing_items = store.search(namespace)
    tool_name = "UserPreferences"
    existing_memories = (
        [(item.key, tool_name, item.value) for item in existing_items]
        if existing_items
        else None
    )

    # Extract new preferences
    trustcall_result = preferences_extractor.invoke(
        {
            "messages": state["messages"][-3:],  # Last 3 messages
            "existing": existing_memories,
        }
    )

    # Save individual preferences
    if trustcall_result["responses"]:
        for pref in trustcall_result["responses"][0].preferences:
            store.put(
                namespace,
                f"{pref.preference_type}_{hash(pref.preference_value)}",
                pref,
            )

    return {
        "messages": [
            f"Preferences updated: {len(trustcall_result['responses'])} changes"
        ]
    }


def update_instructions_memory(
    state: MessagesState, config: RunnableConfig, store: BaseStore
):
    """Update system instructions based on user feedback"""
    user_id = get_user_id(config)
    namespace = ("manager", "instructions", user_id)

    # Get existing instructions
    existing_items = store.search(namespace)
    tool_name = "SystemInstructions"
    existing_memories = (
        [(item.key, tool_name, item.value) for item in existing_items]
        if existing_items
        else None
    )

    # Extract updates
    trustcall_result = instructions_extractor.invoke(
        {"messages": state["messages"][-3:], "existing": existing_memories}
    )

    # Save new instructions
    if trustcall_result["responses"]:
        for instr in trustcall_result["responses"][0].instructions:
            store.put(
                namespace,
                f"{instr.instruction_type}_{hash(instr.instruction_text)}",
                instr,
            )

    return {
        "messages": [
            f"Instructions updated: {len(trustcall_result['responses'])} changes"
        ]
    }


# Decision tool to determine what to update
class UpdateDecision(BaseModel):
    """Decision about which memories to update"""

    update_profile: bool = Field(
        description="Whether to update user profile", default=False
    )
    update_preferences: bool = Field(
        description="Whether to update preferences", default=False
    )
    update_instructions: bool = Field(
        description="Whether to update system instructions", default=False
    )


def decide_what_to_update(state: MessagesState, config: RunnableConfig) -> dict:
    """
    Analyze conversation and decide what memories to update
    """
    # Check last message for keywords indicating preference changes
    last_message = state["messages"][-1].content.lower()

    decision = UpdateDecision()

    # Check for profile updates (name, identity info)
    if any(
        keyword in last_message
        for keyword in ["my name is", "i am from", "i work as", "i'm a"]
    ):
        decision.update_profile = True

    # Check for preference updates (response style, feedback)
    if any(
        keyword in last_message
        for keyword in ["prefer", "like", "don't like", "want", "please"]
    ):
        decision.update_preferences = True

    # Check for instruction updates (how to do things)
    if any(
        keyword in last_message for keyword in ["always", "never", "when", "how to"]
    ):
        decision.update_instructions = True

    return {"update_decision": decision}


# ============= EXPORTS =============

__all__ = [
    "load_manager_memories",
    "update_profile_memory",
    "update_preferences_memory",
    "update_instructions_memory",
    "decide_what_to_update",
    "store",
    "Profile",
    "UserPreference",
    "SystemInstruction",
]
