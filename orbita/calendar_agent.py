"""
Calendar Agent - ORBITA Multi-Agent System
Complete implementation with Google Calendar API integration.

This agent handles calendar operations:
- Reading events from Google Calendar
- Scheduling events with conflict detection
- Finding free time slots
- Summarizing calendar usage

REQUIREMENTS:
1. Google OAuth credentials (client_secret.json)
2. OAuth token file (token.json) - generated on first run
3. Google Calendar API enabled in Google Cloud Console
"""

from typing import Annotated, Sequence, TypedDict, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import os
import uuid

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

# Google Calendar API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pytz

# Configure
load_dotenv()

# ==================== CONFIGURATION ====================

# LLM Setup
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=512,
)

# Google Calendar Configuration
GOOGLE_CLIENT_SECRET_FILE = "client_secret.json"
GOOGLE_TOKEN_FILE = "token.json"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]
DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"

llm = ChatHuggingFace(llm=llm_endpoint)

# ==================== TIME UTILITIES ====================


def get_timezone(tz_name: str = DEFAULT_TIMEZONE):
    """Get timezone object"""
    return pytz.timezone(tz_name)


def now_tz(timezone: str = DEFAULT_TIMEZONE) -> datetime:
    """Get current time with timezone"""
    return datetime.now(get_timezone(timezone))


def to_iso(dt: datetime) -> str:
    """Convert datetime to ISO string"""
    return dt.isoformat()


def from_iso(iso_str: str) -> datetime:
    """Parse ISO string to datetime"""
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def get_today_range(timezone: str = DEFAULT_TIMEZONE) -> Tuple[datetime, datetime]:
    """Start and end of today"""
    tz = get_timezone(timezone)
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start, end


def get_week_range(timezone: str = DEFAULT_TIMEZONE) -> Tuple[datetime, datetime]:
    """Start and end of this week (Mon-Sun)"""
    tz = get_timezone(timezone)
    now = datetime.now(tz)
    start = now - timedelta(days=now.weekday())
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start, end


def get_month_range(timezone: str = DEFAULT_TIMEZONE) -> Tuple[datetime, datetime]:
    """Start and end of this month"""
    tz = get_timezone(timezone)
    now = datetime.now(tz)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if now.month == 12:
        end = now.replace(year=now.year + 1, month=1, day=1)
    else:
        end = now.replace(month=now.month + 1, day=1)
    end = end - timedelta(microseconds=1)

    return start, end


# ==================== GOOGLE CALENDAR CLIENT ====================


class GoogleCalendarClient:
    """Google Calendar API wrapper"""

    def __init__(self):
        self.creds = None
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google Calendar API"""
        # Load existing token
        if os.path.exists(GOOGLE_TOKEN_FILE):
            with open(GOOGLE_TOKEN_FILE, "rb") as token:
                self.creds = pickle.load(token)

        # Refresh or get new credentials
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    GOOGLE_CLIENT_SECRET_FILE, GOOGLE_SCOPES
                )
                self.creds = flow.run_local_server(port=0)

            # Save token
            with open(GOOGLE_TOKEN_FILE, "wb") as token:
                pickle.dump(self.creds, token)

        # Build service
        self.service = build("calendar", "v3", credentials=self.creds)

    def get_events(
        self,
        calendar_id: str = "primary",
        time_min: datetime = None,
        time_max: datetime = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get calendar events within time range"""
        try:
            time_min = time_min or now_tz()
            time_max = time_max or (time_min + timedelta(days=30))

            events_result = (
                self.service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat() + "Z",
                    timeMax=time_max.isoformat() + "Z",
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = []
            for event in events_result.get("items", []):
                events.append(
                    {
                        "id": event["id"],
                        "summary": event.get("summary", "No title"),
                        "description": event.get("description", ""),
                        "start": event["start"].get(
                            "dateTime", event["start"].get("date")
                        ),
                        "end": event["end"].get("dateTime", event["end"].get("date")),
                        "status": event.get("status", "confirmed"),
                        "location": event.get("location", ""),
                        "attendees": len(event.get("attendees", [])),
                    }
                )

            return events

        except Exception as e:
            print(f"Error getting events: {e}")
            return []

    def create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: str = None,
        location: str = None,
        attendees: List[str] = None,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> Dict[str, Any]:
        """Create a new calendar event"""
        try:
            event = {
                "summary": summary,
                "start": {
                    "dateTime": start_time.isoformat(),
                    "timeZone": timezone,
                },
                "end": {
                    "dateTime": end_time.isoformat(),
                    "timeZone": timezone,
                },
            }

            if description:
                event["description"] = description
            if location:
                event["location"] = location
            if attendees:
                event["attendees"] = [{"email": email} for email in attendees]

            created_event = (
                self.service.events().insert(calendarId="primary", body=event).execute()
            )

            return {
                "id": created_event["id"],
                "summary": created_event["summary"],
                "start": created_event["start"]["dateTime"],
                "end": created_event["end"]["dateTime"],
                "status": "confirmed",
            }

        except Exception as e:
            return {"error": str(e)}


# Create global client instance
calendar_client = GoogleCalendarClient()

# ==================== CALENDAR TOOLS ====================


@tool
def get_calendar_events_tool(
    period: str = "today", max_results: int = 20
) -> List[Dict[str, Any]]:
    """
    Get calendar events for a specific period.

    Args:
        period: 'today', 'week', or 'month'
        max_results: Maximum number of events to return
    """
    try:
        if period == "today":
            start_dt, end_dt = get_today_range()
        elif period == "week":
            start_dt, end_dt = get_week_range()
        elif period == "month":
            start_dt, end_dt = get_month_range()
        else:
            start_dt, end_dt = get_today_range()

        events = calendar_client.get_events(
            time_min=start_dt, time_max=end_dt, max_results=max_results
        )

        return events if events else [f"No events found for {period}"]

    except Exception as e:
        return [{"error": f"Failed to get events: {str(e)}"}]


@tool
def schedule_event_tool(
    title: str,
    start_datetime: str,
    duration_minutes: int = 60,
    description: str = "",
    location: str = "",
) -> Dict[str, Any]:
    """
    Schedule a new calendar event.

    Args:
        title: Event title
        start_datetime: Start time in format "YYYY-MM-DD HH:MM[:SS]"
        duration_minutes: Event duration in minutes
        description: Optional event description
        location: Optional location
    """
    try:
        # Parse start time
        start_time = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M")
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Check for conflicts (events within 30 minutes)
        existing_events = calendar_client.get_events(
            time_min=start_time - timedelta(minutes=30),
            time_max=end_time + timedelta(minutes=30),
        )

        if existing_events and not any("error" in e for e in existing_events):
            return {
                "warning": f"Time conflict detected! You have {len(existing_events)} event(s) near this time.",
                "existing_events": existing_events,
                "suggestion": "Please choose a different time or confirm you want to reschedule.",
            }

        # Create event
        result = calendar_client.create_event(
            summary=title,
            start_time=start_time,
            end_time=end_time,
            description=description,
            location=location,
        )

        if "error" in result:
            return {"error": result["error"]}

        return {
            "success": True,
            "message": f"✅ Event scheduled: '{title}'",
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_minutes": duration_minutes,
        }

    except Exception as e:
        return {"error": f"Failed to schedule event: {str(e)}"}


@tool
def find_free_slots_tool(
    date: str, duration_minutes: int = 60, timezone: str = DEFAULT_TIMEZONE
) -> List[Dict[str, Any]]:
    """
    Find free time slots on a specific date.

    Args:
        date: Date in YYYY-MM-DD format
        duration_minutes: Minimum slot duration needed
        timezone: Timezone for the search
    """
    try:
        # Parse the date
        target_date = datetime.strptime(date, "%Y-%m-%d")
        start_dt = target_date.replace(hour=0, minute=0, second=0)
        end_dt = target_date.replace(hour=23, minute=59, second=59)

        # Get existing events for the day
        events = calendar_client.get_events(
            time_min=start_dt, time_max=end_dt, max_results=50
        )

        # If error in events, return empty or error
        if events and any("error" in e for e in events):
            return [{"error": "Could not retrieve events"}]

        # Parse event times
        busy_slots = []
        for event in events:
            if "start" in event and "end" in event:
                try:
                    busy_slots.append(
                        {
                            "start": from_iso(event["start"]),
                            "end": from_iso(event["end"]),
                        }
                    )
                except:
                    pass

        # Find free slots
        free_slots = []
        current_time = start_dt
        slot_duration = timedelta(minutes=duration_minutes)
        work_start = start_dt.replace(hour=8, minute=0)  # 8 AM
        work_end = start_dt.replace(hour=18, minute=0)  # 6 PM

        # Adjust to work hours if needed
        if current_time < work_start:
            current_time = work_start

        # Sort busy slots
        busy_slots.sort(key=lambda x: x["start"])

        # Find gaps between busy slots
        for busy in busy_slots:
            # Find slot before this busy period
            while current_time + slot_duration <= busy["start"]:
                if current_time >= work_start:
                    free_slots.append(
                        {
                            "start": to_iso(current_time),
                            "end": to_iso(current_time + slot_duration),
                            "duration_minutes": duration_minutes,
                        }
                    )
                current_time += timedelta(minutes=30)  # Check every 30 minutes

            # Move past this busy period
            current_time = max(current_time, busy["end"])

        # Check remaining time after last event
        while current_time + slot_duration <= work_end:
            free_slots.append(
                {
                    "start": to_iso(current_time),
                    "end": to_iso(current_time + slot_duration),
                    "duration_minutes": duration_minutes,
                }
            )
            current_time += timedelta(minutes=30)

        return (
            free_slots
            if free_slots
            else [{"message": f"No free slots found on {date}"}]
        )

    except Exception as e:
        return [{"error": f"Could not find free slots: {str(e)}"}]


@tool
def summarize_calendar_tool(period: str = "week") -> Dict[str, Any]:
    """
    Summarize calendar usage for a period.

    Args:
        period: 'today', 'week', or 'month'
    """
    try:
        if period == "today":
            start_dt, end_dt = get_today_range()
            total_minutes = 24 * 60
        elif period == "week":
            start_dt, end_dt = get_week_range()
            total_minutes = 7 * 24 * 60
        elif period == "month":
            start_dt, end_dt = get_month_range()
            days = (end_dt - start_dt).days + 1
            total_minutes = days * 24 * 60
        else:
            return {"error": "Invalid period. Use 'today', 'week', or 'month'."}

        events = get_calendar_events_tool(period=period, max_results=100)

        if isinstance(events, list) and len(events) > 0:
            if "error" in events[0]:
                return events[0]

        total_events = len(events) if isinstance(events, list) else 0
        total_duration = 0

        for event in events:
            if isinstance(event, dict) and "start" in event and "end" in event:
                try:
                    start_time = from_iso(event["start"])
                    end_time = from_iso(event["end"])
                    duration = (end_time - start_time).total_seconds() / 60
                    total_duration += duration
                except:
                    pass

        busy_percentage = (
            (total_duration / total_minutes * 100) if total_minutes > 0 else 0
        )

        return {
            "success": True,
            "period": period,
            "summary": {
                "total_events": total_events,
                "busy_hours": round(total_duration / 60, 1),
                "busy_percentage": round(busy_percentage, 1),
                "free_hours": round((total_minutes - total_duration) / 60, 1),
            },
        }

    except Exception as e:
        return {"error": f"Could not summarize calendar: {str(e)}"}


# ==================== CALENDAR AGENT ====================


class CalendarAgentState(TypedDict):
    calendar_data: Dict
    messages: Annotated[Sequence[BaseMessage], add_messages]


# System prompt
calendar_agent_prompt = """
You are the Calendar Agent, a specialized sub-agent within the ORBITA multi-agent system.

Your role:
- You report to the manager agent named ORBITA
- You handle ALL calendar-related tasks including reading events, scheduling, and calendar management
- You work collaboratively with other specialized agents (Email Agent, Budget Agent, etc.)
- You have access to calendar tools: get_calendar_events_tool, schedule_event_tool, find_free_slots_tool, summarize_calendar_tool

When processing requests:
1. Use the provided tools to perform calendar operations
2. Focus exclusively on calendar tasks - let other agents handle their specialties
3. Provide clear, actionable responses with event details (title, datetime, duration)
4. Check for scheduling conflicts before creating events
5. Report back to ORBITA when tasks are complete

Remember: You are part of a larger system. Stay focused on calendar management.
"""

# Tools list
calendar_tools = [
    get_calendar_events_tool,
    schedule_event_tool,
    find_free_slots_tool,
    summarize_calendar_tool,
]

# LLM with tools
llm = ChatHuggingFace(llm=llm_endpoint).bind_tools(calendar_tools)


def model_call(state: CalendarAgentState) -> CalendarAgentState:
    """Process messages with LLM"""
    system_prompt = SystemMessage(content=calendar_agent_prompt)
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: CalendarAgentState) -> str:
    """Determine next step"""
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def build_calendar_agent():
    """Build the calendar agent graph"""
    graph = StateGraph(CalendarAgentState)

    graph.add_node("llm", model_call)
    tool_node = ToolNode(calendar_tools)
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


# Export the agent
calendar_agent = build_calendar_agent()

if __name__ == "__main__":
    """Example usage and testing"""
    print("🗓️ Testing Calendar Agent...")
    print("=" * 50)

    # Initialize authentication on first run
    print("\n🔐 Authenticating with Google Calendar...")
    try:
        # This will trigger OAuth flow on first run
        calendar_client.service
        print("✅ Authentication successful!")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\nPlease ensure client_secret.json is in this directory")
        print("Visit: https://console.cloud.google.com/apis/credentials")
        exit()

    # Test the agent
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "calendar_test"}}

    # Test 1: Get today's events
    print("\n📅 Test 1: Getting today's events")
    test1 = HumanMessage(content="What events do I have today?")
    result1 = calendar_agent.invoke(
        {"messages": [test1], "calendar_data": {}}, config=config
    )
    print(f"Result: {result1['messages'][-1].content}")

    # Test 2: Schedule an event
    print("\n➕ Test 2: Scheduling an event")
    test2 = HumanMessage(
        content="Schedule a meeting 'Team Sync' tomorrow at 2 PM for 1 hour"
    )
    result2 = calendar_agent.invoke(
        {"messages": [test1, result1["messages"][-1], test2], "calendar_data": {}},
        config=config,
    )
    print(f"Result: {result2['messages'][-1].content}")

    print("\n✅ Calendar agent tests completed!")
    print("\n💡 Next steps:")
    print("1. Get client_secret.json from Google Cloud Console")
    print("2. Place it in this directory")
    print("3. Run: python calendar_agent.py")
    print("4. Authenticate when prompted")
    print("5. token.json will be created automatically")
