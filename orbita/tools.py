import imaplib
import email
import smtplib
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain.tools import tool
from typing import Dict, TypedDict, Annotated, List
import requests
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

"""Email Agent section"""
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
APP_PASSWORD = os.getenv("APP_PASSWORD")


@tool
def fetch_emails_tool(k: int, target_email: str = None) -> list:
    """
    Read the content of the latest k emails. Can filter by sender's email address.
    Args:
        k: Number of emails to read.
        target_email: Sender's email address to filter (if any).
    """
    emails_found = []
    try:
        imap = imaplib.IMAP4_SSL(IMAP_SERVER, 993)
        imap.login(EMAIL_ADDRESS, APP_PASSWORD)
        imap.select("INBOX")

        if target_email:
            search_query = f'FROM "{target_email}"'
        else:
            search_query = "ALL"

        status, messages = imap.search(None, search_query)

        if status == "OK" and messages[0]:
            ids = messages[0].split()[-k:][::-1]

            for mail_id in ids:
                _, msg_data = imap.fetch(mail_id, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])

                subject, encoding = decode_header(msg.get("Subject", ""))[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))

                        if (
                            content_type == "text/plain"
                            and "attachment" not in content_disposition
                        ):
                            payload = part.get_payload(decode=True)
                            body = payload.decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")

                emails_found.append(
                    {
                        "from": msg.get("From"),
                        "subject": subject,
                        "content": body.strip(),
                        "id": mail_id.decode(),
                    }
                )

        imap.logout()
    except Exception as e:
        return [f"Error reading emails: {str(e)}"]

    return emails_found


@tool
def send_email_tool(to_email: str, subject: str, body: str) -> str:
    """
    Send an email to a specific address.
    Args:
        to_email: Recipient's email address.
        subject: Email subject.
        body: Email content.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(SMTP_SERVER, 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, APP_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, [to_email], msg.as_string())
        return f"✅ Successfully sent to {to_email}"
    except Exception as e:
        return f"❌ Error sending email: {str(e)}"


"""Calender Agent section"""
from datetime import datetime, timedelta
from typing import Tuple
import pytz

DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"


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


"""Budget/Hobby Agent section"""

API_TOKEN = os.getenv("SE_PAY_API_TOKEN")


@tool
def get_budget_tool(account_number: str) -> float:
    """
    Get the accumulated budget for a given account number.
    """

    BASE_URL = "https://my.sepay.vn/userapi/transactions/list"

    PARAMS = {
        "account_number": account_number,  # số tài khoản
        "transaction_date_min": "2026-01-01",  # ngày bắt đầu
        "transaction_date_max": "2026-01-31",  # ngày kết thúc
        "limit": 1000,
    }

    HEADERS = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.get(BASE_URL, headers=HEADERS, params=PARAMS)

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    data = response.json()
    transactions = data.get("transactions", [])

    accumulated = 0.0
    if transactions:
        accumulated = float(transactions[0]["amount_in"])

    return accumulated


"""Calendar Agent section"""

from datetime import datetime, timedelta
import json

# In-memory calendar storage (can be replaced with Google Calendar API)
_CALENDAR_EVENTS = []


def _generate_event_id():
    """Generate a unique event ID"""
    return f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


@tool
def get_calendar_events_tool(start_date: str, end_date: str = None) -> List[Dict]:
    """
    Get calendar events between start_date and end_date.
    If end_date is not provided, gets events for just start_date.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional)
    """
    try:
        start = datetime.fromisoformat(start_date)
        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = start + timedelta(days=1)

        events = []
        for event in _CALENDAR_EVENTS:
            event_time = datetime.fromisoformat(event["datetime"])
            if start <= event_time < end:
                events.append(event)

        # Sort by datetime
        events.sort(key=lambda x: x["datetime"])

        return (
            events
            if events
            else [f"No events found between {start_date} and {end_date or start_date}"]
        )

    except Exception as e:
        return [f"Error retrieving calendar events: {str(e)}"]


@tool
def schedule_event_tool(
    title: str, datetime_str: str, duration_minutes: int = 60, description: str = ""
) -> str:
    """
    Schedule a new calendar event.

    Args:
        title: Event title
        datetime_str: Event date and time in YYYY-MM-DD HH:MM format
        duration_minutes: Event duration in minutes (default: 60)
        description: Optional event description
    """
    try:
        # Validate datetime format
        event_time = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))

        # Check for conflicts (events within 30 minutes)
        conflict_window = timedelta(minutes=30)
        for existing in _CALENDAR_EVENTS:
            existing_time = datetime.fromisoformat(existing["datetime"])
            if (
                abs((event_time - existing_time).total_seconds())
                < conflict_window.total_seconds()
            ):
                if existing_time.date() == event_time.date():
                    return f"⚠️ Potential conflict: You have '{existing['title']}' scheduled close to this time."

        # Create event
        event = {
            "id": _generate_event_id(),
            "title": title,
            "datetime": event_time.isoformat(),
            "duration_minutes": duration_minutes,
            "description": description,
        }

        _CALENDAR_EVENTS.append(event)

        return f"✅ Event scheduled: '{title}' on {datetime_str} for {duration_minutes} minutes"

    except Exception as e:
        return f"❌ Error scheduling event: {str(e)}"


# @tool
# def hobby_node(hobby_list: List[str], budget: Dict) -> Dict:
#     """
#     Normalize budget constraints for planning & decision making
#     """
#     accumulated = budget.get("accumulated", 67.0)

#     budget_context = {
#         "currency": "VND",
#         "remaining_budget": accumulated
#     }
#     return {
#         "hobbies": hobby_list,
#         "budget_context": budget_context,
#     }
