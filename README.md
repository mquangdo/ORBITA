# ORBITA - Multi-Agent AI Assistant

A personalized intelligent assistant built with LangGraph, featuring specialized agents for email management, calendar operations, and budget tracking.

![](image01.png)

## 🏗️ Architecture

```
User Input → Manager Agent → Router → Specialized Agents
                                 ↓
                        [Email | Calendar | Budget]
                                 ↓
                        Memory System
```

- **Manager Agent**: Orchestrates the overall system and routes requests
- **Email Agent**: Handles email operations
- **Calendar Agent**: Manages calendar events (Google Calendar integration)
- **Budget Agent**: Tracks and manages budgets

## 📋 Requirements

- Python 3.8+
- NVIDIA AI API access or OpenAI API access
- Google Calendar API credentials (for calendar agent)
- Required Python packages (see setup below)

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd orbita
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file with:
   ```env
   NVIDIA_API_KEY=your_nvidia_api_key
   # or
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_CALENDAR_CREDENTIALS_PATH=client_secret.json
   ```

5. **Set Up Google Calendar API**
   
   Ensure you have:
   - Google Cloud Platform project with Calendar API enabled
   - OAuth 2.0 credentials file downloaded as `client_secret.json`
   - Authorized credentials stored

## 🎯 Usage

### Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

The web interface provides a ChatGPT-like experience with:
- Clean, modern UI
- Real-time message streaming

Interactive CLI with:
- Direct conversation with ORBITA
- Type `exit` or `quit` to end the session

## 🔧 Configuration

### Model Configuration

The system uses NVIDIA's `openai/gpt-oss-120b` model by default. Configure in `manager_agent.py`:

```python
llm = ChatNVIDIA(model="openai/gpt-oss-120b", temperature=0.2)
```

### Memory System

The memory system stores:
- **Profile**: Personal information about the user
- **Preferences**: User likes, dislikes, and habits  
- **Instructions**: Guidelines for agent behavior

Memory is persisted and used to personalize responses across conversations.

## 📁 Project Structure

```
orbita/
├── main.py                    # CLI entry point
├── streamlit_app.py          # Web UI
├── manager_agent.py          # Main orchestration logic
├── manager_memory.py         # Memory management
├── calendar_agent.py         # Calendar operations
├── email_agent.py            # Email management
├── budget_agent.py           # Budget tracking
├── tools.py                  # Shared tools/utilities
├── client_secret.json        # Google API credentials
├── .env                      # Environment variables
└── requirements.txt         # Python dependencies
```

## 🧠 Agent Capabilities

### Manager Agent
- Routes user requests to appropriate specialized agents
- Manages conversation flow and state
- Applies memory context to personalize responses
- Deterministic task routing based on user intent

### Email Agent
- Send and manage emails
- Process email-related queries
- Email composition and scheduling

### Calendar Agent
- Create, read, update, delete calendar events
- Google Calendar integration
- Event scheduling and management

### Budget Agent
- Track expenses and income
- Budget planning and analysis
- Financial insights and reports

## 🔍 Observability

The system integrates with **Opik** for:
- Conversation tracing
- Agent performance monitoring
- Debug insights
- Project: `ORBITA`

Configure in the code:
```python
configure()
tracer = OpikTracer(graph=manager_agent.get_graph(xray=True), project_name='ORBITA')
```

## 🧪 Development

### Linting

The project uses Ruff for linting:
```bash
ruff check .
```

### Testing

Run tests after implementing code changes:
```bash
python -m pytest tests/
```


## 📄 License

This project is part of the ORBITA personal assistant suite.
