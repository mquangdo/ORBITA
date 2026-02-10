# ORBITA - Multi-Agent AI Assistant

A personalized intelligent assistant built with LangGraph, featuring specialized agents for email management, calendar operations, and budget tracking.

## 🚀 Features

- **Multi-Agent System**: Specialized agents for Email, Calendar, and Budget management
- **Intelligent Manager**: Routes tasks to appropriate specialized agents
- **Memory System**: Personalizes responses based on user profile, preferences, and instructions
- **Dual Interface**: Streamlit web UI and command-line interface
- **ChatGPT-like Design**: Modern, responsive web interface
- **Observability**: Integrated with Opik for tracing and monitoring

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

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

The web interface provides a ChatGPT-like experience with:
- Clean, modern UI
- Real-time message streaming
- Sidebar with project information
- Responsive design

### Option 2: Command Line Interface

```bash
python main.py
```

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

## 🚦 Troubleshooting

### Common Issues

1. **NVIDIA API Key Missing**
   - Ensure `.env` file exists with valid API key
   - Verify the key has access to selected model

2. **Google Calendar Authentication**
   - Check `client_secret.json` is valid
   - Re-authorize if token expired
   - Verify Calendar API is enabled in GCP

3. **Streamlit UI Not Loading**
   - Check port 8501 is available
   - Verify all dependencies installed
   - Review browser console for errors

4. **Unicode Errors**
   - The system is configured for UTF-8
   - Ensure terminal supports UTF-8
   - Windows: Check console encoding settings

## 📄 License

This project is part of the ORBITA personal assistant suite.

## 🤝 Contributing

This is a personal project. For questions or feedback, please open an issue on GitHub.

## 📝 Version History

- v1.0 - Initial release with Manager, Email, Calendar, and Budget agents
- v1.1 - Added Streamlit UI with ChatGPT-like interface
- v1.2 - Integrated memory system for personalization

---

**ORBITA** - "Omni-Responsive Bot for Intelligent Task Automation" 🚀
