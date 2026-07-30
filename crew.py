#!/usr/bin/env python3
"""
Fastmail Crew AI Agent - Corrected MCP Integration
"""

import os
import uuid
from typing import cast
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.mcp import MCPServerHTTP
from crewai.flow import Flow, listen
from crewai.experimental import ConversationConfig, RouterConfig, ConversationState
from crewai.flow.persistence import SQLiteFlowPersistence, persist

# Monkey-patch to fix cache_breakpoint issue with Groq
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

# Monkey-patch: crewai always marks native tool schemas as OpenAI "strict"
# mode (crewai/utilities/agent_utils.py::convert_tools_to_openai_schema hardcodes
# "strict": True). Several Fastmail MCP tools (draft_email, create_event,
# update_event, compose_event, list_calendars, ...) declare "open object"
# parameters (e.g. a list of arbitrary {name, email} dicts) whose JSON schema
# fails Groq's strict-mode validator ("'required' present but 'properties' is
# missing"). Because every tool's schema is validated together in a single
# request, one bad schema breaks *every* query regardless of what it asks for.
# Groq's normal (non-strict) function calling doesn't require this level of
# schema conformance, so just turn strict mode off for all tool schemas.
import crewai.utilities.agent_utils as _agent_utils
_original_convert_tools_to_openai_schema = _agent_utils.convert_tools_to_openai_schema

def _convert_tools_to_openai_schema_non_strict(tools):
    openai_tools, available_functions, tool_name_mapping = _original_convert_tools_to_openai_schema(tools)
    for schema in openai_tools:
        schema.get("function", {})["strict"] = False
    return openai_tools, available_functions, tool_name_mapping

_agent_utils.convert_tools_to_openai_schema = _convert_tools_to_openai_schema_non_strict

def load_environment():
    """Load environment variables from .env file"""
    if os.path.exists('.env'):
        load_dotenv()
        print("Environment variables loaded from .env file")
        return True
    else:
        print("Warning: No .env file found")

load_environment()

# Get API keys from environment
FASTMAIL_API_TOKEN = os.getenv('FASTMAIL_API_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in environment")
    exit(1)

if not FASTMAIL_API_TOKEN:
    print("ERROR: FASTMAIL_API_TOKEN not found in environment")
    exit(1)

# Configure Groq LLM.
# NOTE: small models like llama-3.1-8b-instant are not reliable at native
# tool calling once there are 20+ tools with rich descriptions (observed:
# it emits a hallucinated "<function=...>" text block instead of a real
# tool call, which Groq then rejects). llama-3.3-70b-versatile reliably
# issues proper native tool calls against the Fastmail MCP tools.
groq_llm = LLM(
    # model="groq/llama-3.3-70b-versatile",
    model="groq/openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0,
)

# Fastmail MCP server configuration (native CrewAI MCP support)
fastmail_mcp_server = MCPServerHTTP(
    url="https://api.fastmail.com/mcp",
    headers={
        "Authorization": f"Bearer {FASTMAIL_API_TOKEN}"
    },
    streamable=True,
)

# Define the Fastmail expert agent
fastmail_expert = Agent(
    role='Fastmail Assistant',
    goal='Help users interact directly with their Fastmail account using real MCP tools',
    backstory="""You are an AI assistant with direct access to the Fastmail MCP server. 
    You can perform real operations on emails, contacts, and calendar.""",
    llm=groq_llm,
    verbose=True,
    allow_delegation=False,
)

# IMPORTANT: passing mcps=[...] to Agent() has no effect when the agent is run
# via Crew().kickoff() (only Agent.kickoff() consumes the `mcps` field). That
# silently left the agent with zero tools, causing the LLM to hallucinate
# answers instead of calling the real Fastmail MCP server. Resolve the MCP
# tools ourselves and attach them directly to `agent.tools` instead.
print("Connecting to Fastmail MCP server and discovering tools...")
fastmail_expert.tools = fastmail_expert.get_mcp_tools([fastmail_mcp_server])
if not fastmail_expert.tools:
    print("ERROR: No tools discovered from the Fastmail MCP server. "
          "Check FASTMAIL_API_TOKEN and network connectivity.")
    exit(1)
print(f"Discovered {len(fastmail_expert.tools)} Fastmail MCP tools.")

_DB_PATH = "fastmail_chat_state.db"
_SESSION_ID_FILE = ".fastmail_session_id"


def get_or_create_session_id() -> str:
    """Reads a persisted session UUID from disk, or creates and saves a fresh one."""
    if os.path.exists(_SESSION_ID_FILE):
        with open(_SESSION_ID_FILE) as f:
            sid = f.read().strip()
            if sid:
                return sid
    sid = str(uuid.uuid4())
    with open(_SESSION_ID_FILE, "w") as f:
        f.write(sid)
    return sid


_session_id = get_or_create_session_id()
_persistence = SQLiteFlowPersistence(db_path=_DB_PATH)


@persist(_persistence)
class FastmailConversation(Flow):
    # Experimental conversational Flow API — pin crewai==1.15.8 before upgrading
    conversational = True
    conversational_config = ConversationConfig(
        llm=groq_llm,
        answer_from_history_llm=groq_llm,
        router=RouterConfig(
            route_descriptions={
                "fastmail_action": (
                    "User wants to perform or check something in their real Fastmail "
                    "account: search, read, move, or send email; manage calendar "
                    "events or contacts. Requires calling Fastmail MCP tools."
                )
            }
        ),
    )

    @listen("fastmail_action")
    def run_fastmail_action(self) -> str:
        state = cast(ConversationState, self.state)
        user_message = state.current_user_message or ""

        # Build a short prior-turn transcript so the agent has conversational context
        recent = self.conversation_messages
        history_lines = [
            f"{m['role'].capitalize()}: {m.get('content', '')}"
            for m in recent[:-1]
        ]
        context_section = (
            "\nConversation context:\n" + "\n".join(history_lines[-8:]) + "\n"
            if history_lines
            else ""
        )

        task = Task(
            description=(
                f'Execute the user\'s request on their Fastmail account: "{user_message}"'
                f"{context_section}\n"
                "Use the available Fastmail MCP tools to perform the actual operations."
            ),
            agent=fastmail_expert,
            expected_output="The actual results from performing the requested operations",
        )
        crew = Crew(agents=[fastmail_expert], tasks=[task], verbose=True)

        try:
            result = crew.kickoff()
        except Exception as exc:
            result = f"Error: {exc}"

        result_str = str(result)
        self.append_agent_result("fastmail_expert", result_str, visibility="public")
        return result_str


conversation = FastmailConversation()


def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("Fastmail Crew AI Assistant - Real MCP Integration")
    print("=" * 60)
    print("Commands:")
    print("  quit, exit, q  - Exit the chat")
    print("  help, h        - Show this help message")
    print("  config         - Show current configuration")
    print("-" * 60)
    print("Enter your queries about your Fastmail account...")
    print("=" * 60)

def show_config():
    """Show current configuration"""
    print("\nCurrent Configuration:")
    print("-" * 30)
    print(f"Groq API Key: {'SET' if GROQ_API_KEY else 'NOT SET'}")
    print(f"Fastmail API Token: {'SET' if FASTMAIL_API_TOKEN else 'NOT SET'}")
    print(f"Fastmail MCP URL: {fastmail_mcp_server.url}")
    print(f"Streamable HTTP: {fastmail_mcp_server.streamable}")
    print(f"MCP tools attached to agent: {len(fastmail_expert.tools or [])}")
    print(f"Session ID: {_session_id}")
    print(f"State DB: {os.path.abspath(_DB_PATH)}")

def main():
    """Main chat loop"""
    print("Fastmail Crew AI Assistant Initializing...")
    print_welcome()

    while True:
        try:
            user_input = input("\n📧 Fastmail Crew AI > ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() in ['help', 'h']:
                print_welcome()
                continue
            elif user_input.lower() == 'config':
                show_config()
                continue
            elif user_input == "":
                continue

            print("Processing your request... 🤖")

            try:
                result = conversation.handle_turn(user_input, session_id=_session_id)
                print("\n" + "=" * 40)
                print("Response:")
                print("=" * 40)
                print(result)
                print("=" * 40)
            except Exception as e:
                print(f"\nError: {e}")

        except KeyboardInterrupt:
            print("\n\nExiting... 👋")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Check packages
    try:
        import crewai  # noqa: F401
    except ImportError:
        print("Error: Required packages not found.")
        print("Install with: pip install crewai")
        exit(1)

    main()
