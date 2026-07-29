#!/usr/bin/env python3
"""
Fastmail Crew AI Agent
Using Crew AI to interact with Fastmail via MCP
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import json

def load_environment():
    """Load environment variables from .env file"""
    if os.path.exists('.env'):
        load_dotenv()
        print("Environment variables loaded from .env file")
        return True
    else:
        print("Warning: No .env file found, using system environment variables")
        return False

# Load environment variables
load_environment()

# Get API keys from environment
FASTMAIL_API_TOKEN = os.getenv('FASTMAIL_API_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Validate that we have the required API keys
if not GROQ_API_KEY or GROQ_API_KEY == "":
    print("ERROR: GROQ_API_KEY not found or empty in environment variables")
    print("Please ensure your .env file contains: GROQ_API_KEY=your_actual_api_key_here")
    exit(1)

# Set up Groq integration
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_API_MODEL"] = "llama3-8b-8192"

# Define the Fastmail expert agent
fastmail_expert = Agent(
    role='Fastmail Assistant',
    goal='Help users manage their Fastmail account through natural language commands',
    backstory="""You are an AI assistant specialized in helping users interact with their Fastmail account. 
    You can help with emails, contacts, and calendar management. You translate natural language requests 
    into specific actions that can be performed on the user's Fastmail account through the MCP protocol.""",
    verbose=True,
    allow_delegation=False
)

def create_fastmail_task(user_query):
    """Create a task for the Fastmail agent"""
    task = Task(
        description=f"""
        Process the user's request related to their Fastmail account: "{user_query}"

        Since we don't have direct access to the Fastmail MCP server in this environment,
        please provide a detailed explanation of what actions would need to be taken
        to fulfill this request, including:

        1. What specific Fastmail features would be involved (email, contacts, calendar)
        2. What information would need to be retrieved or sent
        3. What the expected response would be

        Provide your response in a clear, structured format.
        """,
        agent=fastmail_expert,
        expected_output="A detailed explanation of how to fulfill the user's request with Fastmail MCP"
    )
    return task

def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("Fastmail Crew AI Assistant")
    print("=" * 60)
    print("This Crew AI implementation provides guidance on how to interact")
    print("with your Fastmail account via MCP.")
    print("Commands:")
    print("  quit, exit, q  - Exit the chat")
    print("  help, h        - Show this help message")
    print("  clear, cls     - Clear the screen")
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
    print(f"API Base URL: {os.environ.get('OPENAI_API_BASE', 'NOT SET')}")
    print(f"API Model: {os.environ.get('OPENAI_API_MODEL', 'NOT SET')}")

def main():
    """Main chat loop"""
    print("Fastmail Crew AI Assistant Initializing...")

    # Print welcome message
    print_welcome()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n📧 Fastmail Crew AI > ").strip()

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() in ['help', 'h']:
                print_welcome()
                continue
            elif user_input.lower() in ['clear', 'cls']:
                # Clear screen (works on most terminals)
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
                continue
            elif user_input.lower() == 'config':
                show_config()
                continue
            elif user_input == "":
                # Skip empty input
                continue

            # Create and run the task
            print("Processing your request with Crew AI... 🤖")

            task = create_fastmail_task(user_input)
            crew = Crew(
                agents=[fastmail_expert],
                tasks=[task],
                verbose=True
            )

            # Execute the crew
            try:
                result = crew.kickoff()

                print("\n" + "=" * 40)
                print("Response:")
                print("=" * 40)
                print(result)
                print("=" * 40)
            except Exception as e:
                print(f"\nError executing Crew AI: {e}")
                print("This could be due to:")
                print("1. Incorrect API key - check your .env file")
                print("2. Network connectivity issues")
                print("3. API access restrictions")
                print("\nTry checking your configuration with the 'config' command")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting... 👋")
            break
        except EOFError:
            print("\n\nEOF received. Exiting... 👋")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import crewai
    except ImportError:
        print("Error: Crew AI not found. Please install it with: pip install crewai")
        exit(1)

    main()
