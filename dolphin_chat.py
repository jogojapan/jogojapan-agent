#!/usr/bin/env python3
"""
Dolphin-MCP Chat Environment
A simple interactive shell for chatting with your Fastmail account via Dolphin-MCP
"""

import os
import sys
import subprocess
import json
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    if os.path.exists('.env'):
        load_dotenv()
        print("Environment variables loaded from .env file")
        return True
    else:
        print("Warning: No .env file found, using system environment variables")
        return False

def validate_config():
    """Validate that the config file exists and is properly formatted"""
    if not os.path.exists('mcp_config.json'):
        print("Error: mcp_config.json not found in current directory")
        return False

    try:
        with open('mcp_config.json', 'r') as f:
            config = json.load(f)

        # Basic validation
        if 'mcp_servers' not in config:
            print("Error: 'mcp_servers' not found in config file")
            return False

        if 'models' not in config:
            print("Error: 'models' not found in config file")
            return False

        print("Configuration file validated successfully")
        return True
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in mcp_config.json: {e}")
        return False
    except Exception as e:
        print(f"Error validating config file: {e}")
        return False

def run_dolphin_query(query):
    """Run a query through Dolphin-MCP"""
    try:
        # Prepare the command
        cmd = [
            "dolphin-mcp-cli",
            "--mcp-config", "mcp_config.json",
            query
        ]

        print(f"Executing: {' '.join(cmd)}")

        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            return result.stdout
        else:
            error_msg = result.stderr
            # Try to parse Python traceback for better error reporting
            if "Traceback" in error_msg:
                lines = error_msg.split('\n')
                # Get the last few lines of the traceback for context
                traceback_lines = []
                in_traceback = False
                for line in reversed(lines):
                    if "Traceback" in line:
                        in_traceback = True
                        traceback_lines.append(line)
                        break
                    if in_traceback:
                        traceback_lines.append(line)

                if traceback_lines:
                    error_msg = "Python Error:\n" + "\n".join(reversed(traceback_lines))

            return f"Command Error: {error_msg}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except FileNotFoundError:
        return "Error: dolphin-mcp-cli not found. Please ensure it's installed and in your PATH."
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("Dolphin-MCP Chat Environment")
    print("=" * 60)
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
    try:
        if os.path.exists('mcp_config.json'):
            with open('mcp_config.json', 'r') as f:
                config = json.load(f)

            print("\nCurrent Configuration:")
            print("-" * 30)
            print(json.dumps(config, indent=2))
        else:
            print("No configuration file found.")
    except Exception as e:
        print(f"Error reading configuration: {e}")

def main():
    """Main chat loop"""
    print("Dolphin-MCP Chat Environment Initializing...")

    # Load environment variables
    load_environment()

    # Validate configuration
    if not validate_config():
        print("Configuration validation failed. Please check your mcp_config.json file.")
        return

    # Print welcome message
    print_welcome()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n📧 Dolphin-MCP > ").strip()

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

            # Run the query
            print("Processing your request... 🤔")
            response = run_dolphin_query(user_input)
            print("\n" + "=" * 40)
            print("Response:")
            print("=" * 40)
            print(response)
            print("=" * 40)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting... 👋")
            break
        except EOFError:
            print("\n\nEOF received. Exiting... 👋")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")

if __name__ == "__main__":
    main()
