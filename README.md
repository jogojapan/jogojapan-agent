# Crew AI Assistant — Fastmail + Nextcloud MCP Integration

A conversational AI assistant that connects to **Fastmail** and **Nextcloud** via MCP (Model Context Protocol) servers, letting you manage email, calendar, files, notes, and more through natural language — plus upload local files to Nextcloud.

## Capabilities

| Route | Agent | What it does |
|---|---|---|
| **Fastmail** | `fastmail_expert` | Search, read, move, send email; manage calendar events and contacts via the Fastmail MCP server |
| **Nextcloud** | `nextcloud_expert` | Manage notes, files (WebDAV), calendar, tasks, contacts, kanban (Deck), wiki (Collectives), recipes, news, mail, talk, tables via the Nextcloud Context Agent MCP server |
| **File Upload** | `file_upload_agent` | Browse the local filesystem (read-only), search for files, and upload any local file (including binary: images, PDFs, archives) to Nextcloud via WebDAV PUT |

A Groq-hosted LLM acts as a **router**, classifying each user message into one of the three routes. The selected agent then executes the request using its MCP tools.

### Key design decisions

- **Binary file uploads**: The Nextcloud MCP's `upload_file` tool only accepts text content. Our custom `upload_local_file` tool reads local files in binary mode and PUTs them directly to the Nextcloud WebDAV endpoint, preserving the correct MIME type.
- **Local filesystem access**: The `@modelcontextprotocol/server-filesystem` npm package provides read-only tools (list, search, read text/media files) for directories you configure in `allowed_dirs.yaml`.
- **Project-local Node.js**: Node.js is installed in `.nodeenv` (via `nodeenv`) — no system-wide install needed. The code auto-detects `.nodeenv/bin/npx` and falls back to system `npx`.
- **Tool count budget**: Groq has a 128-tool limit per call. Cookbook and news tools are dropped from the Nextcloud agent; the file upload agent carries a small curated set (~20 tools total).

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate.fish   # fish
# source .venv/bin/activate      # bash/zsh
pip install -r requirements.txt
```

### 2. Node.js environment (for filesystem MCP server)

```bash
# Install Node.js locally in the project (no sudo needed)
nodeenv --node=22.18.0 .nodeenv
source .nodeenv/bin/activate.fish   # fish
# source .nodeenv/bin/activate       # bash/zsh
```

The filesystem MCP server will be launched via `npx` from `.nodeenv/bin/npx` automatically. If you skip this step, the file upload agent is disabled gracefully — Fastmail and Nextcloud routes still work.

### 3. Environment variables

Copy `.env.example` (or edit `.env` directly) and fill in your credentials:

| Variable | Required | Description |
|---|---|---|
| `FASTMAIL_API_TOKEN` | ✅ | Fastmail API token |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `NC_USER` | ✅ | Nextcloud username |
| `NC_PW` | ✅ | Nextcloud app password |
| `NC_MCP` | ✅ | Nextcloud Context Agent MCP URL (e.g. `https://cloud.example.com/apps/context_agent/api/mcp`) |
| `NC_URL` | ✅ | Nextcloud base URL for WebDAV (e.g. `https://cloud.example.com`) — distinct from `NC_MCP` |

### 4. Allowed directories

Edit `allowed_dirs.yaml` to control which local directories the file upload agent can browse:

```yaml
directories:
  - "$HOME"
  - "/path/to/projects"
```

Environment variables (`$HOME`, `$USER`, etc.) are expanded at startup. Only directories that actually exist are included.

## Running

```bash
source .venv/bin/activate.fish
source .nodeenv/bin/activate.fish   # optional, if .nodeenv exists
python crew.py
```

### In-chat commands

| Command | Action |
|---|---|
| `quit`, `exit`, `q` | Exit |
| `help`, `h` | Show help |
| `config` | Show current configuration |

### Example prompts

- *"Find emails from Alice about the project timeline"*
- *"Create a new note in Nextcloud called Meeting Notes"*
- *"Upload ~/Documents/report.pdf to Nextcloud/Documents"*
- *"What files are in my home directory?"*
- *"Search for .jpg files in ~/Pictures"*

## Project structure

```
crew.py              Main application (agents, MCP connections, Flow, chat loop)
requirements.txt     Python dependencies
allowed_dirs.yaml    Local filesystem directories the agent can read
.env                 Credentials and URLs (not committed)
.nodeenv/            Local Node.js installation (not committed)
.venv/               Python virtual environment (not committed)
```

## Troubleshooting

- **"npx not found"**: Run `nodeenv --node=22.18.0 .nodeenv` to install Node.js locally, or install system-wide (`sudo zypper install nodejs22-npm-bin` on openSUSE).
- **"NC_URL not set"**: Add it to `.env`. This is the WebDAV base URL (e.g. `https://cloud.example.com`), not the MCP endpoint URL. (`NC_BASE_URL` is accepted as a deprecated fallback.)
- **Tool limit exceeded**: The Groq 128-tool limit is handled by dropping cookbook/news tools from the Nextcloud agent and curating the file upload agent's tool set. Check with the `config` command.
- **Session persistence**: Chat state is saved in `fastmail_chat_state.db`; the session ID is stored in `.fastmail_session_id`.
