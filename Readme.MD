`remote_cli_assistant.py` is a general AI-controlled remote CLI assistant.

What it does:

- Opens or connects to a visible CLI window that the assistant can control.
- Lets the user chat with the assistant in one terminal while commands are executed in another visible terminal.
- Supports two execution modes:
  `auto`: the assistant may execute several routine commands in a row until the goal is reached or real user input is needed.
  `confirm`: the assistant proposes one command at a time and waits for `/run`.
- Allows mode switches during the chat with `/mode auto` and `/mode confirm`.
- Supports local and remote setups.
  Local: the controller and the visible CLI run on the same machine.
  Remote: a worker server runs on another machine, opens the visible CLI there, and accepts commands over the network.
- Supports OpenAI-compatible model endpoints such as OpenAI or LM Studio.
- Supports hosted web search or a self-hosted SearXNG research backend.

Main chat commands:

- `/mode auto`
- `/mode confirm`
- `/run`
- `/status`
- `/help`
- `/quit`

Requirements:

- Python
- Network access for the selected LLM endpoint
- Optional network access to a SearXNG instance when `--research-backend searxng` is used
- A terminal launcher supported by the worker machine
  Windows: `cmd.exe`
  macOS: Terminal.app via `open`
  Linux: one of `x-terminal-emulator`, `gnome-terminal`, `konsole`, or `xterm`

Controller usage:

```powershell
python .\remote_cli_assistant.py
python .\remote_cli_assistant.py "Inspect this repository and tell me how to run its tests." --execution-mode confirm
python .\remote_cli_assistant.py --goal "Set up a Python venv, inspect the project, and get the dev server running." --execution-mode auto
```

LM Studio plus SearXNG:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:1234/v1"
export SEARXNG_BASE_URL="http://127.0.0.1:8080"

python remote_cli_assistant.py \
  --planner-model openai/gpt-oss-20b \
  --loop-model openai/gpt-oss-20b \
  --research-backend searxng \
  --execution-mode confirm
```

Remote worker setup:

On the Linux machine that should host the visible CLI window:

```bash
python remote_cli_assistant.py --worker-server --listen-host 0.0.0.0 --port 8765
```

On the Windows machine where the user talks to the assistant:

```powershell
$env:OPENAI_BASE_URL="http://127.0.0.1:1234/v1"
python .\remote_cli_assistant.py --connect 192.168.1.50:8765 --execution-mode confirm
```

Notes:

- The controller uses `planner-model` for fresh user instructions and `loop-model` for routine follow-up turns after command execution.
- The worker runs commands one at a time and streams output back to the controller.
- In `auto` mode the assistant still pauses for obviously risky commands such as destructive deletes, shutdowns, or similar actions.
- `OPENAI_API_KEY` is required when using OpenAI-hosted models.
- `OPENAI_BASE_URL` or `--base-url` can point to LM Studio or another OpenAI-compatible local server.
- `SEARXNG_BASE_URL` or `--search-api-url` enables self-hosted web research through SearXNG.
- `REMOTE_CLI_ASSISTANT_FORCE_CHAT_COMPLETIONS=1` forces the local-server fallback to `/v1/chat/completions`.
