from __future__ import annotations

import json

from .models import ChatMessage, CommandResult


def assistant_prompt(
    *,
    execution_mode: str,
    default_cwd: str,
    worker_label: str,
    system_profile_block: str,
    conversation: list[ChatMessage],
    command_history: list[CommandResult],
    last_result: CommandResult | None,
    research_context: str | None = None,
) -> str:
    conversation_block = _conversation_block(conversation)
    history_block = _history_block(command_history)
    last_result_block = _last_result_block(last_result)
    research_section = f"\nWeb research packet:\n{research_context}\n" if research_context else ""

    return f"""
You are an AI assistant that helps a human control a visible CLI window.

Current execution mode:
{execution_mode}

Execution mode meaning:
- In `auto` mode, routine commands may run automatically until the goal is reached or user input is truly needed.
- In `confirm` mode, propose one command at a time and expect the user to approve execution.
- Even in `auto` mode, ask the user before destructive, risky, credential-related, or high-uncertainty actions.

Remote or local worker:
{worker_label}

Default working directory:
{default_cwd}

Worker system profile:
{system_profile_block}

Conversation so far:
{conversation_block}

Recent command history:
{history_block}

Latest command result:
{last_result_block}
{research_section}
Rules:
- Help with any CLI-oriented goal, not just installation.
- Propose at most one shell command at a time.
- Every command must include an explicit working directory.
- Never use `cd`.
- Prefer safe, incremental commands that reveal information before making big changes.
- Ask the user when the goal is ambiguous, credentials or secrets are needed, a destructive action may be required, or you are blocked by missing context.
- If the goal appears completed, return `finish`.
- If more user input is needed before the next command, return `ask_user`.
- If continuing makes sense, return `continue` with one command.
- Keep `assistant_message` short, helpful, and user-facing.
- Put internal justification in `reason`.
- Return only data matching the schema.
""".strip()


def assistant_schema() -> dict:
    return {
        "type": "json_schema",
        "name": "cli_assistant_decision",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["continue", "ask_user", "finish"]},
                "assistant_message": {"type": "string"},
                "reason": {"type": "string"},
                "next_command": {"type": "string"},
                "cwd": {"type": "string"},
                "expectation": {"type": "string"},
                "goal_status": {"type": "string"},
            },
            "required": [
                "status",
                "assistant_message",
                "reason",
                "next_command",
                "cwd",
                "expectation",
                "goal_status",
            ],
        },
        "strict": True,
    }


def llm_visible_payload(label: str, content: str) -> str:
    payload = {"label": label, "content": content}
    return json.dumps(payload, indent=2)


def _conversation_block(conversation: list[ChatMessage]) -> str:
    if not conversation:
        return "No user goal has been provided yet."
    lines: list[str] = []
    for item in conversation[-12:]:
        lines.append(f"{item.role}: {item.content}")
    return "\n".join(lines)


def _history_block(command_history: list[CommandResult]) -> str:
    if not command_history:
        return "No commands have been executed yet."
    lines: list[str] = []
    for item in command_history[-8:]:
        lines.append(
            f"Step {item.step_number}: command={item.command!r}, cwd={item.cwd!r}, exit_code={item.exit_code}, duration={item.duration_seconds:.1f}s"
        )
        trimmed = item.output.strip()
        if trimmed:
            lines.append(trimmed[-2000:])
    return "\n".join(lines)


def _last_result_block(result: CommandResult | None) -> str:
    if result is None:
        return "No command has run yet."
    return (
        f"step={result.step_number}\n"
        f"command={result.command}\n"
        f"cwd={result.cwd}\n"
        f"exit_code={result.exit_code}\n"
        f"duration_seconds={result.duration_seconds:.1f}\n"
        f"output=\n{result.output[-4000:]}"
    )
