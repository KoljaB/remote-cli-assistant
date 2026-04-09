from __future__ import annotations

import json
from pathlib import Path

from .models import CommandResult, InstallPlan, SystemProfile


TORCH_HINT = (
    "If the repository or goal appears to need PyTorch, CUDA TTS, CUDA ASR, whisper, faster-whisper, "
    "faster-qwen, qwen TTS, RealtimeTTS, or similar GPU inference stacks, you must explicitly research the "
    "best PyTorch wheel for the detected platform and CUDA situation before choosing an install command. "
    "Do not assume the newest CUDA wheel or the website's default latest wheel is automatically best. "
    "A lower CUDA wheel such as cu128 can be correct even when newer options like cu130 exist. "
    "Look closely at nvidia-smi, nvcc --version, and CUDA_PATH-style probe outputs when deciding compatibility. "
    "When CUDA-enabled torch is needed, install torch first inside the venv before the repository's other dependencies."
)


def planner_prompt(
    *,
    repository_url: str,
    install_goal: str,
    install_root: Path,
    system_profile: SystemProfile,
    research_context: str | None = None,
    can_use_web_search: bool = False,
) -> str:
    research_section = ""
    if research_context:
        research_section = f"""
Prepared research packet:
{research_context}
""".rstrip()

    if research_context:
        research_rule = (
            "- Use the provided research packet as your primary external research context.\n"
            "- Base `source_urls` on the packet URLs when they are relevant.\n"
            "- Do not invent external sources that are not supported by the packet."
        )
    elif can_use_web_search:
        research_rule = "- Research the repository using web search before deciding the plan."
    else:
        research_rule = (
            "- No external web research tool is available for this request.\n"
            "- Use the repository URL, install goal, system profile, and later command results carefully, "
            "and be explicit about uncertainty in `research_summary`."
        )

    return f"""
You are planning how to install a GitHub repository on the detected machine.

Repository URL:
{repository_url}

Install goal:
{install_goal}

Target installation directory:
{install_root}

Detected system profile:
{system_profile.to_prompt_block()}

Important CUDA and Torch probe outputs:
{system_profile.torch_probe_block()}

{research_section}

Rules:
{research_rule}
- The plan must be platform-aware. 
- A dedicated virtual environment is mandatory. Return both `venv_path` and `venv_creation_command`.
- Assume every shell command will run independently with an explicit working directory.
- Do not rely on shell state, do not use `cd`, and do not rely on activation side effects.
- Use explicit interpreter paths from the venv after it exists.
- If the user goal implies an optional acceleration feature or model backend, expand the goal into the concrete install objective that would actually satisfy it.
- Decide whether the repository or the install goal requires torch.
- {TORCH_HINT}
- If torch is required, return an explicit `torch_install_command` and explain why that exact torch build is appropriate.
- If torch is not required, return `requires_torch = false` and leave `torch_install_command` empty.
- Keep the first command safe and incremental.
- Provide a concrete verification command and verification working directory.
- The installation is considered finished only after the verification command has succeeded and your finish rule is satisfied.
- Return only structured data matching the requested schema.
""".strip()


def planner_schema() -> dict:
    return {
        "type": "json_schema",
        "name": "installation_plan",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "research_summary": {"type": "string"},
                "install_root": {"type": "string"},
                "venv_path": {"type": "string"},
                "venv_creation_command": {"type": "string"},
                "requires_torch": {"type": "boolean"},
                "torch_install_command": {"type": "string"},
                "torch_reasoning": {"type": "string"},
                "steps_overview": {"type": "array", "items": {"type": "string"}},
                "success_signals": {"type": "array", "items": {"type": "string"}},
                "verification_command": {"type": "string"},
                "verification_cwd": {"type": "string"},
                "finish_rule": {"type": "string"},
                "first_command": {"type": "string"},
                "first_cwd": {"type": "string"},
                "first_reason": {"type": "string"},
                "first_expectation": {"type": "string"},
                "source_urls": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "research_summary",
                "install_root",
                "venv_path",
                "venv_creation_command",
                "requires_torch",
                "torch_install_command",
                "torch_reasoning",
                "steps_overview",
                "success_signals",
                "verification_command",
                "verification_cwd",
                "finish_rule",
                "first_command",
                "first_cwd",
                "first_reason",
                "first_expectation",
                "source_urls",
            ],
        },
        "strict": True,
    }


def action_prompt(
    *,
    repository_url: str,
    install_goal: str,
    plan: InstallPlan,
    last_result: CommandResult | None,
    history: list[CommandResult],
    step_number: int,
    max_steps: int,
    verification_has_succeeded: bool,
    venv_ready: bool,
    torch_ready: bool,
    research_context: str | None = None,
    can_use_web_search: bool = False,
) -> str:
    history_block = _history_block(history)
    last_result_block = _result_block(last_result)
    research_section = ""
    if research_context:
        research_section = f"""
Prepared research packet:
{research_context}
""".rstrip()

    if research_context:
        research_rule = (
            "- Use the provided research packet when it is relevant.\n"
            "- Prefer source-backed decisions over guessing, especially for CUDA, torch, and platform quirks."
        )
    elif can_use_web_search:
        research_rule = (
            "- If a torch, CUDA, cuDNN, wheel compatibility problem appears in the output, use web research before "
            "choosing a fix."
        )
    else:
        research_rule = (
            "- No external web research tool is available for this request.\n"
            "- When the output is ambiguous, favor safe, incremental commands and avoid overconfident guesses."
        )
    return f"""
You are driving a platform-aware installation assistant for a GitHub repository.

Repository URL:
{repository_url}

Install goal:
{install_goal}

Installation root:
{plan.install_root}

Research summary:
{plan.research_summary}

Plan overview:
{chr(10).join(f"- {step}" for step in plan.steps_overview)}

Success signals:
{chr(10).join(f"- {signal}" for signal in plan.success_signals)}

Virtual environment path:
{plan.venv_path}

Virtual environment creation command:
{plan.venv_creation_command}

Torch required:
{"yes" if plan.requires_torch else "no"}

Torch install command:
{plan.torch_install_command or '<not required>'}

Torch reasoning:
{plan.torch_reasoning}

Verification command:
{plan.verification_command}

Verification cwd:
{plan.verification_cwd}

Finish rule:
{plan.finish_rule}

Progress so far:
{history_block}

Latest command result:
{last_result_block}

{research_section}

Control rules:
- You may only propose one next shell command at a time.
- Every command runs independently with an explicit working directory.
- Never use `cd`.
- A virtual environment is mandatory before dependency installation.
- Do not rely on activation side effects; use explicit interpreter paths from the venv.
- If torch is required and not installed yet, install the researched torch build first before other dependency-install steps.
- {TORCH_HINT}
- If the goal mentions an optional model, acceleration backend, or integration, choose next commands that move toward that fully expanded goal rather than a bare minimum install.
{research_rule}
- If the installation appears complete, only return `finish` after a successful verification command or a clearly satisfied success signal.
- If continuing is unsafe or clearly impossible, return `abort`.
- Keep the total session within {max_steps} commands. This is step {step_number} of {max_steps}.
- Return only structured data matching the requested schema.

Verification already succeeded: {"yes" if verification_has_succeeded else "no"}
Virtual environment ready: {"yes" if venv_ready else "no"}
Torch ready: {"yes" if torch_ready else "no"}
""".strip()


def action_schema() -> dict:
    return {
        "type": "json_schema",
        "name": "next_installation_action",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["continue", "finish", "abort"],
                },
                "reason": {"type": "string"},
                "next_command": {"type": "string"},
                "cwd": {"type": "string"},
                "expectation": {"type": "string"},
                "progress_update": {"type": "string"},
            },
            "required": [
                "status",
                "reason",
                "next_command",
                "cwd",
                "expectation",
                "progress_update",
            ],
        },
        "strict": True,
    }


def llm_visible_payload(label: str, content: str) -> str:
    payload = {"label": label, "content": content}
    return json.dumps(payload, indent=2)


def _history_block(history: list[CommandResult]) -> str:
    if not history:
        return "No commands have been executed yet."

    lines: list[str] = []
    for item in history[-8:]:
        lines.append(
            f"Step {item.step_number}: command={item.command!r}, cwd={item.cwd!r}, "
            f"exit_code={item.exit_code}, duration={item.duration_seconds:.1f}s"
        )
        trimmed = item.output.strip()
        if trimmed:
            trimmed = trimmed[-2500:]
            lines.append(trimmed)
    return "\n".join(lines)


def _result_block(result: CommandResult | None) -> str:
    if result is None:
        return "No command has run yet."
    return (
        f"step={result.step_number}\n"
        f"command={result.command}\n"
        f"cwd={result.cwd}\n"
        f"exit_code={result.exit_code}\n"
        f"duration_seconds={result.duration_seconds:.1f}\n"
        f"output=\n{result.output[-6000:]}"
    )
