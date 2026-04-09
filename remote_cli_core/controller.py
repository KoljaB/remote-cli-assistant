from __future__ import annotations

import json
import ntpath
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .agent_prompts import assistant_prompt, assistant_schema, llm_visible_payload
from .models import AssistantDecision, ChatMessage, CommandResult, CommandTask
from .openai_client import OpenAIResponsesClient
from .remote_control import LocalTerminalClient, RemoteTerminalClient, parse_connect_target
from .web_research import ResearchPacket, SearXNGResearchClient, format_research_packet


CHAT_COMMAND_HELP = """Available chat commands:
/mode auto
/mode confirm
/run
/status
/help
/quit"""

RESEARCH_HINTS = (
    "install",
    "setup",
    "configure",
    "configuration",
    "docs",
    "documentation",
    "error",
    "fix",
    "github",
    "repository",
    "cuda",
    "torch",
)
STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "what",
    "when",
    "where",
    "would",
    "could",
    "should",
    "there",
    "which",
    "please",
    "about",
    "into",
    "they",
    "them",
    "will",
    "also",
    "make",
    "just",
}


@dataclass
class ChatAssistantConfig:
    initial_goal: str | None
    planner_model: str
    loop_model: str
    base_url: str | None
    research_backend: str
    search_api_url: str | None
    execution_mode: str
    connect_target: str | None
    max_auto_steps: int


class ChatAssistantController:
    def __init__(self, config: ChatAssistantConfig) -> None:
        self.config = config
        self.client = OpenAIResponsesClient(base_url=config.base_url)
        self.execution_mode = config.execution_mode
        self.research_backend = self._resolve_research_backend()
        self.research_client = self._build_research_client()
        self.conversation: list[ChatMessage] = []
        self.command_history: list[CommandResult] = []
        self.pending_task: CommandTask | None = None
        self.last_result: CommandResult | None = None
        self.next_step = 1

        if config.connect_target:
            host, port = parse_connect_target(config.connect_target)
            self.executor = RemoteTerminalClient(host=host, port=port)
            self.worker_hello = self.executor.hello()
        else:
            self.executor = LocalTerminalClient()
            self.worker_hello = self.executor.hello

        self.default_cwd = self.worker_hello.default_cwd
        self.system_profile_block = self._system_profile_block(self.worker_hello.system_profile)

    def run(self) -> int:
        print(f"Connected CLI worker: {self.worker_hello.worker_label}", flush=True)
        print(f"Execution mode: {self.execution_mode}", flush=True)
        print(CHAT_COMMAND_HELP, flush=True)

        try:
            if self.config.initial_goal:
                self._record_user_message(self.config.initial_goal)
                self._advance_from_user_instruction()

            while True:
                prompt = "you> " if self.pending_task is None else "approve> "
                try:
                    user_input = input(prompt).strip()
                except EOFError:
                    print("", flush=True)
                    return 0
                if not user_input:
                    continue
                if self._handle_chat_command(user_input):
                    continue
                self._record_user_message(user_input)
                self.pending_task = None
                self._advance_from_user_instruction()
        finally:
            self.executor.close()

    def _handle_chat_command(self, user_input: str) -> bool:
        lowered = user_input.strip().lower()
        if lowered == "/help":
            print(CHAT_COMMAND_HELP, flush=True)
            return True
        if lowered == "/status":
            self._show_status()
            return True
        if lowered == "/run":
            if self.pending_task is None:
                print("No pending command to execute.", flush=True)
                return True
            self._execute_pending_and_continue()
            return True
        if lowered == "/mode auto":
            self.execution_mode = "auto"
            print("Execution mode set to auto.", flush=True)
            if self.pending_task is not None and not self._requires_manual_confirmation(self.pending_task.command):
                self._execute_pending_and_continue()
            return True
        if lowered == "/mode confirm":
            self.execution_mode = "confirm"
            print("Execution mode set to confirm.", flush=True)
            return True
        if lowered == "/quit":
            raise SystemExit(0)
        return False

    def _advance_from_user_instruction(self) -> None:
        decision = self._request_decision(user_initiated=True)
        self._handle_decision(decision)

    def _execute_pending_and_continue(self) -> None:
        task = self.pending_task
        if task is None:
            return
        self.pending_task = None
        result = self._execute_task(task)
        self.last_result = result
        self.command_history.append(result)
        decision = self._request_decision(user_initiated=False)
        self._handle_decision(decision)

    def _handle_decision(self, decision: AssistantDecision, *, allow_auto: bool = True) -> None:
        if decision.assistant_message.strip():
            print(f"assistant> {decision.assistant_message}", flush=True)
            self.conversation.append(ChatMessage(role="assistant", content=decision.assistant_message))
        if decision.goal_status.strip():
            print(f"goal> {decision.goal_status}", flush=True)

        if decision.status == "finish":
            self.pending_task = None
            return

        if decision.status == "ask_user":
            self.pending_task = None
            return

        command = decision.next_command.strip()
        if not command:
            self.pending_task = None
            print("assistant> I need more input before proposing a command.", flush=True)
            return

        task = CommandTask(
            step_number=self.next_step,
            command=command,
            cwd=self._normalize_cwd(decision.cwd),
            reason=decision.reason,
            expectation=decision.expectation,
        )
        self.next_step += 1
        self.pending_task = task

        print(f"command[{task.step_number}]> {task.command}", flush=True)
        print(f"cwd[{task.step_number}]> {task.cwd}", flush=True)

        if allow_auto:
            self._auto_drive_if_possible()
        elif self.execution_mode != "auto":
            print("assistant> Use /run to execute this command, or reply with more instructions.", flush=True)

    def _request_decision(self, *, user_initiated: bool) -> AssistantDecision:
        research_packet = self._research_packet(user_initiated=user_initiated)
        prompt = assistant_prompt(
            execution_mode=self.execution_mode,
            default_cwd=self.default_cwd,
            worker_label=self.worker_hello.worker_label,
            system_profile_block=self.system_profile_block,
            conversation=self.conversation,
            command_history=self.command_history,
            last_result=self.last_result,
            research_context=format_research_packet(research_packet) if research_packet is not None else None,
        )
        print(f"llm-prompt> {llm_visible_payload('cli-assistant', prompt)}", flush=True)
        response = self.client.create_json_response(
            model=self.config.planner_model if user_initiated else self.config.loop_model,
            prompt=prompt,
            schema=assistant_schema(),
            tools=[{"type": "web_search"}] if self._uses_hosted_web_search() and self._should_offer_web_search() else None,
            reasoning_effort="high" if user_initiated else "medium",
        )
        print(f"llm-response> {response.text}", flush=True)
        return AssistantDecision.from_dict(json.loads(response.text))

    def _execute_task(self, task: CommandTask) -> CommandResult:
        print(f"assistant> Executing step {task.step_number}...", flush=True)

        def on_event(record: dict) -> None:
            event_type = record.get("event")
            if event_type == "command_started":
                print(f"worker> accepted {record.get('command')} in {record.get('cwd')}", flush=True)
            elif event_type == "command_output":
                text = str(record.get("text", ""))
                if text:
                    print(text, end="" if text.endswith("\n") else "\n", flush=True)
            elif event_type == "command_finished":
                print(
                    f"worker> exit_code={record.get('exit_code')} duration={record.get('duration_seconds')}",
                    flush=True,
                )

        result = self.executor.execute(task, on_event=on_event)
        print(f"assistant> Step {result.step_number} finished with exit code {result.exit_code}.", flush=True)
        return result

    def _record_user_message(self, content: str) -> None:
        self.conversation.append(ChatMessage(role="user", content=content))

    def _auto_drive_if_possible(self) -> None:
        if self.pending_task is None:
            return
        if self.execution_mode != "auto":
            print("assistant> Use /run to execute this command, or reply with more instructions.", flush=True)
            return
        if self._requires_manual_confirmation(self.pending_task.command):
            print("assistant> Manual confirmation required before the next command.", flush=True)
            return

        steps_remaining = self.config.max_auto_steps
        while self.pending_task is not None and self.execution_mode == "auto" and steps_remaining > 0:
            current = self.pending_task
            if current is None:
                return
            if self._requires_manual_confirmation(current.command):
                print("assistant> Manual confirmation required before the next command.", flush=True)
                return
            self.pending_task = None
            result = self._execute_task(current)
            self.last_result = result
            self.command_history.append(result)
            steps_remaining -= 1
            decision = self._request_decision(user_initiated=False)
            self._handle_decision(decision, allow_auto=False)
            if self.pending_task is None:
                return

        if self.pending_task is not None and steps_remaining == 0:
            print("assistant> Auto mode paused after reaching the configured step budget.", flush=True)

    def _show_status(self) -> None:
        print(f"mode> {self.execution_mode}", flush=True)
        print(f"worker> {self.worker_hello.worker_label}", flush=True)
        print(f"default-cwd> {self.default_cwd}", flush=True)
        print(f"history-steps> {len(self.command_history)}", flush=True)
        if self.pending_task is not None:
            print(f"pending-command> {self.pending_task.command}", flush=True)
            print(f"pending-cwd> {self.pending_task.cwd}", flush=True)
        if self.last_result is not None:
            print(f"last-exit> {self.last_result.exit_code}", flush=True)
            print(f"last-command> {self.last_result.command}", flush=True)

    def _resolve_research_backend(self) -> str:
        configured = (self.config.research_backend or "auto").strip().lower()
        if configured != "auto":
            return configured
        if self.config.search_api_url or os.environ.get("SEARXNG_BASE_URL"):
            return "searxng"
        parsed = urlparse(self.client.base_url)
        if parsed.hostname == "api.openai.com":
            return "openai"
        return "none"

    def _build_research_client(self) -> SearXNGResearchClient | None:
        if self.research_backend != "searxng":
            return None
        base_url = self.config.search_api_url or os.environ.get("SEARXNG_BASE_URL")
        if not base_url:
            raise RuntimeError("The SearXNG research backend requires --search-api-url or SEARXNG_BASE_URL.")
        return SearXNGResearchClient(base_url=base_url)

    def _uses_hosted_web_search(self) -> bool:
        return self.research_backend == "openai"

    def _research_packet(self, *, user_initiated: bool) -> ResearchPacket | None:
        if self.research_backend != "searxng" or self.research_client is None:
            return None
        query = self._build_research_query(user_initiated=user_initiated)
        if not query:
            return None
        try:
            return self.research_client.search(
                query=query,
                reason="Gather source-backed context for the current CLI goal or the latest command failure.",
            )
        except RuntimeError as exc:
            print(f"assistant> Research backend unavailable: {exc}", flush=True)
            return None

    def _build_research_query(self, *, user_initiated: bool) -> str:
        latest_user = self._latest_user_message()
        if self.last_result is not None and self.last_result.exit_code != 0:
            error_signal = self._error_signal(self.last_result.output)
            keywords = " ".join(self._keywords((latest_user or "") + " " + self.last_result.output)[:10])
            return " ".join(part for part in (error_signal, keywords) if part).strip()
        if user_initiated and latest_user and self._message_likely_needs_research(latest_user):
            return latest_user
        return ""

    def _should_offer_web_search(self) -> bool:
        latest_user = self._latest_user_message()
        if self.last_result is not None and self.last_result.exit_code != 0:
            return True
        return bool(latest_user and self._message_likely_needs_research(latest_user))

    def _latest_user_message(self) -> str | None:
        for item in reversed(self.conversation):
            if item.role == "user":
                return item.content
        return None

    def _message_likely_needs_research(self, message: str) -> bool:
        lowered = message.lower()
        return "http://" in lowered or "https://" in lowered or any(token in lowered for token in RESEARCH_HINTS)

    def _error_signal(self, output: str) -> str:
        for line in reversed(output.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if any(token in lowered for token in ("error", "failed", "not found", "traceback", "permission denied")):
                return candidate[:180]
        return "command failure"

    def _keywords(self, text: str) -> list[str]:
        keywords: list[str] = []
        for token in re.findall(r"[A-Za-z0-9_./:-]+", text.lower()):
            if len(token) < 4 or token in STOPWORDS:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords

    def _normalize_cwd(self, raw_cwd: str) -> str:
        value = (raw_cwd or "").strip()
        if not value:
            return self.default_cwd
        if self._is_absolute_path(value):
            return value
        base = self.default_cwd
        if "\\" in base or re.match(r"^[A-Za-z]:", base):
            return ntpath.normpath(ntpath.join(base, value))
        return posixpath.normpath(posixpath.join(base, value))

    def _is_absolute_path(self, value: str) -> bool:
        return value.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", value))

    def _requires_manual_confirmation(self, command: str) -> bool:
        lowered = command.lower()
        return any(
            token in lowered
            for token in (
                "rm ",
                "rm -",
                "rmdir",
                "del ",
                "format ",
                "mkfs",
                "shutdown",
                "reboot",
                "passwd",
                "userdel",
                "groupdel",
                "git reset --hard",
                "git clean -fd",
            )
        )

    def _system_profile_block(self, payload: dict) -> str:
        lines = []
        for key, value in payload.items():
            if key == "records":
                continue
            label = key.replace("_", " ")
            lines.append(f"{label}: {value}")
        return "\n".join(lines)
