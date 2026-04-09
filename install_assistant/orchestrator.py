from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .ansi import PARENT_THEME, enable_ansi, paint
from .models import CommandResult, CommandTask, InstallPlan, NextAction, SystemProfile
from .openai_client import OpenAIResponsesClient
from .prompts import action_prompt, action_schema, llm_visible_payload, planner_prompt, planner_schema
from .session_io import SessionPaths, read_json, write_json
from .system_probe import collect_system_profile
from .web_research import ResearchPacket, SearXNGResearchClient, format_research_packet


POLL_INTERVAL_SECONDS = 0.05
REPEAT_LIMIT = 2
SCAN_LIMIT = 10
STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "into",
    "after",
    "before",
    "when",
    "where",
    "have",
    "goal",
    "repo",
    "repository",
    "python",
    "install",
    "installation",
    "system",
}


@dataclass
class AssistantConfig:
    repository_url: str
    install_goal: str
    target_dir: Path | None
    planner_model: str
    loop_model: str
    base_url: str | None
    research_backend: str
    search_api_url: str | None
    max_steps: int


class InstallationOrchestrator:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self.client = OpenAIResponsesClient(base_url=config.base_url)
        self.research_backend = self._resolve_research_backend()
        self.research_client = self._build_research_client()
        self.event_offset = 0

    def run(self) -> int:
        enable_ansi()
        self._headline("System Analysis")
        profile = collect_system_profile()
        self._print_probe_details(profile)

        install_root = self._resolve_install_root(self.config.repository_url, self.config.target_dir)
        install_root.parent.mkdir(parents=True, exist_ok=True)

        self._headline("Planning")
        plan, response_id = self._request_plan(profile, install_root)
        plan.install_root = str(install_root)
        plan.first_cwd = self._normalize_cwd(plan.first_cwd, install_root)
        plan.verification_cwd = self._normalize_cwd(plan.verification_cwd, install_root)
        plan.venv_path = self._normalize_path_hint(plan.venv_path, install_root)
        if not plan.venv_creation_command.strip():
            raise RuntimeError("The planner did not provide a venv creation command.")
        if not plan.verification_command.strip():
            raise RuntimeError("The planner did not provide a verification command.")
        if plan.requires_torch and not plan.torch_install_command.strip():
            raise RuntimeError("The planner marked torch as required but did not provide a torch install command.")
        self._show_plan(plan)

        session = SessionPaths.create()
        write_json(session.status_path, {"state": "running", "message": "Installation in progress."})
        worker_process = self._launch_worker(session)

        history: list[CommandResult] = []
        venv_ready = self._detect_virtual_environment(install_root, history) is not None
        torch_ready = not plan.requires_torch
        verification_has_succeeded = False
        final_state = "aborted"
        final_message = "The installation did not reach a verified completion state."
        next_task = CommandTask(
            step_number=1,
            command=plan.first_command,
            cwd=plan.first_cwd,
            reason=plan.first_reason,
            expectation=plan.first_expectation,
        )
        next_task = self._enforce_prerequisites(next_task, plan, install_root, venv_ready, torch_ready)

        try:
            for step_number in range(1, self.config.max_steps + 1):
                result = self._dispatch_and_wait(session, worker_process, next_task)
                history.append(result)
                venv_ready = self._detect_virtual_environment(install_root, history) is not None
                torch_ready = torch_ready or self._is_torch_step(result, plan)
                verification_has_succeeded = verification_has_succeeded or self._matches_verification(result, plan)
                self._show_command_result(result)

                if step_number == self.config.max_steps:
                    final_state = "aborted"
                    final_message = (
                        f"Stopped after {self.config.max_steps} commands without satisfying the finish rule."
                    )
                    break

                action, response_id = self._request_next_action(
                    plan=plan,
                    response_id=response_id,
                    history=history,
                    last_result=result,
                    verification_has_succeeded=verification_has_succeeded,
                    step_number=step_number + 1,
                    venv_ready=venv_ready,
                    torch_ready=torch_ready,
                )
                self._show_action(action)

                if action.status == "abort":
                    final_state = "aborted"
                    final_message = action.reason
                    break

                if action.status == "finish":
                    if plan.requires_torch and not torch_ready:
                        next_task = CommandTask(
                            step_number=step_number + 1,
                            command=plan.torch_install_command,
                            cwd=self._torch_command_cwd(plan, install_root),
                            reason="Install the researched torch build before finishing.",
                            expectation="Torch should be installed successfully inside the venv.",
                        )
                        next_task = self._enforce_prerequisites(next_task, plan, install_root, venv_ready, torch_ready)
                        continue
                    if verification_has_succeeded:
                        final_state = "completed"
                        final_message = action.reason
                        break
                    next_task = CommandTask(
                        step_number=step_number + 1,
                        command=plan.verification_command,
                        cwd=plan.verification_cwd,
                        reason="Explicit verification is required before the installer can finish.",
                        expectation="The verification command should succeed if the installation is complete.",
                    )
                    next_task = self._enforce_prerequisites(next_task, plan, install_root, venv_ready, torch_ready)
                    continue

                next_command = action.next_command.strip()
                if not next_command:
                    final_state = "aborted"
                    final_message = "The LLM returned continue without a next command."
                    break

                normalized_cwd = self._normalize_cwd(action.cwd, install_root)
                if self._is_repeated_command(history, next_command, normalized_cwd):
                    final_state = "aborted"
                    final_message = (
                        "Stopped because the same command was proposed repeatedly without clear progress."
                    )
                    break

                next_task = CommandTask(
                    step_number=step_number + 1,
                    command=next_command,
                    cwd=normalized_cwd,
                    reason=action.reason,
                    expectation=action.expectation,
                )
                next_task = self._enforce_prerequisites(next_task, plan, install_root, venv_ready, torch_ready)
        finally:
            write_json(session.status_path, {"state": final_state, "message": final_message})

        self._headline("Session Result")
        self._detail(f"Final state: {final_state}")
        self._detail(f"Reason: {final_message}")
        self._show_final_documentation(
            install_root=install_root,
            plan=plan,
            history=history,
            final_state=final_state,
            final_message=final_message,
            verification_has_succeeded=verification_has_succeeded,
            torch_ready=torch_ready,
        )
        self._detail("Waiting for the installation window to be closed...")
        worker_process.wait()
        shutil.rmtree(session.root, ignore_errors=True)
        return 0 if final_state == "completed" else 1

    def _request_plan(self, profile: SystemProfile, install_root: Path) -> tuple[InstallPlan, str]:
        research_packet = self._planner_research_packet(profile)
        if research_packet is not None:
            self._show_research_packet("planner", research_packet)
        prompt = planner_prompt(
            repository_url=self.config.repository_url,
            install_goal=self.config.install_goal,
            install_root=install_root,
            system_profile=profile,
            research_context=format_research_packet(research_packet) if research_packet is not None else None,
            can_use_web_search=self._uses_hosted_web_search(),
        )
        self._llm_prompt("planner", prompt)
        response = self.client.create_json_response(
            model=self.config.planner_model,
            prompt=prompt,
            schema=planner_schema(),
            tools=[{"type": "web_search"}] if self._uses_hosted_web_search() else None,
            reasoning_effort="high",
        )
        self._llm_response("planner", response.text)
        payload = json.loads(response.text)
        plan = InstallPlan.from_dict(payload)
        if not plan.source_urls and research_packet is not None:
            plan.source_urls = research_packet.urls()
        return plan, response.response_id

    def _request_next_action(
        self,
        *,
        plan: InstallPlan,
        response_id: str,
        history: list[CommandResult],
        last_result: CommandResult,
        verification_has_succeeded: bool,
        step_number: int,
        venv_ready: bool,
        torch_ready: bool,
    ) -> tuple[NextAction, str]:
        research_packet = self._action_research_packet(plan, last_result, torch_ready)
        if research_packet is not None:
            self._show_research_packet("next-step", research_packet)
        prompt = action_prompt(
            repository_url=self.config.repository_url,
            install_goal=self.config.install_goal,
            plan=plan,
            last_result=last_result,
            history=history,
            step_number=step_number,
            max_steps=self.config.max_steps,
            verification_has_succeeded=verification_has_succeeded,
            venv_ready=venv_ready,
            torch_ready=torch_ready,
            research_context=format_research_packet(research_packet) if research_packet is not None else None,
            can_use_web_search=self._uses_hosted_web_search(),
        )
        self._llm_prompt("next-step", prompt)
        tools = (
            [{"type": "web_search"}]
            if self._uses_hosted_web_search() and self._needs_torch_research(plan, last_result, torch_ready)
            else None
        )
        response = self.client.create_json_response(
            model=self.config.loop_model,
            prompt=prompt,
            schema=action_schema(),
            previous_response_id=response_id if self._uses_hosted_web_search() else None,
            tools=tools,
            reasoning_effort="medium",
        )
        self._llm_response("next-step", response.text)
        payload = json.loads(response.text)
        return NextAction.from_dict(payload), response.response_id

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
            raise RuntimeError(
                "The SearXNG research backend requires --search-api-url or SEARXNG_BASE_URL."
            )
        return SearXNGResearchClient(base_url=base_url)

    def _uses_hosted_web_search(self) -> bool:
        return self.research_backend == "openai"

    def _planner_research_packet(self, profile: SystemProfile) -> ResearchPacket | None:
        if self.research_backend != "searxng" or self.research_client is None:
            return None
        query = self._build_planner_research_query(profile)
        try:
            return self.research_client.search(
                query=query,
                reason="Research the repository, its install docs, and platform-specific setup before planning.",
            )
        except RuntimeError as exc:
            self._warning(f"Web research is unavailable right now: {exc}")
            return None

    def _action_research_packet(
        self,
        plan: InstallPlan,
        last_result: CommandResult,
        torch_ready: bool,
    ) -> ResearchPacket | None:
        if self.research_backend != "searxng" or self.research_client is None:
            return None
        if not self._needs_local_followup_research(plan, last_result, torch_ready):
            return None
        query = self._build_action_research_query(plan, last_result, torch_ready)
        try:
            return self.research_client.search(
                query=query,
                reason="Research the latest command result and gather source-backed fixes before choosing the next step.",
            )
        except RuntimeError as exc:
            self._warning(f"Follow-up web research is unavailable right now: {exc}")
            return None

    def _needs_local_followup_research(
        self,
        plan: InstallPlan,
        last_result: CommandResult,
        torch_ready: bool,
    ) -> bool:
        if self._needs_torch_research(plan, last_result, torch_ready):
            return True
        lowered = last_result.output.lower()
        return last_result.exit_code != 0 and any(
            token in lowered
            for token in (
                "error",
                "failed",
                "not found",
                "no module named",
                "cuda",
                "torch",
                "wheel",
                "requirements",
            )
        )

    def _build_planner_research_query(self, profile: SystemProfile) -> str:
        repo_terms = self._repository_search_terms()
        goal_terms = " ".join(self._goal_keywords(self.config.install_goal)[:6])
        platform_terms = " ".join(
            term
            for term in (
                profile.platform_family.lower(),
                profile.operating_system.lower(),
                "cuda" if self._goal_mentions_cuda() else "",
                "torch" if self._goal_mentions_torch() else "",
            )
            if term and term != "unknown"
        )
        query_parts = [repo_terms, "installation README requirements", goal_terms, platform_terms]
        return " ".join(part for part in query_parts if part).strip()

    def _build_action_research_query(
        self,
        plan: InstallPlan,
        last_result: CommandResult,
        torch_ready: bool,
    ) -> str:
        repo_terms = self._repository_search_terms()
        signal = self._error_signal(last_result.output)
        keywords = " ".join(self._goal_keywords(last_result.output)[:8])
        focus_terms = []
        if plan.requires_torch and not torch_ready:
            focus_terms.append("torch cuda wheel compatibility")
        if "linux" not in keywords.lower():
            focus_terms.append("linux")
        return " ".join(
            part
            for part in (
                repo_terms,
                signal,
                keywords,
                " ".join(focus_terms),
            )
            if part
        ).strip()

    def _repository_search_terms(self) -> str:
        parsed = urlparse(self.config.repository_url)
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1].removesuffix(".git")
            return f"\"{owner}/{repo}\""
        if path_parts:
            return f"\"{path_parts[-1].removesuffix('.git')}\""
        return f"\"{self.config.repository_url}\""

    def _goal_mentions_cuda(self) -> bool:
        lowered = self.config.install_goal.lower()
        return any(token in lowered for token in ("cuda", "cudnn", "nvidia", "gpu"))

    def _goal_mentions_torch(self) -> bool:
        lowered = self.config.install_goal.lower()
        return any(token in lowered for token in ("torch", "pytorch", "torchaudio", "torchvision"))

    def _error_signal(self, output: str) -> str:
        for line in reversed(output.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if any(
                token in lowered
                for token in ("error", "failed", "not found", "cuda", "torch", "wheel", "module")
            ):
                return candidate[:180]
        return ""

    def _show_research_packet(self, label: str, packet: ResearchPacket) -> None:
        self._headline(f"Web Research: {label}")
        self._detail(f"provider: {packet.provider}")
        self._detail(f"query: {packet.query}")
        if not packet.sources:
            self._warning("No sources were returned by the research backend.")
            return
        for source in packet.sources[:5]:
            self._detail(f"source: {source.url}")
            if source.snippet:
                self._output(source.snippet[:400] + ("\n" if not source.snippet.endswith("\n") else ""))

    def _dispatch_and_wait(
        self,
        session: SessionPaths,
        worker_process: subprocess.Popen[bytes],
        task: CommandTask,
    ) -> CommandResult:
        self._headline(f"Dispatch Step {task.step_number}")
        self._detail(f"cwd: {task.cwd}")
        self._command(task.command + "\n")
        write_json(session.command_path(task.step_number), task.to_dict())

        result_path = session.result_path(task.step_number)
        while True:
            self.event_offset = self._drain_events(session.events_path, self.event_offset)
            if result_path.exists():
                self.event_offset = self._drain_events(session.events_path, self.event_offset)
                return CommandResult.from_dict(read_json(result_path))
            if worker_process.poll() is not None:
                raise RuntimeError("The installation worker window closed unexpectedly.")
            time.sleep(POLL_INTERVAL_SECONDS)

    def _drain_events(self, events_path: Path, offset: int) -> int:
        if not events_path.exists():
            return offset

        with events_path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                self._render_event(event)
            return handle.tell()

    def _render_event(self, event: dict) -> None:
        event_type = event.get("event")
        if event_type == "command_started":
            self._detail(f"Worker accepted step {event.get('step_number')} in {event.get('cwd')}")
            return
        if event_type == "command_output":
            self._output(event.get("text", ""))
            return
        if event_type == "worker_complete":
            self._detail(f"Worker complete: {event.get('message', '')}")

    def _show_plan(self, plan: InstallPlan) -> None:
        self._response(plan.research_summary + "\n")
        if plan.source_urls:
            for url in plan.source_urls:
                self._detail(f"source: {url}")
        self._detail(f"venv path: {plan.venv_path}")
        self._detail(f"venv creation command: {plan.venv_creation_command}")
        self._detail(f"torch required: {'yes' if plan.requires_torch else 'no'}")
        if plan.requires_torch:
            self._detail(f"torch install command: {plan.torch_install_command}")
            self._detail(f"torch reasoning: {plan.torch_reasoning}")
        self._detail(f"verification command: {plan.verification_command}")
        self._detail(f"verification cwd: {plan.verification_cwd}")
        self._detail(f"finish rule: {plan.finish_rule}")

    def _show_action(self, action: NextAction) -> None:
        self._headline("LLM Decision")
        self._detail(f"status: {action.status}")
        self._detail(f"reason: {action.reason}")
        self._detail(f"progress: {action.progress_update}")
        if action.next_command:
            self._detail(f"next cwd: {action.cwd}")
            self._command(action.next_command + "\n")

    def _show_command_result(self, result: CommandResult) -> None:
        self._headline(f"Step {result.step_number} Result")
        self._detail(f"exit code: {result.exit_code}")
        self._detail(f"duration: {result.duration_seconds:.1f}s")
        self._detail(f"last command: {result.command}")
        if result.output.strip():
            self._detail("last output tail:")
            self._output(result.output[-2000:] + ("\n" if not result.output.endswith("\n") else ""))

    def _show_final_documentation(
        self,
        *,
        install_root: Path,
        plan: InstallPlan,
        history: list[CommandResult],
        final_state: str,
        final_message: str,
        verification_has_succeeded: bool,
        torch_ready: bool,
    ) -> None:
        self._headline("Final Documentation")
        self._detail(f"Repository URL: {self.config.repository_url}")
        self._detail(f"Install goal: {self.config.install_goal}")
        self._detail(f"Installation root: {install_root}")
        self._detail(f"Verification succeeded: {'yes' if verification_has_succeeded else 'no'}")
        self._detail(f"Verification command: {plan.verification_command}")
        self._detail(f"Torch required: {'yes' if plan.requires_torch else 'no'}")
        self._detail(f"Torch ready: {'yes' if torch_ready else 'no'}")
        if plan.requires_torch:
            self._detail(f"Torch selection reasoning: {plan.torch_reasoning}")

        successful = [item for item in history if item.exit_code == 0]
        failed = [item for item in history if item.exit_code != 0]

        self._headline("1. Successful Installation Steps")
        if successful:
            for item in successful:
                self._success(f"Step {item.step_number}: success")
                self._detail(f"Purpose: {item.reason}")
                self._detail(f"Expected outcome: {item.expectation}")
                self._detail(f"Working directory: {item.cwd}")
                self._command(item.command + "\n")
        else:
            self._warning("No successful installation commands were recorded.")

        if failed:
            self._headline("Failed Or Diagnostic Steps")
            for item in failed:
                self._error(f"Step {item.step_number}: exit code {item.exit_code}")
                self._detail(f"Working directory: {item.cwd}")
                self._command(item.command + "\n")

        venv_path = self._detect_virtual_environment(install_root, history)
        self._headline("2. Virtual Environment")
        if venv_path:
            self._success(f"Virtual environment folder: {venv_path}")
            for activate_label, activate_command in self._activation_commands(venv_path):
                self._detail(f"{activate_label}:")
                self._command(activate_command + "\n")
            self._detail("Direct Python inside the environment:")
            self._command(f'"{self._venv_python_path(venv_path)}"\n')
        else:
            self._warning(
                "No virtual environment folder was found in the installation root or command history. "
                "The install may be using a global interpreter, which is not the intended outcome."
            )

        repo_info = self._inspect_repository(install_root)
        self._headline("3. Testing And Example Files")
        if repo_info["test_dirs"]:
            self._detail("Test directories found:")
            for path in repo_info["test_dirs"]:
                self._success(f"- {path}")
        else:
            self._warning("No obvious test directory was found.")

        if repo_info["test_files"]:
            self._detail("Example test files:")
            for path in repo_info["test_files"]:
                self._output(f"- {path}\n")

        if repo_info["example_files"]:
            self._detail("Example or demo files:")
            for path in repo_info["example_files"]:
                self._output(f"- {path}\n")
        else:
            self._warning("No obvious example or demo file was found.")

        if repo_info["goal_related_files"]:
            self._detail("Files that look relevant to the chosen goal:")
            for path in repo_info["goal_related_files"]:
                self._output(f"- {path}\n")

        if repo_info["pyproject_scripts"]:
            self._detail("Scripts declared in pyproject.toml:")
            for script_name in repo_info["pyproject_scripts"]:
                self._output(f"- {script_name}\n")

        suggested_python = str(self._venv_python_path(venv_path)) if venv_path else "python"
        if repo_info["has_pytest"]:
            self._detail("Likely first test command:")
            self._command(f'"{suggested_python}" -m pytest\n' if venv_path else "python -m pytest\n")
        elif repo_info["test_dirs"]:
            self._detail("There are test files, but pytest could not be inferred automatically.")
            self._command(f'"{suggested_python}" -m pytest\n' if venv_path else "python -m pytest\n")
        elif repo_info["example_files"]:
            self._detail("A practical first run candidate from the repository:")
            candidate = repo_info["example_files"][0]
            if venv_path:
                self._command(f'"{suggested_python}" "{install_root / candidate}"\n')
            else:
                self._command(f'python "{install_root / candidate}"\n')

        self._headline("Useful Notes")
        self._detail(f"Final outcome: {final_state}")
        self._detail(f"Final reason: {final_message}")
        self._detail(f"Repository present on disk: {'yes' if install_root.exists() else 'no'}")
        if plan.requires_torch:
            self._detail("Torch install command that was planned:")
            self._command(plan.torch_install_command + "\n")
        if repo_info["config_files"]:
            self._detail("Important repository files found:")
            for path in repo_info["config_files"]:
                self._output(f"- {path}\n")
        if repo_info["entrypoint_files"]:
            self._detail("Potential entry-point files:")
            for path in repo_info["entrypoint_files"]:
                self._output(f"- {path}\n")

    def _detect_virtual_environment(self, install_root: Path, history: list[CommandResult]) -> Path | None:
        candidates: list[Path] = []
        for item in history:
            match = re.search(r"-m\s+venv\s+([\"']?)([^\"'\s]+)\1", item.command)
            if match:
                raw_path = Path(match.group(2))
                candidate = raw_path if raw_path.is_absolute() else Path(item.cwd) / raw_path
                candidates.append(candidate)
            for found in re.findall(r"([A-Za-z]:\\[^\"']+?\\Scripts\\python\.exe)", item.command):
                candidates.append(Path(found).parent.parent)
            for found in re.findall(r"(/[^\"'\s]+/(?:bin|Scripts)/python(?:\.exe)?)", item.command):
                candidates.append(Path(found).parent.parent)

        for name in (".venv", "venv", "env", ".env"):
            candidates.append(install_root / name)

        for candidate in candidates:
            if self._looks_like_venv(candidate):
                return candidate.resolve()

        if install_root.exists():
            for pyvenv in install_root.rglob("pyvenv.cfg"):
                return pyvenv.parent.resolve()
        return None

    def _inspect_repository(self, install_root: Path) -> dict[str, object]:
        info: dict[str, object] = {
            "test_dirs": [],
            "test_files": [],
            "example_files": [],
            "goal_related_files": [],
            "entrypoint_files": [],
            "config_files": [],
            "pyproject_scripts": [],
            "has_pytest": False,
        }
        if not install_root.exists():
            return info

        config_files: list[str] = info["config_files"]  # type: ignore[assignment]
        for name in (
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            "environment.yml",
            "setup.py",
            "pytest.ini",
            "tox.ini",
            "README.md",
            "README.rst",
        ):
            path = install_root / name
            if path.exists():
                self._append_unique(config_files, path.relative_to(install_root).as_posix())

        pyproject_path = install_root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
                project_scripts = pyproject_data.get("project", {}).get("scripts", {})
                poetry_scripts = pyproject_data.get("tool", {}).get("poetry", {}).get("scripts", {})
                combined = list(project_scripts.keys()) + list(poetry_scripts.keys())
                pyproject_scripts: list[str] = info["pyproject_scripts"]  # type: ignore[assignment]
                for script_name in combined[:SCAN_LIMIT]:
                    self._append_unique(pyproject_scripts, script_name)
                info["has_pytest"] = "pytest" in pyproject_path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                pass

        goal_keywords = self._goal_keywords(self.config.install_goal)
        excluded_dirs = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv", "venv", "env", ".env", "node_modules"}
        test_dirs: list[str] = info["test_dirs"]  # type: ignore[assignment]
        test_files: list[str] = info["test_files"]  # type: ignore[assignment]
        example_files: list[str] = info["example_files"]  # type: ignore[assignment]
        goal_related_files: list[str] = info["goal_related_files"]  # type: ignore[assignment]
        entrypoint_files: list[str] = info["entrypoint_files"]  # type: ignore[assignment]

        for current_dir, dirnames, filenames in os.walk(install_root):
            dirnames[:] = [name for name in dirnames if name not in excluded_dirs]
            rel_dir = Path(current_dir).relative_to(install_root)
            for dirname in dirnames:
                lower = dirname.lower()
                rel_path = (rel_dir / dirname).as_posix()
                if lower in {"tests", "test", "testing"}:
                    self._append_unique(test_dirs, rel_path)
            for filename in filenames:
                rel_path = (rel_dir / filename).as_posix()
                lower = filename.lower()
                rel_lower = rel_path.lower()
                if lower.startswith("test_") or lower.endswith("_test.py"):
                    self._append_unique(test_files, rel_path)
                if any(part in rel_lower for part in ("example", "examples", "demo", "demos", "sample", "samples")):
                    self._append_unique(example_files, rel_path)
                if lower in {"app.py", "main.py", "run.py", "server.py", "webui.py", "cli.py", "launch.py"}:
                    self._append_unique(entrypoint_files, rel_path)
                if goal_keywords and any(keyword in rel_lower for keyword in goal_keywords):
                    self._append_unique(goal_related_files, rel_path)

        return info

    def _append_unique(self, bucket: list[str], value: str) -> None:
        if value not in bucket and len(bucket) < SCAN_LIMIT:
            bucket.append(value)

    def _goal_keywords(self, text: str) -> list[str]:
        keywords: list[str] = []
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            if len(token) < 4 or token in STOPWORDS:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords

    def _launch_worker(self, session: SessionPaths) -> subprocess.Popen[bytes]:
        config_payload = {
            "commands_dir": str(session.commands_dir),
            "results_dir": str(session.results_dir),
            "artifacts_dir": str(session.artifacts_dir),
            "events_path": str(session.events_path),
            "status_path": str(session.status_path),
        }
        write_json(session.config_path, config_payload)

        script_path = Path(sys.argv[0]).resolve()
        python_executable = Path(sys.executable).resolve()
        self._headline("Worker Window")
        self._detail("Opening the visible installation CLI window...")

        if os.name == "nt":
            launcher_path = session.launcher_path
            launcher_contents = "\r\n".join(
                [
                    "@echo off",
                    "setlocal",
                    f'"{python_executable}" "{script_path}" --worker "{session.config_path}"',
                    "endlocal",
                ]
            )
            launcher_path.write_text(launcher_contents, encoding="utf-8")
            return subprocess.Popen(
                [os.environ.get("COMSPEC", "cmd.exe"), "/k", str(launcher_path)],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

        worker_command = f"{shlex.quote(str(python_executable))} {shlex.quote(str(script_path))} --worker {shlex.quote(str(session.config_path))}"
        launcher_path = session.root / "launch_worker.sh"
        launcher_path.write_text(f"#!/bin/sh\n{worker_command}\nexec \"${{SHELL:-/bin/sh}}\"\n", encoding="utf-8")
        launcher_path.chmod(0o755)

        if sys.platform == "darwin":
            return subprocess.Popen(["open", "-W", "-a", "Terminal", str(launcher_path)])

        terminal_commands = [
            ["x-terminal-emulator", "-e", "sh", "-lc", f"{worker_command}; exec $SHELL"],
            ["gnome-terminal", "--", "sh", "-lc", f"{worker_command}; exec $SHELL"],
            ["konsole", "-e", "sh", "-lc", f"{worker_command}; exec $SHELL"],
            ["xterm", "-e", "sh", "-lc", f"{worker_command}; exec $SHELL"],
        ]
        for command in terminal_commands:
            try:
                return subprocess.Popen(command)
            except FileNotFoundError:
                continue
        raise RuntimeError("No supported terminal launcher was found for this platform.")

    def _normalize_cwd(self, value: str, install_root: Path) -> str:
        text = (value or "").strip()
        if not text:
            return str(install_root)
        path = Path(text)
        if not path.is_absolute():
            path = install_root / path
        return str(path)

    def _normalize_path_hint(self, value: str, install_root: Path) -> str:
        text = (value or "").strip()
        if not text:
            return str(install_root / ".venv")
        path = Path(text)
        if not path.is_absolute():
            path = install_root / path
        return str(path)

    def _matches_verification(self, result: CommandResult, plan: InstallPlan) -> bool:
        expected_commands = {plan.verification_command.strip()}
        expected_commands.add(
            self._rewrite_command_for_venv(
                plan.verification_command,
                self._venv_python_path(Path(plan.venv_path)),
            ).strip()
        )
        return (
            result.command.strip() in expected_commands
            and Path(result.cwd) == Path(plan.verification_cwd)
            and result.exit_code == 0
        )

    def _is_torch_step(self, result: CommandResult, plan: InstallPlan) -> bool:
        if result.exit_code != 0:
            return False
        command = result.command.lower()
        return command == plan.torch_install_command.strip().lower() or self._is_torch_command(command)

    def _is_torch_command(self, command: str) -> bool:
        lowered = command.lower()
        return any(token in lowered for token in ("torch", "torchvision", "torchaudio", "download.pytorch.org"))

    def _is_dependency_install_command(self, command: str) -> bool:
        lowered = command.lower()
        patterns = (
            "pip install",
            "pip3 install",
            "python -m pip install",
            "python3 -m pip install",
            "uv pip install",
            "poetry install",
            "conda install",
        )
        return any(pattern in lowered for pattern in patterns)

    def _needs_torch_research(self, plan: InstallPlan, last_result: CommandResult, torch_ready: bool) -> bool:
        if not plan.requires_torch:
            return False
        if not torch_ready:
            return True
        lowered = last_result.output.lower()
        return any(token in lowered for token in ("torch", "cuda", "cudnn", "wheel", "download.pytorch.org"))

    def _enforce_prerequisites(
        self,
        task: CommandTask,
        plan: InstallPlan,
        install_root: Path,
        venv_ready: bool,
        torch_ready: bool,
    ) -> CommandTask:
        if not venv_ready and self._is_dependency_install_command(task.command):
            return CommandTask(
                step_number=task.step_number,
                command=plan.venv_creation_command,
                cwd=self._venv_command_cwd(plan, install_root),
                reason="Create the required virtual environment before installing dependencies.",
                expectation="The virtual environment should exist and its Python executable should be usable.",
            )
        if plan.requires_torch and venv_ready and not torch_ready and self._is_dependency_install_command(task.command):
            if not self._is_torch_command(task.command):
                return CommandTask(
                    step_number=task.step_number,
                    command=self._rewrite_command_for_venv(
                        plan.torch_install_command,
                        self._venv_python_path(Path(plan.venv_path)),
                    ),
                    cwd=self._torch_command_cwd(plan, install_root),
                    reason="Install the researched torch build before other dependencies.",
                    expectation="Torch should install successfully inside the virtual environment.",
                )
        if venv_ready:
            return self._rewrite_task_for_venv(task, plan)
        return task

    def _rewrite_task_for_venv(self, task: CommandTask, plan: InstallPlan) -> CommandTask:
        if task.command.strip() == plan.venv_creation_command.strip():
            return task
        rewritten = self._rewrite_command_for_venv(task.command, self._venv_python_path(Path(plan.venv_path)))
        if rewritten == task.command:
            return task
        return CommandTask(
            step_number=task.step_number,
            command=rewritten,
            cwd=task.cwd,
            reason=task.reason,
            expectation=task.expectation,
        )

    def _rewrite_command_for_venv(self, command: str, venv_python: Path) -> str:
        stripped = command.strip()
        lowered = stripped.lower()
        venv_python_text = f'"{venv_python}"'

        prefix_map = {
            "python -m pip ": f"{venv_python_text} -m pip ",
            "python3 -m pip ": f"{venv_python_text} -m pip ",
            "pip install ": f"{venv_python_text} -m pip install ",
            "pip3 install ": f"{venv_python_text} -m pip install ",
            "uv pip install ": f"{venv_python_text} -m pip install ",
            "pytest ": f"{venv_python_text} -m pytest ",
        }
        for prefix, replacement in prefix_map.items():
            if lowered.startswith(prefix):
                return replacement + stripped[len(prefix):]
        if lowered == "pytest":
            return f"{venv_python_text} -m pytest"
        if lowered.startswith("python "):
            return f"{venv_python_text} {stripped[7:]}"
        if lowered.startswith("python3 "):
            return f"{venv_python_text} {stripped[8:]}"
        return command

    def _venv_command_cwd(self, plan: InstallPlan, install_root: Path) -> str:
        venv_path = Path(plan.venv_path)
        return str(venv_path.parent if venv_path.parent != Path('.') else install_root)

    def _torch_command_cwd(self, plan: InstallPlan, install_root: Path) -> str:
        venv_path = Path(plan.venv_path)
        return str(venv_path.parent if venv_path.parent != Path('.') else install_root)

    def _is_repeated_command(self, history: list[CommandResult], command: str, cwd: str) -> bool:
        matching = 0
        for item in reversed(history):
            if item.command.strip() == command.strip() and item.cwd == cwd:
                matching += 1
            else:
                break
        return matching >= REPEAT_LIMIT

    def _resolve_install_root(self, repository_url: str, target_dir: Path | None) -> Path:
        if target_dir is not None:
            return target_dir.resolve()
        repository_name = repository_url.rstrip("/").split("/")[-1]
        if repository_name.endswith(".git"):
            repository_name = repository_name[:-4]
        return (Path.cwd() / "installed_repositories" / repository_name).resolve()

    def _venv_python_path(self, venv_path: Path) -> Path:
        if os.name == "nt":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    def _activation_commands(self, venv_path: Path) -> list[tuple[str, str]]:
        if os.name == "nt":
            return [
                ("PowerShell activation command", f'& "{venv_path / "Scripts" / "Activate.ps1"}"'),
                ("cmd.exe activation command", f'"{venv_path / "Scripts" / "activate.bat"}"'),
            ]
        return [("Shell activation command", f'source "{venv_path / "bin" / "activate"}"')]

    def _looks_like_venv(self, path: Path) -> bool:
        return (
            (path / "Scripts" / "python.exe").exists()
            or (path / "bin" / "python").exists()
            or (path / "pyvenv.cfg").exists()
        )

    def _llm_prompt(self, label: str, content: str) -> None:
        self._headline(f"LLM Prompt: {label}")
        self._prompt(llm_visible_payload(label, content) + "\n")

    def _llm_response(self, label: str, content: str) -> None:
        self._headline(f"LLM Response: {label}")
        self._response(content + "\n")

    def _print_probe_details(self, profile: SystemProfile) -> None:
        self._detail(profile.to_prompt_block())
        for record in profile.records:
            self._command(record.command + "\n")
            output = record.output or "<no output>"
            self._output(output + ("\n" if not output.endswith("\n") else ""))

    def _headline(self, text: str) -> None:
        print(paint(f"\n== {text} ==", PARENT_THEME.headline, bold=True), flush=True)

    def _detail(self, text: str) -> None:
        print(paint(text, PARENT_THEME.detail), flush=True)

    def _success(self, text: str) -> None:
        print(paint(text, PARENT_THEME.success, bold=True), flush=True)

    def _warning(self, text: str) -> None:
        print(paint(text, PARENT_THEME.warning, bold=True), flush=True)

    def _error(self, text: str) -> None:
        print(paint(text, PARENT_THEME.error, bold=True), flush=True)

    def _command(self, text: str) -> None:
        sys.stdout.write(paint(text, PARENT_THEME.command, bold=True))
        sys.stdout.flush()

    def _output(self, text: str) -> None:
        sys.stdout.write(paint(text, PARENT_THEME.output))
        sys.stdout.flush()

    def _prompt(self, text: str) -> None:
        sys.stdout.write(paint(text, PARENT_THEME.prompt))
        sys.stdout.flush()

    def _response(self, text: str) -> None:
        sys.stdout.write(paint(text, PARENT_THEME.response))
        sys.stdout.flush()





