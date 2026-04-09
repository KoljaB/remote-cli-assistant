from __future__ import annotations

import sys
import time
from pathlib import Path

from .ansi import WORKER_THEME, enable_ansi, paint
from .models import CommandResult, CommandTask
from .process_runner import stream_cmd_command
from .session_io import append_event, read_json, write_json


POLL_INTERVAL_SECONDS = 0.10


def run_worker(config_path: Path) -> int:
    enable_ansi()
    config = read_json(config_path)
    commands_dir = Path(config["commands_dir"])
    results_dir = Path(config["results_dir"])
    artifacts_dir = Path(config["artifacts_dir"])
    events_path = Path(config["events_path"])
    status_path = Path(config["status_path"])

    next_step = 1
    while True:
        task_path = commands_dir / f"{next_step:04d}.json"
        if task_path.exists():
            task = CommandTask.from_dict(read_json(task_path))
            _execute_task(task, results_dir, artifacts_dir, events_path)
            next_step += 1
            continue

        if status_path.exists():
            status = read_json(status_path)
            state = status.get("state", "running")
            if state != "running":
                append_event(events_path, "worker_complete", {"state": state, "message": status.get("message", "")})
                return 0

        time.sleep(POLL_INTERVAL_SECONDS)


def _execute_task(
    task: CommandTask,
    results_dir: Path,
    artifacts_dir: Path,
    events_path: Path,
) -> None:
    cwd = Path(task.cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / f"{task.step_number:04d}.log"
    output_path.write_text("", encoding="utf-8")

    _write_command(f"[{task.step_number:02d}] {task.command}\n")
    append_event(
        events_path,
        "command_started",
        {
            "step_number": task.step_number,
            "command": task.command,
            "cwd": task.cwd,
            "reason": task.reason,
            "expectation": task.expectation,
        },
    )

    def on_output(text: str) -> None:
        _write_output(text)
        with output_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)
        append_event(
            events_path,
            "command_output",
            {
                "step_number": task.step_number,
                "text": text,
            },
        )

    completed = stream_cmd_command(task.command, cwd=cwd, on_line=on_output)
    result = CommandResult(
        step_number=task.step_number,
        command=task.command,
        cwd=task.cwd,
        reason=task.reason,
        expectation=task.expectation,
        exit_code=completed.exit_code,
        output=completed.output,
        duration_seconds=completed.duration_seconds,
    )
    write_json(results_dir / f"{task.step_number:04d}.json", result.to_dict())
    append_event(
        events_path,
        "command_finished",
        {
            "step_number": task.step_number,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
        },
    )
    _write_output(f"\nExit code: {result.exit_code}\n\n")


def _write_command(text: str) -> None:
    sys.stdout.write(paint(text, WORKER_THEME.command, bold=True))
    sys.stdout.flush()


def _write_output(text: str) -> None:
    sys.stdout.write(paint(text, WORKER_THEME.output))
    sys.stdout.flush()
