from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    commands_dir: Path
    results_dir: Path
    artifacts_dir: Path
    events_path: Path
    status_path: Path
    config_path: Path
    launcher_path: Path

    @classmethod
    def create(cls) -> "SessionPaths":
        root = Path(tempfile.mkdtemp(prefix="github-install-assistant-"))
        commands_dir = root / "commands"
        results_dir = root / "results"
        artifacts_dir = root / "artifacts"
        commands_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            commands_dir=commands_dir,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            events_path=root / "events.jsonl",
            status_path=root / "status.json",
            config_path=root / "session_config.json",
            launcher_path=root / "launch_worker.cmd",
        )

    def command_path(self, step_number: int) -> Path:
        return self.commands_dir / f"{step_number:04d}.json"

    def result_path(self, step_number: int) -> Path:
        return self.results_dir / f"{step_number:04d}.json"

    def output_path(self, step_number: int) -> Path:
        return self.artifacts_dir / f"{step_number:04d}.log"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_event(events_path: Path, event_type: str, payload: dict[str, Any]) -> None:
    record = {"event": event_type, **payload}
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
