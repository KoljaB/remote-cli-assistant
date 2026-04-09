from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from .session_io import SessionPaths, write_json


def launch_visible_worker(session: SessionPaths, *, label: str) -> subprocess.Popen[bytes]:
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

    if os.name == "nt":
        launcher_path = session.launcher_path
        launcher_contents = "\r\n".join(
            [
                "@echo off",
                "setlocal",
                f"title {label}",
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
    launcher_path.write_text(
        "#!/bin/sh\n"
        f"printf '\\033]0;{label}\\007'\n"
        f"{worker_command}\n"
        "exec \"${SHELL:-/bin/sh}\"\n",
        encoding="utf-8",
    )
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
