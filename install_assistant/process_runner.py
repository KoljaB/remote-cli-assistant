from __future__ import annotations

import locale
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ENCODING = locale.getpreferredencoding(False) or "utf-8"
SHELL_EXECUTABLE = os.environ.get("COMSPEC", "cmd.exe") if os.name == "nt" else None


@dataclass
class CompletedCommand:
    command: str
    exit_code: int
    output: str
    duration_seconds: float


def run_cmd_capture(command: str, *, cwd: Path | None = None) -> CompletedCommand:
    started = time.monotonic()
    run_kwargs = {
        "cwd": str(cwd) if cwd else None,
        "capture_output": True,
        "text": True,
        "encoding": ENCODING,
        "errors": "replace",
        "shell": True,
    }
    if SHELL_EXECUTABLE is not None:
        run_kwargs["executable"] = SHELL_EXECUTABLE
    process = subprocess.run(command, **run_kwargs)
    output = process.stdout
    if process.stderr:
        output += process.stderr
    return CompletedCommand(
        command=command,
        exit_code=process.returncode,
        output=output,
        duration_seconds=time.monotonic() - started,
    )


def run_powershell_capture(command: str, *, cwd: Path | None = None) -> CompletedCommand:
    started = time.monotonic()
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding=ENCODING,
        errors="replace",
    )
    output = process.stdout
    if process.stderr:
        output += process.stderr
    return CompletedCommand(
        command=command,
        exit_code=process.returncode,
        output=output,
        duration_seconds=time.monotonic() - started,
    )


def stream_cmd_command(
    command: str,
    *,
    cwd: Path,
    on_line: Callable[[str], None],
) -> CompletedCommand:
    started = time.monotonic()
    popen_kwargs = {
        "cwd": str(cwd),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "encoding": ENCODING,
        "errors": "replace",
        "bufsize": 1,
        "shell": True,
    }
    if SHELL_EXECUTABLE is not None:
        popen_kwargs["executable"] = SHELL_EXECUTABLE
    process = subprocess.Popen(command, **popen_kwargs)

    output_chunks: list[str] = []
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        output_chunks.append(line)
        on_line(line)

    trailing = process.stdout.read()
    if trailing:
        output_chunks.append(trailing)
        on_line(trailing)

    process.stdout.close()
    exit_code = process.wait()
    return CompletedCommand(
        command=command,
        exit_code=exit_code,
        output="".join(output_chunks),
        duration_seconds=time.monotonic() - started,
    )
