from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"


class Color:
    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"
    BRIGHT_BLACK = "\x1b[90m"
    BRIGHT_RED = "\x1b[91m"
    BRIGHT_GREEN = "\x1b[92m"
    BRIGHT_YELLOW = "\x1b[93m"
    BRIGHT_BLUE = "\x1b[94m"
    BRIGHT_MAGENTA = "\x1b[95m"
    BRIGHT_CYAN = "\x1b[96m"
    BRIGHT_WHITE = "\x1b[97m"


@dataclass(frozen=True)
class Theme:
    headline: str
    detail: str
    command: str
    output: str
    prompt: str
    response: str
    success: str
    warning: str
    error: str


PARENT_THEME = Theme(
    headline=Color.BRIGHT_CYAN,
    detail=Color.BRIGHT_BLACK,
    command=Color.BRIGHT_BLUE,
    output=Color.BRIGHT_GREEN,
    prompt=Color.BRIGHT_MAGENTA,
    response=Color.BRIGHT_YELLOW,
    success=Color.BRIGHT_GREEN,
    warning=Color.BRIGHT_YELLOW,
    error=Color.BRIGHT_RED,
)

WORKER_THEME = Theme(
    headline=Color.BRIGHT_CYAN,
    detail=Color.BRIGHT_BLACK,
    command=Color.BRIGHT_CYAN,
    output=Color.BRIGHT_GREEN,
    prompt=Color.BRIGHT_CYAN,
    response=Color.BRIGHT_GREEN,
    success=Color.BRIGHT_GREEN,
    warning=Color.BRIGHT_GREEN,
    error=Color.BRIGHT_GREEN,
)


def enable_ansi() -> None:
    if os.name != "nt":
        return

    kernel32 = ctypes.windll.kernel32
    handles = (-11, -12)
    enable_virtual_terminal_processing = 0x0004

    for handle_id in handles:
        handle = kernel32.GetStdHandle(handle_id)
        if handle == 0:
            continue
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            continue
        kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing)


def paint(text: str, color: str, *, bold: bool = False, dim: bool = False) -> str:
    prefix = color
    if bold:
        prefix += BOLD
    if dim:
        prefix += DIM
    return f"{prefix}{text}{RESET}"
