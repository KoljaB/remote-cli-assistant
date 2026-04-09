from __future__ import annotations

import platform
import re

from .models import ProbeRecord, SystemProfile
from .process_runner import run_cmd_capture, run_powershell_capture


def collect_system_profile() -> SystemProfile:
    records: list[ProbeRecord] = []
    current_platform = platform.system()

    if current_platform == "Windows":
        cmd_specs = [
            ("python_version", "python --version"),
            ("python_locations", "where.exe python"),
            ("python_launcher_versions", "py -0p"),
            ("git_version", "git --version"),
            ("nvidia_smi", "nvidia-smi"),
            ("nvidia_query", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            ("nvcc_version", "nvcc --version"),
            ("cuda_path", "echo %CUDA_PATH%"),
        ]
        ps_specs = [
            ("os_version", "[System.Environment]::OSVersion.VersionString"),
            ("windows_product", "Get-ComputerInfo -Property WindowsProductName,WindowsVersion | Format-List"),
            ("processor_count", "[System.Environment]::ProcessorCount"),
        ]
    else:
        cmd_specs = [
            ("python_version", "python3 --version || python --version"),
            ("python_locations", "which -a python3 python 2>/dev/null"),
            ("python_launcher_versions", "python3 --version 2>/dev/null || python --version"),
            ("git_version", "git --version"),
            ("nvidia_smi", "nvidia-smi"),
            ("nvidia_query", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            ("nvcc_version", "nvcc --version"),
            ("cuda_path", "printf '%s\n' \"$CUDA_PATH\""),
            ("os_version", "uname -a"),
            ("processor_count", "getconf _NPROCESSORS_ONLN"),
        ]
        ps_specs = []

    for name, command in cmd_specs:
        result = run_cmd_capture(command)
        records.append(
            ProbeRecord(
                name=name,
                command=command,
                exit_code=result.exit_code,
                output=result.output.strip(),
            )
        )

    for name, command in ps_specs:
        result = run_powershell_capture(command)
        records.append(
            ProbeRecord(
                name=name,
                command=command,
                exit_code=result.exit_code,
                output=result.output.strip(),
            )
        )

    profile = SystemProfile(records=records)
    profile.platform_family = current_platform
    profile.python_default = _record_output(records, "python_version") or "Unknown"
    profile.python_locations = _split_nonempty_lines(_record_output(records, "python_locations"))
    profile.python_launcher_versions = _split_nonempty_lines(_record_output(records, "python_launcher_versions"))
    profile.git_version = _record_output(records, "git_version") or "Unknown"
    profile.os_version = _record_output(records, "os_version") or platform.platform()
    profile.operating_system = _extract_os_name(current_platform, records)
    profile.processor_count = _record_output(records, "processor_count") or "Unknown"
    profile.cuda_path = _normalize_cuda_path(_record_output(records, "cuda_path"))

    gpu_line = _first_nonempty_line(_record_output(records, "nvidia_query"))
    if gpu_line:
        parts = [part.strip() for part in gpu_line.split(",")]
        if len(parts) >= 3:
            profile.gpu_name = parts[0]
            profile.gpu_vram = parts[1]
            profile.nvidia_driver = parts[2]

    profile.cuda_runtime_version = _extract_cuda_runtime_version(_record_output(records, "nvidia_smi"))
    profile.cuda_toolkit_version = _extract_nvcc_version(_record_output(records, "nvcc_version"))

    return profile


def _record_output(records: list[ProbeRecord], name: str) -> str:
    for record in records:
        if record.name == name and record.output:
            return record.output
    return ""


def _split_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _extract_os_name(current_platform: str, records: list[ProbeRecord]) -> str:
    if current_platform == "Windows":
        text = _record_output(records, "windows_product")
        for line in text.splitlines():
            if "WindowsProductName" in line:
                return line.split(":", 1)[-1].strip() or current_platform
    return current_platform


def _extract_cuda_runtime_version(text: str) -> str:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", text or "")
    return match.group(1) if match else "Unknown"


def _extract_nvcc_version(text: str) -> str:
    match = re.search(r"release\s+([0-9.]+)", text or "")
    return match.group(1) if match else "Unknown"


def _normalize_cuda_path(text: str) -> str:
    value = _first_nonempty_line(text)
    if not value:
        return "Unknown"
    if value in {"%CUDA_PATH%", "$CUDA_PATH"}:
        return "Unknown"
    return value
