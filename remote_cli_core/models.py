from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProbeRecord:
    name: str
    command: str
    exit_code: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemProfile:
    platform_family: str = "Unknown"
    operating_system: str = "Unknown"
    os_version: str = "Unknown"
    python_default: str = "Unknown"
    python_locations: list[str] = field(default_factory=list)
    python_launcher_versions: list[str] = field(default_factory=list)
    git_version: str = "Unknown"
    gpu_name: str = "Unknown"
    gpu_vram: str = "Unknown"
    nvidia_driver: str = "Unknown"
    cuda_runtime_version: str = "Unknown"
    cuda_toolkit_version: str = "Unknown"
    cuda_path: str = "Unknown"
    processor_count: str = "Unknown"
    records: list[ProbeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_family": self.platform_family,
            "operating_system": self.operating_system,
            "os_version": self.os_version,
            "python_default": self.python_default,
            "python_locations": list(self.python_locations),
            "python_launcher_versions": list(self.python_launcher_versions),
            "git_version": self.git_version,
            "gpu_name": self.gpu_name,
            "gpu_vram": self.gpu_vram,
            "nvidia_driver": self.nvidia_driver,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_toolkit_version": self.cuda_toolkit_version,
            "cuda_path": self.cuda_path,
            "processor_count": self.processor_count,
            "records": [record.to_dict() for record in self.records],
        }

    def to_prompt_block(self) -> str:
        lines = [
            f"Platform family: {self.platform_family}",
            f"Operating system: {self.operating_system}",
            f"OS version: {self.os_version}",
            f"Default Python: {self.python_default}",
            f"Python locations: {', '.join(self.python_locations) if self.python_locations else 'Unknown'}",
            "Python launcher versions: "
            f"{', '.join(self.python_launcher_versions) if self.python_launcher_versions else 'Unknown'}",
            f"Git version: {self.git_version}",
            f"GPU: {self.gpu_name}",
            f"GPU VRAM: {self.gpu_vram}",
            f"NVIDIA driver: {self.nvidia_driver}",
            f"CUDA runtime version: {self.cuda_runtime_version}",
            f"CUDA toolkit version: {self.cuda_toolkit_version}",
            f"CUDA_PATH: {self.cuda_path}",
            f"Logical processor count: {self.processor_count}",
        ]
        return "\n".join(lines)

    def probe_output(self, name: str) -> str:
        for record in self.records:
            if record.name == name:
                return record.output or "<no output>"
        return "<no output>"

    def torch_probe_block(self) -> str:
        return "\n".join(
            [
                "nvidia-smi output:",
                self.probe_output("nvidia_smi"),
                "",
                "nvcc --version output:",
                self.probe_output("nvcc_version"),
                "",
                "CUDA_PATH probe output:",
                self.probe_output("cuda_path"),
            ]
        )


@dataclass
class CommandTask:
    step_number: int
    command: str
    cwd: str
    reason: str
    expectation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CommandTask":
        return cls(**payload)


@dataclass
class CommandResult:
    step_number: int
    command: str
    cwd: str
    reason: str
    expectation: str
    exit_code: int
    output: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CommandResult":
        return cls(**payload)


@dataclass
class InstallPlan:
    research_summary: str
    install_root: str
    venv_path: str
    venv_creation_command: str
    requires_torch: bool
    torch_install_command: str
    torch_reasoning: str
    steps_overview: list[str]
    success_signals: list[str]
    verification_command: str
    verification_cwd: str
    finish_rule: str
    first_command: str
    first_cwd: str
    first_reason: str
    first_expectation: str
    source_urls: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InstallPlan":
        return cls(**payload)


@dataclass
class NextAction:
    status: str
    reason: str
    next_command: str
    cwd: str
    expectation: str
    progress_update: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NextAction":
        return cls(**payload)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatMessage":
        return cls(**payload)


@dataclass
class AssistantDecision:
    status: str
    assistant_message: str
    reason: str
    next_command: str
    cwd: str
    expectation: str
    goal_status: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AssistantDecision":
        return cls(**payload)


@dataclass
class WorkerHello:
    worker_label: str
    default_cwd: str
    system_profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerHello":
        return cls(**payload)
