from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class ModelLocation(BaseModel):
    platform: Literal["windows", "wsl2"] = "windows"
    distribution: str | None = None

    @model_validator(mode="after")
    def validate_location(self):
        if self.platform == "wsl2" and (not self.distribution or not self.distribution.strip() or any(c in self.distribution for c in "\0\r\n/\\")):
            raise ValueError("WSL 模型位置必须指定发行版名称")
        return self


class ModelAsset(BaseModel):
    location: ModelLocation | None = None
    id: str
    name: str
    path: str
    format: str
    architecture: str | None = None
    quantization: str | None = None
    size_bytes: int = 0
    files: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=lambda: {"kind": "local"})
    managed: bool = False
    warnings: list[str] = Field(default_factory=list)


class LoadProfile(BaseModel):
    id: str
    model_id: str
    name: str
    backend: str
    runtime_id: str
    alias: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    revision: int = 1
    chat_quick_load: bool = False
    parameter_editor_version: Literal[1] | None = None


class RuntimeInstallation(BaseModel):
    platform: Literal["windows", "wsl2"] = "windows"

    @model_validator(mode="before")
    @classmethod
    def legacy_platform(cls, value):
        if isinstance(value, dict) and "platform" not in value:
            value = {**value, "platform": "wsl2" if value.get("backend") == "vllm" else "windows"}
        return value
    id: str
    backend: str
    name: str
    path: str
    version: str = "unknown"
    managed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Compatibility(BaseModel):
    status: Literal["supported", "unverified", "unsupported"] = "unverified"
    reasons: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=lambda: {"input": ["text"], "output": ["text"], "streaming": True, "tools": False})


class ParameterField(BaseModel):
    key: str
    label: str
    type: Literal["text", "number", "boolean", "select", "file"] = "text"
    default: Any = None
    options: list[Any] = Field(default_factory=list)
    description: str = ""
    advanced: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    exclusive_minimum: bool = False
    integer: bool = False
    step: float | None = None
    special_values: list[int | str] = Field(default_factory=list)


class LaunchContext(BaseModel):
    port: int
    instance_id: str
    work_dir: str
    data_dir: str


class LaunchSpec(BaseModel):
    platform: Literal["windows", "wsl2"] = "windows"
    distribution: str | None = None
    guest_record: str | None = None
    argv: list[str]
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)
    health_path: str = "/v1/models"
    chat_path: str = "/v1/chat/completions"
    health_expected: dict[str, Any] = Field(default_factory=dict)
    model_name: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    # For WSL, stop only this instance's recorded Linux process group.
    cleanup_argv: list[str] = Field(default_factory=list)
    startup_timeout: float = 600


class RuntimeInstance(BaseModel):
    platform: Literal["windows", "wsl2"] = "windows"
    distribution: str | None = None
    guest_pid: int | None = None
    id: str
    model_id: str
    profile_id: str
    profile_revision: int
    alias: str
    backend: str
    runtime_id: str
    state: str = "starting"
    port: int
    pid: int | None = None
    started_at: str
    error: str | None = None
    active_requests: int = 0
    capabilities: dict[str, Any] = Field(default_factory=dict)
    phase: str | None = None
    message: str | None = None


class SourceFile(BaseModel):
    path: str
    size: int | None = None
    role: str = "other"
    group_id: str | None = None
    reason: str | None = None


class SourceRepository(BaseModel):
    source: str
    repo_id: str
    revision: str
    name: str
    description: str = ""
    url: str = ""
    files: list[SourceFile] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    groups: list[dict[str, Any]] = Field(default_factory=list)


class DownloadRequest(BaseModel):
    source: str
    repo_id: str = ""
    revision: str = "main"
    files: list[str] = Field(default_factory=list)
    url: str = ""
    filename: str = ""
    # Destination is assigned by the management core, never an arbitrary remote path.
    destination: str = ""
    include_vision: bool = False
    checked_files: list[SourceFile] | None = None


class AppPaths:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.models = self.root / "models"
        self.runtimes = self.root / "runtimes"
        self.work = self.root / "work"
        self.logs = self.root / "logs"
        for path in (self.root, self.models, self.runtimes, self.work, self.logs):
            path.mkdir(parents=True, exist_ok=True)
