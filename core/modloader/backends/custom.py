"""Explicit local argv templates for an already installed OpenAI-compatible server.

Registration and refresh inspect files and configuration only. The selected program
is executed exclusively by RuntimeManager when the user starts a model instance.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import uuid
from urllib.parse import urlsplit

from ..models import Compatibility, LaunchSpec, ParameterField, RuntimeInstallation
from .adapters import BackendAdapter, OFFLINE_ENV
from ..platforms import (linux_path, runtime_distribution, path_stat, model_target_path,
                         validate_model_runtime, model_exists, location_of)


PLACEHOLDERS = {"model_path", "model_dir", "host", "port", "alias", "context_length"}
GENERATION_PARAMETERS = ("temperature", "top_p", "max_tokens", "top_k", "repeat_penalty", "seed")
DEFAULT_GENERATION_PARAMETERS = ("temperature", "top_p", "max_tokens")
CONFIG_KEYS = {"args", "cwd", "env", "health_path", "chat_path", "api_model_name", "generation_parameters", "stream_usage"}
PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z_0-9]*)\}\}|\{([A-Za-z_][A-Za-z_0-9]*)\}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
SENSITIVE = re.compile(r"api[-_]?key|access[-_]?token|auth[-_]?token|password|passwd|secret|credential|authorization", re.I)
SECRET_TOKEN = re.compile(r"(?:^|[-_])(?:hf[-_]|hugging[-_]face[-_]hub[-_])?token(?:$|[-_])", re.I)
SHELLS = {"cmd", "powershell", "pwsh", "wsl", "bash", "sh", "zsh", "fish", "ssh", "plink", "mshta"}
HOST_FLAGS = {"--host", "--bind", "--bind-address", "--listen-address", "--listen"}
# These common options and the template's own binding options cannot be appended
# by a profile. In particular, -m in a Python module template is not globally banned.
BINDING_FLAGS = HOST_FLAGS | {"--port", "-p", "--model", "--model-path", "--model-dir", "--model-id", "--served-model-name", "--alias", "--listen", "--config"}


def _text(value, label, *, empty=False):
    if not isinstance(value, str) or any(character in value for character in ("\0", "\r", "\n")) or not empty and not value.strip():
        raise ValueError(f"{label}必须是{'可为空的' if empty else '非空'}单行字符串")
    return value


def _sensitive_name(value):
    return bool(SENSITIVE.search(value) or SECRET_TOKEN.search(value.strip("-")))


def _credential(value):
    if re.search(r"(?:authorization|api[-_]?key|password|passwd|secret|access[-_]?token|auth[-_]?token)[\s\"']*[:=]", value, re.I):
        return True
    try:
        parsed = urlsplit(value)
        return bool(parsed.scheme and (parsed.username is not None or parsed.password is not None))
    except ValueError:
        return False


def _argv(values, label):
    if not isinstance(values, list):
        raise ValueError(f"{label}必须是字符串 argv 数组，每项为一个参数")
    for value in values:
        _text(value, label, empty=True)
        if value.startswith("-") and _sensitive_name(value.split("=", 1)[0]) or _credential(value):
            raise ValueError("自定义引擎配置不能保存密钥、令牌或密码；请移除敏感参数")
    return list(values)


def _template(value, replacements=None):
    # Only identifier-shaped placeholders are interpreted. Ordinary JSON braces
    # remain untouched, including nested }}; {{name}} means literal {name}.
    found = set()
    def replace(match):
        if match[1] is not None:
            return "{" + match[1] + "}"
        key = match[2]
        if key not in PLACEHOLDERS:
            raise ValueError("未知占位符：{" + key + "}")
        found.add(key)
        return str(replacements[key]) if replacements is not None else match[0]
    return PLACEHOLDER.sub(replace, value), found


def _local_path(value, label, *, directory=False, check=True):
    _text(value, label)
    path = Path(value).expanduser().resolve()
    if str(path).startswith(("\\\\", "//")):
        raise ValueError(f"{label}必须在本机，不能使用网络或 WSL 路径")
    if check and not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label}{'目录' if directory else '文件'}不存在：{path}")
    return str(path)


def _api_path(value, label):
    _text(value, label)
    parsed = urlsplit(value)
    if not value.startswith("/") or value.startswith("//") or parsed.scheme or parsed.netloc or "\\" in value or parsed.query or parsed.fragment:
        raise ValueError(f"{label}必须是以 / 开头的本机 HTTP 路径，不能包含外链、查询或片段")
    return value


def _binding_flags(args):
    result, previous = set(), None
    for arg in args:
        if arg.startswith("-"):
            previous = arg.split("=", 1)[0]
        _, fields = _template(arg)
        if fields & {"model_path", "model_dir", "host", "port", "alias"} and previous:
            result.add(previous)
    return result


def custom_config(runtime, *, check_paths=True):
    if runtime.platform == 'wsl2':
        distribution = runtime_distribution(runtime)
        executable = linux_path(runtime.path)
        program = PurePosixPath(executable)
        if check_paths:
            status = path_stat(distribution, executable)
            if not status['is_file'] or not status['executable']:
                raise ValueError('WSL 引擎程序不存在或不可执行')
    else:
        executable = _local_path(runtime.path, "引擎程序", check=check_paths)
        program = Path(executable)
    if program.stem.lower() in SHELLS or (runtime.platform == 'windows' and os.name == "nt" and program.suffix.lower() != ".exe"):
        raise ValueError("请选择本机引擎 EXE 或 Python EXE，不支持 shell、WSL、SSH 或启动脚本")
    if not isinstance(runtime.metadata, dict) or not isinstance(runtime.metadata.get("custom"), dict):
        raise ValueError("缺少自定义引擎 metadata.custom 配置")
    raw = runtime.metadata["custom"]
    if set(raw) - CONFIG_KEYS:
        raise ValueError("未知自定义引擎配置项：" + "、".join(sorted(set(raw) - CONFIG_KEYS)))
    args = _argv(raw.get("args"), "启动参数")
    fields = set().union(*(_template(arg)[1] for arg in args)) if args else set()
    if not {"model_path", "port"} <= fields:
        raise ValueError("启动参数必须引用 {model_path} 和 {port}，以使用所选本地模型及实例端口")
    protected = _binding_flags(args)
    for flag in protected:
        if sum(arg.split("=", 1)[0] == flag for arg in args) > 1:
            raise ValueError(f"绑定选项 {flag} 不能在启动模板中重复，避免覆盖所选模型或本机地址")
    for index, arg in enumerate(args):
        flag = arg.split("=", 1)[0]
        if flag in HOST_FLAGS:
            value = arg.split("=", 1)[1] if "=" in arg else args[index + 1] if index + 1 < len(args) else ""
            loopback = {"{host}", "127.0.0.1", "localhost", "::1", "[::1]"}
            local_endpoint = {host + ":{port}" for host in loopback if host != "::1"}
            if value not in loopback | local_endpoint:
                raise ValueError(f"{flag} 只能绑定本机回环地址；请使用 {{host}}（127.0.0.1）")
    cwd = raw.get("cwd") or str(program.parent)
    if runtime.platform == 'wsl2':
        cwd = linux_path(cwd)
        if check_paths and not path_stat(distribution, cwd)['is_dir']:
            raise ValueError('WSL 工作目录不存在')
    else:
        cwd = _local_path(cwd, "工作", directory=True, check=check_paths)
    env = raw.get("env", {})
    if not isinstance(env, dict):
        raise ValueError("环境变量必须是名称到字符串值的对象")
    for key, value in env.items():
        if not isinstance(key, str) or not ENV_NAME.fullmatch(key):
            raise ValueError("环境变量名称只能包含字母、数字和下划线，且不能以数字开头")
        _text(value, "环境变量值", empty=True)
        if _sensitive_name(key) or _credential(value):
            raise ValueError("自定义引擎配置不能明文保存密钥、令牌或密码环境变量")
        if key.upper() in OFFLINE_ENV and value != OFFLINE_ENV[key.upper()]:
            raise ValueError(f"{key} 不能关闭本地离线模式")
    generation = raw.get("generation_parameters", list(DEFAULT_GENERATION_PARAMETERS))
    if not isinstance(generation, list) or any(not isinstance(item, str) or item not in GENERATION_PARAMETERS for item in generation):
        raise ValueError("生成参数仅支持 temperature、top_p、max_tokens、top_k、repeat_penalty、seed")
    if len(generation) != len(set(generation)):
        raise ValueError("生成参数不能重复")
    stream_usage = raw.get("stream_usage", False)
    if type(stream_usage) is not bool:
        raise ValueError("stream_usage 必须为布尔值")
    api_model_name = _text(raw.get("api_model_name", "{alias}"), "API 模型名")
    _template(api_model_name)
    return {"args": args, "cwd": cwd, "env": dict(env),
            "health_path": _api_path(raw.get("health_path", "/v1/models"), "健康检查路径"),
            "chat_path": _api_path(raw.get("chat_path", "/v1/chat/completions"), "对话路径"),
            "api_model_name": api_model_name, "generation_parameters": list(generation), "stream_usage": stream_usage}


def import_custom_runtime(path, metadata=None, platform="windows"):
    runtime = RuntimeInstallation(id=uuid.uuid4().hex, backend="custom", name="自定义引擎", path=path, metadata=dict(metadata or {}), platform=platform)
    config = custom_config(runtime, check_paths=platform != 'wsl2')
    runtime.path = linux_path(runtime.path) if platform == 'wsl2' else str(Path(runtime.path).expanduser().resolve())
    runtime.metadata = {"custom": config, "validation": "configuration" if platform == 'wsl2' else "files_and_configuration",
                        **({'distribution': runtime_distribution(runtime)} if platform == 'wsl2' else {})}
    return runtime


def required_context(config):
    return any("context_length" in _template(value)[1] for value in [*config["args"], config["api_model_name"]])


def custom_fields(runtime=None):
    if runtime is None or not required_context(custom_config(runtime, check_paths=False)):
        return []
    return [ParameterField(key="context_length", label="上下文长度", type="number", integer=True, step=1)]


def custom_extra(profile, config):
    extra = _argv(profile.extra_args, "高级参数")
    protected = BINDING_FLAGS | _binding_flags(config["args"])
    for value in extra:
        if value.split("=", 1)[0] in protected:
            raise ValueError(f"高级参数 {value.split('=', 1)[0]} 不能覆盖启动模板的模型、端口或本机绑定；请编辑引擎模板")
        if _template(value)[1]:
            raise ValueError("高级参数不展开占位符；请把占位符放在引擎启动模板中")
    return extra


def validate_custom_profile(model, profile, runtime):
    validate_model_runtime(model, runtime)
    config = custom_config(runtime, check_paths=runtime.platform != 'wsl2')
    context_required = required_context(config)
    allowed = {"startup_timeout"} | ({"context_length"} if context_required else set())
    unknown = {key for key, value in profile.parameters.items() if key not in allowed and value is not None and value != ""}
    if unknown:
        raise ValueError("自定义引擎不支持这些加载参数：" + "、".join(sorted(unknown)) + "；请在引擎模板中设置实际 argv")
    value = profile.parameters.get("context_length")
    if context_required and (value is None or isinstance(value, str) and not value.strip()):
        raise ValueError("启动模板引用了 {context_length}，请填写该引擎支持的上下文长度")
    if location_of(model).platform == "windows" and not Path(model.path).exists():
        raise ValueError("模型路径不存在")
    custom_extra(profile, config)


class CustomAdapter(BackendAdapter):
    id, label = "custom", "自定义引擎"

    def parameters(self):
        return []

    def inspect(self, model, runtime):
        validate_model_runtime(model, runtime)
        config = custom_config(runtime)
        capabilities = {"input": ["text"], "output": ["text"], "streaming": True, "tools": False,
                        "tools_status": "unverified", "reasoning": "unverified",
                        "generation_parameters": config["generation_parameters"], "stream_usage": config["stream_usage"]}
        if location_of(model).platform == 'wsl2' and model.format != 'custom':
            from ..assets import inspect_asset
            try:
                inspect_asset(model.path, location=model.location)
            except ValueError as error:
                return Compatibility(status='unsupported', reasons=[str(error)], capabilities=capabilities)
        if not model_exists(model) or model.metadata.get("missing_shards"):
            return Compatibility(status="unsupported", reasons=["本地模型路径不存在或登记的模型分片不完整"], capabilities=capabilities)
        return Compatibility(status="unverified", reasons=["文件与启动模板已检查；模型格式、硬件、依赖和 OpenAI 对话兼容性需由此引擎实际加载确认"], capabilities=capabilities)

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        config = custom_config(runtime)
        target = model_target_path(model, runtime)
        path = PurePosixPath(target) if runtime.platform == 'wsl2' else Path(target)
        is_dir = path_stat(runtime_distribution(runtime), target)['is_dir'] if runtime.platform == 'wsl2' else path.is_dir()
        replacements = {"model_path": str(path), "model_dir": str(path if is_dir else path.parent),
                        "host": "127.0.0.1", "port": context.port, "alias": profile.alias,
                        "context_length": profile.parameters.get("context_length", "")}
        args = [_template(value, replacements)[0] for value in config["args"]]
        model_name = _template(config["api_model_name"], replacements)[0]
        if not model_name.strip():
            raise ValueError("展开后的 API 模型名不能为空")
        return LaunchSpec(platform=runtime.platform, distribution=runtime_distribution(runtime) if runtime.platform == 'wsl2' else None,
                          argv=[runtime.path, *args, *custom_extra(profile, config)], cwd=config["cwd"],
                          env={**config["env"], **OFFLINE_ENV}, health_path=config["health_path"], chat_path=config["chat_path"],
                          model_name=model_name, capabilities=self.inspect(model, runtime).capabilities)
