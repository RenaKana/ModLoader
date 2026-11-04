from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath, PurePosixPath
import re
import shlex
import shutil
import subprocess
import uuid
from typing import Any

from ..models import Compatibility, LaunchContext, LaunchSpec, LoadProfile, ModelAsset, ParameterField, RuntimeInstallation
from ..gpu_config import cuda_environment
from ..mtp import apply_mtp, apply_resident_experts, argument
from ..platforms import (windows_to_wsl, model_target_path, model_exists, model_filesystem_path,
                         runtime_distribution, validate_model_runtime, location_of, linux_path, path_stat)

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
# These options can escape the selected local model, activate tools or expose a listener.
RESERVED = {"--host", "--port", "--listen", "--model", "-m", "--alias", "--served-model-name", "--config", "--engine", "--tokenizer", "--mcp-config", "--mcp-servers-config", "--api-key", "--api-key-file", "--ssl-key-file", "--ssl-cert-file", "--model-url", "-mu", "--hf-repo", "-hf", "-hfr", "--hf-file", "-hff", "--hf-token", "-hft", "--hf-repo-draft", "--spec-draft-hf", "-hfd", "-hfrd", "--mmproj-url", "-mmu", "--models-dir", "--models-preset", "--models-autoload", "--model-draft", "-md", "--lora", "--lora-scaled", "--trust-remote-code", "--non-blocking", "--root-path", "--uds", "--rpc", "--remote-expert", "--remote-expert-opt", "--before-load"}
SECRET_RE = re.compile(r"(?:api.?key|token|secret|password|mcp)", re.I)
SENSITIVE_LOG_FLAGS = {"--log-file", "--verbose", "--log-verbose", "--verbose-prompt", "--log-verbosity", "--verbosity", "-lv", "-v"}
# These native C++ string options consume exactly one argv value, even if it
# looks like another option. Other engines' argparse rules are different.
LLAMA_EXTRA_VALUE_FLAGS = {'--chat-template', '--chat-template-file', '--chat-template-kwargs',
                          '--grammar', '--grammar-file', '--json-schema', '-j', '--json-schema-file', '-jf'}


def extra_option_indices(args, runtime):
    index = 0
    while index < len(args):
        yield index
        consumes_value = runtime.backend == 'llama_cpp' and args[index] in LLAMA_EXTRA_VALUE_FLAGS
        index += 2 if consumes_value else 1


def _run_readonly(argv: list[str], timeout: float = 12, *, env=None) -> str:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout, env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return p.stdout + p.stderr if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _identity(backend: str, path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, backend + ":" + os.path.normcase(path)))


def _local_file(path: str, label: str = "文件") -> str:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"{label}不存在：{p}")
    return str(p)


def _strata_supported_flags(flags, version, family=None):
    """Complete native help with the released engine's documented omissions."""
    if not flags:
        return flags  # File-only legacy imports do not have an exhaustive inventory.
    flags = set(flags)
    native_version = re.match(r'^v?(\d+)\.(\d+)\.(\d+)(?:[.-]|$)', version)
    # generate.cpp gained this option in 1e663cc70ee2, first released in v0.1.31;
    # it is still absent from 0.1.56 help. The separate ds4_main.cpp parser does
    # not implement it. Do not infer other options from this help omission.
    if family != 'deepseek41' and native_version and tuple(map(int, native_version.groups())) >= (0, 1, 31):
        flags.add('--resident-budget-gib')
    return sorted(flags)


def _extra(profile: LoadProfile, runtime: RuntimeInstallation) -> list[str]:
    args = profile.extra_args
    known = runtime.metadata.get("supported_flags")
    if runtime.backend == 'strata':
        known = _strata_supported_flags(known, runtime.version, runtime.metadata.get('strata_family'))
    for value in args:
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("高级参数必须是独立的字符串 argv")
    if runtime.backend == 'strata' and profile.parameters.get('startup_expert_prefetch') is False:
        args = [value for value in args if value.split('=', 1)[0] != '--startup-expert-prefetch']
    for index in extra_option_indices(args, runtime):
        value = args[index]
        flag = value.split("=", 1)[0]
        option = flag.startswith('-') and not any(c.isspace() for c in flag)
        if flag in RESERVED or flag in SENSITIVE_LOG_FLAGS or (option and ("mcp" in flag.lower() or "api-key" in flag.lower() or flag.endswith("-default"))):
            raise ValueError(f"高级参数 {flag} 由本地实例管理器保留")
        if known and option and flag.startswith("--") and flag not in known:
            custom_strata = False
            if runtime.backend == 'strata' and flag in ('--raw-token-continuation', '--startup-expert-prefetch'):
                from ..strata_prefetch import select_engine, EXECUTABLE
                custom_strata = Path(select_engine(profile, runtime)[0]).name == EXECUTABLE
            if not custom_strata:
                raise ValueError(f"此运行时的 --help 未列出参数 {flag}")
        if runtime.backend == 'llama_cpp' and value in LLAMA_EXTRA_VALUE_FLAGS and index + 1 == len(args):
            raise ValueError(f'参数 {flag} 缺少值')
    return list(args)


def _mapped(parameters: dict, mapping: dict[str, str]) -> list[str]:
    result = []
    for key, flag in mapping.items():
        value = parameters.get(key)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            if isinstance(value, bool):
                if value:
                    result.append(flag)
            else:
                result.extend([flag, str(value)])
    return result


def _caps(model: ModelAsset, images: bool = False) -> dict:
    return {"input": ["text", "image"] if images else ["text"], "output": ["text"], "streaming": True,
            "tools": False, "tools_status": "unverified", "reasoning": "unverified"}


def _context_caps(capabilities: dict, argv: list[str], flag: str) -> dict:
    """Retain the explicit final launch limit, not an inferred model capacity."""
    try:
        tokens = int(argument(argv, flag))
    except (TypeError, ValueError):
        return capabilities
    return {**capabilities, 'configured_context_tokens': tokens, 'configured_context_flag': flag} if tokens > 0 else capabilities


def _gguf(model: ModelAsset) -> Compatibility:
    if model.format.lower() != "gguf":
        return Compatibility(status="unsupported", reasons=["需要本地 GGUF 权重；HF 原始目录不能直接交给此适配器"])
    missing = model.metadata.get("missing_shards", [])
    if missing:
        return Compatibility(status="unsupported", reasons=["GGUF 分片不完整"])
    if model.metadata.get("is_mmproj") or Path(model.path).name.lower().startswith("mmproj"):
        return Compatibility(status="unsupported", reasons=["mmproj 是视觉投影器，不能单独作为语言模型"])
    return Compatibility(status="unverified", capabilities=_caps(model))


def _hf(model: ModelAsset) -> Compatibility:
    if location_of(model).platform == 'wsl2':
        from ..assets import inspect_asset
        if model.format.lower() not in {'hf', 'huggingface', 'hf_directory', 'safetensors', 'pytorch'}:
            return Compatibility(status='unsupported', reasons=['此适配器需要完整的本地 HF 目录'])
        try:
            inspect_asset(model.path, location=model.location)
        except ValueError as error:
            return Compatibility(status='unsupported', reasons=[str(error)])
        return Compatibility(status='unverified', capabilities=_caps(model, 'image' in model.modalities))
    if model.format.lower() not in {"hf", "huggingface", "hf_directory", "safetensors", "pytorch"} or not model_filesystem_path(model).is_dir():
        return Compatibility(status="unsupported", reasons=["此适配器需要包含 config.json、分词器和完整权重的本地 HF 目录"])
    if not (model_filesystem_path(model) / "config.json").is_file():
        return Compatibility(status="unsupported", reasons=["HF 目录缺少 config.json"])
    return Compatibility(status="unverified", capabilities=_caps(model, "image" in model.modalities))


class BackendAdapter:
    id: str
    label: str

    def parameters(self) -> list[ParameterField]:
        raise NotImplementedError

    def detect(self) -> list[RuntimeInstallation]:
        return []

    def inspect(self, model: ModelAsset, runtime: RuntimeInstallation) -> Compatibility:
        raise NotImplementedError

    def build_launch(self, model: ModelAsset, profile: LoadProfile, runtime: RuntimeInstallation, context: LaunchContext) -> LaunchSpec:
        raise NotImplementedError

    def _validate(self, model, runtime, profile):
        from ..parameter_rules import validate_profile
        validate_model_runtime(model, runtime)
        validate_profile(model, profile, runtime, self, legacy_gpu=True)
        result = self.inspect(model, runtime)
        if result.status == "unsupported":
            raise ValueError("；".join(result.reasons))
        if runtime.backend != self.id:
            raise ValueError("运行时与后端不匹配")
        if not model_exists(model):
            raise ValueError("模型路径不存在")


LLAMA_MAP = {"context_length": "--ctx-size", "gpu_layers": "--n-gpu-layers", "threads": "--threads", "threads_batch": "--threads-batch", "parallel": "--parallel", "batch_size": "--batch-size", "ubatch_size": "--ubatch-size", "cache_type_k": "--cache-type-k", "cache_type_v": "--cache-type-v", "flash_attention": "--flash-attn", "devices": "--device", "split_mode": "--split-mode", "tensor_split": "--tensor-split", "main_gpu": "--main-gpu"}
LLAMA_KV_TYPES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]


def llama_effective_parameters(profile: LoadProfile) -> dict:
    """Resolve the editor's launch defaults without mutating the saved profile."""
    parameters = {field.key: field.default for field in LlamaCppAdapter().parameters() if field.default is not None}
    parameters.update({key: value for key, value in profile.parameters.items()
                       if value is not None and not (isinstance(value, str) and not value.strip())})
    if profile.parameter_editor_version == 1:
        for key, value in profile.parameters.items():
            if value is None:
                parameters.pop(key, None)
    return parameters


class LlamaCppAdapter(BackendAdapter):
    id, label = "llama_cpp", "llama.cpp"

    def parameters(self):
        return [ParameterField(key="context_length", label="上下文长度", type="number", default=32768, minimum=0, maximum=2**31-1, integer=True, step=1, description="0 使用模型默认上下文。"),
                ParameterField(key="gpu_layers", label="GPU 层数", type="number", default=-1, minimum=0, special_values=[-1, -2, 'auto', 'all'], integer=True, step=1),
                ParameterField(key="devices", label="卸载设备", type="text", description="llama.cpp 设备名，逗号分隔；none 关闭卸载。"),
                ParameterField(key="split_mode", label="拆分模式", type="select", options=['none', 'layer', 'row', 'tensor']),
                ParameterField(key="tensor_split", label="GPU 分配比例", type="text"),
                ParameterField(key="main_gpu", label="主 GPU 索引", type="number", minimum=0, maximum=2**31-1, integer=True, step=1),
                ParameterField(key="parallel", label="并发槽位", type="number", default=1, minimum=1, maximum=2**31-1, special_values=[-1], integer=True, step=1, description="-1 由运行环境自动分配。"),
                ParameterField(key="threads", label="CPU 线程", type="number", minimum=-(2**31), maximum=2**31-1, integer=True, step=1, description="非正数由运行环境自动选择。"),
                ParameterField(key="cache_type_k", label="K 缓存类型", type="select", options=LLAMA_KV_TYPES, default="q8_0"),
                ParameterField(key="cache_type_v", label="V 缓存类型", type="select", options=LLAMA_KV_TYPES, default="q8_0"),
                ParameterField(key="flash_attention", label="Flash Attention", type="select", options=["auto", "on", "off"], default="auto", advanced=True),
                ParameterField(key="mmproj", label="视觉投影器 mmproj", type="file", advanced=True)]

    def detect(self):
        found = []
        for p in [shutil.which("llama-server")]:
            if p and Path(p).is_file() and not any(r.path == str(Path(p).resolve()) for r in found):
                found.append(import_runtime(self.id, p))
        return found

    def inspect(self, model, runtime):
        return _gguf(model)

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        # The editor's defaults are ModLoader launch values, not llama.cpp's
        # native auto-sizing defaults. Resolve blanks without changing a profile.
        parameters = llama_effective_parameters(profile)
        argv = [runtime.path, "--model", _local_file(model.path), "--host", "127.0.0.1", "--port", str(context.port), "--alias", profile.alias]
        argv += _mapped(parameters, LLAMA_MAP)
        if profile.parameter_editor_version == 1:
            from ..launch_parameters import LLAMA_GENERATION_MAP
            argv += _mapped(profile.generation, LLAMA_GENERATION_MAP)
            if parameters.get('mtp_enabled') is not False:
                argv += _mapped(parameters, {'mtp_tokens': '--spec-draft-n-max', 'mtp_path': '--spec-draft-model'})
        mmproj = parameters.get("mmproj") if parameters.get('vision_enabled') is not False else None
        if mmproj:
            argv += ["--mmproj", _local_file(mmproj, "视觉投影器")]
        elif parameters.get('vision_enabled') is False and '--no-mmproj' in runtime.metadata.get('supported_flags', []):
            argv.append('--no-mmproj')
        argv += _extra(profile, runtime)
        argv = apply_mtp(argv, model, profile, runtime)
        if "--offline" in runtime.metadata.get("supported_flags", []):
            argv.append("--offline")
        if "--metrics" in runtime.metadata.get("supported_flags", []) and not any(a.split("=", 1)[0] in {"--metrics", "--no-metrics"} for a in argv):
            argv.append("--metrics")
        return LaunchSpec(argv=argv, cwd=str(Path(runtime.path).parent), env={**OFFLINE_ENV, "LLAMA_ARG_OFFLINE": "1", "LLAMA_SERVER_SLOTS_DEBUG": "0"}, model_name=profile.alias,
                          health_path="/health", health_expected={"status": "ok"}, capabilities=_caps(model, bool(mmproj)))


# Only inference fields are retained. In particular env, hooks, API credentials and MCP are never imported.
STRATA_KEYS = {"exe", "args", "cwd", "tokenizer", "model_name", "lib_dirs", "gpu", "gpus_asked", "vision", "backend", "arch", "engine_ready_s", "drafting", "user_args", "mtp", "raw_token_continuation", "parallel"}
STRATA_MAP = {"context_length": "--max-context", "kv_cache": "--kv", "expert_cache": "--expert-cache", "vram_reserve_mib": "--vram-reserve-mib", "spec": "--spec", "pool_workers": "--pool-workers"}
STRATA_BOOL = {"--resident-experts", "--vision", "--no-mmap", "--mlock"}


def _strata_config(raw: dict, source: Path) -> dict:
    cfg = {k: v for k, v in raw.items() if k in STRATA_KEYS}
    cwd = (source.parent / (cfg.get("cwd") or '')).resolve()
    cfg["cwd"] = str(cwd)
    for key in ("exe", "tokenizer"):
        if cfg.get(key):
            cfg[key] = str((cwd / cfg[key]).resolve())
    cfg["lib_dirs"] = [str((cwd / p).resolve()) for p in cfg.get("lib_dirs", [])]
    for key in ('args', 'user_args'):
        if key == 'user_args' and key not in cfg:
            continue
        value = cfg.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f'Strata {key} 必须是字符串数组')
        filtered = _filter_import_args(value, strata=True)
        normalized, index = [], 0
        paths = ('--native', '--pack', '--ple-gguf', '--expert-profile', '--mtp', '--draft-model', '--mtp-draft-vocab')
        while index < len(filtered):
            item = filtered[index]
            flag, sep, inline = item.partition('=')
            if flag in paths:
                if not sep and index + 1 >= len(filtered):
                    raise ValueError(f'Strata {flag} 缺少值')
                value = inline if sep else filtered[index + 1]
                if not (cfg.get('arch') == 'deepseek41' and flag == '--expert-profile' and value == 'none'):
                    value = str((cwd / value).resolve())
                normalized.extend([flag, value])
                index += 1 if sep else 2
            else:
                normalized.append(item)
                index += 1
        cfg[key] = normalized
    if isinstance(cfg.get("vision"), dict):
        cfg["vision"] = {k: v for k, v in cfg["vision"].items() if k in {"exe", "mmproj", "model", "gpu", "max_tokens"}}
        for key in ("exe", "mmproj", "model"):
            if cfg["vision"].get(key):
                cfg["vision"][key] = str((cwd / cfg["vision"][key]).resolve())
    return cfg


def _replace_arg(args: list[str], flag: str, value: str) -> None:
    if flag in args:
        i = args.index(flag)
        if i + 1 == len(args):
            raise ValueError(f"参数 {flag} 缺少值")
        args[i + 1] = value
    else:
        args.extend([flag, value])


class StrataAdapter(BackendAdapter):
    id, label = "strata", "Strata"

    def parameters(self):
        return [ParameterField(key="configuration", label="已有 Strata JSON", type="file", description="只读取推理配置，派生配置去除密钥和 MCP"),
                ParameterField(key="gpu", label="GPU 索引", type="text", description="nvidia-smi 索引，逗号分隔。"),
                ParameterField(key="context_length", label="上下文长度", type="number", default=32768, minimum=1, maximum=2**31-9, integer=True, step=1),
                ParameterField(key="kv_cache", label="KV 缓存", type="select", options=[
                    {"value": "int8", "label": "INT8（K / V 均 8 位）"},
                    {"value": "fp16", "label": "FP16（K / V 均 16 位）"},
                    {"value": "q4_0", "label": "Q4_0（K / V 均 4 位）"},
                    {"value": "k8v4", "label": "K8 / V4（K INT8 · V Q4_0）"},
                ], default="int8"),
                ParameterField(key="expert_cache", label="专家缓存", type="text", default="auto", advanced=True),
                ParameterField(key="startup_expert_prefetch", label="启动专家预取", type="boolean", advanced=True),
                ParameterField(key="vram_reserve_mib", label="显存预留 MiB", type="number", default=2048, minimum=0, integer=True, step=1, advanced=True),
                ParameterField(key="tokenizer", label="本地 Strata tokenizer 目录", type="file", advanced=True),
                ParameterField(key="pack", label="已准备的 Strata pack 目录", type="file", advanced=True),
                ParameterField(key="expert_profile", label="专家访问配置", type="file", advanced=True, description="自动匹配运行环境中与模型层数、专家数对应的配置；可指定本地文件覆盖。"),
                ParameterField(key="mmproj", label="视觉投影器", type="file", advanced=True)]

    def detect(self):
        configured = os.environ.get('MODLOADER_STRATA_DEV_RUNTIME')
        if not configured:
            return []
        p = Path(configured)
        return [import_runtime(self.id, str(p))] if (p / "serve/server.py").exists() else []

    def inspect(self, model, runtime):
        result = _gguf(model)
        from ..strata_deepseek import is_deepseek, engine_path
        if result.status == 'unsupported':
            return result
        if model.architecture in {'engram', 'deepseek41-engram', 'dflash'}:
            return Compatibility(status='unsupported', reasons=['Engram / DSpark 是配套文件，请选择 DeepSeek 主模型'])
        if is_deepseek(model):
            try:
                engine_path(Path(runtime.path))
            except ValueError as error:
                return Compatibility(status='unsupported', reasons=[str(error)])
        elif runtime.metadata.get('strata_family') == 'deepseek41' and not (Path(runtime.path) / 'engine/strata.exe').is_file():
            return Compatibility(status='unsupported', reasons=['此环境只有 DeepSeek V4.1 引擎，请选择该模型对应的 Strata 环境'])
        return result

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        from ..strata_deepseek import is_deepseek
        if is_deepseek(model):
            return self._deepseek_launch(model, profile, runtime, context)
        root = Path(runtime.path)
        p = profile.parameters
        if p.get("configuration"):
            source = Path(_local_file(p["configuration"], "Strata 配置"))
            cfg = _strata_config(json.loads(source.read_text(encoding="utf-8-sig")), source)
        else:
            cfg = _strata_config(p.get("strata_config", {}), root / "strata.json")
        gpu = p.get('gpu')
        if gpu is not None and not (isinstance(gpu, str) and not gpu.strip()):
            cfg['gpu'] = gpu
        cfg["exe"] = str(root / "engine/strata.exe")
        if runtime.metadata.get("engine_backend"):
            cfg["backend"] = runtime.metadata["engine_backend"]
        from ..strata_prefetch import select_engine, KEY as PREFETCH_KEY, FLAG as PREFETCH_FLAG
        cfg["exe"], use_prefetch = select_engine(profile, runtime)
        cfg["lib_dirs"] = list(dict.fromkeys([*cfg.get("lib_dirs", []), *runtime.metadata.get("lib_dirs", [])]))
        if not Path(cfg["exe"]).is_file():
            raise ValueError("Strata 运行时缺少 engine/strata.exe")
        cfg["model_name"] = profile.alias
        cfg["host"], cfg["port"] = "127.0.0.1", context.port
        # Strata defaults the native engine's stderr to DEVNULL when log is absent.
        # Keeping a file here could persist request/debug output independently of the core.
        cfg.pop("log", None)
        from ..strata_drafting import (inspect_drafting, apply_drafting, serve_argv, merge_args, policy_config,
                                      explicit_arguments, supports_policy)
        drafting = inspect_drafting(model, profile, runtime)
        # user_args are safe import-filtered above; merge before structured/UI overrides.
        args = cfg['args'] = merge_args(cfg.get('args', []), cfg.get('user_args', []))
        if cfg.get('raw_token_continuation') is True and '--raw-token-continuation' not in args:
            args.append('--raw-token-continuation')
        if profile.parameter_editor_version == 1:
            from ..mtp import without
            # Settings edited through the full command own every mapped option,
            # including duplicates inherited from a user-selected source JSON.
            args[:] = without(args, *[flag for key, flag in STRATA_MAP.items() if key in p])
        previous_native = args[args.index("--native") + 1] if "--native" in args else None
        if previous_native and Path(previous_native).resolve() != Path(model.path).resolve():
            # Pack/MTP/projection files belong to one model, so do not silently reuse another model's cache.
            raise ValueError("Strata 配置属于另一模型；请导入该模型对应的 pack/tokenizer 配置")
        _replace_arg(args, "--native", _local_file(model.path))
        if p.get("pack"):
            _replace_arg(args, "--pack", str(Path(p["pack"]).resolve()))
        for key, flag in STRATA_MAP.items():
            if p.get(key) is not None:
                _replace_arg(args, flag, str(p[key]))
        if p.get("tokenizer"):
            cfg["tokenizer"] = str(Path(p["tokenizer"]).resolve())
        if not cfg.get("tokenizer") or not (Path(cfg["tokenizer"]) / "vocab.json").is_file():
            raise ValueError("缺少已导出的 Strata tokenizer；请导入已有配置或指定本地 tokenizer")
        if "--pack" not in args or not Path(args[args.index("--pack") + 1]).is_dir():
            raise ValueError("缺少本模型已准备的 Strata pack 目录")
        if p.get('expert_profile'):
            args.extend(['--expert-profile', str(p['expert_profile'])])
        args += _extra(profile, runtime)
        args[:] = merge_args([], args)
        explicit_flags = {arg.split('=', 1)[0] for arg in args if arg.startswith('--')}
        args[:] = apply_resident_experts(apply_drafting(args, model, profile, runtime, drafting), profile)
        if drafting['effective_mode'] == 'mtp' and p.get('mtp_tokens') is not None:
            explicit_flags.add('--spec')
        from ..strata_preparation import expert_profile_path
        expert_path = p.get('expert_profile')
        normalized_args = []
        cursor = 0
        while cursor < len(args):
            arg = args[cursor]
            if arg == '--expert-profile':
                if cursor + 1 >= len(args):
                    raise ValueError('参数 --expert-profile 缺少值')
                expert_path = args[cursor + 1]
                cursor += 2
            elif arg.startswith('--expert-profile='):
                expert_path = arg.split('=', 1)[1]
                cursor += 1
            else:
                normalized_args.append(arg)
                cursor += 1
        if expert_path:
            expert_path = str((Path(cfg['cwd']) / expert_path).resolve())
        args[:] = normalized_args + ['--expert-profile', expert_profile_path(model, runtime, expert_path)]
        # Native packs require lookup drafting and chunked prefill even without MTP.
        for flag, value in (("--spec", "4"), ("--prefill", "auto"), ("--expert-cache", "auto")):
            if not any(arg.split("=", 1)[0] == flag for arg in args):
                args.extend([flag, value])
        for index, arg in enumerate(args):
            flag, _, inline = arg.partition("=")
            if flag == '--expert-cache':
                from ..parameter_rules import validate_strata_expert_cache
                validate_strata_expert_cache(inline if '=' in arg else args[index + 1] if index + 1 < len(args) else '')
            if flag not in ("--spec", "--prefill"):
                continue
            value = inline if "=" in arg else args[index + 1] if index + 1 < len(args) else ""
            try:
                valid = True if flag == '--prefill' and (value == 'auto' or value.startswith('auto:')) else int(value) >= (2 if flag == "--spec" else 1)
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"Strata 原生 pack 的 {flag} 必须为{'至少 2 的整数' if flag == '--spec' else 'auto、auto:N 或正整数'}")
        if p.get('vision_enabled') is False:
            cfg.pop('vision', None)
            args[:] = [arg for arg in args if arg != '--vision']
        elif p.get("mmproj"):
            cfg["vision"] = {"exe": str(root / "engine/strata-vision.exe"), "mmproj": _local_file(p["mmproj"]), "model": model.path, "gpu": True}
            if "--vision" not in args:
                args.append("--vision")
        if cfg.get("vision"):
            cfg["vision"]["model"] = model.path
            _local_file(cfg["vision"].get("mmproj", ""), "视觉投影器")
        if p.get(PREFETCH_KEY) is False:
            args[:] = [arg for arg in args if arg.split('=', 1)[0] != PREFETCH_FLAG]
        if use_prefetch and PREFETCH_FLAG not in args:
            args.append(PREFETCH_FLAG)
        # Requested Auto survives until readiness; its prepared resources and API
        # alias are separate from the policy's model identity.
        cfg['drafting'] = drafting['effective_mode'] if drafting['mode'] == 'legacy' else drafting['mode']
        cfg['mtp'] = cfg['drafting'] != 'off'
        native_policy = supports_policy(runtime)
        if cfg.get('user_args') or native_policy:
            cfg['user_args'] = explicit_arguments(args, explicit_flags)
        if native_policy:
            cfg['_modloader_policy'] = policy_config(model, profile, runtime, drafting)
        out = Path(context.work_dir) / "strata.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        env = {**OFFLINE_ENV, "STRATA_ROUTE_TAIL_SKIP": "0"}
        return LaunchSpec(argv=[*serve_argv(runtime, cfg, out, drafting), "--engine", "strata", "--config", str(out), "--host", "127.0.0.1", "--port", str(context.port)],
                          cwd=str(root), env=env, model_name=profile.alias, health_path="/health",
                          health_expected={"status": "ok", "loaded": True}, capabilities=_caps(model, bool(cfg.get("vision"))))

    def _deepseek_launch(self, model, profile, runtime, context):
        from ..strata_deepseek import engine_path, validate_pack
        from ..mtp import without
        root, p = Path(runtime.path), profile.parameters
        source = Path(p['configuration']) if p.get('configuration') else root / 'strata.json'
        raw = json.loads(source.read_text('utf-8-sig')) if p.get('configuration') else p.get('strata_config', {})
        cfg = _strata_config(raw, source)
        if raw and cfg.get('arch') != 'deepseek41':
            raise ValueError('此 Strata 配置不是 DeepSeek V4.1 配置')
        if p.get('vision_enabled') or cfg.get('vision') and p.get('vision_enabled') is not False:
            raise ValueError('当前 DeepSeek V4.1 引擎未开放图像输入')
        if p.get('mtp_enabled'):
            raise ValueError('DeepSeek V4.1 使用 DSpark；请移除 Qwen MTP 设置')
        exe = engine_path(root)
        build = json.loads((exe.parent / 'BUILD.json').read_text('utf-8-sig')) if (exe.parent / 'BUILD.json').is_file() else {}
        cfg.update(exe=str(exe), arch='deepseek41', model_name=profile.alias, host='127.0.0.1', port=context.port,
                   engine_ready_s=p.get('startup_timeout') or cfg.get('engine_ready_s') or 3600)
        if build.get('backend'):
            cfg['backend'] = build['backend']
        cfg['lib_dirs'] = list(dict.fromkeys([str(exe.parent), *[str((exe.parent / d).resolve()) for d in build.get('lib_dirs', [])],
                                            *cfg.get('lib_dirs', []), *runtime.metadata.get('lib_dirs', [])]))
        cfg.pop('vision', None)
        gpu = p.get('gpu')
        if gpu is not None and gpu != '':
            cfg['gpu'] = gpu
        args = cfg.setdefault('args', [])
        pack = p.get('pack') or argument(args, '--pack')
        if not pack:
            raise ValueError('缺少本模型已准备的 DeepSeek pack 目录')
        pack = str(Path(pack).resolve())
        description = validate_pack(pack, model)
        args = without(args, '--pack') + ['--pack', pack]
        mappings = {'context_length': '--max-context', 'vram_reserve_mib': '--reserve-mib', 'cpu_threads': '--cpu-threads',
                    'ssd_cache': '--ssd-cache', 'dspark_mode': '--dspark', 'expert_profile': '--expert-profile'}
        for key, flag in mappings.items():
            if p.get(key) is not None and p[key] != '':
                args = without(args, flag) + [flag, str(p[key])]
        for flag, value in (('--max-context', '8192'), ('--eos', '1'), ('--dspark', 'auto'), ('--dspark-conf', '0.65')):
            if argument(args, flag) is None:
                args += [flag, value]
        extra = _extra(profile, runtime)
        if any(value.split('=', 1)[0] in {'--pack', '--native'} for value in extra):
            raise ValueError('DeepSeek 模型和 pack 请在模型准备文件中选择，不通过额外参数替换')
        args += extra
        # DSV41's native automatic sentinel is negative; atof("auto") disables it.
        for index, arg in enumerate(args):
            if arg == '--ssd-cache=auto':
                args[index] = '--ssd-cache=-1'
            elif arg == '--ssd-cache' and index + 1 < len(args) and args[index + 1] == 'auto':
                args[index + 1] = '-1'
        mode = argument(args, '--dspark')
        if mode not in ('auto', 'on', 'off', 'none'):
            raise ValueError('DSpark 只接受 auto / on / off / none')
        if mode == 'on' and not description['dspark']:
            raise ValueError('此 DeepSeek pack 不含 DSpark 草稿')
        cfg['args'] = args
        cfg['tokenizer'] = str(Path(p.get('tokenizer') or cfg.get('tokenizer') or Path(pack) / 'tokenizer').resolve())
        from ..strata_preparation import StrataPreparation
        StrataPreparation._validate_tokenizer(cfg['tokenizer'])
        out = Path(context.work_dir) / 'strata.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        return LaunchSpec(argv=[runtime.metadata['python'], '-m', 'serve.server', '--engine', 'strata', '--config', str(out),
                               '--host', '127.0.0.1', '--port', str(context.port)], cwd=str(root), env=OFFLINE_ENV.copy(),
                          model_name=profile.alias, health_path='/health', health_expected={'status': 'ok', 'loaded': True},
                          capabilities=_context_caps(_caps(model), args, '--max-context'), startup_timeout=cfg['engine_ready_s'])


class TransformersAdapter(BackendAdapter):
    id, label = "transformers", "Transformers"

    def parameters(self):
        return [ParameterField(key="device", label="设备", type="text", default="cuda:0"),
                ParameterField(key="dtype", label="数据类型", type="select", options=["auto", "float16", "bfloat16", "float32"], default="auto"),
                ParameterField(key="attn_implementation", label="注意力实现", type="text", default="sdpa", advanced=True),
                ParameterField(key="compile", label="torch.compile", type="boolean", default=False, advanced=True, description="Windows 上可能需要额外编译依赖"),
                ParameterField(key="continuous_batching", label="连续批处理", type="boolean", default=False, advanced=True)]

    def detect(self):
        # Do not import large packages into the management process.
        p = shutil.which("transformers")
        return [import_runtime(self.id, p)] if p else []

    def inspect(self, model, runtime):
        return _hf(model)

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        argv = [runtime.path, "serve", str(Path(model.path).resolve()), "--host", "127.0.0.1", "--port", str(context.port), "--no-trust-remote-code"]
        p = profile.parameters
        argv += _mapped(p, {"device": "--device", "dtype": "--dtype", "attn_implementation": "--attn-implementation"})
        for key in ('compile', 'continuous_batching'):
            if profile.parameter_editor_version == 1 and key in p and p[key] is None:
                continue
            argv.append(('--' if p.get(key, False) else '--no-') + key.replace('_', '-'))
        argv += _extra(profile, runtime)
        argv = apply_mtp(argv, model, profile, runtime)
        # In 5.19.0 ModelManager synchronously loads force_model in its constructor,
        # before Uvicorn binds. /v1/models lists cached models and is not readiness.
        return LaunchSpec(argv=argv, cwd=context.work_dir, env=OFFLINE_ENV.copy(), model_name=str(Path(model.path).resolve()),
                          health_path="/health", health_expected={"status": "ok"}, capabilities=_caps(model, "image" in model.modalities))


class VllmAdapter(BackendAdapter):
    id, label = "vllm", "vLLM (WSL2)"

    def parameters(self):
        return [ParameterField(key="context_length", label="最大上下文", type="number", default=32768, minimum=1, special_values=[-1], integer=True, step=1),
                ParameterField(key="gpu_memory_utilization", label="GPU 显存比例", type="number", default=0.85, minimum=0, maximum=1, exclusive_minimum=True, step=.05),
                ParameterField(key="tensor_parallel_size", label="张量并行", type="number", default=1, minimum=1, integer=True, step=1),
                ParameterField(key="pipeline_parallel_size", label="流水线并行", type="number", minimum=1, integer=True, step=1),
                ParameterField(key="cuda_visible_devices", label="CUDA 可见设备", type="text"),
                ParameterField(key="dtype", label="数据类型", type="text", default="auto"),
                ParameterField(key="enforce_eager", label="禁用 CUDA Graph", type="boolean", default=False, advanced=True),
                ParameterField(key="max_num_seqs", label="最大并行序列", type="number", minimum=1, integer=True, step=1, advanced=True)]

    def inspect(self, model, runtime):
        result = _hf(model)
        validate_model_runtime(model, runtime)
        return result

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        distribution = runtime_distribution(runtime)
        args = [runtime.path, "serve", model_target_path(model, runtime), "--host", "127.0.0.1", "--port", str(context.port), "--served-model-name", profile.alias]
        args += _mapped(profile.parameters, {"context_length": "--max-model-len", "gpu_memory_utilization": "--gpu-memory-utilization", "tensor_parallel_size": "--tensor-parallel-size", "pipeline_parallel_size": "--pipeline-parallel-size", "dtype": "--dtype", "enforce_eager": "--enforce-eager", "max_num_seqs": "--max-num-seqs"})
        args += _extra(profile, runtime)
        args = apply_mtp(args, model, profile, runtime)
        return LaunchSpec(platform='wsl2', distribution=distribution, argv=args, cwd=str(PurePosixPath(runtime.path).parent),
                          env={**OFFLINE_ENV, **cuda_environment(profile, runtime)}, model_name=profile.alias, health_path="/health",
                          capabilities=_context_caps(_caps(model, "image" in model.modalities), args, '--max-model-len'), startup_timeout=900)



from ..platforms import WSL_SUPERVISOR  # compatibility for older callers



def import_runtime(backend: str, path: str, metadata: dict | None = None, platform: str | None = None) -> RuntimeInstallation:
    meta = dict(metadata or {})
    platform = platform or ("wsl2" if backend in {"vllm", "sglang", "ninfer", "sparkinfer"} else "windows")
    version = str(meta.pop("version", "unknown"))
    if backend not in get_adapters():
        raise ValueError("未知后端")
    if backend == "custom":
        from .custom import import_custom_runtime
        return import_custom_runtime(path, meta, platform)
    if backend == 'sglang':
        from .sglang import import_sglang
        return import_sglang(path, {**meta, 'version': version}, platform)
    if backend in {'ninfer', 'sparkinfer'}:
        from .native_engines import import_native
        return import_native(backend, path, {**meta, 'version': version}, platform)
    if backend == "vllm":
        if platform != 'wsl2':
            raise ValueError('vLLM 需要 WSL2 运行环境')
        path = linux_path(path)
        meta['distribution'] = meta.get('distribution', 'ModLoader')
        from ..platforms import distribution_name
        distribution = distribution_name(meta['distribution'])
        return RuntimeInstallation(id=str(uuid.uuid5(uuid.NAMESPACE_URL, backend + ':wsl2:' + distribution + ':' + path)),
                                   backend=backend, platform='wsl2', name=f'vLLM · {distribution} WSL2', path=path, version=version, metadata=meta)
    if platform != 'windows':
        raise ValueError('此后端只支持 Windows；已有 Linux 程序请使用自定义引擎')
    p = Path(path).expanduser().resolve()
    if backend == "llama_cpp":
        p = p / "llama-server.exe" if p.is_dir() else p
        _local_file(str(p), "llama-server")
        output = _run_readonly([str(p), "--version"])
        match = re.search(r"(?:version:|version)\s*([^\r\n]+)", output)
        if match:
            version = match.group(1).strip()
        help_text = _run_readonly([str(p), "--help"])
        meta["supported_flags"] = sorted(set(re.findall(r"(?<!\w)--[a-z][a-z0-9-]*", help_text)))
        if help_text:
            meta['speculative_types'] = sorted(set(re.findall(r'\b(?:draft-[a-z0-9-]+|ngram-[a-z0-9-]+)\b', help_text)))
    elif backend == "strata":
        if p.is_file():
            p = p.parent.parent if p.parent.name == "serve" else p.parent
        if not (p / "serve/server.py").is_file():
            raise ValueError("Strata 目录需要 serve/server.py")
        if meta.get('strata_family') == 'deepseek41' or not (p / 'engine/strata.exe').is_file():
            from ..strata_deepseek import engine_path
            engine = engine_path(p).parent
            meta['strata_family'] = 'deepseek41'
        else:
            engine = p / 'engine'
        python = Path(meta.get("python", p / ".venv/Scripts/python.exe"))
        _local_file(str(python), "Strata Python")
        meta["python"] = str(python.resolve())
        build = engine / "BUILD.json"
        build_info = {}
        if build.is_file():
            build_info = json.loads(build.read_text(encoding="utf-8-sig"))
            version = str(build_info.get("version", version))
        if build_info.get("backend"):
            meta["engine_backend"] = build_info["backend"]
        meta["lib_dirs"] = [str(engine),
                            *[str((engine / directory).resolve()) for directory in build_info.get("lib_dirs", [])],
                            str(p / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64")]
        meta["configurations"] = [str(c) for c in sorted(p.glob("strata-*.json"))]
        # Historical imports are file-only. The new native capability contract
        # starts at 0.1.56; registering one reads help, never starts a model.
        if tuple(int(n) for n in re.findall(r'\d+', version)[:3]) >= (0, 1, 56):
            native = engine / ('strata-dsv41.exe' if meta.get('strata_family') == 'deepseek41' else 'strata.exe')
            environment = {**os.environ, **OFFLINE_ENV}
            environment['PATH'] = os.pathsep.join([*meta['lib_dirs'], environment.get('PATH', '')])
            help_text = _run_readonly([str(native), '--help'], env=environment)
            flags = sorted(set(re.findall(r'(?<!\w)--[a-z][a-z0-9-]*', help_text)))
            if not flags:
                raise ValueError('无法读取 Strata 原生引擎能力（--help）；请检查引擎和 DLL 依赖')
            meta['supported_flags'] = _strata_supported_flags(flags, version, meta.get('strata_family'))
        if meta.get('strata_family') != 'deepseek41':
            from ..strata_prefetch import metadata as prefetch_metadata
            meta.update(prefetch_metadata(p))
    else:
        if p.is_dir():
            p = p / "Scripts/transformers.exe"
        _local_file(str(p), "transformers CLI")
        help_text = _run_readonly([str(p), "serve", "--help"])
        if help_text:
            meta["supported_flags"] = sorted(set(re.findall(r"(?<!\w)--[a-z][a-z0-9-]*", help_text)))
    return RuntimeInstallation(id=_identity(backend, str(p)), backend=backend, name=f"{get_adapters()[backend].label} · {version}", path=str(p), version=version, metadata=meta)


def _filter_import_args(args: list[str], strata: bool = False) -> list[str]:
    result = []
    i = 0
    while i < len(args):
        arg = str(args[i])
        flag = arg.split("=", 1)[0]
        if flag in RESERVED or (strata and flag in SENSITIVE_LOG_FLAGS) or (flag.startswith("-") and SECRET_RE.search(flag)
                and flag not in {"--max-tokens", "--max-new-tokens"}
                and not (strata and flag == '--raw-token-continuation')):
            i += 1
            if "=" not in arg and i < len(args) and not str(args[i]).startswith("--"):
                i += 1
            continue
        if any(x in arg for x in ["\x00", "\n", "\r"]):
            i += 1
            continue
        result.append(arg)
        i += 1
    return result


def _arg_value(args: list[str], *names: str) -> str | None:
    for i, arg in enumerate(args):
        for name in names:
            if arg == name and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def import_configuration(path: str) -> dict[str, Any]:
    source = Path(_local_file(path, "配置"))
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
        if not isinstance(raw, dict) or not isinstance(raw.get("args"), list):
            raise ValueError("此处支持 Strata 推理 JSON；其他服务配置请手工创建加载方案")
        cfg = _strata_config(raw, source)
        model = _arg_value(cfg["args"], "--native")
        if cfg.get('arch') == 'deepseek41':
            from ..strata_deepseek import pack_description
            from ..assets import gguf_metadata
            pack = _arg_value(cfg['args'], '--pack')
            if not pack:
                raise ValueError('DeepSeek 配置缺少 --pack')
            description = pack_description(pack)
            model = next((str(path) for path in description['files'] if path.suffix.lower() == '.gguf'
                          and gguf_metadata(path).get('general.architecture') == 'deepseek41'), None)
        if not model:
            raise ValueError("Strata 配置缺少 --native 模型路径")
        parameters = {"strata_config": cfg}
        if 'gpu' in cfg:
            parameters['gpu'] = cfg['gpu']
        mapping = ({'context_length': '--max-context', 'vram_reserve_mib': '--reserve-mib', 'cpu_threads': '--cpu-threads',
                    'ssd_cache': '--ssd-cache', 'dspark_mode': '--dspark'} if cfg.get('arch') == 'deepseek41' else STRATA_MAP)
        for key, flag in mapping.items():
            value = _arg_value(cfg["args"], flag)
            if value is not None:
                parameters[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
        return {"model_path": str((Path(cfg["cwd"]) / model).resolve()), "name": cfg.get("model_name", source.stem), "backend": "strata", "runtime_path": cfg["cwd"], "parameters": parameters, "extra_args": []}
    if source.suffix.lower() not in {".bat", ".cmd", ".sh"}:
        raise ValueError("仅支持读取 .json/.bat/.cmd/.sh 配置；不会执行脚本")
    variables: dict[str, str] = {}
    text = re.sub(r"\^\r?\n|\\\r?\n", " ", text)
    for line in text.splitlines():
        m = re.match(r'\s*set\s+"?([A-Za-z_][A-Za-z_0-9]*)=(.*?)"?\s*$', line, re.I)
        if m and not SECRET_RE.search(m.group(1)):
            variables[m.group(1).upper()] = m.group(2)
    for line in text.splitlines():
        if "llama-server" not in line.lower() or re.match(r"\s*(rem|echo|::)", line, re.I):
            continue
        line = line.replace("%~dp0", str(source.parent) + os.sep)
        line = re.sub(r"%([A-Za-z_][A-Za-z_0-9]*)%", lambda m: variables.get(m.group(1).upper(), m.group(0)), line)
        args = [a.strip('"') for a in shlex.split(line, posix=False)]
        model = _arg_value(args, "--model", "-m")
        if not model or "%" in model:
            raise ValueError("脚本的模型路径无法静态解析；请直接选择模型")
        name = _arg_value(args, "--alias") or source.stem
        parameters = {}
        for key, flag in LLAMA_MAP.items():
            value = _arg_value(args, flag)
            if value is not None:
                parameters[key] = int(value) if key not in {'devices', 'split_mode', 'tensor_split'} and re.fullmatch(r"-?\d+", value) else value
        filtered = _filter_import_args(args[1:])
        for flag in LLAMA_MAP.values():
            if flag in filtered:
                j = filtered.index(flag)
                del filtered[j:j + 2]
        executable = args[0]
        return {"model_path": str((source.parent / model).resolve()), "name": name, "backend": "llama_cpp", "runtime_path": str((source.parent / executable).resolve()), "parameters": parameters, "extra_args": filtered}
    # Strata launcher points at JSON; only the path is extracted, never its command or environment.
    match = re.search(r'--config\s+(?:"([^"]+)"|(\S+))', text)
    if match:
        config = (match.group(1) or match.group(2)).replace("%~dp0", str(source.parent) + os.sep)
        return import_configuration(str(source.parent / config))
    raise ValueError("未找到可静态导入的 llama-server / Strata 配置")


def get_adapters() -> dict[str, BackendAdapter]:
    from .custom import CustomAdapter
    from .sglang import SglangAdapter
    from .native_engines import NinferAdapter, SparkInferAdapter
    return {a.id: a for a in (LlamaCppAdapter(), StrataAdapter(), TransformersAdapter(), VllmAdapter(), CustomAdapter(), SglangAdapter(), NinferAdapter(), SparkInferAdapter())}
