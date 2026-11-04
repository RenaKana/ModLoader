"""Pre-load memory previews. Native estimates are not a load/admission check.

llama.cpp's fit printer reads model metadata and simulates buffers. It does not
load weights, but backend initialization/recurrent-state simulation can allocate
temporary memory. Host buffers are logical allocations, not additional RSS.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
from .mtp import options as mtp_options, apply_mtp, apply_resident_experts, argument
import re
import subprocess

from .backends.adapters import (LLAMA_MAP, OFFLINE_ENV, STRATA_MAP, _extra,
                                _mapped, _replace_arg, _strata_config,
                                llama_effective_parameters)
from .models import LoadProfile, ModelAsset, RuntimeInstallation
from .gpu_identity import cuda_devices, cuda_devices_in_environment, identity_fields

MIB = 1024 ** 2
NATIVE_TIMEOUT_SECONDS = 45
CACHE_ENTRIES = 32

# Only allocation options understood by the common fit estimator are forwarded.
# Unknown options are deliberately not interpreted as proof of a full estimate.
LLAMA_VALUES = set(LLAMA_MAP.values()) | {
    "--device", "--override-tensor", "--n-cpu-moe", "--rope-scaling",
    "--rope-scale", "--rope-freq-base", "--rope-freq-scale", "--yarn-orig-ctx",
    "--yarn-ext-factor", "--yarn-attn-factor", "--yarn-beta-slow", "--yarn-beta-fast",
    "--numa", "--defrag-thold",
}
LLAMA_SWITCHES = {"--no-kv-offload", "--no-op-offload", "--no-mmap", "--mmap",
                  "--mlock", "--no-host", "--kv-unified", "--cpu-moe"}
LLAMA_ALIASES = {"-c": "--ctx-size", "-ngl": "--n-gpu-layers", "--gpu-layers": "--n-gpu-layers",
                 "-np": "--parallel", "-b": "--batch-size", "-ub": "--ubatch-size",
                 "-ctk": "--cache-type-k", "-ctv": "--cache-type-v", "-fa": "--flash-attn",
                 "-sm": "--split-mode", "-ts": "--tensor-split", "-mg": "--main-gpu",
                 "-t": "--threads", "-tb": "--threads-batch", "-dev": "--device",
                 "-ot": "--override-tensor", "-nkvo": "--no-kv-offload"}
IGNORED_SWITCHES = {"--metrics", "--no-metrics", "--offline", "--no-warmup", "--jinja"}


def _resource(kind, label, value=None, quality="unknown", **kwargs):
    return {"kind": kind, "label": label, "bytes": value, "quality": quality, "notes": [], **kwargs}


def _unknown(note, source="unavailable", diagnostic=None):
    return {"resources": [], "notes": [note], "source": source,
            **({"diagnostic": diagnostic} if diagnostic else {})}


def _stat(path):
    p = Path(path)
    try:
        s = p.stat()
        return str(p.resolve()), s.st_size, s.st_mtime_ns
    except OSError:
        return str(p), None, None


def _options(args):
    """Keep argument order so the final occurrence overrides earlier values."""
    result = []
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("-"):
            result.append((token, None))
        elif "=" in token:
            result.append(tuple(token.split("=", 1)))
        else:
            value = None
            if i + 1 < len(args) and (not args[i + 1].startswith("-") or re.match(r"^-\d", args[i + 1])):
                i += 1
                value = args[i]
            result.append((token, value))
        i += 1
    return result


def llama_estimate_args(profile, runtime):
    parameters = llama_effective_parameters(profile)
    arguments = _mapped(parameters, LLAMA_MAP)
    notes = []
    if mtp_options(profile, runtime)['enabled']:
        notes.append('未计入 MTP 草稿模型和推测解码缓冲的额外占用。')
    if parameters.get("mmproj") and parameters.get('vision_enabled') is not False:
        notes.append("未计入视觉投影器和图像处理缓冲。")
    for flag, value in _options(_extra(profile, runtime)):
        flag = LLAMA_ALIASES.get(flag, flag)
        if flag in LLAMA_VALUES and value is not None:
            arguments.extend([flag, value])
        elif flag in LLAMA_SWITCHES and value is None:
            arguments.append(flag)
        elif flag not in IGNORED_SWITCHES or value is not None:
            notes.append(f"未计入高级选项 {flag} 的影响。")
    return arguments, notes


def _llama_flash_attention_note(diagnostics):
    # Read the resolved path, not just --flash-attn: auto can disable FA after
    # the context reports its initial setting. Unknown logs must stay unknown.
    status = "auto"
    for match in re.finditer(
        r"\bllama_context:\s+flash_attn\s*=\s*(enabled|disabled|auto)\b"
        r"|\bresolve_fused_ops:\s+Flash Attention (enabled|not supported, set to disabled)\b",
        diagnostics,
    ):
        status = match.group(1) or match.group(2)
    description = {
        "enabled": "已启用",
        "disabled": "已关闭",
        "not supported, set to disabled": "已自动关闭（当前配置未通过设备支持检测）",
        "auto": "状态未确认（运行时未返回明确结果）",
    }[status]
    return f"Flash Attention（预估）：{description}。"


def parse_llama_fit(output, limitations=(), diagnostics=""):
    resources = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(Host|CPU|CUDA\d+|Vulkan\d+|SYCL\d+|HIP\d+|Metal\d*)\s+(\d+)\s+(\d+)\s+(\d+)\s*", line)
        if not match:
            continue
        device, *columns = match.groups()
        sizes = [int(value) * MIB for value in columns]
        host = device in {"Host", "CPU"}
        if host:
            # llama_context::memory_breakdown separates model buffers (which
            # may be mmap-backed) from KV/recurrent state and scheduler compute
            # allocations. Only the latter two establish non-model host RAM.
            physical = sizes[1] + sizes[2]
            notes = [f"上下文 {int(columns[1]):,} MiB · 计算 {int(columns[2]):,} MiB"] if physical else []
            if sizes[0]:
                notes.append(f"Host 模型缓冲参考 {int(columns[0]):,} MiB（未计入物理内存）。")
            resources.append(_resource('ram', '系统内存' if physical else '内存缓冲参考',
                physical if physical else sizes[0], 'partial', logical=not bool(physical),
                model_reference_bytes=sizes[0],
                components=[{'label': label, 'bytes': size} for label, size in
                            (('上下文', sizes[1]), ('计算缓冲', sizes[2])) if size],
                notes=[*notes, *limitations]))
        else:
            notes = [f"模型 {int(columns[0]):,} MiB · 上下文 {int(columns[1]):,} MiB · 计算 {int(columns[2]):,} MiB"]
            resources.append(_resource('vram', device, sum(sizes),
                                       'partial' if limitations else 'estimate', device=device,
                                       notes=[*notes, *limitations]))
    if not resources:
        return _unknown("所选运行时没有返回可识别的内存估算。", "llama-fit-params")
    return {"resources": resources, "source": "llama-fit-params", "notes": [
        _llama_flash_attention_note(diagnostics),
        "按当前配置模拟模型、上下文和计算缓冲；不含常规进程、驱动、分配器、文件缓存及请求临时开销。",
    ]}


def strata_estimate(model, profile, runtime, progress=None):
    from .strata_deepseek import is_deepseek
    if is_deepseek(model):
        return _unknown('DeepSeek V4.1 暂无资源估算', 'strata-deepseek')
    p = profile.parameters
    source = Path(p["configuration"]) if p.get("configuration") else Path(runtime.path) / "strata.json"
    raw = json.loads(source.read_text(encoding="utf-8-sig")) if p.get("configuration") else p.get("strata_config", {})
    cfg = _strata_config(raw, source)
    from .strata_drafting import (merge_args, inspect_drafting, apply_drafting, policy_config, plan_args,
                                 explicit_arguments, supports_policy, supports_runtime_plan)
    drafting = inspect_drafting(model, profile, runtime)
    if drafting['status'] != 'ready':
        return _unknown(drafting['message'], 'strata')
    value = p.get('gpu')
    if value is not None and not (isinstance(value, str) and not value.strip()):
        cfg['gpu'] = value
    value = cfg.get('gpu')
    selected = value if isinstance(value, (list, tuple)) else str(value).split(',') if value is not None else []
    selected = [int(str(item).strip()) for item in selected if str(item).strip()]
    single, multi = len(selected) == 1, len(selected) > 1
    args = merge_args(cfg.get('args', []), cfg.get('user_args', []))
    previous = dict(_options(args)).get('--native')
    if previous and Path(previous).resolve() != Path(model.path).resolve():
        raise ValueError('Strata 配置属于另一模型')
    _replace_arg(args, '--native', str(Path(model.path).resolve()))
    if p.get('pack'):
        _replace_arg(args, '--pack', str(Path(p['pack']).resolve()))
    # Mirrors the launcher's imported-config overrides, including its treatment
    # of blanks. Do not call build_launch: that writes a derived config.
    for key, flag in STRATA_MAP.items():
        if p.get(key) is not None:
            _replace_arg(args, flag, str(p[key]))
    if p.get('expert_profile'):
        args.extend(['--expert-profile', str(p['expert_profile'])])
    args.extend(_extra(profile, runtime))
    args = merge_args([], args)
    explicit_flags = {arg.split('=', 1)[0] for arg in args if arg.startswith('--')}
    args = apply_resident_experts(apply_drafting(args, model, profile, runtime, drafting), profile)
    if drafting['effective_mode'] == 'mtp' and p.get('mtp_tokens') is not None:
        explicit_flags.add('--spec')
    if runtime.version == '0.1.43':
        for arg in args:
            if arg.startswith('--') and '=' in arg and not arg.startswith('--expert-profile='):
                raise ValueError('Strata 0.1.43 参数请用空格分隔，不支持 ' + arg.split('=', 1)[0] + '=值')
    options = dict(_options(args))
    if not options.get('--expert-profile'):
        from .strata_preparation import expert_profile_path
        try:
            options['--expert-profile'] = expert_profile_path(model, runtime)
        except (OSError, ValueError):
            pass  # Missing preparation must not hide independently known allocations.
    elif options['--expert-profile'] != 'none':
        options['--expert-profile'] = str((Path(cfg['cwd']) / options['--expert-profile']).resolve())
    from .strata_resource_estimates import allocation_preview
    # The native launcher injects these defaults after user overrides.
    for flag, value in (("--spec", "4"), ("--prefill", "auto"), ("--expert-cache", "auto")):
        options.setdefault(flag, value)
    if p.get('vision_enabled') is False:
        cfg.pop('vision', None)
        options.pop('--vision', None)
    elif p.get('mmproj'):
        options['--vision'] = None
    if p.get('startup_expert_prefetch') is False:
        options.pop('--startup-expert-prefetch', None)
    if supports_runtime_plan(runtime):
        from .strata_prefetch import select_engine
        exe, use_prefetch = select_engine(profile, runtime)
        if use_prefetch:
            options['--startup-expert-prefetch'] = None
        args = [part for key, value in options.items() for part in ([key] if value is None else [key, str(value)])]
        cfg.update(args=args, user_args=explicit_arguments(args, explicit_flags), model_name=profile.alias,
                   drafting=drafting['effective_mode'] if drafting['mode'] == 'legacy' else drafting['mode'],
                   mtp=drafting['effective_mode'] != 'off',
                   exe=exe)
        if supports_policy(runtime):
            cfg['_modloader_policy'] = policy_config(model, profile, runtime, drafting)
        options = dict(_options(plan_args(runtime, cfg)))
    if multi and '--layer-split' not in options and '--peer-device' not in options:
        options['--layer-split'] = 'auto'
    multi = multi or '--layer-split' in options or any(options.get(flag) not in (None, '0')
        for flag in ('--peer-device', '--expert-cache-device1', '--expert-cache-device2', '--expert-cache-device3'))
    result = allocation_preview(model, options, cfg, progress=progress)
    reserve_bytes = None
    try:
        reserve = int(options.get('--vram-reserve-mib', 700))
        if reserve >= 0:
            reserve_bytes = reserve * MIB
    except (ValueError, TypeError):
        pass
    fixed = sum(item['bytes'] for item in result['components'])
    cache = result['cache_bytes']
    dynamic = result['auto'] or cache is not None and cache > 0
    gpu_notes = [f"上下文 {options.get('--max-context', '4096')} · KV {options.get('--kv', 'fp16')} · 专家缓存 {options.get('--expert-cache')}"]
    if options.get('--kv-resident') not in (None, '0'):
        gpu_notes.append(f"启动规划 KV 驻留窗口 {options['--kv-resident']}；加载时按可用显存重算。")
    if result['auto']:
        gpu_notes.append('自动专家缓存按加载时剩余显存分配，预留空间仍供后续缓冲使用。')
    elif cache is not None and cache > 0:
        gpu_notes.append('专家预算按 pack/排名装槽计算；加载时可用显存不足会缩减。')
    gpu_notes.extend(result['notes'])
    gpu = _resource('vram', 'GPU 显存', None if dynamic or not fixed else fixed,
                    'dynamic' if dynamic else 'partial' if fixed else 'unknown', notes=gpu_notes,
                    components=result['components'],
                    **({'device': '多 GPU 合计', 'aggregate': True} if multi else {'device': f'CUDA{selected[0]}'} if single else {}))
    if dynamic and fixed:
        gpu['fixed_bytes'] = fixed
    if cache is not None and cache > 0:
        gpu['budget_bytes'] = fixed + cache
        gpu['cache_budget_bytes'] = cache
    if reserve_bytes is not None:
        gpu['reserve_bytes'] = reserve_bytes
    resources = [gpu]
    mapped = result['host_components']
    host = result['nonmapped_host_components']
    plan = result['resident_plan']
    if plan is not None:
        plan['gpu_device'] = gpu.get('device', 'CUDA0')
        ram = _resource('ram', '系统内存', quality='dynamic' if plan['mode'] == 'resident' else 'partial',
                        components=host, logical=False, resident_plan=plan,
                        notes=['按可用 RAM、commit 和 GPU 专家缓存推演；文件缓存另计。'])
        if plan['mode'] == 'arena':
            ram['bytes'] = plan['host_bytes'] + plan['total_expert_bytes']
        elif plan['mode'] == 'mmap':
            ram['bytes'] = plan['host_bytes']
        ram['expert_total_bytes'] = plan['total_expert_bytes']
        if result['complement_bytes'] is not None:
            ram['complement_bytes'] = result['complement_bytes']
        if mapped:
            ram['mapped_bytes'] = sum(item['bytes'] for item in mapped)
        resources.append(ram)
    elif host:
        resources.append(_resource('ram', '系统内存', sum(item['bytes'] for item in host),
                                   'partial', logical=False, components=host,
                                   notes=['已计入 PLE 主机缓冲；专家驻留布局不可读取。']))
    elif '--resident-experts' in options or '--resident-cpu-experts' in options:
        resources.append(_resource('ram', '系统内存', quality='dynamic',
                                   notes=['专家驻留布局不可读取。']))
    elif mapped:
        resources.append(_resource('ram', '映射内存', sum(item['bytes'] for item in mapped),
                                   'partial', logical=True, components=mapped, notes=['词嵌入映射缓冲。']))
    return {'resources': resources, 'source': 'strata-metadata', 'notes': []}


def _strata_identities(result, runtime):
    gpu = next((item for item in result['resources'] if item['kind'] == 'vram'), None)
    if gpu is None:
        return result
    if gpu.get('aggregate'):
        gpu['identity_status'] = 'unknown'
        return result
    device = gpu.get('device')
    environment = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': device[4:]} if device else {}
    diagnostics = []
    gpu.update(identity_fields(0, cuda_devices_in_environment(environment, runtime.metadata.get('python'), diagnostics)))
    if diagnostics:
        result['diagnostic'] = '\n'.join(filter(None, [result.get('diagnostic'), *diagnostics]))
    return result


def _llama_identities(result):
    rows = [item for item in result['resources'] if re.fullmatch(r'CUDA\d+', item.get('device', ''))]
    if rows:
        identities = cuda_devices()
        for item in rows:
            item.update(identity_fields(int(item['device'][4:]), identities))
    return result



def vllm_estimate(profile, runtime):
    fraction = profile.parameters.get("gpu_memory_utilization")
    options = dict(_options(_extra(profile, runtime)))
    if "--gpu-memory-utilization" in options:
        fraction = options["--gpu-memory-utilization"]
    if '--kv-cache-memory-bytes' in options:
        fraction = None  # An explicit KV pool budget overrides the utilization ratio.
    try:
        fraction = float(fraction)
        if not 0 < fraction <= 1:
            fraction = None
    except (ValueError, TypeError):
        fraction = None
    return {"resources": [_resource("vram", "GPU 显存预算", quality="dynamic",
                                    identity_status='unknown', budget_basis='total',
                                    **({"budget_fraction": fraction} if fraction is not None else {}),
                                    notes=["每张 GPU 的预算，包含模型与 KV 等分配。",
                                           "WSL 可见设备、并行分片和实际可用量需在加载时确认。"])],
            "notes": [], "source": "vllm-config"}


class ResourceEstimator:
    """One native process at a time; shared requests, bounded in-process cache."""

    def __init__(self):
        self._native_lock = asyncio.Lock()
        self._cache = OrderedDict()
        self._inflight = {}

    async def close(self):
        tasks = [entry['task'] for entry in self._inflight.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def estimate(self, model: ModelAsset, profile: LoadProfile, runtime: RuntimeInstallation, progress=None) -> dict:
        from .platforms import validate_model_runtime
        report = progress or (lambda event: None)
        report({'stage': 'model_metadata', 'message': '读取模型信息'})
        try:
            validate_model_runtime(model, runtime)
        except ValueError as error:
            return _unknown(str(error))
        if profile.backend != runtime.backend:
            return _unknown("加载配置与运行环境不匹配。")
        if profile.backend == "strata":
            try:
                result = await asyncio.to_thread(strata_estimate, model, profile, runtime, report)
                report({'stage': 'gpu_identity', 'message': '匹配 GPU 与可用资源'})
                return await asyncio.to_thread(_strata_identities, result, runtime)
            except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                return _unknown("Strata 配置或配套文件不可用，无法估算", "strata-config", str(error))
        if profile.backend in {"vllm", "sglang", "ninfer", "sparkinfer"}:
            from .wsl_resource_estimates import estimate_wsl
            try:
                return await asyncio.to_thread(estimate_wsl, model, profile, runtime, _options, report)
            except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                # Keep the established vLLM budget useful when weight metadata is
                # unavailable; a budget is still not a measured allocation.
                if profile.backend == 'vllm':
                    result = vllm_estimate(profile, runtime)
                    result['notes'].append('模型元数据不可读取，权重与 KV 分项暂不可估算。')
                    result['diagnostic'] = str(error)
                    return result
                return _unknown('模型元数据不可读取，无法估算加载分配。',
                                profile.backend + '-metadata', str(error))
        if profile.backend != "llama_cpp":
            return _unknown("尚未适配此引擎的资源预估。")
        missing = [path for path in dict.fromkeys([model.path, *model.files]) if not Path(path).is_file()]
        if missing:
            return _unknown("模型文件不存在，无法估算", "llama-fit-params", "\n".join(missing))
        executable = Path(runtime.path).with_name("llama-fit-params.exe" if Path(runtime.path).suffix.lower() == ".exe" else "llama-fit-params")
        if not executable.is_file():
            return _unknown("所选 llama.cpp 环境缺少同版本 llama-fit-params，无法可靠估算。")
        args, limitations = llama_estimate_args(profile, runtime)
        files = [model.path, *model.files, str(executable), runtime.path,
                 str(executable.with_name("llama-fit-params-impl.dll"))]
        key = json.dumps([[_stat(f) for f in files], runtime.version, args, limitations], ensure_ascii=False)
        if key in self._cache:
            self._cache.move_to_end(key)
            report({'stage': 'gpu_identity', 'message': '复用估算，匹配 GPU'})
            return await asyncio.to_thread(_llama_identities, deepcopy(self._cache[key]))
        if key not in self._inflight:
            work = {'waiters': 0, 'listeners': [], 'progress': None}
            def native_progress(event):
                work['progress'] = event
                for listener in tuple(work['listeners']):
                    listener(event)
            work['task'] = asyncio.create_task(self._native(executable, model, args, limitations, native_progress))
            self._inflight[key] = work
        entry = self._inflight[key]
        entry['waiters'] += 1
        entry['listeners'].append(report)
        if entry['progress'] is not None:
            report(entry['progress'])
        try:
            result = await asyncio.shield(entry['task'])
            if result["resources"]:
                self._cache[key] = result
                self._cache.move_to_end(key)
                while len(self._cache) > CACHE_ENTRIES:
                    self._cache.popitem(last=False)
            report({'stage': 'gpu_identity', 'message': '匹配 GPU 与可用资源'})
            return await asyncio.to_thread(_llama_identities, deepcopy(result))
        finally:
            entry['listeners'].remove(report)
            entry['waiters'] -= 1
            if entry['waiters'] == 0:
                self._inflight.pop(key, None)
                if not entry['task'].done():
                    entry['task'].cancel()
                    await asyncio.gather(entry['task'], return_exceptions=True)

    async def _native(self, executable, model, args, limitations, progress):
        if self._native_lock.locked():
            progress({'stage': 'estimate_queue', 'message': '等待前一个原生估算完成'})
        async with self._native_lock:
            progress({'stage': 'native_estimate', 'message': '计算模型、上下文与缓冲占用'})
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    str(executable), "--model", str(Path(model.path).resolve()), *args, "--offline", "--fit-print", "on",
                    "--log-verbosity", "5",
                    cwd=str(executable.parent), env={**os.environ, **OFFLINE_ENV, "LLAMA_ARG_OFFLINE": "1"},
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=NATIVE_TIMEOUT_SECONDS)
                if process.returncode != 0:
                    return _unknown("原生估算失败", "llama-fit-params",
                                    f"退出码：{process.returncode}\n" + stderr.decode("utf-8", errors="replace") + stdout.decode("utf-8", errors="replace"))
                return parse_llama_fit(stdout.decode("utf-8", errors="replace"), limitations,
                                       stderr.decode("utf-8", errors="replace"))
            except asyncio.TimeoutError:
                return _unknown("原生估算超时，已停止预览进程。", "llama-fit-params")
            except OSError as error:
                return _unknown("无法启动所选运行环境的原生估算工具。", "llama-fit-params", str(error))
            finally:
                if process is not None and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.communicate()
