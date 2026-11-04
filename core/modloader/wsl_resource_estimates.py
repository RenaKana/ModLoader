"""Metadata-only allocation previews for the four built-in WSL engines.

NInfer rules: legacy 64d10fbf paged KV/state geometry; a754fcf shared context
byte capacity from runtime/engine/context_capacity.cpp. Spark rules: 0.6.34
server/src/model_engine.cpp and runtime/src/kv_cache.cpp. HF pool references
follow vLLM 0.31 KVCacheSpec and SGLang 0.5.21 pool_configurator; their real
pool sizes require startup profiling and are deliberately not predicted here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from .backends.adapters import _extra
from .platforms import location_of, runtime_distribution, wsl_python
from .resource_metadata import read_metadata
from .gpu_config import cuda_environment
from .gpu_identity import identity_fields


def metadata_for(model, profile, runtime):
    prepared = profile.parameters.get('_prepared_artifact') if profile.backend == 'ninfer' else None
    location = location_of(model)
    path = prepared or model.path
    if prepared or location.platform == 'wsl2':
        # Match launch/preflight's selected distribution, including /root paths.
        # Send our stdlib reader over stdin rather than installing into a distro.
        source = Path(__file__).with_name('resource_metadata.py').read_text(encoding='utf-8')
        code = "import json,sys\nn={};exec(compile(sys.stdin.read(),'modloader_resource_metadata','exec'),n)\nprint(json.dumps(n['read_metadata'](sys.argv[1])))"
        return json.loads(wsl_python(runtime_distribution(runtime), code, [path], input=source, timeout=30))
    # A Windows file used by a WSL runtime still lives on Windows: do not test a
    # Linux runtime path with Path.is_file(), or ask WSL to read mounted weights.
    return read_metadata(path)


def _positive(value):
    try:
        n = int(value)
        return n if n > 0 and float(value) == n else None
    except (TypeError, ValueError, OverflowError):
        return None


def _fraction(value):
    try:
        n = float(value)
        return n if 0 < n <= 1 else None
    except (TypeError, ValueError):
        return None


def _human_int(value):
    """vLLM accepts decimal k/m/g/t and integer binary K/M/G/T suffixes."""
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([kKmMgGtT])', str(value).strip())
    if not match:
        return _positive(value)
    number, suffix = match.groups()
    if suffix.isupper() and '.' in number:
        return None
    multiplier = (1024 if suffix.isupper() else 1000) ** ('kmgt'.index(suffix.lower()) + 1)
    return _positive(int(float(number) * multiplier))


def _part(label, value=None, **kwargs):
    return dict(label=label, bytes=value, **kwargs)


def _row(kind, parts, notes, **kwargs):
    known = [part['bytes'] for part in parts if part['bytes'] is not None and not part.get('reference')]
    return dict(kind=kind, label='GPU 显存' if kind == 'vram' else '系统内存',
                bytes=sum(known) if known else None, quality='partial',
                components=parts, notes=notes, **kwargs)


def _options(profile, runtime, options_parser, *, model=None):
    """Use the same defaults and last-extra-argument precedence as launch."""
    p = profile.parameters
    if profile.backend == 'sglang':
        from .backends.sglang import effective_options, extra, normalized_paged_parameters
        normalized = normalized_paged_parameters(profile)
        extra(profile.model_copy(update={'parameters': normalized['normalized_parameters'],
                                          'extra_args': normalized['normalized_extra_args']}), runtime)
        return effective_options(model, profile, runtime)
    get = lambda key, default=None: p[key] if p.get(key) not in (None, '') else default
    if profile.backend == 'ninfer':
        from .backends.native_engines import uses_pinned_ninfer_rules
        from .backends.ninfer_capabilities import ninfer_capacity_mode, validate_capacity_parameters
        validate_capacity_parameters(profile, runtime)
        if uses_pinned_ninfer_rules(runtime) and any(arg.startswith('--') and '=' in arg for arg in profile.extra_args):
            raise ValueError('此 NInfer 版本不接受 --参数=值；请将参数和值分为空格分隔的两个 argv。')
        values = {'--max-context': get('context_length', 8192),
                  '--max-concurrency': get('max_concurrency', 1),
                  '--kv-dtype': get('kv_dtype', 'bf16'),
                  '--spec': p.get('acceleration_mode') or ('mtp' if p.get('mtp_enabled') else 'none'),
                  '--draft-tokens': get('draft_tokens', 3)}
        if ninfer_capacity_mode(runtime) == 'context_bytes':
            values['--context-capacity'] = get('context_capacity', 'auto')
        else:
            values['--kv-capacity'] = p.get('kv_capacity', 'auto')
    elif profile.backend == 'sparkinfer':
        values = {'--ctx': p.get('context_length')}
    elif profile.backend == 'vllm':
        # Unspecified arguments use vLLM defaults, not the editor's field defaults.
        values = {'--max-model-len': p.get('context_length'), '--dtype': p.get('dtype') or 'auto',
                  '--gpu-memory-utilization': get('gpu_memory_utilization', {'0.31.0': .92}.get(runtime.version)),
                  '--tensor-parallel-size': get('tensor_parallel_size', 1),
                  '--pipeline-parallel-size': get('pipeline_parallel_size', 1)}
    aliases = {
                   '-tp': '--tensor-parallel-size', '-pp': '--pipeline-parallel-size',
                   '-dp': '--data-parallel-size', '-q': '--quantization'} if profile.backend == 'vllm' else {}
    values.update((aliases.get(flag, flag), value) for flag, value in options_parser(_extra(profile, runtime)))
    if profile.backend == 'ninfer' and '--kv-capacity' in values and values['--kv-capacity'] in (None, ''):
        # A blank overrides the adapter's auto default and omits the CLI flag.
        # ninfer-serve then uses an explicit pool of max-context tokens.
        values['--kv-capacity'] = values['--max-context']
    return values


def _gpu_environment(profile, runtime):
    # vLLM has never applied the source-build environment at launch. The other
    # WSL adapters do, with an explicit profile selection taking precedence.
    environment = {} if profile.backend == 'vllm' else dict(runtime.metadata.get('build_environment', {}))
    return {**environment, **cuda_environment(profile, runtime)}


def cuda_devices_for(profile, runtime):
    """Read identity only, on demand in the runtime's selected distribution."""
    code = ("import json,os,sys\nos.environ.update(json.loads(sys.argv[1]))\n"
            "n={'__name__':'modloader_gpu_identity'};exec(compile(sys.stdin.read(),'modloader_gpu_identity','exec'),n)\n"
            "print(json.dumps(n['cuda_devices']()))")
    try:
        source = Path(__file__).with_name('gpu_identity.py').read_text(encoding='utf-8')
        environment = {key: value for key, value in _gpu_environment(profile, runtime).items()
                       if key in {'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'}}
        return json.loads(wsl_python(runtime_distribution(runtime), code,
            [json.dumps(environment)], input=source, timeout=10))
    except (OSError, ValueError, TypeError):
        return []


def _cuda_device(index, environment):
    visible = environment.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        devices = str(visible).split(',')
        try:
            index = devices[int(index)].strip()
        except (IndexError, ValueError, TypeError):
            return 'CUDA（可见设备映射）'
    return 'CUDA' + str(index) if str(index).isdecimal() else 'CUDA（可见设备映射）'


def _text_config(metadata):
    config = metadata['config']
    if metadata['format'] != 'gguf':
        text = config.get('text_config', config)
        return text if isinstance(text, dict) else {}
    arch = config.get('general.architecture', '')
    get = lambda name: config.get(arch + '.' + name)
    return dict(num_hidden_layers=get('block_count'), hidden_size=get('embedding_length'),
                num_attention_heads=get('attention.head_count'), num_key_value_heads=get('attention.head_count_kv'),
                head_dim=get('attention.key_length'), v_head_dim=get('attention.value_length'),
                max_position_embeddings=get('context_length'), full_attention_interval=get('full_attention_interval'))


def _geometry(config):
    layers = _positive(config.get('num_hidden_layers'))
    types = config.get('layer_types')
    if types is not None:
        if not isinstance(types, list) or len(types) != layers:
            return None
        # Recurrent and linear-attention layers do not own ordinary paged KV.
        if any(t not in {'full_attention', 'sliding_attention', 'linear_attention', 'mamba'} for t in types):
            return None
        layers = sum(t in {'full_attention', 'sliding_attention'} for t in types)
    else:
        # Transformers' Qwen3.5 config constructs a 3-linear / 1-full stack
        # when layer_types and full_attention_interval are absent in raw JSON.
        interval = config.get('full_attention_interval', 4 if str(config.get('model_type', '')).startswith('qwen3_5') else None)
        if interval is not None:
            interval = _positive(interval)
            layers = layers // interval if layers is not None and interval else None
    heads = _positive(config.get('num_key_value_heads')) or _positive(config.get('num_attention_heads'))
    dimension = _positive(config.get('head_dim'))
    if dimension is None:
        hidden, queries = _positive(config.get('hidden_size')), _positive(config.get('num_attention_heads'))
        if hidden and queries and hidden % queries == 0:
            dimension = hidden // queries
    if layers is None or not heads or not dimension:
        return None
    return layers, heads, dimension


def _ninfer_weights(metadata, options):
    directory = metadata['directory']
    config = metadata['config']
    architecture = str(config.get('model_type', '')) + ' ' + ' '.join(config.get('architectures', []))
    q4 = 'Qwen4Exp' in architecture or 'qwen4' in architecture.lower()
    moe = not q4 and ('Moe' in architecture or 'moe' in architecture)
    if not q4 and 'Qwen3_5' not in architecture and 'qwen3_5' not in architecture:
        return [_part('此架构的 GPU 权重放置')], [_part('此架构的 Host 权重放置')]
    gpu, host, uncertain = set(), set(), set()
    active = {'text'}
    if options.get('--spec') in {'mtp', 'dflash', 'dflash2'}:
        active.add(options['--spec'])
    if '--vision' in options:
        active.add('vision')
    if '--lm-head-draft' in options and options.get('--spec') not in {None, 'none'}:
        active.add('proposal')
    for name, binding in directory['bindings'].items():
        if name.split('/')[0] not in active or '/ple/table' in name:
            continue  # PLE stays disk-backed, not a resident RAM/GPU table.
        if '/proposal/' in name and '--lm-head-draft' not in options:
            continue
        parts = binding.get('parts', [binding])
        ids = {part['object'] for part in parts}
        expert = bool(re.search(r'/moe/experts/(?:\d+/)?(?:gate|up|down)$', name))
        if q4 and expert and name.startswith('text/'):
            host.update(ids)
        elif moe and expert and name.startswith('text/'):
            # Qwen3.5 startup tries resident experts, then CPU placement if the
            # real device budget cannot fit weights + calibrated execution plan.
            uncertain.update(ids)
        else:
            gpu.update(ids)
    objects = {item['id']: item for item in directory['objects']}
    total = lambda ids: sum(objects[i]['bytes'] for i in ids if objects[i]['kind'] == 'tensor')
    gpu_parts = [_part('GPU 权重有效载荷', total(gpu))] if gpu else []
    host_parts = [_part('Host 常驻权重有效载荷', total(host))] if host else []
    if uncertain:
        label = f'专家权重动态放置（有效载荷 {total(uncertain):,} B）'
        gpu_parts.append(_part(label))
        host_parts.append(_part(label))
    resources = set()
    for name in active:
        resources.update(directory['components'].get(name, {}).get('resources', {}).values())
    retained = sum(objects[i]['bytes'] for i in resources if i in objects)
    if retained:
        host_parts.append(_part('内嵌前端资源', retained))
    return gpu_parts, host_parts


def _ninfer(metadata, options, runtime, environment):
    config = _text_config(metadata)
    notes = ['按 NInfer v3 已适配的 Qwen3.5 / Qwen4Exp 放置规则，只统计选中组件的父张量；共享父对象去重。',
             '未计入权重对齐、计算工作区、CUDA Graph、驱动、分配器及请求临时峰值。']
    if metadata['format'] == 'ninfer':
        gpu, host = _ninfer_weights(metadata, options)
    else:
        gpu, host = [_part('待转换 NInfer 权重')], []
        notes.append('尚无转换后的 NInfer artifact，权重占用未知。')
    from .backends.ninfer_capabilities import ninfer_capacity_mode, context_capacity_bytes
    if ninfer_capacity_mode(runtime) == 'context_bytes':
        # The new arena owns both KV and State records. An explicit capacity
        # reserves those bytes once; the old token-page/GDN additions double-count it.
        capacity = options.get('--context-capacity', 'auto')
        budget = None if capacity == 'auto' else context_capacity_bytes(capacity)
        host_capacity = options.get('--host-cache-capacity')
        host_budget = (None if host_capacity in (None, 'default')
                       else context_capacity_bytes(host_capacity, allow_zero=True))
        gpu.extend([_part('KV / State 共享池', budget), _part('工作区 / CUDA Graph 等')])
        host.extend([_part('Host KV / State 共享池', host_budget), _part('进程 / 上下文缓存 / 加载临时峰值')])
        notes = ['KV 与 State 共用按字节设置的 Device 池；auto 容量由启动时的可用显存及执行计划确定。', notes[1]]
        return [_row('vram', gpu, notes, device=_cuda_device(options.get('--device') or 0, environment)),
                _row('ram', host, ['Host 池及目录元数据按启动计划分配。'], logical=False)]
    architecture = str(config.get('model_type', '')) + ' ' + ' '.join(config.get('architectures', []))
    qwen35 = 'qwen3_5' in architecture.lower()
    geometry = _geometry(config) if qwen35 else None
    context = _positive(options.get('--max-context'))
    concurrency = _positive(options.get('--max-concurrency'))
    tokens = _positive(options.get('--kv-capacity'))
    auto = options.get('--kv-capacity') == 'auto'
    if auto and context and concurrency:
        tokens = max((context + 63) // 64, concurrency) * 64
        notes.append('KV auto 按最小共享池估算；并发大于 1 时还可按加载时显存扩容，最终池大小未知。')
    kv = None
    if geometry and tokens and concurrency:
        layers, heads, dimension = geometry
        dtype = options.get('--kv-dtype')
        width = 4 * dimension if dtype == 'bf16' else {
            'int8': 528, 'fp8': 516, 'nvfp4': 288, 'k8v4': 402,
        }.get(dtype) if dimension == 256 else None
        if width is not None:
            pages = (tokens + 63) // 64
            kv = pages * 64 * layers * heads * width
            if options.get('--spec') == 'mtp':
                draft = _positive(options.get('--draft-tokens'))
                if draft:
                    kv += (pages + concurrency * ((draft + 63) // 64)) * 64 * heads * width
                else:
                    kv = None
            notes.append(f'KV 按 64 Token 页、{layers} 层全注意力、{heads} KV 头、{dtype} 数据及缩放平面计算；共享容量不乘并发数。')
    gpu.append(_part('KV 最小共享池' if auto else 'KV 共享池', kv))
    # Persistent Qwen3.5 GDN StateImage: BF16 convolution (kernel-1), FP32
    # recurrence and BF16 continuation hidden, 256-byte aligned fields per slot.
    state = None
    types = config.get('layer_types')
    linear = types.count('linear_attention') if isinstance(types, list) else 0
    dimensions = [_positive(config.get(key)) for key in ('linear_num_key_heads', 'linear_key_head_dim',
                  'linear_num_value_heads', 'linear_value_head_dim', 'linear_conv_kernel_dim', 'hidden_size')]
    if qwen35 and linear and concurrency and all(dimensions):
        kh, kd, vh, vd, kernel, hidden = dimensions
        extra = options.get('--device-state-slots', concurrency)
        try:
            extra = int(extra)
            if extra >= 0:
                align = lambda n: (n + 255) // 256 * 256
                image = linear * (align((2 * kh * kd + vh * vd) * (kernel - 1) * 2) + align(vh * vd * kd * 4)) + align(hidden * 2)
                state = image * (concurrency + extra)
        except (ValueError, TypeError):
            pass
    gpu.extend([_part('GDN 常驻状态', state), _part('工作区 / CUDA Graph 等')])
    host.append(_part('进程 / 上下文缓存 / 加载临时峰值'))
    return [_row('vram', gpu, notes, device=_cuda_device(options.get('--device') or 0, environment)),
            _row('ram', host, ['Host 张量按引擎分配计算；额外进程与文件缓存未知。'], logical=False)]


def _dtype_bytes(config, dtype):
    dtype = str(dtype).lower().replace('torch.', '')
    if dtype == 'auto':
        dtype = str(config.get('dtype') or config.get('torch_dtype') or 'float16').lower().replace('torch.', '')
        if dtype == 'float32':
            dtype = 'float16'  # Both HF engines downcast FP32 in auto mode.
    return {'half': 2, 'float16': 2, 'bfloat16': 2, 'float': 4, 'float32': 4}.get(dtype)


def _hf_tensor_sizes(metadata, dtype, speculative):
    config = _text_config(metadata)
    size = _dtype_bytes(config, dtype)
    for tensor in metadata.get('tensors', []):
        if not speculative and re.search(r'(?:^|\.)(?:mtp|nextn)(?:\.|_)|\.eh_proj\.', tensor['name']):
            continue
        if config.get('tie_word_embeddings') and tensor['name'].endswith('lm_head.weight'):
            continue  # The output parameter aliases the embedding, not a second allocation.
        n = tensor['bytes']
        if tensor['dtype'] in {'F16', 'BF16', 'F32', 'F64'} and 'scale' not in tensor['name']:
            n = math.prod(tensor['shape']) * size if size else None
        yield tensor, n


def _hf_weights(metadata, dtype, speculative):
    sizes = [size for _, size in _hf_tensor_sizes(metadata, dtype, speculative) if size is not None]
    total, known = sum(sizes), bool(sizes)
    return total if known else None


def _paged_hf_weights(metadata, dtype, options):
    """Per-layer slot payload, not a percentage of the whole checkpoint.

    Separated expert IDs and E-first packed tensors (including legacy Granite)
    share this accounting. Preserve quantized weights, scales and biases in their
    serialized dtypes; Marlin repacks and runtime-derived tensors remain gaps.
    """
    from .backends.sglang import paged_model_info
    info = paged_model_info(metadata.get('config', {}), options)
    experts, slots = info['num_experts'], _positive(options.get('--paged-experts-num-resident'))
    quantized = bool(info['quantization'])
    nonexpert, layers, unknown = [], {}, bool(metadata.get('errors'))
    for tensor, n in _hf_tensor_sizes(metadata, dtype, False):
        name = tensor['name']
        # SGLang granitemoe.py's HF mapper sends only these legacy routed weights
        # to experts. Router, shared MLP and unrelated input/output linears stay dense.
        legacy = re.search(r'\.block_sparse_moe\.(input_linear|output_linear)\.weight$', name)
        if legacy:
            projection = {'input_linear': 'gate_up_proj', 'output_linear': 'down_proj'}[legacy[1]]
            name = name[:legacy.start()] + '.block_sparse_moe.experts.' + projection
        if not re.search(r'(?:^|\.)experts\.', name):
            nonexpert.append(n)
            continue
        match = re.search(r'(?:^|\.)(?:layers|blocks|h)\.(\d+)\.(?:.*\.)?experts\.(.*)', name)
        # Legacy tensors are [E, 2I, H] / [E, H, I], not an E-wide dense matrix.
        if not match or not experts or legacy and len(tensor['shape']) != 3:
            unknown = True
            continue
        # Quantization layouts are not activation-dtype arrays: packed weights,
        # scale precision and bias precision are part of the expert store.
        if quantized:
            n = tensor['bytes']
        if n is None:
            unknown = True
            continue
        layer = layers.setdefault(match[1], dict(packed=0, separated={}))
        separated = re.match(r'(\d+)\.', match[2])
        if separated:
            index = int(separated[1])
            layer['separated'][index] = layer['separated'].get(index, 0) + n
        elif tensor['shape'] and tensor['shape'][0] == experts and n % experts == 0:
            layer['packed'] += n
        else:
            unknown = True
    resident = host = 0
    for layer in layers.values():
        sizes = layer['separated']
        if sizes and (set(sizes) != set(range(experts)) or len(set(sizes.values())) != 1):
            unknown = True
            continue
        per_expert = layer['packed'] // experts + (next(iter(sizes.values())) if sizes else 0)
        host += per_expert * experts  # All E, including the K already on the GPU.
        if slots:
            resident += per_expert * min(slots, experts)
    # A config override can change E/layer structure or checkpoint placement.
    if options.get('--json-model-override-args') not in (None, '', '{}'):
        unknown = True
    return dict(nonexpert=sum(n for n in nonexpert if n is not None) if nonexpert and all(n is not None for n in nonexpert) else None,
                resident=None if unknown or not layers or not slots else resident,
                host=None if unknown or not layers else host, quantized=quantized)


def _kv_reference(config, context, dtype, options):
    geometry = _geometry(config)
    if not geometry or not context:
        return None
    layers, heads, dim = geometry
    if config.get('kv_lora_rank') or config.get('sliding_window'):
        return None  # MLA/window/state layouts are not ordinary full-context KV.
    cache_dtype = options.get('--kv-cache-dtype', 'auto')
    width = _dtype_bytes(config, dtype) if cache_dtype == 'auto' else {
        'fp8': 1, 'fp8_e4m3': 1, 'fp8_e5m2': 1, 'bfloat16': 2, 'bf16': 2,
    }.get(cache_dtype)
    vdim = _positive(config.get('v_head_dim')) or dim
    return context * layers * heads * (dim + vdim) * width if width else None


def _hf_engine(metadata, profile, runtime, options):
    config = _text_config(metadata)
    vllm = profile.backend == 'vllm'
    raw_context = options.get('--max-model-len' if vllm else '--context-length')
    context = (_human_int(raw_context) if vllm else _positive(raw_context)) if raw_context not in (None, '') else _positive(config.get('max_position_embeddings'))
    dtype = options.get('--dtype', 'auto')
    tp = _positive(options.get('--tensor-parallel-size' if vllm else '--tp-size'))
    pp = _positive(options.get('--pipeline-parallel-size' if vllm else '--pp-size', 1))
    dp = _positive(options.get('--data-parallel-size' if vllm else '--dp-size', 1))
    multi = (tp, pp, dp) != (1, 1, 1) or _positive(options.get('--nnodes', 1)) != 1
    try:
        offload = float(options.get('--cpu-offload-gb') or 0) > 0 or bool(_positive(options.get('--offload-group-size')))
    except (TypeError, ValueError):
        offload = True
    if vllm:
        spec = json.loads(options.get('--speculative-config') or '{}')
        if not isinstance(spec, dict):
            raise ValueError('--speculative-config 必须是 JSON 对象')
        speculative = bool(spec)
        if profile.parameters.get('mtp_enabled') is False and spec.get('method') == 'mtp':
            speculative = False
        elif profile.parameters.get('mtp_enabled') is True:
            speculative = True
        include_mtp = (speculative and (spec.get('method') == 'mtp' or profile.parameters.get('mtp_enabled') is True)
                       and not profile.parameters.get('mtp_path', spec.get('model')))
    else:
        mode = profile.parameters.get('acceleration_mode') or ('mtp' if profile.parameters.get('mtp_enabled') else 'none')
        speculative = mode != 'none'
        include_mtp = mode == 'mtp'
    override = options.get('--hf-overrides' if vllm else '--json-model-override-args')
    changed_weights = bool(options.get('--quantization') or options.get('--quantization-config')
                           or config.get('quantization_config') or metadata['config'].get('quantization_config')
                           or override not in (None, '', '{}') or '--language-model-only' in options)
    device = str(options.get('--device') or 'cuda').lower()
    parts = [_part('权重张量参考', _hf_weights(metadata, dtype, include_mtp), **({'reference': True} if offload or changed_weights else {})),
             _part(f'单序列 KV 需求（{context:,} Token）' if context else '单序列 KV 需求',
                   _kv_reference(config, context, dtype, options), reference=True),
             _part('实际 KV 池 / 状态 / 工作区 / CUDA Graph')]
    notes = ['权重按 safetensors 张量描述及当前 dtype 估算；量化重排、派生副本、张量并行复制及临时峰值未计入。',
             '单序列 KV 是填满上下文的数据量参考，不是加载时分配；实际 KV 池由剩余预算与运行时 profiling 决定，未加入已估新增。']
    paged = not vllm and '--enable-paged-experts' in options
    paged_weights = _paged_hf_weights(metadata, dtype, options) if paged else None
    if paged:
        placement_changed = offload or multi or override not in (None, '', '{}') or '--language-model-only' in options
        parts[0] = _part('非分页权重张量参考' if changed_weights else '非分页权重张量估算', paged_weights['nonexpert'],
                         **({'reference': True} if placement_changed or changed_weights else {}))
        parts.append(_part('分页专家 GPU K-slot 张量参考' if paged_weights['quantized'] else '分页专家 GPU K-slot 张量估算',
                           paged_weights['resident'], **({'reference': True} if placement_changed or paged_weights['quantized'] else {})))
        if paged_weights['quantized']:
            parts.append(_part('分页专家 GPU 重排 / 填充后分配'))
        notes[0] = '按每个 MoE 层的 min(K, E) 个 GPU 槽位计算；共享专家与其他非分页权重单独计入，全部 E 专家另存锁页 Host。'
        if paged_weights['quantized']:
            notes.append('量化专家保留原始 packed 权重、scale 与 bias 布局；张量字节仅作布局参考，Marlin 重排、padding、派生张量会改变实际分配。')
        if paged_weights['host'] is None:
            notes.append('专家张量布局、分片、专家数或 K 不完整，分页专家分配无法估算。')
    if override not in (None, '', '{}'):
        parts[1]['bytes'] = None
        notes.append('模型配置覆盖改变了结构；源权重总量仅作参考，KV 结构待解析。')
    elif changed_weights and not paged:
        notes.append('量化或禁用多模态会改变加载权重；源张量总量仅作参考。')
    if speculative:
        parts.append(_part('MTP / 草稿模型额外分配'))
    if device == 'cpu':
        parts[2]['label'] = '实际 KV 池 / 状态 / 工作区'
        notes[1] = '单序列 KV 是填满上下文的数据量参考；实际 CPU KV 池、状态与工作区另行分配。'
        return [_row('ram', parts, notes, logical=False)]
    fraction = _fraction(options.get('--gpu-memory-utilization' if vllm else '--mem-fraction-static'))
    if vllm and '--kv-cache-memory-bytes' in options:
        parts.append(_part('KV 池预算', _human_int(options['--kv-cache-memory-bytes']), reference=True))
        fraction = None
        notes.append('显式 KV 池预算覆盖显存比例；实际池还按块与混合布局取整。')
    if vllm and fraction is not None:
        notes.append('vLLM 显存比例以每张卡总显存为基准，预算包含权重与 KV。')
    elif not vllm:
        notes.append('SGLang 静态比例以加载前可用显存为基准；KV 使用加载权重后可用量减去运行时预留，默认比例由引擎按设备选择。')
    gpu_device = '多 GPU 合计' if multi else (_cuda_device(options.get('--base-gpu-id') or 0, _gpu_environment(profile, runtime))
                                                     if device in {'cuda', 'auto'} else device.upper())
    gpu = _row('vram', parts, notes, device=gpu_device,
               **({'budget_fraction': fraction, 'budget_basis': 'total' if vllm else 'available'} if fraction is not None else {}))
    if multi:
        gpu['aggregate'] = True
        gpu['notes'].append('并行配置显示未分片的权重/单序列 KV 合计参考，不预测每张 GPU 的放置与复制，也不套用单卡容量。')
    if offload:
        gpu['notes'].append('CPU offload 改变权重放置；权重总量只作参考。')
    host = [_part('进程 / 权重加载 / CPU 缓存')]
    if not vllm:
        host = [_part('进程与运行时'), _part('加载峰值与文件缓存')]
        if paged:
            host.append(_part('分页专家锁页 Host 张量参考' if paged_weights['quantized'] else '分页专家锁页 Host 张量估算',
                              paged_weights['host'], **({'reference': True} if paged_weights['quantized'] or multi else {})))
            if paged_weights['quantized']:
                host.append(_part('分页专家 Host 重排 / 填充后分配'))
        if offload:
            host.append(_part('CPU offload 权重'))
        if '--enable-hierarchical-cache' in options:
            host.append(_part('CPU KV 缓存'))
    host_notes = ['CPU offload、交换空间与加载文件缓存未独立估算。']
    if paged:
        host_notes.append('锁页 Host 保存每层全部 E 专家，包括 GPU 已驻留的 K。WSL 可锁页容量通常约为系统 RAM 的一半，进程与加载峰值另需内存。')
        if offload:
            host_notes.append('额外 CPU offload 的放置与分页槽位交互待引擎解析。')
    return [gpu, _row('ram', host, host_notes, logical=False)]


def _gpu_identities(resources, profile, runtime, options, identities=None):
    gpu = next((item for item in resources if item['kind'] == 'vram'), None)
    if gpu is None:
        return resources
    hf = profile.backend in {'vllm', 'sglang'}
    if hf and str(options.get('--device') or 'cuda').lower() not in {'cuda', 'auto'}:
        gpu['identity_status'] = 'unknown'
        return resources
    if not hf:
        identities = cuda_devices_for(profile, runtime) if identities is None else identities
        ordinal = (options.get('--device') or 0) if profile.backend == 'ninfer' else 0
        gpu.update(identity_fields(ordinal, identities))
        return resources
    vllm = profile.backend == 'vllm'
    tp = _positive(options.get('--tensor-parallel-size' if vllm else '--tp-size'))
    pp = _positive(options.get('--pipeline-parallel-size' if vllm else '--pp-size', 1))
    dp = _positive(options.get('--data-parallel-size' if vllm else '--dp-size', 1))
    nodes = _positive(options.get('--nnodes', 1))
    distributed = (dp != 1 or nodes != 1 or str(options.get('--node-rank') or '0') != '0'
                   or options.get('--distributed-executor-backend') in {'ray', 'external_launcher'}
                   or any(flag in options for flag in ('--dist-init-addr', '--data-parallel-address',
                           '--data-parallel-rank', '--data-parallel-start-rank', '--data-parallel-size-local')))
    try:
        base = int(options.get('--base-gpu-id') or 0) if not vllm else 0
        step = int(options.get('--gpu-id-step') or 1) if not vllm else 1
    except (ValueError, TypeError):
        base, step = -1, 0
    if distributed or not tp or not pp or base < 0 or step < 1 or gpu.get('aggregate') and gpu.get('budget_fraction') is None:
        gpu['identity_status'] = 'unknown'
        return resources
    identities = cuda_devices_for(profile, runtime)
    if tp * pp > len(identities):
        gpu['identity_status'] = 'unknown'
        return resources
    # SGLang compute_local_gpu_id: PP stages occupy tp devices, with the
    # configurable stride applied to TP ranks only. No cross-node inference.
    ordinals = [base + stage * tp + rank * step for stage in range(pp) for rank in range(tp)]
    resolved = [identity_fields(ordinal, identities) for ordinal in ordinals]
    if len(set(ordinals)) != len(ordinals) or any(item['identity_status'] != 'resolved' for item in resolved):
        gpu['identity_status'] = 'unknown'
        return resources
    if not gpu.get('aggregate'):
        gpu.update(resolved[0])
        return resources
    gpu['identity_status'] = 'unknown'  # The aggregate never belongs to a physical card.
    fraction = gpu.get('budget_fraction')
    if fraction is not None:
        budgets = [dict(kind='vram', label='GPU 显存预算', device=_cuda_device(ordinal, _gpu_environment(profile, runtime)),
                        bytes=None, quality='dynamic', notes=[], budget_fraction=fraction,
                        budget_basis=gpu['budget_basis'], **identity)
                   for ordinal, identity in zip(ordinals, resolved)]
        resources[1:1] = budgets
    return resources


def _spark_host(config, hybrid, environment, runtime):
    # SparkInfer 0.6.34: ModelEngine::load -> enable_prefix_cache ->
    # Qwen35Model::warm_snapshot_pool (qwen35.cpp:5085). Unlike the prefix
    # cache's growth limits, these buffers are allocated and touched at load.
    prefix = str(environment.get('SPARKINFER_PREFIX_CACHE', ''))
    deterministic = ('SPARKINFER_DETERMINISTIC' in environment
                     and not str(environment['SPARKINFER_DETERMINISTIC']).startswith('0'))
    enabled = not prefix.startswith('0') and (prefix.startswith('1') or not deterministic)
    pool = not str(environment.get('SPARKINFER_SNAPSHOT_POOL', '')).startswith('0')
    parts = []
    target = None
    target_expected = False
    if enabled and pool and hybrid:
        layers = _positive(config.get('num_hidden_layers'))
        interval = config.get('full_attention_interval')
        interval = interval if type(interval) is int else None
        gate = not str(environment.get('SPARKINFER_MUSE_GDN_GATE', '')).startswith('0')
        needs_state = not gate or interval is not None and interval > 1
        dimensions = [_positive(config.get(key)) for key in ('linear_num_key_heads',
                      'linear_num_value_heads', 'linear_key_head_dim', 'linear_conv_kernel_dim')]
        if runtime.version == '0.6.34' and layers and interval is not None and all(dimensions):
            if needs_state:
                qheads, vheads, dimension, kernel = dimensions
                slots = layers - layers // interval if interval > 0 else layers
                snapshot = (slots * vheads * dimension * dimension * 4
                            + layers * (kernel - 1) * (2 * qheads + vheads) * dimension * 2)
                # The engine uses atoll, clamps negative MB to zero, and warms
                # min(64, cap / snapshot) buffers in each enabled list.
                match = re.match(r'^\s*([+-]?\d+)', str(environment.get('SPARKINFER_SNAPSHOT_POOL_MB', 4096)))
                cap = max(0, int(match[1]) if match else 0) * 1024 ** 2
                count = min(64, cap // snapshot) if snapshot else 0
                if count:
                    target_expected = True
                    target = count * snapshot
                    parts.append(_part('快照预热池（页锁定）', count * snapshot, reference=True))
                    if not str(environment.get('SPARKINFER_SNAPSHOT_MIGRATE', '')).startswith('0'):
                        target += count * snapshot
                        parts.append(_part('快照预热池（可分页）', count * snapshot, reference=True))
        else:
            target_expected = True
            parts.append(_part('快照预热池'))
    parts.extend([_part('进程与运行时'), _part('加载峰值与文件缓存')])
    if enabled:
        parts.append(_part('按需前缀缓存'))
    result = _row('ram', parts, [], logical=False)
    if target_expected:
        # Allocation failure only shortens the warm lists; it does not fail
        # model loading (qwen35.cpp:5042-5058). This is a component target, not
        # a physical-RAM lower bound or a budget for the whole process.
        result.update(quality='dynamic', budget_component='快照池预热')
        if target is not None:
            result['budget_bytes'] = target
    return result


def _spark_compressed_weights(metadata, config, environment, sm120):
    """Mirror 0.6.34 load_compressed_tensors, not the checkpoint's byte sum.

    NVFP4 carries a 256-byte header, its original UE4M3 scales and (on SM120)
    a separate CUTLASS SFB layout. FFN decode/prefill alias the packed payload;
    only the explicitly selected Q4_K path owns a second decode copy.
    """
    keys = ('num_hidden_layers', 'hidden_size', 'vocab_size', 'num_attention_heads',
            'num_key_value_heads', 'head_dim', 'intermediate_size', 'linear_num_key_heads',
            'linear_num_value_heads', 'linear_key_head_dim', 'linear_conv_kernel_dim')
    dimensions = [_positive(config.get(key)) for key in keys]
    if not all(dimensions):
        return None
    layers, hidden, vocab, qheads, kvheads, head, ffn, lq, lv, ld, kernel = dimensions
    interval = config.get('full_attention_interval', 4)
    tensors = {t['name']: t for t in metadata.get('tensors', [])}
    sizes = dict(plain=0, packed=0, scales=0, header=0, sfb=0, q4=0, vision=0, head=0)
    complete = not metadata.get('errors')
    enabled = lambda name: not str(environment.get(name, '')).startswith('0')
    prefill = enabled('SPARKINFER_QWEN38_PREFILL_NVFP4')

    def plain(name, count, kind='plain'):
        nonlocal complete
        t = tensors.get(name)
        if t and t['dtype'] == 'BF16' and math.prod(t['shape']) == count:
            sizes[kind] += count * 2
        else:
            complete = False

    def projection(prefix, rows, cols, route):
        nonlocal complete
        count = rows * cols
        packed = tensors.get(prefix + '.weight_packed')
        weight = packed or tensors.get(prefix + '.weight')
        scale = tensors.get(prefix + '.weight_scale')
        global_scale = tensors.get(prefix + ('.weight_global_scale' if packed else '.weight_scale_2'))
        if not weight:
            complete = False
            return
        nvfp4 = weight['dtype'] == 'U8'
        if nvfp4:
            if (math.prod(weight['shape']) * 2 != count or not scale or not global_scale
                    or math.prod(scale['shape']) * 16 != count or math.prod(global_scale['shape']) != 1):
                complete = False
                return
        elif math.prod(weight['shape']) != count or weight['dtype'] not in {'BF16', 'F8_E4M3', 'F8_E4M3FN'}:
            complete = False
            return

        native = nvfp4 and (route == 'ffn' and enabled('SPARKINFER_QWEN38_DECODE_NVFP4')
                           or route == 'gdn' and enabled('SPARKINFER_Q38_GDN_NVFP4')
                           or route == 'attn' and enabled('SPARKINFER_Q38_ATTN_NVFP4')
                           and enabled('SPARKINFER_Q38_GDN_NVFP4'))
        keep = nvfp4 and (native or route == 'ffn' and prefill and sm120 is True)
        if keep:
            sizes['packed'] += count // 2
            sizes['scales'] += count // 16
            sizes['header'] += 256
            sfb_on = prefill if route == 'ffn' else enabled('SPARKINFER_Q38_GDN_PREFILL_NVFP4')
            if sfb_on and rows % 128 == 0 and cols % 128 == 0:
                if sm120 is None:
                    complete = False
                elif sm120:
                    sizes['sfb'] += count // 16
        elif route == 'gdn' and not nvfp4 and weight['dtype'] in {'F8_E4M3', 'F8_E4M3FN'}:
            # keep_fp8 stores BF16 per-row scales beside the native FP8 matrix.
            if not scale or math.prod(scale['shape']) != rows:
                complete = False
                return
            sizes['packed'] += count
            sizes['scales'] += rows * 2
            native = True

        if not native:
            if count % 256:
                complete = False
                return
            sizes['q4'] += count // 256 * 144
        if route == 'head' and nvfp4 and enabled('SPARKINFER_Q38_HEAD_NVFP4'):
            # The loader normally keeps this beside Q4_K when its early-load
            # free-memory check clears the configured reserve. It can be
            # released for later requests: include it as an estimate, not a
            # guaranteed lower bound.
            sizes['head'] += 256 + count * 9 // 16
            if prefill and rows % 128 == 0 and cols % 128 == 0:
                if sm120 is None:
                    complete = False
                elif sm120:
                    sizes['head'] += count // 16

    plain('model.language_model.embed_tokens.weight', vocab * hidden)
    plain('model.language_model.norm.weight', hidden)
    projection('lm_head', vocab, hidden, 'head')
    for layer in range(layers):
        prefix = f'model.language_model.layers.{layer}.'
        plain(prefix + 'input_layernorm.weight', hidden)
        plain(prefix + 'post_attention_layernorm.weight', hidden)
        if type(interval) is int and interval > 0 and (layer + 1) % interval:
            b = prefix + 'linear_attn.'
            projection(b + 'in_proj_qkv', (2 * lq + lv) * ld, hidden, 'gdn')
            projection(b + 'in_proj_z', lv * ld, hidden, 'gdn')
            projection(b + 'out_proj', hidden, lv * ld, 'gdn')
            for name, count in [('dt_bias', lv), ('A_log', lv), ('norm.weight', ld),
                                ('conv1d.weight', (2 * lq + lv) * ld * kernel),
                                ('in_proj_a.weight', lv * hidden), ('in_proj_b.weight', lv * hidden)]:
                plain(b + name, count)
        else:
            b = prefix + 'self_attn.'
            for name, rows, cols in [('q_proj', qheads * head * 2, hidden),
                                      ('k_proj', kvheads * head, hidden), ('v_proj', kvheads * head, hidden),
                                      ('o_proj', hidden, qheads * head)]:
                projection(b + name, rows, cols, 'attn')
            plain(b + 'q_norm.weight', head)
            plain(b + 'k_norm.weight', head)
        for name, rows, cols in [('gate_proj', ffn, hidden), ('up_proj', ffn, hidden), ('down_proj', hidden, ffn)]:
            projection(prefix + 'mlp.' + name, rows, cols, 'ffn')

    vision = metadata['config'].get('vision_config')
    if isinstance(vision, dict):
        # qwen_vision.cpp uploads these BF16 tensors at load. Position embeddings
        # alone stay in a host FP32 table and are uploaded after image resampling.
        depth = _positive(vision.get('depth'))
        names = {f'model.visual.{base}.{suffix}' for base in
                 ('patch_embed.proj', 'merger.norm', 'merger.linear_fc1', 'merger.linear_fc2')
                 for suffix in ('weight', 'bias')}
        if depth:
            names.update(f'model.visual.blocks.{layer}.{base}.{suffix}' for layer in range(depth)
                         for base in ('norm1', 'norm2', 'attn.qkv', 'attn.proj', 'mlp.linear_fc1', 'mlp.linear_fc2')
                         for suffix in ('weight', 'bias'))
        else:
            complete = False
        for name in names:
            t = tensors.get(name)
            if t and t['dtype'] == 'BF16':
                sizes['vision'] += math.prod(t['shape']) * 2
            else:
                complete = False
    parts = [_part(label, sizes[key]) for key, label in [
        ('plain', 'BF16 常驻权重'), ('packed', '原生量化权重'), ('scales', '量化缩放'),
        ('header', 'NVFP4 对齐头'), ('sfb', 'CUTLASS SFB 缩放副本'), ('q4', 'Q4_K 转换权重'),
        ('head', 'lm_head 原生副本（按空闲显存保留）'), ('vision', '视觉常驻权重')]]
    if not complete:
        parts.append(_part('未确认的权重分配'))
    return parts, complete


def _spark_fixed_buffers(config, environment):
    """Dense HF constructor buffers and Q4_K row scales, excluding CUB scratch."""
    dimensions = [_positive(config.get(key)) for key in ('num_hidden_layers', 'hidden_size', 'vocab_size',
                  'num_attention_heads', 'num_key_value_heads', 'head_dim', 'intermediate_size',
                  'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_conv_kernel_dim')]
    if not all(dimensions):
        return [_part('GDN 常驻状态'), _part('解码与采样缓冲'), _part('Q4_K 预填充行缩放')]
    layers, hidden, vocab, qheads, kvheads, head, ffn, lq, lv, ld, kernel = dimensions
    interval = config.get('full_attention_interval', 4)
    slots = layers - layers // interval if type(interval) is int and interval > 0 else layers
    state = slots * lv * ld * ld * 4 + layers * (kernel - 1) * (2 * lq + lv) * ld * 2
    qdim, kvdim, lqdim, lvdim = qheads * head, kvheads * head, lq * ld, lv * ld
    decode = 2 * (7 * hidden + 5 * qdim + 2 * kvdim + 4 * lqdim + 5 * lvdim + 2 * lv + 1)
    decode += 9 * vocab * 4 + 1024 * 8 + 1120  # Vocab planes, sparse logit-bias and scalar buffers.
    decode += (ffn + hidden) * 4 + ffn * 7 + (ffn // 16 + 1) * 4
    width = max(hidden, qdim, lvdim)
    decode += 2 * width + 8 * (width // 16 + 1)
    decode += qheads * 256 * (head + 2) * 4  # MAX_NSPLITS, not the current split count.
    width = max(hidden, qdim)
    decode += width + (width // 32) * (8 + 36)
    rows = 0 if str(environment.get('SPARKINFER_Q38_PREFILL_QB', '')).startswith('0') else (
        qdim * 2 + kvdim * 2 + hidden * 2 + ffn * 2) * layers * 4
    return [_part('GDN 常驻状态', state), _part('解码与采样缓冲', decode), _part('Q4_K 预填充行缩放', rows)]


def _spark(metadata, profile, runtime, options, identities=None):
    config = dict(_text_config(metadata))
    hybrid = metadata['format'] == 'safetensors'
    if metadata['format'] == 'safetensors':
        # qwen38_config_from_hf_json ignores layer_types and uses this interval.
        config.pop('layer_types', None)
        config.setdefault('full_attention_interval', 4)
    parts, tensors = [], metadata.get('tensors', [])
    env = _gpu_environment(profile, runtime)
    device = next((item for item in identities or [] if item.get('device') == 'CUDA0'), {})
    capability = device.get('compute_capability')
    sm120 = capability in ([12, 0], (12, 0)) if capability is not None else None
    compressed = (_spark_compressed_weights(metadata, config, env, sm120)
                  if metadata['format'] == 'safetensors' and runtime.version == '0.6.34'
                  and 'quantization_config' in metadata['config'] else None)
    # Spark does not upload arbitrary source tensors unchanged. Only count the
    # paths whose loader allocations follow directly from these descriptors.
    known = []
    if metadata['format'] == 'gguf':
        raw = metadata['config']
        arch = raw.get('general.architecture', '')
        names = {t['name'] for t in tensors}
        dense = arch == 'qwen35' and 'blk.0.ffn_gate.weight' in names and 'blk.0.ffn_gate_exps.weight' not in names
        hybrid = (dense or arch == 'muse-glimmer'
                  or any(name in str(raw.get('general.name', '')) for name in ('Qwen3.5-35B-A3B', 'Qwen3.6-35B-A3B'))
                  or {'blk.0.attn_qkv.weight', 'blk.3.attn_q.weight'} <= names)
        config['full_attention_interval'] = (config.get('full_attention_interval') if dense else 4) if hybrid else None
        if hybrid and dense and config['full_attention_interval'] is None:
            config['full_attention_interval'] = 4
        if arch == 'muse-glimmer':
            config['full_attention_interval'] = 0
        mtp = raw.get(arch + '.nextn_predict_layers', 0)
        if arch != 'muse-glimmer' and _positive(config.get('num_hidden_layers')) and type(mtp) is int and mtp >= 0:
            config['num_hidden_layers'] -= mtp
        for t in tensors:
            layer = re.match(r'blk\.(\d+)\.', t['name'])
            if layer and int(layer[1]) >= (config.get('num_hidden_layers') or 0):
                continue
            if t['name'] == 'token_embd.weight' or 'norm' in t['name']:
                known.append(math.prod(t['shape']) * 2)
    else:
        for t in tensors:
            name = t['name']
            if not name.startswith('model.language_model.'):
                continue
            plain = ('embed_tokens.weight' in name or 'norm.weight' in name
                     or re.search(r'\.linear_attn\.(?:dt_bias|A_log|conv1d\.weight|in_proj_[ab]\.weight)$', name))
            native = ('.linear_attn.' in name and t['dtype'] in {'F8_E4M3', 'F8_E4M3FN'}
                      or t['dtype'] == 'U8' and any(k in name for k in ('.mlp.', '.linear_attn.', '.self_attn.')))
            if plain and t['dtype'] == 'BF16' or native:
                known.append(t['bytes'])
    if compressed is None:
        parts.extend([_part('已确认的权重分配', sum(known) if known else None), _part('其余权重重排 / 量化副本')])
    else:
        parts.extend(compressed[0])
        parts.extend(_spark_fixed_buffers(config, env))
    context = _positive(options.get('--ctx')) or 4096  # Qwen35Config default; not HF max_position_embeddings.
    # The server compacts only a genuinely mixed hybrid stack. With interval=1,
    # interval=0, or no full layers, hybrid_kv_layer_slots returns an empty map:
    # every layer gets a slice and there is no additional trap allocation.
    geometry = _geometry({**config, 'full_attention_interval': None})
    kv = None
    if geometry and context:
        layers, heads, dim = geometry
        int8 = not str(env['SPARKINFER_KV_INT8']).startswith('0') if 'SPARKINFER_KV_INT8' in env else (context >= 4096 if hybrid else True)
        blocks = context // 16 + 8
        interval = _positive(config.get('full_attention_interval'))
        full = layers // interval if hybrid and interval else layers
        slots = full + 1 if 0 < full < layers else layers
        kv = blocks * 16 * slots * heads * (2 * dim * (1 if int8 else 2) + (4 if int8 else 0)) + 256 * blocks * 4
    parts.extend([_part('初始 KV 池（含缩放 / trap slice / 块表）', kv),
                  _part('KV 扩容 / CUDA Graph / CUB / 请求工作区 / 草稿' if compressed else 'KV 扩容 / GDN 状态 / 工作区 / 草稿')])
    notes = ['按 SparkInfer 0.6.34 的直接分配计算。',
            '初始 KV 按 ctx、全注意力层与 16 Token 块计算；' + ('环境已关闭 KV 扩容。' if str(env.get('SPARKINFER_KV_GROW', '')).startswith('0') else '默认还会按空闲显存扩容，最终占用未知。'),
             '未估 CUDA Graph、CUB、草稿及请求预填充工作区。' if compressed else '未估权重转换、额外副本、状态、CUDA Graph、草稿及预填充工作区。']
    gpu = _row('vram', parts, notes, device=_cuda_device(0, env))
    if compressed and compressed[1] and kv is not None:
        gpu['quality'] = 'estimate'
        if not str(env.get('SPARKINFER_KV_GROW', '')).startswith('0'):
            # ModelEngine::grow_kv_pool runs before serving. It preserves this
            # actual engine reserve, not an estimator safety margin. A total-
            # memory upper bound can prove growth impossible without creating
            # a CUDA context or reading another process's current free memory.
            match = re.match(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',
                             str(env.get('SPARKINFER_KV_HEADROOM_GIB', '6')))
            headroom = float(match[1]) if match else 0.
            if not math.isfinite(headroom) or headroom < 0:
                gpu['quality'] = 'partial'
            else:
                reserve = int(headroom * 1024 ** 3)
                fixed = gpu['bytes']
                head = next(p['bytes'] for p in parts if p['label'].startswith('lm_head 原生'))
                total = device.get('memory_total')
                # Even the default head can be dropped; do not use a conditional
                # copy to prove that the remaining device capacity is too small.
                max_extra = total - (fixed - head) - reserve if total is not None else None
                threshold = (blocks // 8) * kv / blocks
                if max_extra is not None and (max_extra <= 0 or max_extra < threshold):
                    notes[1] = '初始 KV 按 ctx、全注意力层与 16 Token 块计算；设备容量不足以触发 KV 扩容。'
                elif str(env.get('SPARKINFER_LMCACHE_ENABLE', '')).startswith('1'):
                    # A successfully attached bridge pins the original layout;
                    # a failed sidecar leaves growth enabled. Metadata alone
                    # cannot decide which startup branch will occur.
                    gpu['quality'] = 'partial'
                else:
                    gpu.update(quality='dynamic', fixed_bytes=fixed, reserve_bytes=reserve)
                    # The source caps total_blocks at 2^22. Usually the GPU is
                    # far smaller; retain that cap when a device can exceed it.
                    maximum = fixed + max(0, (1 << 22) - blocks) * kv // blocks
                    if total is not None and total - reserve > maximum:
                        gpu['budget_bytes'] = maximum
                    notes[1] = '初始 KV 按 ctx、全注意力层与 16 Token 块计算；扩容预算按引擎空闲显存目标计算。'
        draft = (profile.parameters.get('acceleration_mode') not in (None, '', 'none')
                 and profile.parameters.get('draft_model') or options.get('--draft-model'))
        if draft and gpu['quality'] == 'estimate':
            gpu['quality'] = 'partial'
    return [gpu,
            _spark_host(config, hybrid, env, runtime)]


def estimate_wsl(model, profile, runtime, options_parser, progress=None):
    report = progress or (lambda event: None)
    try:
        options = _options(profile, runtime, options_parser, model=model)
    except ValueError as error:
        return dict(resources=[], notes=[str(error)], source=profile.backend + '-parameters')
    try:
        report({'stage': 'model_metadata', 'message': '读取模型权重信息'})
        metadata = metadata_for(model, profile, runtime)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        if profile.backend not in {'vllm', 'sglang'}:
            raise
        # A missing/unreadable weight directory must not erase a configured GPU
        # budget. Allocation components stay unknown, not zero or the budget.
        metadata = dict(format='safetensors', config={}, tensors=[], errors=[str(error)])
    report({'stage': 'allocation', 'message': '计算权重、KV 与工作区占用'})
    if profile.backend == 'ninfer':
        resources = _ninfer(metadata, options, runtime, _gpu_environment(profile, runtime))
    elif profile.backend == 'sparkinfer':
        identities = cuda_devices_for(profile, runtime)
        resources = _spark(metadata, profile, runtime, options, identities)
    else:
        resources = _hf_engine(metadata, profile, runtime, options)
    report({'stage': 'gpu_identity', 'message': '匹配 GPU 与可用资源'})
    resources = _gpu_identities(resources, profile, runtime, options,
                               identities if profile.backend == 'sparkinfer' else None)
    errors = metadata.get('errors', [])
    return dict(resources=resources, notes=['部分模型分片缺失或元数据无效；仅显示可读取的已估项。'] if errors else [],
                source=profile.backend + '-metadata', **({'diagnostic': '\n'.join(errors)} if errors else {}))
