"""SGLang's native local-HF server and embedded NEXTN/MTP launch contract."""
import json
from pathlib import PurePosixPath
import re
import uuid

from ..models import Compatibility, LaunchSpec, ParameterField, RuntimeInstallation
from ..gpu_config import cuda_environment
from ..platforms import distribution_name, linux_path, model_target_path, runtime_distribution, validate_model_runtime
from .adapters import BackendAdapter, OFFLINE_ENV, _caps, _context_caps, _extra, _hf, _mapped
from .sglang_versions import CUDA_COMPONENTS, LEGACY_VERSION, RECIPES, RECOMMENDED_VERSION, recipe

VERSION = RECOMMENDED_VERSION
SOURCE_COMMIT = RECIPES[VERSION]['source_commit']
# DeepGEMM requires NVCC >=12.9; FlashInfer's GEMM/MoE JIT also links NVRTC.
GENERATION_KEYS = ['temperature', 'top_p', 'max_tokens', 'top_k', 'repeat_penalty', 'seed']
SPEC_FIELDS = [
    ParameterField(key='speculative_num_steps', label='MTP 草稿步数', type='number', default=3, minimum=1, integer=True, step=1),
    ParameterField(key='speculative_eagle_topk', label='每步草稿候选', type='number', default=1, minimum=1, integer=True, step=1),
    ParameterField(key='speculative_num_draft_tokens', label='验证窗口 Token', type='number', default=4, minimum=1, integer=True, step=1,
                   description='候选数为 1 时，引擎将验证窗口调整为草稿步数 + 1。'),
]
MAPPING = {'context_length': '--context-length', 'mem_fraction_static': '--mem-fraction-static',
           'tensor_parallel_size': '--tp-size', 'pipeline_parallel_size': '--pp-size', 'max_running_requests': '--max-running-requests',
           'dtype': '--dtype', 'cuda_graph_backend_decode': '--cuda-graph-backend-decode',
           'cuda_graph_backend_prefill': '--cuda-graph-backend-prefill',
           'enable_paged_experts': '--enable-paged-experts', 'paged_experts_num_resident': '--paged-experts-num-resident',
           'moe_runner_backend': '--moe-runner-backend', 'disable_shared_experts_fusion': '--disable-shared-experts-fusion'}
PAGED_FIELDS = [
    ParameterField(key='enable_paged_experts', label='Paged Experts', type='boolean'),
    ParameterField(key='paged_experts_num_resident', label='每层驻留专家 K', type='number', minimum=1, integer=True, step=1),
    ParameterField(key='moe_runner_backend', label='MoE 后端', type='text', options=['auto', 'triton', 'marlin']),
    ParameterField(key='disable_shared_experts_fusion', label='禁用共享专家融合', type='boolean', advanced=True),
]
PAGED_MAPPING = {field.key: MAPPING[field.key] for field in PAGED_FIELDS}
OPTION_ALIASES = {'--tp': '--tp-size', '--tensor-parallel-size': '--tp-size',
                  '--pp': '--pp-size', '--pipeline-parallel-size': '--pp-size',
                  '--ep': '--ep-size', '--expert-parallel-size': '--ep-size',
                  '--dp': '--dp-size', '--data-parallel-size': '--dp-size'}
PROTECTED = {'--model-path', '--tokenizer-path', '--enable-metrics', '--speculative-algorithm', '--speculative-algo',
             '--speculative-draft-model-path', '--speculative-draft-model', '--speculative-num-steps',
             '--speculative-eagle-topk', '--speculative-num-draft-tokens'}


def toolchain_environment(linux_root):
    from .source_builds import BUILD_JOBS
    cache = linux_root + '/cache'
    return {'CUDA_HOME': linux_root + '/cuda', 'CUDACXX': linux_root + '/cuda/bin/nvcc',
            'PATH': linux_root + '/bin:' + linux_root + '/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            'LD_LIBRARY_PATH': linux_root + '/cuda/lib64:' + linux_root + '/cuda/lib:/usr/lib/wsl/lib',
            'SGLANG_CACHE_DIR': cache + '/sglang', 'FLASHINFER_WORKSPACE_BASE': cache + '/sglang',
            'DJ_JIT_CACHE_DIR': cache + '/deep-jit', 'TORCH_EXTENSIONS_DIR': cache + '/torch-extensions',
            'TRITON_CACHE_DIR': cache + '/triton', 'TORCHINDUCTOR_CACHE_DIR': cache + '/inductor', 'CUDA_CACHE_PATH': cache + '/nv',
            'MAX_JOBS': str(BUILD_JOBS), 'FLASHINFER_NVCC_THREADS': '1', 'FLASHINFER_NO_DOWNLOAD': '1'}


async def stage_toolchain(root, linux_root, artifacts, progress, cancel, *, version=LEGACY_VERSION):
    """Stage only approved CUDA archives; also usable to complete an existing venv."""
    from ..platforms import windows_to_wsl
    from .source_builds import stage_archives
    selected = recipe(version)
    components = selected['cuda_components']
    missing = [name for name in components if 'cuda_' + name not in artifacts]
    if missing:
        raise ValueError('SGLang 安装清单缺少独立 JIT 工具链：' + ', '.join(missing) + '；请重新开始安装以获取完整清单')
    archives = [dict(archive=windows_to_wsl(str(artifacts['cuda_' + name].resolve())), destination='cuda') for name in components]
    await stage_archives(root, linux_root, archives, progress, cancel)
    return dict(build_environment=toolchain_environment(linux_root), cuda_release=selected['cuda_release'], cuda_components=list(components))


def import_sglang(path, metadata=None, platform='wsl2'):
    if platform != 'wsl2':
        raise ValueError('本次 SGLang 适配器需要 WSL2')
    metadata = dict(metadata or {})
    distribution = distribution_name(metadata.get('distribution', 'ModLoader'))
    path = linux_path(path)
    metadata.update(distribution=distribution, python=path)
    version = str(metadata.pop('version', 'unknown'))
    return RuntimeInstallation(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f'sglang:wsl2:{distribution}:{path}')),
                               backend='sglang', platform='wsl2', name=f'SGLang · {version}', path=path,
                               version=version, metadata=metadata)


def effective(profile, runtime=None):
    values = {field.key: field.default for field in SglangAdapter().parameters_for_runtime(runtime) if field.default is not None}
    values.update({key: value for key, value in profile.parameters.items()
                   if value is not None and not (isinstance(value, str) and not value.strip())})
    return values


def supports_paged_experts(runtime):
    if runtime is None:
        return True
    known = runtime.metadata.get('supported_flags')
    if known:
        return {'--enable-paged-experts', '--paged-experts-num-resident'} <= set(known)
    return bool(RECIPES.get(runtime.version, {}).get('paged_experts'))


def normalized_paged_parameters(profile):
    """Import only the newly mapped flags, including profiles needing repair.

    Retain last-argv precedence and every other token verbatim. No validation,
    filesystem inspection, or speculative-decoding migration happens here.
    """
    from ..resource_estimates import _options
    reverse = {flag: key for key, flag in PAGED_MAPPING.items()}
    switches = {MAPPING[field.key] for field in PAGED_FIELDS if field.type == 'boolean'}
    parameters, remaining, cursor = dict(profile.parameters), [], 0
    for flag, value in _options(profile.extra_args):
        token = profile.extra_args[cursor]
        width = 1 if '=' in token or value is None else 2
        key = reverse.get(flag)
        if key:
            parameters[key] = (True if value is None and '=' not in token else value) if flag in switches else (
                int(value) if key == 'paged_experts_num_resident' and value is not None and re.fullmatch(r'[+-]?\d+', value) else value)
        else:
            remaining.extend(profile.extra_args[cursor:cursor + width])
        cursor += width
    return dict(normalized_parameters=parameters, normalized_extra_args=remaining)


def paged_model_info(config, options=None):
    """One config interpretation for validation, defaults, UI and allocation."""
    options = options or {}
    override_error = ''
    if options.get('--json-model-override-args'):
        try:
            override = json.loads(options['--json-model-override-args'])
            if not isinstance(override, dict):
                raise ValueError('not an object')
            config = {**config, **override}
        except (TypeError, ValueError):
            override_error = '--json-model-override-args 必须是 JSON 对象'
    text = config.get('text_config') or config
    def positive(value):
        return value if type(value) is int and value > 0 else None
    quant = text.get('quantization_config') or config.get('quantization_config') or {}
    method = str(options.get('--quantization') or quant.get('quant_method') or '').lower().replace('_', '-')
    runner, error, condition = None, '', ''
    if not method:
        runner = 'triton' if config else None
    elif method in {'gptq', 'gptq-marlin', 'awq', 'awq-marlin'}:
        bits = quant.get('bits')
        allowed = (4, 8) if method.startswith('gptq') else (4,)
        if bits is not None and bits not in allowed:
            error = f'{method} 仅支持 ' + '/'.join(map(str, allowed)) + ' bit 专家'
        runner = 'marlin'
    elif method in {'mxfp4', 'modelopt-fp4'} or method == 'modelopt' and 'NVFP4' in str(quant.get('quant_algo', '')).upper():
        runner = 'marlin'
    elif method == 'fp8':
        runner = 'triton'
        if quant.get('quant_method', '').lower() == 'fp8' and not quant.get('weight_block_size'):
            error = 'FP8 专家必须是带 weight_block_size 的序列化 block FP8；per-tensor FP8 不支持'
    elif method in {'w8a8-fp8', 'w8a8-int8', 'blockwise-int8'}:
        runner = 'triton'
    elif method == 'compressed-tensors':
        groups = list((quant.get('config_groups') or {}).values())
        runners = set()
        for group in groups:
            weights, inputs = group.get('weights') or {}, group.get('input_activations')
            bits, kind, strategy = weights.get('num_bits'), weights.get('type'), weights.get('strategy')
            if kind == 'int' and bits in (4, 8) and not inputs:
                runners.add('marlin')
            elif kind == 'float' and bits == 8:
                runners.add('triton')
                if strategy not in {'channel', 'block'} or not inputs or not inputs.get('dynamic'):
                    error = 'compressed-tensors FP8 需要 per-channel/block 权重与 dynamic activations'
            elif kind == 'float' and bits == 4 and strategy == 'tensor_group' and weights.get('group_size') == 16:
                runners.add('marlin')
                if inputs:
                    condition = 'compressed-tensors NVFP4 W4A4 在 Blackwell（SM >= 10.0）不支持分页；需 NVFP4A16 权重'
        if len(runners) == 1:
            runner = runners.pop()
    elif method in {'bitsandbytes', 'gguf', 'hqq', 'quark', 'mxfp8', 'fp4', 'modelopt-fp8', 'w4afp8'}:
        error = f'{method} 专家不在 SGLang 0.5.23 Paged Experts 支持范围内'
    if any('expert' in str(pattern) for pattern in [*(quant.get('ignore') or []), *(quant.get('modules_to_not_convert') or [])]):
        # Global quantization does not establish the method of an excluded MoE
        # layer. Do not reject a valid unquantized/mixed expert checkpoint.
        runner, error = None, ''
    return dict(top_k=positive(text.get('num_experts_per_tok') or text.get('moe_top_k')),
                num_experts=positive(text.get('n_routed_experts') or text.get('num_experts') or text.get('num_local_experts')),
                recommended_runner=runner, quantization=method, quantization_error=error,
                model_override_error=override_error,
                hardware_condition=condition,
                shared_experts=positive(text.get('n_shared_experts') or text.get('num_shared_experts')
                                        or text.get('shared_expert_intermediate_size')),
                fused_shared_experts=positive(text.get('num_fused_shared_experts')))


def _acceleration_mode(profile):
    return profile.parameters.get('acceleration_mode') or ('mtp' if profile.parameters.get('mtp_enabled') else 'none')


def launch_arguments(model, profile, runtime):
    """The effective argument source shared by launch, editor and estimates."""
    from ..resource_estimates import _options
    normalized = normalized_paged_parameters(profile)
    profile = profile.model_copy(update={'parameters': normalized['normalized_parameters'],
                                         'extra_args': normalized['normalized_extra_args']})
    values = effective(profile, runtime)
    if profile.parameter_editor_version == 1:
        values.update({key: None for key, value in profile.parameters.items() if value is None})
    initial = dict(_options([*_mapped(values, MAPPING), *profile.extra_args]))
    enabled = '--enable-paged-experts' in initial
    if not enabled:
        values.pop('paged_experts_num_resident', None)  # Preserve dormant invalid K, but do not send it.
    else:
        info = paged_model_info(model.metadata if model is not None else {}, initial)
        known = runtime.metadata.get('supported_flags')
        if (not initial.get('--moe-runner-backend') and info['recommended_runner']
                and (not known or '--moe-runner-backend' in known)):
            values['moe_runner_backend'] = info['recommended_runner']
        if ('disable_shared_experts_fusion' not in profile.parameters
                and '--disable-shared-experts-fusion' not in initial and '--enforce-shared-experts-fusion' not in initial
                and (not known or '--disable-shared-experts-fusion' in known)):
            values['disable_shared_experts_fusion'] = True
    args = _mapped(values, MAPPING)
    if _acceleration_mode(profile) != 'none':
        args += ['--speculative-algorithm', 'NEXTN']
        args += _mapped(values, {field.key: '--' + field.key.replace('_', '-') for field in SPEC_FIELDS})
    return [*args, *profile.extra_args]


def effective_options(model, profile, runtime):
    from ..resource_estimates import _options
    return {OPTION_ALIASES.get(flag, flag): value for flag, value in _options(launch_arguments(model, profile, runtime))}


def paged_experts(model, profile, runtime, *, devices=None):
    """Editable preflight state; unknown metadata is not admission control."""
    options = effective_options(model, profile, runtime)
    info = paged_model_info(model.metadata, options)
    raw_k = options.get('--paged-experts-num-resident')
    k = int(raw_k) if raw_k is not None and re.fullmatch(r'[+-]?\d+', str(raw_k)) else raw_k
    enabled = '--enable-paged-experts' in options
    result = dict(enabled=enabled, num_resident=k, runner=options.get('--moe-runner-backend'),
                  recommended_runner=info['recommended_runner'], top_k=info['top_k'], num_experts=info['num_experts'],
                  disable_shared_experts_fusion='--disable-shared-experts-fusion' in options,
                  status='disabled' if not enabled else 'available', message='', **normalized_paged_parameters(profile))
    if not enabled:
        result['num_resident'] = result['normalized_parameters'].get('paged_experts_num_resident')
        return result
    problems = []
    if not supports_paged_experts(runtime):
        problems.append(f'Paged Experts 不受此 SGLang {runtime.version} 运行时支持')
    for key in ('enable_paged_experts', 'disable_shared_experts_fusion'):
        if options.get(MAPPING[key]) is not None:
            problems.append(MAPPING[key] + ' 为开关，不接受值')
    if k is None or k == '':
        problems.append('每层驻留专家 K 为必填项')
    elif type(k) is not int:
        problems.append('每层驻留专家 K 必须是整数')
    elif k < 1:
        problems.append('每层驻留专家 K 必须为正整数')
    elif info['top_k'] is not None and k < info['top_k']:
        problems.append(f'每层驻留专家 K 必须不小于 router top-k（{info["top_k"]}）')
    for flag in ('--tp-size', '--pp-size', '--ep-size', '--dp-size'):
        value = str(options.get(flag, 1))
        if not re.fullmatch(r'[+-]?\d+', value) or int(value) != 1:
            problems.append(flag + ' 必须为 1（单 GPU）')
    if options.get('--device') == 'cpu':
        problems.append('需要 GPU，不支持 --device cpu')
    if _acceleration_mode(profile) != 'none' or options.get('--speculative-algorithm') not in (None, '', 'none', 'NONE'):
        problems.append('不支持 MTP / NEXTN 等推测解码')
    for flag in ('--enable-eplb', '--enable-fused-moe-sum-all-reduce'):
        if flag in options:
            problems.append('不支持 ' + flag)
    if options.get('--load-format') == 'dummy':
        problems.append('不支持 --load-format dummy')
    if (info['fused_shared_experts'] or info['shared_experts']) and '--disable-shared-experts-fusion' not in options:
        # Shared expert existence does not by itself prove that fusion is active.
        if info['fused_shared_experts'] or '--enforce-shared-experts-fusion' in options:
            problems.append('不支持共享专家融合；请使用 --disable-shared-experts-fusion')
    graph = dict(decode=options.get('--cuda-graph-backend-decode', 'full'),
                 prefill=options.get('--cuda-graph-backend-prefill', 'breakable'))
    for phase in graph:
        if '--disable-cuda-graph' in options or f'--disable-{phase}-cuda-graph' in options:
            graph[phase] = 'disabled'
    if options.get('--cuda-graph-config'):
        try:
            config = json.loads(options['--cuda-graph-config'])
            for phase in graph:
                graph[phase] = config.get(phase, {}).get('backend', graph[phase])
        except (ValueError, TypeError, AttributeError):
            problems.append('--cuda-graph-config 必须是有效的 CUDA Graph JSON 配置')
    for phase, label, allowed in [('decode', '解码', ('full', 'disabled')), ('prefill', '预填充', ('breakable', 'disabled'))]:
        if graph[phase] not in allowed:
            problems.append(label + ' CUDA Graph 只接受 ' + ' / '.join(allowed))
    runner = result['runner']
    if runner not in (None, '', 'auto', 'triton', 'marlin'):
        problems.append('MoE 后端只接受 triton / marlin（或 auto 由引擎解析）')
    elif runner not in (None, '', 'auto') and info['recommended_runner'] and runner != info['recommended_runner']:
        problems.append(f'{info["quantization"] or "未量化"} 专家需要 {info["recommended_runner"]} MoE 后端')
    if info['quantization_error']:
        problems.append(info['quantization_error'])
    if info['model_override_error']:
        problems.append(info['model_override_error'])
    hardware_unknown = bool(info['hardware_condition'])
    if info['hardware_condition'] and devices:
        capabilities = [d.get('compute_capability') for d in devices]
        if all(isinstance(c, (list, tuple)) and len(c) >= 1 and c[0] >= 10 for c in capabilities):
            problems.append(info['hardware_condition'])
        elif all(isinstance(c, (list, tuple)) and len(c) >= 1 and 8 <= c[0] < 10 for c in capabilities):
            hardware_unknown = False
    if problems:
        result.update(status='unsupported', message='Paged Experts：' + '；'.join(problems))
    elif not info['top_k'] or not info['num_experts'] or not info['recommended_runner'] or hardware_unknown:
        result.update(status='unverified', message=(info['hardware_condition'] if hardware_unknown else '') or '模型专家数、router top-k 或量化元数据缺失，相关约束由引擎校验')
    elif info['shared_experts'] and not result['disable_shared_experts_fusion']:
        result.update(status='unverified', message='共享专家融合状态由引擎解析；若启用了融合，需使用 --disable-shared-experts-fusion')
    elif runner in (None, '', 'auto'):
        result.update(status='unverified', message='auto MoE 后端由引擎解析；Paged Experts 需要 triton / marlin')
    return result


def acceleration(model, profile, runtime):
    mode = _acceleration_mode(profile)
    result = dict(mode=mode, enabled=mode != 'none', status='disabled' if mode == 'none' else 'unverified',
                  kind='mtp', label='原生 MTP / NEXTN', requires_draft=False, fields=[field.model_dump() for field in SPEC_FIELDS],
                  message='')
    if mode not in {'none', 'mtp'}:
        raise ValueError('SGLang 当前只映射 none / mtp（NEXTN）加速模式')
    if mode == 'mtp':
        from ..mtp import model_weight_info
        info = model_weight_info(model)
        if info.get('has_mtp') is False:
            result.update(status='unsupported', message='所选本地权重没有检测到 MTP 张量，不能启用 NEXTN')
        elif info.get('has_mtp') is True:
            result.update(status='available')
    return result


def validate(model, profile, runtime, *, devices=None):
    from ..parameter_rules import validate_number
    normalized = normalized_paged_parameters(profile)
    profile = profile.model_copy(update={'parameters': normalized['normalized_parameters'],
                                         'extra_args': normalized['normalized_extra_args']})
    values = effective(profile, runtime)
    paged = paged_experts(model, profile, runtime, devices=devices)
    if paged['status'] == 'unsupported':
        raise ValueError(paged['message'])
    state = acceleration(model, profile, runtime)
    for field in SglangAdapter().parameters_for_runtime(runtime):
        if not state['enabled'] and field.key in {field.key for field in SPEC_FIELDS}:
            continue
        if field.key == 'paged_experts_num_resident':
            continue  # Check only the effective, active last-argv K.
        if field.type == 'select' and values.get(field.key) not in (None, '') and values[field.key] not in field.options:
            if field.key in {'cuda_graph_backend_decode', 'cuda_graph_backend_prefill'}:
                raise ValueError(f'SGLang {runtime.version} 不支持 {field.label}={values[field.key]}；请选择 ' + ' / '.join(field.options))
            raise ValueError(f'{field.label}不是该引擎支持的选项')
        if field.type == 'number':
            validate_number(field, values.get(field.key))
    if state['status'] == 'unsupported':
        raise ValueError(state['message'])
    # The ordinary NEXTN resolver normalizes linear-chain width to steps + 1.
    # Independent IQuest draft checkpoints have a separate hard constraint;
    # this adapter does not expose their protected draft-model arguments.
    known = runtime.metadata.get('supported_flags')
    if known:
        from ..resource_estimates import _options
        emitted = {flag for flag, _ in _options(launch_arguments(model, profile, runtime))}
        flags = [flag for flag in MAPPING.values() if flag in emitted]
        if state['enabled']:
            flags += ['--speculative-algorithm', *['--' + field.key.replace('_', '-') for field in SPEC_FIELDS]]
        missing = set(flags) - set(known)
        if missing:
            raise ValueError('此 SGLang 运行时 --help 未列出配置所需参数：' + '、'.join(sorted(missing)))
    extra(profile, runtime)


def extra(profile, runtime):
    args = _extra(profile, runtime)
    contract = RECIPES.get(runtime.version, {})
    for index, item in enumerate(args):
        flag, separator, inline = item.partition('=')
        if flag in PROTECTED:
            raise ValueError(f'{flag} 由 SGLang 模型、监控或加速配置管理，不能在高级参数中覆盖')
        if not runtime.metadata.get('supported_flags') and flag in contract.get('removed_flags', ()):
            raise ValueError(f'SGLang {runtime.version} 已移除 {flag}；请从高级参数中移除')
        if runtime.version != '0.5.23':
            continue
        value = inline if separator else args[index + 1] if index + 1 < len(args) else None
        if flag in {'--cuda-graph-backend-decode', '--cuda-graph-backend-prefill'} and value == 'tc_piecewise':
            raise ValueError(f'SGLang {runtime.version} 已移除 tc_piecewise；请选择 full / breakable / disabled')
        if flag == '--cuda-graph-config':
            try:
                config = json.loads(value or '')
            except (ValueError, TypeError):
                raise ValueError('--cuda-graph-config 必须是 JSON 对象') from None
            if not isinstance(config, dict):
                raise ValueError('--cuda-graph-config 必须是 JSON 对象')
            for phase in ('decode', 'prefill'):
                settings = config.get(phase)
                if not isinstance(settings, dict):
                    continue
                if settings.get('backend') == 'tc_piecewise':
                    raise ValueError(f'SGLang {runtime.version} 已移除 tc_piecewise；请选择 full / breakable / disabled')
                if 'tc_compiler' in settings:
                    raise ValueError(f'SGLang {runtime.version} 已移除 tc_compiler；请从 CUDA Graph 配置中移除')
    return args


def map_generation(payload):
    result = dict(payload)
    if 'repeat_penalty' in result:
        value = result.pop('repeat_penalty')
        if 'repetition_penalty' in result and result['repetition_penalty'] != value:
            raise ValueError('repeat_penalty 与 repetition_penalty 冲突')
        result['repetition_penalty'] = value
    return result


class SglangAdapter(BackendAdapter):
    id, label = 'sglang', 'SGLang (WSL2)'

    def parameters(self):
        return self.parameters_for_runtime(None)

    def parameters_for_runtime(self, runtime):
        # Unknown external versions retain the original editor choices; a newer
        # fixed recipe must not impose its restrictions on another runtime.
        version = runtime.version if runtime is not None else LEGACY_VERSION
        graph_backends = RECIPES.get(version, RECIPES[LEGACY_VERSION])['cuda_graph_backends']
        fields = [ParameterField(key='context_length', label='最大上下文', type='number', minimum=1, integer=True, step=1,
                               description='留空使用模型 config.json 的上下文长度。'),
                ParameterField(key='mem_fraction_static', label='静态显存比例', type='number', minimum=0, exclusive_minimum=True, maximum=1, step=.05),
                ParameterField(key='tensor_parallel_size', label='张量并行', type='number', default=1, minimum=1, integer=True, step=1),
                ParameterField(key='pipeline_parallel_size', label='流水线并行', type='number', minimum=1, integer=True, step=1),
                ParameterField(key='cuda_visible_devices', label='CUDA 可见设备', type='text'),
                ParameterField(key='max_running_requests', label='最大并行请求', type='number', minimum=1, integer=True, step=1),
                ParameterField(key='dtype', label='数据类型', type='select', default='auto', options=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32']),
                ParameterField(key='cuda_graph_backend_decode', label='解码 CUDA Graph', type='select', options=list(graph_backends), advanced=True),
                ParameterField(key='cuda_graph_backend_prefill', label='预填充 CUDA Graph', type='select', options=list(graph_backends), advanced=True),
                ParameterField(key='acceleration_mode', label='推测解码', type='select', default='none', options=['none', 'mtp']),
                *[field.model_copy(deep=True) for field in SPEC_FIELDS]]
        if supports_paged_experts(runtime):
            fields += [field.model_copy(deep=True) for field in PAGED_FIELDS]
        return fields

    def inspect(self, model, runtime):
        validate_model_runtime(model, runtime)
        result = _hf(model)
        if runtime.platform != 'wsl2':
            return Compatibility(status='unsupported', reasons=['SGLang 需要 WSL2 环境'])
        if result.status != 'unsupported' and runtime.version not in RECIPES:
            result.status = 'unverified'
            result.reasons.append(f'适配器版本 {" / ".join(RECIPES)}；所选运行时版本 {runtime.version}')
        result.capabilities.update(generation_parameters=GENERATION_KEYS, stream_usage=True)
        return result

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        validate(model, profile, runtime)
        argv = [runtime.path, '-m', 'sglang.launch_server', '--model-path', model_target_path(model, runtime),
                '--host', '127.0.0.1', '--port', str(context.port), '--served-model-name', profile.alias, '--enable-metrics']
        argv += launch_arguments(model, profile, runtime)
        return LaunchSpec(platform='wsl2', distribution=runtime_distribution(runtime), argv=argv,
                          cwd=str(PurePosixPath(runtime.path).parent),
                          env={**runtime.metadata.get('build_environment', {}), **OFFLINE_ENV,
                               'FLASHINFER_NO_DOWNLOAD': '1', 'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION': '0',
                               **cuda_environment(profile, runtime)}, model_name=profile.alias,
                          health_path='/health', capabilities=_context_caps(self.inspect(model, runtime).capabilities, argv, '--context-length'), startup_timeout=900)
