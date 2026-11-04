"""Numeric editor contracts shared by profile saves, previews and launches."""
import math
import re
import struct
from functools import lru_cache
from pathlib import Path

from .models import ParameterField

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
INT64_MIN, INT64_MAX, UINT64_MAX = -(2**63), 2**63 - 1, 2**64 - 1


def validate_number(field, value):
    if value is None or isinstance(value, str) and not value.strip():
        return
    if not isinstance(value, bool) and value in field.special_values:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{field.label}必须是有限数字')
    if field.integer and not isinstance(value, int):
        raise ValueError(f'{field.label}必须是整数')
    if field.minimum is not None and (value < field.minimum or field.exclusive_minimum and value == field.minimum):
        relation = '大于' if field.exclusive_minimum else '不小于'
        special = f'；特殊值：{", ".join(map(str, field.special_values))}' if field.special_values else ''
        raise ValueError(f'{field.label}必须{relation} {field.minimum:g}{special}')
    if field.maximum is not None and value > field.maximum:
        raise ValueError(f'{field.label}不能大于 {field.maximum:g}')


@lru_cache(maxsize=64)
def _model_metadata(path, size, modified):
    from .assets import gguf_metadata
    return gguf_metadata(Path(path))


def gpu_layer_limit(model, runtime=None):
    if runtime is not None and runtime.metadata.get('variant') == 'cpu':
        return 0
    metadata = model.metadata
    # Registrations created before MTP support may omit nextn_predict_layers.
    # Reading the header does not allocate weights or start a model.
    from .platforms import location_of
    if model.format.lower() == 'gguf' and location_of(model).platform == 'windows':
        try:
            path = Path(model.path)
            stamp = path.stat()
            metadata = {**metadata, **_model_metadata(str(path), stamp.st_size, stamp.st_mtime_ns)}
        except (OSError, ValueError):
            pass
    arch = metadata.get('general.architecture') or model.architecture or ''
    blocks = metadata.get(arch + '.block_count')
    # llama.cpp loads BLOCK_COUNT into n_layer_all, including nextn/MTP blocks.
    # load_tensors() and n_gpu_layers() use n_layer_all + 1 (the output layer),
    # NOT n_layer(), which subtracts nextn. Verified at installed commit 0adcc3bb5.
    if model.format.lower() == 'gguf' and type(blocks) is int and blocks > 0:
        return blocks + 1
    return None


def profile_fields(adapter, model=None, runtime=None):
    fields = (adapter.parameters_for_runtime(runtime) if hasattr(adapter, 'parameters_for_runtime')
              else adapter.parameters())
    if model is not None and getattr(adapter, 'id', '') == 'strata':
        from .strata_deepseek import is_deepseek, fields as deepseek_fields
        if is_deepseek(model):
            fields = deepseek_fields()
    if getattr(adapter, 'id', '') == 'custom':
        from .backends.custom import custom_fields
        fields = custom_fields(runtime)
    for field in fields:
        if adapter.id == 'llama_cpp' and field.key == 'gpu_layers' and model is not None:
            field.maximum = gpu_layer_limit(model, runtime)
            field.description = ''
    return [*fields, ParameterField(key='startup_timeout', label='启动等待时间（秒）', type='number',
                                   minimum=0, exclusive_minimum=True, step=1, advanced=True,
                                   description='留空使用该后端的等待时间。')]


def generation_fields(backend, runtime=None):
    # Limits follow the serving API; automatic/clamped inputs remain engine-owned.
    fields = [
        ParameterField(key='temperature', label='温度', type='number', minimum=None if backend in ('llama_cpp', 'strata') else 0, step=.1,
                       description='非正数为贪心采样。' if backend == 'strata' else '0 为贪心采样。'),
        ParameterField(key='top_p', label='Top P', type='number', minimum=0, maximum=None if backend == 'llama_cpp' else 1,
                       exclusive_minimum=backend in ('strata', 'vllm', 'sglang'), step=.05),
        ParameterField(key='max_tokens', label='最大输出 Token', type='number', minimum=None if backend == 'strata' else 0 if backend in ('llama_cpp', 'sglang') else 1, integer=True, step=1,
                       maximum=INT32_MAX if backend in ('llama_cpp', 'ninfer', 'sparkinfer') else None,
                       special_values=[-1, -2] if backend == 'llama_cpp' else [],
                       description='-1 不限长度。' if backend == 'llama_cpp' else '非正数使用剩余上下文。' if backend == 'strata' else ''),
        ParameterField(key='top_k', label='Top K', type='number', minimum=INT32_MIN if backend == 'llama_cpp' else None if backend in ('strata', 'sglang', 'custom') else 0,
                       maximum=INT32_MAX if backend == 'llama_cpp' else INT64_MAX if backend == 'sparkinfer' else None,
                       integer=True, step=1, special_values=[-1] if backend in ('vllm', 'sglang') else [],
                       description='非正数关闭 Top K。' if backend == 'llama_cpp' else '超出 1–64 时，引擎使用 64。' if backend == 'strata' else '-1 或 0 关闭 Top K。' if backend == 'vllm' else '-1 关闭 Top K。' if backend == 'sglang' else '由自定义引擎解释此整数值。' if backend == 'custom' else '0 关闭 Top K。'),
        ParameterField(key='repeat_penalty', label='重复惩罚', type='number', minimum=0, maximum=2 if backend == 'sglang' else None,
                       exclusive_minimum=True, step=.05),
        ParameterField(key='seed', label='随机种子', type='number', integer=True, step=1,
                       minimum=-1 if backend == 'llama_cpp' else None,
                       maximum=4294967295 if backend == 'llama_cpp' else None,
                       description='-1 使用随机种子。' if backend == 'llama_cpp' else ''),
    ]
    if backend == 'vllm' and (runtime is None or runtime.version.lstrip('v') == '0.31.0'):
        next(field for field in fields if field.key == 'temperature').maximum = 2
    seed = next(field for field in fields if field.key == 'seed')
    if backend in ('vllm', 'sglang', 'transformers', 'ninfer'):
        seed.minimum = INT64_MIN
        seed.maximum = UINT64_MAX if backend in ('transformers', 'ninfer') else INT64_MAX
    elif backend == 'sparkinfer':
        seed.maximum = INT64_MAX
    if backend == 'custom':
        from .backends.custom import custom_config, DEFAULT_GENERATION_PARAMETERS
        supported = custom_config(runtime, check_paths=False)['generation_parameters'] if runtime is not None else DEFAULT_GENERATION_PARAMETERS
        fields = [field for field in fields if field.key in supported]
        # Declaring a custom API field does not declare a particular engine's range.
        for field in fields:
            field.minimum = field.maximum = None
            field.exclusive_minimum = False
            field.description = ''
    if backend in {'ninfer', 'sparkinfer'}:
        from .backends.native_engines import NINFER_GENERATION, SPARK_GENERATION
        supported = NINFER_GENERATION if backend == 'ninfer' else SPARK_GENERATION
        fields = [field for field in fields if field.key in supported]
        for field in fields:
            # NInfer 64d10fb/src/serve/translate.cpp and SparkInfer
            # f49c0e4/server/src/chat_tools.cpp reject temperature outside [0, 2].
            if field.key == 'temperature': field.maximum = 2
            if field.key == 'max_tokens' and backend == 'ninfer': field.minimum = 0
            # NInfer's request parser and sampling kernel support at most 20 candidates.
            if field.key == 'top_k' and backend == 'ninfer': field.maximum = 20
            if field.key == 'seed' and backend == 'sparkinfer': field.minimum = 0
        if backend == 'ninfer' and runtime is not None:
            from .backends.native_engines import uses_pinned_ninfer_rules
            if uses_pinned_ninfer_rules(runtime):
                # Verified src/serve/serve_options.h protocol default. Sampling
                # defaults instead depend on the artifact and thinking mode.
                next(field for field in fields if field.key == 'max_tokens').default = 8192
    return fields


def normalize_generation(values, backend):
    values = dict(values)
    if backend in ('vllm', 'sglang') and 'repetition_penalty' in values:
        value = values.pop('repetition_penalty')
        if 'repeat_penalty' in values and values['repeat_penalty'] != value:
            raise ValueError('repeat_penalty 与 repetition_penalty 冲突')
        values['repeat_penalty'] = value
    return values


def validate_generation(values, backend, runtime=None, *, profile=False):
    values = normalize_generation(values, backend)
    fields = generation_fields(backend, runtime)
    if backend == 'custom':
        from .backends.custom import GENERATION_PARAMETERS
        supported = {field.key for field in fields}
        unsupported = {key for key in values if key not in supported and (profile or key in GENERATION_PARAMETERS)}
        if unsupported:
            raise ValueError('此自定义引擎未声明支持生成参数：' + '、'.join(sorted(unsupported)))
    elif backend in {'ninfer', 'sparkinfer'}:
        if 'repeat_penalty' in values or backend == 'sparkinfer' and 'repetition_penalty' in values:
            raise ValueError('此引擎没有可用的重复惩罚采样入口，请移除此参数')
        if backend == 'ninfer' and values.get('repetition_penalty') not in (None, 1, 1.0):
            raise ValueError('NInfer 只接受兼容用途的 repetition_penalty=1；不提供重复惩罚采样器')
    for field in fields:
        value = values.get(field.key)
        validate_number(field, value)
        if backend == 'sglang' and field.key == 'top_k' and isinstance(value, int):
            temperature = values.get('temperature')
            greedy = isinstance(temperature, (int, float)) and 0 <= temperature < 1e-6
            if not greedy and value != -1 and value < 1:
                raise ValueError('非贪心采样的 Top K 必须为 -1 或正整数')
        if backend == 'llama_cpp' and field.key == 'repeat_penalty' and isinstance(value, (int, float)):
            try:
                native = struct.unpack('f', struct.pack('f', value))[0]
                reciprocal = struct.unpack('f', struct.pack('f', 1 / native))[0]
                valid = math.isfinite(native) and math.isfinite(reciprocal)
            except (OverflowError, ZeroDivisionError, struct.error):
                valid = False
            if not valid:
                raise ValueError('重复惩罚及其倒数必须可表示为有限 float32')


def inactive_parameters(profile):
    if profile.backend not in {'sglang', 'ninfer', 'sparkinfer'}:
        return set()
    if profile.backend == 'sglang':
        from .backends.sglang import SPEC_FIELDS, normalized_paged_parameters
        parameters = normalized_paged_parameters(profile)['normalized_parameters']
        inactive = set() if parameters.get('acceleration_mode', 'mtp' if parameters.get('mtp_enabled') else 'none') != 'none' else {field.key for field in SPEC_FIELDS}
        if not parameters.get('enable_paged_experts'):
            inactive.add('paged_experts_num_resident')
        return inactive
    if profile.parameters.get('acceleration_mode', 'mtp' if profile.parameters.get('mtp_enabled') else 'none') != 'none':
        return set()
    from .backends.native_engines import NINFER_ACCEL_FIELDS, SPARK_ACCEL_FIELDS
    return {field.key for field in NINFER_ACCEL_FIELDS + SPARK_ACCEL_FIELDS}


def validate_strata_expert_cache(value):
    # Native atoi accepts a numeric prefix; values below -1 are then rejected.
    match = re.match(r'\s*([+-]?\d+)', str(value))
    if match and int(match[1]) < -1:
        raise ValueError('专家缓存必须为 auto、-1 或非负整数')


def validate_profile(model, profile, runtime, adapter, *, legacy_gpu=False):
    if profile.backend == 'sglang':
        from .backends.sglang import normalized_paged_parameters
        normalized = normalized_paged_parameters(profile)
        profile = profile.model_copy(update={'parameters': normalized['normalized_parameters'],
                                             'extra_args': normalized['normalized_extra_args']})
    from .launch_parameters import validate_editor_profile
    validate_editor_profile(model, profile, runtime)
    from .platforms import validate_model_runtime
    from .gpu_config import validate_gpu_parameters
    validate_model_runtime(model, runtime)
    validate_gpu_parameters(profile, runtime)
    inactive = inactive_parameters(profile)
    for field in profile_fields(adapter, model, runtime):
        if field.key in inactive:
            continue
        if field.type != 'number':
            continue
        value = profile.parameters.get(field.key)
        # Historical llama.cpp profiles use e.g. 999 for all layers. Keep their launch contract.
        if field.key == 'gpu_layers' and legacy_gpu and type(value) is int and field.maximum is not None and value > field.maximum:
            continue
        validate_number(field, value)
    validate_generation(profile.generation, profile.backend, runtime, profile=True)
    if profile.backend == 'strata':
        from .strata_prefetch import validate_profile as validate_prefetch
        validate_prefetch(model, profile, runtime)
        from .strata_deepseek import is_deepseek
        if not is_deepseek(model):
            validate_strata_expert_cache(profile.parameters.get('expert_cache'))
    if profile.backend == 'custom':
        from .backends.custom import validate_custom_profile
        validate_custom_profile(model, profile, runtime)
    elif profile.backend == 'sglang':
        from .backends.sglang import validate
        validate(model, profile, runtime)
    elif profile.backend in {'ninfer', 'sparkinfer'}:
        from .backends.native_engines import validate
        validate(model, profile, runtime)
    from .strata_deepseek import is_deepseek
    if profile.backend not in {'sglang', 'ninfer', 'sparkinfer'} and not (profile.backend == 'strata' and is_deepseek(model)):
        from .mtp import validate_tokens
        validate_tokens(profile, runtime)
