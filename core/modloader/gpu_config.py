"""Native GPU selections shared by launch and resource previews."""
import math
import re


CUDA_ENV_BACKENDS = {'vllm', 'sglang', 'ninfer', 'sparkinfer'}


def _optional_text(value, label):
    if value is None:
        return None
    if not isinstance(value, str) or any(character in value for character in '\0\r\n'):
        raise ValueError(f'{label}必须是单行文本')
    return value if value.strip() else None


def cuda_environment(profile, runtime):
    """Return only the profile override; a blank value inherits the launch environment."""
    value = _optional_text(profile.parameters.get('cuda_visible_devices'), 'CUDA 可见设备')
    return {'CUDA_VISIBLE_DEVICES': value} if value is not None else {}


def validate_gpu_parameters(profile, runtime):
    values = profile.parameters
    if profile.backend in CUDA_ENV_BACKENDS:
        # CUDA owns the index, UUID, MIG and special-value syntax. Do not bind a
        # saved profile to this host's current inventory or normalize its order.
        cuda_environment(profile, runtime)
    elif profile.backend == 'llama_cpp':
        _optional_text(values.get('devices'), '卸载设备')
        _optional_text(values.get('split_mode'), '拆分模式')
        value = values.get('tensor_split')
        # The old launcher importer stored a single integer ratio as a number.
        if type(value) in (int, float):
            value = str(value)
        value = _optional_text(value, 'GPU 分配比例')
        if value is not None:
            try:
                parts = re.split(r'[,/]+', value)
                if parts[-1] == '':
                    parts.pop()  # std::sregex_token_iterator omits the final empty suffix.
                ratios = [float(part.strip()) for part in parts]
            except ValueError:
                raise ValueError('GPU 分配比例必须是逗号或斜线分隔的非负有限数字') from None
            if not all(math.isfinite(part) and part >= 0 for part in ratios):
                raise ValueError('GPU 分配比例必须是非负有限数字')
    elif profile.backend == 'strata':
        value = values.get('gpu')
        if value is None or isinstance(value, str) and not value.strip():
            return
        items = value if isinstance(value, (list, tuple)) else str(value).split(',')
        try:
            for item in items:
                text = str(item).strip()
                if text and int(text) < 0:
                    raise ValueError
        except (TypeError, ValueError):
            raise ValueError('Strata GPU 必须是非负整数索引，以逗号分隔') from None
