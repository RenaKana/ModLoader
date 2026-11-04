"""Version-aware NInfer context contracts; legacy Token budgets stay separate."""
from __future__ import annotations

import re

from .source_builds import resolve_pin

UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1


def uses_new_ninfer_source(runtime):
    return (runtime is not None and runtime.backend == 'ninfer'
            and runtime.metadata.get('version_basis') == 'verified_source_commit'
            and runtime.metadata.get('source_commit') == resolve_pin('ninfer')['commit'])


def ninfer_capacity_mode(runtime):
    """Use verified source identity or the executable's probed flag contract."""
    if runtime is not None and runtime.backend == 'ninfer':
        if uses_new_ninfer_source(runtime):
            return 'context_bytes'
        if '--context-capacity' in (runtime.metadata.get('supported_flags') or []):
            return 'context_bytes'
    return 'kv_tokens'


def context_capacity_bytes(value, *, allow_zero=False):
    """Match a754fcf schema.cpp's exact B/KiB/MiB/GiB quantity grammar."""
    message = '上下文共享容量必须为' + ('非负' if allow_zero else ' auto 或正') + '整数字节数，可使用 B、KiB、MiB、GiB。'
    if isinstance(value, bool):
        raise ValueError(message)
    match = re.fullmatch(r'([0-9]+)(?:\.([0-9]+))? *(B|KiB|MiB|GiB)?', str(value))
    if match:
        whole, fraction, unit = match.groups()
        multiplier = {None: 1, 'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3}[unit]
        fraction = (fraction or '').rstrip('0')
        if len(fraction) <= 30:
            denominator = 10**len(fraction)
            numerator = int(fraction or '0') * multiplier
            result = int(whole) * multiplier + numerator // denominator
            if numerator % denominator == 0 and (0 if allow_zero else 1) <= result <= UINT64_MAX:
                return result
    raise ValueError(message)


def capacity_migration_message(profile, runtime):
    """Return the actionable mismatch without editing the saved profile."""
    shared = ninfer_capacity_mode(runtime) == 'context_bytes'
    legacy = profile.parameters.get('kv_capacity')
    current = profile.parameters.get('context_capacity')
    if shared and legacy not in (None, '', 'auto'):
        return ('此配置保留了旧版 kv_capacity（Token）。请清除旧值，再设置新版 context_capacity'
                '（KV / State 共享字节容量，或 auto）。')
    if not shared and current not in (None, '', 'auto'):
        return '此 NInfer 运行环境不支持 context_capacity；请选择新版运行环境，或清除共享字节容量并设置旧版 KV Token 容量。'
    return ''


def validate_capacity_parameters(profile, runtime):
    message = capacity_migration_message(profile, runtime)
    if message:
        raise ValueError(message)
    if ninfer_capacity_mode(runtime) != 'context_bytes':
        return
    value = profile.parameters.get('context_capacity')
    if value not in (None, '', 'auto'):
        context_capacity_bytes(value)
    for index, argument in enumerate(profile.extra_args):
        flag, separator, value = argument.partition('=')
        if flag == '--kv-capacity':
            raise ValueError('新版 NInfer 已移除 --kv-capacity（Token）；请改用 --context-capacity 的共享字节容量，或 auto。')
        if flag == '--context-capacity':
            if not separator:
                if index + 1 >= len(profile.extra_args):
                    raise ValueError('--context-capacity 缺少上下文共享容量值。')
                value = profile.extra_args[index + 1]
            if value != 'auto':
                context_capacity_bytes(value)
