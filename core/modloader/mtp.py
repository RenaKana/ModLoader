"""MTP configuration and local-file inspection; never downloads or loads a model."""
from __future__ import annotations

from functools import lru_cache
import errno
import json
from pathlib import Path
import re
import struct

from .assets import gguf_metadata, GGUF_SHARD
from .models import ParameterField

# Strata 0.1.43/0.1.44: include/strata/kernels/verify_kernels.hpp kVerifyMaxT.
# A verify window includes the current token, so it holds at most 7 predictions.
STRATA_VERIFY_MAX = 8


def strata_spec_field(profile, runtime):
    """0.1.56 long lookup is separate from the MTP drafter's 7 predictions."""
    try:
        args, _ = inherited_args(profile, runtime)
    except OSError:
        args = profile.extra_args  # Resource repair must not block the command editor.
    long_lookup = ('--lookup-max-t' in runtime.metadata.get('supported_flags', [])
                   and runtime.metadata.get('engine_backend') != 'metal'
                   and int(argument(args, '--lookup-max-t') or 0) > 0
                   and int(argument(args, '--suffix-draft') or 3) > 0)
    return ParameterField(key='spec', label='Strata 验证窗口 --spec', type='number',
                          minimum=2, maximum=16 if long_lookup else STRATA_VERIFY_MAX, integer=True)


def token_field(backend):
    return ParameterField(key='mtp_tokens', label='最大预测 Token', type='number', minimum=0 if backend == 'llama_cpp' else 1,
                          maximum=STRATA_VERIFY_MAX - 1 if backend == 'strata' else 2**31 - 1 if backend == 'llama_cpp' else None,
                          integer=True, step=1)


def _number(value, label):
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{label}必须是数字') from None


def validate_tokens(profile, runtime):
    from .parameter_rules import validate_number
    field = token_field(profile.backend)
    args, _ = inherited_args(profile, runtime)
    config, inherited_enabled = {}, False
    if profile.backend == 'strata':
        inherited_enabled = bool(argument(args, '--mtp'))
    elif profile.backend == 'llama_cpp':
        inherited_enabled = 'draft-mtp' in (argument(args, '--spec-type') or '').split(',')
    elif profile.backend == 'vllm':
        config = json.loads(argument(args, '--speculative-config') or '{}')
        inherited_enabled = isinstance(config, dict) and config.get('method') == 'mtp'
    enabled = profile.parameters.get('mtp_enabled', inherited_enabled)
    if profile.backend == 'strata':
        from .strata_drafting import mode_of
        mode = mode_of(profile, runtime)
        if mode != 'legacy':
            from .strata_drafting import supports_dflash
            enabled = mode == 'mtp' or mode == 'auto' and not supports_dflash(runtime)
    value = profile.parameters.get('mtp_tokens') if enabled else None
    if value is not None:
        validate_number(field, value)
    if profile.backend == 'strata':
        spec = argument(profile.extra_args, '--spec') or profile.parameters.get('spec', argument(args, '--spec'))
        if spec is not None and value is None:
            validate_number(strata_spec_field(profile, runtime), _number(spec, '--spec'))
    elif profile.backend == 'llama_cpp':
        if not enabled:
            return
        maximum = value if value is not None else argument(args, '--spec-draft-n-max')
        if maximum is not None:
            validate_number(field, _number(maximum, field.label))
    elif profile.backend == 'vllm':
        if enabled and isinstance(config, dict) and config.get('method') == 'mtp' and value is None and config.get('num_speculative_tokens') is not None:
            validate_number(field, config['num_speculative_tokens'])


def argument(args, flag):
    value = None
    for i, item in enumerate(args):
        if item == flag and i + 1 < len(args):
            value = args[i + 1]
        elif item.startswith(flag + '='):
            value = item.split('=', 1)[1]
    return value


def without(args, *flags, switches=()):
    result, i = [], 0
    while i < len(args):
        flag = args[i].split('=', 1)[0]
        if flag in flags or flag in switches:
            i += 1 if '=' in args[i] or flag in switches else 2
        else:
            result.append(args[i])
            i += 1
    return result


def inherited_args(profile, runtime):
    args, cwd = [], Path(runtime.path).parent
    if profile.backend == 'strata':
        from .backends.adapters import _strata_config
        source = Path(profile.parameters.get('configuration') or Path(runtime.path) / 'strata.json')
        raw = json.loads(source.read_text('utf-8-sig')) if profile.parameters.get('configuration') else profile.parameters.get('strata_config', {})
        cfg = _strata_config(raw, source)
        args, cwd = [*cfg.get('args', []), *cfg.get('user_args', [])], Path(cfg['cwd'])
        if cfg.get('raw_token_continuation') is True and '--raw-token-continuation' not in args:
            args.append('--raw-token-continuation')
    return [*args, *profile.extra_args], cwd


def options(profile, runtime):
    args, cwd = inherited_args(profile, runtime)
    path, enabled, tokens = '', False, None
    if profile.backend == 'strata':
        path = argument(args, '--mtp') or ''
        enabled = bool(path)
        window = _number(argument(profile.extra_args, '--spec') or profile.parameters.get('spec', argument(args, '--spec') or 4), '--spec')
        cap = _number(argument(args, '--mtp-max-t') or 0, '--mtp-max-t')
        tokens = (min(window, cap) if cap > 0 else window) - 1
    elif profile.backend == 'llama_cpp':
        enabled = 'draft-mtp' in (argument(args, '--spec-type') or '').split(',')
        path = argument(args, '--spec-draft-model') or argument(args, '--model-draft') or argument(args, '-md') or ''
        raw = argument(args, '--spec-draft-n-max')
        tokens = _number(raw, '--spec-draft-n-max') if raw is not None else None
    elif profile.backend == 'vllm':
        config = json.loads(argument(args, '--speculative-config') or '{}')
        if not isinstance(config, dict):
            raise ValueError('--speculative-config 必须是 JSON 对象')
        enabled = config.get('method') == 'mtp'
        path = config.get('model', '') if enabled else ''
        tokens = config.get('num_speculative_tokens', 1) if enabled else 1
    path = profile.parameters.get('mtp_path', path) or ''
    if path and profile.backend != 'vllm':
        path = str((cwd / path).resolve())
    return {'enabled': profile.parameters.get('mtp_enabled', enabled), 'path': path,
            'tokens': profile.parameters.get('mtp_tokens', tokens), 'token_field': token_field(profile.backend).model_dump(),
            'resident_experts': profile.parameters.get('resident_experts', '--resident-experts' in args)}


def _stamp(path):
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=32)
def _gguf_info(path, size, modified):
    return gguf_metadata(Path(path), tensor_names=True)


def _has_mtp(names):
    return any(re.search(r'(?:^|\.)(?:mtp|nextn)(?:\.|_)|\.eh_proj\.', name) for name in names)


def weight_info(path):
    """Read GGUF descriptors or HF config/weight headers, not tensor payloads."""
    path = Path(path)
    # A missing GGUF is not an HF directory. Preserve the actual missing path.
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, '权重文件不存在', str(path))
    if path.is_file():
        info = _gguf_info(*_stamp(path))
        names = info['tensor_names']
        split = GGUF_SHARD.match(path.name)
        if split:
            names = []
            for index in range(1, int(split.group(3)) + 1):
                shard = path.with_name(f'{split.group(1)}-{index:05d}-of-{split.group(3)}{split.group(4)}')
                names.extend(_gguf_info(*_stamp(shard))['tensor_names'])
        arch = info.get('general.architecture', '')
        return {'has_mtp': _has_mtp(names), 'architecture': arch,
                'embedding': info.get(arch + '.embedding_length')}
    config = json.loads((path / 'config.json').read_text('utf-8'))
    text = config.get('text_config') or config
    names = []
    indexes = list(path.glob('*.safetensors.index.json')) + list(path.glob('pytorch_model*.index.json'))
    if indexes:
        for index in indexes:
            weights = json.loads(index.read_text('utf-8')).get('weight_map', {})
            for file in set(weights.values()):
                if not (path / file).is_file():
                    raise ValueError(f'MTP 检查发现缺少权重分片：{file}')
            names.extend(weights)
    else:
        for weights in path.glob('*.safetensors'):
            with weights.open('rb') as stream:
                length = struct.unpack('<Q', stream.read(8))[0]
                if length > weights.stat().st_size - 8:
                    raise ValueError('Safetensors 文件头超出文件边界')
                names.extend(json.loads(stream.read(length)))
    return {'has_mtp': _has_mtp(names) if names else None,
            'architecture': text.get('model_type') or config.get('model_type'),
            'embedding': text.get('hidden_size')}


@lru_cache(maxsize=16)
def _llama_mtp(path, size, modified):
    from .backends.adapters import _run_readonly
    help_text = _run_readonly([path, '--help'])
    return 'draft-mtp' in help_text if help_text else None


def runtime_support(runtime):
    if runtime.backend == 'custom':
        return False, '自定义引擎没有已验证的通用 MTP 参数映射；请使用引擎自身的启动模板'
    if runtime.backend == 'strata':
        return True, ''
    if runtime.backend == 'llama_cpp':
        types = runtime.metadata.get('speculative_types')
        if types is not None:
            return 'draft-mtp' in types, '此 llama.cpp 版本未提供 draft-mtp'
        try:
            supported = _llama_mtp(*_stamp(Path(runtime.path)))
        except OSError:
            supported = None
        return supported, '未确认此 llama.cpp 可执行程序支持 draft-mtp；请检查运行环境'
    if runtime.backend == 'vllm':
        flags = runtime.metadata.get('supported_flags')
        if flags:
            return '--speculative-config' in flags, '此 vLLM 版本未提供 --speculative-config'
        return None, '运行环境尚未记录推测解码能力，需在加载时由 vLLM 校验'
    return False, 'Transformers serve 未提供原生 MTP 参数入口'


def model_weight_info(model):
    from .platforms import location_of, remote_asset_operation
    location = location_of(model)
    if location.platform == 'wsl2':
        return remote_asset_operation(location.distribution, 'weight_info', {'path': model.path})
    return weight_info(model.path)


def check_external(path, model, backend, runtime=None):
    if runtime is not None and runtime.platform == 'wsl2':
        from .platforms import runtime_distribution, windows_to_wsl, remote_asset_operation
        target_path = path if path.startswith('/') else windows_to_wsl(path, runtime_distribution(runtime))
        target = model_weight_info(model)
        draft = remote_asset_operation(runtime_distribution(runtime), 'weight_info', {'path': target_path})
        if draft['has_mtp'] is False:
            raise ValueError('所选权重没有发现 MTP 张量；普通草稿模型不是 MTP 配套文件')
        for key in ('architecture', 'embedding'):
            if target[key] and draft[key] and target[key] != draft[key]:
                raise ValueError('MTP 配套权重的架构或隐藏维度与当前模型不匹配')
        return target_path
    path = Path(path)
    if backend == 'strata':
        for name in ('dense.bin', 'dense.txt', 'experts.bin'):
            item = path / name
            if not item.is_file() or not item.stat().st_size:
                raise ValueError(f'MTP 运行目录缺少 {name}；请选择已准备的 rt 目录，不是 GGUF 文件')
        embedding = model_weight_info(model)['embedding']
        rows = [line.split() for line in (path / 'dense.txt').read_text('utf-8').splitlines() if line.strip()]
        fc = next((row for row in rows if row[0] == 'fc_embedding.weight'), None)
        if not fc or len(fc) < 6:
            raise ValueError('MTP dense.txt 缺少 fc_embedding.weight 描述')
        if embedding and (int(fc[2]) != embedding or int(fc[3]) != embedding):
            raise ValueError('MTP 的隐藏维度与当前模型不匹配')
    else:
        if not path.exists():
            raise ValueError('MTP 配套文件不存在')
        target, draft = model_weight_info(model), weight_info(path)
        if draft['has_mtp'] is False:
            raise ValueError('所选权重没有发现 MTP 张量；普通草稿模型不是 MTP 配套文件')
        for key in ('architecture', 'embedding'):
            if target[key] and draft[key] and target[key] != draft[key]:
                raise ValueError('MTP 配套权重的架构或隐藏维度与当前模型不匹配')
    return str(path.resolve())


def inspect_mtp(model, profile, runtime, profiles=()):
    result = {'status': 'unverified', 'message': '', 'enabled': False, 'path': '',
              'path_kind': 'directory' if profile.backend in ('strata', 'vllm') else 'file',
              'needs_path': profile.backend == 'strata', 'candidates': [], 'resident_experts': False,
              'backend_supported': None}
    try:
        from .platforms import validate_model_runtime
        validate_model_runtime(model, runtime)
        result.update(options(profile, runtime))
        if profile.backend == 'strata':
            from .strata_drafting import mode_of, inspect_drafting
            if mode_of(profile, runtime) != 'legacy':
                drafting = inspect_drafting(model, profile, runtime)
                if drafting['effective_mode'] != 'mtp':
                    result.update(enabled=False, path='', status='disabled', backend_supported=True)
                    return result
                result['enabled'] = True
                if drafting['status'] == 'needs_preparation':
                    result.update(status='preparable', message=drafting['message'], backend_supported=True)
                    return result
        supported, reason = runtime_support(runtime)
        result['backend_supported'] = supported
        if supported is False:
            result.update(status='unsupported', message=reason)
            return result
        from .backends.adapters import get_adapters
        compatibility = get_adapters()[profile.backend].inspect(model, runtime)
        if compatibility.status == 'unsupported':
            result.update(status='unsupported', backend_supported=False, message='；'.join(compatibility.reasons))
            return result
        if result['path']:
            check_external(result['path'], model, profile.backend, runtime)
            result.update(status='external', message='')
        elif profile.backend == 'strata':
            # Reuse is suggested only through the same registered model, never by name.
            from .models import LoadProfile
            for item in profiles:
                if item['model_id'] != model.id or item['backend'] != 'strata' or item['id'] == profile.id:
                    continue
                try:
                    path = options(LoadProfile(**item), runtime)['path']
                    if path:
                        path = check_external(path, model, 'strata')
                        if not any(candidate['path'] == path for candidate in result['candidates']):
                            result['candidates'].append({'path': path, 'label': item['name']})
                except (OSError, ValueError, TypeError):
                    continue
            result.update(status='external' if result['candidates'] else 'missing',
                          message='可复用此模型已有的 MTP 文件' if result['candidates'] else '请选择本地 MTP 运行目录')
        else:
            info = model_weight_info(model)
            if info['has_mtp']:
                result.update(status='builtin', message='')
            elif info['has_mtp'] is False:
                result.update(status='missing', needs_path=True, message='当前权重不含 MTP，请选择配套权重')
            else:
                result.update(message='未识别到 MTP 层信息')
        if supported is None:
            result.update(status='unverified', message=result['message'] + ' ' + reason)
    except FileNotFoundError as error:
        result.update(status='unverified', message='文件不存在，无法检查 MTP', diagnostic=error.filename or str(error))
    except OSError as error:
        result.update(status='unverified', message='文件读取失败，无法检查 MTP', diagnostic=str(error))
    except (ValueError, TypeError, struct.error) as error:
        result.update(status='missing' if result['path'] else 'unverified', message=str(error))
    return result


def apply_mtp(args, model, profile, runtime):
    """Only explicit UI overrides change legacy argv; untouched profiles retain their contract."""
    enabled = profile.parameters.get('mtp_enabled')
    tokens = profile.parameters.get('mtp_tokens')
    validate_tokens(profile, runtime)
    if enabled is None and tokens is None:
        return args
    if enabled is None:
        enabled = options(profile, runtime)['enabled']
    if not isinstance(enabled, bool):
        raise ValueError('MTP 开关必须为布尔值')
    settings = options(profile, runtime)
    path = settings['path']
    if enabled:
        supported, reason = runtime_support(runtime)
        if supported is False:
            raise ValueError(reason)
        if path:
            path = check_external(path, model, profile.backend, runtime)
        elif profile.backend == 'strata':
            raise ValueError('启用 MTP 前请选择配套的本地运行目录')
        elif model_weight_info(model)['has_mtp'] is False:
            raise ValueError('模型权重不含 MTP 层，请选择配套文件或关闭 MTP')
    if profile.backend == 'strata':
        args = without(args, '--mtp')
        if enabled and tokens is not None:
            args = without(args, '--spec', '--mtp-max-t') + ['--spec', str(int(tokens) + 1)]
        return [*args, '--mtp', path] if enabled else args
    if profile.backend == 'llama_cpp':
        types = [value for value in (argument(args, '--spec-type') or '').split(',') if value and value not in ('draft-mtp', 'none')]
        if enabled and any(value.startswith('draft-') for value in types):
            raise ValueError('额外参数已指定另一种草稿方法，请先移除后再启用 MTP')
        args = without(args, '--spec-type')
        if not enabled and 'draft-mtp' in (argument(profile.extra_args, '--spec-type') or '').split(','):
            args = without(args, '--spec-draft-model', '--model-draft', '-md')
        if enabled:
            types.append('draft-mtp')
            args = without(args, '--spec-draft-model', '--model-draft', '-md')
            if path:
                args += ['--spec-draft-model', path]
            if tokens is not None:
                args = without(args, '--spec-draft-n-max') + ['--spec-draft-n-max', str(int(tokens))]
        if types or argument(profile.extra_args, '--spec-type') is not None:
            args += ['--spec-type', ','.join(types) or 'none']
        return args
    if profile.backend == 'vllm':
        config = json.loads(argument(args, '--speculative-config') or '{}')
        if enabled and config.get('method') not in (None, 'mtp'):
            raise ValueError('额外参数已指定另一种推测解码方法，请先移除后再启用 MTP')
        if not enabled and config.get('method') != 'mtp':
            return args
        args = without(args, '--speculative-config')
        if enabled:
            from .backends.adapters import windows_to_wsl
            config.update(method='mtp')
            config.setdefault('num_speculative_tokens', 1)
            if tokens is not None:
                config['num_speculative_tokens'] = int(tokens)
            if path:
                from .platforms import runtime_distribution
                config['model'] = path if path.startswith('/') else windows_to_wsl(path, runtime_distribution(runtime))
            else:
                config.pop('model', None)
            args += ['--speculative-config', json.dumps(config, ensure_ascii=False)]
        return args
    return args


def apply_resident_experts(args, profile):
    enabled = profile.parameters.get('resident_experts')
    if enabled is None:
        return args
    if not isinstance(enabled, bool):
        raise ValueError('专家常驻 RAM 开关必须为布尔值')
    if '--resident-cpu-experts' in args or argument(args, '--resident-budget-gib') is not None:
        raise ValueError('额外参数已设置其他专家驻留模式，请先移除再使用专家常驻 RAM 开关')
    args = without(args, switches=('--resident-experts',))
    return [*args, '--resident-experts'] if enabled else args
