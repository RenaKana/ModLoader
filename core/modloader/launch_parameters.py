"""Side-effect-free startup editor: argv parsing, backend-scoped rules and rendering.

Only editor-versioned profiles use this stricter contract. Legacy argv precedence
and launch behavior remain unchanged until a user explicitly edits the command.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path, PureWindowsPath
import re

from .models import ParameterField
from .parameter_rules import profile_fields, generation_fields, validate_number, validate_generation, inactive_parameters, INT32_MIN, INT32_MAX
from .backends.adapters import LLAMA_MAP, STRATA_MAP, RESERVED, SENSITIVE_LOG_FLAGS, LLAMA_EXTRA_VALUE_FLAGS, extra_option_indices, get_adapters, llama_effective_parameters

LLAMA_GENERATION_MAP = {'temperature': '--temp', 'top_p': '--top-p', 'top_k': '--top-k',
                        'repeat_penalty': '--repeat-penalty', 'seed': '--seed', 'max_tokens': '--n-predict'}
PARAMETER_ALIASES = {
    ('llama_cpp', 'flash_attention'): {'1': 'on', 'true': 'on', 'enabled': 'on', '0': 'off', 'false': 'off', 'disabled': 'off', '-1': 'auto'},
    ('transformers', 'dtype'): {'half': 'float16', 'float': 'float32'},
    ('strata', 'ssd_cache'): {'auto': '-1'},
}


def canonical_parameter(backend, key, value):
    return PARAMETER_ALIASES.get((backend, key), {}).get(value, value) if isinstance(value, str) else value


@dataclass
class Token:
    value: str
    start: int
    end: int


class InputError(ValueError):
    def __init__(self, message, start, end):
        super().__init__(message)
        self.start, self.end = start, end


def tokenize(text):
    """Parse literal argv only; never interpolate variables or execute a shell."""
    if text.lstrip().startswith(('[', '{')):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise InputError('JSON argv 格式错误：' + error.msg, error.pos, error.pos + 1) from None
        values = raw.get('args') if isinstance(raw, dict) else raw
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise InputError('JSON 必须是字符串 argv 数组或包含 args 数组的对象', 0, len(text))
        # Locate JSON strings in the selected array; decoding gives exact token spans.
        offset = 0
        if isinstance(raw, dict):
            match = re.search(r'"args"\s*:\s*\[', text)
            offset = match.end() - 1 if match else 0
        decoder, result, cursor = json.JSONDecoder(), [], text.find('[', offset) + 1
        for expected in values:
            while cursor < len(text) and text[cursor] in ' \r\n\t,':
                cursor += 1
            value, end = decoder.raw_decode(text, cursor)
            result.append(Token(value, cursor, end))
            cursor = end
        return result
    result, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        continuation = re.match(r'[\\^`]\r?\n', text[i:])
        if continuation:
            i += len(continuation[0])
            continue
        start, value, quote = i, [], None
        while i < len(text):
            c = text[i]
            if quote is None and c.isspace():
                break
            if quote is None and c in '|&;<>`':
                raise InputError('这里只接受启动参数，不执行 Shell 操作符', i, i + 1)
            if c == '\\' and quote != "'":
                end = i
                while end < len(text) and text[end] == '\\':
                    end += 1
                if end < len(text) and text[end] == '"':
                    value.extend('\\' * ((end - i) // 2))
                    if (end - i) % 2:
                        value.append('"')
                    else:
                        quote = None if quote == '"' else '"'
                    i = end + 1
                    continue
            if c in ('"', "'") and (quote is None or c == quote):
                quote = None if quote == c else c
            else:
                value.append(c)
            i += 1
        if quote:
            raise InputError('引号未闭合', start, i)
        result.append(Token(''.join(value), start, i))
    return result


def quote_arg(value):
    value = str(value)
    if not value or re.search(r'[\s"\'|&;<>`]', value):
        # Windows argv escaping also round-trips paths ending with a backslash.
        escaped = re.sub(r'(\\*)"', lambda m: '\\' * (len(m[1]) * 2 + 1) + '"', value)
        escaped = re.sub(r'\\+$', lambda m: m[0] * 2, escaped)
        return '"' + escaped + '"'
    return value


def format_argv(argv):
    lines = []
    for item in argv:
        if item.startswith('-') and not re.match(r'^-\d', item):
            lines.append(quote_arg(item))
        elif lines:
            lines[-1] += ' ' + quote_arg(item)
        else:
            lines.append(quote_arg(item))
    return '\n'.join(lines)


def mappings(model, profile, runtime):
    backend = profile.backend
    if backend == 'llama_cpp':
        mapping = {**LLAMA_MAP, 'mmproj': '--mmproj', 'mtp_tokens': '--spec-draft-n-max', 'mtp_path': '--spec-draft-model'}
    elif backend == 'strata':
        from .strata_deepseek import is_deepseek
        mapping = ({'context_length': '--max-context', 'vram_reserve_mib': '--reserve-mib', 'cpu_threads': '--cpu-threads',
                    'ssd_cache': '--ssd-cache', 'dspark_mode': '--dspark', 'expert_profile': '--expert-profile'} if is_deepseek(model)
                   else {**STRATA_MAP, 'resident_experts': '--resident-experts', 'mtp_path': '--mtp',
                         'dflash2_path': '--draft-model', 'expert_profile': '--expert-profile'})
    elif backend == 'vllm':
        mapping = {'context_length': '--max-model-len', 'gpu_memory_utilization': '--gpu-memory-utilization',
                   'tensor_parallel_size': '--tensor-parallel-size', 'pipeline_parallel_size': '--pipeline-parallel-size',
                   'dtype': '--dtype', 'enforce_eager': '--enforce-eager', 'max_num_seqs': '--max-num-seqs'}
    elif backend == 'transformers':
        mapping = {'device': '--device', 'dtype': '--dtype', 'attn_implementation': '--attn-implementation',
                   'compile': '--compile', 'continuous_batching': '--continuous-batching'}
    elif backend == 'sglang':
        from .backends.sglang import MAPPING, SPEC_FIELDS
        mapping = {**MAPPING, **{field.key: '--' + field.key.replace('_', '-') for field in SPEC_FIELDS}}
    elif backend == 'ninfer':
        from .backends.ninfer_capabilities import ninfer_capacity_mode
        capacity = 'context_capacity' if ninfer_capacity_mode(runtime) == 'context_bytes' else 'kv_capacity'
        mapping = {'context_length': '--max-context', 'max_concurrency': '--max-concurrency', capacity: '--' + capacity.replace('_', '-'),
                   'kv_dtype': '--kv-dtype', 'draft_tokens': '--draft-tokens'}
    elif backend == 'sparkinfer':
        mapping = {'context_length': '--ctx', 'draft_model': '--draft-model'}
    else:
        mapping = {}
        from .backends.custom import custom_config
        args = custom_config(runtime, check_paths=False)['args']
        for index, arg in enumerate(args):
            if arg == '{context_length}' and index and args[index - 1].startswith('-'):
                mapping['context_length'] = args[index - 1]
            elif '={context_length}' in arg:
                mapping['context_length'] = arg.split('=', 1)[0]
    return mapping


def rules(model, profile, runtime):
    fields = {field.key: field for field in profile_fields(get_adapters()[profile.backend], model, runtime)}
    if profile.backend == 'sglang':
        from .backends.sglang import PAGED_FIELDS
        # Hidden runtime controls still need their real grammar when repairing
        # an imported command; runtime capability checks own acceptance.
        for field in PAGED_FIELDS:
            fields.setdefault(field.key, field)
    if profile.backend == 'llama_cpp':
        fields.update({key: ParameterField(key=key, label=label, type='number', minimum=minimum,
                                           maximum=INT32_MAX, integer=True)
                       for key, label, minimum in [('threads_batch', '批处理线程', INT32_MIN), ('batch_size', '逻辑批大小', 1),
                                                   ('ubatch_size', '物理批大小', 0)]})
        from .mtp import token_field
        fields['mtp_tokens'] = token_field('llama_cpp')
    if profile.backend == 'strata':
        from .mtp import strata_spec_field
        try:
            spec_field = strata_spec_field(profile, runtime)
        except (OSError, ValueError):
            spec_field = ParameterField(key='spec', label='Strata 验证窗口 --spec', type='number', minimum=2, maximum=8, integer=True)
        fields.update(spec=spec_field,
                      resident_experts=ParameterField(key='resident_experts', label='专家常驻 RAM', type='boolean'),
                      pool_workers=ParameterField(key='pool_workers', label='工作线程', type='number', minimum=0, integer=True))
    mapping = mappings(model, profile, runtime)
    result = {flag: ('parameters', key, fields.get(key, ParameterField(key=key, label=flag)), None) for key, flag in mapping.items()}
    if profile.backend == 'llama_cpp':
        generation = {field.key: field for field in generation_fields('llama_cpp', runtime)}
        result.update({flag: ('generation', key, generation[key], None) for key, flag in LLAMA_GENERATION_MAP.items()})
        aliases = {'-c': '--ctx-size', '-ngl': '--n-gpu-layers', '--gpu-layers': '--n-gpu-layers', '-t': '--threads',
                   '-tb': '--threads-batch', '-b': '--batch-size', '-ub': '--ubatch-size', '-np': '--parallel',
                   '-ctk': '--cache-type-k', '-ctv': '--cache-type-v', '-fa': '--flash-attn', '-dev': '--device',
                   '-sm': '--split-mode', '-ts': '--tensor-split', '-mg': '--main-gpu', '-n': '--n-predict',
                   '--predict': '--n-predict', '-s': '--seed', '--temperature': '--temp'}
        result.update({alias: result[flag] for alias, flag in aliases.items()})
    elif profile.backend == 'transformers':
        for key in ('compile', 'continuous_batching'):
            result['--no-' + key.replace('_', '-')] = ('parameters', key, fields[key], False)
    elif profile.backend == 'vllm':
        result['-tp'] = result['--tensor-parallel-size']
        result['-pp'] = result['--pipeline-parallel-size']
    special = {'llama_cpp': ['--spec-type'], 'vllm': ['--speculative-config'],
               'sglang': ['--speculative-algorithm', '--speculative-algo'], 'ninfer': ['--spec']}
    for flag in special.get(profile.backend, []):
        result[flag] = ('acceleration', flag, ParameterField(key=flag, label=flag), None)
    return result


def converted(field, raw, backend=None):
    if backend == 'strata' and field.key == 'expert_cache':
        from .parameter_rules import validate_strata_expert_cache
        validate_strata_expert_cache(raw)
    raw = canonical_parameter(backend, field.key, raw)
    if field.key == 'context_length' and backend == 'vllm':
        if raw.lower() == 'auto':
            raw = '-1'
        else:
            match = re.fullmatch(r'(\d+(?:\.\d+)?)([kKmMgGtT])', raw.strip())
            if match:
                number, suffix = match.groups()
                power = 'kmgt'.index(suffix.lower()) + 1
                if suffix.isupper():
                    if '.' in number:
                        raise ValueError('二进制单位只接受整数')
                    raw = str(int(number) * 1024**power)
                else:
                    try:
                        raw = str(int(float(number) * 1000**power))
                    except (ValueError, OverflowError):
                        raise ValueError('上下文长度必须为有效整数') from None
            else:
                try:
                    raw = str(int(raw))
                except ValueError:
                    raise ValueError('上下文长度必须为整数、auto 或有效单位格式') from None
    if field.key == 'context_length' and backend == 'sglang':
        match = re.fullmatch(r'(\d+(?:\.\d+)?)(Ki|Mi|Gi|Ti|k|M|G|T)', raw)
        if match:
            number, suffix = match.groups()
            if suffix.endswith('i') and '.' in number:
                raise ValueError('IEC 单位只接受整数')
            multiplier = {'k': 10**3, 'M': 10**6, 'G': 10**9, 'T': 10**12,
                          'Ki': 2**10, 'Mi': 2**20, 'Gi': 2**30, 'Ti': 2**40}[suffix]
            raw = str(int(Decimal(number) * multiplier))
        else:
            try:
                raw = str(int(raw))
            except ValueError:
                raise ValueError('上下文长度必须为整数或有效 SI / IEC 单位格式') from None
    if field.type == 'number':
        if raw in field.special_values:
            return raw
        if field.integer:
            if not re.fullmatch(r'[+-]?\d+', raw):
                raise ValueError(f'{field.label}必须是整数')
            value = int(raw)
        else:
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f'{field.label}必须是有限数字') from None
        validate_number(field, value)
        return value
    if field.type == 'select':
        options = [option.get('value') if isinstance(option, dict) else option for option in field.options]
        if raw not in options:
            raise ValueError(f'{field.label}只接受：' + ' / '.join(map(str, options)))
    if not raw:
        raise ValueError(f'{field.label}缺少值')
    return raw


def _path_equal(left, right):
    return left.replace('\\', '/').rstrip('/').casefold() == str(right).replace('\\', '/').rstrip('/').casefold()


def _strata_args(profile, runtime):
    from .mtp import inherited_args
    try:
        args, _ = inherited_args(profile, runtime)
        return (args[:-len(profile.extra_args)] if profile.extra_args else args), None
    except (OSError, ValueError) as error:
        # Editing settings remains available when the inherited file needs repair.
        return [], '无法读取继承配置：' + str(error)


def _managed(model, profile, runtime):
    # Values are checked then discarded, never used to redirect the instance.
    model_paths = {model.path}
    if runtime.platform == 'wsl2' and re.match(r'^[A-Za-z]:[\\/]', model.path):
        model_paths.add('/mnt/' + model.path[0].lower() + '/' + model.path[3:].replace('\\', '/'))
    values = {flag: model_paths for flag in ('--model', '-m', '--model-path', '--native')}
    values.update({flag: {profile.alias} for flag in ('--alias', '--served-model-name', '--model-id', '--model-name')})
    values.update({flag: {'127.0.0.1', 'localhost'} for flag in ('--host', '--bind', '--bind-address', '--listen-address')})
    values['--engine'] = {'strata'} if profile.backend == 'strata' else set()
    for flag, key in [('--pack', 'pack'), ('--tokenizer', 'tokenizer')]:
        if profile.parameters.get(key):
            values[flag] = {str(profile.parameters[key])}
    if profile.backend == 'strata':
        from .mtp import argument
        args, _ = _strata_args(profile, runtime)
        for flag in ('--pack', '--tokenizer'):
            value = argument(args, flag)
            if value:
                values.setdefault(flag, set()).add(value)
    elif profile.backend == 'custom':
        from .backends.custom import custom_config, _binding_flags, _template
        args = custom_config(runtime, check_paths=False)['args']
        protected = _binding_flags(args)
        path = Path(model.path)
        replacements = dict(model_path=model.path, model_dir=str(path if path.is_dir() else path.parent), host='127.0.0.1',
                            port='{port}', alias=profile.alias, context_length=profile.parameters.get('context_length', ''))
        for index, item in enumerate(args):
            flag, separator, inline = item.partition('=')
            if flag in protected:
                raw = inline if separator else args[index + 1] if index + 1 < len(args) else ''
                values[flag] = {_template(raw, replacements)[0]}
    return values


def _model_binding_matches(raw, model, runtime, managed):
    if any(_path_equal(raw, path) for path in managed['--model']):
        return True
    if runtime.platform == 'wsl2':
        from .platforms import model_target_path
        target = model_target_path(model, runtime)
        for flag in ('--model', '-m', '--model-path', '--native'):
            managed[flag].add(target)
    return any(_path_equal(raw, path) for path in managed['--model'])


def _custom_prefix(tokens, model, profile, runtime):
    from .backends.custom import custom_config, _template
    config = custom_config(runtime, check_paths=False)
    path = Path(model.path)
    replacements = dict(model_path=model.path, model_dir=str(path if path.is_dir() else path.parent),
                        host='127.0.0.1', port='{port}', alias=profile.alias, context_length='{context_length}')
    retained, index = [], 1
    for template in config['args']:
        if index >= len(tokens):
            raise InputError('完整命令缺少自定义引擎模板参数', tokens[-1].start, tokens[-1].end)
        token = tokens[index]
        expected = _template(template, replacements)[0]
        if '{context_length}' in expected:
            if '=' not in expected and index > 1:
                retained.append(tokens[index - 1])
            retained.append(token)
        elif '{port}' in expected:
            pattern = re.escape(expected).replace(re.escape('{port}'), r'\d+')
            if not re.fullmatch(pattern, token.value):
                raise InputError('端口由管理器分配', token.start, token.end)
        elif not _path_equal(expected, token.value):
            raise InputError('完整命令不能更改自定义引擎的固定启动模板', token.start, token.end)
        index += 1
    return [*retained, *tokens[index:]]


def _acceleration(flag, raw, parameters, extra, unchecked, runtime):
    if runtime.backend == 'llama_cpp':
        types = raw.split(',')
        known = runtime.metadata.get('speculative_types')
        if known is not None and any(value not in [*known, 'none'] for value in types):
            raise ValueError('此运行环境未提供推测解码类型：' + raw)
        if 'draft-mtp' in types or raw == 'none':
            parameters['mtp_enabled'] = 'draft-mtp' in types
            types = [value for value in types if value not in ('draft-mtp', 'none')]
        if types:
            extra.extend([flag, ','.join(types)])
            unchecked.append(dict(flag=flag, message='模式参数未校验'))
    elif runtime.backend == 'vllm':
        try:
            config = json.loads(raw)
        except ValueError:
            raise ValueError('--speculative-config 必须是 JSON 对象') from None
        if not isinstance(config, dict):
            raise ValueError('--speculative-config 必须是 JSON 对象')
        if config.get('method') == 'mtp':
            from .mtp import token_field
            if 'num_speculative_tokens' in config:
                validate_number(token_field('vllm'), config['num_speculative_tokens'])
                parameters['mtp_tokens'] = config.pop('num_speculative_tokens')
            parameters['mtp_enabled'] = True
            parameters['mtp_path'] = config.pop('model', '')
            config.pop('method')
        if config:
            extra.extend([flag, json.dumps(config, ensure_ascii=False)])
            unchecked.append(dict(flag=flag, message='配置字段未校验'))
    else:
        allowed = ('NEXTN', 'none') if runtime.backend == 'sglang' else ('mtp', 'none')
        if raw not in allowed:
            raise ValueError(flag + ' 只接受：' + ' / '.join(allowed))
        parameters['acceleration_mode'] = 'none' if raw == 'none' else 'mtp'


def _diagnostic(text, start, end, flag, message, value_start=None, value_end=None):
    offset = lambda n: len(text[:n].encode('utf-16-le')) // 2
    result = dict(start=offset(start), end=offset(end), flag=flag, message=message)
    if value_start is not None:
        result.update(value_start=offset(value_start), value_end=offset(value_end))
    return result


def parse_parameters(model, profile, runtime, text):
    parameters, generation, extra, errors, unchecked = dict(profile.parameters), dict(profile.generation), [], [], []
    parsed_keys = set()
    spark_paths = None
    try:
        tokens = tokenize(text)
    except InputError as error:
        return dict(parameters=profile.parameters, generation=profile.generation, extra_args=profile.extra_args,
                    errors=[_diagnostic(text, error.start, error.end, '', str(error))], unchecked=[])
    rule_profile = profile
    if profile.backend == 'strata':
        rule_profile = profile.model_copy(update={'extra_args': [token.value for token in tokens]})
    rule, managed, i = rules(model, rule_profile, runtime), _managed(model, profile, runtime), 0
    # Accept only the selected executable (or its basename), never a shell command.
    if tokens and not tokens[0].value.startswith('-'):
        executable = tokens[0].value
        expected = [runtime.path]
        if profile.backend == 'strata':
            expected += [runtime.metadata.get('python', ''), str(Path(runtime.path) / 'engine/strata.exe'),
                         str(profile.parameters.get('strata_config', {}).get('exe', ''))]
            from .strata_deepseek import is_deepseek, engine_path
            try:
                if is_deepseek(model):
                    expected.append(str(engine_path(Path(runtime.path))))
                else:
                    from .strata_prefetch import select_engine
                    expected.append(select_engine(profile, runtime)[0])
            except (OSError, ValueError):
                pass  # Missing launch resources do not authorize a different executable.
        def basename(path): return PureWindowsPath(path.replace('/', '\\')).name.casefold()
        if any(path and (_path_equal(executable, path) or executable.casefold() == basename(path)) for path in expected):
            i = 1
            if profile.backend == 'custom':
                try:
                    tokens, i = _custom_prefix(tokens, model, profile, runtime), 0
                except InputError as error:
                    errors.append(_diagnostic(text, error.start, error.end, '', str(error)))
                    tokens = []
            elif profile.backend in ('vllm', 'transformers') and i < len(tokens) and tokens[i].value == 'serve':
                i += 1
                if i < len(tokens) and not tokens[i].value.startswith('-'):
                    if not _model_binding_matches(tokens[i].value, model, runtime, managed):
                        errors.append(_diagnostic(text, tokens[i].start, tokens[i].end, '', '模型由当前配置管理'))
                    i += 1
            elif profile.backend in ('strata', 'sglang') and i + 1 < len(tokens) and tokens[i].value == '-m':
                module = 'serve.server' if profile.backend == 'strata' else 'sglang.launch_server'
                if tokens[i + 1].value != module:
                    errors.append(_diagnostic(text, tokens[i].start, tokens[i + 1].end, '-m', '不能更改启动模块'))
                i += 2
            elif (profile.backend == 'strata' and i < len(tokens)
                  and PureWindowsPath(tokens[i].value.replace('/', '\\')).name == 'strata-serve.py'):
                # Generated child shim from a command preview; never executed
                # or retained as a user-selected executable.
                i += 1
            elif profile.backend == 'ninfer' and i < len(tokens) and not tokens[i].value.startswith('-'):
                expected_artifact = profile.parameters.get('_prepared_artifact') or model.path
                if not _path_equal(tokens[i].value, expected_artifact):
                    errors.append(_diagnostic(text, tokens[i].start, tokens[i].end, '', '模型 artifact 由准备流程管理'))
                i += 1
        else:
            errors.append(_diagnostic(text, tokens[0].start, tokens[0].end, '', '可执行程序由所选运行环境管理'))
            i = 1
    while i < len(tokens):
        token = tokens[i]
        flag, sep, inline = token.value.partition('=')
        entry = rule.get(flag)
        # These zero-argument switches are generated by the manager itself.
        if flag in ({'--enable-metrics'} if profile.backend == 'sglang' else {'--no-trust-remote-code'} if profile.backend == 'transformers' else set()):
            if sep:
                errors.append(_diagnostic(text, token.start, token.end, flag, '此参数为开关，不接受值'))
            i += 1
            continue
        is_managed = flag in managed or flag in RESERVED or flag in SENSITIVE_LOG_FLAGS or flag in {'--port', '--model-dir', '--tokenizer-path', '--pack'}
        if not entry and not is_managed:
            consumes_value = profile.backend == 'llama_cpp' and not sep and flag in LLAMA_EXTRA_VALUE_FLAGS
            group = tokens[i:i + 2] if consumes_value else [token]
            try:
                candidate = profile.model_copy(update={'extra_args': [item.value for item in group]})
                if profile.backend == 'custom':
                    from .backends.custom import custom_config, custom_extra
                    custom_extra(candidate, custom_config(runtime, check_paths=False))
                else:
                    from .backends.adapters import _extra
                    _extra(candidate, runtime)
            except ValueError as error:
                end = token.end
                if not sep and i + 1 < len(tokens) and (not tokens[i + 1].value.startswith('-') or re.match(r'^-\d|^-\.\d', tokens[i + 1].value)):
                    end = tokens[i + 1].end
                errors.append(_diagnostic(text, token.start, end, flag, str(error)))
                i += 1
                continue
            for position, item in enumerate(group):
                literal_value = profile.backend == 'llama_cpp' and flag in LLAMA_EXTRA_VALUE_FLAGS and (position or sep)
                forbidden = '\0' if literal_value else '\0\r\n'
                if any(c in item.value for c in forbidden):
                    errors.append(_diagnostic(text, item.start, item.end, flag, '参数不能包含 NUL 或换行'))
                extra.append(item.value)
            if flag.startswith('-') and not re.match(r'^-\d', flag) and not any(c.isspace() for c in flag):
                unchecked.append(dict(flag=flag, message='缺少校验规则'))
            i += len(group)
            continue
        switch = entry is not None and entry[2].type == 'boolean'
        end, raw, value_start, value_end = token.end, inline, token.start + len(flag) + 1, token.end
        if switch:
            if sep:
                errors.append(_diagnostic(text, token.start, end, flag, '此参数为开关，不接受值'))
            elif i + 1 < len(tokens) and not tokens[i + 1].value.startswith('-'):
                errors.append(_diagnostic(text, token.start, tokens[i + 1].end, flag, '此参数为开关，不接受值'))
                i += 1
            else:
                parameters[entry[1]] = entry[3] if entry[3] is not None else True
            i += 1
            continue
        if not sep:
            if i + 1 == len(tokens) or tokens[i + 1].value.startswith('-') and not re.match(r'^-\d|^-\.\d', tokens[i + 1].value):
                errors.append(_diagnostic(text, token.start, end, flag, flag + ' 缺少值'))
                i += 1
                continue
            raw, end = tokens[i + 1].value, tokens[i + 1].end
            value_start, value_end = tokens[i + 1].start, end
            i += 1
        try:
            if profile.backend in ('vllm', 'sglang') and flag in ('--model', '-m', '--model-path', '--native'):
                _model_binding_matches(raw, model, runtime, managed)
            if (profile.backend == 'sparkinfer' and flag in ('--model', '-m', '--model-path', '--native', '--tokenizer')
                    and not any(_path_equal(raw, allowed) for allowed in managed.get(flag, set()))):
                # Resolve only pasted launch bindings; ordinary parameter edits remain
                # available without model resources or a running WSL distribution.
                from .backends.native_engines import spark_launch_paths
                if spark_paths is None:
                    spark_paths = spark_launch_paths(model, profile, runtime)
                managed.setdefault(flag, set()).add(spark_paths[1 if flag == '--tokenizer' else 0])
            if entry:
                section, key, field, _ = entry
                if section == 'acceleration':
                    _acceleration(flag, raw, parameters, extra, unchecked, runtime)
                else:
                    value = converted(field, raw, profile.backend)
                    if section == 'generation':
                        validate_generation({key: value}, profile.backend, runtime)
                    if section == 'parameters':
                        from .gpu_config import validate_gpu_parameters
                        validate_gpu_parameters(profile.model_copy(update={'parameters': {key: value}}), runtime)
                        parsed_keys.add(key)
                    (parameters if section == 'parameters' else generation)[key] = value
                    if profile.backend == 'custom' and key == 'context_length':
                        unchecked.append(dict(flag=flag, message='整数格式已校验，取值范围未校验'))
                    if field.type == 'text' and key not in ('devices', 'tensor_split', 'gpu'):
                        message = ('设备可用性未校验' if key == 'device' else
                                   '可用模式未校验' if key in ('dtype', 'attn_implementation') else
                                   '资源路径未校验' if key in ('mtp_path', 'expert_profile') else
                                   '取值格式未校验')
                        unchecked.append(dict(flag=flag, message=message))
                    if profile.backend == 'strata':
                        if key == 'mtp_path':
                            parameters['mtp_enabled'] = True
                        elif key == 'dflash2_path':
                            parameters['strata_drafting'] = 'dflash2'
            elif flag == '--port' or managed.get(flag) == {'{port}'}:
                if not raw.isdigit() or not 1 <= int(raw) <= 65535:
                    raise ValueError('端口必须是 1–65535 的整数；实际端口由管理器分配')
            elif flag == '--config' and profile.backend == 'strata' and PureWindowsPath(raw.replace('/', '\\')).name == 'strata.json':
                pass  # Preview wrapper only. Never open or save this pasted path.
            elif flag in managed and any(_path_equal(raw, allowed) for allowed in managed[flag]):
                pass
            else:
                raise ValueError(flag + ' 由本地实例管理器保留，不能在这里覆盖')
        except ValueError as error:
            errors.append(_diagnostic(text, token.start, end, flag, str(error), value_start, value_end))
        i += 1
    if profile.backend == 'strata' and 'spec' in parsed_keys and parameters.get('mtp_enabled'):
        if parameters.get('strata_drafting') not in ('auto', 'dflash2', 'off') and parameters['spec'] <= 8:
            parameters['mtp_tokens'] = parameters['spec'] - 1
    candidate = profile.model_copy(update=dict(parameters=parameters, generation=generation, extra_args=extra))
    # Shared GPU grammar checks cross-field values without filesystem preflight.
    if not errors:
        try:
            from .gpu_config import validate_gpu_parameters
            validate_gpu_parameters(candidate, runtime)
        except ValueError as error:
            errors.append(_diagnostic(text, 0, len(text), '', str(error)))
    if errors:
        return dict(parameters=dict(profile.parameters), generation=dict(profile.generation), extra_args=list(profile.extra_args), errors=errors, unchecked=unchecked)
    return dict(parameters=parameters, generation=generation, extra_args=extra, errors=[], unchecked=unchecked)


def parameter_argv(model, profile, runtime):
    if profile.backend == 'sglang':
        from .backends.sglang import launch_arguments
        return launch_arguments(model, profile, runtime)
    mapping = mappings(model, profile, runtime)
    values = profile.parameters
    if profile.backend == 'llama_cpp':
        values = llama_effective_parameters(profile)
    elif profile.backend == 'ninfer':
        from .backends.ninfer_capabilities import ninfer_capacity_mode
        capacity = 'context_capacity' if ninfer_capacity_mode(runtime) == 'context_bytes' else 'kv_capacity'
        values = {'max_concurrency': 1, capacity: 'auto', 'kv_dtype': 'bf16', **values}
    args = []
    if profile.backend == 'strata':
        from .mtp import without
        args = _strata_args(profile, runtime)[0]
    for key, flag in mapping.items():
        value = values.get(key)
        if profile.backend == 'strata' and key == 'ssd_cache' and value == 'auto':
            value = '-1'
        if profile.backend == 'strata' and profile.parameter_editor_version == 1 and key in values:
            args = without(args, flag, switches=(flag,) if key == 'resident_experts' else ())
        if value is None or value == '':
            continue
        if key in ('mtp_path', 'mtp_tokens') and values.get('mtp_enabled') is False:
            continue
        if profile.backend in ('sglang', 'ninfer') and key.startswith(('speculative_', 'draft_tokens')) and values.get('acceleration_mode', 'none') == 'none':
            continue
        if profile.backend == 'strata':
            args = without(args, flag, switches=(flag,) if isinstance(value, bool) else ())
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif profile.backend == 'transformers':
                args.append('--no-' + key.replace('_', '-'))
        else:
            args += [flag, str(value)]
    enabled = values.get('mtp_enabled')
    if profile.backend == 'llama_cpp' and enabled is not None:
        args += ['--spec-type', 'draft-mtp' if enabled else 'none']
    elif profile.backend == 'vllm' and enabled:
        config = dict(method='mtp', num_speculative_tokens=values.get('mtp_tokens', 1))
        if values.get('mtp_path'):
            config['model'] = values['mtp_path']
        args += ['--speculative-config', json.dumps(config, ensure_ascii=False)]
    elif profile.backend == 'strata':
        from .mtp import argument, without
        from .strata_deepseek import is_deepseek
        if not is_deepseek(model):
            from .strata_drafting import mode_of
            try:
                drafting_mode = mode_of(profile, runtime)
            except (OSError, ValueError):
                drafting_mode = profile.parameters.get('strata_drafting') or 'legacy'
            if enabled is False:
                args = without(args, '--mtp')
            if drafting_mode == 'legacy' and enabled and values.get('mtp_tokens') is not None:
                args = without(args, '--spec') + ['--spec', str(values['mtp_tokens'] + 1)]
            for flag, default in (('--spec','4'),('--prefill','auto'),('--expert-cache','auto')):
                if argument(args,flag) is None:
                    args += [flag,default]
            from .strata_drafting import mode_of, inspect_drafting, apply_drafting
            try:
                state = inspect_drafting(model, profile, runtime) if mode_of(profile, runtime) != 'legacy' else None
            except (OSError, ValueError):
                # The command editor stays usable while a source/resource path
                # is being repaired; preparation owns filesystem diagnostics.
                state = None
            if state and state['status'] == 'ready':
                return apply_drafting([*args, *profile.extra_args], model, profile, runtime, state)
    elif profile.backend in ('sglang', 'ninfer') and values.get('acceleration_mode', 'none') != 'none':
        args += ['--speculative-algorithm', 'NEXTN'] if profile.backend == 'sglang' else ['--spec', 'mtp']
    if profile.backend == 'llama_cpp':
        for key, flag in LLAMA_GENERATION_MAP.items():
            if profile.generation.get(key) is not None:
                args += [flag, str(profile.generation[key])]
    return [*args, *profile.extra_args]


def _displayed_fields(text, model, profile, runtime):
    entries = rules(model, profile, runtime)
    result = set()
    tokens = tokenize(text)
    for index in extra_option_indices([token.value for token in tokens], runtime):
        token = tokens[index]
        entry = entries.get(token.value.split('=', 1)[0])
        if not entry:
            continue
        section, key, _, _ = entry
        if section == 'acceleration':
            keys = ('mtp_enabled', 'mtp_tokens', 'mtp_path') if profile.backend == 'vllm' else ('mtp_enabled',) if profile.backend == 'llama_cpp' else ('acceleration_mode',)
            result.update(('parameters', key) for key in keys)
        else:
            result.add((section, key))
    return result


def edit_parameters(model, profile, runtime, *, text=None, previous_text=None):
    if runtime.backend != profile.backend:
        raise ValueError('所选运行环境与后端不匹配')
    if text is not None and not isinstance(text, str):
        raise ValueError('启动参数必须是文本')
    original = profile
    profile = profile.model_copy(update={'parameters': {key: canonical_parameter(profile.backend, key, value)
                                                       for key, value in profile.parameters.items()}})
    if text is not None and previous_text is not None:
        if not isinstance(previous_text, str):
            raise ValueError('上次启动参数必须是文本')
        # A fragment paste omits previous_text; only an explicit full-text edit
        # can clear fields, and only those actually present in the earlier view.
        try:
            removed = _displayed_fields(previous_text, model, profile, runtime) - _displayed_fields(text, model, profile, runtime)
        except InputError:
            removed = set()  # The parser below returns the new text's diagnostic.
        parameters, generation = dict(profile.parameters), dict(profile.generation)
        for section, key in removed:
            if section == 'generation':
                generation.pop(key, None)
            else:
                parameters[key] = False if key == 'mtp_enabled' else 'none' if key == 'acceleration_mode' else None
        profile = profile.model_copy(update=dict(parameters=parameters, generation=generation, parameter_editor_version=1))
    if text is None:
        result = dict(parameters=dict(profile.parameters), generation=dict(profile.generation), extra_args=list(profile.extra_args), errors=[], unchecked=[])
        rendered = format_argv(parameter_argv(model, profile, runtime))
        inspected = parse_parameters(model, profile, runtime, rendered)
        result.update(errors=inspected['errors'], unchecked=inspected['unchecked'])
    else:
        result = parse_parameters(model, profile, runtime, text)
    if result['errors']:
        result.update(parameters=dict(original.parameters), generation=dict(original.generation), extra_args=list(original.extra_args))
    effective = profile.model_copy(update={**{key: result[key] for key in ('parameters', 'generation', 'extra_args')},
                                          **({'parameter_editor_version': 1} if text is not None else {})})
    result['text'] = text if result['errors'] and text is not None else format_argv(parameter_argv(model, effective, runtime))
    result['parameter_editor_version'] = 1
    if profile.backend == 'strata':
        _, problem = _strata_args(profile, runtime)
        if problem:
            result['preview_error'] = problem
    return result


def validate_editor_profile(model, profile, runtime):
    if profile.parameter_editor_version != 1:
        return
    if profile.backend == 'sglang':
        from .backends.sglang import normalized_paged_parameters
        normalized = normalized_paged_parameters(profile)
        profile = profile.model_copy(update={'parameters': normalized['normalized_parameters'],
                                             'extra_args': normalized['normalized_extra_args']})
    mapped = rules(model, profile, runtime)
    inactive = inactive_parameters(profile)
    checked = set()
    for section, key, field, _ in mapped.values():
        if section not in ('parameters', 'generation') or (section, key) in checked:
            continue
        checked.add((section, key))
        if section == 'parameters' and key in inactive:
            continue
        value = getattr(profile, section).get(key)
        if value is None or value == '':
            continue
        if field.type == 'number':
            validate_number(field, value)
        elif field.type == 'boolean' and not isinstance(value, bool):
            raise ValueError(field.label + '必须是布尔值')
        elif field.type == 'select':
            allowed = [option.get('value') if isinstance(option, dict) else option for option in field.options]
            if canonical_parameter(profile.backend, key, value) not in allowed:
                raise ValueError(field.label + '只接受：' + ' / '.join(map(str, allowed)))
    result = edit_parameters(model, profile, runtime)
    if result['errors']:
        raise ValueError('；'.join(error['message'] for error in result['errors']))
    for index in extra_option_indices(profile.extra_args, runtime):
        value = profile.extra_args[index]
        entry = mapped.get(value.split('=', 1)[0])
        if entry and entry[0] in ('parameters', 'generation'):
            raise ValueError('已映射的启动参数必须先同步到配置，不能保留在额外 argv 中：' + value.split('=', 1)[0])
