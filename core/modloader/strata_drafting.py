"""Strata drafter selection, independent resources, and an isolated serve bridge."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import struct
import sys

from .mtp import argument, without

MODES = ('auto', 'mtp', 'dflash2', 'off')
PACK_NAMES = ('qwen3.8-flash-next-dflash2.q4_0-selq4.pack',
              'qwen3.8-flash-next-dflash2.q4_0.pack', 'qwen3.8-flash-next-dflash2.q8.pack')
DRAFT_FLAGS = ('--draft', '--draft-model', '--draft-conf', '--draft-fallback', '--draft-mtp-spec',
               '--draft-cap', '--draft-rule', '--draft-mtp-min-p')
MTP_FLAGS = ('--mtp', '--mtp-q4', '--mtp-hnorm', '--mtp-window', '--mtp-draft-vocab', '--spec-min-p')


def merge_args(args, overrides):
    """Strata long-option groups; an unknown switch must not eat the next flag."""
    def groups(values):
        result, index = [], 0
        while index < len(values):
            group = [values[index]]
            if '=' not in values[index] and index + 1 < len(values) and not values[index + 1].startswith('--'):
                group.append(values[index + 1])
                index += 1
            result.append(group)
            index += 1
        return result
    result = groups(args)
    for group in groups(overrides):
        flag = group[0].split('=', 1)[0]
        result = [item for item in result if item[0].split('=', 1)[0] != flag]
        result.append(group)
    return [value for group in result for value in group]


def explicit_arguments(args, supplied_flags):
    """Project user-owned options from the final, normalized launch arguments.

    Keep provenance, not an earlier copy of values: removed options must not be
    replayed by Control, and resolved paths must have one meaning in both lists.
    """
    result, keep = [], False
    for arg in args:
        if arg.startswith('--'):
            keep = arg.split('=', 1)[0] in supplied_flags
        if keep:
            result.append(arg)
    return result


def configuration(profile, runtime):
    from .backends.adapters import _strata_config
    source = Path(profile.parameters.get('configuration') or Path(runtime.path) / 'strata.json')
    raw = json.loads(source.read_text('utf-8-sig')) if profile.parameters.get('configuration') else profile.parameters.get('strata_config', {})
    return _strata_config(raw, source)


def mode_of(profile, runtime):
    cfg = configuration(profile, runtime)
    mode = profile.parameters.get('strata_drafting') or cfg.get('drafting') or 'legacy'
    if mode not in (*MODES, 'legacy'):
        raise ValueError('Strata drafting 必须是 auto、mtp、dflash2 或 off')
    return mode


def _module_functions(path):
    if not path.is_file():
        return set()
    return {node.name for node in ast.parse(path.read_text('utf-8-sig')).body if isinstance(node, ast.FunctionDef)}


def supports_policy(runtime):
    """The bridge needs the serving-drafter and start-fit APIs, not their files."""
    root = Path(runtime.path) / 'strata_control'
    return ({'engine_flags', 'conf_for', 'slots_of', 'strip_mtp_flags'} <= _module_functions(root / 'drafting.py')
            and 'fit_start' in _module_functions(root / 'vram_fit.py'))


def supports_runtime_plan(runtime):
    return bool(runtime.metadata.get('python')) and 'engine_args' in _module_functions(Path(runtime.path) / 'serve/server.py')


def supports_dflash(runtime):
    root = Path(runtime.path)
    flags = runtime.metadata.get('supported_flags', [])
    if not supports_policy(runtime):
        return False
    if flags:
        return {'--draft', '--draft-fallback', '--mtp-draft-vocab'} <= set(flags)
    from .strata_prefetch import engine_identity
    identity = engine_identity(root)
    version = tuple(int(n) for n in re.findall(r'\d+', identity['version'])[:3])
    return version >= (0, 1, 56) and identity['backend'] in ('cuda', 'hip', 'metal', 'sycl')


def supported_model(model):
    # Use asset identity, never the editable API alias. These are upstream's
    # supported base/Gyro/Unsloth families, not Swift, Coder or NVFP4 fine-tunes.
    name = ' '.join((model.name, Path(model.path).name, str(model.metadata.get('general.name', '')))).lower()
    return bool(re.search(r'qwen3[._-]8[-_ .]*flash[-_ .]*next', name)) and not any(x in name for x in ('swift', 'coder', 'nvfp4'))


def pack_description(path):
    """Read DF2P descriptors, not tensor payloads; reject truncated resources."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'DF2P':
            raise ValueError('DFlash2 资源必须是 DF2P .pack 文件，不是 MTP rt 目录')
        version, length = struct.unpack('<II', header[4:])
        if version != 1 or length > size - 12:
            raise ValueError('DFlash2 pack 版本或索引长度无效')
        rows = stream.read(length).decode('utf-8').splitlines()
    tensors, config = {}, {}
    for line in rows:
        row = line.split()
        if row[:1] == ['cfg'] and len(row) >= 3:
            config[row[1]] = ' '.join(row[2:])
        elif row[:1] == ['t']:
            if len(row) != 7:
                raise ValueError('DFlash2 pack tensor 索引无效')
            offset, count = int(row[5]), int(row[6])
            if offset < 0 or count <= 0 or 12 + length + offset + count > size:
                raise ValueError('DFlash2 pack 数据被截断')
            tensors[row[1]] = {'dtype': row[2], 'bytes': count}
    if not tensors or 'fc.weight' not in tensors or 'end' not in [row.strip() for row in rows]:
        raise ValueError('DFlash2 pack 缺少权重索引')
    return {'path': str(path.resolve()), 'tensors': tensors, 'config': config,
            'selector_q4': tensors.get('candidate_selector.predecessor_codebook', {}).get('dtype') == 'q4_0'}


def inspect_drafting(model, profile, runtime, resource_root=None):
    from .mtp import options, check_external
    cfg = configuration(profile, runtime)
    args = merge_args(merge_args(cfg.get('args', []), cfg.get('user_args', [])), profile.extra_args)
    mode = mode_of(profile, runtime)
    state = dict(mode=mode, effective_mode=mode, status='ready', path='', vocab='', message='', download_bytes=0,
                 supported=supports_dflash(runtime))
    if mode == 'legacy':
        enabled = options(profile, runtime)['enabled']
        if 'mtp_enabled' not in profile.parameters and cfg.get('mtp') is False:
            enabled = False
        state['effective_mode'] = 'mtp' if enabled else 'off'
        return state
    if mode == 'off':
        return state
    effective = mode
    capable = supports_dflash(runtime) and supported_model(model)
    blocked = (int(argument(args, '--pipeline-windows') or 0) > 0
               or any(int(argument(args, flag) or 0) > 1 for flag in ('--batch', '--slots', '--parallel'))
               or type(cfg.get('parallel')) is int and cfg['parallel'] > 1
               or '--batch-mtp' in args)
    if mode == 'auto':
        effective = 'dflash2' if capable and not blocked else 'mtp'
    elif mode == 'dflash2' and (not state['supported'] or blocked):
        raise ValueError('DFlash2 需要支持该功能的 Strata 引擎和单并发')
    if effective == 'dflash2':
        architecture = model.architecture or model.metadata.get('general.architecture')
        embedding = model.metadata.get('qwen4exp.embedding_length')
        if architecture and architecture != 'qwen4exp' or embedding is not None and int(embedding) != 2560:
            raise ValueError('此 DFlash2 引擎需要 qwen4exp 架构、2560 隐藏维度')
    state['effective_mode'] = effective
    if effective == 'mtp':
        path = options(profile, runtime)['path']
        if not path and resource_root:
            candidate = Path(resource_root) / 'mtp/rt'
            if candidate.is_dir():
                path = str(candidate)
        if path:
            state['path'] = check_external(path, model, 'strata', runtime)
        else:
            state.update(status='needs_preparation', message='首次加载需准备 MTP 运行资源', download_bytes=5_200_000_000)
            if not supported_model(model):
                raise ValueError('请选择此模型对应的本地 MTP 运行目录')
        return state
    path = profile.parameters.get('dflash2_path') or argument(args, '--draft-model')
    if path:
        path = str((Path(cfg['cwd']) / path).resolve())
    elif resource_root:
        path = next((str(Path(resource_root) / 'dflash2' / name) for name in PACK_NAMES
                     if (Path(resource_root) / 'dflash2' / name).is_file()), '')
    vocab = argument(args, '--mtp-draft-vocab')
    candidates = ([Path(cfg['cwd']) / vocab] if vocab else [])
    if path:
        description = pack_description(path)
        state.update(path=description['path'], selector_q4=description['selector_q4'])
        candidates.append(Path(path).parent / 'draft_vocab.bin')
    candidates.append(Path(runtime.path) / 'data/draft_vocab.bin')
    state['vocab'] = next((str(p.resolve()) for p in candidates if p.is_file() and p.stat().st_size), '')
    if vocab and not candidates[0].is_file():
        raise ValueError(f'DFlash2 词表不存在：{candidates[0]}')
    if not path:
        state.update(status='needs_preparation', message='首次加载需下载并准备 DFlash2', download_bytes=1_206_141_244)
    elif not state['vocab']:
        raise ValueError('DFlash2 缺少 draft_vocab.bin，请选择词表或补全运行环境 data 目录')
    return state


def policy_config(model, profile, runtime, state):
    """Only resolved resource identities cross into the runtime policy process."""
    from .mtp import options, check_external, inherited_args
    mtp_args = []
    mtp_path = (state['path'] or options(profile, runtime)['path']) if state['effective_mode'] == 'mtp' else ''
    if state['mode'] == 'auto' and state['effective_mode'] == 'dflash2':
        candidate = options(profile, runtime)['path']
        if candidate:
            try:
                mtp_path = check_external(candidate, model, 'strata', runtime)
            except (OSError, ValueError):
                pass  # An unused/incomplete MTP is not a prerequisite for DFlash2.
        if mtp_path:
            # Dormant MTP tuning belongs only to its readiness retry, not to
            # user_args, which Control merges into the initial DFlash2 start.
            inherited, cwd = inherited_args(profile, runtime)
            for flag in MTP_FLAGS:
                value = argument(inherited, flag)
                if flag != '--mtp' and value is not None:
                    if flag == '--mtp-draft-vocab':
                        value = str((cwd / value).resolve())
                    mtp_args += [flag, value]
    return dict(model_name=' '.join((model.name, Path(model.path).name,
                                     str(model.metadata.get('general.name', '')))),
                initial=state['effective_mode'], mtp_path=mtp_path, mtp_args=mtp_args,
                path=state['path'] if state['effective_mode'] == 'dflash2' else '', vocab=state['vocab'])


def apply_drafting(args, model, profile, runtime, state=None):
    """Resolve exactly one drafter; old profiles retain their MTP switch semantics."""
    from .mtp import apply_mtp
    state = state or inspect_drafting(model, profile, runtime)
    if state['mode'] == 'legacy':
        if state['effective_mode'] == 'off' and configuration(profile, runtime).get('mtp') is False:
            args = without(args, *MTP_FLAGS, *DRAFT_FLAGS, switches=('--batch-mtp',))
        return apply_mtp(args, model, profile, runtime)
    if state['status'] != 'ready':
        raise ValueError(state['message'])
    effective = state['effective_mode']
    confidence = argument(args, '--draft-conf')
    original = list(args)
    args = without(args, *DRAFT_FLAGS)
    if effective == 'off':
        return without(args, *MTP_FLAGS, switches=('--batch-mtp',))
    if effective == 'mtp':
        tokens = profile.parameters.get('mtp_tokens')
        if tokens is not None:
            from .mtp import token_field
            from .parameter_rules import validate_number
            validate_number(token_field('strata'), tokens)
            args = without(args, '--spec', '--mtp-max-t') + ['--spec', str(int(tokens) + 1)]
        return without(args, '--mtp') + ['--mtp', state['path']]
    # Only defaults are changed: extra/user args and the structured spec win.
    cfg = configuration(profile, runtime)
    explicit = [*cfg.get('args', []), *cfg.get('user_args', [])]
    spec = argument(profile.extra_args, '--spec') or profile.parameters.get('spec') or argument(explicit, '--spec')
    args = without(args, '--spec', *MTP_FLAGS, switches=('--batch-mtp',))
    # Native drafter tuning remains the user's; only mutually exclusive resource
    # selection/fallback is owned by the one-drafter policy.
    tuning = [flag for flag in DRAFT_FLAGS if flag not in ('--draft', '--draft-model', '--draft-conf', '--draft-fallback')]
    for flag in tuning:
        value = argument(original, flag)
        if value is not None:
            args += [flag, value]
    return args + ['--draft', 'dflash2', '--draft-model', state['path'], '--draft-fallback', 'none',
                   '--mtp-draft-vocab', state['vocab'], '--spec', str(spec if spec is not None else 6),
                   *(['--draft-conf', confidence] if confidence is not None else [])]


def serve_argv(runtime, cfg, out, state):
    """The runtime bridge binds prepared resources without freezing Auto retries."""
    argv = [runtime.metadata['python'], '-m', 'serve.server']
    if '_modloader_policy' in cfg:
        argv = [runtime.metadata['python'], str(policy_script()), runtime.path]
    return argv


def policy_script():
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1])) / 'modloader/strata_policy.py'


def plan_args(runtime, cfg):
    """Ask installed Control for the same plan as serve, without starting engines."""
    import subprocess
    from .runtime import local_process_environment
    from .backends.adapters import OFFLINE_ENV
    result = subprocess.run([runtime.metadata['python'], str(policy_script()), runtime.path, '--plan'],
                            input=json.dumps(cfg), text=True, encoding='utf-8', capture_output=True,
                            timeout=45, env=local_process_environment({**OFFLINE_ENV, 'STRATA_NO_ACQUIRE': '1'}),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise ValueError('Strata 启动规划失败：' + result.stderr.strip())
    return json.loads(result.stdout)['args']
