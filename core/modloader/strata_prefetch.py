"""Strata's opt-in startup-prefetch engine, bound to the installed engine identity.

Discovery only reads local files. Building is an explicit managed-runtime operation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

KEY = 'startup_expert_prefetch'
FLAG = '--startup-expert-prefetch'
EXECUTABLE = 'strata-startup-prefetch.exe'
MANIFEST = 'startup-prefetch.json'


def _json(path):
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} 必须为 JSON 对象')
    return value


def engine_identity(root):
    """The engine commit is deliberately independent of the Python service source."""
    engine = Path(root) / 'engine'
    build = _json(engine / 'BUILD.json')
    component = _json(engine / 'engine-component.json')
    version = str(build.get('version') or component.get('version') or '')
    if build.get('version') and component.get('version') and str(component['version']) != version:
        raise ValueError('Strata 引擎版本清单不一致')
    commit = str(build.get('commit') or '')
    build_id = str(component.get('build_id') or '')
    if commit and build_id and not build_id.startswith(commit):
        raise ValueError('Strata 引擎提交与 build_id 不一致')
    backend = str(component.get('backend') or build.get('backend') or ('cuda' if build.get('cuda') else ''))
    return dict(version=version, commit=commit, backend=backend, build_id=build_id)


def _sidecar(directory, identity):
    value = _json(Path(directory) / MANIFEST)
    if (value.get('schema_version') != 1 or value.get('feature') != KEY
            or value.get('engine_identity') != identity or value.get('executable') != EXECUTABLE
            or not isinstance(value.get('patch_revision'), str) or not value['patch_revision']):
        raise ValueError('启动预取产物与当前 Strata 引擎不匹配，请重新准备')
    commit = value.get('source_commit')
    if (not isinstance(commit, str) or not re.fullmatch('[0-9a-fA-F]{40}', commit)
            or not identity.get('commit') or not commit.lower().startswith(identity['commit'].lower())):
        raise ValueError('启动预取产物源码提交与当前 Strata 引擎不匹配')
    exe = Path(directory) / EXECUTABLE
    if not exe.is_file() or exe.stat().st_size == 0 or exe.resolve().parent != Path(directory).resolve():
        raise ValueError('启动预取引擎文件缺失或不在对应引擎目录')
    return value


def legacy_selected(runtime):
    selected = os.environ.get('MODLOADER_STRATA_DEV_RUNTIME')
    return bool(runtime.platform == 'windows' and selected and Path(selected).resolve() == Path(runtime.path).resolve())


def effective_enabled(profile, runtime):
    if KEY in profile.parameters:
        value = profile.parameters[KEY]
        if not isinstance(value, bool):
            raise ValueError('启动预取必须为布尔值')
        return value
    from .mtp import inherited_args
    args, _ = inherited_args(profile, runtime)
    return legacy_selected(runtime) or FLAG in args


def capability(runtime):
    result = dict(status='missing', available=False, preparable=False, source=None,
                  message='尚未准备启动预取引擎', engine_identity={})
    if runtime.backend != 'strata' or runtime.platform != 'windows' or runtime.metadata.get('strata_family') == 'deepseek41':
        return {**result, 'status': 'unsupported', 'message': '启动预取仅支持 Windows CUDA GGUF 引擎'}
    try:
        identity = engine_identity(runtime.path)
        result['engine_identity'] = identity
        result['preparable'] = bool(runtime.managed and identity['backend'] == 'cuda' and identity['version']
                                    and re.fullmatch('[0-9a-fA-F]{7,40}', identity['commit']))
        sidecar = Path(runtime.path) / 'engine' / MANIFEST
        if sidecar.is_file():
            if identity['backend'] != 'cuda':
                raise ValueError('启动预取产物仅支持 CUDA 引擎')
            _sidecar(sidecar.parent, identity)
            return {**result, 'status': 'ready', 'available': True, 'source': 'managed', 'message': ''}
        if identity['backend'] and identity['backend'] != 'cuda':
            return {**result, 'status': 'unsupported', 'message': '启动预取仅支持 CUDA 引擎'}
    except (OSError, ValueError, TypeError) as error:
        result.update(status='mismatch', message=f'启动预取：{error}')
    if legacy_selected(runtime):
        exe = os.environ.get('MODLOADER_STRATA_DEV_EXE')
        if exe and Path(exe).is_file():
            return {**result, 'status': 'ready', 'available': True, 'source': 'legacy', 'message': ''}
        result['message'] = ('未设置 Strata 开发版引擎 MODLOADER_STRATA_DEV_EXE' if not exe
                             else f'Strata 开发版引擎不存在：{Path(exe).resolve()}')
    return result


def metadata(root):
    """Refresh recorded metadata without trusting a stale runtime registration."""
    try:
        identity = engine_identity(root)
    except (OSError, ValueError, TypeError):
        return {'engine_identity': {}, KEY: None}
    try:
        value = _sidecar(Path(root) / 'engine', identity)
    except (OSError, ValueError, TypeError):
        value = None
    return {'engine_identity': identity, KEY: value}


def profile_status(model, profile, runtime):
    result = capability(runtime)
    enabled = effective_enabled(profile, runtime)
    result['enabled'] = enabled
    if not enabled:
        return result
    # Historical profiles retain the process-scoped development contract.
    if KEY not in profile.parameters:
        return result
    from .strata_deepseek import is_deepseek
    try:
        if not result['available']:
            raise ValueError(result['message'])
        if runtime.platform != 'windows' or model.format.lower() != 'gguf' or is_deepseek(model):
            raise ValueError('启动预取仅支持 Windows CUDA mapped GGUF')
    except (OSError, ValueError, TypeError) as error:
        result.update(status='unsupported', message=f'启动预取：{error}')
    return result


def validate_profile(model, profile, runtime):
    result = profile_status(model, profile, runtime)
    if profile.parameters.get(KEY) is True and (not result['available'] or result['status'] != 'ready'):
        raise ValueError(result['message'])
    return result


def select_engine(profile, runtime):
    from .mtp import inherited_args
    inherited, _ = inherited_args(profile, runtime)
    if ('--raw-token-continuation' in inherited
            and '--raw-token-continuation' not in runtime.metadata.get('supported_flags', [])):
        sidecar = _json(Path(runtime.path) / 'engine' / MANIFEST)
        if 'raw_token_continuation' in sidecar.get('features', []):
            state = capability(runtime)
            if not state['available']:
                raise ValueError(state['message'])
            return str(Path(runtime.path) / 'engine' / EXECUTABLE), effective_enabled(profile, runtime)
        # Historical runtimes may not have a recorded help inventory. Only a
        # known incompatible official engine requires the newly prepared build.
        identity = engine_identity(runtime.path)
        if runtime.metadata.get('supported_flags') or identity['version'] == '0.1.56':
            raise ValueError('此配置需要 raw-token-continuation；请先为目标环境准备自定义引擎')
    if profile.parameters.get(KEY) is False:
        return str(Path(runtime.path) / 'engine/strata.exe'), False
    if KEY not in profile.parameters and not legacy_selected(runtime):
        return str(Path(runtime.path) / 'engine/strata.exe'), False
    if KEY not in profile.parameters or capability(runtime)['source'] == 'legacy':
        exe = os.environ.get('MODLOADER_STRATA_DEV_EXE')
        if not exe:
            raise ValueError('未设置 Strata 开发版引擎 MODLOADER_STRATA_DEV_EXE')
        if not Path(exe).is_file():
            raise ValueError(f'Strata 开发版引擎不存在：{Path(exe).resolve()}')
        return str(Path(exe).resolve()), True
    state = capability(runtime)
    if not state['available']:
        raise ValueError(state['message'])
    return str(Path(runtime.path) / 'engine' / EXECUTABLE), True


def build_script():
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
    return root / 'scripts/build-strata-prefetch.ps1'


async def prepare_runtime(runtime, paths, progress, cancel, *, installing=False, proxy=None):
    from .backends import installers
    if not installing:
        installers.validate_managed_runtime(runtime, paths)
    state = capability(runtime)
    if state['source'] == 'managed' and state['available']:
        updated = runtime.model_copy(deep=True)
        updated.metadata.update(metadata(runtime.path))
        return updated
    if not state['preparable']:
        raise ValueError('此环境缺少可追溯的 Windows CUDA 引擎，无法准备启动预取')
    script = build_script()
    if not script.is_file():
        raise ValueError('启动预取构建脚本缺失，请更新 ModLoader')
    work = paths.work / ('strata-prefetch-' + uuid.uuid4().hex)
    stage = work / 'output'
    stage.mkdir(parents=True)
    try:
        return await _build_and_publish(runtime, paths, state, script, work, stage, progress, cancel, proxy)
    finally:
        # This is product-owned job scratch, after the builder has exited; never the runtime or a shared cache.
        if work.resolve().parent != paths.work.resolve() or not work.name.startswith('strata-prefetch-'):
            raise ValueError('启动预取临时目录越界，拒绝清理')
        await installers._thread(shutil.rmtree, work, cancel=cancel)


async def _build_and_publish(runtime, paths, state, script, work, stage, progress, cancel, proxy):
    from .backends import installers
    # Use the application's selected route, not the desktop launcher's proxy or bypass list.
    proxy_environment = {key: proxy or '' if key.lower() in {'http_proxy', 'https_proxy'} else ''
                         for key in os.environ if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'}}
    proxy_environment.update(HTTP_PROXY=proxy or '', HTTPS_PROXY=proxy or '', ALL_PROXY='', NO_PROXY='')
    await installers._emit(progress, phase='prepare', message='准备同版本 Strata 启动预取引擎', can_pause=False)
    await installers._command(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                               '-RuntimeRoot', runtime.path, '-OutputDirectory', str(stage), '-WorkDirectory', str(work),
                               '-ToolchainConfig', str(paths.root / 'runtime-tools/strata-prefetch.json')],
                              progress, cancel, timeout=7200, owned_root=work, env=proxy_environment)
    installers._check(cancel)
    identity = engine_identity(runtime.path)
    if identity != state['engine_identity']:
        raise ValueError('构建期间 Strata 引擎已变化，未发布启动预取产物')
    _sidecar(stage, identity)
    target = Path(runtime.path) / 'engine'
    # Publish without an await/cancellation point; restore the previous pair if either replace fails.
    backups, published = {}, []
    for name in (EXECUTABLE, MANIFEST):
        destination = target / name
        if destination.exists():
            backup = work / ('previous-' + name)
            shutil.copy2(destination, backup)
            backups[name] = backup
    try:
        for name in (EXECUTABLE, MANIFEST):
            os.replace(stage / name, target / name)
            published.append(name)
    except OSError:
        for name in reversed(published):
            if name in backups:
                os.replace(backups[name], target / name)
            else:
                (target / name).unlink()
        raise
    updated = runtime.model_copy(deep=True)
    updated.metadata.update(metadata(runtime.path))
    return updated
