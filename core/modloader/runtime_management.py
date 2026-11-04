"""Application registration changes; imported runtime directories are read-only here."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from fastapi import Request

from .models import LoadProfile, ModelAsset, RuntimeInstallation


def _within(value, root):
    if not isinstance(value, str) or not value:
        return False
    value = value.split('=', 1)[-1] if value.startswith('--') and '=' in value else value
    path_type = PurePosixPath if str(root).startswith('/') and not str(root).startswith('//') else Path
    try:
        path = path_type(value)
        if not path.is_absolute():
            return False
        base = path_type(root)
        if path_type is Path:
            path, base = path.resolve(), base.resolve()
        return path.is_relative_to(base)
    except (OSError, ValueError):
        return False


def references_root(value, root):
    if isinstance(value, dict):
        return any(references_root(item, root) for item in value.values())
    if isinstance(value, list):
        return any(references_root(item, root) for item in value)
    return _within(value, root)


def protect_runtime_resources(controller, runtime, replacements):
    """Uninstall must not delete registered model files or retained profile resources."""
    roots = [runtime.metadata['managed_root']]
    if runtime.platform == 'wsl2':
        roots.append(runtime.metadata.get('linux_root', runtime.path))
    updated = {p['id']: p for p in replacements}
    from .platforms import location_of, runtime_distribution, windows_to_wsl, windows_path
    mapped_roots = {}
    def guest_host_root(distribution):
        if distribution not in mapped_roots:
            mapped_roots[distribution] = windows_to_wsl(roots[0], distribution)
        return mapped_roots[distribution]
    for kind in ('models', 'profiles'):
        for original in controller.store.all(kind):
            row = updated.get(original['id'], original) if kind == 'profiles' else original
            candidate_roots = [roots[0]]
            if kind == 'models':
                location = location_of(row)
                if location.platform == 'wsl2':
                    candidate_roots = [guest_host_root(location.distribution)]
                    if runtime.platform == 'wsl2' and location.distribution == runtime_distribution(runtime):
                        candidate_roots += roots[1:]
                elif runtime.platform == 'wsl2':
                    candidate_roots += [windows_path(runtime_distribution(runtime), value) for value in roots[1:]]
            else:
                referenced = controller.store.get('runtimes', row.get('runtime_id'))
                if referenced:
                    owner = RuntimeInstallation(**referenced)
                    if owner.platform == 'wsl2':
                        candidate_roots.append(guest_host_root(runtime_distribution(owner)))
                        if runtime.platform == 'wsl2' and runtime_distribution(owner) == runtime_distribution(runtime):
                            candidate_roots += roots[1:]
            if any(references_root(row, root) for root in candidate_roots):
                raise ValueError(f"{row.get('name', row['id'])} 仍使用此环境目录中的资源，请先调整资源路径；可以仅移除登记")



def _strata_parameters(profile, source, target):
    from .backends.adapters import _strata_config
    parameters = deepcopy(profile.parameters)
    warnings = []
    source_root, target_root = Path(source.path).resolve(), Path(target.path).resolve()

    def runtime_file(value):
        if not _within(value, str(source_root)):
            return value
        translated = target_root / Path(value).resolve().relative_to(source_root)
        if not translated.exists():
            raise ValueError(f'新环境缺少对应运行文件：{translated}；请先调整该配置')
        return str(translated)

    config_file = parameters.get('configuration')
    if config_file or parameters.get('strata_config'):
        location = Path(config_file) if config_file else source_root / 'strata.json'
        raw = json.loads(location.read_text('utf-8-sig')) if config_file else parameters['strata_config']
        cfg = _strata_config(raw, location)
        if cfg.get('arch') == 'deepseek41':
            from .strata_deepseek import engine_path
            cfg['exe'] = str(engine_path(target_root))
            cfg['lib_dirs'] = [item for item in cfg.get('lib_dirs', []) if not _within(item, str(source_root))]
            cfg['lib_dirs'] += target.metadata.get('lib_dirs', [])
        else:
            cfg['exe'] = str(target_root / 'engine/strata.exe')
        cfg['cwd'] = str(target_root)
        cfg['lib_dirs'] = [runtime_file(item) for item in cfg.get('lib_dirs', [])]
        if not cfg['lib_dirs']:
            cfg['lib_dirs'] = target.metadata.get('lib_dirs', [])
        if cfg.get('vision', {}).get('exe'):
            cfg['vision']['exe'] = runtime_file(cfg['vision']['exe'])
        for key in ('args', 'user_args'):
            args = cfg.get(key, [])
            for index, value in enumerate(args):
                if value in ('--expert-profile', '--mtp', '--draft-model', '--mtp-draft-vocab') and index + 1 < len(args):
                    args[index + 1] = runtime_file(args[index + 1])
        parameters.pop('configuration', None)
        parameters['strata_config'] = cfg
        if config_file:
            warnings.append('已将导入的 Strata 配置保存为独立快照，原配置文件不修改')
    for key in ('expert_profile', 'mtp_path', 'dflash2_path'):
        if parameters.get(key):
            parameters[key] = runtime_file(parameters[key])
    extra = list(profile.extra_args)
    for index, value in enumerate(extra):
        flag = value.split('=', 1)[0]
        if flag in ('--expert-profile', '--mtp', '--draft-model', '--mtp-draft-vocab') and '=' not in value and index + 1 < len(extra):
            extra[index + 1] = runtime_file(extra[index + 1])
        elif flag in ('--expert-profile', '--mtp', '--draft-model', '--mtp-draft-vocab') and '=' in value:
            extra[index] = flag + '=' + runtime_file(value.split('=', 1)[1])
    return parameters, extra, warnings


def switch_prefetch(controller, source, target, profile_ids, disable=False):
    from .strata_prefetch import effective_enabled, capability
    if not isinstance(disable, bool):
        raise ValueError('关闭启动预取必须为布尔值')
    affected = []
    if source.backend == 'strata':
        for identifier in profile_ids:
            profile = LoadProfile(**controller.require('profiles', identifier))
            if profile.runtime_id != source.id:
                raise ValueError('配置已不再引用此环境，请刷新后重试')
            if effective_enabled(profile, source):
                affected.append({'id': profile.id, 'name': profile.name})
    status = ('disabled' if disable else 'preserved' if capability(target)['available'] else 'choice_required') if affected else 'unchanged'
    return {'status': status, 'affected_profiles': affected,
            'message': '目标环境没有匹配的启动预取引擎；请先准备，或明确关闭预取后切换' if status == 'choice_required' else ''}


async def prepare_switch(controller, source, target, profile_ids, *, disable_startup_expert_prefetch=False):
    if source.id == target.id:
        raise ValueError('请选择不同的运行环境')
    if source.backend != target.backend:
        raise ValueError('替换环境必须属于同一后端')
    if not isinstance(profile_ids, list) or any(not isinstance(item, str) for item in profile_ids):
        raise ValueError('请选择要切换的加载配置')
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError('加载配置不能重复')
    prefetch = switch_prefetch(controller, source, target, profile_ids, disable_startup_expert_prefetch)
    if prefetch['status'] == 'choice_required':
        raise ValueError(prefetch['message'])
    expected, replacements, warnings = [], [], []
    for identifier in profile_ids:
        original = controller.require('profiles', identifier)
        profile = LoadProfile(**original)
        if profile.runtime_id != source.id or profile.backend != source.backend:
            raise ValueError(f'{profile.name} 已不再引用此环境，请刷新后重试')
        model = ModelAsset(**controller.require('models', profile.model_id))
        next_profile = profile.model_copy(deep=True)
        next_profile.runtime_id, next_profile.revision = target.id, profile.revision + 1
        if profile.backend == 'strata':
            next_profile.parameters, next_profile.extra_args, notes = _strata_parameters(profile, source, target)
            from .strata_prefetch import KEY, effective_enabled
            next_profile.parameters[KEY] = effective_enabled(profile, source) and not disable_startup_expert_prefetch
            warnings.extend(f'{profile.name}：{note}' for note in notes)
        preview = await controller.runtime.preview(model, next_profile, target)
        compatibility = preview.get('compatibility') or {}
        if compatibility.get('status') == 'unsupported':
            raise ValueError(f"{profile.name}：" + '；'.join(compatibility.get('reasons', [])))
        preparation = preview.get('preparation') or {}
        if preparation.get('status') in ('error', 'choice_required'):
            raise ValueError(f"{profile.name}：{preparation.get('message', '请先处理准备文件')}")
        if preparation.get('status') == 'needs_preparation':
            warnings.append(f'{profile.name}：下次加载需要为新环境准备本模型的运行资源')
        expected.append(original)
        replacements.append({**original, **next_profile.model_dump()})
    return expected, replacements, list(dict.fromkeys(warnings))


def register_runtime_routes(app, controller):
    from fastapi import HTTPException
    from .runtime import ACTIVE_STATES
    from .store import UNFINISHED_JOBS, TERMINAL_JOBS
    from .backends import catalog, installers
    from .operation_progress import operation_response

    def owned_runtime(identifier):
        runtime = RuntimeInstallation(**controller.require('runtimes', identifier))
        installers.validate_managed_runtime(runtime, controller.paths)
        return runtime

    @app.get('/api/runtimes/versions')
    async def versions(backend: str, cursor: str | None = None):
        return await catalog.list_versions(backend, cursor=cursor, proxy=controller.proxy())

    @app.post('/api/runtimes/plan')
    async def plan(body: dict):
        backend = body.get('backend')
        if backend not in controller.adapters:
            raise ValueError('未知后端')
        update_from = body.get('update_from')
        if update_from:
            source = owned_runtime(update_from)
            if source.backend != backend:
                raise ValueError('更新目标必须属于同一后端')
        options = body.get('options') or {}
        if not isinstance(options, dict) or not options.get('version'):
            raise ValueError('请选择明确的软件版本')
        selected = {key: options[key] for key in ('version', 'variant', 'python') if key in options}
        prefetch = options.get('startup_expert_prefetch', False)
        if not isinstance(prefetch, bool) or prefetch and backend != 'strata':
            raise ValueError('启动预取选项只适用于 Strata，且必须为布尔值')
        value = await catalog.plan_install(backend, controller.paths, selected, proxy=controller.proxy())
        if backend == 'strata':
            if prefetch and (value.get('variant') or '').startswith(('hip', 'dsv41')):
                raise ValueError('启动预取仅支持 Windows CUDA GGUF 引擎')
            value.setdefault('options', {})['startup_expert_prefetch'] = prefetch
        value['update_from'] = update_from
        if body.get('retry_of'):
            previous = controller.require('jobs', body['retry_of'])
            old_options = previous.get('request', {}).get('options', {})
            if (previous.get('kind') == 'install' and previous.get('state') in TERMINAL_JOBS
                    and previous.get('state') not in {'completed', 'complete'}
                    and previous.get('backend') == backend
                    and str(old_options.get('version')) == value['version']
                    and old_options.get('variant', old_options.get('flavor')) == value['variant']
                    and old_options.get('python') == selected.get('python')):
                stage = next((s for s in reversed(previous.get('stages', [])) if s.get('title') == 'Python 依赖' and s.get('files')), None)
                if stage:
                    value['_retry_packages'] = {'backend': backend, 'version': value['version'], 'stage': deepcopy(stage)}
                    value['dependency_notice'] = '重试将优先复用上次依赖清单；目标 Python 环境一致时校验缓存、补下载缺失文件并继续安装。'
        # Only the server's resolved manifest can authorize an installation.
        controller.install_plans[value['id']] = deepcopy(value)
        return {key: item for key, item in value.items() if key != '_retry_packages'}

    @app.post('/api/runtimes/install')
    async def install(body: dict):
        async with controller.environment_lock:
            value = controller.install_plans.get(body.get('plan_id'))
            backend = value['backend'] if value else body.get('backend')
            if any(j.get('backend') == backend for j in controller.store.jobs('install', UNFINISHED_JOBS)):
                raise HTTPException(409, '该后端已有未结束的安装任务，请先完成或停止')
            if value is None:
                raise HTTPException(409, '安装清单已失效，请重新开始安装')
            if value.get('update_from'):
                owned_runtime(value['update_from'])
            request = {'backend': value['backend'], 'options': {**value.get('options', {}), 'version':value['version'],
                       'variant':value.get('variant'), '_plan':deepcopy(value)}}
            if value.get('update_from'):
                request['update_from'] = value['update_from']
            job = controller.new_job('install', request)
            controller.install_plans.pop(value['id'])
            return job

    @app.post('/api/runtimes/{identifier}/prepare-startup-prefetch')
    async def prepare_startup_prefetch(identifier: str):
        async with controller.environment_lock:
            runtime = owned_runtime(identifier)
            if runtime.backend != 'strata':
                raise ValueError('只有 Strata 环境可以准备启动预取')
            if any(job.get('backend') == 'strata' for job in controller.store.jobs('install', UNFINISHED_JOBS)):
                raise HTTPException(409, 'Strata 已有未结束的环境任务，请先完成或停止')
            return controller.new_job('install', {'backend': 'strata', 'runtime_id': identifier,
                                                  'operation': 'prepare_startup_prefetch'})

    @app.post('/api/runtimes/{identifier}/refresh')
    async def refresh(identifier: str, request: Request):
        async def operation(progress):
            progress({'stage': 'runtime_verify', 'message': '等待运行环境检查'})
            async with controller.environment_lock:
                controller.assert_runtime_available(identifier)
                runtime = RuntimeInstallation(**controller.require('runtimes', identifier))
                progress({'stage': 'runtime_verify', 'message': '检查自定义引擎配置' if runtime.backend == 'custom' else '检查运行入口与版本'})
                def checked(event):
                    if event.get('message'):
                        progress({'stage': 'runtime_verify', 'message': event['message']})
                try:
                    result = await installers.validate_runtime(runtime, paths=controller.paths, progress=checked)
                except (RuntimeError, TimeoutError) as error:
                    raise ValueError(str(error)) from error
                if result is not None:
                    runtime = result
                return controller.store.put('runtimes', runtime.model_dump())
        return await operation_response(request, operation)

    @app.patch('/api/runtimes/{identifier}')
    async def edit_runtime(identifier: str, body: dict):
        from .backends.custom import import_custom_runtime, _text
        async with controller.environment_lock:
            controller.assert_runtime_available(identifier)
            current = RuntimeInstallation(**controller.require('runtimes', identifier))
            if current.backend != 'custom' or body.get('backend', 'custom') != 'custom':
                raise ValueError('此入口只编辑自定义引擎；现有后端请使用原有环境管理流程')
            active = [i.alias for i in controller.runtime.instances.values() if i.runtime_id == identifier and i.state in ACTIVE_STATES]
            if active:
                raise HTTPException(409, '请先停止使用此引擎的实例：' + '、'.join(active))
            updated = import_custom_runtime(body.get('path', current.path), body.get('metadata', current.metadata), body.get('platform', current.platform))
            updated.id, updated.version = current.id, current.version
            updated.name = _text(body.get('name', current.name), '引擎名称').strip()
            return controller.store.put('runtimes', updated.model_dump())

    async def switch_data(identifier, body, *, preview=False):
        source = RuntimeInstallation(**controller.require('runtimes', identifier))
        target = RuntimeInstallation(**controller.require('runtimes', body.get('target_id')))
        if source.id == target.id or source.backend != target.backend:
            raise ValueError('请选择同一后端的不同运行环境')
        profile_ids = body.get('profile_ids', [])
        if not isinstance(profile_ids, list) or not profile_ids or any(not isinstance(value, str) for value in profile_ids):
            raise ValueError('请选择至少一套加载配置')
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError('加载配置不能重复')
        disabled = body.get('disable_startup_expert_prefetch', False)
        prefetch = switch_prefetch(controller, source, target, profile_ids, disabled)
        if preview and prefetch['status'] == 'choice_required':
            return [], [], {'profiles': [{'id': p, 'name': controller.require('profiles', p)['name']} for p in profile_ids],
                            'warnings': [], 'source_id': source.id, 'target_id': target.id, 'startup_expert_prefetch': prefetch}
        expected, replacements, warnings = await prepare_switch(controller, source, target, profile_ids,
                                                                disable_startup_expert_prefetch=disabled)
        if not expected:
            raise ValueError('请选择至少一套加载配置')
        public = {'profiles':[{'id':p['id'], 'name':p['name']} for p in replacements], 'warnings':warnings,
                  'source_id':source.id, 'target_id':target.id, 'startup_expert_prefetch': prefetch}
        return expected, replacements, public

    @app.post('/api/runtimes/{identifier}/switch-preview')
    async def switch_preview(identifier: str, body: dict):
        async with controller.environment_lock:
            _, _, result = await switch_data(identifier, body, preview=True)
            return result

    @app.post('/api/runtimes/{identifier}/switch')
    async def switch(identifier: str, body: dict):
        async with controller.environment_lock:
            controller.assert_runtime_available(identifier)
            controller.assert_runtime_available(body.get('target_id'))
            expected, replacements, result = await switch_data(identifier, body)
            controller.store.apply_runtime_switch(expected, replacements)
            return result

    @app.delete('/api/runtimes/{identifier}')
    async def remove(identifier: str, delete_files: bool = False, replacement_id: str | None = None,
                     disable_startup_expert_prefetch: bool = False):
        async with controller.environment_lock:
            controller.assert_runtime_available(identifier)
            runtime = RuntimeInstallation(**controller.require('runtimes', identifier))
            active = [i.alias for i in controller.runtime.instances.values() if i.runtime_id == identifier and i.state in ACTIVE_STATES]
            if active:
                raise HTTPException(409, '请先停止使用此环境的实例：' + '、'.join(active))
            if any(j.get('update_from') == identifier for j in controller.store.jobs('install', UNFINISHED_JOBS)):
                raise HTTPException(409, '此环境仍有关联更新任务，请先完成或停止')
            if any(j.get('runtime_id') == identifier for j in controller.store.jobs('install', UNFINISHED_JOBS)):
                raise HTTPException(409, '此环境仍有准备任务，请先完成或停止')
            refs = [p for p in controller.store.all('profiles') if p['runtime_id'] == identifier]
            expected, replacements, warnings = [], [], []
            if refs:
                if not replacement_id:
                    raise HTTPException(409, '请为关联加载配置选择替换环境：' + '、'.join(p['name'] for p in refs))
                target = RuntimeInstallation(**controller.require('runtimes', replacement_id))
                expected, replacements, warnings = await prepare_switch(controller, runtime, target, [p['id'] for p in refs],
                                                                        disable_startup_expert_prefetch=disable_startup_expert_prefetch)
            if delete_files:
                installers.validate_managed_runtime(runtime, controller.paths)
                protect_runtime_resources(controller, runtime, replacements)
                await installers.uninstall_runtime(runtime, controller.paths)
            controller.store.apply_runtime_switch(expected, replacements, remove_runtime=identifier)
            return {'ok':True, 'warnings':warnings}

    @app.get('/api/installations/history')
    async def history(page: int = 1):
        if page < 1:
            raise ValueError('页码必须从 1 开始')
        return controller.store.job_history('install', TERMINAL_JOBS, page)

    def delete_job(identifier):
        job = controller.require('jobs', identifier)
        task = controller.job_tasks.get(identifier)
        if job.get('kind') != 'install':
            raise ValueError('此入口只处理环境安装记录')
        if job['state'] not in TERMINAL_JOBS or (task and not task.done()):
            raise HTTPException(409, '任务尚未结束，请停止后再删除记录')
        controller.store.remove('jobs', identifier)
        for values in (controller.job_records, controller.job_tasks, controller.cancel_events,
                       controller.download_pauses, controller.stage_confirmations):
            values.pop(identifier, None)

    @app.delete('/api/jobs/{identifier}')
    async def remove_job(identifier: str):
        delete_job(identifier)
        return {'ok':True}

    @app.delete('/api/installations/history')
    async def clear_history():
        removed = 0
        for job in controller.store.jobs('install', TERMINAL_JOBS):
            task = controller.job_tasks.get(job['id'])
            if not task or task.done():
                delete_job(job['id'])
                removed += 1
        return {'ok':True, 'removed':removed}

    @app.post('/api/jobs/{identifier}/confirm')
    async def confirm(identifier: str, body: dict):
        job = controller.job_records.get(identifier) or controller.require('jobs', identifier)
        pending = controller.stage_confirmations.get(identifier)
        if job['state'] != 'awaiting_confirmation' or not pending or pending[0] != body.get('stage_id'):
            raise HTTPException(409, '待确认清单已变化或已经确认，请刷新后重试')
        if controller.cancel_events[identifier].is_set():
            raise HTTPException(409, '任务正在停止')
        job.update(state='running', phase='prepare', message='清单已确认，继续安装')
        controller.store.put('jobs', job)
        pending[1].set()
        return job
