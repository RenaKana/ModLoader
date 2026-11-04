import asyncio
import json
from pathlib import Path
import sys

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.models import RuntimeInstallation, RuntimeInstance, ModelAsset, LoadProfile
from test_core import FixtureAdapter, add_profile
from test_strata_preparation import write_pack, write_expert_profile


@pytest.fixture
async def management(tmp_path):
    controller = Controller(tmp_path / 'data', 'management-secret')
    controller.adapters['fixture'] = FixtureAdapter()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization':'Bearer management-secret'}) as client:
        yield controller, client
    await controller.close()


def target_runtime(controller, tmp_path, backend='fixture', managed=False):
    root = controller.paths.runtimes / 'target' if managed else tmp_path / 'external-target'
    root.mkdir(parents=True)
    exe = root / 'engine.exe'
    exe.write_text('external sentinel')
    runtime = RuntimeInstallation(id='target', backend=backend, name='Target', path=str(exe), managed=managed,
                                  version='1', metadata={'managed_root':str(root), 'install_root':str(root)} if managed else {})
    if managed:
        (root / 'modloader-runtime.json').write_text(runtime.model_dump_json())
    controller.store.put('runtimes', runtime.model_dump())
    return runtime


async def test_state_exposes_verified_runtime_defaults_without_saving_them(management, tmp_path):
    from modloader.backends.source_builds import PINS

    controller, client = management
    verified = RuntimeInstallation(id='ninfer-pinned', backend='ninfer', name='Pinned NInfer',
                                   platform='wsl2', path='/fixture/ninfer-serve',
                                   metadata={'source_commit': PINS['ninfer']['commit'],
                                             'version_basis': 'verified_source_commit'})
    unknown = verified.model_copy(update={'id': 'ninfer-unknown', 'metadata': {}})
    for runtime in (verified, unknown):
        controller.store.put('runtimes', runtime.model_dump())
    other = target_runtime(controller, tmp_path)
    before = controller.store.all('runtimes')

    response = await client.get('/api/state')
    assert response.status_code == 200, response.text
    runtimes = {runtime['id']: runtime for runtime in response.json()['runtimes']}
    defaults = lambda identifier: {field['key']: field['default'] for field in runtimes[identifier]['generation_parameters']}
    assert defaults(verified.id)['max_tokens'] == 8192
    assert defaults(unknown.id)['max_tokens'] is None
    assert 'generation_parameters' not in runtimes[other.id]
    assert controller.store.all('runtimes') == before


async def test_remove_external_migrates_references_without_touching_directory(management, tmp_path):
    controller, client = management
    _, profile, source = add_profile(controller, tmp_path, 'one')
    target = target_runtime(controller, tmp_path)
    assert (await client.delete('/api/runtimes/' + source.id)).status_code == 409
    response = await client.delete('/api/runtimes/' + source.id, params={'replacement_id':target.id})
    assert response.status_code == 200, response.text
    updated = controller.require('profiles', profile.id)
    assert updated['runtime_id'] == target.id and updated['revision'] == profile.revision + 1
    assert updated['parameters'] == profile.parameters and updated['alias'] == profile.alias
    assert controller.store.get('runtimes', source.id) is None
    assert Path(source.path).exists() and Path(target.path).read_text() == 'external sentinel'


async def test_running_environment_cannot_be_removed_after_profile_switch(management, tmp_path):
    controller, client = management
    model, profile, source = add_profile(controller, tmp_path, 'live')
    target = target_runtime(controller, tmp_path)
    instance = RuntimeInstance(id='live', model_id=model.id, profile_id=profile.id, profile_revision=1, alias='live',
                               backend='fixture', runtime_id=source.id, state='ready', port=0, started_at='fixture')
    controller.runtime.instances[instance.id] = instance
    response = await client.post('/api/runtimes/' + source.id + '/switch', json={'target_id':target.id, 'profile_ids':[profile.id]})
    assert response.status_code == 200, response.text
    assert instance.runtime_id == source.id and instance.state == 'ready'
    response = await client.delete('/api/runtimes/' + source.id)
    assert response.status_code == 409 and 'live' in response.text
    controller.runtime.instances.clear()


async def test_switch_failure_is_atomic_and_preserves_profile_settings(management, tmp_path, monkeypatch):
    controller, client = management
    _, first, source = add_profile(controller, tmp_path, 'one')
    _, second, _ = add_profile(controller, tmp_path, 'two')
    target = target_runtime(controller, tmp_path)
    before = controller.store.all('profiles')
    original = controller.runtime.preview
    async def preview(model, profile, runtime):
        if profile.id == second.id:
            raise ValueError('该配置含新环境不支持的参数')
        return await original(model, profile, runtime)
    monkeypatch.setattr(controller.runtime, 'preview', preview)
    response = await client.post('/api/runtimes/' + source.id + '/switch', json={'target_id':target.id, 'profile_ids':[first.id,second.id]})
    assert response.status_code == 400 and '不支持' in response.text
    assert controller.store.all('profiles') == before


async def test_external_update_and_uninstall_are_rejected(management, tmp_path):
    controller, client = management
    runtime = target_runtime(controller, tmp_path, backend='llama_cpp')
    before = Path(runtime.path).read_bytes()
    response = await client.post('/api/runtimes/plan', json={'backend':'llama_cpp', 'update_from':runtime.id,
                                                          'options':{'version':'b11609','variant':'cpu'}})
    assert response.status_code == 400 and '外部' in response.text
    response = await client.delete('/api/runtimes/' + runtime.id, params={'delete_files':'true'})
    assert response.status_code == 400
    assert Path(runtime.path).read_bytes() == before and controller.store.get('runtimes', runtime.id)


async def test_uninstall_preserves_registered_model_resource(management, tmp_path):
    controller, client = management
    runtime = target_runtime(controller, tmp_path, managed=True)
    model = ModelAsset(id='shared', name='Shared model', path=runtime.path, format='gguf', files=[runtime.path])
    controller.store.put('models', model.model_dump())
    response = await client.delete('/api/runtimes/' + runtime.id, params={'delete_files':'true'})
    assert response.status_code == 400 and 'Shared model' in response.text
    assert Path(runtime.path).exists()


async def test_install_history_pagination_and_record_only_deletion(management, tmp_path):
    controller, client = management
    runtime = target_runtime(controller, tmp_path)
    for i in range(23):
        controller.store.put('jobs', {'id':f'old-{i}', 'kind':'install', 'state':'cancelled', 'request':{}, 'runtime_id':runtime.id})
    for state in ('running','paused','awaiting_confirmation','cancelling'):
        controller.store.put('jobs', {'id':state, 'kind':'install', 'state':state, 'request':{}})
    controller.store.put('jobs', {'id':'model-download', 'kind':'download', 'state':'completed', 'request':{}})
    first = (await client.get('/api/installations/history')).json()
    third = (await client.get('/api/installations/history?page=3')).json()
    assert first['total'] == 23 and len(first['items']) == 10 and first['items'][0]['id'] == 'old-22'
    assert len(third['items']) == 3 and 'created_at' not in third['items'][0]
    state = (await client.get('/api/state')).json()
    assert {j['state'] for j in state['installations']} == {'running','paused','awaiting_confirmation','cancelling'}
    for state in ('running','paused','awaiting_confirmation','cancelling'):
        assert (await client.delete('/api/jobs/' + state)).status_code == 409
    assert (await client.delete('/api/jobs/old-22')).status_code == 200
    assert (await client.delete('/api/installations/history')).json()['removed'] == 22
    assert (await client.get('/api/installations/history')).json()['total'] == 0
    assert controller.store.get('runtimes', runtime.id) and Path(runtime.path).exists()
    assert controller.store.get('jobs', 'model-download')


@pytest.mark.parametrize('cached', [False, True])
async def test_dependency_stages_continue_automatically_and_reject_duplicate_start(management, monkeypatch, cached):
    controller, client = management
    import modloader.backends as backends
    from modloader.backends import catalog
    base = {'id':'base', 'title':'Base', 'files':[{'id':'base-file','name':'engine.zip','size_bytes':12,'cached':False}]}
    dependency = {'id':'deps', 'title':'Dependencies', 'files':[{'id':'wheel','name':'package.whl','size_bytes':None,'cached':cached}]}
    ready, finish = asyncio.Event(), asyncio.Event()
    observed = []
    async def plan(*args, **kwargs):
        return {'id':'preview', 'backend':'llama_cpp', 'version':'b11609', 'variant':'cpu', 'options':{}, 'stage':base}
    async def installer(backend, paths, options, progress, cancel, pause=None, confirm=None, proxy=None):
        assert options['_plan']['stage'] == base
        await progress({'manifest':{**base, 'files':[{**base['files'][0], 'status':'completed'}]}})
        await confirm(dependency)
        observed.append(controller.store.jobs('install')[0]['state'])
        ready.set()
        await finish.wait()
        await confirm({'id':'build-deps', 'title':'Build dependencies', 'files':[]})
        observed.append(controller.store.jobs('install')[0]['state'])
        return RuntimeInstallation(id='installed', backend=backend, name='Installed', path='fixture', version='b11609')
    monkeypatch.setattr(catalog, 'plan_install', plan)
    monkeypatch.setattr(backends, 'install_backend', installer)
    assert (await client.post('/api/runtimes/install', json={'backend':'llama_cpp'})).status_code == 409
    planned = await client.post('/api/runtimes/plan', json={'backend':'llama_cpp','options':{'version':'b11609','variant':'cpu'}})
    assert planned.status_code == 200
    response = await client.post('/api/runtimes/install', json={'plan_id':'preview'})
    assert response.status_code == 200
    identifier = response.json()['id']
    await asyncio.wait_for(ready.wait(), 2)
    job = controller.require('jobs', identifier)
    assert job['state'] == 'running' and len(job['stages']) == 2
    assert not controller.stage_confirmations
    assert '_plan' not in job['request']['options']
    assert job['stages'][0]['files'][0]['status'] == 'completed'
    assert job['manifest']['files'][0]['size_bytes'] is None
    assert (await client.post('/api/runtimes/install', json={'plan_id':'preview'})).status_code == 409
    assert (await client.post(f'/api/jobs/{identifier}/confirm', json={'stage_id':'deps'})).status_code == 409
    finish.set()
    await asyncio.wait_for(controller.job_tasks[identifier], 2)
    assert (await client.post(f'/api/jobs/{identifier}/confirm', json={'stage_id':'deps'})).status_code == 409
    job = controller.require('jobs', identifier)
    assert observed == ['running', 'running'] and len(job['stages']) == 3
    assert job['state'] == 'completed' and job['created_at'] and job['started_at'] and job['finished_at']
    assert not (await client.get('/api/state')).json()['installations']


async def test_cancel_between_dependency_stages_never_installs(management, monkeypatch):
    controller, client = management
    import modloader.backends as backends
    ready = asyncio.Event()
    async def installer(backend, paths, options, progress, cancel, pause=None, confirm=None, proxy=None):
        await confirm({'id':'deps','title':'Dependencies','files':[]})
        ready.set()
        await cancel.wait()
        await confirm({'id':'next','title':'Next dependencies','files':[]})
        pytest.fail('cancelled stage must not install')
    monkeypatch.setattr(backends, 'install_backend', installer)
    job = controller.new_job('install', {'backend':'llama_cpp', 'options':{}})
    await asyncio.wait_for(ready.wait(), 2)
    assert controller.require('jobs', job['id'])['state'] == 'running'
    response = await client.post('/api/jobs/' + job['id'] + '/cancel')
    assert response.json()['state'] == 'cancelled'
    assert not controller.stage_confirmations and not controller.store.all('runtimes')
    assert (await client.post('/api/downloads/' + job['id'] + '/retry')).status_code == 409


async def test_legacy_waiting_stage_continues_only_with_current_manifest(management):
    controller, client = management
    identifier = 'legacy-waiting'
    job = {'id':identifier, 'kind':'install', 'state':'awaiting_confirmation'}
    event = asyncio.Event()
    controller.store.put('jobs', job)
    controller.job_records[identifier] = job
    controller.stage_confirmations[identifier] = ('deps', event)
    controller.cancel_events[identifier] = asyncio.Event()
    assert (await client.post(f'/api/jobs/{identifier}/confirm', json={'stage_id':'base'})).status_code == 409
    assert not event.is_set()
    assert (await client.post(f'/api/jobs/{identifier}/confirm', json={'stage_id':'deps'})).status_code == 200
    assert event.is_set() and job['state'] == 'running'
    assert (await client.post(f'/api/jobs/{identifier}/confirm', json={'stage_id':'deps'})).status_code == 409


async def test_retry_plan_only_reuses_server_manifest_for_unchanged_selection(management, monkeypatch):
    from modloader.backends import catalog
    controller, client = management
    stage = {'id': 'old-deps', 'title': 'Python 依赖', 'files': [{'id': 'wheel', 'name': 'pkg.whl'}]}
    original = {'version': '0.31.0', 'variant': 'wsl2'}
    previous = {'id': 'failed-install', 'kind': 'install', 'state': 'failed', 'backend': 'vllm', 'request': {'options': original}, 'stages': [stage]}
    controller.store.put('jobs', previous)
    async def plan(backend, paths, options, **kwargs):
        return {'id': 'retry-preview', 'backend': backend, **options, 'options': options, 'stage': {'id': 'base', 'files': []}}
    monkeypatch.setattr(catalog, 'plan_install', plan)
    for changes, reused in [({}, True), ({'version': '0.32.0'}, False), ({'variant': 'other'}, False), ({'python': 'other-python'}, False)]:
        result = await client.post('/api/runtimes/plan', json={'backend': 'vllm', 'options': {**original, **changes},
                                                              'retry_of': previous['id'], '_retry_packages': {'stage': {'files': ['untrusted']}}})
        assert result.status_code == 200, result.text
        assert '_retry_packages' not in result.json()
        saved = controller.install_plans['retry-preview']
        assert ('_retry_packages' in saved) == reused
        if reused:
            assert saved['_retry_packages']['stage'] == stage
    previous['state'] = 'completed'
    controller.store.put('jobs', previous)
    await client.post('/api/runtimes/plan', json={'backend': 'vllm', 'options': original, 'retry_of': previous['id']})
    assert '_retry_packages' not in controller.install_plans['retry-preview']


async def test_exit_interrupts_dependency_work_without_restarting_on_reopen(tmp_path, monkeypatch):
    import modloader.backends as backends
    ready = asyncio.Event()
    async def installer(backend, paths, options, progress, cancel, pause=None, confirm=None, proxy=None):
        await confirm({'id':'deps','title':'Dependencies','files':[]})
        ready.set()
        await asyncio.Event().wait()
        pytest.fail('shutdown must not continue dependency work')
    monkeypatch.setattr(backends, 'install_backend', installer)
    controller = Controller(tmp_path / 'restart-data', 'management-secret')
    job = controller.new_job('install', {'backend':'llama_cpp','options':{}})
    await asyncio.wait_for(ready.wait(), 2)
    await controller.close()
    reopened = Controller(tmp_path / 'restart-data', 'management-secret')
    try:
        assert reopened.require('jobs', job['id'])['state'] == 'interrupted'
        assert not reopened.job_tasks and not reopened.stage_confirmations
    finally:
        await reopened.close()


async def test_strata_switch_rebinds_programs_without_editing_external_config(management, tmp_path):
    controller, client = management
    roots = [tmp_path / name for name in ('old-strata','new-strata')]
    runtimes = []
    for index, root in enumerate(roots):
        (root / 'engine').mkdir(parents=True)
        (root / 'engine/strata.exe').write_text('fixture')
        (root / 'engine/strata-vision.exe').write_text('fixture')
        (root / 'libs').mkdir()
        write_expert_profile(root / 'data/expert-profile.bin')
        runtime = RuntimeInstallation(id=f'strata-{index}', name=root.name, backend='strata', path=str(root),
                                      metadata={'python':sys.executable, 'lib_dirs':[str(root / 'libs')]})
        controller.store.put('runtimes', runtime.model_dump())
        runtimes.append(runtime)
    pack = write_pack(tmp_path / 'pack')
    model_file = tmp_path / 'model.gguf'
    model_file.write_bytes(b'GGUF')
    projector = tmp_path / 'mmproj.gguf'
    projector.write_bytes(b'GGUF')
    model = ModelAsset(id='strata-model', name='Model', path=str(model_file), format='gguf',
                       metadata={'qwen4exp.block_count':48,'qwen4exp.expert_count':512})
    config_file = roots[0] / 'original.json'
    config_file.write_text(json.dumps({'cwd':str(roots[0]), 'args':['--native',str(model_file),'--pack',str(pack)],
                                      'tokenizer':str(pack / 'tokenizer'), 'lib_dirs':['libs'],
                                      'vision':{'exe':'engine/strata-vision.exe','mmproj':str(projector)}}))
    before = config_file.read_bytes()
    profile = LoadProfile(id='strata-profile', model_id=model.id, name='Strata profile', runtime_id=runtimes[0].id,
                          backend='strata', alias='strata', parameters={'configuration':str(config_file), 'context_length':8192},
                          generation={'temperature':0.2})
    controller.store.put('models', model.model_dump())
    controller.store.put('profiles', profile.model_dump())
    response = await client.post('/api/runtimes/' + runtimes[0].id + '/switch', json={'target_id':runtimes[1].id, 'profile_ids':[profile.id]})
    assert response.status_code == 200, response.text
    updated = controller.require('profiles', profile.id)
    config = updated['parameters']['strata_config']
    assert config['vision']['exe'] == str(roots[1] / 'engine/strata-vision.exe')
    assert config['lib_dirs'] == [str(roots[1] / 'libs')]
    assert str(pack) in config['args'] and config['tokenizer'] == str(pack / 'tokenizer')
    assert updated['parameters']['context_length'] == 8192 and updated['generation'] == profile.generation
    assert config_file.read_bytes() == before


async def test_deepseek_switch_keeps_pack_and_selects_new_dsv41_engine(management, tmp_path):
    from test_strata_deepseek_preparation import write_gguf, write_ds_pack
    controller, client = management
    roots = [tmp_path / name for name in ('old-dsv41', 'new-dsv41')]
    runtimes = []
    for index, root in enumerate(roots):
        engine = root / 'engine-dsv41' / f'0.1.49.{index}'
        engine.mkdir(parents=True)
        (engine / 'strata-dsv41.exe').write_bytes(b'not executed')
        runtime = RuntimeInstallation(id=f'dsv41-{index}', name=root.name, backend='strata', path=str(root),
                                      metadata={'python': sys.executable, 'strata_family': 'deepseek41', 'lib_dirs': [str(engine)]})
        controller.store.put('runtimes', runtime.model_dump())
        runtimes.append(runtime)
    source = write_gguf(tmp_path / 'model' / 'ds.gguf', hashes=True, embedded=True)
    pack = tmp_path / 'external-pack'
    write_ds_pack(pack, [source])
    model = ModelAsset(id='ds-model', name='DeepSeek', path=str(source), format='gguf', architecture='deepseek41')
    config_file = roots[0] / 'original.json'
    config_file.write_text(json.dumps({'arch': 'deepseek41', 'cwd': str(roots[0]),
                                      'args': ['--pack', str(pack), '--expert-profile', 'none'],
                                      'tokenizer': str(pack / 'tokenizer'), 'lib_dirs': runtimes[0].metadata['lib_dirs']}))
    before = config_file.read_bytes()
    profile = LoadProfile(id='ds-profile', model_id=model.id, name='DeepSeek profile', runtime_id=runtimes[0].id,
                          backend='strata', alias='ds', parameters={'configuration': str(config_file), 'context_length': 65536},
                          generation={'temperature': 0.2})
    controller.store.put('models', model.model_dump())
    controller.store.put('profiles', profile.model_dump())
    response = await client.post('/api/runtimes/' + runtimes[0].id + '/switch', json={'target_id': runtimes[1].id, 'profile_ids': [profile.id]})
    assert response.status_code == 200, response.text
    updated = controller.require('profiles', profile.id)
    cfg = updated['parameters']['strata_config']
    assert cfg['arch'] == 'deepseek41'
    assert cfg['exe'] == str(roots[1] / 'engine-dsv41/0.1.49.1/strata-dsv41.exe')
    assert cfg['lib_dirs'] == runtimes[1].metadata['lib_dirs']
    assert cfg['args'] == ['--pack', str(pack), '--expert-profile', 'none']
    assert updated['parameters']['context_length'] == 65536 and updated['generation'] == profile.generation
    assert config_file.read_bytes() == before
