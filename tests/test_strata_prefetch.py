import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

from modloader.backends import get_adapters, import_runtime
from modloader.models import RuntimeInstallation, LoadProfile, ModelAsset, LaunchContext
from test_runtime_management import management
from test_strata_preparation import write_expert_profile, write_pack


IDENTITY = {'version': '0.1.48', 'commit': '6f8800aefe27', 'backend': 'cuda', 'build_id': '6f8800aefe27+cuda13.0'}


def runtime_fixture(root, identifier='strata', managed=False, paths=None, optimized=False):
    (root / 'engine').mkdir(parents=True)
    (root / 'serve').mkdir()
    (root / 'serve/server.py').write_text('# never executed')
    (root / 'engine/strata.exe').write_text('official engine sentinel')
    (root / 'engine/BUILD.json').write_text(json.dumps({'version': IDENTITY['version'], 'commit': IDENTITY['commit'], 'cuda': '13.0'}))
    (root / 'engine/engine-component.json').write_text(json.dumps({'version': IDENTITY['version'], 'backend': 'cuda', 'build_id': IDENTITY['build_id']}))
    write_expert_profile(root / 'data/expert-profile.bin')
    runtime = RuntimeInstallation(id=identifier, name=identifier, backend='strata', path=str(root), managed=managed,
                                  metadata={'python': sys.executable, 'provenance': {'source_commit': 'service-source-not-engine'}})
    if managed:
        runtime.metadata.update(managed_root=str(root), install_root=str(root))
        (root / 'modloader-runtime.json').write_text(runtime.model_dump_json())
    if optimized:
        sidecar_fixture(root / 'engine')
    return runtime


def sidecar_fixture(directory, **changes):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'strata-startup-prefetch.exe').write_text('optimized engine sentinel')
    value = dict(schema_version=1, feature='startup_expert_prefetch', engine_identity=IDENTITY,
                 source_commit=IDENTITY['commit'] + '0' * 28, executable='strata-startup-prefetch.exe', patch_revision='prefetch-v1')
    value.update(changes)
    (directory / 'startup-prefetch.json').write_text(json.dumps(value))


def profile_fixture(tmp_path, runtime, enabled=True):
    model_file = tmp_path / 'model.gguf'
    model_file.write_bytes(b'GGUF')
    pack = write_pack(tmp_path / 'pack')
    model = ModelAsset(id='model', name='Model', path=str(model_file), format='gguf',
                       metadata={'qwen4exp.block_count': 48, 'qwen4exp.expert_count': 512})
    parameters = {'pack': str(pack), 'tokenizer': str(pack / 'tokenizer'), 'context_length': 8192}
    if enabled is not None:
        parameters['startup_expert_prefetch'] = enabled
    profile = LoadProfile(id='profile', model_id=model.id, name='Profile', backend='strata', runtime_id=runtime.id,
                          alias='model', parameters=parameters)
    return model, profile


def test_managed_prefetch_selects_same_engine_sidecar_without_changing_official(tmp_path, monkeypatch):
    monkeypatch.delenv('MODLOADER_STRATA_DEV_RUNTIME', raising=False)
    runtime = runtime_fixture(tmp_path / 'runtime', optimized=True)
    model, profile = profile_fixture(tmp_path, runtime)
    context = LaunchContext(port=12345, instance_id='test', work_dir=str(tmp_path / 'work'), data_dir=str(tmp_path))
    get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text())
    assert cfg['exe'] == str(Path(runtime.path) / 'engine/strata-startup-prefetch.exe')
    assert cfg['args'].count('--startup-expert-prefetch') == 1
    assert (Path(runtime.path) / 'engine/strata.exe').read_text() == 'official engine sentinel'


def test_explicit_false_overrides_legacy_environment_and_argument(tmp_path, monkeypatch):
    runtime = runtime_fixture(tmp_path / 'runtime')
    model, profile = profile_fixture(tmp_path, runtime, False)
    monkeypatch.setenv('MODLOADER_STRATA_DEV_RUNTIME', runtime.path)
    monkeypatch.setenv('MODLOADER_STRATA_DEV_EXE', str(tmp_path / 'missing-legacy.exe'))
    profile.extra_args = ['--startup-expert-prefetch']
    context = LaunchContext(port=12345, instance_id='test', work_dir=str(tmp_path / 'work'), data_dir=str(tmp_path))
    get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text())
    assert cfg['exe'].endswith('engine\\strata.exe') or cfg['exe'].endswith('engine/strata.exe')
    assert '--startup-expert-prefetch' not in cfg['args']


@pytest.mark.parametrize('change', [{'engine_identity': {**IDENTITY, 'commit': 'old-commit'}}, {'executable': '../other.exe'}, {'source_commit': 'unrelated'}])
def test_prefetch_rejects_stale_or_escaped_sidecar(tmp_path, change):
    runtime = runtime_fixture(tmp_path / 'runtime')
    sidecar_fixture(Path(runtime.path) / 'engine', **change)
    model, profile = profile_fixture(tmp_path, runtime)
    from modloader.parameter_rules import validate_profile
    with pytest.raises(ValueError, match='预取'):
        validate_profile(model, profile, runtime, get_adapters()['strata'])


@pytest.mark.parametrize('args', [['--no-mmap'], ['--expert-cache', '0'], ['--expert-cache', 'layer:4']])
def test_prefetch_preserves_legal_non_global_cache_and_nonmapped_arguments(tmp_path, args):
    runtime = runtime_fixture(tmp_path / 'runtime', optimized=True)
    model, profile = profile_fixture(tmp_path, runtime)
    profile.extra_args = args
    from modloader.parameter_rules import validate_profile
    validate_profile(model, profile, runtime, get_adapters()['strata'])
    context = LaunchContext(port=12345, instance_id='test', work_dir=str(tmp_path / 'work'), data_dir=str(tmp_path))
    get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text())
    assert '--startup-expert-prefetch' in cfg['args']
    assert all(arg in cfg['args'] for arg in args)


async def test_legacy_prefetch_switch_requires_explicit_atomic_disable(management, tmp_path, monkeypatch):
    controller, client = management
    source = runtime_fixture(tmp_path / 'source', 'source')
    target = runtime_fixture(tmp_path / 'target', 'target')
    model, profile = profile_fixture(tmp_path, source, None)
    dev = tmp_path / 'legacy.exe'
    dev.write_text('legacy')
    monkeypatch.setenv('MODLOADER_STRATA_DEV_RUNTIME', source.path)
    monkeypatch.setenv('MODLOADER_STRATA_DEV_EXE', str(dev))
    for runtime in (source, target):
        controller.store.put('runtimes', runtime.model_dump())
    controller.store.put('models', model.model_dump())
    controller.store.put('profiles', profile.model_dump())
    before = controller.require('profiles', profile.id)
    body = {'target_id': target.id, 'profile_ids': [profile.id]}
    preview = await client.post('/api/runtimes/source/switch-preview', json=body)
    assert preview.status_code == 200, preview.text
    assert preview.json()['startup_expert_prefetch']['status'] == 'choice_required'
    assert (await client.post('/api/runtimes/source/switch', json=body)).status_code == 400
    assert (await client.delete('/api/runtimes/source', params={'replacement_id': target.id})).status_code == 400
    assert controller.require('profiles', profile.id) == before
    body['disable_startup_expert_prefetch'] = True
    preview = await client.post('/api/runtimes/source/switch-preview', json=body)
    assert preview.json()['startup_expert_prefetch']['status'] == 'disabled'
    assert controller.require('profiles', profile.id) == before
    result = await client.post('/api/runtimes/source/switch', json=body)
    assert result.status_code == 200, result.text
    saved = controller.require('profiles', profile.id)
    assert saved['runtime_id'] == target.id and saved['parameters']['startup_expert_prefetch'] is False


@pytest.mark.parametrize('proxy', ['http://fixture-user:fixture-password@proxy.invalid:17890', ''])
async def test_prepare_runtime_job_publishes_valid_sidecar_without_loading_model(management, monkeypatch, proxy):
    controller, client = management
    runtime = runtime_fixture(controller.paths.runtimes / 'prefetch', managed=True)
    controller.store.put('runtimes', runtime.model_dump())
    from modloader.backends import installers
    script = controller.paths.work / 'build-fixture.ps1'
    script.write_text('# subprocess is stubbed')
    monkeypatch.setattr('modloader.strata_prefetch.build_script', lambda: script)
    monkeypatch.setattr(controller, 'proxy', lambda: proxy)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY'):
        monkeypatch.setenv(key, 'inherited-launcher-setting')
    parent_environment = dict(os.environ)
    calls = []
    async def command(argv, progress, cancel, **kwargs):
        calls.append(argv)
        assert kwargs['env']['HTTP_PROXY'] == kwargs['env']['HTTPS_PROXY'] == proxy
        assert kwargs['env']['ALL_PROXY'] == kwargs['env']['NO_PROXY'] == ''
        assert proxy not in argv
        sidecar_fixture(Path(argv[argv.index('-OutputDirectory') + 1]))
        return ''
    monkeypatch.setattr(installers, '_command', command)
    response = await client.post('/api/runtimes/strata/prepare-startup-prefetch')
    assert response.status_code == 200, response.text
    job = response.json()
    await controller.job_tasks[job['id']]
    result = controller.require('jobs', job['id'])
    assert result['state'] == 'completed', result
    assert result['operation'] == 'prepare_startup_prefetch'
    assert len(calls) == 1
    state = (await client.get('/api/state')).json()
    assert state['runtimes'][0]['startup_expert_prefetch']['available'] is True
    saved = controller.require('runtimes', runtime.id)
    assert saved['metadata']['engine_identity'] == IDENTITY
    assert saved['metadata']['provenance'] == runtime.metadata['provenance']
    assert not controller.runtime.instances
    assert dict(os.environ) == parent_environment
    assert 'fixture-password' not in json.dumps(result)


async def test_preflight_and_save_reject_enabled_missing_prefetch(management, tmp_path):
    controller, client = management
    runtime = runtime_fixture(tmp_path / 'runtime')
    model, profile = profile_fixture(tmp_path, runtime)
    controller.store.put('runtimes', runtime.model_dump())
    controller.store.put('models', model.model_dump())
    response = await client.post('/api/profiles/preflight', json=profile.model_dump())
    assert response.status_code == 200, response.text
    assert response.json()['status'] == 'error'
    assert response.json()['startup_expert_prefetch']['enabled'] is True
    assert (await client.post('/api/profiles', json=profile.model_dump())).status_code == 400


def test_import_identity_comes_from_engine_not_service_source(tmp_path, monkeypatch):
    runtime = runtime_fixture(tmp_path / 'runtime', optimized=True)
    monkeypatch.setattr('modloader.backends.adapters._run_readonly', lambda *a, **kw: pytest.fail('external import must not execute'))
    imported = import_runtime('strata', runtime.path, runtime.metadata)
    assert imported.metadata['engine_identity'] == IDENTITY
    assert imported.metadata['provenance'] == runtime.metadata['provenance']
    assert imported.metadata['startup_expert_prefetch']['source_commit'].startswith(IDENTITY['commit'])


@pytest.mark.parametrize('failure', ['build', 'cancel', 'wrong_identity', 'publication'])
async def test_prepare_failure_keeps_previous_engine_pair(tmp_path, monkeypatch, failure):
    from modloader.models import AppPaths
    from modloader.strata_prefetch import prepare_runtime
    from modloader.backends import installers
    paths = AppPaths(tmp_path / 'data')
    runtime = runtime_fixture(paths.runtimes / 'runtime', managed=True)
    engine = Path(runtime.path) / 'engine'
    sidecar_fixture(engine, engine_identity={**IDENTITY, 'commit': 'old'})
    before = {name: (engine / name).read_bytes() for name in ('strata.exe', 'strata-startup-prefetch.exe', 'startup-prefetch.json')}
    cancel = asyncio.Event()
    script = paths.work / 'fixture.ps1'
    script.touch()
    monkeypatch.setattr('modloader.strata_prefetch.build_script', lambda: script)
    async def command(argv, *args, **kwargs):
        output = Path(argv[argv.index('-OutputDirectory') + 1])
        if failure == 'build':
            raise RuntimeError('fixture compiler failure')
        sidecar_fixture(output, **({'engine_identity': {**IDENTITY, 'version': 'other'}} if failure == 'wrong_identity' else {}))
        if failure == 'cancel':
            cancel.set()
    monkeypatch.setattr(installers, '_command', command)
    if failure == 'publication':
        import os
        original_replace = os.replace
        def replace(source, target):
            if Path(source).name == 'startup-prefetch.json':
                raise OSError('fixture publication denied')
            return original_replace(source, target)
        monkeypatch.setattr('modloader.strata_prefetch.os.replace', replace)
    with pytest.raises((RuntimeError, ValueError, OSError, asyncio.CancelledError)):
        await prepare_runtime(runtime, paths, None, cancel)
    assert {name: (engine / name).read_bytes() for name in before} == before
    assert runtime.metadata.get('startup_expert_prefetch') is None


async def test_switch_rechecks_target_after_successful_preview(management, tmp_path):
    controller, client = management
    source = runtime_fixture(tmp_path / 'source', 'source', optimized=True)
    target = runtime_fixture(tmp_path / 'target', 'target', optimized=True)
    model, profile = profile_fixture(tmp_path, source)
    for runtime in (source, target):
        controller.store.put('runtimes', runtime.model_dump())
    controller.store.put('models', model.model_dump())
    controller.store.put('profiles', profile.model_dump())
    body = {'target_id': target.id, 'profile_ids': [profile.id]}
    preview = await client.post('/api/runtimes/source/switch-preview', json=body)
    assert preview.status_code == 200, preview.text
    assert preview.json()['startup_expert_prefetch']['status'] == 'preserved'
    (Path(target.path) / 'engine/BUILD.json').write_text(json.dumps({'version': '0.1.49', 'commit': 'new'}))
    apply = await client.post('/api/runtimes/source/switch', json=body)
    assert apply.status_code == 400
    assert controller.require('profiles', profile.id)['runtime_id'] == source.id


async def test_legacy_switch_to_ready_target_persists_enabled_without_old_exe(management, tmp_path, monkeypatch):
    controller, client = management
    source = runtime_fixture(tmp_path / 'source', 'source')
    target = runtime_fixture(tmp_path / 'target', 'target', optimized=True)
    model, profile = profile_fixture(tmp_path, source, None)
    dev = tmp_path / 'legacy.exe'
    dev.write_text('legacy')
    monkeypatch.setenv('MODLOADER_STRATA_DEV_RUNTIME', source.path)
    monkeypatch.setenv('MODLOADER_STRATA_DEV_EXE', str(dev))
    for runtime in (source, target):
        controller.store.put('runtimes', runtime.model_dump())
    controller.store.put('models', model.model_dump())
    controller.store.put('profiles', profile.model_dump())
    response = await client.post('/api/runtimes/source/switch', json={'target_id': target.id, 'profile_ids': [profile.id]})
    assert response.status_code == 200, response.text
    saved = controller.require('profiles', profile.id)
    assert saved['parameters']['startup_expert_prefetch'] is True
    assert str(dev) not in json.dumps(saved)


@pytest.mark.parametrize('enabled,failure', [(False, False), (True, False), (True, True)])
async def test_install_prepares_only_selected_optimization_before_registration(tmp_path, monkeypatch, enabled, failure):
    from modloader.models import AppPaths
    from modloader.backends import installers
    from modloader.strata_prefetch import capability
    paths = AppPaths(tmp_path / 'data')
    calls = []
    class Transfers:
        def __init__(self, *args): pass
        async def add(self, stage): return stage
        async def fetch(self, stage): return {'source': Path('source.zip'), 'engine': Path('engine.tar.gz')}
    async def unpack(artifact, root, *args):
        if artifact.name == 'source.zip':
            runtime_fixture(root / 'Strata-fixture')
    async def venv(*args): return sys.executable
    async def packages(*args, **kwargs): return []
    async def validate(runtime, **kwargs): return runtime
    async def command(argv, *args, **kwargs):
        calls.append(argv)
        assert kwargs['env']['HTTP_PROXY'] == kwargs['env']['HTTPS_PROXY'] == 'http://fixture-proxy.invalid:17890'
        if failure:
            raise RuntimeError('fixture build unavailable')
        sidecar_fixture(Path(argv[argv.index('-OutputDirectory') + 1]))
    script = paths.work / 'fixture.ps1'
    script.touch()
    monkeypatch.setattr('modloader.strata_prefetch.build_script', lambda: script)
    monkeypatch.setattr(installers, '_Transfers', Transfers)
    monkeypatch.setattr(installers, '_unpack', unpack)
    monkeypatch.setattr(installers, '_verify_source', lambda *args: None)
    monkeypatch.setattr(installers, '_approved_venv', venv)
    monkeypatch.setattr(installers, '_package_stage', packages)
    monkeypatch.setattr(installers, 'validate_runtime', validate)
    monkeypatch.setattr(installers, '_command', command)
    plan = dict(id='plan', backend='strata', version='0.2.4', variant='cuda13', stage={},
                provenance={'source_commit': 'service-source', 'source_tree': []},
                options={'startup_expert_prefetch': enabled, 'python': sys.executable})
    if failure:
        with pytest.raises(RuntimeError, match='fixture build unavailable'):
            await installers.install_backend('strata', paths, {'_plan': plan}, None, asyncio.Event(), proxy='http://fixture-proxy.invalid:17890')
        assert not list(paths.runtimes.glob('*/modloader-runtime.json'))
    else:
        runtime = await installers.install_backend('strata', paths, {'_plan': plan}, None, asyncio.Event(), proxy='http://fixture-proxy.invalid:17890')
        assert capability(runtime)['available'] is enabled
        assert (Path(runtime.metadata['managed_root']) / 'modloader-runtime.json').is_file()
        assert runtime.metadata['source_commit'] == 'service-source'
    assert len(calls) == int(enabled)


async def test_catalog_keeps_opted_in_prefetch_in_plan(tmp_path, monkeypatch):
    from modloader.backends import catalog
    from modloader.models import AppPaths
    from test_strata_installation import release_asset
    async def metadata(url, proxy=None):
        if '/releases/tags/' in url:
            return {'id': 1, 'assets': [release_asset('strata-engine-0.1.48-windows-x86_64-cuda-cuda13.tar.gz')]}
        if '/git/ref/tags/' in url:
            return {'object': {'type': 'commit', 'sha': '9' * 40}}
        return {'tree': [], 'truncated': False}
    monkeypatch.setattr(catalog, '_json', metadata)
    plan = await catalog.plan_install('strata', AppPaths(tmp_path / 'data'),
                                      {'version': '0.2.4', 'python': sys.executable, 'startup_expert_prefetch': True})
    assert plan['options']['startup_expert_prefetch'] is True


async def test_active_prepare_blocks_duplicate_and_removal_then_cancel_preserves_runtime(management, monkeypatch):
    controller, client = management
    runtime = runtime_fixture(controller.paths.runtimes / 'runtime', managed=True)
    controller.store.put('runtimes', runtime.model_dump())
    before = controller.require('runtimes', runtime.id)
    started = asyncio.Event()
    script = controller.paths.work / 'fixture.ps1'
    script.touch()
    monkeypatch.setattr('modloader.strata_prefetch.build_script', lambda: script)
    async def command(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr('modloader.backends.installers._command', command)
    response = await client.post('/api/runtimes/strata/prepare-startup-prefetch')
    assert response.status_code == 200, response.text
    job = response.json()
    await asyncio.wait_for(started.wait(), 2)
    assert (await client.post('/api/runtimes/strata/prepare-startup-prefetch')).status_code == 409
    assert (await client.delete('/api/runtimes/strata')).status_code == 409
    result = await client.post('/api/jobs/' + job['id'] + '/cancel')
    assert result.status_code == 200, result.text
    assert result.json()['state'] == 'cancelled'
    assert controller.require('runtimes', runtime.id) == before
    assert not (Path(runtime.path) / 'engine/startup-prefetch.json').exists()
    assert not list(controller.paths.work.glob('strata-prefetch-*'))


async def test_update_plan_and_job_keep_prefetch_choice_without_switching_source(management, monkeypatch):
    controller, client = management
    runtime = runtime_fixture(controller.paths.runtimes / 'runtime', managed=True)
    controller.store.put('runtimes', runtime.model_dump())
    before = controller.require('runtimes', runtime.id)
    async def plan(backend, paths, options, proxy=None):
        return dict(id='plan', backend=backend, version='0.2.5', variant='cuda13', options={}, stage={})
    observed = []
    async def install(backend, paths, options, *args, **kwargs):
        observed.append(options)
        raise RuntimeError('fixture failed build; original retained')
    monkeypatch.setattr('modloader.backends.catalog.plan_install', plan)
    monkeypatch.setattr('modloader.backends.install_backend', install)
    response = await client.post('/api/runtimes/plan', json={'backend': 'strata', 'update_from': runtime.id,
                                                           'options': {'version': '0.2.5', 'startup_expert_prefetch': True}})
    assert response.status_code == 200, response.text
    assert response.json()['options']['startup_expert_prefetch'] is True
    response = await client.post('/api/runtimes/install', json={'plan_id': 'plan'})
    assert response.status_code == 200, response.text
    job = response.json()
    await controller.job_tasks[job['id']]
    assert observed[0]['_plan']['options']['startup_expert_prefetch'] is True
    assert controller.require('jobs', job['id'])['state'] == 'failed'
    assert controller.require('runtimes', runtime.id) == before
    assert len(controller.store.all('runtimes')) == 1


def test_ordinary_external_import_keeps_working_with_unusable_optional_component_metadata(tmp_path):
    runtime = runtime_fixture(tmp_path / 'runtime')
    (Path(runtime.path) / 'engine/engine-component.json').write_text('not valid JSON')
    imported = import_runtime('strata', runtime.path, runtime.metadata)
    assert imported.version == IDENTITY['version']
    assert imported.metadata['engine_identity'] == {}
