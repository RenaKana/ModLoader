"""CPU-only integration coverage of Strata preparation through management and gateway APIs."""
import asyncio
import inspect
import json
from pathlib import Path
import sys

import httpx
import psutil
import pytest

from modloader.app import Controller, build_app
from modloader.models import Compatibility, LaunchSpec, LoadProfile, ModelAsset, RuntimeInstallation
from test_strata_preparation import write_pack, write_expert_profile


class PreparedFixtureAdapter:
    label = 'Prepared CPU fixture'

    def __init__(self):
        self.launches = []

    def parameters(self):
        return []

    def inspect(self, *args):
        return Compatibility(status='supported')

    def build_launch(self, model, profile, runtime, context):
        assert Path(profile.parameters['pack'], 'native_experts.txt').is_file()
        assert Path(profile.parameters['tokenizer'], 'tokenizer.json').is_file()
        config = {'model_name': profile.alias, 'tokenizer': profile.parameters['tokenizer'],
                  'args': ['--native', model.path, '--pack', profile.parameters['pack']]}
        Path(context.work_dir, 'strata.json').write_text(json.dumps(config), encoding='utf-8')
        if context.instance_id != 'preview':
            self.launches.append(profile.model_copy(deep=True))
        return LaunchSpec(argv=[sys.executable, str(Path(__file__).with_name('fake_backend.py')),
                                '--port', str(context.port)], cwd=str(Path(__file__).parent),
                          model_name='fixture', startup_timeout=15,
                          capabilities={'input': ['text'], 'output': ['text'], 'streaming': True, 'tools': False})


@pytest.fixture
async def lifecycle(tmp_path):
    controller = Controller(tmp_path / 'data', 'management-secret')
    adapter = PreparedFixtureAdapter()
    controller.adapters['strata'] = adapter
    model_dir = controller.paths.models / 'fixture-download'
    model_dir.mkdir()
    source = model_dir / 'fixture.gguf'
    source.write_bytes(b'CPU fixture source; never read as tensors')
    model = ModelAsset(id='model', name='Fixture', path=str(source), format='gguf', files=[str(source)],
                       metadata={'qwen4exp.block_count':48, 'qwen4exp.expert_count':512},
                       managed=True, source={'kind': 'download', 'managed_root': str(model_dir)})
    runtime_dir = tmp_path / 'runtime'
    (runtime_dir / 'tools').mkdir(parents=True)
    write_expert_profile(runtime_dir / 'data' / 'expert-profile.bin')
    # A gate allows tests to observe preparing without relying on timing or large files.
    script = '''import argparse, json, os, time
from pathlib import Path
''' + inspect.getsource(write_pack) + '''
parser = argparse.ArgumentParser()
parser.add_argument('--gguf', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
root = Path(__file__).parent.parent
with (root / 'starts.txt').open('a') as stream:
    stream.write('started\\n')
(root / 'pack.pid').write_text(str(os.getpid()))
print('waiting for fixture release', flush=True)
while not (root / 'release').exists():
    time.sleep(.02)
write_pack(Path(args.out))
print('fixture complete', flush=True)
'''
    (runtime_dir / 'tools' / 'iq_pack.py').write_text(script)
    runtime = RuntimeInstallation(id='runtime', backend='strata', name='CPU fixture', path=str(runtime_dir),
                                  metadata={'python': sys.executable})
    profile = LoadProfile(id='new', name='Independent', model_id=model.id, runtime_id=runtime.id,
                          backend='strata', alias='independent',
                          parameters={'context_length': 65536, 'kv_cache': 'q8', 'spec': 4, 'startup_timeout': 15},
                          generation={'temperature': .8})
    for kind, record in [('models', model), ('runtimes', runtime), ('profiles', profile)]:
        controller.store.put(kind, record.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as manager:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
            try:
                yield controller, adapter, model, profile, runtime, manager, public
            finally:
                await controller.close()


async def wait_until(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.02)


async def start_preparing(lifecycle):
    controller, adapter, model, profile, runtime, manager, public = lifecycle
    response = await asyncio.wait_for(manager.post(f'/api/profiles/{profile.id}/start'), 2)
    assert response.status_code == 200, response.text
    assert response.json()['state'] == 'preparing'
    instance = controller.runtime.instances[response.json()['id']]
    assert instance.id not in controller.runtime.specs
    assert adapter.launches == []
    await wait_until(lambda: (Path(runtime.path) / 'pack.pid').is_file())
    return instance


async def test_preflight_and_missing_preview_do_not_prepare_or_persist(lifecycle):
    controller, adapter, model, profile, runtime, manager, public = lifecycle
    before = controller.store.all('profiles')
    preflight = await manager.post('/api/profiles/preflight', json=profile.model_dump())
    assert preflight.status_code == 200
    assert preflight.json()['status'] == 'needs_preparation'
    preview = await manager.post(f'/api/profiles/{profile.id}/preview')
    assert preview.status_code == 200
    assert preview.json()['preparation']['status'] == 'needs_preparation'
    assert preview.json()['argv'] == []
    assert not controller.preparation.root.exists()
    assert not (Path(runtime.path) / 'starts.txt').exists()
    assert controller.store.all(controller.preparation.KIND) == []
    assert controller.store.all('profiles') == before
    assert list(controller.paths.work.iterdir()) == []
    assert adapter.launches == []


async def test_preparing_completion_gateway_and_owned_deletion(lifecycle, tmp_path):
    controller, adapter, model, profile, runtime, manager, public = lifecycle
    before = controller.store.get('profiles', profile.id)
    instance = await start_preparing(lifecycle)
    await wait_until(lambda: instance.message == 'waiting for fixture release')
    assert instance.phase == 'preparing'
    assert (await public.get('/v1/models')).json()['data'] == []
    response = await public.post('/v1/chat/completions', json={'model': profile.alias, 'messages': []})
    assert response.status_code == 409
    (Path(runtime.path) / 'release').touch()
    await wait_until(lambda: instance.state not in ('preparing', 'starting'))
    assert instance.state == 'ready', instance.error
    assert instance.phase == 'ready' and instance.message == '模型已就绪'
    assert '正在检查 Strata pack 与 tokenizer' in controller.runtime.logs[instance.id]
    assert {row['id'] for row in (await public.get('/v1/models')).json()['data']} == {profile.alias}
    chat = await public.post('/v1/chat/completions', json={'model': profile.alias, 'messages': [{'role': 'user', 'content': 'hi'}]})
    assert chat.status_code == 200 and chat.json()['model'] == profile.alias
    assert len(adapter.launches) == 1
    resolved = adapter.launches[0]
    assert {k: v for k, v in resolved.parameters.items() if k not in ('pack', 'tokenizer', 'expert_profile')} == profile.parameters
    assert resolved.parameters['expert_profile'] == str(Path(runtime.path) / 'data/expert-profile.bin')
    assert resolved.generation == profile.generation and resolved.extra_args == profile.extra_args
    assert controller.store.get('profiles', profile.id) == before
    assert (await manager.post(f'/api/instances/{instance.id}/stop', json={})).status_code == 200

    # Completed resources are reused without running the converter a second time.
    second = await manager.post(f'/api/profiles/{profile.id}/start')
    assert second.status_code == 200 and second.json()['state'] == 'starting'
    second_instance = controller.runtime.instances[second.json()['id']]
    await wait_until(lambda: second_instance.state != 'starting')
    assert second_instance.state == 'ready', second_instance.error
    assert (Path(runtime.path) / 'starts.txt').read_text().splitlines() == ['started']
    await manager.post(f'/api/instances/{second_instance.id}/stop', json={})

    # A second profile may point outside the app; it does not make that pack app-owned.
    external = write_pack(tmp_path / 'external-pack')
    old = profile.model_copy(deep=True)
    old.id, old.alias = 'external-profile', 'external'
    old.parameters.update(pack=str(external), tokenizer=str(external / 'tokenizer'), context_length=123)
    controller.store.put('profiles', old.model_dump())
    preview = await manager.post(f'/api/models/{model.id}/delete-preview')
    assert preview.status_code == 200
    listed = set(preview.json()['files'])
    owned = set(controller.preparation.owned_files(model))
    assert listed == owned | {model.path}
    assert not any(Path(p).is_relative_to(external) for p in listed)
    assert controller.store.get('profiles', old.id) == old.model_dump()
    assert (await manager.delete(f'/api/models/{model.id}?delete_files=true')).status_code == 200
    assert not Path(model.path).exists()
    assert all(not Path(p).exists() for p in owned)
    assert (external / 'native_experts.txt').is_file()
    assert controller.store.all(controller.preparation.KIND) == []


async def test_cancel_blocks_duplicate_preparation_and_model_profile_deletion(lifecycle):
    controller, adapter, model, profile, runtime, manager, public = lifecycle
    instance = await start_preparing(lifecycle)
    pid = int((Path(runtime.path) / 'pack.pid').read_text())
    process = psutil.Process(pid)
    other = profile.model_copy(update={'id': 'other', 'alias': 'other'})
    controller.store.put('profiles', other.model_dump())
    duplicate = await manager.post('/api/profiles/other/start')
    assert duplicate.status_code == 400 and '准备' in duplicate.text
    assert len(controller.runtime.instances) == 1
    assert (Path(runtime.path) / 'starts.txt').read_text().splitlines() == ['started']
    assert (await manager.delete(f'/api/profiles/{profile.id}')).status_code == 409
    assert (await manager.delete(f'/api/models/{model.id}')).status_code == 409
    assert (await manager.delete(f'/api/models/{model.id}?delete_files=true')).status_code == 409
    stopped = await manager.post(f'/api/instances/{instance.id}/stop', json={})
    assert stopped.status_code == 200 and stopped.json()['state'] == 'stopped'
    await asyncio.to_thread(process.wait, 5)
    assert not process.is_running()
    assert adapter.launches == [] and instance.id not in controller.runtime.processes
    assert controller.store.all(controller.preparation.KIND) == []
    assert list(controller.preparation.root.iterdir()) == []
    assert Path(model.path).exists()
    assert (await manager.delete(f'/api/profiles/{profile.id}')).status_code == 200


async def test_preparation_startup_timeout_fails_and_cleans_without_launch(lifecycle):
    controller, adapter, model, profile, runtime, manager, public = lifecycle
    profile.parameters['startup_timeout'] = 1
    controller.store.put('profiles', profile.model_dump())
    instance = await start_preparing(lifecycle)
    pid = int((Path(runtime.path) / 'pack.pid').read_text())
    process = psutil.Process(pid)
    await wait_until(lambda: instance.state != 'preparing', timeout=5)
    assert instance.state == 'failed' and '超时' in instance.error
    assert instance.phase == 'failed' and instance.message == instance.error
    await asyncio.to_thread(process.wait, 5)
    assert not process.is_running()
    assert adapter.launches == []
    assert controller.store.all(controller.preparation.KIND) == []
    assert list(controller.preparation.root.iterdir()) == []
    assert (await public.get('/v1/models')).json()['data'] == []


async def test_existing_resources_preview_and_load_preserve_profile_independence(lifecycle, tmp_path):
    controller, adapter, model, profile, runtime, manager, public = lifecycle
    pack = write_pack(tmp_path / 'imported-pack')
    old = profile.model_copy(deep=True)
    old.id, old.alias, old.name = 'old', 'old', 'Imported'
    old.parameters = {'strata_config': {'args': ['--native', model.path, '--pack', str(pack)],
                                      'tokenizer': str(pack / 'tokenizer')},
                      'context_length': 1000, 'kv_cache': 'q4', 'spec': 12}
    old.generation = {'temperature': .1}
    controller.store.put('profiles', old.model_dump())
    before = controller.store.all('profiles')
    preflight = await manager.post('/api/profiles/preflight', json=profile.model_dump())
    assert preflight.status_code == 200 and preflight.json()['status'] == 'ready'
    assert preflight.json()['source']['profile_id'] == old.id
    preview = await manager.post(f'/api/profiles/{profile.id}/preview')
    assert preview.status_code == 200 and preview.json()['argv'], preview.text
    assert preview.json()['config'] == {'model_name': profile.alias, 'tokenizer': str(pack / 'tokenizer'),
                                      'args': ['--native', model.path, '--pack', str(pack)]}
    assert not controller.preparation.root.exists()
    assert list(controller.paths.work.iterdir()) == []
    started = await manager.post(f'/api/profiles/{profile.id}/start')
    assert started.status_code == 200 and started.json()['state'] == 'starting'
    instance = controller.runtime.instances[started.json()['id']]
    await wait_until(lambda: instance.state != 'starting')
    assert instance.state == 'ready', instance.error
    resolved = adapter.launches[0]
    assert resolved.parameters == {**profile.parameters, 'pack': str(pack), 'tokenizer': str(pack / 'tokenizer'), 'expert_profile':str(Path(runtime.path) / 'data/expert-profile.bin')}
    assert resolved.generation == profile.generation
    assert controller.store.all('profiles') == before
    assert not (Path(runtime.path) / 'starts.txt').exists()
    assert controller.store.all(controller.preparation.KIND) == []
