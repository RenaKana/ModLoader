import asyncio
import json
from pathlib import Path
import struct
import sys
import uuid
import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.assets import inspect_asset
from modloader.models import ModelAsset,LoadProfile,RuntimeInstallation,RuntimeInstance,LaunchSpec,Compatibility
from modloader.secrets import Vault


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows system proxy settings')
@pytest.mark.parametrize('system_proxy', ['', 'http://127.0.0.1:7890'])
def test_system_proxy_ignores_launcher_environment(controller, monkeypatch, system_proxy):
    import urllib.request
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:17890')
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:17890')
    monkeypatch.setattr(urllib.request, 'getproxies_registry', lambda: {'https': system_proxy})
    controller.settings['proxy_mode'] = 'system'
    assert controller.proxy() == system_proxy
    controller.settings.update(proxy_mode='manual', proxy_url='http://127.0.0.1:23456')
    assert controller.proxy() == 'http://127.0.0.1:23456'
    controller.settings['proxy_mode'] = 'direct'
    assert controller.proxy() == ''


def gguf(path):
    path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0))


def test_download_groups_register_each_model_and_associate_root_projector(controller):
    directory = controller.paths.models / 'multi-download'
    for folder in ('Q3', 'Q4'):
        (directory / folder).mkdir(parents=True)
        gguf(directory / folder / 'model.gguf')
    gguf(directory / 'mmproj.gguf')
    gguf(directory / 'mtp.gguf')
    groups = [{'id': folder, 'name': folder, 'format': 'gguf', 'files': [folder + '/model.gguf']} for folder in ('Q3', 'Q4')]
    request = {'source': 'hf', 'repo_id': 'fixture/models', 'destination': str(directory)}
    result = {'model_groups': groups, 'revision': 'fixed',
              'vision_groups': [{'id': 'vision', 'files': ['mmproj.gguf']}],
              'vision_assignments': {'Q3': ['vision'], 'Q4': ['vision']}}
    ids, errors = controller.register_download(request, result)
    assert len(ids) == 2 and not errors
    for identifier in ids:
        model = controller.store.get('models', identifier)
        assert model['metadata']['mmproj_candidates'] == [str(directory / 'mmproj.gguf')]
        assert model['source']['revision'] == 'fixed'
        assert len(model['files']) == 1
    repeated, errors = controller.register_download(request, result)
    assert repeated == ids and not errors
    assert controller.register_download(request, {'model_groups': [], 'revision': 'fixed'}) == ([], [])
    assert len(controller.store.all('models')) == 2


def test_download_partial_registration_retains_success_and_reports_failure(controller):
    directory = controller.paths.models / 'partial-download'
    directory.mkdir()
    gguf(directory / 'good.gguf')
    ids, errors = controller.register_download({'source': 'hf', 'destination': str(directory)}, {
        'model_groups': [{'id': 'ok', 'name': 'ok', 'format': 'gguf', 'files': ['good.gguf']},
                         {'id': 'missing', 'name': 'missing', 'format': 'gguf', 'files': ['missing.gguf']}]})
    assert len(ids) == 1 and len(errors) == 1
    assert errors[0]['path'] == 'missing'


def test_missing_gguf_shard_and_external_reference(tmp_path):
    model=tmp_path/'model-00001-of-00002.gguf'
    gguf(model)
    with pytest.raises(ValueError,match='缺少分片'): inspect_asset(str(model))
    gguf(tmp_path/'model-00002-of-00002.gguf')
    asset=inspect_asset(str(model))
    assert len(asset.files)==2 and asset.path==str(model) and not asset.managed


def test_hf_incomplete_shards(tmp_path):
    (tmp_path/'config.json').write_text('{"model_type":"qwen3"}')
    (tmp_path/'model-1.safetensors').write_bytes(b'fixture')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'model-2.safetensors'}}))
    with pytest.raises(ValueError,match='分片'): inspect_asset(str(tmp_path))


def test_hf_named_shards_without_index_must_be_complete(tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"qwen2_5_vl","vision_config":{}}')
    (tmp_path / 'tokenizer.json').write_text('{}')
    (tmp_path / 'model-00001-of-00002.safetensors').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='分片不完整'):
        inspect_asset(str(tmp_path))
    (tmp_path / 'model-00002-of-00002.safetensors').write_bytes(b'fixture')
    assert len(inspect_asset(str(tmp_path)).files) == 4


def test_projector_candidates_do_not_accept_hf_tower_or_json(tmp_path):
    from modloader.assets import projector_candidates
    (tmp_path / 'IQ3').mkdir()
    gguf(tmp_path / 'IQ3/model.gguf')
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'vision.safetensors').write_bytes(b'fixture')
    gguf(tmp_path / 'mmproj.gguf')
    model = inspect_asset(str(tmp_path / 'IQ3/model.gguf'), source={'managed_root':str(tmp_path)})
    model.metadata['mmproj_candidates'] = [str(tmp_path/'config.json'),str(tmp_path/'vision.safetensors')]
    assert projector_candidates(model) == [str(tmp_path/'mmproj.gguf')]


def test_credentials_encrypted_at_rest(tmp_path):
    vault=Vault(tmp_path/'credentials.json')
    token='test-secret-'+uuid.uuid4().hex
    vault.set('hf',token)
    assert token not in (tmp_path/'credentials.json').read_text()
    assert Vault(tmp_path/'credentials.json').get('hf')==token


class FixtureAdapter:
    label='fixture'
    def parameters(self): return []
    def inspect(self,*args): return Compatibility(status='supported')
    def build_launch(self,model,profile,runtime,context):
        return LaunchSpec(argv=[sys.executable,str(Path(__file__).with_name('fake_backend.py')),'--port',str(context.port)],cwd=str(Path(__file__).parent),model_name='fixture',startup_timeout=15)


@pytest.fixture
def controller(tmp_path):
    c=Controller(tmp_path/'data','management-secret')
    c.adapters['fixture']=FixtureAdapter()
    return c


def add_profile(c,tmp_path,alias):
    path=tmp_path/(alias+'.gguf')
    gguf(path)
    model=inspect_asset(str(path))
    runtime=RuntimeInstallation(id='fixture',backend='fixture',name='fixture',path=sys.executable)
    profile=LoadProfile(id=alias,model_id=model.id,name=alias,backend='fixture',runtime_id='fixture',alias=alias)
    for kind,value in [('models',model),('runtimes',runtime),('profiles',profile)]: c.store.put(kind,value.model_dump())
    return model,profile,runtime


async def wait_ready(instance):
    async with asyncio.timeout(20):
        while instance.state=='starting': await asyncio.sleep(.05)
    assert instance.state=='ready',instance.error


async def test_auth_profiles_and_external_file_protection(controller,tmp_path):
    model,profile,runtime=add_profile(controller,tmp_path,'one')
    app=build_app(controller)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
        assert (await client.get('/api/state')).status_code==401
        client.headers['Authorization']='Bearer management-secret'
        response=await client.post('/api/profiles',json={**profile.model_dump(),'id':'two'})
        assert response.status_code==409
        assert (await client.delete('/api/models/'+model.id+'?delete_files=true')).status_code==400
        assert Path(model.path).exists()
        assert (await client.delete('/api/models/'+model.id)).status_code==200
        assert Path(model.path).exists()
    await controller.close()


async def test_profile_copy_creates_independent_record_without_changing_model_or_original(controller, tmp_path):
    model, profile, _ = add_profile(controller, tmp_path, 'original')
    profile.parameters = {'strata_config': {'args': ['--mtp', 'local-mtp', '--resident-experts']}, 'unknown_future_setting': {'nested': [1, 2]}}
    profile.generation = {'temperature': .6, 'custom': {'enabled': True}}
    profile.extra_args = ['--spec', '4']
    original = controller.store.put('profiles', profile.model_dump())
    body = {**original, 'name': 'Original · 副本', 'alias': 'original-copy'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
        response = await client.post('/api/profiles', json=body)
        assert response.status_code == 200
        duplicate = response.json()
        assert duplicate['id'] != profile.id and duplicate['model_id'] == model.id
        for key in ('parameters', 'generation', 'extra_args', 'runtime_id', 'backend'):
            assert duplicate[key] == original[key]
        assert (await client.post('/api/profiles', json=body)).status_code == 409
        assert controller.store.get('profiles', profile.id) == original
        assert len(controller.store.all('models')) == 1 and len(controller.store.all('profiles')) == 2
    await controller.close()


async def test_numeric_profile_contract_rejects_before_save_and_preserves_sentinels(controller, tmp_path):
    from test_mtp import base
    model, profile, runtime, _ = base(tmp_path)
    model.metadata['qwen35.block_count'] = 32
    for kind, item in [('models', model), ('profiles', profile), ('runtimes', runtime)]:
        controller.store.put(kind, item.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
        preflight = (await client.post('/api/profiles/preflight', json=profile.model_dump())).json()
        gpu = next(field for field in preflight['parameters'] if field['key'] == 'gpu_layers')
        assert gpu['maximum'] == 33
        cases = [('parameters', {'gpu_layers': 1000}), ('parameters', {'gpu_layers': -3}),
                 ('parameters', {'context_length': -1}), ('parameters', {'threads': 1.5}),
                 ('parameters', {'startup_timeout': 0}), ('generation', {'top_p': -0.01}),
                 ('generation', {'max_tokens': -3}), ('generation', {'seed': 4294967296})]
        for section, values in cases:
            response = await client.patch('/api/profiles/' + profile.id, json={section: values})
            assert response.status_code == 400, response.text
            assert controller.store.get('profiles', profile.id) == profile.model_dump()
        response = await client.post('/api/profiles', json={**profile.model_dump(), 'alias': 'invalid', 'parameters': {'gpu_layers': 1000}})
        assert response.status_code == 400
        body = {'parameters': {'gpu_layers': 33, 'context_length': 0, 'threads': -1, 'parallel': -1},
                'generation': {'temperature': 3.5, 'top_p': 0, 'top_k': 0, 'max_tokens': -1, 'seed': -1}}
        assert (await client.patch('/api/profiles/' + profile.id, json=body)).status_code == 200
        preview = (await client.post('/api/profiles/' + profile.id + '/preview')).json()
        assert preview['argv'][preview['argv'].index('--n-gpu-layers') + 1] == '33'
        body['parameters'].update(gpu_layers=-2, threads=-2)
        body['generation'].update(top_p=2, max_tokens=0)
        response = await client.patch('/api/profiles/' + profile.id, json=body)
        assert response.status_code == 200, response.text
        assert response.json()['parameters'] == body['parameters']
        assert response.json()['generation'] == body['generation']
        preview = (await client.post('/api/profiles/' + profile.id + '/preview')).json()
        assert preview['argv'][preview['argv'].index('--n-gpu-layers') + 1] == '-2'
        assert preview['argv'][preview['argv'].index('--threads') + 1] == '-2'
        assert not controller.runtime.instances
    await controller.close()


async def test_mtp_length_survives_real_save_and_command_preview(controller, tmp_path):
    from test_mtp import base
    model, profile, runtime, _ = base(tmp_path)
    for kind, item in [('models', model), ('runtimes', runtime)]:
        controller.store.put(kind, item.model_dump())
    profile.parameters = {'mtp_enabled': True, 'mtp_tokens': 6}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
        response = await client.post('/api/profiles', json=profile.model_dump())
        assert response.status_code == 200, response.text
        saved = response.json()
        preview = (await client.post('/api/profiles/' + saved['id'] + '/preview')).json()
        assert preview['argv'][preview['argv'].index('--spec-draft-n-max') + 1] == '6'
        status = (await client.post('/api/profiles/preflight', json=saved)).json()
        assert status['mtp']['tokens'] == 6
        assert not controller.runtime.instances
    await controller.close()


async def test_two_processes_route_stream_and_independent_stop(controller,tmp_path):
    first=await controller.runtime.start(*add_profile(controller,tmp_path,'one'))
    second=await controller.runtime.start(*add_profile(controller,tmp_path,'two'))
    try:
        await asyncio.gather(wait_ready(first),wait_ready(second))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller,public=True)),base_url='http://testserver',headers={'Authorization':'Bearer '+controller.api_key}) as client:
            data=(await client.get('/v1/models')).json()
            assert {x['id'] for x in data['data']}=={'one','two'}
            response=await client.post('/v1/chat/completions',json={'model':'one','messages':[{'role':'user','content':'hi'}]})
            assert response.json()['model']=='one'
            streamed=await client.post('/v1/chat/completions',json={'model':'two','messages':[{'role':'user','content':'hi'}],'stream':True})
            assert '[DONE]' in streamed.text and '"model": "two"' in streamed.text
            assert second.active_requests==0
            tools = [{'type':'function', 'function':{'name':'lookup', 'parameters':{'type':'object', 'properties':{}}}}]
            assert (await client.post('/v1/chat/completions',json={'model':'two','messages':[],'tools':tools})).status_code==200
            await controller.runtime.stop(first.id)
            assert first.state=='stopped' and second.state=='ready'
            assert (await client.post('/v1/chat/completions',json={'model':'one','messages':[]})).status_code==409
    finally:
        await controller.close()


async def test_load_cancel_releases_owned_process(controller,tmp_path):
    instance=await controller.runtime.start(*add_profile(controller,tmp_path,'cancel'))
    await asyncio.sleep(.05)
    await controller.runtime.stop(instance.id)
    assert instance.state=='stopped'
    process=controller.runtime.processes.get(instance.id)
    assert not process or process.returncode is not None
    await controller.close()


async def test_first_metric_sample_same_clock_tick_still_updates_memory(controller,monkeypatch):
    from types import SimpleNamespace
    import modloader.app as app_module
    monkeypatch.setattr(app_module,'time',SimpleNamespace(monotonic=lambda:1.0))
    monkeypatch.setattr(app_module.shutil,'which',lambda _:None)
    controller.metrics_task=asyncio.create_task(controller.metrics())
    await asyncio.sleep(0)
    assert controller.hardware['memory_total']>0
    assert controller.hardware['disk_read_bytes_per_second'] is None
    assert 'error' not in controller.hardware
    await controller.close()


async def test_signed_download_url_is_encrypted_in_job_storage(controller,monkeypatch):
    async def no_download(job): pass
    monkeypatch.setattr(controller,'run_job',no_download)
    url='https://example.invalid/model.gguf?signature=test-private-signature'
    job=controller.new_job('download',{'source':'url','url':url})
    persisted=controller.store.get('jobs',job['id'])
    assert 'test-private-signature' not in json.dumps(persisted)
    assert controller.vault.get('download_url_'+job['id'])==url
    await controller.close()


async def test_job_controls_follow_download_and_install_stage_capabilities(controller, monkeypatch):
    import asyncio
    import modloader.backends as backends

    prepare, download, install = asyncio.Event(), asyncio.Event(), asyncio.Event()
    allow_download, allow_install = asyncio.Event(), asyncio.Event()
    async def installer(backend, paths, options, progress, cancel, pause=None, confirm=None, proxy=None):
        await progress({'phase': 'prepare', 'can_pause': False})
        prepare.set()
        await allow_download.wait()
        await progress({'phase': 'download', 'can_pause': True, 'downloaded': 1024})
        download.set()
        while not allow_install.is_set():
            await pause.checkpoint(cancel)
            await asyncio.sleep(.005)
        await pause.checkpoint(cancel)
        await progress({'phase': 'install', 'can_pause': False})
        install.set()
        await cancel.wait()
        raise asyncio.CancelledError
    monkeypatch.setattr(backends, 'install_backend', installer)
    controller.install_plans['controls'] = {'id':'controls', 'backend':'llama_cpp', 'version':'fixture', 'variant':'cpu', 'options':{}, 'stage':{'id':'base','files':[]}}
    app = build_app(controller)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
        response = await client.post('/api/runtimes/install', json={'plan_id': 'controls'})
        identifier = response.json()['id']
        task = controller.job_tasks[identifier]
        try:
            await asyncio.wait_for(prepare.wait(), 2)
            assert (await client.post(f'/api/jobs/{identifier}/pause')).status_code == 409
            allow_download.set()
            await asyncio.wait_for(download.wait(), 2)
            response = await client.post(f'/api/jobs/{identifier}/pause')
            assert response.status_code == 200 and response.json()['state'] == 'pausing'
            async with asyncio.timeout(2):
                while controller.require('jobs', identifier)['state'] != 'paused':
                    await asyncio.sleep(.005)
            assert (await client.post(f'/api/jobs/{identifier}/pause')).status_code == 409
            assert (await client.post(f'/api/downloads/{identifier}/retry')).status_code == 409
            assert (await client.post('/api/runtimes/install', json={'backend': 'llama_cpp'})).status_code == 409
            response = await client.post(f'/api/jobs/{identifier}/resume')
            assert response.json()['state'] == 'running' and response.json()['id'] == identifier
            assert controller.job_tasks[identifier] is task
            assert controller.require('jobs', identifier)['downloaded_bytes'] == 1024
            allow_install.set()
            await asyncio.wait_for(install.wait(), 2)
            assert (await client.post(f'/api/jobs/{identifier}/pause')).status_code == 409
            response = await client.post(f'/api/jobs/{identifier}/cancel')
            assert response.json()['state'] == 'cancelled' and response.json()['can_pause'] is False
            assert response.json()['message'] == '已停止；可重试重新安装'
            assert task.done()
            assert (await client.post(f'/api/jobs/{identifier}/resume')).status_code == 409
        finally:
            await controller.close()


@pytest.mark.parametrize('state', ['pausing', 'paused'])
async def test_restart_marks_paused_job_interrupted_without_live_coroutine(controller, state):
    controller.store.put('jobs', {'id': 'old-paused', 'kind': 'download', 'state': state, 'can_pause': True, 'request': {}})
    root = controller.paths.root
    await controller.close()
    reopened = Controller(root, 'management-secret')
    try:
        job = reopened.require('jobs', 'old-paused')
        assert job['state'] == 'interrupted' and job['can_pause'] is False
        assert not reopened.job_tasks and not reopened.download_pauses
    finally:
        await reopened.close()


async def test_new_pause_ignores_worker_acknowledgement_for_previous_request():
    from modloader.sources.control import DownloadPause

    states = []
    async def paused():
        states.append('paused')
    control = DownloadPause(paused)
    control.request()
    previous = control.generation
    control.resume()
    control.request()
    await control.acknowledge(previous)
    assert states == []
    await control.acknowledge(control.generation)
    assert states == ['paused']
    control.resume()


async def test_recursive_folder_import_preserves_existing_profiles_and_reports_partial_errors(controller,tmp_path):
    folder=tmp_path/'library'
    folder.mkdir()
    model,profile,_=add_profile(controller,folder,'existing')
    original=controller.store.get('models',model.id)
    original['name']='我的显示名称'
    controller.store.put('models',original)
    deep=folder/'vendor'/'family'/'revision'/'quantization'
    deep.mkdir(parents=True)
    gguf(deep/'new.gguf')
    (deep/'broken.gguf').write_bytes(b'invalid')
    app=build_app(controller)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',headers={'Authorization':'Bearer management-secret'}) as client:
            first,second=await asyncio.gather(*(client.post('/api/models/import-folder',json={'path':str(folder)}) for _ in range(2)))
            assert first.status_code==second.status_code==200
            results=[first.json(),second.json()]
            assert sum(len(result['imported']) for result in results)==1
            assert sorted(len(result['existing']) for result in results)==[1,2]
            assert all(len(result['errors'])==1 and result['errors'][0]['path']==str(deep/'broken.gguf') for result in results)
            assert len(controller.store.all('models'))==2
            assert controller.store.get('models',model.id)==original
            assert controller.store.get('profiles',profile.id)==profile.model_dump()
            assert (deep/'new.gguf').exists() and (deep/'broken.gguf').read_bytes()==b'invalid'
            assert (await client.post('/api/models/import-folder',json={'path':str(deep/'new.gguf')})).status_code==400
    finally:
        await controller.close()


class GatewayStream(httpx.AsyncByteStream):
    def __init__(self, chunks, waiting=None):
        self.chunks = chunks
        self.waiting = waiting
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
        if self.waiting is not None:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def gateway_client(controller, tmp_path, monkeypatch, handler, aliases=('one',)):
    """Exercise the real gateway without a model, process or persistent chat data."""
    instances = {}
    for alias in aliases:
        model, profile, _ = add_profile(controller, tmp_path, alias)
        instance = RuntimeInstance(id=alias+'-instance', model_id=model.id, profile_id=profile.id,
                                   profile_revision=profile.revision, alias=alias, backend='fixture',
                                   runtime_id='fixture', state='ready', port=12345, started_at='',
                                   capabilities={'input':['text'], 'tools':False})
        controller.runtime.instances[instance.id] = instance
        controller.runtime.specs[instance.id] = LaunchSpec(argv=[], cwd=str(tmp_path), model_name='fixture')
        controller.runtime.requests[instance.id] = set()
        controller.runtime.logs[instance.id] = []
        instances[alias] = instance
    original_client = httpx.AsyncClient
    client = original_client(transport=httpx.ASGITransport(app=build_app(controller)),
                             base_url='http://testserver', headers={'Authorization':'Bearer management-secret'})
    backend_clients = []
    def backend_client(*args, **kwargs):
        result = original_client(*args, **kwargs, transport=httpx.MockTransport(handler))
        backend_clients.append(result)
        return result
    monkeypatch.setattr('modloader.app.httpx.AsyncClient', backend_client)
    return client, instances, backend_clients


@pytest.mark.parametrize('endpoint', ['/api/chat', '/v1/chat/completions'])
@pytest.mark.parametrize('backend_status', [200, 400])
async def test_unverified_tools_reach_backend_and_preserve_its_result(controller, tmp_path, monkeypatch, endpoint, backend_status):
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {}}}}]
    calls = []
    backend_result = {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call-1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
    ]}, 'finish_reason': 'tool_calls'}]} if backend_status == 200 else {'error': {'message': 'tools unsupported'}}

    async def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(backend_status, json=backend_result)

    client, instances, backend_clients = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            response = await client.post(endpoint, json={'model': 'one', 'messages': [], 'tools': tools, 'tool_choice': 'auto'})
            assert response.status_code == backend_status
            actual = response.json()
            if endpoint == '/api/chat':
                report = actual.pop('modloader_request')
                assert report['status'] == ('completed' if backend_status == 200 else 'failed')
                assert report['upstream']['status'] == backend_status
                assert report['counters'] == {'input': None, 'output': None, 'cached': None}
            assert actual == ({**backend_result, 'model': 'one'} if backend_status == 200 else backend_result)
            assert len(calls) == 1
            assert calls[0]['tools'] == tools and calls[0]['tool_choice'] == 'auto'
            assert instances['one'].active_requests == 0
            assert all(client.is_closed for client in backend_clients)
    finally:
        await controller.close()


def sse(chunk, prefix='data: '):
    data = chunk if isinstance(chunk,str) else json.dumps(chunk)
    return (prefix+data+'\n\n').encode()


@pytest.mark.parametrize('prefix', ['data: ', 'data:'])
@pytest.mark.parametrize('endpoint', ['/api/chat', '/v1/chat/completions'])
async def test_gateway_stream_usage_tail_reasoning_and_numeric_only_retention(controller,tmp_path,monkeypatch,prefix,endpoint):
    private_prompt, private_reasoning, private_answer = 'private prompt sentinel', 'private reasoning sentinel', 'private answer sentinel'
    messages = [{'role':'user','content':'previous question'},
                {'role':'assistant','content':private_answer,'reasoning_content':private_reasoning},
                {'role':'user','content':private_prompt}]
    stream = GatewayStream([
        sse({'model':'fixture','choices':[{'delta':{'role':'assistant'}}]},prefix),
        sse({'model':'fixture','choices':[{'delta':{'reasoning_content':private_reasoning}}],'usage':{'completion_tokens':2}},prefix),
        sse({'model':'fixture','choices':[{'delta':{'content':private_answer}}]},prefix),
        sse({'model':'fixture','choices':[{'delta':{},'finish_reason':'stop'}]},prefix),
        sse({'model':'fixture','choices':[], 'usage':{'prompt_tokens':9,'completion_tokens':5,'total_tokens':14,'note':private_prompt},
             'timings':{'predicted_n':5,'predicted_ms':250,'note':private_answer}},prefix),
        sse('[DONE]',prefix),
    ])
    async def handler(request):
        payload = json.loads(request.content)
        assert payload['model']=='fixture'
        assert payload['messages']==messages
        assert payload['preserve_thinking'] is True
        assert payload['stream_options']=={'include_usage':True,'continuous_usage_stats':False}
        return httpx.Response(200,stream=stream)
    client, instances, backend_clients = gateway_client(controller,tmp_path,monkeypatch,handler)
    instance = instances['one']
    try:
        async with client:
            before = (await client.get('/api/instances/'+instance.id+'/telemetry')).json()
            assert not any(group['id']=='api_request' for group in before['groups'])
            response = await client.post(endpoint,json={'model':'one','messages':messages,'preserve_thinking':True,
                                                          'stream':True,'stream_options':{'continuous_usage_stats':False}})
            assert response.status_code==200 and '[DONE]' in response.text
            chunks = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith('data:') and line[5:].strip()!='[DONE]']
            reports = [chunk['modloader_request'] for chunk in chunks if 'modloader_request' in chunk]
            assert len(reports) == (1 if endpoint == '/api/chat' else 0)
            chunks = [chunk for chunk in chunks if 'modloader_request' not in chunk]
            assert all(chunk['model']=='one' for chunk in chunks)
            assert chunks[1]['choices'][0]['delta']['reasoning_content']==private_reasoning
            assert chunks[2]['choices'][0]['delta']['content']==private_answer
            assert chunks[-1]['usage']['completion_tokens']==5
            record = controller.request_metrics[instance.id]
            assert record['status']=='completed' and 0<=record['first_output_s']<=record['duration_s']
            assert record['usage']=={'prompt_tokens':9,'completion_tokens':5,'total_tokens':14}
            assert record['timings']=={'predicted_n':5,'predicted_ms':250}
            assert not any(text in json.dumps(record) for text in (private_prompt,private_reasoning,private_answer))
            groups = (await client.get('/api/instances/'+instance.id+'/telemetry')).json()['groups']
            assert groups[0]['id']=='api_request'
            metrics = {metric['id']:metric for metric in groups[0]['metrics']}
            assert metrics['output_speed']['value']==20 and metrics['output_speed']['label']=='后端解码速度'
            assert metrics['output_speed']['rate_basis']=='decode'
            assert metrics['ended_at']['value']==record['ended_at'] and metrics['ended_at']['kind']=='timestamp'
            assert instance.active_requests==0 and not controller.runtime.requests[instance.id]
            assert not controller.active_request_tokens[instance.id]
            assert stream.closed and all(client.is_closed for client in backend_clients)
    finally:
        await controller.close()
    persisted = (controller.paths.root/'modloader.sqlite3').read_bytes()
    assert not any(text.encode() in persisted for text in (private_prompt,private_reasoning,private_answer))
    reopened = Controller(controller.paths.root,'management-secret')
    try:
        assert reopened.request_metrics=={}
    finally:
        await reopened.close()


async def test_gateway_nonstream_usage_average_and_instance_isolation(controller,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import modloader.request_metrics as request_module
    clock = iter((1.0,2.0,3.0,5.0))
    monkeypatch.setattr(request_module,'time',SimpleNamespace(monotonic=lambda:next(clock),time=lambda:100.0))
    completion_tokens = iter((4,11))
    async def handler(request):
        return httpx.Response(200,json={'model':'fixture','choices':[{'message':{'role':'assistant','content':'answer'},'finish_reason':'stop'}],
                                        'usage':{'prompt_tokens':3,'completion_tokens':next(completion_tokens)}})
    client, instances, backend_clients = gateway_client(controller,tmp_path,monkeypatch,handler,('one','two'))
    try:
        async with client:
            for alias in instances:
                response = await client.post('/api/chat',json={'model':alias,'messages':[]})
                assert response.status_code==200 and response.json()['model']==alias
            first, second = (controller.request_metrics[instances[alias].id] for alias in ('one','two'))
            assert first['usage']['completion_tokens']==4 and second['usage']['completion_tokens']==11
            for record, expected in ((first,4),(second,5.5)):
                metrics = {metric['id']:metric for metric in request_module.request_metrics_group(record)['metrics']}
                assert metrics['output_speed']['value']==expected
                assert metrics['output_speed']['label']=='端到端平均速度' and '含排队、预填充' in metrics['output_speed']['description']
                assert metrics['output_speed']['rate_basis']=='end_to_end'
                assert 'first_output_s' not in metrics
            await controller.runtime.stop(instances['one'].id)
            stopped = (await client.get('/api/instances/'+instances['one'].id+'/telemetry')).json()
            assert stopped['state']=='stopped' and stopped['groups'][0]['id']=='api_request'
            assert all(instance.active_requests==0 for instance in instances.values())
            assert all(client.is_closed for client in backend_clients)
    finally:
        await controller.close()


@pytest.mark.parametrize('streaming,complete,include_usage', [(False,True,False),(True,True,False),(True,False,True)])
async def test_gateway_absent_usage_and_incomplete_stream_do_not_invent_speed(controller,tmp_path,monkeypatch,streaming,complete,include_usage):
    async def handler(request):
        payload = json.loads(request.content)
        if not streaming:
            return httpx.Response(200,json={'model':'fixture','choices':[{'message':{'content':'hello'}}]})
        assert payload['stream_options']=={'include_usage':include_usage,'continuous_usage_stats':True}
        chunks = [sse({'model':'fixture','choices':[{'delta':{'content':'hello'}}]})]
        if include_usage:
            chunks.append(sse({'choices':[],'usage':{'completion_tokens':3}}))
        if complete:
            chunks.append(sse('[DONE]'))
        return httpx.Response(200,stream=GatewayStream(chunks))
    client, instances, _ = gateway_client(controller,tmp_path,monkeypatch,handler)
    instance = instances['one']
    try:
        async with client:
            response = await client.post('/api/chat',json={'model':'one','messages':[],'stream':streaming,
                                                          'stream_options':{'include_usage':include_usage,'continuous_usage_stats':True}})
            assert response.status_code==200
            record = controller.request_metrics[instance.id]
            assert record['status']==('completed' if complete else 'incomplete')
            assert record['usage']==({'completion_tokens':3} if include_usage else {})
            from modloader.request_metrics import request_metrics_group
            metrics = {metric['id']:metric for metric in request_metrics_group(record)['metrics']}
            assert metrics['output_speed']['value'] is None
            assert instance.active_requests==0 and not controller.runtime.requests[instance.id]
    finally:
        await controller.close()


@pytest.mark.parametrize('failure', ['http','stream','connection','read'])
async def test_gateway_failed_requests_release_resources_and_do_not_report_full_speed(controller,tmp_path,monkeypatch,failure):
    streams = []
    async def handler(request):
        if failure=='connection':
            raise httpx.ConnectError('fixture unavailable',request=request)
        if failure=='http':
            return httpx.Response(503,json={'error':{'message':'fixture failure'}})
        chunks = [sse({'choices':[{'delta':{'content':'partial'}}],'usage':{'completion_tokens':2},'timings':{'predicted_per_second':99}})]
        chunks.extend([sse({'error':{'message':'fixture stream failure'}}),sse('[DONE]')] if failure=='stream' else [httpx.ReadError('fixture disconnect')])
        stream = GatewayStream(chunks)
        streams.append(stream)
        return httpx.Response(200,stream=stream)
    client, instances, backend_clients = gateway_client(controller,tmp_path,monkeypatch,handler)
    instance = instances['one']
    try:
        async with client:
            response = await client.post('/api/chat',json={'model':'one','messages':[],'stream':failure in ('stream','read')})
            assert response.status_code==({'http':503,'connection':502}.get(failure,200))
            assert 'error' in response.text or 'detail' in response.text
            record = controller.request_metrics[instance.id]
            assert record['status']=='failed'
            from modloader.request_metrics import request_metrics_group
            group = request_metrics_group(record)
            assert group['request_key'] == record['request_key'] and group['request_status'] == 'failed'
            metrics = {metric['id']:metric for metric in group['metrics']}
            assert metrics['output_speed']['value'] is None
            assert instance.active_requests==0 and not controller.runtime.requests[instance.id]
            assert all(stream.closed for stream in streams) and all(client.is_closed for client in backend_clients)
    finally:
        await controller.close()


async def test_gateway_stream_cancellation_retains_status_and_releases_backend(controller,tmp_path,monkeypatch):
    waiting = asyncio.Event()
    stream = GatewayStream([sse({'choices':[{'delta':{'reasoning_content':'partial thought'}}],
                                'usage':{'completion_tokens':2},'timings':{'predicted_per_second':99}})],waiting)
    async def handler(request):
        return httpx.Response(200,stream=stream)
    client, instances, backend_clients = gateway_client(controller,tmp_path,monkeypatch,handler)
    instance = instances['one']
    try:
        async with client:
            task = asyncio.create_task(client.post('/api/chat',json={'model':'one','messages':[],'stream':True},
                                                  headers={'X-Modloader-Request-Id': 'cancelled-assistant'}))
            await asyncio.wait_for(waiting.wait(),5)
            assert instance.active_requests==1
            token = next(iter(controller.active_request_tokens[instance.id]))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            record = controller.request_metrics[instance.id]
            assert record['status']=='cancelled' and record['first_output_s'] is not None
            from modloader.request_metrics import request_metrics_group
            group = request_metrics_group(record)
            assert group['request_key'] == token and group['request_status'] == 'cancelled'
            assert group['client_request_id'] == 'cancelled-assistant'
            metrics = {metric['id']:metric for metric in group['metrics']}
            assert metrics['output_speed']['value'] is None
            assert instance.active_requests==0 and not controller.runtime.requests[instance.id]
            assert not controller.active_request_tokens[instance.id]
            assert stream.closed and all(client.is_closed for client in backend_clients)
    finally:
        await controller.close()


def test_request_metadata_rejects_invalid_numbers_and_prefers_backend_speed(monkeypatch):
    from types import SimpleNamespace
    import modloader.request_metrics as request_module
    clock = iter((1.0,3.0))
    monkeypatch.setattr(request_module,'time',SimpleNamespace(monotonic=lambda:next(clock),time=lambda:100.0))
    metrics = request_module.RequestMetrics(False)
    metrics.observe({'usage':{'prompt_tokens':False,'completion_tokens':6,'total_tokens':float('nan')},
                     'timings':{'predicted_per_second':12.5,'predicted_n':-1,'predicted_ms':'50','private':'body'}})
    record = metrics.finish('completed')
    assert record['usage']=={'completion_tokens':6} and record['timings']=={'predicted_per_second':12.5}
    values = {item['id']:item for item in request_module.request_metrics_group(record)['metrics']}
    assert values['output_speed']['value']==12.5 and values['output_speed']['label']=='后端解码速度'
    assert values['prompt_tokens']['value'] is None
    assert metrics.finish('failed') is record and record['status']=='completed'


async def test_gateway_done_then_reader_close_keeps_completion_and_latest_request(controller,tmp_path,monkeypatch):
    from starlette.requests import Request
    waiting = asyncio.Event()
    stream = GatewayStream([sse({'choices':[],'usage':{'completion_tokens':2}}),sse('[DONE]')],waiting)
    async def handler(request):
        if json.loads(request.content).get('stream'):
            return httpx.Response(200,stream=stream)
        return httpx.Response(200,json={'model':'fixture','choices':[], 'usage':{'completion_tokens':7},
                                        'timings':{'predicted_per_second':42}})
    client, instances, backend_clients = gateway_client(controller,tmp_path,monkeypatch,handler)
    instance = instances['one']
    try:
        async with client:
            app = client._transport.app
            endpoint = next(route.endpoint for route in app.routes if route.path=='/api/chat')
            async def receive():
                return {'type':'http.request','body':json.dumps({'model':'one','messages':[],'stream':True}).encode(),'more_body':False}
            response = await endpoint(Request({'type':'http','method':'POST','path':'/api/chat','headers':[]},receive))
            async for line in response.body_iterator:
                if '[DONE]' in line:
                    break
            completed = controller.request_metrics[instance.id]
            assert completed['status']=='completed' and completed['usage']['completion_tokens']==2
            # A later request may finish while the first reader is closing.
            await client.post('/api/chat',json={'model':'one','messages':[]})
            latest = controller.request_metrics[instance.id]
            assert latest['usage']['completion_tokens']==7
            later_window = {'token':'later-request-token'}
            controller.generation_samples[instance.id]=later_window
            await response.body_iterator.aclose()
            assert controller.request_metrics[instance.id] is latest
            assert controller.generation_samples[instance.id] is later_window
            assert completed['status']=='completed'
            assert instance.active_requests==0 and not controller.runtime.requests[instance.id]
            assert stream.closed and all(client.is_closed for client in backend_clients)
    finally:
        await controller.close()


@pytest.mark.parametrize('source,before,after,span,expected', [
    ('/metrics live.generated',132,291,1.0243212,155.22573),
    ('/slots next_token.n_decoded',89,154,.9759905,66.598998),
])
def test_generation_window_uses_real_failure_counts_instead_of_backend_mean(source,before,after,span,expected):
    from modloader.request_metrics import generation_window
    live = {'scope':'model','source':source,'backend_running':1,'backend_queued':0,
            'tokens_per_second':140.2 if 'metrics' in source else 0,'window_s':2,
            '_generated_tokens':before,'_elapsed_s':1.6,'_slot_id':0}
    previous, first = generation_window(None,'request-one',live,10.0)
    assert first['tokens_per_second'] is None and not first['available_for_chat']
    _, second = generation_window(previous,'request-one',{**live,'_generated_tokens':after,'_elapsed_s':2.6},10+span)
    assert second['tokens_per_second']==pytest.approx(expected,rel=1e-5)
    assert second['window_s']==pytest.approx(span) and second['available_for_chat']
    assert not any(key.startswith('_') for key in second)


@pytest.mark.parametrize('change,new_token,new_time', [
    ({},'another-request',11.0),
    ({'_slot_id':1},'request-one',11.0),
    ({'source':'SSE usage.completion_tokens'},'request-one',11.0),
    ({'_generated_tokens':50},'request-one',11.0),
    ({'_elapsed_s':.5},'request-one',11.0),
    ({},'request-one',9.0),
    ({},'request-one',10.0),
    ({'backend_running':2},'request-one',11.0),
    ({'backend_queued':1},'request-one',11.0),
    ({},None,11.0),
    ({'_generated_tokens':0},'request-one',11.0),
    ({'_generated_tokens':None},'request-one',11.0),
])
def test_generation_window_resets_at_request_and_counter_boundaries(change,new_token,new_time):
    from modloader.request_metrics import generation_window
    live = {'scope':'model','source':'/slots next_token.n_decoded','backend_running':1,'backend_queued':0,
            '_generated_tokens':100,'_elapsed_s':1.0,'_slot_id':0}
    previous, _ = generation_window(None,'request-one',live,10.0)
    next_state, result = generation_window(previous,new_token,{**live,'_generated_tokens':200,'_elapsed_s':2.0,**change},new_time)
    assert result['tokens_per_second'] is None and result['window_s'] is None and not result['available_for_chat']
    if next_state is not None:
        assert next_state['time']==new_time  # Reset baseline, never reuse the old interval.


def test_generation_window_reports_zero_for_a_stalled_positive_counter():
    from modloader.request_metrics import generation_window
    live = {'scope':'model','source':'/metrics live.generated','backend_running':1,'backend_queued':0,
            '_generated_tokens':100,'_elapsed_s':1.0,'_slot_id':None}
    previous, _ = generation_window(None,'request-one',live,10.0)
    _, result = generation_window(previous,'request-one',{**live,'_elapsed_s':2.0},11.0)
    assert result['tokens_per_second']==0 and result['available_for_chat']


def test_generation_window_grows_then_rolls_without_changing_request_mean():
    from modloader.request_metrics import generation_window
    live = {'scope':'model','source':'/metrics live.generated','backend_running':1,'backend_queued':0,'_slot_id':0}
    state = None
    for second, count, expected in [(0,100,None),(1,110,10),(2,140,20),(3,200,100/3),(4,220,110/3)]:
        state, result = generation_window(state,'one',{**live,'_generated_tokens':count},10+second)
        if expected is None:
            assert result['tokens_per_second'] is None and result['window_s'] is None
        else:
            assert result['tokens_per_second'] == pytest.approx(expected)
            assert result['window_s'] == min(second,3)
            assert result['average_tokens_per_second'] == pytest.approx((count-100)/second)
    assert state['samples'] == [(11,110),(12,140),(13,200),(14,220)]
    assert state['start_time'] == 10 and state['start_count'] == 100
    assert result['average_tokens_per_second'] == 30
    assert result['tokens_per_second'] != result['average_tokens_per_second']


def test_generation_window_weights_jitter_and_short_dropped_polls_by_real_time():
    from modloader.request_metrics import generation_window
    live = {'scope':'model','source':'/slots next_token.n_decoded','backend_running':1,'backend_queued':0,'_slot_id':0}
    state = None
    for sampled_at, count in [(10,100),(10.75,130),(12.25,145),(13.25,205),(14,220)]:
        state, result = generation_window(state,'one',{**live,'_generated_tokens':count},sampled_at)
        if sampled_at == 13.25:
            assert result['window_s'] == 3.25
            assert result['tokens_per_second'] == pytest.approx(105/3.25)
            assert result['tokens_per_second'] != pytest.approx((40+10+60)/3)
    assert state['samples'] == [(10.75,130),(12.25,145),(13.25,205),(14,220)]
    assert result['window_s'] == 3.25
    assert result['tokens_per_second'] == pytest.approx(90/3.25)
    assert result['average_tokens_per_second'] == 30


def test_generation_window_includes_short_stalls_and_batched_token_counts():
    from modloader.request_metrics import generation_window
    live = {'scope':'request','source':'SSE usage.completion_tokens','_slot_id':None}
    state = None
    for second, count, expected in [(0,100,None),(1,160,60),(2,160,30),(3,160,20),
                                    (4,220,20),(5,220,20),(6,220,20),(7,220,0)]:
        state, result = generation_window(state,'one',{**live,'_generated_tokens':count},10+second)
        if expected is not None:
            assert result['tokens_per_second'] == expected
            assert result['available_for_chat']
    assert result['average_tokens_per_second'] == pytest.approx(120/7)


def test_generation_window_uses_nearest_observed_boundary_after_sparse_polls():
    from modloader.request_metrics import generation_window
    live = {'scope':'request','source':'SSE usage.completion_tokens','_slot_id':None}
    state = None
    for sampled_at, count in [(10,100),(12.9,130),(15.8,200)]:
        state, result = generation_window(state,'one',{**live,'_generated_tokens':count},sampled_at)
    assert state['samples'] == [(12.9,130),(15.8,200)]
    assert result['window_s'] == pytest.approx(2.9)
    assert result['tokens_per_second'] == pytest.approx(70/2.9)
    assert result['average_tokens_per_second'] == pytest.approx(100/5.8)


def test_generation_window_restarts_current_rate_after_long_gap_but_keeps_request_mean():
    from modloader.request_metrics import generation_window
    live = {'scope':'model','source':'/metrics live.generated','backend_running':1,'backend_queued':0,
            '_generated_tokens':100,'_elapsed_s':1,'_slot_id':0}
    state, _ = generation_window(None,'one',live,10)
    state, _ = generation_window(state,'one',{**live,'_generated_tokens':140,'_elapsed_s':2},11)
    state, resumed = generation_window(state,'one',{**live,'_generated_tokens':240,'_elapsed_s':5.01},14.01)
    assert resumed['tokens_per_second'] is None and resumed['window_s'] is None
    assert not resumed['available_for_chat']
    assert state['samples'] == [(14.01,240)]
    assert resumed['average_tokens_per_second'] == pytest.approx(140/4.01)
    _, measured = generation_window(state,'one',{**live,'_generated_tokens':250,'_elapsed_s':6.01},15.01)
    assert measured['tokens_per_second'] == 10 and measured['window_s'] == 1
    assert measured['average_tokens_per_second'] == pytest.approx(150/5.01)


@pytest.mark.parametrize('backend', ['ninfer', 'llama_cpp', 'vllm', 'sglang'])
@pytest.mark.parametrize('chat_path', ['/api/chat', '/v1/chat/completions'])
async def test_gateway_exposes_exact_stream_counters_before_final_usage(controller, tmp_path, monkeypatch, backend, chat_path):
    """Exercise request forwarding, SSE observation and the actual telemetry route."""
    from starlette.requests import Request
    from types import SimpleNamespace
    import modloader.app as app_module
    clock, running = [0.0], [2]
    native_timings = backend in {'ninfer', 'llama_cpp'}
    monkeypatch.setattr(app_module, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000+clock[0]))
    first = {'choices': [{'delta': {'reasoning_content': 'one whole SSE chunk'}}]}
    second = {'choices': [{'delta': {'content': 'another whole chunk'}}]}
    if native_timings:
        first['timings'] = {'prompt_n': 40, 'cache_n': 60, 'predicted_n': 10, 'predicted_per_second': 90}
        second['timings'] = {'prompt_n': 40, 'cache_n': 60, 'predicted_n': 40, 'predicted_per_second': 100}
        native = ('ninfer_requests_running {running}\nninfer_requests_waiting 1\nninfer_max_context_tokens 128000\nninfer_generation_tokens_total 999999\n'
                  if backend == 'ninfer' else 'llamacpp:requests_processing {running}\nllamacpp:requests_deferred 1\n')
        progress = [sse({'choices': [{'delta': {}}], 'prompt_progress': {'total': 100, 'processed': 60, 'cache': 60}})]
    else:
        first['usage'] = {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}
        second['usage'] = {'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140}
        prefix, active, waiting = ('vllm:', 'num_requests_running', 'num_requests_waiting') if backend == 'vllm' else ('sglang:', 'num_running_reqs', 'num_queue_reqs')
        native = f'{prefix}{active} {{running}}\n{prefix}{waiting} 1\n{prefix}generation_tokens_total 999999\n'
        progress = []
    more = []
    for count in (100,120,150):
        chunk = {'choices': [{'delta': {'content': 'another batch'}}]}
        if native_timings:
            chunk['timings'] = {**second['timings'],'predicted_n':count}
        else:
            chunk['usage'] = {**second['usage'],'completion_tokens':count,'total_tokens':100+count}
        more.append(sse(chunk))
    stream = GatewayStream([*progress, sse(first), sse(second), *more,
                            sse({'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 150, 'total_tokens': 250}}),
                            sse('[DONE]')])
    async def handler(request):
        if request.method == 'GET':
            if backend == 'llama_cpp' and request.url.path == '/slots':
                return httpx.Response(200, json=[{'id': 0, 'is_processing': True, 'n_ctx': 128000,
                    'n_prompt_tokens': 60, 'n_prompt_tokens_processed': 0, 'n_prompt_tokens_cache': 60,
                    'next_token': [{'n_decoded': 0}]}])
            return httpx.Response(200, json={'status': 'ok'}) if request.url.path == '/health' else httpx.Response(200, text=native.format(running=running[0]))
        forwarded = json.loads(request.content)
        assert 'X-Modloader-Request-Id' not in request.headers
        assert 'client_request_id' not in forwarded
        if native_timings:
            assert forwarded['timings_per_token'] is True and forwarded['return_progress'] is True
        else:
            assert forwarded['stream_options'] == {'include_usage': True, 'continuous_usage_stats': True}
        return httpx.Response(200, stream=stream)
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    instance.backend = backend
    if backend == 'llama_cpp':
        controller.runtime.specs[instance.id].env['LLAMA_SERVER_SLOTS_DEBUG'] = '0'
    response = None
    try:
        async with client:
            endpoint = next(route.endpoint for route in client._transport.app.routes if route.path == chat_path)
            preflight = await client.options(chat_path, headers={'Origin': 'null', 'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'authorization,content-type,x-modloader-request-id'})
            assert preflight.status_code == 200
            telemetry_path = '/api/instances/'+instance.id+'/telemetry'
            assert (await client.get(telemetry_path)).json()['live_generation']['request_key'] is None
            async def receive():
                return {'type': 'http.request', 'body': json.dumps({'model': 'one', 'messages': [], 'stream': True}).encode(), 'more_body': False}
            response = await endpoint(Request({'type': 'http', 'method': 'POST', 'path': chat_path,
                'headers': [(b'x-modloader-request-id', b'assistant-message-one')]}, receive))
            async def advance():
                while True:
                    line = await response.body_iterator.__anext__()
                    if line.startswith('data:'):
                        return line
            waiting = (await client.get(telemetry_path)).json()['live_generation']
            assert waiting['request_key'] and waiting['phase'] == 'waiting'
            assert waiting['client_request_id'] == ('assistant-message-one' if chat_path == '/api/chat' else None)
            assert waiting['tokens_per_second'] is None and waiting['prefill_tokens_per_second'] is None
            assert waiting['generated_tokens'] is None and waiting['request_elapsed_s'] >= 0
            if native_timings:
                await advance()
                clock[0] = .5
                prefill = (await client.get(telemetry_path)).json()['live_generation']
                assert prefill['phase'] == 'prefill' and prefill['context_tokens'] == 100
                assert prefill['request_key'] == waiting['request_key']
                assert prefill['prefill_processed_tokens'] == 0 and prefill['prefill_total_tokens'] == 100
                assert prefill['prefill_cached_tokens'] == 60
                assert prefill['context_capacity'] == 128000 and prefill['tokens_per_second'] is None
            clock[0] = 1
            await advance()
            first_sample = (await client.get(telemetry_path)).json()['live_generation']
            assert first_sample['client_request_id'] == ('assistant-message-one' if chat_path == '/api/chat' else None)
            assert first_sample['request_key'] and first_sample['scope'] == 'request'
            assert first_sample['request_key'] == waiting['request_key'] and first_sample['phase'] == 'thinking'
            assert first_sample['generated_tokens'] == 10 and first_sample['context_tokens'] == 110
            assert first_sample['tokens_per_second'] is None
            assert not controller.request_metrics  # No terminal usage/result yet.
            clock[0] = 1.1
            await advance()
            cached = (await client.get(telemetry_path)).json()['live_generation']
            assert cached['request_key'] == waiting['request_key'] and cached['phase'] == 'output'
            assert cached['prefill_tokens_per_second'] is None
            clock[0] = 2
            measured = (await client.get(telemetry_path)).json()['live_generation']
            assert measured['available_for_chat'] and measured['tokens_per_second'] == 30
            assert measured['window_s'] == 1 and measured['context_tokens'] == 140
            assert measured['prompt_tokens'] == 100 and measured['generated_tokens'] == 40
            assert measured['phase'] == 'output' and measured['prefill_tokens_per_second'] is None
            assert measured['average_tokens_per_second'] == (100 if native_timings else 30)
            assert measured['average_basis'] == ('backend_decode' if native_timings else 'sampled_decode')
            # The stream belongs to this request, even while other backend work
            # exists. Multiple gateway requests are still ambiguous to this UI.
            assert measured['backend_running'] == 2 and measured['backend_queued'] == 1
            for second, count, speed, mean in [(3,100,45,45),(4,120,110/3,110/3),(5,150,110/3,35)]:
                clock[0] = second
                await advance()
                rolling = (await client.get(telemetry_path)).json()['live_generation']
                assert rolling['tokens_per_second'] == pytest.approx(speed)
                assert rolling['window_s'] == min(second-1,3)
                assert rolling['generated_tokens'] == count and rolling['context_tokens'] == 100+count
                assert rolling['average_tokens_per_second'] == pytest.approx(100 if native_timings else mean)
            instance.active_requests = 2
            ambiguous = (await client.get(telemetry_path)).json()['live_generation']
            assert ambiguous['client_request_id'] is None
            assert not ambiguous['available_for_chat'] and ambiguous['context_tokens'] is None
            assert ambiguous['tokens_per_second'] is None
            instance.active_requests = 1
            async for _ in response.body_iterator:
                pass
            assert controller.request_metrics[instance.id]['status'] == 'completed'
            assert instance.active_requests == 0
            completed = (await client.get(telemetry_path)).json()['live_generation']
            assert completed['client_request_id'] is None and completed['context_tokens'] is None
    finally:
        if response is not None:
            await response.body_iterator.aclose()
        await controller.close()


@pytest.mark.parametrize('backend', ['ninfer', 'llama_cpp'])
async def test_gateway_preserves_explicitly_disabled_native_stream_metrics(controller, tmp_path, monkeypatch, backend):
    async def handler(request):
        forwarded = json.loads(request.content)
        assert forwarded['timings_per_token'] is False and forwarded['return_progress'] is False
        return httpx.Response(200, stream=GatewayStream([
            sse({'choices': [{'delta': {'content': 'a whole response chunk'}, 'finish_reason': 'stop'}]}),
            sse('[DONE]')]))
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instances['one'].backend = backend
    try:
        async with client:
            response = await client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': True,
                'timings_per_token': False, 'return_progress': False})
            assert response.status_code == 200 and '[DONE]' in response.text
            record = controller.request_metrics[instances['one'].id]
            assert record['status'] == 'completed' and record['usage'] == record['timings'] == record['prefill'] == {}
    finally:
        await controller.close()


@pytest.mark.parametrize('backend', ['ninfer', 'llama_cpp'])
def test_request_stream_counts_do_not_guess_missing_counts_or_combine_multiple_choices(backend):
    from modloader.request_metrics import RequestMetrics, generation_window
    missing = RequestMetrics(True, backend=backend)
    missing.observe({'choices': [{'delta': {'content': 'arbitrarily many words in a chunk'}}]})
    _, view = generation_window(None, missing.token, missing.live_generation(), 1)
    assert view['tokens_per_second'] is None and view['context_tokens'] is None
    assert view['generated_tokens'] is None
    multiple = RequestMetrics(True, backend='vllm', single_choice=False)
    multiple.observe({'usage': {'prompt_tokens': 100, 'completion_tokens': 50}})
    _, view = generation_window(None, multiple.token, multiple.live_generation(), 1)
    assert view['request_key'] == multiple.token
    assert view['prompt_tokens'] == 100 and view['generated_tokens'] == 50
    assert view['tokens_per_second'] is None and view['context_tokens'] is None
    assert view['average_tokens_per_second'] is None and not view['available_for_chat']
    assert multiple.snapshot()['counters'] == {'input': 100, 'output': 50, 'cached': None}
    assert multiple.snapshot()['measurements']['decode']['reported'] == 0


@pytest.mark.parametrize('backend', ['ninfer', 'llama_cpp'])
def test_native_prefill_progress_and_mean_stay_separate_from_decode(backend):
    from modloader.request_metrics import RequestMetrics, generation_window, request_metrics_group
    metrics = RequestMetrics(True, backend=backend, client_request_id='assistant-one')
    state = None
    # NInfer 64d10fbf9f8e and llama.cpp b11609 share this native SSE shape.
    for second, processed in enumerate((0, 21, 28)):
        metrics.observe({'choices': [{'delta': {}, 'finish_reason': None, 'index': 0}], 'usage': None,
                         'prompt_progress': {'cache': 0, 'processed': processed, 'time_ms': second*34, 'total': 28}})
        state, view = generation_window(state, metrics.token, metrics.live_generation(), 10+second)
        assert view['request_key'] == metrics.token and view['phase'] == 'prefill'
        assert view['prefill_processed_tokens'] == processed and view['prefill_total_tokens'] == 28
        assert view['prefill_cached_tokens'] == 0
        assert view['tokens_per_second'] is None and view['average_tokens_per_second'] is None
        assert not view['available_for_chat']
    assert view['prefill_tokens_per_second'] == 14 and view['prefill_window_s'] == 2
    assert view['average_prefill_tokens_per_second'] == 14 and view['prefill_average_basis'] == 'sampled_prefill'
    timings = {'cache_n': 0, 'predicted_ms': 0, 'predicted_n': 1, 'predicted_per_second': 0,
               'prompt_ms': 93.509687, 'prompt_n': 28, 'prompt_per_second': 299.43421797572694}
    metrics.observe({'choices': [{'delta': {'reasoning_content': 'private reasoning'}}], 'timings': timings})
    state, thinking = generation_window(state, metrics.token, metrics.live_generation(), 13)
    assert thinking['phase'] == 'thinking' and thinking['generated_tokens'] == 1
    assert thinking['prefill_tokens_per_second'] is None and thinking['prefill_window_s'] is None
    assert thinking['average_prefill_tokens_per_second'] == timings['prompt_per_second']
    assert thinking['prefill_average_basis'] == 'backend_prefill'
    assert thinking['tokens_per_second'] is None and metrics.first_output_s is not None
    # Delayed progress must not re-enter prefill after the first reasoning token.
    metrics.observe({'prompt_progress': {'processed': 28, 'total': 28}, 'choices': [{'delta': {}}]})
    state, delayed = generation_window(state, metrics.token, metrics.live_generation(), 14)
    assert delayed['phase'] == 'thinking' and delayed['prefill_tokens_per_second'] is None
    assert delayed['tokens_per_second'] is None and delayed['window_s'] is None
    metrics.observe({'choices': [{'delta': {'content': 'private answer'}}], 'timings': {**timings, 'predicted_n': 5}})
    state, resumed = generation_window(state, metrics.token, metrics.live_generation(), 15)
    assert resumed['phase'] == 'output' and resumed['tokens_per_second'] is None
    # A progress-only frame broke decode continuity. Two new observations are
    # required; do not reuse its retained predicted_n as a middle sample.
    metrics.observe({'choices': [{'delta': {'content': 'private tail'}}], 'timings': {**timings, 'predicted_n': 7}})
    _, output = generation_window(state, metrics.token, metrics.live_generation(), 16)
    assert output['phase'] == 'output' and output['tokens_per_second'] == 2
    assert output['window_s'] == 1 and output['generated_tokens'] == 7
    record = metrics.finish('completed')
    group = request_metrics_group(record)
    values = {metric['id']: metric for metric in group['metrics']}
    assert group['request_key'] == metrics.token and group['request_status'] == 'completed'
    assert group['client_request_id'] == 'assistant-one'
    assert values['prefill_speed']['value'] == timings['prompt_per_second'] and values['prefill_speed']['rate_basis'] == 'prefill'
    assert values['output_speed']['rate_basis'] == 'decode'
    assert 'private ' not in json.dumps(record)


def test_llama_progress_with_per_token_timings_keeps_full_prompt_and_advances_from_zero():
    from modloader.request_metrics import RequestMetrics, generation_window
    metrics = RequestMetrics(True, backend='llama_cpp')
    state = None
    # b11609 server-context.cpp send_partial_response copies stats on every
    # progress frame; server-task.cpp attaches both fields to the chat chunk.
    for second, position, computed in ((10, 4096, 0), (11, 6144, 2048), (12, 8192, 4096)):
        timings = {'cache_n': 4096, 'prompt_n': computed, 'prompt_ms': (second-10)*1000,
                   'prompt_per_token_ms': 1000/2048 if computed else 0,
                   'prompt_per_second': 2048 if computed else 0,
                   'predicted_n': 0, 'predicted_ms': 0, 'predicted_per_token_ms': 0, 'predicted_per_second': 0}
        metrics.observe({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None}, 'finish_reason': None}],
                         'prompt_progress': {'total': 8192, 'cache': 4096, 'processed': position, 'time_ms': (second-10)*1000},
                         'timings': timings})
        state, view = generation_window(state, metrics.token, metrics.live_generation(), second)
        assert view['phase'] == 'prefill' and metrics.first_output_s is None
        assert view['prompt_tokens'] == view['prefill_total_tokens'] == view['context_tokens'] == 8192
        assert view['prefill_processed_tokens'] == computed and view['prefill_cached_tokens'] == 4096
        assert view['generated_tokens'] == 0 and view['tokens_per_second'] is None
        assert view['average_prefill_tokens_per_second'] == timings['prompt_per_second']
        assert view['prefill_average_basis'] == 'backend_prefill'
    assert view['prefill_tokens_per_second'] == 2048 and view['prefill_window_s'] == 2
    metrics.observe({'choices': [{'delta': {'content': 'several tokens in one chunk'}}],
                     'timings': {**timings, 'predicted_n': 16, 'predicted_ms': 2000, 'predicted_per_second': 8}})
    _, output = generation_window(state, metrics.token, metrics.live_generation(), 13)
    assert output['phase'] == 'output' and output['generated_tokens'] == 16
    assert output['prompt_tokens'] == 8192 and output['context_tokens'] == 8208
    assert output['prefill_processed_tokens'] == 4096 and output['average_tokens_per_second'] == 8


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('chat_path', ['/api/chat', '/v1/chat/completions'])
async def test_sparkinfer_gateway_retains_native_terminal_usage_without_fake_timings(controller, tmp_path, monkeypatch, streaming, chat_path):
    usage = {'prompt_tokens': 200, 'completion_tokens': 50, 'total_tokens': 250,
             'prompt_tokens_details': {'cached_tokens': 128}, 'ttft_ms': 120.5, 'generation_ms': 1000, 'decode_tps': 50}
    async def handler(request):
        if request.method == 'GET':
            if request.url.path == '/metrics':
                return httpx.Response(200, text='sparkinfer_active_requests 0\nsparkinfer_waiting_requests 0\n')
            return httpx.Response(200, json={'max_context': 8192} if request.url.path == '/v1/info' else {'status': 'ok'})
        forwarded = json.loads(request.content)
        assert 'timings_per_token' not in forwarded and 'return_progress' not in forwarded
        if streaming:
            assert forwarded['stream_options'] == {'include_usage': True}
            return httpx.Response(200, stream=GatewayStream([
                sse({'choices': [{'delta': {'content': 'many tokens delivered as one chunk'}}]}),
                sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}),
                sse({'choices': [], 'usage': usage}), sse('[DONE]')]))
        return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}], 'usage': usage})
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    instance.backend = 'sparkinfer'
    try:
        async with client:
            response = await client.post(chat_path, json={'model': 'one', 'messages': [], 'stream': streaming},
                                         headers={'X-Modloader-Call-Id': 'spark-call'})
            assert response.status_code == 200
            if streaming:
                chunks = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith('data:') and '[DONE]' not in line]
                assert next(chunk['usage'] for chunk in chunks if 'usage' in chunk) == usage
                reports = [chunk['modloader_request'] for chunk in chunks if 'modloader_request' in chunk]
                assert len(reports) == (1 if chat_path == '/api/chat' else 0)
                report = reports[0] if reports else controller.request_metrics[instance.id]
            else:
                body = response.json()
                assert body['usage'] == usage and ('modloader_request' in body) == (chat_path == '/api/chat')
                report = body.get('modloader_request', controller.request_metrics[instance.id])
            assert report['status'] == 'completed' and report['usage'] == usage
            assert report['timings'] == report['prefill'] == {}
            if chat_path == '/api/chat':
                retained = (await client.get(f'/api/instances/{instance.id}/request-metrics/spark-call')).json()
                assert retained == report
            else:
                assert report['client_call_id'] is None
            groups = (await client.get(f'/api/instances/{instance.id}/telemetry')).json()['groups']
            metrics = {metric['id']: metric for metric in groups[0]['metrics']}
            assert metrics['output_speed']['value'] == 50 and metrics['output_speed']['rate_basis'] == 'end_to_end'
            assert metrics['backend_average_tps']['value'] == 50 and metrics['backend_average_tps']['unit'] == 'token/s'
            assert metrics['backend_average_tps']['basis'] == 'usage.decode_tps'
            assert metrics['backend_ttft_ms']['value'] == 120.5 and metrics['backend_generation_ms']['value'] == 1000
    finally:
        await controller.close()


@pytest.mark.parametrize('status', ['completed', 'failed', 'cancelled', 'incomplete'])
@pytest.mark.parametrize('native', [dict(ttft_ms=120.5, generation_ms=1000, decode_tps=50),
                                   dict(ttft_ms=0, generation_ms=0, decode_tps=0)])
def test_sparkinfer_native_usage_keeps_gateway_clocks_and_terminal_rate_basis(monkeypatch, status, native):
    from types import SimpleNamespace
    import modloader.request_metrics as request_module
    clock = [0.0]
    monkeypatch.setattr(request_module, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000+clock[0]))
    metrics = request_module.RequestMetrics(True, backend='sparkinfer')
    clock[0] = 2
    metrics.observe({'choices': [{'delta': {'content': 'answer'}}]})
    clock[0] = 5
    metrics.observe({'usage': {'prompt_tokens': 200, 'completion_tokens': 50, 'total_tokens': 250, **native}})
    record = metrics.finish(status)
    assert record['duration_s'] == 5 and record['first_output_s'] == 2
    assert record['timings'] == {} and {key: record['usage'][key] for key in native} == native
    group = {metric['id']: metric for metric in request_module.request_metrics_group(record)['metrics']}
    assert group['duration_s']['value'] == 5 and group['duration_s']['unit'] == 's'
    assert group['first_output_s']['value'] == 2 and group['first_output_s']['unit'] == 's'
    for field in ('ttft_ms', 'generation_ms'):
        metric = group['backend_' + field]
        assert metric['value'] == native[field] and metric['unit'] == 'ms' and metric['basis'] == 'usage.' + field
    speed = group['output_speed']
    assert speed['value'] == (native['decode_tps'] if status == 'completed' else None)
    assert speed['rate_basis'] == ('end_to_end' if status == 'completed' else None)
    if status == 'completed':
        assert speed['label'] == '后端平均速度' and '含首 token 等待' in speed['description']
    average = group['backend_average_tps']
    assert average['value'] == (native['decode_tps'] if status == 'completed' else None)
    assert average['unit'] == 'token/s' and average['basis'] == 'usage.decode_tps'
    assert average['rate_basis'] == 'end_to_end'


@pytest.mark.parametrize('timings, expected', [
    ({'predicted_per_second': 80, 'predicted_n': 50, 'predicted_ms': 2000}, 80),
    ({'predicted_n': 50, 'predicted_ms': 2000}, 25),
    ({'predicted_per_second': 0}, 0),
])
def test_request_metrics_prefers_native_decode_timings_over_sparkinfer_average(timings, expected):
    from modloader.request_metrics import RequestMetrics, request_metrics_group
    metrics = RequestMetrics(True, backend='llama_cpp')
    metrics.observe({'usage': {'completion_tokens': 50, 'generation_ms': 1000, 'decode_tps': 50},
                     'timings': timings})
    group = {metric['id']: metric for metric in request_metrics_group(metrics.finish('completed'))['metrics']}
    assert group['output_speed']['value'] == expected and group['output_speed']['rate_basis'] == 'decode'
    assert group['backend_average_tps']['value'] == 50 and group['backend_average_tps']['rate_basis'] == 'end_to_end'


def test_sparkinfer_terminal_usage_does_not_turn_into_a_live_sample():
    from modloader.request_metrics import RequestMetrics, generation_window
    metrics = RequestMetrics(True, backend='sparkinfer')
    metrics.observe({'choices': [{'delta': {'content': 'one complete chunk'}}]})
    metrics.observe({'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 50,
                     'ttft_ms': 100, 'generation_ms': 1000, 'decode_tps': 50}})
    state = None
    for sampled_at in (10, 11):
        state, live = generation_window(state, metrics.token, metrics.live_generation(), sampled_at)
        assert live['generated_tokens'] is None and live['tokens_per_second'] is None
        assert live['average_tokens_per_second'] is None and not live['available_for_chat']
    assert metrics.finish('completed')['usage']['completion_tokens'] == 50


@pytest.mark.parametrize('native', [{}, {'ttft_ms': -1, 'generation_ms': float('nan'), 'decode_tps': True}])
def test_sparkinfer_missing_or_invalid_native_usage_keeps_existing_final_fallback(native):
    from modloader.request_metrics import RequestMetrics, request_metrics_group
    metrics = RequestMetrics(False, backend='sparkinfer')
    metrics.observe({'usage': {'completion_tokens': 20, **native}})
    record = {**metrics.finish('completed'), 'duration_s': 5}
    assert record['usage'] == {'completion_tokens': 20} and record['timings'] == {}
    group = {metric['id']: metric for metric in request_metrics_group(record)['metrics']}
    assert not {'backend_ttft_ms', 'backend_generation_ms', 'backend_average_tps'} & set(group)
    assert group['output_speed']['value'] == 4 and group['output_speed']['rate_basis'] == 'end_to_end'
    assert group['output_speed']['label'] == '端到端平均速度'


@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
async def test_gateway_preserves_explicitly_disabled_continuous_usage(controller, tmp_path, monkeypatch, backend):
    async def handler(request):
        forwarded = json.loads(request.content)
        assert forwarded['stream_options'] == {'include_usage': False, 'continuous_usage_stats': False}
        return httpx.Response(200, stream=GatewayStream([
            sse({'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'stop'}]}), sse('[DONE]')]))
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instances['one'].backend = backend
    try:
        async with client:
            response = await client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': True,
                'stream_options': {'include_usage': False, 'continuous_usage_stats': False}})
            assert response.status_code == 200 and '[DONE]' in response.text
            assert controller.request_metrics[instances['one'].id]['usage'] == {}
    finally:
        await controller.close()


async def test_gateway_strata_progress_preserves_computed_cache_and_native_decode_counters(controller, tmp_path, monkeypatch):
    from starlette.requests import Request
    from types import SimpleNamespace
    import modloader.app as app_module
    clock = [0.0]
    monkeypatch.setattr(app_module, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000+clock[0]))
    native = {'engine': {'max_context': 4096}, 'live': {
        'state': 'reading', 'running': 1, 'queued': 0, 'prompt_tokens': 1024,
        'prompt_read': 640, 'prompt_total': 1024, 'generated': 0, 'prefill_tok_s_mean': 128}}
    # Strata v0.1.43 reports a position including the cached prompt prefix in
    # both SSE processed and /metrics live.prompt_read.
    stream = GatewayStream([
        sse({'choices': [{'delta': {}}], 'prompt_progress': {'cache': 640, 'processed': 640, 'total': 1024, 'time_ms': 0}}),
        sse({'choices': [{'delta': {}}], 'prompt_progress': {'cache': 640, 'processed': 896, 'total': 1024, 'time_ms': 1000}}),
        sse({'choices': [{'delta': {'reasoning_content': 'private thought'}}]}),
        sse({'choices': [{'delta': {'content': 'private answer'}}]}),
        sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}), sse('[DONE]')])
    async def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'loaded': True} if request.url.path == '/health' else native)
        forwarded = json.loads(request.content)
        assert forwarded['return_progress'] is True and 'timings_per_token' not in forwarded
        return httpx.Response(200, stream=stream)
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    instance.backend = 'strata'
    response = None
    try:
        async with client:
            endpoint = next(route.endpoint for route in client._transport.app.routes if route.path == '/api/chat')
            async def receive():
                return {'type': 'http.request', 'body': json.dumps({'model': 'one', 'messages': [], 'stream': True}).encode(), 'more_body': False}
            response = await endpoint(Request({'type': 'http', 'method': 'POST', 'path': '/api/chat',
                'headers': [(b'x-modloader-request-id', b'strata-assistant')]}, receive))
            async def advance():
                while True:
                    if (await response.body_iterator.__anext__()).startswith('data:'):
                        return
            async def sample():
                return (await client.get('/api/instances/'+instance.id+'/telemetry')).json()['live_generation']
            await advance()
            first = await sample()
            assert first['phase'] == 'prefill' and first['client_request_id'] == 'strata-assistant'
            assert first['prompt_tokens'] == first['prefill_total_tokens'] == 1024
            assert first['prefill_processed_tokens'] == 0 and first['prefill_cached_tokens'] == 640
            assert first['prefill_tokens_per_second'] is None
            clock[0] = 1
            native['live']['prompt_read'] = 896
            await advance()
            prefill = await sample()
            assert prefill['prefill_processed_tokens'] == 256 and prefill['prefill_cached_tokens'] == 640
            assert prefill['prefill_tokens_per_second'] == 256 and prefill['prefill_window_s'] == 1
            assert prefill['average_prefill_tokens_per_second'] == 128 and prefill['prefill_average_basis'] == 'backend_prefill'
            clock[0] = 2
            native['live'].update(state='generating', prompt_read=None, prompt_total=None, generated=5, tok_s_mean=10)
            await advance()
            thinking = await sample()
            assert thinking['phase'] == 'thinking' and thinking['generated_tokens'] == 5
            assert thinking['context_tokens'] == 1029 and thinking['average_tokens_per_second'] == 10
            assert thinking['average_basis'] == 'backend_decode' and thinking['prefill_tokens_per_second'] is None
            clock[0] = 3
            native['live'].update(generated=25, tok_s_mean=15)
            await advance()
            output = await sample()
            assert output['phase'] == 'output' and output['generated_tokens'] == 25
            assert output['tokens_per_second'] == 20 and output['average_tokens_per_second'] == 15
            assert output['prefill_processed_tokens'] == 256 and output['prefill_cached_tokens'] == 640
            clock[0] = 4
            native['live'].update(running=2, queued=1)
            concurrent = await sample()
            assert concurrent['request_key'] == first['request_key']
            assert concurrent['generated_tokens'] is None and concurrent['average_tokens_per_second'] is None
            assert concurrent['prefill_processed_tokens'] == 256 and concurrent['prefill_cached_tokens'] == 640
            instance.active_requests = 2
            ambiguous = await sample()
            assert ambiguous['request_key'] is None and ambiguous['prefill_cached_tokens'] is None
            instance.active_requests = 1
            async for _ in response.body_iterator:
                pass
            record = controller.request_metrics[instance.id]
            assert record['status'] == 'completed' and record['usage'] == record['timings'] == {}
            assert record['prefill']['prefill_processed_tokens'] == 256 and record['prefill']['prefill_cached_tokens'] == 640
    finally:
        if response is not None:
            await response.body_iterator.aclose()
        await controller.close()


@pytest.mark.parametrize('backend', ['ninfer', 'llama_cpp', 'strata'])
@pytest.mark.parametrize('cache,position,computed', [(None, 4096, None), (0, 0, 0), (0, 21, 21), (4096, 4096, 0), (4096, 6144, 2048)])
def test_native_progress_keeps_missing_cache_distinct_from_zero_and_does_not_invent_usage(backend, cache, position, computed):
    from modloader.request_metrics import RequestMetrics, generation_window
    metrics = RequestMetrics(True, backend=backend)
    progress = {'processed': position, 'total': 8192, **({'cache': cache} if cache is not None else {})}
    metrics.observe({'choices': [{'delta': {}}], 'prompt_progress': progress})
    _, view = generation_window(None, metrics.token, metrics.live_generation(), 10)
    assert view['phase'] == 'prefill' and view['prefill_processed_tokens'] == computed
    assert view['prefill_cached_tokens'] == cache and view['prompt_tokens'] == 8192
    assert view['generated_tokens'] == (None if backend == 'strata' else 0)
    assert view['average_tokens_per_second'] is None and view['tokens_per_second'] is None
    record = metrics.snapshot()
    assert record['usage'] == record['timings'] == {}
    assert ('prefill_cached_tokens' in record['prefill']) == (cache is not None)
    assert ('prefill_processed_tokens' in record['prefill']) == (computed is not None)


@pytest.mark.parametrize('streaming,single_choice', [(True, True), (True, False), (False, True)])
def test_request_identity_survives_missing_backend_and_zero_output(streaming, single_choice):
    from modloader.request_metrics import RequestMetrics, generation_window
    metrics = RequestMetrics(streaming, backend='custom', single_choice=single_choice)
    metrics.observe({'choices': [{'delta': {'role': 'assistant', 'content': ''}}]})
    _, view = generation_window(None, metrics.token, metrics.live_generation(), 10)
    assert view['request_key'] == metrics.token and view['elapsed_s'] >= 0
    assert view['phase'] == 'waiting' and view['generated_tokens'] is None
    assert view['tokens_per_second'] is None and view['prefill_tokens_per_second'] is None


def test_backend_reading_without_progress_is_waiting_and_cannot_regress_decode():
    from modloader.request_metrics import RequestMetrics, generation_window
    metrics = RequestMetrics(True, backend='strata')
    live = {'scope': 'model', 'source': '/metrics live.generated', 'backend_running': 1, 'backend_queued': 0,
            '_generated_tokens': None, '_prefill_source': '/metrics live.prompt_read',
            '_overview': {'phase': 'waiting', 'prompt_tokens': 100}}
    state, waiting = generation_window(None, metrics.token, metrics.live_generation(live), 10)
    assert waiting['phase'] == 'waiting' and waiting['prompt_tokens'] == 100
    state, decode = generation_window(state, metrics.token, metrics.live_generation({**live,
        '_overview': {'phase': 'generating', 'generated_tokens': 0}}), 11)
    assert decode['phase'] == 'generating' and decode['tokens_per_second'] is None
    _, delayed = generation_window(state, metrics.token, metrics.live_generation({**live,
        '_overview': {'phase': 'prefill', 'prefill_processed_tokens': 20}}), 12)
    assert delayed['phase'] == 'generating' and delayed['prefill_tokens_per_second'] is None


def test_prefill_window_restarts_on_counter_source_reset_and_long_gap():
    from modloader.request_metrics import generation_window
    live = {'scope': 'request', 'source': 'SSE prompt_progress.processed', '_generated_tokens': 0,
            '_prefill_source': 'SSE prompt_progress.processed'}
    state = None
    for second, processed in ((10, 0), (11, 30), (15, 60)):
        state, view = generation_window(state, 'one', {**live, '_overview': {
            'phase': 'prefill', 'prefill_processed_tokens': processed}}, second)
    assert view['prefill_tokens_per_second'] is None and view['prefill_window_s'] is None
    assert view['average_prefill_tokens_per_second'] == 12
    state, resumed = generation_window(state, 'one', {**live, '_overview': {
        'phase': 'prefill', 'prefill_processed_tokens': 75}}, 16)
    assert resumed['prefill_tokens_per_second'] == 15 and resumed['prefill_window_s'] == 1
    state, reset = generation_window(state, 'one', {**live, '_overview': {
        'phase': 'prefill', 'prefill_processed_tokens': 2}}, 17)
    assert reset['prefill_tokens_per_second'] is None and reset['average_prefill_tokens_per_second'] is None
    _, changed = generation_window(state, 'one', {**live, '_prefill_source': 'another counter', '_overview': {
        'phase': 'prefill', 'prefill_processed_tokens': 100}}, 18)
    assert changed['prefill_tokens_per_second'] is None and changed['average_prefill_tokens_per_second'] is None


@pytest.mark.parametrize('restore_original', [False,True])
async def test_telemetry_discards_a_snapshot_if_request_lifecycle_changes_during_fetch(controller,tmp_path,monkeypatch,restore_original):
    import modloader.backends as backends
    from modloader.request_metrics import RequestMetrics
    entered, release = asyncio.Event(), asyncio.Event()
    async def collect(instance,spec):
        entered.set()
        await release.wait()
        return {'groups':[],'unavailable':[], 'live_generation':{'scope':'model','source':'/metrics live.generated',
                'backend_running':1,'backend_queued':0,'_generated_tokens':100,'_elapsed_s':1.0,'_slot_id':None}}
    monkeypatch.setattr(backends,'collect_telemetry',collect)
    async def unused(request):
        raise AssertionError('backend chat was not requested')
    client, instances, _ = gateway_client(controller,tmp_path,monkeypatch,unused)
    instance=instances['one']
    instance.active_requests=1
    first, second = RequestMetrics(True, backend='strata'), RequestMetrics(True, backend='strata')
    controller.active_request_metrics.update({first.token: first, second.token: second})
    controller.active_request_tokens[instance.id]=frozenset({first.token})
    try:
        async with client:
            task=asyncio.create_task(client.get('/api/instances/'+instance.id+'/telemetry'))
            await asyncio.wait_for(entered.wait(),5)
            controller.active_request_tokens[instance.id]=frozenset({first.token,second.token})
            current = first if restore_original else second
            controller.active_request_tokens[instance.id]=frozenset({current.token})
            release.set()
            result=(await task).json()
            assert not result['live_generation']['available_for_chat'] and result['live_generation']['tokens_per_second'] is None
            assert result['live_generation']['request_key'] == current.token and result['live_generation']['phase'] == 'waiting'
            assert result['live_generation']['generated_tokens'] is None
            assert instance.id not in controller.generation_samples
    finally:
        instance.active_requests=0
        await controller.close()


@pytest.mark.parametrize('platform,guest_pid', [('wsl2',4321),('wsl2',None),('windows',4321)])
async def test_telemetry_process_group_distinguishes_host_and_linux_service_pid(controller,tmp_path,monkeypatch,platform,guest_pid):
    from types import SimpleNamespace
    import modloader.backends as backends
    async def collect(instance,spec):
        return {'groups':[], 'unavailable':[]}
    async def unused(request):
        raise AssertionError('backend chat was not requested')
    def process(pid):
        assert pid == 12345  # Never treat a Linux service PID as a Windows process ID.
        return SimpleNamespace(children=lambda recursive:[], memory_info=lambda:SimpleNamespace(rss=4096), is_running=lambda:True)
    monkeypatch.setattr(backends,'collect_telemetry',collect)
    monkeypatch.setattr('modloader.app.psutil.Process',process)
    client,instances,_=gateway_client(controller,tmp_path,monkeypatch,unused)
    instance=instances['one']
    instance.platform=platform
    instance.distribution='模型 Ubuntu' if platform=='wsl2' else None
    instance.pid,instance.guest_pid=12345,guest_pid
    try:
        async with client:
            response=await client.get('/api/instances/'+instance.id+'/telemetry')
            assert response.status_code==200
            result=response.json()
            group=next(group for group in result['groups'] if group['id']=='process')
            metrics={metric['id']:metric for metric in group['metrics']}
            assert metrics['pid']['value']==12345 and metrics['requests']['value']==0
            if platform=='wsl2':
                assert metrics['platform']['value']=='wsl2' and metrics['distribution']['value']=='模型 Ubuntu'
                assert metrics['pid']['label']=='宿主启动进程 PID'
                assert metrics['guest_pid']=={'id':'guest_pid','label':'Linux 服务进程 PID','value':guest_pid,'kind':'text'}
                assert 'rss' not in metrics
                assert any('Linux guest' in message and '内存' in message for message in result['unavailable'])
            else:
                assert list(metrics)==['pid','requests','rss']
                assert metrics['pid']['label']=='托管进程 PID' and metrics['rss']['value']==4096
    finally:
        await controller.close()


@pytest.mark.parametrize('all_finished', [True,False])
async def test_gateway_finish_reason_at_eof_requires_all_choices(controller,tmp_path,monkeypatch,all_finished):
    async def handler(request):
        chunks = [sse({'choices':[{'index':0,'delta':{'content':'one'},'finish_reason':'stop'},
                                  {'index':1,'delta':{'content':'two'},'finish_reason':'stop' if all_finished else None}]}),
                  sse({'choices':[],'timings':{'predicted_n':4,'predicted_ms':200}})]
        return httpx.Response(200,stream=GatewayStream(chunks))
    client, instances, _ = gateway_client(controller,tmp_path,monkeypatch,handler)
    instance = instances['one']
    try:
        async with client:
            response = await client.post('/api/chat',json={'model':'one','messages':[],'stream':True,'n':2})
            assert response.status_code==200 and '[DONE]' not in response.text
            record = controller.request_metrics[instance.id]
            assert record['status']==('completed' if all_finished else 'incomplete')
            from modloader.request_metrics import request_metrics_group
            metrics = {metric['id']:metric for metric in request_metrics_group(record)['metrics']}
            assert record['usage'] == record['timings'] == {}
            assert record['counters']['output'] is None and record['measurements']['decode']['reported'] == 0
            assert metrics['completion_tokens']['value'] is None
            assert 'predicted_n' not in metrics['completion_tokens']['description']
            assert metrics['output_speed']['value'] is None
    finally:
        await controller.close()


async def test_telemetry_shared_samples_timer_jitter_and_current_request_ownership(controller,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import modloader.app as app_module
    import modloader.backends as backends
    clock, calls = [0.0], []
    counter = [100]
    backend_counts = {'running':1,'queued':0}
    monkeypatch.setattr(app_module,'time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:1000+clock[0]))
    sample_data = {'groups':[{'id':'strata_live','title':'当前请求与吞吐','metrics':[]}], 'unavailable':[],
                   'live_generation':{'tokens_per_second':None,'window_s':None,'scope':'model','source':'/metrics live.generated',
                                      'backend_running':1,'backend_queued':0,'available_for_chat':False,'reason':None,
                                      '_generated_tokens':100,'_elapsed_s':None,'_slot_id':None}}
    async def collect(instance,spec):
        calls.append(instance.id)
        await asyncio.sleep(0)
        return {**sample_data,'live_generation':{**sample_data['live_generation'],
                                               'backend_running':backend_counts['running'],'backend_queued':backend_counts['queued'],
                                               '_generated_tokens':counter[0]}}
    monkeypatch.setattr(backends,'collect_telemetry',collect)
    async def unused(request):
        raise AssertionError('backend chat was not requested')
    client, instances, _ = gateway_client(controller,tmp_path,monkeypatch,unused,('one','two'))
    first, second = instances['one'], instances['two']
    first.active_requests=1
    controller.active_request_tokens[first.id]={'request-one'}
    try:
        async with client:
            endpoint='/api/instances/'+first.id+'/telemetry'
            responses=await asyncio.gather(client.get(endpoint),client.get(endpoint))
            payloads=[response.json() for response in responses]
            assert calls==[first.id] and all(not payload['live_generation']['available_for_chat'] for payload in payloads)
            assert payloads[0]['updated_at']==payloads[1]['updated_at']==1000
            assert all(sum(group['id']=='process' for group in payload['groups'])==1 for payload in payloads)
            assert len(sample_data['groups'])==1  # Cache entries never acquire dynamic groups.
            # A 1s timer firing slightly early still creates a fresh backend sample.
            clock[0]=0.95
            counter[0]=150
            measured=(await client.get(endpoint)).json()
            assert measured['live_generation']['available_for_chat']
            assert measured['live_generation']['tokens_per_second']==pytest.approx(50/.95)
            assert measured['live_generation']['average_tokens_per_second']==pytest.approx(50/.95)
            assert measured['live_generation']['average_basis']=='sampled_decode'
            assert measured['live_generation']['window_s']==pytest.approx(.95)
            assert calls==[first.id,first.id] and measured['updated_at']==1000.95
            assert not any(key.startswith('_') for key in measured['live_generation'])
            first.active_requests=2
            clock[0]=1.2
            ambiguous=(await client.get(endpoint)).json()
            assert not ambiguous['live_generation']['available_for_chat'] and '唯一' in ambiguous['live_generation']['reason']
            assert ambiguous['live_generation']['tokens_per_second'] is None
            assert ambiguous['live_generation']['average_tokens_per_second'] is None
            assert ambiguous['live_generation']['average_basis'] is None
            assert calls==[first.id,first.id] and ambiguous['updated_at']==1000.95
            first.active_requests=1
            clock[0]=1.95
            counter[0]=200
            resumed=(await client.get(endpoint)).json()
            assert not resumed['live_generation']['available_for_chat']  # Reset after concurrency.
            assert resumed['live_generation']['average_tokens_per_second'] is None
            clock[0]=2.95
            counter[0]=250
            steady=(await client.get(endpoint)).json()
            assert steady['live_generation']['tokens_per_second']==pytest.approx(50)
            assert steady['live_generation']['average_tokens_per_second']==pytest.approx(50)
            clock[0]=3.95
            backend_counts['running']=2
            busy=(await client.get(endpoint)).json()
            assert calls==[first.id]*5 and busy['updated_at']==1003.95
            assert not busy['live_generation']['available_for_chat'] and '并发' in busy['live_generation']['reason']
            other=(await client.get('/api/instances/'+second.id+'/telemetry')).json()
            assert calls==[first.id]*5+[second.id] and not other['live_generation']['available_for_chat']
            first.state='stopped'
            stopped=(await client.get(endpoint)).json()
            assert stopped['live_generation'] is None and first.id not in controller.telemetry_samples
    finally:
        first.active_requests=0
        await controller.close()


async def test_telemetry_missing_sample_breaks_rolling_rate_then_recovers(controller,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import modloader.app as app_module
    import modloader.backends as backends
    from modloader.request_metrics import RequestMetrics
    clock, counter, missing = [0.0], [100], [False]
    monkeypatch.setattr(app_module,'time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:1000+clock[0]))
    async def collect(instance,spec):
        if missing[0]:
            return {'groups':[],'unavailable':['/metrics 连接不可用']}
        return {'groups':[],'unavailable':[], 'live_generation':{'scope':'model','source':'/metrics live.generated',
                'backend_running':1,'backend_queued':0,'_generated_tokens':counter[0],'_slot_id':None}}
    monkeypatch.setattr(backends,'collect_telemetry',collect)
    async def unused(request):
        raise AssertionError('backend chat was not requested')
    client, instances, _ = gateway_client(controller,tmp_path,monkeypatch,unused)
    instance = instances['one']
    instance.active_requests = 1
    request = RequestMetrics(True, backend='strata')
    controller.active_request_metrics[request.token] = request
    controller.active_request_tokens[instance.id] = frozenset({request.token})
    try:
        async with client:
            path = '/api/instances/'+instance.id+'/telemetry'
            assert (await client.get(path)).json()['live_generation']['tokens_per_second'] is None
            clock[0], counter[0] = 1, 140
            measured = (await client.get(path)).json()['live_generation']
            assert measured['tokens_per_second'] == measured['average_tokens_per_second'] == 40
            clock[0], missing[0] = 2, True
            failed = (await client.get(path)).json()
            assert failed['unavailable'] == ['/metrics 连接不可用']
            assert failed['live_generation']['request_key'] == request.token
            assert failed['live_generation']['tokens_per_second'] is None and failed['live_generation']['generated_tokens'] is None
            assert controller.generation_samples[instance.id]['samples'] == []
            clock[0], counter[0], missing[0] = 3, 200, False
            resumed = (await client.get(path)).json()['live_generation']
            assert resumed['tokens_per_second'] is None and not resumed['available_for_chat']
            assert resumed['window_s'] is None and resumed['average_tokens_per_second'] == pytest.approx(100/3)
            clock[0], counter[0] = 4, 210
            steady = (await client.get(path)).json()['live_generation']
            assert steady['tokens_per_second'] == 10 and steady['window_s'] == 1
            assert steady['average_tokens_per_second'] == 27.5
            # Ending the request invalidates even a cached positive window.
            instance.active_requests = 0
            controller.active_request_tokens[instance.id] = frozenset()
            stopped = (await client.get(path)).json()['live_generation']
            assert stopped['tokens_per_second'] is None and stopped['window_s'] is None
            assert stopped['request_key'] is None and instance.id not in controller.generation_samples
    finally:
        instance.active_requests = 0
        await controller.close()


async def test_gpu_sampling_error_is_not_a_device_name_and_recovers(controller, monkeypatch):
    from types import SimpleNamespace
    identities = [{'device': 'CUDA0', 'gpu_id': 'GPU-second', 'pci_bus_id': '0000:02:00.0'},
                  {'device': 'CUDA1', 'gpu_id': 'GPU-first', 'pci_bus_id': '0000:01:00.0'}]
    monkeypatch.setattr('modloader.app.GpuProbe', lambda: SimpleNamespace(lib=None, cuda_devices=identities, close=lambda: None))
    samples = []
    class Process:
        def __init__(self, code, output):
            self.returncode, self.output = code, output
        async def communicate(self):
            return self.output, b''
    processes = iter([Process(1, b'Failed to initialize NVML: Unknown Error'),
                      Process(0, b'0, GPU-first, GPU fixture, 32768, 8192, 24576, 23, 51, 120, 00000000:01:00.0\n'
                                 b'1, GPU-second, GPU fixture, 24576, 4096, 20480, 0, 0, 0, 00000000:02:00.0')])
    async def spawn(*args, **kwargs):
        return next(processes)
    async def sleep(_):
        samples.append(dict(controller.hardware))
        if len(samples) == 2:
            raise asyncio.CancelledError
    monkeypatch.setattr('modloader.app.shutil.which', lambda _: 'nvidia-smi')
    monkeypatch.setattr('modloader.app.asyncio.create_subprocess_exec', spawn)
    monkeypatch.setattr('modloader.app.asyncio.sleep', sleep)
    with pytest.raises(asyncio.CancelledError):
        await controller.metrics()
    assert samples[0]['gpu'] is None
    assert samples[0]['gpu_error'] == 'Failed to initialize NVML: Unknown Error'
    assert samples[1]['gpu']['name'] == 'GPU fixture'
    assert samples[1]['gpu']['memory_used'] == 8192 * 1024 ** 2
    assert [gpu['gpu_id'] for gpu in samples[1]['gpus']] == ['GPU-first', 'GPU-second']
    assert [gpu['device'] for gpu in samples[1]['gpus']] == ['CUDA1', 'CUDA0']
    assert samples[1]['gpus'][1]['memory_free'] == 20480 * 1024 ** 2
    assert samples[1]['gpus'][1]['power'] == 0 and samples[1]['gpus'][1]['temperature'] == 0
    assert samples[1]['gpus'][1]['temperature_source'].startswith('nvidia-smi ·')
    assert 'gpu_error' not in samples[1]


async def test_failed_hardware_poll_clears_dynamic_values(controller, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('modloader.app.GpuProbe', lambda: SimpleNamespace(lib=None, close=lambda: None))
    monkeypatch.setattr('modloader.app.system_specs', lambda: {})
    def fail():
        raise OSError('fixture sensor failure')
    monkeypatch.setattr('modloader.app.psutil.virtual_memory', fail)
    controller.hardware.update(cpu_percent=50, memory_used=1234, disk_read_bytes_per_second=123,
                               gpus=[{'id':'gpu', 'name':'device', 'utilization':90, 'pcie_rx_bytes_s':999}])
    async def stop(_):
        raise asyncio.CancelledError
    monkeypatch.setattr('modloader.app.asyncio.sleep', stop)
    with pytest.raises(asyncio.CancelledError):
        await controller.metrics()
    assert controller.hardware['cpu_percent'] is None
    assert controller.hardware['memory_used'] is None
    assert controller.hardware['disk_read_bytes_per_second'] is None
    assert controller.hardware['gpu']['id'] == 'gpu'
    assert controller.hardware['gpu']['utilization'] is None
    assert controller.hardware['gpu']['pcie_rx_bytes_s'] is None


@pytest.mark.parametrize('streaming', [False, True])
async def test_chat_final_request_report_keeps_short_calls_and_cumulative_usage(controller, tmp_path, monkeypatch, streaming):
    tail = {'choices': [{'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105,
                      'prompt_tokens_details': {'cached_tokens': 20, 'private': 'not retained'},
                      'completion_tokens_details': {'reasoning_tokens': 2}},
            'timings': {'predicted_n': 5, 'predicted_ms': 50, 'prompt_n': 80, 'cache_n': 20, 'prompt_ms': 20}}
    async def handler(request):
        if not streaming:
            return httpx.Response(200, json=tail)
        return httpx.Response(200, stream=GatewayStream([
            sse({'choices': [{'delta': {'content': 'done'}}], 'usage': {'completion_tokens': 2}}),
            sse(tail), sse('[DONE]')]))
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    try:
        async with client:
            response = await client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': streaming},
                                         headers={'X-Modloader-Request-Id': 'turn-short', 'X-Modloader-Call-Id': 'call-short'})
            assert response.status_code == 200
            if streaming:
                chunks = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith('data:') and line[5:].strip() != '[DONE]']
                reports = [chunk['modloader_request'] for chunk in chunks if 'modloader_request' in chunk]
                assert len(reports) == 1
                report = reports[0]
                assert response.text.index('modloader_request') < response.text.index('[DONE]')
            else:
                report = response.json()['modloader_request']
            assert report['status'] == 'completed'
            assert report['client_request_id'] == 'turn-short' and report['client_call_id'] == 'call-short'
            assert report['usage']['completion_tokens'] == 5
            assert report['usage']['prompt_tokens_details'] == {'cached_tokens': 20}
            assert report['usage']['completion_tokens_details'] == {'reasoning_tokens': 2}
            assert report['timings']['predicted_ms'] == 50 and report['timings']['cache_n'] == 20
            retained = await client.get(f'/api/instances/{instance.id}/request-metrics/call-short')
            assert retained.status_code == 200 and retained.json() == report
            assert (await client.get(f'/api/instances/{instance.id}/request-metrics/not-this-call')).status_code == 404
            # The original single recent slot remains bounded; chat messages own
            # the ledger, rather than a second backend request-history service.
            next_response = await client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': streaming},
                                             headers={'X-Modloader-Call-Id': 'call-next'})
            assert next_response.status_code == 200
            assert (await client.get(f'/api/instances/{instance.id}/request-metrics/call-short')).status_code == 404
    finally:
        await controller.close()
