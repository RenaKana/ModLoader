import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.backends import get_adapters, import_runtime
from modloader.models import LaunchContext, LoadProfile, ModelAsset, RuntimeInstallation
from modloader.parameter_rules import generation_fields, profile_fields


SERVER = Path(__file__).parent / 'fixtures/custom_engine_server.py'


def configuration(**changes):
    return {**{'args': [str(SERVER), '--model', '{model_path}', '--host', '{host}', '--port', '{port}',
                       '--api-model-name', 'upstream-{alias}'], 'api_model_name': 'upstream-{alias}'}, **changes}


def engine(tmp_path, **changes):
    return import_runtime('custom', sys.executable, {'custom': configuration(**changes)})


def profile_for(tmp_path, runtime, **changes):
    path = tmp_path / 'arbitrary weights.bin'
    path.write_bytes(b'not loaded as weights')
    model = ModelAsset(id='model', name='Custom model', path=str(path), format='custom')
    profile = LoadProfile(id='profile', name='Custom profile', model_id=model.id, backend='custom', runtime_id=runtime.id, alias='custom-alias', **changes)
    return model, profile


@pytest.fixture
async def api(tmp_path):
    controller = Controller(tmp_path / 'isolated-data', 'management-secret')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer management-secret'}) as client:
        yield controller, client
    await controller.close()


async def register(client, **changes):
    response = await client.post('/api/runtimes/import', json={'backend': 'custom', 'path': sys.executable,
                                                            'name': 'Fixture custom engine', 'metadata': {'custom': configuration(**changes)}})
    assert response.status_code == 200, response.text
    return RuntimeInstallation(**response.json())


async def create_profile(controller, client, tmp_path, runtime, **changes):
    model, profile = profile_for(tmp_path, runtime, **changes)
    controller.store.put('models', model.model_dump())
    response = await client.post('/api/profiles', json=profile.model_dump())
    assert response.status_code == 200, response.text
    return model, LoadProfile(**response.json())


async def wait_state(instance, expected):
    async with asyncio.timeout(10):
        while instance.state in ('starting', 'stopping'):
            await asyncio.sleep(.025)
    assert instance.state == expected, instance.error


async def test_registration_edit_and_refresh_never_execute_program_and_keep_references(api, tmp_path, monkeypatch):
    controller, client = api
    def forbid(*args, **kwargs):
        raise AssertionError('Registration/refresh must never execute a custom program')
    monkeypatch.setattr(subprocess, 'run', forbid)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbid)
    runtime = await register(client)
    second = await register(client)
    assert runtime.id != second.id  # A shared Python may serve multiple engines.
    _, profile = await create_profile(controller, client, tmp_path, runtime)
    for name in ('Renamed', 'Renamed again'):
        response = await client.patch('/api/runtimes/' + runtime.id, json={'name': name, 'backend': 'custom',
                                                                         'metadata': {'custom': configuration(env={'CUSTOM_THREADS': '2'})}})
        assert response.status_code == 200, response.text
        assert response.json()['id'] == runtime.id and response.json()['name'] == name
        assert controller.require('profiles', profile.id)['runtime_id'] == runtime.id
    refreshed = await client.post('/api/runtimes/' + runtime.id + '/refresh')
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()['metadata']['validation'] == 'files_and_configuration'
    assert refreshed.json()['name'] == 'Renamed again'
    assert controller.store.all('jobs') == []
    capabilities = (await client.get('/api/state')).json()['backends']
    custom = next(item for item in capabilities if item['id'] == 'custom')
    assert not custom['installable'] and custom['parameters'][0]['key'] == 'startup_timeout'
    assert custom['supported_platforms'] == ['windows', 'wsl2']
    assert [field['key'] for field in custom['generation_parameters']] == ['temperature', 'top_p', 'max_tokens']


@pytest.mark.parametrize('change,match', [
    ({'args': ['--model', '{model_path}']}, 'model_path'),
    ({'args': ['--model', '{model_path}', '--port', '{potr}']}, 'potr'),
    ({'args': ['--model', '{model_path}', '--port', '{port}', '--port', '9000']}, '重复'),
    ({'args': ['--model', '{model_path}', '--port', '{port}', '--host', '0.0.0.0']}, '回环'),
    ({'args': ['--model', '{model_path}', '--port', '{port}', '--api-key', 'dont-store-this']}, '敏感参数'),
    ({'env': {'HF_TOKEN': 'dont-store-this'}}, '明文'),
    ({'env': {'OTHER': 'https://user:password@example.test/'}}, '明文'),
    ({'env': {'HF_HUB_OFFLINE': '0'}}, '离线'),
    ({'health_path': '//example.test/models'}, '本机'),
    ({'chat_path': 'https://example.test/chat'}, '本机'),
    ({'generation_parameters': ['made_up']}, '生成参数'),
])
def test_invalid_custom_configuration_is_rejected_before_registration(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        engine(tmp_path, **change)


def test_native_executable_and_working_directory_are_required(tmp_path):
    fake = tmp_path / 'cmd.exe'
    fake.write_text('never run')
    with pytest.raises(ValueError, match='shell'):
        import_runtime('custom', str(fake), {'custom': configuration()})
    with pytest.raises(ValueError, match='不存在'):
        engine(tmp_path, cwd=str(tmp_path / 'missing'))


def test_template_preserves_argv_python_module_json_and_literal_braces(tmp_path):
    args = ['-m', 'some_local_module', '--model={model_path}', '--bind', '{host}:{port}',
            '--json', '{"nested":{"literal":"{{alias}}","name":"{alias}"}}']
    runtime = engine(tmp_path, args=args, api_model_name='{model_path}')
    model, profile = profile_for(tmp_path, runtime, extra_args=['--tokenizer', 'local tokenizer'])
    context = LaunchContext(port=12345, instance_id='fixture', work_dir=str(tmp_path), data_dir=str(tmp_path))
    spec = get_adapters()['custom'].build_launch(model, profile, runtime, context)
    assert spec.argv[1:3] == ['-m', 'some_local_module']
    assert spec.argv[3] == '--model=' + model.path
    assert spec.argv[spec.argv.index('--bind') + 1] == '127.0.0.1:12345'
    assert json.loads(spec.argv[spec.argv.index('--json') + 1]) == {'nested': {'literal': '{alias}', 'name': profile.alias}}
    assert spec.argv[-2:] == profile.extra_args and spec.model_name == model.path
    assert spec.env['HF_HUB_OFFLINE'] == '1'
    for extra in (['--port=8000'], ['--model', 'other'], ['--api-key=secret'], ['--json', '{}']):
        profile.extra_args = extra
        with pytest.raises(ValueError):
            get_adapters()['custom'].build_launch(model, profile, runtime, context)


async def test_per_runtime_parameters_and_generation_are_validated_on_save_and_preflight(api, tmp_path):
    controller, client = api
    args = [*configuration()['args'], '--context-length', '{context_length}']
    runtime = await register(client, args=args, generation_parameters=['temperature', 'max_tokens', 'top_k'])
    model, profile = profile_for(tmp_path, runtime)
    controller.store.put('models', model.model_dump())
    status = await client.post('/api/profiles/preflight', json=profile.model_dump())
    assert status.status_code == 200, status.text
    assert status.json()['status'] == 'error' and 'context_length' in status.json()['message']
    field = status.json()['parameters'][0]
    assert field['key'] == 'context_length' and field['default'] is None and field['minimum'] is None
    assert [field['key'] for field in status.json()['generation_parameters']] == ['temperature', 'max_tokens', 'top_k']
    profile.parameters['context_length'] = 1000000  # No undocumented model-context cap.
    profile.generation['top_k'] = -17  # Unknown engine semantics, not another adapter's sentinel.
    response = await client.post('/api/profiles', json=profile.model_dump())
    assert response.status_code == 200, response.text
    profile = LoadProfile(**response.json())
    before = deepcopy(controller.require('profiles', profile.id))
    for section, value in [('generation', {'top_p': .5}), ('generation', {'max_tokens': 1.5}),
                           ('generation', {'top_k': 1.5}), ('parameters', {'context_length': 1.5}),
                           ('parameters', {'gpu_layers': 1}), ('extra_args', ['--port', '3333'])]:
        result = await client.patch('/api/profiles/' + profile.id, json={section: value})
        assert result.status_code == 400, result.text
        assert controller.require('profiles', profile.id) == before
    default = engine(tmp_path)
    assert [field.key for field in profile_fields(get_adapters()['custom'], model, default)] == ['startup_timeout']
    assert [field.key for field in generation_fields('custom', default)] == ['temperature', 'top_p', 'max_tokens']


async def test_explicit_custom_model_import_does_not_change_format_scan_or_existing_metadata(api, tmp_path):
    controller, client = api
    path = tmp_path / 'native model.pack'
    path.write_bytes(b'custom model fixture')
    assert (await client.post('/api/models/import', json={'path': str(path)})).status_code == 400
    result = await client.post('/api/models/import-custom', json={'path': str(path), 'name': 'Native fixture'})
    assert result.status_code == 200, result.text
    model = result.json()
    assert model['format'] == 'custom' and not model['managed'] and model['size_bytes'] == path.stat().st_size
    assert model['metadata']['size_bytes_known'] is True and model['files'] == [str(path)]
    model['metadata']['preserved'] = 'original metadata'
    controller.store.put('models', model)
    again = await client.post('/api/models/import-custom', json={'path': str(path), 'name': 'Do not overwrite'})
    assert again.json() == model
    directory = tmp_path / 'opaque-directory'
    directory.mkdir()
    (directory / 'do-not-scan.bin').write_bytes(b'fixture')
    result = (await client.post('/api/models/import-custom', json={'path': str(directory)})).json()
    assert result['format'] == 'custom' and result['files'] == []
    assert result['metadata']['size_bytes_known'] is False and '未统计' in result['warnings'][0]
    assert (await client.post('/api/models/import-custom', json={'path': str(tmp_path / 'missing')})).status_code == 400


@pytest.mark.parametrize('stream_usage', [False, True])
async def test_custom_process_health_gateway_stream_usage_metrics_and_stop(api, tmp_path, stream_usage):
    controller, client = api
    args = [*configuration()['args'], '--health-path', '/custom-health', '--chat-path', '/custom-chat']
    if not stream_usage:
        args += ['--reject-stream-usage']
    runtime = await register(client, args=args, health_path='/custom-health', chat_path='/custom-chat', stream_usage=stream_usage)
    _, profile = await create_profile(controller, client, tmp_path, runtime, generation={'temperature': .3, 'max_tokens': 16})
    preview = await client.post('/api/profiles/' + profile.id + '/preview')
    assert preview.status_code == 200, preview.text
    assert preview.json()['argv'][0] == sys.executable
    assert controller.runtime.processes == {}
    start = await client.post('/api/profiles/' + profile.id + '/start')
    assert start.status_code == 200, start.text
    instance = controller.runtime.instances[start.json()['id']]
    await wait_state(instance, 'ready')
    assert instance.capabilities['stream_usage'] is stream_usage
    assert instance.capabilities['generation_parameters'] == ['temperature', 'top_p', 'max_tokens']
    assert (await client.patch('/api/runtimes/' + runtime.id, json={'name': 'Blocked while active'})).status_code == 409
    ordinary = {'model': profile.alias, 'messages': [{'role': 'user', 'content': 'hello fixture'}],
                'stop': ['END'], 'response_format': {'type': 'json_object'}}
    response = await client.post('/api/chat', json=ordinary)
    assert response.status_code == 200, response.text
    assert response.json()['model'] == profile.alias
    received = response.json()['received']
    assert received['model'] == 'upstream-' + profile.alias
    assert received['temperature'] == .3 and received['stop'] == ['END'] and received['response_format'] == ordinary['response_format']
    assert (await client.post('/api/chat', json={**ordinary, 'top_k': 8})).status_code == 400
    # The public OpenAI entry point applies the same declared generation contract.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
        assert (await public.post('/v1/chat/completions', json={**ordinary, 'seed': 42})).status_code == 400
    if not stream_usage:
        denied = await client.post('/api/chat', json={**ordinary, 'stream': True, 'stream_options': {'include_usage': True}})
        assert denied.status_code == 400 and 'include_usage' in denied.text
    stream = await client.post('/api/chat', json={**ordinary, 'stream': True})
    assert stream.status_code == 200 and 'data: [DONE]' in stream.text
    chunks = [json.loads(line[5:]) for line in stream.text.splitlines() if line.startswith('data:') and '[DONE]' not in line]
    reports = [chunk['modloader_request'] for chunk in chunks if set(chunk) == {'modloader_request'}]
    model_chunks = [chunk for chunk in chunks if set(chunk) != {'modloader_request'}]
    assert model_chunks and all(chunk['model'] == profile.alias for chunk in model_chunks)
    assert len(reports) == 1 and reports[0]['status'] == 'completed'
    assert reports[0]['counters']['input'] == (4 if stream_usage else None)
    assert reports[0]['counters']['output'] == (3 if stream_usage else None)
    usage = [chunk['usage'] for chunk in chunks if 'usage' in chunk]
    assert bool(usage) is stream_usage
    metrics = await client.get('/api/instances/' + instance.id + '/telemetry')
    assert metrics.status_code == 200, metrics.text
    data = metrics.json()
    assert data['backend_status'] == 'ready' and data['live_generation'] is None
    assert any('没有通用' in text for text in data['unavailable'])
    request = next(group for group in data['groups'] if group['id'] == 'api_request')
    completion = next(metric for metric in request['metrics'] if metric['id'] == 'completion_tokens')
    assert completion['value'] == (3 if stream_usage else None)
    assert controller.runtime.processes[instance.id].returncode is None
    stopped = await client.post('/api/instances/' + instance.id + '/stop', json={})
    assert stopped.status_code == 200 and stopped.json()['state'] == 'stopped'
    assert controller.runtime.processes[instance.id].returncode is not None
    assert instance.active_requests == 0 and not controller.runtime.requests[instance.id]
    failed_chat = await client.post('/api/chat', json=ordinary)
    assert failed_chat.status_code == 409
    assert (await client.patch('/api/runtimes/' + runtime.id, json={'name': 'Editable after stop'})).status_code == 200


@pytest.mark.parametrize('flag,match,timeout', [('--exit-immediately', '退出', 5), ('--never-ready', '超时', .15)])
async def test_startup_failure_and_timeout_clean_up_owned_process(api, tmp_path, flag, match, timeout):
    controller, client = api
    runtime = await register(client, args=[*configuration()['args'], flag])
    _, profile = await create_profile(controller, client, tmp_path, runtime, parameters={'startup_timeout': timeout})
    result = await client.post('/api/profiles/' + profile.id + '/start')
    assert result.status_code == 200, result.text
    instance = controller.runtime.instances[result.json()['id']]
    await wait_state(instance, 'failed')
    await controller.runtime.tasks[instance.id]
    assert match in instance.error
    assert controller.runtime.processes[instance.id].returncode is not None


async def test_backend_error_and_forced_stop_release_inflight_gateway_request(api, tmp_path):
    controller, client = api
    runtime = await register(client)
    _, profile = await create_profile(controller, client, tmp_path, runtime)
    result = await client.post('/api/profiles/' + profile.id + '/start')
    instance = controller.runtime.instances[result.json()['id']]
    await wait_state(instance, 'ready')
    ordinary = {'model': profile.alias, 'messages': [{'role': 'user', 'content': 'fixture only'}]}
    failed = await client.post('/api/chat', json={**ordinary, 'fixture_error': True})
    assert failed.status_code == 503 and instance.active_requests == 0
    assert controller.request_metrics[instance.id]['status'] == 'failed'
    task = asyncio.create_task(client.post('/api/chat', json={**ordinary, 'stream': True, 'fixture_delay': 2}))
    async with asyncio.timeout(5):
        while not instance.active_requests or not controller.runtime.requests[instance.id]:
            await asyncio.sleep(.01)
    assert (await client.post('/api/instances/' + instance.id + '/stop', json={})).status_code == 409
    stopped = await client.post('/api/instances/' + instance.id + '/stop', json={'force': True})
    assert stopped.status_code == 200
    response = await asyncio.wait_for(task, 5)
    assert response.status_code == 200
    assert instance.active_requests == 0 and controller.runtime.processes[instance.id].returncode is not None
    assert controller.request_metrics[instance.id]['status'] == 'cancelled'
