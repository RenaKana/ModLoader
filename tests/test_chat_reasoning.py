"""Reasoning snapshots and gateway payloads; no inference runtime is launched."""
import json
import uuid

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.conversations import DETAIL_KIND
from modloader.models import LaunchSpec, RuntimeInstance


@pytest.fixture
async def controller(tmp_path):
    value = Controller(tmp_path / 'data', 'reasoning-fixture')
    try:
        yield value
    finally:
        value.runtime.instances.clear()
        await value.close()


@pytest.fixture
async def client(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer reasoning-fixture'}) as value:
        yield value


def conversation(effort=None):
    return {'id': str(uuid.uuid4()), 'title': '推理快照', 'selection': None, 'draft': {'text': '保留草稿', 'images': []},
            'settings': {'system': '', 'toolsDraft': '', 'stream': True, **({} if effort is None else {'reasoningEffort': effort})}}


async def test_legacy_default_and_named_numeric_snapshots_round_trip(client, controller):
    for effort in (None, 'none', 'low', 'medium', 'high', 'xhigh', 'max', 1, 17, 50, 100):
        body = conversation(effort)
        response = await client.post('/api/conversations', json=body)
        assert response.status_code == 200, response.text
        actual = (await client.get('/api/conversations/' + body['id'])).json()
        assert actual['settings'] == body['settings']
        stored = controller.store.get(DETAIL_KIND, body['id'])
        assert 'reasoningEffort' not in stored['settings']  # Older clients can still edit the legacy settings object.
        if effort is not None:
            assert stored['reasoning_effort'] == effort
            stored['settings']['system'] = '旧版更新系统词'
            controller.store.put(DETAIL_KIND, stored)
            assert (await client.get('/api/conversations/' + body['id'])).json()['settings']['reasoningEffort'] == effort
        reset = {field: value for field, value in body['settings'].items() if field != 'reasoningEffort'}
        assert (await client.patch('/api/conversations/' + body['id'], json={'settings': reset})).status_code == 200
        assert 'reasoningEffort' not in (await client.get('/api/conversations/' + body['id'])).json()['settings']
        assert 'reasoning_effort' not in controller.store.get(DETAIL_KIND, body['id'])


async def test_presets_can_gain_change_and_remove_reasoning_without_rewriting_conversations(client):
    body = {'id': str(uuid.uuid4()), 'name': '旧预设', 'system': '', 'toolsDraft': ''}
    original = (await client.post('/api/chat-presets', json=body)).json()['items'][0]
    assert 'reasoningEffort' not in original
    saved = conversation('xhigh')
    await client.post('/api/conversations', json=saved)
    for effort in ('xhigh', 'max', 1, 17, 100, None):
        response = await client.patch('/api/chat-presets/' + body['id'], json={'reasoningEffort': effort, 'make_default': True})
        assert response.status_code == 200, response.text
        catalog = response.json()
        assert catalog['default_id'] == body['id']
        if effort is None:
            assert 'reasoningEffort' not in catalog['items'][0]
        else:
            assert catalog['items'][0]['reasoningEffort'] == effort
        assert (await client.get('/api/conversations/' + saved['id'])).json()['settings']['reasoningEffort'] == 'xhigh'


@pytest.mark.parametrize('effort', [True, False, 0, 101, 1.5, [], {}, 'ultra', 'budget', '50'])
async def test_invalid_reasoning_cannot_partially_change_snapshots(client, effort):
    body = conversation('low')
    await client.post('/api/conversations', json=body)
    response = await client.patch('/api/conversations/' + body['id'], json={'title': '不得写入', 'settings': {**body['settings'], 'reasoningEffort': effort}})
    assert response.status_code == 400
    actual = (await client.get('/api/conversations/' + body['id'])).json()
    assert actual['title'] == body['title'] and actual['settings'] == body['settings']
    preset = {'id': str(uuid.uuid4()), 'name': '预设', 'system': '', 'toolsDraft': '', 'reasoningEffort': effort}
    assert (await client.post('/api/chat-presets', json=preset)).status_code == 400
    assert (await client.get('/api/chat-presets')).json()['items'] == []


@pytest.mark.parametrize('backend', ['strata', 'llama_cpp', 'ninfer'])
async def test_gateway_preserves_streaming_nonstreaming_overrides_and_omits_default(controller, tmp_path, monkeypatch, backend):
    controller.store.put('profiles', {'id': 'profile', 'generation': {}})
    instance = RuntimeInstance(id='fixture', model_id='model', profile_id='profile', profile_revision=1, alias='fixture',
                               backend=backend, runtime_id='runtime', state='ready', port=12345, started_at='', capabilities={'input': ['text']})
    controller.runtime.instances[instance.id] = instance
    controller.runtime.specs[instance.id] = LaunchSpec(argv=[], cwd=str(tmp_path), model_name='backend-model')
    controller.runtime.requests[instance.id] = set()
    original = httpx.AsyncClient
    requests = []

    def inference(request):
        payload = json.loads(request.content)
        requests.append(payload)
        chunk = {'choices': [{'message': {'content': '回答'}, 'finish_reason': 'stop'}]}
        if payload.get('stream'):
            return httpx.Response(200, content=('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode(), headers={'Content-Type': 'text/event-stream'})
        return httpx.Response(200, json=chunk)

    async with original(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer reasoning-fixture'}) as client:
        monkeypatch.setattr('modloader.app.httpx.AsyncClient', lambda **options: original(transport=httpx.MockTransport(inference), **options))
        for stream in (False, True):
            efforts = (None, 'none', 'low', 'medium', 'xhigh') if backend == 'ninfer' else (None, 'none', 'xhigh', 17)
            for effort in efforts:
                override = {} if effort is None else {'reasoning_effort': effort}
                payload = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream, **override}
                if stream and effort == 'none':
                    payload['return_progress'] = False
                    if backend in {'ninfer', 'llama_cpp'}:
                        payload['timings_per_token'] = False
                if override and backend == 'llama_cpp':
                    payload['chat_template_kwargs'] = {'enable_thinking': override['reasoning_effort'] != 'none'}
                response = await client.post('/api/chat', json=payload)
                assert response.status_code == 200, response.text
                expected = {**payload, 'model': 'backend-model', **({'stream_options': {'include_usage': True}} if stream else {})}
                if stream and backend in {'ninfer', 'llama_cpp'}:
                    expected.setdefault('timings_per_token', True)
                if stream:
                    expected.setdefault('return_progress', True)
                assert requests[-1] == expected
    assert instance.active_requests == 0
