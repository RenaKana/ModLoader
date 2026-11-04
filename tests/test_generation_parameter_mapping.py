from contextlib import asynccontextmanager
import json

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.models import LaunchSpec, LoadProfile, ModelAsset, RuntimeInstance, RuntimeInstallation


@asynccontextmanager
async def gateway(tmp_path, monkeypatch, *, backend='vllm', generation=None, public=False, argv=None,
                  model_defaults=None, generation_defaults=None):
    """Use the real gateway with an isolated store and an in-memory upstream."""
    controller = Controller(tmp_path / 'isolated-data', 'mapping-fixture')
    runtime = RuntimeInstallation(id='runtime', backend=backend, name='Fixture', path='fixture',
                                  version='0.31.0' if backend == 'vllm' else '5.19.0')
    controller.store.put('runtimes', runtime.model_dump())
    if backend == 'transformers':
        path = tmp_path / 'local-model'
        path.mkdir()
        (path / 'config.json').write_text(json.dumps(model_defaults or {}), encoding='utf-8')
        if generation_defaults is not None:
            (path / 'generation_config.json').write_text(json.dumps(generation_defaults), encoding='utf-8')
        model = ModelAsset(id='model', name='Fixture', path=str(path), format='safetensors')
        controller.store.put('models', model.model_dump())
    profile = LoadProfile(id='profile', model_id='model', name='Fixture', backend=backend,
                          runtime_id='runtime', alias='fixture', generation=generation or {})
    controller.store.put('profiles', profile.model_dump())
    instance = RuntimeInstance(id='instance', model_id='model', profile_id=profile.id,
                               profile_revision=profile.revision, alias=profile.alias, backend=backend,
                               runtime_id=profile.runtime_id, state='ready', port=12345, started_at='',
                               capabilities={'input': ['text']})
    controller.runtime.instances[instance.id] = instance
    controller.runtime.specs[instance.id] = LaunchSpec(argv=argv or [], cwd=str(tmp_path), model_name='upstream')
    controller.runtime.requests[instance.id] = set()
    controller.runtime.logs[instance.id] = []
    received = []

    async def upstream(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                                                      'finish_reason': 'stop'}]})

    original = httpx.AsyncClient
    client = original(transport=httpx.ASGITransport(app=build_app(controller, public=public)),
                      base_url='http://testserver',
                      headers={'Authorization': 'Bearer ' + (controller.api_key if public else controller.token)})
    monkeypatch.setattr('modloader.app.httpx.AsyncClient',
                        lambda *args, **kwargs: original(*args, **kwargs, transport=httpx.MockTransport(upstream)))
    try:
        async with client:
            yield client, received
    finally:
        await controller.close()


@pytest.mark.parametrize('public,endpoint', [(False, '/api/chat'), (True, '/v1/chat/completions')])
@pytest.mark.parametrize('defaults,request_values,expected', [
    ({'repeat_penalty': 1.2}, {}, 1.2),
    ({}, {'repeat_penalty': 1.3}, 1.3),
    ({}, {'repetition_penalty': 1.4}, 1.4),
    ({'repeat_penalty': 1.2}, {'repetition_penalty': 1.4}, 1.4),
    ({'repetition_penalty': 1.2}, {'repeat_penalty': 1.4}, 1.4),
    ({}, {'repeat_penalty': 1.2, 'repetition_penalty': 1.2}, 1.2),
])
async def test_vllm_repetition_penalty_mapping_preserves_request_priority(
        tmp_path, monkeypatch, public, endpoint, defaults, request_values, expected):
    async with gateway(tmp_path, monkeypatch, generation=defaults, public=public) as (client, received):
        response = await client.post(endpoint, json={'model': 'fixture', 'messages': [],
                                                     'stop': ['END'], **request_values})
        assert response.status_code == 200, response.text
        assert len(received) == 1
        assert received[0]['repetition_penalty'] == expected
        assert 'repeat_penalty' not in received[0]
        assert received[0]['stop'] == ['END']


@pytest.mark.parametrize('public,endpoint', [(False, '/api/chat'), (True, '/v1/chat/completions')])
@pytest.mark.parametrize('defaults,request_values', [
    ({}, {'repeat_penalty': 1.2, 'repetition_penalty': 1.3}),
    ({'repeat_penalty': 1.2, 'repetition_penalty': 1.3}, {}),
])
async def test_vllm_repetition_penalty_alias_conflicts_are_rejected(
        tmp_path, monkeypatch, public, endpoint, defaults, request_values):
    async with gateway(tmp_path, monkeypatch, generation=defaults, public=public) as (client, received):
        response = await client.post(endpoint, json={'model': 'fixture', 'messages': [], **request_values})
        assert response.status_code == 400
        assert '冲突' in response.text
        assert received == []


@pytest.mark.parametrize('field', ['repeat_penalty', 'repetition_penalty'])
async def test_public_vllm_requests_validate_both_repetition_penalty_names(tmp_path, monkeypatch, field):
    async with gateway(tmp_path, monkeypatch, public=True) as (client, received):
        response = await client.post('/v1/chat/completions', json={
            'model': 'fixture', 'messages': [], field: 0})
        assert response.status_code == 400
        assert received == []


@pytest.mark.parametrize('public,endpoint', [(False, '/api/chat'), (True, '/v1/chat/completions')])
@pytest.mark.parametrize('defaults,request_values,expected', [
    ({'top_k': 0, 'repeat_penalty': 1.2}, {}, {'top_k': 0, 'repetition_penalty': 1.2}),
    ({'top_k': 10, 'repeat_penalty': 1.2}, {'top_k': 5, 'repetition_penalty': 1.4},
     {'top_k': 5, 'repetition_penalty': 1.4}),
    ({'top_k': 10, 'repeat_penalty': 1.2},
     {'generation_config': '{"top_k":7,"repetition_penalty":1.5,"eos_token_id":[1,2],"do_sample":true}'},
     {'top_k': 7, 'repetition_penalty': 1.5, 'eos_token_id': [1, 2], 'do_sample': True}),
])
async def test_transformers_sampling_fields_reach_native_generation_config(
        tmp_path, monkeypatch, public, endpoint, defaults, request_values, expected):
    async with gateway(tmp_path, monkeypatch, backend='transformers', generation=defaults,
                       public=public) as (client, received):
        response = await client.post(endpoint, json={
            'model': 'fixture', 'messages': [], 'max_tokens': 128, 'stop': ['END'], **request_values})
        assert response.status_code == 200, response.text
        assert len(received) == 1
        config = json.loads(received[0]['generation_config'])
        assert config == expected
        assert not {'top_k', 'repeat_penalty', 'repetition_penalty'} & received[0].keys()
        assert received[0]['max_tokens'] == 128 and received[0]['stop'] == ['END']


@pytest.mark.parametrize('request_values', [
    {'repeat_penalty': 1.2, 'repetition_penalty': 1.3},
    {'top_k': 7, 'generation_config': '{"top_k":8}'},
    {'repeat_penalty': 1.2, 'generation_config': '{"repetition_penalty":1.3}'},
    {'repeat_penalty': 1.2, 'frequency_penalty': .5},
])
async def test_transformers_conflicting_sampling_sources_are_not_silently_overwritten(
        tmp_path, monkeypatch, request_values):
    async with gateway(tmp_path, monkeypatch, backend='transformers') as (client, received):
        response = await client.post('/api/chat', json={'model': 'fixture', 'messages': [], **request_values})
        assert response.status_code == 400
        assert '冲突' in response.text
        assert received == []


@pytest.mark.parametrize('argv,expected_status', [
    (['--continuous-batching'], 400),
    (['--no-continuous-batching', '--continuous-batching'], 400),
    (['--continuous-batching', '--no-continuous-batching'], 200),
])
async def test_transformers_sampling_mapping_checks_effective_continuous_batching_mode(
        tmp_path, monkeypatch, argv, expected_status):
    async with gateway(tmp_path, monkeypatch, backend='transformers', argv=argv,
                       generation={'top_k': 5, 'repeat_penalty': 1.2}) as (client, received):
        response = await client.post('/api/chat', json={'model': 'fixture', 'messages': [], 'max_tokens': 128})
        assert response.status_code == expected_status
        if expected_status == 400:
            assert '关闭连续批处理' in response.text
            assert received == []
        else:
            assert len(received) == 1


async def test_transformers_direct_top_k_is_rejected_in_continuous_batching(tmp_path, monkeypatch):
    async with gateway(tmp_path, monkeypatch, backend='transformers', argv=['--continuous-batching'],
                       public=True) as (client, received):
        response = await client.post('/v1/chat/completions', json={
            'model': 'fixture', 'messages': [], 'top_k': 0})
        assert response.status_code == 400
        assert '关闭连续批处理' in response.text
        assert received == []


async def test_transformers_existing_native_config_remains_usable_in_continuous_batching(tmp_path, monkeypatch):
    native = '{"top_k":7,"max_new_tokens":24,"do_sample":true}'
    async with gateway(tmp_path, monkeypatch, backend='transformers', argv=['--continuous-batching']) as (client, received):
        response = await client.post('/api/chat', json={
            'model': 'fixture', 'messages': [], 'generation_config': native})
        assert response.status_code == 200, response.text
        assert received[0]['generation_config'] == native


@pytest.mark.parametrize('defaults,request_values,expected', [
    ({'repeat_penalty': 1.2}, {'frequency_penalty': .5}, 1.5),
    ({'frequency_penalty': .5}, {'repeat_penalty': 1.2}, 1.2),
])
async def test_transformers_request_penalty_overrides_profile_across_native_names(
        tmp_path, monkeypatch, defaults, request_values, expected):
    async with gateway(tmp_path, monkeypatch, backend='transformers', generation=defaults) as (client, received):
        response = await client.post('/api/chat', json={
            'model': 'fixture', 'messages': [], 'max_tokens': 128, **request_values})
        assert response.status_code == 200, response.text
        assert json.loads(received[0]['generation_config'])['repetition_penalty'] == expected
        assert 'frequency_penalty' not in received[0]


async def test_transformers_null_sampling_fields_do_not_replace_native_defaults(tmp_path, monkeypatch):
    async with gateway(tmp_path, monkeypatch, backend='transformers') as (client, received):
        response = await client.post('/api/chat', json={
            'model': 'fixture', 'messages': [], 'top_k': None, 'repeat_penalty': None})
        assert response.status_code == 200, response.text
        assert 'generation_config' not in received[0]
        assert 'top_k' not in received[0] and 'repeat_penalty' not in received[0]


@pytest.mark.parametrize('generation_defaults,model_defaults,expected', [
    ({'max_new_tokens': 4096}, {}, 4096),
    ({}, {'max_new_tokens': 8192}, 1024),
    (None, {}, 1024),
    (None, {'max_new_tokens': 64}, 1024),
    (None, {'max_new_tokens': 4096}, 4096),
    (None, {'text_config': {'max_new_tokens': 4096}}, 4096),
    (None, {'max_new_tokens': 2048, 'text_config': {'max_new_tokens': 4096}}, 2048),
    (None, {'is_encoder_decoder': True, 'decoder_max_new_tokens': 4096}, 4096),
])
async def test_transformers_mapping_preserves_wrapper_default_output_length(
        tmp_path, monkeypatch, generation_defaults, model_defaults, expected):
    async with gateway(tmp_path, monkeypatch, backend='transformers', model_defaults=model_defaults,
                       generation_defaults=generation_defaults) as (client, received):
        response = await client.post('/api/chat', json={'model': 'fixture', 'messages': [], 'top_k': 0})
        assert response.status_code == 200, response.text
        assert json.loads(received[0]['generation_config'])['max_new_tokens'] == expected


@pytest.mark.parametrize('request_values,expected', [
    ({'max_tokens': 128}, {'top_k': 0}),
    ({'generation_config': '{"max_new_tokens":24,"do_sample":true}'},
     {'max_new_tokens': 24, 'do_sample': True, 'top_k': 0}),
])
async def test_transformers_explicit_output_length_does_not_need_model_defaults(
        tmp_path, monkeypatch, request_values, expected):
    async with gateway(tmp_path, monkeypatch, backend='transformers',
                       generation_defaults='invalid config shape') as (client, received):
        response = await client.post('/api/chat', json={
            'model': 'fixture', 'messages': [], 'top_k': 0, **request_values})
        assert response.status_code == 200, response.text
        assert json.loads(received[0]['generation_config']) == expected
