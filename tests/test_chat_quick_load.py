import json

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.models import LoadProfile, RuntimeInstance
from test_core import FixtureAdapter, add_profile


@pytest.fixture
async def quick_controller(tmp_path):
    controller = Controller(tmp_path / 'data', 'quick-load-fixture')
    controller.adapters['fixture'] = FixtureAdapter()
    try:
        yield controller
    finally:
        controller.runtime.instances.clear()
        await controller.close()


def instance_for(profile, state='preparing'):
    return RuntimeInstance(id='quick-instance', model_id=profile.model_id, profile_id=profile.id,
                           profile_revision=profile.revision, alias=profile.alias, backend=profile.backend,
                           runtime_id=profile.runtime_id, state=state, port=12345, started_at='2026-10-27T00:00:00Z')


async def test_quick_load_toggle_persists_without_changing_running_revision_or_revalidating(quick_controller, tmp_path, monkeypatch):
    controller = quick_controller
    _, profile, _ = add_profile(controller, tmp_path, 'quick')
    legacy = {**profile.model_dump(exclude={'chat_quick_load'}), 'external_metadata': {'keep': True}}
    controller.store.put('profiles', legacy)
    running = instance_for(profile, 'ready')
    controller.runtime.instances[running.id] = running

    def unexpected_validation(*args, **kwargs):
        pytest.fail('快捷列表修改不能重新校验启动参数')

    monkeypatch.setattr('modloader.parameter_rules.validate_profile', unexpected_validation)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer quick-load-fixture'}) as client:
        for enabled in (True, False):
            response = await client.patch('/api/profiles/quick', json={'chat_quick_load': enabled})
            assert response.status_code == 200, response.text
            expected = {**legacy, 'chat_quick_load': enabled}
            assert response.json() == expected
            assert controller.store.get('profiles', profile.id) == expected
            assert running.state == 'ready' and running.profile_revision == expected['revision'] == 1
            state = (await client.get('/api/state')).json()
            assert next(item for item in state['profiles'] if item['id'] == profile.id)['chat_quick_load'] is enabled
    assert LoadProfile(**legacy).chat_quick_load is False


async def test_normal_profile_edit_preserves_quick_load_setting_and_updates_revision(quick_controller, tmp_path):
    controller = quick_controller
    _, profile, _ = add_profile(controller, tmp_path, 'quick')
    controller.store.put('profiles', {**profile.model_dump(), 'chat_quick_load': True})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer quick-load-fixture'}) as client:
        response = await client.patch('/api/profiles/quick', json={'name': '重命名配置'})
    assert response.status_code == 200, response.text
    assert response.json()['chat_quick_load'] is True
    assert response.json()['name'] == '重命名配置'
    assert response.json()['revision'] == 2


@pytest.mark.parametrize('streaming', [False, True])
async def test_profile_start_negotiates_progress_and_keeps_existing_preparation_path(quick_controller, tmp_path, monkeypatch, streaming):
    controller = quick_controller
    model, profile, runtime = add_profile(controller, tmp_path, 'quick')
    expected = instance_for(profile)
    calls = []

    async def start(actual_model, actual_profile, actual_runtime):
        calls.append((actual_model, actual_profile, actual_runtime))
        return expected

    monkeypatch.setattr(controller.runtime, 'start', start)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer quick-load-fixture', **({'Accept': 'application/x-ndjson'} if streaming else {})}) as client:
        response = await client.post('/api/profiles/quick/start')
    assert response.status_code == 200, response.text
    assert calls == [(model, profile, runtime)]
    if streaming:
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[0] == {'type': 'progress', 'stage': 'startup', 'message': '检查启动条件'}
        assert events[-1] == {'type': 'result', 'value': expected.model_dump()}
    else:
        assert response.json() == expected.model_dump()


@pytest.mark.parametrize('streaming', [False, True])
async def test_duplicate_start_is_rejected_without_stopping_existing_instance(quick_controller, tmp_path, streaming):
    controller = quick_controller
    _, profile, _ = add_profile(controller, tmp_path, 'quick')
    existing = instance_for(profile, 'ready')
    controller.runtime.instances[existing.id] = existing
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer quick-load-fixture', **({'Accept': 'application/x-ndjson'} if streaming else {})}) as client:
        response = await client.post('/api/profiles/quick/start')
    if streaming:
        error = json.loads(response.text.splitlines()[-1])
        assert error['type'] == 'error' and error['status'] == 400
        assert '已有运行实例' in error['detail']
    else:
        assert response.status_code == 400 and '已有运行实例' in response.json()['detail']
    assert existing.state == 'ready' and list(controller.runtime.instances) == [existing.id]


async def test_parallel_instances_route_exactly_and_keep_legacy_alias(quick_controller, tmp_path):
    import asyncio
    from test_core import wait_ready
    controller = quick_controller
    args = add_profile(controller, tmp_path, 'parallel')
    first = await controller.runtime.start(*args)
    second = await controller.runtime.start(*args, allow_parallel=True)
    try:
        await asyncio.gather(wait_ready(first), wait_ready(second))
        assert first.alias == 'parallel' and second.alias == 'parallel--2'
        assert controller.store.get('profiles', 'parallel')['alias'] == 'parallel'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer quick-load-fixture'}) as client:
            response = await client.post('/api/chat', json={'model': first.alias, 'instance_id': second.id, 'messages': []})
            assert response.status_code == 200, response.text
            assert second.id in controller.request_metrics and first.id not in controller.request_metrics
            assert (await client.post('/api/chat', json={'model': first.alias, 'instance_id': 'missing', 'messages': []})).status_code == 409
            first.active_requests = 1
            assert (await client.post(f'/api/instances/{first.id}/stop', json={})).status_code == 409
            assert first.state == 'ready'
            first.active_requests = 0
            stopped = await client.post(f'/api/instances/{first.id}/stop', json={})
            assert stopped.json()['state'] == 'stopped' and second.state == 'ready'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)), base_url='http://testserver', headers={'Authorization': 'Bearer ' + controller.api_key}) as client:
            assert (await client.post('/v1/chat/completions', json={'model': second.alias, 'messages': []})).status_code == 200
    finally:
        await controller.runtime.stop(first.id, True)
        await controller.runtime.stop(second.id, True)


async def test_chat_settings_defaults_validation_and_persistence(quick_controller):
    controller = quick_controller
    assert controller.settings['chat_unload_before_load'] is True
    assert controller.settings['chat_tool_details'] == 'compact'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer quick-load-fixture'}) as client:
        assert (await client.patch('/api/settings', json={'chat_unload_before_load': False, 'chat_tool_details': 'full'})).status_code == 200
        saved = controller.store.get('settings', 'main')
        assert saved['chat_unload_before_load'] is False and saved['chat_tool_details'] == 'full'
        for invalid in ({'chat_unload_before_load': 'false'}, {'chat_tool_details': 'other'}):
            assert (await client.patch('/api/settings', json=invalid)).status_code == 400
