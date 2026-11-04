"""API/job integration and the model/profile transaction, with isolated records."""
import asyncio
from copy import deepcopy

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.models import ModelAsset, LoadProfile, RuntimeInstallation


def records(controller):
    model = ModelAsset(id='model', name='迁移模型', path='D:/fixture/model.ninfer', format='ninfer').model_dump()
    runtime = RuntimeInstallation(id='runtime', name='NInfer', backend='ninfer', platform='wsl2',
                                  path='/opt/ninfer', metadata={'distribution': 'ModLoader'}).model_dump()
    profile = LoadProfile(id='profile', name='加载配置', model_id='model', runtime_id='runtime',
                          backend='ninfer', alias='model').model_dump()
    for kind, value in [('models', model), ('runtimes', runtime), ('profiles', profile)]:
        controller.store.put(kind, value)
    return model, runtime, profile


@pytest.mark.asyncio
async def test_migration_queued_job_blocks_conflicting_actions(tmp_path):
    controller = Controller(tmp_path / 'data', 'fixture-token')
    records(controller)
    controller.store.put('jobs', {'id': 'queued-migration', 'kind': 'model_migration', 'state': 'queued',
                                  'request': {'model_id': 'model'}})
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer fixture-token'}) as client:
            for method, path, body in [
                ('POST', '/api/profiles/profile/start', {}),
                ('PATCH', '/api/profiles/profile', {'name': '不应保存'}),
                ('DELETE', '/api/profiles/profile', None),
                ('DELETE', '/api/models/model', None),
                ('POST', '/api/models/model/storage-migrate', {'plan_id': 'another'}),
                ('POST', '/api/runtimes/runtime/refresh', {}),
                ('POST', '/api/runtimes/runtime/switch', {'target_id': 'runtime', 'profile_ids': ['profile']}),
                ('DELETE', '/api/runtimes/runtime', None),
            ]:
                response = await client.request(method, path, json=body)
                assert response.status_code == 409, (path, response.text)
                assert '迁移' in response.text
            state = (await client.get('/api/state')).json()
            assert state['migrations'][0]['id'] == 'queued-migration'
            assert controller.store.get('profiles', 'profile')['name'] == '加载配置'
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_migration_job_uses_existing_progress_and_cancellation(tmp_path, monkeypatch):
    controller = Controller(tmp_path / 'data', 'fixture-token')
    records(controller)
    entered = asyncio.Event()
    released = asyncio.Event()

    async def preview(model_id):
        return {'id': 'plan', 'model_id': model_id, 'eligible': True}

    async def run(request, job_id, progress, cancel):
        assert request == {'model_id': 'model', 'plan_id': 'plan', 'delete_source': False}
        await progress({'phase': 'copying', 'downloaded_bytes': 64, 'total_bytes': 256, 'message': '复制模型'})
        entered.set()
        await released.wait()
        return {'model_id': 'native-model', 'model_ids': ['native-model'], 'source_model_id': 'model',
                'source_deleted': False, 'switched_profile_ids': ['profile'], 'warnings': []}

    monkeypatch.setattr(controller.model_storage, 'preview', preview)
    monkeypatch.setattr(controller.model_storage, 'run', run)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer fixture-token'}) as client:
            response = await client.post('/api/models/model/storage-preview')
            assert response.json()['id'] == 'plan'
            response = await client.post('/api/models/model/storage-migrate', json={'plan_id': 'plan'})
            assert response.status_code == 200, response.text
            job_id = response.json()['id']
            await asyncio.wait_for(entered.wait(), 2)
            state = (await client.get('/api/state')).json()
            assert state['migrations'][0]['downloaded_bytes'] == 64
            cancelled = await client.post(f'/api/jobs/{job_id}/cancel')
            assert cancelled.json()['state'] == 'cancelled'
            assert '迁移' in cancelled.json()['message']
            assert controller.store.get('models', 'model')

            entered.clear()
            response = await client.post('/api/models/model/storage-migrate', json={'plan_id': 'plan'})
            job_id = response.json()['id']
            await asyncio.wait_for(entered.wait(), 2)
            released.set()
            await controller.job_tasks[job_id]
            result = controller.store.get('jobs', job_id)
            assert result['state'] == 'completed'
            assert result['model_id'] == 'native-model'
            assert result['source_deleted'] is False
            assert '原文件保留' in result['message']
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_model_storage_transaction_rejects_changed_profiles_and_rolls_back(tmp_path):
    controller = Controller(tmp_path / 'data', 'fixture-token')
    model, runtime, profile = records(controller)
    native = {**model, 'id': 'native', 'path': '/opt/modloader/models/native/model.ninfer',
              'location': {'platform': 'wsl2', 'distribution': 'ModLoader'}}
    replacement = {**profile, 'model_id': native['id'], 'revision': 2}
    extra = {**profile, 'id': 'another', 'alias': 'another'}
    controller.store.put('profiles', extra)
    try:
        with pytest.raises(ValueError, match='关联加载配置已变化'):
            controller.store.apply_model_storage_migration(model, [profile], [runtime], native, [replacement])
        assert controller.store.get('models', 'native') is None
        assert controller.store.get('profiles', 'profile') == profile
        # A failure after inserting the model must still roll back the model and profile together.
        preparation = {'id': 'existing-preparation', 'model_id': 'model'}
        controller.store.put('ninfer_preparations', preparation)
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            controller.store.apply_model_storage_migration(model, [profile, extra], [runtime], native, [replacement],
                                                            new_preparations=[deepcopy(preparation)])
        assert controller.store.get('models', 'native') is None
        assert controller.store.get('profiles', 'profile') == profile
        controller.store.apply_model_storage_migration(model, [profile, extra], [runtime], native, [replacement])
        assert controller.store.get('models', 'model') == model
        assert controller.store.get('models', 'native') == native
        assert controller.store.get('profiles', 'profile') == replacement
        assert controller.store.get('profiles', 'another') == extra
    finally:
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('prepared', [True, False])
async def test_resource_preview_reuses_prepared_ninfer_without_changing_profile(tmp_path, monkeypatch, prepared):
    controller = Controller(tmp_path / 'data', 'fixture-token')
    model, runtime, profile = records(controller)
    model.update(path='D:/fixture/hf', format='safetensors')
    controller.store.put('models', model)
    from modloader import parameter_rules
    monkeypatch.setattr(parameter_rules, 'validate_profile', lambda *args, **kwargs: None)

    def resolved(model, profile, runtime):
        if not prepared:
            raise ValueError('首次加载需要转换')
        result = profile.model_copy(deep=True)
        result.parameters['_prepared_artifact'] = '/opt/modloader/models/prepared/model.ninfer'
        return result

    async def estimate(model, effective, runtime, progress=None):
        assert effective.parameters.get('_prepared_artifact') == ('/opt/modloader/models/prepared/model.ninfer' if prepared else None)
        return {'resources': [{'kind': 'vram', 'bytes': 1024, 'quality': 'partial'}], 'notes': [], 'source': 'metadata'}

    monkeypatch.setattr(controller.preparation, 'resolved_profile', resolved)
    monkeypatch.setattr(controller.resource_estimator, 'estimate', estimate)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer fixture-token'}) as client:
            response = await client.post('/api/resources/estimate', json=profile)
            assert response.status_code == 200, response.text
            assert response.json()['resources'][0]['bytes'] == 1024
            assert bool(response.json()['notes']) is not prepared
            assert controller.store.get('profiles', profile['id']) == profile
            assert not controller.store.all('jobs')
    finally:
        await controller.close()


async def test_storage_preview_accepts_direction_and_destination(tmp_path, monkeypatch):
    controller = Controller(tmp_path / 'data', 'fixture-token')
    records(controller)
    calls = []
    async def preview(identifier, options=None):
        calls.append((identifier, options))
        return {'id': 'plan', 'eligible': True, 'target_platform': options.get('target_platform') if options else 'wsl2'}
    monkeypatch.setattr(controller.model_storage, 'preview', preview)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer fixture-token'}) as client:
            response = await client.post('/api/models/model/storage-preview', json={'target_platform': 'windows', 'destination': 'D:/chosen'})
            assert response.status_code == 200 and response.json()['target_platform'] == 'windows'
            response = await client.post('/api/models/model/storage-preview')
            assert response.status_code == 200 and response.json()['target_platform'] == 'wsl2'
        assert calls == [('model', {'target_platform': 'windows', 'destination': 'D:/chosen'}), ('model', None)]
    finally:
        await controller.close()
