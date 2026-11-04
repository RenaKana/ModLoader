import httpx
import pytest

from modloader.app import Controller, build_app
from test_sglang_backend import fixture


@pytest.mark.parametrize('enabled', [False, True])
async def test_preflight_exposes_editable_paging_values_without_hiding_conflicts(tmp_path, monkeypatch, enabled):
    model, runtime, profile, _ = fixture(tmp_path, monkeypatch)
    runtime.version = '0.5.23'
    model.metadata = {'config': {'model_type': 'qwen3_moe', 'num_experts': 128, 'num_experts_per_tok': 4}}
    profile.parameters = {'acceleration_mode': 'mtp' if enabled else 'none'}
    profile.extra_args = ['--paged-experts-num-resident', '16', '--moe-runner-backend', 'triton']
    if enabled:
        profile.extra_args.insert(0, '--enable-paged-experts')
    controller = Controller(tmp_path / 'preflight-data', 'test-secret')
    for kind, record in [('models', model), ('runtimes', runtime)]:
        controller.store.put(kind, record.model_dump())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                     base_url='http://testserver', headers={'Authorization': 'Bearer test-secret'}) as client:
            response = await client.post('/api/profiles/preflight', json=profile.model_dump())
        assert response.status_code == 200, response.text
        body = response.json()
        paging = body['paged_experts']
        assert paging['enabled'] is enabled
        assert paging['num_resident'] == 16
        assert paging['runner'] == 'triton'
        assert paging['normalized_parameters']['paged_experts_num_resident'] == 16
        assert paging['normalized_extra_args'] == []
        if enabled:
            assert body['status'] == 'error'
            assert '推测解码' in body['message']
        else:
            assert body['status'] == 'ready'
    finally:
        await controller.close()
