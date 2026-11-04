"""Version selection and shared integration contracts; no installed engines are run."""
from types import SimpleNamespace

import pytest

from modloader.backends import catalog, get_adapters, installers
from modloader.models import AppPaths, RuntimeInstallation
from modloader.parameter_rules import profile_fields


NINFER_NEW = 'a754fcf3bf524494aa927c2f54e0cbfb58f569ab'
NINFER_OLD = '64d10fbf9f8e'


async def test_sglang_catalog_keeps_old_recipe_and_offers_new(monkeypatch):
    async def metadata(url, proxy=None):
        assert '/pypi/sglang/' in url
        return {'urls': [{'packagetype': 'bdist_wheel', 'digests': {'sha256': 'a' * 64}}]}
    monkeypatch.setattr(catalog, '_json', metadata)
    versions = (await catalog.list_versions('sglang'))['versions']
    assert {v['version'] for v in versions} == {'0.5.21', '0.5.23'}
    assert [v['version'] for v in versions if v['recommended']] == ['0.5.23']


async def test_ninfer_catalog_offers_both_source_contracts():
    versions = (await catalog.list_versions('ninfer'))['versions']
    assert {v['version'] for v in versions} == {NINFER_OLD, NINFER_NEW[:12]}
    assert [v['version'] for v in versions if v['recommended']] == [NINFER_NEW[:12]]


def test_profile_fields_use_the_selected_runtime_contract():
    runtime = object()
    called = []
    adapter = SimpleNamespace(id='ninfer', parameters=lambda: [],
                              parameters_for_runtime=lambda selected: called.append(selected) or [])
    profile_fields(adapter, runtime=runtime)
    assert called == [runtime]


@pytest.mark.parametrize('version', ['0.5.21', '0.5.23'])
async def test_sglang_plan_keeps_version_specific_provenance(tmp_path, monkeypatch, version):
    from modloader.backends import source_builds
    paths = AppPaths(tmp_path)
    (paths.runtimes / 'ModLoader-wsl-owned.json').write_text('{}')
    async def metadata(url, proxy=None):
        return {'urls': [{'packagetype': 'bdist_wheel', 'digests': {'sha256': 'a' * 64}}]}
    async def cuda_files(*args, **kwargs):
        return []
    monkeypatch.setattr(catalog, '_json', metadata)
    monkeypatch.setattr(source_builds, 'cuda_files', cuda_files)
    plan = await catalog.plan_install('sglang', paths, {'version': version, 'variant': 'wsl2'})
    assert plan['version'] == version
    assert plan['provenance']['tag'] == 'v' + version
    assert plan['provenance']['source_commit']
    tiles = [f for f in plan['stage']['files'] if f.get('role') == 'cuda_tile']
    if version == '0.5.21':
        assert tiles and tiles[0]['version'] == '1.6.0rc5'
    else:
        assert not tiles, 'The old CUDA Tile prerelease must not leak into the new recipe.'


async def test_new_ninfer_runtime_validation_uses_its_own_source_pin(tmp_path, monkeypatch):
    runtime = RuntimeInstallation(id='new-ninfer', backend='ninfer', name='NInfer',
        platform='wsl2', path='/opt/modloader/new/source/build/apps/ninfer-serve', managed=False,
        metadata={'distribution': 'ModLoader', 'source_commit': NINFER_NEW,
                  'source_root': '/opt/modloader/new/source', 'python': '/opt/modloader/new/venv/bin/python',
                  'requested_version': NINFER_NEW[:12]})
    async def flags(*args, **kwargs):
        return ['--host', '--port', '--spec', '--draft-tokens', '--context-capacity']
    async def command(*args, **kwargs):
        return 'usage: tools'
    monkeypatch.setattr(installers, '_help_flags', flags)
    monkeypatch.setattr(installers, '_command', command)
    result = await installers.validate_runtime(runtime, paths=AppPaths(tmp_path), _installing=True)
    assert result.version == NINFER_NEW[:12]
    assert result.metadata['source_commit'] == NINFER_NEW
    assert runtime.version != result.version


async def test_external_ninfer_keeps_legacy_import_contract(tmp_path, monkeypatch):
    runtime = RuntimeInstallation(id='external', backend='ninfer', name='NInfer', platform='wsl2',
                                  path='/opt/external/ninfer-serve', metadata={'distribution': 'ModLoader'})
    async def flags(*args, **kwargs):
        return ['--host', '--port', '--spec', '--draft-tokens']
    monkeypatch.setattr(installers, '_help_flags', flags)
    result = await installers.validate_runtime(runtime, paths=AppPaths(tmp_path))
    assert result.version == 'unknown'
    assert result.metadata['supported_flags'] == ['--host', '--port', '--spec', '--draft-tokens']


async def test_preflight_exposes_explicit_capacity_migration_action(tmp_path, monkeypatch):
    import httpx
    from modloader.app import Controller, build_app
    from modloader.models import ModelAsset, LoadProfile
    controller = Controller(tmp_path / 'data', 'fixture-token')
    model = ModelAsset(id='model', name='Qwen', path=str(tmp_path / 'model.ninfer'), format='ninfer')
    runtime = RuntimeInstallation(id='new', name='NInfer', backend='ninfer', platform='wsl2',
        path='/opt/new/ninfer-serve', metadata={'distribution': 'ModLoader', 'supported_flags': ['--context-capacity']})
    profile = LoadProfile(id='profile', name='Original', model_id=model.id, runtime_id=runtime.id,
        backend='ninfer', alias='custom', parameters={'kv_capacity': 200000, 'context_length': 100000})
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    controller.store.put('profiles', profile.model_dump())
    monkeypatch.setattr(controller.preparation, 'preflight', lambda *args: {
        'status': 'ready', 'message': '', 'errors': [], 'candidates': []})
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                    headers={'Authorization': 'Bearer fixture-token'}) as client:
            response = await client.post('/api/profiles/preflight', json=profile.model_dump())
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['parameter_migration']['parameter'] == 'kv_capacity'
        assert 'Token' in result['parameter_migration']['message']
        assert 'context_capacity' in {f['key'] for f in result['parameters']}
        assert controller.store.get('profiles', profile.id) == profile.model_dump()
    finally:
        await controller.close()
