"""Pinned GDN planning limits, including the actual Qwen3.5-4B failure."""
import json
from pathlib import Path

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.backends import get_adapters, native_engines as native
from modloader.engine_preparation import NativePreparation
from modloader.models import AppPaths
from test_native_engines import MemoryStore, local, write_hf_weights


artifact_info = native.artifact_info
FOUR_B = json.loads((Path(__file__).parent / 'fixtures/ninfer-qwen35-4b-gdn.json').read_text())


@pytest.fixture
def fixed(local):
    runtime = local[1]
    runtime.metadata.update(source_commit=native.PINS['ninfer']['commit'], version_basis='verified_source_commit')
    return local


def bindings(shape, *, merged=False, dtype='bf16'):
    count = shape[0] * shape[1]
    return {'components': {'text': {}, 'mtp': {}}, 'gdn_bindings': {
        'text/layers/0/gdn/a_projection': [dict(id='weight/a', shape=shape, format=dtype,
                                              range=[0, count // 2 if merged else count])],
        'text/layers/0/gdn/b_projection': [dict(id='weight/a' if merged else 'weight/b', shape=shape, format=dtype,
                                              range=[count // 2 if merged else 0, count])],
    }}


def test_gdn_guard_keeps_known_pinned_27b_paired_geometry(fixed):
    native.validate_gdn_artifact(bindings([48, 5120]), fixed[1])


@pytest.mark.parametrize('shape,merged,dtype', [
    ([32, 2560], False, 'bf16'),  # Actual 4B converted parents.
    ([48, 2560], False, 'bf16'),  # Weight input accepts it; paired control does not.
    ([32, 2048], False, 'bf16'),
    ([48, 5120], False, 'fp16'),
])
def test_gdn_guard_uses_parent_topology_shape_and_format(fixed, shape, merged, dtype):
    with pytest.raises(ValueError, match='64d10fbf9f8e.*GDN.*' + str(shape[1])):
        native.validate_gdn_artifact(bindings(shape, merged=merged, dtype=dtype), fixed[1])


@pytest.mark.parametrize('dtype', ['F16', 'F32'])
def test_hf_guard_does_not_confuse_source_dtype_with_converted_parent_format(fixed, dtype):
    model, runtime, profile, context, _ = fixed
    tensors = {name: {'dtype': dtype, 'shape': [48, 5120]} for name in FOUR_B['tensors']}
    write_hf_weights(Path(model.path) / 'model.safetensors', tensors)
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    state = NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore()).preflight(model, profile, runtime)
    assert state['status'] == 'needs_preparation'


@pytest.mark.parametrize('mode', ['none', 'mtp'])
async def test_real_4b_header_failure_blocks_hf_preflight_preview_start_and_reuse(fixed, mode):
    model, runtime, profile, context, source = fixed
    write_hf_weights(Path(model.path) / 'model.safetensors', FOUR_B['tensors'])
    # The fixture deliberately retains a misleading 5120 config/name: the
    # descriptors of the actual weights, not model size/name, decide this guard.
    model.name = 'Not a model-name compatibility rule'
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b', 'acceleration_mode': mode, 'draft_tokens': 1}
    metadata = native.resource_info(model.path, runtime, gdn=True)
    assert metadata['gdn_weights'] == FOUR_B['tensors']
    prep = NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'error'
    assert all(text in state['message'] for text in ('64d10fbf9f8e', 'paired', 'BF16[32, 2560]', '关闭 MTP'))
    assert not prep.root.exists()

    async def unexpected_conversion(event):
        pytest.fail('An unsupported source started conversion')
    with pytest.raises(ValueError, match='GDN'):
        await prep.ensure(model, profile, runtime, unexpected_conversion)

    controller = Controller(Path(context.data_dir) / 'geometry-api', 'fixture-secret')
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    controller.store.put('profiles', profile.model_dump())
    # Even a pre-existing converted-output record must not bypass source checks.
    controller.store.put(prep.KIND, {'id': 'old-4b-artifact', 'model_id': model.id, 'artifact': '/old/model.ninfer'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        try:
            checked = await manager.post('/api/profiles/preflight', json=profile.model_dump())
            assert checked.status_code == 200 and checked.json()['status'] == 'error', checked.text
            preview = await manager.post('/api/profiles/' + profile.id + '/preview')
            assert preview.status_code == 200 and preview.json()['argv'] == [], preview.text
            assert preview.json()['preparation']['message'] == state['message']
            started = await manager.post('/api/profiles/' + profile.id + '/start')
            assert started.status_code == 400 and state['message'] in started.text, started.text
            assert not controller.runtime.instances and not controller.runtime.tasks
            assert not controller.preparation.native.root.exists() and not (source / 'pid').exists()
        finally:
            await controller.close()


@pytest.mark.parametrize('mode', ['none', 'mtp'])
def test_official_artifact_query_rejects_actual_4b_parents_in_preflight_and_launch(fixed, monkeypatch, mode):
    model, runtime, profile, context, _ = fixed
    runtime.metadata['source_root'] = '/opt/fixture/source'
    path = Path(model.path).parent / 'existing.ninfer'
    path.write_bytes(b'query fixture; framing and semantic reader tested independently')
    model = model.model_copy(update={'path': path.as_posix(), 'format': 'ninfer'})
    profile.parameters = {'acceleration_mode': mode, 'draft_tokens': 1}
    queries = []
    def query(argv, **kwargs):
        queries.append(argv)
        return json.dumps(bindings([32, 2560]))
    monkeypatch.setattr(native, 'artifact_info', artifact_info)
    monkeypatch.setattr(native, '_run', query)
    state = NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore()).preflight(model, profile, runtime)
    assert state['status'] == 'error' and 'bf16[32, 2560]' in state['message']
    assert state['preparation_fields'] == []
    with pytest.raises(ValueError, match='GDN'):
        get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    assert queries and 'from tools.artifact.schema import binding_parts' in queries[0][queries[0].index('-c') + 1]


def test_official_artifact_query_keeps_summary_small_and_non_gdn_paths_unverified(fixed, monkeypatch):
    _, runtime, _, _, _ = fixed
    runtime.metadata['source_root'] = '/opt/fixture/source'
    monkeypatch.setattr(native, '_run', lambda *args, **kwargs: json.dumps(bindings([48, 5120])))
    summary = artifact_info('/model.ninfer', runtime)
    assert summary == {'components': {'text': {}, 'mtp': {}}}
    # Absence/complexity is not interpreted as evidence that a geometry works.
    native.validate_gdn_artifact({'components': {'text': {}}}, runtime)
    native.validate_gdn_artifact(bindings([96, 2560], merged=True), runtime)


@pytest.mark.parametrize('identity', [
    {},  # Display version alone does not identify an external executable.
    {'source_commit': 'f' * 40, 'version_basis': 'verified_source_commit'},
    {'source_commit': native.PINS['ninfer']['commit'], 'version_basis': 'external_unknown_no_version_flag'},
])
def test_other_or_unknown_external_pin_uses_original_official_check_path(local, monkeypatch, identity):
    model, runtime, profile, context, _ = local
    runtime.version = native.PINS['ninfer']['version']
    runtime.metadata.update(identity)
    assert not native.uses_pinned_ninfer_rules(runtime)
    native.validate_gdn_source({'gdn_weights': FOUR_B['tensors']}, runtime)
    native.validate_gdn_artifact(bindings([32, 2560]), runtime)

    write_hf_weights(Path(model.path) / 'model.safetensors', FOUR_B['tensors'])
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    original_run = native._run
    def metadata_probe(argv, **kwargs):
        assert argv[-1] != 'gdn', 'A fixed-pin HF metadata extension was applied to an external environment'
        return original_run(argv, **kwargs)
    monkeypatch.setattr(native, '_run', metadata_probe)
    state = NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore()).preflight(model, profile, runtime)
    assert state['status'] == 'needs_preparation'
    assert get_adapters()['ninfer'].inspect(model, runtime).status == 'unverified'

    runtime.metadata['source_root'] = '/opt/external/source'
    def official_query(argv, **kwargs):
        assert argv[-1] == '', 'A fixed-pin binding API was requested from another official inspector'
        return json.dumps(bindings([32, 2560]))
    monkeypatch.setattr(native, '_run', official_query)
    assert artifact_info('/existing.ninfer', runtime) == {'components': {'text': {}, 'mtp': {}}}
