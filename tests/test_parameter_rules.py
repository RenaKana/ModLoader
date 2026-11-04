import pytest

from modloader.backends import get_adapters
from modloader.parameter_rules import generation_fields, gpu_layer_limit, profile_fields, validate_generation, validate_profile
from test_mtp import base


@pytest.mark.parametrize('metadata, expected', [
    ({'source_commit': '64d10fbf9f8e6bfc79ab0570b94674c5b9997929', 'version_basis': 'verified_source_commit'}, 8192),
    ({'source_commit': '64d10fbf9f8e6bfc79ab0570b94674c5b9997929'}, None),
    ({'source_commit': 'unknown', 'version_basis': 'verified_source_commit'}, None),
])
def test_generation_defaults_require_verified_ninfer_source(tmp_path, metadata, expected):
    _, _, runtime, _ = base(tmp_path, 'ninfer')
    runtime.metadata = metadata
    fields = {field.key: field for field in generation_fields('ninfer', runtime)}
    assert fields['max_tokens'].default == expected
    assert fields['temperature'].default is None
    assert next(field for field in generation_fields('ninfer') if field.key == 'max_tokens').default is None


@pytest.mark.parametrize('backend,section,key,value', [
    ('strata', 'parameters', 'context_length', -1),
    ('strata', 'parameters', 'vram_reserve_mib', -1),
    ('vllm', 'parameters', 'gpu_memory_utilization', 0),
    ('vllm', 'parameters', 'gpu_memory_utilization', 1.01),
    ('vllm', 'parameters', 'tensor_parallel_size', 1.5),
    ('vllm', 'parameters', 'max_num_seqs', 0),
    ('strata', 'generation', 'top_k', 1.5),
    ('strata', 'generation', 'max_tokens', 1.5),
    ('transformers', 'generation', 'top_p', -0.1),
    ('vllm', 'generation', 'temperature', -1),
])
def test_invalid_numbers_use_backend_contract(tmp_path, backend, section, key, value):
    model, profile, runtime, _ = base(tmp_path, backend)
    getattr(profile, section)[key] = value
    with pytest.raises(ValueError):
        validate_profile(model, profile, runtime, get_adapters()[backend])


def test_bounds_do_not_invent_hardware_or_sample_limits(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    profile.parameters = {'context_length': 1048576, 'threads': 1000, 'gpu_layers': 'auto'}
    validate_profile(model, profile, runtime, get_adapters()['llama_cpp'])
    validate_generation({'temperature': 4, 'max_tokens': -2, 'top_k': -1}, 'llama_cpp')
    # Strata serve's request path defines temperature <= 0 as greedy.
    validate_generation({'temperature': -1, 'top_k': 64, 'seed': -1}, 'strata')
    validate_generation({'gpu_memory_utilization': 1, 'temperature': 2}, 'vllm')
    fields = {field.key: field for field in generation_fields('vllm')}
    assert fields['temperature'].maximum == 2
    assert fields['max_tokens'].maximum is None


@pytest.mark.parametrize('ratio', ['3,1', '1/0/2', '0,0', '1,,2', '1,', 1])
def test_llama_native_split_ratios_keep_automatic_and_legacy_values(tmp_path, ratio):
    model, profile, runtime, context = base(tmp_path)
    profile.parameters = {'tensor_split': ratio, 'main_gpu': 999}
    adapter = get_adapters()['llama_cpp']
    validate_profile(model, profile, runtime, adapter)
    argv = adapter.build_launch(model, profile, runtime, context).argv
    assert argv[argv.index('--tensor-split') + 1] == str(ratio)
    assert argv[argv.index('--main-gpu') + 1] == '999'


@pytest.mark.parametrize('backend,key,value', [
    ('llama_cpp', 'tensor_split', '-1,2'),
    ('llama_cpp', 'tensor_split', 'NaN,1'),
    ('llama_cpp', 'tensor_split', 'inf'),
    ('llama_cpp', 'tensor_split', 'one,2'),
    ('llama_cpp', 'tensor_split', ',1'),
    ('llama_cpp', 'main_gpu', -1),
    ('llama_cpp', 'main_gpu', 1.5),
    ('vllm', 'pipeline_parallel_size', 0),
    ('sglang', 'pipeline_parallel_size', 1.5),
    ('vllm', 'cuda_visible_devices', ['0', '1']),
    ('sglang', 'cuda_visible_devices', '0\n1'),
    ('ninfer', 'cuda_visible_devices', True),
    ('sparkinfer', 'cuda_visible_devices', '0\0'),
    ('strata', 'gpu', -1),
    ('strata', 'gpu', [0, 1.5]),
    ('strata', 'gpu', 'GPU-22222222'),
])
def test_native_gpu_invalid_values_are_rejected_by_shared_profile_rules(tmp_path, backend, key, value):
    model, profile, runtime, _ = base(tmp_path, backend)
    profile.parameters[key] = value
    with pytest.raises(ValueError):
        validate_profile(model, profile, runtime, get_adapters()[backend])


@pytest.mark.parametrize('backend', ['llama_cpp', 'vllm'])
async def test_native_gpu_api_save_preview_and_clear(tmp_path, monkeypatch, backend):
    import httpx
    from modloader.app import Controller, build_app
    from modloader.backends import adapters
    from test_backends import fixtures
    model, profile, runtime, _ = fixtures(tmp_path, backend, hf=backend == 'vllm')
    if backend == 'vllm':
        runtime.path = '/opt/vllm/bin/vllm'
        runtime.metadata = {'distribution': 'Fixture', 'build_environment': {'CUDA_VISIBLE_DEVICES': 'legacy-build-only'}}
        monkeypatch.setattr(adapters, 'model_target_path', lambda model, runtime: '/models/fixture')
        parameters = {'tensor_parallel_size': 8, 'pipeline_parallel_size': 4,
                      'cuda_visible_devices': 'GPU-22222222-3333-4444-5555-666666666666'}
        expected = {'--tensor-parallel-size': '8', '--pipeline-parallel-size': '4'}
        invalid = {**parameters, 'pipeline_parallel_size': 0}
        extra = ['--pipeline-parallel-size', '2']
    else:
        parameters = {'devices': 'CUDA2,CUDA0', 'split_mode': 'tensor', 'tensor_split': '0,0', 'main_gpu': 0}
        expected = {'--device': 'CUDA2,CUDA0', '--split-mode': 'tensor', '--tensor-split': '0,0', '--main-gpu': '0'}
        invalid = {**parameters, 'tensor_split': '-1,2'}
        extra = ['--tensor-split', '3,1']
    profile.parameters, profile.extra_args = parameters, extra
    controller = Controller(tmp_path / 'gpu-api-data', 'fixture-management')
    for kind, value in [('models', model), ('runtimes', runtime)]:
        controller.store.put(kind, value.model_dump())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                    headers={'Authorization': 'Bearer fixture-management'}) as client:
            saved = await client.post('/api/profiles', json=profile.model_dump())
            assert saved.status_code == 200, saved.text
            identifier = saved.json()['id']
            assert saved.json()['parameters'] == parameters
            preview = await client.post('/api/profiles/' + identifier + '/preview')
            assert preview.status_code == 200, preview.text
            spec = preview.json()
            for flag, value in expected.items():
                assert spec['argv'][spec['argv'].index(flag) + 1] == value
            assert spec['argv'][-len(extra):] == extra
            if backend == 'vllm':
                assert spec['env']['CUDA_VISIBLE_DEVICES'] == parameters['cuda_visible_devices']
            rejected = await client.patch('/api/profiles/' + identifier, json={'parameters': invalid})
            assert rejected.status_code == 400, rejected.text
            assert controller.store.get('profiles', identifier)['parameters'] == parameters
            cleared = await client.patch('/api/profiles/' + identifier,
                                         json={'parameters': {key: '' for key in parameters}, 'extra_args': []})
            assert cleared.status_code == 200, cleared.text
            preview = await client.post('/api/profiles/' + identifier + '/preview')
            assert preview.status_code == 200, preview.text
            assert not set(expected) & set(preview.json()['argv'])
            if backend == 'vllm':
                assert 'CUDA_VISIBLE_DEVICES' not in preview.json()['env']
            assert not controller.runtime.instances
    finally:
        await controller.close()


def test_gpu_limit_includes_all_blocks_and_output_and_keeps_legacy_all(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    model.metadata.update({'qwen35.block_count': 65, 'qwen35.nextn_predict_layers': 1})
    assert gpu_layer_limit(model) == 66
    profile.parameters['gpu_layers'] = 999
    with pytest.raises(ValueError, match='GPU'):
        validate_profile(model, profile, runtime, get_adapters()['llama_cpp'])
    validate_profile(model, profile, runtime, get_adapters()['llama_cpp'], legacy_gpu=True)
    model.metadata.clear()
    assert gpu_layer_limit(model) is None


@pytest.mark.parametrize('layers', [0, 12, 66, -1, 'all', 'auto', None])
def test_gpu_launch_values_and_unset_preserve_their_contract(tmp_path, layers):
    model, profile, runtime, context = base(tmp_path)
    model.metadata.update({'qwen35.block_count': 65, 'qwen35.nextn_predict_layers': 1})
    if layers is not None:
        profile.parameters['gpu_layers'] = layers
    before = profile.model_dump()
    adapter = get_adapters()['llama_cpp']
    validate_profile(model, profile, runtime, adapter)
    argv = adapter.build_launch(model, profile, runtime, context).argv
    assert argv[argv.index('--n-gpu-layers') + 1] == str(-1 if layers is None else layers)
    assert profile.model_dump() == before


@pytest.mark.parametrize('layers', [-3, 67, 1.5, float('nan'), True])
def test_new_gpu_numeric_values_reject_invalid_input(tmp_path, layers):
    model, profile, runtime, _ = base(tmp_path)
    model.metadata['qwen35.block_count'] = 65
    profile.parameters['gpu_layers'] = layers
    with pytest.raises(ValueError, match='GPU'):
        validate_profile(model, profile, runtime, get_adapters()['llama_cpp'])


def test_layer_limits_are_model_and_backend_specific(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    model.metadata['qwen35.block_count'] = 48
    assert gpu_layer_limit(model) == 49
    runtime.metadata['variant'] = 'cpu'
    assert gpu_layer_limit(model, runtime) == 0
    field = next(f for f in profile_fields(get_adapters()['llama_cpp'], model, runtime) if f.key == 'gpu_layers')
    assert field.maximum == 0
    assert not any(f.key == 'gpu_layers' for f in profile_fields(get_adapters()['strata'], model))
    runtime.metadata.clear()
    model.metadata.clear()
    assert gpu_layer_limit(model, runtime) is None


async def test_gpu_legacy_api_save_copy_and_preview(tmp_path):
    import httpx
    from modloader.app import Controller, build_app
    model, profile, runtime, _ = base(tmp_path)
    model.metadata['qwen35.block_count'] = 65
    profile.parameters = {'gpu_layers': 999}
    controller = Controller(tmp_path / 'data', 'test-management')
    for kind, value in [('models', model), ('profiles', profile), ('runtimes', runtime)]:
        controller.store.put(kind, value.model_dump())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer test-management'}) as client:
            response = await client.patch('/api/profiles/profile', json={'name': 'Renamed'})
            assert response.status_code == 200, response.text
            assert response.json()['parameters']['gpu_layers'] == 999
            duplicate = {**profile.model_dump(), 'copy_from': profile.id, 'alias': 'copy'}
            response = await client.post('/api/profiles', json=duplicate)
            assert response.status_code == 200, response.text
            assert response.json()['parameters']['gpu_layers'] == 999
            for layers in [0, 17, 66]:
                response = await client.patch('/api/profiles/profile', json={'parameters': {'gpu_layers': layers}})
                assert response.status_code == 200, response.text
                saved = controller.store.get('profiles', profile.id)
                assert saved['parameters']['gpu_layers'] == layers
                preview = await client.post('/api/profiles/profile/preview')
                assert preview.status_code == 200, preview.text
                argv = preview.json()['argv']
                assert argv[argv.index('--n-gpu-layers') + 1] == str(layers)
            response = await client.patch('/api/profiles/profile', json={'parameters': {'gpu_layers': 67}})
            assert response.status_code == 400
            assert controller.store.get('profiles', profile.id)['parameters']['gpu_layers'] == 66
            model.metadata.clear()
            controller.store.put('models', model.model_dump())
            response = await client.patch('/api/profiles/profile', json={'parameters': {'gpu_layers': 1000}})
            assert response.status_code == 200, response.text
            preview = await client.post('/api/profiles/profile/preview')
            assert preview.status_code == 200, preview.text
            argv = preview.json()['argv']
            assert argv[argv.index('--n-gpu-layers') + 1] == '1000'
    finally:
        await controller.close()


async def test_gpu_import_preserves_legacy_all_argv_instead_of_rewriting(tmp_path, monkeypatch):
    import struct
    import httpx
    from modloader.app import Controller, build_app
    from modloader.backends import adapters

    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw

    weights = tmp_path / 'model.gguf'
    content = b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
    content += string('general.architecture') + struct.pack('<I', 8) + string('llama')
    content += string('llama.block_count') + struct.pack('<II', 4, 32)
    weights.write_bytes(content)
    executable = tmp_path / 'llama-server.exe'
    executable.touch()
    script = tmp_path / 'legacy.cmd'
    script.write_text(f'"{executable}" --model "{weights}" --n-gpu-layers 999 --ctx-size 1048576\n')
    monkeypatch.setattr(adapters, '_run_readonly', lambda argv: '')
    controller = Controller(tmp_path / 'data', 'test-management')
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer test-management'}) as client:
            response = await client.post('/api/config/import', json={'path': str(script)})
            assert response.status_code == 200, response.text
            profile = response.json()['profile']
            assert profile['parameters'] == {'gpu_layers': 999, 'context_length': 1048576}
            preview = await client.post(f"/api/profiles/{profile['id']}/preview")
            assert preview.status_code == 200, preview.text
            argv = preview.json()['argv']
            assert argv[argv.index('--n-gpu-layers') + 1] == '999'
            assert argv[argv.index('--ctx-size') + 1] == '1048576'
    finally:
        await controller.close()


def test_gpu_limit_refreshes_real_gguf_header_over_registration(tmp_path):
    import struct
    model, _, _, _ = base(tmp_path)
    model.metadata['qwen35.block_count'] = 65
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    from pathlib import Path
    content = b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
    content += string('general.architecture') + struct.pack('<I', 8) + string('llama')
    content += string('llama.block_count') + struct.pack('<II', 4, 32)
    Path(model.path).write_bytes(content)
    assert gpu_layer_limit(model) == 33
