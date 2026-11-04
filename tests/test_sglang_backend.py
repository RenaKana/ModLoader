import asyncio
import json
from pathlib import Path
import struct

import pytest

from modloader.backends import catalog, get_adapters, import_runtime, installers
from modloader.backends import sglang
from modloader.backends import source_builds
from modloader.backends.sglang_versions import RECIPES, recipe
from modloader.backends.telemetry import Metrics, _prometheus
from modloader.models import AppPaths, LaunchContext, LoadProfile, ModelAsset
from modloader.parameter_rules import validate_generation, validate_profile


def fixture(tmp_path, monkeypatch):
    weights = tmp_path / 'local hf'
    weights.mkdir()
    (weights / 'config.json').write_text('{"model_type":"qwen3_5"}')
    header = json.dumps({'mtp.layers.0.weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
    (weights / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0\0')
    model = ModelAsset(id='model', name='Local HF', path=str(weights), format='safetensors', architecture='qwen3_5')
    runtime = import_runtime('sglang', '/opt/sglang/bin/python', {'distribution': 'ExternalUbuntu'})
    profile = LoadProfile(id='profile', model_id=model.id, name='SGLang', backend='sglang', runtime_id=runtime.id, alias='local-model')
    context = LaunchContext(port=12345, instance_id='fixture', work_dir=str(tmp_path), data_dir=str(tmp_path))
    monkeypatch.setattr(sglang, 'model_target_path', lambda model, runtime: '/mnt/d/local hf')
    return model, runtime, profile, context


def test_sglang_native_argv_and_optional_context_preserve_external_distribution(tmp_path, monkeypatch):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.platform == 'wsl2' and spec.distribution == 'ExternalUbuntu'
    assert spec.argv[:5] == ['/opt/sglang/bin/python', '-m', 'sglang.launch_server', '--model-path', '/mnt/d/local hf']
    assert '--context-length' not in spec.argv and '--enable-metrics' in spec.argv
    assert spec.argv[spec.argv.index('--host') + 1] == '127.0.0.1'
    assert spec.health_path == '/health' and '/health_generate' not in spec.argv
    assert spec.env['SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION'] == '0'
    assert spec.env['FLASHINFER_NO_DOWNLOAD'] == '1'
    assert spec.capabilities['stream_usage'] is True
    assert get_adapters()['sglang'].inspect(model, runtime).status == 'unverified'
    assert not any('wsl.exe' in arg for arg in spec.argv)
    with pytest.raises(ValueError, match='WSL2'):
        import_runtime('sglang', 'python.exe', platform='windows')


def test_sglang_launch_uses_its_own_toolchain_and_bounded_offline_jit_caches(tmp_path, monkeypatch):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    root = '/opt/modloader/sglang-fixture'
    runtime.metadata['build_environment'] = sglang.toolchain_environment(root)
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.env['CUDA_HOME'] == root + '/cuda' and spec.env['CUDACXX'] == root + '/cuda/bin/nvcc'
    assert spec.env['PATH'].startswith(root + '/bin:' + root + '/cuda/bin:')
    assert spec.env['DJ_JIT_CACHE_DIR'] == root + '/cache/deep-jit'
    assert spec.env['FLASHINFER_WORKSPACE_BASE'] == root + '/cache/sglang'
    assert spec.env['TORCH_EXTENSIONS_DIR'].startswith(root + '/') and spec.env['MAX_JOBS'] == '4'
    assert spec.env['FLASHINFER_NVCC_THREADS'] == '1' and spec.env['FLASHINFER_NO_DOWNLOAD'] == '1'
    assert 'SGLANG_ENABLE_JIT_DEEPGEMM' not in spec.env  # Keep the actual kernel capability enabled.


def test_sglang_native_tp_pp_and_profile_cuda_environment_keep_extra_priority(tmp_path, monkeypatch):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    runtime.metadata['build_environment'] = {'CUDA_VISIBLE_DEVICES': '7', 'BUILD_VALUE': 'keep'}
    profile.parameters = {'tensor_parallel_size': 8, 'pipeline_parallel_size': 4,
                          'cuda_visible_devices': 'GPU-22222222-3333-4444-5555-666666666666,GPU-88888888'}
    profile.extra_args = ['--pipeline-parallel-size', '2', '--tensor-parallel-size=16']
    before = profile.model_dump()
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--tp-size') + 1] == '8'
    assert spec.argv[spec.argv.index('--pp-size') + 1] == '4'
    assert spec.argv[-len(profile.extra_args):] == profile.extra_args
    assert spec.env['CUDA_VISIBLE_DEVICES'] == profile.parameters['cuda_visible_devices']
    assert spec.env['BUILD_VALUE'] == 'keep'
    assert runtime.metadata['build_environment']['CUDA_VISIBLE_DEVICES'] == '7'
    assert profile.model_dump() == before
    profile.parameters = {'pipeline_parallel_size': '  ', 'cuda_visible_devices': '  '}
    profile.extra_args = []
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert '--pp-size' not in spec.argv and spec.env['CUDA_VISIBLE_DEVICES'] == '7'
    runtime.metadata.pop('build_environment')
    assert 'CUDA_VISIBLE_DEVICES' not in get_adapters()['sglang'].build_launch(model, profile, runtime, context).env


def test_sglang_mtp_has_explicit_window_semantics_and_no_arbitrary_cap(tmp_path, monkeypatch):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    profile.parameters = {'acceleration_mode': 'mtp', 'speculative_num_steps': 100, 'speculative_num_draft_tokens': 101}
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--speculative-algorithm') + 1] == 'NEXTN'
    assert spec.argv[spec.argv.index('--speculative-num-steps') + 1] == '100'
    assert sglang.acceleration(model, profile, runtime)['status'] == 'available'
    profile.parameters['speculative_num_draft_tokens'] = 100
    validate_profile(model, profile, runtime, get_adapters()['sglang'])
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--speculative-num-draft-tokens') + 1] == '100'
    assert profile.parameters['speculative_num_draft_tokens'] == 100
    profile.parameters = {'acceleration_mode': 'none', 'speculative_num_steps': 100}
    assert sglang.acceleration(model, profile, runtime)['fields']
    profile.parameters.update(speculative_num_steps='unused-old-value', speculative_eagle_topk=-99)
    validate_profile(model, profile, runtime, get_adapters()['sglang'])
    runtime.metadata['supported_flags'] = ['--dtype', '--tp-size']
    profile.parameters['cuda_graph_backend_decode'] = 'disabled'
    with pytest.raises(ValueError, match='help'):
        validate_profile(model, profile, runtime, get_adapters()['sglang'])


@pytest.mark.parametrize('extra', [['--model-path', 'remote/repo'], ['--speculative-algorithm', 'EAGLE'], ['--port', '8']])
def test_sglang_advanced_args_cannot_override_local_model_or_acceleration(tmp_path, monkeypatch, extra):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    profile.extra_args = extra
    with pytest.raises(ValueError):
        get_adapters()['sglang'].build_launch(model, profile, runtime, context)


@pytest.mark.parametrize('extra', [['--tool-call-parser', 'qwen3_coder'], ['--enable-auto-tool-choice']])
def test_sglang_tool_output_options_are_not_tool_execution(tmp_path, monkeypatch, extra):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    runtime.metadata['supported_flags'] = ['--dtype', '--tp-size', '--tool-call-parser', '--enable-auto-tool-choice']
    profile.extra_args = extra
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[-len(extra):] == extra


def test_sglang_sampling_contract_maps_without_silently_dropping_fields():
    payload = {'repeat_penalty': 1.1, 'top_k': -1, 'max_tokens': 0, 'stop': ['END'], 'response_format': {'type': 'json_object'}}
    validate_generation(payload, 'sglang')
    mapped = sglang.map_generation(payload)
    assert mapped['repetition_penalty'] == 1.1 and 'repeat_penalty' not in mapped
    assert mapped['stop'] == ['END'] and mapped['response_format'] == payload['response_format']
    for invalid in ({'top_k': 0}, {'top_k': -2}, {'top_p': 0}, {'repeat_penalty': 2.1}, {'repetition_penalty': 0}):
        with pytest.raises(ValueError):
            validate_generation(invalid, 'sglang')
    with pytest.raises(ValueError, match='冲突'):
        sglang.map_generation({'repeat_penalty': 1.1, 'repetition_penalty': 1.2})


def test_sglang_metrics_keep_counters_window_acceptance_and_model_rate_distinct():
    metrics = Metrics()
    _prometheus('''sglang:prompt_tokens_total 100
sglang:generation_tokens_total 80
sglang:gen_throughput 25
sglang:token_usage 0.4
sglang:spec_accept_length 2.4
sglang:spec_accept_rate 0.7
''', 'sglang', metrics)
    values = {metric['id']: metric for group in metrics.result()['groups'] for metric in group['metrics']}
    assert values['generation_tokens_total']['kind'] == 'counter' and values['generation_tokens_total']['unit'] == 'token'
    assert values['gen_throughput']['kind'] == 'gauge' and values['gen_throughput']['unit'] == 'token/s'
    assert values['token_usage']['value'] == .4
    assert '40' in values['spec_accept_rate']['description'] and 'bonus' in values['spec_accept_length']['description']
    assert 'bonus' not in values['spec_accept_rate']['description']


def test_sglang_token_counters_preserve_streaming_series_and_stable_identity():
    # 0.5.21 TokenizerMetricsCollector adds is_streaming to its base labels in
    # observe_one_finished_request. These values reproduce the real MTP run.
    text = '''# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="local-model",engine_type="unified",is_streaming="false"} 112
sglang:prompt_tokens_total{model_name="local-model",engine_type="unified",is_streaming="true"} 32
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="local-model",engine_type="unified",is_streaming="false"} 136
sglang:generation_tokens_total{model_name="local-model",engine_type="unified",is_streaming="true"} 128
'''
    result = Metrics()
    _prometheus(text, 'sglang', result)
    metrics = result.groups['sglang_totals']['metrics']
    assert len(metrics) == len({metric['id'] for metric in metrics}) == 4
    for name, expected in [('prompt_tokens_total', {'false': 112, 'true': 32}),
                           ('generation_tokens_total', {'false': 136, 'true': 128})]:
        for streaming, value in expected.items():
            found = next(metric for metric in metrics if metric['id'].startswith(name)
                         and f'is_streaming={streaming}' in metric['label'])
            assert found['value'] == value and found['kind'] == 'counter'
            assert 'model_name=local-model' in found['label'] and 'engine_type=unified' in found['label']
            assert ('；流式请求序列' if streaming == 'true' else '；非流式请求序列') in found['description']

    # Neither scrape order nor native label order changes a series identity.
    reversed_text = '\n'.join(reversed(text.splitlines())).replace(
        'model_name="local-model",engine_type="unified",', 'engine_type="unified",model_name="local-model",')
    reordered = Metrics()
    _prometheus(reversed_text, 'sglang', reordered)
    assert {metric['id']: metric for metric in reordered.groups['sglang_totals']['metrics']} == {
        metric['id']: metric for metric in metrics}


@pytest.mark.parametrize('dimension', ['model_name', 'engine_type', 'moe_ep_rank', 'priority', 'deployment'])
def test_sglang_scheduler_preserves_native_and_configured_extra_label_dimensions(dimension):
    # SchedulerMetricsCollector supports these native labels plus explicitly
    # configured extra_metric_labels; none may collapse into another series.
    lines = []
    for value in ['first', 'second']:
        labels = {'model_name': 'local-model', 'engine_type': 'unified', dimension: value}
        encoded = ','.join(f'{key}={json.dumps(value)}' for key, value in labels.items())
        lines.append(f'sglang:num_queue_reqs{{{encoded}}} 1')
    result = Metrics()
    _prometheus('\n'.join(lines), 'sglang', result)
    metrics = result.groups['sglang_requests']['metrics']
    assert len(metrics) == len({metric['id'] for metric in metrics}) == 2
    assert all(metric['value'] == 1 for metric in metrics)
    assert all(any(f'{dimension}={value}' in metric['label'] for metric in metrics) for value in ['first', 'second'])


@pytest.mark.parametrize('version', ['0.5.21', '0.5.23'])
async def test_sglang_plan_pins_wheel_version_and_reuses_existing_owned_distro(tmp_path, monkeypatch, version):
    selected = recipe(version)
    paths = AppPaths(tmp_path)
    (paths.runtimes / 'ModLoader-wsl-owned.json').write_text('{}')
    async def metadata(url, proxy=None):
        if url.endswith('redistrib_13.4.3.json'):
            return {name: {'linux-x86_64': {'relative_path': name + '/linux-x86_64/archive.tar.xz', 'sha256': 'b' * 64, 'size': 42}}
                    for name in sglang.CUDA_COMPONENTS}
        assert any(url.endswith('/sglang/' + item + '/json') for item in RECIPES)
        return {'urls': [{'packagetype': 'bdist_wheel', 'digests': {'sha256': 'a' * 64}}]}
    monkeypatch.setattr(catalog, '_json', metadata)
    plan = await catalog.plan_install('sglang', paths, {'version': version, 'variant': 'wsl2'})
    assert [item.get('role') for item in plan['stage']['files']] == [
        *[item['role'] for item in selected['approved_wheels']], *['cuda_' + name for name in sglang.CUDA_COMPONENTS]]
    assert plan['variant'] == 'wsl2' and plan['provenance']['cuda_release'] == '13.4.3'
    assert plan['provenance']['cuda_components'] == list(sglang.CUDA_COMPONENTS)
    assert 'cuda_nvml_dev' not in sglang.CUDA_COMPONENTS
    if selected['approved_wheels']:
        assert plan['stage']['files'][0]['url'].startswith('https://pypi.nvidia.com/cuda-tile/')
    else:
        assert not any('pypi.nvidia.com/cuda-tile/' in item['url'] for item in plan['stage']['files'])
    assert plan['provenance']['source_commit'] == selected['source_commit']
    versions = await catalog.list_versions('sglang')
    assert [item['version'] for item in versions['versions']] == ['0.5.23', '0.5.21']
    assert [item['version'] for item in versions['versions'] if item['recommended']] == ['0.5.23']
    with pytest.raises(ValueError, match='只支持'):
        await catalog.plan_install('sglang', paths, {'version': '0.5.17'})


@pytest.mark.parametrize('version', ['0.5.21', '0.5.23'])
async def test_sglang_install_uses_isolated_venv_approved_stages_and_entrypoint(tmp_path, monkeypatch, version):
    selected = recipe(version)
    paths, commands, requirements = AppPaths(tmp_path), [], []
    async def command(argv, *args, **kwargs): commands.append(argv); return ''
    async def packages(python, requested, root, transfer, confirm, **kwargs):
        assert kwargs['wsl'] is True and confirm is not None
        if selected['approved_wheels']:
            assert kwargs['approved_wheels'][0]['url'].startswith('https://pypi.nvidia.com/cuda-tile/')
        else:
            assert kwargs['approved_wheels'] is None
        requirements.extend(requested)
        return [{'package': 'sglang', 'version': version}]
    async def proxy(*args): return 'http://configured-proxy:17890'
    async def apt(*args, **kwargs): assert kwargs['resolver_proxy'] == 'http://configured-proxy:17890'
    async def validate(runtime, **kwargs): return runtime
    async def artifacts(self, stage): return {'cuda_tile': tmp_path / 'cuda_tile.whl',
                                            **{'cuda_' + name: tmp_path / (name + '.tar.xz') for name in sglang.CUDA_COMPONENTS}}
    from modloader import platforms
    monkeypatch.setattr(platforms, 'windows_to_wsl', lambda value: '/mnt/fixture/' + Path(value).name)
    monkeypatch.setattr(installers, 'list_wsl', lambda: {'available': True, 'distributions': [{'name': 'ModLoader', 'version': 2}]})
    monkeypatch.setattr(installers, '_verify_wsl_owner', lambda _: None)
    monkeypatch.setattr(installers, '_wsl_command', command)
    monkeypatch.setattr(installers, '_wsl_proxy', proxy)
    monkeypatch.setattr(installers, '_apt_stage', apt)
    monkeypatch.setattr(installers, '_package_stage', packages)
    monkeypatch.setattr(installers, 'validate_runtime', validate)
    monkeypatch.setattr(installers._Transfers, 'fetch', artifacts)
    monkeypatch.setattr(installers, 'windows_to_wsl', lambda value: '/mnt/fixture/' + Path(value).name)
    plan = dict(id='plan', backend='sglang', version=version, variant='wsl2', stage={'id': 'base', 'title': '基础文件', 'files': []})
    runtime = await installers.install_backend('sglang', paths, {'_plan': plan}, None, asyncio.Event(), confirm=lambda stage: None)
    if selected['approved_wheels']:
        assert requirements[:3] == list(selected['requirements'])
        assert requirements[3].startswith('cuda-tile @ file:///mnt/fixture/cuda_tile.whl#sha256=')
    else:
        assert requirements == list(selected['requirements'])
    assert commands[0] == ['/usr/bin/python3', '-m', 'venv', runtime.metadata['linux_root']]
    assert len(commands) == 2 and commands[1][1].endswith('/stage-native-build.py')
    stage = json.loads((Path(runtime.metadata['managed_root']) / 'stage-native-build.json').read_text())
    assert len(stage['archives']) == len(sglang.CUDA_COMPONENTS) and all(item['destination'] == 'cuda' for item in stage['archives'])
    assert runtime.path == runtime.metadata['linux_root'] + '/bin/python' and runtime.managed
    assert runtime.metadata['build_environment'] == sglang.toolchain_environment(runtime.metadata['linux_root'])
    assert runtime.metadata['cuda_release'] == source_builds.CUDA_RELEASE
    assert (Path(runtime.metadata['managed_root']) / 'modloader-runtime.json').is_file()


async def test_sglang_toolchain_repair_rejects_incomplete_confirmed_manifest_before_staging(tmp_path):
    with pytest.raises(ValueError, match='cuda_nvrtc'):
        await sglang.stage_toolchain(tmp_path, '/opt/modloader/sglang-fixture',
                                    {'cuda_' + name: tmp_path / name for name in sglang.CUDA_COMPONENTS if name != 'cuda_nvrtc'},
                                    None, asyncio.Event())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('cuda_home', [None, '/opt/modloader/ninfer-other/cuda'])
async def test_managed_sglang_cannot_validate_with_missing_or_borrowed_toolchain(monkeypatch, cuda_home):
    runtime = import_runtime('sglang', '/opt/modloader/sglang-fixture/bin/python',
                             {'linux_root': '/opt/modloader/sglang-fixture',
                              'build_environment': {'CUDA_HOME': cuda_home} if cuda_home else {}})
    async def forbidden(*args, **kwargs): raise AssertionError('invalid environment must not be executed')
    monkeypatch.setattr(installers, '_command', forbidden)
    with pytest.raises(ValueError, match='自身独立'):
        await installers.validate_runtime(runtime, _installing=True)


@pytest.mark.parametrize('version', ['0.5.21', '0.5.23'])
async def test_sglang_refresh_reads_version_and_help_without_loading_model(tmp_path, monkeypatch, version):
    runtime = import_runtime('sglang', '/opt/sglang/bin/python', {'distribution': 'ExternalUbuntu'})
    commands = []
    async def command(argv, *args, **kwargs):
        commands.append(argv)
        return version if '-c' in argv else '--model-path --host --port --served-model-name --enable-metrics'
    monkeypatch.setattr(installers, '_command', command)
    refreshed = await installers.validate_runtime(runtime)
    assert refreshed.version == version
    assert all(argv[:4] == ['wsl.exe', '--distribution', 'ExternalUbuntu', '--exec'] for argv in commands)
    assert commands[1][-3:] == ['-m', 'sglang.launch_server', '--help']
    assert refreshed.metadata['validation'] == 'entrypoint'


@pytest.mark.parametrize('version,options', [
    ('0.5.21', ['full', 'breakable', 'tc_piecewise', 'disabled']),
    ('0.5.23', ['full', 'breakable', 'disabled']),
    ('unknown', ['full', 'breakable', 'tc_piecewise', 'disabled']),
])
def test_sglang_graph_fields_follow_the_selected_runtime(version, options):
    runtime = import_runtime('sglang', '/opt/sglang/bin/python', {'version': version})
    fields = {field.key: field for field in get_adapters()['sglang'].parameters_for_runtime(runtime)}
    assert fields['cuda_graph_backend_decode'].options == options
    assert fields['cuda_graph_backend_prefill'].options == options


@pytest.mark.parametrize('key', ['cuda_graph_backend_decode', 'cuda_graph_backend_prefill'])
def test_sglang_0523_rejects_retired_graph_values_without_migrating_saved_profiles(tmp_path, monkeypatch, key):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    profile.parameters[key] = 'tc_piecewise'
    before = profile.model_dump()
    runtime.version = '0.5.21'
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index(sglang.MAPPING[key]) + 1] == 'tc_piecewise'
    runtime.version = '0.5.23'
    for operation in (lambda: validate_profile(model, profile, runtime, get_adapters()['sglang']),
                      lambda: get_adapters()['sglang'].build_launch(model, profile, runtime, context)):
        with pytest.raises(ValueError, match=r'0\.5\.23.*tc_piecewise.*full.*breakable.*disabled'):
            operation()
    assert profile.model_dump() == before


@pytest.mark.parametrize('extra', [
    ['--cuda-graph-backend-decode', 'tc_piecewise'],
    ['--cuda-graph-backend-prefill=tc_piecewise'],
    ['--cuda-graph-config', '{"decode":{"backend":"tc_piecewise"}}'],
    ['--cuda-graph-config={"prefill":{"tc_compiler":"inductor"}}'],
])
def test_sglang_0523_reports_retired_graph_values_in_advanced_arguments(tmp_path, monkeypatch, extra):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    runtime.version = '0.5.23'
    profile.extra_args = extra
    before = profile.model_dump()
    with pytest.raises(ValueError, match=r'0\.5\.23.*(tc_piecewise|tc_compiler)'):
        get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert profile.model_dump() == before
    runtime.version = '0.5.21'
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[-len(extra):] == extra


def test_sglang_0523_reports_removed_speculative_flag_without_discarding_user_arguments(tmp_path, monkeypatch):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    profile.extra_args = ['--speculative-skip-dp-mlp-sync']
    before = profile.model_dump()
    runtime.version = '0.5.23'
    with pytest.raises(ValueError, match=r'0\.5\.23.*--speculative-skip-dp-mlp-sync'):
        get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert profile.model_dump() == before
    runtime.version = '0.5.21'
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[-1] == profile.extra_args[0]


def test_sglang_actual_flag_inventory_takes_precedence_over_a_known_removed_flag(tmp_path, monkeypatch):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    runtime.version = '0.5.23'
    profile.extra_args = ['--speculative-skip-dp-mlp-sync']
    runtime.metadata['supported_flags'] = ['--dtype', '--tp-size', *profile.extra_args]
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[-1] == profile.extra_args[0]
    runtime.metadata['supported_flags'].remove(profile.extra_args[0])
    with pytest.raises(ValueError, match='help.*--speculative-skip-dp-mlp-sync'):
        get_adapters()['sglang'].build_launch(model, profile, runtime, context)


@pytest.mark.parametrize('version', ['0.5.21', '0.5.23', '0.5.99'])
def test_sglang_known_versions_keep_native_nextn_and_unknown_advanced_capabilities(tmp_path, monkeypatch, version):
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    runtime.version = version
    profile.parameters = {'acceleration_mode': 'mtp', 'cuda_graph_backend_decode': 'breakable'}
    profile.extra_args = ['--tool-call-parser', 'qwen3_coder', '--new-plugin-flag', 'custom']
    before = profile.model_dump()
    spec = get_adapters()['sglang'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--speculative-algorithm') + 1] == 'NEXTN'
    assert spec.argv[-len(profile.extra_args):] == profile.extra_args
    assert spec.health_path == '/health' and spec.chat_path == '/v1/chat/completions'
    assert profile.model_dump() == before
    reasons = get_adapters()['sglang'].inspect(model, runtime).reasons
    assert any('所选运行时版本' in reason for reason in reasons) is (version == '0.5.99')


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16-le'])
def test_installer_shared_wsl_listing_accepts_utf8_and_legacy_output(monkeypatch, encoding):
    from types import SimpleNamespace
    from modloader import platforms
    monkeypatch.setenv('WSL_UTF8', '1')
    output = '  NAME          STATE           VERSION\r\n* ModLoader     Running         2\r\n'.encode(encoding)
    monkeypatch.setattr(platforms.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output))
    result = installers.list_wsl()
    assert result['available'] and result['distributions'][0]['name'] == 'ModLoader'
    assert result['distributions'][0]['version'] == 2
