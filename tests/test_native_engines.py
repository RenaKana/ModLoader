"""Native backend contracts and real CPU-only preparation lifecycle fixtures."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import psutil
import pytest

from modloader.app import Controller, build_app
from modloader.backends import adapters, get_adapters, import_runtime, installers
from modloader.backends import native_engines as native
from modloader.backends.telemetry import Metrics, _prometheus
from modloader import engine_preparation as preparation
from modloader.models import AppPaths, LaunchContext, LaunchSpec, LoadProfile, ModelAsset, RuntimeInstallation
from modloader.parameter_rules import profile_fields, validate_generation, validate_profile


class MemoryStore:
    def __init__(self):
        self.records = {}

    def all(self, kind):
        return deepcopy(list(self.records.get(kind, {}).values()))

    def put(self, kind, row):
        self.records.setdefault(kind, {})[row['id']] = deepcopy(row)

    def remove(self, kind, identifier):
        self.records.get(kind, {}).pop(identifier)


def write_hf_weights(path, tensors):
    records, offset = {}, 0
    for name, tensor in tensors.items():
        size = {'BF16': 2, 'F16': 2, 'F32': 4}[tensor['dtype']]
        for dim in tensor['shape']:
            size *= dim
        records[name] = {**tensor, 'data_offsets': [offset, offset + size]}
        offset += size
    header = json.dumps(records).encode()
    header += b' ' * (-len(header) % 8)
    path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(offset))


def tokenizer_documents(pipeline=None):
    """Small complete resource schema; inference still uses a CPU server fixture."""
    tokenizer = deepcopy(pipeline or {})
    tokenizer['model'] = {'type': 'BPE', 'vocab': {'a': 0, 'b': 1}, 'merges': [], **tokenizer.get('model', {})}
    tokenizer['added_tokens'] = [dict(id=index, content=content, single_word=False, lstrip=False,
                                     rstrip=False, normalized=False, special=True)
                                 for index, content in [(2, '<|endoftext|>'), (3, '<|im_end|>')]]
    config = {'added_tokens_decoder': {str(item['id']): {key: value for key, value in item.items() if key != 'id'}
                                       for item in tokenizer['added_tokens']},
              'add_bos_token': False, 'add_prefix_space': False, 'pad_token': '<|endoftext|>'}
    return tokenizer, config


@pytest.mark.parametrize('backend,path,source', [
    ('ninfer', '/opt/manual/ninfer/build/apps/ninfer-serve', '/opt/manual/ninfer'),
    ('sparkinfer', '/opt/manual/spark/build/server/sparkinfer_server', '/opt/manual/spark'),
])
async def test_native_import_refresh_uses_help_and_truthful_source_version(monkeypatch, backend, path, source):
    runtime = import_runtime(backend, path, {'distribution': 'ExternalUbuntu'})
    assert runtime.metadata['source_root'] == source and runtime.version == 'unknown'
    assert import_runtime(backend, path, {'distribution': 'OtherUbuntu'}).id != runtime.id
    commands = []
    async def command(argv, *args, **kwargs):
        commands.append(argv)
        return '--host --port --spec --draft-tokens --ctx --tokenizer --draft-model'
    monkeypatch.setattr(installers, '_command', command)
    runtime.version = 'user-supplied-version'
    refreshed = await installers.validate_runtime(runtime)
    assert refreshed.version == 'unknown' and refreshed.metadata['validation'] == 'entrypoint'
    assert len(commands) == 1 and commands[0][-2:] == [path, '--help']
    assert commands[0][:4] == ['wsl.exe', '--distribution', 'ExternalUbuntu', '--exec']
    assert '--version' not in commands[0]
    with pytest.raises(ValueError, match='WSL2'):
        import_runtime(backend, path, platform='windows')


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Run stdlib metadata probes locally, never WSL or the real engine."""
    root = tmp_path / 'HF source'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps({'architectures': ['Qwen3_5ForConditionalGeneration'], 'text_config': {'hidden_size': 5120}}))
    write_hf_weights(root / 'model.safetensors', {
        'model.language_model.layers.0.linear_attn.in_proj_a.weight': {'dtype': 'BF16', 'shape': [48, 5120]},
        'model.language_model.layers.0.linear_attn.in_proj_b.weight': {'dtype': 'BF16', 'shape': [48, 5120]},
    })
    (root / 'chat_template.jinja').write_text('{{ messages }}')
    tokenizer, tokenizer_config = tokenizer_documents()
    (root / 'tokenizer.json').write_text(json.dumps(tokenizer))
    (root / 'tokenizer_config.json').write_text(json.dumps(tokenizer_config))
    (root / 'generation_config.json').write_text('{"eos_token_id":[2,3]}')
    source = tmp_path / 'official-tools-fixture'
    (source / 'tools/convert').mkdir(parents=True)
    (source / 'tools/convert/__main__.py').write_text('# converter-presence fixture')
    model = ModelAsset(id='model', name='Local fixture', path=root.as_posix(), format='safetensors')
    runtime = RuntimeInstallation(id='runtime', backend='ninfer', platform='wsl2', name='Native fixture',
                                  path='/opt/fixture/source/build/apps/ninfer-serve',
                                  metadata={'distribution': 'ExternalUbuntu', 'source_root': source.as_posix(), 'python': sys.executable})
    profile = LoadProfile(id='profile', name='Native', model_id=model.id, runtime_id=runtime.id,
                          backend='ninfer', alias='local-fixture')
    context = LaunchContext(port=12345, instance_id='fixture', work_dir=str(tmp_path), data_dir=str(tmp_path))

    def run(argv, **kwargs):
        index = argv.index('-c')
        result = subprocess.run([sys.executable, *argv[index:]], text=True, capture_output=True, timeout=kwargs.get('timeout', 10))
        if result.returncode:
            raise ValueError(result.stderr)
        return result.stdout.strip()

    def summary(path, runtime):
        return json.loads(Path(path).read_text())

    monkeypatch.setattr(native, '_run', run)
    monkeypatch.setattr(preparation, '_run', run)
    monkeypatch.setattr(native, 'model_target_path', lambda model, runtime: model.path)
    monkeypatch.setattr(preparation, 'model_target_path', lambda model, runtime: model.path)
    monkeypatch.setattr(native, 'resource_target', lambda value, runtime: str(value))
    monkeypatch.setattr(native, 'artifact_info', summary)
    monkeypatch.setattr(adapters, 'model_exists', lambda model: Path(model.path).exists())
    monkeypatch.setattr(preparation, 'windows_to_wsl', lambda value, distribution: Path(value).as_posix())
    return model, runtime, profile, context, source


def artifact(local, components=None):
    model, runtime, profile, context, source = local
    path = Path(model.path).parent / 'model.ninfer'
    path.write_text(json.dumps({'components': components or {'text': {}, 'mtp': {}}}))
    return model.model_copy(update={'path': path.as_posix(), 'format': 'custom'}), runtime, profile, context, source


@pytest.fixture
def inspection_cache(tmp_path, monkeypatch):
    """Exercise the real cache with small files in place of official inspection."""
    paths = [tmp_path / name for name in ('model.ninfer', 'part.ninfer', 'inspect.py', 'python')]
    paths[0].write_text(json.dumps({'components': {'text': {}, 'mtp': {}}}))
    for path in paths[1:]:
        path.write_text('fixture')
    calls = {'inspect': 0, 'stat': 0}
    def stamps(names):
        rows = []
        for name in names:
            path = Path(name)
            stat = path.stat()
            rows.append([str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_mode])
        return rows
    def inspect(path, runtime):
        calls['inspect'] += 1
        return json.loads(Path(path).read_text()), stamps([path, *paths[1:]])
    def check(names, runtime):
        calls['stat'] += 1
        return stamps(names)
    monkeypatch.setattr(native, '_inspect_artifact', inspect)
    monkeypatch.setattr(native, '_artifact_stamps', check)
    monkeypatch.setattr(preparation, 'model_target_path', lambda model, runtime: model.path)
    def no_process(*args, **kwargs):
        raise AssertionError('The cache fixture must not start WSL or an engine')
    monkeypatch.setattr(native, '_run', no_process)
    model = ModelAsset(id='model', name='Fixture', path=str(paths[0]), format='custom')
    runtime = RuntimeInstallation(id=str(tmp_path), backend='ninfer', platform='wsl2', name='Fixture',
        path='/opt/fixture/ninfer-serve', metadata={'distribution': 'Fixture', 'source_root': '/opt/fixture'})
    profile = LoadProfile(id='profile', name='Fixture', backend='ninfer', model_id=model.id,
        runtime_id=runtime.id, alias='fixture')
    prep = preparation.NativePreparation(AppPaths(tmp_path / 'data'), MemoryStore())
    return paths, calls, model, runtime, profile, prep


def test_ninfer_cached_inspection_preserves_current_parameters_and_copy_isolation(inspection_cache):
    paths, calls, model, runtime, profile, prep = inspection_cache
    first = prep.preflight(model, profile, runtime)
    assert first['status'] == 'ready' and first['acceleration']['status'] == 'disabled'
    first['artifact_summary']['components'].clear()
    profile.parameters = {'context_length': 8192, 'acceleration_mode': 'mtp', 'draft_tokens': 1}
    second = prep.preflight(model, profile, runtime)
    assert second['status'] == 'ready' and second['acceleration']['status'] == 'available'
    assert set(second['artifact_summary']['components']) == {'text', 'mtp'}
    # Even an immediately sequential request must stat, not reuse an earlier ready result.
    assert calls == {'inspect': 1, 'stat': 1}
    profile.parameters['context_length'] = 0
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    profile.parameters['context_length'] = 4096
    profile.extra_args = ['--spec', 'mtp']
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    assert calls == {'inspect': 1, 'stat': 1}
    profile.extra_args = []
    paths[0].write_text(json.dumps({'components': {'text': {}}}))
    assert prep.preflight(model, profile, runtime)['status'] == 'error'  # MTP was removed.
    profile.parameters['acceleration_mode'] = 'none'
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert calls['inspect'] == 2


@pytest.mark.parametrize('dependency', [0, 1, 2, 3])
def test_ninfer_inspection_cache_invalidates_each_file_dependency(inspection_cache, dependency):
    paths, calls, model, runtime, profile, prep = inspection_cache
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    paths[dependency].write_text(paths[dependency].read_text() + ' ')
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert calls == {'inspect': 2, 'stat': 1}


def test_ninfer_missing_continuation_and_failed_inspection_never_return_cached_ready(inspection_cache):
    paths, calls, model, runtime, profile, prep = inspection_cache
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    paths[1].unlink()
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    paths[1].write_text('restored continuation')
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    paths[0].write_text('broken artifact')
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    paths[0].write_text('{"components":{"text":{}}}')
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert calls['inspect'] == 5


def test_ninfer_inspection_cache_isolates_models_and_runtime_identity(inspection_cache):
    paths, calls, model, runtime, profile, prep = inspection_cache
    native.artifact_info(model.path, runtime)
    for update in ({'id': 'other-runtime'}, {'version': 'changed'},
                   {'metadata': {**runtime.metadata, 'source_root': '/opt/other-source'}}):
        native.artifact_info(model.path, runtime.model_copy(deep=True, update=update))
    other = paths[0].with_name('other.ninfer')
    other.write_text('{"components":{"text":{}}}')
    assert set(native.artifact_info(str(other), runtime)['components']) == {'text'}
    assert calls['inspect'] == 5


def test_ninfer_concurrent_cold_inspections_share_one_successful_read(inspection_cache, monkeypatch):
    paths, calls, model, runtime, profile, prep = inspection_cache
    original = native._inspect_artifact
    barrier = threading.Barrier(3)
    def inspect(*args):
        time.sleep(.05)
        return original(*args)
    monkeypatch.setattr(native, '_inspect_artifact', inspect)
    def request():
        barrier.wait(timeout=3)
        return native.artifact_info(model.path, runtime)
    with ThreadPoolExecutor(max_workers=3) as pool:
        summaries = list(pool.map(lambda _: request(), range(3)))
    assert all('mtp' in summary['components'] for summary in summaries)
    assert calls == {'inspect': 1, 'stat': 0}
    native.artifact_info(model.path, runtime)
    assert calls == {'inspect': 1, 'stat': 1}


def test_spark_windows_preflight_checks_source_locally_and_retains_linux_launch_paths(local, monkeypatch):
    model, runtime, profile, context, _ = local
    runtime.backend = profile.backend = 'sparkinfer'
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    monkeypatch.setattr(preparation, 'model_target_path', lambda model, runtime: '/mnt/fixture/model')
    def no_remote(*args, **kwargs):
        raise AssertionError('Windows source and default tokenizer need no guest metadata process')
    monkeypatch.setattr(native, 'resource_info', no_remote)
    access = []
    def guest_access(target, tokenizer, info, runtime):
        access.append((target, tokenizer))
    monkeypatch.setattr(native, 'check_spark_access', guest_access)
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'ready'
    assert state['artifact'] == '/mnt/fixture/model' and state['tokenizer'] == '/mnt/fixture/model/tokenizer.json'
    assert access == [('/mnt/fixture/model', '/mnt/fixture/model/tokenizer.json')]
    def unavailable_mount(*args):
        raise ValueError('SparkInfer WSL model path is inaccessible')
    monkeypatch.setattr(native, 'check_spark_access', unavailable_mount)
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    monkeypatch.setattr(native, 'check_spark_access', guest_access)
    root = Path(model.path)
    index = root / 'model.safetensors.index.json'
    index.write_text('{"weight_map":{"a":"missing.safetensors"}}')
    assert 'missing.safetensors' in prep.preflight(model, profile, runtime)['message']
    index.write_text('{"weight_map":{"a":"../escape.safetensors"}}')
    assert 'escaped' in prep.preflight(model, profile, runtime)['message']
    index.write_text('{"weight_map":{"a":"model.safetensors"}}')
    (root / 'tokenizer.json').unlink()
    assert 'tokenizer.json' in prep.preflight(model, profile, runtime)['message']
    (root / 'tokenizer.json').write_text('{}')
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    (root / 'config.json').write_text('[]')
    assert prep.preflight(model, profile, runtime)['status'] == 'error'


def test_resource_inspection_local_and_guest_checks_match(local):
    model, runtime, *_ = local
    assert native.resource_info(model.path, runtime, gdn=True) == native.local_resource_info(model.path, gdn=True)
    assert native.resource_info(str(Path(model.path) / 'tokenizer.json'), runtime) == native.local_resource_info(str(Path(model.path) / 'tokenizer.json'))


def test_spark_batched_guest_checks_keep_missing_target_tokenizer_and_shard_errors(local):
    model, runtime, *_ = local
    root = Path(model.path)
    index = root / 'model.safetensors.index.json'
    index.write_text('{"weight_map":{"a":"model.safetensors"}}')
    info = native.local_resource_info(model.path)
    native.check_spark_access(model.path, str(root / 'tokenizer.json'), info, runtime)
    with pytest.raises(ValueError, match='WSL 模型路径不可访问'):
        native.check_spark_access(str(root / 'unmounted'), str(root / 'tokenizer.json'), info, runtime)
    with pytest.raises(ValueError, match='WSL tokenizer.json 不可访问'):
        native.check_spark_access(model.path, str(root / 'missing-tokenizer.json'), info, runtime)
    (root / 'model.safetensors').unlink()
    (root / 'other.safetensors').write_bytes(b'header-only fixture')
    with pytest.raises(ValueError, match='缺少分片'):
        native.check_spark_access(model.path, str(root / 'tokenizer.json'), info, runtime)


def test_ninfer_native_argv_uses_artifact_and_explicit_mtp_window(local):
    model, runtime, profile, context, _ = artifact(local)
    profile.parameters = {'context_length': 2048, 'acceleration_mode': 'mtp', 'draft_tokens': 1, 'kv_capacity': 4096}
    spec = get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    assert spec.platform == 'wsl2' and spec.distribution == 'ExternalUbuntu'
    assert spec.argv[:2] == [runtime.path, model.path]
    assert spec.argv[spec.argv.index('--spec') + 1] == 'mtp'
    assert spec.argv[spec.argv.index('--draft-tokens') + 1] == '1'
    assert spec.argv[spec.argv.index('--max-context') + 1] == '2048'
    assert spec.argv[spec.argv.index('--kv-capacity') + 1] == '4096'
    assert spec.argv[spec.argv.index('--host') + 1] == '127.0.0.1'
    assert spec.health_path == '/health' and spec.capabilities['stream_usage']
    assert 'repeat_penalty' not in spec.capabilities['generation_parameters']
    profile.parameters['draft_tokens'] = 6
    with pytest.raises(ValueError, match='5'):
        validate_profile(model, profile, runtime, get_adapters()['ninfer'])
    profile.parameters['acceleration_mode'] = 'none'
    profile.parameters['draft_tokens'] = 'dormant-invalid-value'
    profile.parameters['conversion_device'] = 'old-unused-device'
    profile.parameters['conversion_recipe'] = 'old-unused-recipe'
    validate_profile(model, profile, runtime, get_adapters()['ninfer'])
    assert '--draft-tokens' not in get_adapters()['ninfer'].build_launch(model, profile, runtime, context).argv


def test_ninfer_new_runtime_uses_shared_byte_capacity_without_removed_token_flag(local):
    model, runtime, profile, context, _ = artifact(local)
    runtime.metadata.update(source_commit='a754fcf3bf524494aa927c2f54e0cbfb58f569ab',
                            version_basis='verified_source_commit')
    profile.parameters = {'context_length': 2048, 'context_capacity': '1.5 GiB'}
    spec = get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    assert '--kv-capacity' not in spec.argv
    assert spec.argv[spec.argv.index('--context-capacity') + 1] == '1.5 GiB'
    assert spec.argv[spec.argv.index('--max-context') + 1] == '2048'


def test_ninfer_new_runtime_rejects_old_saved_token_capacity_without_rewriting_profile(local):
    model, runtime, profile, context, _ = artifact(local)
    runtime.metadata['supported_flags'] = ['--context-capacity']
    profile.parameters = {'context_length': 2048, 'kv_capacity': 4096}
    original = deepcopy(profile.parameters)
    with pytest.raises(ValueError, match='Token.*context_capacity'):
        get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    assert profile.parameters == original


def test_ninfer_new_runtime_default_and_blank_capacity_follow_auto(local):
    model, runtime, profile, context, _ = artifact(local)
    runtime.metadata['supported_flags'] = ['--context-capacity']
    adapter = get_adapters()['ninfer']
    fields = {field.key: field for field in adapter.parameters_for_runtime(runtime)}
    assert 'kv_capacity' not in fields and fields['context_capacity'].default == 'auto'
    assert fields['context_length'].maximum == 2**32 - 1
    spec = adapter.build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--context-capacity') + 1] == 'auto'
    profile.parameters = {'kv_capacity': 'auto', 'context_capacity': ''}
    spec = adapter.build_launch(model, profile, runtime, context)
    assert '--context-capacity' not in spec.argv and '--kv-capacity' not in spec.argv


@pytest.mark.parametrize('capacity', ['1', '1B', '1 KiB', '1.5MiB', '18446744073709551615B'])
def test_ninfer_new_runtime_preserves_valid_upstream_byte_quantities(local, capacity):
    model, runtime, profile, context, _ = artifact(local)
    runtime.metadata['supported_flags'] = ['--context-capacity']
    profile.parameters = {'context_capacity': capacity}
    spec = get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--context-capacity') + 1] == capacity


@pytest.mark.parametrize('capacity', ['0', '-1B', '1GB', '1.1B', '18446744073709551616B', True])
def test_ninfer_new_runtime_rejects_invalid_or_fractional_byte_quantities(local, capacity):
    model, runtime, profile, context, _ = artifact(local)
    runtime.metadata['supported_flags'] = ['--context-capacity']
    profile.parameters = {'context_capacity': capacity}
    with pytest.raises(ValueError, match='上下文.*容量'):
        get_adapters()['ninfer'].build_launch(model, profile, runtime, context)


def test_ninfer_legacy_runtime_preserves_tokens_and_does_not_accept_new_byte_field(local):
    model, runtime, profile, context, _ = artifact(local)
    profile.parameters = {'context_length': 2048, 'kv_capacity': 4096}
    spec = get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--kv-capacity') + 1] == '4096'
    profile.parameters['context_capacity'] = '1GiB'
    with pytest.raises(ValueError, match='context.capacity'):
        get_adapters()['ninfer'].build_launch(model, profile, runtime, context)


def test_ninfer_new_qwen4_conditional_generation_alias_is_version_scoped(local):
    _, runtime, _, _, _ = local
    config = {'architectures': ['Qwen4ExpForConditionalGeneration']}
    runtime.metadata.update(source_commit='a754fcf3bf524494aa927c2f54e0cbfb58f569ab',
                            version_basis='verified_source_commit')
    native.validate_conversion_source(config, runtime=runtime)
    native.validate_conversion_source(config, 'qwen3_8_flash_next', runtime=runtime)
    with pytest.raises(ValueError, match='不匹配'):
        native.validate_conversion_source(config, 'qwen3_8_27b', runtime=runtime)
    runtime.metadata = {}
    with pytest.raises(ValueError, match='不支持该源架构'):
        native.validate_conversion_source(config, runtime=runtime)


@pytest.mark.parametrize('metadata', [{}, {'source_commit': 'a754fcf3bf524494aa927c2f54e0cbfb58f569ab'}])
def test_ninfer_display_version_or_unverified_commit_does_not_change_legacy_contract(local, metadata):
    _, runtime, _, _, _ = local
    runtime.version = 'a754fcf3bf52'
    runtime.metadata = metadata
    fields = {field.key for field in native.NinferAdapter().parameters_for_runtime(runtime)}
    assert 'kv_capacity' in fields and 'context_capacity' not in fields


@pytest.mark.parametrize('has_vision,selected,additional,enabled', [
    (True, None, [], True),
    (True, True, [], True),
    (True, False, [], False),
    (True, None, ['--vision'], True),
    (False, None, [], False),
    (False, False, [], False),
])
def test_ninfer_vision_startup_preflight_and_capabilities_agree(local, has_vision, selected, additional, enabled):
    components = {'text': {}, 'mtp': {}, **({'vision': {}} if has_vision else {})}
    model, runtime, profile, context, _ = artifact(local, components)
    # A source's image metadata cannot enable a missing artifact component.
    model.modalities = ['text', 'image']
    runtime.metadata['supported_flags'] = ['--vision']
    if selected is not None:
        profile.parameters['vision_enabled'] = selected
    profile.extra_args = additional
    before = profile.model_dump()
    spec = get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'ready' and state['vision_enabled'] is enabled
    assert spec.argv.count('--vision') == int(enabled)
    assert ('image' in spec.capabilities['input']) is enabled
    assert state['preparation_fields'] == []
    assert profile.model_dump() == before


@pytest.mark.parametrize('has_vision,parameters,additional,flags,message', [
    (False, {'vision_enabled': True}, [], ['--vision'], '未包含视觉组件'),
    (False, {}, ['--vision'], ['--vision'], '未包含视觉组件'),
    (True, {'vision_enabled': False}, ['--vision'], ['--vision'], '移除额外启动参数'),
    (True, {}, [], ['--host'], '运行环境未提供 --vision'),
    (True, {'vision_enabled': 'false'}, [], ['--vision'], '必须为布尔值'),
])
def test_ninfer_vision_rejects_missing_components_flags_or_conflicting_choices(local, has_vision, parameters, additional, flags, message):
    components = {'text': {}, **({'vision': {}} if has_vision else {})}
    model, runtime, profile, context, _ = artifact(local, components)
    profile.parameters, profile.extra_args = parameters, additional
    runtime.metadata['supported_flags'] = flags
    with pytest.raises(ValueError, match=message):
        get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    state = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore()).preflight(model, profile, runtime)
    assert state['status'] == 'error' and message in state['message']


@pytest.mark.parametrize('has_vision', [False, True])
async def test_ninfer_management_preflight_and_preview_keep_effective_vision(local, has_vision):
    model, runtime, profile, context, _ = artifact(local, {'text': {}, **({'vision': {}} if has_vision else {})})
    controller = Controller(Path(context.data_dir) / 'vision-api', 'fixture-secret')
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    controller.store.put('profiles', profile.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        try:
            for selected in (None, False, True):
                profile.parameters = {} if selected is None else {'vision_enabled': selected}
                controller.store.put('profiles', profile.model_dump())
                checked = await manager.post('/api/profiles/preflight', json=profile.model_dump())
                assert checked.status_code == 200, checked.text
                preview = await manager.post('/api/profiles/' + profile.id + '/preview')
                if selected is True and not has_vision:
                    assert checked.json()['status'] == 'error'
                    assert preview.status_code == 200 and preview.json()['argv'] == []
                    assert preview.json()['preparation']['status'] == 'error' and '未包含视觉组件' in preview.text
                else:
                    enabled = has_vision and selected is not False
                    assert checked.json()['status'] == 'ready' and checked.json()['vision_enabled'] is enabled
                    assert preview.status_code == 200, preview.text
                    assert ('--vision' in preview.json()['argv']) is enabled
        finally:
            await controller.close()


@pytest.mark.parametrize('backend,values', [
    ('ninfer', {'temperature': 2, 'top_p': 0, 'top_k': 20, 'max_tokens': 0, 'seed': -1}),
    ('sparkinfer', {'temperature': 2, 'top_p': 0, 'top_k': 1000, 'max_tokens': 1, 'seed': 0}),
])
def test_native_sampling_preserves_real_backend_boundaries(backend, values):
    validate_generation(values, backend)
    payload = {**values, 'stop': ['END'], 'response_format': {'type': 'json_object'}}
    assert native.map_generation(payload, backend) == payload
    for bad in ({'temperature': 2.1}, {'top_k': -1}, {'top_k': 1.5}, {'repeat_penalty': 1.1}):
        with pytest.raises(ValueError):
            validate_generation(bad, backend)
    with pytest.raises(ValueError):
        validate_generation({'repetition_penalty': 1.1}, backend)
    if backend == 'ninfer':
        with pytest.raises(ValueError):
            validate_generation({'top_k': 21}, backend)
        validate_generation({'repetition_penalty': 1}, backend)
    else:
        with pytest.raises(ValueError):
            validate_generation({'seed': -1}, backend)


def test_spark_requires_matching_local_draft_type_not_mtp_gguf(local):
    model, runtime, profile, context, _ = local
    runtime.backend, profile.backend = 'sparkinfer', 'sparkinfer'
    runtime.path = '/opt/fixture/source/build/server/sparkinfer_server'
    profile.parameters = {'context_length': 2048, 'acceleration_mode': 'dflash2'}
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    draft = Path(model.path).parent / 'draft'
    draft.mkdir()
    (draft / 'config.json').write_text('{"architectures":["DFlash2DraftModel"],"block_size":4}')
    (draft / 'model.safetensors').write_bytes(b'tiny fixture')
    profile.parameters['draft_model'] = draft.as_posix()
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'ready' and state['acceleration']['status'] == 'available'
    assert next(field for field in state['acceleration']['fields'] if field['key'] == 'draft_proposals')['maximum'] == 2**31 - 1
    profile.parameters['draft_proposals'] = 1
    spec = get_adapters()['sparkinfer'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--draft-model') + 1] == draft.as_posix()
    assert spec.argv[spec.argv.index('--ctx') + 1] == '2048'
    assert '--spec' not in spec.argv and '--draft-tokens' not in spec.argv
    assert spec.env['SPARKINFER_DFLASH_PROPOSALS'] == '1'
    profile.parameters['draft_proposals'] = 4
    validate_profile(model, profile, runtime, get_adapters()['sparkinfer'])
    # The engine caps the effective depth at block_size - 1, not the app input.
    assert get_adapters()['sparkinfer'].build_launch(model, profile, runtime, context).env['SPARKINFER_DFLASH_PROPOSALS'] == '4'
    profile.parameters['draft_proposals'] = 0
    profile.parameters['acceleration_mode'] = 'dspark'
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    (draft / 'config.json').write_text('{"dflash_config":{"block_size":16}}')
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    profile.parameters['acceleration_mode'] = 'mtp'
    with pytest.raises(ValueError):
        validate_profile(model, profile, runtime, get_adapters()['sparkinfer'])


def _spark_output_fixture(local):
    model, runtime, profile, context, _ = local
    runtime.backend = profile.backend = 'sparkinfer'
    runtime.path = '/opt/fixture/source/build/server/sparkinfer_server'
    profile.parameters = {'context_length': 4096, 'acceleration_mode': 'none'}
    profile.generation = {'max_tokens': 13}
    return model, runtime, profile, context


@pytest.mark.parametrize('explicit_tokenizer', [False, True])
async def test_spark_launch_editor_api_preview_roundtrip(local, explicit_tokenizer):
    model, runtime, profile, context = _spark_output_fixture(local)
    if explicit_tokenizer:
        selected = Path(model.path) / 'selected-tokenizer.json'
        selected.write_text('{}')
        profile.parameters['tokenizer'] = selected.as_posix()
    original_parameters = dict(profile.parameters)
    controller = Controller(Path(context.data_dir) / 'spark-launch-editor-api', 'fixture-secret')
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        try:
            preview = await manager.post('/api/profiles/launch-parameters', json={**profile.model_dump(), 'preview': True})
            assert preview.status_code == 200, preview.text
            result = preview.json()
            assert not result['errors'] and result['argv'], result
            expected = list(result['argv'])
            port_index = expected.index('--port') + 1
            for _ in range(2):
                parsed = await manager.post('/api/profiles/launch-parameters', json={
                    **profile.model_dump(),
                    **{key: result[key] for key in ('parameters', 'generation', 'extra_args', 'parameter_editor_version')},
                    'text': result['text'], 'preview': True,
                })
                assert parsed.status_code == 200, parsed.text
                result = parsed.json()
                assert not result['errors'] and not result.get('preview_error'), result
                assert result['parameters'] == original_parameters and result['extra_args'] == []
                # A fresh preview reserves a new port; all model and argv bindings stay fixed.
                actual = list(result['argv'])
                actual[port_index] = expected[port_index]
                assert actual == expected
            saved = await manager.post('/api/profiles', json={
                **profile.model_dump(),
                **{key: result[key] for key in ('parameters', 'generation', 'extra_args', 'parameter_editor_version')},
            })
            assert saved.status_code == 200, saved.text
            assert saved.json()['parameters'] == original_parameters
            assert saved.json()['extra_args'] == []
        finally:
            await controller.close()


@pytest.mark.parametrize('backend', ['ninfer', 'sparkinfer'])
def test_native_cuda_selection_overrides_only_the_profile_environment(local, backend):
    if backend == 'ninfer':
        model, runtime, profile, context, _ = artifact(local)
    else:
        model, runtime, profile, context = _spark_output_fixture(local)
    runtime.metadata['build_environment'] = {'CUDA_VISIBLE_DEVICES': '3', 'BUILD_VALUE': 'keep'}
    adapter = get_adapters()[backend]
    baseline = adapter.build_launch(model, profile, runtime, context)
    for value in ['GPU-22222222-3333-4444-5555-666666666666', 'MIG-GPU-22222222/1/2', '-1', '0,9']:
        profile.parameters['cuda_visible_devices'] = value
        spec = adapter.build_launch(model, profile, runtime, context)
        assert spec.env['CUDA_VISIBLE_DEVICES'] == value
        assert spec.env['BUILD_VALUE'] == 'keep' and spec.argv == baseline.argv
    for blank in [None, '', ' \t']:
        profile.parameters['cuda_visible_devices'] = blank
        assert adapter.build_launch(model, profile, runtime, context) == baseline
    runtime.metadata.pop('build_environment')
    assert 'CUDA_VISIBLE_DEVICES' not in adapter.build_launch(model, profile, runtime, context).env


async def test_spark_server_output_tokens_shared_schema_and_api_preview(local, monkeypatch):
    model, runtime, profile, context = _spark_output_fixture(local)
    adapter = get_adapters()['sparkinfer']
    field = next(item for item in profile_fields(adapter, model, runtime)
                 if item.key == 'server_max_output_tokens')
    assert (field.label, field.type, field.minimum, field.maximum, field.integer) == (
        '服务最大输出 Token', 'number', -(2**31), 2**31 - 1, True)
    assert 'server_max_output_tokens' not in {item.key for item in profile_fields(get_adapters()['ninfer'], model, runtime)}

    # Inspect the real LaunchSpec built by the management API's preview path;
    # the API's existing secret filter intentionally omits token-named env keys.
    launch_specs = []
    controller = Controller(Path(context.data_dir) / 'spark-output-token-api', 'fixture-secret')
    preview_adapter = controller.adapters['sparkinfer']
    original = preview_adapter.build_launch
    def capture(*args):
        spec = original(*args)
        launch_specs.append(spec)
        return spec
    monkeypatch.setattr(preview_adapter, 'build_launch', capture)
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        try:
            saved = await manager.post('/api/profiles', json=profile.model_dump())
            assert saved.status_code == 200, saved.text
            profile_data = saved.json()
            for value in (32768, 8192, 2**31 - 1, 0, -1, -(2**31)):
                profile_data['parameters']['server_max_output_tokens'] = value
                updated = await manager.patch('/api/profiles/' + profile_data['id'], json=profile_data)
                assert updated.status_code == 200, updated.text
                profile_data = updated.json()
                preview = await manager.post('/api/profiles/' + profile_data['id'] + '/preview')
                assert preview.status_code == 200, preview.text
                assert launch_specs, preview.json()
                spec = launch_specs[-1]
                assert spec.env['SPARKINFER_MAX_OUTPUT_TOKENS'] == str(value)
                assert 'SPARKINFER_MAX_OUTPUT_TOKENS' not in spec.argv
                assert str(value) not in spec.argv
                assert profile_data['generation'] == {'max_tokens': 13}

            # Clearing the saved field removes the explicit override on the next launch.
            profile_data['parameters']['server_max_output_tokens'] = ''
            cleared = await manager.patch('/api/profiles/' + profile_data['id'], json=profile_data)
            assert cleared.status_code == 200, cleared.text
            profile_data = cleared.json()
            preview = await manager.post('/api/profiles/' + profile_data['id'] + '/preview')
            assert preview.status_code == 200, preview.text
            assert 'SPARKINFER_MAX_OUTPUT_TOKENS' not in launch_specs[-1].env
            assert profile_data['generation'] == {'max_tokens': 13}
        finally:
            await controller.close()


async def test_spark_server_output_tokens_reject_invalid_save_and_launch(local):
    model, runtime, profile, context = _spark_output_fixture(local)
    adapter = get_adapters()['sparkinfer']
    controller = Controller(Path(context.data_dir) / 'spark-output-token-invalid-api', 'fixture-secret')
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        try:
            for value in (-(2**31) - 1, 1.5, True, 2**31):
                invalid = profile.model_copy(deep=True)
                invalid.parameters['server_max_output_tokens'] = value
                saved = await manager.post('/api/profiles', json=invalid.model_dump())
                assert saved.status_code == 400, f'{value!r}: {saved.text}'
                with pytest.raises(ValueError):
                    adapter.build_launch(model, invalid, runtime, context)
        finally:
            await controller.close()


def test_spark_server_output_tokens_leave_existing_or_inherited_environment(local):
    model, runtime, profile, context = _spark_output_fixture(local)
    adapter = get_adapters()['sparkinfer']
    for value in (None, '   ', 'missing'):
        profile.parameters.pop('server_max_output_tokens', None)
        if value != 'missing':
            profile.parameters['server_max_output_tokens'] = value
        runtime.metadata['build_environment'] = {'SPARKINFER_MAX_OUTPUT_TOKENS': '70000', 'SPARK_CUSTOM': 'keep'}
        spec = adapter.build_launch(model, profile, runtime, context)
        assert spec.env['SPARKINFER_MAX_OUTPUT_TOKENS'] == '70000'
        assert spec.env['SPARK_CUSTOM'] == 'keep'

        # Without a build-time override, omission leaves the WSL process environment inherited.
        runtime.metadata.pop('build_environment')
        spec = adapter.build_launch(model, profile, runtime, context)
        assert 'SPARKINFER_MAX_OUTPUT_TOKENS' not in spec.env


def test_ninfer_preflight_never_guesses_recipe_or_starts_conversion(local):
    model, runtime, profile, context, source = local
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'choice_required' and len(state['recipes']) == 6
    assert next(field for field in state['preparation_fields'] if field['key'] == 'conversion_recipe')['default'] is None
    assert not prep.root.exists()
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b', 'acceleration_mode': 'mtp', 'draft_tokens': 1}
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'needs_preparation' and not prep.root.exists()
    assert 'conversion_source' not in [field['key'] for field in state['preparation_fields']]
    profile.parameters['conversion_recipe'] = 'qwen3_8_27b_nvfp4'
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'error' and '额外' in state['message']
    assert 'conversion_source' in [field['key'] for field in state['preparation_fields']]
    (Path(model.path) / 'config.json').write_text('{"architectures":["Qwen3ForCausalLM"]}')
    assert '不支持该源架构' in prep.preflight(model, profile, runtime)['message']
    (Path(model.path) / 'config.json').write_text('{"architectures":["Qwen3_5MoeForCausalLM"]}')
    profile.parameters['conversion_recipe'] = 'qwen3_6_27b'
    assert '不匹配' in prep.preflight(model, profile, runtime)['message']


def test_native_preflight_blocks_indexed_missing_shards_and_invalid_boolean(local):
    model, runtime, profile, context, source = local
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    index = Path(model.path) / 'model.safetensors.index.json'
    index.write_text('{"weight_map":{"a":"missing-00002.safetensors","b":"model.safetensors"}}')
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    assert 'missing-00002.safetensors' in prep.preflight(model, profile, runtime)['message']
    profile.parameters['prepare_mtp'] = 'false'
    assert '布尔值' in prep.preflight(model, profile, runtime)['message']
    profile.parameters.pop('prepare_mtp')
    index.write_text('{"weight_map":{"a":"../outside.safetensors"}}')
    assert 'escaped' in prep.preflight(model, profile, runtime)['message']


def test_preconverted_ninfer_checks_mtp_components_without_conversion_fields(local):
    model, runtime, profile, context, _ = artifact(local, {'text': {}})
    profile.parameters = {'acceleration_mode': 'mtp', 'draft_tokens': 1}
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'error' and 'MTP' in state['message']
    assert state['preparation_fields'] == [] and state['recipes'] == []
    profile.parameters['acceleration_mode'] = 'none'
    profile.parameters['draft_tokens'] = -99
    profile.parameters['generation_config'] = '/old/unused/missing-generation-config.json'
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert profile.parameters['generation_config'] == '/old/unused/missing-generation-config.json'


@pytest.mark.parametrize('role', native.NINFER_TEXT_RESOURCES)
def test_ninfer_preflight_blocks_missing_required_text_resources_before_conversion(local, role):
    model, runtime, profile, context, source = local
    (Path(model.path) / role).unlink()
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'error' and role in state['message']
    assert not prep.root.exists() and not (source / 'pid').exists()
    assert 'generation_config' in {field['key'] for field in state['preparation_fields']}


@pytest.mark.parametrize('role', native.NINFER_VISION_RESOURCES)
def test_ninfer_preflight_requires_vision_resources_only_when_selected(local, role):
    model, runtime, profile, context, source = local
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    for name in native.NINFER_VISION_RESOURCES:
        if name != role:
            (Path(model.path) / name).write_text('{"patch_size":16,"temporal_patch_size":2,"merge_size":2}')
    profile.parameters['prepare_vision'] = True
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'error' and role in state['message']
    assert not prep.root.exists()


@pytest.mark.parametrize('content', [b'', b'[]', b'not json', b'\xff'])
def test_ninfer_preflight_rejects_empty_nonobject_or_invalid_resource_bytes(local, content):
    model, runtime, profile, context, source = local
    (Path(model.path) / 'generation_config.json').write_bytes(content)
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    state = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore()).preflight(model, profile, runtime)
    assert state['status'] == 'error' and 'generation_config.json' in state['message']
    assert not (source / 'pid').exists()


def write_converter(source):
    script = source / 'convert-fixture.py'
    script.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
for flag in ['model','recipe','components','name','device','out','source']: p.add_argument('--'+flag)
p.add_argument('--resource',action='append')
p.add_argument('--proposal',action='store_true')
a=p.parse_args()
root=Path(__file__).parent
(root/'pid').write_text(str(os.getpid()))
(root/'argv.json').write_text(json.dumps(vars(a)))
print('fixture converter waiting',flush=True)
while not (root/'release').exists(): time.sleep(.02)
if (root/'fail').exists(): raise SystemExit(7)
Path(a.out).write_text(json.dumps({'components':{name:{} for name in a.components.split(',')}}))
print('fixture converted',flush=True)
''')
    return script


async def wait_file(path):
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(.02)


@pytest.mark.parametrize('vision', [False, True])
async def test_real_cpu_conversion_reuse_cleanup_and_source_identity(local, monkeypatch, vision):
    model, runtime, profile, context, source = local
    model.name = '本地 Qwen3.5 4B 中文模型'
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b', 'conversion_device': 'cpu', 'acceleration_mode': 'mtp', 'draft_tokens': 1}
    if vision:
        profile.parameters['prepare_vision'] = True
        for role in native.NINFER_VISION_RESOURCES:
            (Path(model.path) / role).write_text('{"patch_size":16,"temporal_patch_size":2,"merge_size":2}')
    store = MemoryStore()
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), store)
    script = write_converter(source)
    calls, events = [], []

    def launch(spec, context):
        calls.append(spec)
        return LaunchSpec(argv=[sys.executable, '-u', str(script), *spec.argv[4:]], cwd=str(source), model_name='fixture')

    monkeypatch.setattr(preparation, 'prepare_launch', launch)
    async def progress(event): events.append(event)
    before = (Path(model.path) / 'model.safetensors').read_bytes()
    task = asyncio.create_task(prep.ensure(model, profile, runtime, progress))
    await wait_file(source / 'pid')
    assert psutil.pid_exists(int((source / 'pid').read_text()))
    assert store.all(prep.KIND) == []
    (source / 'release').touch()
    resolved = await task
    target = resolved.parameters['_prepared_artifact']
    assert Path(target).is_file() and target != model.path and profile.parameters.get('_prepared_artifact') is None
    assert (Path(model.path) / 'model.safetensors').read_bytes() == before
    command = json.loads((source / 'argv.json').read_text())
    assert command['recipe'] == 'qwen3_6_27b' and command['components'] == ('text,vision,mtp' if vision else 'text,mtp') and command['device'] == 'cpu'
    assert command['name'] == model.name and command['name'] != command['recipe']
    assert command['resource'] == ['chat_template.jinja=' + str(Path(model.path) / 'chat_template.jinja').replace('\\', '/')]
    assert events and any('fixture converted' in event.get('message', '') for event in events)
    assert [event['phase'] for event in events[-2:]] == ['validating', 'prepared']
    assert len(store.all(prep.KIND)) == 1 and prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert 'tokenizer_directory' not in store.all(prep.KIND)[0]['identity']['parameters']
    assert (await prep.ensure(model, profile, runtime, progress)).parameters['_prepared_artifact'] == target
    assert len(calls) == 1
    launch = get_adapters()['ninfer'].build_launch(model, resolved, runtime, context)
    assert ('--vision' in launch.argv) is vision
    assert ('image' in launch.capabilities['input']) is vision
    assert prep.preflight(model, profile, runtime)['vision_enabled'] is vision
    if vision:
        profile.parameters['vision_enabled'] = False
        state = prep.preflight(model, profile, runtime)
        assert state['status'] == 'ready' and state['artifact'] == target and state['vision_enabled'] is False
        assert (await prep.ensure(model, profile, runtime, progress)).parameters['_prepared_artifact'] == target
        assert len(calls) == 1  # A launch toggle must not rebuild immutable components.
        profile.parameters.pop('vision_enabled')
        Path(target).write_text('{"components":{"text":{},"mtp":{}}}')
        assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    # Same directory but changed source shard must not reuse a stale artifact.
    (Path(model.path) / 'model.safetensors').write_bytes(before + b'changed')
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    prep.remove_owned(model)
    assert not Path(target).exists() and Path(model.path).is_dir() and store.all(prep.KIND) == []


@pytest.mark.parametrize('selected', ['D:\\official resources\\generation_config.json', '/official resources/generation_config.json'])
async def test_real_cpu_conversion_uses_explicit_generation_sidecar_and_invalidates_reuse(local, monkeypatch, selected):
    model, runtime, profile, context, source = local
    original_generation = Path(model.path) / 'generation_config.json'
    original_generation.unlink()  # Reproduce the frozen 4B snapshot's absent file.
    sidecar = source.parent / 'official generation config.json'
    sidecar.write_text('{"eos_token_id":[2,3]}')
    mapped = []
    def resource_target(value, owner):
        mapped.append((value, owner.metadata['distribution']))
        return sidecar.as_posix() if value == selected else str(value)
    monkeypatch.setattr(native, 'resource_target', resource_target)
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b', 'generation_config': selected}
    store = MemoryStore()
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), store)
    script = write_converter(source)
    (source / 'release').touch()
    monkeypatch.setattr(preparation, 'prepare_launch', lambda spec, context: spec.model_copy(update={
        'platform': 'windows', 'argv': [sys.executable, '-u', str(script), *spec.argv[4:]], 'cwd': str(source)}))
    async def progress(event): pass
    result = await prep.ensure(model, profile, runtime, progress)
    command = json.loads((source / 'argv.json').read_text())
    assert command['resource'] == ['chat_template.jinja=' + str(Path(model.path) / 'chat_template.jinja').replace('\\', '/'),
                                   'generation_config.json=' + sidecar.as_posix()]
    assert (selected, 'ExternalUbuntu') in mapped and profile.parameters['generation_config'] == selected
    assert not original_generation.exists() and Path(result.parameters['_prepared_artifact']).is_file()
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    stamp = store.all(prep.KIND)[0]['identity']['resources']['generation_config.json']
    assert stamp['size'] == sidecar.stat().st_size and Path(stamp['path']) == sidecar
    sidecar.write_text('{"eos_token_id":[2,3],"updated":true}')
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'


async def test_real_cpu_conversion_cancel_stops_only_fixture_and_removes_partial(local, monkeypatch):
    model, runtime, profile, context, source = local
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    store = MemoryStore()
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), store)
    script = write_converter(source)
    monkeypatch.setattr(preparation, 'prepare_launch', lambda spec, context: LaunchSpec(
        argv=[sys.executable, '-u', str(script), *spec.argv[4:]], cwd=str(source), model_name='fixture'))
    async def progress(event): pass
    task = asyncio.create_task(prep.ensure(model, profile, runtime, progress))
    await wait_file(source / 'pid')
    pid = int((source / 'pid').read_text())
    assert psutil.pid_exists(pid)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not psutil.pid_exists(pid)
    assert store.all(prep.KIND) == [] and list(prep.root.iterdir()) == []
    assert (Path(model.path) / 'model.safetensors').is_file()


@pytest.mark.parametrize('failure', ['nonzero', 'source_changed', 'resource_changed'])
async def test_failed_real_conversion_never_registers_partial_or_stale_artifact(local, monkeypatch, failure):
    model, runtime, profile, context, source = local
    profile.parameters = {'conversion_recipe': 'qwen3_6_27b'}
    if failure == 'resource_changed':
        sidecar = source.parent / 'generation-sidecar.json'
        sidecar.write_text('{"eos_token_id":[2,3]}')
        profile.parameters['generation_config'] = sidecar.as_posix()
    store = MemoryStore()
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), store)
    script = write_converter(source)
    monkeypatch.setattr(preparation, 'prepare_launch', lambda spec, context: LaunchSpec(
        argv=[sys.executable, '-u', str(script), *spec.argv[4:]], cwd=str(source), model_name='fixture'))
    async def progress(event): pass
    task = asyncio.create_task(prep.ensure(model, profile, runtime, progress))
    await wait_file(source / 'pid')
    if failure == 'nonzero':
        (source / 'fail').touch()
    elif failure == 'source_changed':
        (Path(model.path) / 'model.safetensors').write_bytes(b'changed while converting')
    else:
        sidecar.write_text('{"eos_token_id":[2,3],"updated":true}')
    (source / 'release').touch()
    with pytest.raises((RuntimeError, ValueError), match='7|发生变化'):
        await task
    assert store.all(prep.KIND) == [] and list(prep.root.iterdir()) == []
    assert (Path(model.path) / 'model.safetensors').is_file()


async def test_spark_save_metadata_probe_does_not_block_other_api_requests(local, monkeypatch):
    model, runtime, profile, context, source = local
    profile.backend = runtime.backend = 'sparkinfer'
    draft = Path(model.path).parent / 'draft'
    draft.mkdir()
    (draft / 'config.json').write_text('{"architectures":["DFlash2DraftModel"],"block_size":4}')
    (draft / 'model.safetensors').write_bytes(b'fixture')
    profile.parameters = {'acceleration_mode': 'dflash2', 'draft_model': draft.as_posix(), 'draft_proposals': 1}
    entered, release = threading.Event(), threading.Event()
    original = native.resource_info
    def probe(path, runtime):
        if path == draft.as_posix():
            entered.set()
            if not release.wait(3):
                raise ValueError('Fixture metadata probe was not released')
        return original(path, runtime)
    monkeypatch.setattr(native, 'resource_info', probe)
    controller = Controller(Path(context.data_dir) / 'concurrent-save', 'fixture-secret')
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        pending = asyncio.create_task(manager.post('/api/profiles', json=profile.model_dump()))
        try:
            async with asyncio.timeout(5):
                while not entered.is_set(): await asyncio.sleep(.01)
            assert not pending.done(), 'A synchronous metadata probe blocked the event loop until its timeout'
            assert (await asyncio.wait_for(manager.get('/api/health'), 1)).status_code == 200
            release.set()
            response = await pending
            assert response.status_code == 200, response.text
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await controller.close()


@pytest.mark.parametrize('backend,text,key,unit', [
    ('ninfer', 'ninfer_spec_decode_rounds_total 8\nninfer_spec_decode_draft_tokens_total 20\nninfer_spec_decode_accepted_tokens_total 12\nninfer_spec_decode_fallback_steps_total 2\n', 'spec_decode_rounds_total', '轮'),
    ('sparkinfer', 'sparkinfer_tokens_total{kind="completion"} 33\nsparkinfer_speculative_runs_total 2\nsparkinfer_active_requests 1\n', 'speculative_runs_total', '个'),
])
def test_native_metrics_counters_never_become_speed_or_accept_rate(backend, text, key, unit):
    result = Metrics()
    _prometheus(text, backend, result)
    values = [metric for group in result.result()['groups'] for metric in group['metrics']]
    found = next(metric for metric in values if metric['id'] == key)
    assert found['kind'] == 'counter' and found['unit'] == unit
    assert '重启归零' in found['description']
    assert not any(metric.get('unit') == 'token/s' or 'accept_rate' in metric['id'] for metric in values)
    if backend == 'sparkinfer':
        assert any('kind=completion' in metric['label'] for metric in values)
    else:
        assert {metric['id']: metric['value'] for metric in values} == {
            'spec_decode_rounds_total': 8, 'spec_decode_draft_tokens_total': 20,
            'spec_decode_accepted_tokens_total': 12, 'spec_decode_fallback_steps_total': 2}
    assert result.unavailable


def test_ninfer_native_metrics_sample_distinguishes_reported_zero_from_missing():
    text = (Path(__file__).parent / 'fixtures/ninfer-basic-metrics.prom').read_text()
    result = Metrics()
    _prometheus(text, 'ninfer', result)
    spec = {metric['id']: metric for metric in result.groups['ninfer_spec']['metrics']}
    assert set(spec) == {'spec_decode_rounds_total', 'spec_decode_draft_tokens_total',
                         'spec_decode_accepted_tokens_total', 'spec_decode_fallback_steps_total'}
    assert all(metric['value'] == 0 and metric['kind'] == 'counter' for metric in spec.values())
    assert spec['spec_decode_draft_tokens_total']['unit'] == spec['spec_decode_accepted_tokens_total']['unit'] == 'token'
    assert spec['spec_decode_rounds_total']['unit'] == '轮' and spec['spec_decode_fallback_steps_total']['unit'] == '步'
    assert {metric['id']: metric['value'] for metric in result.groups['ninfer_tokens']['metrics']} == {
        'prompt_tokens_total': 34, 'generation_tokens_total': 4}

    missing = Metrics()
    _prometheus('\n'.join(line for line in text.splitlines() if 'ninfer_spec_decode_accepted_tokens_total' not in line),
                'ninfer', missing)
    assert 'spec_decode_accepted_tokens_total' not in {metric['id'] for metric in missing.groups['ninfer_spec']['metrics']}


@pytest.mark.parametrize('backend', ['sglang', 'ninfer', 'sparkinfer'])
async def test_formal_backend_gateway_real_process_nonstream_sse_and_stop(local, monkeypatch, backend):
    """Formal launch contracts, then a CPU service substituting only the native executable."""
    from modloader.backends import sglang
    model, runtime, profile, context, source = artifact(local) if backend == 'ninfer' else local
    profile.backend = runtime.backend = backend
    profile.generation = {'temperature': .3, 'max_tokens': 4, 'top_k': 1}
    if backend == 'sglang':
        runtime.path = '/opt/fixture/bin/python'
        profile.generation['repeat_penalty'] = 1.1
        monkeypatch.setattr(sglang, 'model_target_path', lambda model, runtime: model.path)
    original = get_adapters()[backend]
    server = Path(__file__).parent / 'fixtures/custom_engine_server.py'
    class CpuExecutableAdapter:
        id, label = backend, 'CPU executable substitute'
        def parameters(self): return original.parameters()
        def inspect(self, *args): return original.inspect(*args)
        def build_launch(self, model, profile, runtime, context):
            spec = original.build_launch(model, profile, runtime, context)
            assert spec.platform == 'wsl2' and spec.distribution == 'ExternalUbuntu'
            return spec.model_copy(update=dict(platform='windows', distribution=None, argv=[
                sys.executable, str(server), '--model', model.path, '--port', str(context.port)],
                cwd=str(server.parent), health_path='/v1/models', health_expected={}, startup_timeout=10))
    controller = Controller(Path(context.data_dir) / ('gateway-' + backend), 'fixture-secret')
    controller.adapters[backend] = CpuExecutableAdapter()
    controller.store.put('models', model.model_dump())
    controller.store.put('runtimes', runtime.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        try:
            saved = await manager.post('/api/profiles', json=profile.model_dump())
            assert saved.status_code == 200, saved.text
            profile = LoadProfile(**saved.json())
            preflight = await manager.post('/api/profiles/preflight', json=profile.model_dump())
            assert preflight.status_code == 200 and preflight.json()['status'] == 'ready', preflight.text
            assert [field['key'] for field in preflight.json()['generation_parameters']] == (
                ['temperature', 'top_p', 'max_tokens', 'top_k', 'repeat_penalty', 'seed'] if backend == 'sglang'
                else ['temperature', 'top_p', 'max_tokens', 'top_k', 'seed'])
            started = await manager.post('/api/profiles/' + profile.id + '/start')
            assert started.status_code == 200, started.text
            instance = controller.runtime.instances[started.json()['id']]
            async with asyncio.timeout(15):
                while instance.state in {'preparing', 'starting'}:
                    await asyncio.sleep(.02)
            assert instance.state == 'ready', instance.error
            ordinary = {'model': profile.alias, 'messages': [{'role': 'user', 'content': 'fixture'}], 'stop': ['END']}
            nonstream = await manager.post('/api/chat', json=ordinary)
            assert nonstream.status_code == 200, nonstream.text
            received = nonstream.json()['received']
            assert received['stop'] == ['END'] and received['temperature'] == .3 and received['top_k'] == 1
            if backend == 'sglang':
                assert received['repetition_penalty'] == 1.1 and 'repeat_penalty' not in received
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)), base_url='http://testserver',
                                        headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
                if backend != 'sglang':
                    assert (await public.post('/v1/chat/completions', json={**ordinary, 'repeat_penalty': 1.2})).status_code == 400
                streamed = await public.post('/v1/chat/completions', json={**ordinary, 'stream': True})
                assert streamed.status_code == 200 and 'data: [DONE]' in streamed.text
                chunks = [json.loads(line[5:]) for line in streamed.text.splitlines() if line.startswith('data:') and '[DONE]' not in line]
                assert any(chunk.get('usage', {}).get('completion_tokens') == 3 for chunk in chunks)
            process = controller.runtime.processes[instance.id]
            assert process.returncode is None
            stopped = await manager.post('/api/instances/' + instance.id + '/stop', json={})
            assert stopped.status_code == 200 and process.returncode is not None
            assert not controller.runtime.requests[instance.id] and instance.active_requests == 0
        finally:
            await controller.close()
