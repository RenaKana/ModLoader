import asyncio
import json
from pathlib import Path
import struct
import sys

import pytest

from modloader.backends.adapters import get_adapters, import_configuration
from modloader.models import AppPaths, LaunchContext, LoadProfile, ModelAsset, RuntimeInstallation
from modloader.parameter_rules import profile_fields
from modloader.store import Store
from modloader.strata_preparation import StrataPreparation
from modloader import strata_deepseek as ds


def write_gguf(path, arch='deepseek41', *, hashes=False, embedded=False):
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    metadata = [('general.architecture', 8, string(arch))]
    if hashes:
        metadata.append(('deepseek41.engram.token_map', 9, struct.pack('<IQI', 4, 1, 0)))
    names = ['embed.weight'] + (['per_layer_token_embd.weight'] if embedded else [])
    header = b'GGUF' + struct.pack('<IQQ', 3, len(names), len(metadata))
    for name, kind, value in metadata:
        header += string(name) + struct.pack('<I', kind) + value
    for i, name in enumerate(names):
        header += string(name) + struct.pack('<IQIQ', 1, 1, 0, i * 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + bytes((-len(header)) % 32) + bytes(4 * len(names)))
    return path


def write_ds_pack(output, sources, engram=None, dspark=None):
    output.mkdir(parents=True, exist_ok=True)
    root = Path(sources[0]).parent
    (output / 'ds4.txt').write_text(f'format gguf\nmodel_dir {root}\nnum_hidden_layers 40\nn_routed_experts 384\n')
    files = [*map(Path, sources), *([Path(engram)] if engram else []), *([Path(dspark)] if dspark else [])]
    import os
    lines = [f'file {i} {os.path.relpath(p, root)}' for i, p in enumerate(files)]
    for i, p in enumerate(files):
        name = 'mtp.0.weight' if dspark and p == Path(dspark) else f'weight.{i}'
        lines.append(f't {name} F32 {i} 0 4 1')
    (output / 'tensors.txt').write_text('\n'.join(lines))
    (output / 'engram.bin').write_bytes(b'fixture hashes')
    tok = output / 'tokenizer'
    tok.mkdir(exist_ok=True)
    for name, value in [('vocab.json', {'test': 0}), ('token_type.json', [1]), ('tokenizer.json', {'pre': 'deepseek3'})]:
        (tok / name).write_text(json.dumps(value))
    (tok / 'merges.txt').write_text('')
    (tok / 'chat_template.jinja').write_text('{{ messages }}')


@pytest.fixture
def setup(tmp_path):
    paths = AppPaths(tmp_path / 'data')
    store = Store(paths.root / 'test.sqlite')
    model_dir = tmp_path / 'models'
    source = write_gguf(model_dir / 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00002.gguf')
    write_gguf(source.with_name(source.name.replace('00001-of', '00002-of')))
    write_gguf(model_dir / 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf', 'deepseek41-engram')
    write_gguf(model_dir / 'DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf', hashes=True)
    write_gguf(model_dir / 'dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf', 'dflash')
    root = tmp_path / 'runtime'
    (root / 'tools/dsv41').mkdir(parents=True)
    (root / 'engine-dsv41').mkdir()
    (root / 'engine-dsv41/strata-dsv41.exe').write_bytes(b'not executed')
    (root / 'engine-dsv41/BUILD.json').write_text('{"version":"0.1.49.2","backend":"cuda"}')
    (root / 'data/dsv41').mkdir(parents=True)
    (root / 'data/dsv41/profile-2.00.bin').write_bytes(b'hot expert fixture')
    import inspect
    script = 'import argparse, json, sys\nfrom pathlib import Path\n' + inspect.getsource(write_ds_pack) + '''
p = argparse.ArgumentParser()
for name in ('model', 'out', 'engram', 'engram-meta', 'dspark-gguf', 'dspark-noise-token', 'dspark-markov-rank'):
    p.add_argument('--' + name)
p.add_argument('--shards', nargs='+')
a = p.parse_args()
assert '--native' not in sys.argv and '--skip-tensors' not in sys.argv
out = Path(a.out)
write_ds_pack(out, [Path(a.model) / n for n in a.shards], a.engram, a.dspark_gguf)
(out / 'argv.json').write_text(json.dumps(sys.argv))
'''
    (root / 'tools/dsv41/pack_gguf.py').write_text(script)
    model = ModelAsset(id='ds', name='DeepSeek', path=str(source), format='gguf', architecture='deepseek41')
    runtime = RuntimeInstallation(id='r', name='DS', backend='strata', path=str(root), version='0.1.49.2', metadata={'python': sys.executable})
    profile = LoadProfile(id='p', name='DS', model_id=model.id, backend='strata', runtime_id=runtime.id, alias='ds')
    yield StrataPreparation(paths, store), model, profile, runtime
    store.close()


async def ignore(event):
    pass


@pytest.mark.parametrize('parameters,extra,expected', [
    ({'ssd_cache': 'auto'}, [], '-1'), ({}, ['--ssd-cache', 'auto'], '-1'),
    ({}, ['--ssd-cache=auto'], '-1'), ({'ssd_cache': '0'}, [], '0'),
])
async def test_ssd_cache_auto_matches_preview_and_native_semantics(setup, tmp_path, parameters, extra, expected):
    from modloader.mtp import argument
    from modloader.launch_parameters import edit_parameters, tokenize
    prep, model, profile, runtime = setup
    resolved = await prep.ensure(model, profile, runtime, ignore)
    resolved.parameters.update(parameters)
    resolved.extra_args = extra
    edited = edit_parameters(model, resolved, runtime, text=edit_parameters(model, resolved, runtime)['text'])
    assert not edited['errors'], edited
    assert argument([token.value for token in tokenize(edited['text'])], '--ssd-cache') == expected
    context = LaunchContext(port=18099, instance_id='cache', work_dir=str(tmp_path / 'cache-instance'), data_dir=str(prep.paths.root))
    get_adapters()['strata'].build_launch(model, resolved, runtime, context)
    config = json.loads((Path(context.work_dir) / 'strata.json').read_text())
    assert argument(config['args'], '--ssd-cache') == expected


async def test_complete_offline_prepare_reuse_and_deepseek_launch(setup, tmp_path):
    prep, model, profile, runtime = setup
    before = prep.preflight(model, profile, runtime)
    assert before['status'] == 'needs_preparation', before
    assert set(before['dependencies']) == {'engram', 'engram_meta', 'dspark'}
    assert not prep.root.exists()
    resolved = await prep.ensure(model, profile, runtime, ignore)
    pack = Path(resolved.parameters['pack'])
    argv = json.loads((pack / 'argv.json').read_text())
    assert len(argv[argv.index('--shards') + 1:argv.index('--out')]) == 2
    assert argv[argv.index('--dspark-noise-token') + 1] == '128799'
    assert argv[argv.index('--dspark-markov-rank') + 1] == '256'
    assert (pack / 'expert-profile.bin').read_bytes() == b'hot expert fixture'
    assert (await prep.ensure(model, profile, runtime, ignore)).parameters['pack'] == str(pack)
    resolved.parameters.update(context_length=262144, dspark_mode='on', vram_reserve_mib=4096, cpu_threads=16)
    context = LaunchContext(port=18099, instance_id='test', work_dir=str(tmp_path / 'instance'), data_dir=str(prep.paths.root))
    launch = get_adapters()['strata'].build_launch(model, resolved, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text())
    assert cfg['arch'] == 'deepseek41'
    assert cfg['exe'] == str(ds.engine_path(Path(runtime.path)))
    assert '--native' not in cfg['args'] and '--kv' not in cfg['args'] and '--spec' not in cfg['args']
    assert cfg['args'][cfg['args'].index('--reserve-mib') + 1] == '4096'
    assert cfg['args'][cfg['args'].index('--dspark') + 1] == 'on'
    assert launch.startup_timeout == 3600
    assert launch.capabilities['configured_context_tokens'] == 262144
    assert launch.env['HF_HUB_OFFLINE'] == '1'
    imported = import_configuration(str(Path(context.work_dir) / 'strata.json'))
    assert imported['model_path'] == model.path
    assert imported['parameters']['dspark_mode'] == 'on'


@pytest.mark.parametrize('key', ['engram', 'engram_meta', 'dspark'])
async def test_dependency_change_invalidates_preparation(setup, key):
    prep, model, profile, runtime = setup
    await prep.ensure(model, profile, runtime, ignore)
    path = Path(ds.dependencies(model, profile)[key])
    path.write_bytes(path.read_bytes() + b'changed')
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'


@pytest.mark.parametrize('key,match', [('engram', 'Engram'), ('engram_meta', '哈希元数据')])
async def test_missing_required_dependency_fails_before_creating_output(setup, key, match):
    prep, model, profile, runtime = setup
    Path(ds.dependencies(model, profile)[key]).unlink()
    result = prep.preflight(model, profile, runtime)
    assert result['status'] == 'error' and match in result['message']
    with pytest.raises(ValueError, match=match):
        await prep.ensure(model, profile, runtime, ignore)
    assert not prep.root.exists()


async def test_dspark_is_optional_but_required_when_forced_on(setup, tmp_path):
    prep, model, profile, runtime = setup
    Path(ds.dependencies(model, profile)['dspark']).unlink()
    profile.parameters['dspark_mode'] = 'none'
    resolved = await prep.ensure(model, profile, runtime, ignore)
    assert not ds.pack_description(resolved.parameters['pack'])['dspark']
    profile.extra_args = ['--dspark', 'on']
    context = LaunchContext(port=18099, instance_id='test', work_dir=str(tmp_path / 'instance'), data_dir=str(prep.paths.root))
    with pytest.raises(ValueError, match='DSpark'):
        get_adapters()['strata'].build_launch(model, resolved.model_copy(update={'extra_args': profile.extra_args}), runtime, context)
    profile.extra_args = []
    profile.parameters['dspark_mode'] = 'on'
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    resolved.parameters['dspark_mode'] = 'on'
    assert prep.preflight(model, resolved, runtime)['status'] == 'error'


def test_explicit_dependencies_are_not_replaced_and_multiple_choices_require_selection(setup):
    prep, model, profile, runtime = setup
    profile.parameters['engram'] = model.path
    assert '架构不匹配' in prep.preflight(model, profile, runtime)['message']
    profile.parameters.clear()
    write_gguf(Path(model.path).parent / 'other-engram.gguf', 'deepseek41-engram')
    assert '多个 engram' in prep.preflight(model, profile, runtime)['message']


async def test_pack_cannot_bind_another_model_or_missing_references(setup, tmp_path):
    prep, model, profile, runtime = setup
    resolved = await prep.ensure(model, profile, runtime, ignore)
    other = model.model_copy(update={'id': 'other', 'path': str(write_gguf(tmp_path / 'other.gguf'))})
    assert '另一模型' in prep.preflight(other, resolved, runtime)['message'] or '不匹配' in prep.preflight(other, resolved, runtime)['message']
    Path(ds.dependencies(model, profile)['engram']).unlink()
    assert prep.preflight(model, resolved, runtime)['status'] == 'error'


async def test_failure_cleans_owned_output_and_preserves_inputs(setup):
    prep, model, profile, runtime = setup
    (Path(runtime.path) / 'tools/dsv41/pack_gguf.py').write_text('import sys; sys.exit(7)')
    sources = ds.shards(model)
    with pytest.raises(ValueError, match='退出码 7'):
        await prep.ensure(model, profile, runtime, ignore)
    assert list(prep.root.iterdir()) == []
    assert all(path.is_file() for path in sources)
    assert prep.store.all(prep.KIND) == []


async def test_packer_must_not_adopt_unselected_neighbor_files(setup, tmp_path):
    prep, model, profile, runtime = setup
    neighbor = write_gguf(tmp_path / 'unrelated.gguf', 'dflash')
    script = Path(runtime.path) / 'tools/dsv41/pack_gguf.py'
    script.write_text(script.read_text() + '\nwith (out / "tensors.txt").open("a") as f: f.write(' + repr(
        f'\nfile 9 {neighbor}\nt mtp.unselected F32 9 0 4 1\n') + ')\n')
    with pytest.raises(ValueError, match='未选择的配套文件'):
        await prep.ensure(model, profile, runtime, ignore)
    assert list(prep.root.iterdir()) == [] and neighbor.is_file()


async def test_native_qwen_controls_are_not_applied_and_timeout_remains_configurable(setup, tmp_path):
    prep, model, profile, runtime = setup
    resolved = await prep.ensure(model, profile, runtime, ignore)
    resolved.parameters.update(startup_timeout=7200, kv_cache='int8', expert_cache='auto', spec=4)
    context = LaunchContext(port=18099, instance_id='test', work_dir=str(tmp_path / 'instance'), data_dir=str(prep.paths.root))
    adapter = get_adapters()['strata']
    launch = adapter.build_launch(model, resolved, runtime, context)
    assert launch.startup_timeout == 7200
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text())
    assert cfg['engine_ready_s'] == 7200
    assert '--kv' not in cfg['args'] and '--expert-cache' not in cfg['args'] and '--spec' not in cfg['args']
    resolved.parameters['vision_enabled'] = True
    with pytest.raises(ValueError, match='图像输入'):
        adapter.build_launch(model, resolved, runtime, context)


def test_model_specific_fields_do_not_expose_qwen_controls(setup):
    _, model, _, runtime = setup
    adapter = get_adapters()['strata']
    keys = {field.key for field in profile_fields(adapter, model, runtime)}
    assert {'engram', 'engram_meta', 'dspark', 'dspark_mode'} <= keys
    assert not {'kv_cache', 'expert_cache', 'spec'} & keys
    qwen = model.model_copy(update={'architecture': 'qwen4exp'})
    qwen_keys = {field.key for field in profile_fields(adapter, qwen, runtime)}
    assert 'kv_cache' in qwen_keys and 'dspark' not in qwen_keys


def test_deepseek_needs_its_own_engine(setup):
    prep, model, profile, runtime = setup
    ds.engine_path(Path(runtime.path)).unlink()
    (Path(runtime.path) / 'engine').mkdir()
    (Path(runtime.path) / 'engine/strata.exe').write_bytes(b'Qwen')
    assert 'strata-dsv41.exe' in prep.preflight(model, profile, runtime)['message']
    assert get_adapters()['strata'].inspect(model, runtime).status == 'unsupported'


@pytest.mark.parametrize('source', ['parameter', 'extra'])
async def test_native_expert_profile_none_and_caller_last_dspark(setup, tmp_path, source):
    from modloader.mtp import argument
    prep, model, profile, runtime = setup
    profile.extra_args = ['--dspark', 'none']
    if source == 'extra':
        profile.extra_args += ['--expert-profile', 'none']
    else:
        profile.parameters['expert_profile'] = 'none'
    resolved = await prep.ensure(model, profile, runtime, ignore)
    assert (Path(resolved.parameters['pack']) / 'expert-profile.bin').exists()
    context = LaunchContext(port=18099, instance_id='test', work_dir=str(tmp_path / 'instance'), data_dir=str(prep.paths.root))
    get_adapters()['strata'].build_launch(model, resolved, runtime, context)
    cfg_path = Path(context.work_dir) / 'strata.json'
    cfg = json.loads(cfg_path.read_text())
    assert argument(cfg['args'], '--expert-profile') == 'none'
    assert argument(cfg['args'], '--dspark') == 'none'
    imported = import_configuration(str(cfg_path))
    assert argument(imported['parameters']['strata_config']['args'], '--expert-profile') == 'none'
    profile.extra_args = []
    profile.parameters.pop('expert_profile', None)
    assert prep.preflight(model, profile, runtime)['pack'] == resolved.parameters['pack']
