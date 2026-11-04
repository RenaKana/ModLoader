import asyncio
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import psutil
import pytest

from modloader.models import AppPaths, LoadProfile, ModelAsset, RuntimeInstallation
from modloader.store import Store
from modloader.strata_preparation import StrataPreparation, expert_profile_path


def write_expert_profile(path, layers=48, experts=512, *, version=1, slots=1, pairs=((0, 0),)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'STRP' + struct.pack('<5I', version, layers, experts, slots, len(pairs)) +
                     b''.join(struct.pack('<HH', *pair) for pair in pairs))
    return path


def write_pack(pack):
    pack.mkdir(parents=True, exist_ok=True)
    (pack / 'dense.bin').write_bytes(b'dense')
    (pack / 'index.txt').write_text('# strata pack index v3\nweight ' + ' '.join(['0'] * 18) + '\n')
    (pack / 'conversions.json').write_text('{"schema":1,"tensors":[]}')
    (pack / 'native_experts.txt').write_text('# strata native experts v3\n0 17 20 0 10 0 1 2\n')
    tokenizer = pack / 'tokenizer'
    tokenizer.mkdir(exist_ok=True)
    for name, data in [('vocab.json', {'test': 0}), ('token_type.json', [1]), ('tokenizer.json', {'model': 'gpt2'})]:
        (tokenizer / name).write_text(json.dumps(data))
    (tokenizer / 'merges.txt').write_text('')
    return pack


@pytest.fixture
def fixture(tmp_path):
    paths = AppPaths(tmp_path / 'data')
    store = Store(paths.root / 'database.sqlite')
    source = tmp_path / 'model.gguf'
    source.write_bytes(b'fixture, no tensors')
    model = ModelAsset(id='model', name='Model', path=str(source), format='gguf',
                       metadata={'qwen4exp.block_count': 48, 'qwen4exp.expert_count': 512})
    runtime_root = tmp_path / 'runtime'
    (runtime_root / 'tools').mkdir(parents=True)
    script = runtime_root / 'tools' / 'iq_pack.py'
    script.write_text('print("placeholder")')
    write_expert_profile(runtime_root / 'data' / 'expert-profile.bin')
    runtime = RuntimeInstallation(id='runtime', name='Strata', backend='strata', path=str(runtime_root),
                                  metadata={'python': sys.executable})
    profile = LoadProfile(id='new', name='New', model_id=model.id, runtime_id=runtime.id, backend='strata', alias='new')
    preparation = StrataPreparation(paths, store)
    yield preparation, model, profile, runtime
    store.close()


def install_fake(runtime, mode='success'):
    # Real subprocess exercises lifecycle and the production argv; it never opens model tensors.
    import inspect
    helper = inspect.getsource(write_pack)
    script = '''import argparse, json, subprocess, sys, time
from pathlib import Path
''' + helper + '''
parser = argparse.ArgumentParser()
parser.add_argument('--gguf', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
out = Path(args.out)
print('api_key=SECRET preparing local fixture', flush=True)
MODE = ''' + repr(mode) + '''
if MODE in ('cancel', 'fail_tree'):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    (Path(__file__).parent.parent / 'child.pid').write_text(str(child.pid))
    print('child ready', flush=True)
    if MODE == 'fail_tree':
        print('trigger progress error', flush=True)
    time.sleep(120)
if MODE == 'failure':
    (out / 'dense.bin').write_bytes(b'partial')
    sys.exit(7)
write_pack(out)
if MODE == 'incomplete':
    (out / 'native_experts.txt').unlink()
'''
    (Path(runtime.path) / 'tools' / 'iq_pack.py').write_text(script)


def imported(preparation, model, runtime, pack, name='Existing'):
    profile = LoadProfile(id=name, model_id=model.id, runtime_id=runtime.id, backend='strata', name=name, alias=name,
                          parameters={'strata_config': {'cwd': str(pack.parent),
                                      'args': ['--native', model.path, '--pack', str(pack)],
                                      'tokenizer': str(pack / 'tokenizer'), 'vision': {'gpu': 5}},
                                      'context_length': 123, 'kv_cache': 'q4', 'spec': 12}, generation={'temperature': 0.2})
    preparation.store.put('profiles', profile.model_dump())
    return profile


def test_preflight_is_read_only_and_reuses_only_resources(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    pack = write_pack(tmp_path / 'external')
    old = imported(prep, model, runtime, pack)
    profile.parameters = {'context_length': 500, 'spec': 4}
    profile.generation = {'temperature': 0.8}
    before = prep.store.all('profiles')
    result = prep.preflight(model, profile, runtime)
    resolved = prep.resolved_profile(model, profile, runtime)
    assert result['status'] == 'ready'
    assert result['source']['profile_id'] == old.id
    assert resolved.parameters == {**profile.parameters, 'pack': str(pack), 'tokenizer': str(pack / 'tokenizer'),
                                   'expert_profile': str(Path(runtime.path) / 'data' / 'expert-profile.bin')}
    assert resolved.generation == profile.generation
    assert resolved.extra_args == profile.extra_args
    assert prep.store.all('profiles') == before
    assert not prep.root.exists()
    assert 'pack' not in profile.parameters
    json.dumps(result)


def test_distinct_candidates_require_choice_and_duplicate_pairs_collapse(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    a = write_pack(tmp_path / 'a')
    b = write_pack(tmp_path / 'b')
    imported(prep, model, runtime, a, 'A')
    imported(prep, model, runtime, a, 'Duplicate A')
    imported(prep, model, runtime, b, 'B')
    result = prep.preflight(model, profile, runtime)
    assert result['status'] == 'choice_required'
    assert len(result['candidates']) == 2
    profile.parameters['strata_resource_id'] = result['candidates'][1]['id']
    assert prep.resolved_profile(model, profile, runtime).parameters['pack'] == str(b)
    profile.parameters['strata_resource_id'] = 'gone'
    assert prep.preflight(model, profile, runtime)['status'] == 'error'


def test_other_models_and_wrong_native_are_not_candidates(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    pack = write_pack(tmp_path / 'external')
    wrong = imported(prep, model, runtime, pack)
    wrong.parameters['strata_config']['args'][1] = str(tmp_path / 'another.gguf')
    prep.store.put('profiles', wrong.model_dump())
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    assert prep.preflight(model, wrong, runtime)['status'] == 'error'
    wrong.model_id = 'other-model'
    prep.store.put('profiles', wrong.model_dump())
    assert prep.preflight(model, profile, runtime)['candidates'] == []


def test_explicit_resources_override_and_invalid_paths_never_fall_back(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    pack = write_pack(tmp_path / 'external')
    old = imported(prep, model, runtime, pack)
    profile.parameters = {'pack': str(tmp_path / 'missing')}
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    profile.parameters = {'tokenizer': str(tmp_path / 'missing')}
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    replacement = write_pack(tmp_path / 'replacement')
    old.parameters.update(pack=str(replacement), tokenizer=str(replacement / 'tokenizer'))
    assert prep.resolved_profile(model, old, runtime).parameters['pack'] == str(replacement)


def test_configuration_file_relative_paths_and_no_mutation(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    pack = write_pack(tmp_path / 'pack')
    config = tmp_path / 'config.json'
    text = json.dumps({'args': ['--native', 'model.gguf', '--pack', 'pack'], 'tokenizer': 'pack/tokenizer'})
    config.write_text(text)
    profile.parameters = {'configuration': str(config)}
    assert prep.preflight(model, profile, runtime)['pack'] == str(pack)
    assert config.read_text() == text


@pytest.mark.parametrize('damage', ['marker', 'index', 'json'])
def test_incomplete_existing_resources_are_rejected(fixture, tmp_path, damage):
    prep, model, profile, runtime = fixture
    pack = write_pack(tmp_path / 'external')
    if damage == 'marker':
        (pack / 'native_experts.txt').unlink()
    elif damage == 'index':
        (pack / 'index.txt').write_text('# header only')
    else:
        (pack / 'tokenizer' / 'vocab.json').write_text('{bad')
    profile.parameters = {'pack': str(pack)}
    assert prep.preflight(model, profile, runtime)['status'] == 'error'


@pytest.mark.asyncio
async def test_prepare_registers_reuses_and_invalidates_source_and_runtime(fixture):
    prep, model, profile, runtime = fixture
    install_fake(runtime)
    messages = []
    async def progress(event):
        messages.append(event)
    resolved = await prep.ensure(model, profile, runtime, progress)
    assert 'SECRET' not in json.dumps(messages)
    assert '[REDACTED]' in json.dumps(messages)
    assert all('percent' not in event for event in messages)
    assert len(prep.store.all(prep.KIND)) == 1
    assert profile.parameters == {}
    assert await prep.ensure(model, profile, runtime, progress) == resolved
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    Path(model.path).write_bytes(b'changed source')
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    # A persisted/manual pack path must not bypass generated source invalidation.
    assert prep.preflight(model, resolved, runtime)['status'] == 'error'
    stale_profile = resolved.model_copy(update={'id': 'stale'})
    prep.store.put('profiles', stale_profile.model_dump())
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    second = await prep.ensure(model, profile, runtime, progress)
    assert second.parameters['pack'] != resolved.parameters['pack']
    assert Path(resolved.parameters['pack']).exists()
    runtime.version = 'new-version'
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    assert len(prep.owned_files(model)) == 16
    prep.remove_owned(model)
    assert prep.store.all(prep.KIND) == []
    assert list(prep.root.iterdir()) == []


@pytest.mark.parametrize('mode', ['failure', 'incomplete'])
@pytest.mark.asyncio
async def test_failure_removes_only_exclusive_output_and_does_not_register(fixture, tmp_path, mode):
    prep, model, profile, runtime = fixture
    external = write_pack(tmp_path / 'external')
    install_fake(runtime, mode)
    async def progress(event):
        pass
    with pytest.raises(ValueError):
        await prep.ensure(model, profile, runtime, progress)
    assert prep.store.all(prep.KIND) == []
    assert list(prep.root.iterdir()) == []
    assert external.exists()


@pytest.mark.parametrize('mode', ['cancel', 'fail_tree'])
@pytest.mark.asyncio
async def test_cancel_or_callback_error_stops_process_tree_and_cleans(fixture, mode):
    prep, model, profile, runtime = fixture
    install_fake(runtime, mode)
    ready = asyncio.Event()
    child = None
    async def progress(event):
        nonlocal child
        if event['message'] == 'child ready':
            pid = int((Path(runtime.path) / 'child.pid').read_text())
            child = psutil.Process(pid)
            ready.set()
        if event['message'] == 'trigger progress error':
            raise RuntimeError('callback failed')
    task = asyncio.create_task(prep.ensure(model, profile, runtime, progress))
    await asyncio.wait_for(ready.wait(), 10)
    assert child is not None
    if mode == 'cancel':
        task.cancel()
    with pytest.raises(asyncio.CancelledError if mode == 'cancel' else RuntimeError):
        await asyncio.wait_for(task, 10)
    await asyncio.to_thread(child.wait, 5)
    assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    assert prep.store.all(prep.KIND) == []
    assert list(prep.root.iterdir()) == []


def test_split_source_identity_includes_all_shards_and_rejects_missing(fixture):
    prep, model, profile, runtime = fixture
    first = Path(model.path).with_name('m-00001-of-00002.gguf')
    second = first.with_name('m-00002-of-00002.gguf')
    first.write_bytes(b'first')
    model.path = str(first)
    assert prep.preflight(model, profile, runtime)['status'] == 'error'
    second.write_bytes(b'second')
    identity = prep._identity(model, runtime)
    assert [item['path'] for item in identity['source']] == [str(first), str(second)]


def test_owned_removal_rejects_external_paths(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    external = write_pack(tmp_path / 'external')
    prep.store.put(prep.KIND, {'id': '../external', 'model_id': model.id, 'pack': str(external)})
    with pytest.raises(ValueError):
        prep.remove_owned(model)
    assert external.exists()


def test_expert_profile_defaults_follow_geometry_and_ignore_other_backend_files(fixture):
    prep, model, profile, runtime = fixture
    data = Path(runtime.path) / 'data'
    assert expert_profile_path(model, runtime) == str(data / 'expert-profile.bin')
    model.metadata['qwen4exp.expert_count'] = 256
    coder = write_expert_profile(data / 'expert-profile-coder.bin', experts=256)
    assert expert_profile_path(model, runtime) == str(coder)
    coder.unlink()
    write_expert_profile(data / 'expert-profile-sycl.bin', experts=256)
    with pytest.raises(ValueError, match='256'):
        expert_profile_path(model, runtime)


@pytest.mark.parametrize('damage', ['magic', 'version', 'geometry', 'slots', 'truncated', 'range', 'empty'])
def test_explicit_expert_profile_validation_matches_engine_contract(fixture, tmp_path, damage):
    prep, model, profile, runtime = fixture
    path = tmp_path / 'custom.bin'
    if damage == 'version':
        write_expert_profile(path, version=2)
    elif damage == 'geometry':
        write_expert_profile(path, experts=256)
    elif damage == 'slots':
        write_expert_profile(path, slots=0)
    elif damage == 'range':
        write_expert_profile(path, pairs=((48, 0),))
    elif damage == 'empty':
        write_expert_profile(path, slots=0, pairs=())
    else:
        write_expert_profile(path)
        original = path.read_bytes()
        path.write_bytes(b'FAIL' + original[4:] if damage == 'magic' else original[:-1])
    profile.parameters['expert_profile'] = str(path)
    result = prep.preflight(model, profile, runtime)
    assert result['status'] == 'error'
    assert 'expert profile' in result['message']
    assert not prep.root.exists()
    # A valid installed default must never silently replace an invalid explicit choice.
    with pytest.raises(ValueError):
        expert_profile_path(model, runtime, path)


def test_expert_profile_reader_accepts_upstream_optional_trailing_table(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    path = write_expert_profile(tmp_path / 'custom.bin', slots=3, pairs=((0, 0), (47, 511)))
    assert expert_profile_path(model, runtime, path) == str(path)
    with path.open('ab') as stream:
        stream.write(struct.pack('<i', -1) * (48 * 512))
    assert expert_profile_path(model, runtime, path) == str(path)


def test_expert_profile_explicit_imported_and_extra_precedence(fixture, tmp_path):
    prep, model, profile, runtime = fixture
    pack = write_pack(tmp_path / 'pack')
    old = imported(prep, model, runtime, pack)
    imported_path = write_expert_profile(tmp_path / 'imported.bin')
    explicit_path = write_expert_profile(tmp_path / 'explicit.bin')
    extra_path = write_expert_profile(tmp_path / 'extra.bin')
    old.parameters['strata_config']['args'] += ['--expert-profile', imported_path.name]
    before = old.model_dump()
    assert prep.preflight(model, old, runtime)['expert_profile'] == str(imported_path)
    assert old.model_dump() == before
    old.parameters['expert_profile'] = str(explicit_path)
    assert prep.preflight(model, old, runtime)['expert_profile'] == str(explicit_path)
    old.extra_args = ['--expert-profile=' + extra_path.name]
    resolved = prep.resolved_profile(model, old, runtime)
    assert resolved.parameters['expert_profile'] == str(extra_path)
    assert resolved.extra_args == old.extra_args
    old.extra_args = ['--expert-profile', str(tmp_path / 'missing.bin')]
    assert prep.preflight(model, old, runtime)['status'] == 'error'


def test_expert_profile_geometry_falls_back_to_readonly_gguf_metadata(fixture):
    prep, model, profile, runtime = fixture
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    metadata = b''.join(string(key) + struct.pack('<II', 4, value) for key, value in
                        [('qwen4exp.block_count', 48), ('qwen4exp.expert_count', 512)])
    original = b'GGUF' + struct.pack('<IQQ', 3, 0, 2) + metadata
    Path(model.path).write_bytes(original)
    model.metadata = {'qwen4exp.block_count': 48}
    assert expert_profile_path(model, runtime).endswith('expert-profile.bin')
    assert model.metadata == {'qwen4exp.block_count': 48}
    assert Path(model.path).read_bytes() == original
    changed = original.replace(struct.pack('<II', 4, 48), struct.pack('<II', 4, 49), 1)
    source = Path(model.path)
    stamp = source.stat()
    source.write_bytes(changed)
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match='49'):
        expert_profile_path(model, runtime)


async def test_missing_expert_profile_fails_before_preparation_subprocess(fixture):
    prep, model, profile, runtime = fixture
    (Path(runtime.path) / 'data' / 'expert-profile.bin').unlink()
    install_fake(runtime)
    messages = []
    async def progress(event):
        messages.append(event)
    with pytest.raises(ValueError, match='expert profile'):
        await prep.ensure(model, profile, runtime, progress)
    assert messages == [] and not prep.root.exists()
    assert prep.store.all(prep.KIND) == []


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows inherited stdin pipe initialization regression')
def test_prepare_completes_when_parent_thread_is_reading_stdin_pipe(fixture, tmp_path):
    from modloader.runtime import WindowsJob

    prep, model, profile, runtime = fixture
    install_fake(runtime)
    child_pid_file = tmp_path / 'pack-child.pid'
    tool = Path(runtime.path) / 'tools' / 'iq_pack.py'
    tool.write_text('import os\nfrom pathlib import Path\n' +
                    f'Path({str(child_pid_file)!r}).write_text(str(os.getpid()))\n' + tool.read_text())
    request = tmp_path / 'preparation-request.json'
    request.write_text(json.dumps({'root': str(prep.paths.root), 'model': model.model_dump(),
                                   'profile': profile.model_dump(), 'runtime': runtime.model_dump()}))
    helper = tmp_path / 'stdin-reader-helper.py'
    helper.write_text('''import asyncio, json, os, sys, threading, time, traceback
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from modloader.models import AppPaths, ModelAsset, LoadProfile, RuntimeInstallation
from modloader.store import Store
from modloader.strata_preparation import StrataPreparation

# Reproduce the management core's exit-pipe listener. The outer test keeps stdin
# open: inheriting this handle can block the converter in Python stdio startup.
threading.Thread(target=sys.stdin.buffer.read, daemon=True).start()
time.sleep(.1)
request = json.loads(Path(sys.argv[2]).read_text())
paths = AppPaths(Path(request['root']))
store = Store(paths.root / 'database.sqlite')
preparation = StrataPreparation(paths, store)
async def progress(event):
    pass
async def run():
    return await asyncio.wait_for(preparation.ensure(
        ModelAsset(**request['model']), LoadProfile(**request['profile']),
        RuntimeInstallation(**request['runtime']), progress), timeout=5)
code = 0
try:
    result = asyncio.run(run())
    assert Path(result.parameters['pack'], 'native_experts.txt').is_file()
    assert len(store.all(preparation.KIND)) == 1
except BaseException:
    traceback.print_exc()
    code = 1
finally:
    store.close()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid interpreter shutdown waiting on the deliberately blocked reader.
    os._exit(code)
''')
    process = subprocess.Popen([sys.executable, '-u', str(helper),
                                str(Path(__file__).resolve().parents[1] / 'core'), str(request)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    job = None
    try:
        job = WindowsJob(process.pid)
        # communicate() would close stdin and erase the exact condition under test.
        process.wait(timeout=15)
        job.close()  # No descendant can keep the captured stdout pipe open after the helper exits.
        output = process.stdout.read().decode('utf-8', errors='replace')
        assert process.returncode == 0, output
        assert child_pid_file.is_file(), output
        assert not psutil.pid_exists(int(child_pid_file.read_text()))
        assert prep.preflight(model, profile, runtime)['status'] == 'ready'
        records = prep.store.all(prep.KIND)
        assert len(records) == 1
        assert (Path(records[0]['pack']) / 'dense.bin').read_bytes() == b'dense'
    finally:
        if job:
            job.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


@pytest.mark.parametrize('relative', ['dense.bin', 'index.txt', 'conversions.json', 'native_experts.txt',
                                     'tokenizer/vocab.json', 'tokenizer/merges.txt',
                                     'tokenizer/token_type.json', 'tokenizer/tokenizer.json'])
def test_preflight_success_cache_tracks_every_read_file_not_load_parameters(fixture, tmp_path, monkeypatch, relative):
    prep, model, profile, runtime = fixture
    pack = write_pack(tmp_path / 'pack')
    profile.parameters = {'pack': str(pack), 'context_length': 4096}
    original = StrataPreparation._validate_tokenizer
    validations = []
    def counted(path):
        validations.append(str(path))
        return original(path)
    monkeypatch.setattr(StrataPreparation, '_validate_tokenizer', staticmethod(counted))
    assert prep.resolved_profile(model, profile, runtime).parameters['pack'] == str(pack)
    profile.parameters['context_length'] = 8192
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert len(validations) == 1
    target = pack / relative
    stamp = target.stat()
    with target.open('ab') as stream:
        stream.write(b'\n')
    os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert len(validations) == 2
    target.unlink()
    assert prep.preflight(model, profile, runtime)['status'] == 'error'


def test_expert_profile_cache_invalidates_file_and_does_not_remember_validation_failure(fixture, monkeypatch):
    prep, model, profile, runtime = fixture
    target = Path(runtime.path) / 'data' / 'expert-profile.bin'
    original_open = Path.open
    reads = []
    def counted(path, *args, **kwargs):
        if path == target and args and args[0] == 'rb':
            reads.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    assert expert_profile_path(model, runtime) == str(target)
    assert expert_profile_path(model, runtime) == str(target) and len(reads) == 1
    write_expert_profile(target, experts=256)
    for _ in range(2):
        with pytest.raises(ValueError, match='256'):
            expert_profile_path(model, runtime, target)
    assert len(reads) == 3
    write_expert_profile(target)
    assert expert_profile_path(model, runtime) == str(target) and len(reads) == 4
