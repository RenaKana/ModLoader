"""Copy/registration contracts using tiny real containers and a local guest bridge."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from modloader import assets, engine_preparation, model_storage, platforms
from modloader.backends import installers, native_engines
from modloader.models import AppPaths, Compatibility, LoadProfile, ModelAsset, ModelLocation, RuntimeInstallation
from modloader.store import Store
from test_ninfer_assets import write_ninfer


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Production copy/parser code, with only the WSL transport translated."""
    paths = AppPaths(tmp_path / 'data')
    backing = paths.runtimes / 'wsl-ModLoader'
    backing.mkdir()
    guest = tmp_path / 'guest'
    guest.mkdir()
    base = guest / 'models'
    store = Store(paths.root / 'test.sqlite')
    runtime = SimpleNamespace(instances={}, adapters={'sparkinfer': SimpleNamespace(inspect=lambda *args: Compatibility(status='supported'))})
    service = model_storage.ModelStorageMigration(paths, store, runtime, asyncio.Lock())
    monkeypatch.setattr(installers, '_verify_wsl_owner', lambda paths: None)
    monkeypatch.setattr(platforms, 'require_wsl2', lambda distribution: None)
    monkeypatch.setattr(model_storage, 'require_wsl2', lambda distribution: None)

    def translate(value):
        if isinstance(value, str) and value.startswith(model_storage.STORAGE_BASE):
            return str(base) + value[len(model_storage.STORAGE_BASE):].replace('/', '\\')
        if isinstance(value, list):
            return [translate(item) for item in value]
        if isinstance(value, dict):
            return {key: translate(item) for key, item in value.items()}
        return value

    def native(value):
        if isinstance(value, str):
            if value.startswith(str(base)):
                return model_storage.STORAGE_BASE + value[len(str(base)):].replace('\\', '/')
            return value
        if isinstance(value, list):
            return [native(item) for item in value]
        if isinstance(value, dict):
            return {key: native(item) for key, item in value.items()}
        return value

    def code_local(code):
        code = code.replace("'/opt/modloader/models'", repr(str(base))).replace("Path('/')", 'Path(' + repr(base.anchor) + ')')
        # Windows has no directory-fsync handle API; real WSL exercises this
        # durability boundary, while this fixture checks file/registration logic.
        code = code.replace('for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True):',
                            'for directory in []:')
        if 'os.statvfs' in code:
            code = ('import os,shutil\nfrom types import SimpleNamespace\n'
                    'os.statvfs=lambda p: SimpleNamespace(f_bavail=shutil.disk_usage(p).free,f_frsize=1)\n') + code
        return code

    def python(distribution, code, args=(), *, timeout=60, input=None):
        assert distribution == 'ModLoader'
        payload = json.dumps(translate(json.loads(input))) if input is not None else None
        converted = []
        for arg in args:
            try:
                parsed = json.loads(arg)
                converted.append(json.dumps(translate(parsed)) if isinstance(parsed, (dict, list)) else translate(arg))
            except (TypeError, ValueError):
                converted.append(translate(arg))
        command = [sys.executable, '-X', 'utf8', '-c', code_local(code), *map(str, converted)]
        result = subprocess.run(command, input=payload, text=True, capture_output=True, encoding='utf-8', timeout=timeout)
        if result.returncode:
            raise ValueError(result.stderr)
        try:
            return json.dumps(native(json.loads(result.stdout)))
        except ValueError:
            return result.stdout

    def launch(spec, context):
        request_path = Path(spec.argv[-1])
        request = json.loads(request_path.read_text(encoding='utf-8'))
        request_path.write_text(json.dumps(translate(request)), encoding='utf-8')
        return spec.model_copy(update={'platform': 'windows', 'argv': [sys.executable, '-X', 'utf8', '-u', '-c', code_local(spec.argv[-2]), str(request_path)],
                                      'cwd': str(guest), 'cleanup_argv': []})

    monkeypatch.setattr(model_storage, 'wsl_python', python)
    monkeypatch.setattr(engine_preparation, 'wsl_python', python)
    monkeypatch.setattr(platforms, 'wsl_python', python)
    monkeypatch.setattr(platforms, 'windows_path', lambda distribution, value: translate(value))
    monkeypatch.setattr(model_storage, 'windows_to_wsl', lambda value, distribution: Path(value).as_posix())
    monkeypatch.setattr(model_storage, 'prepare_launch', launch)
    monkeypatch.setattr(engine_preparation, 'windows_path', lambda distribution, value: translate(value))
    machine = SimpleNamespace(paths=paths, store=store, service=service, runtime=runtime,
                              guest=guest, base=base, translate=translate, native=native,
                              code_local=code_local, launch=launch, events=[])

    async def progress(event):
        machine.events.append(event)
    machine.progress = progress
    yield machine
    store.close()


def register(machine, *, fmt='safetensors', managed=False, source_root=None):
    root = source_root or machine.paths.models / 'source'
    root.mkdir(parents=True)
    if fmt == 'ninfer':
        selected, _, _ = write_ninfer(root / 'model.ninfer')
    elif fmt == 'gguf':
        import struct
        content = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
        selected = root / 'weights-00001-of-00002.gguf'
        selected.write_bytes(content)
        (root / 'weights-00002-of-00002.gguf').write_bytes(content)
    else:
        selected = root
        (root / 'config.json').write_text('{"model_type":"llama"}')
        (root / 'model.safetensors').write_bytes(b'tiny metadata-only weight fixture')
        (root / 'tokenizer.json').write_text('{}')
        (root / 'nested').mkdir()
        (root / 'nested' / 'extra_config.json').write_text('{}')
    if fmt != 'safetensors':
        (root / 'tokenizer.json').write_text('{}')
        (root / 'chat_template.jinja').write_text('{{ messages }}')
    model = assets.inspect_asset(str(selected), managed=managed,
                                 source={'kind': 'download', 'managed_root': str(root)} if managed else None)
    machine.store.put('models', model.model_dump())
    runtime = RuntimeInstallation(id='native', platform='wsl2', backend='sparkinfer', name='Native', path='/opt/spark', metadata={'distribution': 'ModLoader'})
    profile = LoadProfile(id='native-profile', model_id=model.id, name='Native', backend=runtime.backend, runtime_id=runtime.id, alias='native',
                          parameters={'tokenizer': str(root / 'tokenizer.json')})
    machine.store.put('runtimes', runtime.model_dump())
    machine.store.put('profiles', profile.model_dump())
    return model, profile, runtime


async def migrate(machine, model, *, delete_source=False, progress=None):
    preview = await machine.service.preview(model.id)
    assert preview['eligible'], preview
    result = await machine.service.run(dict(model_id=model.id, plan_id=preview['id'], delete_source=delete_source),
                                       'fixture-job', progress or machine.progress, asyncio.Event())
    return preview, result


@pytest.mark.parametrize('fmt', ['safetensors', 'gguf', 'ninfer'])
async def test_copy_complete_model_and_switch_only_modloader_profiles(storage, fmt):
    model, profile, runtime = register(storage, fmt=fmt)
    windows = RuntimeInstallation(id='windows', backend='custom', name='Windows', path='C:\\fixture.exe')
    old_profile = profile.model_copy(update={'id': 'windows-profile', 'runtime_id': windows.id, 'backend': 'custom'})
    storage.store.put('runtimes', windows.model_dump())
    storage.store.put('profiles', old_profile.model_dump())
    before = deepcopy(storage.store.get('models', model.id))
    preview, result = await migrate(storage, model)
    target = storage.store.get('models', result['model_id'])
    assert target['location'] == {'platform': 'wsl2', 'distribution': 'ModLoader'}
    assert target['managed'] and target['source']['original_model_id'] == model.id
    assert storage.store.get('models', model.id) == before
    assert storage.store.get('profiles', old_profile.id) == old_profile.model_dump()
    switched = storage.store.get('profiles', profile.id)
    assert switched['model_id'] == target['id'] and switched['revision'] == 2
    assert switched['parameters']['tokenizer'].startswith('/opt/modloader/models/')
    assert preview['preserved_profiles'][0]['id'] == old_profile.id
    assert not preview['delete_source_allowed']
    assert result['source_deleted'] is False
    assert all(Path(value).exists() for value in model.files)
    for row in storage.service.plans.values():
        assert row['id'] != preview['id']
    if fmt in {'gguf', 'ninfer'}:
        assert len(target['files']) == 2
        assert Path(storage.translate(str(Path(target['path']).parent).replace('\\', '/') + '/tokenizer.json')).is_file()
    else:
        assert any(path.endswith('/nested/extra_config.json') for path in target['files'])
    assert storage.events[-1]['completed_files'] == preview['file_count']
    assert storage.events[-1]['copied_bytes'] == preview['total_bytes']


async def test_managed_delete_verifies_transfer_and_removes_only_copied_sources(storage):
    model, profile, runtime = register(storage, fmt='ninfer', managed=True)
    unrelated = Path(model.path).parent / 'another.gguf'
    unrelated.write_bytes(b'not selected')
    preview, result = await migrate(storage, model, delete_source=True)
    assert preview['delete_source_allowed']
    assert result['source_deleted'] and storage.store.get('models', model.id) is None
    assert all(not Path(value).exists() for value in model.files)
    assert unrelated.read_bytes() == b'not selected'
    assert any(event['phase'] == 'verifying' for event in storage.events)
    assert storage.store.get('profiles', profile.id)['model_id'] == result['model_id']


@pytest.mark.parametrize('fmt', ['safetensors', 'gguf', 'ninfer'])
@pytest.mark.parametrize('delete_source', [False, True], ids=['keep-original', 'delete-original'])
async def test_external_import_migration_honors_source_file_choice(storage, fmt, delete_source):
    root = storage.paths.root.parent / 'external-library' / 'imported-model'
    model, profile, runtime = register(storage, fmt=fmt, source_root=root)
    assert not model.managed and 'managed_root' not in model.source
    assert not root.is_relative_to(storage.paths.models)
    sibling = (root.parent if fmt == 'safetensors' else root) / 'unrelated.bin'
    sibling.write_bytes(b'unrelated user file')
    if fmt == 'safetensors':
        copied = [root / name for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'nested/extra_config.json')]
    else:
        copied = [*[Path(value) for value in model.files], root / 'tokenizer.json', root / 'chat_template.jinja']
    originals = {path: path.read_bytes() for path in copied}
    original_record = storage.store.get('models', model.id)

    preview, result = await migrate(storage, model, delete_source=delete_source)

    assert preview['delete_source_allowed'] and not preview['delete_source_reasons']
    assert preview['delete_source_bytes'] == sum(len(value) for value in originals.values())
    assert result['source_deleted'] is delete_source
    target = storage.store.get('models', result['model_id'])
    target_root = Path(storage.translate(target['path']))
    if fmt != 'safetensors':
        target_root = target_root.parent
    for path, content in originals.items():
        assert (target_root / path.relative_to(root)).read_bytes() == content
        if delete_source:
            assert not path.exists()
        else:
            assert path.read_bytes() == content
    assert sibling.read_bytes() == b'unrelated user file'
    assert root.is_dir()  # Only the copied files are removed, never their source directory.
    assert storage.store.get('models', model.id) == (None if delete_source else original_record)
    assert storage.store.get('profiles', profile.id)['model_id'] == target['id']
    assert any(event['phase'] == 'verifying' for event in storage.events) is delete_source


async def test_external_and_other_model_resource_references_prevent_source_deletion(storage):
    model, profile, runtime = register(storage, managed=True)
    other = ModelAsset(id='other', name='Other', path=model.path, format=model.format, files=model.files)
    storage.store.put('models', other.model_dump())
    preview = await storage.service.preview(model.id)
    assert not preview['delete_source_allowed']
    assert any('其他模型' in value for value in preview['delete_source_reasons'])
    with pytest.raises(ValueError, match='其他模型'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id'], delete_source=True),
                                  'job', storage.progress, asyncio.Event())
    assert storage.store.get('models', model.id) and not storage.base.exists()


async def test_changed_profiles_reject_registration_and_clean_only_owned_target(storage):
    model, profile, runtime = register(storage)
    sibling = storage.base / 'unrelated'
    sibling.mkdir(parents=True)
    (sibling / 'keep').write_text('keep')
    async def progress(event):
        await storage.progress(event)
        if event['phase'] == 'copied':
            storage.store.put('profiles', profile.model_copy(update={'name': 'Changed while copying'}).model_dump())
    preview = await storage.service.preview(model.id)
    with pytest.raises(ValueError, match='已变化'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'job', progress, asyncio.Event())
    assert [value['id'] for value in storage.store.all('models')] == [model.id]
    assert storage.store.get('profiles', profile.id)['name'] == 'Changed while copying'
    assert not Path(storage.translate(preview['target_directory'])).exists()
    assert (sibling / 'keep').read_text() == 'keep'
    assert not storage.service.active


async def test_cancel_and_failed_destination_inspection_keep_original(storage):
    model, profile, runtime = register(storage, fmt='ninfer')
    cancellation = asyncio.Event()
    async def progress(event):
        if event['phase'] == 'copying':
            cancellation.set()
    preview = await storage.service.preview(model.id)
    with pytest.raises(asyncio.CancelledError):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'job', progress, cancellation)
    assert not Path(storage.translate(preview['target_directory'])).exists()
    assert storage.store.get('profiles', profile.id)['model_id'] == model.id
    assert Path(model.path).exists()
    async def corrupt(event):
        if event['phase'] == 'copied':
            Path(storage.translate(next_preview['target_path'])).write_bytes(b'broken')
    next_preview = await storage.service.preview(model.id)
    with pytest.raises(ValueError, match='NInfer'):
        await storage.service.run(dict(model_id=model.id, plan_id=next_preview['id']), 'job-2', corrupt, asyncio.Event())
    assert storage.store.get('profiles', profile.id)['model_id'] == model.id
    assert not Path(storage.translate(next_preview['target_directory'])).exists()


async def test_running_model_space_and_foreign_distro_are_reported(storage, monkeypatch):
    model, profile, runtime = register(storage)
    storage.runtime.instances['i'] = SimpleNamespace(model_id=model.id, state='ready')
    assert not (await storage.service.preview(model.id))['eligible']
    storage.runtime.instances.clear()
    with monkeypatch.context() as patch:
        patch.setattr(storage.service, '_disk', lambda: {'host_free_bytes': 0, 'linux_free_bytes': 10**12})
        preview = await storage.service.preview(model.id)
        assert not preview['eligible'] and '空间不足' in preview['errors'][0]
    def denied(paths):
        raise ValueError('当前发行版所有权不同')
    monkeypatch.setattr(installers, '_verify_wsl_owner', denied)
    preview = await storage.service.preview(model.id)
    assert not preview['eligible'] and '所有权不同' in preview['errors'][0]


async def test_unknown_format_never_scans_arbitrary_model_tree(storage, monkeypatch):
    model, profile, runtime = register(storage)
    storage.store.put('models', {**model.model_dump(), 'format': 'custom'})
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported model tree was inspected')
    monkeypatch.setattr(model_storage, 'inspect_asset', forbidden)
    preview = await storage.service.preview(model.id)
    assert not preview['eligible'] and '已识别' in preview['errors'][0]
    assert not storage.base.exists()


async def test_source_inventory_and_late_reference_changes_preserve_source(storage):
    model, profile, runtime = register(storage, managed=True)
    preview = await storage.service.preview(model.id)
    async def add_file(event):
        if event['phase'] == 'copied':
            (Path(model.path) / 'new_resource.json').write_text('{}')
    with pytest.raises(ValueError, match='清单发生变化'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'job', add_file, asyncio.Event())
    assert not Path(storage.translate(preview['target_directory'])).exists()
    async def add_reference(event):
        if event['phase'] == 'copied':
            storage.store.put('models', {**model.model_dump(), 'id': 'late-other'})
    _, result = await migrate(storage, model, delete_source=True, progress=add_reference)
    assert result['source_deleted'] is False
    assert any('其他模型' in value for value in result['warnings'])
    assert Path(model.path).exists()
    assert storage.store.get('profiles', profile.id)['model_id'] == result['model_id']


async def test_ninfer_prepared_artifact_is_copied_and_rekeyed_without_changing_original(storage, monkeypatch):
    model, old_profile, old_runtime = register(storage, managed=True)
    runtime = old_runtime.model_copy(update={'backend': 'ninfer'})
    profile = old_profile.model_copy(update={'backend': 'ninfer', 'parameters': {'conversion_recipe': 'qwen3_8_27b'}})
    storage.store.put('runtimes', runtime.model_dump())
    storage.store.put('profiles', profile.model_dump())
    original_output = storage.paths.root / 'prepared/ninfer/original'
    artifact, parts, _ = write_ninfer(original_output / 'model.ninfer')
    def identity(self, asset, configuration, installation):
        return dict(model_id=asset.id, target=asset.path, parameters=deepcopy(configuration.parameters), runtime_id=installation.id)
    monkeypatch.setattr(engine_preparation.NativePreparation, '_identity', identity)
    original = dict(id='original', model_id=model.id, runtime_id=runtime.id, output=str(original_output.resolve()),
                    identity=identity(None, model, profile, runtime), artifact=artifact.as_posix())
    storage.store.put('ninfer_preparations', original)
    def inspect(path, installation):
        metadata, _, _ = assets.ninfer_metadata(Path(storage.translate(path)))
        return {'components': {name: {} for name in metadata['ninfer_components']}}
    monkeypatch.setattr(native_engines, 'artifact_info', inspect)
    preview, result = await migrate(storage, model)
    assert preview['prepared_count'] == 1 and preview['prepared_bytes'] == sum(path.stat().st_size for path in parts)
    records = storage.store.all('ninfer_preparations')
    assert len(records) == 2 and storage.store.get('ninfer_preparations', 'original') == original
    new = next(value for value in records if value['id'] != 'original')
    assert new['model_id'] == result['model_id']
    assert new['artifact'].startswith(preview['target_directory'] + '/.prepared/')
    assert new['identity']['target'] == storage.store.get('models', result['model_id'])['path']
    assert all(path.exists() for path in parts)
    preparation = engine_preparation.NativePreparation(storage.paths, storage.store)
    assert preparation._owned(new).is_dir()
    native_model = ModelAsset(**storage.store.get('models', result['model_id']))
    future = preparation._output_record(native_model, runtime, 'future', {})
    output = preparation._create_output(future)
    assert output.is_dir() and future['output'].startswith('/opt/modloader/models/')
    assert future['output'] != str(storage.paths.root / 'prepared/ninfer/future')
    outside = {**future, 'output': '/opt/modloader/another'}
    with pytest.raises(ValueError, match='所有权'):
        preparation._owned(outside)


async def test_retry_cleans_only_previous_matching_unregistered_partial(storage):
    model, profile, runtime = register(storage)
    prior_id = 'a' * 32
    root = storage.base / prior_id
    root.mkdir(parents=True)
    (root / 'partial.bin').write_bytes(b'partial')
    owner = dict(version=1, model_id=prior_id, source_model_id=model.id, job_id='interrupted')
    (root / model_storage.STORAGE_MARKER).write_text(json.dumps(owner))
    storage.store.put('jobs', dict(id='interrupted', kind='model_migration', state='interrupted',
                                  request={'model_id': model.id, 'plan_id': prior_id},
                                  migration_root=model_storage.STORAGE_BASE + '/' + prior_id))
    preview, result = await migrate(storage, model)
    assert any('中断任务' in value for value in preview['warnings'])
    assert preview['disk']['partial_reclaimable_bytes'] > 0
    assert not root.exists()
    assert Path(storage.translate(preview['target_directory'])).is_dir()


async def test_unfinished_download_writers_block_preview_and_recheck(storage):
    model, profile, runtime = register(storage)
    job = dict(id='writer', kind='download', state='paused', request={'destination': model.path})
    storage.store.put('jobs', job)
    preview = await storage.service.preview(model.id)
    assert not preview['eligible'] and '未完成的下载' in preview['errors'][0]
    storage.store.remove('jobs', job['id'])
    preview = await storage.service.preview(model.id)
    storage.store.put('jobs', job)
    with pytest.raises(ValueError, match='未完成的下载'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'job', storage.progress, asyncio.Event())
    assert not storage.base.exists()
    storage.store.remove('jobs', job['id'])
    storage.service.active[model.id] = 'active-migration'
    with pytest.raises(ValueError, match='正在迁移'):
        storage.service.assert_download_available(str(Path(model.path) / 'child'))
    storage.service.assert_download_available(str(storage.paths.models / 'independent'))


async def test_explicit_in_root_resources_move_but_external_resources_stay(storage):
    model, profile, runtime = register(storage, fmt='gguf', managed=True)
    root = Path(model.path).parent
    resource = root / 'vision' / 'mmproj.gguf'
    resource.parent.mkdir()
    resource.write_bytes(Path(model.path).read_bytes())
    external = storage.paths.root / 'external-draft'
    external.mkdir()
    profile.parameters.update(mmproj=str(resource), draft_model=str(external))
    storage.store.put('profiles', profile.model_dump())
    preview, result = await migrate(storage, model)
    switched = storage.store.get('profiles', profile.id)
    assert switched['parameters']['mmproj'].startswith('/opt/modloader/models/')
    assert Path(storage.translate(switched['parameters']['mmproj'])).read_bytes() == resource.read_bytes()
    assert switched['parameters']['draft_model'] == str(external)
    assert preview['external_resources'] == [dict(profile_id=profile.id, field='draft_model', path=str(external))]
    assert any('原路径' in value for value in preview['warnings'])


async def test_shared_adjacent_resources_block_source_delete_and_custom_literals_are_preserved(storage):
    model, profile, runtime = register(storage, fmt='gguf', managed=True)
    sibling = Path(model.path).parent / 'another.gguf'
    sibling.write_bytes(Path(model.path).read_bytes())
    other = assets.inspect_asset(str(sibling))
    storage.store.put('models', other.model_dump())
    preview = await storage.service.preview(model.id)
    assert not preview['delete_source_allowed'] and any('其他模型' in value for value in preview['delete_source_reasons'])
    custom = runtime.model_copy(update={'backend': 'custom', 'metadata': {**runtime.metadata, 'custom': {'args': ['--model', str(Path(model.path).parent)], 'cwd': '/opt/fixture'}}})
    profile.backend = 'custom'
    storage.store.put('runtimes', custom.model_dump())
    storage.store.put('profiles', profile.model_dump())
    preview = await storage.service.preview(model.id)
    assert not preview['profiles'] and any('固定引用' in value['reason'] for value in preview['preserved_profiles'])


async def test_copy_launch_cwd_never_pins_disposable_work_directory(storage, monkeypatch):
    model, profile, runtime = register(storage)
    command = installers._command
    calls = []
    async def launch(argv, progress, cancel, **kwargs):
        cwd = Path(kwargs['cwd']).resolve()
        assert cwd == Path(storage.paths.work.anchor).resolve()
        assert not cwd.is_relative_to(storage.paths.work)
        calls.append(cwd)
        return await command(argv, progress, cancel, **kwargs)
    monkeypatch.setattr(installers, '_command', launch)
    _, result = await migrate(storage, model)
    assert len(calls) == 1 and storage.store.get('models', result['model_id'])
    assert not list(storage.paths.work.glob('model-migration-*'))


async def test_custom_env_source_reference_keeps_selected_profile_and_shared_runtime(storage, monkeypatch):
    model, profile, runtime = register(storage, managed=True)
    source = Path(model.path).resolve()
    mapping = model_storage.windows_to_wsl
    monkeypatch.setattr(model_storage, 'windows_to_wsl', lambda value, distribution:
                        '/mnt/d/source' if Path(value).resolve() == source else mapping(value, distribution))
    custom = runtime.model_copy(update={'backend': 'custom', 'metadata': {**runtime.metadata, 'custom': {
        'args': ['--model', '{model_path}'], 'cwd': '/opt/fixture',
        'env': {'TOKENIZER_PATH': '/mnt/d/source/tokenizer.json', 'UNRELATED': 'kept'}}}})
    profile.backend = 'custom'
    storage.store.put('runtimes', custom.model_dump())
    storage.store.put('profiles', profile.model_dump())
    preview = await storage.service.preview(model.id)
    assert preview['eligible'] and not preview['profiles']
    assert any('环境变量固定引用' in value['reason'] for value in preview['preserved_profiles'])
    assert not preview['delete_source_allowed']
    assert storage.store.get('runtimes', runtime.id) == custom.model_dump()
    assert storage.store.get('profiles', profile.id) == profile.model_dump()


async def test_other_custom_env_source_reference_blocks_delete_without_rewriting_env(storage, monkeypatch):
    model, profile, runtime = register(storage, managed=True)
    source = Path(model.path).resolve()
    mapping = model_storage.windows_to_wsl
    monkeypatch.setattr(model_storage, 'windows_to_wsl', lambda value, distribution:
                        '/mnt/d/source' if Path(value).resolve() == source else mapping(value, distribution))
    other = ModelAsset(id='other', name='Other', format='custom', path=str(storage.paths.models / 'unrelated'))
    custom = runtime.model_copy(update={'id': 'shared-env-runtime', 'backend': 'custom', 'metadata': {
        **runtime.metadata, 'custom': {'args': ['--model', '{model_path}'], 'cwd': '/opt/fixture',
                                     'env': {'TOKENIZER_PATH': '/mnt/d/source/tokenizer.json'}}}})
    other_profile = profile.model_copy(update={'id': 'other-profile', 'model_id': other.id,
                                               'backend': 'custom', 'runtime_id': custom.id, 'parameters': {}})
    storage.store.put('models', other.model_dump())
    storage.store.put('runtimes', custom.model_dump())
    storage.store.put('profiles', other_profile.model_dump())
    preview = await storage.service.preview(model.id)
    assert preview['eligible'] and preview['profiles'][0]['id'] == profile.id
    assert not preview['delete_source_allowed']
    assert any('环境变量引用' in value for value in preview['delete_source_reasons'])
    with pytest.raises(ValueError, match='环境变量引用'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id'], delete_source=True),
                                  'job', storage.progress, asyncio.Event())
    assert storage.store.get('runtimes', custom.id) == custom.model_dump()
    assert storage.store.get('profiles', other_profile.id) == other_profile.model_dump()
    assert Path(model.path).exists() and not storage.base.exists()


async def test_future_native_conversion_ensure_uses_native_output(storage, monkeypatch):
    model, profile, runtime = register(storage)
    _, migrated = await migrate(storage, model)
    model = ModelAsset(**storage.store.get('models', migrated['model_id']))
    runtime = runtime.model_copy(update={'backend': 'ninfer', 'metadata': {**runtime.metadata, 'source_root': '/opt/fixture'}})
    profile = profile.model_copy(update={'model_id': model.id, 'backend': 'ninfer', 'parameters': {'conversion_recipe': 'qwen3_8_27b'}})
    preparation = engine_preparation.NativePreparation(storage.paths, storage.store)
    monkeypatch.setattr(preparation, 'preflight', lambda *args: {'status': 'needs_preparation'})
    monkeypatch.setattr(preparation, '_identity', lambda *args: {'target': model.path})
    monkeypatch.setattr(native_engines, 'conversion_resources', lambda *args: {'chat_template.jinja': {'path': '/fixture/chat_template.jinja'}})
    monkeypatch.setattr(native_engines, 'model_target_path', lambda asset, runtime: asset.path)
    monkeypatch.setattr(engine_preparation, 'model_target_path', lambda asset, runtime: asset.path)
    def summary(path, runtime):
        metadata, _, _ = assets.ninfer_metadata(Path(storage.translate(path)))
        return {'components': {name: {} for name in metadata['ninfer_components']}}
    monkeypatch.setattr(native_engines, 'artifact_info', summary)
    seen = []
    async def convert(argv, progress, cancel, **kwargs):
        seen.append(argv)
        output = argv[argv.index('--out') + 1]
        write_ninfer(Path(storage.translate(output)))
    monkeypatch.setattr(installers, '_command', convert)
    monkeypatch.setattr(engine_preparation, 'prepare_launch', lambda spec, context: spec)
    resolved = await preparation.ensure(model, profile, runtime, storage.progress)
    assert len(seen) == 1
    artifact = resolved.parameters['_prepared_artifact']
    assert artifact.startswith(model.source['managed_root'] + '/.prepared/')
    assert Path(storage.translate(artifact)).is_file()
    record = storage.store.all('ninfer_preparations')[0]
    assert record['artifact'] == artifact and record['location']['platform'] == 'wsl2'
    assert not (storage.paths.root / 'prepared/ninfer').exists()
    inspected = assets.inspect_asset(model.path, location=model.location)
    assert not any('/.prepared/' in value for value in inspected.files)




def register_wsl(machine, *, fmt='safetensors'):
    model, profile, runtime = register(machine, fmt=fmt, source_root=machine.base / 'direct-import')
    model.path = machine.native(model.path)
    model.files = [machine.native(path) for path in model.files]
    model.location = ModelLocation(platform='wsl2', distribution='ModLoader')
    profile.parameters = {key: machine.native(value) for key, value in profile.parameters.items()}
    machine.store.put('models', model.model_dump())
    machine.store.put('profiles', profile.model_dump())
    return model, profile, runtime


@pytest.mark.parametrize('fmt', ['safetensors', 'gguf', 'ninfer'])
@pytest.mark.parametrize('delete_source', [False, True])
async def test_direct_wsl_import_moves_to_selected_windows_parent(storage, fmt, delete_source):
    model, profile, runtime = register_wsl(storage, fmt=fmt)
    source = Path(storage.translate(model.path))
    selected_files = {Path(storage.translate(path)): Path(storage.translate(path)).read_bytes() for path in model.files}
    sibling = (source.parent if source.is_file() else source.parent) / 'unselected.bin'
    sibling.write_bytes(b'unrelated')
    destination = storage.paths.root / 'chosen-parent'
    destination.mkdir()
    preview = await storage.service.preview(model.id, {'target_platform': 'windows', 'destination': str(destination)})
    assert preview['eligible'], preview
    assert preview['delete_source_allowed'], preview
    assert Path(preview['target_directory']).parent == destination
    assert Path(preview['target_directory']).name == model.name
    assert preview['source_platform'] == 'wsl2' and preview['target_platform'] == 'windows'
    result = await storage.service.run(dict(model_id=model.id, plan_id=preview['id'], delete_source=delete_source),
                                       'windows-job', storage.progress, asyncio.Event())
    target = storage.store.get('models', result['model_id'])
    assert platforms.location_of(target).platform == 'windows'
    assert target['path'] == preview['target_path'] and Path(target['path']).exists()
    assert storage.store.get('profiles', profile.id)['model_id'] == target['id']
    assert storage.store.get('profiles', profile.id)['parameters']['tokenizer'] == (Path(preview['target_path']) if fmt == 'safetensors' else Path(preview['target_path']).parent).joinpath('tokenizer.json').as_posix()
    assert result['source_deleted'] is delete_source
    assert storage.store.get('models', model.id) == (None if delete_source else model.model_dump())
    for path, content in selected_files.items():
        assert path.exists() is not delete_source
        relative = path.relative_to(source if fmt == 'safetensors' else source.parent)
        output = (Path(target['path']) if fmt == 'safetensors' else Path(target['path']).parent) / relative
        assert output.read_bytes() == content
    assert sibling.read_bytes() == b'unrelated'


async def test_readable_names_do_not_overwrite_same_named_directories(storage):
    model, _, _ = register(storage)
    root = storage.base / model.name
    root.mkdir(parents=True)
    (root / 'keep').write_text('untouched')
    preview, result = await migrate(storage, model)
    assert preview['target_directory'] == model_storage.STORAGE_BASE + '/' + model.name + ' (2)'
    assert (root / 'keep').read_text() == 'untouched'
    migrated = ModelAsset(**storage.store.get('models', result['model_id']))
    windows_parent = storage.paths.root / 'destination'
    windows_parent.mkdir()
    existing = windows_parent / model.name
    existing.mkdir()
    (existing / 'keep').write_text('untouched')
    back = await storage.service.preview(migrated.id, {'target_platform': 'windows', 'destination': str(windows_parent)})
    assert back['eligible'], back
    assert Path(back['target_directory']).name == model.name + ' (2)'
    assert (existing / 'keep').read_text() == 'untouched'


async def test_wsl_windows_cancel_conflict_and_failure_preserve_original(storage):
    model, profile, _ = register_wsl(storage, fmt='ninfer')
    destination = storage.paths.root / 'destination'
    destination.mkdir()
    options = {'target_platform': 'windows', 'destination': str(destination)}
    cancel = asyncio.Event()
    async def cancelling(event):
        if event['phase'] == 'copying':
            cancel.set()
    preview = await storage.service.preview(model.id, options)
    with pytest.raises(asyncio.CancelledError):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'cancel-job', cancelling, cancel)
    assert not Path(preview['target_directory']).exists()
    assert Path(storage.translate(model.path)).exists()
    assert storage.store.get('profiles', profile.id) == profile.model_dump()
    preview = await storage.service.preview(model.id, options)
    Path(preview['target_directory']).mkdir()
    (Path(preview['target_directory']) / 'keep').write_text('collision')
    with pytest.raises((ValueError, OSError)):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'conflict-job', storage.progress, asyncio.Event())
    assert (Path(preview['target_directory']) / 'keep').read_text() == 'collision'
    preview = await storage.service.preview(model.id, options)
    async def corrupt(event):
        if event['phase'] == 'copied':
            Path(preview['target_path']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='NInfer'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'broken-job', corrupt, asyncio.Event())
    assert not Path(preview['target_directory']).exists()
    assert storage.store.get('profiles', profile.id) == profile.model_dump()


async def test_wsl_shared_reference_blocks_source_delete(storage):
    model, _, _ = register_wsl(storage, fmt='gguf')
    storage.store.put('models', {**model.model_dump(), 'id': 'shared'})
    destination = storage.paths.root / 'destination'
    destination.mkdir()
    preview = await storage.service.preview(model.id, {'target_platform': 'windows', 'destination': str(destination)})
    assert preview['eligible'] and not preview['delete_source_allowed'], preview
    assert any('其他模型' in reason for reason in preview['delete_source_reasons'])
    with pytest.raises(ValueError, match='其他模型'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id'], delete_source=True), 'job', storage.progress, asyncio.Event())
    assert not Path(preview['target_directory']).exists()


async def test_wsl_windows_maps_windows_and_native_profiles_preserving_other_distro(storage):
    model, native_profile, runtime = register_wsl(storage)
    windows = RuntimeInstallation(id='windows', platform='windows', backend='custom', name='Windows', path='C:/fixture.exe')
    win_profile = native_profile.model_copy(update={'id': 'windows-profile', 'runtime_id': windows.id, 'backend': 'custom'})
    foreign = runtime.model_copy(update={'id': 'foreign', 'metadata': {'distribution': 'Ubuntu'}})
    foreign_profile = native_profile.model_copy(update={'id': 'foreign-profile', 'runtime_id': foreign.id})
    for row in (windows, foreign): storage.store.put('runtimes', row.model_dump())
    for row in (win_profile, foreign_profile): storage.store.put('profiles', row.model_dump())
    destination = storage.paths.root / 'destination'; destination.mkdir()
    preview = await storage.service.preview(model.id, {'target_platform': 'windows', 'destination': str(destination)})
    assert preview['eligible'], preview
    assert {p['id'] for p in preview['profiles']} == {native_profile.id, win_profile.id}
    assert [p['id'] for p in preview['preserved_profiles']] == [foreign_profile.id]
    assert not preview['delete_source_allowed']
    result = await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'job', storage.progress, asyncio.Event())
    expected = Path(preview['target_path']) / 'tokenizer.json'
    assert storage.store.get('profiles', win_profile.id)['parameters']['tokenizer'] == str(expected)
    assert storage.store.get('profiles', native_profile.id)['parameters']['tokenizer'] == expected.as_posix()
    assert storage.store.get('profiles', foreign_profile.id) == foreign_profile.model_dump()
    assert storage.store.get('models', result['model_id'])


async def test_native_prepared_copy_to_windows_keeps_identity_and_owned_record(storage, monkeypatch):
    model, old_profile, old_runtime = register(storage)
    runtime = old_runtime.model_copy(update={'backend': 'ninfer'})
    profile = old_profile.model_copy(update={'backend': 'ninfer', 'parameters': {'conversion_recipe': 'qwen3_8_27b'}})
    storage.store.put('runtimes', runtime.model_dump()); storage.store.put('profiles', profile.model_dump())
    def identity(self, asset, configuration, installation):
        return dict(model_id=asset.id, target=asset.path, parameters=deepcopy(configuration.parameters), runtime_id=installation.id)
    monkeypatch.setattr(engine_preparation.NativePreparation, '_identity', identity)
    output = storage.paths.root / 'prepared/ninfer/original'
    artifact, _, _ = write_ninfer(output / 'model.ninfer')
    original = dict(id='original', model_id=model.id, runtime_id=runtime.id, output=str(output), artifact=str(artifact),
                    identity=identity(None, model, profile, runtime))
    storage.store.put('ninfer_preparations', original)
    monkeypatch.setattr(native_engines, 'artifact_info', lambda path, runtime: {'components': list(assets.ninfer_metadata(Path(storage.translate(path)))[0]['ninfer_components'])})
    _, native = await migrate(storage, model)
    native_model = ModelAsset(**storage.store.get('models', native['model_id']))
    destination = storage.paths.root / 'destination'; destination.mkdir()
    preview = await storage.service.preview(native_model.id, {'target_platform': 'windows', 'destination': str(destination)})
    assert preview['eligible'] and preview['prepared_count'] == 1, preview
    result = await storage.service.run(dict(model_id=native_model.id, plan_id=preview['id'], delete_source=True),
                                       'to-windows', storage.progress, asyncio.Event())
    assert result['source_deleted'], result
    record = next(row for row in storage.store.all('ninfer_preparations') if row['model_id'] == result['model_id'])
    assert record['location'] == {'platform': 'windows'}
    assert Path(record['output']).is_dir()
    assert record['identity']['target'] == preview['target_path']
    preparation = engine_preparation.NativePreparation(storage.paths, storage.store)
    assert preparation._owned(record) == Path(record['output'])
    assert (preparation._owned(record) / 'model.ninfer').is_file()
    assert storage.store.get('ninfer_preparations', 'original') == original


async def test_windows_copy_snapshot_conflict_cleans_target_without_switch(storage):
    model, profile, _ = register_wsl(storage)
    destination = storage.paths.root / 'destination'; destination.mkdir()
    preview = await storage.service.preview(model.id, {'target_platform': 'windows', 'destination': str(destination)})
    async def mutate(event):
        if event['phase'] == 'copied':
            storage.store.put('profiles', profile.model_copy(update={'name': 'user edit'}).model_dump())
    with pytest.raises(ValueError, match='已变化'):
        await storage.service.run(dict(model_id=model.id, plan_id=preview['id']), 'job', mutate, asyncio.Event())
    assert not Path(preview['target_directory']).exists()
    assert storage.store.get('profiles', profile.id)['name'] == 'user edit'
    assert storage.store.get('models', model.id) == model.model_dump()
