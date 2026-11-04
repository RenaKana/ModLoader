"""No-engine NInfer v3 container fixtures, based on the pinned framing contract."""
import json
from pathlib import Path
import struct

import httpx
import pytest

from modloader import assets
from modloader.app import Controller, build_app
from modloader.platforms import model_identity


HEADER = struct.Struct('<8sQ16s')
MAGIC, PART_MAGIC, IDENTIFIER = b'NINFER\0\3', b'NINPRT\0\3', bytes(range(16))


def write_ninfer(path, *, lengths=(4, 4), change=None, raw_json=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = {
        'components': {
            'text': {'config': {'model_type': 'qwen3_5_text', 'architectures': ['Qwen3_5ForCausalLM']}},
            'mtp': {'config': {'architectures': ['Qwen3_5MTP']}, 'target': 'text'},
        },
        'objects': [{'id': 'w', 'kind': 'tensor', 'shape': [sum(lengths) // 2], 'format': 'bf16',
                     'layout': 'contiguous_le_v1', 'offset': 0, 'bytes': sum(lengths)}],
        'bindings': {'text/final_norm': {'object': 'w'}},
        'uses': [],
        'files': [{'path': None if index == 0 else path.name + f'.part-{index:04d}', 'payload_bytes': size}
                  for index, size in enumerate(lengths)],
        'metadata': {'name': 'Tiny structural fixture'},
        'provenance': {'fixture': True},
    }
    if change:
        change(directory)
    raw = raw_json if raw_json is not None else json.dumps(directory, separators=(',', ':')).encode()
    offset = (HEADER.size + len(raw) + 4095) // 4096 * 4096
    path.write_bytes(HEADER.pack(MAGIC, len(raw), IDENTIFIER) + raw + b'\0' * (offset - HEADER.size - len(raw)) + b'\0' * lengths[0])
    parts = [path]
    for index, length in enumerate(lengths[1:], 1):
        part = path.parent / (path.name + f'.part-{index:04d}')
        part.write_bytes(HEADER.pack(PART_MAGIC, index, IDENTIFIER) + b'\0' * (4096 - HEADER.size) + b'\0' * length)
        parts.append(part)
    return path, parts, directory


def test_native_file_and_single_artifact_directory_import_measure_all_continuations(tmp_path):
    path, files, _ = write_ninfer(tmp_path / 'artifact' / 'model.NINFER')
    model = assets.inspect_asset(str(path))
    assert model.format == 'ninfer' and model.path == str(path)
    assert model.files == list(map(str, files))
    assert model.size_bytes == sum(part.stat().st_size for part in files)
    assert model.architecture == 'qwen3_5_text' and model.quantization == 'bf16'
    assert model.metadata['ninfer_version'] == 3 and model.metadata['ninfer_artifact_id'] == IDENTIFIER.hex()
    assert model.metadata['ninfer_validation'] == 'container'
    assert model.warnings == []
    assert model.metadata['ninfer_components'] == ['text', 'mtp']
    source = {'kind': 'download', 'managed_root': str(path.parent)}
    same = assets.inspect_asset(str(path.parent), 'Selected name', source, True)
    assert same.name == 'Selected name' and same.managed and same.source == source
    assert model_identity(same) == model_identity(model)


@pytest.mark.parametrize('case', ['header', 'magic', 'version', 'zero_json', 'oversize_json', 'invalid_json', 'duplicate_json', 'nonfinite_json'])
def test_invalid_native_header_and_directory_never_pass_on_suffix(tmp_path, case):
    path, _, _ = write_ninfer(tmp_path / 'bad.ninfer', lengths=(8,))
    content = path.read_bytes()
    if case == 'header':
        content = b'NINFER'
    elif case == 'magic':
        content = b'NOTMDL\0\3' + content[8:]
    elif case == 'version':
        content = b'NINFER\0\2' + content[8:]
    elif case == 'zero_json':
        content = HEADER.pack(MAGIC, 0, IDENTIFIER) + content[HEADER.size:]
    elif case == 'oversize_json':
        content = HEADER.pack(MAGIC, len(content), IDENTIFIER) + content[HEADER.size:]
    else:
        raw = b'{' if case == 'invalid_json' else b'{"files":[],"files":[]}' if case == 'duplicate_json' else b'{"value":NaN}'
        write_ninfer(path, lengths=(8,), raw_json=raw)
        content = path.read_bytes()
    path.write_bytes(content)
    with pytest.raises(ValueError, match='NInfer'):
        assets.inspect_asset(str(path))


@pytest.mark.parametrize('case', ['missing', 'truncated', 'long', 'identity', 'index', 'magic', 'traversal', 'duplicate', 'entry_path', 'negative_payload'])
def test_invalid_continuation_inventory_and_identity_are_rejected(tmp_path, case):
    path, files, _ = write_ninfer(tmp_path / 'bad.ninfer')
    if case in {'traversal', 'duplicate', 'entry_path', 'negative_payload'}:
        def change(directory):
            if case == 'traversal': directory['files'][1]['path'] = '../outside.ninfer'
            if case == 'duplicate': directory['files'][1]['path'] = path.name
            if case == 'entry_path': directory['files'][0]['path'] = path.name
            if case == 'negative_payload': directory['files'][1]['payload_bytes'] = -1
        write_ninfer(path, change=change)
    elif case == 'missing':
        files[1].unlink()
    elif case == 'truncated':
        files[1].write_bytes(files[1].read_bytes()[:-1])
    elif case == 'long':
        files[1].write_bytes(files[1].read_bytes() + b'x')
    else:
        header = (PART_MAGIC, 1, IDENTIFIER)
        if case == 'identity': header = (PART_MAGIC, 1, b'0' * 16)
        if case == 'index': header = (PART_MAGIC, 2, IDENTIFIER)
        if case == 'magic': header = (MAGIC, 1, IDENTIFIER)
        files[1].write_bytes(HEADER.pack(*header) + files[1].read_bytes()[HEADER.size:])
    with pytest.raises(ValueError, match='NInfer'):
        assets.inspect_asset(str(path))


@pytest.mark.parametrize('change', [
    lambda d: d.pop('objects'),
    lambda d: d.update(components={'vision': {'config': {}}}),
    lambda d: d.update(objects=[]),
    lambda d: d['objects'][0].update(offset=9),
    lambda d: d['objects'][0].update(bytes=0),
    lambda d: d['objects'][0].update(shape=[-1]),
    lambda d: d.update(bindings=[]),
])
def test_invalid_container_records_are_rejected(tmp_path, change):
    path, _, _ = write_ninfer(tmp_path / 'bad.ninfer', change=change)
    with pytest.raises(ValueError, match='NInfer'):
        assets.inspect_asset(str(path))


def test_ninfer_scanner_finds_nested_artifacts_and_does_not_treat_mtp_named_entry_as_gguf_auxiliary(tmp_path):
    from test_asset_scan import write_gguf, write_hf
    first, parts, _ = write_ninfer(tmp_path / 'one/two/mtp-model.ninfer')
    second, _, _ = write_ninfer(tmp_path / 'HF/exports/native.ninfer', lengths=(8,))
    gguf = write_gguf(tmp_path / 'language.gguf')
    hf = write_hf(tmp_path / 'HF')
    bad = tmp_path / 'bad.ninfer'
    bad.write_bytes(b'not an artifact')
    (tmp_path / 'unknown.pack').write_bytes(b'unknown')
    result = assets.scan_model_folder(str(tmp_path))
    assert {model.path for model in result['models']} == {str(first), str(second), str(gguf), str(hf)}
    assert result['errors'] == [{'path': str(bad), 'error': 'NInfer 入口文件头截断'}]
    assert result['skipped'] == []
    assert all(model.metadata['import_root'] == str(tmp_path) for model in result['models'])
    assert len([model for model in result['models'] if model.format == 'ninfer']) == 2


def test_ninfer_directory_with_multiple_entries_is_explicitly_ambiguous(tmp_path):
    write_ninfer(tmp_path / 'one.ninfer', lengths=(8,))
    write_ninfer(tmp_path / 'two.ninfer', lengths=(8,))
    with pytest.raises(ValueError, match='直接选择'):
        assets.inspect_asset(str(tmp_path))


def test_directory_and_scan_distinguish_entry_from_arbitrarily_named_continuation(tmp_path):
    path, files, _ = write_ninfer(tmp_path / 'mmproj-native.ninfer',
                                  change=lambda d: d['files'][1].update(path='00-continuation.ninfer'))
    renamed = tmp_path / '00-continuation.ninfer'
    files[1].rename(renamed)
    model = assets.inspect_asset(str(tmp_path))
    assert model.path == str(path) and model.files == [str(path), str(renamed)]
    scanned = assets.scan_model_folder(str(tmp_path))
    assert [asset.path for asset in scanned['models']] == [str(path)] and not scanned['errors']
    assert scanned['skipped'] == [{'path': str(renamed), 'reason': 'NInfer 续卷需随入口工件导入，不能独立导入'}]
    with pytest.raises(ValueError, match='续卷'):
        assets.inspect_asset(str(renamed))


def test_native_only_export_uses_embedded_config_and_does_not_swallow_nested_hf(tmp_path):
    from test_asset_scan import write_hf
    path, _, _ = write_ninfer(tmp_path / 'export.ninfer', lengths=(8,))
    (tmp_path / 'config.json').write_text('{"model_type":"original_hf_metadata"}')
    child = write_hf(tmp_path / 'separate-HF')
    assert assets.inspect_asset(str(tmp_path)).path == str(path)
    result = assets.scan_model_folder(str(tmp_path))
    assert {model.path for model in result['models']} == {str(path), str(child)}
    assert result['errors'] == []


def test_container_check_reads_only_entry_json_and_continuation_headers(tmp_path, monkeypatch):
    path, files, _ = write_ninfer(tmp_path / 'model.ninfer')
    original = Path.open
    reads = []
    class MetadataReader:
        def __init__(self, selected, stream):
            self.selected, self.stream = selected, stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def read(self, length=-1):
            reads.append((self.selected, self.stream.tell(), length))
            assert length > 0
            return self.stream.read(length)
    def open_file(selected, *args, **kwargs):
        stream = original(selected, *args, **kwargs)
        return MetadataReader(selected, stream) if selected in files else stream
    monkeypatch.setattr(Path, 'open', open_file)
    model = assets.inspect_asset(str(path))
    assert model.format == 'ninfer' and len(reads) == 3
    assert reads[0] == (path, 0, HEADER.size)
    assert reads[1][0] == path and reads[1][1] == HEADER.size
    assert reads[2] == (files[1], 0, HEADER.size)


async def test_normal_import_and_folder_apis_register_ninfer_once_without_runtime_or_user_database(tmp_path, monkeypatch):
    path, _, _ = write_ninfer(tmp_path / 'models/native.ninfer')
    from modloader.backends import native_engines
    def forbid(*args, **kwargs): raise AssertionError('Model registration must not run the engine inspector')
    monkeypatch.setattr(native_engines, 'artifact_info', forbid)
    controller = Controller(tmp_path / 'isolated-data', 'fixture-secret')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as client:
        try:
            imported = await client.post('/api/models/import', json={'path': str(path), 'name': 'Native model'})
            assert imported.status_code == 200, imported.text
            original = imported.json()
            assert original['format'] == 'ninfer' and original['size_bytes'] > 0 and not original['managed']
            again = await client.post('/api/models/import', json={'path': str(path.parent), 'name': 'Do not overwrite'})
            assert again.json() == original
            folder = await client.post('/api/models/import-folder', json={'path': str(path.parent)})
            assert folder.status_code == 200 and folder.json()['imported'] == [] and folder.json()['existing'] == [original]
            assert len(controller.store.all('models')) == 1
            unknown = path.parent / 'unknown.pack'
            unknown.write_bytes(b'unknown format fixture')
            assert (await client.post('/api/models/import', json={'path': str(unknown)})).status_code == 400
            assert (await client.post('/api/models/import-custom', json={'path': str(unknown)})).json()['format'] == 'custom'
            invalid = path.parent / 'bad.ninfer'
            invalid.write_bytes(b'not native')
            assert (await client.post('/api/models/import', json={'path': str(invalid)})).status_code == 400
            assert len(controller.store.all('models')) == 2
        finally:
            await controller.close()
