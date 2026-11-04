import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

import modloader.assets as assets
from modloader.assets import inspect_asset, scan_model_folder


def write_gguf(path, metadata=None, tensor_count=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    content = b'GGUF' + struct.pack('<IQQ', 3, tensor_count, len(metadata))
    for key, value in metadata.items():
        key_bytes = key.encode('utf-8')
        content += struct.pack('<Q', len(key_bytes)) + key_bytes
        if isinstance(value, str):
            value_bytes = value.encode('utf-8')
            content += struct.pack('<IQ', 8, len(value_bytes)) + value_bytes
        else:
            content += struct.pack('<II', 4, value)
    path.write_bytes(content)
    return path


def write_hf(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(json.dumps({'model_type': 'qwen3'}), encoding='utf-8')
    (path / 'model-00001-of-00002.safetensors').write_bytes(b'first-weight-fixture')
    (path / 'model-00002-of-00002.safetensors').write_bytes(b'second-weight-fixture')
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'first': 'model-00001-of-00002.safetensors',
        'second': 'model-00002-of-00002.safetensors',
    }}), encoding='utf-8')
    (path / 'tokenizer.json').write_text('{}', encoding='utf-8')
    return path


def directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != 'nt':
            pytest.skip('Directory symlink creation is unavailable')
    script = "New-Item -ItemType Junction -Path '{}' -Target '{}' -ErrorAction Stop | Out-Null".format(
        str(link).replace("'", "''"), str(target).replace("'", "''"),
    )
    process = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                             capture_output=True, text=True)
    if process.returncode:
        pytest.skip('Directory junction creation is unavailable: ' + process.stderr.strip())


def test_scan_three_and_five_levels_and_distinct_quantizations(tmp_path):
    three = write_gguf(tmp_path / 'one' / 'two' / 'three' / 'model-Q4_K_M.gguf', {'general.file_type': 15})
    five = write_gguf(tmp_path / 'one' / 'two' / 'three' / 'four' / 'five' / 'model-Q8_0.GGUF', {'general.file_type': 7})
    (tmp_path / 'notes.txt').write_text('ordinary non-model file', encoding='utf-8')
    for ignored in ('.git', '.cache'):
        (tmp_path / ignored).mkdir()
        (tmp_path / ignored / 'broken.gguf').write_bytes(b'not a model')

    result = scan_model_folder(str(tmp_path))

    assert {model.path for model in result['models']} == {str(three), str(five)}
    assert {model.quantization for model in result['models']} == {'Q4_K_M', 'Q8_0'}
    assert result['errors'] == []
    assert result['skipped'] == []
    assert all(not model.managed for model in result['models'])


def test_scan_shards_once_and_inspect_uppercase_from_second_shard(tmp_path, monkeypatch):
    first = write_gguf(tmp_path / 'model-00001-of-00003.GGUF', {'general.name': 'first-shard'})
    second = write_gguf(tmp_path / 'model-00002-of-00003.GGUF', {'general.name': 'second-shard'})
    third = write_gguf(tmp_path / 'model-00003-of-00003.GGUF')
    asset = inspect_asset(str(second))
    assert asset.path == str(first)
    assert asset.files == [str(first), str(second), str(third)]
    assert asset.metadata['general.name'] == 'first-shard'
    inspected = []
    original = assets.inspect_asset

    def inspect(path, *args, **kwargs):
        inspected.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(assets, 'inspect_asset', inspect)
    result = scan_model_folder(str(tmp_path))

    assert len(result['models']) == 1
    assert result['models'][0].files == [str(first), str(second), str(third)]
    assert len(inspected) == 1
    assert result['errors'] == []


def test_scan_second_shard_file_link_before_primary(tmp_path, monkeypatch):
    first = write_gguf(tmp_path / 'model-00001-of-00002.gguf')
    second = write_gguf(tmp_path / 'model-00002-of-00002.gguf')
    alias = tmp_path / '00-first-entry.gguf'
    try:
        alias.symlink_to(second)
    except OSError:
        pytest.skip('File symlink creation is unavailable')
    inspected = []
    original = assets.inspect_asset

    def inspect(path, *args, **kwargs):
        inspected.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(assets, 'inspect_asset', inspect)
    result = scan_model_folder(str(tmp_path))

    assert inspected == [second]
    assert [model.path for model in result['models']] == [str(first)]
    assert result['errors'] == []


def test_projection_files_are_skipped_by_name_type_and_architecture(tmp_path):
    model = write_gguf(tmp_path / 'language.gguf', {'general.architecture': 'llama'})
    vision = write_gguf(tmp_path / 'multimodal.gguf', {'general.architecture': 'llava'})
    write_gguf(tmp_path / 'mmproj-model.gguf')
    write_gguf(tmp_path / 'projected.gguf', {'general.type': 'projector', 'general.architecture': 'llama'})
    write_gguf(tmp_path / 'image-encoder.GGUF', {'general.architecture': 'clip'})
    (tmp_path / 'image.png').write_bytes(b'ordinary file')

    result = scan_model_folder(str(tmp_path))

    assert {asset.path for asset in result['models']} == {str(model), str(vision)}
    assert len(result['skipped']) == 3
    assert all('视觉投影器' in item['reason'] for item in result['skipped'])
    assert result['errors'] == []


def test_vocabulary_only_and_adapter_gguf_are_auxiliary_files(tmp_path):
    model = write_gguf(tmp_path / 'language.gguf', {'general.architecture': 'llama'})
    vocabulary = write_gguf(tmp_path / 'ggml-vocab-bert-bge.gguf', {
        'general.architecture': 'bert', 'general.name': 'bert-bge',
    }, tensor_count=0)
    adapter = write_gguf(tmp_path / 'qwen-chinese-taste-lora.gguf', {
        'general.type': 'adapter', 'general.architecture': 'qwen35',
    }, tensor_count=512)

    result = scan_model_folder(str(tmp_path))

    assert [asset.path for asset in result['models']] == [str(model)]
    assert {item['path'] for item in result['skipped']} == {str(vocabulary), str(adapter)}
    assert result['errors'] == []
    # Single-resource inspection keeps its existing format acceptance.
    assert inspect_asset(str(vocabulary)).metadata['tensor_count'] == 0
    assert inspect_asset(str(adapter)).metadata['general.type'] == 'adapter'


def test_hf_weights_and_vision_components_are_one_model_with_nested_gguf(tmp_path):
    hf = write_hf(tmp_path / 'full-hf')
    write_hf(hf / 'vision-tower')
    converted = write_gguf(hf / 'exports' / 'deep' / 'converted.gguf')
    for ignored in ('.git', '.cache'):
        (hf / ignored).mkdir()
        (hf / ignored / 'broken.gguf').write_bytes(b'not a model')

    result = scan_model_folder(str(tmp_path))

    assert {model.path for model in result['models']} == {str(hf), str(converted)}
    hf_asset = next(model for model in result['models'] if model.format == 'safetensors')
    assert str(hf / 'vision-tower' / 'config.json') in hf_asset.files
    assert str(converted) in hf_asset.files
    assert not any('.git' in Path(path).parts or '.cache' in Path(path).parts for path in hf_asset.files)
    assert result['errors'] == []
    assert result['skipped'] == []


def test_invalid_models_report_errors_and_continue_to_valid_children(tmp_path):
    bad_gguf = tmp_path / 'bad.gguf'
    bad_gguf.write_bytes(b'not a GGUF')
    bad_json = tmp_path / 'bad-json'
    bad_json.mkdir()
    (bad_json / 'config.json').write_text('{broken', encoding='utf-8')
    list_config = write_hf(tmp_path / 'list-config')
    (list_config / 'config.json').write_text('[]', encoding='utf-8')
    missing_weights = tmp_path / 'missing-weights'
    missing_weights.mkdir()
    (missing_weights / 'config.json').write_text('{}', encoding='utf-8')
    (missing_weights / 'model.safetensors.index.json').write_text('{}', encoding='utf-8')
    child = write_hf(missing_weights / 'complete-child')
    loose_weights = tmp_path / 'loose-weights'
    loose_weights.mkdir()
    (loose_weights / 'first.safetensors').write_bytes(b'fixture')
    (loose_weights / 'second.safetensors').write_bytes(b'fixture')
    nested_gguf = write_gguf(loose_weights / 'child' / 'good.gguf')
    missing_shard = write_hf(tmp_path / 'missing-hf-shard')
    (missing_shard / 'model-00002-of-00002.safetensors').unlink()
    valid = write_gguf(tmp_path / 'valid.gguf')

    result = scan_model_folder(str(tmp_path))

    assert {model.path for model in result['models']} == {str(child), str(nested_gguf), str(valid)}
    assert {item['path'] for item in result['errors']} == {
        str(bad_gguf), str(bad_json), str(list_config), str(missing_weights), str(loose_weights), str(missing_shard),
    }
    assert len(result['errors']) == 6
    assert result['skipped'] == []


def test_missing_first_gguf_shard_is_one_group_error(tmp_path):
    write_gguf(tmp_path / 'broken-00002-of-00003.gguf')
    write_gguf(tmp_path / 'broken-00003-of-00003.gguf')
    valid = write_gguf(tmp_path / 'valid.gguf')

    result = scan_model_folder(str(tmp_path))

    assert [model.path for model in result['models']] == [str(valid)]
    assert len(result['errors']) == 1
    assert 'broken-00001-of-00003.gguf' in result['errors'][0]['error']


def test_inaccessible_directory_and_file_do_not_stop_other_models(tmp_path, monkeypatch):
    denied_dir = tmp_path / 'denied-directory'
    denied_dir.mkdir()
    denied_file = write_gguf(tmp_path / 'denied-file.gguf')
    valid = write_gguf(tmp_path / 'nested' / 'valid.gguf')
    original_iterdir = Path.iterdir
    original_metadata = assets.gguf_metadata

    def iterdir(path):
        if path == denied_dir:
            raise PermissionError('fixture directory access denied')
        return original_iterdir(path)

    def metadata(path):
        if path == denied_file:
            raise PermissionError('fixture model access denied')
        return original_metadata(path)

    monkeypatch.setattr(Path, 'iterdir', iterdir)
    monkeypatch.setattr(assets, 'gguf_metadata', metadata)
    result = scan_model_folder(str(tmp_path))

    assert [model.path for model in result['models']] == [str(valid)]
    assert {item['path'] for item in result['errors']} == {str(denied_dir), str(denied_file)}


def test_directory_links_deduplicate_stop_cycles_and_stay_within_root(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    real = root / 'real'
    model = write_gguf(real / 'deep' / 'valid.gguf')
    outside = tmp_path / 'outside'
    write_gguf(outside / 'external.gguf')
    directory_link(root / 'alias', real)
    directory_link(real / 'loop', root)
    directory_link(root / 'external-link', outside)
    root_alias = tmp_path / 'root-alias'
    directory_link(root_alias, root)

    result = scan_model_folder(str(root_alias))

    assert [asset.path for asset in result['models']] == [str(model)]
    assert result['errors'] == []
    assert len(result['skipped']) == 3
    assert any(item['path'] == str(root / 'external-link') and '根目录之外' in item['reason'] for item in result['skipped'])
    assert sum('已扫描' in item['reason'] for item in result['skipped']) == 2


def test_hf_file_aggregation_also_stops_directory_link_cycles(tmp_path):
    hf = write_hf(tmp_path / 'hf')
    outside = tmp_path / 'outside'
    write_gguf(outside / 'external.gguf')
    directory_link(hf / 'loop', hf)
    directory_link(hf / 'external-link', outside)

    result = scan_model_folder(str(hf))

    assert [asset.path for asset in result['models']] == [str(hf)]
    assert result['errors'] == []
    assert len(result['skipped']) == 2
    assert not any('external.gguf' in path for path in result['models'][0].files)


def test_invalid_scan_root_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_model_folder(str(tmp_path / 'not-present'))
    file = write_gguf(tmp_path / 'model.gguf')
    with pytest.raises(ValueError, match='目录'):
        scan_model_folder(str(file))
    for path in ('', '  '):
        with pytest.raises(ValueError, match='根目录'):
            scan_model_folder(path)


def test_directory_names_preserve_model_version_and_explicit_names(tmp_path):
    hf = write_hf(tmp_path / 'Qwen--Qwen3.5-4B')
    assert inspect_asset(str(hf)).name == hf.name
    assert scan_model_folder(str(tmp_path))['models'][0].name == hf.name
    assert inspect_asset(str(hf), name='SGLang 测试模型').name == 'SGLang 测试模型'
    custom = tmp_path / 'custom-model-v1.2'
    custom.mkdir()
    assert assets.inspect_custom_asset(str(custom)).name == custom.name
    assert assets.inspect_custom_asset(str(custom), name='自定义模型').name == '自定义模型'
    gguf = write_gguf(tmp_path / 'Qwen3.8-Q4.gguf')
    assert inspect_asset(str(gguf)).name == 'Qwen3.8-Q4'
    assert assets.inspect_custom_asset(str(gguf)).name == 'Qwen3.8-Q4'


def test_scan_progress_counts_directories_without_guessing_total(tmp_path):
    write_gguf(tmp_path / 'nested' / 'good.gguf')
    bad = tmp_path / 'other' / 'bad.gguf'
    bad.parent.mkdir()
    bad.write_bytes(b'invalid')
    events = []

    result = scan_model_folder(str(tmp_path), progress=events.append)

    assert len(result['models']) == 1 and len(result['errors']) == 1
    assert all(event['stage'] == 'scanning' and 'total' not in event for event in events)
    counts = [event['completed'] for event in events]
    assert counts == sorted(counts) and counts[-1] == 3
    assert events[-1]['message'] == '已扫描 3 个目录 · 发现 1 个模型'
    assert any('正在检查 good.gguf' in event['message'] for event in events)


def test_scan_progress_can_interrupt_before_model_inspection(tmp_path, monkeypatch):
    write_gguf(tmp_path / 'model.gguf')
    inspected = []
    monkeypatch.setattr(assets, 'inspect_asset', lambda path: inspected.append(path))

    def progress(event):
        if '正在检查 model.gguf' in event['message']:
            raise RuntimeError('scan interrupted')

    with pytest.raises(RuntimeError, match='scan interrupted'):
        scan_model_folder(str(tmp_path), progress=progress)
    assert inspected == []


def test_wsl_scan_progress_preserves_existing_remote_call(tmp_path, monkeypatch):
    from modloader import platforms
    model = inspect_asset(str(write_gguf(tmp_path / 'model.gguf')))
    calls, events = [], []
    result = {'models': [model], 'errors': [], 'skipped': []}
    def inspect(operation, path, location):
        calls.append((operation, path, location.distribution))
        return result
    monkeypatch.setattr(platforms, 'inspect_wsl_asset', inspect)

    assert scan_model_folder('/home/models', {'platform': 'wsl2', 'distribution': 'Fixture'}, events.append) is result
    assert calls == [('scan_model_folder', '/home/models', 'Fixture')]
    assert events == [{'stage': 'scanning', 'message': '正在扫描 WSL 模型目录'},
                      {'stage': 'scanning', 'message': '扫描完成 · 发现 1 个模型'}]
