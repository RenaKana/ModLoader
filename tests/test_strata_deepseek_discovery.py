import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest

from modloader.models import AppPaths, RuntimeInstallation
from modloader.runtime_discovery import discover_runtimes


def write(path: Path, content='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def deepseek(root: Path, layout='flat', version='0.1.49.2'):
    write(root / 'serve/server.py')
    write(root / '.venv/Scripts/python.exe')
    folder = root / ('engine' if layout == 'legacy' else 'engine-dsv41')
    if layout == 'version-folder':
        older = folder / '0.1.49.1-windows-cuda'
        write(older / 'strata-dsv41.exe')
        write(older / 'BUILD.json', json.dumps({'version': '0.1.49.1'}))
        folder /= version + '-windows-cuda'
    write(folder / 'strata-dsv41.exe')
    write(folder / 'BUILD.json', json.dumps({'version': version, 'backend': 'cuda'}))
    return folder


@pytest.fixture
def paths(tmp_path, monkeypatch):
    def forbid_execution(*args, **kwargs):
        raise AssertionError('Runtime discovery must not execute any program')
    monkeypatch.setattr(subprocess, 'run', forbid_execution)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbid_execution)
    return AppPaths(tmp_path / 'data')


@pytest.mark.parametrize('layout', ['flat', 'version-folder', 'legacy'])
def test_deepseek_only_discovery_reads_selected_engine_build(paths, tmp_path, layout):
    root = tmp_path / 'Strata'
    deepseek(root, layout)
    result = discover_runtimes(paths, search_path=str(root))
    assert len(result['runtimes']) == 1
    runtime = result['runtimes'][0]
    assert runtime['backend'] == 'strata' and runtime['path'] == str(root)
    assert runtime['version'] == '0.1.49.2'
    assert runtime['metadata']['strata_family'] == 'deepseek41'
    assert not runtime['registered'] and result['errors'] == []


def test_dual_engine_discovery_keeps_qwen_default_and_registered_deepseek(paths, tmp_path):
    root = tmp_path / 'Strata'
    deepseek(root)
    write(root / 'engine/strata.exe')
    write(root / 'engine/BUILD.json', json.dumps({'version': '0.1.48'}))
    result = discover_runtimes(paths, search_path=str(root))
    assert result['runtimes'][0]['version'] == '0.1.48'
    assert 'strata_family' not in result['runtimes'][0]['metadata']
    registered = RuntimeInstallation(id='deepseek', backend='strata', name='DeepSeek runtime', path=str(root),
                                     version='0.1.49.2', metadata={'strata_family': 'deepseek41'}).model_dump()
    result = discover_runtimes(paths, [registered], search_path=str(root))
    assert len(result['runtimes']) == 1
    runtime = result['runtimes'][0]
    assert runtime['registered'] and runtime['id'] == 'deepseek'
    assert runtime['version'] == '0.1.49.2' and runtime['metadata']['strata_family'] == 'deepseek41'
    assert result['errors'] == []


def test_missing_registered_deepseek_engine_is_not_validated_by_qwen(paths, tmp_path):
    root = tmp_path / 'Strata'
    write(root / 'serve/server.py')
    write(root / '.venv/Scripts/python.exe')
    write(root / 'engine/strata.exe')
    registered = RuntimeInstallation(id='deepseek', backend='strata', name='DeepSeek runtime', path=str(root),
                                     metadata={'strata_family': 'deepseek41'}).model_dump()
    result = discover_runtimes(paths, [registered], search_path=str(root))
    assert len(result['runtimes']) == 1 and not result['runtimes'][0]['registered']
    assert len(result['errors']) == 1 and result['errors'][0]['path'] == str(root)


def test_invalid_deepseek_build_preserves_unknown_candidate(paths, tmp_path):
    root = tmp_path / 'Strata'
    folder = deepseek(root)
    (folder / 'BUILD.json').write_text('not json', encoding='utf-8')
    result = discover_runtimes(paths, search_path=str(root))
    assert len(result['runtimes']) == 1 and result['runtimes'][0]['version'] == 'unknown'
    assert len(result['errors']) == 1 and result['errors'][0]['path'] == str(folder / 'BUILD.json')


def test_deepseek_discovery_does_not_follow_external_engine_folder(paths, tmp_path, monkeypatch):
    root = tmp_path / 'Strata'
    write(root / 'serve/server.py')
    write(root / '.venv/Scripts/python.exe')
    outside = tmp_path / 'external'
    write(outside / 'strata-dsv41.exe')
    write(outside / 'BUILD.json', json.dumps({'version': '0.1.49.2'}))
    try:
        if sys.platform == 'win32':
            import _winapi
            _winapi.CreateJunction(str(outside), str(root / 'engine-dsv41'))
        else:
            (root / 'engine-dsv41').symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f'Directory symlinks unavailable: {error}')
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path.resolve().is_relative_to(outside):
            raise AssertionError('Discovery must not read metadata outside its search boundary')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    result = discover_runtimes(paths, search_path=str(root))
    assert result['runtimes'] == []
    assert str(outside) not in result['searched']
