"""Request progress uses isolated files/SDK fixtures, never installed environments."""
import asyncio
import json
from pathlib import Path
import subprocess
import threading

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.models import AppPaths, DownloadRequest, RuntimeInstallation, SourceFile, SourceRepository
from modloader.runtime_discovery import discover_runtimes
from modloader.sources import SourceService


def test_runtime_progress_reports_current_directory_and_real_counts(tmp_path, monkeypatch):
    selected = tmp_path / 'selected'
    for name in ('first', 'second'):
        directory = selected / name
        directory.mkdir(parents=True)
        (directory / 'llama-server.exe').write_text('fixture, never executed')

    def forbid(*args, **kwargs):
        pytest.fail('Discovery must not execute a candidate')

    monkeypatch.setattr(subprocess, 'run', forbid)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbid)
    events = []
    original = Path.iterdir

    def entries(path):
        if path == selected:
            assert events and str(selected) in events[-1]['message']
        return original(path)

    monkeypatch.setattr(Path, 'iterdir', entries)
    result = discover_runtimes(AppPaths(tmp_path / 'data'), search_path=str(selected), progress=events.append)
    assert len(result['runtimes']) == 2
    assert all(event['stage'] == 'runtime_discovery' for event in events)
    assert all('total' not in event and 'percent' not in event for event in events)
    assert '所选文件夹' in events[-1]['message']
    assert str(selected / 'second') in events[-1]['message']
    assert '已检查 3 个目录 · 已发现 2 个环境' in events[-1]['message']


def test_runtime_progress_cancellation_keeps_found_entries_and_stops_later_work(tmp_path):
    selected = tmp_path / 'selected'
    selected.mkdir()
    (selected / 'llama-server.exe').write_text('fixture')
    (selected / 'later').mkdir()
    (selected / 'later' / 'llama-server.exe').write_text('fixture')
    cancel = threading.Event()

    def progress(event):
        if '已发现 1 个环境' in event['message']:
            cancel.set()

    result = discover_runtimes(AppPaths(tmp_path / 'data'), search_path=str(selected), cancel=cancel, progress=progress)
    assert [runtime['path'] for runtime in result['runtimes']] == [str(selected / 'llama-server.exe')]
    assert str(selected / 'later') not in result['searched']
    assert any('已取消' in note for note in result['notes'])


def service(tmp_path):
    return SourceService(AppPaths(tmp_path / 'data'), lambda _: None, lambda: None)


async def test_repository_plan_progress_counts_only_actual_index_and_config_reads(tmp_path, monkeypatch):
    source = service(tmp_path)
    paths = [f'{folder}/{name}' for folder in ('first', 'second')
             for name in ('model.safetensors.index.json', 'weights.safetensors', 'config.json')]
    repository = SourceRepository(source='hf', repo_id='fixture/model', revision='commit', name='model',
                                  files=[SourceFile(path=path, size=8) for path in paths])
    events, reads = [], []

    async def worker(payload, provider, progress=None):
        assert provider == 'hf'
        if payload['action'] == 'resolve':
            assert events[-1]['stage'] == 'repository'
            return repository.model_dump()
        assert payload['action'] == 'read_json'
        path = payload['file']
        reads.append(path)
        assert events[-1]['message'].endswith(path)
        return {'weight_map': {'weight': 'weights.safetensors'}} if path.endswith('index.json') else {}

    monkeypatch.setattr(source, '_worker', worker)
    plan = await source.plan(DownloadRequest(source='hf', repo_id='fixture/model',
                                             files=['first/weights.safetensors', 'second/weights.safetensors']), progress=events.append)
    assert len(plan['files']) == 6
    assert reads == ['first/model.safetensors.index.json', 'second/model.safetensors.index.json',
                     'first/config.json', 'second/config.json']
    assert [event['stage'] for event in events] == ['repository', *['weight_indexes'] * 4, *['model_config'] * 4]
    for stage in ('weight_indexes', 'model_config'):
        counted = [event for event in events if event['stage'] == stage]
        assert [(event['completed'], event['total']) for event in counted] == [(0, 2), (1, 2), (1, 2), (2, 2)]


async def test_cancel_repository_read_stops_before_following_index_or_configuration(tmp_path, monkeypatch):
    source = service(tmp_path)
    entered, stopped = asyncio.Event(), asyncio.Event()
    reads = []

    async def worker(payload, provider, progress=None):
        if payload['action'] == 'resolve':
            return SourceRepository(source='hf', repo_id='fixture/model', revision='commit', name='model',
                                    files=[SourceFile(path=path, size=8) for path in ('model.safetensors.index.json', 'weights.safetensors', 'config.json')]).model_dump()
        reads.append(payload['file'])
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(source, '_worker', worker)
    task = asyncio.create_task(source.plan(DownloadRequest(source='hf', repo_id='fixture/model', files=['weights.safetensors']), progress=lambda event: None))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and reads == ['model.safetensors.index.json']


async def test_readme_progress_keeps_explicit_document_and_result(tmp_path, monkeypatch):
    source = service(tmp_path)
    events = []
    document = {'path': 'docs/README.zh-CN.md', 'revision': 'commit', 'content': '# fixture'}

    async def worker(payload, provider, progress=None):
        assert payload['action'] == 'readme' and payload['file_path'] == document['path']
        assert events == [{'stage': 'readme', 'message': '读取 docs/README.zh-CN.md · fixture/model'}]
        return document

    monkeypatch.setattr(source, '_worker', worker)
    assert await source.readme('hf', 'fixture/model', 'commit', document['path'], progress=events.append) == document


@pytest.mark.parametrize('stream', [False, True])
async def test_runtime_refresh_forwards_specific_checks_and_preserves_json_contract(tmp_path, monkeypatch, stream):
    from modloader.backends import installers

    controller = Controller(tmp_path / 'isolated-data', 'management-secret')
    runtime = RuntimeInstallation(id='fixture-runtime', backend='llama_cpp', name='Fixture', path=str(tmp_path / 'never-executed.exe'))
    controller.store.put('runtimes', runtime.model_dump())

    async def validate(value, paths=None, *, progress=None, **kwargs):
        assert value.id == runtime.id and progress is not None
        progress({'phase': 'verify', 'message': '检查 llama.cpp --version'})
        await asyncio.sleep(0)
        return value.model_copy(update={'version': 'fixture-b2'})

    monkeypatch.setattr(installers, 'validate_runtime', validate)
    try:
        headers = {'Authorization': 'Bearer management-secret', 'Accept': 'application/x-ndjson' if stream else 'application/json'}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers=headers) as client:
            response = await client.post('/api/runtimes/fixture-runtime/refresh')
        assert response.status_code == 200
        if stream:
            frames = [json.loads(line) for line in response.text.splitlines()]
            assert any(frame['type'] == 'progress' and frame['message'] == '检查 llama.cpp --version' for frame in frames)
            result = frames[-1]['value']
        else:
            result = response.json()
        assert result['id'] == runtime.id and result['version'] == 'fixture-b2'
        assert controller.require('runtimes', runtime.id)['version'] == 'fixture-b2'
        assert controller.store.all('jobs') == []
    finally:
        await controller.close()
        controller.store.close()
