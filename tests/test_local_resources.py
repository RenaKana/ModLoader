"""Controlled local attachments, with isolated files and stubbed OS handoff."""
import base64
import io
import os
from pathlib import Path
import subprocess
import zipfile

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest

from modloader import local_resources as resources
from modloader.local_tools import LocalToolError
from modloader.store import Store

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


@pytest.fixture
def environment(tmp_path):
    directory = tmp_path / 'authorized'
    directory.mkdir()
    store = Store(tmp_path / 'isolated.sqlite3')
    store.put('conversations', {'id': 'test-conversation'})
    store.put('conversation_details', {'id': 'test-conversation', 'tool_directory': str(directory)})
    service = resources.LocalResources(store)
    try:
        yield directory, store, service
    finally:
        store.close()


def attach(environment, name, content):
    directory, _, service = environment
    (directory / name).write_bytes(content)
    return service.attach('test-conversation', str(directory), name)['resource']


def client_for(service):
    app = FastAPI()

    @app.exception_handler(LocalToolError)
    async def error(_, issue):
        return JSONResponse({'error': str(issue)}, status_code=issue.status_code)

    app.include_router(resources.create_local_resource_router(service))
    return TestClient(app)


def test_image_reference_is_small_persistent_and_streams_original_bytes(environment):
    directory, store, service = environment
    image = attach(environment, '图片.dat', PNG)
    assert image['kind'] == 'image' and image['mime'] == 'image/png'
    assert image['size'] == len(PNG) and image['can_open'] is False
    assert 'url' not in image and 'content' not in image
    restored = resources.LocalResources(store)
    assert restored.metadata(image['id']) == image
    with client_for(restored) as client:
        result = client.get(f"/api/chat-tools/resources/{image['id']}/content")
        assert result.status_code == 200 and result.content == PNG
        assert result.headers['content-type'] == 'image/png'
        assert result.headers['cache-control'] == 'no-store'
        assert result.headers['x-content-type-options'] == 'nosniff'


@pytest.mark.parametrize('name,content,mime,can_open', [
    ('ordinary.txt', '完整文本🙂'.encode(), 'text/plain', True),
    ('pretend.png', b'not an image', 'text/plain', False),
    ('fake.txt', b'MZ' + b'\x00' * 40, 'application/octet-stream', False),
    ('run.exe', PNG, 'image/png', False),
    ('run.ps1', b'Write-Output hello', 'text/plain', False),
    ('run.cmd', b'@echo hello', 'text/plain', False),
    ('run.py', b'print(1)', 'text/plain', False),
    ('run.txt', b'#!/usr/bin/env python\nprint(1)', 'text/plain', False),
    ('link.lnk', b'\x4c\x00\x00\x00\x01\x14\x02\x00', 'application/octet-stream', False),
    ('link.url', b'[InternetShortcut]\nURL=file:///C:/run.exe', 'text/plain', False),
    ('picture.png', PNG, 'image/png', True),
    ('document.pdf', b'%PDF-1.7\n%%EOF', 'application/pdf', True),
    ('invalid.txt', b'first\n' + b'a' * 65536 + b'\xff', 'application/octet-stream', False),
], ids=lambda value: 'bytes' if isinstance(value, bytes) else str(value))
def test_type_comes_from_bytes_and_default_app_is_gated(environment, name, content, mime, can_open):
    item = attach(environment, name, content)
    assert (item['mime'], item['can_open']) == (mime, can_open)


def test_actions_check_current_authority_and_never_report_opened(environment, monkeypatch):
    directory, store, service = environment
    item = attach(environment, 'ordinary.txt', b'hello')
    calls = []
    monkeypatch.setattr(resources, '_system_action', lambda path, action: calls.append((path, action)))
    assert service.action(item['id'], 'open') == {'handed_off': True}
    assert calls == [(directory / 'ordinary.txt', 'open')]
    executable = attach(environment, 'run.cmd', b'@echo bad')
    with pytest.raises(LocalToolError, match='不允许直接打开'):
        service.action(executable['id'], 'open')
    assert service.action(executable['id'], 'reveal') == {'handed_off': True}
    assert calls[-1] == (directory / 'run.cmd', 'reveal')
    store.put('conversation_details', {'id': 'test-conversation', 'tool_directory': ''})
    with pytest.raises(LocalToolError, match='撤回'):
        service.action(item['id'], 'open')
    with client_for(service) as client:
        assert client.get(f"/api/chat-tools/resources/{item['id']}").status_code == 403


def test_office_documents_are_identified_from_archive_contents_and_macros_not_opened(environment):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('[Content_Types].xml', '<Types/>')
        archive.writestr('word/document.xml', '<document/>')
    item = attach(environment, 'document.docx', buffer.getvalue())
    assert item['mime'].endswith('wordprocessingml.document') and item['can_open']
    disguised = attach(environment, 'document.xlsx', buffer.getvalue())
    assert not disguised['can_open']
    with zipfile.ZipFile(buffer, 'a') as archive:
        archive.writestr('word/vbaProject.bin', b'macro')
    macro = attach(environment, 'macro.docx', buffer.getvalue())
    assert not macro['can_open']


def test_handoff_failure_propagates(environment, monkeypatch):
    item = attach(environment, 'ordinary.txt', b'hello')
    def fail(*_):
        raise LocalToolError('找不到默认应用', 409)
    monkeypatch.setattr(resources, '_system_action', fail)
    with client_for(environment[2]) as client:
        result = client.post(f"/api/chat-tools/resources/{item['id']}/action", json={'action': 'open'})
        assert result.status_code == 409 and result.json()['error'] == '找不到默认应用'


def test_changed_deleted_missing_conversation_and_switched_directory_fail(environment):
    directory, store, service = environment
    changed = attach(environment, 'changed.txt', b'before')
    (directory / 'changed.txt').write_bytes(b'after')
    with pytest.raises(LocalToolError, match='变化'):
        service.metadata(changed['id'])
    missing = attach(environment, 'missing.txt', b'hello')
    (directory / 'missing.txt').unlink()
    with pytest.raises(LocalToolError, match='不存在'):
        service.metadata(missing['id'])
    valid = attach(environment, 'valid.txt', b'hello')
    other = directory.parent / 'other'
    other.mkdir()
    store.put('conversation_details', {'id': 'test-conversation', 'tool_directory': str(other)})
    with pytest.raises(LocalToolError, match='不再授权'):
        service.metadata(valid['id'])
    with pytest.raises(LocalToolError, match='已更换'):
        service.attach('test-conversation', str(directory), 'valid.txt')
    store.remove('conversations', 'test-conversation')
    with pytest.raises(LocalToolError, match='对话不存在'):
        service.metadata(valid['id'])


@pytest.mark.parametrize('path', ['../escape.txt', '/absolute.txt', 'C:/absolute.txt', '.env', '.ssh/id_rsa', 'file.txt:stream', 'NUL', '.modloader-chat-recovery-kept'])
def test_protected_and_escaping_paths_are_rejected(environment, path):
    directory, _, service = environment
    with pytest.raises(LocalToolError):
        service.attach('test-conversation', str(directory), path)


def test_replaced_root_and_file_hardlink_rejected(environment):
    directory, _, service = environment
    item = attach(environment, 'file.txt', b'hello')
    os.link(directory / 'file.txt', directory / 'hardlink.txt')
    with pytest.raises(LocalToolError, match='硬链接'):
        service.metadata(item['id'])
    old = directory.with_name('old-directory')
    directory.rename(old)
    directory.mkdir()
    (directory / 'file.txt').write_bytes(b'hello')
    with pytest.raises(LocalToolError, match='目录已被替换'):
        service.metadata(item['id'])


@pytest.mark.skipif(os.name != 'nt', reason='Windows junction policy')
def test_windows_junction_rejected(environment):
    directory, _, service = environment
    outside = directory.parent / 'outside'
    outside.mkdir()
    (outside / 'file.txt').write_bytes(b'outside')
    junction = directory / 'junction'
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), str(outside)], capture_output=True)
    assert result.returncode == 0
    with pytest.raises(LocalToolError, match='junction|reparse'):
        service.attach('test-conversation', str(directory), 'junction/file.txt')


def test_existing_text_read_and_approved_write_contract_still_works(environment):
    from modloader.local_tools import LocalTools
    directory, _, _ = environment
    service = LocalTools()
    try:
        (directory / 'text.txt').write_text('before', encoding='utf-8')
        assert service.read(str(directory), 'text.txt')['content'] == 'before'
        pending = service.prepare_write(str(directory), 'text.txt', content='after')
        assert service.write(pending['preparation_id'], approved=True)['written']
        assert service.read(str(directory), 'text.txt')['content'] == 'after'
    finally:
        service.close()


def test_explicit_history_replacement_keeps_retained_references(environment):
    _, store, service = environment
    keep = attach(environment, 'keep.txt', b'kept')
    discard = attach(environment, 'discard.txt', b'discarded')
    with store.lock, store.connection:
        resources.forget_conversation_resources(store, 'test-conversation', [keep['id']])
    assert service.metadata(keep['id']) == keep
    with pytest.raises(LocalToolError, match='不存在'):
        service.metadata(discard['id'])
    with store.lock, store.connection:
        resources.forget_conversation_resources(store, 'test-conversation')
    with pytest.raises(LocalToolError, match='不存在'):
        service.metadata(keep['id'])
