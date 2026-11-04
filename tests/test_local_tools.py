"""Isolated local-tool safety and management-API contracts; no live user files."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import time

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader import local_tools as local
from modloader.local_tools import LocalTools, LocalToolError


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / 'selected'
    root.mkdir()
    return root


@pytest.fixture
def tools():
    service = LocalTools()
    try:
        yield service
    finally:
        service.close()


def save(root, name, content):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode('utf-8') if isinstance(content, str) else content)
    return target


def test_unicode_bom_pagination_and_exact_edit(tools, workspace):
    original = '\ufeff第一行\r\n第二行🙂\r第三行\n末行'
    target = save(workspace, '文档/日本語.txt', original)
    result = tools.read(str(workspace), '文档\\日本語.txt', 2, 2)
    assert result['content'] == '第二行🙂\r第三行\n'
    assert (result['total_lines'], result['next_start_line'], result['next_start_column']) == (4, 4, 1)
    assert result['encoding'] == 'utf-8-sig' and result['has_more']
    prepared = tools.prepare_write(str(workspace), '文档/日本語.txt', edits=[
        {'old_text': '第二行🙂', 'new_text': '已编辑🙂'}, {'old_text': '末行', 'new_text': '结尾'},
    ])
    assert prepared['original_content'] == original[1:]
    assert target.read_bytes() == original.encode('utf-8')  # Preparation is read-only.
    assert tools.write(prepared['preparation_id'], approved=True)['written']
    assert target.read_bytes() == original.replace('第二行🙂', '已编辑🙂').replace('末行', '结尾').encode('utf-8')
    assert list(workspace.rglob('.modloader-chat-*')) == []


@pytest.mark.parametrize('content,old', [('same same', 'same'), ('aaa', 'aa'), ('text', 'missing')])
def test_exact_edit_requires_a_unique_match(tools, workspace, content, old):
    target = save(workspace, 'file.txt', content)
    with pytest.raises(LocalToolError, match='唯一'):
        tools.prepare_write(str(workspace), 'file.txt', edits=[{'old_text': old, 'new_text': 'new'}])
    assert target.read_text('utf-8') == content


def test_create_approval_is_single_use_and_cancellation_releases_data(tools, workspace):
    prepared = tools.prepare_write(str(workspace), 'new.txt', content='')
    assert prepared['operation'] == 'create' and prepared['original_content'] is None
    assert not (workspace / 'new.txt').exists()
    with pytest.raises(LocalToolError, match='确认'):
        tools.write(prepared['preparation_id'], approved=False)
    tools.write(prepared['preparation_id'], approved=True)
    assert (workspace / 'new.txt').read_bytes() == b''
    with pytest.raises(LocalToolError, match='失效'):
        tools.write(prepared['preparation_id'], approved=True)
    pending = tools.prepare_write(str(workspace), 'cancelled.txt', content='not written')
    assert tools.cancel(pending['preparation_id']) == {'cancelled': True}
    assert not tools._pending and tools._timer is None and not (workspace / 'cancelled.txt').exists()


def test_expiration_and_shutdown_drop_prepared_contents(tools, workspace, monkeypatch):
    prepared = tools.prepare_write(str(workspace), 'expired.txt', content='not written')
    identifier = prepared['preparation_id']
    tools._pending[identifier] = replace(tools._pending[identifier], expires_at=time.monotonic() - 1)
    with pytest.raises(LocalToolError, match='失效'):
        tools.write(identifier, approved=True)
    monkeypatch.setattr(local, 'PREPARATION_TTL_SECONDS', 0.03)
    tools.prepare_write(str(workspace), 'timer.txt', content='expire in memory')
    deadline = time.monotonic() + 2
    while tools._pending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not tools._pending
    tools.prepare_write(str(workspace), 'closed.txt', content='drop on shutdown')
    tools.close()
    assert not tools._pending and tools._timer is None
    assert not any(workspace.iterdir())


@pytest.mark.parametrize('path', [
    '../outside.txt', 'folder/../../outside.txt', '..\\outside.txt', '/outside.txt', '\\outside.txt',
    'C:\\outside.txt', 'C:outside.txt', '\\\\host\\share\\outside.txt', '\\\\?\\C:\\outside.txt',
    '\\\\.\\PhysicalDrive0', 'file.txt:stream', 'NUL', 'com1.txt', 'folder./file', 'folder /file',
])
def test_paths_cannot_choose_another_root(tools, workspace, path):
    for operation in (lambda: tools.read(str(workspace), path),
                      lambda: tools.list(str(workspace), path),
                      lambda: tools.prepare_write(str(workspace), path, content='not written')):
        with pytest.raises(LocalToolError) as caught:
            operation()
        assert caught.value.status_code == 403
    assert not any(workspace.iterdir())


@pytest.mark.parametrize('directory', ['', None, 'relative', '\\\\host\\share', '\\\\?\\C:\\'])
def test_an_explicit_local_absolute_directory_is_required(tools, directory):
    with pytest.raises(LocalToolError):
        tools.list(directory)


@pytest.mark.parametrize('path', [
    '.env', '.ENV.production', '.ssh/id_rsa', '.aws/credentials', '.config/gcloud/token.txt',
    '.config/gh/hosts.yml', '.codex/auth.json', 'credentials.dpapi.json', 'private.key', '.netrc',
])
def test_sensitive_paths_are_blocked_before_access(tools, workspace, path):
    for operation in (lambda: tools.read(str(workspace), path),
                      lambda: tools.prepare_write(str(workspace), path, content='not written')):
        with pytest.raises(LocalToolError) as caught:
            operation()
        assert caught.value.status_code == 403


def test_selected_sensitive_directory_is_not_a_bypass(tools, workspace):
    protected = workspace / '.ssh'
    protected.mkdir()
    with pytest.raises(LocalToolError, match='凭据'):
        tools.list(str(protected))


def test_list_is_paged_and_marks_protected_entries(tools, workspace):
    save(workspace, 'folder/file.txt', 'text')
    save(workspace, 'a.txt', 'a')
    save(workspace, '.env', 'fixture not a real credential')
    result = tools.list(str(workspace), limit=1)
    assert result['entries'][0]['name'] == 'folder' and result['next_offset'] == 1
    result = tools.list(str(workspace), offset=1, limit=5000)
    assert result['total_entries'] == 3 and result['next_offset'] is None
    assert next(entry for entry in result['entries'] if entry['name'] == '.env')['blocked'] is True


def test_hard_links_cannot_read_or_overwrite_outside_files(tools, workspace, tmp_path):
    outside = save(tmp_path, 'outside/credential.txt', 'outside fixture secret')
    link = workspace / 'innocent.txt'
    os.link(outside, link)
    assert tools.list(str(workspace))['entries'][0]['blocked'] is True
    for operation in (lambda: tools.read(str(workspace), link.name),
                      lambda: tools.prepare_write(str(workspace), link.name, content='changed')):
        with pytest.raises(LocalToolError, match='硬链接') as caught:
            operation()
        assert caught.value.status_code == 403
    assert outside.read_text('utf-8') == 'outside fixture secret'


def make_directory_link(link, target):
    if os.name == 'nt':
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True)
        assert result.returncode == 0, result.stderr
    else:
        link.symlink_to(target, target_is_directory=True)


def test_junction_or_symlink_escape_and_selected_link_are_rejected(tools, workspace, tmp_path):
    outside = tmp_path / 'outside'
    save(outside, 'file.txt', 'outside')
    link = workspace / 'link'
    make_directory_link(link, outside)
    assert tools.list(str(workspace))['entries'][0]['blocked'] is True
    for operation in (lambda: tools.read(str(workspace), 'link/file.txt'),
                      lambda: tools.prepare_write(str(workspace), 'link/new.txt', content='not written'),
                      lambda: tools.list(str(link))):
        with pytest.raises(LocalToolError) as caught:
            operation()
        assert caught.value.status_code == 403
    assert not (outside / 'new.txt').exists()


@pytest.mark.parametrize('content', [b'\xff\xfea\x00', b'\xfe\xff\x00a', b'valid\x00binary', b'bad\xffutf8'])
def test_non_utf8_and_binary_files_are_not_edited(tools, workspace, content):
    target = save(workspace, 'file.bin', content)
    for operation in (lambda: tools.read(str(workspace), target.name),
                      lambda: tools.prepare_write(str(workspace), target.name, content='replacement')):
        with pytest.raises(LocalToolError) as caught:
            operation()
        assert caught.value.status_code == 415
    assert target.read_bytes() == content


def test_binary_reads_are_bounded_and_stop_before_full_allocation(tools, workspace, monkeypatch):
    target = save(workspace, 'model.gguf', b'GGUF\x00' + b'x' * (local.IO_CHUNK_BYTES * 3))
    actual_open = local._Directory.open_regular
    sizes = []

    class BoundedReader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def __getattr__(self, name):
            return getattr(self.source, name)

        def read(self, size=-1):
            assert 0 < size <= local.IO_CHUNK_BYTES, 'unbounded file allocation'
            sizes.append(size)
            return self.source.read(size)

    monkeypatch.setattr(local._Directory, 'open_regular', lambda self, name: BoundedReader(actual_open(self, name)))
    for operation in (lambda: tools.read(str(workspace), target.name),
                      lambda: tools.prepare_write(str(workspace), target.name, content='replacement')):
        with pytest.raises(LocalToolError) as caught:
            operation()
        assert caught.value.status_code == 415
    assert sum(sizes) < target.stat().st_size


def test_long_lines_have_lossless_column_continuation(tools, workspace):
    text = '中' * (local.READ_PAGE_CHARACTERS + 7) + '\r\nsecond\rthird\n'
    save(workspace, 'long.txt', text)
    first = tools.read(str(workspace), 'long.txt')
    assert len(first['content']) == local.READ_PAGE_CHARACTERS
    assert first['next_start_line'] == 1 and first['next_start_column'] == local.READ_PAGE_CHARACTERS + 1
    second = tools.read(str(workspace), 'long.txt', start_line=first['next_start_line'],
                        start_column=first['next_start_column'])
    assert first['content'] + second['content'] == text
    assert second['total_lines'] == 3 and not second['has_more']
    assert tools.read(str(workspace), 'long.txt', 500, 10000)['content'] == ''


def test_crlf_split_at_io_boundary_and_empty_file(tools, workspace):
    text = 'a' * (local.IO_CHUNK_BYTES - 1) + '\r\nb'
    save(workspace, 'boundary.txt', text)
    first = tools.read(str(workspace), 'boundary.txt', line_count=1)
    second = tools.read(str(workspace), 'boundary.txt', start_line=first['next_start_line'],
                        start_column=first['next_start_column'])
    assert first['content'] + second['content'] == text
    assert second['total_lines'] == 2
    save(workspace, 'empty.txt', '')
    empty = tools.read(str(workspace), 'empty.txt')
    assert empty['content'] == '' and empty['total_lines'] == 0 and not empty['has_more']


def test_edit_memory_budget_does_not_limit_paged_reads(tools, workspace, monkeypatch):
    monkeypatch.setattr(local, 'EDIT_MEMORY_BUDGET_BYTES', 256)
    save(workspace, 'large.txt', 'text\n' * 100)
    assert tools.read(str(workspace), 'large.txt', line_count=500)['total_lines'] == 100
    with pytest.raises(LocalToolError, match='内存预算') as caught:
        tools.prepare_write(str(workspace), 'large.txt', content='new')
    assert caught.value.status_code == 413
    save(workspace, 'small.txt', 'a' * 10)
    first = tools.prepare_write(str(workspace), 'small.txt', content='b' * 10)
    with pytest.raises(LocalToolError, match='内存预算'):
        tools.prepare_write(str(workspace), 'next.txt', content='xx')
    tools.cancel(first['preparation_id'])
    assert tools.prepare_write(str(workspace), 'next.txt', content='xx')['operation'] == 'create'


def test_changed_target_and_concurrent_creation_are_preserved(tools, workspace):
    target = save(workspace, 'file.txt', 'approved original')
    first = tools.prepare_write(str(workspace), target.name, content='proposed')
    target.write_text('external edit', 'utf-8')
    with pytest.raises(LocalToolError, match='确认期间'):
        tools.write(first['preparation_id'], approved=True)
    assert target.read_text('utf-8') == 'external edit'
    second = tools.prepare_write(str(workspace), 'new.txt', content='proposed')
    save(workspace, 'new.txt', 'created externally')
    with pytest.raises(LocalToolError, match='确认期间'):
        tools.write(second['preparation_id'], approved=True)
    assert (workspace / 'new.txt').read_text('utf-8') == 'created externally'
    assert list(workspace.glob('.modloader-chat-*')) == []


@pytest.mark.parametrize('safe_save', [False, True])
def test_final_commit_race_restores_external_save_and_preserves_versions(tools, workspace, monkeypatch, safe_save):
    target = save(workspace, 'file.txt', 'approved original')
    prepared = tools.prepare_write(str(workspace), target.name, content='approved proposal')
    actual_replace = local._Directory.replace
    raced = False

    def racing_replace(self, temporary, name, *, existing, backup=None):
        nonlocal raced
        if existing and name == target.name and not raced:
            raced = True
            if safe_save:
                external = save(workspace, 'external-save.txt', 'external saved version')
                os.replace(external, target)
            else:
                target.write_text('external saved version', 'utf-8')
        return actual_replace(self, temporary, name, existing=existing, backup=backup)

    monkeypatch.setattr(local._Directory, 'replace', racing_replace)
    with pytest.raises(LocalToolError) as caught:
        tools.write(prepared['preparation_id'], approved=True)
    assert caught.value.status_code == 409 and raced
    assert target.read_text('utf-8') == 'external saved version'
    recovered = {Path(path).read_bytes() for path in caught.value.recovery_paths}
    assert {b'approved original', b'approved proposal'} <= recovered
    assert not list(workspace.glob(local.TEMPORARY_PREFIX + '*'))
    assert all(entry.get('blocked') for entry in tools.list(str(workspace))['entries'] if entry['name'].startswith(local.RECOVERY_PREFIX))


def test_external_save_during_rollback_is_retained(tools, workspace, monkeypatch):
    target = save(workspace, 'file.txt', 'original')
    prepared = tools.prepare_write(str(workspace), target.name, content='proposal')
    actual_replace = local._Directory.replace
    count = 0

    def racing_replace(self, temporary, name, *, existing, backup=None):
        nonlocal count
        if existing and name == target.name:
            count += 1
            target.write_text('external ' + str(count), 'utf-8')
        return actual_replace(self, temporary, name, existing=existing, backup=backup)

    monkeypatch.setattr(local._Directory, 'replace', racing_replace)
    with pytest.raises(LocalToolError) as caught:
        tools.write(prepared['preparation_id'], approved=True)
    values = {target.read_bytes(), *(Path(path).read_bytes() for path in caught.value.recovery_paths)}
    assert {b'original', b'proposal', b'external 1', b'external 2'} <= values
    assert count == 2  # A continuously changing target cannot cause an infinite rollback.


@pytest.mark.skipif(os.name != 'nt', reason='Windows mandatory sharing contract')
def test_open_external_writer_keeps_the_actual_replaced_file(tools, workspace):
    import ctypes
    import msvcrt

    target = save(workspace, 'file.txt', 'original')
    prepared = tools.prepare_write(str(workspace), target.name, content='proposal')
    handle = local._kernel.CreateFileW(str(target), 0x40000000, 0x1 | 0x2 | 0x4, None, 3, 0, None)
    assert handle != local.wintypes.HANDLE(-1).value
    descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    try:
        with pytest.raises(LocalToolError) as caught:
            tools.write(prepared['preparation_id'], approved=True)
        assert caught.value.status_code == 409 and caught.value.recovery_paths
        assert target.read_bytes() == b'proposal'
        os.lseek(descriptor, 0, os.SEEK_END)
        os.write(descriptor, b' external after comparison')
        assert any(Path(path).read_bytes() == b'original external after comparison'
                   for path in caught.value.recovery_paths)
    finally:
        os.close(descriptor)


@pytest.mark.skipif(os.name != 'nt', reason='Windows mandatory sharing contract')
def test_success_cleanup_denies_writes_until_delete_by_handle(tools, workspace, monkeypatch):
    import ctypes

    target = save(workspace, 'file.txt', 'original')
    prepared = tools.prepare_write(str(workspace), target.name, content='proposal')
    actual_match = local._Directory._source_matches
    attempted = []

    def match_then_write(self, source, name, original, *, renamed=False):
        result = actual_match(self, source, name, original, renamed=renamed)
        if renamed and result:
            attempted.append(name)
            handle = local._kernel.CreateFileW(str(self.path / name), 0x40000000, 0x1 | 0x2 | 0x4, None, 3, 0, None)
            if handle != local.wintypes.HANDLE(-1).value:
                local._kernel.CloseHandle(handle)
                pytest.fail('external write opened between backup comparison and deletion')
            assert ctypes.get_last_error() == 32
        return result

    monkeypatch.setattr(local._Directory, '_source_matches', match_then_write)
    assert tools.write(prepared['preparation_id'], approved=True)['written']
    assert attempted and target.read_bytes() == b'proposal'
    assert not list(workspace.glob('.modloader-chat-*'))


def test_write_failure_leaves_original_and_cleans_own_temporary(tools, workspace, monkeypatch):
    target = save(workspace, 'file.txt', 'original')
    prepared = tools.prepare_write(str(workspace), target.name, content='proposal')

    def failing_replace(*args, **kwargs):
        raise OSError('replacement denied by fixture')

    monkeypatch.setattr(local._Directory, 'replace', failing_replace)
    with pytest.raises(LocalToolError):
        tools.write(prepared['preparation_id'], approved=True)
    assert target.read_bytes() == b'original'
    assert list(workspace.glob(local.TEMPORARY_PREFIX + '*')) == []


def test_orphan_recovery_preserves_active_and_recovery_files(tools, workspace):
    orphan = local.TEMPORARY_PREFIX + 'a' * 32 + '.tmp'
    save(workspace, orphan, 'orphan')
    save(workspace, orphan.removesuffix('.tmp') + '.lock', '')
    recovery = save(workspace, local.RECOVERY_PREFIX + 'retained.previous', 'external recovery data')
    assert tools.list(str(workspace))['total_entries'] == 1
    assert not (workspace / orphan).exists()
    assert recovery.read_text('utf-8') == 'external recovery data'
    active = local.TEMPORARY_PREFIX + 'b' * 32 + '.tmp'
    with local._directory(workspace) as anchor:
        with anchor.temporary_lock(active):
            with anchor.temporary(active) as output:
                output.write(b'active')
            tools.list(str(workspace))
            assert (workspace / active).read_bytes() == b'active'
            anchor.unlink(active)
    assert not list(workspace.glob(local.TEMPORARY_PREFIX + '*'))


async def test_management_api_contract_approval_and_public_separation(tmp_path):
    controller = Controller(tmp_path / 'data', 'local-tools-fixture')
    root = tmp_path / 'project'
    root.mkdir()
    target = save(root, 'file.txt', 'original')
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver') as client:
            assert (await client.post('/api/chat-tools/list', json={'directory': str(root)})).status_code == 401
            client.headers['Authorization'] = 'Bearer local-tools-fixture'
            assert (await client.post('/api/chat-tools/list', json={'directory': str(root)})).json()['entries'][0]['name'] == 'file.txt'
            result = await client.post('/api/chat-tools/read', json={'directory': str(root), 'path': target.name, 'start_column': 2})
            assert result.status_code == 200 and result.json()['content'] == 'riginal'
            prepared = await client.post('/api/chat-tools/prepare-write', json={'directory': str(root), 'path': target.name, 'content': 'approved'})
            assert prepared.status_code == 200 and prepared.json()['original_content'] == 'original'
            identifier = prepared.json()['preparation_id']
            assert target.read_bytes() == b'original'
            denied = await client.post('/api/chat-tools/write', json={'preparation_id': identifier, 'approved': False})
            assert denied.status_code == 403 and target.read_bytes() == b'original'
            changed = await client.post('/api/chat-tools/write', json={'preparation_id': identifier, 'approved': True, 'content': 'unapproved'})
            assert changed.status_code == 400
            result = await client.post('/api/chat-tools/write', json={'preparation_id': identifier, 'approved': True})
            assert result.status_code == 200 and target.read_bytes() == b'approved'
            assert (await client.post('/api/chat-tools/write', json={'preparation_id': identifier, 'approved': True})).status_code == 409
            pending = (await client.post('/api/chat-tools/prepare-write', json={'directory': str(root), 'path': 'cancel.txt', 'content': 'no'})).json()
            assert (await client.delete('/api/chat-tools/prepared/' + pending['preparation_id'])).json() == {'cancelled': True}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
            assert (await public.post('/api/chat-tools/read', json={'directory': str(root), 'path': target.name})).status_code == 404
    finally:
        await controller.close()


def test_empty_directory_path_lists_root_but_does_not_relax_file_paths(tools, workspace):
    save(workspace, 'note.txt', 'content')
    assert tools.list(str(workspace), '') == tools.list(str(workspace), '.')
    with pytest.raises(LocalToolError):
        tools.read(str(workspace), '')
    with pytest.raises(LocalToolError):
        tools.prepare_write(str(workspace), '', content='no')
