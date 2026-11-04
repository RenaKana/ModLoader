"""Persistent, revocable references to files in a conversation's selected directory."""
from contextlib import contextmanager
import codecs
import json
import os
from pathlib import Path
import secrets
import subprocess
import zipfile

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .conversations import DETAIL_KIND, SUMMARY_KIND
from .local_tools import (LocalToolError, _directory, _relative, _root,
                          _version, IO_CHUNK_BYTES)

RESOURCE_KIND = 'chat_local_resources'
TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.tsv', '.json', '.yaml', '.yml', '.log', '.xml'}
IMAGE_EXTENSIONS = {
    'image/png': {'.png'}, 'image/jpeg': {'.jpg', '.jpeg', '.jpe'},
    'image/gif': {'.gif'}, 'image/webp': {'.webp'}, 'image/bmp': {'.bmp'},
    'image/x-icon': {'.ico'},
    'image/avif': {'.avif'},
}


def forget_conversation_resources(store, conversation_id, keep_ids=()):
    """Call inside conversation deletion/explicit clear, never autosave."""
    store.connection.execute("DELETE FROM records WHERE kind=? AND json_extract(body, '$.conversation_id')=? "
                             "AND id NOT IN (SELECT value FROM json_each(?))",
                             (RESOURCE_KIND, conversation_id, json.dumps(list(keep_ids))))


def _classify(source, name):
    """Inspect bytes; the extension only gates safe default-app dispatch."""
    head = source.read(64)
    source.seek(0)
    mime = None
    if head.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif head.startswith(b'\xff\xd8\xff'):
        mime = 'image/jpeg'
    elif head.startswith((b'GIF87a', b'GIF89a')):
        mime = 'image/gif'
    elif head.startswith(b'RIFF') and head[8:12] == b'WEBP':
        mime = 'image/webp'
    elif head.startswith(b'BM'):
        mime = 'image/bmp'
    elif head.startswith(b'\x00\x00\x01\x00'):
        mime = 'image/x-icon'
    elif head[4:8] == b'ftyp' and (head[8:12] in (b'avif', b'avis') or any(head[index:index + 4] in (b'avif', b'avis') for index in range(16, len(head) - 3, 4))):
        mime = 'image/avif'
    suffix = Path(name).suffix.casefold()
    if mime:
        return mime, 'image', suffix in IMAGE_EXTENSIONS[mime]
    if head.startswith(b'%PDF-'):
        return 'application/pdf', 'file', suffix == '.pdf'
    if head.startswith((b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08')):
        try:
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())
                if '[Content_Types].xml' in names:
                    for entry, extension, mime in (
                        ('word/document.xml', '.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
                        ('xl/workbook.xml', '.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
                        ('ppt/presentation.xml', '.pptx', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
                    ):
                        if entry in names:
                            macros = any(name.casefold().endswith('/vbaproject.bin') for name in names)
                            return mime, 'file', suffix == extension and not macros
            return 'application/zip', 'file', suffix == '.zip'
        except zipfile.BadZipFile:
            return 'application/octet-stream', 'file', False
        finally:
            source.seek(0)
    if head.startswith((b'MZ', b'\x7fELF', b'\x4c\x00\x00\x00\x01\x14\x02\x00')):
        return 'application/octet-stream', 'file', False
    # Validate the complete text using bounded reads, not just its first page.
    decoder = codecs.getincrementaldecoder('utf-8-sig')()
    try:
        while chunk := source.read(IO_CHUNK_BYTES):
            if b'\x00' in chunk:
                raise UnicodeError()
            decoder.decode(chunk)
        decoder.decode(b'', final=True)
        text = True
    except UnicodeError:
        text = False
    finally:
        source.seek(0)
    return ('text/plain' if text else 'application/octet-stream'), 'file', text and suffix in TEXT_EXTENSIONS and not head.lstrip().startswith(b'#!')


def _system_action(path, action):
    if os.name != 'nt':
        raise LocalToolError('请在 Windows 桌面应用中打开本地附件', 409)
    try:
        if action == 'open':
            os.startfile(str(path))
        else:
            subprocess.Popen(['explorer.exe', '/select,', str(path)])
    except OSError as error:
        raise LocalToolError('无法交给系统处理：' + (error.strerror or str(error)), 409) from error


class LocalResources:
    def __init__(self, store):
        self.store = store

    def _authorized_root(self, conversation_id):
        if not isinstance(conversation_id, str) or not self.store.get(SUMMARY_KIND, conversation_id):
            raise LocalToolError('附件所属对话不存在', 404)
        detail = self.store.get(DETAIL_KIND, conversation_id) or {}
        directory = detail.get('tool_directory', detail.get('settings', {}).get('toolDirectory', ''))
        if not directory:
            raise LocalToolError('本地目录授权已撤回', 403)
        return _root(directory)

    def attach(self, conversation_id, directory, path):
        root, parts = self._authorized_root(conversation_id), _relative(path)
        if root != _root(directory):
            raise LocalToolError('本地目录授权已更换，请重试', 403)
        with _directory(root, parts[:-1]) as anchor:
            with anchor.open_regular(parts[-1], deny_write=True, deny_delete=True) as source:
                before = os.fstat(source.fileno())
                mime, kind, can_open = _classify(source, parts[-1])
                if _version(before) != _version(os.fstat(source.fileno())):
                    raise LocalToolError('附件在读取时发生变化，请重新引用', 409)
                record = {'id': secrets.token_urlsafe(32), 'conversation_id': conversation_id,
                          'directory': str(root), 'path': '/'.join(parts), 'name': parts[-1],
                          'size': before.st_size, 'mime': mime, 'kind': kind, 'can_open': can_open,
                          'root_identity': list(anchor.root_identity), 'file_version': _json_version(before)}
                with self.store.lock:
                    if self._authorized_root(conversation_id) != root:
                        raise LocalToolError('本地目录授权已更换，请重试', 403)
                    self.store.put(RESOURCE_KIND, record)
        return {'resource': self._public(record), 'reference': 'modloader-resource:' + record['id']}

    @staticmethod
    def _public(record):
        return {key: record[key] for key in ('id', 'name', 'path', 'directory', 'size', 'mime', 'kind', 'can_open')}

    @contextmanager
    def opened(self, identifier):
        record = self.store.get(RESOURCE_KIND, identifier)
        if record is None:
            raise LocalToolError('附件引用不存在', 404)
        root = self._authorized_root(record['conversation_id'])
        if root != _root(record['directory']):
            raise LocalToolError('附件所属目录已不再授权', 403)
        parts = _relative(record['path'])
        with _directory(root, parts[:-1]) as anchor:
            if list(anchor.root_identity) != record['root_identity']:
                raise LocalToolError('授权目录已被替换，请重新引用附件', 409)
            with anchor.open_regular(parts[-1], deny_write=True, deny_delete=True) as source:
                if _json_version(os.fstat(source.fileno())) != record['file_version']:
                    raise LocalToolError('附件已发生变化，请重新引用', 409)
                yield record, source, anchor.path / parts[-1]

    def metadata(self, identifier):
        with self.opened(identifier) as (record, _, __):
            return self._public(record)

    def image(self, identifier):
        # Enter before sending headers so missing/revoked/changed files return a
        # normal API error, then retain the pinned handle through streaming.
        context = self.opened(identifier)
        record, source, _ = context.__enter__()
        if record['kind'] != 'image':
            context.__exit__(None, None, None)
            raise LocalToolError('此附件不是可内嵌图片', 415)

        def chunks():
            try:
                while chunk := source.read(IO_CHUNK_BYTES):
                    if self._authorized_root(record['conversation_id']) != _root(record['directory']):
                        raise LocalToolError('本地目录授权已更换', 403)
                    if self.store.get(RESOURCE_KIND, identifier) is None:
                        raise LocalToolError('附件引用已移除', 404)
                    yield chunk
            finally:
                context.__exit__(None, None, None)
        return StreamingResponse(chunks(), media_type=record['mime'], background=BackgroundTask(context.__exit__, None, None, None), headers={
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "default-src 'none'; sandbox",
        })

    def action(self, identifier, action):
        if action not in ('open', 'reveal'):
            raise LocalToolError('附件操作无效')
        with self.store.lock, self.opened(identifier) as (record, _, path):
            if action == 'open' and not record['can_open']:
                raise LocalToolError('此文件类型不允许直接打开，请使用文件夹定位', 403)
            _system_action(path, action)
        return {'handed_off': True}


def _json_version(info):
    identity, *rest = _version(info)
    return [list(identity), *rest]


def create_local_resource_router(service):
    router = APIRouter(prefix='/api/chat-tools')

    @router.post('/attach')
    def attach(body: dict):
        return service.attach(body.get('conversation_id'), body.get('directory'), body.get('path'))

    @router.get('/resources/{identifier}')
    def metadata(identifier: str):
        return service.metadata(identifier)

    @router.get('/resources/{identifier}/content')
    def content(identifier: str):
        return service.image(identifier)

    @router.post('/resources/{identifier}/action')
    def action(identifier: str, body: dict):
        return service.action(identifier, body.get('action'))

    return router
