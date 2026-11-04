"""Text-only chat tools, scoped to a directory explicitly supplied by the UI.

Path checks follow the MCP filesystem server's real-path/parent validation and
no-follow approach. Directory handles stay pinned while I/O runs; neither a
model path nor a pending write can select a different root. Writes deliberately
use an atomic replacement, rather than MCP's current in-place overwrite.
https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import codecs
import ntpath
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time


PREPARATION_TTL_SECONDS = 15 * 60
IO_CHUNK_BYTES = 64 * 1024
READ_PAGE_CHARACTERS = 64 * 1024
# Full edit previews retain original/replacement bytes, decoded strings and
# JSON representations. Reserve 12 bytes per input UTF-8 byte for their peak
# coexistence; pagination does not have this full-document allocation.
EDIT_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
EDIT_MEMORY_MULTIPLIER = 12
TEMPORARY_PREFIX = '.modloader-chat-write-'
TEMPORARY_NAME = re.compile(re.escape(TEMPORARY_PREFIX) + r'[0-9a-f]{32}\.(?:tmp|lock)\Z')
RECOVERY_PREFIX = '.modloader-chat-recovery-'

# This is a name-based guard, not a content classifier. Apply it to the full
# path, including a selected root, so selecting .ssh itself cannot bypass it.
PROTECTED_DIRECTORIES = frozenset({
    '.ssh', '.aws', '.azure', '.gnupg', '.kube', '.docker', '.gcloud', '.git', '.codex',
})
PROTECTED_FILES = frozenset({
    '.netrc', '_netrc', '.npmrc', '.pypirc', '.git-credentials', '.boto', '.s3cfg', '.authinfo',
    'credentials', 'credentials.json', 'credentials.dpapi.json',
    'secrets.json', 'secrets.yaml', 'secrets.yml', 'tokens.json', 'token.json',
    'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519',
    'application_default_credentials.json', 'auth.json', 'oauth_creds.json', '.credentials.json',
    'service-account.json', 'service_account.json',
})
PRIVATE_KEY_SUFFIXES = ('.pem', '.key', '.p12', '.pfx')
WINDOWS_DEVICES = frozenset({
    'con', 'prn', 'aux', 'nul', 'conin$', 'conout$',
    *(f'{prefix}{number}' for prefix in ('com', 'lpt') for number in '123456789¹²³'),
})


class LocalToolError(ValueError):
    def __init__(self, message: str, status_code: int = 400, *, recovery_paths=None):
        super().__init__(message)
        self.status_code = status_code
        self.recovery_paths = recovery_paths or []


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LocalToolError(f'{name} 必须是大于或等于 {minimum} 的整数')
    return value


def _protected(parts):
    names = [part.casefold() for part in parts]
    if any(name in PROTECTED_DIRECTORIES for name in names):
        return True
    if any(names[index:index + 2] in (['.config', 'gcloud'], ['.config', 'gh']) for index in range(len(names) - 1)):
        return True
    return any(name in PROTECTED_FILES or name == '.env' or name.startswith('.env.')
               or name.endswith(PRIVATE_KEY_SUFFIXES) for name in names)


def _safe_components(parts):
    for part in parts:
        if part == '..' or '\x00' in part or ':' in part or part.endswith((' ', '.')):
            raise LocalToolError('路径不能包含上级目录、设备路径、数据流或末尾空格/句点', 403)
        if part.split('.')[0].casefold() in WINDOWS_DEVICES:
            raise LocalToolError('不允许访问设备路径', 403)


def _root(directory):
    if not isinstance(directory, str) or not directory or '\x00' in directory:
        raise LocalToolError('请先选择工作目录')
    if directory.startswith(('\\\\', '//')) or ntpath.splitdrive(directory)[0].startswith('\\\\'):
        raise LocalToolError('工作目录必须是本机目录，不能使用 UNC 或设备路径', 403)
    root = Path(directory)
    if not root.is_absolute():
        raise LocalToolError('工作目录必须是绝对路径')
    _safe_components(root.parts[1:])
    if _protected(root.parts):
        raise LocalToolError('不能将凭据目录或凭据文件用作工作目录', 403)
    return root


def _relative(path, *, allow_root=False):
    if not isinstance(path, str) or not path or '\x00' in path:
        raise LocalToolError('path 必须是相对于工作目录的路径')
    if ntpath.isabs(path) or ntpath.splitdrive(path)[0] or path.startswith(('/', '\\')):
        raise LocalToolError('path 只能使用相对路径', 403)
    raw = path.replace('\\', '/').split('/')
    _safe_components(part for part in raw if part not in ('', '.'))
    parts = tuple(part for part in raw if part not in ('', '.'))
    if not parts and not allow_root:
        raise LocalToolError('请指定文件路径')
    if _protected(parts):
        raise LocalToolError('不允许工具访问凭据文件或目录', 403)
    if any(TEMPORARY_NAME.fullmatch(part) or part.startswith(RECOVERY_PREFIX) for part in parts):
        raise LocalToolError('不允许工具访问写入临时文件或恢复副本', 403)
    return parts


def _is_link(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _identity(info):
    return info.st_dev, info.st_ino


def _version(info):
    return _identity(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode


if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _kernel.CreateFileW.restype = wintypes.HANDLE
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL
    _kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
    _kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _kernel.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    _kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _kernel.ReplaceFileW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                    wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID]
    _kernel.ReplaceFileW.restype = wintypes.BOOL
    _kernel.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _kernel.SetFileInformationByHandle.restype = wintypes.BOOL

    class _AttributeTagInfo(ctypes.Structure):
        _fields_ = [('attributes', wintypes.DWORD), ('tag', wintypes.DWORD)]

    class _DispositionInfo(ctypes.Structure):
        _fields_ = [('delete_file', ctypes.c_ubyte)]

    def _windows_handle(path, *, directory=False, create=False, exclusive=False, deny_write=False, deny_delete=False, delete=False):
        # OPEN_REPARSE_POINT opens the entry itself, never its target. Keeping
        # ancestor directories open without write/delete sharing prevents both
        # junction conversion and renaming while a child path is being used.
        access = 0x80 if directory else 0x80000000 | (0x40000000 if create else 0)
        if delete:
            access |= 0x10000
        sharing = 0 if exclusive else 0x1 if directory else 0x1 | 0x2 | 0x4
        if deny_write:
            sharing &= ~0x2
        if deny_delete:
            sharing &= ~0x4
        flags = 0x00200000 | (0x02000000 if directory else 0)
        handle = _kernel.CreateFileW(str(path), access, sharing, None, 1 if create else 3, flags, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = _AttributeTagInfo()
            if not _kernel.GetFileInformationByHandleEx(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.attributes & 0x400:
                raise LocalToolError('不允许通过符号链接、junction 或 reparse point 访问文件', 403)
            if directory and not info.attributes & 0x10:
                raise LocalToolError('路径不是目录')
            size = _kernel.GetFinalPathNameByHandleW(handle, None, 0, 0)
            if not size:
                raise ctypes.WinError(ctypes.get_last_error())
            final = ctypes.create_unicode_buffer(size + 1)
            if not _kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0):
                raise ctypes.WinError(ctypes.get_last_error())
            canonical = final.value
            if canonical.startswith('\\\\?\\UNC\\'):
                raise LocalToolError('不允许通过映射盘访问 UNC 目录', 403)
            if canonical.startswith('\\\\?\\'):
                canonical = canonical[4:]
            if not Path(canonical).is_absolute() or _protected(Path(canonical).parts):
                raise LocalToolError('不允许工具访问凭据文件或目录', 403)
            return handle
        except BaseException:
            _kernel.CloseHandle(handle)
            raise


def _lock_temporary(descriptor):
    # Windows uses an exclusive handle; POSIX requires an advisory lock.
    if os.name != 'nt':
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


class _Directory:
    """A pinned parent; POSIX uses openat, Windows locks every ancestor."""
    def __init__(self, path, descriptor=None):
        self.path, self.descriptor = path, descriptor
        self.identity = _identity(path.stat() if descriptor is None else os.fstat(descriptor))

    def info(self, name):
        return ((self.path / name).lstat() if self.descriptor is None
                else os.stat(name, dir_fd=self.descriptor, follow_symlinks=False))

    def open_regular(self, name, *, deny_write=False, deny_delete=False, delete=False):
        before = self.info(name)
        if _is_link(before):
            raise LocalToolError('不允许通过符号链接、junction 或 reparse point 访问文件', 403)
        if not stat.S_ISREG(before.st_mode):
            raise LocalToolError('只能访问普通文件')
        if before.st_nlink != 1:
            raise LocalToolError('不允许访问多硬链接文件，无法确认其他链接是否位于工作目录内', 403)
        if os.name == 'nt':
            handle = _windows_handle(self.path / name, deny_write=deny_write, deny_delete=deny_delete, delete=delete)
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            except BaseException:
                _kernel.CloseHandle(handle)
                raise
        else:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.descriptor)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or _identity(before) != _identity(opened):
                raise LocalToolError('文件在打开期间发生变化，请重新读取', 409)
            return os.fdopen(descriptor, 'rb')
        except BaseException:
            os.close(descriptor)
            raise

    def snapshot(self, name, *, missing=False, byte_budget=None):
        try:
            source = self.open_regular(name)
        except FileNotFoundError:
            if missing:
                return None
            raise
        with source:
            before = os.fstat(source.fileno())
            budget = EDIT_MEMORY_BUDGET_BYTES // EDIT_MEMORY_MULTIPLIER if byte_budget is None else byte_budget
            if before.st_size > budget:
                raise _edit_budget_error()
            chunks, size = [], 0
            for raw, _ in _text_chunks(source):
                size += len(raw)
                if size > budget:
                    raise _edit_budget_error()
                chunks.append(raw)
            content = b''.join(chunks)
            after = os.fstat(source.fileno())
            if after.st_nlink != 1 or _version(before) != _version(after) or _identity(after) != _identity(self.info(name)):
                raise LocalToolError('文件在读取期间发生变化，请重新读取', 409)
            return _Snapshot(content, _version(after), stat.S_IMODE(after.st_mode))

    def matches(self, name, original, *, renamed=False):
        try:
            source = self.open_regular(name)
        except FileNotFoundError:
            return original is None
        with source:
            return self._source_matches(source, name, original, renamed=renamed)

    def _source_matches(self, source, name, original, *, renamed=False):
        if original is None:
            return False
        before = os.fstat(source.fileno())
        if renamed:
            if (_identity(before), before.st_size, before.st_mtime_ns, before.st_mode) != (
                    original.version[0], original.version[1], original.version[2], original.version[4]):
                return False
        elif _version(before) != original.version:
            return False
        offset = 0
        while chunk := source.read(IO_CHUNK_BYTES):
            if chunk != original.content[offset:offset + len(chunk)]:
                return False
            offset += len(chunk)
        after = os.fstat(source.fileno())
        return (offset == len(original.content) and after.st_nlink == 1 and _version(before) == _version(after)
                and _identity(before) == _identity(self.info(name)))

    def discard_confirmed(self, name, original):
        # The same handle denies concurrent writes from before comparison until
        # deletion. Delete by handle, not by the checked name: a concurrent
        # rename cannot make us unlink another file at that name.
        with self.open_regular(name, deny_write=True, delete=True) as source:
            if not self._source_matches(source, name, original, renamed=True):
                return False
            info = _DispositionInfo(1)
            if not _kernel.SetFileInformationByHandle(msvcrt.get_osfhandle(source.fileno()), 4,
                                                      ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            return True

    def temporary(self, name):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = (os.open(self.path / name, flags, 0o666) if self.descriptor is None
                      else os.open(name, flags, 0o666, dir_fd=self.descriptor))
        return os.fdopen(descriptor, 'wb')

    @contextmanager
    def temporary_lock(self, temporary, *, create=True):
        name = temporary.removesuffix('.tmp') + '.lock'
        if os.name == 'nt':
            handle = _windows_handle(self.path / name, create=create, exclusive=True)
            try:
                descriptor = msvcrt.open_osfhandle(handle, (os.O_RDWR if create else os.O_RDONLY) | os.O_BINARY)
            except BaseException:
                _kernel.CloseHandle(handle)
                raise
        else:
            descriptor = os.open(name, os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if create else 0),
                                 0o600, dir_fd=self.descriptor)
        acquired = False
        locked_identity = None
        try:
            _lock_temporary(descriptor)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or _identity(opened) != _identity(self.info(name)):
                raise LocalToolError('临时文件锁发生变化', 409)
            acquired, locked_identity = True, _identity(os.fstat(descriptor))
            yield
        finally:
            os.close(descriptor)
            if acquired:
                try:
                    if _identity(self.info(name)) == locked_identity:
                        self.unlink(name)
                except FileNotFoundError:
                    pass

    def recover_temporaries(self):
        with os.scandir(self.path if self.descriptor is None else self.descriptor) as entries:
            for entry in entries:
                if not TEMPORARY_NAME.fullmatch(entry.name) or not entry.name.endswith('.lock'):
                    continue
                try:
                    temporary = entry.name.removesuffix('.lock') + '.tmp'
                    with self.temporary_lock(temporary, create=False):
                        try:
                            info = self.info(temporary)
                        except FileNotFoundError:
                            continue
                        if stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not _is_link(info):
                            self.unlink(temporary)
                except (OSError, LocalToolError):
                    # A locked file is active; a link/special file is not ours.
                    continue

    def unlink(self, name):
        if self.descriptor is None:
            (self.path / name).unlink(missing_ok=True)
        else:
            try:
                os.unlink(name, dir_fd=self.descriptor)
            except FileNotFoundError:
                pass

    def replace(self, temporary, name, *, existing, backup=None):
        if os.name == 'nt':
            if existing:
                # ReplaceFile keeps the target's ACL and metadata. Failure does
                # not fall back to an in-place/truncating write.
                if not _kernel.ReplaceFileW(str(self.path / name), str(self.path / temporary),
                                           str(self.path / backup) if backup is not None else None, 0, None, None):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                os.rename(self.path / temporary, self.path / name)  # Windows: no-clobber.
        elif existing:
            # renameat2 exchange captures the exact entry replaced at the
            # commit boundary, not an earlier hard-link/snapshot of its name.
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            exchange = getattr(libc, 'renameat2', None)
            if exchange is None:
                raise LocalToolError('当前平台缺少安全原子交换，不能覆盖现有文件', 501)
            exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            exchange.restype = ctypes.c_int
            if exchange(self.descriptor, os.fsencode(temporary), self.descriptor, os.fsencode(name), 2) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            if backup is not None:
                os.rename(temporary, backup, src_dir_fd=self.descriptor, dst_dir_fd=self.descriptor)
            else:
                self.unlink(temporary)
        else:
            # link is an atomic, no-clobber installation of the completed file.
            os.link(temporary, name, src_dir_fd=self.descriptor, dst_dir_fd=self.descriptor, follow_symlinks=False)
            self.unlink(temporary)


@contextmanager
def _directory(root, children=()):
    try:
        with ExitStack() as stack:
            paths = [Path(root.anchor)]
            for part in (*root.parts[1:], *children):
                paths.append(paths[-1] / part)
            descriptor = None
            root_identity = None
            for index, path in enumerate(paths):
                if os.name == 'nt':
                    handle = _windows_handle(path, directory=True)
                    stack.callback(_kernel.CloseHandle, handle)
                else:
                    descriptor = os.open(path if index == 0 else path.name,
                                         os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                         **({} if index == 0 else {'dir_fd': descriptor}))
                    stack.callback(os.close, descriptor)
                if index == len(root.parts) - 1:
                    root_identity = _identity(path.stat() if os.name == 'nt' else os.fstat(descriptor))
            result = _Directory(paths[-1], descriptor)
            result.root_identity = root_identity
            result.recover_temporaries()
            yield result
    except FileNotFoundError as error:
        raise LocalToolError('文件或父目录不存在，请先选择已有目录', 404) from error
    except NotADirectoryError as error:
        raise LocalToolError('路径不是目录或包含符号链接', 403) from error
    except PermissionError as error:
        raise LocalToolError('无法访问文件或目录，请检查权限或文件占用', 403) from error
    except OSError as error:
        raise LocalToolError('文件操作失败：' + (error.strerror or '系统拒绝操作')) from error


@dataclass(frozen=True)
class _Snapshot:
    content: bytes
    version: tuple
    mode: int


@dataclass(frozen=True)
class _PreparedWrite:
    root: Path
    parts: tuple[str, ...]
    root_identity: tuple
    parent_identity: tuple
    original: _Snapshot | None
    replacement: bytes
    expires_at: float


def _decode(content):
    if content.startswith((b'\xff\xfe', b'\xfe\xff')) or b'\x00' in content:
        raise LocalToolError('只支持 UTF-8 文本文件，不支持二进制或 UTF-16 文件', 415)
    try:
        return content.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise LocalToolError('文件不是有效的 UTF-8 文本', 415) from error


def _entry_exists(anchor, name):
    try:
        anchor.info(name)
        return True
    except FileNotFoundError:
        return False


def _edit_budget_error():
    return LocalToolError(f'完整编辑预览超出 {EDIT_MEMORY_BUDGET_BYTES / 1024 / 1024:g} MiB 内存预算，'
                          '请改用分页读取或缩小编辑文件', 413)


def _text_chunks(source):
    decoder = codecs.getincrementaldecoder('utf-8-sig')()
    first = True
    try:
        while raw := source.read(IO_CHUNK_BYTES):
            if (first and raw.startswith((b'\xff\xfe', b'\xfe\xff'))) or b'\x00' in raw:
                raise LocalToolError('只支持 UTF-8 文本文件，不支持二进制或 UTF-16 文件', 415)
            first = False
            yield raw, decoder.decode(raw)
        final = decoder.decode(b'', final=True)
        if final:
            yield b'', final
    except UnicodeDecodeError as error:
        raise LocalToolError('文件不是有效的 UTF-8 文本', 415) from error


def _line_parts(source):
    pending_cr = ''
    for _, text in _text_chunks(source):
        text = pending_cr + text
        pending_cr = ''
        if text.endswith('\r'):
            text, pending_cr = text[:-1], '\r'
        for part in re.split(r'(\r\n|\r|\n)', text):
            if part:
                yield part, part in ('\r', '\n', '\r\n')
    if pending_cr:
        yield pending_cr, True


class LocalTools:
    def __init__(self):
        self._pending: dict[str, _PreparedWrite] = {}
        self._lock = threading.RLock()
        self._timer = None
        self._closed = False

    def _expire(self):
        with self._lock:
            self._timer = None
            now = time.monotonic()
            self._pending = {key: value for key, value in self._pending.items() if value.expires_at > now}
            if self._pending and not self._closed:
                delay = max(0, min(value.expires_at for value in self._pending.values()) - now)
                self._timer = threading.Timer(delay, self._expire)
                self._timer.daemon = True
                self._timer.start()

    def close(self):
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()

    def list(self, directory, path='.', offset=0, limit=100):
        root, parts = _root(directory), _relative('.' if path == '' else path, allow_root=True)
        offset, limit = _integer(offset, 'offset', 0), _integer(limit, 'limit', 1)
        entries = []
        with _directory(root, parts) as anchor:
            with os.scandir(anchor.path if anchor.descriptor is None else anchor.descriptor) as scanned:
                for entry in scanned:
                    if TEMPORARY_NAME.fullmatch(entry.name):
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:  # A concurrently removed entry is not a listing failure.
                        continue
                    kind = 'link' if _is_link(info) else 'directory' if stat.S_ISDIR(info.st_mode) else 'file' if stat.S_ISREG(info.st_mode) else 'other'
                    item = {'name': entry.name, 'path': '/'.join((*parts, entry.name)), 'type': kind}
                    if kind == 'file':
                        item['size'] = info.st_size
                    if (kind in ('link', 'other') or (kind == 'file' and info.st_nlink != 1)
                            or _protected((*root.parts, *parts, entry.name)) or entry.name.startswith(RECOVERY_PREFIX)):
                        item['blocked'] = True
                    entries.append(item)
        entries.sort(key=lambda item: (item['type'] != 'directory', item['name'].casefold(), item['name']))
        end = offset + limit
        return {'path': '/'.join(parts) or '.', 'entries': entries[offset:end], 'total_entries': len(entries),
                'next_offset': end if end < len(entries) else None}

    def read(self, directory, path, start_line=1, line_count=80, start_column=1):
        root, parts = _root(directory), _relative(path)
        start_line, line_count = _integer(start_line, 'start_line', 1), _integer(line_count, 'line_count', 1)
        start_column = _integer(start_column, 'start_column', 1)
        selected, total, line, column = [], 0, 1, 1
        remaining, next_position = READ_PAGE_CHARACTERS, None
        end_line, end_column = start_line - 1, start_column
        with _directory(root, parts[:-1]) as anchor:
            with anchor.open_regular(parts[-1]) as source:
                before = os.fstat(source.fileno())
                beginning = source.read(3)
                encoding = 'utf-8-sig' if beginning.startswith(b'\xef\xbb\xbf') else 'utf-8'
                source.seek(0)
                for part, newline in _line_parts(source):
                    total = line
                    if start_line <= line < start_line + line_count and next_position is None:
                        skip = max(0, start_column - column) if line == start_line else 0
                        piece = part if newline and skip == 0 else '' if newline else part[skip:]
                        if piece:
                            take = min(remaining, len(piece))
                            if newline and take < len(piece):
                                take = 0  # Do not split a CRLF pair across pages.
                            if take:
                                selected.append(piece[:take])
                                end_line, end_column = line, column + skip + take - 1
                                remaining -= take
                            if take < len(piece):
                                next_position = line, column + skip + take
                    elif line >= start_line + line_count and next_position is None:
                        next_position = line, 1
                    if newline:
                        line, column = line + 1, 1
                    else:
                        column += len(part)
                after = os.fstat(source.fileno())
                if after.st_nlink != 1 or _version(before) != _version(after) or _identity(after) != _identity(anchor.info(parts[-1])):
                    raise LocalToolError('文件在读取期间发生变化，请重新读取', 409)
        return {'path': '/'.join(parts), 'content': ''.join(selected), 'start_line': start_line,
                'start_column': start_column, 'end_line': end_line, 'end_column': end_column, 'total_lines': total,
                'next_start_line': next_position[0] if next_position else None,
                'next_start_column': next_position[1] if next_position else None,
                'has_more': next_position is not None, 'encoding': encoding}

    def prepare_write(self, directory, path, *, content=None, edits=None):
        root, parts = _root(directory), _relative(path)
        if (content is None) == (edits is None):
            raise LocalToolError('请提供 content 或 edits，不能同时提供')
        with self._lock, _directory(root, parts[:-1]) as anchor:
            if self._closed:
                raise LocalToolError('应用正在退出', 503)
            now = time.monotonic()
            self._pending = {key: value for key, value in self._pending.items() if value.expires_at > now}
            held_bytes = sum(len(value.replacement) + (len(value.original.content) if value.original else 0)
                             for value in self._pending.values())
            byte_budget = EDIT_MEMORY_BUDGET_BYTES // EDIT_MEMORY_MULTIPLIER - held_bytes
            incoming_bytes = 0
            if content is not None:
                if not isinstance(content, str) or '\x00' in content:
                    raise LocalToolError('content 必须是 UTF-8 文本，不能包含 NUL 字符')
                if len(content) > byte_budget:
                    raise _edit_budget_error()
                try:
                    incoming_bytes = len(content.encode('utf-8'))
                except UnicodeEncodeError as error:
                    raise LocalToolError('content 包含无效的 Unicode 字符') from error
            original = anchor.snapshot(parts[-1], missing=True, byte_budget=max(0, byte_budget - incoming_bytes))
            original_content = _decode(original.content) if original is not None else None
            if edits is not None:
                if original is None:
                    raise LocalToolError('编辑目标不存在；创建文件请使用 content', 404)
                if not isinstance(edits, list) or not edits:
                    raise LocalToolError('edits 必须是非空的精确替换数组')
                content = original_content
                for edit in edits:
                    if not isinstance(edit, dict) or set(edit) != {'old_text', 'new_text'}:
                        raise LocalToolError('每项编辑必须包含 old_text 和 new_text')
                    old, new = edit['old_text'], edit['new_text']
                    if not isinstance(old, str) or not old or not isinstance(new, str):
                        raise LocalToolError('old_text 必须是非空文本，new_text 必须是文本')
                    first = content.find(old)
                    if first < 0 or content.find(old, first + 1) >= 0:
                        raise LocalToolError('精确替换未唯一匹配，请提供只出现一处的 old_text', 409)
                    if len(content) - len(old) + len(new) > byte_budget:
                        raise _edit_budget_error()
                    content = content.replace(old, new, 1)
            if not isinstance(content, str) or '\x00' in content:
                raise LocalToolError('content 必须是 UTF-8 文本，不能包含 NUL 字符')
            try:
                replacement = content.encode('utf-8')
            except UnicodeEncodeError as error:
                raise LocalToolError('content 包含无效的 Unicode 字符') from error
            bom = original is not None and original.content.startswith(b'\xef\xbb\xbf')
            if bom:
                replacement = b'\xef\xbb\xbf' + replacement
            if len(replacement) + (len(original.content) if original else 0) > byte_budget:
                raise _edit_budget_error()
            identifier = secrets.token_urlsafe(32)
            self._pending[identifier] = _PreparedWrite(root, parts, anchor.root_identity, anchor.identity, original,
                                                       replacement, time.monotonic() + PREPARATION_TTL_SECONDS)
            if self._timer is None:
                self._timer = threading.Timer(PREPARATION_TTL_SECONDS, self._expire)
                self._timer.daemon = True
                self._timer.start()
        return {'preparation_id': identifier, 'directory': str(root), 'path': '/'.join(parts),
                'operation': 'edit' if edits is not None else 'overwrite' if original is not None else 'create',
                'original_content': original_content, 'content': content, 'expires_in_seconds': PREPARATION_TTL_SECONDS,
                'encoding': 'utf-8-sig' if bom else 'utf-8'}

    def cancel(self, identifier):
        with self._lock:
            result = self._pending.pop(identifier, None) is not None
            self._stop_empty_timer()
            return {'cancelled': result}

    def _stop_empty_timer(self):
        if not self._pending and self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _commit_existing(self, anchor, temporary, name, prepared):
        identifier = temporary[len(TEMPORARY_PREFIX):].removesuffix('.tmp')
        backup = RECOVERY_PREFIX + identifier + '.previous'
        restore_backup = RECOVERY_PREFIX + identifier + '.displaced'
        if os.name != 'nt':
            # An exchanged-out entry must already have a recovery name if the
            # process exits immediately after renameat2 succeeds.
            anchor.replace(temporary, backup, existing=False)
            temporary = backup
        try:
            anchor.replace(temporary, name, existing=True, backup=backup)
            try:
                confirmed = (anchor.discard_confirmed(backup, prepared.original) if os.name == 'nt'
                             else anchor.matches(backup, prepared.original, renamed=True))
            except OSError as error:
                path = str(anchor.path / backup)
                raise LocalToolError('文件已替换，旧文件仍被其他程序占用；恢复副本已保留：' + path,
                                     409, recovery_paths=[path]) from error
            except LocalToolError:
                confirmed = False
            if confirmed:
                # POSIX has no Windows-style mandatory deny-write sharing.
                # Keep its recovery entry rather than deleting an open writer's
                # file after comparison.
                return [] if os.name == 'nt' else [str(anchor.path / backup)]
        except OSError as error:
            if not _entry_exists(anchor, backup):
                raise
            paths = [str(anchor.path / backup)]
            raise LocalToolError('原子替换未完成，请检查恢复副本：' + paths[0], 409, recovery_paths=paths) from error

        # Preserve the approved pair as well as every unconfirmed entry moved
        # by the rollback. These are recovery data, never orphan-temp cleanup.
        recovery = [backup]
        for suffix, data in (('.approved-original', prepared.original.content), ('.proposed', prepared.replacement)):
            saved = RECOVERY_PREFIX + identifier + suffix
            try:
                with anchor.temporary(saved) as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                recovery.append(saved)
            except OSError:
                if _entry_exists(anchor, saved):
                    recovery.append(saved)
        restored = False
        try:
            anchor.replace(backup, name, existing=True, backup=restore_backup)
            recovery.append(restore_backup)
            restored = True
        except (OSError, LocalToolError):
            pass
        paths = [str(anchor.path / item) for item in recovery if _entry_exists(anchor, item)]
        message = ('文件在最终写入时被其他程序修改，已回滚；请检查恢复副本：' if restored
                   else '文件在最终写入时被其他程序修改，自动回滚未完成；请从恢复副本恢复：')
        raise LocalToolError(message + '；'.join(paths), 409, recovery_paths=paths)

    def write(self, preparation_id, *, approved=False):
        if approved is not True:
            raise LocalToolError('写入前需要用户确认', 403)
        if not isinstance(preparation_id, str) or not preparation_id:
            raise LocalToolError('preparation_id 不能为空')
        with self._lock:
            prepared = self._pending.pop(preparation_id, None)
            self._stop_empty_timer()
            if prepared is None or prepared.expires_at <= time.monotonic():
                raise LocalToolError('写入确认已失效，请重新准备并确认', 409)
            with _directory(prepared.root, prepared.parts[:-1]) as anchor:
                if (anchor.root_identity, anchor.identity) != (prepared.root_identity, prepared.parent_identity):
                    raise LocalToolError('工作目录在确认期间发生变化，请重新准备并确认', 409)
                retained = []
                if not anchor.matches(prepared.parts[-1], prepared.original):
                    raise LocalToolError('文件在确认期间发生变化，未写入；请重新读取并确认', 409)
                temporary = TEMPORARY_PREFIX + secrets.token_hex(16) + '.tmp'
                with anchor.temporary_lock(temporary):
                    try:
                        with anchor.temporary(temporary) as target:
                            target.write(prepared.replacement)
                            target.flush()
                            if os.name != 'nt' and prepared.original is not None:
                                os.fchmod(target.fileno(), prepared.original.mode)
                            os.fsync(target.fileno())
                        # Recheck after preparing the temporary, not only before it.
                        if not anchor.matches(prepared.parts[-1], prepared.original):
                            raise LocalToolError('文件在写入期间发生变化，未写入；请重新读取并确认', 409)
                        if prepared.original is not None:
                            retained = self._commit_existing(anchor, temporary, prepared.parts[-1], prepared)
                        else:
                            anchor.replace(temporary, prepared.parts[-1], existing=False)
                    finally:
                        anchor.unlink(temporary)
        result = {'path': '/'.join(prepared.parts), 'written': True, 'bytes_written': len(prepared.replacement)}
        if retained:
            result['recovery_paths'] = retained
        return result
