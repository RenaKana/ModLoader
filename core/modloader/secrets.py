"""Windows user-bound encrypted credentials. No global login files are read."""
import base64
import ctypes
import json
import os
import secrets
import threading
from pathlib import Path
from ctypes import wintypes


class Blob(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def crypt(data: bytes, decrypt=False) -> bytes:
    if os.name != 'nt':
        raise RuntimeError('凭据存储需要 Windows DPAPI')
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source, result = Blob(len(data), buffer), Blob()
    dll = ctypes.WinDLL('crypt32', use_last_error=True)
    fn = dll.CryptUnprotectData if decrypt else dll.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(result.data, result.length)
    finally:
        free = ctypes.WinDLL('kernel32').LocalFree
        free.argtypes = [ctypes.c_void_p]
        free(ctypes.cast(result.data, ctypes.c_void_p))


class Vault:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.values = json.loads(path.read_text('utf-8')) if path.exists() else {}

    def get(self, key):
        with self.lock:
            raw = self.values.get(key)
            return crypt(base64.b64decode(raw), decrypt=True).decode('utf-8') if raw else None

    def set(self, key, value):
        with self.lock:
            if value:
                self.values[key] = base64.b64encode(crypt(value.encode('utf-8'))).decode('ascii')
            else:
                self.values.pop(key, None)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.values), encoding='utf-8')
            temporary.replace(self.path)

    def ensure(self, key):
        value = self.get(key)
        if not value:
            value = secrets.token_urlsafe(32)
            self.set(key, value)
        return value
