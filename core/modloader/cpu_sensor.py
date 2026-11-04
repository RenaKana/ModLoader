"""Per-parent CPU/GPU Windows sensor reader with an explicit enable operation.

Normal polling never elevates. Only ``enable`` asks Windows to start the bundled
helper; its private pipe and process handle belong to this reader instance.
"""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as w
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid


STARTUP_TIMEOUT_S = 30.0
UAC_TIMEOUT_S = 60.0
SAMPLE_TIMEOUT_S = 8.0
# JavaScriptSerializer accepts 2,097,152 UTF-16 code units by default. UTF-8
# needs at most three bytes per code unit, plus the newline. Match that actual
# helper format limit rather than imposing a GPU-count limit on its array.
MAX_REPLY_BYTES = 3 * 2_097_152 + 2
_STATUSES = {'disabled', 'driver_missing', 'permission_required', 'starting', 'ready', 'error'}
_INVALID_HANDLE = c.c_void_p(-1).value


def _result(status, reason=None, value=None, source=None, gpu_temperatures=None, gpu_temperature_reason=None,
            power=None, power_source=None, power_reason=None):
    return {'cpu_temperature_c': value, 'cpu_temperature_source': source,
            'cpu_temperature_reason': reason, 'cpu_temperature_status': status,
            'gpu_temperatures': [] if gpu_temperatures is None else gpu_temperatures,
            'gpu_temperature_reason': gpu_temperature_reason,
            'cpu_power_w': power, 'cpu_power_source': power_source,
            'cpu_power_reason': power_reason if power is not None else power_reason or reason or 'CPU 未提供封装功率传感器'}


class _ReaderError(Exception):
    def __init__(self, reason, status='error'):
        super().__init__(reason)
        self.status = status


class _ShellExecuteInfo(c.Structure):
    # SHELLEXECUTEINFOW; the hIcon/hMonitor union has the size of one HANDLE.
    _fields_ = [('cbSize', w.DWORD), ('fMask', w.ULONG), ('hwnd', w.HWND),
                ('lpVerb', w.LPCWSTR), ('lpFile', w.LPCWSTR), ('lpParameters', w.LPCWSTR),
                ('lpDirectory', w.LPCWSTR), ('nShow', c.c_int), ('hInstApp', w.HINSTANCE),
                ('lpIDList', c.c_void_p), ('lpClass', w.LPCWSTR), ('hkeyClass', w.HKEY),
                ('dwHotKey', w.DWORD), ('hIconOrMonitor', w.HANDLE), ('hProcess', w.HANDLE)]


class _WindowsApi:
    """Small Win32 binding. No service, driver-install, or process-kill APIs."""
    def __init__(self):
        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        self.shell = c.WinDLL('shell32', use_last_error=True)
        signatures = {
            'CloseHandle': ([w.HANDLE], w.BOOL),
            'GetProcessId': ([w.HANDLE], w.DWORD),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'CreateFileW': ([w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE], w.HANDLE),
            'GetNamedPipeServerProcessId': ([w.HANDLE, c.POINTER(w.ULONG)], w.BOOL),
            'SetNamedPipeHandleState': ([w.HANDLE, c.POINTER(w.DWORD), c.POINTER(w.DWORD), c.POINTER(w.DWORD)], w.BOOL),
            'ReadFile': ([w.HANDLE, c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p], w.BOOL),
            'WriteFile': ([w.HANDLE, c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.kernel, name)
            fn.argtypes, fn.restype = args, result
        self.shell.ShellExecuteExW.argtypes = [c.POINTER(_ShellExecuteInfo)]
        self.shell.ShellExecuteExW.restype = w.BOOL

    def launch(self, helper, pipe_name):
        info = _ShellExecuteInfo()
        info.cbSize = c.sizeof(info)
        info.fMask = 0x40 | 0x100 | 0x400  # NOCLOSEPROCESS | NOASYNC | FLAG_NO_UI
        info.lpVerb, info.lpFile = 'runas', str(helper)
        info.lpParameters = subprocess.list2cmdline(['--pipe', pipe_name, '--parent-pid', str(os.getpid())])
        info.lpDirectory, info.nShow = str(helper.parent), 0
        if not self.shell.ShellExecuteExW(c.byref(info)):
            error = c.get_last_error()
            if error == 1223:  # ERROR_CANCELLED: the user declined UAC.
                raise _ReaderError('已取消管理员授权；可再次启用硬件温度', 'permission_required')
            raise _ReaderError(f'硬件温度读取器启动失败（Windows 错误 {error}）')
        if not info.hProcess:
            raise _ReaderError('Windows 未返回硬件温度读取器的进程句柄')
        pid = self.kernel.GetProcessId(info.hProcess)
        if not pid:
            self.close_handle(info.hProcess)
            raise _ReaderError('无法确认硬件温度读取器的进程身份')
        return info.hProcess, pid

    def close_handle(self, handle):
        if handle and handle != _INVALID_HANDLE:
            self.kernel.CloseHandle(handle)

    def running(self, handle):
        state = self.kernel.WaitForSingleObject(handle, 0)
        if state == 0xFFFFFFFF:
            raise _ReaderError('无法查询硬件温度读取器的进程状态')
        return state == 0x102  # WAIT_TIMEOUT means still alive.

    def connect(self, name, pid, process, stopping):
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not stopping.is_set():
            if not self.running(process):
                raise _ReaderError('硬件温度读取器已退出，未建立连接；请检查读取器及 .NET 运行库')
            # Identification-only SQOS: this client does not grant impersonation.
            handle = self.kernel.CreateFileW('\\\\.\\pipe\\' + name, 0xC0000000, 0,
                                             None, 3, 0x100000 | 0x10000, None)
            if handle != _INVALID_HANDLE:
                try:
                    server_pid = w.ULONG()
                    if not self.kernel.GetNamedPipeServerProcessId(handle, c.byref(server_pid)) or server_pid.value != pid:
                        raise _ReaderError('硬件温度读取器的管道进程身份不匹配')
                    # PIPE_NOWAIT keeps both reads and tiny command writes bounded.
                    # The server remains a blocking, byte-mode named pipe.
                    mode = w.DWORD(1)
                    if not self.kernel.SetNamedPipeHandleState(handle, c.byref(mode), None, None):
                        raise _ReaderError(f'无法设置硬件温度读取管道（Windows 错误 {c.get_last_error()}）')
                    return _Pipe(self, handle, stopping)
                except Exception:
                    self.close_handle(handle)
                    raise
            error = c.get_last_error()
            if error not in (2, 231):  # Not created yet / pipe busy.
                raise _ReaderError(f'无法连接硬件温度读取器（Windows 错误 {error}）')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _ReaderError('硬件温度读取器连接超时；请重新启用')
            stopping.wait(min(0.05, remaining))
        raise _ReaderError('硬件温度读取已关闭', 'disabled')


class _Pipe:
    def __init__(self, api, handle, stopping):
        self.api, self.handle, self.stopping = api, handle, stopping

    def _pause(self, deadline, wait=True):
        if self.stopping.is_set():
            raise _ReaderError('硬件温度读取已关闭', 'disabled')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _ReaderError('硬件温度读取超时；请重新启用')
        if wait:
            self.stopping.wait(min(0.02, remaining))

    def send(self, command, deadline):
        payload = command.encode('ascii') + b'\n'
        while payload:
            count = w.DWORD()
            data = c.create_string_buffer(payload)
            if not self.api.kernel.WriteFile(self.handle, data, len(payload), c.byref(count), None):
                raise _ReaderError(f'硬件温度读取管道已断开（Windows 错误 {c.get_last_error()}）')
            payload = payload[count.value:]
            if payload:
                self._pause(deadline)

    def request(self):
        deadline = time.monotonic() + SAMPLE_TIMEOUT_S
        self.send('sample', deadline)
        received = bytearray()
        while True:
            buffer, count = c.create_string_buffer(4096), w.DWORD()
            ok = self.api.kernel.ReadFile(self.handle, buffer, len(buffer), c.byref(count), None)
            if not ok:
                error = c.get_last_error()
                if error != 232:  # ERROR_NO_DATA in non-blocking mode.
                    raise _ReaderError(f'硬件温度读取管道已断开（Windows 错误 {error}）')
            elif count.value == 0:
                raise _ReaderError('硬件温度读取器已关闭连接')
            received.extend(buffer.raw[:count.value])
            if len(received) > MAX_REPLY_BYTES:
                raise _ReaderError('硬件温度读取器返回了过长的响应')
            if b'\n' in received:
                line, trailing = received.split(b'\n', 1)
                if trailing:
                    raise _ReaderError('硬件温度读取器返回了多余的响应')
                return _decode_reply(line)
            self._pause(deadline, wait=not ok)

    def close(self):
        if self.handle is not None:
            try:
                self.send('close', time.monotonic() + 0.2)
            except _ReaderError:
                pass
            finally:
                self.api.close_handle(self.handle)
                self.handle = None


def _decode_reply(line):
    try:
        value = json.loads(line.decode('utf-8'))
        status, temperature, source, reason = (value[key] for key in (
            'cpu_temperature_status', 'cpu_temperature_c', 'cpu_temperature_source', 'cpu_temperature_reason'))
        if status not in _STATUSES or (reason is not None and not isinstance(reason, str)):
            raise ValueError
        if status == 'ready':
            if type(temperature) not in (int, float) or not math.isfinite(temperature) or not isinstance(source, str) or not source or reason is not None:
                raise ValueError
        elif temperature is not None or source is not None or not reason:
            raise ValueError
        gpus = value.get('gpu_temperatures', [])  # Older CPU-only helpers remain readable.
        gpu_reason = value.get('gpu_temperature_reason')
        if not isinstance(gpus, list) or (gpu_reason is not None and not isinstance(gpu_reason, str)):
            raise ValueError
        normalized = []
        for gpu in gpus:
            identifier, name, gpu_temperature, gpu_source, unavailable, pci = (gpu[key] for key in (
                'identifier', 'name', 'temperature_c', 'source', 'reason', 'pci_bus_id'))
            if not isinstance(identifier, str) or not identifier or not isinstance(name, str) or not name:
                raise ValueError
            if gpu_source not in (None, 'GPU Core') or (unavailable is not None and not isinstance(unavailable, str)):
                raise ValueError
            if gpu_temperature is None:
                if not unavailable:
                    raise ValueError
            elif type(gpu_temperature) not in (int, float) or not math.isfinite(gpu_temperature) or gpu_source != 'GPU Core' or unavailable is not None:
                raise ValueError
            if pci is not None and (not isinstance(pci, str) or not re.fullmatch(r'[0-9a-fA-F]{2}:[01][0-9a-fA-F]\.[0-7]', pci)):
                raise ValueError
            normalized.append({'identifier': identifier, 'name': name, 'temperature_c': gpu_temperature,
                               'source': gpu_source, 'reason': unavailable, 'pci_bus_id': pci.lower() if pci else None})
        power = value.get('cpu_power_w')
        power_source = value.get('cpu_power_source')
        power_reason = value.get('cpu_power_reason', reason or 'CPU 未提供封装功率传感器')
        if power is not None:
            if type(power) not in (int, float) or not math.isfinite(power) or power < 0 or not isinstance(power_source, str) or not power_source or power_reason is not None:
                raise ValueError
        elif power_source is not None or not isinstance(power_reason, str) or not power_reason:
            raise ValueError
        return _result(status, reason, temperature, source, normalized, gpu_reason, power, power_source, power_reason)
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError):
        raise _ReaderError('硬件温度读取器返回了无效响应') from None


def _bundled_helper():
    directory = Path(__file__).resolve().parent / 'sensors'
    helper = (directory / 'ModLoader.CpuSensors.exe').resolve()
    if helper.parent != directory or not helper.is_file():
        raise _ReaderError('缺少内置硬件温度读取器，请重新安装完整的 ModLoader')
    return helper


class CpuSensorReader:
    def __init__(self):
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._api = self._pipe = self._process = None
        self._pending_launch = None
        self._state = _result('disabled', '尚未启用硬件温度读取')

    def sample(self):
        """Poll an existing connection only. This method never prompts for UAC."""
        if not self._lock.acquire(blocking=False):
            return self._state.copy()
        try:
            return self._sample_locked()
        finally:
            if self._stopping.is_set():
                self._disconnect()
                self._state = _result('disabled', '硬件温度读取已关闭')
            self._lock.release()

    def enable(self):
        """Explicit UAC action; start only the helper bundled with this module."""
        if not self._lock.acquire(blocking=False):
            return self._state.copy()
        try:
            if self._stopping.is_set():
                return _result('disabled', '硬件温度读取已关闭')
            if self._pipe is not None and self._state.get('gpu_temperature_reason'):
                # A failed GPU group cannot be reinitialized by LHM polling.
                # Only this explicit retry replaces our existing helper.
                self._disconnect()
            if self._pipe is not None:
                return self._sample_locked()
            if self._pending_launch is not None and not self._pending_launch.is_set():
                self._state = _result('permission_required', '上一次管理员授权仍在等待；请先处理系统授权窗口，再重新启用')
                return self._state.copy()
            if os.name != 'nt':
                self._state = _result('error', '内置硬件温度读取器仅支持 Windows')
                return self._state.copy()
            self._state = _result('starting', '正在等待管理员授权并连接硬件温度读取器')
            try:
                helper = _bundled_helper()
                self._api = _WindowsApi()
                name = 'ModLoader.CpuSensors.' + uuid.uuid4().hex
                self._process, pid = self._launch_helper(helper, name)
                if self._stopping.is_set():
                    raise _ReaderError('硬件温度读取已关闭', 'disabled')
                self._pipe = self._api.connect(name, pid, self._process, self._stopping)
                return self._sample_locked()
            except _ReaderError as error:
                self._disconnect()
                self._state = _result(error.status, str(error))
            except OSError as error:
                self._disconnect()
                code = getattr(error, 'winerror', None) or error.errno
                self._state = _result('error', f'硬件温度读取器不可用（Windows 错误 {code}）')
            return self._state.copy()
        finally:
            if self._stopping.is_set():
                self._disconnect()
                self._state = _result('disabled', '硬件温度读取已关闭')
            self._lock.release()

    def _launch_helper(self, helper, name):
        # ShellExecuteEx can wait for an unanswered secure-desktop UAC prompt.
        # Keep that wait outside asyncio's executor, and release a late result
        # without retaining an unusable elevated process handle.
        api, stopping = self._api, self._stopping
        completed, delivery_lock = threading.Event(), threading.Lock()
        delivery = {'abandoned': False, 'outcome': None}
        self._pending_launch = completed

        def launch():
            try:
                outcome = api.launch(helper, name)
            except Exception as error:
                outcome = error
            with delivery_lock:
                if delivery['abandoned'] or stopping.is_set():
                    if not isinstance(outcome, Exception):
                        api.close_handle(outcome[0])
                else:
                    delivery['outcome'] = outcome
                completed.set()

        threading.Thread(target=launch, name='cpu-sensor-uac', daemon=True).start()
        deadline = time.monotonic() + UAC_TIMEOUT_S
        while True:
            if stopping.is_set() or time.monotonic() >= deadline:
                with delivery_lock:
                    delivery['abandoned'] = True
                    outcome = delivery['outcome']
                    if outcome is not None and not isinstance(outcome, Exception):
                        api.close_handle(outcome[0])
                    delivery['outcome'] = None
                if stopping.is_set():
                    raise _ReaderError('硬件温度读取已关闭', 'disabled')
                raise _ReaderError('管理员授权等待超时；请先处理系统授权窗口，再重新启用', 'permission_required')
            if completed.wait(0.05):
                with delivery_lock:
                    outcome = delivery['outcome']
                    delivery['outcome'] = None
                if stopping.is_set():
                    if outcome is not None and not isinstance(outcome, Exception):
                        api.close_handle(outcome[0])
                    raise _ReaderError('硬件温度读取已关闭', 'disabled')
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

    def _sample_locked(self):
        if self._stopping.is_set():
            self._state = _result('disabled', '硬件温度读取已关闭')
        elif self._pipe is not None:
            try:
                self._state = self._pipe.request()
                if self._state['cpu_temperature_status'] != 'ready' and self._state.get('cpu_power_w') is None and not self._state.get('gpu_temperatures'):
                    self._disconnect()
            except _ReaderError as error:
                self._disconnect()
                self._state = _result(error.status, str(error))
        return self._state.copy()

    def _disconnect(self):
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None
        if self._process is not None:
            self._api.close_handle(self._process)
            self._process = None
        self._api = None

    def close(self):
        # Do not block app shutdown behind a pending UAC dialog. The enabling
        # thread will see this flag and release its own handles when it returns.
        self._stopping.set()
        if self._lock.acquire(blocking=False):
            try:
                self._disconnect()
                self._state = _result('disabled', '硬件温度读取已关闭')
            finally:
                self._lock.release()
