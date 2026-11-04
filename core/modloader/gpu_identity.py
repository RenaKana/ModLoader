"""CUDA ordinals and physical identities, without contexts or allocations.

This file is also sent to a selected WSL distribution's stdlib Python. Keep it
independent of ModLoader packages and inference libraries.
"""
from __future__ import annotations

import ctypes as c
import json
import os
import re
import subprocess
import sys
import uuid


def pci_address(value):
    match = re.fullmatch(r'([0-9a-f]{4,8}):([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])',
                         str(value or '').strip().lower())
    return f'{int(match[1], 16):04x}:{match[2]}' if match else None


def gpu_uuid(value):
    value = str(value or '').strip()
    return value if value.lower().startswith(('gpu-', 'mig-')) else None


def cuda_devices():
    """Return the calling process's CUDA-visible order and UUID/PCI identities."""
    try:
        lib = c.CDLL('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
        lib.cuInit.argtypes, lib.cuInit.restype = [c.c_uint], c.c_int
        lib.cuDeviceGetCount.argtypes, lib.cuDeviceGetCount.restype = [c.POINTER(c.c_int)], c.c_int
        lib.cuDeviceGet.argtypes, lib.cuDeviceGet.restype = [c.POINTER(c.c_int), c.c_int], c.c_int
        if lib.cuInit(0) != 0:
            return []
        count = c.c_int()
        if lib.cuDeviceGetCount(c.byref(count)) != 0:
            return []
        get_uuid = getattr(lib, 'cuDeviceGetUuid_v2', None) or getattr(lib, 'cuDeviceGetUuid', None)
        if get_uuid:
            get_uuid.argtypes, get_uuid.restype = [c.c_void_p, c.c_int], c.c_int
        get_pci = getattr(lib, 'cuDeviceGetPCIBusId', None)
        if get_pci:
            get_pci.argtypes, get_pci.restype = [c.c_char_p, c.c_int, c.c_int], c.c_int
        get_capability = getattr(lib, 'cuDeviceComputeCapability', None)
        if get_capability:
            get_capability.argtypes, get_capability.restype = [c.POINTER(c.c_int), c.POINTER(c.c_int), c.c_int], c.c_int
        get_total = getattr(lib, 'cuDeviceTotalMem_v2', None) or getattr(lib, 'cuDeviceTotalMem', None)
        if get_total:
            get_total.argtypes, get_total.restype = [c.POINTER(c.c_size_t), c.c_int], c.c_int
        devices = []
        for ordinal in range(count.value):
            handle = c.c_int()
            if lib.cuDeviceGet(c.byref(handle), ordinal) != 0:
                continue
            raw_uuid, bus = (c.c_ubyte * 16)(), c.create_string_buffer(32)
            gpu_id = None
            if get_uuid and get_uuid(c.byref(raw_uuid), handle.value) == 0 and any(raw_uuid):
                gpu_id = 'GPU-' + str(uuid.UUID(bytes=bytes(raw_uuid)))
            pci = bus.value.decode('ascii') if get_pci and get_pci(bus, len(bus), handle.value) == 0 else None
            device = dict(device=f'CUDA{ordinal}', gpu_id=gpu_id, pci_bus_id=pci_address(pci))
            major, minor = c.c_int(), c.c_int()
            if get_capability and get_capability(c.byref(major), c.byref(minor), handle.value) == 0:
                device['compute_capability'] = [major.value, minor.value]
            total = c.c_size_t()
            if get_total and get_total(c.byref(total), handle.value) == 0 and total.value:
                device['memory_total'] = total.value
            devices.append(device)
        return devices
    except (OSError, AttributeError):
        return []


def matching_device(identity, devices):
    """A UUID conflict must never be repaired by guessing from an index/PCI."""
    key = gpu_uuid(identity.get('gpu_id') or identity.get('id'))
    ids = [gpu_uuid(item.get('gpu_id') or item.get('id')) for item in devices]
    if key:
        matches = [item for item, other in zip(devices, ids) if other and other.lower() == key.lower()]
        if matches:
            return matches[0] if len(matches) == 1 else None
    pci = pci_address(identity.get('pci_bus_id'))
    if not pci:
        return None
    matches = [item for item, other in zip(devices, ids)
               if pci_address(item.get('pci_bus_id')) == pci and (not key or not other)]
    return matches[0] if len(matches) == 1 else None


def identity_fields(ordinal, devices):
    matches = [item for item in devices if item.get('device') == f'CUDA{ordinal}']
    if len(matches) == 1:
        identity = matches[0]
        gpu_id, pci = gpu_uuid(identity.get('gpu_id')), pci_address(identity.get('pci_bus_id'))
        if (gpu_id or pci) and matching_device(identity, devices) is identity:
            return dict(identity_status='resolved', logical_device=f'CUDA{ordinal}', **({'gpu_id': gpu_id} if gpu_id else {}),
                        **({'pci_bus_id': pci} if pci else {}))
    return dict(identity_status='unknown')


def hardware_devices(devices, identities):
    result = []
    for device in devices:
        match = matching_device(device, identities)
        result.append({**device, 'device': match['device'] if match else None})
    return result


def cuda_devices_in_environment(environment, python=None, diagnostics=None):
    # CUDA latches visibility at cuInit. A different launch environment must be
    # enumerated in a fresh process, never by changing the monitor's environment.
    if not environment:
        return cuda_devices()
    if not python and getattr(sys, 'frozen', False):
        if diagnostics is not None:
            diagnostics.append('GPU 身份查询缺少 Strata Python 解释器')
        return []
    try:
        result = subprocess.run([python or sys.executable, os.path.abspath(__file__)],
            env={**os.environ, **environment}, capture_output=True, encoding='utf-8',
            # The core is already reading its inherited stdin as a parent-lifetime
            # pipe. Sharing that handle can block Python's standard-stream setup
            # before this script runs; the identity reader accepts no stdin.
            stdin=subprocess.DEVNULL,
            timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode != 0:
            if diagnostics is not None:
                diagnostics.append(f'GPU 身份查询退出码 {result.returncode}：{result.stderr.strip()}')
            return []
        devices = json.loads(result.stdout)
        if not devices and diagnostics is not None:
            diagnostics.append('CUDA 驱动未返回所选设备的身份')
        return devices
    except subprocess.TimeoutExpired:
        if diagnostics is not None:
            diagnostics.append('GPU 身份查询超过 10 秒')
        return []
    except (OSError, ValueError) as error:
        if diagnostics is not None:
            diagnostics.append(f'GPU 身份查询失败：{error}')
        return []


if __name__ == '__main__':
    print(json.dumps(cuda_devices()))
