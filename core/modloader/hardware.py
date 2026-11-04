"""Read-only hardware specifications and NVIDIA driver counters."""
from __future__ import annotations

import ctypes as c
import csv
import json
import math
import os
import platform
import re
import subprocess
import threading

import psutil

from .gpu_identity import cuda_devices, gpu_uuid, hardware_devices


def memory_commit_available():
    """Windows commit headroom, used by Strata's resident expert allocator."""
    if os.name != 'nt':
        return None
    class MemoryStatus(c.Structure):
        _fields_ = [('length', c.c_uint32), ('load', c.c_uint32),
                    ('total_phys', c.c_uint64), ('available_phys', c.c_uint64),
                    ('total_pagefile', c.c_uint64), ('available_pagefile', c.c_uint64),
                    ('total_virtual', c.c_uint64), ('available_virtual', c.c_uint64),
                    ('available_extended_virtual', c.c_uint64)]
    status = MemoryStatus()
    status.length = c.sizeof(status)
    try:
        query = c.windll.kernel32.GlobalMemoryStatusEx
        query.argtypes = [c.POINTER(MemoryStatus)]
        query.restype = c.c_int
        return int(status.available_pagefile) if query(c.byref(status)) else None
    except (AttributeError, OSError):
        return None


def cpu_temperature():
    """Read CPU-owned sensors only; never load a driver or use ACPI zones."""
    missing = {'cpu_temperature_c': None, 'cpu_temperature_source': None,
               'cpu_temperature_reason': '未检测到可读取的 CPU 温度传感器'}
    if os.name != 'nt':
        read = getattr(psutil, 'sensors_temperatures', None)
        if read is None:
            return missing
        try:
            sensors = [(f'{driver} · {sensor.label or "CPU"}', sensor.current)
                       for driver, entries in read().items() if driver in ('coretemp', 'k10temp', 'zenpower')
                       for sensor in entries if type(sensor.current) in (int, float) and math.isfinite(sensor.current)]
        except (OSError, RuntimeError):
            return {**missing, 'cpu_temperature_reason': 'CPU 温度传感器读取失败'}
    else:
        # Both providers expose the Hardware parent, so a motherboard sensor
        # merely named "CPU" is not enough to qualify. No ACPI/board fallback.
        script = r"""[Console]::OutputEncoding = [Text.UTF8Encoding]::new();
        $providers = @();
        foreach ($provider in @('LibreHardwareMonitor', 'OpenHardwareMonitor')) {
            $ns = 'root\' + $provider;
            try {
                $hardware = @(Get-CimInstance -Namespace $ns -ClassName Hardware -ErrorAction Stop | Select-Object HardwareType,Identifier);
                $sensors = @(Get-CimInstance -Namespace $ns -ClassName Sensor -ErrorAction Stop -Filter "SensorType='Temperature'" | Select-Object Parent,Name,Value);
                $providers += @{source=$provider; hardware=$hardware; sensors=$sensors};
            } catch {}
        }
        @{providers=$providers} | ConvertTo-Json -Compress -Depth 6"""
        try:
            proc = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                                  capture_output=True, encoding='utf-8', timeout=5, creationflags=0x08000000)
            if proc.returncode:
                return {**missing, 'cpu_temperature_reason': 'CPU 温度传感器查询失败'}
            raw = json.loads(proc.stdout.lstrip('\ufeff'))
            sensors = []
            for provider in raw.get('providers', []):
                cpus = {item['Identifier'] for item in provider['hardware'] if str(item.get('HardwareType')).lower() == 'cpu'}
                sensors.extend((f'{provider["source"]} · {sensor["Name"]}', sensor['Value'])
                               for sensor in provider['sensors'] if sensor.get('Parent') in cpus
                               and type(sensor.get('Value')) in (int, float) and math.isfinite(sensor['Value']))
            if not sensors:
                return {**missing, 'cpu_temperature_reason': '已运行的硬件监控工具未提供 CPU 温度' if raw.get('providers') else '未读到 CPU 温度；请检查传感器工具及读取权限'}
        except subprocess.TimeoutExpired:
            return {**missing, 'cpu_temperature_reason': 'CPU 温度传感器查询超时'}
        except (OSError, ValueError, KeyError, TypeError):
            return {**missing, 'cpu_temperature_reason': 'CPU 温度传感器读取失败'}
    if not sensors:
        return missing
    source, value = max(sensors, key=lambda sensor: sensor[1])
    return {'cpu_temperature_c': value, 'cpu_temperature_source': f'{source} · CPU 传感器最高值',
            'cpu_temperature_reason': None}


def pcie_bandwidth(generation, width):
    # PCIe 1–5, encoded payload rate per direction (not measured throughput).
    rates = {1: 2.5e9 * 8 / 10, 2: 5e9 * 8 / 10,
             3: 8e9 * 128 / 130, 4: 16e9 * 128 / 130, 5: 32e9 * 128 / 130}
    rate = rates.get(generation)
    return rate * width / 8 if rate and width and width > 0 else None


def pci_bdf(value):
    """Windows PnP exposes bus/device/function, without a PCI domain."""
    match = re.fullmatch(r'(?:[0-9a-f]{4,8}:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])', str(value or '').lower())
    return match.group(1) if match else None


def gpu_temperatures(devices, readings, reason=None):
    """Prefer LHM's current GPU Core value for an unambiguous physical GPU.

    Keep existing driver counters (including temperature when LHM cannot read
    it) available. Never match identical model names or enumeration indices.
    """
    devices = devices or []
    readings = readings or []
    device_keys = [pci_bdf(device.get('pci_bus_id')) for device in devices]
    reading_keys = [pci_bdf(reading.get('pci_bus_id')) for reading in readings]
    result = []
    for device, key in zip(devices, device_keys):
        merged = {**device, 'temperature_source': f"{device.get('driver_source', 'NVML')} · {device.get('name') or 'GPU'}",
                  'temperature_reason': reason or '未读到匹配的硬件温度传感器，保留 NVIDIA 驱动读数'}
        if key and device_keys.count(key) == 1 and reading_keys.count(key) == 1:
            reading = readings[reading_keys.index(key)]
            value = reading.get('temperature_c')
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                merged.update(temperature=value,
                              temperature_source=f"LibreHardwareMonitor · {reading['name']} · {reading['source']}",
                              temperature_reason=None)
            elif reading.get('reason'):
                merged['temperature_reason'] = reading['reason'] + '；保留 NVIDIA 驱动读数'
        elif key and (device_keys.count(key) > 1 or reading_keys.count(key) > 1):
            merged['temperature_reason'] = 'PCI 地址无法唯一匹配，保留 NVIDIA 驱动读数'
        result.append(merged)
    return result


def system_specs():
    result = {'cpu_info': {'name': platform.processor() or None, 'cores': psutil.cpu_count(logical=False),
                           'threads': psutil.cpu_count()}, 'memory_modules': []}
    if os.name != 'nt':
        return result
    script = """[Console]::OutputEncoding = [Text.UTF8Encoding]::new();
    @{cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize);
      memory=@(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,SMBIOSMemoryType,ConfiguredClockSpeed,DeviceLocator)} | ConvertTo-Json -Compress -Depth 4"""
    try:
        proc = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                              capture_output=True, encoding='utf-8', timeout=8, creationflags=0x08000000)
        if proc.returncode:
            raise ValueError('系统硬件规格读取失败')
        raw = json.loads(proc.stdout.lstrip('\ufeff'))
        cpus = raw.get('cpu') or []
        if cpus:
            cpu = cpus[0]
            result['cpu_info'].update(name=str(cpu['Name']).strip(), max_clock_mhz=cpu.get('MaxClockSpeed'),
                                      l2_bytes=(cpu.get('L2CacheSize') or 0) * 1024 or None,
                                      l3_bytes=(cpu.get('L3CacheSize') or 0) * 1024 or None)
        types = {26: 'DDR4', 34: 'DDR5', 30: 'LPDDR4', 35: 'LPDDR5', 24: 'DDR3'}
        result['memory_modules'] = [{'capacity': item.get('Capacity'), 'type': types.get(item.get('SMBIOSMemoryType')),
                                     'speed_mt_s': item.get('ConfiguredClockSpeed') or None,
                                     'slot': item.get('DeviceLocator')} for item in raw.get('memory', [])]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        result['specs_error'] = '系统未返回完整硬件规格'
    return result


class GpuProbe:
    """Small binding to installed NVML; unsupported optional counters stay null.

    Signatures/units follow the CUDA SDK nvml.h. No driver or clock changes.
    """
    def __init__(self):
        self.lib = None
        self.specs = {}
        self.cuda_devices = cuda_devices()
        self.lock = threading.Lock()
        try:
            lib = c.CDLL('nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
            if lib.nvmlInit_v2() == 0:
                self.lib = lib
        except (OSError, AttributeError):
            pass

    def close(self):
        with self.lock:
            if self.lib:
                self.lib.nvmlShutdown()
                self.lib = None

    def number(self, name, *args):
        value = c.c_uint()
        fn = getattr(self.lib, name, None)
        return value.value if fn and fn(*args, c.byref(value)) == 0 else None

    def string(self, name, handle):
        value = c.create_string_buffer(128)
        fn = getattr(self.lib, name, None)
        return value.value.decode('utf-8', errors='replace') if fn and fn(handle, value, len(value)) == 0 else None

    def pci_bus_id(self, handle):
        # nvmlPciInfo_t from nvml.h; v3 fills the full domain-qualified bus ID.
        class PciInfo(c.Structure):
            _fields_ = [('busIdLegacy', c.c_char * 16), ('domain', c.c_uint),
                        ('bus', c.c_uint), ('device', c.c_uint), ('pciDeviceId', c.c_uint),
                        ('pciSubSystemId', c.c_uint), ('busId', c.c_char * 32)]
        info = PciInfo()
        fn = getattr(self.lib, 'nvmlDeviceGetPciInfo_v3', None)
        return info.busId.decode('ascii') if fn and fn(handle, c.byref(info)) == 0 else None

    def sample(self):
        with self.lock:
            return self._sample()

    def _sample(self):
        if not self.lib:
            return None
        count = self.number('nvmlDeviceGetCount_v2')
        if count is None:
            raise RuntimeError('NVIDIA 驱动未返回设备信息')
        devices = []
        for index in range(count):
            handle = c.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(index, c.byref(handle)) != 0:
                continue
            uuid = self.string('nvmlDeviceGetUUID', handle) or str(index)
            if uuid not in self.specs:
                major, minor = c.c_int(), c.c_int()
                fn = getattr(self.lib, 'nvmlDeviceGetCudaComputeCapability', None)
                compute = f'{major.value}.{minor.value}' if fn and fn(handle, c.byref(major), c.byref(minor)) == 0 else None
                bus = self.number('nvmlDeviceGetMemoryBusWidth', handle)
                clock = self.number('nvmlDeviceGetMaxClockInfo', handle, 2)  # NVML_CLOCK_MEM, MHz
                self.specs[uuid] = {'id': uuid, 'gpu_id': gpu_uuid(uuid), 'index': index, 'name': self.string('nvmlDeviceGetName', handle),
                    'pci_bus_id': self.pci_bus_id(handle),
                    'compute_capability': compute, 'cuda_cores': self.number('nvmlDeviceGetNumGpuCores', handle),
                    'memory_bus_bits': bus, 'memory_max_clock_mhz': clock,
                    # NVML memory clock is DDR clock, not an inferred SKU bandwidth.
                    'memory_bandwidth_bytes_s': clock * 1e6 * 2 * bus / 8 if clock and bus else None,
                    'pcie_gen_max': self.number('nvmlDeviceGetMaxPcieLinkGeneration', handle),
                    'pcie_width_max': self.number('nvmlDeviceGetMaxPcieLinkWidth', handle)}
                self.specs[uuid]['pcie_bandwidth_max_bytes_s'] = pcie_bandwidth(
                    self.specs[uuid]['pcie_gen_max'], self.specs[uuid]['pcie_width_max'])
            class Memory(c.Structure):
                _fields_ = [('total', c.c_ulonglong), ('free', c.c_ulonglong), ('used', c.c_ulonglong)]
            class Util(c.Structure):
                _fields_ = [('gpu', c.c_uint), ('memory', c.c_uint)]
            memory, util = Memory(), Util()
            memory_ok = self.lib.nvmlDeviceGetMemoryInfo(handle, c.byref(memory)) == 0
            util_ok = self.lib.nvmlDeviceGetUtilizationRates(handle, c.byref(util)) == 0
            gen = self.number('nvmlDeviceGetCurrPcieLinkGeneration', handle)
            width = self.number('nvmlDeviceGetCurrPcieLinkWidth', handle)
            rx = self.number('nvmlDeviceGetPcieThroughput', handle, 1)
            tx = self.number('nvmlDeviceGetPcieThroughput', handle, 0)
            power = self.number('nvmlDeviceGetPowerUsage', handle)
            devices.append({**self.specs[uuid], 'index': index, 'driver_source': 'NVML', 'memory_total': memory.total if memory_ok else None,
                'memory_used': memory.used if memory_ok else None, 'memory_free': memory.free if memory_ok else None,
                'utilization': util.gpu if util_ok else None, 'temperature': self.number('nvmlDeviceGetTemperature', handle, 0),
                'power': power / 1000 if power is not None else None,
                'power_source': 'NVML · nvmlDeviceGetPowerUsage' if power is not None else None,
                'power_reason': None if power is not None else 'NVIDIA 驱动未提供功率读数', 'pcie_gen': gen, 'pcie_width': width,
                'pcie_bandwidth_bytes_s': pcie_bandwidth(gen, width),
                'pcie_rx_bytes_s': rx * 1024 if rx is not None else None,
                'pcie_tx_bytes_s': tx * 1024 if tx is not None else None})
        return hardware_devices(devices, getattr(self, 'cuda_devices', []))


def parse_nvidia_smi(output):
    """Read all query rows; unsupported counters stay null, including zero power."""
    devices = []
    for fields in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(fields) != 10 or not fields[0].strip().isdecimal():
            continue
        fields = [value.strip() for value in fields]
        def number(index):
            try:
                value = float(fields[index])
                return value if math.isfinite(value) else None
            except ValueError:
                return None
        gpu_id, index = gpu_uuid(fields[1]), int(fields[0])
        power = number(8)
        devices.append(dict(id=gpu_id or fields[9] or f'nvidia-smi-{index}', gpu_id=gpu_id,
            index=index, name=fields[2], pci_bus_id=fields[9], driver_source='nvidia-smi',
            **{key: number(column) * 1024**2 if number(column) is not None else None
               for key, column in (('memory_total', 3), ('memory_used', 4), ('memory_free', 5))},
            utilization=number(6), temperature=number(7), power=power,
            power_source='nvidia-smi · power.draw' if power is not None else None,
            power_reason=None if power is not None else 'NVIDIA 驱动未提供功率读数'))
    return devices
