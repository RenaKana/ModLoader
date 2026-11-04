import ctypes
import asyncio
import json
from types import SimpleNamespace

import pytest

from modloader.hardware import GpuProbe, cpu_temperature, gpu_temperatures, pcie_bandwidth, pci_bdf, system_specs, parse_nvidia_smi
from modloader.gpu_identity import hardware_devices, identity_fields, matching_device, pci_address
from modloader.request_metrics import generation_window


@pytest.mark.parametrize('available, success', [(0, True), (12 * 1024**3, True), (0, False)])
def test_commit_headroom_preserves_zero_and_query_failure(monkeypatch, available, success):
    import modloader.hardware as hardware
    if hardware.os.name != 'nt':
        pytest.skip('Windows commit counter')
    def query(pointer):
        assert pointer._obj.length == 64
        pointer._obj.available_pagefile = available
        return success
    monkeypatch.setattr(hardware.c, 'windll', SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=query)))
    assert hardware.memory_commit_available() == (available if success else None)


@pytest.mark.parametrize('generation,width,expected', [(1, 16, 4e9), (2, 8, 4e9), (4, 16, 16e9*128/130*2), (5, 16, 32e9*128/130*2), (6, 16, None), (4, None, None)])
def test_pcie_capacity_is_encoded_one_direction_not_throughput(generation, width, expected):
    value = pcie_bandwidth(generation, width)
    assert value == pytest.approx(expected) if expected is not None else value is None


def test_nvml_preserves_zero_and_unsupported_fields():
    probe = object.__new__(GpuProbe)
    def zero(output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint)).contents.value = 0
        return 0
    probe.lib = SimpleNamespace(counter=zero, unsupported=lambda output: 3)
    assert probe.number('counter') == 0
    assert probe.number('unsupported') is None
    assert probe.number('missing') is None


@pytest.mark.parametrize('milliwatts,watts', [(0, 0), (155358, 155.358), (None, None)])
def test_nvml_power_is_reported_device_watts_not_an_estimate(milliwatts, watts):
    probe = object.__new__(GpuProbe)
    probe.specs = {'fixture': {'id': 'fixture'}}
    probe.lib = SimpleNamespace(nvmlDeviceGetHandleByIndex_v2=lambda *args: 0,
                                nvmlDeviceGetMemoryInfo=lambda *args: 3,
                                nvmlDeviceGetUtilizationRates=lambda *args: 3)
    probe.string = lambda *args: 'fixture'
    probe.number = lambda name, *args: {'nvmlDeviceGetCount_v2': 1, 'nvmlDeviceGetPowerUsage': milliwatts}.get(name)
    gpu = probe._sample()[0]
    assert gpu['power'] == watts
    assert gpu['power_source'] == ('NVML · nvmlDeviceGetPowerUsage' if watts is not None else None)
    assert (gpu['power_reason'] is None) == (watts is not None)


def test_nvml_samples_every_card_and_maps_cuda_by_identity_not_nvml_index():
    probe = object.__new__(GpuProbe)
    probe.specs = {key: dict(id=key, gpu_id=key, index=index) for index, key in enumerate(['GPU-a', 'GPU-b'])}
    probe.cuda_devices = [dict(device='CUDA0', gpu_id='GPU-b'), dict(device='CUDA1', gpu_id='GPU-a')]
    def handle(index, output):
        output._obj.value = index + 10
        return 0
    probe.lib = SimpleNamespace(nvmlDeviceGetHandleByIndex_v2=handle,
        nvmlDeviceGetMemoryInfo=lambda *args: 3, nvmlDeviceGetUtilizationRates=lambda *args: 3)
    probe.string = lambda name, handle: ['GPU-a', 'GPU-b'][handle.value - 10]
    probe.number = lambda name, *args: 2 if name == 'nvmlDeviceGetCount_v2' else None
    devices = probe._sample()
    assert [item['index'] for item in devices] == [0, 1]
    assert [item['device'] for item in devices] == ['CUDA1', 'CUDA0']
    assert all(item['memory_used'] is None for item in devices)


def test_smi_parses_all_cards_and_unsupported_values_without_false_zeroes():
    devices = parse_nvidia_smi('0, GPU-a, same model, 32768, 8192, 24576, 0, 0, 0, 00000000:01:00.0\n'
                               '1, GPU-b, same model, 24576, [N/A], [N/A], [N/A], [N/A], [N/A], 00000000:02:00.0')
    assert [item['gpu_id'] for item in devices] == ['GPU-a', 'GPU-b']
    assert devices[0]['memory_free'] == 24576 * 1024**2
    assert devices[0]['temperature'] == 0 and devices[0]['power'] == 0
    assert devices[1]['memory_used'] is None and devices[1]['temperature'] is None
    assert devices[1]['power_source'] is None and devices[1]['power_reason']
    assert parse_nvidia_smi('Failed to initialize NVML: Unknown Error') == []


def test_identity_matching_never_guesses_from_indices_names_or_conflicting_uuid():
    devices = [dict(device='CUDA0', gpu_id='GPU-b', pci_bus_id='0000:02:00.0'),
               dict(device='CUDA1', gpu_id='GPU-a', pci_bus_id='0000:01:00.0')]
    cards = [dict(id='GPU-a', name='same', index=0, pci_bus_id='00000000:01:00.0'),
             dict(id='GPU-b', name='same', index=1, pci_bus_id='00000000:02:00.0')]
    assert [item['device'] for item in hardware_devices(cards, devices)] == ['CUDA1', 'CUDA0']
    assert matching_device(dict(gpu_id='GPU-x', pci_bus_id='0000:01:00.0'), devices) is None
    assert hardware_devices([dict(index=0, name='same')], devices)[0]['device'] is None
    assert identity_fields(0, devices)['gpu_id'] == 'GPU-b'
    assert identity_fields(2, devices) == {'identity_status': 'unknown'}
    assert identity_fields(0, [*devices, dict(device='CUDA2', gpu_id='GPU-b')]) == {'identity_status': 'unknown'}
    assert pci_address('00000001:AB:00.0') == '0001:ab:00.0'
    assert pci_address('ab:00.0') is None


@pytest.mark.parametrize('capability_status', [None, 0, 1])
def test_cuda_driver_enumerates_identity_without_creating_contexts(monkeypatch, capability_status):
    import modloader.gpu_identity as identity
    def init(flags):
        assert flags == 0
        return 0
    def count(output):
        output._obj.value = 2
        return 0
    def device(output, index):
        output._obj.value = index + 10
        return 0
    def uuid(output, handle):
        output._obj[:] = bytes([handle] * 16)
        return 0
    def pci(output, length, handle):
        output.value = f'0000:{handle:02x}:00.0'.encode()
        return 0
    lib = SimpleNamespace(cuInit=init, cuDeviceGetCount=count, cuDeviceGet=device,
                          cuDeviceGetUuid_v2=uuid, cuDeviceGetPCIBusId=pci)
    if capability_status is not None:
        def capability(major, minor, handle):
            major._obj.value, minor._obj.value = 12, handle - 10
            return capability_status
        lib.cuDeviceComputeCapability = capability
        def total(output, handle):
            output._obj.value = (handle + 1) * 1024**3
            return capability_status
        lib.cuDeviceTotalMem_v2 = total
    monkeypatch.setattr(identity.c, 'CDLL', lambda name: lib)
    devices = identity.cuda_devices()
    assert [item['device'] for item in devices] == ['CUDA0', 'CUDA1']
    assert devices[0]['gpu_id'] == 'GPU-0a0a0a0a-0a0a-0a0a-0a0a-0a0a0a0a0a0a'
    assert devices[1]['pci_bus_id'] == '0000:0b:00.0'
    if capability_status == 0:
        assert devices[0]['compute_capability'] == [12, 0]
        assert devices[1]['compute_capability'] == [12, 1]
        assert devices[0]['memory_total'] == 11 * 1024**3
    else:
        assert all('compute_capability' not in item for item in devices)
        assert all('memory_total' not in item for item in devices)


def test_frozen_identity_reader_uses_strata_python_and_never_restarts_core(monkeypatch):
    import modloader.gpu_identity as identity
    monkeypatch.setattr(identity.sys, 'frozen', True, raising=False)
    calls = []
    def read(argv, **kwargs):
        calls.append(argv)
        assert kwargs['stdin'] == identity.subprocess.DEVNULL
        assert kwargs['env']['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2'
        return SimpleNamespace(returncode=0, stdout='[{"device":"CUDA0","gpu_id":"GPU-selected"}]')
    monkeypatch.setattr(identity.subprocess, 'run', read)
    environment = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '2'}
    assert identity.cuda_devices_in_environment(environment) == []
    assert calls == []
    assert identity.cuda_devices_in_environment(environment, 'D:/Strata/python/python.exe')[0]['gpu_id'] == 'GPU-selected'
    assert calls[0][0] == 'D:/Strata/python/python.exe'


@pytest.mark.parametrize('failure', ['timeout', 'exit', 'json'])
def test_identity_reader_reports_failure_instead_of_silent_empty_list(monkeypatch, failure):
    import modloader.gpu_identity as identity
    def read(argv, **kwargs):
        if failure == 'timeout':
            raise identity.subprocess.TimeoutExpired(argv, 10)
        return SimpleNamespace(returncode=1 if failure == 'exit' else 0,
                               stdout='invalid-json', stderr='fixture interpreter error')
    monkeypatch.setattr(identity.subprocess, 'run', read)
    diagnostics = []
    assert identity.cuda_devices_in_environment({'CUDA_VISIBLE_DEVICES': '0'}, 'python', diagnostics) == []
    assert len(diagnostics) == 1
    assert {'timeout': '超过 10 秒', 'exit': '退出码 1', 'json': 'GPU 身份查询失败'}[failure] in diagnostics[0]


def test_system_specs_use_reported_cache_units_and_memory_rate(monkeypatch):
    import modloader.hardware as hardware
    monkeypatch.setattr(hardware.os, 'name', 'nt')
    monkeypatch.setattr(hardware.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{"cpu":[{"Name":" CPU fixture ","L3CacheSize":131072,"MaxClockSpeed":4300}],"memory":[{"Capacity":34359738368,"SMBIOSMemoryType":34,"ConfiguredClockSpeed":6400}]}'))
    result = system_specs()
    assert result['cpu_info']['name'] == 'CPU fixture'
    assert result['cpu_info']['l3_bytes'] == 128 * 1024**2
    assert result['memory_modules'][0]['speed_mt_s'] == 6400
    assert result['memory_modules'][0]['type'] == 'DDR5'


def test_current_request_overview_is_not_retained_across_ambiguity():
    live = {'source': 'fixture', 'backend_running': 1, 'backend_queued': 0, '_generated_tokens': 40,
            '_overview': {'average_tokens_per_second': 20, 'context_tokens': 1040, 'context_capacity': 4096}}
    _, current = generation_window(None, 'request-one', live, 10)
    assert current['average_tokens_per_second'] == 20
    assert current['context_tokens'] == 1040
    assert current['tokens_per_second'] is None  # One sample is not a rate.
    for token, running in [(None, 1), ('request-two', 2)]:
        _, invalid = generation_window(None, token, {**live, 'backend_running': running}, 11)
        assert invalid['request_key'] is None
        assert invalid['average_tokens_per_second'] is None
        assert invalid['context_tokens'] is None


def test_live_average_accumulates_real_decode_samples_not_request_elapsed():
    live = {'source': '/slots next_token.n_decoded', 'backend_running': 1, 'backend_queued': 0,
            '_generated_tokens': 100, '_elapsed_s': 101, '_slot_id': 0}
    state, first = generation_window(None, 'one', live, 10)
    assert first['average_tokens_per_second'] is None
    state, second = generation_window(state, 'one', {**live, '_generated_tokens': 140, '_elapsed_s': 102}, 11)
    assert second['average_tokens_per_second'] == 40
    assert second['average_basis'] == 'sampled_decode'
    state, third = generation_window(state, 'one', {**live, '_generated_tokens': 170, '_elapsed_s': 104}, 13)
    assert third['tokens_per_second'] == pytest.approx(70 / 3)  # Three-second rolling window.
    assert third['average_tokens_per_second'] == pytest.approx(70 / 3)  # All sampled decode.
    assert third['average_tokens_per_second'] != pytest.approx(170 / 104)  # Includes prefill.
    for token, change in [('two', {}), ('one', {'_slot_id': 1}), ('one', {'_generated_tokens': 10}),
                           ('one', {'backend_running': 2}), ('one', {'backend_queued': 1})]:
        _, reset = generation_window(state, token, {**live, '_generated_tokens': 200, **change}, 14)
        assert reset['average_tokens_per_second'] is None
        assert reset['average_basis'] is None


def test_native_decode_mean_and_actual_zero_take_priority_over_sampled_mean():
    live = {'source': '/metrics live.generated', 'backend_running': 1, 'backend_queued': 0,
            '_generated_tokens': 100, '_overview': {'average_tokens_per_second': 8.5}}
    state, first = generation_window(None, 'one', live, 10)
    assert first['average_tokens_per_second'] == 8.5 and first['average_basis'] == 'backend_decode'
    _, second = generation_window(state, 'one', {**live, '_generated_tokens': 140}, 11)
    assert second['average_tokens_per_second'] == 8.5 and second['average_basis'] == 'backend_decode'
    _, stopped = generation_window(state, 'one', {**live, '_overview': {'average_tokens_per_second': 0}}, 11)
    assert stopped['average_tokens_per_second'] == 0 and stopped['average_basis'] == 'backend_decode'


def test_windows_temperature_requires_cpu_hardware_parent_and_preserves_zero(monkeypatch):
    import modloader.hardware as hardware
    monkeypatch.setattr(hardware.os, 'name', 'nt')
    provider = {'source': 'LibreHardwareMonitor', 'hardware': [
        {'HardwareType': 'Cpu', 'Identifier': '/cpu/0'}, {'HardwareType': 'Motherboard', 'Identifier': '/board/0'}],
        'sensors': [{'Name': 'CPU', 'Parent': '/board/0', 'Value': 99},
                    {'Name': 'CPU Package', 'Parent': '/cpu/0', 'Value': 0},
                    {'Name': 'CPU Core #1', 'Parent': '/cpu/0', 'Value': None}]}
    monkeypatch.setattr(hardware.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps({'providers': [provider]})))
    value = cpu_temperature()
    assert value['cpu_temperature_c'] == 0
    assert value['cpu_temperature_source'] == 'LibreHardwareMonitor · CPU Package · CPU 传感器最高值'
    assert value['cpu_temperature_reason'] is None
    provider['sensors'].append({'Name': 'CPU Core #2', 'Parent': '/cpu/0', 'Value': 57.5})
    assert cpu_temperature()['cpu_temperature_c'] == 57.5
    provider['hardware'] = provider['hardware'][1:]  # Board sensor named CPU cannot qualify.
    board_only = cpu_temperature()
    assert board_only['cpu_temperature_c'] is None
    assert '未提供 CPU 温度' in board_only['cpu_temperature_reason']


def test_missing_windows_temperature_provider_and_timeout_are_explicit(monkeypatch):
    import modloader.hardware as hardware
    monkeypatch.setattr(hardware.os, 'name', 'nt')
    monkeypatch.setattr(hardware.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{"providers":[]}'))
    missing = cpu_temperature()
    assert missing['cpu_temperature_c'] is None and missing['cpu_temperature_source'] is None
    assert '传感器工具及读取权限' in missing['cpu_temperature_reason']
    def timeout(*args, **kwargs):
        raise hardware.subprocess.TimeoutExpired('fixture', 5)
    monkeypatch.setattr(hardware.subprocess, 'run', timeout)
    assert cpu_temperature()['cpu_temperature_reason'] == 'CPU 温度传感器查询超时'


async def test_cpu_temperature_missing_source_retries_slowly_without_blocking_resource_poll(monkeypatch):
    import modloader.app as app_module
    controller = object.__new__(app_module.Controller)
    controller.hardware = {}
    controller.cpu_sensor_lock = asyncio.Lock()
    controller.cpu_temperature_wake = asyncio.Event()
    controller.temperature_auto_started = True
    controller.external_cpu_retry_at = 0
    controller.cpu_sensor = SimpleNamespace(sample=lambda: {'cpu_temperature_c': None, 'cpu_temperature_status': 'disabled'})
    samples = iter([{'cpu_temperature_c': None}, {'cpu_temperature_c': 0}, {'cpu_temperature_c': None}])
    monkeypatch.setattr(app_module, 'cpu_temperature', lambda: next(samples))
    waits = []
    async def wait_for(coroutine, seconds):
        coroutine.close()
        controller.external_cpu_retry_at = 0  # This fixture advances directly to the next scheduled read.
        waits.append(seconds)
        if len(waits) == 3:
            raise asyncio.CancelledError
        raise TimeoutError
    monkeypatch.setattr(app_module.asyncio, 'wait_for', wait_for)
    with pytest.raises(asyncio.CancelledError):
        await controller._sample_cpu_temperature()
    assert waits == [app_module.CPU_TEMPERATURE_RETRY_S, app_module.CPU_TEMPERATURE_INTERVAL_S, app_module.CPU_TEMPERATURE_RETRY_S]
    assert controller.hardware['cpu_temperature_c'] is None


def test_native_temperature_precedes_external_but_preserves_existing_provider(monkeypatch):
    import modloader.app as app_module
    controller = object.__new__(app_module.Controller)
    controller.external_cpu_retry_at = 0
    native = {'cpu_temperature_c': 0, 'cpu_temperature_status': 'ready', 'cpu_temperature_source': 'native',
              'gpu_temperatures': [{'temperature_c': 49.25}]}
    controller.cpu_sensor = SimpleNamespace(sample=lambda: native.copy())
    external_calls = []
    def external():
        external_calls.append(True)
        return {'cpu_temperature_c': 57.5, 'cpu_temperature_source': 'WMI', 'cpu_temperature_reason': None}
    monkeypatch.setattr(app_module, 'cpu_temperature', external)
    assert controller._read_cpu_temperature() == native
    assert external_calls == []
    native.update(cpu_temperature_c=None, cpu_temperature_status='permission_required')
    result = controller._read_cpu_temperature()
    assert result['cpu_temperature_c'] == 57.5 and result['cpu_temperature_source'] == 'WMI'
    assert result['cpu_temperature_status'] == 'ready'
    assert result['gpu_temperatures'] == native['gpu_temperatures']


def test_missing_external_cpu_provider_does_not_delay_live_gpu_sampling(monkeypatch):
    import modloader.app as app_module
    controller = object.__new__(app_module.Controller)
    controller.external_cpu_retry_at = 0
    native = {'cpu_temperature_c': None, 'cpu_temperature_status': 'driver_missing',
              'gpu_temperatures': [{'temperature_c': 49.125}]}
    calls = []
    controller.cpu_sensor = SimpleNamespace(sample=lambda: native.copy())
    monkeypatch.setattr(app_module, 'cpu_temperature', lambda: calls.append(True) or {'cpu_temperature_c': None})
    clock = [100]
    monkeypatch.setattr(app_module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    assert controller._read_cpu_temperature()['gpu_temperatures'] == native['gpu_temperatures']
    clock[0] += 1
    assert controller._read_cpu_temperature()['gpu_temperatures'] == native['gpu_temperatures']
    assert len(calls) == 1
    clock[0] += 30
    controller._read_cpu_temperature()
    assert len(calls) == 2


async def test_cpu_enable_is_management_authenticated_explicit_and_closed(tmp_path):
    import httpx
    from modloader.app import Controller, build_app
    controller = Controller(tmp_path / 'data', 'management-test')
    calls = []
    sample = {'cpu_temperature_c': 55.25, 'cpu_temperature_status': 'ready',
              'cpu_temperature_source': 'test CPU', 'cpu_temperature_reason': None}
    controller.cpu_sensor = SimpleNamespace(enable=lambda: calls.append('enable') or sample.copy(),
                                            close=lambda: calls.append('close'))
    path = '/api/hardware/temperatures/enable'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver') as client:
        assert (await client.post(path)).status_code == 401
        assert (await client.post(path, headers={'Authorization': 'Bearer ' + controller.api_key})).status_code == 401
        assert calls == []
        response = await client.post(path, headers={'Authorization': 'Bearer management-test'})
        assert response.status_code == 200 and response.json() == sample
        assert controller.hardware['cpu_temperature_c'] == 55.25
        assert controller.cpu_temperature_wake.is_set()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)), base_url='http://testserver') as client:
        assert (await client.post(path, headers={'Authorization': 'Bearer ' + controller.api_key})).status_code == 404
    await controller.close()
    assert calls == ['enable', 'close']


async def test_temperature_autostart_once_cancel_does_not_reprompt_and_manual_retry_works(tmp_path, monkeypatch):
    import modloader.app as app_module
    controller = app_module.Controller(tmp_path / 'data', 'test')
    monkeypatch.setattr(app_module, 'os', SimpleNamespace(name='nt', environ={}))
    calls = []
    denied = {'cpu_temperature_c': None, 'cpu_temperature_source': None,
              'cpu_temperature_status': 'permission_required', 'cpu_temperature_reason': 'cancelled'}
    controller.cpu_sensor = SimpleNamespace(enable=lambda: calls.append('enable') or denied.copy(),
                                            close=lambda: None)
    await controller._auto_start_temperatures()
    await controller._auto_start_temperatures()
    assert calls == ['enable'] and controller.hardware == {**controller.hardware, **denied}
    await controller.enable_cpu_temperature()
    assert calls == ['enable', 'enable']
    await controller.close()


async def test_headless_temperature_optout_never_starts_helper(tmp_path, monkeypatch):
    import modloader.app as app_module
    controller = app_module.Controller(tmp_path / 'data', 'test')
    monkeypatch.setattr(app_module, 'os', SimpleNamespace(name='nt', environ={'MODLOADER_SENSOR_AUTO_START': '0'}))
    calls = []
    controller.cpu_sensor = SimpleNamespace(enable=lambda: calls.append('enable'), close=lambda: None)
    await controller._auto_start_temperatures()
    assert calls == []
    await controller.close()


def test_gpu_temperature_matches_physical_pci_not_name_or_index():
    cards = [{'id': 'one', 'name': 'same GPU', 'pci_bus_id': '00000000:01:00.0', 'temperature': 48, 'utilization': 12},
             {'id': 'two', 'name': 'same GPU', 'pci_bus_id': '00000000:02:00.0', 'temperature': 50, 'utilization': 9}]
    readings = [{'name': 'same GPU', 'pci_bus_id': '02:00.0', 'temperature_c': 50.125, 'source': 'GPU Core'},
                {'name': 'same GPU', 'pci_bus_id': '01:00.0', 'temperature_c': 0, 'source': 'GPU Core'}]
    result = gpu_temperatures(cards, readings)
    assert [gpu['temperature'] for gpu in result] == [0, 50.125]
    assert [gpu['utilization'] for gpu in result] == [12, 9]
    assert all(gpu['temperature_source'] == 'LibreHardwareMonitor · same GPU · GPU Core' for gpu in result)
    assert all(gpu['temperature_reason'] is None for gpu in result)
    assert cards[0]['temperature'] == 48  # Do not change the underlying driver sample.
    # Missing identity is not guessed even with one model name.
    assert gpu_temperatures(cards, [{**readings[0], 'pci_bus_id': None}])[1]['temperature'] == 50
    # Windows lacks a domain: identical BDF in two domains is ambiguous.
    ambiguous = [cards[0], {**cards[1], 'pci_bus_id': '00000001:01:00.0'}]
    assert [gpu['temperature'] for gpu in gpu_temperatures(ambiguous, readings)] == [48, 50]
    duplicate = gpu_temperatures(cards, readings + [readings[0]])
    assert duplicate[1]['temperature'] == 50
    assert '无法唯一匹配' in duplicate[1]['temperature_reason']


def test_gpu_temperature_unavailable_preserves_driver_value_and_explains_source():
    cards = [{'id': 'one', 'name': 'GPU', 'pci_bus_id': '0000:01:00.0', 'temperature': 48}]
    readings = [{'name': 'GPU', 'pci_bus_id': '01:00.0', 'temperature_c': None,
                 'source': 'GPU Core', 'reason': '传感器缺测'}]
    result = gpu_temperatures(cards, readings)[0]
    assert result['temperature'] == 48 and result['temperature_source'] == 'NVML · GPU'
    assert '传感器缺测' in result['temperature_reason']
    assert gpu_temperatures(cards, [], '读取器失败')[0]['temperature_reason'] == '读取器失败'
    assert pci_bdf('00000000:AB:1F.7') == 'ab:1f.7'
    assert pci_bdf(None) is None and pci_bdf('index 0') is None


async def test_cpu_enable_concurrent_requests_share_authorization_and_close_unblocks(tmp_path):
    import threading
    from modloader.app import Controller
    controller = Controller(tmp_path / 'data', 'test')
    started, closed = threading.Event(), threading.Event()
    calls = []
    def enable():
        calls.append('enable')
        started.set()
        assert closed.wait(3)
        return {'cpu_temperature_c': None, 'cpu_temperature_status': 'disabled',
                'cpu_temperature_source': None, 'cpu_temperature_reason': 'closed'}
    controller.cpu_sensor = SimpleNamespace(enable=enable, close=closed.set)
    first = asyncio.create_task(controller.enable_cpu_temperature())
    assert await asyncio.to_thread(started.wait, 3)
    second = asyncio.create_task(controller.enable_cpu_temperature())
    await asyncio.sleep(0)
    assert controller.hardware['cpu_temperature_status'] == 'starting'
    await asyncio.wait_for(controller.close(), 3)
    assert (await first) == (await second)
    assert calls == ['enable']
