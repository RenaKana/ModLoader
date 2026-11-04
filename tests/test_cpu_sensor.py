import ctypes
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import pytest

import modloader.cpu_sensor as sensor


def reply(status='ready', value=42.5, source='LibreHardwareMonitor · CPU · Core (Tctl/Tdie)', reason=None,
          gpus=None, gpu_reason=None):
    if status != 'ready':
        value, source = None, None
        reason = reason or 'fixture unavailable'
    return sensor._result(status, reason, value, source, gpus, gpu_reason)


def gpu_reply(value=36.421875, identifier='/gpu-nvidia/0', name='NVIDIA GeForce RTX 5090',
              pci='01:00.0', source='GPU Core', reason=None):
    return {'identifier': identifier, 'name': name, 'pci_bus_id': pci,
            'temperature_c': value, 'source': source, 'reason': reason}


class FakePipe:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.closed = False
        self.samples = 0

    def request(self):
        self.samples += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


class FakeApi:
    def __init__(self, pipe):
        self.pipe = pipe
        self.launched = []
        self.connected = []
        self.closed = []
        self.failure = None

    def launch(self, path, name):
        self.launched.append((path, name))
        if self.failure:
            raise self.failure
        return 101, 202

    def connect(self, name, pid, process, stopping):
        self.connected.append((name, pid, process))
        return self.pipe

    def close_handle(self, handle):
        self.closed.append(handle)


def install_api(monkeypatch, pipe):
    api = FakeApi(pipe)
    monkeypatch.setattr(sensor.os, 'name', 'nt')
    monkeypatch.setattr(sensor, '_WindowsApi', lambda: api)
    monkeypatch.setattr(sensor, '_bundled_helper', lambda: Path('C:/bundled/sensors/ModLoader.CpuSensors.exe'))
    return api


def test_polling_is_disabled_without_launching_or_inspecting_privileged_components(monkeypatch):
    def forbidden():
        pytest.fail('ordinary polling must not create a Windows API or resolve/run the helper')
    monkeypatch.setattr(sensor, '_WindowsApi', forbidden)
    monkeypatch.setattr(sensor, '_bundled_helper', forbidden)
    reader = sensor.CpuSensorReader()
    for _ in range(3):
        assert reader.sample() == sensor._result('disabled', '尚未启用硬件温度读取')
    reader.close()
    assert reader.sample()['cpu_temperature_status'] == 'disabled'


def test_explicit_enable_uses_private_random_pipe_and_keeps_legitimate_zero(monkeypatch):
    pipe = FakePipe([reply(value=0), reply(value=57.25), reply(value=58)])
    api = install_api(monkeypatch, pipe)
    reader = sensor.CpuSensorReader()
    assert reader.enable()['cpu_temperature_c'] == 0
    path, name = api.launched[0]
    assert path.name == 'ModLoader.CpuSensors.exe'
    assert re.fullmatch(r'ModLoader\.CpuSensors\.[0-9a-f]{32}', name)
    assert api.connected == [(name, 202, 101)]
    assert reader.sample()['cpu_temperature_c'] == 57.25
    assert reader.enable()['cpu_temperature_c'] == 58
    assert len(api.launched) == 1
    reader.close()
    reader.close()
    assert pipe.closed and api.closed == [101]
    assert reader.sample()['cpu_temperature_c'] is None


def test_each_reader_gets_its_own_pipe_name(monkeypatch):
    pipe = FakePipe([reply(), reply()])
    api = install_api(monkeypatch, pipe)
    first, second = sensor.CpuSensorReader(), sensor.CpuSensorReader()
    first.enable()
    second.enable()
    assert api.launched[0][1] != api.launched[1][1]
    first.close()
    second.close()


@pytest.mark.parametrize('status', ['driver_missing', 'permission_required', 'error'])
def test_helper_unavailability_disconnects_and_never_reprompts_on_poll(monkeypatch, status):
    pipe = FakePipe([reply(status=status)])
    api = install_api(monkeypatch, pipe)
    reader = sensor.CpuSensorReader()
    result = reader.enable()
    assert result['cpu_temperature_status'] == status
    assert result['cpu_temperature_c'] is None
    assert pipe.closed and api.closed == [101]
    assert reader.sample() == result
    assert len(api.launched) == 1


@pytest.mark.parametrize('status', ['driver_missing', 'permission_required', 'error'])
def test_gpu_samples_continue_when_cpu_is_unavailable_without_fabricating_cpu_zero(monkeypatch, status):
    pipe = FakePipe([reply(status, gpus=[gpu_reply(value=0)]),
                     reply(status, gpus=[gpu_reply(value=36.421875)])])
    api = install_api(monkeypatch, pipe)
    reader = sensor.CpuSensorReader()
    first = reader.enable()
    assert first['cpu_temperature_c'] is None
    assert first['cpu_temperature_status'] == status
    assert first['gpu_temperatures'][0]['temperature_c'] == 0
    assert reader.sample()['gpu_temperatures'][0]['temperature_c'] == 36.421875
    assert not pipe.closed and api.closed == [] and len(api.launched) == 1
    reader.close()
    assert pipe.closed and api.closed == [101]


def test_enumerated_gpu_can_recover_from_missing_value_without_new_uac(monkeypatch):
    missing = gpu_reply(value=None, reason='GPU 未提供有效的 GPU Core 当前温度')
    pipe = FakePipe([reply('permission_required', gpus=[missing]),
                     reply('permission_required', gpus=[gpu_reply()])])
    api = install_api(monkeypatch, pipe)
    reader = sensor.CpuSensorReader()
    assert reader.enable()['gpu_temperatures'][0]['temperature_c'] is None
    assert reader.sample()['gpu_temperatures'][0]['temperature_c'] == 36.421875
    assert not pipe.closed and len(api.launched) == 1
    reader.close()


def test_gpu_initialization_failure_does_not_discard_valid_cpu(monkeypatch):
    result = reply(gpu_reason='硬件温度读取库不可用：GPU API fixture')
    assert sensor._decode_reply(json.dumps(result).encode()) == result
    pipe = FakePipe([result, result])
    api = install_api(monkeypatch, pipe)
    reader = sensor.CpuSensorReader()
    assert reader.enable()['cpu_temperature_c'] == 42.5
    assert reader.sample()['gpu_temperature_reason'] == result['gpu_temperature_reason']
    assert not pipe.closed and len(api.launched) == 1
    reader.close()


def test_explicit_retry_rebuilds_gpu_initialization_failure_but_not_healthy_gpu_with_missing_cpu(monkeypatch):
    failed = reply(gpu_reason='硬件温度读取失败：GPU group initialization fixture')
    original = FakePipe([failed, failed])
    api = install_api(monkeypatch, original)
    reader = sensor.CpuSensorReader()
    assert reader.enable()['gpu_temperature_reason'] == failed['gpu_temperature_reason']
    assert reader.sample()['cpu_temperature_c'] == 42.5
    assert len(api.launched) == 1 and not original.closed  # Polling does not restart or elevate.

    recovered = reply('permission_required', gpus=[gpu_reply()])
    replacement = FakePipe([recovered, recovered])
    api.pipe = replacement
    assert reader.enable() == recovered
    assert original.closed and api.closed == [101]
    assert len(api.launched) == 2 and api.launched[0][1] != api.launched[1][1]
    assert reader.enable() == recovered  # Healthy GPU alone is enough to reuse the connection.
    assert len(api.launched) == 2 and not replacement.closed
    reader.close()


def test_uac_cancellation_is_actionable_not_a_crash(monkeypatch):
    api = install_api(monkeypatch, FakePipe([]))
    api.failure = sensor._ReaderError('已取消管理员授权', 'permission_required')
    reader = sensor.CpuSensorReader()
    assert reader.enable()['cpu_temperature_status'] == 'permission_required'
    assert reader.sample()['cpu_temperature_status'] == 'permission_required'
    assert not api.connected and not api.closed and len(api.launched) == 1


def test_failed_or_timed_out_read_clears_old_temperature_and_releases_own_handles(monkeypatch):
    pipe = FakePipe([reply(value=66), sensor._ReaderError('CPU 温度读取超时；请重新启用')])
    api = install_api(monkeypatch, pipe)
    reader = sensor.CpuSensorReader()
    reader.enable()
    result = reader.sample()
    assert result['cpu_temperature_status'] == 'error'
    assert result['cpu_temperature_c'] is None and result['cpu_temperature_source'] is None
    assert '超时' in result['cpu_temperature_reason']
    assert pipe.closed and api.closed == [101]
    assert reader.sample() == result and len(api.launched) == 1


def test_close_during_pending_uac_does_not_block_or_later_connect(monkeypatch):
    pipe = FakePipe([])
    api = install_api(monkeypatch, pipe)
    entered, resume = threading.Event(), threading.Event()
    def waiting_launch(path, name):
        entered.set()
        assert resume.wait(2)
        return 101, 202
    api.launch = waiting_launch
    reader = sensor.CpuSensorReader()
    thread = threading.Thread(target=reader.enable)
    thread.start()
    assert entered.wait(2)
    assert reader.sample()['cpu_temperature_status'] == 'starting'
    reader.close()
    thread.join(0.5)
    assert not thread.is_alive()  # Executor work ends even with UAC unanswered.
    resume.set()
    assert reader._pending_launch.wait(2)
    assert api.closed == [101] and not api.connected
    assert reader.sample()['cpu_temperature_status'] == 'disabled'
    assert reader.enable()['cpu_temperature_status'] == 'disabled'


def test_uac_timeout_does_not_stack_prompts_and_releases_a_late_process(monkeypatch):
    api = install_api(monkeypatch, FakePipe([]))
    entered, resume = threading.Event(), threading.Event()
    def waiting_launch(path, name):
        api.launched.append((path, name))
        entered.set()
        assert resume.wait(2)
        return 101, 202
    api.launch = waiting_launch
    monkeypatch.setattr(sensor, 'UAC_TIMEOUT_S', 0.01)
    reader = sensor.CpuSensorReader()
    result = reader.enable()
    assert entered.is_set()
    assert result['cpu_temperature_status'] == 'permission_required'
    assert '超时' in result['cpu_temperature_reason']
    assert reader.enable()['cpu_temperature_status'] == 'permission_required'
    assert len(api.launched) == 1 and not api.connected
    resume.set()
    assert reader._pending_launch.wait(2)
    assert api.closed == [101]
    reader.close()


def test_only_existing_bundled_path_can_launch(monkeypatch, tmp_path):
    module = tmp_path / 'modloader' / 'cpu_sensor.py'
    module.parent.mkdir()
    module.write_text('# fixture')
    monkeypatch.setattr(sensor, '__file__', str(module))
    with pytest.raises(sensor._ReaderError, match='缺少内置'):
        sensor._bundled_helper()
    helper = module.parent / 'sensors' / 'ModLoader.CpuSensors.exe'
    helper.parent.mkdir()
    helper.write_bytes(b'fixture; never executed')
    assert sensor._bundled_helper() == helper.resolve()


@pytest.mark.parametrize('change', [
    {'cpu_temperature_status': 'unknown'}, {'cpu_temperature_c': True},
    {'cpu_temperature_c': float('nan')}, {'cpu_temperature_c': float('inf')},
    {'cpu_temperature_c': None}, {'cpu_temperature_source': None},
    {'cpu_temperature_reason': 'a ready value cannot have a failure reason'},
    {'cpu_temperature_status': 'permission_required'},
])
def test_malformed_or_contradictory_helper_results_are_rejected(change):
    invalid = {**reply(), **change}
    with pytest.raises(sensor._ReaderError, match='无效响应'):
        sensor._decode_reply(json.dumps(invalid).encode())


@pytest.mark.parametrize('line', [b'[]', b'{}', b'not-json', b'\xff', b'null'])
def test_protocol_parse_failures_do_not_echo_received_data(line):
    with pytest.raises(sensor._ReaderError) as error:
        sensor._decode_reply(line)
    assert str(error.value) == '硬件温度读取器返回了无效响应'


def test_protocol_preserves_ready_zero_and_explicit_driver_permission_failures():
    for value in [reply(value=0), reply('permission_required'), reply('driver_missing')]:
        assert sensor._decode_reply(json.dumps(value).encode()) == value


def test_legacy_cpu_only_reply_remains_usable():
    legacy = reply()
    del legacy['gpu_temperatures']
    del legacy['gpu_temperature_reason']
    result = sensor._decode_reply(json.dumps(legacy).encode())
    assert result == reply()


@pytest.mark.parametrize('watts', [0, 74.125])
def test_cpu_package_power_keeps_real_watts_and_its_own_source(watts):
    value = {**reply(), 'cpu_power_w': watts,
             'cpu_power_source': 'LibreHardwareMonitor · CPU fixture · CPU Package', 'cpu_power_reason': None}
    assert sensor._decode_reply(json.dumps(value).encode()) == value


@pytest.mark.parametrize('change', [
    {'cpu_power_w': True}, {'cpu_power_w': -1}, {'cpu_power_w': float('nan')},
    {'cpu_power_w': float('inf')}, {'cpu_power_w': 10**400},
    {'cpu_power_source': None}, {'cpu_power_reason': 'contradictory failure'},
])
def test_cpu_power_protocol_rejects_invalid_or_contradictory_samples(change):
    value = {**reply(), 'cpu_power_w': 74.125,
             'cpu_power_source': 'LibreHardwareMonitor · CPU Package', 'cpu_power_reason': None, **change}
    with pytest.raises(sensor._ReaderError, match='无效响应'):
        sensor._decode_reply(json.dumps(value).encode())


def test_cpu_only_legacy_helper_does_not_imply_zero_power():
    value = reply()
    for key in ['cpu_power_w', 'cpu_power_source', 'cpu_power_reason']:
        del value[key]
    decoded = sensor._decode_reply(json.dumps(value).encode())
    assert decoded['cpu_power_w'] is None
    assert decoded['cpu_power_source'] is None
    assert decoded['cpu_power_reason'] == 'CPU 未提供封装功率传感器'


def test_power_sample_can_continue_without_temperature_and_clears_on_read_failure(monkeypatch):
    value = {**reply('error'), 'cpu_power_w': 0,
             'cpu_power_source': 'LibreHardwareMonitor · Package', 'cpu_power_reason': None}
    pipe = FakePipe([value, sensor._ReaderError('硬件读取超时')])
    api = install_api(monkeypatch, pipe)
    reader = sensor.CpuSensorReader()
    assert reader.enable()['cpu_power_w'] == 0
    assert not pipe.closed
    failed = reader.sample()
    assert failed['cpu_power_w'] is None and failed['cpu_power_source'] is None
    assert failed['cpu_power_reason'] == '硬件读取超时'
    assert pipe.closed and api.closed == [101]
    reader.close()


def test_gpu_protocol_preserves_fractions_exact_identity_and_unavailable_records():
    first = gpu_reply(pci='0A:1F.7')
    second = gpu_reply(value=None, identifier='/gpu-nvidia/1', pci=None, reason='缺少 GPU Core 当前值')
    value = reply('permission_required', gpus=[first, second])
    result = sensor._decode_reply(json.dumps(value).encode())
    assert result['gpu_temperatures'][0] == {**first, 'pci_bus_id': '0a:1f.7'}
    assert result['gpu_temperatures'][1] == second
    assert result['gpu_temperatures'][0]['name'] == result['gpu_temperatures'][1]['name']
    assert result['cpu_temperature_c'] is None


@pytest.mark.parametrize('change', [
    {'temperature_c': True}, {'temperature_c': float('nan')}, {'temperature_c': float('inf')},
    {'temperature_c': 10**400}, {'temperature_c': None}, {'source': 'GPU Hot Spot'},
    {'source': 'GPU Memory Junction'}, {'reason': 'contradictory failure'},
    {'pci_bus_id': '00000000:01:00.0'}, {'pci_bus_id': '01:20.0'}, {'pci_bus_id': '01:00.8'},
    {'identifier': ''}, {'name': None},
])
def test_invalid_gpu_record_is_a_protocol_failure(change):
    value = reply(gpus=[{**gpu_reply(), **change}])
    with pytest.raises(sensor._ReaderError, match='无效响应'):
        sensor._decode_reply(json.dumps(value).encode())


@pytest.mark.parametrize('gpus', [None, {}, ['not a record'], [{}]])
def test_gpu_array_shape_is_checked(gpus):
    value = {**reply(), 'gpu_temperatures': gpus}
    with pytest.raises(sensor._ReaderError, match='无效响应'):
        sensor._decode_reply(json.dumps(value).encode())


class PipeKernel:
    def __init__(self, reads):
        self.reads = iter(reads)
        self.written = bytearray()

    def WriteFile(self, handle, data, length, count, overlapped):
        self.written.extend(ctypes.string_at(data, length))
        ctypes.cast(count, ctypes.POINTER(sensor.w.DWORD)).contents.value = length
        return 1

    def ReadFile(self, handle, buffer, length, count, overlapped):
        data = next(self.reads)
        if data is None:
            return 0
        ctypes.memmove(buffer, data, len(data))
        ctypes.cast(count, ctypes.POINTER(sensor.w.DWORD)).contents.value = len(data)
        return 1


def make_pipe(reads):
    kernel = PipeKernel(reads)
    closed = []
    api = SimpleNamespace(kernel=kernel, close_handle=closed.append)
    return sensor._Pipe(api, 555, threading.Event()), kernel, closed


def test_pipe_uses_only_fixed_commands_and_accepts_fragmented_json(monkeypatch):
    wire = json.dumps(reply(value=0)).encode() + b'\n'
    pipe, kernel, closed = make_pipe([wire[:20], None, wire[20:]])
    monkeypatch.setattr(sensor.c, 'get_last_error', lambda: 232, raising=False)
    assert pipe.request()['cpu_temperature_c'] == 0
    pipe.close()
    assert kernel.written == b'sample\nclose\n'
    assert closed == [555]


@pytest.mark.parametrize('wire,match', [(b'{}\n{}\n', '多余'), (b'x' * 4096, '过长')])
def test_pipe_rejects_extra_lines_and_bounds_response_memory(monkeypatch, wire, match):
    monkeypatch.setattr(sensor, 'MAX_REPLY_BYTES', 8192)  # Exercise the bound with a small fixture.
    pipe, _, _ = make_pipe([wire] * 3)
    with pytest.raises(sensor._ReaderError, match=match):
        pipe.request()
    pipe.close()


def test_gpu_array_is_not_limited_by_the_old_four_scalar_reply_size():
    gpus = [gpu_reply(identifier=f'/gpu-nvidia/{index}', pci=f'{index:02x}:00.0') for index in range(64)]
    value = reply('permission_required', gpus=gpus)
    wire = json.dumps(value).encode() + b'\n'
    assert len(wire) > 8192
    pipe, _, _ = make_pipe([wire[index:index+4096] for index in range(0, len(wire), 4096)])
    assert pipe.request() == value
    pipe.close()


def test_pipe_eof_and_timeout_finish_with_clear_errors(monkeypatch):
    pipe, _, _ = make_pipe([b''])
    with pytest.raises(sensor._ReaderError, match='关闭连接'):
        pipe.request()
    pipe.close()
    pipe, _, _ = make_pipe([None])
    ticks = iter([0, sensor.SAMPLE_TIMEOUT_S + 1])
    monkeypatch.setattr(sensor.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(sensor.c, 'get_last_error', lambda: 232, raising=False)
    with pytest.raises(sensor._ReaderError, match='超时'):
        pipe.request()
    # Restore the real clock before close's bounded graceful write.
    monkeypatch.undo()
    pipe.close()


def test_pipe_zero_progress_write_has_a_deadline(monkeypatch):
    pipe, kernel, _ = make_pipe([])
    kernel.WriteFile = lambda *args: 1
    monkeypatch.setattr(sensor.time, 'monotonic', lambda: 100)
    with pytest.raises(sensor._ReaderError, match='超时'):
        pipe.send('sample', 99)
    monkeypatch.undo()
    pipe.close()


def test_connect_pins_actual_server_pid_and_closes_untrusted_handle():
    api = object.__new__(sensor._WindowsApi)
    closed = []
    def server_pid(handle, output):
        ctypes.cast(output, ctypes.POINTER(sensor.w.ULONG)).contents.value = 666
        return 1
    api.kernel = SimpleNamespace(CreateFileW=lambda *args: 555, WaitForSingleObject=lambda *args: 0x102,
                                 GetNamedPipeServerProcessId=server_pid, CloseHandle=lambda handle: closed.append(handle))
    with pytest.raises(sensor._ReaderError, match='身份不匹配'):
        api.connect('ModLoader.CpuSensors.' + 'a' * 32, 202, 101, threading.Event())
    assert closed == [555]


def test_connect_bounds_startup_wait_and_does_not_retry_unrelated_errors(monkeypatch):
    api = object.__new__(sensor._WindowsApi)
    api.kernel = SimpleNamespace(CreateFileW=lambda *args: sensor._INVALID_HANDLE,
                                 WaitForSingleObject=lambda *args: 0x102)
    ticks = iter([0, sensor.STARTUP_TIMEOUT_S + 1])
    monkeypatch.setattr(sensor.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(sensor.c, 'get_last_error', lambda: 2, raising=False)
    with pytest.raises(sensor._ReaderError, match='连接超时'):
        api.connect('fixture', 202, 101, threading.Event())
    monkeypatch.setattr(sensor.time, 'monotonic', lambda: 0)
    monkeypatch.setattr(sensor.c, 'get_last_error', lambda: 5)
    with pytest.raises(sensor._ReaderError, match='Windows 错误 5'):
        api.connect('fixture', 202, 101, threading.Event())


def test_win32_launch_is_hidden_runas_with_only_private_pipe_and_parent_pid():
    api = object.__new__(sensor._WindowsApi)
    captured = {}
    def execute(pointer):
        info = ctypes.cast(pointer, ctypes.POINTER(sensor._ShellExecuteInfo)).contents
        captured.update(verb=info.lpVerb, path=info.lpFile, params=info.lpParameters,
                        show=info.nShow, size=info.cbSize, flags=info.fMask)
        info.hProcess = 101
        return 1
    api.shell = SimpleNamespace(ShellExecuteExW=execute)
    api.kernel = SimpleNamespace(GetProcessId=lambda handle: 202)
    path = Path('C:/bundled/sensors/ModLoader.CpuSensors.exe')
    name = 'ModLoader.CpuSensors.' + 'a' * 32
    assert api.launch(path, name) == (101, 202)
    assert captured == {'verb': 'runas', 'path': str(path),
                        'params': f'--pipe {name} --parent-pid {sensor.os.getpid()}',
                        'show': 0, 'size': ctypes.sizeof(sensor._ShellExecuteInfo), 'flags': 0x540}


def test_win32_declined_uac_maps_to_permission_required(monkeypatch):
    api = object.__new__(sensor._WindowsApi)
    api.shell = SimpleNamespace(ShellExecuteExW=lambda *args: 0)
    monkeypatch.setattr(sensor.c, 'get_last_error', lambda: 1223, raising=False)
    with pytest.raises(sensor._ReaderError) as error:
        api.launch(Path('C:/bundled/sensors/ModLoader.CpuSensors.exe'), 'fixture')
    assert error.value.status == 'permission_required'
    assert '已取消' in str(error.value)


@pytest.mark.skipif(sensor.os.name != 'nt', reason='native helper targets Windows')
def test_built_native_partial_success_and_pci_format_contract():
    helper = Path(sensor.__file__).resolve().parent / 'sensors' / 'ModLoader.CpuSensors.exe'
    if not helper.is_file():
        pytest.skip('native helper has not been built')
    # Invoke only pure methods in the actual candidate. This does not create a
    # Computer, open PawnIO/GPU APIs, runas, or launch a persistent helper.
    script = r'''
    $ErrorActionPreference = 'Stop';
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new();
    $assembly = [Reflection.Assembly]::LoadFrom($env:MODLOADER_SENSOR_TEST_HELPER);
    $flags = [Reflection.BindingFlags]'NonPublic,Static';
    $program = $assembly.GetType('Program');
    $result = $program.GetMethod('Result', $flags);
    $keep = $program.GetMethod('KeepSession', $flags);
    $has = $program.GetMethod('HasTemperature', $flags);
    $cases = @();
    foreach ($kind in @('missing','cpu','gpu-zero','gpu-missing')) {
        $status = if ($kind -eq 'cpu') { 'ready' } else { 'permission_required' };
        $value = $result.Invoke($null, [object[]]@($status,'fixture',$null,$null));
        if ($kind -eq 'cpu') { $value['cpu_temperature_c']=40.0; $value['cpu_temperature_source']='CPU fixture'; $value['cpu_temperature_reason']=$null; }
        if ($kind -like 'gpu-*') {
            $temperature = if ($kind -eq 'gpu-zero') { 0.0 } else { $null };
            $gpu = New-Object 'Collections.Generic.Dictionary[string,object]';
            $gpu.Add('identifier','/gpu-nvidia/0'); $gpu.Add('name','GPU fixture'); $gpu.Add('pci_bus_id','01:00.0');
            $gpu.Add('source','GPU Core'); $gpu.Add('temperature_c',$temperature);
            $gpu.Add('reason',$(if ($null -eq $temperature) { 'fixture missing' } else { $null }));
            $value['gpu_temperatures'].Add($gpu);
        }
        $cases += @{kind=$kind; keep=$keep.Invoke($null,[object[]]@($value)); has=$has.Invoke($null,[object[]]@($value))};
    }
    $format = $assembly.GetType('Native').GetMethod('FormatPciBusId', $flags);
    $pci = @();
    foreach ($pair in @(@(255,2031623),@(256,0),@(0,2097152),@(0,8))) {
        $pci += $format.Invoke($null,[object[]]@([uint32]$pair[0],[uint32]$pair[1]));
    }
    @{cases=$cases; pci=$pci} | ConvertTo-Json -Depth 5 -Compress;
    '''
    env = {**sensor.os.environ, 'MODLOADER_SENSOR_TEST_HELPER': str(helper)}
    process = sensor.subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                                    capture_output=True, encoding='utf-8', timeout=15, creationflags=0x08000000, env=env)
    assert process.returncode == 0, process.stderr
    assert not process.stderr
    result = json.loads(process.stdout.lstrip('\ufeff'))
    assert result['cases'] == [
        {'kind': 'missing', 'keep': False, 'has': False},
        {'kind': 'cpu', 'keep': True, 'has': True},
        {'kind': 'gpu-zero', 'keep': True, 'has': True},
        {'kind': 'gpu-missing', 'keep': True, 'has': False},
    ]
    assert result['pci'] == ['ff:1f.7', None, None, None]
