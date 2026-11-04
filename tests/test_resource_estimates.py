import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from modloader.models import LoadProfile, ModelAsset, RuntimeInstallation
from modloader.resource_estimates import (MIB, ResourceEstimator, llama_estimate_args,
                                         parse_llama_fit, strata_estimate, vllm_estimate)


@pytest.fixture(autouse=True)
def synthetic_cuda_identity(monkeypatch):
    monkeypatch.setattr('modloader.resource_estimates.cuda_devices', lambda: [])
    monkeypatch.setattr('modloader.resource_estimates.cuda_devices_in_environment', lambda *args: [])


def fixtures(tmp_path, backend="llama_cpp"):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    model = ModelAsset(id="model", name="fixture", path=str(model_path), format="gguf")
    runtime = RuntimeInstallation(id="runtime", backend=backend, name="fixture", path=str(tmp_path / "llama-server.exe"))
    (tmp_path / "llama-fit-params.exe").write_bytes(b"fixture, never execute")
    profile = LoadProfile(id="profile", model_id="model", name="fixture", backend=backend, runtime_id="runtime", alias="test")
    return model, profile, runtime


def test_launch_defaults_blanks_and_last_extra_override(tmp_path):
    _, profile, runtime = fixtures(tmp_path)
    profile.parameters = {"context_length": "  ", "gpu_layers": 0, "parallel": 3}
    profile.extra_args = ["-c", "65536", "--cache-type-k=f16", "--n-gpu-layers", "-1", "--metrics"]
    args, notes = llama_estimate_args(profile, runtime)
    assert args[:6] == ["--ctx-size", "32768", "--n-gpu-layers", "0", "--parallel", "3"]
    assert args[-6:] == ["--ctx-size", "65536", "--cache-type-k", "f16", "--n-gpu-layers", "-1"]
    assert not notes
    assert profile.parameters["context_length"] == "  "
    assert "--host" not in args and "--port" not in args


def test_unsupported_and_projection_remain_partial(tmp_path):
    _, profile, runtime = fixtures(tmp_path)
    profile.parameters["mmproj"] = "vision.gguf"
    profile.extra_args = ["--spec-type", "mtp", "--future-allocation", "12"]
    args, notes = llama_estimate_args(profile, runtime)
    assert "--spec-type" not in args
    assert len(notes) == 3
    result = parse_llama_fit("CUDA0 18865 1237 527\nHost 2425 0 52\n", notes)
    assert result["resources"][0]["bytes"] == (18865 + 1237 + 527) * MIB
    assert all(r["quality"] == "partial" for r in result["resources"])


def test_disabled_vision_does_not_report_omitted_projector(tmp_path):
    _, profile, runtime = fixtures(tmp_path)
    profile.parameters.update(mmproj='saved-vision.gguf', vision_enabled=False)
    _, notes = llama_estimate_args(profile, runtime)
    assert not any('视觉' in note for note in notes)


def test_native_parser_multiple_devices_and_host_are_not_rss():
    result = parse_llama_fit("diagnostics ignored\nCUDA0 100 20 3\nCUDA1 50 10 2\nHost 2 0 1\n")
    assert [r["kind"] for r in result["resources"]] == ["vram", "vram", "ram"]
    assert result["resources"][1]["device"] == "CUDA1"
    assert result["resources"][0]["quality"] == "estimate"
    assert result["resources"][2]["quality"] == "partial"
    assert result["resources"][2]["label"] == "系统内存"
    assert result["resources"][2]["bytes"] == MIB
    assert result["resources"][2]["logical"] is False
    assert result["resources"][2]["model_reference_bytes"] == 2*MIB
    assert "Host 模型缓冲参考" in result["resources"][2]["notes"][1]
    assert parse_llama_fit("CUDA0 -1 2 3\n")["resources"] == []


@pytest.mark.asyncio
async def test_native_fit_rows_use_cuda_identity_not_hardware_enumeration_order(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path)
    estimator = ResourceEstimator()
    async def native(*args):
        return parse_llama_fit('CUDA0 100 20 3\nCUDA1 50 10 2\n')
    monkeypatch.setattr(estimator, '_native', native)
    monkeypatch.setattr('modloader.resource_estimates.cuda_devices', lambda: [
        dict(device='CUDA0', gpu_id='GPU-nvml-index-1', pci_bus_id='0000:02:00.0'),
        dict(device='CUDA1', gpu_id='GPU-nvml-index-0', pci_bus_id='0000:01:00.0')])
    rows = (await estimator.estimate(model, profile, runtime))['resources']
    assert [row['gpu_id'] for row in rows] == ['GPU-nvml-index-1', 'GPU-nvml-index-0']
    assert rows[1]['bytes'] == 62 * MIB and rows[1]['identity_status'] == 'resolved'
    monkeypatch.setattr('modloader.resource_estimates.cuda_devices', lambda: [])
    cached = (await estimator.estimate(model, profile, runtime))['resources']
    assert all(row['identity_status'] == 'unknown' and 'gpu_id' not in row for row in cached)


@pytest.mark.asyncio
@pytest.mark.parametrize('selection,index', [(0, 0), ('0', 0), ([0], 0), (2, 2), ('2', 2), ([2], 2), (' 2 ', 2)])
async def test_strata_single_selection_maps_child_visibility_environment(tmp_path, monkeypatch, selection, index):
    model, profile, runtime = fixtures(tmp_path, 'strata')
    runtime.metadata['python'] = 'D:/Strata/python/python.exe'
    profile.parameters.update(gpu=selection, strata_config={'gpu': 0})
    environments = []
    def read(environment, python, diagnostics):
        assert diagnostics == []
        environments.append((environment, python))
        return [dict(device='CUDA0', gpu_id='GPU-selected', pci_bus_id='0000:03:00.0')]
    monkeypatch.setattr('modloader.resource_estimates.cuda_devices_in_environment', read)
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['device'] == f'CUDA{index}' and gpu['gpu_id'] == 'GPU-selected'
    assert environments == [({'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': str(index)}, runtime.metadata['python'])]


@pytest.mark.asyncio
async def test_strata_identity_failure_keeps_estimate_and_reports_diagnostic(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path, 'strata')
    profile.parameters.update(gpu='0', strata_config={'gpu': 0})
    before = strata_estimate(model, profile, runtime)
    def read(environment, python, diagnostics):
        diagnostics.append('GPU 身份查询超过 10 秒')
        return []
    monkeypatch.setattr('modloader.resource_estimates.cuda_devices_in_environment', read)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert result['diagnostic'] == 'GPU 身份查询超过 10 秒'
    assert result['resources'][0] == {**before['resources'][0], 'identity_status': 'unknown'}
    assert result['resources'][1:] == before['resources'][1:]


def test_strata_multi_card_remains_aggregate_and_blank_selection_keeps_imported_gpu(tmp_path):
    model, profile, runtime = fixtures(tmp_path, 'strata')
    profile.parameters.update(gpu='1,3', strata_config={'gpu': 0})
    gpu = strata_estimate(model, profile, runtime)['resources'][0]
    assert gpu['aggregate'] and gpu['device'] == '多 GPU 合计'
    profile.parameters.update(gpu='  ', strata_config={'gpu': 3})
    assert strata_estimate(model, profile, runtime)['resources'][0]['device'] == 'CUDA3'
    profile.extra_args = ['--layer-split', 'auto']
    assert strata_estimate(model, profile, runtime)['resources'][0]['aggregate']
    profile.parameters.clear()
    assert strata_estimate(model, profile, runtime)['resources'][0]['aggregate']


def test_llama_real_all_gpu_output_still_has_host_compute():
    # Retained Qwen3.8-27B-NVFP4-MTP-HIGH --fit-print output. Its verbose
    # transcript reports 66/66 layers offloaded, and CUDA_Host compute 52.02 MiB.
    result = parse_llama_fit('CUDA0 15536 2197 505\nHost 994 0 52\n')
    gpu, ram = result['resources']
    assert gpu['bytes'] == (15536+2197+505)*MIB
    assert ram['bytes'] == 52*MIB and ram['logical'] is False
    assert ram['model_reference_bytes'] == 994*MIB
    assert ram['components'] == [{'label':'计算缓冲','bytes':52*MIB}]


def test_llama_cpu_context_is_physical_but_model_reference_is_separate():
    ram = parse_llama_fit('CPU 4096 128 64\n')['resources'][0]
    assert ram['bytes'] == 192*MIB and ram['logical'] is False
    assert ram['model_reference_bytes'] == 4096*MIB
    assert sum(item['bytes'] for item in ram['components']) == ram['bytes']


@pytest.mark.parametrize('model_bytes', [0, 994])
def test_llama_pure_host_reference_is_explicitly_logical(model_bytes):
    ram = parse_llama_fit(f'Host {model_bytes} 0 0\n')['resources'][0]
    assert ram['logical'] is True and ram['bytes'] == model_bytes*MIB
    assert ram['components'] == []


@pytest.mark.parametrize("diagnostics, expected", [
    ("0.00.429.895 D llama_context: flash_attn = auto\n"
     "0.00.436.348 D resolve_fused_ops: Flash Attention enabled\n", "已启用"),
    ("0.00.403.339 D llama_context: flash_attn = auto\n"
     "0.00.410.215 D resolve_fused_ops: layer 3 is assigned to device CUDA0 but Flash Attention is assigned to device CPU (usually due to missing support)\n"
     "0.00.410.217 D resolve_fused_ops: Flash Attention not supported, set to disabled\n",
     "已自动关闭（当前配置未通过设备支持检测）"),
    ("llama_context: flash_attn = disabled\n", "已关闭"),
    ("llama_init_from_model: enabling flash_attn since it is required for quantized V cache\n"
     "llama_context: flash_attn = enabled\n", "已启用"),
    ("llama_context: flash_attn = auto\n", "状态未确认（运行时未返回明确结果）"),
    ("", "状态未确认（运行时未返回明确结果）"),
])
def test_flash_attention_note_uses_resolved_native_log(diagnostics, expected):
    result = parse_llama_fit("CUDA0 15536 1717 1732\nHost 994 0 84\n", diagnostics=diagnostics)
    assert result["notes"][0] == f"Flash Attention（预估）：{expected}。"
    assert result["resources"][0]["bytes"] == (15536 + 1717 + 1732) * MIB


@pytest.mark.asyncio
async def test_native_estimate_collects_flash_attention_diagnostics(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path)
    argv = []
    class Process:
        returncode = 0
        async def communicate(self):
            return (b"CUDA0 15536 1717 1732\n",
                    b"llama_context: flash_attn = auto\n"
                    b"resolve_fused_ops: Flash Attention not supported, set to disabled\n")
    async def create(*args, **kwargs):
        argv.extend(args)
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert argv[argv.index("--log-verbosity") + 1] == "5"
    assert result["notes"][0] == "Flash Attention（预估）：已自动关闭（当前配置未通过设备支持检测）。"


def test_strata_import_overrides_dynamic_reserve_and_no_writes(tmp_path):
    model, profile, runtime = fixtures(tmp_path, "strata")
    source = tmp_path / "import.json"
    raw = {"args": ["--max-context", "262144", "--expert-cache", "auto", "--kv", "int8",
                    "--resident-experts", "--vram-reserve-mib", "2048", "--mtp", "mtp"], "gpu": 0}
    source.write_text(json.dumps(raw))
    profile.parameters = {"configuration": str(source), "context_length": 65536}
    profile.extra_args = ["--vram-reserve-mib", "1024"]
    result = strata_estimate(model, profile, runtime)
    assert result["resources"][0]["reserve_bytes"] == 1024 * MIB
    assert result["resources"][0]["bytes"] is None
    assert result["resources"][0]["quality"] == "dynamic"
    assert "65536" in result["resources"][0]["notes"][0]
    assert result["resources"][1]["quality"] == "dynamic"
    assert json.loads(source.read_text()) == raw
    assert not (tmp_path / "strata.json").exists()
    profile.extra_args = ["--vram-reserve-mib", "-10"]
    assert "reserve_bytes" not in strata_estimate(model, profile, runtime)["resources"][0]


def test_vllm_budget_not_claimed_as_allocation(tmp_path):
    _, profile, runtime = fixtures(tmp_path, "vllm")
    profile.parameters["gpu_memory_utilization"] = .85
    profile.extra_args = ["--gpu-memory-utilization=.7"]
    gpu = vllm_estimate(profile, runtime)["resources"][0]
    assert gpu["budget_fraction"] == .7 and gpu["bytes"] is None and gpu["quality"] == "dynamic"
    profile.extra_args = ["--gpu-memory-utilization", "-1"]
    assert "budget_fraction" not in vllm_estimate(profile, runtime)["resources"][0]
    profile.extra_args = ['--gpu-memory-utilization', '.7', '--kv-cache-memory-bytes', '1073741824']
    assert 'budget_fraction' not in vllm_estimate(profile, runtime)['resources'][0]


@pytest.mark.asyncio
async def test_same_configuration_shares_native_work_and_stat_invalidates_cache(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path)
    estimator = ResourceEstimator()
    calls = []
    started, finish = asyncio.Event(), asyncio.Event()
    async def native(*args):
        calls.append(args)
        started.set()
        await finish.wait()
        return parse_llama_fit("CUDA0 100 20 3\n")
    monkeypatch.setattr(estimator, "_native", native)
    first = asyncio.create_task(estimator.estimate(model, profile, runtime))
    second = asyncio.create_task(estimator.estimate(model, profile, runtime))
    await started.wait()
    finish.set()
    one, two = await asyncio.gather(first, second)
    assert len(calls) == 1
    one["resources"][0]["bytes"] = 0
    assert two["resources"][0]["bytes"] == 123 * MIB
    assert (await estimator.estimate(model, profile, runtime))["resources"][0]["bytes"] == 123 * MIB
    assert len(calls) == 1
    Path(model.path).write_bytes(b"GGUF changed")
    await estimator.estimate(model, profile, runtime)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cancel_last_waiter_cancels_obsolete_work(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path)
    estimator = ResourceEstimator()
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def native(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    monkeypatch.setattr(estimator, "_native", native)
    task = asyncio.create_task(estimator.estimate(model, profile, runtime))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cancelled.wait(), 1)
    assert not estimator._inflight


@pytest.mark.asyncio
async def test_native_progress_queue_shared_waiter_and_cancellation(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path)
    estimator = ResourceEstimator()
    started, release = asyncio.Event(), asyncio.Event()
    processes = []
    class Process:
        returncode = None
        async def communicate(self):
            started.set()
            await release.wait()
            self.returncode = 0
            return b'CUDA0 100 20 3\n', b''
        def kill(self):
            pytest.fail('A remaining subscriber still needs the native estimate')
    async def create(*args, **kwargs):
        process = Process()
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create)
    first_events, shared_events, queued_events = [], [], []
    first = asyncio.create_task(estimator.estimate(model, profile, runtime, first_events.append))
    await started.wait()
    shared = asyncio.create_task(estimator.estimate(model, profile, runtime, shared_events.append))
    other_profile = profile.model_copy(update={'parameters': {'context_length': 8192}})
    queued = asyncio.create_task(estimator.estimate(model, other_profile, runtime, queued_events.append))
    for _ in range(5):
        await asyncio.sleep(0)
    assert any(event['stage'] == 'native_estimate' for event in shared_events)
    assert any(event['stage'] == 'estimate_queue' for event in queued_events)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert len(processes) == 1
    release.set()
    one, two = await asyncio.gather(shared, queued)
    assert one['resources'] and two['resources'] and len(processes) == 2
    assert [event['stage'] for event in queued_events][-2:] == ['native_estimate', 'gpu_identity']
    assert not any('total' in event or 'percent' in event for event in shared_events + queued_events)


@pytest.mark.asyncio
async def test_native_timeout_kills_process_without_forwarding_listener_args(tmp_path, monkeypatch):
    import modloader.resource_estimates as module
    model, profile, runtime = fixtures(tmp_path)
    argv = []
    class Process:
        returncode = None
        killed = False
        async def communicate(self):
            if self.killed:
                return b"", b""
            await asyncio.Event().wait()
        def kill(self):
            self.killed = True
            self.returncode = -1
    process = Process()
    async def create(*args, **kwargs):
        argv.extend(args)
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(module, "NATIVE_TIMEOUT_SECONDS", .01)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert process.killed and not result["resources"]
    assert "--fit-print" in argv and "--offline" in argv
    assert argv[argv.index("--fit-print") + 1] == "on"
    assert "--host" not in argv and "--port" not in argv and "--alias" not in argv


@pytest.mark.asyncio
async def test_unknown_backend_and_missing_estimator_do_not_spawn(tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail('Unsupported previews must not spawn an engine')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    model, profile, runtime = fixtures(tmp_path)
    (tmp_path / "llama-fit-params.exe").unlink()
    assert not (await ResourceEstimator().estimate(model, profile, runtime))["resources"]
    for backend in ('transformers', 'custom'):
        profile.backend = runtime.backend = backend
        result = await ResourceEstimator().estimate(model, profile, runtime)
        assert not result['resources']
        assert result['notes'] == ['尚未适配此引擎的资源预估。']


@pytest.mark.asyncio
async def test_missing_model_reports_missing_file_without_spawning(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path)
    Path(model.path).unlink()
    async def no_process(*args, **kwargs):
        pytest.fail('Missing weights must not spawn the native estimator')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', no_process)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert result['resources'] == []
    assert result['notes'] == ['模型文件不存在，无法估算']
    assert result['diagnostic'] == model.path


@pytest.mark.asyncio
async def test_failed_native_estimate_retains_actual_diagnostics(tmp_path, monkeypatch):
    model, profile, runtime = fixtures(tmp_path)
    class Process:
        returncode = 3221226505
        async def communicate(self):
            return b'', b'gguf_init_from_file: failed to open GGUF file (No such file or directory)'
    async def create(*args, **kwargs):
        return Process()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert result['notes'] == ['原生估算失败']
    assert '3221226505' in result['diagnostic']
    assert 'No such file or directory' in result['diagnostic']


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
async def test_endpoint_estimates_unsaved_draft_without_mutating_store(tmp_path, monkeypatch, streaming):
    import httpx
    from modloader.app import Controller, build_app
    model, profile, runtime = fixtures(tmp_path)
    controller = Controller(tmp_path / "data", "fixture-token")
    for kind, item in [("models", model), ("profiles", profile), ("runtimes", runtime)]:
        controller.store.put(kind, item.model_dump())
    calls = []
    async def estimate(model_arg, draft, runtime_arg, progress=None):
        calls.append((model_arg, draft, runtime_arg))
        if progress:
            progress({'stage': 'native_estimate', 'message': '计算模型、上下文与缓冲占用'})
            await asyncio.sleep(.01)
        return parse_llama_fit("CUDA0 100 20 3\n")
    monkeypatch.setattr(controller.resource_estimator, "estimate", estimate)
    draft = {"model_id": model.id, "runtime_id": runtime.id, "backend": "llama_cpp",
             "parameters": {"context_length": 123456}, "extra_args": ["--parallel", "3"]}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url="http://testserver",
                                    headers={"authorization": "Bearer fixture-token", **({'accept': 'application/x-ndjson'} if streaming else {})}) as client:
            response = await client.post("/api/resources/estimate", json=draft)
            assert response.status_code == 200
            if streaming:
                events = [json.loads(line) for line in response.text.splitlines()]
                assert any(event['type'] == 'progress' and event['stage'] == 'native_estimate' for event in events)
                assert events[-1]['type'] == 'result'
                result = events[-1]['value']
            else:
                result = response.json()
            assert result["resources"][0]["bytes"] == 123 * MIB
            assert calls[0][1].parameters == draft["parameters"]
            assert calls[0][1].extra_args == draft["extra_args"]
            assert controller.store.all("profiles") == [profile.model_dump()]
            assert not controller.runtime.instances
            bad = await client.post("/api/resources/estimate", json={**draft, "backend": "strata"})
            assert bad.status_code == 400 and len(calls) == 1
    finally:
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['ready', 'missing', 'invalid_explicit'])
async def test_strata_endpoint_resolves_resources_without_preparing_or_loading(tmp_path, monkeypatch, state):
    import httpx
    from modloader.app import Controller, build_app
    model, profile, runtime = fixtures(tmp_path, 'strata')
    controller = Controller(tmp_path / 'data', 'fixture-token')
    for kind, item in [('models', model), ('profiles', profile), ('runtimes', runtime)]:
        controller.store.put(kind, item.model_dump())
    before = controller.store.all('profiles')
    calls = []
    def preflight(model_arg, draft, runtime_arg):
        assert model_arg.id == model.id and runtime_arg.id == runtime.id
        assert draft.parameters['context_length'] == 8192
        return {'status': 'ready' if state == 'ready' else 'error', 'message': 'fixture missing pack',
                'pack': 'resolved-pack', 'tokenizer': 'resolved-tokenizer', 'expert_profile': 'resolved-ranking'}
    async def estimate(model_arg, draft, runtime_arg, progress=None):
        calls.append(draft)
        return {'resources': [{'kind': 'vram', 'label': 'GPU 显存', 'bytes': None,
                               'quality': 'dynamic', 'fixed_bytes': 123, 'notes': []}], 'notes': []}
    async def forbidden(*args, **kwargs):
        pytest.fail('Preview must not prepare weights or start an engine')
    monkeypatch.setattr(controller.preparation.strata, 'preflight', preflight)
    monkeypatch.setattr(controller.preparation, 'ensure', forbidden)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    monkeypatch.setattr(controller.resource_estimator, 'estimate', estimate)
    parameters = {'context_length': 8192}
    if state == 'invalid_explicit':
        parameters['pack'] = 'stale-pack'
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                    headers={'authorization': 'Bearer fixture-token'}) as client:
            response = await client.post('/api/resources/estimate', json={
                'model_id': model.id, 'runtime_id': runtime.id, 'backend': 'strata',
                'parameters': parameters, 'extra_args': []})
            assert response.status_code == 200
            result = response.json()
            if state == 'invalid_explicit':
                assert result['resources'] == [] and not calls
            else:
                assert result['resources'][0]['fixed_bytes'] == 123
                assert len(calls) == 1
                if state == 'ready':
                    assert calls[0].parameters == {**parameters, 'pack': 'resolved-pack',
                                                   'tokenizer': 'resolved-tokenizer', 'expert_profile': 'resolved-ranking'}
                else:
                    assert calls[0].parameters == parameters
            if state != 'ready':
                assert 'fixture missing pack' in result['diagnostic']
            assert controller.store.all('profiles') == before
            assert not controller.runtime.instances
            assert not controller.preparation.strata.root.exists()
    finally:
        await controller.close()
