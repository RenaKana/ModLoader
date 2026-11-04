import asyncio
import ast
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from modloader.backends import get_adapters, import_configuration, import_runtime, install_backend, uninstall_runtime, collect_telemetry
from modloader.backends.adapters import windows_to_wsl, WSL_SUPERVISOR
from modloader.backends.installers import _extract
from modloader.models import AppPaths, LaunchContext, LaunchSpec, LoadProfile, ModelAsset, RuntimeInstallation, RuntimeInstance


def test_legacy_strata_detection_uses_explicit_environment_not_machine_paths(tmp_path, monkeypatch):
    runtime = tmp_path / 'Strata'
    (runtime / 'serve').mkdir(parents=True)
    (runtime / 'serve/server.py').write_text('# fixture', encoding='utf-8')
    monkeypatch.setenv('MODLOADER_STRATA_DEV_RUNTIME', str(runtime))
    monkeypatch.setattr('modloader.backends.adapters.import_runtime', lambda backend, path: (backend, path))
    assert get_adapters()['strata'].detect() == [('strata', str(runtime))]
    monkeypatch.delenv('MODLOADER_STRATA_DEV_RUNTIME')
    assert get_adapters()['strata'].detect() == []


def fixtures(tmp_path, backend="llama_cpp", hf=False):
    model_path = tmp_path / ("hf model" if hf else "part-00001-of-00002.gguf")
    if hf:
        model_path.mkdir()
        (model_path / "config.json").write_text('{"model_type":"unknown_future"}')
    else:
        model_path.write_bytes(b"GGUF")
    model = ModelAsset(id="model", name="Fixture", path=str(model_path), format="hf" if hf else "gguf")
    runtime = RuntimeInstallation(id="runtime", backend=backend, name="Fixture", path=str(tmp_path / "engine.exe"))
    profile = LoadProfile(id="profile", model_id=model.id, name="Fixture", backend=backend, runtime_id=runtime.id, alias="test-model")
    work = tmp_path / "instance"
    work.mkdir()
    context = LaunchContext(port=18764, instance_id="unique-instance", work_dir=str(work), data_dir=str(tmp_path))
    return model, profile, runtime, context


@pytest.mark.parametrize("blank", ["missing", None, "", " \t "])
def test_llama_blank_parameters_apply_editor_defaults_without_saving_them(tmp_path, blank):
    model, profile, runtime, context = fixtures(tmp_path)
    expected = {"context_length": ("--ctx-size", "32768"), "gpu_layers": ("--n-gpu-layers", "-1"),
                "parallel": ("--parallel", "1"), "cache_type_k": ("--cache-type-k", "q8_0"),
                "cache_type_v": ("--cache-type-v", "q8_0"), "flash_attention": ("--flash-attn", "auto")}
    if blank != "missing":
        profile.parameters = {key: blank for key in expected}
    original = dict(profile.parameters)
    spec = get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)
    for flag, value in expected.values():
        assert spec.argv.count(flag) == 1
        assert spec.argv[spec.argv.index(flag) + 1] == value
    assert "--threads" not in spec.argv
    assert profile.parameters == original


def test_llama_explicit_zero_and_false_are_not_replaced_by_defaults(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path)
    profile.parameters = {"context_length": 0, "gpu_layers": 0, "parallel": 2,
                          "cache_type_k": "f16", "cache_type_v": "f16", "flash_attention": False}
    spec = get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)
    for flag, value in [("--ctx-size", "0"), ("--n-gpu-layers", "0"), ("--parallel", "2"),
                        ("--cache-type-k", "f16"), ("--cache-type-v", "f16")]:
        assert spec.argv[spec.argv.index(flag) + 1] == value
    assert "--flash-attn" not in spec.argv
    assert profile.parameters["flash_attention"] is False


def test_llama_extra_args_still_follow_resolved_defaults(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path)
    profile.extra_args = ["--ctx-size", "65536", "--cache-type-k=f16", "--parallel", "2"]
    spec = get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index("--ctx-size") + 1] == "32768"
    assert spec.argv[-len(profile.extra_args):] == profile.extra_args
    assert profile.parameters == {}


@pytest.mark.parametrize('mode', ['none', 'layer', 'row', 'tensor', 'future-native-mode'])
def test_llama_native_multi_gpu_parameters_preserve_extra_arg_priority(tmp_path, mode):
    model, profile, runtime, context = fixtures(tmp_path)
    profile.parameters = {'devices': 'CUDA2,CUDA0', 'split_mode': mode, 'tensor_split': '3/1', 'main_gpu': 0}
    profile.extra_args = ['--device', 'CUDA3', '--split-mode=row', '--tensor-split', '1,2', '--main-gpu', '2']
    before = profile.model_dump()
    argv = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context).argv
    for flag, value in [('--device', 'CUDA2,CUDA0'), ('--split-mode', mode),
                        ('--tensor-split', '3/1'), ('--main-gpu', '0')]:
        assert argv[argv.index(flag) + 1] == value
    assert argv[-len(profile.extra_args):] == profile.extra_args
    assert profile.model_dump() == before


@pytest.mark.parametrize('blank', [None, '', ' \t'])
def test_llama_optional_gpu_fields_do_not_change_legacy_launch(tmp_path, blank):
    model, profile, runtime, context = fixtures(tmp_path)
    baseline = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context)
    profile.parameters = {key: blank for key in ['devices', 'split_mode', 'tensor_split', 'main_gpu']}
    assert get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context) == baseline


def test_llama_gpu_launcher_import_preserves_device_text_and_single_ratio(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path)
    script = tmp_path / 'multi-gpu.cmd'
    script.write_text('llama-server.exe --model "' + model.path + '" --device none '
                      '--split-mode tensor --tensor-split 1 --main-gpu 0')
    imported = import_configuration(str(script))
    assert imported['parameters'] == {'devices': 'none', 'split_mode': 'tensor', 'tensor_split': '1', 'main_gpu': 0}
    assert imported['extra_args'] == []
    profile.parameters = imported['parameters']
    argv = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context).argv
    assert argv[argv.index('--device') + 1] == 'none'


@pytest.mark.parametrize('gpu', [None, '', '  ', 7, [4, 9], '4,9'])
def test_strata_gpu_override_keeps_native_types_and_blank_inheritance(tmp_path, gpu):
    from test_mtp import strata
    model, profile, runtime, context, _ = strata(tmp_path)
    profile.parameters['strata_config']['gpu'] = [2, 8]
    profile.parameters['gpu'] = gpu
    before = profile.model_dump()
    get_adapters()['strata'].build_launch(model, profile, runtime, context)
    generated = json.loads((Path(context.work_dir) / 'strata.json').read_text())
    assert generated['gpu'] == ([2, 8] if gpu is None or isinstance(gpu, str) and not gpu.strip() else gpu)
    assert profile.model_dump() == before


@pytest.mark.parametrize('gpu', [2, [0, 5], '0,5'])
def test_strata_gpu_import_keeps_native_value(tmp_path, gpu):
    source = tmp_path / 'strata-gpu.json'
    source.write_text(json.dumps({'gpu': gpu, 'args': ['--native', 'model.gguf']}))
    imported = import_configuration(str(source))
    assert imported['parameters']['gpu'] == gpu
    assert imported['parameters']['strata_config']['gpu'] == gpu


def test_llama_uses_local_model_and_no_shell(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path)
    profile.parameters = {"context_length": 65536, "gpu_layers": -1}
    spec = get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index("--model") + 1] == str(Path(model.path).resolve())
    assert spec.argv[spec.argv.index("--host") + 1] == "127.0.0.1"
    assert spec.argv[spec.argv.index("--ctx-size") + 1] == "65536"
    assert spec.env["LLAMA_ARG_OFFLINE"] == "1"
    assert spec.model_name == profile.alias
    assert spec.health_path == "/health" and spec.health_expected == {"status": "ok"}


@pytest.mark.parametrize("flag", ["--host", "--port=9000", "--api-key", "--mcp-servers-json", "--hf-repo", "--mmproj-url", "--trust-remote-code", "--models-dir", "--gpt-oss-20b-default"])
def test_extra_args_cannot_escape_instance_boundary(tmp_path, flag):
    model, profile, runtime, context = fixtures(tmp_path)
    profile.extra_args = [flag, "value"]
    with pytest.raises(ValueError):
        get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)


def test_known_runtime_help_rejects_unknown_advanced_flag(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path)
    runtime.metadata["supported_flags"] = ["--threads"]
    profile.extra_args = ["--unknown-new-version-only", "1"]
    with pytest.raises(ValueError, match="--help"):
        get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)


def test_compatibility_is_not_inferred_from_suffix(tmp_path):
    model, _, runtime, _ = fixtures(tmp_path)
    adapters = get_adapters()
    assert adapters["llama_cpp"].inspect(model, runtime).status == "unverified"
    model.metadata["missing_shards"] = ["part-00002-of-00002.gguf"]
    assert adapters["llama_cpp"].inspect(model, runtime).status == "unsupported"
    model.metadata = {"is_mmproj": True}
    assert adapters["strata"].inspect(model, runtime).status == "unsupported"
    model.format = "hf"
    assert adapters["llama_cpp"].inspect(model, runtime).status == "unsupported"


@pytest.mark.parametrize("kv_cache", ["int8", "fp16", "q4_0", "k8v4"])
def test_strata_import_and_derived_config_preserve_inference_only(tmp_path, kv_cache, monkeypatch):
    monkeypatch.delenv("MODLOADER_STRATA_DEV_RUNTIME", raising=False)
    monkeypatch.delenv("MODLOADER_STRATA_DEV_EXE", raising=False)
    from test_strata_preparation import write_expert_profile
    model, profile, runtime, context = fixtures(tmp_path, "strata")
    model.metadata.update({'qwen4exp.block_count':48, 'qwen4exp.expert_count':512})
    root = tmp_path / "Strata"
    (root / "engine").mkdir(parents=True)
    (root / "engine/strata.exe").touch()
    write_expert_profile(root / 'data' / 'expert-profile.bin')
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.json").write_text("{}")
    pack = tmp_path / "pack"
    pack.mkdir()
    source = tmp_path / "original.json"
    raw = dict(exe="engine/strata.exe", cwd=str(root), tokenizer=str(tokenizer), model_name="original", lib_dirs=[str(tmp_path / "original-cuda")],
               args=["--native", model.path, "--pack", str(pack), "--max-context", "32768", "--kv", "int8"],
               api_key="fixture-secret-do-not-copy", mcp_servers=[{"command": "never-run"}], env={"SECRET": "fixture-secret-do-not-copy"}, before_load="never-run")
    original = json.dumps(raw)
    source.write_text(original)
    imported = import_configuration(str(source))
    assert "fixture-secret" not in json.dumps(imported)
    assert "mcp" not in json.dumps(imported)
    profile.parameters = imported["parameters"]
    profile.parameters["context_length"] = 65536
    profile.parameters["kv_cache"] = kv_cache
    runtime.path, runtime.metadata = str(root), {"python": "python.exe"}
    spec = get_adapters()["strata"].build_launch(model, profile, runtime, context)
    generated = json.loads((Path(context.work_dir) / "strata.json").read_text())
    assert generated["cwd"] == str(root)
    assert generated["lib_dirs"] == raw["lib_dirs"]
    assert generated["host"] == "127.0.0.1" and generated["port"] == context.port
    assert generated["model_name"] == profile.alias
    assert generated["args"][generated["args"].index("--max-context") + 1] == "65536"
    assert generated["args"].count("--kv") == 1
    assert generated["args"][generated["args"].index("--kv") + 1] == kv_cache
    assert "api_key" not in generated and "mcp_servers" not in generated and "env" not in generated
    assert spec.cwd == str(root)
    assert spec.health_path == "/health" and spec.health_expected == {"status": "ok", "loaded": True}
    assert source.read_text() == original
    assert "log" not in generated
    assert generated['args'][generated['args'].index('--spec') + 1] == '4'
    assert generated['args'][generated['args'].index('--prefill') + 1] == 'auto'
    assert generated['args'][generated['args'].index('--expert-cache') + 1] == 'auto'
    assert '--mtp' not in generated['args']
    assert generated['args'][generated['args'].index('--expert-profile') + 1] == str(root / 'data' / 'expert-profile.bin')
    assert generated['exe'] == str(root / 'engine/strata.exe')
    assert '--startup-expert-prefetch' not in generated['args']


@pytest.mark.parametrize("mode", ["enabled", "existing-flag", "other-runtime", "missing-exe"])
def test_strata_dev_engine_is_process_scoped(tmp_path, monkeypatch, mode):
    from test_strata_preparation import write_expert_profile
    model, profile, runtime, context = fixtures(tmp_path, "strata")
    model.metadata.update({'qwen4exp.block_count': 48, 'qwen4exp.expert_count': 512})
    root = tmp_path / "Strata"
    (root / "engine").mkdir(parents=True)
    (root / "engine/strata.exe").touch()
    write_expert_profile(root / "data/expert-profile.bin")
    pack, tokenizer = tmp_path / "pack", tmp_path / "tokenizer"
    pack.mkdir()
    tokenizer.mkdir()
    (tokenizer / "vocab.json").write_text("{}")
    runtime.path, runtime.metadata = str(root), {"python": "python.exe", "lib_dirs": [str(tmp_path / "cuda")]}
    profile.parameters = {"pack": str(pack), "tokenizer": str(tokenizer), "context_length": 8192, "kv_cache": "q4_0"}
    if mode == "existing-flag":
        profile.extra_args = ["--startup-expert-prefetch"]
    before = profile.model_dump()
    monkeypatch.delenv("MODLOADER_STRATA_DEV_RUNTIME", raising=False)
    adapter = get_adapters()["strata"]
    baseline_spec = adapter.build_launch(model, profile, runtime, context)
    output = Path(context.work_dir) / "strata.json"
    baseline = json.loads(output.read_text())
    candidate = tmp_path / "strata-dev.exe"
    if mode != "missing-exe":
        candidate.touch()
    target = tmp_path / "other-strata" if mode == "other-runtime" else root / ".." / "Strata"
    monkeypatch.setenv("MODLOADER_STRATA_DEV_RUNTIME", str(target))
    monkeypatch.setenv("MODLOADER_STRATA_DEV_EXE", str(candidate))
    if mode == "missing-exe":
        with pytest.raises(ValueError, match="Strata 开发版引擎不存在"):
            adapter.build_launch(model, profile, runtime, context)
        assert json.loads(output.read_text()) == baseline
    else:
        spec = adapter.build_launch(model, profile, runtime, context)
        generated = json.loads(output.read_text())
        expected = baseline
        if mode != "other-runtime":
            expected["exe"] = str(candidate.resolve())
            if "--startup-expert-prefetch" not in expected["args"]:
                expected["args"].append("--startup-expert-prefetch")
            assert generated["args"].count("--startup-expert-prefetch") == 1
        assert generated == expected
        assert spec == baseline_spec
    assert profile.model_dump() == before


def test_llama_independent_kv_types_reach_launch_and_estimation(tmp_path):
    from modloader.resource_estimates import llama_estimate_args
    model, profile, runtime, context = fixtures(tmp_path)
    profile.parameters = {"cache_type_k": "q8_0", "cache_type_v": "q4_0", "flash_attention": "on"}
    launch = get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)
    estimated, _ = llama_estimate_args(profile, runtime)
    for arguments in [launch.argv, estimated]:
        assert arguments[arguments.index("--cache-type-k") + 1] == "q8_0"
        assert arguments[arguments.index("--cache-type-v") + 1] == "q4_0"


def test_strata_missing_prepared_assets_is_actionable(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path, "strata")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine/strata.exe").touch()
    runtime.path = str(tmp_path)
    with pytest.raises(ValueError, match="tokenizer"):
        get_adapters()["strata"].build_launch(model, profile, runtime, context)


def test_launcher_import_does_not_copy_credentials_or_mcp(tmp_path):
    model = tmp_path / "model with spaces.gguf"
    model.touch()
    script = tmp_path / "start.bat"
    script.write_text('@echo off\nset API_KEY=fixture-secret\nllama-server.exe ^\n --model "' + str(model) + '" ^\n --alias imported --ctx-size 32768 --api-key "%API_KEY%" --mcp-servers-config "tools.json" --temp 0.7\n')
    imported = import_configuration(str(script))
    assert imported["model_path"] == str(model)
    assert imported["parameters"]["context_length"] == 32768
    assert imported["extra_args"] == ["--temp", "0.7"]
    assert "API_KEY" not in json.dumps(imported) and "mcp" not in json.dumps(imported)


def test_llama_imported_parameters_take_priority_over_editor_defaults(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path)
    script = tmp_path / "start.bat"
    script.write_text('llama-server.exe --model "' + model.path + '" --ctx-size 65536 --n-gpu-layers 4 '
                      '--parallel 3 --cache-type-k f16 --cache-type-v f16 --flash-attn off --temp 0.7')
    imported = import_configuration(str(script))
    profile.parameters = imported["parameters"]
    profile.extra_args = imported["extra_args"]
    original = dict(profile.parameters)
    spec = get_adapters()["llama_cpp"].build_launch(model, profile, runtime, context)
    for flag, value in [("--ctx-size", "65536"), ("--n-gpu-layers", "4"), ("--parallel", "3"),
                        ("--cache-type-k", "f16"), ("--cache-type-v", "f16"), ("--flash-attn", "off")]:
        assert spec.argv.count(flag) == 1
        assert spec.argv[spec.argv.index(flag) + 1] == value
    assert spec.argv[-2:] == ["--temp", "0.7"]
    assert profile.parameters == original


def test_transformers_native_serve_pins_directory_offline(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path, "transformers", hf=True)
    adapter = get_adapters()["transformers"]
    assert adapter.inspect(model, runtime).status == "unverified"
    spec = adapter.build_launch(model, profile, runtime, context)
    assert spec.argv[1:3] == ["serve", str(Path(model.path).resolve())]
    assert {"--no-compile", "--no-continuous-batching", "--no-trust-remote-code"}.issubset(spec.argv)
    assert spec.env["HF_HUB_OFFLINE"] == "1"
    assert spec.health_path == "/health" and spec.health_expected == {"status": "ok"}


def test_wsl_mapping_uses_selected_distribution_mounts(monkeypatch):
    from modloader import platforms
    calls = []
    def translated(argv, **kwargs):
        calls.append(argv)
        return '/drives/d/Models/my model/config.json'
    monkeypatch.setattr(platforms, '_run', translated)
    assert windows_to_wsl(r"D:\Models\my model\config.json", 'External') == '/drives/d/Models/my model/config.json'
    assert calls[0] == ['wsl.exe', '--distribution', 'External', '--exec', 'wslpath', '-a', '-u', r"D:\Models\my model\config.json"]
    with pytest.raises(ValueError):
        windows_to_wsl('relative/model')


def test_wsl_external_registration_and_supervisor_syntax(tmp_path, monkeypatch):
    from modloader import platforms
    ast.parse(WSL_SUPERVISOR)
    external = import_runtime('vllm', '/opt/vllm/bin/vllm', {'distribution': 'Ubuntu'})
    assert external.platform == 'wsl2' and external.metadata['distribution'] == 'Ubuntu'
    model, profile, runtime, context = fixtures(tmp_path, 'vllm', hf=True)
    runtime.path = '/opt/modloader/test/bin/vllm'
    runtime.metadata = {'distribution': 'ModLoader'}
    monkeypatch.setattr(platforms, 'windows_to_wsl', lambda path, distribution='ModLoader': '/custom-mount/' + Path(path).name)
    monkeypatch.setattr(platforms, 'require_wsl2', lambda distribution: {'version': 2})
    spec = get_adapters()['vllm'].build_launch(model, profile, runtime, context)
    assert spec.platform == 'wsl2' and spec.distribution == 'ModLoader'
    assert spec.argv[0] == runtime.path and not spec.cleanup_argv
    assert spec.health_path == '/health' and spec.health_expected == {}
    wrapped = platforms.prepare_launch(spec, context)
    assert wrapped.argv[:5] == ['wsl.exe', '--distribution', 'ModLoader', '--exec', '/usr/bin/python3']
    assert wrapped.cleanup_argv[-1] == context.instance_id and 'stop' in wrapped.cleanup_argv
    assert wrapped.env['WSL_UTF8'] == '1'
    assert not any(a in ('--shutdown', '--terminate') for a in wrapped.argv + wrapped.cleanup_argv)


@pytest.mark.parametrize("backend", ["transformers", "vllm"])
def test_hf_image_metadata_is_exposed_without_claiming_verified_support(tmp_path, backend, monkeypatch):
    from modloader import platforms
    monkeypatch.setattr(platforms, 'windows_to_wsl', lambda path, distribution='ModLoader': '/models/hf')
    model, profile, runtime, context = fixtures(tmp_path, backend, hf=True)
    model.modalities = ["text", "image"]
    adapter = get_adapters()[backend]
    result = adapter.inspect(model, runtime)
    assert result.status == "unverified"
    assert "image" in result.capabilities["input"]
    if backend == "vllm":
        if sys.platform != "win32":
            return
        runtime.path = "/opt/modloader/test/bin/vllm"
        runtime.metadata = {"distribution": "ModLoader"}
    spec = adapter.build_launch(model, profile, runtime, context)
    assert "image" in spec.capabilities["input"]


def test_archive_path_escape_is_rejected(tmp_path):
    zpath = tmp_path / "bad.zip"
    with zipfile.ZipFile(zpath, "w") as archive:
        archive.writestr("../escaped.txt", "bad")
    target = tmp_path / "extract"
    target.mkdir()
    with pytest.raises(ValueError):
        _extract(zpath, target)
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize('kind', ['traversal', 'symlink', 'hardlink', 'device'])
def test_engine_tar_rejects_unsafe_members_before_writing(tmp_path, kind):
    import io
    import tarfile
    archive = tmp_path / 'engine.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        good = tarfile.TarInfo('strata.exe')
        good.size = 6
        output.addfile(good, io.BytesIO(b'engine'))
        bad = tarfile.TarInfo('../escaped.txt' if kind == 'traversal' else 'unsafe')
        if kind != 'traversal':
            bad.type = {'symlink': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE, 'device': tarfile.CHRTYPE}[kind]
            bad.linkname = '../escaped.txt'
        output.addfile(bad)
    target = tmp_path / 'extract'
    with pytest.raises(ValueError, match='目录穿越|特殊文件'):
        _extract(archive, target)
    assert not target.exists()
    assert not (tmp_path / 'escaped.txt').exists()


def test_cancelled_install_does_not_start_or_download(tmp_path):
    cancel = asyncio.Event()
    cancel.set()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(install_backend("llama_cpp", AppPaths(tmp_path), {}, None, cancel))
    assert not list((tmp_path / "runtimes").iterdir())


@pytest.mark.parametrize('format', ['zip', 'tar.gz'])
async def test_extract_cancellation_waits_until_archive_writer_stops(tmp_path, monkeypatch, format):
    import io
    import tarfile
    import threading
    import time
    from modloader.backends.installers import _unpack

    archive, destination = tmp_path / ('package.' + format), tmp_path / 'unpacked'
    content = b'x' * (8 * 1024 * 1024)
    if format == 'zip':
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('large.bin', content)
        reader = zipfile.ZipExtFile
    else:
        with tarfile.open(archive, 'w:gz') as output:
            member = tarfile.TarInfo('large.bin')
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
        reader = tarfile.ExFileObject
    started = threading.Event()
    original = reader.read
    def slow_read(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        started.set()
        time.sleep(.02)
        return result
    monkeypatch.setattr(reader, 'read', slow_read)
    cancel, events = asyncio.Event(), []
    async def progress(event):
        events.append(event)
    task = asyncio.create_task(_unpack(archive, destination, progress, cancel))
    try:
        async with asyncio.timeout(3):
            while not started.is_set():
                await asyncio.sleep(.005)
        cancel.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        target = destination / 'large.bin'
        size = target.stat().st_size
        await asyncio.sleep(.06)
        assert target.stat().st_size == size < 8 * 1024 * 1024
        assert events == [{'phase': 'extract', 'message': f'解压 {archive.name}', 'can_pause': False, 'percent': None}]
    finally:
        cancel.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_install_command_stop_terminates_actual_process_writer(tmp_path):
    from modloader.backends.installers import _command

    marker = tmp_path / 'installer-output'
    script = "import sys,time; f=open(sys.argv[1],'wb');\nfor n in range(1000):\n f.write(b'x'*1024); f.flush(); print(n,flush=True); time.sleep(.02)"
    cancel = asyncio.Event()
    task = asyncio.create_task(_command([sys.executable, '-u', '-c', script, str(marker)], None, cancel))
    try:
        async with asyncio.timeout(3):
            while not marker.exists() or not marker.stat().st_size:
                await asyncio.sleep(.01)
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        size = marker.stat().st_size
        await asyncio.sleep(.08)
        assert marker.stat().st_size == size
    finally:
        cancel.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_uninstall_refuses_external_and_escaped_paths(tmp_path):
    paths = AppPaths(tmp_path / "data")
    original = tmp_path / "original-runtime"
    original.mkdir()
    sentinel = original / "keep.txt"
    sentinel.write_text("keep")
    runtime = RuntimeInstallation(id="external", backend="llama_cpp", name="External", path=str(original / "llama-server.exe"))
    with pytest.raises(ValueError, match="外部"):
        asyncio.run(uninstall_runtime(runtime, paths))
    runtime.managed = True
    runtime.metadata["managed_root"] = str(original)
    with pytest.raises(ValueError, match="之外"):
        asyncio.run(uninstall_runtime(runtime, paths))
    assert sentinel.read_text() == "keep"


def test_frozen_installer_uses_private_python_not_application_executable(tmp_path, monkeypatch):
    import modloader.backends.installers as installers
    calls = []
    async def download(url, target, progress, cancel, digest=None):
        calls.append(("download", url, digest))
    def extract(archive, destination, cancel=None):
        (destination / "tools").mkdir()
        (destination / "tools/python.exe").touch()
    async def command(argv, *args, **kwargs):
        calls.append(("command", argv))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(installers, "_download", download)
    monkeypatch.setattr(installers, "_extract", extract)
    monkeypatch.setattr(installers, "_command", command)
    result = asyncio.run(installers._venv(tmp_path / "venv", {}, None, asyncio.Event()))
    command_args = next(c[1] for c in calls if c[0] == "command")
    assert command_args[0] == str(tmp_path / "python-3.12.10/tools/python.exe")
    assert command_args[1:3] == ["-m", "venv"]
    assert result.endswith("python.exe") if sys.platform == "win32" else result.endswith("python")


def test_llama_metrics_flag_is_version_gated_and_not_duplicated(tmp_path):
    model, profile, runtime, context = fixtures(tmp_path)
    adapter = get_adapters()["llama_cpp"]
    assert "--metrics" not in adapter.build_launch(model, profile, runtime, context).argv
    runtime.metadata["supported_flags"] = ["--metrics"]
    assert adapter.build_launch(model, profile, runtime, context).argv.count("--metrics") == 1
    profile.extra_args = ["--metrics"]
    spec = adapter.build_launch(model, profile, runtime, context)
    assert spec.argv.count("--metrics") == 1
    assert spec.env["LLAMA_SERVER_SLOTS_DEBUG"] == "0"
    profile.extra_args = ["--log-file", "chat.log"]
    with pytest.raises(ValueError):
        adapter.build_launch(model, profile, runtime, context)


def telemetry_fixture(backend, monkeypatch, responses, spec_env=None, spec_caps=None):
    import httpx
    import modloader.backends.telemetry as telemetry
    calls = []
    def handler(request):
        calls.append(request.url.path)
        status, data = responses[request.url.path]
        return httpx.Response(status, json=data) if not isinstance(data, str) else httpx.Response(status, text=data)
    client_type = httpx.AsyncClient
    monkeypatch.setattr(telemetry.httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs))
    instance = RuntimeInstance(id="fixture", model_id="m", profile_id="p", profile_revision=1, alias="model", backend=backend,
                               runtime_id="r", port=18999, started_at="2026-10-14", state="ready")
    expected = {} if backend == "vllm" else {"status": "ok"}
    if backend == "strata":
        expected["loaded"] = True
    spec = LaunchSpec(argv=["engine"], cwd=".", model_name="local-model", health_path="/health", health_expected=expected, env=spec_env or {}, capabilities=spec_caps or {})
    result = asyncio.run(collect_telemetry(instance, spec))
    return result, calls


def telemetry_values(result):
    return {m["id"]: m for group in result["groups"] for m in group["metrics"]}


def test_strata_telemetry_preserves_units_nulls_and_excludes_content(monkeypatch):
    response = {"engine": {"max_context": 32768, "expert_slots": 8000, "expert_cache_mib": 22000, "arena_mib": 55000,
                            "kv": "int8", "kv_resident": 4096, "model": "do-not-copy-model-path"},
                "live": {"state": "idle", "queued": 0, "tok_s": None, "slots": [{"slot": 0, "state": "idle", "held_tokens": 4096}]},
                "totals": {"requests": 3, "output_tokens": 123},
                "conversation_cache": {"enabled": True, "budget_mib": 1024, "parked": 0, "restores": 0, "last_reused": 5},
                "requests": [{"hit_rate": 0.8, "pcie_share": 0.1, "ram_blobs": 32, "file_mb": 4.5, "decode_tok_s": 100,
                              "input": "fixture-chat-secret", "output": "fixture-response-secret"}],
                "hardware": {"gpu_mem_used": 999, "cpu_percent": 50}, "api_key": "fixture-key"}
    result, calls = telemetry_fixture("strata", monkeypatch, {"/health": (200, {"status": "ok", "loaded": True}), "/metrics": (200, response)})
    values = telemetry_values(result)
    assert values["expert_cache_mib"]["value"] == 22000 and values["expert_cache_mib"]["unit"] == "MiB"
    assert values["arena_mib"]["value"] == 55000
    assert values["tok_s"]["value"] is None
    assert values["queued"]["value"] == 0
    assert values["output_tokens"]["kind"] == "counter"
    assert values["hit_rate"]["value"] == 0.8
    assert values["held_tokens"]["value"] == 4096
    assert not {"parked", "restores", "gpu_mem_used", "cpu_percent"}.intersection(values)
    assert "fixture-chat" not in json.dumps(result) and "fixture-key" not in json.dumps(result)
    assert "do-not-copy-model" not in json.dumps(result)
    assert set(calls) == {"/health", "/metrics"}
    assert result["backend_status"] == "idle"


def test_vllm_prometheus_histogram_mean_counters_and_labels(monkeypatch):
    prometheus = '''# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="secret-model-label",engine="0"} 2
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.25
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{engine="0"} 120
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="1",engine="0"} 2
vllm:time_to_first_token_seconds_bucket{le="+Inf",engine="0"} 4
vllm:time_to_first_token_seconds_count{engine="0"} 4
vllm:time_to_first_token_seconds_sum{engine="0"} 6
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 100000
'''
    result, calls = telemetry_fixture("vllm", monkeypatch, {"/health": (200, {}), "/metrics": (200, prometheus)})
    metrics = [m for g in result["groups"] for m in g["metrics"]]
    assert next(m for m in metrics if m["id"].startswith("generation_tokens_total"))["kind"] == "counter"
    assert next(m for m in metrics if m["id"].endswith("_mean"))["value"] == 1.5
    assert not any("bucket" in m["id"] or "process_" in m["id"] for m in metrics)
    assert "secret-model-label" not in json.dumps(result)
    assert result["backend_status"] == "ready"
    assert set(calls) == {"/health", "/metrics"}


def test_llama_telemetry_does_not_fetch_slots_without_debug_disabled(monkeypatch):
    result, calls = telemetry_fixture("llama_cpp", monkeypatch, {"/health": (200, {"status": "ok"}), "/metrics": (404, {"error": "fixture-sensitive-error"})})
    assert "/slots" not in calls
    assert any("slots" in item for item in result["unavailable"])
    assert "fixture-sensitive" not in json.dumps(result)


def test_llama_slots_and_scrape_window_rates_are_distinct(monkeypatch):
    text = '# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total 40\n# TYPE llamacpp:predicted_tokens_seconds gauge\nllamacpp:predicted_tokens_seconds 10\n'
    slot = {"id": 0, "n_ctx": 8192, "n_prompt_tokens": 100, "n_prompt_tokens_cache": 90, "is_processing": False,
            "prompt": "fixture-private-content", "params": {"stop": ["fixture-private-stop"]}, "next_token": [{"n_decoded": 5}]}
    result, calls = telemetry_fixture("llama_cpp", monkeypatch, {"/health": (200, {"status": "ok"}), "/metrics": (200, text), "/slots": (200, [slot])}, {"LLAMA_SERVER_SLOTS_DEBUG": "0"})
    values = telemetry_values(result)
    assert values["tokens_predicted_total"]["kind"] == "counter"
    assert values["predicted_tokens_seconds"]["kind"] == "gauge"
    assert "采集" in values["predicted_tokens_seconds"]["description"]
    assert values["n_ctx"]["value"] == 8192 and values["n_decoded"]["value"] == 5
    assert "fixture-private" not in json.dumps(result)


def test_transformers_has_only_supported_service_telemetry(monkeypatch):
    result, calls = telemetry_fixture("transformers", monkeypatch, {"/health": (200, {"status": "ok"})})
    assert calls == ["/health"]
    assert result["backend_status"] == "ready"
    assert any("未提供请求、KV、专家或吞吐指标端点" in item for item in result["unavailable"])
    assert not any(m["id"] in {"kv_cache_usage", "tokens_per_second"} for g in result["groups"] for m in g["metrics"])


@pytest.mark.parametrize("executable", ["python.exe", "wsl.exe"])
def test_install_command_scopes_cache_and_temp_to_owned_root(tmp_path, monkeypatch, executable):
    import os
    import modloader.backends.installers as installers
    captured = {}
    original_temp = os.environ.get("TEMP")
    monkeypatch.setenv("WSL_UTF8", "0")
    class Output:
        async def readline(self):
            return b""
    class Process:
        returncode = 0
        stdout = Output()
        async def wait(self):
            return 0
    async def spawn(*argv, **kwargs):
        captured.update(kwargs)
        return Process()
    monkeypatch.setattr(installers.asyncio, "create_subprocess_exec", spawn)
    asyncio.run(installers._command([executable, "--version"], None, asyncio.Event(), owned_root=tmp_path))
    assert captured["env"]["PIP_CACHE_DIR"] == str(tmp_path / "cache/pip")
    assert all(captured["env"][key] == str(tmp_path / "tmp") for key in ("TEMP", "TMP", "TMPDIR"))
    assert captured["cwd"] == str(tmp_path)
    assert os.environ.get("TEMP") == original_temp
    assert captured["env"]["WSL_UTF8"] == ("1" if executable == "wsl.exe" else "0")
    assert os.environ["WSL_UTF8"] == "0"


def test_vllm_no_samples_does_not_invent_zero_latency(monkeypatch):
    text = '# TYPE vllm:request_decode_time_seconds histogram\nvllm:request_decode_time_seconds_sum 0\nvllm:request_decode_time_seconds_count 0\n'
    result, _ = telemetry_fixture("vllm", monkeypatch, {"/health": (200, {}), "/metrics": (200, text)})
    values = telemetry_values(result)
    assert values["request_decode_time_seconds_count"]["value"] == 0
    assert values["request_decode_time_seconds_mean"]["value"] is None


def test_telemetry_endpoint_failure_reports_unavailable_without_error_body(monkeypatch):
    result, _ = telemetry_fixture("strata", monkeypatch, {"/health": (503, {"error": "fixture-private-data"}), "/metrics": (404, {"error": "fixture-private-data"})})
    assert result["groups"] == []
    assert result["backend_status"] == "unreachable"
    assert len(result["unavailable"]) == 2
    assert "fixture-private-data" not in json.dumps(result)


def test_strata_live_window_and_request_mean_remain_distinct_model_rates(monkeypatch):
    response = {'live':{'state':'generating','queued':0,'generated':250,'tok_s':37.5,'tok_s_mean':12.5,'tok_s_window_s':2,
                        'request_id':'private-unsupported-id','tail':'private-output'}}
    result, _ = telemetry_fixture('strata',monkeypatch,{'/health':(200,{'status':'ok','loaded':True}),'/metrics':(200,response)})
    values = telemetry_values(result)
    assert values['tok_s']['value']==37.5 and values['tok_s_mean']['value']==12.5
    assert '0.25s' in values['tok_s']['description'] and '首 token' in values['tok_s_mean']['description']
    live = result['live_generation']
    assert live['scope']=='model' and live['tokens_per_second'] is None and live['window_s'] is None
    assert live['_generated_tokens']==250 and live['source']=='/metrics live.generated'
    assert live['backend_running']==1 and live['backend_queued']==0
    assert not live['available_for_chat']  # The gateway, not the adapter, checks ownership.
    assert 'private-' not in json.dumps(result)


@pytest.mark.parametrize('state,running,count', [('generating',2,50),('reading',1,50),('idle',0,50),('generating',None,'invalid')])
def test_strata_model_live_rate_preserves_concurrency_and_invalid_sources(monkeypatch,state,running,count):
    response = {'live':{'state':state,'parallel':4,'queued':1,'running':running,'generated':count,'tok_s':50,'tok_s_window_s':2}}
    result, _ = telemetry_fixture('strata',monkeypatch,{'/health':(200,{'status':'ok','loaded':True}),'/metrics':(200,response)})
    live = result['live_generation']
    assert live['backend_running']==running and live['backend_queued']==1
    assert live['tokens_per_second'] is None
    assert live['_generated_tokens']==(50 if state=='generating' and running==2 else None)
    assert not live['available_for_chat']


def test_llama_live_generation_uses_slot_counter_when_scrape_gauge_is_zero(monkeypatch):
    text = ('# TYPE llamacpp:requests_processing gauge\nllamacpp:requests_processing 1\n'
            '# TYPE llamacpp:requests_deferred gauge\nllamacpp:requests_deferred 0\n'
            '# TYPE llamacpp:predicted_tokens_seconds gauge\nllamacpp:predicted_tokens_seconds 0\n')
    slots = [{'id':0,'is_processing':True,'next_token':[{'n_decoded':154}],'prompt':'private-conversation'}]
    result, _ = telemetry_fixture('llama_cpp',monkeypatch,{'/health':(200,{'status':'ok'}),'/metrics':(200,text),'/slots':(200,slots)}, {'LLAMA_SERVER_SLOTS_DEBUG':'0'})
    live = result['live_generation']
    assert live['tokens_per_second'] is None and live['backend_running']==1 and live['backend_queued']==0
    assert live['_generated_tokens']==154 and live['_slot_id']==0
    assert live['scope']=='model' and live['window_s'] is None
    assert 'n_decoded' in live['source'] and 'private-conversation' not in json.dumps(result)


def test_multiple_labeled_llama_engines_do_not_get_an_arbitrary_chat_rate(monkeypatch):
    text = ('# TYPE llamacpp:requests_processing gauge\nllamacpp:requests_processing{engine="0"} 1\nllamacpp:requests_processing{engine="1"} 1\n'
            '# TYPE llamacpp:predicted_tokens_seconds gauge\nllamacpp:predicted_tokens_seconds{engine="0"} 20\nllamacpp:predicted_tokens_seconds{engine="1"} 30\n')
    result, _ = telemetry_fixture('llama_cpp',monkeypatch,{'/health':(200,{'status':'ok'}),'/metrics':(200,text)})
    assert result['live_generation']['tokens_per_second'] is None
    assert result['live_generation']['backend_running'] is None


def test_strata_current_overview_uses_native_mean_and_logical_context(monkeypatch):
    from modloader.request_metrics import generation_window
    response = {'engine': {'max_context': 4096},
                'live': {'state': 'generating', 'queued': 0, 'running': 1, 'prompt_tokens': 128,
                         'generated': 16, 'tok_s': 90, 'tok_s_mean': 8.5, 'elapsed_s': 4}}
    result, _ = telemetry_fixture('strata', monkeypatch, {'/health': (200, {'loaded': True}), '/metrics': (200, response)})
    _, live = generation_window(None, 'single-request', result['live_generation'], 10)
    assert live['average_tokens_per_second'] == 8.5  # Neither tok_s nor generated/elapsed.
    assert live['context_tokens'] == 144 and live['context_capacity'] == 4096
    assert live['prompt_tokens'] == 128 and live['generated_tokens'] == 16
    assert live['tokens_per_second'] is None


def test_llama_context_is_held_tokens_without_adding_decoded_twice(monkeypatch):
    from modloader.request_metrics import generation_window
    text = 'llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\n'
    slots = [{'id': 0, 'is_processing': True, 'n_prompt_tokens': 144, 'n_ctx': 4096,
              'next_token': [{'n_decoded': 16}]}]
    result, _ = telemetry_fixture('llama_cpp', monkeypatch, {'/health': (200, {'status': 'ok'}), '/metrics': (200, text), '/slots': (200, slots)}, {'LLAMA_SERVER_SLOTS_DEBUG': '0'})
    _, live = generation_window(None, 'single-request', result['live_generation'], 10)
    assert live['context_tokens'] == 144 and live['context_capacity'] == 4096
    assert live['average_tokens_per_second'] is None


def test_ninfer_idle_capacity_and_request_counts_are_not_cumulative_token_usage(monkeypatch):
    from modloader.request_metrics import generation_window
    text = ('ninfer_requests_running 0\nninfer_requests_waiting 0\nninfer_requests_paused 0\n'
            'ninfer_max_context_tokens 128000\nninfer_generation_tokens_total 8372\n'
            'ninfer_prompt_tokens_total 4098\n')
    result, calls = telemetry_fixture('ninfer', monkeypatch, {'/health': (200, {'status': 'ok'}), '/metrics': (200, text)})
    _, live = generation_window(None, None, result['live_generation'], 10)
    assert live['backend_running'] == live['backend_queued'] == 0
    assert live['backend_paused'] == 0
    assert live['context_capacity'] == 128000
    assert live['context_tokens'] is None and live['generated_tokens'] is None
    assert live['tokens_per_second'] is None and not live['available_for_chat']
    assert set(calls) == {'/health', '/metrics'}


def test_spark_preserves_official_context_capacity_but_has_no_live_token_counter(monkeypatch):
    from modloader.request_metrics import generation_window
    text = 'sparkinfer_active_requests 1\nsparkinfer_waiting_requests 0\nsparkinfer_tokens_total{kind="completion"} 9999\n'
    result, calls = telemetry_fixture('sparkinfer', monkeypatch,
                                     {'/health': (200, {'status': 'ok'}), '/metrics': (200, text),
                                      '/v1/info': (200, {'model': 'private-model-path', 'max_context': 8192, 'max_output_tokens': 4096})})
    _, live = generation_window(None, 'one-request', result['live_generation'], 10)
    assert live['context_capacity'] == 8192 and live['context_tokens'] is None
    assert live['tokens_per_second'] is None and live['generated_tokens'] is None
    assert not live['available_for_chat']
    assert 'private-model-path' not in json.dumps(result)
    assert any('仅在请求结束结算' in reason for reason in result['unavailable'])
    assert set(calls) == {'/health', '/metrics', '/v1/info'}


@pytest.mark.parametrize('cache_enabled', [False, True])
def test_sparkinfer_conditional_prefix_cache_keeps_native_dimensions_and_no_request_rates(monkeypatch, cache_enabled):
    from modloader.request_metrics import RequestMetrics, generation_window
    text = ('sparkinfer_active_requests 1\nsparkinfer_waiting_requests 0\nsparkinfer_free_kv_blocks 7\n'
            'sparkinfer_tokens_total{kind="prompt"} 4096\nsparkinfer_tokens_total{kind="completion"} 512\n')
    # SparkInfer 0.6.34 /metrics emits this block only when prefix_cache_stats.enabled.
    cache = {
        'prefix_cache_lookups_total': (12, '次', 'counter'),
        'prefix_cache_hits_total': (9, '次', 'counter'),
        'prefix_cache_tokens_reused_total': (8192, 'token', 'counter'),
        'prefix_cache_evictions_total': (2, '个', 'counter'),
        'prefix_cache_entries': (4, '个', 'gauge'),
        'prefix_cache_kv_blocks': (32, '块', 'gauge'),
        'prefix_cache_host_bytes': (1048576, 'bytes', 'gauge'),
        'prefix_cache_host_hits_total': (0, '次', 'counter'),
        'prefix_cache_demotions_total': (3, '个', 'counter'),
        'prefix_cache_host_entries': (0, '个', 'gauge'),
        'prefix_cache_host_kv_bytes': (2097152, 'bytes', 'gauge'),
    }
    if cache_enabled:
        text += ''.join(f'sparkinfer_{field} {value}\n' for field, (value, _, _) in cache.items())
    result, calls = telemetry_fixture('sparkinfer', monkeypatch, {
        '/health': (200, {'status': 'ok'}), '/metrics': (200, text), '/v1/info': (200, {'max_context': 8192})})
    values = telemetry_values(result)
    assert values['free_kv_blocks']['value'] == 7 and values['free_kv_blocks']['unit'] == '块'
    if cache_enabled:
        group = next(group for group in result['groups'] if group['id'] == 'sparkinfer_prefix_cache')
        assert {metric['id'] for metric in group['metrics']} == set(cache)
        for field, (value, unit, kind) in cache.items():
            assert values[field]['value'] == value and values[field]['unit'] == unit and values[field]['kind'] == kind
            assert f'/metrics · sparkinfer_{field}' in values[field]['description']
        assert '快照' in values['prefix_cache_host_bytes']['label']
        assert 'KV' in values['prefix_cache_host_kv_bytes']['label']
    else:
        assert not (set(cache) & set(values))
        assert not any(group['id'] == 'sparkinfer_prefix_cache' for group in result['groups'])
    totals = [metric for metric in values.values() if metric['id'].startswith('tokens_total_')]
    assert len(totals) == 2 and {metric['value'] for metric in totals} == {4096, 512}
    assert any('kind=prompt' in metric['label'] for metric in totals)
    assert any('kind=completion' in metric['label'] for metric in totals)
    assert not any(metric.get('unit') == 'token/s' or 'accept_rate' in metric['id'] for metric in values.values())
    request = RequestMetrics(True, backend='sparkinfer')
    request.observe({'choices': [{'delta': {'content': 'many words in one SSE chunk'}}]})
    _, live = generation_window(None, request.token, request.live_generation(result['live_generation']), 10)
    assert live['generated_tokens'] is None and live['tokens_per_second'] is None
    assert live['average_tokens_per_second'] is None and live['context_capacity'] == 8192
    assert set(calls) == {'/health', '/metrics', '/v1/info'}


def test_vllm_waiting_reason_dimensions_do_not_hide_the_actual_waiting_gauge(monkeypatch):
    text = ('vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'
            'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 0\n')
    result, _ = telemetry_fixture('vllm', monkeypatch, {'/health': (200, {}), '/metrics': (200, text)})
    assert result['live_generation']['backend_running'] == result['live_generation']['backend_queued'] == 0


@pytest.mark.parametrize('backend,flag', [('vllm', '--max-model-len'), ('sglang', '--context-length')])
def test_configured_context_limit_survives_final_argv_and_wsl_wrapper(tmp_path, monkeypatch, backend, flag):
    import modloader.backends.adapters as adapters
    import modloader.backends.sglang as sglang
    import modloader.platforms as platforms
    from modloader.request_metrics import generation_window
    model, profile, runtime, context = fixtures(tmp_path, backend, hf=True)
    runtime.platform, runtime.path = 'wsl2', '/opt/engine/bin/python'
    runtime.metadata['distribution'] = 'External'
    profile.parameters['context_length'] = 2048
    profile.extra_args = [flag+'=4096']
    monkeypatch.setattr(adapters, 'model_target_path', lambda *_: '/models/local-hf')
    monkeypatch.setattr(sglang, 'model_target_path', lambda *_: '/models/local-hf')
    spec = get_adapters()[backend].build_launch(model, profile, runtime, context)
    assert spec.capabilities['configured_context_tokens'] == 4096
    assert spec.capabilities['configured_context_flag'] == flag
    assert profile.parameters['context_length'] == 2048 and profile.extra_args == [flag+'=4096']
    monkeypatch.setattr(platforms, 'require_wsl2', lambda *_: None)
    monkeypatch.setattr(platforms, 'windows_to_wsl', lambda *_: '/tmp/instance-record.json')
    wrapped = platforms.prepare_launch(spec, context)
    assert wrapped.capabilities == spec.capabilities
    text = 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n' if backend == 'vllm' else 'sglang:num_running_reqs 0\nsglang:num_queue_reqs 0\n'
    result, _ = telemetry_fixture(backend, monkeypatch, {'/health': (200, {}), '/metrics': (200, text)}, spec_caps=wrapped.capabilities)
    _, live = generation_window(None, None, result['live_generation'], 10)
    assert live['context_capacity'] == 4096 and live['context_capacity_basis'] == '启动配置上限'
    assert live['context_tokens'] is None
    metric = next(metric for group in result['groups'] for metric in group['metrics'] if metric['id'] == 'configured_context_tokens')
    assert flag in metric['description'] and '不是实时 KV 池容量' in metric['description']


@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
def test_context_limit_is_not_inferred_when_launch_did_not_set_one(monkeypatch, backend):
    text = 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n' if backend == 'vllm' else 'sglang:num_running_reqs 0\nsglang:num_queue_reqs 0\n'
    result, _ = telemetry_fixture(backend, monkeypatch, {'/health': (200, {}), '/metrics': (200, text)})
    assert result['live_generation']['_overview']['context_capacity'] is None
    assert not any(metric['id'] == 'configured_context_tokens' for group in result['groups'] for metric in group['metrics'])


def test_strata_prefill_position_does_not_count_cached_tokens_as_processed(monkeypatch):
    from modloader.request_metrics import generation_window
    response = {'engine': {'max_context': 4096}, 'live': {'state': 'reading', 'phase': 'reading the prompt',
                'queued': 0, 'running': 1, 'prompt_tokens': 1024, 'prompt_read': 640, 'prompt_total': 1024,
                'generated': 0, 'elapsed_s': 2, 'prefill_tok_s_mean': 128}}
    def collect():
        with monkeypatch.context() as patch:
            return telemetry_fixture('strata', patch, {'/health': (200, {'loaded': True}),
                '/metrics': (200, response)})[0]['live_generation']
    state, first = generation_window(None, 'request-one', collect(), 10)
    assert first['request_key'] == 'request-one' and first['phase'] == 'prefill'
    assert first['prefill_processed_tokens'] is None and first['prefill_cached_tokens'] is None
    assert first['prefill_total_tokens'] == 1024
    assert first['prefill_tokens_per_second'] is None and first['average_prefill_tokens_per_second'] == 128
    assert first['prefill_average_basis'] == 'backend_prefill' and first['tokens_per_second'] is None
    response['live'].update(prompt_read=896, elapsed_s=3, prefill_tok_s_mean=150)
    state, sampled = generation_window(state, 'request-one', collect(), 11)
    assert sampled['prefill_processed_tokens'] is None and sampled['prefill_cached_tokens'] is None
    assert sampled['prefill_tokens_per_second'] is None and sampled['prefill_window_s'] is None
    assert sampled['average_prefill_tokens_per_second'] == 150  # Native mean is not the current window.
    assert sampled['tokens_per_second'] is None and sampled['average_tokens_per_second'] is None
    response['live'].update(state='generating', phase='thinking', prompt_read=None, prompt_total=None,
                            generated=5, tok_s_mean=2.5)
    _, decode = generation_window(state, 'request-one', collect(), 12)
    assert decode['phase'] == 'generating'  # Thinking/output require the actual SSE delta.
    assert decode['prefill_tokens_per_second'] is None and decode['prefill_window_s'] is None
    assert decode['average_prefill_tokens_per_second'] == 150 and decode['average_tokens_per_second'] == 2.5


def test_strata_reading_without_explicit_progress_is_not_assumed_prefill(monkeypatch):
    from modloader.request_metrics import generation_window
    response = {'live': {'state': 'reading', 'queued': 0, 'prompt_tokens': 1024, 'generated': 0,
                         'prompt_read': None, 'prompt_total': None}}
    result, _ = telemetry_fixture('strata', monkeypatch, {'/health': (200, {'loaded': True}), '/metrics': (200, response)})
    _, view = generation_window(None, 'one', result['live_generation'], 10)
    assert view['phase'] == 'waiting' and view['request_key'] == 'one'
    assert view['prefill_tokens_per_second'] is None and view['prefill_processed_tokens'] is None


def test_llama_prefill_counter_excludes_cache_and_held_context_is_not_a_target(monkeypatch):
    from modloader.request_metrics import RequestMetrics, generation_window
    text = 'llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\nllamacpp:prompt_tokens_seconds 999\n'
    slots = [{'id': 0, 'is_processing': True, 'n_prompt_tokens': 900, 'n_ctx': 4096,
              'n_prompt_tokens_processed': 0, 'n_prompt_tokens_cache': 900, 'next_token': [{'n_decoded': 0}]}]
    def collect():
        with monkeypatch.context() as patch:
            return telemetry_fixture('llama_cpp', patch, {'/health': (200, {'status': 'ok'}),
                '/metrics': (200, text), '/slots': (200, slots)}, {'LLAMA_SERVER_SLOTS_DEBUG': '0'})[0]['live_generation']
    state, first = generation_window(None, 'one', collect(), 10)
    assert first['phase'] == 'prefill' and first['prefill_processed_tokens'] == 0
    assert first['prefill_cached_tokens'] == 900 and first['prefill_total_tokens'] is None
    assert first['average_prefill_tokens_per_second'] is None
    slots[0].update(n_prompt_tokens=924, n_prompt_tokens_processed=24)
    _, second = generation_window(state, 'one', collect(), 11)
    assert second['prefill_tokens_per_second'] == 24 and second['prefill_average_basis'] == 'sampled_prefill'
    assert second['context_tokens'] == 924 and second['prefill_total_tokens'] is None
    assert second['tokens_per_second'] is None and second['average_tokens_per_second'] is None

    # SSE can advance while /slots is still returning the previous snapshot.
    request = RequestMetrics(True, backend='llama_cpp')
    slots[0]['next_token'][0]['n_decoded'] = 4
    request.observe({'choices': [{'delta': {}}], 'prompt_progress': {'total': 1024, 'processed': 950, 'cache': 900}})
    state, first = generation_window(None, request.token, request.live_generation(collect()), 12)
    assert first['phase'] == 'prefill'
    assert first['prefill_processed_tokens'] == 50 and first['prefill_total_tokens'] == 1024
    assert first['prefill_cached_tokens'] == 900 and first['generated_tokens'] == 0
    request.observe({'choices': [{'delta': {}}], 'prompt_progress': {'total': 1024, 'processed': 974, 'cache': 900}})
    live = request.live_generation(collect())
    _, progressed = generation_window(state, request.token, live, 13)
    assert live['_prefill_source'] == 'SSE prompt_progress.processed - cache'
    assert progressed['prefill_processed_tokens'] == 74 and progressed['prefill_tokens_per_second'] == 24
    assert request.snapshot()['prefill']['prefill_processed_tokens'] == 74

    # With timings explicitly disabled, existing /slots decode counters remain
    # usable once the SSE delta has established that generation started.
    request.observe({'choices': [{'delta': {'content': 'one whole SSE chunk'}}]})
    slots[0]['next_token'][0]['n_decoded'] = 7
    live = request.live_generation(collect())
    _, output = generation_window(None, request.token, live, 14)
    assert live['source'] == '/slots next_token.n_decoded' and output['generated_tokens'] == 7
    assert output['phase'] == 'output' and output['prefill_processed_tokens'] == 74

    request.observe({'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': 1024, 'completion_tokens': 10},
                     'timings': {'prompt_n': 124, 'cache_n': 900, 'prompt_ms': 2000,
                                 'prompt_per_second': 62, 'predicted_n': 10, 'predicted_per_second': 5}})
    # A late progress frame and another old poll cannot undo native timings.
    request.observe({'prompt_progress': {'total': 1024, 'processed': 974, 'cache': 900}})
    live = request.live_generation(collect())
    _, completed = generation_window(None, request.token, live, 15)
    assert live['_prefill_source'] == 'SSE timings.prompt_n'
    assert completed['phase'] == 'output' and completed['prefill_processed_tokens'] == 124
    assert completed['average_prefill_tokens_per_second'] == 62 and completed['prefill_average_basis'] == 'backend_prefill'
    assert completed['generated_tokens'] == 10 and completed['average_tokens_per_second'] == 5
    assert request.finish('completed')['prefill']['prefill_processed_tokens'] == 124


@pytest.mark.parametrize('backend,text', [
    ('vllm', 'vllm:num_requests_running 1\nvllm:num_requests_waiting 0\nvllm:avg_prompt_throughput_toks_per_s 9000\nvllm:prompt_tokens_total 500000\n'),
    ('sglang', 'sglang:num_running_reqs 1\nsglang:num_queue_reqs 0\nsglang:prompt_tokens_total 500000\n'),
])
def test_engine_prefill_totals_and_throughput_are_not_current_request_prefill(monkeypatch, backend, text):
    from modloader.request_metrics import RequestMetrics, generation_window
    result, _ = telemetry_fixture(backend, monkeypatch, {'/health': (200, {}), '/metrics': (200, text)})
    request = RequestMetrics(True, backend=backend)
    _, view = generation_window(None, request.token, request.live_generation(result['live_generation']), 10)
    assert view['request_key'] == request.token and view['phase'] == 'waiting'
    assert view['prefill_tokens_per_second'] is None and view['average_prefill_tokens_per_second'] is None
    assert view['prefill_processed_tokens'] is None and view['prefill_total_tokens'] is None
