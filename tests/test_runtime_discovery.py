import asyncio
import json
import ntpath
import os
from pathlib import Path
import subprocess
import threading

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.models import AppPaths, RuntimeInstallation
from modloader import runtime_discovery as discovery


def file(path: Path, content="fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def transformers(root: Path, version="5.19.0"):
    file(root / "Scripts/python.exe")
    executable = file(root / "Scripts/transformers.exe")
    file(root / f"Lib/site-packages/transformers-{version}.dist-info/METADATA", f"Name: transformers\nVersion: {version}\n\n")
    return executable


def strata(root: Path, version="0.1.43"):
    file(root / "serve/server.py")
    file(root / "engine/strata.exe")
    file(root / ".venv/Scripts/python.exe")
    file(root / "engine/BUILD.json", json.dumps({"version": version}))
    return root


@pytest.fixture
def machine(tmp_path, monkeypatch):
    application = tmp_path / "tools/ModLoader"
    paths = AppPaths(application / "data")
    monkeypatch.setattr(discovery, "_application_roots", lambda _: [application])
    monkeypatch.setattr(discovery, "_registered_python_roots", lambda: [])
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData", "CONDA_PREFIX", "CONDA_ENVS_PATH"):
        monkeypatch.delenv(name, raising=False)
    def forbid_execution(*args, **kwargs):
        raise AssertionError("Automatic discovery must not execute any program")
    monkeypatch.setattr(subprocess, "run", forbid_execution)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbid_execution)
    return paths


def test_default_discovers_non_path_layouts_and_only_complete_entries(machine):
    tools = machine.root.parent.parent
    llama = file(tools / "llama.cpp/build/bin/Release/llama-server.exe")
    engine = strata(tools / "Strata")
    cli = transformers(tools / "other-project/.venv")
    managed = file(machine.runtimes / "llama_cpp-b9999-test/release/llama-server.exe")
    file(machine.runtimes / "transformers-incomplete/venv/Scripts/python.exe")
    file(tools / "Strata-unprepared/serve/server.py")
    file(tools / "Strata-unprepared/engine/strata.exe")
    result = discovery.discover_runtimes(machine)
    found = {runtime["path"]: runtime for runtime in result["runtimes"]}
    assert set(found) == {str(llama), str(engine), str(cli), str(managed)}
    assert found[str(engine)]["version"] == "0.1.43"
    assert found[str(cli)]["version"] == "5.19.0"
    assert found[str(llama)]["version"] == "unknown"
    assert all(not runtime["registered"] for runtime in found.values())
    assert found[str(managed)]["metadata"]["discovery_source"] == "应用管理目录"
    assert result["searched"] and result["errors"] == []


def test_windows_python_and_conda_install_locations(machine, monkeypatch, tmp_path):
    local_python = transformers(tmp_path / "local/Programs/Python/Python312")
    registered_python = transformers(tmp_path / "registered-python")
    conda = transformers(tmp_path / "home/miniconda3/envs/inference")
    user_conda = transformers(tmp_path / "home/.conda/envs/other")
    custom_conda = transformers(tmp_path / "other-env-root/runtime")
    monkeypatch.setattr(discovery, "_registered_python_roots", lambda: [registered_python.parent.parent])
    monkeypatch.setenv("CONDA_ENVS_PATH", str(custom_conda.parent.parent.parent))
    result = discovery.discover_runtimes(machine)
    assert {runtime["path"] for runtime in result["runtimes"]} == {str(path) for path in (local_python, registered_python, conda, user_conda, custom_conda)}


def test_custom_folder_has_no_depth_limit_and_skips_caches_and_library_internals(machine, tmp_path):
    selected = tmp_path / "chosen"
    deep = file(selected / "vendor/family/revision/platform/build/release/llama-server.exe")
    cli = transformers(selected / "project/.venv")
    for relative in (".git", "node_modules", "models", ".cache/huggingface", "project/.venv/Lib/internal", "other/site-packages"):
        file(selected / relative / "llama-server.exe")
    result = discovery.discover_runtimes(machine, search_path=str(selected))
    assert {runtime["path"] for runtime in result["runtimes"]} == {str(deep), str(cli)}
    assert str(deep.parent) in result["searched"]
    assert all(runtime["metadata"]["discovery_source"] == "所选文件夹" for runtime in result["runtimes"])


def test_selecting_system_python_scripts_reads_fixed_dependencies_only(machine, tmp_path):
    environment = tmp_path / "system-python"
    file(environment / "python.exe")
    cli = file(environment / "Scripts/transformers.exe")
    file(environment / "Lib/site-packages/transformers-5.19.0.dist-info/METADATA", "Name: transformers\nVersion: 5.19.0\n\n")
    file(environment / "llama-server.exe")
    file(environment / "another-tool/llama-server.exe")
    result = discovery.discover_runtimes(machine, search_path=str(cli.parent))
    assert [runtime["path"] for runtime in result["runtimes"]] == [str(cli)]
    assert result["runtimes"][0]["version"] == "5.19.0"
    assert str(environment / "another-tool") not in result["searched"]


def test_path_aliases_deduplicate_and_registered_metadata_and_id_survive(machine, monkeypatch):
    executable = file(machine.root.parent / "bin/llama-server.exe")
    monkeypatch.setenv("PATH", os.pathsep.join((str(executable.parent), str(executable.parent / "../bin"))))
    registered = RuntimeInstallation(id="existing-id", backend="llama_cpp", name="My runtime", path=str(executable), version="b123", metadata={"supported_flags": ["--threads"]}).model_dump()
    result = discovery.discover_runtimes(machine, [registered])
    assert len(result["runtimes"]) == 1
    runtime = result["runtimes"][0]
    assert runtime["registered"] and runtime["id"] == "existing-id"
    assert runtime["name"] == "My runtime" and runtime["version"] == "b123"
    assert runtime["metadata"]["supported_flags"] == ["--threads"]
    assert discovery._path_key(executable) == ntpath.normcase(str(executable.resolve()))
    assert discovery._path_key(executable) == discovery._path_key(Path(str(executable).upper()))


def test_python_without_transformers_cli_and_unregistered_wsl_are_not_candidates(machine):
    file(machine.root.parent / ".venv/Scripts/python.exe")
    file(machine.runtimes / "transformers-half-installed/venv/Scripts/python.exe")
    file(machine.runtimes / "vllm-incomplete/bin/vllm")
    registered = [RuntimeInstallation(id="wsl", backend="vllm", name="Existing vLLM", path="/opt/modloader/vllm/bin/vllm", metadata={"distribution": "ModLoader"}).model_dump(),
                  RuntimeInstallation(id="unrelated-wsl", backend="vllm", name="Ubuntu", path="/opt/vllm/bin/vllm", metadata={"distribution": "Ubuntu"}).model_dump()]
    result = discovery.discover_runtimes(machine, registered)
    assert {runtime["id"] for runtime in result["runtimes"]} == {"wsl", "unrelated-wsl"}
    assert result["runtimes"][0]["registered"]
    assert "WSL 未启动验证" in result["runtimes"][0]["metadata"]["discovery_source"]


def test_formal_wsl_backends_remain_registered_without_starting_distribution(machine):
    registered = [RuntimeInstallation(id=backend, backend=backend, platform='wsl2', name=backend,
                                     path='/opt/' + backend + '/bin/service', metadata={'distribution': 'ExternalUbuntu'}).model_dump()
                  for backend in ('sglang', 'ninfer', 'sparkinfer')]
    result = discovery.discover_runtimes(machine, registered)
    assert {runtime['id'] for runtime in result['runtimes']} == {'sglang', 'ninfer', 'sparkinfer'}
    assert all(runtime['registered'] and 'WSL 未启动验证' in runtime['metadata']['discovery_source'] for runtime in result['runtimes'])


def test_permission_error_returns_other_candidates_once(machine, tmp_path, monkeypatch):
    selected = tmp_path / "chosen"
    executable = file(selected / "available/llama-server.exe")
    denied = selected / "denied"
    denied.mkdir()
    original = Path.iterdir
    def entries(path):
        if path == denied:
            raise PermissionError("fixture permission denied")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", entries)
    result = discovery.discover_runtimes(machine, search_path=str(selected))
    assert [runtime["path"] for runtime in result["runtimes"]] == [str(executable)]
    assert result["errors"] == [{"path": str(denied), "error": "fixture permission denied"}]


def test_invalid_static_metadata_keeps_candidate_with_unknown_version(machine, tmp_path):
    selected = strata(tmp_path / "chosen")
    (selected / "engine/BUILD.json").write_text("not json")
    result = discovery.discover_runtimes(machine, search_path=str(selected))
    assert result["runtimes"][0]["version"] == "unknown"
    assert len(result["errors"]) == 1 and result["errors"][0]["path"].endswith("BUILD.json")


def test_custom_folder_does_not_follow_external_links_or_cycles(machine, tmp_path):
    selected = tmp_path / "chosen"
    local = file(selected / "local/llama-server.exe")
    outside = file(tmp_path / "outside/llama-server.exe")
    try:
        (selected / "escape").symlink_to(outside.parent, target_is_directory=True)
        (selected / "cycle").symlink_to(selected, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    result = discovery.discover_runtimes(machine, search_path=str(selected))
    assert [runtime["path"] for runtime in result["runtimes"]] == [str(local)]
    assert str(outside.parent) not in result["searched"]


def test_cancel_stops_recursive_work_and_preserves_partial_results(machine, tmp_path, monkeypatch):
    selected = tmp_path / "chosen"
    executable = file(selected / "llama-server.exe")
    file(selected / "later/llama-server.exe")
    cancel = threading.Event()
    original = Path.iterdir
    def entries(path):
        if path == selected / "later":
            cancel.set()
        return original(path)
    monkeypatch.setattr(Path, "iterdir", entries)
    result = discovery.discover_runtimes(machine, search_path=str(selected), cancel=cancel)
    assert [runtime["path"] for runtime in result["runtimes"]] == [str(executable)]
    assert any("已取消" in note for note in result["notes"])
    assert str(selected / "later") not in result["searched"]


@pytest.mark.parametrize("choice", ["missing", "file", "wsl"])
def test_invalid_custom_root_is_rejected(machine, tmp_path, choice):
    selected = {"missing": str(tmp_path / "missing"), "file": str(file(tmp_path / "file.txt")), "wsl": "\\\\wsl$\\Ubuntu\\opt"}[choice]
    with pytest.raises(ValueError):
        discovery.discover_runtimes(machine, search_path=selected)


async def test_detect_endpoint_requires_auth_and_never_registers_candidates(machine, tmp_path):
    selected = tmp_path / "chosen"
    executable = file(selected / "llama-server.exe")
    controller = Controller(machine.root, "management-secret")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url="http://testserver") as client:
            assert (await client.get("/api/runtimes/detect", params={"path": str(selected)})).status_code == 401
            client.headers["Authorization"] = "Bearer management-secret"
            response = await client.get("/api/runtimes/detect", params={"path": str(selected)})
            assert response.status_code == 200
            result = response.json()
            assert [runtime["path"] for runtime in result["runtimes"]] == [str(executable)]
            assert set(result) == {"runtimes", "searched", "errors", "notes"}
            assert controller.store.all("runtimes") == []
            assert (await client.get("/api/runtimes/detect", params={"path": str(executable)})).status_code == 400
    finally:
        await controller.close()
        controller.store.close()
