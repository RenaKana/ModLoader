"""Read-only runtime discovery. Executables are validated only on explicit import."""
from __future__ import annotations

import json
import ntpath
import os
from pathlib import Path
import stat
import sys
import threading
import time
from typing import Callable, Iterable
import uuid

from .models import AppPaths, RuntimeInstallation


LABELS = {"llama_cpp": "llama.cpp", "strata": "Strata", "transformers": "Transformers", "vllm": "vLLM", "custom": "自定义引擎",
          "sglang": "SGLang", "ninfer": "NInfer", "sparkinfer": "SparkInfer"}
SKIP_DIRECTORIES = {".git", "node_modules", "__pycache__", ".cache", "cache", "caches", "models", "weights", "huggingface", ".huggingface", "hub", "blobs", "snapshots", "site-packages", "wsl-modloader"}
BINARY_LAYOUTS = ("bin", "build/bin", "build/bin/Release", "build/bin/Debug", "build/bin/RelWithDebInfo", "build/bin/MinSizeRel", "bin/Release", "Release")
ENV_LAYOUTS = (".venv", "venv", "env")


def _path_key(path: Path) -> str:
    # Discovery targets Windows, including when fixture tests run elsewhere.
    return ntpath.normcase(str(path.resolve()))


def _wsl_path(path: Path) -> bool:
    return ntpath.normcase(str(path)).startswith(("\\\\wsl$\\", "\\\\wsl.localhost\\"))


def _registered_python_roots() -> list[Path]:
    """PEP 514 install locations, without invoking Python or its launcher."""
    try:
        import winreg
    except ImportError:
        return []
    roots = []
    def subkeys(key):
        index = 0
        while True:
            try:
                yield winreg.EnumKey(key, index)
            except OSError:
                return
            index += 1
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, "Software\\Python", 0, winreg.KEY_READ | view) as companies:
                    for company in subkeys(companies):
                        try:
                            with winreg.OpenKey(companies, company) as tags:
                                for tag in subkeys(tags):
                                    try:
                                        with winreg.OpenKey(tags, tag + "\\InstallPath") as installation:
                                            value, _ = winreg.QueryValueEx(installation, None)
                                            if isinstance(value, str) and value:
                                                roots.append(Path(value))
                                    except OSError:
                                        pass
                        except OSError:
                            pass
            except OSError:
                pass
    return roots


class _Discovery:
    def __init__(self, cancel: threading.Event | None, progress: Callable[[dict], None] | None = None):
        self.cancel = cancel
        self.progress = progress
        self.runtimes: dict[tuple[str, str], dict] = {}
        self.searched: dict[str, str] = {}
        self.errors: dict[str, dict] = {}
        self.inspected: set[str] = set()
        self.walked: set[str] = set()
        self.checked = 0
        self.current_source = ""
        self.current_path = ""
        self.last_progress = 0.0

    def report(self, path: Path | None = None, source: str | None = None, *, force=False):
        changed_source = source is not None and source != self.current_source
        if source is not None:
            self.current_source = source
        if path is not None:
            self.current_path = str(path)
        now = time.monotonic()
        if self.progress and (force or changed_source or now - self.last_progress >= .1):
            self.last_progress = now
            location = " · ".join(value for value in (self.current_source, self.current_path) if value)
            self.progress({"stage": "runtime_discovery",
                           "message": f"已检查 {self.checked} 个目录 · 已发现 {len(self.runtimes)} 个环境 · {location}"})

    def stopped(self) -> bool:
        return bool(self.cancel and self.cancel.is_set())

    def error(self, path: Path, error: Exception | str):
        self.errors.setdefault(ntpath.normcase(str(path)), {"path": str(path), "error": str(error)})

    def resolve(self, path: Path) -> Path | None:
        if self.stopped() or _wsl_path(path):
            return None
        try:
            return path.expanduser().resolve()
        except (OSError, RuntimeError) as error:
            self.error(path, error)
            return None

    def exists(self, path: Path, *, directory=False, boundary: Path | None = None) -> bool:
        if self.stopped() or _wsl_path(path):
            return False
        try:
            if boundary is not None and not path.resolve().is_relative_to(boundary):
                return False
            mode = path.stat().st_mode
            return stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
        except (FileNotFoundError, NotADirectoryError):
            return False
        except (OSError, RuntimeError) as error:
            self.error(path, error)
            return False

    def children(self, path: Path) -> list[Path]:
        if self.stopped():
            return []
        self.report(path)
        try:
            children = []
            for child in path.iterdir():
                if self.stopped():
                    break
                children.append(child)
            return sorted(children, key=lambda item: item.name.casefold())
        except OSError as error:
            self.error(path, error)
            return []

    def add(self, backend: str, path: Path, source: str, version="unknown", metadata=None):
        resolved = self.resolve(path)
        if resolved is None:
            return
        key = (backend, _path_key(resolved))
        if key in self.runtimes:
            return
        name = LABELS[backend] if version == "unknown" else f"{LABELS[backend]} · {version}"
        runtime = RuntimeInstallation(id=str(uuid.uuid5(uuid.NAMESPACE_URL, backend + ":" + key[1])),
                                      backend=backend, name=name, path=str(resolved),
                                      version=version, metadata={**(metadata or {}), "discovery_source": source})
        self.runtimes[key] = {**runtime.model_dump(), "registered": False}
        self.report(force=True)

    def python(self, root: Path, boundary: Path | None = None) -> Path | None:
        for path in (root / "Scripts/python.exe", root / "python.exe"):
            if self.exists(path, boundary=boundary):
                return path
        return None

    def transformers(self, executable: Path, source: str, boundary: Path | None):
        if not self.exists(executable, boundary=boundary):
            return
        environment = executable.parent.parent if executable.parent.name.casefold() == "scripts" else executable.parent
        metadata_boundary = boundary
        # A user may select Scripts itself. Check only its owning environment's
        # fixed dependencies; do not traverse its parent or external junctions.
        if boundary == executable.parent and executable.parent.name.casefold() == "scripts":
            metadata_boundary = self.resolve(environment)
            if metadata_boundary is None or executable.parent.parent != metadata_boundary:
                return
            self.searched.setdefault(_path_key(metadata_boundary), str(metadata_boundary))
        python = self.python(environment, metadata_boundary)
        if python is None:
            return
        version = "unknown"
        packages = environment / "Lib/site-packages"
        if self.exists(packages, directory=True, boundary=metadata_boundary):
            matches = [package for package in self.children(packages)
                       if package.name.casefold().startswith("transformers-") and package.name.casefold().endswith(".dist-info")]
            # Multiple metadata directories can be leftovers from an interrupted upgrade.
            if len(matches) == 1:
                metadata = matches[0] / "METADATA"
                if self.exists(metadata, boundary=metadata_boundary):
                    try:
                        headers = {}
                        with metadata.open(encoding="utf-8", errors="replace") as stream:
                            for line in stream:
                                if self.stopped() or not line.strip():
                                    break
                                key, separator, value = line.partition(":")
                                if separator and key in ("Name", "Version"):
                                    headers[key] = value.strip()
                        if headers.get("Name", "").casefold() == "transformers":
                            version = headers.get("Version") or "unknown"
                    except OSError as error:
                        self.error(metadata, error)
        self.add("transformers", executable, source, version, {"python": str(python.resolve())})

    def strata(self, root: Path, source: str, boundary: Path | None = None, python_override: str | None = None,
               family: str | None = None):
        if not self.exists(root / "serve/server.py", boundary=boundary):
            return
        engine = root / "engine/strata.exe"
        deepseek = family == "deepseek41"
        if deepseek or not self.exists(engine, boundary=boundary):
            from .strata_deepseek import engine_path
            try:
                engine = engine_path(root, boundary=boundary)
            except ValueError:
                return
            if not self.exists(engine, boundary=boundary):
                return
            deepseek = True
        python = Path(python_override) if python_override else self.python(root / ".venv", boundary)
        if python is None or not self.exists(python, boundary=boundary):
            return
        version = "unknown"
        build = engine.parent / "BUILD.json"
        if self.exists(build, boundary=boundary):
            try:
                metadata = json.loads(build.read_text(encoding="utf-8-sig"))
                if isinstance(metadata, dict) and isinstance(metadata.get("version"), (str, int, float)):
                    version = str(metadata["version"])
            except (OSError, ValueError) as error:
                self.error(build, error)
        self.add("strata", root, source, version,
                 {"python": str(python.resolve()), **({"strata_family": "deepseek41"} if deepseek else {})})
        return True

    def inspect(self, directory: Path, source: str, boundary: Path | None = None):
        self.report(directory, source)
        directory = self.resolve(directory)
        if directory is None or not self.exists(directory, directory=True, boundary=boundary):
            return
        key = _path_key(directory)
        if key in self.inspected:
            return
        self.inspected.add(key)
        self.searched.setdefault(key, str(directory))
        for filename in ("llama-server.exe", "llama-server"):
            executable = directory / filename
            if self.exists(executable, boundary=boundary):
                self.add("llama_cpp", executable, source)
        self.transformers(directory / "transformers.exe", source, boundary)
        self.strata(directory, source, boundary)
        if not self.stopped():
            self.checked += 1
        self.report()

    def layout(self, directory: Path, source: str):
        self.inspect(directory, source)
        for relative in BINARY_LAYOUTS:
            self.inspect(directory / relative, source)
        self.inspect(directory / "Scripts", source)
        for relative in ENV_LAYOUTS:
            self.inspect(directory / relative / "Scripts", source)

    def conda(self, directory: Path):
        self.report(directory, "Conda 环境")
        self.inspect(directory / "Scripts", "Conda 环境")
        environments = directory / "envs"
        if self.exists(environments, directory=True):
            self.searched.setdefault(_path_key(environments), str(environments.resolve()))
            for environment in self.children(environments):
                self.inspect(environment / "Scripts", "Conda 环境")

    def tree(self, root: Path, source: str):
        self.report(root, source)
        root = self.resolve(root)
        if root is None or not self.exists(root, directory=True):
            return
        pending = [root]
        while pending and not self.stopped():
            directory = self.resolve(pending.pop())
            if directory is None or not directory.is_relative_to(root):
                continue
            key = _path_key(directory)
            if key in self.walked:
                continue
            self.walked.add(key)
            self.report(directory, source)
            entries = self.children(directory)
            if self.stopped():
                break
            if ntpath.normcase(str(directory)) in self.errors:
                self.searched.setdefault(key, str(directory))
                continue
            self.inspect(directory, source, root)
            children = []
            for child in entries:
                if self.stopped():
                    break
                name = child.name.casefold()
                if name in SKIP_DIRECTORIES or (name == "lib" and self.python(directory, root) is not None):
                    continue
                if self.exists(child, directory=True, boundary=root):
                    children.append(child)
            pending.extend(reversed(children))

    def registered(self, records: Iterable[dict]):
        for record in records:
            if self.stopped():
                return
            try:
                runtime = RuntimeInstallation.model_validate(record)
            except ValueError:
                continue
            if runtime.backend not in LABELS:
                continue
            if runtime.platform == "wsl2":
                if runtime.metadata.get("distribution") and runtime.path.startswith("/"):
                    self.runtimes[(runtime.backend, runtime.metadata["distribution"], runtime.path)] = {**runtime.model_dump(), "registered": True,
                        "metadata": {**runtime.metadata, "discovery_source": "已登记（WSL 未启动验证）"}}
                    self.report(Path(runtime.path), "已登记（WSL）", force=True)
                continue
            path = self.resolve(Path(runtime.path))
            if path is None:
                continue
            key = (runtime.backend, _path_key(path))
            self.report(path, "已登记")
            valid = True
            if runtime.backend == "strata":
                valid = self.strata(path, "已登记", python_override=runtime.metadata.get("python"),
                                    family=runtime.metadata.get("strata_family"))
            else:
                self.inspect(path.parent, "已登记")
            if valid and key in self.runtimes:
                candidate = self.runtimes[key]
                self.runtimes[key] = {**runtime.model_dump(), "path": str(path), "registered": True,
                    "metadata": {**candidate["metadata"], **runtime.metadata, "discovery_source": "已登记"}}
            else:
                self.error(path, "已登记环境缺少必要的运行入口，请检查路径或重新导入")


def _application_roots(paths: AppPaths) -> list[Path]:
    roots = [paths.root.parent, Path.cwd()]
    executable = Path(sys.executable)
    if getattr(sys, "frozen", False):
        roots.append(executable.parent)
        if executable.parent.name.casefold() == "core" and executable.parent.parent.name.casefold() == "resources":
            roots.append(executable.parent.parent.parent)
    elif executable.parent.name.casefold() == "scripts":
        roots.append(executable.parent.parent.parent)
    return list(dict.fromkeys(roots))


def _default_search(discovery: _Discovery, paths: AppPaths):
    discovery.report(source="PATH", force=True)
    for value in os.environ.get("PATH", "").split(os.pathsep):
        if value.strip().strip('"'):
            discovery.inspect(Path(value.strip().strip('"')), "PATH")
    discovery.inspect(paths.runtimes, "应用管理目录")
    discovery.report(paths.runtimes, "应用管理目录")
    for child in discovery.children(paths.runtimes):
        if child.name.casefold().startswith(("llama_cpp-", "strata-", "transformers-")):
            discovery.tree(child, "应用管理目录")
    roots = _application_roots(paths)
    parents = set()
    for root in roots:
        discovery.layout(root, "应用附近")
        for parent in (root, root.parent):
            resolved = discovery.resolve(parent)
            if resolved is None or _path_key(resolved) in parents:
                continue
            parents.add(_path_key(resolved))
            if not discovery.exists(resolved, directory=True):
                continue
            discovery.report(resolved, "应用附近")
            discovery.searched.setdefault(_path_key(resolved), str(resolved))
            for child in discovery.children(resolved):
                name = child.name.casefold()
                if name in SKIP_DIRECTORIES:
                    continue
                if name.startswith(("llama.cpp", "llama_cpp-", "llama-b", "strata")):
                    discovery.layout(child, "应用附近")
                elif name.startswith(("miniconda", "anaconda")):
                    discovery.conda(child)
                else:
                    for layout in ENV_LAYOUTS:
                        discovery.inspect(child / layout / "Scripts", "应用附近的 Python 环境")
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    local = Path(os.environ.get("LOCALAPPDATA") or home / "AppData/Local")
    python_locations = [local / "Programs/Python", home / "AppData/Local/Programs/Python"]
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(variable):
            python_locations.append(Path(os.environ[variable]))
    python_roots = _registered_python_roots()
    for location in dict.fromkeys(python_locations):
        if discovery.exists(location, directory=True):
            discovery.report(location, "Python 安装目录")
            discovery.searched.setdefault(_path_key(location), str(location.resolve()))
            for child in discovery.children(location):
                if child.name.casefold().startswith("python"):
                    python_roots.append(child)
    for root in python_roots:
        discovery.inspect(root / "Scripts", "Python 安装目录")
    conda_roots = [home / "miniconda3", home / "anaconda3", local / "miniconda3", local / "anaconda3"]
    if os.environ.get("ProgramData"):
        conda_roots.extend(Path(os.environ["ProgramData"]) / name for name in ("miniconda3", "anaconda3"))
    if os.environ.get("CONDA_PREFIX"):
        conda_roots.append(Path(os.environ["CONDA_PREFIX"]))
    for root in conda_roots:
        discovery.conda(root)
    environment_locations = [home / ".conda/envs"]
    environment_locations.extend(Path(value) for value in os.environ.get("CONDA_ENVS_PATH", "").split(os.pathsep) if value)
    for location in environment_locations:
        if discovery.exists(location, directory=True):
            discovery.report(location, "Conda 环境")
            discovery.searched.setdefault(_path_key(location), str(location.resolve()))
            for child in discovery.children(location):
                discovery.inspect(child / "Scripts", "Conda 环境")


def discover_runtimes(paths: AppPaths, registered: Iterable[dict] = (), search_path: str | None = None,
                      *, cancel: threading.Event | None = None, progress: Callable[[dict], None] | None = None) -> dict:
    discovery = _Discovery(cancel, progress)
    if search_path is not None:
        selected = Path(search_path).expanduser()
        if not search_path.strip() or _wsl_path(selected):
            raise ValueError("请选择 Windows 本机文件夹；WSL 环境需显式导入验证")
        try:
            if not stat.S_ISDIR(selected.stat().st_mode):
                raise ValueError("搜索路径必须是文件夹")
        except (FileNotFoundError, NotADirectoryError):
            raise ValueError("搜索文件夹不存在，请重新选择") from None
        discovery.tree(selected, "所选文件夹")
    else:
        _default_search(discovery, paths)
    discovery.registered(registered)
    discovery.report(force=True)
    notes = ["搜索仅检查本机文件与静态版本信息，不启动后端；候选需导入后验证。",
             "默认搜索 PATH、应用管理目录、应用附近的常见后端 / Python 环境、Python 安装目录与 Conda 环境。",
             "选择文件夹后递归搜索全部层级，跳过 Git、node_modules、模型缓存和 Python 库内部，不跟随超出搜索范围的链接。",
             "发现 Scripts/transformers.exe 时，可读取所属 Python 目录的固定入口与版本元数据，不递归搜索父目录。",
             "WSL 仅列出已登记的运行环境；搜索不启动 WSL，Linux 入口需在显式导入或加载时验证。"]
    if discovery.stopped():
        notes.append("搜索已取消，仅返回停止前发现的结果。")
    return {"runtimes": sorted(discovery.runtimes.values(), key=lambda item: (item["backend"], ntpath.normcase(item["path"]))),
            "searched": list(discovery.searched.values()), "errors": list(discovery.errors.values()), "notes": notes}
