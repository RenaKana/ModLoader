"""Explicit, cancellable installation into AppPaths; never called by detection."""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import contextmanager, nullcontext
import errno
import hashlib
import ipaddress
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid
import zipfile
import copy
import re
import shlex
from urllib.parse import unquote, urlsplit

import httpx
import psutil

from ..models import AppPaths, RuntimeInstallation
from ..sources.control import DownloadPause, PAUSE_CHUNK_SIZE
from .adapters import import_runtime
from ..platforms import windows_to_wsl, WSL_SUPERVISOR, runtime_distribution, list_wsl
from . import catalog

VERSIONS = catalog.RECOMMENDED
# Per-installation worker budget; all files remain eligible for download.
INSTALL_DOWNLOAD_CONCURRENCY = 4
# Keep a bounded diagnostic excerpt, separate from the final parallel-build warnings.
COMMAND_TAIL_LINES = 15
COMMAND_DIAGNOSTIC_LINES = 48
COMMAND_DIAGNOSTIC_LINE_CHARS = 1600
_COMMAND_FAILURE = re.compile(r'(?:^FAILED:|\bfatal error:|\berror:|\bCMake Error\b|\bnvlink error\b|^Traceback \(most recent call last\):)', re.I)
UBUNTU_URL = "https://cloud-images.ubuntu.com/wsl/releases/24.04/20240423/ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz"
UBUNTU_SHA256 = "2a790896740b14d637dbdc583cce1ba081ac53b9e9cdb46dc09a2f73abbd9934"
PYTHON_URL = "https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg"
# PSF's official NuGet package includes venv, ensurepip and pip, unlike the embeddable zip.
PYTHON_SHA256 = "0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8"


async def _emit(progress, **event):
    if progress:
        value = progress(event)
        if inspect.isawaitable(value):
            await value


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise asyncio.CancelledError("安装已取消；已有运行时保持不变")


async def _thread(operation, *args, cancel, pause=None):
    # Cancelling to_thread alone leaves its writer running after the job ends.
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancel.set()
        if pause:
            pause.resume()
        await asyncio.gather(task, return_exceptions=True)
        raise


async def _json(url: str, proxy=None):
    return await catalog._json(url, proxy)


async def _download(url, target: Path, progress, cancel, digest: str | None = None, pause=None, proxy=None):
    _check(cancel)
    filename = re.sub(r'\.[a-f0-9]{32}\.part$', '', target.name)
    await _emit(progress, phase="download", message=f"下载 {filename}", can_pause=pause is not None)
    loop = asyncio.get_running_loop()
    def transfer():
        req = urllib.request.Request(url, headers={"User-Agent": "ModLoader/0.1"})
        hasher, count, last_percent = hashlib.sha256(), 0, -1
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {}))
        with opener.open(req, timeout=45) as response, target.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            while True:
                _check(cancel)
                if pause:
                    out.flush()
                    asyncio.run_coroutine_threadsafe(pause.checkpoint(cancel), loop).result()
                # read(n) waits to fill n bytes on slow streams. read1 returns
                # available data after one socket read, so pause is checked promptly.
                chunk = response.read1(PAUSE_CHUNK_SIZE)
                _check(cancel)
                if pause:
                    asyncio.run_coroutine_threadsafe(pause.checkpoint(cancel), loop).result()
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
                count += len(chunk)
                percent = int(count * 100 / total) if total else 0
                if percent != last_percent:
                    future = asyncio.run_coroutine_threadsafe(_emit(progress, phase="download", message=filename, downloaded=count, total=total, percent=percent), loop)
                    future.result()
                    last_percent = percent
            if total and count != total:
                raise ValueError(f'下载长度不匹配：{filename}（实收 {count} B / 应收 {total} B）')
        if digest and hasher.hexdigest() != digest.removeprefix("sha256:"):
            raise ValueError("官方发行件 SHA256 校验失败，未安装")
    try:
        await _thread(transfer, cancel=cancel, pause=pause)
    finally:
        await _emit(progress, can_pause=False)


@contextmanager
def _install_file_access(path):
    try:
        yield
    except OSError as error:
        if error.errno != errno.EBADF:
            raise
        raise ValueError(f"安装文件访问被中断：{path}（系统错误 9）。请检查安全软件的拦截或隔离记录，处理后重试；已下载文件会复用。") from error


def _extract(archive: Path, destination: Path, cancel=None):
    base = destination.resolve()
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            members = []
            for info in tar:
                _check(cancel)
                member = (base / info.name).resolve()
                if not member.is_relative_to(base) or not (info.isfile() or info.isdir()):
                    raise ValueError("发行包包含目录穿越、链接或特殊文件")
                members.append(info)
            for info in members:
                _check(cancel)
                member = (base / info.name).resolve()
                if info.isdir():
                    member.mkdir(parents=True, exist_ok=True)
                    continue
                member.parent.mkdir(parents=True, exist_ok=True)
                with _install_file_access(member), tar.extractfile(info) as source, member.open('wb') as output:
                    while True:
                        _check(cancel)
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        _check(cancel)
                        output.write(chunk)
        return
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            _check(cancel)
            member = (base / info.filename).resolve()
            if not member.is_relative_to(base) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("发行包包含目录穿越或符号链接")
        for info in z.infolist():
            _check(cancel)
            member = (base / info.filename).resolve()
            if info.is_dir():
                member.mkdir(parents=True, exist_ok=True)
                continue
            member.parent.mkdir(parents=True, exist_ok=True)
            with _install_file_access(member), z.open(info) as source, member.open('wb') as output:
                while True:
                    _check(cancel)
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    _check(cancel)
                    output.write(chunk)


async def _unpack(archive, destination, progress, cancel):
    _check(cancel)
    await _emit(progress, phase='extract', message=f'解压 {archive.name}', can_pause=False, percent=None)
    await _thread(_extract, archive, destination, cancel, cancel=cancel)


def _verify_source(checkout, expected_files, cancel):
    actual_files = {path.relative_to(checkout).as_posix(): path for path in checkout.rglob("*") if path.is_file()}
    if actual_files.keys() != expected_files.keys():
        missing = sorted(expected_files.keys() - actual_files.keys())
        if missing:
            raise ValueError(f"源码包内容与确认的官方 Git 树不一致，缺少文件：{checkout / missing[0]}。请检查安全软件的拦截或隔离记录，处理后重试。")
        raise ValueError("源码包内容与确认的官方 Git 树不一致")
    def blob(data):
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    attributes = actual_files.get(".gitattributes")
    crlf_scripts = set()
    if attributes:
        with _install_file_access(attributes):
            data = attributes.read_bytes()
        if blob(data) == expected_files[".gitattributes"]:
            rules = {line.strip() for line in data.splitlines()}
            crlf_scripts = {suffix for suffix in (".bat", ".cmd", ".ps1")
                            if ("*" + suffix + " text eol=crlf").encode() in rules}
    for relative, path in actual_files.items():
        _check(cancel)
        with _install_file_access(path), path.open("rb") as source:
            digest = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
            while chunk := source.read(1024 * 1024):
                _check(cancel)
                digest.update(chunk)
        if digest.hexdigest() != expected_files[relative]:
            # GitHub applies Strata's verified CRLF attribute when archiving Windows
            # launchers. Git stores their blobs with LF. Do not normalize binary
            # files or paths whose subtree has its own attribute rules.
            overridden = any((parent / ".gitattributes").as_posix() in actual_files
                             for parent in Path(relative).parents if parent != Path("."))
            if path.suffix in crlf_scripts and not overridden:
                with _install_file_access(path):
                    normalized = blob(path.read_bytes().replace(b"\r\n", b"\n"))
                if normalized == expected_files[relative]:
                    continue
            raise ValueError(f"源码 Git blob 校验失败（{relative}），未执行源码")


async def _command(argv, progress, cancel, cwd=None, cleanup=None, timeout=3600, owned_root: Path | None = None,
                   env: dict[str, str] | None = None):
    _check(cancel)
    await _emit(progress, phase="install", message=f"运行 {Path(argv[0]).name}", can_pause=False)
    environment = dict(os.environ)
    if env is not None:
        environment.update(env)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if Path(argv[0]).name.lower() == "wsl.exe":
        # WSL's own diagnostics otherwise use UTF-16LE alongside Linux UTF-8 output.
        environment["WSL_UTF8"] = "1"
    if owned_root is not None:
        root = Path(owned_root).resolve()
        cache, temp = root / "cache/pip", root / "tmp"
        cache.mkdir(parents=True, exist_ok=True)
        temp.mkdir(parents=True, exist_ok=True)
        environment.update(PIP_CACHE_DIR=str(cache), TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp))
        cwd = cwd or str(root)
    process = await asyncio.create_subprocess_exec(*argv, cwd=cwd, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    async def output():
        tail = deque(maxlen=COMMAND_TAIL_LINES)
        context = deque(maxlen=2)
        diagnostics = []
        following = 0
        def retain(entries):
            for number, value in entries:
                if len(diagnostics) < COMMAND_DIAGNOSTIC_LINES and (not diagnostics or number > diagnostics[-1][0]):
                    diagnostics.append((number, value[:COMMAND_DIAGNOSTIC_LINE_CHARS]))
        line_number = 0
        while line := await process.stdout.readline():
            value = line.decode("utf-8", errors="replace").strip()
            # Install commands contain only public package URLs and versions, never credentials.
            line_number += 1
            entry = (line_number, value)
            tail.append(entry)
            if _COMMAND_FAILURE.search(value):
                retain([*context, entry])
                following = 2
            elif following:
                retain([entry])
                following -= 1
            context.append(entry)
            await _emit(progress, phase="install", message=value)
        await process.wait()
        if process.returncode:
            diagnostic_ids = {number for number, _ in diagnostics}
            ending = [value[:COMMAND_DIAGNOSTIC_LINE_CHARS] for number, value in tail if number not in diagnostic_ids]
            detail = '关键失败输出：\n' + '\n'.join(value for _, value in diagnostics) + '\n末尾输出：\n' if diagnostics else ''
            raise RuntimeError(f"安装命令失败 ({process.returncode}): " + detail + "\n".join(ending))
        return "\n".join(value for _, value in tail)
    task = asyncio.create_task(output())
    cancelled = asyncio.create_task(cancel.wait())
    try:
        done, _ = await asyncio.wait([task, cancelled], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        if cancelled in done:
            raise asyncio.CancelledError("安装已取消")
        raise TimeoutError("安装步骤超时，已停止该安装进程")
    finally:
        cancelled.cancel()
        if process.returncode is None:
            if process.stdin:
                process.stdin.close()
            if cleanup:
                cleaner = await asyncio.create_subprocess_exec(*cleanup, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    await asyncio.wait_for(cleaner.wait(), 15)
                except asyncio.TimeoutError:
                    cleaner.kill()
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    await asyncio.wait_for(killer.wait(), 10)
                except asyncio.TimeoutError:
                    killer.kill()
                    await killer.wait()
            else:
                process.kill()
            await asyncio.wait_for(process.wait(), 5)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, cancelled, return_exceptions=True)


async def _release(repo, tag, asset, target, progress, cancel, pause=None):
    release = await _json(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    item = next((a for a in release["assets"] if a["name"] == asset), None)
    if not item:
        raise ValueError(f"官方固定版本 {tag} 没有发行件 {asset}")
    if not item.get("digest", "").startswith("sha256:"):
        raise ValueError("官方 API 未提供此发行件 SHA256，改用导入已安装运行时")
    await _download(item["browser_download_url"], target, progress, cancel, item["digest"], **({'pause': pause} if pause else {}))


async def _venv(root, options, progress, cancel, pause=None):
    python = options.get("python")
    if not python and not getattr(sys, "frozen", False):
        python = sys.executable
    if not python:
        managed = root.parent / "python-3.12.10"
        managed.mkdir(exist_ok=False)
        archive = managed / "python.nupkg"
        await _download(PYTHON_URL, archive, progress, cancel, PYTHON_SHA256, **({'pause': pause} if pause else {}))
        await _unpack(archive, managed, progress, cancel)
        python = str(managed / "tools/python.exe")
    await _command([python, "-m", "venv", str(root)], progress, cancel, owned_root=root.parent)
    return str(root / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


class _Transfers:
    """A single installation's stage/file accounting over the shared verified cache."""
    def __init__(self, paths, progress, cancel, pause, proxy):
        self.paths, self.progress, self.cancel, self.pause, self.proxy = paths, progress, cancel, pause, proxy
        self.stages = []
        self.received = {}

    async def publish(self, stage, **event):
        files = [item for entry in self.stages for item in entry["files"]]
        pending = [f for f in files if not f.get("cached")]
        total = sum(f["size_bytes"] for f in pending) if all(f.get("size_bytes") is not None for f in pending) else None
        count = sum(self.received.values())
        active = sum(item.get("status") == "downloading" for item in files)
        if event.get("phase") == "download" and active > 1:
            event["message"] = f"正在并行下载 {active} 个文件"
        await _emit(self.progress, manifest=copy.deepcopy(stage), downloaded_bytes=count, total_bytes=total,
                    completed_files=sum(f.get("status") in {"complete", "cached"} for f in files), total_files=len(files),
                    **{**dict(percent=int(count * 100 / total) if total else None), **event})

    async def add(self, stage, confirm=None):
        stage = copy.deepcopy(stage)
        self.stages.append(stage)
        await self.publish(stage, phase="prepare", can_pause=False, message=stage["title"])
        if confirm:
            await confirm(copy.deepcopy(stage))
            _check(self.cancel)
        return stage

    async def fetch(self, stage):
        result = {}
        remaining = iter(stage["files"])
        failure = None

        async def fetch_file(item, pause):
            nonlocal failure
            _check(self.cancel)
            if pause:
                await pause.checkpoint(self.cancel)
            path = catalog.cache_path(self.paths, item)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Re-check at use time: an old preview or another failed job cannot bless a corrupt cache.
            valid = await asyncio.to_thread(catalog.cached_file, self.paths, item)
            if valid:
                self.received[item["id"]] = 0
                item.update(status="cached", cached=True, downloaded_bytes=0, size_bytes=path.stat().st_size)
                await self.publish(stage, phase="download", file=item["name"], file_downloaded_bytes=0, file_total_bytes=0,
                                   message=f"复用已校验缓存 {item['name']}")
            else:
                item.update(status="downloading", cached=False, downloaded_bytes=0)
                expected_size = item.get("size_bytes")
                partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
                async def update(event):
                    if "downloaded" in event:
                        self.received[item["id"]] = event["downloaded"]
                        item["downloaded_bytes"] = event["downloaded"]
                    if event.get("total"):
                        item["size_bytes"] = event["total"]
                    # One completed worker must not disable pause for the others.
                    forwarded = {k: v for k, v in event.items() if k not in {"downloaded", "total", "percent", "can_pause"}}
                    await self.publish(stage, file=item["name"], file_downloaded_bytes=self.received.get(item["id"], 0),
                                       file_total_bytes=item.get("size_bytes"), can_pause=self.pause is not None and not self.cancel.is_set(), **forwarded)
                try:
                    await _download(item["url"], partial, update, self.cancel, item.get("digest"), pause, self.proxy)
                    _check(self.cancel)
                    size = partial.stat().st_size
                    if expected_size is not None and size != expected_size:
                        raise ValueError(f"发行清单与实际下载长度不匹配：{item['name']}（实收 {size} B / 应收 {expected_size} B）")
                    partial.replace(path)
                    if not item.get("digest"):
                        with path.open("rb") as stream:
                            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                        path.with_suffix(path.suffix + ".sha256").write_text(checksum)
                    self.received[item["id"]] = size
                    item.update(status="complete", downloaded_bytes=size, size_bytes=size)
                    await self.publish(stage, phase="download", file=item["name"], message=f"已校验 {item['name']}")
                except BaseException as error:
                    item["status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
                    if not isinstance(error, asyncio.CancelledError) and failure is None:
                        failure = error
                    self.cancel.set()
                    if self.pause:
                        self.pause.resume()
                    await self.publish(stage)
                    raise
                finally:
                    partial.unlink(missing_ok=True)
            result[item["id"]] = path

        async def worker():
            context = self.pause.participant() if self.pause else nullcontext()
            async with context as pause:
                for item in remaining:
                    await fetch_file(item, pause)

        workers = [asyncio.create_task(worker()) for _ in range(min(INSTALL_DOWNLOAD_CONCURRENCY, len(stage["files"])))]
        try:
            await asyncio.gather(*workers)
            if self.pause:
                await self.pause.checkpoint(self.cancel)
            _check(self.cancel)
        except BaseException:
            # Join every writer before failing/cancelling the installation.
            self.cancel.set()
            if self.pause:
                self.pause.resume()
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if failure is not None:
                raise failure
            raise
        finally:
            await self.publish(stage, can_pause=False, file=None, file_downloaded_bytes=0, file_total_bytes=0)
        # Offline requirements pair each path with its manifest hash, not completion order.
        return {item.get("role", item["id"]): result[item["id"]] for item in stage["files"]}


async def _approved_venv(root, options, artifacts, progress, cancel):
    python = options.get("python")
    if not python:
        base = root.parent / "python-3.12.10"
        await _unpack(artifacts["python"], base, progress, cancel)
        python = str(base / "tools/python.exe")
    await _command([python, "-m", "venv", str(root)], progress, cancel, owned_root=root.parent)
    return str(root / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


async def _package_stage(python, requirements, root, transfer, confirm, *, wsl=False, resolver_proxy=None, retry=None, approved_wheels=None):
    """Resolve metadata, record the frozen manifest, download, then install offline."""
    if confirm is None:
        raise ValueError("请通过环境安装任务下载和安装依赖")
    from . import resolve_packages
    native = (lambda p: windows_to_wsl(str(p.resolve()))) if wsl else str
    phase = "resolve"
    async def resolve_progress(event):
        await _emit(transfer.progress, **{**event, "phase": phase, "can_pause": False, "total_bytes": None, "percent": None})
    async def command(argv):
        if wsl:
            return await _wsl_command(argv, root, resolve_progress, transfer.cancel)
        return await _command(argv, resolve_progress, transfer.cancel, owned_root=root)

    context = dict(requirements=resolve_packages.requirements_identity(requirements), resolver=hashlib.sha256(resolve_packages.SOURCE.encode()).hexdigest())
    if approved_wheels:
        context['approved_wheels'] = [{key: item[key] for key in ('url', 'digest')} for item in approved_wheels]
    items = None
    if retry:
        previous = await asyncio.to_thread(resolve_packages.previous_resolution, transfer.paths, retry, requirements)
        if previous and all(previous.get(key) == value for key, value in context.items()):
            await resolve_progress(dict(message="正在检查目标 Python 环境，以复用上次依赖清单"))
            output = await command([python, "-c", "import json; from pip._vendor.packaging.markers import default_environment; print(json.dumps(default_environment()))"])
            environment = json.loads(output.strip().splitlines()[-1])
            if previous.get("environment") == environment:
                context["environment"] = environment
                items = retry["stage"]["files"]
                phase = "verify_cache"
    if items is None:
        helper, report = root / "resolve-packages.py", root / "packages.json"
        helper.write_text(resolve_packages.SOURCE, encoding="utf-8")
        argv = [python, native(helper), "--report", native(report), "--index-url", "https://pypi.org/simple"]
        if approved_wheels:
            approved = root / 'approved-metadata-wheels.json'
            approved.write_text(json.dumps(approved_wheels), encoding='utf-8')
            argv += ['--modloader-approved-wheels', native(approved)]
        if any(req.startswith("torch") for req in requirements):
            argv += ["--extra-index-url", "https://download.pytorch.org/whl/cu130"]
        if resolver_proxy is None:
            resolver_proxy = transfer.proxy
        if resolver_proxy:
            argv += ["--proxy", resolver_proxy]
        argv += requirements
        await resolve_progress(dict(message="正在读取依赖元数据；安装包下载大小待解析"))
        await command(argv)
        data = json.loads(report.read_text(encoding="utf-8"))
        context["environment"] = data.get("environment")
        items = resolve_packages.package_files(data)

    completed = 0
    label = "正在复用依赖清单并校验本地缓存" if phase == "verify_cache" else "正在读取依赖文件大小"
    await resolve_progress(dict(message=f"{label}（0 / {len(items)}）"))
    semaphore = asyncio.Semaphore(INSTALL_DOWNLOAD_CONCURRENCY)
    async with httpx.AsyncClient(proxy=transfer.proxy or None, trust_env=False, follow_redirects=True, timeout=45) as client:
        async def inspect_file(item):
            nonlocal completed
            async with semaphore:
                _check(transfer.cancel)
                file = await catalog.manifest_file(transfer.paths, item["name"], item["purpose"], item["url"], item["digest"], item.get("size_bytes"),
                                                   package=item["package"], version=item["version"])
                if file["cached"]:
                    file["size_bytes"] = catalog.cache_path(transfer.paths, file).stat().st_size
                elif file["size_bytes"] is None:
                    try:
                        response = await client.head(file["url"])
                        if response.is_success and response.headers.get("content-length", "").isdigit():
                            file["size_bytes"] = int(response.headers["content-length"])
                    except httpx.HTTPError:
                        pass  # Size is optional; the approved SHA256 is mandatory.
                _check(transfer.cancel)
                completed += 1
                await resolve_progress(dict(message=f"{label}（{completed} / {len(items)}）"))
                return file
        tasks = [asyncio.create_task(inspect_file(item)) for item in items]
        try:
            files = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    stage = await transfer.add(dict(id=uuid.uuid4().hex, title="Python 依赖", files=files, resolution=context), confirm)
    artifacts = await transfer.fetch(stage)
    # Exact local wheel paths and no dependencies prohibit a second online resolution.
    from urllib.parse import quote
    wheel_paths = ["file://" + quote(native(path)) if wsl else path.resolve().as_uri() for path in artifacts.values()]
    locked = root / "confirmed-packages.txt"
    locked.write_text("\n".join(f"{path} --hash={item['digest']}" for path, item in zip(wheel_paths, stage["files"])), encoding="utf-8")
    install = [python, "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--no-index", "--no-deps", "--require-hashes", "-r", native(locked)]
    if files:
        if wsl:
            await _wsl_command(install, root, transfer.progress, transfer.cancel)
        else:
            await _command(install, transfer.progress, transfer.cancel, owned_root=root)
    check = [python, "-m", "pip", "--isolated", "check"]
    if wsl:
        await _wsl_command(check, root, transfer.progress, transfer.cancel)
    else:
        await _command(check, transfer.progress, transfer.cancel, owned_root=root)
    return [{k: item[k] for k in ("package", "version", "digest", "url")} for item in stage["files"]]


async def _wsl_command(argv, root, progress, cancel, timeout=3600):
    supervisor = root / "supervise-install.py"
    supervisor.write_text(WSL_SUPERVISOR, encoding="utf-8")
    record = windows_to_wsl(str((root / "install-process.json").resolve()))
    identity = uuid.uuid4().hex
    prefix = ["wsl.exe", "--distribution", "ModLoader", "--exec", "/usr/bin/python3", windows_to_wsl(str(supervisor.resolve()))]
    return await _command(prefix + ["run", record, identity] + argv, progress, cancel,
                          cleanup=prefix + ["stop", record, identity], timeout=timeout, owned_root=root)


_WSL_PROXY_HOST = r'''
import json, socket, struct, sys
from pathlib import Path
host, port = sys.argv[1], int(sys.argv[2])
def reachable(address):
    try:
        with socket.create_connection((address, port), timeout=3): return True
    except OSError: return False
if reachable(host):
    print(json.dumps(host))
    raise SystemExit(0)
for line in Path('/proc/net/route').read_text().splitlines()[1:]:
    fields = line.split()
    if fields[1] == '00000000' and int(fields[3], 16) & 2:
        gateway = socket.inet_ntoa(struct.pack('<I', int(fields[2], 16)))
        if reachable(gateway):
            print(json.dumps(gateway))
            raise SystemExit(0)
raise RuntimeError('WSL 无法连接已配置的 Windows 本机代理（端口 %d）；请检查该代理是否允许 WSL 访问' % port)
'''


async def _wsl_proxy(proxy, root, progress, cancel):
    """Translate a Windows loopback proxy for NAT WSL, keeping the selected port."""
    if not proxy:
        return proxy
    parsed = urlsplit(proxy)
    host = parsed.hostname
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        return proxy
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    result = await _wsl_command(["/usr/bin/python3", "-c", _WSL_PROXY_HOST, host, str(port)], root, progress, cancel, timeout=30)
    gateway = json.loads(result.strip().splitlines()[-1])
    if gateway == host:
        return proxy
    local_addresses = {address.address for entries in psutil.net_if_addrs().values() for address in entries}
    if gateway not in local_addresses:
        raise ValueError("WSL 默认网关不是本机 Windows 地址，未更换代理")
    return parsed._replace(netloc=f"{gateway}:{port}").geturl()


_APT_RESOLVE = r'''
import json, os, pathlib, shlex, subprocess, sys
from urllib.parse import unquote
destination, proxy = sys.argv[1:3]
packages = json.loads(sys.argv[3]) if len(sys.argv) > 3 else ['python3-venv', 'python3-pip', 'libnuma1']
install_options = ['--no-install-recommends'] if len(sys.argv) > 4 and sys.argv[4] == 'no-recommends' else []
os.environ['LC_ALL'] = 'C'
def installed(package):
    result = subprocess.run(['dpkg-query', '-W', '-f=${db:Status-Abbrev} ${Version}', package], text=True, capture_output=True)
    return result.stdout[3:].strip() if result.returncode == 0 and result.stdout.startswith('ii ') else None
missing = [package for package in packages if not installed(package)]
if not missing:
    pathlib.Path(destination).write_text('[]')
    raise SystemExit(0)
options = ['-o', 'Acquire::Retries=0', '-o', 'Acquire::http::Timeout=45', '-o', 'Acquire::https::Timeout=45']
if proxy:
    options += ['-o', 'Acquire::http::Proxy='+proxy, '-o', 'Acquire::https::Proxy='+proxy]
subprocess.run(['apt-get', *options, 'update'], check=True)
text = subprocess.check_output(['apt-get', *options, *install_options, '--print-uris', '--yes', '--download-only', '--no-upgrade', 'install', *missing], text=True)
files = []
for line in text.splitlines():
    if not line.startswith("'"): continue
    url, filename, size, *_ = shlex.split(line)
    package = filename.split('_')[0]
    paragraphs = subprocess.check_output(['apt-cache', 'show', package], text=True).split('\n\n')
    matches = []
    for paragraph in paragraphs:
        fields = dict(row.split(': ', 1) for row in paragraph.splitlines() if ': ' in row and not row.startswith(' '))
        if fields.get('Filename', '').rsplit('/', 1)[-1] == unquote(url.rsplit('/', 1)[-1]): matches.append(fields)
    if not matches or not matches[0].get('SHA256') or len({(v.get('SHA256'),v.get('Version')) for v in matches}) != 1:
        raise RuntimeError('Cannot freeze official apt package SHA256: '+filename)
    data = matches[0]
    previous = installed(data['Package'])
    if previous and previous != data['Version']:
        raise RuntimeError('WSL dependency would upgrade shared package '+data['Package']+'; existing environments retained, installation stopped')
    files.append(dict(name=filename, url=url, size_bytes=int(size), digest='sha256:'+data['SHA256'], package=data['Package'], version=data['Version']))
pathlib.Path(destination).write_text(json.dumps(files))
'''


async def _apt_stage(root, transfer, confirm, *, resolver_proxy=None, packages=None, install_recommends=True):
    if confirm is None:
        raise ValueError("请通过环境安装任务下载和安装 WSL 系统依赖")
    report = root / "apt-packages.json"
    script = root / "resolve-apt.py"
    script.write_text(_APT_RESOLVE, encoding="utf-8")
    async def resolve_progress(event):
        await _emit(transfer.progress, **{**event, "phase": "resolve", "can_pause": False, "total_bytes": None, "percent": None})
    await resolve_progress(dict(message="更新 Ubuntu 软件索引并解析系统依赖，不下载软件包"))
    proxy = transfer.proxy if resolver_proxy is None else resolver_proxy
    if packages is not None and (not isinstance(packages, (list, tuple)) or any(not isinstance(item, str) or not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', item) for item in packages)):
        raise ValueError('无效的系统依赖包名称')
    argv = ["/usr/bin/python3", windows_to_wsl(str(script.resolve())), windows_to_wsl(str(report.resolve())), proxy or ""]
    if packages is not None or not install_recommends:
        argv.append(json.dumps(list(packages) if packages is not None else ['python3-venv', 'python3-pip', 'libnuma1']))
    if not install_recommends:
        argv.append('no-recommends')
    await _wsl_command(argv, root, resolve_progress, transfer.cancel)
    items = json.loads(report.read_text(encoding="utf-8"))
    files = []
    for item in items:
        if Path(item["name"]).name != item["name"] or not re.fullmatch(r"sha256:[a-f0-9]{64}", item["digest"]):
            raise ValueError("无效的 Ubuntu 软件包清单")
        files.append(await catalog.manifest_file(transfer.paths, item["name"], f"WSL 系统依赖 {item['package']} {item['version']}", item["url"], item["digest"], item["size_bytes"], package=item["package"], version=item["version"]))
    if not files:
        return
    stage = await transfer.add(dict(id=uuid.uuid4().hex, title="WSL 系统依赖", files=files), confirm)
    downloaded = await transfer.fetch(stage)
    archive = root / "apt-archives"
    archive.mkdir()
    for path in downloaded.values():
        shutil.copy2(path, archive / path.name)
    argv = ["apt-get", "--yes", "--no-download", *(['--no-install-recommends'] if not install_recommends else []),
            "-o", "Dir::Cache::archives=" + windows_to_wsl(str(archive.resolve())),
            "install", *[f"{item['package']}={item['version']}" for item in stage["files"]]]
    await _wsl_command(argv, root, transfer.progress, transfer.cancel)


async def _help_flags(argv, progress, cancel, **options):
    """Read option names from the whole help stream without retaining its full text."""
    flags = set()
    async def collect(event):
        flags.update(re.findall(r'(?<!\w)--[a-z][a-z0-9-]*', event.get('message') or ''))
        await _emit(progress, **event)
    tail = await _command(argv, collect, cancel, **options)
    flags.update(re.findall(r'(?<!\w)--[a-z][a-z0-9-]*', tail))
    return sorted(flags)


async def validate_runtime(runtime, paths=None, *, cancel=None, progress=None, _installing=False):
    """Read installed versions and verify the service entry point without starting models."""
    cancel = cancel or asyncio.Event()
    outer_progress = progress
    async def verify_progress(event):
        await _emit(outer_progress, **{**event, "phase": "verify", "can_pause": False})
    progress = verify_progress
    if runtime.managed:
        if paths is None:
            raise ValueError("验证托管环境需要应用路径")
        validate_managed_runtime(runtime, paths)
    updated = runtime.model_copy(deep=True)
    backend = runtime.backend
    if backend == "custom":
        from .custom import custom_config
        updated.metadata = {**runtime.metadata, "custom": await asyncio.to_thread(custom_config, runtime), "validation": "files_and_configuration"}
        return updated
    elif backend == "llama_cpp":
        output = await _command([runtime.path, "--version"], progress, cancel, timeout=30)
        match = re.search(r"(?:version:|version)\s*([^\r\n]+)", output)
        if not match:
            raise ValueError("llama-server 未返回可识别的实际版本")
        updated.version = match[1].strip()
        imported = await asyncio.to_thread(import_runtime, backend, runtime.path, runtime.metadata)
        updated.metadata.update(imported.metadata)
    elif backend == "strata":
        imported = await asyncio.to_thread(import_runtime, backend, runtime.path, runtime.metadata)
        updated.version = imported.version
        updated.metadata.update(imported.metadata)
        expected = runtime.metadata.get("provenance", {}).get("engine_version")
        if expected and updated.version != expected:
            raise ValueError("Strata 引擎实际版本与确认发行件不一致")
        if runtime.managed or _installing:
            await _command([updated.metadata["python"], "-m", "serve.server", "--help"], progress, cancel, cwd=runtime.path, timeout=60)
    elif backend in {'ninfer', 'sparkinfer'}:
        from .source_builds import PINS, NINFER_PINS, resolve_pin
        from ..platforms import wsl_prefix
        environment = [key + '=' + value for key, value in runtime.metadata.get('build_environment', {}).items()]
        flags = await _help_flags([*wsl_prefix(runtime_distribution(runtime)), '/usr/bin/env', *environment, runtime.path, '--help'], progress, cancel, timeout=120)
        required = {'--host', '--port', '--spec', '--draft-tokens'} if backend == 'ninfer' else {'--host', '--port', '--ctx', '--tokenizer', '--draft-model'}
        if not required <= set(flags):
            raise ValueError('运行时 --help 未包含此正式适配器所需的服务参数：' + ', '.join(sorted(required - set(flags))))
        updated.metadata['supported_flags'] = flags
        if runtime.managed or _installing:
            pins = NINFER_PINS.values() if backend == 'ninfer' else [PINS[backend]]
            pin = next((item for item in pins if item['commit'] == runtime.metadata.get('source_commit')), None)
            if pin is None:
                raise ValueError('托管原生引擎缺少与已确认配方一致的固定源码身份')
            if backend == 'ninfer' and pin['commit'] != PINS['ninfer']['commit'] and '--context-capacity' not in flags:
                raise ValueError('新版 NInfer --help 未提供共享容量接口 --context-capacity')
            requested = runtime.metadata.get('requested_version')
            if requested and resolve_pin(backend, requested)['commit'] != pin['commit']:
                raise ValueError('原生引擎源码身份与所选安装版本不一致')
            updated.version = pin['version']
            if backend == 'ninfer':
                source, python = runtime.metadata.get('source_root'), runtime.metadata.get('python')
                if not source or not python:
                    raise ValueError('托管 NInfer 缺少官方准备工具的源码目录或独立 Python 环境')
                prefix = [*wsl_prefix(runtime_distribution(runtime)), '/usr/bin/env', '--chdir=' + source, *environment, python]
                for module in ('tools.artifact.inspect', 'tools.convert'):
                    await _command([*prefix, '-m', module, '--help'], progress, cancel, timeout=120)
                updated.metadata['preparation_validation'] = 'tool_entrypoints'
        else:
            updated.version = 'unknown'  # --help establishes the entry point, not a binary version.
        updated.metadata['version_basis'] = 'verified_source_commit' if runtime.managed or _installing else 'external_unknown_no_version_flag'
    else:
        python = runtime.metadata.get("python") or str(Path(runtime.path).parent / "python.exe")
        code = "import importlib.metadata; print(importlib.metadata.version(" + repr(backend) + "))"
        if backend in {"vllm", "sglang"}:
            if backend == 'sglang' and (runtime.managed or _installing):
                linux_root = runtime.metadata.get('linux_root')
                if not linux_root or runtime.metadata.get('build_environment', {}).get('CUDA_HOME') != linux_root + '/cuda':
                    raise ValueError('托管 SGLang 缺少自身独立的 CUDA JIT 工具链；请补充已确认组件后再验证')
            python = runtime.path.rsplit("/", 1)[0] + "/python"
            prefix = ["wsl.exe", "--distribution", runtime_distribution(runtime), "--exec"]
            if backend == 'sglang' and runtime.metadata.get('build_environment'):
                prefix += ['/usr/bin/env', *[key + '=' + value for key, value in runtime.metadata['build_environment'].items()]]
            argv = [*prefix, python, "-c", code]
            help_argv = ([*prefix, python, "-m", "sglang.launch_server", "--help"]
                         if backend == 'sglang' else [*prefix, runtime.path, "serve", "--help"])
        else:
            argv = [python, "-c", code]
            help_argv = [runtime.path, "serve", "--help"]
        updated.version = (await _command(argv, progress, cancel, timeout=60)).strip().splitlines()[-1]
        updated.metadata['supported_flags'] = await _help_flags(help_argv, progress, cancel, timeout=120)
        required = {'--model-path', '--host', '--port', '--served-model-name', '--enable-metrics'} if backend == 'sglang' else set()
        if not required <= set(updated.metadata['supported_flags']):
            raise ValueError('SGLang --help 未包含此适配器所需的本地服务、模型别名或监控参数：' + ', '.join(sorted(required - set(updated.metadata['supported_flags']))))
        expected = runtime.metadata.get("requested_version")
        if expected and updated.version != expected:
            raise ValueError("安装后实际版本与确认版本不一致")
        if backend == 'sglang' and (runtime.managed or _installing):
            output = await _command([*prefix, linux_root + '/cuda/bin/nvcc', '--version'], progress, cancel, timeout=30)
            match = re.search(r'release\s+(\d+)\.(\d+)', output)
            if not match or (int(match[1]), int(match[2])) < (12, 9):
                raise ValueError('SGLang DeepGEMM JIT 需要实际可用的 NVCC >= 12.9')
            updated.metadata['toolchain_validation'] = 'nvcc'
    updated.name = runtime.name.rsplit(" · ", 1)[0] + " · " + updated.version
    updated.metadata["validation"] = "files" if backend == "strata" and not (runtime.managed or _installing) else "entrypoint"
    return updated


def _verify_wsl_owner(paths: AppPaths):
    """A stale marker must not authorize modifying a replacement distro of the same name."""
    import winreg
    marker = paths.runtimes / "ModLoader-wsl-owned.json"
    expected = (paths.runtimes / "wsl-ModLoader").resolve()
    if not marker.is_file():
        raise ValueError("缺少专属 WSL 发行版所有权记录")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("distribution") != "ModLoader" or Path(record.get("location", "")).resolve() != expected:
        raise ValueError("WSL 所有权记录路径不匹配")
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Lxss") as registry:
        for index in range(winreg.QueryInfoKey(registry)[0]):
            with winreg.OpenKey(registry, winreg.EnumKey(registry, index)) as distribution:
                name = winreg.QueryValueEx(distribution, "DistributionName")[0]
                if name == "ModLoader":
                    actual = winreg.QueryValueEx(distribution, "BasePath")[0]
                    if Path(str(actual).removeprefix("\\\\?\\")).resolve() != expected:
                        raise ValueError("当前 ModLoader WSL 注册路径不同，拒绝修改")
                    return
    raise ValueError("找不到已登记的专属 ModLoader WSL 发行版")


async def install_backend(backend: str, paths: AppPaths, options: dict, progress, cancel: asyncio.Event, pause: DownloadPause | None = None, *, confirm=None, proxy=None) -> RuntimeInstallation:
    """Execute exactly a server-owned preview; only confirmed artifacts are downloaded."""
    _check(cancel)
    if backend not in VERSIONS:
        raise ValueError("未知后端")
    if options.get("import_path"):
        return await asyncio.to_thread(import_runtime, backend, options["import_path"], options.get("metadata"))
    plan = options.get("_plan")
    if not isinstance(plan, dict) or plan.get("backend") != backend:
        raise ValueError("安装清单已失效，请重新开始安装")
    if os.name != "nt":
        raise ValueError("这些环境安装配方面向 Windows；其他系统可导入已安装运行时")
    plan = copy.deepcopy(plan)
    version, variant = plan["version"], plan["variant"]
    if not re.fullmatch(r"[A-Za-z0-9.+_-]+", version):
        raise ValueError("无效安装版本")
    if backend == 'sglang':
        from .sglang_versions import recipe
        sglang_recipe = recipe(version)
    wsl_present = False
    if backend in {"vllm", "sglang", "ninfer", "sparkinfer"}:
        # Check ownership before transferring a rootfs that an existing foreign distro cannot use.
        listing = await asyncio.to_thread(list_wsl)
        if not listing['available']:
            raise RuntimeError("WSL2 不可用，请先启用 Windows WSL2 功能")
        registered = next((item for item in listing['distributions'] if item['name'] == 'ModLoader'), None)
        wsl_present = registered is not None
        if registered and registered['version'] != 2:
            raise ValueError('已存在的 ModLoader 发行版不是 WSL2；未修改或重新导入该发行版')
        if wsl_present:
            _verify_wsl_owner(paths)
        _check(cancel)
    root = paths.runtimes / f"{backend}-{version}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    transfer = _Transfers(paths, progress, cancel, pause, proxy)
    stage = await transfer.add(plan["stage"])
    artifacts = await transfer.fetch(stage)
    meta = {"version": version, "variant": variant, "provenance": {k: v for k, v in plan.get("provenance", {}).items() if k != "source_tree"}, "plan_id": plan["id"]}
    if 'sources' in meta['provenance']:
        meta['provenance']['sources'] = [{key: value for key, value in item.items() if key not in {'tree', 'submodules'}} for item in meta['provenance']['sources']]
    selected = plan.get("options", {})
    if backend == "llama_cpp":
        await _unpack(artifacts["engine"], root, progress, cancel)
        if "cuda" in artifacts:
            await _unpack(artifacts["cuda"], root, progress, cancel)
        candidates = list(root.rglob("llama-server.exe"))
        if len(candidates) != 1:
            raise ValueError("发行件中找不到唯一 llama-server.exe")
        for dll in root.rglob("*.dll"):
            destination = candidates[0].parent / dll.name
            if dll != destination and not destination.exists():
                shutil.copy2(dll, destination)
        runtime = await asyncio.to_thread(import_runtime, backend, str(candidates[0]), meta)
    elif backend == "strata":
        await _unpack(artifacts["source"], root, progress, cancel)
        checkouts = list(root.glob("Strata-*"))
        if len(checkouts) != 1:
            raise ValueError("源码包目录不唯一")
        checkout = checkouts[0]
        await _thread(_verify_source, checkout, plan["provenance"]["source_tree"], cancel, cancel=cancel)
        from ..strata_patches import apply_server_patch
        if await asyncio.to_thread(apply_server_patch, checkout, plan['provenance']['source_commit']):
            meta['raw_token_continuation_server'] = True
        deepseek = variant == "dsv41-cuda13"
        await _unpack(artifacts["engine"], checkout / ("engine-dsv41" if deepseek else "engine"), progress, cancel)
        if deepseek:
            from ..strata_deepseek import engine_path
            engine_path(checkout)
            meta["strata_family"] = "deepseek41"
        python = await _approved_venv(checkout / ".venv", selected, artifacts, progress, cancel)
        requirements = ["-r", str(checkout / "requirements.txt")]
        if artifacts["engine"].name.endswith(".zip"):
            requirements += ["nvidia-cublas==13.0.2.14", "nvidia-cuda-runtime==13.0.96"]
        else:
            meta["engine_backend"] = "hip" if variant.startswith("hip-") else "cuda"
        meta["packages"] = await _package_stage(python, requirements, root, transfer, confirm, retry=plan.get("_retry_packages"))
        meta.update(python=python, source_commit=plan["provenance"]["source_commit"])
        runtime = await asyncio.to_thread(import_runtime, backend, str(checkout), meta)
    elif backend == "transformers":
        python = await _approved_venv(root / "venv", selected, artifacts, progress, cancel)
        meta["packages"] = await _package_stage(python, [f"transformers[serving]=={version}", "torch==2.14.1", "accelerate==1.12.0"], root, transfer, confirm, retry=plan.get("_retry_packages"))
        meta["python"] = python
        runtime = await asyncio.to_thread(import_runtime, backend, str(Path(python).parent / "transformers.exe"), meta)
    else:
        ownership = paths.runtimes / "ModLoader-wsl-owned.json"
        if wsl_present:
            _verify_wsl_owner(paths)
        else:
            if "rootfs" not in artifacts:
                raise ValueError("专属 WSL 发行版已变化，请重新预览基础文件清单")
            distro_dir = paths.runtimes / "wsl-ModLoader"
            await _command(["wsl.exe", "--import", "ModLoader", str(distro_dir), str(artifacts["rootfs"]), "--version", "2"], progress, cancel, timeout=600, owned_root=root)
            ownership.write_text(json.dumps({"distribution": "ModLoader", "location": str(distro_dir), "rootfs": catalog.UBUNTU_URL}), encoding="utf-8")
        _verify_wsl_owner(paths)
        linux_root = "/opt/modloader/" + root.name
        resolver_proxy = await _wsl_proxy(proxy, root, progress, cancel)
        if backend in {'ninfer', 'sparkinfer'}:
            from .source_builds import install as source_install, resolve_pin
            meta.update(await source_install(backend, root, linux_root, artifacts, plan, transfer, confirm, resolver_proxy))
            entrypoint = '/' + resolve_pin(backend, version)['entrypoint']
        else:
            await _apt_stage(root, transfer, confirm, resolver_proxy=resolver_proxy)
            await _wsl_command(["/usr/bin/python3", "-m", "venv", linux_root], root, progress, cancel)
            if backend == 'sglang':
                from .sglang import stage_toolchain
                meta.update(await stage_toolchain(root, linux_root, artifacts, progress, cancel, version=version))
            requirements = list(sglang_recipe['requirements']) if backend == 'sglang' else [f"vllm=={version}"]
            approved_wheels = None
            if backend == 'sglang' and sglang_recipe['approved_wheels']:
                from urllib.parse import quote
                approved_wheels = []
                for item in sglang_recipe['approved_wheels']:
                    if item['role'] not in artifacts:
                        raise ValueError('SGLang 安装清单缺少 ' + item['name'] + '，请重新预览安装')
                    wheel = windows_to_wsl(str(artifacts[item['role']].resolve()))
                    approved_wheels.append(dict(path=wheel, url=item['url'], digest=item['digest']))
                    requirements += [item['package'] + ' @ file://' + quote(wheel) + '#sha256=' + item['digest'].removeprefix('sha256:')]
            meta["packages"] = await _package_stage(linux_root + "/bin/python", requirements, root, transfer, confirm, wsl=True, resolver_proxy=resolver_proxy, retry=plan.get("_retry_packages"), approved_wheels=approved_wheels)
            entrypoint = '/bin/python' if backend == 'sglang' else '/bin/vllm'
        runtime = import_runtime(backend, linux_root + entrypoint, {**meta, "distribution": "ModLoader", "distribution_owned": True, "linux_root": linux_root})
    _check(cancel)
    runtime.metadata["requested_version"] = version
    runtime.metadata["install_root"] = str(root)
    runtime.metadata["managed_root"] = str(root)
    runtime = await validate_runtime(runtime, paths=paths, cancel=cancel, progress=progress, _installing=True)
    runtime.managed = True
    if backend == 'strata' and plan.get('options', {}).get('startup_expert_prefetch') is True:
        from ..strata_prefetch import prepare_runtime
        runtime = await prepare_runtime(runtime, paths, progress, cancel, installing=True, proxy=proxy)
    (root / "modloader-runtime.json").write_text(runtime.model_dump_json(indent=2), encoding="utf-8")
    await _emit(progress, phase="complete", message=f"{runtime.name} 安装完成", percent=100, can_pause=False)
    return runtime


def validate_managed_runtime(runtime: RuntimeInstallation, paths: AppPaths) -> Path:
    """Verify on-disk ownership without modifying or removing the runtime."""
    if not runtime.managed or not runtime.metadata.get("managed_root"):
        raise ValueError("外部导入的运行时只能移除记录，不能删除原文件")
    root = Path(runtime.metadata["managed_root"]).resolve()
    base = paths.runtimes.resolve()
    if root == base or not root.is_relative_to(base) or root.parent != base:
        raise ValueError("拒绝删除托管运行时目录之外的路径")
    manifest = root / "modloader-runtime.json"
    if not manifest.is_file():
        raise ValueError("运行时所有权清单缺失，拒绝删除")
    recorded = RuntimeInstallation.model_validate_json(manifest.read_text(encoding="utf-8"))
    if recorded.id != runtime.id or recorded.path != runtime.path or recorded.backend != runtime.backend or not recorded.managed or Path(recorded.metadata.get("managed_root", "")).resolve() != root:
        raise ValueError("运行时所有权与清单不匹配")
    if runtime.backend in {"vllm", "sglang", "ninfer", "sparkinfer"}:
        if runtime.metadata.get("distribution") != "ModLoader" or not runtime.metadata.get("distribution_owned"):
            raise ValueError("不能修改非托管 WSL 发行版")
        _verify_wsl_owner(paths)
        linux_root = runtime.metadata.get("linux_root")
        entrypoint = '/bin/python' if runtime.backend == 'sglang' else '/bin/vllm'
        if runtime.backend in {'ninfer', 'sparkinfer'}:
            from .source_builds import PINS
            entrypoint = '/' + PINS[runtime.backend]['entrypoint']
        if linux_root != "/opt/modloader/" + root.name or runtime.path != linux_root + entrypoint:
            raise ValueError("WSL 运行时目录与所有权不匹配")
    elif not Path(runtime.path).resolve().is_relative_to(root):
        raise ValueError("运行时可执行路径不属于托管目录")
    return root


async def uninstall_runtime(runtime: RuntimeInstallation, paths: AppPaths, progress=None, cancel: asyncio.Event | None = None) -> None:
    """Remove only this managed installation. Caller must stop/reject active instances first.

    A shared WSL distribution is retained: other imported venvs can still use it.
    External runtimes are only removable from the caller's registry, never deleted here.
    """
    cancel = cancel or asyncio.Event()
    _check(cancel)
    root = validate_managed_runtime(runtime, paths)
    if runtime.backend in {"vllm", "sglang", "ninfer", "sparkinfer"}:
        linux_root = runtime.metadata["linux_root"]
        # Pass the verified path as a separate argv; no path interpolation into shell code.
        code = "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); assert p.parent==pathlib.Path('/opt/modloader') and p.resolve()==p; shutil.rmtree(p)"
        await _command(["wsl.exe", "--distribution", "ModLoader", "--exec", "/usr/bin/python3", "-c", code, linux_root], progress, cancel, timeout=120, owned_root=root)
    _check(cancel)
    await asyncio.to_thread(shutil.rmtree, root)
    await _emit(progress, phase="complete", message="已删除此托管运行时；外部模型与其他运行时保持不变", percent=100)
