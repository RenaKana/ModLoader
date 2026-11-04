from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Awaitable, Callable

from ..models import AppPaths, DownloadRequest, SourceFile, SourceRepository
from ._common import NETWORK_ATTEMPTS, NETWORK_TIMEOUT, WORKER_IDLE_TIMEOUT, SourceError, destination_path, file_target, relative_path, repository_id, revision_name, source_name
from .http_download import download_url, inspect_url
from .control import DownloadPause
from .planning import build_plan, classify_repository


__all__ = ["SourceService", "SourceError"]
Progress = Callable[[dict], Awaitable[None]]


async def _with_cancel(operation: Awaitable, cancel: asyncio.Event):
    if cancel.is_set():
        if hasattr(operation, "close"):
            operation.close()
        raise asyncio.CancelledError
    task = asyncio.create_task(operation)
    watcher = asyncio.create_task(cancel.wait())
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done and cancel.is_set():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise asyncio.CancelledError
        return await task
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        if not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await task


class SourceService:
    """Official Hub SDKs in isolated workers, with app-owned cache and auth."""

    def __init__(self, paths: AppPaths, credential_provider: Callable[[str], str | None], proxy_provider: Callable[[], str | None], models_root: Path | None = None):
        self.paths = paths
        self.credential_provider = credential_provider
        self.proxy_provider = proxy_provider
        self.models_root = (models_root or paths.models).resolve()
        self.cache = self.paths.root / "cache" / "sources"

    def _environment(self, source: str) -> dict[str, str]:
        env = os.environ.copy()
        for key in list(env):
            if key.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MODELSCOPE_API_TOKEN", "HF_ENDPOINT", "MODELSCOPE_ENDPOINT", "MODELSCOPE_DOMAIN"}:
                env.pop(key, None)
        proxy = self.proxy_provider() or ""
        token = self.credential_provider(source) or ""
        env.update({
            "MODLOADER_SOURCE_TOKEN": token, "MODLOADER_SOURCE_PROXY": proxy,
            "HF_HOME": str(self.cache / "hf"), "HF_HUB_CACHE": str(self.cache / "hf" / "hub"),
            "HF_XET_CACHE": str(self.cache / "hf" / "xet"),
            "HF_TOKEN_PATH": str(self.cache / "hf" / "unused-token"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_ETAG_TIMEOUT": str(int(NETWORK_TIMEOUT)), "HF_HUB_DOWNLOAD_TIMEOUT": str(int(NETWORK_TIMEOUT)),
            "MODELSCOPE_HOME": str(self.cache / "modelscope" / "settings"),
            "MODELSCOPE_CACHE": str(self.cache / "modelscope" / "hub"),
            "MODELSCOPE_API_TOKEN": "", "MODELSCOPE_DOWNLOAD_INTRA_CLOUD": "0",
            "MODELSCOPE_DOWNLOAD_MAX_RETRIES": str(NETWORK_ATTEMPTS - 1),
            "MODELSCOPE_DOWNLOAD_TIMEOUT": str(int(NETWORK_TIMEOUT)),
            "MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS": "1", "MODELSCOPE_LOG_LEVEL": "ERROR",
            "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        })
        if proxy:
            env["HTTP_PROXY"] = proxy
            env["HTTPS_PROXY"] = proxy
        core = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = core + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    async def _worker(self, payload: dict, source: str, progress: Progress | None = None, pause: DownloadPause | None = None) -> dict:
        payload = {**payload, "source": source, "cache": str(self.cache), "models_root": str(self.models_root)}
        environment = self._environment(source)
        if pause is not None:
            payload['pausable'] = True
            # Native Xet reconstruction writes outside Python's cooperative
            # stream checkpoints. Keep pausable jobs on the SDK HTTP path.
            environment['HF_HUB_DISABLE_XET'] = '1'
        argv = [sys.executable, "--source-worker"] if getattr(sys, "frozen", False) else [sys.executable, "-m", "modloader.sources.worker"]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        result: dict | None = None
        commands = None
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
            await process.stdin.drain()
            if pause is None:
                process.stdin.close()
            else:
                async def control():
                    sent = 0
                    while True:
                        if pause.generation == sent:
                            pause.changed.clear()
                            await pause.changed.wait()
                        sent = pause.generation
                        message = {'command': 'pause' if pause.requested else 'resume', 'generation': sent}
                        process.stdin.write(json.dumps(message).encode('utf-8') + b'\n')
                        await process.stdin.drain()
                commands = asyncio.create_task(control())
            buffer = bytearray()
            while True:
                try:
                    chunk = await asyncio.wait_for(process.stdout.read(64 * 1024), timeout=WORKER_IDLE_TIMEOUT)
                except asyncio.TimeoutError:
                    if pause is not None and pause.acknowledged:
                        continue
                    raise SourceError("来源任务长时间没有响应，已停止；请检查网络后重试。", "source_timeout") from None
                if not chunk:
                    break
                buffer.extend(chunk)
                while (index := buffer.find(b"\n")) != -1:
                    line = bytes(buffer[:index])
                    del buffer[: index + 1]
                    try:
                        message = json.loads(line)
                    except ValueError:
                        raise SourceError("来源 SDK 返回了无效响应。", "worker_error") from None
                    if message.get("type") == "progress" and progress:
                        await progress(message["data"])
                    elif message.get('type') == 'paused' and pause is not None:
                        await pause.acknowledge(message.get('generation', -1))
                    elif message.get("type") == "error":
                        raise SourceError(message.get("message", "来源任务失败。"), message.get("code", "source_error"))
                    elif message.get("type") == "result":
                        result = message["data"]
                        # Let the control reader leave stdin before Python
                        # finalizes its buffered input stream.
                        if pause is not None:
                            process.stdin.close()
            status = await process.wait()
            if status != 0 or result is None:
                raise SourceError("来源 SDK 子进程未完成任务。", "worker_error")
            return result
        finally:
            if commands:
                commands.cancel()
                await asyncio.gather(commands, return_exceptions=True)
            if process.stdin:
                process.stdin.close()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

    async def search(self, source: str, query: str, cursor: str | None = None) -> dict:
        source = source_name(source)
        if source == "url":
            repository = await self.resolve(source, query)
            return {"items": [{"id": repository.url, "name": repository.name, "description": "HTTP(S) 文件直链", "url": repository.url}], "next_cursor": None}
        return await self._worker({"action": "search", "query": query.strip(), "cursor": cursor}, source)

    async def resolve(self, source: str, repo_id: str, revision: str = "main", *, progress: Callable[[dict], None] | None = None) -> SourceRepository:
        source = source_name(source)
        if source == "url":
            if progress:
                progress({"stage": "file_info", "message": "检查链接与文件大小"})
            info = await inspect_url(repo_id, proxy=self.proxy_provider())
            return SourceRepository(source="url", repo_id="", revision="", name=info["path"], url=info["url"], files=[SourceFile(path=info["path"], size=info["size"])])
        if progress:
            progress({"stage": "repository", "message": f"读取仓库清单 · {repo_id}"})
        data = await self._worker({"action": "resolve", "repo_id": repository_id(repo_id, source), "revision": revision_name(revision)}, source)
        repository = SourceRepository.model_validate(data)

        async def read_json(path: str) -> dict:
            return await self._worker({"action": "read_json", "repo_id": repository.repo_id, "revision": repository.revision, "file": path}, source)

        return await classify_repository(repository, read_json, progress=progress)

    async def readme(self, source: str, repo_id: str, revision: str = "main", file_path: str | None = None,
                     *, progress: Callable[[dict], None] | None = None) -> dict:
        source = source_name(source)
        if source == "url":
            raise SourceError("直链来源没有仓库 README，请查看原始页面。", "unsupported_source")
        path = relative_path(file_path) if file_path is not None else None
        if progress:
            progress({"stage": "readme", "message": f"读取 {path or 'README'} · {repo_id}"})
        return await self._worker({"action": "readme", "repo_id": repository_id(repo_id, source), "revision": revision_name(revision), "file_path": path}, source)

    async def plan(self, request: DownloadRequest, *, progress: Callable[[dict], None] | None = None) -> dict:
        source = source_name(request.source)
        if request.destination:
            destination_path(self.models_root, request.destination)
        if source == "url":
            if progress:
                progress({"stage": "file_info", "message": "检查链接与文件大小"})
            info = await inspect_url(request.url or request.repo_id, request.filename, self.proxy_provider())
            return {"source": "url", "repo_id": "", "revision": "", "files": [{"path": info["path"], "size": info["size"], "role": "other", "reason": "直接链接文件"}], "total_bytes": info["size"] or 0, "unknown_size_files": [info["path"]] if info["size"] is None else [], "warnings": [], "groups": [], "model_groups": [], "automatic_files": [], "optional_files": [], "url": info["url"]}
        feedback = {"progress": progress} if progress is not None else {}
        repository = await self.resolve(source, request.repo_id, request.revision, **feedback)

        async def read_json(path: str) -> dict:
            return await self._worker({"action": "read_json", "repo_id": repository.repo_id, "revision": repository.revision, "file": path}, source)

        return await build_plan(request, repository, read_json, progress=progress)

    async def download(self, request: DownloadRequest, progress: Progress, cancel: asyncio.Event, pause: DownloadPause | None = None) -> dict:
        source = source_name(request.source)
        destination = destination_path(self.models_root, request.destination)
        await progress({"phase": "planning", "message": "正在准备下载清单", "downloaded_bytes": 0, "total_bytes": None, "can_pause": False})
        if source == "url":
            if request.checked_files is not None:
                plan = await self.plan(request)
                if {relative_path(file.path): file.size for file in request.checked_files} != {file["path"]: file["size"] for file in plan["files"]}:
                    raise SourceError("下载文件已变化，请重新检查下载。", "resource_changed")
            return await _with_cancel(download_url(request.url or request.repo_id, request.filename, destination, progress, cancel, self.proxy_provider(), pause=pause), cancel)

        async def operation() -> dict:
            plan = await self.plan(request)
            if plan.get("requires_vision_selection"):
                raise SourceError("请先选择匹配的视觉组件，再检查下载。", "vision_selection_required")
            if request.checked_files is not None:
                approved = {relative_path(file.path): file.size for file in request.checked_files}
                current = {file["path"]: file["size"] for file in plan["files"]}
                if request.revision != plan["revision"] or approved != current:
                    raise SourceError("仓库文件已变化，请重新检查下载。", "resource_changed")
            for file in plan["files"]:
                file_target(destination, file["path"])
            destination.mkdir(parents=True, exist_ok=True)
            controls = {'pause': pause} if pause is not None else {}
            result = await self._worker({"action": "download", "repo_id": plan["repo_id"], "revision": plan["revision"], "files": plan["files"], "destination": str(destination)}, source, progress, **controls)
            result["warnings"] = plan["warnings"]
            result["plan_files"] = plan["files"]
            for key in ("model_groups", "vision_groups", "required_vision_groups", "vision_candidates", "vision_assignments"):
                result[key] = plan.get(key, [] if key != "vision_assignments" else {})
            return result

        return await _with_cancel(operation(), cancel)
