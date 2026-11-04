"""One-shot SDK worker. Its stdin and environment carry private request data."""
from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ._common import NETWORK_ATTEMPTS, NETWORK_TIMEOUT, SEARCH_PAGE_SIZE, SourceError, destination_path, error_message, file_target, relative_path, repository_id, revision_name
from .control import PAUSE_CHUNK_SIZE


_OUTPUT = sys.stdout
_OUTPUT_LOCK = threading.Lock()


class _DownloadControl:
    """Only the thread consuming SDK response bytes acknowledges a pause."""
    def __init__(self):
        self.condition = threading.Condition()
        self.requested = False
        self.generation = 0

    def listen(self, stream):
        try:
            for line in stream:
                try:
                    message = json.loads(line)
                    command, generation = message['command'], int(message['generation'])
                except (ValueError, TypeError, KeyError):
                    continue
                if command not in ('pause', 'resume'):
                    continue
                with self.condition:
                    if generation <= self.generation:
                        continue
                    self.generation = generation
                    self.requested = command == 'pause'
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.requested = False
                self.condition.notify_all()

    def checkpoint(self):
        with self.condition:
            announced = -1
            while self.requested:
                if announced != self.generation:
                    emit('paused', generation=self.generation)
                    announced = self.generation
                self.condition.wait()

    def stream(self, chunks):
        emit('progress', {'can_pause': True})
        try:
            iterator = iter(chunks)
            while True:
                self.checkpoint()
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                self.checkpoint()
                yield chunk
        finally:
            emit('progress', {'can_pause': False})


def emit(kind: str, data: dict | None = None, **fields) -> None:
    message = {"type": kind, **fields}
    if data is not None:
        message["data"] = data
    with _OUTPUT_LOCK:
        # Frozen Windows workers may use GBK stdout despite PYTHONUTF8.
        # ASCII JSON escapes keep the pipe protocol independent of that codec.
        _OUTPUT.write(json.dumps(message, ensure_ascii=True) + "\n")
        _OUTPUT.flush()


def _page(cursor: str | None, default: int) -> int:
    try:
        result = int(cursor) if cursor else default
    except (ValueError, TypeError):
        raise SourceError("搜索分页标记无效。", "invalid_cursor") from None
    if result < default:
        raise SourceError("搜索分页标记无效。", "invalid_cursor")
    return result


class _HuggingFace:
    def __init__(self, cache: Path):
        import httpx
        from huggingface_hub import HfApi, set_client_factory

        self.token = os.environ.get("MODLOADER_SOURCE_TOKEN") or False
        self.cache = cache / "hf" / "hub"
        self.requests: dict[tuple[str, str], int] | None = None
        self.control: _DownloadControl | None = None

        def request_budget(request: httpx.Request) -> None:
            if self.requests is None:
                return
            parsed = urlsplit(str(request.url))
            key = request.method, urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
            self.requests[key] = self.requests.get(key, 0) + 1
            if self.requests[key] > NETWORK_ATTEMPTS:
                raise SourceError("来源重试次数已用尽，可再次启动任务继续下载。", "retry_exhausted")

        def download_stream(response: httpx.Response) -> None:
            if self.control is not None and response.request.method == 'GET':
                original = response.iter_bytes
                response.iter_bytes = lambda chunk_size=None: self.control.stream(original(chunk_size=PAUSE_CHUNK_SIZE))

        set_client_factory(lambda: httpx.Client(
            proxy=os.environ.get("MODLOADER_SOURCE_PROXY") or None,
            trust_env=False, follow_redirects=True, timeout=NETWORK_TIMEOUT,
            event_hooks={"request": [request_budget], 'response': [download_stream]},
        ))
        self.api = HfApi(endpoint="https://huggingface.co", token=self.token)

    def search(self, query: str, cursor: str | None) -> dict:
        offset = _page(cursor, 0)
        # The official iterator follows Hub pagination. Its public API does
        # not accept a cursor, so an offset resumes that iterator without a
        # fixed cap on the number of search results.
        models = list(itertools.islice(self.api.list_models(search=query or None, sort="downloads", limit=offset + SEARCH_PAGE_SIZE + 1, token=self.token), offset, offset + SEARCH_PAGE_SIZE + 1))
        items = [{
            "id": model.id, "name": model.id.rsplit("/", 1)[-1],
            "description": getattr(model, "pipeline_tag", None) or "",
            "url": f"https://huggingface.co/{model.id}",
            "downloads": getattr(model, "downloads", None),
            "gated": getattr(model, "gated", False),
        } for model in models[:SEARCH_PAGE_SIZE]]
        return {"items": items, "next_cursor": str(offset + SEARCH_PAGE_SIZE) if len(models) > SEARCH_PAGE_SIZE else None}

    def resolve(self, repo_id: str, revision: str) -> dict:
        info = self.api.model_info(repo_id, revision=revision, files_metadata=True, timeout=NETWORK_TIMEOUT, token=self.token)
        files, rejected = [], 0
        for item in info.siblings or []:
            try:
                path = relative_path(item.rfilename)
            except SourceError:
                rejected += 1
                continue
            files.append({"path": path, "size": item.size})
        metadata = {"requested_revision": revision, "gated": info.gated, "private": info.private, "pipeline_tag": info.pipeline_tag, "tags": info.tags or [], "revision_is_commit": bool(info.sha), "warnings": []}
        if rejected:
            metadata["warnings"].append(f"已排除 {rejected} 个不能安全保存的仓库路径。")
        return {"source": "hf", "repo_id": repo_id, "revision": info.sha or revision, "name": repo_id.rsplit("/", 1)[-1], "description": info.pipeline_tag or "", "url": f"https://huggingface.co/{repo_id}", "files": files, "metadata": metadata}

    def get_file(self, repo_id: str, revision: str, path: str, local: Path | None = None, tqdm_class=None) -> Path:
        from huggingface_hub import hf_hub_download

        self.requests = {}
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=relative_path(path), revision=revision, token=self.token, cache_dir=self.cache, local_dir=local, endpoint="https://huggingface.co", etag_timeout=NETWORK_TIMEOUT, tqdm_class=tqdm_class))
        finally:
            self.requests = None


class _ModelScope:
    def __init__(self, cache: Path):
        import requests
        from modelscope_hub import HubApi, HubConfig
        from modelscope_hub._legacy_api import LegacyClient
        from modelscope_hub._openapi import OpenAPIClient

        self.cache = cache / "modelscope" / "hub"
        self.control: _DownloadControl | None = None
        token = os.environ.get("MODLOADER_SOURCE_TOKEN") or ""
        config = HubConfig(endpoint="https://modelscope.cn", token=token, cache_dir=self.cache, config_dir=cache / "modelscope" / "settings")
        proxy = os.environ.get("MODLOADER_SOURCE_PROXY")

        def session() -> requests.Session:
            result = requests.Session()
            result.trust_env = False  # includes implicit .netrc credentials
            if proxy:
                result.proxies.update({"http": proxy, "https": proxy})
            return result

        self.api = HubApi(config)
        # ModelScope's facade has no proxy/timeout constructor arguments. Its
        # documented composed clients accept a Session and retry settings.
        self.api._openapi = OpenAPIClient(config, session=session(), timeout=NETWORK_TIMEOUT, max_retries=NETWORK_ATTEMPTS - 1)
        self.api._legacy = LegacyClient(token=token, endpoint=config.endpoint, timeout=int(NETWORK_TIMEOUT), max_retries=NETWORK_ATTEMPTS - 1)
        self.api._legacy._session.trust_env = False
        if proxy:
            self.api._legacy._session.proxies.update({"http": proxy, "https": proxy})
        original_stream = self.api._legacy.download_stream

        def download_stream(*args, **kwargs):
            response = original_stream(*args, **kwargs)
            if self.control is not None:
                original = response.iter_content
                response.iter_content = lambda chunk_size=1, decode_unicode=False: self.control.stream(original(chunk_size=PAUSE_CHUNK_SIZE, decode_unicode=decode_unicode))
            return response

        self.api._legacy.download_stream = download_stream

    def search(self, query: str, cursor: str | None) -> dict:
        page = _page(cursor, 1)
        data = self.api.list_repos("model", search=query or None, sort="downloads", page_number=page, page_size=SEARCH_PAGE_SIZE)
        items = [{"id": item.repo_id, "name": item.display_name or item.name or item.repo_id, "description": item.description or "", "url": f"https://modelscope.cn/models/{item.repo_id}", "downloads": item.downloads, "gated": item.gated or item.login_required or False} for item in data.items]
        has_more = page * SEARCH_PAGE_SIZE < data.total_count if data.total_count else len(items) == SEARCH_PAGE_SIZE
        return {"items": items, "next_cursor": str(page + 1) if has_more else None}

    def resolve(self, repo_id: str, revision: str) -> dict:
        requested = revision
        warnings: list[str] = []
        if revision == "main":
            refs = self.api.list_repo_revisions(repo_id, "model")
            names = {ref.get("Revision") or ref.get("name") if isinstance(ref, dict) else ref for ref in refs if isinstance(ref, (dict, str))}
            if "main" not in names and "master" in names:
                revision = "master"
                warnings.append("此 ModelScope 仓库默认分支为 master，已使用该分支。")
        info = self.api.get_repo(repo_id, "model", revision=revision)
        files, rejected = [], 0
        for item in self.api.list_repo_files(repo_id, "model", revision=revision, recursive=True):
            if item.is_dir:
                continue
            try:
                path = relative_path(item.path)
            except SourceError:
                rejected += 1
                continue
            files.append({"path": path, "size": item.size})
        if rejected:
            warnings.append(f"已排除 {rejected} 个不能安全保存的仓库路径。")
        return {"source": "modelscope", "repo_id": repo_id, "revision": revision, "name": info.display_name or info.name or repo_id, "description": info.description or "", "url": f"https://modelscope.cn/models/{repo_id}", "files": files, "metadata": {"requested_revision": requested, "revision_is_commit": False, "gated": info.gated or info.login_required or False, "warnings": warnings}}

    def get_file(self, repo_id: str, revision: str, path: str, local: Path | None = None, expected_sha256: str | None = None, **_) -> Path:
        return self.api.download_file(repo_id=repo_id, repo_type="model", file_path=relative_path(path), revision=revision, cache_dir=self.cache, local_dir=local, expected_sha256=expected_sha256)


def _download(client, request: dict, control: _DownloadControl | None = None) -> dict:
    from tqdm.auto import tqdm

    destination = destination_path(Path(request["models_root"]), request["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    files = request["files"]
    for item in files:
        if control:
            control.checkpoint()
        file_target(destination, item["path"])
    total = sum(item["size"] or 0 for item in files) if all(item["size"] is not None for item in files) else None
    completed = 0
    hashes: dict[str, str | None] = {}
    if isinstance(client, _ModelScope):
        current = {item.path: item for item in client.api.list_repo_files(request["repo_id"], "model", revision=request["revision"], recursive=True) if not item.is_dir}
        for item in files:
            remote = current.get(item["path"])
            if remote is None or (item["size"] is not None and item["size"] != remote.size):
                raise SourceError("仓库文件在规划后发生变化，请重新规划下载。", "resource_changed")
            hashes[item["path"]] = remote.sha256

    for item in files:
        path = item["path"]
        expected = item["size"]
        target = file_target(destination, path)
        file_target(destination, path + ".incomplete")
        if isinstance(client, _HuggingFace):
            file_target(destination, ".cache/huggingface/download/" + path)

        def report(size: int, phase: str = "downloading", transferred: int | None = None) -> None:
            message = f"正在校验 {path}" if phase == "verifying" else f"已下载 {path}" if phase == "file_complete" else f"正在下载 {path}"
            if transferred is not None and phase == "downloading":
                message += f"（网络已接收 {transferred / (1024 * 1024):.1f} MiB，已重建 {size / (1024 * 1024):.1f} MiB）"
            emit("progress", {"phase": phase, "message": message, "file": path, "file_downloaded_bytes": size, "file_transferred_bytes": transferred, "file_total_bytes": expected, "downloaded_bytes": completed + size, "total_bytes": total})

        class FileProgress(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.pop("name", None)
                kwargs["disable"] = True
                self.downloaded = kwargs.get("initial", 0)
                self.transferred = 0
                self.has_transfer = False
                self.last_emit = 0.0
                self.progress_closed = False
                super().__init__(*args, **kwargs)
                report(self.downloaded)

            def publish(self, force=False):
                if force or time.monotonic() - self.last_emit >= 0.15:
                    report(self.downloaded, transferred=self.transferred if self.has_transfer else None)
                    self.last_emit = time.monotonic()

            def update(self, size=1):
                self.downloaded += max(0, size)
                self.n = self.downloaded
                self.publish()
                return super().update(size)

            def update_transfer(self, size=1):
                # HF 1.33's Xet reporter recognizes this method and routes both
                # counters through one bar. Transfer bytes are compressed/
                # deduplicated network traffic, not reconstructed model bytes.
                first_transfer = not self.has_transfer
                self.has_transfer = True
                self.transferred += max(0, size)
                self.publish(force=first_transfer)

            def set_transfer_postfix_str(self, *_, **__):
                # The SDK's terminal speed suffix has no UI output here; the
                # progress message already reports the actual transfer count.
                pass

            def close(self):
                if not self.progress_closed:
                    self.progress_closed = True
                    self.publish(force=True)
                return super().close()

        stop = threading.Event()

        def monitor() -> None:
            previous = -1
            heartbeat = time.monotonic()
            while not stop.wait(0.15):
                partial = target.with_suffix(target.suffix + ".incomplete")
                try:
                    size = partial.stat().st_size if partial.is_file() else target.stat().st_size if target.is_file() else 0
                    phase = "verifying" if target.is_file() and not partial.is_file() else "downloading"
                    if size != previous or time.monotonic() - heartbeat > 15:
                        report(size, phase)
                        previous, heartbeat = size, time.monotonic()
                except OSError:
                    pass

        report(0)
        thread = None
        try:
            client.control = control
            if isinstance(client, _ModelScope):
                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                result = Path(client.get_file(request["repo_id"], request["revision"], path, local=destination, expected_sha256=hashes[path]))
            else:
                result = client.get_file(request["repo_id"], request["revision"], path, local=destination, tqdm_class=FileProgress)
            if result.resolve() != target.resolve() or not target.is_file():
                raise SourceError("SDK 未生成本次下载所需的文件。", "download_failed")
            size = target.stat().st_size
            if expected is not None and size != expected:
                raise SourceError("下载文件大小与仓库信息不一致。", "size_mismatch")
            report(size, "file_complete")
            completed += size
        finally:
            client.control = None
            stop.set()
            if thread:
                thread.join(timeout=1)
    emit("progress", {"phase": "complete", "downloaded_bytes": completed, "total_bytes": total or completed, 'can_pause': False})
    return {"path": str(destination), "files": [item["path"] for item in files], "revision": request["revision"], "source": request["source"], "repo_id": request["repo_id"], "size_bytes": completed}


def _readme(client, repo_id: str, revision: str, file_path: str | None = None) -> dict:
    path = relative_path(file_path) if file_path is not None else None
    if path and not re.fullmatch(r"README(?:[._-][^.]+)*\.(?:md|markdown|rst|txt)|README", path.rsplit("/", 1)[-1], re.IGNORECASE):
        raise SourceError("只能读取仓库内的 README 文档。", "invalid_readme")
    repository = client.resolve(repo_id, revision)
    if path:
        if path not in {item["path"] for item in repository["files"]}:
            raise SourceError("指定的 README 文件不在此仓库版本中。", "not_found")
    else:
        root_files = {item["path"] for item in repository["files"] if "/" not in item["path"]}
        for name in ("README.md", "README.markdown", "README.rst", "README.txt", "README", "README_zh.md", "README_cn.md", "README_zh-CN.md", "README.zh-CN.md", "README-zh.md"):
            matches = sorted(item for item in root_files if item.casefold() == name.casefold())
            if matches:
                path = name if name in matches else matches[0]
                break
    content = client.get_file(repo_id, repository["revision"], path).read_text(encoding="utf-8-sig") if path else ""
    return {"content": content, "path": path, "url": repository["url"], "revision": repository["revision"]}


def execute(request: dict, control: _DownloadControl | None = None) -> dict:
    cache = Path(request["cache"])
    source = request["source"]
    client = _HuggingFace(cache) if source == "hf" else _ModelScope(cache)
    action = request["action"]
    if action == "search":
        return client.search(request.get("query", ""), request.get("cursor"))
    repo_id = repository_id(request["repo_id"], source)
    revision = revision_name(request["revision"])
    if action == "resolve":
        return client.resolve(repo_id, revision)
    if action == "readme":
        return _readme(client, repo_id, revision, request.get("file_path"))
    if action == "read_json":
        data = json.loads(client.get_file(repo_id, revision, relative_path(request["file"])).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SourceError("模型配置或索引不是 JSON 对象。", "invalid_index")
        return data
    if action == "download":
        return _download(client, {**request, "repo_id": repo_id, "revision": revision}, control)
    raise SourceError("未知来源任务。", "worker_error")


def main() -> int:
    source = "Hub"
    control_thread = None
    try:
        request = json.loads(sys.stdin.buffer.readline())
        control = None
        if request.get('action') == 'download' and request.get('pausable'):
            control = _DownloadControl()
            control_thread = threading.Thread(target=control.listen, args=(sys.stdin.buffer,), daemon=True)
            control_thread.start()
        source = request.get("source", "Hub")
        logging.disable(logging.CRITICAL)
        # SDKs occasionally print notices; only our JSONL protocol is returned.
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(request, control)
        emit("result", result)
        return 0
    except Exception as error:
        message, code = error_message(error, source)
        emit("error", message=message, code=code)
        return 1
    finally:
        if control_thread:
            control_thread.join(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
