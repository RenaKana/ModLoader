from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from ._common import NETWORK_ATTEMPTS, NETWORK_TIMEOUT, SourceError, direct_url, file_target, public_url, url_filename
from .control import DownloadPause, PAUSE_CHUNK_SIZE


Progress = Callable[[dict], Awaitable[None]]
_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _length(headers: httpx.Headers) -> int | None:
    value = headers.get("content-length")
    if value is None:
        return None
    try:
        result = int(value)
    except ValueError:
        raise SourceError("来源返回了无效的文件长度。", "invalid_response") from None
    if result < 0:
        raise SourceError("来源返回了无效的文件长度。", "invalid_response")
    return result


def _raise_status(response: httpx.Response) -> None:
    if response.status_code in {401, 403}:
        raise SourceError("下载地址需要授权，或已过期。", "authorization_required")
    if response.status_code == 404:
        raise SourceError("下载文件不存在。", "not_found")
    if response.status_code >= 400:
        raise SourceError(f"下载服务器返回 HTTP {response.status_code}。", "http_error")


async def inspect_url(value: str, filename: str = "", proxy: str | None = None) -> dict:
    url = direct_url(value)
    name = url_filename(url, filename)
    async with httpx.AsyncClient(proxy=proxy or None, trust_env=False, follow_redirects=True, timeout=NETWORK_TIMEOUT) as client:
        try:
            response = await client.head(url, headers={"Accept-Encoding": "identity"})
            if response.status_code in {405, 501}:
                async with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                    _raise_status(response)
                    size = _length(response.headers)
            else:
                _raise_status(response)
                size = _length(response.headers)
        except httpx.HTTPError:
            raise SourceError("无法读取下载地址，请检查网络、代理或链接有效期。", "network_error") from None
    return {"path": name, "size": size, "url": public_url(url)}


def _validator(headers: httpx.Headers) -> tuple[str, str] | None:
    etag = headers.get("etag")
    if etag and not etag.startswith("W/"):
        return "etag", etag
    modified = headers.get("last-modified")
    if modified:
        return "last_modified", modified
    return None


def _metadata(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_metadata(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if not temporary.resolve().is_relative_to(path.parent.resolve()):
        raise SourceError("下载状态文件路径越过了本次下载目录。", "unsafe_path")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


async def download_url(
    value: str,
    filename: str,
    destination: Path,
    progress: Progress,
    cancel: asyncio.Event,
    proxy: str | None = None,
    pause: DownloadPause | None = None,
) -> dict:
    url = direct_url(value)
    name = url_filename(url, filename)
    target = file_target(destination, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    # These private files retain only a fingerprint and HTTP validators, never
    # the signed URL. The .part is closed on cancellation before returning.
    part = file_target(destination, name + ".modloader-part")
    journal = file_target(destination, name + ".modloader-meta.json")
    fingerprint = hashlib.sha256(url.encode("utf-8")).hexdigest()
    last_failure: SourceError | None = None
    restart_used = False
    async with httpx.AsyncClient(proxy=proxy or None, trust_env=False, follow_redirects=True, timeout=NETWORK_TIMEOUT) as client:
        for attempt in range(NETWORK_ATTEMPTS):
            if cancel.is_set():
                raise asyncio.CancelledError
            state = _metadata(journal)
            offset = part.stat().st_size if part.is_file() else 0
            if offset and (state.get("fingerprint") != fingerprint or not state.get("validator")):
                part.unlink(missing_ok=True)
                journal.unlink(missing_ok=True)
                offset = 0
                state = {}
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                headers["If-Range"] = state["validator"][1]
            try:
                async with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416 and offset and not restart_used:
                        # An already-full partial or a changed resource cannot
                        # be proven from a 416 alone; safely request it again.
                        part.unlink(missing_ok=True)
                        journal.unlink(missing_ok=True)
                        restart_used = True
                        continue
                    if response.status_code in {429, 500, 502, 503, 504}:
                        last_failure = SourceError(f"下载服务器暂时不可用（HTTP {response.status_code}）。", "http_error")
                    else:
                        _raise_status(response)
                        if response.headers.get("content-encoding", "identity").lower() != "identity":
                            raise SourceError("下载服务器未提供原始文件字节，无法安全恢复下载。", "invalid_response")
                        validator = _validator(response.headers)
                        total = _length(response.headers)
                        append = False
                        if response.status_code == 206:
                            match = _RANGE.fullmatch(response.headers.get("content-range", ""))
                            if not match:
                                raise SourceError("服务器返回的 Content-Range 无效，已停止以避免损坏文件。", "invalid_range")
                            start, end, remote_total = map(int, match.groups())
                            if start != offset or end < start or end >= remote_total or (total is not None and total != end - start + 1):
                                raise SourceError("服务器返回的下载范围与本地进度不一致。", "invalid_range")
                            if offset and (validator is None or list(validator) != state.get("validator") or (state.get("total") is not None and state["total"] != remote_total)):
                                if restart_used:
                                    raise SourceError("下载资源在恢复期间发生变化，请重新下载。", "resource_changed")
                                part.unlink(missing_ok=True)
                                journal.unlink(missing_ok=True)
                                restart_used = True
                                continue
                            total = remote_total
                            append = offset > 0
                        elif response.status_code == 200:
                            # Ignored Range or failed If-Range: never append a
                            # whole response to an old partial.
                            offset = 0
                        else:
                            raise SourceError(f"服务器返回了不支持的下载状态 HTTP {response.status_code}。", "invalid_response")
                        _save_metadata(journal, {"fingerprint": fingerprint, "validator": list(validator) if validator else None, "total": total})
                        downloaded = offset
                        await progress({"phase": "downloading", "message": f"{'正在继续下载' if append else '正在下载'} {name}", "file": name,
                                        "downloaded_bytes": downloaded, "total_bytes": total,
                                        "file_downloaded_bytes": downloaded, "file_total_bytes": total,
                                        "resumed": append, "can_pause": pause is not None})
                        emitted = time.monotonic()
                        with part.open("ab" if append else "wb") as output:
                            chunks = response.aiter_raw(chunk_size=PAUSE_CHUNK_SIZE).__aiter__()
                            while True:
                                if pause:
                                    output.flush()
                                    await pause.checkpoint(cancel)
                                try:
                                    chunk = await anext(chunks)
                                except StopAsyncIteration:
                                    break
                                if cancel.is_set():
                                    raise asyncio.CancelledError
                                if pause:
                                    await pause.checkpoint(cancel)
                                output.write(chunk)
                                downloaded += len(chunk)
                                if total is not None and downloaded > total:
                                    raise SourceError("下载内容超过声明的文件长度，已停止。", "size_mismatch")
                                if time.monotonic() - emitted >= 0.1:
                                    output.flush()
                                    await progress({"phase": "downloading", "message": f"正在下载 {name}", "file": name,
                                                    "downloaded_bytes": downloaded, "total_bytes": total,
                                                    "file_downloaded_bytes": downloaded, "file_total_bytes": total})
                                    emitted = time.monotonic()
                        if cancel.is_set():
                            raise asyncio.CancelledError
                        if total is not None and downloaded != total:
                            raise SourceError("下载内容不完整，请重试以继续下载。", "size_mismatch")
                        # Recheck an existing symlink/junction immediately before
                        # the atomic final placement.
                        file_target(destination, name)
                        os.replace(part, target)
                        journal.unlink(missing_ok=True)
                        await progress({"phase": "complete", "message": f"文件下载完成：{name}", "file": name,
                                        "downloaded_bytes": downloaded, "total_bytes": downloaded,
                                        "file_downloaded_bytes": downloaded, "file_total_bytes": downloaded,
                                        "can_pause": False})
                        return {"path": str(destination), "files": [name], "revision": "", "size_bytes": downloaded, "source": "url", "url": public_url(url)}
            except (httpx.HTTPError, OSError):
                last_failure = SourceError("下载连接中断或文件写入失败；可重试以继续下载。", "download_failed")
            if attempt + 1 < NETWORK_ATTEMPTS:
                await asyncio.sleep(0.5 * (attempt + 1))
        raise last_failure or SourceError("无法完成下载，请检查链接是否发生变化。", "download_failed")
