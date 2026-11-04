from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit


# These are transport limits, not limits on model or repository size.
NETWORK_ATTEMPTS = 3
NETWORK_TIMEOUT = 30.0
WORKER_IDLE_TIMEOUT = 300.0
SEARCH_PAGE_SIZE = 30


class SourceError(ValueError):
    def __init__(self, message: str, code: str = "source_error"):
        super().__init__(message)
        self.code = code


def source_name(source: str) -> str:
    value = source.strip().lower()
    if value in {"hf", "huggingface", "huggingface_hub"}:
        return "hf"
    if value in {"modelscope", "ms"}:
        return "modelscope"
    if value in {"url", "direct"}:
        return "url"
    raise SourceError("不支持的模型来源。", "unsupported_source")


def relative_path(value: str) -> str:
    # Windows ADS, drive paths, reserved names and trailing dots are unsafe even
    # when pathlib is running on a non-Windows test host.
    parts = value.split("/")
    if not value or "\\" in value or any(part in {"", ".", ".."} for part in parts):
        raise SourceError("文件路径必须是仓库内的相对路径。", "unsafe_path")
    for part in parts:
        if re.search(r'[<>:"|?*\x00-\x1f]', part) or part.endswith((".", " ")):
            raise SourceError("文件路径包含 Windows 不支持的字符。", "unsafe_path")
        stem = part.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", stem):
            raise SourceError("文件路径包含 Windows 保留名称。", "unsafe_path")
    return PurePosixPath(value).as_posix()


def repository_id(value: str, source: str = "hf") -> str:
    value = value.strip()
    parts = relative_path(value).split("/")
    if len(parts) not in ({1, 2} if source == "hf" else {2}):
        raise SourceError("仓库 ID 应为 owner/model（Hugging Face 也支持无命名空间的仓库）。", "invalid_repository")
    return value


def revision_name(value: str) -> str:
    return relative_path(value.strip() or "main")


def destination_path(models_root: Path, value: str) -> Path:
    if not value:
        raise SourceError("下载目录必须由管理核心指定。", "unsafe_destination")
    root = models_root.resolve()
    destination = Path(value).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise SourceError("下载目录必须位于应用的模型目录内。", "unsafe_destination")
    return destination


def file_target(destination: Path, value: str) -> Path:
    target = destination.joinpath(*relative_path(value).split("/"))
    if not target.resolve().is_relative_to(destination.resolve()):
        raise SourceError("文件路径越过了本次下载目录。", "unsafe_path")
    return target


def direct_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        valid = parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise SourceError("直链下载只支持有效的 HTTP(S) 地址。", "invalid_url")
    if parsed.username is not None or parsed.password is not None:
        raise SourceError("直链地址不能包含账号或密码。", "invalid_url")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, parsed.query, ""))


def public_url(value: str) -> str:
    parsed = urlsplit(direct_url(value))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def url_filename(value: str, supplied: str = "") -> str:
    name = supplied or unquote(urlsplit(direct_url(value)).path.rsplit("/", 1)[-1])
    if not name:
        raise SourceError("此地址没有文件名，请指定保存文件名。", "missing_filename")
    # A direct URL is one file; do not accept directory components in its name.
    name = relative_path(name)
    if "/" in name:
        raise SourceError("直链文件名不能包含目录。", "unsafe_path")
    return name


def error_message(error: Exception, source: str) -> tuple[str, str]:
    """Classify SDK failures without emitting signed URLs or credentials."""
    if isinstance(error, SourceError):
        return str(error), error.code
    name = type(error).__name__
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None) or getattr(error, "status_code", None)
    if "Gated" in name or status == 403 or name in {"PermissionDeniedError", "PermissionError"}:
        return "此仓库需要授权；请在来源网站取得访问权限并配置具有读取权限的令牌。", "authorization_required"
    if status == 401 or name == "AuthenticationError":
        return "来源认证失败；请检查令牌及仓库读取权限。", "authentication_failed"
    if status == 404 or name in {"RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError", "NotExistError", "NotFoundError"}:
        return "仓库、版本或文件不存在，或当前账号无权查看。", "not_found"
    if status == 429 or name == "RateLimitError":
        return "来源请求频率受限，请稍后重试。", "rate_limited"
    if isinstance(error, ImportError):
        return f"缺少 {source} 官方 SDK，请安装项目依赖。", "missing_dependency"
    if isinstance(error, (OSError,)) and not status:
        return "下载或文件写入失败，请检查网络、磁盘空间和目录权限。", "download_failed"
    return f"{source} 来源请求失败（{name}），请检查网络、代理和仓库信息。", "source_error"
