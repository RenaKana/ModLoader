"""Official runtime releases and immutable, user-reviewable install manifests."""
from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from pathlib import Path

import httpx

RECOMMENDED = {"llama_cpp": "b11609", "strata": "0.2.10", "transformers": "5.19.0", "vllm": "0.31.0", "sglang": "0.5.23", "ninfer": "a754fcf3bf52", "sparkinfer": "0.6.34"}
REPOSITORIES = {"llama_cpp": "ggml-org/llama.cpp", "strata": "Niko1221/Strata"}
# Published Control 0.1.49.2 fixes DeepSeek's unrelated parent mmproj selection (#3917/#3927).
STRATA_SOURCE_OVERRIDES = {("v0.2.4", "dsv41-cuda13"): "dfa3f4025c9f1bc4af8cee201b20a2d223061813"}
PYTHON_URL = "https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg"
PYTHON_SHA256 = "0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8"
UBUNTU_URL = "https://cloud-images.ubuntu.com/wsl/releases/24.04/20240423/ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz"
UBUNTU_SHA256 = "2a790896740b14d637dbdc583cce1ba081ac53b9e9cdb46dc09a2f73abbd9934"


def backend_capabilities():
    return {
        "llama_cpp": dict(description="GGUF 模型原生推理", subtitle="Windows 原生程序", symbol="L", installable=True, import_kind="file", import_hint="选择 llama-server.exe", python_option=False),
        "strata": dict(description="MoE 模型与原生 pack 推理", subtitle="Windows · Python + 原生引擎", symbol="S", installable=True, import_kind="directory", import_hint="选择 Strata 根目录（含 serve 和引擎）", python_option=True),
        "transformers": dict(description="完整 Hugging Face 模型目录", subtitle="Windows · 独立 Python 环境", symbol="T", installable=True, import_kind="file", import_hint="选择 transformers.exe", python_option=True),
        "vllm": dict(description="高吞吐 Hugging Face 模型推理", subtitle="专属 ModLoader WSL2", symbol="V", installable=True, import_kind="wsl", import_hint="ModLoader 发行版内的 /绝对路径/bin/vllm", python_option=False),
        "custom": dict(description="已安装的本机 OpenAI 兼容服务", subtitle="自定义 EXE / Python · 手动配置", symbol="C", installable=False, supported_platforms=['windows', 'wsl2'], import_kind="file", import_hint="选择本机引擎 EXE 或 Python EXE；导入不运行程序", python_option=False),
        "sglang": dict(description="本地 HF 推理与原生 NEXTN / MTP", subtitle="WSL2 · 独立 Python 环境", symbol="G", installable=True, supported_platforms=['wsl2'], import_kind="wsl", import_hint="选择发行版内安装 SGLang 的 /绝对路径/bin/python", python_option=False),
        "ninfer": dict(description="原生 .ninfer artifact 与 MTP", subtitle="WSL2 · 固定源码 · RTX 5090", symbol="N", installable=True, supported_platforms=['wsl2'], import_kind="wsl", import_hint="完整构建中的 /绝对路径/build/apps/ninfer-serve", python_option=False),
        "sparkinfer": dict(description="GGUF / 原生 NVFP4 与 DFlash2 / DSpark", subtitle="WSL2 · 固定源码与独立工具链", symbol="P", installable=True, supported_platforms=['wsl2'], import_kind="wsl", import_hint="完整构建中的 /绝对路径/build/server/sparkinfer_server", python_option=False),
    }


async def _request(url, proxy=None):
    try:
        async with httpx.AsyncClient(proxy=proxy or None, trust_env=False, follow_redirects=True, timeout=45) as client:
            response = await client.get(url, headers={"User-Agent": "ModLoader/0.1", "Accept": "application/json"})
            response.raise_for_status()
            return response
    except httpx.HTTPError as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status == 429 or (status == 403 and response.headers.get("x-ratelimit-remaining") == "0"):
            raise ValueError("官方版本目录的请求额度已用尽，请稍后重试；本次未开始下载") from exc
        raise ValueError(f"官方版本目录读取失败{f'（HTTP {status}）' if status else '（连接或超时）'}；请检查代理后重试") from exc


async def _json(url, proxy=None):
    return (await _request(url, proxy)).json()


def _digest(asset):
    value = asset.get("digest") or ""
    return value if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value) else None


def _strata_assets(release):
    flavors = {
        ("engine", "cuda-cuda13"): ("cuda13", "CUDA 13 · NVIDIA"),
        ("engine", "cuda-cuda12"): ("cuda12", "CUDA 12 · NVIDIA"),
        ("engine", "hip-rx9070"): ("hip-rx9070", "HIP · RX 9000 / R9700"),
        ("engine", "hip-rdna3"): ("hip-rdna3", "HIP · RX 7000 / Strix Halo"),
        ("dsv41", "cuda-cuda13"): ("dsv41-cuda13", "DeepSeek V4.1 · CUDA 13 · NVIDIA"),
    }
    choices = {}
    for asset in release.get("assets", []):
        if not _digest(asset):
            continue
        name = asset["name"]
        if name == "strata-windows-x64.zip":
            choices["windows-x64"] = dict(label="Windows x64", asset=asset)
            continue
        match = re.fullmatch(r"strata-(engine|dsv41)-(\d+(?:\.\d+)+)-windows-x86_64-(.+)\.tar\.gz", name)
        if not match or (match[1], match[3]) not in flavors:
            continue
        variant, label = flavors[(match[1], match[3])]
        # A release can carry a later engine hotfix without changing its app tag.
        previous = choices.get(variant)
        if previous and tuple(map(int, previous["engine_version"].split("."))) >= tuple(map(int, match[2].split("."))):
            continue
        choices[variant] = dict(label=label, asset=asset, engine_version=match[2])
    order = [item[0] for (family, _), item in flavors.items() if family == "engine"] + ["windows-x64", "dsv41-cuda13"]
    return {variant: choices[variant] for variant in order if variant in choices}


def _variants(backend, release):
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    if backend == "strata":
        return [{"id": variant, "label": item["label"]} for variant, item in _strata_assets(release).items()]
    variants = []
    for name, asset in assets.items():
        match = re.fullmatch(r"llama-.+-bin-win-(.+)-x64\.zip", name)
        if not match or not _digest(asset):
            continue
        variant = match[1]
        if variant.startswith("cuda") and not _digest(assets.get(f"cudart-llama-bin-win-{variant}-x64.zip", {})):
            continue
        variants.append({"id": variant, "label": variant.upper().replace("-", " ") + " · x64"})
    return variants


async def list_versions(backend, cursor=None, proxy=None):
    if backend not in RECOMMENDED:
        raise ValueError("未知后端")
    if backend == 'sglang':
        from .sglang_versions import RECIPES
        versions = []
        for version in RECIPES:
            data = await _json(f'https://pypi.org/pypi/sglang/{version}/json', proxy)
            available = any(item.get('packagetype') == 'bdist_wheel' and not item.get('yanked') and item.get('digests', {}).get('sha256')
                            for item in data.get('urls', []))
            versions.append(dict(version=version, label=version, status='available' if available else 'unavailable',
                                 recommended=version == RECOMMENDED[backend], reason='固定安装配方；其他版本可手动导入。',
                                 variants=[{'id': 'wsl2', 'label': 'WSL2 · Linux x64'}] if available else []))
        return {'versions': versions, 'next_cursor': None}
    if backend in {'ninfer', 'sparkinfer'}:
        from .source_builds import PINS, NINFER_PINS
        pins = NINFER_PINS.values() if backend == 'ninfer' else [PINS[backend]]
        return {'versions': [dict(version=pin['version'], label=pin['version'] + ' · ' + pin['commit'][:12], status='available',
                                 recommended=pin['version'] == RECOMMENDED[backend], reason='固定源码版本',
                                 variants=[{'id': 'wsl2', 'label': 'WSL2 · RTX 5090 · 独立构建'}]) for pin in pins], 'next_cursor': None}
    if backend in REPOSITORIES:
        page = int(cursor or 1)
        if page < 1:
            raise ValueError("无效发行目录游标")
        url = f"https://api.github.com/repos/{REPOSITORIES[backend]}/releases?per_page=30&page={page}"
        response = await _request(url, proxy)
        versions = []
        for release in response.json():
            if release.get("draft"):
                continue
            version = release["tag_name"].removeprefix("v") if backend == "strata" else release["tag_name"]
            variants = _variants(backend, release)
            status = ("unverified" if release.get("prerelease") else "available") if variants else "unavailable"
            item = dict(version=version, label=release.get("name") or version, status=status,
                        recommended=version == RECOMMENDED[backend], variants=variants)
            if not variants:
                item["reason"] = "没有包含官方 SHA256 的 Windows x64 兼容发行件"
            elif release.get("prerelease"):
                item["reason"] = "预发行版本"
            versions.append(item)
        return {"versions": versions, "next_cursor": str(page + 1) if response.links.get("next") else None}
    data = await _json(f"https://pypi.org/pypi/{backend}/json", proxy)
    versions = []
    # PyPI's project API supplies its complete release map, without pagination.
    def ordering(value):
        return tuple(int(piece) if piece.isdigit() else -1 for piece in re.split(r"[.a-z+-]", value))
    for version, files in sorted(data.get("releases", {}).items(), key=lambda pair: ordering(pair[0]), reverse=True):
        usable = [item for item in files if item.get("packagetype") == "bdist_wheel" and not item.get("yanked") and item.get("digests", {}).get("sha256")]
        stable = bool(re.fullmatch(r"\d+(?:\.\d+)+", version))
        supported = bool(usable)
        status = "unverified" if supported else "unavailable"
        reason = ""
        if not supported:
            reason = "没有可安装的官方 wheel"
        if supported and stable and version == RECOMMENDED[backend]:
            status = "available"
            reason = ""
        variants = [{"id": "cuda-13", "label": "CUDA 13 · Windows x64"}] if backend == "transformers" else [{"id": "wsl2", "label": "WSL2 · Linux x64"}]
        versions.append(dict(version=version, label=version, status=status, recommended=version == RECOMMENDED[backend], reason=reason, variants=variants if supported else []))
    return {"versions": versions, "next_cursor": None}


def cache_path(paths, item):
    identity = item.get("digest") or item.get("source_commit")
    if not identity:
        raise ValueError("发行件缺少可验证身份")
    key = hashlib.sha256((item["url"] + "\n" + identity).encode()).hexdigest()
    return paths.runtimes / ".artifact-cache" / key / item["name"]


def cached_file(paths, item):
    path = cache_path(paths, item)
    if not path.is_file():
        return False
    digest = item.get("digest")
    if not digest:
        receipt = path.with_suffix(path.suffix + ".sha256")
        if not receipt.is_file():
            return False
        digest = "sha256:" + receipt.read_text().strip()
    algorithm, expected = digest.split(":", 1)
    if algorithm != "sha256":
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


async def manifest_file(paths, name, purpose, url, digest=None, size=None, **extra):
    item = dict(id=uuid.uuid4().hex, name=name, purpose=purpose, size_bytes=size, url=url, digest=digest, **extra)
    item["cached"] = await asyncio.to_thread(cached_file, paths, item)
    item["status"] = "cached" if item["cached"] else "pending"
    return item


async def plan_install(backend, paths, options, proxy=None):
    if backend not in RECOMMENDED:
        raise ValueError("未知后端")
    if backend in {'ninfer', 'sparkinfer'}:
        from .source_builds import plan_install as source_plan
        return await source_plan(backend, paths, options, proxy)
    version = str(options.get("version") or RECOMMENDED[backend])
    if not re.fullmatch(r"[A-Za-z0-9.+_-]+", version):
        raise ValueError("无效版本")
    if backend == 'sglang':
        from .sglang_versions import recipe
        sglang_recipe = recipe(version)
    files, provenance = [], {}
    variant = options.get("variant") or options.get("flavor")
    if backend in REPOSITORIES:
        tag = "v" + version if backend == "strata" else version
        release = await _json(f"https://api.github.com/repos/{REPOSITORIES[backend]}/releases/tags/{tag}", proxy)
        variants = _variants(backend, release)
        variant = variant or next((v["id"] for v in variants if v["id"] == "cuda-13.4"), variants[0]["id"] if variants else None)
        if variant not in {v["id"] for v in variants}:
            raise ValueError("此版本没有所选变体或缺少官方 SHA256")
        if backend == "strata":
            engine = _strata_assets(release)[variant]
            names = [engine["asset"]["name"]]
            if engine.get("engine_version"):
                provenance["engine_version"] = engine["engine_version"]
        else:
            names = [f"llama-{version}-bin-win-{variant}-x64.zip"]
        if backend == "llama_cpp" and variant.startswith("cuda"):
            names.append(f"cudart-llama-bin-win-{variant}-x64.zip")
        for name in names:
            asset = next(a for a in release["assets"] if a["name"] == name)
            files.append(await manifest_file(paths, name, "推理引擎" if name == names[0] else "CUDA 运行库", asset["browser_download_url"], _digest(asset), asset.get("size"), role="engine" if name == names[0] else "cuda"))
        provenance.update(repository=REPOSITORIES[backend], tag=tag, release_id=release["id"])
        if backend == "strata":
            obj = (await _json(f"https://api.github.com/repos/Niko1221/Strata/git/ref/tags/{tag}", proxy))["object"]
            if obj["type"] == "tag":
                obj = (await _json(obj["url"], proxy))["object"]
            commit = obj["sha"]
            if not re.fullmatch(r"[a-f0-9]{40}", commit):
                raise ValueError("无效 Strata 源码提交")
            commit = STRATA_SOURCE_OVERRIDES.get((tag, variant), commit)
            provenance["source_commit"] = commit
            tree = await _json(f"https://api.github.com/repos/Niko1221/Strata/git/trees/{commit}?recursive=1", proxy)
            if tree.get("truncated"):
                raise ValueError("官方源码树未完整返回，无法验证源码归属")
            provenance["source_tree"] = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
            files.append(await manifest_file(paths, f"Strata-{commit}.zip", "服务源码（固定官方 Git 提交；源站未提供 SHA256）", f"https://codeload.github.com/Niko1221/Strata/zip/{commit}", role="source", source_commit=commit))
    else:
        data = await _json(f"https://pypi.org/pypi/{backend}/{version}/json", proxy)
        if not any(f.get("packagetype") == "bdist_wheel" and not f.get("yanked") for f in data.get("urls", [])):
            raise ValueError("此版本没有官方 wheel")
        if backend == "transformers":
            info = data.get("info", {})
            serving = "serving" in (info.get("provides_extra") or []) or any(
                re.search(r"\bextra\s*==\s*['\"]serving['\"]", requirement)
                for requirement in (info.get("requires_dist") or [])
            )
            if not serving:
                raise ValueError("此版本官方包元数据未声明 serving 依赖组，无法生成当前 serve 适配器的安装配方")
        variant = variant or ("cuda-13" if backend == "transformers" else "wsl2")
        if variant != ("cuda-13" if backend == "transformers" else "wsl2"):
            raise ValueError("此版本没有所选变体")
        provenance["index"] = "https://pypi.org/simple"
        if backend == 'sglang':
            provenance.update({key: sglang_recipe[key] for key in ('repository', 'tag', 'source_commit')})
            for wheel in sglang_recipe['approved_wheels']:
                files.append(await manifest_file(paths, wheel['name'], wheel['purpose'], wheel['url'], wheel['digest'],
                             role=wheel['role'], package=wheel['package'], version=wheel['version']))
            from .source_builds import cuda_files
            files += await cuda_files(paths, proxy, components=sglang_recipe['cuda_components'])
            provenance.update(cuda_release=sglang_recipe['cuda_release'], cuda_components=list(sglang_recipe['cuda_components']))
    if backend in {"strata", "transformers"} and not options.get("python"):
        files.append(await manifest_file(paths, "python.3.12.10.nupkg", "独立 Python 3.12.10（含 pip/venv）", PYTHON_URL, "sha256:" + PYTHON_SHA256, role="python"))
    if backend in {"vllm", "sglang"} and not (paths.runtimes / "ModLoader-wsl-owned.json").is_file():
        files.append(await manifest_file(paths, "ubuntu-24.04.rootfs.tar.gz", "专属 WSL2 Ubuntu 基础系统", UBUNTU_URL, "sha256:" + UBUNTU_SHA256, role="rootfs"))
    return dict(id=uuid.uuid4().hex, backend=backend, version=version, variant=variant,
                options={k: v for k, v in options.items() if k in {"python", "version", "variant"}
                         or backend == 'strata' and k == 'startup_expert_prefetch'}, provenance=provenance,
                stage=dict(id=uuid.uuid4().hex, title="基础文件", files=files),
                dependency_notice="基础环境就绪后自动解析并安装所需依赖，文件清单和下载进度显示在任务详情中。" if backend != "llama_cpp" else None)
