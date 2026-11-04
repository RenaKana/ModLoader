"""Metadata-only resolver program, executed by the destination Python."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

SGLANG_CUDA_TILE_URL = 'https://pypi.nvidia.com/cuda-tile/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl'
SGLANG_CUDA_TILE_SHA256 = 'b74c20348210d2182cd998a0ecb60c518989a79b28592d72eb8294b38ddb93d7'


def requirements_identity(requirements):
    # Installation roots change on retry; a requirements file is identified by content.
    return ["sha256:" + hashlib.sha256(Path(value).read_bytes()).hexdigest()
            if index and requirements[index - 1] == "-r" else value
            for index, value in enumerate(requirements)]


def package_files(report):
    files = []
    for package in report["install"]:
        info = package["download_info"]
        url = info["url"]
        checksum = info.get("archive_info", {}).get("hashes", {}).get("sha256")
        parsed = urlsplit(url)
        name = unquote(parsed.path.rsplit("/", 1)[-1])
        nvidia_tile = url == SGLANG_CUDA_TILE_URL and checksum == SGLANG_CUDA_TILE_SHA256
        if (parsed.scheme != "https" or not (parsed.hostname in {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"} or nvidia_tile)
                or not name.endswith(".whl") or Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", checksum or "")):
            raise ValueError("依赖报告缺少官方 wheel URL / SHA256，未下载依赖")
        metadata = package["metadata"]
        files.append(dict(name=name, purpose=f"{metadata['name']} {metadata['version']}", url=url,
                          digest="sha256:" + checksum, size_bytes=None, package=metadata["name"], version=metadata["version"]))
    return files


def previous_resolution(paths, retry, requirements):
    stage = retry["stage"]
    if stage.get("resolution"):
        return stage["resolution"]
    # Older jobs already retain their pip reports but did not link the environment
    # to the manifest. Reuse only an exact report match with the same pinned inputs.
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9_.+-]+", req) for req in requirements):
        return None
    def requested_key(name, version, extras=()):
        return (re.sub(r"[-_.]+", "-", name).lower(), version, tuple(sorted(extras)))
    wanted = []
    for req in requirements:
        name, version = req.split("==")
        base, _, extras = name.partition("[")
        wanted.append(requested_key(base, version, extras.rstrip("]").split(",") if extras else ()))
    def identities(files):
        return [(item["url"], item["digest"], item["package"], item["version"]) for item in files]
    for path in paths.runtimes.glob(f"{retry['backend']}-{retry['version']}-*/packages.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            requested = [requested_key(p["metadata"]["name"], p["metadata"]["version"], p.get("requested_extras") or ())
                         for p in report["install"] if p.get("requested")]
            if sorted(requested) == sorted(wanted) and identities(package_files(report)) == identities(stage["files"]) and report.get("environment"):
                return dict(environment=report["environment"], requirements=requirements_identity(requirements),
                            resolver=hashlib.sha256(SOURCE.encode()).hexdigest())
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


SOURCE = r'''
"""Run with the destination venv's Python; resolve wheels using metadata only.

pip dry-run ordinarily downloads wheels after resolution. This bounded adapter
prevents that step and refuses candidates without PEP 658 metadata. It never
installs packages; its JSON report is input to the separate confirmation stage.
"""
from __future__ import annotations

import sys
import json
import os
from urllib.parse import urlsplit


def main(argv=None):
    from pip._internal.cli.main import main as pip_main
    from pip._internal.network.session import PipSession
    from pip._internal.operations.prepare import RequirementPreparer
    from pip._internal.utils.direct_url_helpers import direct_url_from_link
    from pip._internal.distributions.wheel import WheelDistribution

    original_request = PipSession.request
    argv = list(argv or sys.argv[1:])
    approved = {}
    if '--modloader-approved-wheels' in argv:
        index = argv.index('--modloader-approved-wheels')
        with open(argv[index + 1], encoding='utf-8') as source:
            entries = json.load(source)
        del argv[index:index + 2]
        for item in entries:
            if (item.get('url') != 'https://pypi.nvidia.com/cuda-tile/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl'
                    or item.get('digest') != 'sha256:b74c20348210d2182cd998a0ecb60c518989a79b28592d72eb8294b38ddb93d7'):
                raise RuntimeError('Unapproved local metadata wheel')
            approved[os.path.realpath(item['path'])] = item
    # CUDA is a choice for torch, not an alternate source for every dependency.
    cuda_index = "--extra-index-url" in argv
    if cuda_index:
        index = argv.index("--extra-index-url")
        if argv[index + 1] != "https://download.pytorch.org/whl/cu130":
            raise RuntimeError("Unsupported CUDA package index")
        del argv[index:index + 2]

    def metadata_request(self, method, url, *args, **kwargs):
        self.trust_env = False
        parsed = urlsplit(url)
        if cuda_index and parsed.hostname == "pypi.org" and parsed.path == "/simple/torch/":
            url = "https://download.pytorch.org/whl/cu130/torch/"
            parsed = urlsplit(url)
        path = parsed.path.lower()
        allowed = (parsed.hostname == "pypi.org" and path.startswith("/simple/")) or (
            parsed.hostname == "download.pytorch.org" and path.startswith("/whl/") and not path.endswith(".whl")) or (
            parsed.hostname in {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"} and path.endswith(".whl.metadata"))
        if method.upper() not in {"GET", "HEAD"} or parsed.scheme != "https" or not allowed:
            raise RuntimeError("依赖解析阶段禁止下载软件包；该来源未提供可独立获取的官方元数据")
        kwargs["allow_redirects"] = False
        result = original_request(self, method, url, *args, **kwargs)
        if result.is_redirect:
            raise RuntimeError("依赖元数据发生重定向；停止解析以避免未确认的软件包下载")
        return result

    def metadata_only(self, req, parallel_builds=False):
        if not req.link or not req.link.is_wheel or not req.link.hash or req.link.hash_name != "sha256":
            raise RuntimeError(f"{req.name} 必须具有官方 SHA256 和兼容 wheel；不在确认前构建源码包")
        if req.link.is_file and os.path.realpath(req.link.file_path) in approved:
            item = approved[os.path.realpath(req.link.file_path)]
            if req.link.hash != item['digest'].removeprefix('sha256:'):
                raise RuntimeError('Approved wheel identity mismatch')
            req.local_file_path = req.link.file_path
            dist = WheelDistribution(req).get_metadata_distribution()
            if str(dist.canonical_name) != 'cuda-tile' or str(dist.version) != '1.6.0rc5':
                raise RuntimeError('Approved CUDA Tile wheel metadata mismatch')
            req.download_info = direct_url_from_link(req.link)
            req.download_info.url = item['url']
            req.get_dist = lambda: dist
            return dist
        dist = self._fetch_metadata_using_link_data_attr(req)
        if dist is None:
            raise RuntimeError(f"{req.name} 没有 PEP 658 独立元数据；未下载软件包，可改用已有环境")
        req.download_info = direct_url_from_link(req.link)
        req.get_dist = lambda: dist
        return dist

    def deny_payload(*args, **kwargs):
        raise RuntimeError("解析阶段禁止软件包下载")

    # This helper is short-lived and patches only its own destination interpreter.
    PipSession.request = metadata_request
    RequirementPreparer.prepare_linked_requirement = metadata_only
    RequirementPreparer._prepare_linked_requirement = deny_payload
    RequirementPreparer.prepare_linked_requirements_more = lambda *args, **kwargs: None
    return pip_main(["--isolated", "install", "--dry-run", "--ignore-installed", "--only-binary=:all:",
                     "--disable-pip-version-check", "--no-cache-dir", "--retries", "0", "--timeout", "45", *argv])


if __name__ == "__main__":
    raise SystemExit(main())

'''
