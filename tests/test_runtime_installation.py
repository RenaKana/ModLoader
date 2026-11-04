import asyncio
import hashlib
import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from modloader.backends import catalog, installers
from modloader.models import AppPaths, RuntimeInstallation


async def test_command_environment_override_is_child_scoped_and_not_logged(monkeypatch):
    import os
    monkeypatch.setenv('MODLOADER_FIXTURE_INHERITED', 'kept')
    monkeypatch.setenv('HTTP_PROXY', 'original-parent-value')
    child_proxy = 'http://fixture-user:fixture-password@proxy.invalid:17890'
    before = dict(os.environ)
    events = []
    code = ("import os; print(os.environ['HTTP_PROXY'] == os.environ['HTTPS_PROXY'] "
            "and os.environ['HTTP_PROXY'].endswith('@proxy.invalid:17890') "
            "and os.environ['MODLOADER_FIXTURE_INHERITED'] == 'kept')")
    output = await installers._command([sys.executable, '-c', code], events.append, asyncio.Event(),
                                       env={'HTTP_PROXY': child_proxy, 'HTTPS_PROXY': child_proxy})
    assert output == 'True'
    assert dict(os.environ) == before
    assert 'fixture-password' not in json.dumps(events)


@pytest.mark.parametrize('proxy', [None, '', 'http://proxy.example:7890', 'https://192.168.1.2:8443'])
async def test_wsl_proxy_preserves_non_loopback_settings(proxy, tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs): raise AssertionError('unneeded WSL probe')
    monkeypatch.setattr(installers, '_wsl_command', forbidden)
    assert await installers._wsl_proxy(proxy, tmp_path, None, asyncio.Event()) == proxy


@pytest.mark.parametrize('proxy,host,port', [
    ('http://127.0.0.1:7890', '127.0.0.1', '7890'),
    ('http://localhost:17890', 'localhost', '17890'),
    ('https://[::1]', '::1', '443'),
])
async def test_wsl_proxy_translates_only_host_and_retains_port(proxy, host, port, tmp_path, monkeypatch):
    from types import SimpleNamespace
    async def command(argv, *args, **kwargs):
        assert argv[-2:] == [host, port]
        return 'wsl: diagnostic\n"172.20.112.1"'
    monkeypatch.setattr(installers, '_wsl_command', command)
    monkeypatch.setattr(installers.psutil, 'net_if_addrs', lambda: {'WSL': [SimpleNamespace(address='172.20.112.1')]})
    result = await installers._wsl_proxy(proxy, tmp_path, None, asyncio.Event())
    assert result == proxy.split('://')[0] + '://172.20.112.1:' + port


async def test_wsl_proxy_keeps_reachable_loopback_and_surfaces_failure(tmp_path, monkeypatch):
    async def mirrored(*args, **kwargs): return '"127.0.0.1"'
    monkeypatch.setattr(installers, '_wsl_command', mirrored)
    assert await installers._wsl_proxy('http://127.0.0.1:7890', tmp_path, None, asyncio.Event()) == 'http://127.0.0.1:7890'
    async def unreachable(*args, **kwargs): raise RuntimeError('WSL 无法连接已配置的 Windows 本机代理')
    monkeypatch.setattr(installers, '_wsl_command', unreachable)
    with pytest.raises(RuntimeError, match='无法连接'):
        await installers._wsl_proxy('http://127.0.0.1:17890', tmp_path, None, asyncio.Event())


async def test_wsl_proxy_rejects_gateway_outside_windows_host(tmp_path, monkeypatch):
    async def router(*args, **kwargs): return '"192.168.1.1"'
    monkeypatch.setattr(installers, '_wsl_command', router)
    monkeypatch.setattr(installers.psutil, 'net_if_addrs', lambda: {})
    with pytest.raises(ValueError, match='不是本机 Windows 地址'):
        await installers._wsl_proxy('http://127.0.0.1:7890', tmp_path, None, asyncio.Event())


@pytest.mark.parametrize('status,headers,quota', [(403, {'x-ratelimit-remaining':'0'}, True), (429, {}, True), (403, {}, False)])
async def test_catalog_distinguishes_upstream_quota_from_other_network_errors(monkeypatch, status, headers, quota):
    original_client = httpx.AsyncClient
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(status, headers=headers, json={'message':'upstream error'})
    monkeypatch.setattr(catalog.httpx, 'AsyncClient', lambda **kwargs: original_client(transport=httpx.MockTransport(handle)))
    with pytest.raises(ValueError) as error:
        await catalog.list_versions('llama_cpp', proxy='http://127.0.0.1:17890')
    assert ('额度已用尽' in str(error.value)) is quota
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_catalog_follows_official_pages_and_rejects_missing_digest(monkeypatch):
    urls = []
    async def request(url, proxy=None):
        urls.append((url, proxy))
        return httpx.Response(200, request=httpx.Request("GET", url), headers={"Link": '<https://api.github.com/next>; rel="next"'}, json=[
            {"tag_name": "b99999", "assets": [{"name": "llama-b99999-bin-win-cpu-x64.zip", "digest": "sha256:" + "a" * 64}]},
            {"tag_name": "b99998", "assets": [{"name": "llama-b99998-bin-win-cpu-x64.zip"}]}])
    monkeypatch.setattr(catalog, "_request", request)
    result = await catalog.list_versions("llama_cpp", cursor="4", proxy="http://proxy:80")
    assert result["next_cursor"] == "5"
    assert "page=4" in urls[0][0]
    assert urls[0][1] == "http://proxy:80"
    assert result["versions"][0]["variants"] == [{"id": "cpu", "label": "CPU · x64"}]
    assert result["versions"][1]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_plan_freezes_official_version_variant_files_without_download(tmp_path, monkeypatch):
    paths = AppPaths(tmp_path)
    async def get_json(url, proxy=None):
        return {"id": 7, "tag_name": "b99999", "assets": [{"name": "llama-b99999-bin-win-cpu-x64.zip", "digest": "sha256:" + "a" * 64, "size": 17, "browser_download_url": "https://github.com/engine.zip"}]}
    monkeypatch.setattr(catalog, "_json", get_json)
    plan = await catalog.plan_install("llama_cpp", paths, {"version": "b99999", "variant": "cpu"})
    assert (plan["version"], plan["variant"]) == ("b99999", "cpu")
    assert plan["stage"]["files"][0]["size_bytes"] == 17
    assert not list(paths.runtimes.iterdir())


@pytest.mark.asyncio
async def test_confirm_blocks_payload_and_reuses_verified_cache(tmp_path, monkeypatch):
    paths, calls, events = AppPaths(tmp_path), [], []
    content = b"approved wheel"
    checksum = "sha256:" + hashlib.sha256(content).hexdigest()
    item = await catalog.manifest_file(paths, "a.whl", "a 1", "https://files.pythonhosted.org/a.whl", checksum, len(content))
    async def download(url, target, progress, cancel, digest=None, pause=None, proxy=None):
        calls.append(url)
        target.write_bytes(content)
        await progress(dict(downloaded=len(content), total=len(content), phase="download"))
    monkeypatch.setattr(installers, "_download", download)
    confirm_entered, approved = asyncio.Event(), asyncio.Event()
    async def confirm(stage):
        assert stage["files"][0]["digest"] == checksum
        confirm_entered.set()
        await approved.wait()
    transfer = installers._Transfers(paths, events.append, asyncio.Event(), None, None)
    async def execute():
        stage = await transfer.add(dict(id="dependencies", title="Dependencies", files=[item]), confirm)
        return await transfer.fetch(stage)
    task = asyncio.create_task(execute())
    await confirm_entered.wait()
    assert not calls
    approved.set()
    await task
    assert calls == [item["url"]]
    next_transfer = installers._Transfers(paths, events.append, asyncio.Event(), None, None)
    stage = await next_transfer.add(dict(id="retry", title="Dependencies", files=[item]))
    await next_transfer.fetch(stage)
    assert len(calls) == 1
    assert events[-1]["manifest"]["files"][0]["status"] == "cached"
    assert events[-1]["downloaded_bytes"] == 0
    assert events[-1]["total_bytes"] == 0
    assert events[-1]["file_downloaded_bytes"] == 0
    catalog.cache_path(paths, item).write_bytes(b"corrupt")
    stage = await next_transfer.add(dict(id="repair", title="Dependencies", files=[dict(item, id="repair-file")]))
    await next_transfer.fetch(stage)
    assert len(calls) == 2


@pytest.mark.parametrize('header_length', [3, 6])
async def test_real_short_download_error_names_manifest_file_and_received_expected_bytes(tmp_path, header_length):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', str(header_length))
            self.end_headers()
            self.wfile.write(b'abc')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    paths = AppPaths(tmp_path)
    item = await catalog.manifest_file(paths, 'original_engine.whl', 'Engine', f'http://127.0.0.1:{server.server_port}/download',
                                       'sha256:' + hashlib.sha256(b'abc').hexdigest(), 6)
    transfer = installers._Transfers(paths, None, asyncio.Event(), None, None)
    stage = await transfer.add(dict(id='short', title='Fixture', files=[item]))
    try:
        with pytest.raises(ValueError) as error:
            await transfer.fetch(stage)
        message = str(error.value)
        assert 'original_engine.whl' in message and '实收 3 B / 应收 6 B' in message and '.part' not in message
        assert stage['files'][0]['status'] == 'error'
        assert not catalog.cache_path(paths, item).exists()
        assert not list(paths.runtimes.rglob('*.part'))
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(1)


@pytest.mark.asyncio
async def test_cancelled_confirmation_never_downloads(tmp_path, monkeypatch):
    cancel = asyncio.Event()
    async def confirm(stage):
        cancel.set()
    transfer = installers._Transfers(AppPaths(tmp_path), None, cancel, None, None)
    with pytest.raises(asyncio.CancelledError):
        await transfer.add(dict(id="deps", title="Dependencies", files=[]), confirm)


@pytest.mark.asyncio
async def test_confirmed_package_report_installs_only_hashed_local_wheels(tmp_path, monkeypatch):
    root = tmp_path / "install"
    root.mkdir()
    paths = AppPaths(tmp_path / "data")
    calls, confirmations, events = [], [], []
    content = b"wheel"
    digest = hashlib.sha256(content).hexdigest()
    async def command(argv, *args, **kwargs):
        calls.append(argv)
        if "--report" in argv:
            await args[0]({"phase": "install", "message": "Downloading pkg.whl.metadata"})
            Path(argv[argv.index("--report") + 1]).write_text(json.dumps({"install": [{"download_info": {"url": "https://files.pythonhosted.org/pkg-1-py3-none-any.whl", "archive_info": {"hashes": {"sha256": digest}}}, "metadata": {"name": "pkg", "version": "1"}}]}))
    async def download(url, target, *args, **kwargs):
        assert confirmations
        target.write_bytes(content)
        await args[0]({"phase": "download", "downloaded": len(content), "total": len(content)})
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def head(self, url):
            assert events[-1]["phase"] == "resolve" and events[-1]["total_bytes"] is None
            return httpx.Response(200, headers={"content-length": str(len(content))})
    async def confirm(stage): confirmations.append(stage)
    monkeypatch.setattr(installers, "_command", command)
    monkeypatch.setattr(installers, "_download", download)
    monkeypatch.setattr(installers.httpx, "AsyncClient", Client)
    transfer = installers._Transfers(paths, events.append, asyncio.Event(), None, None)
    result = await installers._package_stage("python", ["pkg==1"], root, transfer, confirm)
    assert result[0]["version"] == "1"
    install = next(argv for argv in calls if "--require-hashes" in argv)
    assert "--no-index" in install and "--no-deps" in install
    requirements = (root / "confirmed-packages.txt").read_text()
    assert "file:///" in requirements and "--hash=sha256:" + digest in requirements
    assert "https://" not in requirements
    metadata_event = next(e for e in events if e.get('message') == 'Downloading pkg.whl.metadata')
    assert metadata_event['phase'] == 'resolve' and metadata_event['total_bytes'] is None
    assert any(e.get('message') == '正在读取依赖文件大小（1 / 1）' for e in events)
    prepared = next(e for e in events if e.get('phase') == 'prepare')
    assert prepared['total_bytes'] == len(content)
    assert not any(e.get('phase') in {'confirmation', 'awaiting_confirmation'} for e in events)
    assert any(e.get('downloaded_bytes') == len(content) and e.get('total_bytes') == len(content) for e in events)


@pytest.mark.parametrize('legacy,changed_environment', [(False, False), (True, False), (False, True)])
async def test_package_retry_reuses_resolution_and_sizes_but_rechecks_cached_files(tmp_path, monkeypatch, legacy, changed_environment):
    from modloader.backends import resolve_packages
    paths, root = AppPaths(tmp_path / 'data'), tmp_path / 'retry'
    root.mkdir()
    contents = {f'pkg{i}': bytes([i]) * (100 + i) for i in range(6)}
    report = {'environment': {'python_full_version': '3.12.3'}, 'install': [
        {'requested': i == 0, 'metadata': {'name': name, 'version': '1'},
         'download_info': {'url': f'https://files.pythonhosted.org/{name}-1-py3-none-any.whl',
                           'archive_info': {'hashes': {'sha256': hashlib.sha256(data).hexdigest()}}}}
        for i, (name, data) in enumerate(contents.items())]}
    items = resolve_packages.package_files(report)
    for item in items:
        item.update(size_bytes=len(contents[item['package']]), status='complete', cached=True)
    stage = {'id': 'previous', 'title': 'Python 依赖', 'files': items}
    if legacy:
        old = paths.runtimes / 'vllm-1-old'
        old.mkdir()
        (old / 'packages.json').write_text(json.dumps(report))
    else:
        stage['resolution'] = dict(environment=report['environment'], requirements=['pkg0==1'], resolver=hashlib.sha256(resolve_packages.SOURCE.encode()).hexdigest())
    for item in items[:3]:
        path = catalog.cache_path(paths, item)
        path.parent.mkdir(parents=True)
        path.write_bytes(contents[item['package']] if item != items[2] else b'corrupted cached wheel')
    commands, heads, downloads, confirmations, events = [], [], [], [], []
    active, peak, clients = 0, 0, 0
    async def command(argv, *args, **kwargs):
        commands.append(argv)
        if '-c' in argv:
            return json.dumps({'python_full_version': '3.13.0'} if changed_environment else report['environment'])
        if '--report' in argv:
            Path(argv[argv.index('--report') + 1]).write_text(json.dumps(report))
    class Client:
        def __init__(self, **kwargs):
            nonlocal clients
            clients += 1
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def head(self, url):
            nonlocal active, peak
            heads.append(url)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.02)
            active -= 1
            item = next(f for f in items if f['url'] == url)
            return httpx.Response(200, headers={'content-length': str(item['size_bytes'])})
    async def download(url, target, progress, *args, **kwargs):
        assert confirmations
        downloads.append(url)
        item = next(f for f in items if f['url'] == url)
        await asyncio.sleep(.001 * (6 - items.index(item)))
        target.write_bytes(contents[item['package']])
    async def confirm(value): confirmations.append(value)
    monkeypatch.setattr(installers, '_command', command)
    monkeypatch.setattr(installers, '_download', download)
    monkeypatch.setattr(installers.httpx, 'AsyncClient', Client)
    transfer = installers._Transfers(paths, events.append, asyncio.Event(), None, None)
    await installers._package_stage('python', ['pkg0==1'], root, transfer, confirm, retry={'backend': 'vllm', 'version': '1', 'stage': stage})
    assert bool([c for c in commands if '--report' in c]) == changed_environment
    assert clients == 1
    assert set(heads) == ({f['url'] for f in items[2:]} if changed_environment else set())
    if changed_environment:
        assert peak == installers.INSTALL_DOWNLOAD_CONCURRENCY
    assert set(downloads) == {f['url'] for f in items[2:]}
    assert [f['cached'] for f in confirmations[0]['files']] == [True, True, False, False, False, False]
    locked = (root / 'confirmed-packages.txt').read_text().splitlines()
    assert len(locked) == 6
    for line, item in zip(locked, items):
        assert item['name'] in line and f"--hash={item['digest']}" in line
    assert any(e.get('phase') == ('resolve' if changed_environment else 'verify_cache') for e in events)


async def test_pause_at_final_file_does_not_advance_to_install(tmp_path, monkeypatch):
    from modloader.sources.control import DownloadPause
    paused = asyncio.Event()
    async def acknowledged(): paused.set()
    pause, cancel = DownloadPause(acknowledged), asyncio.Event()
    async def progress(event):
        if event.get('completed_files') == 1 and pause.generation == 0:
            pause.request()
    async def download(url, target, *args, **kwargs): target.write_bytes(b'last')
    monkeypatch.setattr(installers, '_download', download)
    paths = AppPaths(tmp_path)
    item = await catalog.manifest_file(paths, 'last.whl', 'Last', 'https://files.pythonhosted.org/last.whl', 'sha256:' + hashlib.sha256(b'last').hexdigest(), 4)
    transfer = installers._Transfers(paths, progress, cancel, pause, None)
    stage = await transfer.add(dict(id='last', title='Last', files=[item]))
    task = asyncio.create_task(transfer.fetch(stage))
    try:
        await asyncio.wait_for(paused.wait(), 2)
        assert not task.done()
        pause.resume()
        await asyncio.wait_for(task, 2)
    finally:
        cancel.set()
        pause.resume()
        await asyncio.gather(task, return_exceptions=True)


def test_managed_owner_validates_without_removing_files(tmp_path):
    paths = AppPaths(tmp_path)
    root = paths.runtimes / "owned"
    root.mkdir()
    binary = root / "llama-server.exe"
    binary.touch()
    runtime = RuntimeInstallation(id="owned", backend="llama_cpp", name="Owned", path=str(binary), managed=True, metadata={"managed_root": str(root)})
    (root / "modloader-runtime.json").write_text(runtime.model_dump_json(), encoding="utf-8")
    assert installers.validate_managed_runtime(runtime, paths) == root
    assert binary.is_file()
    runtime.path = str(tmp_path / "external.exe")
    with pytest.raises(ValueError, match="不匹配"):
        installers.validate_managed_runtime(runtime, paths)


def test_metadata_resolver_refuses_payload_and_missing_metadata(monkeypatch):
    from modloader.backends.resolve_packages import SOURCE
    from pip._internal.network.session import PipSession
    from pip._internal.operations.prepare import RequirementPreparer
    import pip._internal.cli.main as cli
    namespace = {"__name__": "resolver_test"}
    exec(SOURCE, namespace)
    originals = {name: getattr(RequirementPreparer, name) for name in ("prepare_linked_requirement", "_prepare_linked_requirement", "prepare_linked_requirements_more")}
    request = PipSession.request
    def fake_main(argv):
        session = PipSession()
        with pytest.raises(RuntimeError, match="禁止下载"):
            session.request("GET", "https://files.pythonhosted.org/pkg.whl")
        from types import SimpleNamespace
        preparer = SimpleNamespace(_fetch_metadata_using_link_data_attr=lambda req: None)
        req = SimpleNamespace(link=SimpleNamespace(is_wheel=True, is_file=False, hash="a"*64, hash_name="sha256"), name="pkg")
        with pytest.raises(RuntimeError, match="PEP 658"):
            RequirementPreparer.prepare_linked_requirement(preparer, req)
        return 0
    monkeypatch.setattr(cli, "main", fake_main)
    try:
        assert namespace["main"](["pkg"]) == 0
    finally:
        PipSession.request = request
        for name, function in originals.items():
            setattr(RequirementPreparer, name, function)


@pytest.mark.asyncio
async def test_external_strata_validation_does_not_execute_project(tmp_path, monkeypatch):
    root = tmp_path / "external"
    (root / "serve").mkdir(parents=True)
    (root / "engine").mkdir()
    (root / "serve/server.py").write_text("raise AssertionError('must not execute')")
    (root / "engine/strata.exe").touch()
    (root / "python.exe").touch()
    (root / "engine/BUILD.json").write_text('{"version":"0.1.44.1"}')
    runtime = RuntimeInstallation(id="external", backend="strata", name="External", path=str(root), metadata={"python": str(root / "python.exe")})
    async def forbidden(*args, **kwargs): raise AssertionError("external project executed")
    monkeypatch.setattr(installers, "_command", forbidden)
    result = await installers.validate_runtime(runtime)
    assert result.version == "0.1.44.1" and result.id == runtime.id
    assert not list(root.rglob("__pycache__"))


async def test_strata_install_validates_its_actual_entrypoint_and_required_imports(tmp_path):
    root = tmp_path / 'strata'
    (root / 'serve').mkdir(parents=True)
    (root / 'engine').mkdir()
    (root / 'serve/__init__.py').touch()
    (root / 'serve/server.py').write_text('import argparse\nargparse.ArgumentParser().parse_args()\n')
    (root / 'engine/strata.exe').touch()
    (root / 'engine/BUILD.json').write_text('{"version":"0.1.44.1"}')
    for module in ('fastapi', 'uvicorn'):
        (root / (module + '.py')).write_text("raise RuntimeError('Strata does not require this web framework')")
    runtime = installers.import_runtime('strata', str(root), {'python': sys.executable})
    result = await installers.validate_runtime(runtime, _installing=True)
    assert result.version == '0.1.44.1'
    assert result.metadata['validation'] == 'entrypoint'
    (root / 'serve/server.py').write_text('import missing_strata_dependency\n')
    with pytest.raises(RuntimeError, match='missing_strata_dependency'):
        await installers.validate_runtime(runtime, _installing=True)


def test_wsl_installed_base_packages_skip_apt_and_updates(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    report = tmp_path / "apt.json"
    calls = []
    def installed(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "dpkg-query"
        return SimpleNamespace(returncode=0, stdout="ii  1.2.3")
    monkeypatch.setattr(subprocess, "run", installed)
    monkeypatch.setattr(sys, "argv", ["resolve-apt.py", str(report), ""])
    with pytest.raises(SystemExit) as result:
        exec(installers._APT_RESOLVE, {})
    assert result.value.code == 0 and json.loads(report.read_text()) == []
    assert len(calls) == 3


def test_source_commit_blob_check_rejects_modified_or_extra_files(tmp_path):
    content = b"print('official source')\n"
    source = tmp_path / "server.py"
    source.write_bytes(content)
    expected = {"server.py": hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()}
    installers._verify_source(tmp_path, expected, asyncio.Event())
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="blob"):
        installers._verify_source(tmp_path, expected, asyncio.Event())
    source.write_bytes(content)
    (tmp_path / "extra.py").touch()
    with pytest.raises(ValueError, match="不一致"):
        installers._verify_source(tmp_path, expected, asyncio.Event())


def test_source_verification_names_interrupted_and_missing_file(tmp_path, monkeypatch):
    import os
    source = tmp_path / 'tool_flow_e2e.py'
    data = b'# fixture, never executed\n'
    source.write_bytes(data)
    expected = {source.name: hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()}
    original_open = Path.open
    def interrupted(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == source:
            os.close(stream.fileno())
        return stream
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', interrupted)
        with pytest.raises(ValueError) as error:
            installers._verify_source(tmp_path, expected, asyncio.Event())
        assert str(source) in str(error.value) and '安全软件' in str(error.value)
        assert '系统错误 9' in str(error.value)
    installers._verify_source(tmp_path, expected, asyncio.Event())
    source.unlink()
    with pytest.raises(ValueError) as error:
        installers._verify_source(tmp_path, expected, asyncio.Event())
    assert '缺少文件' in str(error.value) and str(source) in str(error.value)


def test_archive_write_interruption_names_destination(tmp_path, monkeypatch):
    import os
    import zipfile
    archive, destination = tmp_path / 'source.zip', tmp_path / 'extract'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('script.py', b'# fixture, never executed\n')
    target = destination / 'script.py'
    original_open = Path.open
    def interrupted(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == target:
            os.close(stream.fileno())
        return stream
    monkeypatch.setattr(Path, 'open', interrupted)
    with pytest.raises(ValueError) as error:
        installers._extract(archive, destination)
    assert str(target) in str(error.value) and '系统错误 9' in str(error.value)


@pytest.mark.parametrize('suffix', ['.bat', '.cmd', '.ps1'])
def test_source_archive_honors_verified_script_line_endings_without_accepting_other_changes(tmp_path, suffix):
    def blob(data): return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    attributes = ('*' + suffix + ' text eol=crlf\n').encode()
    source = b'@echo off\necho official\n'
    (tmp_path / '.gitattributes').write_bytes(attributes)
    script = tmp_path / 'nested' / ('setup' + suffix)
    script.parent.mkdir()
    script.write_bytes(source.replace(b'\n', b'\r\n'))
    expected = {'.gitattributes': blob(attributes), 'nested/setup' + suffix: blob(source)}
    installers._verify_source(tmp_path, expected, asyncio.Event())
    assert script.read_bytes() == source.replace(b'\n', b'\r\n')
    script.write_bytes(b'@echo off\r\necho tampered\r\n')
    with pytest.raises(ValueError, match='nested/setup'):
        installers._verify_source(tmp_path, expected, asyncio.Event())
    script.write_bytes(source.replace(b'\n', b'\r\n'))
    (tmp_path / '.gitattributes').write_bytes(('*' + suffix + ' -text\n').encode())
    with pytest.raises(ValueError, match='gitattributes'):
        installers._verify_source(tmp_path, expected, asyncio.Event())
    (tmp_path / '.gitattributes').write_bytes(attributes)
    override = ('*' + suffix + ' -text\n').encode()
    (script.parent / '.gitattributes').write_bytes(override)
    expected['nested/.gitattributes'] = blob(override)
    with pytest.raises(ValueError, match='nested/setup'):
        installers._verify_source(tmp_path, expected, asyncio.Event())
    script.write_bytes(source)
    binary = tmp_path / 'raw.dat'
    binary.write_bytes(b'binary\r\n')
    expected['raw.dat'] = blob(b'binary\n')
    with pytest.raises(ValueError, match='raw.dat'):
        installers._verify_source(tmp_path, expected, asyncio.Event())


@pytest.mark.asyncio
async def test_foreign_wsl_owner_rejected_before_payload_or_install_directory(tmp_path, monkeypatch):
    paths = AppPaths(tmp_path)
    def foreign(*args): raise ValueError("缺少专属 WSL 发行版所有权记录")
    monkeypatch.setattr(installers, 'list_wsl', lambda: {'available': True, 'distributions': [{'name': 'ModLoader', 'version': 2}]})
    monkeypatch.setattr(installers, "_verify_wsl_owner", foreign)
    plan = {"id": "frozen", "backend": "vllm", "version": "0.31.0", "variant": "wsl2", "stage": {"id": "base", "files": []}}
    with pytest.raises(ValueError, match="所有权"):
        await installers.install_backend("vllm", paths, {"_plan": plan}, None, asyncio.Event())
    assert not list(paths.runtimes.iterdir())


@pytest.mark.asyncio
async def test_transformers_v4_catalog_is_unverified_instead_of_arbitrarily_excluded(monkeypatch):
    async def get_json(url, proxy=None):
        return {"releases": {"4.57.3": [{"packagetype": "bdist_wheel", "digests": {"sha256": "a" * 64}}]}}
    monkeypatch.setattr(catalog, "_json", get_json)
    result = await catalog.list_versions("transformers")
    assert result["versions"][0]["version"] == "4.57.3"
    assert result["versions"][0]["status"] == "unverified"
    assert result["versions"][0]["variants"]


@pytest.mark.asyncio
@pytest.mark.parametrize("info", [{"provides_extra": ["serving"]}, {"requires_dist": ['fastapi; extra == "serving"']}])
async def test_transformers_v4_plan_uses_official_serving_metadata(tmp_path, monkeypatch, info):
    async def get_json(url, proxy=None):
        assert "/4.57.3/json" in url
        return {"info": info, "urls": [{"packagetype": "bdist_wheel", "digests": {"sha256": "a" * 64}}]}
    monkeypatch.setattr(catalog, "_json", get_json)
    plan = await catalog.plan_install("transformers", AppPaths(tmp_path), {"version": "4.57.3"})
    assert plan["version"] == "4.57.3"
    assert plan["variant"] == "cuda-13"


@pytest.mark.asyncio
async def test_transformers_plan_rejects_absent_serving_metadata_regardless_of_major(tmp_path, monkeypatch):
    async def get_json(url, proxy=None):
        return {"info": {"provides_extra": ["torch"]}, "urls": [{"packagetype": "bdist_wheel", "digests": {"sha256": "a" * 64}}]}
    monkeypatch.setattr(catalog, "_json", get_json)
    with pytest.raises(ValueError, match="未声明 serving"):
        await catalog.plan_install("transformers", AppPaths(tmp_path), {"version": "5.19.0"})
