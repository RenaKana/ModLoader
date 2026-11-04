import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from modloader.backends import catalog, installers


DEEPSEEK_024_SOURCE = 'dfa3f4025c9f1bc4af8cee201b20a2d223061813'


def release_asset(name, content=b'engine', *, version='0.2.1'):
    return dict(name=name, digest='sha256:' + hashlib.sha256(content).hexdigest(), size=len(content),
                browser_download_url=f'https://github.com/Niko1221/Strata/releases/download/v{version}/' + name)


async def test_strata_catalog_lists_split_engines_and_keeps_legacy(monkeypatch):
    names = [
        'strata-engine-0.1.46-windows-x86_64-cuda-cuda13.tar.gz',
        'strata-engine-0.1.46-windows-x86_64-cuda-cuda12.tar.gz',
        'strata-engine-0.1.46-windows-x86_64-hip-rx9070.tar.gz',
        'strata-engine-0.1.46.1-windows-x86_64-hip-rdna3.tar.gz',
        'strata-control-0.1.46.3-windows-x86_64.zip',
        'strata-engine-0.1.46-linux-x86_64-cuda-cuda13.tar.gz',
        'Strata-0.2.1-windows-cuda13.exe',
    ]
    releases = [dict(tag_name='v0.2.1', assets=[release_asset(name) for name in names]),
                dict(tag_name='v0.1.44.1', assets=[release_asset('strata-windows-x64.zip')]),
                dict(tag_name='v0.2.0-beta', prerelease=True, assets=[release_asset(names[0])]),
                dict(tag_name='v0.2.2', assets=[dict(name=names[0])]),
                dict(tag_name='v0.2.10', assets=[release_asset(
                    'strata-engine-0.1.56-windows-x86_64-cuda-cuda13.tar.gz', version='0.2.10')])]
    async def request(url, proxy=None):
        return httpx.Response(200, request=httpx.Request('GET', url), json=releases)
    monkeypatch.setattr(catalog, '_request', request)
    versions = (await catalog.list_versions('strata'))['versions']
    assert [choice['id'] for choice in versions[0]['variants']] == ['cuda13', 'cuda12', 'hip-rx9070', 'hip-rdna3']
    assert not versions[0]['recommended'] and versions[0]['status'] == 'available'
    assert [item['version'] for item in versions if item['recommended']] == ['0.2.10']
    assert versions[1]['variants'] == [{'id': 'windows-x64', 'label': 'Windows x64'}]
    assert versions[2]['status'] == 'unverified'
    assert versions[3]['status'] == 'unavailable'
    hotfix = release_asset('strata-engine-0.1.46.2-windows-x86_64-hip-rdna3.tar.gz')
    releases[0]['assets'].insert(0, hotfix)
    selected = catalog._strata_assets(releases[0])['hip-rdna3']
    assert selected['asset'] == hotfix and selected['engine_version'] == '0.1.46.2'


async def test_strata_catalog_keeps_deepseek_hotfix_and_qwen_versions_separate(tmp_path, monkeypatch):
    from modloader.models import AppPaths

    qwen = release_asset('strata-engine-0.1.48-windows-x86_64-cuda-cuda13.tar.gz', version='0.2.4')
    deepseek = release_asset('strata-dsv41-0.1.49.2-windows-x86_64-cuda-cuda13.tar.gz', version='0.2.4')
    release = dict(id=418129260, tag_name='v0.2.4', assets=[
        deepseek,
        release_asset('strata-dsv41-0.1.49-windows-x86_64-cuda-cuda13.tar.gz'),
        qwen,
        release_asset('strata-dsv41-0.1.49.1-windows-x86_64-cuda-cuda13.tar.gz'),
        release_asset('strata-dsv41-0.1.99-linux-x86_64-cuda-cuda13.tar.gz'),
        release_asset('strata-dsv41-0.1.99-windows-x86_64-cuda-cuda12.tar.gz'),
        dict(name='strata-dsv41-0.1.99-windows-x86_64-cuda-cuda13.tar.gz'),
        release_asset('strata-control-0.1.49.2-windows-x86_64.zip'),
    ])
    commit = '9' * 40
    expected_source = commit
    async def metadata(url, proxy=None):
        if '/releases/tags/' in url:
            return release
        if '/git/ref/tags/' in url:
            assert url.endswith('/' + release['tag_name'])
            return {'object': {'type': 'commit', 'sha': commit}}
        assert url.endswith(f'/git/trees/{expected_source}?recursive=1')
        return dict(tree=[], truncated=False)
    monkeypatch.setattr(catalog, '_json', metadata)

    variants = catalog._variants('strata', release)
    assert [item['id'] for item in variants] == ['cuda13', 'dsv41-cuda13']
    assert variants[1]['label'] == 'DeepSeek V4.1 · CUDA 13 · NVIDIA'
    selected = catalog._strata_assets(release)
    assert selected['cuda13']['engine_version'] == '0.1.48'
    assert selected['dsv41-cuda13']['asset'] == deepseek
    assert selected['dsv41-cuda13']['engine_version'] == '0.1.49.2'

    paths = AppPaths(tmp_path / 'data')
    options = {'version': '0.2.4', 'python': sys.executable}
    default = await catalog.plan_install('strata', paths, options)
    assert default['variant'] == 'cuda13'
    assert default['provenance']['engine_version'] == '0.1.48'
    assert default['provenance']['source_commit'] == commit
    expected_source = DEEPSEEK_024_SOURCE
    plan = await catalog.plan_install('strata', paths, {**options, 'variant': 'dsv41-cuda13'})
    assert plan['stage']['files'][0]['name'] == deepseek['name']
    assert plan['provenance']['engine_version'] == '0.1.49.2'
    assert plan['provenance']['tag'] == 'v0.2.4'
    assert plan['provenance']['source_commit'] == DEEPSEEK_024_SOURCE
    assert plan['stage']['files'][1]['url'].endswith('/' + DEEPSEEK_024_SOURCE)
    assert plan['stage']['files'][1]['source_commit'] == DEEPSEEK_024_SOURCE
    assert catalog.RECOMMENDED['strata'] == '0.2.10'
    expected_source = commit
    release['assets'] = [deepseek, release_asset('strata-windows-x64.zip')]
    legacy = await catalog.plan_install('strata', paths, options)
    assert legacy['variant'] == 'windows-x64'
    assert 'engine_version' not in legacy['provenance']
    assert legacy['provenance']['source_commit'] == commit
    release['tag_name'] = 'v0.2.5'
    other = await catalog.plan_install('strata', paths, {**options, 'version': '0.2.5', 'variant': 'dsv41-cuda13'})
    assert other['provenance']['source_commit'] == commit


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows managed installation')
@pytest.mark.parametrize('variant,flavor,engine_version,layout', [
    ('cuda13', 'cuda-cuda13', '0.1.46', 'flat'),
    ('cuda12', 'cuda-cuda12', '0.1.46', 'flat'),
    ('hip-rx9070', 'hip-rx9070', '0.1.46', 'flat'),
    ('hip-rdna3', 'hip-rdna3', '0.1.46.1', 'flat'),
    ('windows-x64', None, '0.1.44.1', 'flat'),
    ('dsv41-cuda13', 'cuda-cuda13', '0.1.49.2', 'flat'),
    ('dsv41-cuda13', 'cuda-cuda13', '0.1.49.2', 'version-folder'),
    ('dsv41-cuda13', 'cuda-cuda13', '0.1.49.2', 'wrong-engine'),
    ('dsv41-cuda13', 'cuda-cuda13', '0.1.49.2', 'wrong-version'),
])
async def test_strata_plan_download_unpack_register_and_launch_config(tmp_path, monkeypatch, variant, flavor, engine_version, layout):
    """Real local HTTP/archive/entrypoint flow; engine binaries and dependency provisioning are fixtures."""
    from modloader.backends import get_adapters
    from modloader.models import AppPaths, LaunchContext, LoadProfile, ModelAsset
    from test_strata_preparation import write_expert_profile

    paths = AppPaths(tmp_path / 'data')
    tag_commit = '9' * 40
    deepseek = variant == 'dsv41-cuda13'
    commit = DEEPSEEK_024_SOURCE if deepseek else tag_commit
    release_version = '0.2.4' if deepseek else '0.2.1' if flavor else '0.1.44.1'
    source_files = {
        'serve/__init__.py': b'',
        'serve/server.py': b'import argparse\nargparse.ArgumentParser().parse_args()\n',
        'requirements.txt': b'numpy>=1.24\n',
    }
    tree = [dict(path=name, type='blob', sha=hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest())
            for name, data in source_files.items()]
    source_buffer = io.BytesIO()
    with zipfile.ZipFile(source_buffer, 'w') as archive:
        for name, data in source_files.items():
            archive.writestr('Strata-' + commit + '/' + name, data)

    build = dict(version='0.1.49.1' if layout == 'wrong-version' else engine_version)
    if variant.startswith('hip-'):
        build.update(backend='hip', lib_dirs=['rocm/bin'])
    executable = 'strata-dsv41.exe' if deepseek and layout != 'wrong-engine' else 'strata.exe'
    engine_files = {'BUILD.json': json.dumps(build).encode(), executable: b'fixture engine'}
    library = 'rocm/bin/amdhip64.dll' if variant.startswith('hip-') else 'cublas64_13.dll' if variant in {'cuda13', 'dsv41-cuda13'} else 'cublas64_12.dll'
    if flavor:
        engine_files[library] = b'fixture runtime library'
    engine_buffer = io.BytesIO()
    if flavor:
        engine_name = f'strata-{"dsv41" if deepseek else "engine"}-{engine_version}-windows-x86_64-{flavor}.tar.gz'
        with tarfile.open(fileobj=engine_buffer, mode='w:gz') as archive:
            for name, data in engine_files.items():
                info = tarfile.TarInfo((engine_version + '-windows-cuda/' if layout == 'version-folder' else '') + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    else:
        engine_name = 'strata-windows-x64.zip'
        with zipfile.ZipFile(engine_buffer, 'w') as archive:
            for name, data in engine_files.items():
                archive.writestr(name, data)
    asset = release_asset(engine_name, engine_buffer.getvalue(), version=release_version)
    payloads = {'/engine': engine_buffer.getvalue(), '/source': source_buffer.getvalue()}
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            requests.append(self.path)
            content = payloads[self.path]
            self.send_response(200)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def metadata(url, proxy=None):
        if '/releases/tags/' in url:
            return dict(id=418129260, assets=[release_asset('strata-engine-0.1.48-windows-x86_64-cuda-cuda13.tar.gz'), asset] if deepseek else [asset])
        if '/git/ref/tags/' in url:
            return {'object': {'type': 'commit', 'sha': tag_commit}}
        assert url.endswith(f'/git/trees/{commit}?recursive=1')
        return dict(tree=tree, truncated=False)
    original_download = installers._download
    async def download(url, *args, **kwargs):
        route = {asset['browser_download_url']: '/engine', f'https://codeload.github.com/Niko1221/Strata/zip/{commit}': '/source'}[url]
        return await original_download(f'http://127.0.0.1:{server.server_port}' + route, *args, **kwargs)
    requirements_seen = []
    async def venv(*args): return sys.executable
    async def packages(python, requirements, *args, **kwargs):
        requirements_seen.extend(requirements)
        return []
    monkeypatch.setattr(catalog, '_json', metadata)
    monkeypatch.setattr(installers, '_download', download)
    monkeypatch.setattr(installers, '_approved_venv', venv)
    monkeypatch.setattr(installers, '_package_stage', packages)
    try:
        plan = await catalog.plan_install('strata', paths, {'version': release_version, 'variant': variant, 'python': sys.executable})
        assert requests == []
        assert [item['role'] for item in plan['stage']['files']] == ['engine', 'source']
        assert plan['stage']['files'][0]['name'] == engine_name
        assert plan['provenance'].get('engine_version') == (engine_version if flavor else None)
        assert plan['provenance']['tag'] == 'v' + release_version
        assert plan['provenance']['source_commit'] == commit
        assert plan['stage']['files'][1]['source_commit'] == commit
        events = []
        if layout in {'wrong-engine', 'wrong-version'}:
            with pytest.raises(ValueError):
                await installers.install_backend('strata', paths, {'_plan': plan}, events.append, asyncio.Event())
            assert sorted(requests) == ['/engine', '/source']
            assert not list(paths.runtimes.glob('strata-*/modloader-runtime.json'))
            assert all(event.get('phase') != 'complete' for event in events)
            if layout == 'wrong-engine':
                assert requirements_seen == []
            return
        runtime = await installers.install_backend('strata', paths, {'_plan': plan}, events.append, asyncio.Event())
        assert sorted(requests) == ['/engine', '/source']
        assert runtime.version == engine_version and runtime.managed
        assert runtime.metadata['requested_version'] == release_version
        assert runtime.metadata['provenance']['source_commit'] == commit
        root = Path(runtime.path)
        engine_root = root / ('engine-dsv41' if deepseek else 'engine')
        if layout == 'version-folder':
            engine_root /= engine_version + '-windows-cuda'
        assert (engine_root / executable).read_bytes() == b'fixture engine'
        if flavor:
            assert (engine_root / library).is_file()
            assert requirements_seen == ['-r', str(root / 'requirements.txt')]
        else:
            assert 'nvidia-cublas==13.0.2.14' in requirements_seen
            assert 'nvidia-cuda-runtime==13.0.96' in requirements_seen
        assert runtime.metadata['validation'] == 'entrypoint'
        managed = installers.validate_managed_runtime(runtime, paths)
        assert (managed / 'modloader-runtime.json').is_file()
        assert events[-1]['phase'] == 'complete'
        if deepseek:
            assert not (root / 'engine').exists()
            assert runtime.metadata['strata_family'] == 'deepseek41'
            assert runtime.metadata['engine_backend'] == 'cuda'
            assert runtime.metadata['provenance']['engine_version'] == engine_version
            assert str(engine_root) in runtime.metadata['lib_dirs']
            return

        model_path = tmp_path / 'model.gguf'
        model_path.write_bytes(b'GGUF')
        model = ModelAsset(id='model', name='fixture', path=str(model_path), format='gguf',
                           metadata={'qwen4exp.block_count': 48, 'qwen4exp.expert_count': 512})
        tokenizer, pack = tmp_path / 'tokenizer', tmp_path / 'pack'
        tokenizer.mkdir()
        pack.mkdir()
        (tokenizer / 'vocab.json').write_text('{}')
        write_expert_profile(root / 'data/expert-profile.bin')
        profile = LoadProfile(id='profile', name='fixture', model_id=model.id, backend='strata', runtime_id=runtime.id, alias='fixture',
                              parameters={'pack': str(pack), 'tokenizer': str(tokenizer),
                                          'strata_config': {'lib_dirs': [str(tmp_path / 'custom-libs')]}})
        context = LaunchContext(port=18764, instance_id='fixture', work_dir=str(tmp_path / 'instance'), data_dir=str(paths.root))
        launch = get_adapters()['strata'].build_launch(model, profile, runtime, context)
        config = json.loads((Path(context.work_dir) / 'strata.json').read_text())
        assert launch.argv[:5] == [sys.executable, '-m', 'serve.server', '--engine', 'strata']
        assert config['exe'] == str(root / 'engine/strata.exe')
        assert config['lib_dirs'][:2] == [str(tmp_path / 'custom-libs'), str(root / 'engine')]
        if flavor:
            assert config['backend'] == ('hip' if variant.startswith('hip-') else 'cuda')
        if variant.startswith('hip-'):
            assert str(root / 'engine/rocm/bin') in config['lib_dirs']
        if variant == 'cuda13':
            (root / 'engine/BUILD.json').write_text('{"version":"0.1.45"}')
            with pytest.raises(ValueError, match='引擎实际版本'):
                await installers.validate_runtime(runtime, paths=paths)
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(1)
