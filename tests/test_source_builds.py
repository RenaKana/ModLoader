import asyncio
import ast
import io
import json
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from modloader.backends import source_builds as builds, catalog, installers
from modloader.models import AppPaths, RuntimeInstallation


def test_embedded_guest_helpers_compile_and_cmake_network_fetches_are_disabled():
    compile(builds.STAGE_SOURCE, '<stage-native>', 'exec')
    module = ast.parse(Path(builds.__file__).read_text(encoding='utf-8'))
    helpers = [node.value for node in ast.walk(module) if isinstance(node, ast.Constant)
               and isinstance(node.value, str) and node.value.startswith('import ')]
    assert len(helpers) == 2  # Plus STAGE_SOURCE above, whose leading newline is intentional.
    for code in helpers:
        compile(code, '<guest-helper>', 'exec')
    options = builds.spark_cmake_options('/opt/modloader/fixture')
    for dependency in ('NLOHMANN_JSON', 'TOKENIZERS_CPP', 'RE2', 'XGRAMMAR', 'DLPACK', 'CUTLASS'):
        assert any(value.startswith('-DFETCHCONTENT_SOURCE_DIR_' + dependency + '=') for value in options)
    assert '-DSPARKINFER_NVFP4=ON' in options
    assert builds.build_environment('/opt/modloader/fixture')['CARGO_NET_OFFLINE'] == 'true'
    assert builds.build_environment('/opt/modloader/fixture')['TORCH_EXTENSIONS_DIR'] == '/opt/modloader/fixture/torch-extensions'
    assert builds.build_environment('/opt/modloader/fixture')['MAX_JOBS'] == '4'
    assert builds.BUILD_JOBS == 4


def test_source_extractor_rejects_paths_outside_isolated_stage(tmp_path):
    module = ast.parse(builds.STAGE_SOURCE)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'extract')
    namespace = dict(zipfile=zipfile, tarfile=tarfile, stat=stat)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<extract-fixture>', 'exec'), namespace)
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as archive:
        archive.writestr('../outside', 'must not escape')
    with pytest.raises(ValueError, match='Invalid archive path'):
        namespace['extract'](bad, tmp_path / 'payload')
    assert not (tmp_path / 'outside').exists()


def test_cuda_redist_layout_preserves_lib_and_exposes_nvcc_lib64_search(tmp_path, monkeypatch):
    module = ast.parse(builds.STAGE_SOURCE)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'cuda_layout')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<cuda-layout-fixture>', 'exec'), namespace)
    cuda = tmp_path / 'cuda'
    library = cuda / 'lib'
    library.mkdir(parents=True)
    for name in ('libcudart_static.a', 'libcudadevrt.a'):
        (library / name).write_bytes(b'approved archive fixture')
    links = []
    monkeypatch.setattr(Path, 'symlink_to', lambda path, target, **kwargs: links.append((path, target, kwargs)))
    namespace['cuda_layout'](cuda)
    assert links == [(cuda / 'lib64', 'lib', {'target_is_directory': True})]
    assert (library / 'libcudart_static.a').read_bytes() == b'approved archive fixture'
    (cuda / 'lib64').mkdir()
    namespace['cuda_layout'](cuda)
    assert len(links) == 1  # Do not replace an existing layout.
    (library / 'libcudadevrt.a').unlink()
    with pytest.raises(ValueError, match='static runtime'):
        namespace['cuda_layout'](cuda)


@pytest.mark.parametrize('component,header', [('libcurand', 'curand_kernel.h'), ('cuda_profiler_api', 'cuda_profiler_api.h'),
                                            ('cuda_nvtx', 'nvtx3/nvToolsExt.h')])
def test_cuda_delta_stage_merges_headers_without_restaging_source_or_build(tmp_path, component, header):
    root = tmp_path / 'owned-root'
    cuda = root / 'cuda'
    (cuda / 'lib').mkdir(parents=True)
    (cuda / 'lib64').mkdir()  # Existing layout avoids Windows symlink privileges in this fixture.
    for name in ('libcudart_static.a', 'libcudadevrt.a'):
        (cuda / 'lib' / name).write_bytes(b'previously approved CUDA runtime')
    built = root / 'source/build/previous-kernel.o'
    built.parent.mkdir(parents=True)
    built.write_bytes(b'keep compiled output')
    archive = tmp_path / (component + '.tar.xz')
    content = b'// official component header fixture\n'
    with tarfile.open(archive, 'w:xz') as tar:
        member = tarfile.TarInfo(component + '-linux-x86_64-archive/include/' + header)
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    namespace = dict(root=root, request={'archives': [{'archive': str(archive), 'destination': 'cuda'}]})
    exec('import hashlib,json,os,pathlib,shutil,stat,sys,tarfile,tempfile,zipfile', namespace)
    module = ast.parse(builds.STAGE_SOURCE)
    # Run the actual staging functions and archive loop, substituting only the /opt root boundary.
    nodes = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.For))]
    nodes += [node for node in module.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == 'cuda_layout']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<cuda-delta-stage>', 'exec'), namespace)
    assert (cuda / 'include' / header).read_bytes() == content
    assert (cuda / 'lib/libcudart_static.a').read_bytes() == b'previously approved CUDA runtime'
    assert built.read_bytes() == b'keep compiled output'


async def test_ninfer_plan_fixed_source_toolchain_manifest_and_no_payload(tmp_path, monkeypatch):
    paths = AppPaths(tmp_path)
    (paths.runtimes / 'ModLoader-wsl-owned.json').write_text('{}')
    urls = []
    async def metadata(url, proxy=None):
        urls.append(url)
        if '/git/trees/' in url:
            return {'tree': [{'type': 'blob', 'path': 'CMakeLists.txt', 'sha': 'b' * 40, 'mode': '100644'}]}
        return {name: {'linux-x86_64': {'relative_path': name + '/linux-x86_64/archive.tar.xz', 'sha256': 'a' * 64, 'size': 42}}
                for name in builds.ENGINE_CUDA_COMPONENTS['ninfer']}
    monkeypatch.setattr(catalog, '_json', metadata)
    plan = await builds.plan_install('ninfer', paths, {'version': builds.PINS['ninfer']['version']})
    assert plan['provenance']['source_commit'] == builds.PINS['ninfer']['commit']
    assert len(plan['stage']['files']) == 1 + len(builds.ENGINE_CUDA_COMPONENTS['ninfer'])
    assert plan['provenance']['cuda_components'] == [*builds.CUDA_COMPONENTS, 'cuda_nvtx']
    assert all(item.get('digest') or item.get('source_commit') for item in plan['stage']['files'])
    assert plan['stage']['files'][0]['url'].endswith(builds.PINS['ninfer']['commit'])
    assert any('redistrib_13.4.3.json' in url for url in urls)
    assert list(paths.models.iterdir()) == []
    with pytest.raises(ValueError):
        await builds.plan_install('ninfer', paths, {'version': 'master'})


@pytest.mark.parametrize('version,commit', [
    (None, 'a754fcf3bf524494aa927c2f54e0cbfb58f569ab'),
    ('a754fcf3bf52', 'a754fcf3bf524494aa927c2f54e0cbfb58f569ab'),
    ('64d10fbf9f8e', '64d10fbf9f8e6bfc79ab0570b94674c5b9997929'),
])
async def test_ninfer_version_choice_pins_new_default_and_keeps_legacy_recipe(tmp_path, monkeypatch, version, commit):
    paths = AppPaths(tmp_path)
    (paths.runtimes / 'ModLoader-wsl-owned.json').write_text('{}')
    async def metadata(url, proxy=None):
        if '/git/trees/' in url:
            assert commit in url
            return {'tree': [{'type': 'blob', 'path': 'CMakeLists.txt', 'sha': 'b' * 40, 'mode': '100644'}]}
        return {name: {'linux-x86_64': {'relative_path': name + '/linux-x86_64/archive.tar.xz',
                                       'sha256': 'a' * 64, 'size': 42}}
                for name in builds.ENGINE_CUDA_COMPONENTS['ninfer']}
    monkeypatch.setattr(catalog, '_json', metadata)
    plan = await builds.plan_install('ninfer', paths, {} if version is None else {'version': version})
    assert plan['provenance']['source_commit'] == commit
    assert plan['version'] == commit[:12] and plan['options']['version'] == commit[:12]
    assert builds.PINS['ninfer']['commit'] == '64d10fbf9f8e6bfc79ab0570b94674c5b9997929'


@pytest.mark.parametrize('commit,jpeg', [
    ('a754fcf3bf524494aa927c2f54e0cbfb58f569ab', True),
    ('64d10fbf9f8e6bfc79ab0570b94674c5b9997929', False),
])
async def test_ninfer_build_adds_jpeg_only_for_new_source_recipe(tmp_path, monkeypatch, commit, jpeg):
    packages = []
    async def command(*args, **kwargs): pass
    async def apt(*args, **kwargs): packages.extend(kwargs['packages'])
    async def python_packages(*args, **kwargs): return []
    from modloader import platforms
    monkeypatch.setattr(platforms, 'windows_to_wsl', lambda value: '/mounted/' + Path(value).name)
    monkeypatch.setattr(installers, '_wsl_command', command)
    monkeypatch.setattr(installers, '_apt_stage', apt)
    monkeypatch.setattr(installers, '_package_stage', python_packages)
    artifacts = {'cuda_' + name: tmp_path / name for name in builds.ENGINE_CUDA_COMPONENTS['ninfer']}
    transfer = SimpleNamespace(progress=None, cancel=asyncio.Event())
    await builds.install('ninfer', tmp_path, '/opt/modloader/fixture', artifacts,
                         {'provenance': {'source_commit': commit, 'sources': []}}, transfer, None, None)
    assert ('libjpeg-dev' in packages) is jpeg


async def test_ninfer_nvtx_delta_has_official_identity_and_does_not_add_spark_components(tmp_path, monkeypatch):
    entry = {'relative_path': 'cuda_nvtx/linux-x86_64/cuda_nvtx-linux-x86_64-13.4.92-archive.tar.xz',
             'sha256': 'd767ac8b8bb40f5d58602ce900e8bf2fe073c1ebb4a7d405ff67674009fa83ce', 'size': '98560'}
    async def metadata(url, proxy=None):
        assert url == builds.CUDA_BASE + 'redistrib_13.4.3.json'
        assert proxy == 'http://approved:17890'
        return {'cuda_nvtx': {'linux-x86_64': entry}}
    monkeypatch.setattr(catalog, '_json', metadata)
    files = await builds.cuda_files(AppPaths(tmp_path), 'http://approved:17890', components=('cuda_nvtx',))
    assert len(files) == 1 and files[0]['role'] == 'cuda_cuda_nvtx'
    assert files[0]['url'] == builds.CUDA_BASE + entry['relative_path'] and files[0]['size_bytes'] == 98_560
    assert files[0]['digest'] == 'sha256:' + entry['sha256']
    assert 'cuda_nvtx' not in builds.ENGINE_CUDA_COMPONENTS['sparkinfer']
    assert 'libcurand' not in builds.ENGINE_CUDA_COMPONENTS['ninfer']


async def test_spark_plan_and_incremental_cuda_manifest_add_only_required_components(tmp_path, monkeypatch):
    paths = AppPaths(tmp_path)
    (paths.runtimes / 'ModLoader-wsl-owned.json').write_text('{}')
    extras = {
        'libcurand': {'relative_path': 'libcurand/linux-x86_64/libcurand-linux-x86_64-10.4.4.72-archive.tar.xz',
                     'sha256': '6f725f9187dc5f675308b830f786c163f5b1b499d86986f94fb60aa701c13c0b', 'size': '86992868'},
        'cuda_profiler_api': {'relative_path': 'cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-13.4.92-archive.tar.xz',
                              'sha256': '9d405e9b0fd9a5c29e680d6f6750b9ca79b7df91e253d4dffea3495057a75f08', 'size': '17112'},
    }
    urls = []
    async def metadata(url, proxy=None):
        assert proxy == 'http://approved:17890'
        urls.append(url)
        if '/git/trees/' in url:
            return {'tree': [{'type': 'blob', 'path': 'CMakeLists.txt', 'sha': 'b' * 40, 'mode': '100644'}]}
        return {**{name: {'linux-x86_64': {'relative_path': name + '/linux-x86_64/archive.tar.xz', 'sha256': 'a' * 64, 'size': 42}}
                   for name in builds.CUDA_COMPONENTS}, **{name: {'linux-x86_64': entry} for name, entry in extras.items()}}
    async def dependencies(paths, files, sources, proxy):
        return files, sources
    monkeypatch.setattr(catalog, '_json', metadata)
    monkeypatch.setattr(builds, 'spark_dependencies', dependencies)
    plan = await builds.plan_install('sparkinfer', paths, {}, 'http://approved:17890')
    assert plan['provenance']['cuda_components'] == [*builds.CUDA_COMPONENTS, *extras]
    assert len(plan['stage']['files']) == 1 + len(builds.CUDA_COMPONENTS) + len(extras)
    for name, entry in extras.items():
        added = next(item for item in plan['stage']['files'] if item['role'] == 'cuda_' + name)
        assert added['size_bytes'] == int(entry['size']) and added['digest'] == 'sha256:' + entry['sha256']
        assert added['url'] == builds.CUDA_BASE + entry['relative_path']
        assert name not in builds.ENGINE_CUDA_COMPONENTS['ninfer']
        urls.clear()
        delta = await builds.cuda_files(paths, 'http://approved:17890', components=(name,))
        assert len(delta) == 1 and delta[0]['url'] == added['url']
        assert urls == [builds.CUDA_BASE + 'redistrib_13.4.3.json']
    assert not list(paths.models.iterdir()) and not list(paths.runtimes.rglob('*.tar.xz'))


@pytest.mark.parametrize('backend', ['ninfer', 'sparkinfer'])
async def test_native_build_only_uses_confirmed_sources_isolated_toolchain_and_four_jobs(tmp_path, monkeypatch, backend):
    root, linux = tmp_path, '/opt/modloader/fixture'
    calls = []
    async def command(argv, *args, **kwargs):
        calls.append(('command', argv))
    async def apt(*args, **kwargs):
        assert kwargs['install_recommends'] is False
        calls.append(('apt', kwargs['packages']))
    async def packages(python, requirements, *args, **kwargs):
        assert kwargs['wsl'] and kwargs['resolver_proxy'] == 'http://approved:17890'
        calls.append(('packages', requirements))
        return [{'package': 'torch', 'version': '2.13.0'}]
    from modloader import platforms
    monkeypatch.setattr(platforms, 'windows_to_wsl', lambda value: '/mounted/' + Path(value).name)
    monkeypatch.setattr(installers, '_wsl_command', command)
    monkeypatch.setattr(installers, '_apt_stage', apt)
    monkeypatch.setattr(installers, '_package_stage', packages)
    async def cargo(*args, **kwargs):
        calls.append(('cargo', []))
    monkeypatch.setattr(builds, 'prepare_spark_cargo', cargo)
    transfer = SimpleNamespace(progress=None, cancel=asyncio.Event())
    artifacts = {name: root / (name + '.tar.xz') for name in ['source', *['cuda_' + key for key in builds.ENGINE_CUDA_COMPONENTS[backend]]]}
    plan = {'provenance': {'source_commit': builds.PINS[backend]['commit'], 'sources': [
        {'role': 'source', 'destination': 'source', 'tree': {'CMakeLists.txt': {'sha': 'a' * 40, 'mode': '100644'}}}]}}
    record = await builds.install(backend, root, linux, artifacts, plan, transfer, lambda stage: None, 'http://approved:17890')
    assert calls[0][0] == 'apt' and calls[1][1] == ['/usr/bin/python3', '-m', 'venv', linux + '/venv']
    configure = next(argv for kind, argv in calls if kind == 'command' and '-S' in argv)
    build = calls[-1][1]
    assert '-DFETCHCONTENT_FULLY_DISCONNECTED=ON' in configure
    assert ('-DCMAKE_CUDA_ARCHITECTURES=120a' if backend == 'ninfer' else '-DSPARKINFER_NVFP4=ON') in configure
    assert build[build.index('--parallel') + 1] == '4'
    assert build[-1] == ('ninfer-serve' if backend == 'ninfer' else 'sparkinfer_server')
    assert record['source_commit'] == builds.PINS[backend]['commit'] and record['python'] == linux + '/venv/bin/python'
    request = json.loads((root / 'stage-native-build.json').read_text())
    assert len(request['archives']) == len(artifacts) and all(value['destination'] in {'cuda', 'source'} for value in request['archives'])


@pytest.mark.parametrize('backend,missing', [('sparkinfer', 'libcurand'), ('sparkinfer', 'cuda_profiler_api'), ('ninfer', 'cuda_nvtx')])
async def test_old_native_manifest_missing_component_fails_before_shared_or_build_changes(tmp_path, backend, missing):
    artifacts = {'cuda_' + name: tmp_path / name for name in builds.ENGINE_CUDA_COMPONENTS[backend] if name != missing}
    with pytest.raises(ValueError, match='缺少必需的 CUDA 组件：' + missing) as error:
        await builds.install(backend, tmp_path, '/opt/modloader/fixture', artifacts, {'provenance': {}},
                             SimpleNamespace(), None, None)
    assert '重新开始安装以获取完整清单' in str(error.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('backend', ['ninfer', 'sparkinfer'])
async def test_native_runtime_retains_provenance_but_not_blob_trees_and_plan_stays_complete(tmp_path, monkeypatch, backend):
    import copy
    paths = AppPaths(tmp_path)
    pin = builds.PINS[backend]
    tree = {'CMakeLists.txt': {'sha': 'a' * 40, 'mode': '100644'}}
    source = dict(role='source', repository=pin['repository'], commit=pin['commit'], destination='source',
                  tree=tree, submodules=[{'path': 'submodule', 'sha': 'b' * 40}])
    dependency = dict(role='dependency', repository='official/dependency', commit='c' * 40, destination='deps/dependency',
                      tree=copy.deepcopy(tree), submodules=[])
    provenance = dict(repository=pin['repository'], source_commit=pin['commit'], source_tree=tree,
                      sources=[source, dependency], cuda_release=builds.CUDA_RELEASE,
                      cuda_components=list(builds.ENGINE_CUDA_COMPONENTS[backend]))
    plan = dict(id='approved-plan', backend=backend, version=pin['version'], variant='wsl2', provenance=provenance,
                stage={'id': 'base', 'title': 'Approved source manifest', 'files': []})
    original = copy.deepcopy(plan)
    checked = []
    class Transfers:
        def __init__(self, *args): pass
        async def add(self, stage): return stage
        async def fetch(self, stage): return {}
    async def proxy(*args): return 'http://approved:17890'
    async def source_install(selected, root, linux, artifacts, complete_plan, *args):
        assert selected == backend and complete_plan == original
        checked.append('complete source inventory available for validation')
        return dict(source_commit=pin['commit'], source_root=linux + '/source')
    def import_runtime(selected, path, metadata):
        return RuntimeInstallation(id='native-runtime', backend=selected, name=selected, path=path,
                                   platform='wsl2', metadata=metadata)
    async def validate(runtime, **kwargs):
        assert checked and kwargs['_installing'] is True
        runtime.version = pin['version']
        return runtime
    monkeypatch.setattr(installers, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(installers, 'list_wsl', lambda: {'available': True, 'distributions': [{'name': 'ModLoader', 'version': 2}]})
    monkeypatch.setattr(installers, '_verify_wsl_owner', lambda paths: None)
    monkeypatch.setattr(installers, '_Transfers', Transfers)
    monkeypatch.setattr(installers, '_wsl_proxy', proxy)
    monkeypatch.setattr(builds, 'install', source_install)
    monkeypatch.setattr(installers, 'import_runtime', import_runtime)
    monkeypatch.setattr(installers, 'validate_runtime', validate)
    runtime = await installers.install_backend(backend, paths, {'_plan': plan}, None, asyncio.Event())
    compact = runtime.metadata['provenance']
    assert 'source_tree' not in compact and compact['source_commit'] == pin['commit']
    assert compact['repository'] == pin['repository'] and compact['cuda_release'] == builds.CUDA_RELEASE
    assert compact['cuda_components'] == list(builds.ENGINE_CUDA_COMPONENTS[backend])
    assert compact['sources'] == [{key: value for key, value in item.items() if key not in {'tree', 'submodules'}}
                                  for item in original['provenance']['sources']]
    assert plan == original  # Confirmation/recovery plan still has every blob and submodule entry.
    manifest = Path(runtime.metadata['managed_root']) / 'modloader-runtime.json'
    recorded = RuntimeInstallation.model_validate_json(manifest.read_text(encoding='utf-8'))
    assert recorded.metadata['provenance'] == compact and recorded.managed


@pytest.mark.parametrize('recommendations', [True, False])
async def test_native_apt_resolution_and_offline_install_share_recommendations_policy(tmp_path, monkeypatch, recommendations):
    paths = AppPaths(tmp_path / 'isolated')
    commands = []
    item = dict(name='required_1.0_amd64.deb', url='https://archive.ubuntu.com/required.deb', size_bytes=1,
                digest='sha256:' + 'a' * 64, package='required', version='1.0')
    payload = tmp_path / item['name']
    payload.write_bytes(b'x')
    async def command(argv, *args, **kwargs):
        commands.append(argv)
        if argv[0] == '/usr/bin/python3':
            (tmp_path / 'apt-packages.json').write_text(json.dumps([item]))
    async def add(stage, confirm):
        assert confirm is not None
        return stage
    async def fetch(stage):
        return {'required': payload}
    monkeypatch.setattr(installers, 'windows_to_wsl', lambda value: '/mounted/' + Path(value).name)
    monkeypatch.setattr(installers, '_wsl_command', command)
    transfer = SimpleNamespace(paths=paths, proxy=None, progress=None, cancel=asyncio.Event(), add=add, fetch=fetch)
    await installers._apt_stage(tmp_path, transfer, lambda stage: None, packages=['required'], install_recommends=recommendations)
    assert ('no-recommends' in commands[0]) is not recommendations
    assert ('--no-install-recommends' in commands[-1]) is not recommendations
    assert '--no-download' in commands[-1] and commands[-1][-1] == 'required=1.0'


@pytest.mark.parametrize('upgrade', [False, True])
def test_no_recommends_guest_resolver_keeps_shared_package_upgrade_guard(tmp_path, monkeypatch, upgrade):
    import subprocess
    import sys
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'dpkg-query':
            installed = upgrade and argv[-1] == 'shared'
            return SimpleNamespace(returncode=0 if installed else 1, stdout='ii  0.9' if installed else '')
        return SimpleNamespace(returncode=0)
    def output(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'apt-get':
            assert '--no-install-recommends' in argv and '--no-upgrade' in argv
            return "'https://archive.ubuntu.com/shared_1.0_amd64.deb' shared_1.0_amd64.deb 1 SHA256:" + 'a' * 64
        return 'Package: shared\nVersion: 1.0\nFilename: pool/shared_1.0_amd64.deb\nSHA256: ' + 'a' * 64
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'check_output', output)
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['resolve-apt.py', str(report), '', '["required"]', 'no-recommends'])
    if upgrade:
        with pytest.raises(RuntimeError, match='upgrade shared package shared'):
            exec(installers._APT_RESOLVE, {})
        assert not report.exists()
    else:
        exec(installers._APT_RESOLVE, {})
        assert json.loads(report.read_text())[0]['package'] == 'shared'
