"""Approved, offline native WSL builds for the two source-only engine recipes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import uuid

PINS = {
    'ninfer': {'repository': 'Neroued/ninfer', 'version': '64d10fbf9f8e',
               'commit': '64d10fbf9f8e6bfc79ab0570b94674c5b9997929', 'entrypoint': 'source/build/apps/ninfer-serve'},
    'sparkinfer': {'repository': 'gittensor-ai-lab/sparkinfer', 'version': '0.6.34',
                  'commit': 'f49c0e4afa612f9e6ddaf12893a1e08a3713cc61', 'entrypoint': 'source/build/server/sparkinfer_server'},
}
NINFER_PINS = {
    'a754fcf3bf52': {'repository': 'Neroued/ninfer', 'version': 'a754fcf3bf52',
                     'commit': 'a754fcf3bf524494aa927c2f54e0cbfb58f569ab',
                     'entrypoint': 'source/build/apps/ninfer-serve'},
    PINS['ninfer']['version']: PINS['ninfer'],
}


def resolve_pin(backend, version=None):
    """Select an approved recipe without changing existing runtime identities."""
    if backend == 'ninfer':
        selected = next(iter(NINFER_PINS)) if version is None else version
        if isinstance(selected, str) and selected in NINFER_PINS:
            return NINFER_PINS[selected]
    else:
        pin = PINS[backend]
        if version is None or version == pin['version']:
            return pin
    raise ValueError('请选择此适配器已经固定并核实的 WSL2 构建版本')


CUDA_RELEASE = '13.4.3'
CUDA_COMPONENTS = ('cuda_nvcc', 'cuda_crt', 'cuda_cudart', 'cccl', 'libnvvm', 'cuda_nvml_dev')
# Required by production sources: NInfer core/nvtx.h; Spark's sampling/DFlash
# kernels and qwen35.cpp. Profiling-only NInfer benchmarks are not built by this recipe.
ENGINE_CUDA_COMPONENTS = {'ninfer': (*CUDA_COMPONENTS, 'cuda_nvtx'),
                          'sparkinfer': (*CUDA_COMPONENTS, 'libcurand', 'cuda_profiler_api')}
CUDA_BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
BUILD_JOBS = 4
NINFER_SYSTEM_PACKAGES = ['python3-venv', 'python3-pip', 'build-essential', 'cmake', 'ninja-build', 'pkg-config',
                          'libavformat-dev', 'libavcodec-dev', 'libavutil-dev', 'libswscale-dev', 'libhwloc-dev',
                          'liburing-dev', 'libcurl4-openssl-dev']


async def source_file(paths, repository, commit, role, destination, proxy=None):
    from . import catalog
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('构建源码缺少固定的官方 Git 提交')
    tree = await catalog._json(f'https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1', proxy)
    if tree.get('truncated'):
        raise ValueError('官方源码树未完整返回，不能执行未核实的源码')
    entries = {entry['path']: {'sha': entry['sha'], 'mode': entry['mode']} for entry in tree['tree'] if entry['type'] == 'blob'}
    item = await catalog.manifest_file(paths, repository.rsplit('/', 1)[1] + '-' + commit + '.zip',
                                       f'{repository} 固定源码 {commit[:12]}', f'https://codeload.github.com/{repository}/zip/{commit}',
                                       role=role, source_commit=commit)
    return item, dict(role=role, repository=repository, commit=commit, destination=destination, tree=entries,
                      submodules=[entry for entry in tree['tree'] if entry['type'] == 'commit'])


async def cuda_files(paths, proxy=None, *, components=CUDA_COMPONENTS):
    """Resolve official archives; an explicit subset also supports confirmed build repairs."""
    from . import catalog
    data = await catalog._json(CUDA_BASE + f'redistrib_{CUDA_RELEASE}.json', proxy)
    result = []
    for name in components:
        entry = data[name]['linux-x86_64']
        relative = entry['relative_path']
        if not re.fullmatch(r'[A-Za-z0-9_./+-]+', relative) or '..' in Path(relative).parts:
            raise ValueError('无效 CUDA 官方重分发路径')
        result.append(await catalog.manifest_file(paths, relative.rsplit('/', 1)[1], f'独立 CUDA {CUDA_RELEASE} 工具链 · {name}',
                      CUDA_BASE + relative, 'sha256:' + entry['sha256'], int(entry['size']), role='cuda_' + name))
    return result


async def plan_install(backend, paths, options, proxy=None):
    from . import catalog
    pin = resolve_pin(backend, options.get('version'))
    if options.get('variant', 'wsl2') != 'wsl2':
        raise ValueError('请选择此适配器已经固定并核实的 WSL2 构建版本')
    source, record = await source_file(paths, pin['repository'], pin['commit'], 'source', 'source', proxy)
    components = ENGINE_CUDA_COMPONENTS[backend]
    files = [source, *await cuda_files(paths, proxy, components=components)]
    sources = [record]
    if backend == 'sparkinfer':
        files, sources = await spark_dependencies(paths, files, sources, proxy)
    if not (paths.runtimes / 'ModLoader-wsl-owned.json').is_file():
        files.append(await catalog.manifest_file(paths, 'ubuntu-24.04.rootfs.tar.gz', '专属 WSL2 Ubuntu 基础系统',
                    catalog.UBUNTU_URL, 'sha256:' + catalog.UBUNTU_SHA256, role='rootfs'))
    return dict(id=uuid.uuid4().hex, backend=backend, version=pin['version'], variant='wsl2', options={'version': pin['version'], 'variant': 'wsl2'},
                provenance={'repository': pin['repository'], 'source_commit': pin['commit'], 'sources': sources,
                            'cuda_release': CUDA_RELEASE, 'cuda_components': list(components)},
                stage=dict(id=uuid.uuid4().hex, title='固定源码与独立 CUDA 工具链', files=files),
                dependency_notice='基础环境就绪后自动解析并安装系统、转换工具及构建依赖，文件清单显示在任务详情中；离线构建最多使用 4 个作业。')


def build_environment(linux_root):
    return {'CUDA_HOME': linux_root + '/cuda', 'CUDACXX': linux_root + '/cuda/bin/nvcc',
            'PATH': linux_root + '/rust/bin:' + linux_root + '/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            'LD_LIBRARY_PATH': linux_root + '/cuda/lib:' + linux_root + '/cuda/lib64:/usr/lib/wsl/lib',
            'CARGO_HOME': linux_root + '/cargo-home', 'CARGO_NET_OFFLINE': 'true', 'CARGO_BUILD_JOBS': str(BUILD_JOBS),
            'CMAKE_BUILD_PARALLEL_LEVEL': str(BUILD_JOBS),
            # NInfer's CUDA converter lazily builds stored_scale from its verified source.
            'TORCH_EXTENSIONS_DIR': linux_root + '/torch-extensions', 'MAX_JOBS': str(BUILD_JOBS)}


STAGE_SOURCE = r'''
import hashlib,json,os,pathlib,shutil,stat,sys,tarfile,tempfile,zipfile
root=pathlib.Path(sys.argv[1]); request=json.load(open(sys.argv[2]))
assert root.parent==pathlib.Path('/opt/modloader') and root.resolve()==root
root.mkdir(parents=True,exist_ok=True)
def destination(value):
    p=root/value
    if p!=root and p.resolve().is_relative_to(root): return p
    raise ValueError('Build destination escaped its isolated root')
def extract(archive,target):
    target.mkdir(parents=True,exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                p=target/item.filename
                if not p.resolve().is_relative_to(target): raise ValueError('Invalid archive path')
                if item.is_dir(): p.mkdir(parents=True,exist_ok=True); continue
                p.parent.mkdir(parents=True,exist_ok=True)
                if stat.S_ISLNK(item.external_attr>>16):
                    link=z.read(item).decode()
                    if not (p.parent/link).resolve().is_relative_to(target): raise ValueError('Archive symlink escaped root')
                    p.symlink_to(link)
                else:
                    p.write_bytes(z.read(item)); p.chmod((item.external_attr>>16)&0o777 or 0o644)
    else:
        with tarfile.open(archive) as tar: tar.extractall(target,filter='data')
    entries=list(target.iterdir())
    return entries[0] if len(entries)==1 and entries[0].is_dir() else target
def cuda_layout(cuda):
    library=cuda/'lib'
    if not all((library/name).is_file() for name in ('libcudart_static.a','libcudadevrt.a')):
        raise ValueError('CUDA redist is missing its static runtime libraries')
    # NVIDIA redist stores libraries in lib/, while its nvcc.profile searches lib64/.
    # Keep the approved files in place; supply the expected layout only in this root.
    alternate=cuda/'lib64'
    if not alternate.exists(): alternate.symlink_to('lib',target_is_directory=True)
for item in request['archives']:
    target=destination(item['destination'])
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        unpacked=extract(item['archive'],pathlib.Path(temporary)/'payload')
        tree=item.get('tree')
        if tree is not None:
            actual={str(p.relative_to(unpacked)):p for p in unpacked.rglob('*') if p.is_file() or p.is_symlink()}
            if actual.keys()!=tree.keys(): raise ValueError('Official Git source inventory mismatch')
            for name,p in actual.items():
                data=os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
                if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=tree[name]['sha']:
                    raise ValueError('Official Git blob mismatch: '+name)
        shutil.copytree(unpacked,target,dirs_exist_ok=True,symlinks=True)
cuda_layout(root/'cuda')
print('Verified source and toolchain files staged in isolated environment',flush=True)
'''


async def stage_archives(root, linux_root, archives, progress, cancel):
    """Merge verified archives into one owned environment, retaining existing build output."""
    from . import installers
    from ..platforms import windows_to_wsl
    request = root / 'stage-native-build.json'
    request.write_text(json.dumps({'archives': archives}), encoding='utf-8')
    helper = root / 'stage-native-build.py'
    helper.write_text(STAGE_SOURCE, encoding='utf-8')
    await installers._wsl_command(['/usr/bin/python3', windows_to_wsl(str(helper.resolve())), linux_root,
                                   windows_to_wsl(str(request.resolve()))], root, progress, cancel)


async def install(backend, root, linux_root, artifacts, plan, transfer, confirm, resolver_proxy):
    from . import installers
    from ..platforms import windows_to_wsl
    provenance = plan['provenance']
    components = ENGINE_CUDA_COMPONENTS[backend]
    missing = [name for name in components if 'cuda_' + name not in artifacts]
    if missing:
        raise ValueError('安装清单缺少必需的 CUDA 组件：' + ', '.join(missing) + '；请重新开始安装以获取完整清单')
    archives = []
    for source in provenance['sources']:
        archives.append(dict(archive=windows_to_wsl(str(artifacts[source['role']].resolve())), destination=source['destination'], tree=source['tree']))
    for component in components:
        archives.append(dict(archive=windows_to_wsl(str(artifacts['cuda_' + component].resolve())), destination='cuda'))
    packages = list(NINFER_SYSTEM_PACKAGES) if backend == 'ninfer' else ['python3-venv', 'python3-pip', 'build-essential', 'cmake', 'ninja-build', 'pkg-config']
    if backend == 'ninfer' and provenance['source_commit'] == resolve_pin('ninfer')['commit']:
        # a754fcf cmake/Dependencies.cmake requires JPEG for product media.
        packages.append('libjpeg-dev')
    await installers._apt_stage(root, transfer, confirm, resolver_proxy=resolver_proxy, packages=packages, install_recommends=False)
    await installers._wsl_command(['/usr/bin/python3', '-m', 'venv', linux_root + '/venv'], root, transfer.progress, transfer.cancel)
    await stage_archives(root, linux_root, archives, transfer.progress, transfer.cancel)
    # These are preparation dependencies, not implicit downloads when converting a model.
    python_packages = await installers._package_stage(linux_root + '/venv/bin/python', ['torch==2.13.0', 'numpy'], root,
                        transfer, confirm, wsl=True, resolver_proxy=resolver_proxy, retry=plan.get('_retry_packages')) if backend == 'ninfer' else []
    env = build_environment(linux_root)
    if backend == 'sparkinfer':
        await prepare_spark_cargo(root, linux_root, artifacts, plan, transfer, confirm, resolver_proxy)
    source = linux_root + '/source'
    configure = ['cmake', '-S', source, '-B', source + '/build', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                 '-DCUDAToolkit_ROOT=' + linux_root + '/cuda', '-DCMAKE_CUDA_COMPILER=' + linux_root + '/cuda/bin/nvcc',
                 '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON']
    if backend == 'ninfer':
        configure += ['-DCMAKE_CUDA_ARCHITECTURES=120a', '-DNINFER_BUILD_APPS=ON', '-DBUILD_TESTING=OFF', '-DNINFER_BUILD_BENCHMARKS=OFF']
        target = 'ninfer-serve'
    else:
        configure += spark_cmake_options(linux_root)
        target = 'sparkinfer_server'
    prefix = ['/usr/bin/env', *[key + '=' + value for key, value in env.items()]]
    await installers._wsl_command([*prefix, *configure], root, transfer.progress, transfer.cancel, timeout=1800)
    await installers._wsl_command([*prefix, 'cmake', '--build', source + '/build', '--parallel', str(BUILD_JOBS), '--target', target],
                                  root, transfer.progress, transfer.cancel, timeout=7200)
    return dict(python=linux_root + '/venv/bin/python', source_root=source, source_commit=provenance['source_commit'],
                packages=python_packages, build_environment=env, build_jobs=BUILD_JOBS, cuda_release=CUDA_RELEASE)


async def resolve_commit(repository, ref, proxy=None):
    from . import catalog
    result = await catalog._json(f'https://api.github.com/repos/{repository}/git/ref/tags/{ref}', proxy)
    value = result['object']
    if value['type'] == 'tag':
        value = (await catalog._json(value['url'], proxy))['object']
    if value['type'] != 'commit' or not re.fullmatch('[a-f0-9]{40}', value['sha']):
        raise ValueError('构建依赖的官方 tag 未解析到固定提交')
    return value['sha']


async def spark_dependencies(paths, files, sources, proxy=None):
    from . import catalog
    pinned = [
        ('nlohmann_json', 'https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz', '42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa'),
        ('re2', 'https://github.com/google/re2/archive/refs/tags/2023-03-01.tar.gz', '7a9a4824958586980926a300b4717202485c4b4115ac031822e29aa4ef207e48'),
        ('dlpack', 'https://github.com/dmlc/dlpack/archive/bbd2f4d32427e548797929af08cfe2a9cbb3cf12.tar.gz', 'f5dcb30f8a3d1a41d48b5d5b3fe1631dfce11de76f9e7e5bd22c6dc75a758885'),
    ]
    for role, url, checksum in pinned:
        name = role + '-' + url.rsplit('/', 1)[1]
        files.append(await catalog.manifest_file(paths, name, 'SparkInfer 官方构建依赖 · ' + role, url, 'sha256:' + checksum, role=role))
        sources.append(dict(role=role, destination='deps/' + role, tree=None))
    token, token_source = await source_file(paths, 'mlc-ai/tokenizers-cpp', 'c586c52f93f7b060753bd2388eb96a105cb7374d', 'tokenizers_cpp', 'deps/tokenizers_cpp', proxy)
    files.append(token); sources.append(token_source)
    for path, repository in [('msgpack', 'msgpack/msgpack-c'), ('sentencepiece', 'google/sentencepiece')]:
        sha = next((entry['sha'] for entry in token_source['submodules'] if entry['path'] == path), None)
        if sha is None:
            raise ValueError('官方 tokenizer 子模块清单发生变化，未执行不完整构建')
        item, source = await source_file(paths, repository, sha, 'token_' + path, 'deps/tokenizers_cpp/' + path, proxy)
        files.append(item); sources.append(source)
    cutlass = await resolve_commit('NVIDIA/cutlass', 'v4.7.0', proxy)
    item, source = await source_file(paths, 'NVIDIA/cutlass', cutlass, 'cutlass', 'deps/cutlass', proxy)
    files.append(item); sources.append(source)
    xgrammar = await resolve_commit('mlc-ai/xgrammar', 'v0.2.6', proxy)
    item, source = await source_file(paths, 'mlc-ai/xgrammar', xgrammar, 'xgrammar', 'deps/xgrammar', proxy)
    files.append(item); sources.append(source)
    picojson = next((entry['sha'] for entry in source['submodules'] if entry['path'] == '3rdparty/picojson'), None)
    if picojson:
        item, source = await source_file(paths, 'kazuho/picojson', picojson, 'picojson', 'deps/xgrammar/3rdparty/picojson', proxy)
        files.append(item); sources.append(source)
    rust_url = 'https://static.rust-lang.org/dist/2025-11-10/rust-1.91.1-x86_64-unknown-linux-gnu.tar.xz'
    files.append(await catalog.manifest_file(paths, rust_url.rsplit('/', 1)[1], '独立 Rust 1.91.1 工具链（官方发行清单 SHA256）', rust_url,
                 'sha256:db178028da493a663b2e344a142076fb0f72adea4120043983f488de9cf95214', role='rust'))
    sources.append(dict(role='rust', destination='toolchain/rust', tree=None))
    return files, sources


def spark_cmake_options(linux_root):
    return ['-DCMAKE_CUDA_ARCHITECTURES=120', '-DBUILD_SERVER=ON', '-DBUILD_TESTS=OFF', '-DBUILD_EXAMPLES=OFF',
            '-DBUILD_BENCHMARKS=OFF', '-DBUILD_CUTE_KERNELS=OFF', '-DSPARKINFER_NVFP4=ON', '-DMLC_ENABLE_SENTENCEPIECE_TOKENIZER=OFF',
            *['-DFETCHCONTENT_SOURCE_DIR_' + name.upper() + '=' + linux_root + '/deps/' + name
              for name in ('nlohmann_json', 'tokenizers_cpp', 're2', 'xgrammar', 'dlpack', 'cutlass')]]


async def prepare_spark_cargo(root, linux_root, artifacts, plan, transfer, confirm, resolver_proxy):
    from . import catalog, installers
    from ..platforms import windows_to_wsl
    if confirm is None:
        raise ValueError('请通过环境安装任务下载和安装 Rust crate 依赖')
    env = build_environment(linux_root)
    await installers._wsl_command(['/bin/sh', linux_root + '/toolchain/rust/install.sh', '--prefix=' + linux_root + '/rust', '--disable-ldconfig'],
                                  root, transfer.progress, transfer.cancel, timeout=600)
    lock_report = root / 'cargo-lock.json'
    source = linux_root + '/deps/tokenizers_cpp/rust'
    code = '''import json,os,pathlib,subprocess,sys,tomllib
root,source,report,proxy=sys.argv[1:]
env=dict(os.environ,CARGO_HOME=root+'/cargo-home',CARGO_HTTP_PROXY=proxy,CARGO_NET_OFFLINE='false',CARGO_BUILD_JOBS='4')
subprocess.run([root+'/rust/bin/cargo','generate-lockfile','--manifest-path',source+'/Cargo.toml'],env=env,check=True)
data=tomllib.loads(pathlib.Path(source+'/Cargo.lock').read_text())
items=[]
for p in data['package']:
    if 'source' not in p: continue
    if p['source']!='registry+https://github.com/rust-lang/crates.io-index': raise ValueError('Unapproved cargo source')
    items.append(dict(package=p['name'],version=p['version'],digest='sha256:'+p['checksum']))
pathlib.Path(report).write_text(json.dumps(items))'''
    await installers._emit(transfer.progress, phase='resolve', message='解析 Rust registry 元数据并冻结 Cargo.lock；尚未下载 crate 源码', can_pause=False)
    await installers._wsl_command(['/usr/bin/python3', '-c', code, linux_root, source, windows_to_wsl(str(lock_report.resolve())), resolver_proxy or ''],
                                  root, transfer.progress, transfer.cancel, timeout=900)
    files = []
    for item in json.loads(lock_report.read_text(encoding='utf-8')):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', item['package']) or not re.fullmatch(r'[A-Za-z0-9_.+-]+', item['version']) or not re.fullmatch('sha256:[a-f0-9]{64}', item['digest']):
            raise ValueError('无效的 Cargo.lock 依赖身份')
        name = item['package'] + '-' + item['version'] + '.crate'
        url = f"https://static.crates.io/crates/{item['package']}/{name}"
        files.append(await catalog.manifest_file(transfer.paths, name, f"Rust crate {item['package']} {item['version']}", url, item['digest'], package=item['package'], version=item['version']))
    stage = await transfer.add(dict(id=uuid.uuid4().hex, title='Rust 构建依赖', files=files), confirm)
    crates = await transfer.fetch(stage)
    request = root / 'stage-crates.json'
    request.write_text(json.dumps({'archives': [dict(archive=windows_to_wsl(str(path.resolve())), destination='vendor/' + item['package'] + '-' + item['version'])
                                               for path, item in zip(crates.values(), stage['files'])]}), encoding='utf-8')
    await installers._wsl_command(['/usr/bin/python3', windows_to_wsl(str((root / 'stage-native-build.py').resolve())), linux_root,
                                   windows_to_wsl(str(request.resolve()))], root, transfer.progress, transfer.cancel)
    # Cargo's directory source requires its own per-file integrity manifest.
    code = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); items=json.load(open(sys.argv[2]))
for item in items:
    p=root/'vendor'/(item['package']+'-'+item['version'])
    files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and f.name!='.cargo-checksum.json'}
    (p/'.cargo-checksum.json').write_text(json.dumps(dict(package=item['digest'].removeprefix('sha256:'),files=files)))
home=root/'cargo-home'; home.mkdir(exist_ok=True)
lines=['[source.crates-io]','replace-with = "modloader-vendored"','[source.modloader-vendored]',
       'directory = '+json.dumps(str(root/'vendor')),'[net]','offline = true']
(home/'config.toml').write_text(chr(10).join(lines)+chr(10))
deps=root/'source/build/_deps'; deps.mkdir(parents=True,exist_ok=True)
(deps/'cutlass-src').symlink_to(root/'deps/cutlass',target_is_directory=True)
print('Cargo directory source verified; CMake and Cargo network fetches disabled')'''
    await installers._wsl_command(['/usr/bin/python3', '-c', code, linux_root, windows_to_wsl(str(lock_report.resolve()))], root, transfer.progress, transfer.cancel)
