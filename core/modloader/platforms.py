"""Platform execution and explicit WSL filesystem access.

Listing never starts a distribution. Checks, model reads and launch preparation are
explicit operations; engines supply native argv and never construct WSL shells.
"""
from __future__ import annotations
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess

from .models import ModelLocation

WSL_SUPERVISOR = r'''import json, os, select, signal, subprocess, sys, time
from pathlib import Path
mode, record, identity = sys.argv[1:4]
path = Path(record)
saved_record = None
def starttime(pid):
    try: return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError): return None
def owned_group(pid, saved_start):
    if saved_start is not None and starttime(pid) == saved_start: return True
    # An engine can exit before its workers. Verify the remaining group by the
    # per-instance environment marker rather than treating a stale PID as ours.
    marker = ('MODLOADER_INSTANCE_ID=' + identity).encode()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pid and marker in (entry / 'environ').read_bytes().split(b'\x00'): return True
        except (OSError, IndexError, ValueError): pass
    return False
def stop():
    try: saved = json.loads(path.read_text())
    except (OSError, ValueError): saved = saved_record
    if saved is None: return
    pid = saved.get('pid', 0)
    if saved.get('identity') != identity or not isinstance(pid, int) or pid <= 1: return
    if not owned_group(pid, saved.get('starttime')): return
    try: os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError: return
    for _ in range(50):
        if not owned_group(pid, saved.get('starttime')): return
        time.sleep(.1)
    if owned_group(pid, saved.get('starttime')):
        try: os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError: pass
if mode == 'stop':
    stop(); sys.exit(0)
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', MODLOADER_INSTANCE_ID=identity)
for k in list(env):
    if any(word in k.upper() for word in ('API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN', 'MCP')) or k.upper() in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'): env.pop(k, None)
config = json.loads(sys.argv[4]) if mode == 'run-config' else {'argv': sys.argv[4:]}
env.update(config.get('env', {}))
env['MODLOADER_INSTANCE_ID'] = identity
proc = subprocess.Popen(config['argv'], cwd=config.get('cwd'), env=env, start_new_session=True, stdin=subprocess.DEVNULL)
saved_record = dict(identity=identity, pid=proc.pid, starttime=starttime(proc.pid))
try:
    path.write_text(json.dumps(saved_record))
    print('MODLOADER_GUEST_PID=' + str(proc.pid), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: stop())
    signal.signal(signal.SIGINT, lambda *_: stop())
    # Monitor the parent pipe without a buffered reader thread surviving shutdown.
    while proc.poll() is None:
        readable, _, _ = select.select([sys.stdin.fileno()], [], [], .1)
        if readable and not os.read(sys.stdin.fileno(), 4096):
            stop()
            break
    sys.exit(proc.wait())
finally:
    stop()
    path.unlink(missing_ok=True)
'''
def distribution_name(value):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in '\0\r\n/\\'):
        raise ValueError('请选择有效的 WSL 发行版')
    return value


def linux_path(value):
    if not isinstance(value, str) or not value.startswith('/') or any(c in value for c in '\0\r\n'):
        raise ValueError('WSL 路径必须是 Linux 绝对路径')
    return str(PurePosixPath(value))


def location_of(model):
    value = model.get('location') if isinstance(model, dict) else model.location
    return ModelLocation.model_validate(value or {'platform': 'windows'})


def runtime_distribution(runtime):
    return distribution_name(runtime.metadata.get('distribution', 'ModLoader' if runtime.backend == 'vllm' else None))


def model_identity(model):
    location = location_of(model)
    path = model['path'] if isinstance(model, dict) else model.path
    if location.platform == 'wsl2':
        return ('wsl2', location.distribution, linux_path(path))
    return ('windows', os.path.normcase(str(Path(path).resolve())))


def validate_model_runtime(model, runtime):
    location = location_of(model)
    if location.platform == 'wsl2':
        if runtime.platform != 'wsl2':
            raise ValueError('Windows 引擎不能直接使用 WSL 内部模型；请选择同发行版的 WSL 引擎')
        if location.distribution != runtime_distribution(runtime):
            raise ValueError('WSL 内部模型只能由同一发行版的引擎使用')


def wsl_prefix(distribution):
    return ['wsl.exe', '--distribution', distribution_name(distribution), '--exec']


def _run(argv, *, timeout=30, input=None):
    try:
        result = subprocess.run(argv, input=input, capture_output=True, encoding='utf-8', errors='replace',
                                timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f'WSL 操作失败：{error}') from error
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or f'WSL 退出码 {result.returncode}').strip())
    return result.stdout.replace('\x00', '').lstrip('\ufeff').strip()


def list_wsl(paths=None):
    try:
        # UTF-16LE is used by older wsl.exe even with WSL_UTF8 set; removing NULs
        # also accepts that ASCII column encoding, while UTF-8 supports new WSL.
        result = subprocess.run(['wsl.exe', '--list', '--verbose'], capture_output=True, timeout=15,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'distributions': [], 'error': str(error)}
    raw = result.stdout
    text = raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-16-le' if b'\x00' in raw else 'utf-8', errors='replace') if isinstance(raw, bytes) else raw
    import re
    distributions = []
    for line in text.splitlines():
        match = re.match(r'^\s*\*?\s*(.+?)\s{2,}(\S+)\s+(\d+)\s*$', line)
        if match:
            name, state, version = match.groups()
            distributions.append({'name': name.strip(), 'state': state, 'version': int(version), 'managed': False})
    if paths is not None and any(item['name'] == 'ModLoader' for item in distributions):
        from .backends.installers import _verify_wsl_owner
        try:
            _verify_wsl_owner(paths)
        except (OSError, ValueError, ImportError):
            pass
        else:
            next(item for item in distributions if item['name'] == 'ModLoader')['managed'] = True
    return {'available': result.returncode == 0, 'distributions': distributions}


def require_wsl2(distribution):
    distribution_name(distribution)
    listing = list_wsl()
    found = next((item for item in listing['distributions'] if item['name'] == distribution), None)
    if not found or found['version'] != 2:
        raise ValueError('所选发行版未登记为 WSL2；请先安装或转换，再重新检查')
    return found


def wsl_python(distribution, code, args=(), *, timeout=60, input=None):
    code = 'import signal; signal.alarm(' + str(int(timeout)) + ')\n' + code
    return _run([*wsl_prefix(distribution), '/usr/bin/python3', '-c', code, *map(str, args)], timeout=timeout + 2, input=input)


def check_wsl(distribution):
    found = require_wsl2(distribution)
    result = {'distribution': distribution, 'version': found['version'], 'python': None, 'gpu': None, 'ok': False, 'errors': []}
    try:
        result['python'] = _run([*wsl_prefix(distribution), '/usr/bin/python3', '--version'])
    except ValueError as error:
        result['errors'].append(str(error))
    try:
        result['gpu'] = wsl_python(distribution, "import shutil,subprocess; p=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'; subprocess.run([p,'--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],check=True)")
    except ValueError as error:
        result['errors'].append(str(error))
    result['ok'] = not result['errors']
    return result


def windows_to_wsl(path, distribution='ModLoader'):
    # Query the selected distribution's actual mount configuration (including UNC
    # mounts); do not assume /mnt/<drive> or narrow legacy Windows path support.
    if not PureWindowsPath(path).is_absolute():
        raise ValueError('Windows 模型和工作目录必须是绝对路径')
    return linux_path(_run([*wsl_prefix(distribution), 'wslpath', '-a', '-u', str(path)]))


def windows_path(distribution, path):
    return '\\\\wsl.localhost\\' + distribution_name(distribution) + linux_path(path).replace('/', '\\')


def model_filesystem_path(model, path=None):
    location = location_of(model)
    value = path or (model['path'] if isinstance(model, dict) else model.path)
    return Path(windows_path(location.distribution, value) if location.platform == 'wsl2' else value)


def model_target_path(model, runtime):
    validate_model_runtime(model, runtime)
    if location_of(model).platform == 'wsl2':
        return linux_path(model.path)
    if runtime.platform == 'wsl2':
        return windows_to_wsl(str(Path(model.path).resolve()), runtime_distribution(runtime))
    return str(Path(model.path).resolve())


def path_stat(distribution, path):
    require_wsl2(distribution)
    code = "import os,json,sys; p=sys.argv[1]; s=os.stat(p); print(json.dumps(dict(path=os.path.realpath(p),is_file=os.path.isfile(p),is_dir=os.path.isdir(p),executable=os.access(p,os.X_OK),size_bytes=s.st_size)))"
    return json.loads(wsl_python(distribution, code, [linux_path(path)]))


def model_exists(model):
    location = location_of(model)
    if location.platform == 'windows':
        return Path(model.path).exists()
    try:
        return bool(path_stat(location.distribution, model.path))
    except ValueError:
        return False


def browse_wsl(distribution, path):
    require_wsl2(distribution)
    code = """import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
if not root.is_dir(): raise ValueError('请选择目录')
entries=[]
for entry in sorted(root.iterdir(), key=lambda p:(not p.is_dir(),p.name)):
    try:
        entries.append(dict(name=entry.name,path=str(entry),type='directory' if entry.is_dir() else 'file',size_bytes=entry.stat().st_size if entry.is_file() else 0))
    except OSError: continue
print(json.dumps(dict(path=str(root),parent=str(root.parent),entries=entries),ensure_ascii=False))
"""
    return {'distribution': distribution, **json.loads(wsl_python(distribution, code, [linux_path(path or '/')]))}


def prepare_launch(spec, context):
    if spec.platform != 'wsl2':
        return spec
    distribution = distribution_name(spec.distribution)
    require_wsl2(distribution)
    work = Path(context.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    record = work / 'wsl-process.json'
    linux_record = windows_to_wsl(str(record.resolve()), distribution)
    prefix = [*wsl_prefix(distribution), '/usr/bin/python3', '-u', '-c', WSL_SUPERVISOR]
    payload = json.dumps({'argv': spec.argv, 'cwd': linux_path(spec.cwd), 'env': spec.env}, ensure_ascii=False)
    return spec.model_copy(update={'argv': [*prefix, 'run-config', linux_record, context.instance_id, payload],
                                   'cwd': str(work), 'env': {'WSL_UTF8': '1'}, 'guest_record': str(record),
                                   'cleanup_argv': [*prefix, 'stop', linux_record, context.instance_id]})


def register_platform_routes(app, paths=None):
    import asyncio
    @app.get('/api/platforms/wsl')
    async def listing():
        return await asyncio.to_thread(list_wsl, paths)

    @app.post('/api/platforms/wsl/check')
    async def check(body: dict):
        return await asyncio.to_thread(check_wsl, body.get('distribution'))

    @app.post('/api/platforms/wsl/browse')
    async def browse(body: dict):
        return await asyncio.to_thread(browse_wsl, body.get('distribution'), body.get('path', '/'))

def remote_asset_operation(distribution, operation, payload):
    """Run the same metadata parser inside Linux, never copy model payloads.

    The guest uses Python's standard library only; pydantic remains in the host.
    The parser source is sent over stdin, not installed into external distros.
    """
    source = Path(__file__).with_name('assets.py').read_text(encoding='utf-8')
    source = source.replace('from .models import ModelAsset',
                            'from types import SimpleNamespace\nclass ModelAsset(SimpleNamespace):\n    def model_dump(self): return vars(self)')
    mtp_source = Path(__file__).with_name('mtp.py').read_text(encoding='utf-8').replace('from .assets import gguf_metadata, GGUF_SHARD', '').replace('from .models import ParameterField', 'ParameterField = ModelAsset') if operation == 'weight_info' else ''
    code = """import json,sys
from pathlib import Path
request=json.load(sys.stdin)
namespace={}
exec(compile(request['source'],'modloader_asset_inspection','exec'),namespace)
if request.get('mtp_source'): exec(compile(request['mtp_source'],'modloader_mtp_inspection','exec'),namespace)
operation=request['operation']; payload=request['payload']
if operation=='projector_candidates':
    payload.pop('location',None)
    result=namespace[operation](namespace['ModelAsset'](**payload))
else:
    result=namespace[operation](**payload)
print(json.dumps(result,default=lambda value:vars(value),ensure_ascii=False))
"""
    return json.loads(wsl_python(distribution, code, timeout=300,
                                input=json.dumps({'source': source, 'operation': operation, 'payload': payload, 'mtp_source': mtp_source})))


def inspect_wsl_asset(operation, path, location, **kwargs):
    from .models import ModelAsset
    location = ModelLocation.model_validate(location)
    require_wsl2(location.distribution)
    result = remote_asset_operation(location.distribution, operation, {'path': linux_path(path), **kwargs})
    def model(value):
        return ModelAsset(**{**value, 'location': location.model_dump(), 'managed': False})
    if operation == 'scan_model_folder':
        result['models'] = [model(value) for value in result['models']]
        return result
    return model(result)
