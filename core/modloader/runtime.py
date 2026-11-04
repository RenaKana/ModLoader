import asyncio
import ctypes
import json
import os
import re
import socket
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil
from .models import LaunchContext, LoadProfile, ModelAsset, RuntimeInstallation, RuntimeInstance

from .platforms import validate_model_runtime, model_exists, runtime_distribution, prepare_launch

ACTIVE_STATES = ('preparing', 'starting', 'ready', 'stopping')


class WindowsJob:
    """Kernel-owned lifetime boundary for one model process tree."""
    def __init__(self, pid):
        self.handle = None
        if os.name != 'nt':
            return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [('per_process', ctypes.c_int64), ('per_job', ctypes.c_int64), ('flags', w.DWORD), ('minimum', ctypes.c_size_t), ('maximum', ctypes.c_size_t), ('active', w.DWORD), ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        process = self.kernel.OpenProcess(0x0100 | 0x0001, False, pid)
        try:
            if not process or not self.kernel.AssignProcessToJobObject(self.handle, process):
                self.close()
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if process:
                self.kernel.CloseHandle(process)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def scrub(line):
    line = re.sub(r'''(?i)((?:authorization|api[-_]?key|token|password)["']?\s*[:=]\s*)(["']).*?\2''', r'\1\2[REDACTED]\2', line)
    line = re.sub(r'''(?i)(authorization[\s"'=:]+)(?:bearer|basic)\s+[^\s,"'}]+''', r'\1[REDACTED]', line)
    line = re.sub(r'''(?i)(authorization|api[-_]?key|token|password)([\s"'=:]+)[^\s,"'}]+''', r'\1\2[REDACTED]', line)
    line = re.sub(r'https?://\S+\?\S+', '[URL with query redacted]', line)
    return line


def loading_summary(line):
    """Keep the latest diagnostic readable without turning it into estimated progress."""
    line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', str(line))
    line = scrub(' '.join(line.split()))
    # Only the compact readout is shortened; the diagnostic log keeps the full line.
    return line if len(line) <= 240 else line[:237] + '…'


def local_process_environment(overrides=None):
    environment = os.environ.copy()
    for key in list(environment):
        if any(word in key.upper() for word in ('API_KEY','ACCESS_TOKEN','AUTH_TOKEN','MCP')) or key.upper().startswith('LLAMA_ARG_') or key.upper() in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','STRATA_DEBUG','STRATA_BATCH_TRACE','STRATA_REQUEST_LINES','LLAMA_LOG_FILE','LLAMA_LOG_VERBOSITY','LLAMA_SERVER_SLOTS_DEBUG'):
            environment.pop(key)
    environment.update(overrides or {})
    environment.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    return environment


class RuntimeManager:
    def __init__(self, paths, adapters, preparation=None):
        self.paths, self.adapters = paths, adapters
        self.preparation = preparation
        self.instances = {}
        self.processes = {}
        self.jobs = {}
        self.specs = {}
        self.logs = {}
        self.tasks = {}
        self.requests = {}
        self.lock = asyncio.Lock()

    def snapshot(self):
        return [instance.model_dump() for instance in self.instances.values()]

    async def preview(self, model, profile, runtime):
        import tempfile
        validate_model_runtime(model, runtime)
        preparation = None
        if profile.backend == 'strata':
            from .strata_prefetch import validate_profile as validate_prefetch
            await asyncio.to_thread(validate_prefetch, model, profile, runtime)
        if profile.backend in {'strata', 'ninfer', 'sparkinfer'} and self.preparation:
            preparation = await asyncio.to_thread(self.preparation.preflight, model, profile, runtime)
            if preparation['status'] != 'ready':
                return {'argv': [], 'cwd': runtime.path, 'preparation': preparation,
                        'compatibility': self.adapters[profile.backend].inspect(model, runtime).model_dump()}
            profile = await asyncio.to_thread(self.preparation.resolved_profile, model, profile, runtime)
        with tempfile.TemporaryDirectory(dir=self.paths.work) as temporary:
            context = LaunchContext(port=free_port(), instance_id='preview', work_dir=temporary, data_dir=str(self.paths.root))
            adapter = self.adapters[profile.backend]
            compatibility = await asyncio.to_thread(adapter.inspect, model, runtime)
            spec = await asyncio.to_thread(adapter.build_launch, model, profile, runtime, context)
            config = None
            if profile.backend == 'strata':
                # Materialize the effective engine configuration before the preview
                # directory is released; a dangling --config path is not a preview.
                config = json.loads((Path(temporary) / 'strata.json').read_text(encoding='utf-8'))
            return {'argv': spec.argv, 'cwd': spec.cwd, 'platform': spec.platform, 'distribution': spec.distribution, 'env': {k: scrub(v) for k,v in spec.env.items() if not any(word in k.lower() for word in ('key','token','secret'))}, 'config': config, 'compatibility': compatibility.model_dump(), 'preparation': preparation}

    async def start(self, model: ModelAsset, profile: LoadProfile, runtime: RuntimeInstallation, *, allow_parallel=False):
        async with self.lock:
            if profile.backend in {'strata', 'ninfer', 'sparkinfer'} and self.preparation:
                from .parameter_rules import validate_profile
                await asyncio.to_thread(validate_profile, model, profile, runtime, self.adapters[profile.backend], legacy_gpu=True)
            if not allow_parallel and any((x.profile_id == profile.id or x.alias == profile.alias) and x.state in ACTIVE_STATES for x in self.instances.values()):
                raise ValueError('该模型别名已有运行实例，请先卸载或使用另一套别名')
            if profile.backend in {'strata', 'ninfer'} and any(x.model_id == model.id and x.backend == profile.backend and x.state == 'preparing' for x in self.instances.values()):
                raise ValueError('该模型正在准备运行文件，请等待准备完成后再加载其他配置')
            validate_model_runtime(model, runtime)
            if not await asyncio.to_thread(model_exists, model):
                raise ValueError('模型文件已移动或不可访问，请重新导入')
            adapter = self.adapters[profile.backend]
            compatibility = await asyncio.to_thread(adapter.inspect, model, runtime)
            if compatibility.status == 'unsupported':
                raise ValueError('；'.join(compatibility.reasons))
            wait_seconds = None
            if profile.parameters.get('startup_timeout') is not None:
                import math
                wait_seconds = float(profile.parameters['startup_timeout'])
                if not math.isfinite(wait_seconds) or wait_seconds <= 0:
                    raise ValueError('启动等待时间必须大于零')
            preparing = False
            if profile.backend in {'strata', 'ninfer', 'sparkinfer'} and self.preparation:
                status = await asyncio.to_thread(self.preparation.preflight, model, profile, runtime)
                if status['status'] in ('error', 'choice_required'):
                    raise ValueError('；'.join(status.get('errors') or [status['message']]))
                preparing = status['status'] == 'needs_preparation'
                if not preparing:
                    profile = await asyncio.to_thread(self.preparation.resolved_profile, model, profile, runtime)
            identifier = uuid.uuid4().hex
            alias = profile.alias
            used_aliases = {x.alias for x in self.instances.values() if x.state in ACTIVE_STATES}
            suffix = 2
            while alias in used_aliases:
                alias = f'{profile.alias}--{suffix}'
                suffix += 1
            work = self.paths.work / identifier
            work.mkdir(parents=True)
            port = free_port()
            context = LaunchContext(port=port, instance_id=identifier, work_dir=str(work), data_dir=str(self.paths.root))
            spec = None if preparing else await asyncio.to_thread(adapter.build_launch, model, profile, runtime, context)
            if spec and wait_seconds is not None:
                spec.startup_timeout = wait_seconds
            instance = RuntimeInstance(id=identifier, model_id=model.id, profile_id=profile.id, profile_revision=profile.revision, alias=alias, backend=profile.backend, runtime_id=runtime.id, platform=runtime.platform, distribution=runtime_distribution(runtime) if runtime.platform == 'wsl2' else None, port=port, started_at=datetime.now(timezone.utc).isoformat(), capabilities=(spec.capabilities if spec else None) or compatibility.capabilities,
                                       state='preparing' if preparing else 'starting', phase='preparing' if preparing else 'loading', message='正在生成本地运行文件' if preparing else '正在加载模型')
            self.instances[identifier] = instance
            if spec:
                self.specs[identifier] = spec
            self.logs[identifier] = deque(maxlen=1000)  # Bounded diagnostic buffer, not a model/data limit.
            self.requests[identifier] = set()
            if preparing:
                self.tasks[identifier] = asyncio.create_task(self._prepare_and_run(instance, model, profile, runtime, context, wait_seconds or (7200 if profile.backend == 'ninfer' else 600)))
            else:
                self.tasks[identifier] = asyncio.create_task(self._run(instance, spec))
            return instance

    async def _prepare_and_run(self, instance, model, profile, runtime, context, timeout):
        started = time.monotonic()
        async def progress(update):
            instance.phase = update.get('phase', 'preparing')
            message = scrub(str(update.get('message') or '正在生成本地运行文件'))
            instance.message = loading_summary(message)
            self.logs[instance.id].append(message)
        try:
            resolved = await asyncio.wait_for(self.preparation.ensure(model, profile, runtime, progress), timeout)
            spec = await asyncio.to_thread(self.adapters[profile.backend].build_launch, model, resolved, runtime, context)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError()
            spec.startup_timeout = remaining
            self.specs[instance.id] = spec
            instance.capabilities = spec.capabilities
            instance.state, instance.phase, instance.message = 'starting', 'loading', '运行文件已准备，正在加载模型'
            self.logs[instance.id].append(instance.message)
            await self._run(instance, spec)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            instance.state = 'failed'
            instance.phase = 'failed'
            instance.error = '模型准备超时；可检查日志后调整启动等待时间' if isinstance(error, TimeoutError) else scrub(str(error))
            instance.message = instance.error
            self.logs[instance.id].append(instance.error)

    async def _run(self, instance, spec):
        identifier = instance.id
        try:
            instance.phase, instance.message = 'launching', '正在启动后端进程'
            context = LaunchContext(port=instance.port, instance_id=identifier, work_dir=str(self.paths.work / identifier), data_dir=str(self.paths.root))
            spec = await asyncio.to_thread(prepare_launch, spec, context)
            self.specs[identifier] = spec
            environment = local_process_environment(spec.env)
            process = await asyncio.create_subprocess_exec(*spec.argv, cwd=spec.cwd, env=environment, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, creationflags=0x08000000 if os.name == 'nt' else 0, limit=1024*1024)
            self.processes[identifier] = process
            instance.pid = process.pid
            instance.phase, instance.message = 'loading', '后端已启动，正在加载模型'
            try:
                self.jobs[identifier] = WindowsJob(process.pid)
            except Exception:
                process.kill()
                await process.wait()
                raise
            reader = asyncio.create_task(self._read_output(instance, process))
            deadline = time.monotonic() + spec.startup_timeout
            async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
                while instance.state == 'starting':
                    if process.returncode is not None:
                        raise RuntimeError(f'后端退出，退出码 {process.returncode}。请查看加载日志。')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('模型加载超时；可检查日志后调整启动等待时间')
                    try:
                        response = await client.get(f'http://127.0.0.1:{instance.port}{spec.health_path}')
                        ready = response.is_success
                        if ready and spec.health_expected:
                            try:
                                status = response.json()
                                ready = all(status.get(k) == value for k,value in spec.health_expected.items())
                            except (ValueError, AttributeError):
                                ready = False
                        if ready:
                            instance.state = 'ready'
                            instance.phase, instance.message = 'ready', '模型已就绪'
                            self.logs[identifier].append('模型已就绪')
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.5)
            code = await process.wait()
            await reader
            if instance.state not in ('stopping','stopped'):
                instance.state = 'failed'
                instance.phase = 'failed'
                instance.error = f'后端意外退出（{code}）'
                instance.message = instance.error
        except asyncio.CancelledError:
            pass
        except Exception as error:
            instance.state = 'failed'
            instance.phase = 'failed'
            instance.error = scrub(str(error))
            instance.message = instance.error
            self.logs[identifier].append(instance.error)
        finally:
            await self._cleanup(identifier)
            if instance.state == 'stopping':
                instance.state = 'stopped'

    async def _read_output(self, instance, process):
        while True:
            try:
                raw = await process.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError):
                raw = await process.stdout.read(65536)
            if not raw:
                break
            # Runtime stdout may include prompts at verbose log levels. Do not retain
            # inference-time output; management logs contain lifecycle diagnostics only.
            if raw.startswith(b'MODLOADER_GUEST_PID='):
                try:
                    instance.guest_pid = int(raw.split(b'=', 1)[1].strip())
                except ValueError:
                    pass
                continue
            if instance.state == 'starting':
                line = scrub(raw.decode('utf-8', errors='replace').rstrip())
                self.logs[instance.id].append(line)
                summary = loading_summary(line)
                if summary:
                    instance.message = summary

    async def _cleanup(self, identifier):
        spec = self.specs.get(identifier)
        process = self.processes.get(identifier)
        if process and process.stdin:
            try:
                process.stdin.close()  # WSL supervisor terminates its own group on EOF.
            except Exception:
                pass
        if spec and spec.cleanup_argv and process:
            try:
                cleanup = await asyncio.create_subprocess_exec(*spec.cleanup_argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, creationflags=0x08000000 if os.name == 'nt' else 0)
                try:
                    await asyncio.wait_for(cleanup.wait(), 15)
                except TimeoutError:
                    cleanup.kill()
                    await cleanup.wait()
            except OSError:
                pass
        job = self.jobs.pop(identifier, None)
        if job:
            job.close()
        if process and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    async def stop(self, identifier, force=False):
        instance = self.instances[identifier]
        if instance.active_requests and not force:
            raise RuntimeError('模型仍有请求正在生成，确认后将中断这些请求并卸载')
        if instance.state in ('stopped','failed'):
            return instance
        instance.state = 'stopping'
        instance.phase, instance.message = 'stopping', '正在停止所属后端进程'
        for response in list(self.requests[identifier]):
            await response.aclose()
        task = self.tasks.get(identifier)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        instance.state = 'stopped'
        instance.phase, instance.message = 'stopped', '实例已停止'
        self.logs[identifier].append('实例已停止，所属后端进程已清理')
        return instance

    async def shutdown(self):
        await asyncio.gather(*(self.stop(key, True) for key in self.instances), return_exceptions=True)
