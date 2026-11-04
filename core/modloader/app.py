from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
import urllib.request

import httpx
import psutil
from anyio import CancelScope
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .assets import inspect_asset, projector_candidates
from .platforms import model_identity, location_of, model_filesystem_path, validate_model_runtime
from .conversations import Conversations, ConversationNotFound
from .chat_presets import ChatPresets, ChatPresetNotFound
from .chat_documents import read_chat_document
from .local_tools import LocalTools, LocalToolError
from .local_resources import LocalResources, create_local_resource_router
from .models import AppPaths, ModelAsset, LoadProfile, RuntimeInstallation, DownloadRequest
from .runtime import RuntimeManager, scrub, ACTIVE_STATES
from .runtime_discovery import discover_runtimes
from .operation_progress import operation_response
from .resource_estimates import ResourceEstimator
from .request_history import RequestHistory, validate_date
from .request_metrics import RequestMetrics, request_metrics_group, generation_window, generation_metrics_group, SNAPSHOT_FIELDS, DECODE_PHASES
from .generation_events import GenerationEvents, request_accounting
from .hardware import GpuProbe, cpu_temperature, gpu_temperatures, system_specs, memory_commit_available, parse_nvidia_smi
from .gpu_identity import hardware_devices
from .cpu_sensor import CpuSensorReader
from .secrets import Vault
from .store import Store, UNFINISHED_JOBS, TERMINAL_JOBS
from .web_search import DEFAULT_SEARXNG_URL, DEFAULT_IMAGE_ENGINE, validate_http_url, validate_image_engine
from .network_services import DEFAULT_NETWORK_SETTINGS, WEB_SERVICES, WEB_CREDENTIAL_KEYS, PROVIDER_SETTINGS, FALLBACK_SETTINGS, require_service, run_network_service, run_network_route, validate_service_url, validate_service_secret, validate_openai_search_model, validate_openai_image_model, validate_openai_read_model, validate_fallbacks

CPU_TEMPERATURE_INTERVAL_S = 1
CPU_TEMPERATURE_RETRY_S = 30


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def _transformers_generation_source(values):
    result = dict(values)
    raw = result.get('generation_config')
    if raw is None:
        config = {}
    else:
        try:
            config = json.loads(raw) if isinstance(raw, str) else None
        except ValueError as error:
            raise ValueError('generation_config 必须是 JSON 对象字符串') from error
        if not isinstance(config, dict):
            raise ValueError('generation_config 必须是 JSON 对象字符串')
    overrides = {key: config[key] for key in ('top_k', 'repetition_penalty') if key in config}
    for key, native in [('top_k', 'top_k'), ('repeat_penalty', 'repetition_penalty'),
                        ('repetition_penalty', 'repetition_penalty')]:
        if key in result:
            value = result.pop(key)
            if native in overrides and overrides[native] != value:
                raise ValueError(f'{key} 与 {native} / generation_config 冲突')
            overrides[native] = value
    if result.get('frequency_penalty') is not None:
        try:
            value = 1.0 + float(result.pop('frequency_penalty'))
        except (TypeError, ValueError) as error:
            raise ValueError('frequency_penalty 必须是有限数字') from error
        if 'repetition_penalty' in overrides and overrides['repetition_penalty'] != value:
            raise ValueError('重复惩罚与 frequency_penalty 冲突')
        overrides['repetition_penalty'] = value
    return result, config, overrides


def _transformers_default_output(model):
    """Preserve serve's max(1024, model max_new_tokens) when adding request JSON."""
    if model is None:
        raise ValueError('当前模型记录缺失，无法读取默认输出长度')
    root = model_filesystem_path(model)
    path = root / 'generation_config.json'
    from_model = False
    try:
        raw = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        path = root / 'config.json'
        raw = path.read_text(encoding='utf-8')
        from_model = True
    try:
        config = json.loads(raw)
    except ValueError as error:
        raise ValueError(f'模型 {path.name} 不是有效 JSON') from error
    if not isinstance(config, dict):
        raise ValueError(f'模型 {path.name} 必须是 JSON 对象')
    value = config.get('max_new_tokens')
    if from_model and value is None:
        # GenerationConfig.from_model_config first uses outer attributes, then
        # fills unset attributes from get_text_config(decoder=True).
        nested = [config[name] for name in ('decoder', 'generator', 'text_config') if config.get(name) is not None]
        if len(nested) > 1:
            raise ValueError('模型 config.json 包含多个文本配置，无法确定默认输出长度')
        if nested:
            if not isinstance(nested[0], dict):
                raise ValueError('模型文本配置必须是 JSON 对象')
            value = nested[0].get('max_new_tokens')
        elif config.get('is_encoder_decoder'):
            value = config.get('decoder_max_new_tokens')
    if value is None:
        return 1024
    if not isinstance(value, (int, float)) or isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'模型 {path.name} 的 max_new_tokens 必须是有限数字')
    return max(1024, value)


def _transformers_generation(defaults, values, argv, model):
    managed = {'top_k', 'repeat_penalty', 'repetition_penalty'}
    if not any(source.get(key) is not None for source in (defaults, values) for key in managed):
        return {key: value for key, value in {**defaults, **values}.items() if key not in managed}
    base, base_config, base_overrides = _transformers_generation_source(defaults)
    request, request_config, request_overrides = _transformers_generation_source(values)
    result = {**base, **request}
    config = dict(request_config if 'generation_config' in values else base_config)
    overrides = {**base_overrides, **request_overrides}
    if not any(value is not None for value in overrides.values()):
        return result
    continuous_batching = False
    for arg in argv:
        if arg in {'--continuous-batching', '--no-continuous-batching'}:
            continuous_batching = arg == '--continuous-batching'
    if continuous_batching:
        raise ValueError('Transformers 的 Top K 和重复惩罚逐请求设置需要关闭连续批处理')
    from .parameter_rules import validate_generation
    validate_generation({'top_k': overrides.get('top_k'), 'repeat_penalty': overrides.get('repetition_penalty')},
                        'transformers')
    if result.get('generation_config') is None and result.get('max_tokens') is None:
        config['max_new_tokens'] = _transformers_default_output(model)
    config.update(overrides)
    result['generation_config'] = json.dumps(config, ensure_ascii=False, allow_nan=False)
    return result


class Controller:
    def __init__(self, root, token):
        from .backends import get_adapters
        from .sources import SourceService
        self.paths = AppPaths(Path(root))
        self.store = Store(self.paths.root / 'modloader.sqlite3')
        self.conversations = Conversations(self.store)
        self.chat_presets = ChatPresets(self.store)
        self.chat_tools = LocalTools()
        self.local_resources = LocalResources(self.store)
        self.conversations.interrupt_generating()
        self.vault = Vault(self.paths.root / 'credentials.dpapi.json')
        self.api_key = self.vault.ensure('inference_api_key')
        self.token = token
        self.settings = {'id':'main', 'api_port':12340, 'close_to_tray':True, 'chat_unload_before_load':True, 'chat_tool_details':'compact', 'font_scale':1, 'proxy_mode':'system', 'proxy_url':'', 'searxng_url':DEFAULT_SEARXNG_URL, 'searxng_image_engine':DEFAULT_IMAGE_ENGINE, 'model_directory':str(self.paths.models), 'api_history_enabled':True, 'api_history_retention_days':0, 'api_history_directory':str(self.paths.root / 'logs' / 'api-requests'), **DEFAULT_NETWORK_SETTINGS, **(self.store.get('settings', 'main') or {})}
        self.adapters = get_adapters()
        from .engine_preparation import PreparationDispatcher
        self.preparation = PreparationDispatcher(self.paths, self.store, proxy_provider=self.proxy)
        self.runtime = RuntimeManager(self.paths, self.adapters, self.preparation)
        self.resource_estimator = ResourceEstimator()
        self.sources = SourceService(self.paths, self.vault.get, self.proxy)
        self.job_tasks, self.cancel_events, self.download_pauses, self.job_records = {}, {}, {}, {}
        self.install_plans, self.stage_confirmations = {}, {}
        self.environment_lock = asyncio.Lock()
        from .model_storage import ModelStorageMigration
        self.model_storage = ModelStorageMigration(self.paths, self.store, self.runtime, self.environment_lock)
        self.shutting_down = False
        self.hardware = {'cpu_percent':None, 'memory_total':None, 'memory_used':None, 'memory_commit_available':None, 'gpu':None}
        self.cpu_sensor = CpuSensorReader()
        self.cpu_sensor_lock = asyncio.Lock()
        self.cpu_temperature_wake = asyncio.Event()
        self.cpu_enable_task = None
        self.temperature_auto_started = False
        self.external_cpu_retry_at = 0.0
        self.metrics_task = None
        self.public_server = None
        self.public_task = None
        self.api_error = None
        self.request_metrics = {}
        self.request_history = RequestHistory(self.settings['api_history_directory'], self.settings['api_history_enabled'], self.settings['api_history_retention_days'], infer_turns=False)
        self.generation_events = GenerationEvents(self.request_history.run_id)
        self.telemetry_samples, self.telemetry_locks = {}, {}
        self.active_request_tokens, self.generation_samples = {}, {}
        self.active_request_metrics = {}
        for job in self.store.all('jobs'):
            changed = 'can_pause' not in job
            job.setdefault('can_pause', False)
            if job['state'] in UNFINISHED_JOBS:
                job.update(state='interrupted', can_pause=False, error='应用上次退出时任务未完成，可以重试继续')
                changed = True
            if changed:
                self.store.put('jobs', job)

    def generation_event_metrics(self, identifier):
        instance = self.runtime.instances.get(identifier)
        tokens = self.active_request_tokens.get(identifier, frozenset())
        if instance is not None and instance.active_requests == 1 and len(tokens) == 1:
            return self.active_request_metrics.get(next(iter(tokens)))
        return None

    def generation_event_basis(self, metrics):
        return self.request_history.generation_basis(metrics.token) if metrics is not None else None

    def publish_generation_event(self, identifier, chunk=None, *, reset=False, finalized=None):
        if self.generation_events.subscribed(identifier):
            current = self.generation_event_metrics(identifier)
            closed = [request_accounting(finalized, self.generation_event_basis(finalized))] if finalized is not None else []
            self.generation_events.publish(identifier, current, chunk, reset=reset,
                turn=self.generation_event_basis(current), finalized=closed)

    def proxy(self):
        mode = self.settings.get('proxy_mode', 'system')
        if mode == 'direct':
            return ''
        if mode == 'manual':
            return self.settings.get('proxy_url') or ''
        # System mode follows Windows settings, not proxy variables inherited from a launcher.
        proxies = urllib.request.getproxies_registry() if os.name == 'nt' else urllib.request.getproxies()
        return proxies.get('https') or proxies.get('http') or ''

    def require(self, kind, identifier):
        item = self.store.get(kind, identifier)
        if item is None:
            raise HTTPException(404, '记录不存在')
        return item

    def assert_model_available(self, identifier):
        if any(job.get('request', {}).get('model_id') == identifier
               for job in self.store.jobs('model_migration', UNFINISHED_JOBS)):
            raise HTTPException(409, '模型正在迁移，请等待完成或停止迁移后再操作')
        self.model_storage.assert_available(identifier)

    def assert_runtime_available(self, identifier):
        for profile in self.store.all('profiles'):
            if profile['runtime_id'] == identifier:
                self.assert_model_available(profile['model_id'])

    def safe_settings(self):
        return {**self.settings, 'api_history_status':self.request_history.archive.status(), 'hf_configured':bool(self.vault.get('hf')), 'modelscope_configured':bool(self.vault.get('modelscope')), 'api_key_configured':True, 'api_error':self.api_error, 'api_url':f"http://127.0.0.1:{self.settings['api_port']}/v1",
                'web_credentials':{key:bool(self.vault.get(key)) for key in WEB_CREDENTIAL_KEYS}, 'web_services':WEB_SERVICES}

    async def metrics(self):
        probe = GpuProbe()
        specs = asyncio.create_task(asyncio.to_thread(system_specs))
        temperature = asyncio.create_task(self._sample_cpu_temperature())
        last_disk, last_time = psutil.disk_io_counters(), time.monotonic()
        primed = False
        try:
            await self._sample_metrics(probe, specs, last_disk, last_time, primed)
        finally:
            temperature.cancel()
            await asyncio.gather(temperature, return_exceptions=True)
            await asyncio.to_thread(probe.close)
            if not specs.done():
                specs.cancel()

    async def _sample_cpu_temperature(self):
        await self._auto_start_temperatures()
        # After the one startup attempt, polling never requests elevation again.
        while True:
            self.cpu_temperature_wake.clear()
            async with self.cpu_sensor_lock:
                sample = await asyncio.to_thread(self._read_cpu_temperature)
                self.hardware.update(sample)
            available = sample['cpu_temperature_c'] is not None or sample.get('cpu_power_w') is not None or any(
                item.get('temperature_c') is not None for item in sample.get('gpu_temperatures', []))
            interval = CPU_TEMPERATURE_INTERVAL_S if available else CPU_TEMPERATURE_RETRY_S
            try:
                await asyncio.wait_for(self.cpu_temperature_wake.wait(), interval)
            except TimeoutError:
                pass

    async def _auto_start_temperatures(self):
        if self.temperature_auto_started:
            return
        self.temperature_auto_started = True
        # Headless/fixture runs can opt out; normal desktop startup is automatic.
        if os.name == 'nt' and os.environ.get('MODLOADER_SENSOR_AUTO_START') != '0':
            await self.enable_cpu_temperature()

    def _read_cpu_temperature(self):
        native = self.cpu_sensor.sample()
        if native['cpu_temperature_c'] is not None:
            return native
        # Keep the existing external sensor path usable without new privileges.
        # A missing CPU provider must not delay otherwise healthy GPU samples.
        now = time.monotonic()
        if now < self.external_cpu_retry_at:
            return native
        external = cpu_temperature()
        self.external_cpu_retry_at = now + (CPU_TEMPERATURE_RETRY_S if external['cpu_temperature_c'] is None else 0)
        if external['cpu_temperature_c'] is not None:
            return {**native, **external, 'cpu_temperature_status':'ready'}
        return native if os.name == 'nt' else {**native, **external, 'cpu_temperature_status':'disabled'}

    async def enable_cpu_temperature(self):
        if self.shutting_down:
            raise HTTPException(503, '应用正在退出')
        if self.cpu_enable_task is None or self.cpu_enable_task.done():
            async def enable():
                async with self.cpu_sensor_lock:
                    self.hardware.update(cpu_temperature_c=None, cpu_temperature_source=None, cpu_power_w=None, cpu_power_source=None,
                                         cpu_power_reason='正在连接硬件传感器',
                                         cpu_temperature_status='starting', cpu_temperature_reason='正在连接硬件温度读取器，请确认 Windows 授权')
                    sample = await asyncio.to_thread(self.cpu_sensor.enable)
                    self.hardware.update(sample)
                    self.cpu_temperature_wake.set()
                    return sample
            self.cpu_enable_task = asyncio.create_task(enable())
        # A closing browser request must not orphan a helper already authorized.
        return await asyncio.shield(self.cpu_enable_task)

    async def _sample_metrics(self, probe, specs, last_disk, last_time, primed):
        while True:
            try:
                memory = psutil.virtual_memory()
                self.hardware['memory_commit_available'] = memory_commit_available()
                disk, now = psutil.disk_io_counters(), time.monotonic()
                elapsed=now-last_time
                cpu, cores = psutil.cpu_percent(), psutil.cpu_percent(percpu=True)
                self.hardware.update(cpu_percent=cpu if primed else None, cpu_per_core=cores if primed else [], memory_total=memory.total, memory_available=memory.available, memory_used=memory.total-memory.available, cpu_count=psutil.cpu_count(), disk_read_bytes_per_second=(disk.read_bytes-last_disk.read_bytes)/elapsed if disk and last_disk and elapsed>0 and disk.read_bytes>=last_disk.read_bytes else None)
                primed = True
                if specs is not None and specs.done():
                    self.hardware.update(specs.result())
                    specs = None
                self.hardware.pop('error',None)
                last_disk, last_time = disk, now
                if probe.lib:
                    try:
                        devices = await asyncio.to_thread(probe.sample)
                        devices = self._gpu_temperature_readings(devices)
                        self.hardware.update(gpus=devices, gpu=devices[0] if devices else None)
                        self.hardware.pop('gpu_error', None)
                    except Exception as error:
                        devices = [{**spec, **dict.fromkeys(('memory_used','memory_free','utilization','temperature','power','pcie_rx_bytes_s','pcie_tx_bytes_s','pcie_gen','pcie_width'))} for spec in probe.specs.values()]
                        devices = hardware_devices(devices, getattr(probe, 'cuda_devices', []))
                        self.hardware.update(gpus=devices, gpu=devices[0] if devices else None, gpu_error=scrub(str(error)))
                elif shutil.which('nvidia-smi'):
                    proc = await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,pci.bus_id','--format=csv,noheader,nounits', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, creationflags=0x08000000 if os.name=='nt' else 0)
                    try:
                        out,_ = await asyncio.wait_for(proc.communicate(), 5)
                        text = out.decode(errors='replace').strip()
                        devices = parse_nvidia_smi(text) if proc.returncode == 0 else []
                        if devices:
                            devices = hardware_devices(devices, getattr(probe, 'cuda_devices', []))
                            devices = self._gpu_temperature_readings(devices)
                            self.hardware.update(gpus=devices, gpu=devices[0])
                            self.hardware.pop('gpu_error', None)
                        else:
                            self.hardware.update(gpus=[], gpu=None)
                            self.hardware['gpu_error'] = scrub(text) or 'GPU 采样未返回有效数据'
                    except (TimeoutError, IndexError):
                        self.hardware.update(gpus=[], gpu=None)
                        self.hardware['gpu_error'] = 'GPU 采样超时' if proc.returncode is None else 'GPU 采样未返回有效数据'
                        if proc.returncode is None:
                            proc.kill()
                            await proc.wait()
            except asyncio.CancelledError:
                return
            except Exception as error:
                # A failed poll is a gap, never a repeat of the last good value.
                self.hardware.update(cpu_percent=None, cpu_per_core=[], memory_used=None,
                                     memory_available=None, memory_commit_available=None, disk_read_bytes_per_second=None)
                devices = [{**g, **dict.fromkeys(('memory_used','memory_free','utilization','temperature','power','pcie_rx_bytes_s','pcie_tx_bytes_s','pcie_gen','pcie_width'))}
                           for g in self.hardware.get('gpus', [])]
                self.hardware.update(gpus=devices, gpu=devices[0] if devices else None)
                self.hardware['error']=scrub(str(error))
            self.hardware['sampled_at'] = datetime.now(timezone.utc).timestamp()
            await asyncio.sleep(1)

    def _gpu_temperature_readings(self, devices):
        return gpu_temperatures(devices, self.hardware.get('gpu_temperatures'),
                                self.hardware.get('gpu_temperature_reason'))

    def new_job(self, kind, request):
        if kind == 'download':
            self.model_storage.assert_download_available(request.get('destination'))
        identifier = uuid.uuid4().hex
        request = dict(request)
        install_plan = None
        if kind == 'install':
            request['options'] = dict(request.get('options', {}))
            install_plan = request['options'].pop('_plan', None)
        if request.get('url'):
            from .sources._common import public_url
            self.vault.set('download_url_'+identifier,request['url'])
            request['url']=public_url(request['url'])
        job = {'id':identifier, 'kind':kind, 'state':'queued', 'phase':'queued', 'message':'等待开始', 'error':None, 'can_pause':False, 'downloaded_bytes':0, 'total_bytes':None, 'percent':None, 'created_at':timestamp(), 'started_at':None, 'finished_at':None, 'request':request, **{k:v for k,v in request.items() if k in ('source','repo_id','backend','destination','update_from')}}
        if kind == 'install':
            plan = install_plan or {}
            job.update(operation='update' if request.get('update_from') else 'install', version=plan.get('version'),
                       manifest=plan.get('stage'), stages=[plan['stage']] if plan.get('stage') else [])
            if request.get('operation') == 'prepare_startup_prefetch':
                job.update(operation='prepare_startup_prefetch', runtime_id=request['runtime_id'])
        self.store.put('jobs', job)
        self.job_records[identifier] = job
        self.cancel_events[identifier] = asyncio.Event()
        from .sources.control import DownloadPause
        async def paused():
            if job['state'] == 'pausing' and not self.cancel_events[identifier].is_set():
                job.update(state='paused')
                self.store.put('jobs', job)
        self.download_pauses[identifier] = DownloadPause(paused)
        self.job_tasks[identifier] = asyncio.create_task(self.run_job(job, install_plan)) if install_plan else asyncio.create_task(self.run_job(job))
        return job

    def register_download(self, request, result):
        from .assets import scan_model_folder
        from .sources._common import relative_path
        directory = Path(request['destination']).resolve()
        source = {'kind': request['source'], 'repo_id': request.get('repo_id'), 'revision': result.get('revision'), 'managed_root': str(directory)}
        models, errors = [], []
        groups = result.get('model_groups', [])
        if 'model_groups' in result:
            for group in groups:
                try:
                    paths = [(directory / relative_path(p)).resolve() for p in group['files']]
                    if any(not p.is_relative_to(directory) for p in paths):
                        raise ValueError('模型文件超出下载目录')
                    if group['format'].lower() == 'gguf':
                        selected = sorted(p for p in paths if p.suffix.lower() == '.gguf')[0]
                    else:
                        selected = next((p.parent for p in paths if p.name == 'config.json'), paths[0].parent)
                    model = inspect_asset(str(selected), source=source, managed=True)
                    vision_ids = result.get('vision_assignments', {}).get(group['id'], [])
                    projected = [str(directory / relative_path(p)) for v in result.get('vision_groups', []) if v['id'] in vision_ids for p in v['files']]
                    if projected:
                        model.metadata['mmproj_candidates'] = projected
                    model.metadata['mmproj_candidates'] = projector_candidates(model)
                    models.append(model)
                except (ValueError, OSError, KeyError, IndexError) as error:
                    errors.append({'path': group.get('name', ''), 'error': scrub(str(error))})
        else:
            selected = Path(result.get('path') or directory)
            if selected.is_file():
                try:
                    models.append(inspect_asset(str(selected), source=source, managed=True))
                except (ValueError, OSError) as error:
                    errors.append({'path': str(selected), 'error': scrub(str(error))})
            else:
                scan = scan_model_folder(str(selected))
                models, errors = scan['models'], scan['errors']
                for model in models:
                    model.source, model.managed = dict(source), True
        registered = {model_identity(m): m for m in self.store.all('models')}
        ids = []
        for model in models:
            key = model_identity(model)
            if key in registered:
                ids.append(registered[key]['id'])
            else:
                self.store.put('models', model.model_dump())
                registered[key] = model.model_dump()
                ids.append(model.id)
        return ids, errors

    async def run_job(self, job, install_plan=None):
        from .backends import install_backend
        identifier = job['id']
        cancel = self.cancel_events[identifier]
        pause = self.download_pauses[identifier]
        async def progress(update):
            # The UI polls persistent progress; neither tokens nor chat payloads enter it.
            allowed = ('phase','message','file','downloaded_bytes','total_bytes','percent','can_pause',
                       'manifest','file_downloaded_bytes','file_total_bytes','completed_files','total_files')
            if cancel.is_set(): update={**update,'can_pause':False}
            if 'downloaded' in update: update={**update,'downloaded_bytes':update['downloaded']}
            if 'total' in update: update={**update,'total_bytes':update['total']}
            if update.get('phase',job['phase'])!=job['phase'] and 'percent' not in update: job['percent']=None
            job.update({k:v for k,v in update.items() if k in allowed})
            if update.get('manifest'):
                stage = json.loads(json.dumps(update['manifest']))
                stages = job.setdefault('stages', [])
                previous = next((i for i, value in enumerate(stages) if value['id'] == stage['id']), None)
                if previous is None:
                    stages.append(stage)
                else:
                    stages[previous] = stage
            self.store.put('jobs', job)
        async def record_stage(stage):
            if cancel.is_set():
                raise asyncio.CancelledError()
            # Starting the installation covers its required dependency stages.
            # Keep each resolved manifest visible before its files are fetched.
            await progress({'manifest':stage, 'phase':'prepare', 'can_pause':False,
                            'percent':None, 'message':stage['title']})
            if cancel.is_set():
                raise asyncio.CancelledError()
        try:
            if cancel.is_set():
                raise asyncio.CancelledError()
            job.update(state='running', started_at=timestamp())
            self.store.put('jobs', job)
            if job['kind'] == 'install':
                options = {**job['request'].get('options',{}), **({'_plan':install_plan} if install_plan else {})}
                if job.get('operation') == 'prepare_startup_prefetch':
                    from .strata_prefetch import prepare_runtime
                    original = RuntimeInstallation(**self.require('runtimes', job['request']['runtime_id']))
                    record = await prepare_runtime(original, self.paths, progress, cancel, proxy=self.proxy())
                else:
                    record = await install_backend(job['request']['backend'], self.paths, options, progress, cancel, pause=pause, confirm=record_stage, proxy=self.proxy())
                if cancel.is_set():
                    raise asyncio.CancelledError()
                async with self.environment_lock:
                    self.store.put('runtimes', record.model_dump())
                job['runtime_id'] = record.id
            elif job['kind'] == 'model_migration':
                job.update(await self.model_storage.run(job['request'], identifier, progress, cancel))
            else:
                from .sources import SourceService
                request = dict(job['request'])
                original_url = self.vault.get('download_url_'+identifier)
                if original_url: request['url']=original_url
                # Each job retains its selected root if settings change during download.
                sources = SourceService(self.paths,self.vault.get,self.proxy,models_root=Path(request['destination']).parent)
                result = await sources.download(DownloadRequest(**request), progress, cancel, pause=pause)
                if cancel.is_set():
                    raise asyncio.CancelledError()
                await progress({'phase': 'registering', 'message': '登记模型', 'can_pause': False})
                async with self.environment_lock:
                    ids, errors = self.register_download(job['request'], result)
                job['model_ids'], job['import_errors'] = ids, errors
                if ids:
                    job['model_id'] = ids[0]
                if not ids and errors:
                    raise ValueError('文件已下载，模型登记失败：' + '；'.join(item['error'] for item in errors))
            job.update(state='completed', phase='completed', message='已完成', percent=100)
            if job['kind'] == 'download':
                job['message'] = f"下载完成，已登记 {len(job.get('model_ids', []))} 个模型" + (f"；{len(job['import_errors'])} 项未登记，请查看详情" if job.get('import_errors') else '')
            elif job['kind'] == 'model_migration':
                target = 'Windows' if job.get('target_platform') == 'windows' else 'WSL 内部存储'
                job['message'] = f'已迁移到 {target}，原文件已删除' if job.get('source_deleted') else f'已切换到 {target} 副本，原文件保留'
        except asyncio.CancelledError:
            if job.get('operation') == 'prepare_startup_prefetch':
                job.update(state='interrupted' if self.shutting_down else 'cancelled', message='启动预取准备已停止，可重新准备')
            elif job['kind'] == 'model_migration':
                job.update(state='interrupted' if self.shutting_down else 'cancelled', message='迁移已中断，原模型保持可用；重新检查后可重试')
            elif self.shutting_down:
                job.update(state='interrupted', message='应用退出，任务已中断；可重试重新安装' if job['kind']=='install' else '应用退出，下载已中断；可重试')
            else:
                job.update(state='cancelled', message='已停止；可重试重新安装' if job['kind']=='install' else '已停止；可重试下载')
        except Exception as error:
            job.update(state='failed', error=scrub(str(error)), message='任务失败')
        finally:
            job['can_pause'] = False
            job['finished_at'] = timestamp()
            self.stage_confirmations.pop(identifier, None)
            self.store.put('jobs', job)

    async def close(self):
        self.shutting_down = True
        self.chat_tools.close()
        # Signal the reader before awaiting a pending UAC request or sample.
        await asyncio.to_thread(self.cpu_sensor.close)
        if self.metrics_task:
            self.metrics_task.cancel()
            await asyncio.gather(self.metrics_task, return_exceptions=True)
        if self.cpu_enable_task:
            await asyncio.gather(self.cpu_enable_task, return_exceptions=True)
        for event in self.cancel_events.values():
            event.set()
        for pause in self.download_pauses.values():
            pause.resume()
        for task in self.job_tasks.values():
            if not task.done(): task.cancel()
        await asyncio.gather(*self.job_tasks.values(), return_exceptions=True)
        await self.resource_estimator.close()
        await self.runtime.shutdown()
        if self.public_server:
            self.public_server.should_exit = True
        if self.public_task:
            await self.public_task
        await asyncio.to_thread(self.request_history.archive.drain)
        self.store.close()


def build_app(controller: Controller, public=False):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1','localhost','testserver'])
    if not public:
        app.add_middleware(CORSMiddleware, allow_origins=['null','http://127.0.0.1:5173','http://localhost:5173'], allow_methods=['GET','POST','PATCH','DELETE'], allow_headers=['Authorization','Content-Type','X-Modloader-Request-Id','X-Modloader-Call-Id'])

    class Authentication:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            # Pure ASGI preserves the server's disconnect events for requests and streams.
            if scope['type'] == 'http' and scope['method'] != 'OPTIONS':
                value = Headers(scope=scope).get('authorization', '')
                expected = controller.api_key if public else controller.token
                if not hmac.compare_digest(value, 'Bearer '+expected):
                    response = JSONResponse({'error':{'message':'认证失败','type':'authentication_error'}}, status_code=401)
                    return await response(scope, receive, send)
            await self.app(scope, receive, send)

    app.add_middleware(Authentication)

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse({'detail':scrub(str(error))}, status_code=400)

    @app.exception_handler(ConversationNotFound)
    async def missing_conversation(request, error):
        return JSONResponse({'detail':str(error)}, status_code=404)

    @app.exception_handler(ChatPresetNotFound)
    async def missing_chat_preset(request, error):
        return JSONResponse({'detail':str(error)}, status_code=404)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request, error):
        return JSONResponse({'detail':'文件或目录不存在，请重新选择'}, status_code=400)

    @app.exception_handler(PermissionError)
    async def denied_file(request, error):
        return JSONResponse({'detail':'无法访问文件或目录，请检查权限'}, status_code=403)

    async def models():
        return {'object':'list', 'data':[{'id':i.alias,'object':'model','created':0,'owned_by':'local'} for i in controller.runtime.instances.values() if i.state=='ready']}

    async def chat(request: Request):
        payload = await request.json()
        alias = payload.get('model')
        instance_id = payload.pop('instance_id', None) if not public else None
        instance = next((i for i in controller.runtime.instances.values() if (i.id == instance_id if instance_id is not None else i.alias == alias) and i.state == 'ready'), None)
        if not instance:
            return JSONResponse({'error':{'message':'模型未加载或尚未就绪，请先在模型库加载','code':'model_not_loaded'}}, status_code=409)
        if not isinstance(payload.get('messages'), list):
            raise HTTPException(400, 'messages 必须为消息数组')
        images = any(isinstance(m.get('content'),list) and any(p.get('type')=='image_url' for p in m['content'] if isinstance(p,dict)) for m in payload['messages'])
        if images and 'image' not in instance.capabilities.get('input',[]):
            raise HTTPException(400, '当前加载配置未启用图片输入')
        spec = controller.runtime.specs[instance.id]
        profile = controller.require('profiles', instance.profile_id)
        from .parameter_rules import normalize_generation, validate_generation
        defaults = normalize_generation(profile.get('generation', {}), instance.backend)
        payload = normalize_generation(payload, instance.backend)
        payload = (_transformers_generation(defaults, payload, spec.argv, controller.store.get('models', instance.model_id)) if instance.backend == 'transformers'
                   else {**defaults, **payload})
        payload['model'] = spec.model_name
        if not public or instance.backend in {'custom', 'vllm', 'transformers', 'sglang', 'ninfer', 'sparkinfer'}:
            runtime = RuntimeInstallation(**controller.require('runtimes', instance.runtime_id)) if instance.backend in {'custom', 'vllm'} else None
            validate_generation(payload, instance.backend, runtime)
        if instance.backend in {'vllm', 'sglang'}:
            if 'repeat_penalty' in payload:
                payload['repetition_penalty'] = payload.pop('repeat_penalty')
        elif instance.backend in {'ninfer', 'sparkinfer'}:
            from .backends.native_engines import map_generation
            payload = map_generation(payload, instance.backend)
        if payload.get('stream'):
            options = payload.get('stream_options')
            if instance.backend == 'custom' and not spec.capabilities.get('stream_usage', False):
                if options is not None and not isinstance(options, dict):
                    raise HTTPException(400, 'stream_options 必须是对象')
                if isinstance(options, dict) and options.get('include_usage'):
                    raise HTTPException(400, '此自定义引擎未声明支持流式 usage，请关闭 include_usage')
            elif options is None or isinstance(options,dict):
                defaults = {'include_usage': True}
                if instance.backend in {'vllm', 'sglang'}:
                    defaults['continuous_usage_stats'] = True
                payload['stream_options'] = {**defaults, **(options or {})}
            if instance.backend in {'ninfer', 'llama_cpp'}:
                # Official cumulative decode counters; SSE chunks are not tokens.
                payload.setdefault('timings_per_token', True)
            if instance.backend in {'ninfer', 'llama_cpp', 'strata'}:
                payload.setdefault('return_progress', True)
        # This client only targets the literal HTTP loopback URL below and never
        # follows redirects. Loading a TLS trust store would block SSE startup.
        client = httpx.AsyncClient(trust_env=False, verify=False, follow_redirects=False, timeout=httpx.Timeout(600, connect=10))
        instance.active_requests += 1
        metrics = RequestMetrics(payload.get('stream'), backend=instance.backend, single_choice=payload.get('n', 1) == 1,
                                 source='internal' if request.url.path == '/api/chat' else 'external',
                                 client_request_id=request.headers.get('X-Modloader-Request-Id') if request.url.path == '/api/chat' else None,
                                 client_call_id=request.headers.get('X-Modloader-Call-Id') if request.url.path == '/api/chat' else None)
        if instance.active_requests > 1:
            controller.request_history.mark_concurrent(instance.id)
        report_metrics = request.url.path == '/api/chat'
        if not report_metrics:
            controller.request_history.start(metrics, instance, payload['messages'], client_context={
                'source': 'opencode', 'session_id': request.headers.get('X-Opencode-Session-Id')})
        def observe(chunk):
            metrics.observe(chunk)
            if not report_metrics:
                controller.request_history.observe(metrics.token, chunk)
            controller.publish_generation_event(instance.id, chunk)
        controller.active_request_metrics[metrics.token] = metrics
        # Replace the immutable token set on each lifecycle change, so a poll
        # can detect even a brief concurrent request during its backend await.
        controller.active_request_tokens[instance.id] = controller.active_request_tokens.get(instance.id,frozenset()) | {metrics.token}
        controller.generation_samples.pop(instance.id,None)
        controller.telemetry_samples.pop(instance.id,None)
        controller.publish_generation_event(instance.id, reset=True)
        response = None
        cleaned = False
        def release_token():
            controller.active_request_metrics.pop(metrics.token, None)
            tokens=controller.active_request_tokens.get(instance.id,frozenset())
            if metrics.token in tokens:
                controller.active_request_tokens[instance.id]=tokens-{metrics.token}
        def finish(status):
            release_token()
            previous=controller.generation_samples.get(instance.id)
            if previous and previous['token']==metrics.token:
                controller.generation_samples.pop(instance.id,None)
            if metrics.result is None:
                controller.request_metrics[instance.id] = metrics.finish(status)
                if not report_metrics:
                    controller.request_history.finish(metrics)
                controller.telemetry_samples.pop(instance.id,None)
                controller.publish_generation_event(instance.id, reset=True, finalized=metrics)
        def failure_status():
            return 'cancelled' if instance.state in ('stopping','stopped') else 'failed'
        async def cleanup():
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            release_token()
            # StreamingResponse cancels its task group on client disconnect.
            # Complete resource release even inside that cancellation scope.
            with CancelScope(shield=True):
                try:
                    if response:
                        controller.runtime.requests[instance.id].discard(response)
                        await response.aclose()
                finally:
                    try:
                        await client.aclose()
                    finally:
                        instance.active_requests = max(0, instance.active_requests-1)
                        controller.publish_generation_event(instance.id, reset=True)
        try:
            response = await client.send(client.build_request('POST',f'http://127.0.0.1:{instance.port}{spec.chat_path}',json=payload), stream=True)
            metrics.observe_upstream(response.status_code)
            controller.runtime.requests[instance.id].add(response)
            if response.status_code >= 400:
                data = await response.aread()
                try: error = json.loads(data)
                except Exception: error = {'error':{'message':f'后端返回 HTTP {response.status_code}'}}
                observe(error)
                metrics.observe_upstream(error=error)
                finish('failed')
                await cleanup()
                if report_metrics and isinstance(error, dict): error['modloader_request'] = metrics.result
                return JSONResponse(error, status_code=response.status_code)
            if payload.get('stream'):
                async def events():
                    status, done = 'incomplete', False
                    seen_choices, finished_choices = set(), set()
                    try:
                        async for line in response.aiter_lines():
                            if line.startswith('data:'):
                                data = line[5:].lstrip()
                                if data == '[DONE]':
                                    done = True
                                    if status != 'failed': status = 'completed'
                                    # Clients commonly stop reading at [DONE]; that
                                    # disconnect must not undo a completed request.
                                    finish(status)
                                    if report_metrics:
                                        yield 'data: '+json.dumps({'modloader_request': metrics.result}, ensure_ascii=False)+'\n\n'
                                else:
                                    try:
                                        chunk = json.loads(data)
                                        observe(chunk)
                                        if isinstance(chunk,dict):
                                            if chunk.get('error'):
                                                status = 'failed'
                                                metrics.observe_upstream(error=chunk, cause='stream_error')
                                            choices = chunk.get('choices')
                                            if isinstance(choices,list):
                                                for index,choice in enumerate(choices):
                                                    if not isinstance(choice,dict): continue
                                                    key = choice.get('index',index)
                                                    if type(key) is not int: key = index
                                                    seen_choices.add(key)
                                                    if choice.get('finish_reason') is not None:
                                                        finished_choices.add(key)
                                            if 'model' in chunk: chunk['model'] = alias
                                            line = 'data: '+json.dumps(chunk, ensure_ascii=False)
                                    except json.JSONDecodeError:
                                        pass
                            yield line+'\n'
                        expected_choices = payload.get('n',1)
                        finished = bool(finished_choices) and seen_choices <= finished_choices
                        if type(expected_choices) is int:
                            finished = finished and len(finished_choices) >= expected_choices
                        if (done or finished) and status != 'failed':
                            status = 'completed'
                        if report_metrics and not done:
                            finish(status)
                            yield 'data: '+json.dumps({'modloader_request': metrics.result}, ensure_ascii=False)+'\n\n'
                    except (asyncio.CancelledError, GeneratorExit):
                        if status != 'failed':
                            status = 'cancelled'
                        raise
                    except (httpx.HTTPError, httpx.StreamClosed) as error:
                        # A completed request can race the instance closing its
                        # response. Its terminal frame has already been sent.
                        if metrics.result is not None:
                            return
                        upstream_failed = status == 'failed'
                        if not upstream_failed:
                            status = failure_status()
                            metrics.observe_upstream(cause=type(error).__name__)
                        finish(status)
                        if upstream_failed:
                            if report_metrics:
                                yield 'data: '+json.dumps({'modloader_request': metrics.result})+'\n\n'
                        else:
                            yield 'data: '+json.dumps({'error':{'message':'后端连接已结束'}, **({'modloader_request': metrics.result} if report_metrics else {})})+'\n\n'
                    except BaseException:
                        status = 'failed'
                        metrics.observe_upstream(cause='internal_error')
                        raise
                    finally:
                        finish(status)
                        await cleanup()
                return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
            data = json.loads(await response.aread())
            observe(data)
            data['model'] = alias
            if data.get('error'):
                metrics.observe_upstream(error=data, cause='response_error')
            finish('failed' if data.get('error') else 'completed')
            if report_metrics: data['modloader_request'] = metrics.result
            await cleanup()
            return data
        except (httpx.HTTPError, httpx.StreamClosed) as error:
            metrics.observe_upstream(cause=type(error).__name__)
            finish(failure_status())
            await cleanup()
            raise HTTPException(502, '无法连接模型后端，请检查实例状态') from error
        except asyncio.CancelledError:
            finish('cancelled')
            await cleanup()
            raise
        except BaseException:
            metrics.observe_upstream(cause='internal_error')
            finish('failed')
            await cleanup()
            raise

    app.add_api_route('/v1/models', models, methods=['GET'])
    app.add_api_route('/v1/chat/completions', chat, methods=['POST'])
    if public:
        return app
    app.add_api_route('/api/chat', chat, methods=['POST'])

    @app.get('/api/instances/{identifier}/generation-events')
    async def generation_events(identifier: str):
        if identifier not in controller.runtime.instances:
            raise HTTPException(404, '实例不存在')
        async def events():
            current = controller.generation_event_metrics(identifier)
            queue = controller.generation_events.subscribe(identifier, current,
                active=bool(controller.active_request_tokens.get(identifier)), turn=controller.generation_event_basis(current))
            try:
                yield ': connected\n\n'
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ': heartbeat\n\n'
                    else:
                        yield 'event: generation\ndata: ' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n\n'
            finally:
                controller.generation_events.unsubscribe(identifier, queue)
        return StreamingResponse(events(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/api/request-history/runs')
    async def request_history_runs():
        saved, issues = await asyncio.to_thread(controller.request_history.archive.read)
        return controller.request_history.catalog(saved, issues)

    @app.get('/api/request-history/dates')
    async def request_history_dates():
        saved, issues = await asyncio.to_thread(controller.request_history.archive.read)
        internal = await asyncio.to_thread(controller.conversations.request_history)
        return controller.request_history.dates(saved, issues, internal)

    @app.get('/api/request-history/usage')
    async def request_history_usage(run_id: str | None = None, instance_id: str | None = None,
                                    date_from: str | None = None, date_to: str | None = None,
                                    source: str | None = Query(None, pattern='^(external|internal)$'),
                                    conversation_id: str | None = None):
        # Read the complete retained metric inventory, independently of list paging.
        history = controller.request_history
        saved, issues = await asyncio.to_thread(history.archive.read)
        internal = await asyncio.to_thread(controller.conversations.request_history)
        if conversation_id is not None:
            # Reuse the conversation identity validation without loading its text.
            from .conversations import _identifier
            _identifier(conversation_id)
        try:
            return history.usage(saved, issues, run_id=run_id, instance_id=instance_id, date_from=date_from,
                                 date_to=date_to, source=source, conversation_id=conversation_id, internal=internal)
        except KeyError:
            raise HTTPException(404, '运行历史不存在')

    @app.get('/api/request-history')
    async def request_history(run_id: str | None = None, instance_id: str | None = None,
                              turn_id: str | None = None, request_key: str | None = None, offset: int = Query(0, ge=0),
                              limit: int | None = Query(None, ge=1), date_from: str | None = None, date_to: str | None = None,
                              group_by: str | None = Query(None, pattern='^turn$'),
                              external_group_by: str | None = Query(None, pattern='^request$'),
                              source: str | None = Query(None, pattern='^(external|internal)$'),
                              conversation_id: str | None = None, message_id: str | None = None):
        history = controller.request_history
        if date_from is not None:
            validate_date(date_from)
        if date_to is not None:
            validate_date(date_to)
        if date_from and date_to and date_from > date_to:
            raise ValueError('开始日期不得晚于结束日期')
        if message_id is not None and conversation_id is None:
            raise ValueError('查询内置消息需要 conversation_id')
        if conversation_id is not None:
            if source == 'external' or run_id is not None or turn_id is not None or request_key is not None:
                raise ValueError('内置对话与外部请求身份不可混用')
            source, group_by = 'internal', 'turn'
            if message_id is not None:
                date_from = date_to = None
        if source == 'internal' and (run_id is not None or turn_id is not None or request_key is not None):
            raise ValueError('内置对话不使用外部运行或请求身份')
        if source is not None and group_by is None:
            raise ValueError('来源筛选需要按轮查询')
        if run_id and (turn_id or request_key):
            date_from = date_to = None
        dated = date_from is not None or date_to is not None
        grouped = group_by == 'turn' and turn_id is None and request_key is None
        saved, issues = ({}, []) if source == 'internal' or run_id == history.run_id or (run_id is None and not dated and not grouped) else await asyncio.to_thread(
            history.archive.read, run_id, None if grouped else date_from, None if grouped else date_to)
        internal = None
        if grouped and source != 'external' and run_id is None:
            internal = await asyncio.to_thread(controller.conversations.request_history, date_from, date_to,
                                               instance_id, conversation_id, message_id)
        try:
            return history.page(saved, issues, run_id, instance_id, turn_id, offset, limit, request_key,
                                date_from, date_to, group_by, internal, source != 'internal', external_group_by)
        except KeyError:
            raise HTTPException(404, '运行历史不存在')

    @app.post('/api/request-history/retry')
    async def retry_request_history():
        return controller.request_history.archive.retry()

    app.include_router(create_local_resource_router(controller.local_resources))

    @app.post('/api/chat/attachments/text')
    async def read_chat_attachment(request: Request):
        content = await request.body()
        return await asyncio.to_thread(read_chat_document, request.query_params.get('name'), content)

    @app.get('/api/instances/{identifier}/request-metrics/{call_id}')
    async def chat_request_metrics(identifier: str, call_id: str):
        # Cancellation cannot deliver a terminal SSE frame. Read only this
        # exact in-flight / most-recent call; chat messages own retained history.
        for token in controller.active_request_tokens.get(identifier, ()):
            metrics = controller.active_request_metrics.get(token)
            if metrics is not None and metrics.client_call_id == call_id:
                return metrics.result or metrics.snapshot()
        recent = controller.request_metrics.get(identifier)
        if recent and recent.get('client_call_id') == call_id:
            return recent
        raise HTTPException(404, '本次请求的计量记录不可用')

    @app.exception_handler(LocalToolError)
    async def local_tool_error(request, error):
        detail = {'detail': str(error)}
        if error.recovery_paths:
            detail['recovery_paths'] = error.recovery_paths
        return JSONResponse(detail, status_code=error.status_code)

    @app.post('/api/chat-tools/list')
    def list_local_directory(body:dict):
        return controller.chat_tools.list(body.get('directory'), body.get('path', '.'),
                                          body.get('offset', 0), body.get('limit', 100))

    @app.post('/api/chat-tools/read')
    def read_local_text(body:dict):
        return controller.chat_tools.read(body.get('directory'), body.get('path'),
                                          body.get('start_line', 1), body.get('line_count', 80), body.get('start_column', 1))

    @app.post('/api/chat-tools/prepare-write')
    def prepare_local_write(body:dict):
        return controller.chat_tools.prepare_write(body.get('directory'), body.get('path'),
                                                   content=body.get('content'), edits=body.get('edits'))

    @app.post('/api/chat-tools/write')
    def write_local_text(body:dict):
        if set(body) - {'preparation_id', 'approved'}:
            raise LocalToolError('确认写入时不能更改目录、路径或内容')
        return controller.chat_tools.write(body.get('preparation_id'), approved=body.get('approved'))

    @app.delete('/api/chat-tools/prepared/{identifier}')
    def cancel_local_write(identifier:str):
        return controller.chat_tools.cancel(identifier)

    @app.post('/api/web-search')
    async def web_search(body:dict, request:Request):
        async def perform(progress):
            category = body.get('category', 'general')
            if category not in ('general', 'images'):
                raise ValueError('搜索类别必须为 general 或 images')
            capability = 'images' if category == 'images' else 'search'
            return await run_network_route(capability, controller.settings, controller.vault.get, controller.proxy(),
                                             query=body.get('query'), progress=progress)
        return await operation_response(request, perform, cancel_on_disconnect=True)

    @app.post('/api/web-fetch')
    async def web_fetch(body:dict, request:Request):
        async def perform(progress):
            arguments = {'url': body.get('url'), 'start_line': body.get('start_line', 1),
                         'line_count': body.get('line_count', 80), 'progress': progress}
            if 'provider' in body:
                return await run_network_service(body['provider'], 'fetch', controller.settings, controller.vault.get,
                                                 controller.proxy(), **arguments)
            return await run_network_route('fetch', controller.settings, controller.vault.get, controller.proxy(), **arguments)
        return await operation_response(request, perform, cancel_on_disconnect=True)

    @app.post('/api/web-reverse-image')
    async def web_reverse_image(body:dict, request:Request):
        if set(body) - {'image', 'filename'}:
            raise ValueError('反向搜图只接受 image 与 filename，不接受本机路径或图片网址')
        async def perform(progress):
            return await run_network_service('saucenao', 'reverse', controller.settings, controller.vault.get,
                                             controller.proxy(), image=body.get('image'), filename=body.get('filename'), progress=progress)
        return await operation_response(request, perform, cancel_on_disconnect=True)

    @app.post('/api/web-services/test')
    async def test_web_service(body:dict, request:Request):
        async def perform(progress):
            return await run_network_service(body.get('provider'), body.get('capability'), controller.settings,
                                             controller.vault.get, controller.proxy(), query=body.get('query'), url=body.get('url'),
                                             start_line=body.get('start_line', 1), line_count=body.get('line_count', 80), progress=progress,
                                             **({'image': body.get('image'), 'filename': body.get('filename')} if body.get('capability') == 'reverse' else {}))
        return await operation_response(request, perform, cancel_on_disconnect=True)

    @app.get('/api/chat-presets')
    def list_chat_presets():
        return controller.chat_presets.list()

    @app.post('/api/chat-presets')
    def create_chat_preset(body:dict):
        return controller.chat_presets.create(body)

    @app.patch('/api/chat-presets/default')
    def set_default_chat_preset(body:dict):
        return controller.chat_presets.set_default(body)

    @app.patch('/api/chat-presets/{identifier}')
    def patch_chat_preset(identifier:str, body:dict):
        return controller.chat_presets.patch(identifier, body)

    @app.get('/api/conversations')
    def list_conversations():
        return controller.conversations.list()

    @app.post('/api/conversations')
    def create_conversation(body:dict):
        return controller.conversations.create(body)

    @app.get('/api/conversations/{identifier}')
    def get_conversation(identifier:str):
        return controller.conversations.get(identifier)

    @app.patch('/api/conversations/{identifier}')
    def patch_conversation(identifier:str, body:dict):
        return controller.conversations.patch(identifier, body)

    @app.delete('/api/conversations/{identifier}', status_code=204)
    def delete_conversation(identifier:str):
        controller.conversations.delete(identifier)

    @app.get('/api/health')
    async def health(): return {'status':'ok','version':'0.1.0'}

    @app.post('/api/hardware/temperatures/enable')
    @app.post('/api/hardware/cpu-temperature/enable')
    async def enable_cpu_temperature():
        return await controller.enable_cpu_temperature()

    @app.get('/api/state')
    async def state():
        from .backends.catalog import backend_capabilities
        from .parameter_rules import profile_fields, generation_fields
        capabilities = backend_capabilities()
        backends=[{'id':key,'label':adapter.label,**capabilities.get(key, {}),'parameters':[p.model_dump() for p in profile_fields(adapter)],
                   'generation_parameters':[p.model_dump() for p in generation_fields(key)]} for key,adapter in controller.adapters.items()]
        runtimes = controller.store.all('runtimes')
        for runtime in runtimes:
            if runtime['backend'] == 'strata':
                from .strata_prefetch import capability
                runtime['startup_expert_prefetch'] = capability(RuntimeInstallation(**runtime))
            if runtime['backend'] == 'ninfer':
                runtime['generation_parameters'] = [field.model_dump() for field in generation_fields('ninfer', RuntimeInstallation(**runtime))]
        return {'models':controller.store.all('models'), 'profiles':controller.store.all('profiles'), 'runtimes':runtimes, 'instances':controller.runtime.snapshot(), 'downloads':controller.store.jobs('download'), 'installations':controller.store.jobs('install', UNFINISHED_JOBS), 'migrations':controller.store.jobs('model_migration'), 'settings':controller.safe_settings(), 'backends':backends, 'hardware':controller.hardware,
                'generation_parameters':[field.model_dump() for field in generation_fields('')]}

    @app.post('/api/models/import')
    async def import_model(body:dict):
        model = await asyncio.to_thread(inspect_asset, body['path'], body.get('name'), location=body.get('location'))
        async with controller.environment_lock:
            await asyncio.to_thread(check_import_files, model)
            existing = next((m for m in controller.store.all('models') if model_identity(m)==model_identity(model)),None)
            if existing: return existing
            return controller.store.put('models',model.model_dump())

    def check_import_files(model):
        # A migration can finish while inspection is waiting for the write lock.
        location = location_of(model)
        selected = {model.path, *model.files}
        if location.platform == 'wsl2':
            from .platforms import wsl_python
            code = 'import json,sys; from pathlib import Path; print(json.dumps(all(Path(value).exists() for value in json.load(sys.stdin))))'
            exists = json.loads(wsl_python(location.distribution, code, input=json.dumps(list(selected))))
        else:
            exists = all(Path(value).exists() for value in selected)
        if not exists:
            raise ValueError('检查后模型文件已移动或删除，请重新选择模型路径')

    @app.post('/api/models/{identifier}/storage-preview')
    async def model_storage_preview(identifier: str, body: dict | None = None):
        controller.require('models', identifier)
        controller.assert_model_available(identifier)
        return await controller.model_storage.preview(identifier, body) if body else await controller.model_storage.preview(identifier)

    @app.post('/api/models/{identifier}/storage-migrate')
    async def model_storage_migrate(identifier: str, body: dict):
        async with controller.environment_lock:
            model = controller.require('models', identifier)
            controller.assert_model_available(identifier)
            if any(i.model_id == identifier and i.state in ACTIVE_STATES for i in controller.runtime.instances.values()):
                raise HTTPException(409, '请先停止该模型，再迁移文件')
            if type(body.get('delete_source', False)) is not bool or not isinstance(body.get('plan_id'), str):
                raise ValueError('请先检查迁移清单并选择原文件处理方式')
            job = controller.new_job('model_migration', {'model_id': identifier, 'plan_id': body['plan_id'],
                                                        'delete_source': body.get('delete_source', False)})
            job['name'] = model['name'] + ' · 存储迁移'
            controller.store.put('jobs', job)
            return job

    @app.post('/api/models/import-custom')
    async def import_custom_model(body:dict):
        from .assets import inspect_custom_asset
        model = await asyncio.to_thread(inspect_custom_asset, body.get('path'), body.get('name'), location=body.get('location'))
        async with controller.environment_lock:
            await asyncio.to_thread(check_import_files, model)
            existing = next((m for m in controller.store.all('models') if model_identity(m) == model_identity(model)), None)
            if existing:
                return existing
            return controller.store.put('models', model.model_dump())

    @app.post('/api/models/import-folder')
    async def import_folder(body:dict, request:Request):
        from .assets import scan_model_folder
        if not isinstance(body.get('path'),str) or not body['path'].strip():
            raise ValueError('请选择要递归导入的文件夹')
        async def perform(progress):
            cancelled = threading.Event()
            def scan_progress(event):
                if cancelled.is_set():
                    raise asyncio.CancelledError()
                progress(event)
            try:
                scan = await asyncio.to_thread(scan_model_folder,body['path'], location=body.get('location'), progress=scan_progress)
                imported, existing = [], []
                async with controller.environment_lock:
                    registered = {model_identity(model):model for model in controller.store.all('models')}
                    for index, model in enumerate(scan['models']):
                        progress({'stage': 'registering', 'message': f'校验并登记模型 · {model.name}', 'completed': index, 'total': len(scan['models'])})
                        try:
                            await asyncio.to_thread(check_import_files, model)
                        except ValueError as error:
                            scan['errors'].append({'path': model.path, 'error': str(error)})
                            continue
                        key = model_identity(model)
                        if key in registered:
                            existing.append(registered[key])
                        else:
                            record = controller.store.put('models',model.model_dump())
                            registered[key] = record
                            imported.append(record)
                return {'root':body['path'] if (body.get('location') or {}).get('platform') == 'wsl2' else str(Path(body['path']).resolve()),'imported':imported,'existing':existing,'errors':scan['errors'],'skipped':scan['skipped']}
            finally:
                cancelled.set()
        return await operation_response(request, perform)

    def model_files(identifier):
        model = controller.require('models', identifier)
        if location_of(model).platform == 'wsl2' or not model['managed'] or not model.get('source',{}).get('managed_root'):
            raise HTTPException(400,'外部模型只能移除登记，不能由应用删除文件')
        root = Path(model['source']['managed_root']).resolve()
        files = [Path(p).resolve() for p in model['files']]
        if not all(p.is_relative_to(root) for p in files):
            raise HTTPException(400,'文件超出该下载任务的目录')
        return files

    @app.post('/api/models/{identifier}/delete-preview')
    async def delete_preview(identifier:str):
        files=model_files(identifier)
        files += [Path(p) for p in controller.preparation.owned_files(ModelAsset(**controller.require('models',identifier)))]
        return {'files':[str(p) for p in files], 'total_bytes':sum(p.stat().st_size for p in files if p.exists())}

    def delete_model_record(identifier:str, delete_files:bool=False):
        controller.assert_model_available(identifier)
        model = ModelAsset(**controller.require('models',identifier))
        if any(i.model_id==identifier and i.state in ACTIVE_STATES for i in controller.runtime.instances.values()):
            raise HTTPException(409,'请先卸载该模型的运行实例')
        if delete_files:
            files=model_files(identifier)
            files += [Path(path) for path in controller.preparation.owned_files(model)]
            others = [m for m in controller.store.all('models') if m['id'] != identifier]
            other_files = {model_identity({**m, 'path': p}) for m in others for p in m['files']}
            candidates = {model_identity({'path': str(p)}) for p in files}
            from .platforms import windows_to_wsl
            for distribution in {location_of(m).distribution for m in others if location_of(m).platform == 'wsl2'}:
                candidates.update(('wsl2', distribution, windows_to_wsl(str(p), distribution)) for p in files)
            if candidates & other_files:
                raise HTTPException(409,'文件被其他模型登记引用，请先移除其他登记')
            for path in files:
                if path.is_file(): path.unlink()
            controller.preparation.remove_owned(model)
        for profile in controller.store.all('profiles'):
            if profile['model_id']==identifier: controller.store.remove('profiles',profile['id'])
        controller.store.remove('models',identifier)
        return {'ok':True}

    @app.delete('/api/models/{identifier}')
    async def delete_model(identifier:str, delete_files:bool=False):
        async with controller.environment_lock:
            return delete_model_record(identifier, delete_files)

    def save_profile(body, identifier=None, imported=False):
        previous = controller.require('profiles',identifier) if identifier else {}
        if previous and set(body) == {'chat_quick_load'}:
            profile = LoadProfile(**{**previous, **body})
            return controller.store.put('profiles', {**previous, 'chat_quick_load': profile.chat_quick_load})
        data = {**previous,**body,'id':identifier or uuid.uuid4().hex,'revision':previous.get('revision',0)+1}
        if previous.get('parameter_editor_version') == 1:
            data['parameter_editor_version'] = 1
        profile=LoadProfile(**data)
        controller.assert_model_available(profile.model_id)
        if previous and previous['model_id'] != profile.model_id:
            controller.assert_model_available(previous['model_id'])
        model = ModelAsset(**controller.require('models',profile.model_id))
        runtime=controller.require('runtimes',profile.runtime_id)
        if runtime['backend']!=profile.backend or profile.backend not in controller.adapters:
            raise ValueError('所选运行环境与后端不匹配')
        from .parameter_rules import validate_profile
        gpu = profile.parameters.get('gpu_layers')
        source = previous or (controller.require('profiles', body['copy_from']) if body.get('copy_from') else {})
        legacy_gpu = imported or (bool(source) and profile.model_id == source['model_id'] and profile.backend == source['backend'] and gpu == source.get('parameters', {}).get('gpu_layers'))
        validate_profile(model, profile, RuntimeInstallation(**runtime), controller.adapters[profile.backend], legacy_gpu=legacy_gpu)
        if not profile.alias.strip() or any(c.isspace() for c in profile.alias):
            raise ValueError('API 别名不能为空或包含空白')
        if any(p['id']!=profile.id and p['alias']==profile.alias for p in controller.store.all('profiles')):
            raise HTTPException(409,'此 API 别名已被另一套配置使用')
        return controller.store.put('profiles',profile.model_dump())

    @app.post('/api/resources/estimate')
    async def resource_estimate(body: dict, request: Request):
        profile = LoadProfile(**{**body, 'id': 'resource-preview', 'name': '资源预览', 'alias': 'resource-preview'})
        model = ModelAsset(**controller.require('models', profile.model_id))
        runtime = RuntimeInstallation(**controller.require('runtimes', profile.runtime_id))
        if runtime.backend != profile.backend or profile.backend not in controller.adapters:
            raise ValueError('所选运行环境与后端不匹配')
        from .parameter_rules import validate_profile
        async def estimate(progress):
            progress({'stage': 'profile_validation', 'message': '检查估算参数与模型信息'})
            await asyncio.to_thread(validate_profile, model, profile, runtime, controller.adapters[profile.backend], legacy_gpu=True)
            effective = profile
            preparation_note = None
            if profile.backend == 'strata' or profile.backend == 'ninfer' and not model.path.lower().endswith('.ninfer'):
                try:
                    progress({'stage': 'preparation', 'message': '检查已有准备文件'})
                    # Resolve existing files only; ensure() is the separate preparation/load path.
                    effective = await asyncio.to_thread(controller.preparation.resolved_profile, model, profile, runtime)
                except ValueError as error:
                    if profile.backend == 'strata' and any(profile.parameters.get(key) for key in ('pack', 'configuration', 'strata_config')):
                        # An explicitly selected invalid pack can belong to a stale model.
                        # Do not estimate its weights as if launch would accept them.
                        return {'resources': [], 'source': 'strata-metadata', 'notes': ['Strata 准备资源不可用'], 'diagnostic': str(error)}
                    preparation_note = '准备资源不可用：' + str(error)
            result = await controller.resource_estimator.estimate(model, effective, runtime, progress=progress)
            if preparation_note:
                result['notes'] = [*result.get('notes', []), preparation_note]
                result['diagnostic'] = '\n'.join(filter(None, [result.get('diagnostic'), preparation_note]))
            return result
        return await operation_response(request, estimate, cancel_on_disconnect=True)

    @app.post('/api/profiles/launch-parameters')
    async def launch_parameters(body: dict):
        from .launch_parameters import edit_parameters, format_argv
        profile = LoadProfile(**{**body, 'id': body.get('id') or 'parameter-editor',
                                 'name': body.get('name') or '启动参数', 'alias': body.get('alias') or 'parameter-editor'})
        model = ModelAsset(**controller.require('models', profile.model_id))
        runtime = RuntimeInstallation(**controller.require('runtimes', profile.runtime_id))
        result = await asyncio.to_thread(edit_parameters, model, profile, runtime, text=body.get('text'), previous_text=body.get('previous_text'))
        if body.get('preview') and not result['errors']:
            effective = profile.model_copy(update={key: result[key] for key in ('parameters', 'generation', 'extra_args', 'parameter_editor_version')})
            if 'text' not in body:
                effective.parameter_editor_version = profile.parameter_editor_version
            try:
                preview = await controller.runtime.preview(model, effective, runtime)
                result.update(preview)
                if preview.get('argv'):
                    config = preview.get('config')
                    result['text'] = format_argv([config['exe'], *config.get('args', [])] if config else preview['argv'])
            except (ValueError, OSError) as error:
                result['preview_error'] = str(error)
        return result

    @app.post('/api/profiles/preflight')
    async def profile_preflight(body:dict, request:Request):
        profile = LoadProfile(**{**body, 'id': body.get('id') or 'preflight', 'name': body.get('name') or '配置预检', 'alias': body.get('alias') or 'preflight'})
        model = ModelAsset(**controller.require('models', profile.model_id))
        runtime = RuntimeInstallation(**controller.require('runtimes', profile.runtime_id))
        if runtime.backend != profile.backend or profile.backend not in controller.adapters:
            raise ValueError('所选运行环境与后端不匹配')
        async def check(progress):
            progress({'stage': 'preparation', 'message': '检查运行环境与准备文件'})
            validate_model_runtime(model, runtime)
            if profile.backend in {'strata', 'ninfer', 'sparkinfer'}:
                status = await asyncio.to_thread(controller.preparation.preflight, model, profile, runtime)
            else:
                status = {'status':'ready','source':None,'pack':None,'tokenizer':None,'candidates':[],'errors':[],'message':''}
            if profile.backend == 'strata':
                from .strata_prefetch import profile_status
                try:
                    prefetch = await asyncio.to_thread(profile_status, model, profile, runtime)
                    status['startup_expert_prefetch'] = prefetch
                    if profile.parameters.get('startup_expert_prefetch') is True and prefetch['status'] != 'ready':
                        status.update(status='error', errors=[*status.get('errors', []), prefetch['message']], message=prefetch['message'])
                except ValueError as error:
                    status.update(status='error', errors=[*status.get('errors', []), str(error)], message=str(error))
            if profile.backend == 'custom':
                from .parameter_rules import validate_profile
                try:
                    await asyncio.to_thread(validate_profile, model, profile, runtime, controller.adapters[profile.backend])
                    status['compatibility'] = (await asyncio.to_thread(controller.adapters[profile.backend].inspect, model, runtime)).model_dump()
                except ValueError as error:
                    status.update(status='error', errors=[str(error)], message=str(error))
            if profile.backend == 'sglang':
                from .backends.sglang import acceleration, paged_experts, validate
                devices = controller.hardware.get('gpus', [])
                status['paged_experts'] = await asyncio.to_thread(paged_experts, model, profile, runtime,
                                                                  devices=devices)
                try:
                    status['acceleration'] = await asyncio.to_thread(acceleration, model, profile, runtime)
                    await asyncio.to_thread(validate, model, profile, runtime, devices=devices)
                except ValueError as error:
                    status.update(status='error', errors=[str(error)], message=str(error))
            if profile.backend == 'ninfer':
                from .backends.ninfer_capabilities import capacity_migration_message, ninfer_capacity_mode
                message = capacity_migration_message(profile, runtime)
                if message:
                    field = 'kv_capacity' if ninfer_capacity_mode(runtime) == 'context_bytes' else 'context_capacity'
                    status['parameter_migration'] = {'parameter': field, 'message': message}
            # Only projector-based engines derive vision state from mmproj.
            # Native engines own the state reported by their artifact preflight.
            if profile.backend in {'llama_cpp', 'strata'}:
                progress({'stage': 'vision', 'message': '检查视觉配套文件'})
                status['mmproj_candidates'] = await asyncio.to_thread(projector_candidates, model)
                inherited_vision = {}
                if profile.backend == 'strata':
                    from .backends.adapters import _strata_config
                    try:
                        source = Path(profile.parameters.get('configuration') or (Path(runtime.path) / 'strata.json'))
                        raw = json.loads(source.read_text('utf-8-sig')) if profile.parameters.get('configuration') else profile.parameters.get('strata_config', {})
                        inherited_vision = _strata_config(raw, source).get('vision') or {}
                    except (OSError, ValueError, TypeError):
                        pass  # Resource preflight already reports an invalid imported configuration.
                status['mmproj'] = profile.parameters.get('mmproj') or inherited_vision.get('mmproj')
                status['vision_enabled'] = profile.parameters.get('vision_enabled', bool(status['mmproj'])) is not False
                if status['vision_enabled'] and status['mmproj'] and not Path(status['mmproj']).is_file():
                    status['mmproj_error'] = '视觉投影器文件不存在'
            if profile.backend not in {'sglang', 'ninfer', 'sparkinfer'}:
                from .mtp import inspect_mtp
                progress({'stage': 'mtp', 'message': '检查 MTP 支持与配套文件'})
                from .strata_deepseek import is_deepseek
                if profile.backend == 'strata' and is_deepseek(model):
                    status['strata_family'] = 'deepseek41'
                    status['vision_enabled'] = False
                    status['mmproj_candidates'] = []
                else:
                    status['mtp'] = await asyncio.to_thread(inspect_mtp, model, profile, runtime, controller.store.all('profiles'))
            progress({'stage': 'profile_fields', 'message': '读取参数范围'})
            from .parameter_rules import profile_fields, generation_fields
            status['parameters'] = [field.model_dump() for field in profile_fields(controller.adapters[profile.backend], model, runtime)]
            status['generation_parameters'] = [field.model_dump() for field in generation_fields(profile.backend, runtime)]
            return status
        return await operation_response(request, check)

    @app.post('/api/profiles')
    async def create_profile(body:dict):
        async with controller.environment_lock:
            return await asyncio.to_thread(save_profile, body)
    @app.patch('/api/profiles/{identifier}')
    async def edit_profile(identifier:str,body:dict):
        async with controller.environment_lock:
            return await asyncio.to_thread(save_profile, body, identifier)
    @app.delete('/api/profiles/{identifier}')
    async def delete_profile(identifier:str):
        async with controller.environment_lock:
            controller.assert_model_available(controller.require('profiles', identifier)['model_id'])
            if any(i.profile_id==identifier and i.state in ACTIVE_STATES for i in controller.runtime.instances.values()): raise HTTPException(409,'请先卸载该配置的实例')
            controller.store.remove('profiles',identifier)
            return {'ok':True}

    def profile_args(identifier):
        profile=LoadProfile(**controller.require('profiles',identifier))
        return ModelAsset(**controller.require('models',profile.model_id)),profile,RuntimeInstallation(**controller.require('runtimes',profile.runtime_id))

    @app.post('/api/profiles/{identifier}/preview')
    async def preview(identifier:str): return await controller.runtime.preview(*profile_args(identifier))
    @app.post('/api/profiles/{identifier}/start')
    async def start(identifier:str, request: Request):
        body = await request.json() if await request.body() else {}
        if not isinstance(body, dict) or type(body.get('allow_parallel', False)) is not bool:
            raise HTTPException(400, 'allow_parallel 必须为布尔值')
        async def launch(progress):
            async with controller.environment_lock:
                progress({'stage': 'startup', 'message': '检查启动条件'})
                controller.assert_model_available(controller.require('profiles', identifier)['model_id'])
                return (await controller.runtime.start(*profile_args(identifier), **({'allow_parallel': True} if body.get('allow_parallel') else {}))).model_dump()
        return await operation_response(request, launch)
    @app.post('/api/instances/{identifier}/stop')
    async def stop(identifier:str,body:dict):
        if identifier not in controller.runtime.instances: raise HTTPException(404,'实例不存在')
        try: return (await controller.runtime.stop(identifier,body.get('force',False))).model_dump()
        except RuntimeError as error: raise HTTPException(409,str(error))
    @app.get('/api/instances/{identifier}/logs')
    async def logs(identifier:str): return {'lines':list(controller.runtime.logs.get(identifier,[]))}

    @app.get('/api/instances/{identifier}/telemetry')
    async def telemetry(identifier:str):
        from .backends import collect_telemetry
        instance=controller.runtime.instances.get(identifier)
        if not instance: raise HTTPException(404,'实例不存在')
        recent=controller.request_metrics.get(identifier)
        recent_groups=[request_metrics_group(recent)] if recent else []
        if instance.state!='ready':
            controller.telemetry_samples.pop(identifier,None)
            controller.generation_samples.pop(identifier,None)
            return {'instance_id':identifier,'backend':instance.backend,'state':instance.state,'updated_at':time.time(),'groups':recent_groups,'unavailable':['模型尚未就绪，监控数据暂不可用'],'live_generation':None,'external_history':controller.request_history.latest(identifier)}
        # Share close-in-time Chat/Monitor polls. A .5s TTL tolerates the jitter
        # of 1s frontend timers without silently stretching refreshes to 2s.
        lock=controller.telemetry_locks.setdefault(identifier,asyncio.Lock())
        async with lock:
            now=time.monotonic()
            sample=controller.telemetry_samples.get(identifier)
            if sample is None or now-sample[0]>=.5:
                tokens_before=controller.active_request_tokens.get(identifier,frozenset())
                single_before=instance.active_requests==1
                data=await collect_telemetry(instance,controller.runtime.specs[identifier])
                live=data.get('live_generation')
                tokens=controller.active_request_tokens.get(identifier,frozenset())
                token=next(iter(tokens)) if tokens is tokens_before and single_before and len(tokens)==1 and instance.active_requests==1 else None
                current=controller.active_request_metrics.get(token)
                if current is not None:
                    live=current.live_generation(live)
                if live:
                    previous,live=generation_window(controller.generation_samples.get(identifier),token,live,time.monotonic())
                    if previous is None: controller.generation_samples.pop(identifier,None)
                    else: controller.generation_samples[identifier]=previous
                    data={**data,'live_generation':live}
                elif identifier in controller.generation_samples:
                    # Break the current-rate window on a failed/missing sample,
                    # while retaining the independent request-mean origin.
                    previous=controller.generation_samples[identifier]
                    prefill=previous.get('prefill')
                    controller.generation_samples[identifier]={**previous,'samples':[],
                        'prefill':{**prefill,'samples':[]} if prefill else None}
                sample=(now,time.time(),data)
                controller.telemetry_samples[identifier]=sample
        result={**sample[2], 'groups':list(sample[2].get('groups',[])), 'unavailable':list(sample[2].get('unavailable',[]))}
        live=result.get('live_generation')
        tokens=controller.active_request_tokens.get(identifier,set())
        current=controller.active_request_metrics.get(next(iter(tokens))) if instance.active_requests==1 and len(tokens)==1 else None
        if current is not None and (not live or live.get('request_key') != current.token):
            # A lifecycle change invalidates backend ownership, not the gateway's
            # identity. Return a fresh request-only snapshot without a rate window.
            _, request_live=generation_window(None,current.token,current.live_generation(),time.monotonic())
            if live:
                request_live.update({key:live.get(key) for key in ('context_capacity','context_capacity_basis')})
            live=request_live
        if live:
            live={**live}
            tokens=controller.active_request_tokens.get(identifier,set())
            previous=controller.generation_samples.get(identifier)
            if instance.active_requests!=1 or len(tokens)!=1:
                controller.generation_samples.pop(identifier,None)
                live.update(tokens_per_second=None,window_s=None,available_for_chat=False,reason='统一 API 当前没有唯一的在途请求，模型速度无法归属此对话')
            elif live.get('scope') != 'request' and (live.get('backend_running')!=1 or live.get('backend_queued')!=0):
                controller.generation_samples.pop(identifier,None)
                live.update(tokens_per_second=None,window_s=None,available_for_chat=False,reason='后端存在并发、排队或缺少请求数，模型速度无法归属此对话')
            elif (live.get('available_for_chat') or live.get('prefill_tokens_per_second') is not None) and (previous is None or previous['token'] not in tokens):
                live.update(tokens_per_second=None,window_s=None,prefill_tokens_per_second=None,prefill_window_s=None,available_for_chat=False,reason='等待同一请求的两次连续 token 采样')
            if instance.active_requests != 1 or len(tokens) != 1 or live.get('request_key') not in tokens or (live.get('scope') != 'request' and (live.get('backend_running') != 1 or live.get('backend_queued') != 0)):
                live.update({key: None for key in SNAPSHOT_FIELDS})
            current = controller.active_request_metrics.get(live.get('request_key'))
            # Correlate only the validated live request, never a cached/finished
            # request or an ambiguous model-wide sample after a chat switch.
            live['client_request_id'] = current.client_request_id if current is not None else None
            live['client_call_id'] = current.client_call_id if current is not None else None
            if current is not None:
                live.update(phase=current.phase, first_output_s=current.first_output_s, request_elapsed_s=max(0, time.monotonic()-current.started))
                if current.phase in DECODE_PHASES:
                    live.update(prefill_tokens_per_second=None, prefill_window_s=None)
            result['live_generation']=live
            if current is not None:
                controller.request_history.capture_live(current.token, live)
        recent=controller.request_metrics.get(identifier)
        result['groups']=([request_metrics_group(recent)] if recent else [])+([generation_metrics_group(live)] if live else [])+result.get('groups',[])
        rss=None
        if instance.pid:
            try:
                process=psutil.Process(instance.pid)
                rss=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running())
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        # Windows process RSS doesn't describe Linux guest RAM or GPU allocations.
        if instance.platform == 'wsl2':
            runtime_metrics=[{'id':'platform','label':'运行平台','value':instance.platform,'kind':'text'},
                             {'id':'distribution','label':'WSL 发行版','value':instance.distribution,'kind':'text'},
                             {'id':'pid','label':'宿主启动进程 PID','value':instance.pid,'kind':'text'},
                             {'id':'guest_pid','label':'Linux 服务进程 PID','value':instance.guest_pid,'kind':'text'}]
        else:
            runtime_metrics=[{'id':'pid','label':'托管进程 PID','value':instance.pid,'kind':'text'}]
        runtime_metrics.append({'id':'requests','label':'通过统一 API 的在途请求','value':instance.active_requests,'unit':'个','kind':'gauge'})
        if instance.platform != 'wsl2' and instance.backend != 'vllm':
            runtime_metrics.append({'id':'rss','label':'进程树驻留内存','value':rss,'unit':'bytes','kind':'gauge','description':'操作系统 RSS，包含程序开销；不同进程的共享页可能重复计数'})
        else:
            result['unavailable'].append('未采集 Linux guest 进程内存')
        result.setdefault('groups',[]).append({'id':'process','title':'实例进程','metrics':runtime_metrics})
        return {'instance_id':identifier,'backend':instance.backend,'state':instance.state,'updated_at':sample[1],**result,'external_history':controller.request_history.latest(identifier)}

    @app.get('/api/runtimes/detect')
    async def detect(request:Request,path:str|None=None):
        async def scan(progress):
            cancel = threading.Event()
            try:
                return await asyncio.to_thread(discover_runtimes, controller.paths, controller.store.all('runtimes'), path, cancel=cancel, progress=progress)
            finally:
                cancel.set()
        return await operation_response(request, scan, cancel_on_disconnect=True)
    @app.post('/api/runtimes/import')
    async def import_runtime(body:dict, request:Request):
        async def perform(progress):
            from .backends import import_runtime
            progress({'stage': 'runtime_import', 'message': '检查运行入口与版本'})
            runtime=await asyncio.to_thread(import_runtime,body['backend'],body['path'],body.get('metadata'), platform=body.get('platform'))
            if runtime.backend == 'custom' and 'name' in body:
                from .backends.custom import _text
                runtime.name = _text(body['name'], '引擎名称').strip()
            progress({'stage': 'runtime_import', 'message': '登记运行环境'})
            async with controller.environment_lock:
                previous = controller.store.get('runtimes', runtime.id)
                if previous is None and runtime.platform == 'wsl2' and runtime.backend != 'custom':
                    from .platforms import runtime_distribution
                    previous = next((row for row in controller.store.all('runtimes')
                                     if row['backend'] == runtime.backend and row['path'] == runtime.path
                                     and RuntimeInstallation(**row).platform == 'wsl2'
                                     and runtime_distribution(RuntimeInstallation(**row)) == runtime_distribution(runtime)), None)
                    if previous:
                        runtime.id = previous['id']
                if previous:
                    # Re-importing the same path does not transfer or discard ownership.
                    controller.assert_runtime_available(previous['id'])
                    runtime.managed = previous.get('managed', False)
                    ownership = ('managed_root','install_root','distribution_owned','linux_root','distribution')
                    runtime.metadata = {**previous.get('metadata', {}), **runtime.metadata,
                                        **{key:previous['metadata'][key] for key in ownership if key in previous.get('metadata', {})}}
                return controller.store.put('runtimes',runtime.model_dump())
        return await operation_response(request, perform)
    from .runtime_management import register_runtime_routes
    register_runtime_routes(app, controller)
    from .platforms import register_platform_routes
    register_platform_routes(app, controller.paths)

    @app.post('/api/config/import')
    async def import_config(body:dict):
        from .backends import import_configuration,import_runtime
        config=await asyncio.to_thread(import_configuration,body['path'])
        model=await asyncio.to_thread(inspect_asset,config['model_path'],config.get('name'))
        async with controller.environment_lock:
            await asyncio.to_thread(check_import_files, model)
            existing=next((m for m in controller.store.all('models') if m['path']==model.path),None)
            if existing: model=ModelAsset(**existing)
            controller.store.put('models',model.model_dump())
        runtime_path=config.get('runtime_path')
        if runtime_path:
            runtime=await asyncio.to_thread(import_runtime,config['backend'],runtime_path,config.get('runtime_metadata'))
        else:
            candidates=await asyncio.to_thread(controller.adapters[config['backend']].detect)
            if not candidates: raise ValueError('请先导入该配置对应的后端运行环境')
            runtime=candidates[0]
        async with controller.environment_lock:
            previous = controller.store.get('runtimes', runtime.id)
            if previous is None and runtime.platform == 'wsl2' and runtime.backend != 'custom':
                from .platforms import runtime_distribution
                previous = next((row for row in controller.store.all('runtimes')
                                 if row['backend'] == runtime.backend and row['path'] == runtime.path
                                 and RuntimeInstallation(**row).platform == 'wsl2'
                                 and runtime_distribution(RuntimeInstallation(**row)) == runtime_distribution(runtime)), None)
                if previous:
                    runtime.id = previous['id']
            if previous:
                runtime = RuntimeInstallation(**previous)
            controller.store.put('runtimes',runtime.model_dump())
            alias=re.sub(r'[^\w.-]','-',config.get('alias') or model.name).strip('-') or 'local-model'
            used={p['alias'] for p in controller.store.all('profiles')}
            if alias in used: alias+='-'+uuid.uuid4().hex[:6]
            profile=await asyncio.to_thread(save_profile, {'model_id':model.id,'name':config.get('name') or '导入配置','backend':config['backend'],'runtime_id':runtime.id,'alias':alias,'parameters':config.get('parameters',{}),'extra_args':config.get('extra_args',[])}, imported=True)
        return {'model':model.model_dump(),'profile':profile,'runtime':runtime.model_dump()}

    @app.get('/api/sources/{source}/search')
    async def search(source:str, q:str, request:Request, cursor:str|None=None):
        async def perform(progress):
            progress({'stage': 'repository_search', 'message': '搜索模型仓库'})
            return await controller.sources.search(source, q, cursor)
        return await operation_response(request, perform)
    @app.post('/api/sources/resolve')
    async def resolve(body:dict, request:Request):
        async def perform(progress):
            result = await controller.sources.resolve(body['source'], body['repo_id'], body.get('revision', 'main'), progress=progress)
            return result.model_dump()
        return await operation_response(request, perform)
    @app.post('/api/sources/readme')
    async def readme(body:dict, request:Request):
        async def perform(progress):
            return await controller.sources.readme(body['source'], body['repo_id'], body.get('revision', 'main'), body.get('file_path'), progress=progress)
        return await operation_response(request, perform)
    @app.post('/api/downloads/plan')
    async def plan(body:DownloadRequest, request:Request):
        async def perform(progress):
            return await controller.sources.plan(body, progress=progress)
        return await operation_response(request, perform)
    @app.post('/api/downloads')
    async def download(body:DownloadRequest):
        directory=Path(controller.settings['model_directory']).expanduser().resolve()
        directory.mkdir(parents=True,exist_ok=True)
        slug=re.sub(r'[^\w.-]','-',body.repo_id or body.filename or 'download')
        body.destination=str(directory/(slug+'-'+uuid.uuid4().hex[:8]))
        return controller.new_job('download',body.model_dump())
    @app.post('/api/downloads/{identifier}/retry')
    async def retry(identifier:str):
        job=controller.require('jobs',identifier)
        if job['state'] in UNFINISHED_JOBS: raise HTTPException(409,'任务尚未结束')
        if job['kind'] == 'install':
            raise HTTPException(409, '请在运行环境中选择版本并重新开始安装')
        request=dict(job['request'])
        original_url=controller.vault.get('download_url_'+identifier)
        if original_url: request['url']=original_url
        return controller.new_job(job['kind'],request)
    @app.post('/api/jobs/{identifier}/pause')
    async def pause(identifier:str):
        job=controller.job_records.get(identifier) or controller.require('jobs',identifier)
        control=controller.download_pauses.get(identifier)
        task=controller.job_tasks.get(identifier)
        if not control or not task or task.done() or job['state']!='running' or not job.get('can_pause'):
            raise HTTPException(409,'当前阶段不能暂停；安装阶段可停止')
        control.request()
        job.update(state='pausing')
        controller.store.put('jobs',job)
        return job
    @app.post('/api/jobs/{identifier}/resume')
    async def resume(identifier:str):
        job=controller.job_records.get(identifier) or controller.require('jobs',identifier)
        control=controller.download_pauses.get(identifier)
        task=controller.job_tasks.get(identifier)
        if not control or not task or task.done() or job['state'] not in ('pausing','paused'):
            raise HTTPException(409,'任务当前未暂停')
        if job['kind'] == 'download':
            controller.model_storage.assert_download_available(job['request'].get('destination'))
        control.resume()
        job.update(state='running')
        controller.store.put('jobs',job)
        return job
    @app.post('/api/jobs/{identifier}/cancel')
    async def cancel(identifier:str):
        job=controller.job_records.get(identifier) or controller.require('jobs',identifier)
        event=controller.cancel_events.get(identifier)
        if event: event.set()
        control=controller.download_pauses.get(identifier)
        if control: control.resume()
        task=controller.job_tasks.get(identifier)
        if task and not task.done():
            job.update(state='cancelling',can_pause=False)
            controller.store.put('jobs',job)
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            if job['state']=='cancelling':
                job.update(state='cancelled',message='迁移已停止，可重新检查后重试' if job['kind']=='model_migration' else '已停止；可重试重新安装' if job['kind']=='install' else '已停止；可重试下载')
                controller.store.put('jobs',job)
        return controller.require('jobs',identifier)

    @app.patch('/api/settings')
    async def settings(body:dict):
        allowed={'api_history_enabled','api_history_retention_days','api_history_directory','chat_unload_before_load','chat_tool_details','api_port','close_to_tray','font_scale','proxy_mode','proxy_url','model_directory','searxng_url','searxng_image_engine', *DEFAULT_NETWORK_SETTINGS}
        update={k:v for k,v in body.items() if k in allowed}
        if 'api_history_enabled' in update and type(update['api_history_enabled']) is not bool:
            raise ValueError('历史保存开关必须为布尔值')
        if 'api_history_retention_days' in update and (type(update['api_history_retention_days']) is not int or update['api_history_retention_days'] not in (0, 7, 30)):
            raise ValueError('历史保留期必须为 0、7 或 30 天')
        if 'api_history_directory' in update:
            directory = update['api_history_directory']
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError('历史保存目录不能为空')
            update['api_history_directory'] = os.path.abspath(Path(directory).expanduser())
        if 'searxng_url' in update:
            validate_http_url(update['searxng_url'], 'SearXNG 地址')
        if 'searxng_image_engine' in update:
            validate_image_engine(update['searxng_image_engine'])
        if 'openai_search_model' in update:
            validate_openai_search_model(update['openai_search_model'])
        if 'openai_image_model' in update:
            validate_openai_image_model(update['openai_image_model'])
        if 'openai_read_model' in update:
            validate_openai_read_model(update['openai_read_model'])
        for field, capability in PROVIDER_SETTINGS.items():
            if field in update:
                require_service(update[field], capability)
        for field, capability in FALLBACK_SETTINGS.items():
            if field in update:
                validate_fallbacks(update[field], capability)
        for service in WEB_SERVICES:
            if service['credential_key'] and service['url_setting'] in update:
                validate_service_url(update[service['url_setting']], service['label'])
        secrets = body.get('secrets', {})
        if not isinstance(secrets, dict):
            raise ValueError('secrets 必须为凭据对象')
        for key in WEB_CREDENTIAL_KEYS:
            if key in secrets:
                validate_service_secret(secrets[key])
        if 'api_port' in update and (type(update['api_port']) is not int or not 1<=update['api_port']<=65535): raise ValueError('端口必须为 1–65535 的整数')
        if 'chat_unload_before_load' in update and type(update['chat_unload_before_load']) is not bool:
            raise ValueError('快捷加载卸载策略必须为布尔值')
        if 'chat_tool_details' in update and update['chat_tool_details'] not in ('compact', 'full'):
            raise ValueError('工具详情必须为 compact 或 full')
        if 'font_scale' in update:
            import math
            if not isinstance(update['font_scale'],(int,float)) or not math.isfinite(update['font_scale']) or update['font_scale']<=0: raise ValueError('文字比例必须是大于零的有限数字')
        if update.get('proxy_mode',controller.settings['proxy_mode']) not in ('system','direct','manual'): raise ValueError('未知代理模式')
        if update.get('proxy_url'):
            parsed=urlparse(update['proxy_url'])
            if parsed.scheme not in ('http','https') or not parsed.hostname: raise ValueError('代理地址需要有效的 HTTP(S) URL')
            if parsed.username or parsed.password: raise ValueError('请使用不含凭据的本地代理地址')
        if 'model_directory' in update:
            path=Path(update['model_directory']).expanduser().resolve()
            path.mkdir(parents=True,exist_ok=True)
            update['model_directory']=str(path)
        for key,value in secrets.items():
            if key in ('hf','modelscope', *WEB_CREDENTIAL_KEYS): controller.vault.set(key,value)
        if 'api_key' in body:
            if not body['api_key']: raise ValueError('API Key 不能为空')
            controller.vault.set('inference_api_key',body['api_key'])
            controller.api_key=body['api_key']
        changed_port=update.get('api_port',controller.settings['api_port'])!=controller.settings['api_port']
        controller.settings.update(update)
        controller.store.put('settings',controller.settings)
        if 'api_history_enabled' in update or 'api_history_directory' in update or 'api_history_retention_days' in update:
            controller.request_history.configure(controller.settings['api_history_enabled'], controller.settings['api_history_directory'], controller.settings['api_history_retention_days'])
        if changed_port: controller.api_error='API 端口修改将在重启应用后生效'
        return controller.safe_settings()
    @app.get('/api/settings/api-key')
    async def api_key(): return {'key':controller.api_key}
    return app
