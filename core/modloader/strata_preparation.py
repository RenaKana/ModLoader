"""Resolve and prepare local Strata resources independently of load settings."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import re
import shutil
import signal
import struct
import uuid
from threading import RLock

import psutil

from .models import LoadProfile, ModelAsset, RuntimeInstallation
from . import strata_deepseek as deepseek


_VALIDATION_LOCK = RLock()


def _file_stamp(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_mode


@lru_cache(maxsize=32)
def _validated_expert_profile(name, layers, experts, stamp):
    path = Path(name)
    with path.open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] != b'STRP':
            raise ValueError(f'Strata expert profile 的 STRP 文件头无效：{path}')
        version, nl, ne, slots, ranked = struct.unpack('<5I', header[4:])
        if version != 1:
            raise ValueError(f'Strata expert profile 版本 {version} 不受支持（需要版本 1）：{path}')
        if (nl, ne) != (layers, experts):
            raise ValueError(f'Strata expert profile 为 {nl}×{ne}，模型需要 {layers}×{experts}：{path}')
        if ranked > slots:
            raise ValueError(f'Strata expert profile 排名数超过槽位数：{path}')
        if not ranked:
            raise ValueError(f'Strata expert profile 排名为空，无法初始化推理验证器：{path}')
        if path.stat().st_size < 24 + 4 * ranked:
            raise ValueError(f'Strata expert profile 排名列表被截断：{path}')
        for _ in range(ranked):
            pair = stream.read(4)
            if len(pair) != 4:
                raise ValueError(f'Strata expert profile 排名列表被截断：{path}')
            layer, expert = struct.unpack('<HH', pair)
            if layer >= layers or expert >= experts:
                raise ValueError(f'Strata expert profile 排名 ({layer}, {expert}) 超出模型范围：{path}')
    return name


def expert_profile_path(model: ModelAsset, runtime: RuntimeInstallation, explicit=None) -> str:
    """Resolve the runtime's expert ranking, not model weights or a load profile.

    STRP v1 validation follows src/core/expert_cache.cpp::read_expert_profile:
    the engine consumes the header and ranked pairs, not the optional trailing table.
    """
    metadata = model.metadata
    keys = ('qwen4exp.block_count', 'qwen4exp.expert_count')
    if any(metadata.get(key) is None for key in keys):
        from .resource_metadata import read_metadata
        metadata = {**metadata, **read_metadata(model.path)['config']}
    layers, experts = (metadata.get(key) for key in keys)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (layers, experts)):
        raise ValueError('无法读取模型的 qwen4exp 层数和专家数，不能选择 Strata expert profile')

    def validate(path):
        with _VALIDATION_LOCK:
            return _validated_expert_profile(str(path.resolve()), layers, experts, _file_stamp(path))

    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise ValueError(f'Strata expert profile 不存在：{path}')
        return validate(path)
    errors = []
    for name in ('expert-profile.bin', 'expert-profile-coder.bin'):
        path = Path(runtime.path) / 'data' / name
        if path.is_file():
            try:
                return validate(path)
            except ValueError as error:
                errors.append(str(error))
    detail = '；'.join(errors) if errors else f'运行时 data 目录没有标准 expert profile：{runtime.path}'
    raise ValueError(f'缺少匹配 {layers}×{experts} 的 Strata expert profile；{detail}')


def _argument(args, flag):
    value = None
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 == len(args) or str(args[index + 1]).startswith('--'):
                raise ValueError(f'Strata 配置 {flag} 缺少值')
            value = args[index + 1]
        elif isinstance(arg, str) and arg.startswith(flag + '='):
            value = arg.split('=', 1)[1]
            if not value:
                raise ValueError(f'Strata 配置 {flag} 缺少值')
    return value


class StrataPreparation:
    KIND = 'strata_preparations'

    def __init__(self, paths, store, proxy_provider=None):
        self.paths, self.store = paths, store
        self.proxy_provider = proxy_provider

    @property
    def root(self):
        return self.paths.root / 'prepared' / 'strata'

    @staticmethod
    def _file_identity(path):
        stat = path.stat()
        return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

    def _identity(self, model, runtime, profile=None):
        source = Path(model.path).absolute()
        match = re.fullmatch(r'(.+)-(\d+)-of-(\d+)\.gguf', source.name, re.I)
        shards = [source]
        if match:
            stem, number, count = match.groups()
            shards = [source.with_name(f'{stem}-{i:0{len(number)}d}-of-{count}.gguf')
                      for i in range(1, int(count) + 1)]
        for shard in shards:
            if not shard.is_file():
                raise ValueError(f'模型分片不存在：{shard}')
        is_ds = deepseek.is_deepseek(model)
        tool = Path(runtime.path) / ('tools/dsv41/pack_gguf.py' if is_ds else 'tools/iq_pack.py')
        identity = {'source': [self._file_identity(p) for p in shards],
                'runtime': {'id': runtime.id, 'path': str(Path(runtime.path).resolve()),
                            'version': runtime.version,
                            'tool': self._file_identity(tool) if tool.is_file() else None}}
        if is_ds and profile is not None:
            identity['dependencies'] = {key: self._file_identity(Path(path)) for key, path in deepseek.dependencies(model, profile).items()}
            if profile.parameters.get('expert_profile') and profile.parameters['expert_profile'] != 'none':
                identity['expert_profile'] = self._file_identity(Path(profile.parameters['expert_profile']))
        return identity

    @staticmethod
    def _validate(pack, tokenizer, model=None):
        """Check completion and small metadata, without reading tensor payloads."""
        if model is not None and deepseek.is_deepseek(model):
            deepseek.validate_pack(pack, model)
            StrataPreparation._validate_tokenizer(tokenizer)
            return
        pack, tokenizer = Path(pack).resolve(), Path(tokenizer).resolve()
        files = [pack / name for name in ('dense.bin', 'index.txt', 'conversions.json', 'native_experts.txt')] + [
                tokenizer / name for name in ('vocab.json', 'merges.txt', 'token_type.json', 'tokenizer.json')]
        for path in files:
            if not path.is_file():
                raise ValueError(f'Strata 资源缺少文件：{path}')
        stamps = tuple(_file_stamp(path) for path in files)
        with _VALIDATION_LOCK:
            StrataPreparation._validate_cached(str(pack), str(tokenizer), stamps)

    @staticmethod
    @lru_cache(maxsize=32)
    def _validate_cached(pack, tokenizer, stamps):
        pack, tokenizer = Path(pack), Path(tokenizer)
        for name, numeric_start, minimum in [('native_experts.txt', 0, 8), ('index.txt', 1, 19)]:
            rows = 0
            with (pack / name).open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if not line.strip() or line.lstrip().startswith('#'):
                        continue
                    fields = line.split()
                    if len(fields) < minimum:
                        raise ValueError(f'Strata 资源格式不完整：{pack / name}')
                    try:
                        [int(value) for value in fields[numeric_start:minimum]]
                    except ValueError as error:
                        raise ValueError(f'Strata 资源格式无效：{pack / name}') from error
                    rows += 1
            if not rows:
                raise ValueError(f'Strata 资源没有数据：{pack / name}')
        with (pack / 'conversions.json').open(encoding='utf-8-sig') as stream:
            conversions = json.load(stream)
        if not isinstance(conversions, dict) or not conversions:
            raise ValueError(f'Strata 资源 JSON 无效：{pack / "conversions.json"}')
        StrataPreparation._validate_tokenizer(tokenizer)

    @staticmethod
    def _validate_tokenizer(tokenizer):
        tokenizer = Path(tokenizer)
        if not (tokenizer / 'merges.txt').is_file():
            raise ValueError(f'Strata 资源缺少文件：{tokenizer / "merges.txt"}')
        for path, types in [(tokenizer / 'vocab.json', (dict,)), (tokenizer / 'token_type.json', (dict, list)),
                            (tokenizer / 'tokenizer.json', (dict,))]:
            with path.open(encoding='utf-8-sig') as stream:
                value = json.load(stream)
            if not isinstance(value, types) or not value:
                raise ValueError(f'Strata 资源 JSON 无效：{path}')

    @staticmethod
    def _configured(model, profile, runtime):
        p = profile.parameters
        pack, tokenizer = p.get('pack'), p.get('tokenizer')
        expert_profile = p.get('expert_profile')
        cwd = Path(runtime.path).resolve()
        imported = bool(p.get('configuration') or p.get('strata_config'))
        if imported:
            source = Path(p['configuration']) if p.get('configuration') else Path(runtime.path) / 'strata.json'
            raw = json.loads(source.read_text(encoding='utf-8-sig')) if p.get('configuration') else p['strata_config']
            if not isinstance(raw, dict) or not isinstance(raw.get('args', []), list):
                raise ValueError('Strata 配置格式无效')
            from .backends.adapters import _strata_config
            from .strata_drafting import merge_args
            raw = _strata_config(raw, source)
            cwd = Path(raw['cwd'])
            args = merge_args(raw.get('args', []), raw.get('user_args', []))

            native = _argument(args, '--native')
            if deepseek.is_deepseek(model):
                if raw.get('arch') != 'deepseek41':
                    raise ValueError('此 Strata 配置不是 DeepSeek V4.1 配置')
            elif not native or (cwd / native).resolve() != Path(model.path).resolve():
                raise ValueError('Strata 配置属于另一模型或缺少 --native；请选择本模型的资源')
            configured_pack = _argument(args, '--pack')
            configured_experts = _argument(args, '--expert-profile')
            expert_profile = expert_profile or ('none' if deepseek.is_deepseek(model) and configured_experts == 'none' else str((cwd / configured_experts).resolve()) if configured_experts else None)
            pack = pack or (str((cwd / configured_pack).resolve()) if configured_pack else None)
            tokenizer = tokenizer or (str((cwd / raw['tokenizer']).resolve()) if raw.get('tokenizer') else None)
        pack = str(Path(pack).resolve()) if pack else None
        tokenizer = str(Path(tokenizer).resolve()) if tokenizer else (str(Path(pack) / 'tokenizer') if pack else None)
        extra_experts = _argument(profile.extra_args, '--expert-profile')
        if extra_experts:
            expert_profile = 'none' if deepseek.is_deepseek(model) and extra_experts == 'none' else str((cwd / extra_experts).resolve())
        return pack, tokenizer, imported, expert_profile

    @staticmethod
    def _candidate(pack, tokenizer, source):
        key = json.dumps([os.path.normcase(pack), os.path.normcase(tokenizer)])
        return {'id': hashlib.sha256(key.encode()).hexdigest()[:24], 'label': source['label'],
                'pack': pack, 'tokenizer': tokenizer, 'source': source}

    def _owned_path(self, record):
        expected = self.root / str(record['id'])
        resolved = expected.resolve()
        # Registered output is always an exclusive immediate child, never an imported path.
        if (expected.name != record['id'] or resolved.parent != self.root.resolve()
                or self.root.resolve() != self.root.absolute()
                or not self.root.resolve().is_relative_to(self.paths.root.resolve())
                or expected.is_symlink() or resolved != Path(record.get('pack', '')).resolve()):
            raise ValueError('Strata 准备资源路径越出受管理目录')
        return resolved

    def _check_generated_source(self, pack, model, identity, records):
        if Path(pack).resolve().is_relative_to(self.root.resolve()):
            matching = [r for r in records if Path(r.get('pack', '')).resolve() == Path(pack).resolve()]
            if not any(r.get('model_id') == model.id and r.get('identity') == identity for r in matching):
                raise ValueError('已准备的 Strata 资源与当前模型或运行时不匹配；请重新准备')

    def _draft_root(self, model):
        # UUID-derived segment also supports imported non-UUID model identifiers.
        base = self.paths.root / 'prepared' / 'strata-drafting'
        root = base / uuid.uuid5(uuid.NAMESPACE_URL, model.id).hex
        if base.resolve() != base.absolute() or root.resolve().parent != base.resolve() or root.is_symlink():
            raise ValueError('Strata 草稿准备目录存在链接或越界')
        return root

    def preflight(self, model: ModelAsset, profile: LoadProfile, runtime: RuntimeInstallation) -> dict:
        result = self._base_preflight(model, profile, runtime)
        if deepseek.is_deepseek(model):
            return result
        try:
            from .strata_drafting import inspect_drafting
            drafting = inspect_drafting(model, profile, runtime, self._draft_root(model))
            result['drafting'] = drafting
            if result['status'] == 'ready' and drafting['status'] != 'ready':
                result.update(status=drafting['status'], message=drafting['message'])
        except (OSError, ValueError, TypeError) as error:
            result.update(status='error', errors=[str(error)], message=str(error))
        return result

    def _base_preflight(self, model: ModelAsset, profile: LoadProfile, runtime: RuntimeInstallation) -> dict:
        result = {'status': 'needs_preparation', 'source': None, 'pack': None, 'tokenizer': None, 'expert_profile': None,
                  'candidates': [], 'errors': [], 'message': '首次加载需要在本地准备 Strata 资源'}
        try:
            from .platforms import validate_model_runtime
            validate_model_runtime(model, runtime)
            is_ds = deepseek.is_deepseek(model)
            if is_ds:
                deepseek.engine_path(Path(runtime.path))
            identity = self._identity(model, runtime, profile)
            if is_ds:
                result['dependencies'] = {key: value['path'] for key, value in identity.get('dependencies', {}).items()}
            records = self.store.all(self.KIND)
            pack, tokenizer, imported, explicit_experts = self._configured(model, profile, runtime)
            result['expert_profile'] = explicit_experts if is_ds else expert_profile_path(model, runtime, explicit_experts)
            if pack:
                self._check_generated_source(pack, model, identity, records)
                self._validate(pack, tokenizer, model)
                if is_ds and profile.parameters.get('dspark_mode') == 'on' and not deepseek.pack_description(pack)['dspark']:
                    raise ValueError('此 DeepSeek pack 不含 DSpark；请选择含草稿的 pack 或重新准备')
                result.update(status='ready', pack=pack, tokenizer=tokenizer,
                              source={'kind': 'explicit' if profile.parameters.get('pack') or profile.parameters.get('tokenizer') else 'imported',
                                      'label': profile.name, 'profile_id': profile.id}, message='使用配置指定的 Strata 资源')
                return result
            if imported:
                raise ValueError('已有 Strata 配置缺少 pack；请补全或移除该配置后准备资源')
            if tokenizer:
                self._validate_tokenizer(tokenizer)
            candidates = {}
            for record in records:
                if record.get('model_id') != model.id or record.get('identity') != identity:
                    continue
                try:
                    path = self._owned_path(record)
                    tok = tokenizer or record['tokenizer']
                    self._validate(path, tok, model)
                    item = self._candidate(str(path), tok, {'kind': 'prepared', 'label': '已准备资源'})
                    candidates[item['id']] = item
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            for row in self.store.all('profiles'):
                if row.get('model_id') != model.id or row.get('backend') != 'strata' or row.get('id') == profile.id:
                    continue
                try:
                    other = LoadProfile.model_validate(row)
                    other_pack, other_tokenizer, _, _ = self._configured(model, other, runtime)
                    if not other_pack:
                        continue
                    self._check_generated_source(other_pack, model, identity, records)
                    tok = tokenizer or other_tokenizer
                    self._validate(other_pack, tok, model)
                    item = self._candidate(other_pack, tok, {'kind': 'profile', 'label': other.name, 'profile_id': other.id})
                    candidates.setdefault(item['id'], item)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            result['candidates'] = [{k: v for k, v in c.items() if k != 'source'} for c in candidates.values()]
            selected = profile.parameters.get('strata_resource_id')
            if selected and selected not in candidates:
                raise ValueError('所选 Strata 资源已失效；请重新选择资源')
            if selected or len(candidates) == 1:
                item = candidates[selected] if selected else next(iter(candidates.values()))
                result.update(status='ready', source=item['source'], pack=item['pack'], tokenizer=item['tokenizer'], message='复用本模型已有的 Strata 资源')
            elif len(candidates) > 1:
                result.update(status='choice_required', message='本模型存在多组 Strata 资源，请选择要复用的一组')
            elif not runtime.metadata.get('python') or not identity['runtime']['tool']:
                raise ValueError('Strata 运行时缺少 Python 或 ' + ('tools/dsv41/pack_gguf.py' if is_ds else 'tools/iq_pack.py') + '，无法准备资源')
            if is_ds and result['status'] == 'needs_preparation':
                _, deps, default_experts = deepseek.preparation_inputs(model, profile, runtime)
                result['expert_profile'] = explicit_experts or default_experts
                result['dependencies'] = deps
            if is_ds and result['status'] == 'ready' and profile.parameters.get('dspark_mode') == 'on':
                if not deepseek.pack_description(result['pack'])['dspark']:
                    raise ValueError('此 DeepSeek pack 不含 DSpark；请选择含草稿的 pack 或重新准备')
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status='error', errors=[str(error)], message=str(error))
        return result

    @staticmethod
    def _resolved(profile, pack, tokenizer, expert_profile):
        resolved = profile.model_copy(deep=True)
        resolved.parameters.update(pack=pack, tokenizer=tokenizer, expert_profile=expert_profile)
        return resolved

    def resolved_profile(self, model, profile, runtime) -> LoadProfile:
        result = self.preflight(model, profile, runtime)
        if result['status'] != 'ready':
            raise ValueError(result['message'])
        return self._draft_resolved(self._resolved(profile, result['pack'], result['tokenizer'], result['expert_profile']), result.get('drafting'))

    @staticmethod
    def _draft_resolved(profile, state):
        if state and state['mode'] != 'legacy':
            if state['effective_mode'] == 'dflash2':
                profile.parameters['dflash2_path'] = state['path']
            elif state['effective_mode'] == 'mtp':
                profile.parameters['mtp_path'] = state['path']
        return profile

    async def ensure(self, model, profile, runtime, progress) -> LoadProfile:
        resolved = await self._ensure_base(model, profile, runtime, progress)
        if deepseek.is_deepseek(model):
            return resolved
        from .strata_drafting import inspect_drafting
        state = await asyncio.to_thread(inspect_drafting, model, resolved, runtime, self._draft_root(model))
        if state['status'] != 'ready':
            await self._prepare_drafting(model, resolved, runtime, state, progress)
            state = await asyncio.to_thread(inspect_drafting, model, resolved, runtime, self._draft_root(model))
            if state['status'] != 'ready':
                raise ValueError(state['message'])
        resolved = self._draft_resolved(resolved, state)
        # Auto may reuse an MTP prepared earlier, but never downloads one just
        # for a possible readiness retry. Missing/invalid inactive paths stay idle.
        if state['mode'] == 'auto' and state['effective_mode'] == 'dflash2' and not resolved.parameters.get('mtp_path'):
            candidate = self._draft_root(model) / 'mtp/rt'
            if candidate.is_dir():
                resolved.parameters['mtp_path'] = str(candidate)
        return resolved

    async def _ensure_base(self, model, profile, runtime, progress) -> LoadProfile:
        result = await asyncio.to_thread(self._base_preflight, model, profile, runtime)
        if result['status'] == 'ready':
            return self._resolved(profile, result['pack'], result['tokenizer'], result['expert_profile'])
        if result['status'] != 'needs_preparation':
            raise ValueError(result['message'])
        from .runtime import WindowsJob, scrub, local_process_environment

        identity = self._identity(model, runtime, profile)
        identifier = uuid.uuid4().hex
        record = {'id': identifier, 'model_id': model.id, 'runtime_id': runtime.id,
                  'identity': identity, 'pack': str(self.root / identifier),
                  'tokenizer': str(self.root / identifier / 'tokenizer')}
        output = self._owned_path(record)
        output.mkdir(parents=True, exist_ok=False)
        process = job = spawning = None
        registered = False
        try:
            await progress({'phase': 'preparing', 'message': '正在生成 Strata pack 与 tokenizer'})
            if deepseek.is_deepseek(model):
                sources, deps, expert = deepseek.preparation_inputs(model, profile, runtime)
                argv = [str(runtime.metadata['python']), '-u', str(Path(runtime.path) / 'tools/dsv41/pack_gguf.py'),
                        '--model', str(sources[0].parent), '--shards', *[p.name for p in sources], '--out', str(output)]
                for key, flag in (('engram', '--engram'), ('engram_meta', '--engram-meta'), ('dspark', '--dspark-gguf')):
                    if deps.get(key):
                        argv.extend([flag, deps[key]])
                if deps.get('dspark'):
                    # DeepSeek V4.1's published DSpark geometry (upstream setup.py dsv41_pack_cmd).
                    argv += ['--dspark-noise-token', '128799', '--dspark-markov-rank', '256']
            else:
                argv = [str(runtime.metadata['python']), '-u', str(Path(runtime.path) / 'tools' / 'iq_pack.py'),
                        '--gguf', str(Path(model.path).absolute()), '--out', str(output)]
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                *argv, cwd=runtime.path, env=local_process_environment({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 1024, **({'start_new_session': True} if os.name != 'nt' else {'creationflags': 0x08000000})))
            process = await asyncio.shield(spawning)
            job = WindowsJob(process.pid)
            while raw := await process.stdout.readline():
                line = scrub(raw.decode('utf-8', errors='replace').strip())
                if line:
                    await progress({'phase': 'preparing', 'message': line})
            code = await process.wait()
            if code:
                raise ValueError(f'Strata 资源准备失败，进程退出码 {code}')
            await progress({'phase': 'validating', 'message': '正在检查 Strata pack 与 tokenizer'})
            self._validate(output, record['tokenizer'], model)
            if deepseek.is_deepseek(model):
                allowed = {*sources, *[Path(path).resolve() for path in deps.values()]}
                unexpected = [path for path in deepseek.pack_description(output)['files']
                              if path not in allowed and not path.is_relative_to(output)]
                if unexpected:
                    raise ValueError('打包工具自动引用了未选择的配套文件，请更新 Strata 准备工具或调整配套目录：' + '、'.join(map(str, unexpected)))
                if expert:
                    shutil.copyfile(expert, output / 'expert-profile.bin')
            if identity != self._identity(model, runtime, profile):
                raise ValueError('模型或 Strata 运行时在准备期间发生变化；本次资源未保存')
            tokenizer = profile.parameters.get('tokenizer') or record['tokenizer']
            if tokenizer != record['tokenizer']:
                self._validate(output, tokenizer, model)
            await progress({'phase': 'prepared', 'message': 'Strata 资源准备完成'})
            self.store.put(self.KIND, record)
            registered = True
            return self._resolved(profile, str(output), str(Path(tokenizer).resolve()), result['expert_profile'])
        finally:
            if spawning is not None and process is None:
                try:
                    process = await spawning
                except (Exception, asyncio.CancelledError):
                    pass
            if job:
                job.close()
            if process:
                if os.name != 'nt':
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.returncode is None and not job:
                    try:
                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass
                if process.returncode is None:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
            if not registered:
                shutil.rmtree(self._owned_path(record))

    async def _prepare_drafting(self, model, profile, runtime, state, progress):
        from .runtime import local_process_environment, scrub, WindowsJob
        root = self._draft_root(model)
        root.mkdir(parents=True, exist_ok=True)
        python = str(runtime.metadata['python'])
        # urllib's empty environment otherwise falls back to Windows registry
        # proxies. Set its policy before official helpers construct their own
        # openers, so direct/manual routes match the application setting.
        route = ('import os, urllib.request\n'
                 'proxy = os.environ.get("HTTPS_PROXY", "")\n'
                 'urllib.request.getproxies = lambda: {"http": proxy, "https": proxy} if proxy else {}\n'
                 'urllib.request.proxy_bypass = lambda host: False\n')
        if state['effective_mode'] == 'dflash2':
            script = (route + 'from strata_control import drafting_fetch\n'
                      'import sys\n'
                      'drafting_fetch.ensure(sys.argv[1], progress=lambda text, *n: print(text, flush=True))\n'
                      'drafting_fetch.convert_selector(sys.argv[1])\n')
            commands = [[python, '-u', '-c', script, str(root)]]
            required = ['strata_control/drafting_fetch.py', 'tools/dflash2_pack.py']
        else:
            mtp = root / 'mtp'
            commands = [[python, '-u', str(Path(runtime.path) / 'tools/mtp_fetch.py'), 'fetch', '--out', str(mtp)],
                        [python, '-u', str(Path(runtime.path) / 'tools/mtp_pack.py'), '--src', str(mtp),
                         '--experts', 'q2_0', '--out', str(mtp / 'mtp-q2_0.gguf')],
                        [python, '-u', str(Path(runtime.path) / 'tools/mtp_rt.py'), '--gguf',
                         str(mtp / 'mtp-q2_0.gguf'), '--out', str(mtp / 'rt')]]
            required = ['tools/mtp_fetch.py', 'tools/mtp_pack.py', 'tools/mtp_rt.py']
            commands = [[python, '-u', '-c', route + 'import runpy, sys\nscript = sys.argv.pop(1)\nrunpy.run_path(script, run_name="__main__")\n',
                         *argv[2:]] for argv in commands]
        for name in required:
            if not (Path(runtime.path) / name).is_file():
                raise ValueError(f'Strata 环境缺少 {name}，无法准备草稿资源')
        env = local_process_environment()
        proxy = self.proxy_provider() if self.proxy_provider else ''
        for key in list(env):
            if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
                env.pop(key)
        env.update(HTTP_PROXY=proxy or '', HTTPS_PROXY=proxy or '', ALL_PROXY='', NO_PROXY='',
                   HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0', STRATA_NO_ACQUIRE='0',
                   HF_HUB_DISABLE_TELEMETRY='1', PYTHONPATH=runtime.path)
        temporary = root / 'tmp'
        temporary.mkdir(exist_ok=True)
        env.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
        for argv in commands:
            process = job = None
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                *argv, cwd=runtime.path, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 1024, **({'start_new_session': True} if os.name != 'nt' else {'creationflags': 0x08000000})))
            try:
                process = await asyncio.shield(spawning)
                job = WindowsJob(process.pid)
                from collections import deque
                tail = deque(maxlen=20)
                while raw := await process.stdout.readline():
                    line = scrub(raw.decode('utf-8', errors='replace').strip())
                    if line:
                        tail.append(line)
                        await progress({'phase': 'preparing', 'message': line})
                code = await process.wait()
                if code:
                    raise ValueError(f'Strata 草稿资源准备失败（{code}）：' + '\n'.join(tail))
            finally:
                if process is None:
                    try:
                        process = await spawning
                        job = WindowsJob(process.pid)
                    except (Exception, asyncio.CancelledError):
                        pass
                if job:
                    job.close()
                if process and process.returncode is None:
                    if os.name != 'nt':
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    await process.wait()
        if state['effective_mode'] == 'mtp':
            vocab = Path(runtime.path) / 'data/draft_vocab.bin'
            if vocab.is_file():
                shutil.copyfile(vocab, root / 'mtp/rt/draft_vocab.bin')

    def owned_files(self, model) -> list[str]:
        files = []
        for record in self.store.all(self.KIND):
            if record.get('model_id') == model.id:
                root = self._owned_path(record)
                files.extend(str(p) for p in root.rglob('*') if p.is_file() and p.resolve().is_relative_to(root))
        draft_root = self._draft_root(model)
        files.extend(str(p) for p in draft_root.rglob('*') if p.is_file() and p.resolve().is_relative_to(draft_root.resolve()))
        return files

    def remove_owned(self, model) -> None:
        for record in self.store.all(self.KIND):
            if record.get('model_id') == model.id:
                root = self._owned_path(record)
                if root.exists():
                    shutil.rmtree(root)
                self.store.remove(self.KIND, record['id'])
        draft_root = self._draft_root(model)
        if draft_root.exists():
            if draft_root.resolve() != draft_root.absolute():
                raise ValueError('Strata 草稿目录存在链接，拒绝清理')
            shutil.rmtree(draft_root)
