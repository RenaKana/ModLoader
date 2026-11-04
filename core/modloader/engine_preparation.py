"""Native-engine preparation dispatch; Strata retains its original implementation."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePosixPath
import shutil
import uuid

from .models import LaunchContext, LaunchSpec
from .platforms import (model_target_path, runtime_distribution, windows_to_wsl, prepare_launch,
                        wsl_prefix, _run, location_of, windows_path, wsl_python)
from .backends import native_engines as native
from .backends.adapters import OFFLINE_ENV


class NativePreparation:
    KIND = 'ninfer_preparations'

    def __init__(self, paths, store):
        self.paths, self.store = paths, store
        self.root = paths.root / 'prepared/ninfer'

    @staticmethod
    def _base(kind):
        return dict(status='ready', kind=kind, artifact=None, recipes=[], preparation_fields=[], candidates=[], errors=[], message='', source=None, pack=None, tokenizer=None)

    def _owned(self, record, operation='check'):
        if record.get('location', {}).get('platform') == 'wsl2':
            from .backends.installers import _verify_wsl_owner
            _verify_wsl_owner(self.paths)
            storage = record.get('storage_root', '')
            output = str(PurePosixPath(storage) / '.prepared' / record['id'])
            if (record.get('location', {}).get('distribution') != 'ModLoader'
                    or PurePosixPath(storage).parent != PurePosixPath('/opt/modloader/models') or record['output'] != output
                    or PurePosixPath(record['id']).name != record['id']
                    or PurePosixPath(record['model_id']).name != record['model_id']):
                raise ValueError('NInfer Linux 派生资源所有权路径不匹配')
            code = '''import json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); output=Path(sys.argv[2])
if root.resolve()!=root or output.resolve()!=output or json.loads((root/'.modloader-model-storage.json').read_text())['model_id']!=sys.argv[3]:
    raise ValueError('NInfer Linux 派生资源所有权标记不匹配')
if sys.argv[4]=='create':
    output.parent.mkdir(exist_ok=True); output.mkdir(exist_ok=False)
elif sys.argv[4]=='remove' and output.exists(): shutil.rmtree(output)'''
            wsl_python('ModLoader', code, [storage, output, record['model_id'], operation])
            return Path(windows_path('ModLoader', output))
        root = Path(record['output']).resolve()
        if record.get('storage_root'):
            storage = Path(record['storage_root'])
            if (storage.resolve() != storage or root != storage / '.prepared' / record['id']
                    or Path(record['id']).name != record['id']
                    or json.loads((storage / '.modloader-model-storage.json').read_text(encoding='utf-8'))['model_id'] != record['model_id']):
                raise ValueError('NInfer Windows 派生资源所有权路径不匹配')
            return root
        if root.parent != self.root.resolve() or root.name != record['id']:
            raise ValueError('NInfer 派生资源所有权路径不匹配')
        return root

    def _output_record(self, model, runtime, identifier, identity):
        """Migrated native models keep both existing and future artifacts on ext4."""
        record = dict(id=identifier, model_id=model.id, runtime_id=runtime.id, identity=identity)
        storage = model.source.get('managed_root', '')
        if (location_of(model).platform == 'wsl2' and location_of(model).distribution == 'ModLoader'
                and model.managed and model.source.get('kind') == 'storage_migration'
                and PurePosixPath(storage).parent == PurePosixPath('/opt/modloader/models')):
            record.update(output=str(PurePosixPath(storage) / '.prepared' / identifier),
                          location={'platform': 'wsl2', 'distribution': 'ModLoader'}, storage_root=storage)
        else:
            record['output'] = str((self.root / identifier).resolve())
        return record

    def _create_output(self, record):
        if record.get('location', {}).get('platform') == 'wsl2':
            return self._owned(record, 'create')
        output = self._owned(record)
        output.mkdir(parents=True, exist_ok=False)
        return output

    def _remove_output(self, record):
        if record.get('location', {}).get('platform') == 'wsl2':
            self._owned(record, 'remove')
        else:
            root = self._owned(record)
            if root.exists():
                shutil.rmtree(root)

    def _identity(self, model, profile, runtime, selected_resources=None):
        target = model_target_path(model, runtime)
        # Stat only registered source files, never hash or recursively scan weights.
        source_info = native.resource_info(target, runtime) if not model.files else None
        files = model.files or [model.path]
        targets = [model_target_path(model.model_copy(update={'path': value}), runtime) for value in files]
        source = runtime.metadata.get('source_root')
        code = '''import json,sys
from pathlib import Path
rows=[]
for value in json.loads(sys.argv[1]):
    p=Path(value); s=p.stat(); rows.append(dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns))
print(json.dumps(rows))'''
        stamps = source_info['file_stats'] if source_info and source_info.get('is_dir') else json.loads(_run([*wsl_prefix(runtime_distribution(runtime)), '/usr/bin/python3', '-c', code, json.dumps(targets)]))
        fields = ('conversion_recipe', 'prepare_vision', 'prepare_mtp', 'prepare_proposal', 'conversion_source', 'conversion_device', 'chat_template', 'generation_config', 'tokenizer_directory')
        parameters = {key: profile.parameters.get(key) for key in fields}
        if not profile.parameters.get('tokenizer_directory'):
            parameters.pop('tokenizer_directory')  # Keep identities of existing default-resource preparations reusable.
        selected_resources = selected_resources if selected_resources is not None else native.conversion_resources(target, profile, runtime)
        resources = {role: value['stamp'] for role, value in selected_resources.items()}
        if profile.parameters.get('conversion_source'):
            info = native.resource_info(native.resource_target(profile.parameters['conversion_source'], runtime), runtime)
            resources['conversion_source'] = info.get('stamp') or info.get('file_stats')
        return dict(source=stamps, target=target, model_id=model.id, runtime_id=runtime.id,
                    runtime_path=runtime.path, source_root=source, source_commit=runtime.metadata.get('source_commit'),
                    parameters=parameters, resources=resources, mtp=native.mode(profile) == 'mtp')

    def _ninfer(self, model, profile, runtime):
        state = self._base('ninfer_artifact')
        target = model_target_path(model, runtime)
        if model.path.lower().endswith('.ninfer'):
            summary = native.artifact_info(target, runtime)
            accel = native.acceleration(model, profile, runtime, summary)
            state.update(artifact=target, artifact_summary=summary, acceleration=accel,
                         vision_enabled=native.ninfer_vision_enabled(profile, runtime, summary),
                         source={'kind': 'native', 'label': '已选本地 NInfer artifact'})
            if accel['status'] == 'unsupported':
                state.update(status='error', errors=[accel['message']], message=accel['message'])
            return state
        recipes = [dict(id=value, label=value.replace('_', ' ')) for value in native.RECIPES]
        recipe = profile.parameters.get('conversion_recipe')
        fields = [field.model_dump() for field in native.NINFER_PREPARATION_FIELDS
                  if field.key != 'conversion_source' or recipe in {'qwen3_6_27b_nvfp4', 'qwen3_8_27b_nvfp4', 'qwen3_8_flash_next'}]
        state.update(recipes=recipes, preparation_fields=fields, source={'kind': 'hf', 'label': '原始本地 HF 权重（保持不变）'},
                     acceleration=native.acceleration(model, profile, runtime))
        info = native.resource_info(target, runtime, gdn=native.uses_pinned_ninfer_rules(runtime))
        if not info.get('is_dir') or not info.get('config') or not any(name.endswith('.safetensors') for name in info.get('files', [])):
            raise ValueError('NInfer 转换需要完整本地 HF 目录或已有 .ninfer 文件；不会猜测其他格式')
        if info.get('missing_shards'):
            raise ValueError('本地 HF 源缺少分片：' + '、'.join(info['missing_shards']))
        native.validate_conversion_source(info['config'], runtime=runtime)
        native.validate_gdn_source(info, runtime)
        planned = {'components': {'vision': {}} if profile.parameters.get('prepare_vision') else {}}
        state['vision_enabled'] = native.ninfer_vision_enabled(profile, runtime, planned)
        if not recipe:
            return {**state, 'status': 'choice_required', 'message': '请选择与模型匹配的官方配方；不会默认使用 27B 配方。'}
        if recipe not in native.RECIPES:
            raise ValueError('所选转换配方不属于此固定版本的官方配方')
        native.validate_conversion_source(info['config'], recipe, runtime=runtime)
        source_root = runtime.metadata.get('source_root')
        if not source_root or not native.resource_info(str(PurePosixPath(source_root) / 'tools/convert/__main__.py'), runtime).get('is_file'):
            raise ValueError('此 NInfer 环境缺少官方转换工具，请使用完整构建或配置 metadata.source_root')
        if any(field['key'] == 'conversion_source' for field in fields):
            source = profile.parameters.get('conversion_source')
            if not source or not native.resource_info(native.resource_target(source, runtime), runtime).get('is_dir'):
                raise ValueError('该官方配方需要额外的本地源目录，请选择完整来源；不会自动下载')
        resources = native.conversion_resources(target, profile, runtime)
        identity = self._identity(model, profile, runtime, resources)
        for record in self.store.all(self.KIND):
            if record.get('identity') == identity:
                self._owned(record)
                try:
                    summary = native.artifact_info(record['artifact'], runtime)
                except (ValueError, OSError):
                    continue  # A moved/incomplete derived output must be prepared again.
                if profile.parameters.get('prepare_vision') and 'vision' not in summary.get('components', {}):
                    continue
                accel = native.acceleration(model, profile, runtime, summary)
                if accel['status'] == 'unsupported':
                    continue
                return {**state, 'artifact': record['artifact'], 'artifact_summary': summary, 'acceleration': accel,
                        'vision_enabled': native.ninfer_vision_enabled(profile, runtime, summary),
                        'message': '已复用同源模型与配方的完整 NInfer 派生 artifact。'}
        return {**state, 'status': 'needs_preparation', 'message': '首次加载将按所选配方生成独立 NInfer artifact，保留原权重。'}

    def _spark(self, model, profile, runtime):
        state = self._base('sparkinfer_local')
        target = model_target_path(model, runtime)
        local = location_of(model).platform == 'windows'
        info = native.local_resource_info(model.path) if local else native.resource_info(target, runtime)
        if not info.get('exists'):
            raise ValueError('SparkInfer 本地模型路径不存在')
        if not info.get('is_dir') and not target.lower().endswith('.gguf'):
            raise ValueError('SparkInfer 需要本地 GGUF 或原生 HF / NVFP4 目录')
        if info.get('is_dir') and (not info.get('config') or not any(name.endswith('.safetensors') for name in info.get('files', []))):
            raise ValueError('SparkInfer 原生目录缺少 config.json 或 safetensors 权重')
        if info.get('missing_shards'):
            raise ValueError('本地模型目录缺少分片：' + '、'.join(info['missing_shards']))
        tokenizer = native.resource_target(profile.parameters['tokenizer'], runtime) if profile.parameters.get('tokenizer') else str(PurePosixPath(target if info['is_dir'] else str(PurePosixPath(target).parent)) / 'tokenizer.json')
        local_tokenizer = None
        if profile.parameters.get('tokenizer') and not profile.parameters['tokenizer'].startswith('/'):
            local_tokenizer = Path(profile.parameters['tokenizer']).resolve()
        elif local and not profile.parameters.get('tokenizer'):
            local_tokenizer = (Path(model.path) if info['is_dir'] else Path(model.path).parent) / 'tokenizer.json'
        tokenizer_exists = (local_tokenizer is None or local_tokenizer.is_file()) if local else native.resource_info(tokenizer, runtime).get('is_file')
        if not tokenizer_exists:
            raise ValueError('缺少本地 tokenizer.json；请选择匹配文件，不会自动下载')
        if local:
            native.check_spark_access(target, tokenizer, info, runtime)
        accel = native.acceleration(model, profile, runtime)
        field = next(field for field in native.SparkInferAdapter().parameters() if field.key == 'tokenizer')
        state.update(artifact=target, tokenizer=tokenizer, acceleration=accel, preparation_fields=[field.model_dump()],
                     source={'kind': 'native', 'label': '直接读取本地 GGUF / 原生目录'})
        if accel['status'] == 'unsupported':
            state.update(status='error', errors=[accel['message']], message=accel['message'])
        return state

    def preflight(self, model, profile, runtime):
        try:
            native.validate(model, profile, runtime)
            return self._ninfer(model, profile, runtime) if profile.backend == 'ninfer' else self._spark(model, profile, runtime)
        except (ValueError, OSError) as error:
            state = self._base('ninfer_artifact' if profile.backend == 'ninfer' else 'sparkinfer_local')
            if profile.backend == 'ninfer' and not model.path.lower().endswith('.ninfer'):
                state.update(recipes=[dict(id=value, label=value.replace('_', ' ')) for value in native.RECIPES],
                             preparation_fields=[field.model_dump() for field in native.NINFER_PREPARATION_FIELDS])
            elif profile.backend == 'sparkinfer':
                state['preparation_fields'] = [field.model_dump() for field in native.SparkInferAdapter().parameters() if field.key == 'tokenizer']
            state.update(status='error', errors=[str(error)], message=str(error))
            try:
                state['acceleration'] = native.acceleration(model, profile, runtime)
            except (ValueError, OSError):
                pass
            return state

    @staticmethod
    def _resolved(profile, state):
        result = profile.model_copy(deep=True)
        if profile.backend == 'ninfer':
            result.parameters['_prepared_artifact'] = state['artifact']
        return result

    def resolved_profile(self, model, profile, runtime):
        state = self.preflight(model, profile, runtime)
        if state['status'] != 'ready':
            raise ValueError(state['message'])
        return self._resolved(profile, state)

    async def ensure(self, model, profile, runtime, progress):
        state = await asyncio.to_thread(self.preflight, model, profile, runtime)
        if state['status'] == 'ready':
            return self._resolved(profile, state)
        if state['status'] != 'needs_preparation' or profile.backend != 'ninfer':
            raise ValueError(state['message'])
        from .backends.installers import _command
        identity = await asyncio.to_thread(self._identity, model, profile, runtime)
        identifier = uuid.uuid4().hex
        record = self._output_record(model, runtime, identifier, identity)
        linux_output = record['output'] if record.get('location') else await asyncio.to_thread(windows_to_wsl, record['output'], runtime_distribution(runtime))
        artifact = str(PurePosixPath(linux_output) / 'model.ninfer')
        source = runtime.metadata['source_root']
        components = ['text']
        if profile.parameters.get('prepare_vision'):
            components.append('vision')
        if profile.parameters.get('prepare_mtp') or native.mode(profile) == 'mtp':
            components.append('mtp')
        recipe = profile.parameters['conversion_recipe']
        argv = [runtime.metadata.get('python', '/usr/bin/python3'), '-u', '-m', 'tools.convert', '--model', model_target_path(model, runtime),
                '--recipe', recipe, '--components', ','.join(components), '--name', model.name, '--device', profile.parameters.get('conversion_device') or 'cpu', '--out', artifact]
        extra_source = profile.parameters.get('conversion_source')
        if extra_source and recipe in {'qwen3_6_27b_nvfp4', 'qwen3_8_27b_nvfp4', 'qwen3_8_flash_next'}:
            key = 'nvidia' if recipe == 'qwen3_8_flash_next' else 'quantized'
            argv += ['--source', key + '=' + native.resource_target(extra_source, runtime)]
        if profile.parameters.get('prepare_proposal', True):
            argv.append('--proposal')
        # Preserve the model's template rather than applying a 27B example to all recipes.
        resources = await asyncio.to_thread(native.conversion_resources, model_target_path(model, runtime), profile, runtime)
        argv += ['--resource', 'chat_template.jinja=' + resources['chat_template.jinja']['path']]
        if profile.parameters.get('generation_config'):
            argv += ['--resource', 'generation_config.json=' + resources['generation_config.json']['path']]
        if profile.parameters.get('tokenizer_directory'):
            for role in ('tokenizer.json', 'tokenizer_config.json'):
                argv += ['--resource', role + '=' + resources[role]['path']]
        work = self.paths.work / ('ninfer-prepare-' + identifier)
        work.mkdir(parents=True, exist_ok=False)
        context = LaunchContext(port=0, instance_id='prepare-' + identifier, work_dir=str(work), data_dir=str(self.paths.root))
        spec = LaunchSpec(platform='wsl2', distribution=runtime_distribution(runtime), argv=argv, cwd=source,
                          env={**runtime.metadata.get('build_environment', {}), **OFFLINE_ENV}, model_name=profile.alias)
        registered = False
        await asyncio.to_thread(self._create_output, record)
        try:
            prepared = await asyncio.to_thread(prepare_launch, spec, context)
            async def update(event):
                await progress({**event, 'phase': 'preparing'})
            await progress({'phase': 'preparing', 'message': '正在转换 NInfer artifact'})
            await _command(prepared.argv, update, asyncio.Event(), cwd=prepared.cwd, cleanup=prepared.cleanup_argv,
                           timeout=profile.parameters.get('startup_timeout') or 7200)
            await progress({'phase': 'validating', 'message': '正在检查 NInfer artifact 与所选组件'})
            summary = await asyncio.to_thread(native.artifact_info, artifact, runtime)
            if identity != await asyncio.to_thread(self._identity, model, profile, runtime):
                raise ValueError('准备期间源模型或配方发生变化，本次派生文件不登记')
            if not set(components) <= set(summary.get('components', {})):
                raise ValueError('转换输出未包含所请求的完整组件，本次派生文件不登记')
            record.update(artifact=artifact, artifact_summary=summary)
            self.store.put(self.KIND, record)
            registered = True
            await progress({'phase': 'prepared', 'message': 'NInfer artifact 已完整检查，开始加载。'})
            return self._resolved(profile, {'artifact': artifact})
        finally:
            if not registered:
                await asyncio.to_thread(self._remove_output, record)

    def owned_files(self, model):
        files = []
        for record in self.store.all(self.KIND):
            if record.get('model_id') == model.id:
                root = self._owned(record)
                files.extend(str(path) for path in root.rglob('*') if path.is_file() and path.resolve().is_relative_to(root))
        return files

    def remove_owned(self, model):
        for record in self.store.all(self.KIND):
            if record.get('model_id') == model.id:
                self._remove_output(record)
                self.store.remove(self.KIND, record['id'])


class PreparationDispatcher:
    def __init__(self, paths, store, proxy_provider=None):
        from .strata_preparation import StrataPreparation
        self.strata = StrataPreparation(paths, store, proxy_provider=proxy_provider)
        self.native = NativePreparation(paths, store)

    def __getattr__(self, name):
        return getattr(self.strata, name)

    def for_profile(self, profile):
        return self.native if profile.backend in {'ninfer', 'sparkinfer'} else self.strata

    def preflight(self, model, profile, runtime): return self.for_profile(profile).preflight(model, profile, runtime)
    def resolved_profile(self, model, profile, runtime): return self.for_profile(profile).resolved_profile(model, profile, runtime)
    async def ensure(self, model, profile, runtime, progress): return await self.for_profile(profile).ensure(model, profile, runtime, progress)
    def owned_files(self, model): return self.strata.owned_files(model) + self.native.owned_files(model)
    def remove_owned(self, model):
        self.strata.remove_owned(model)
        self.native.remove_owned(model)
