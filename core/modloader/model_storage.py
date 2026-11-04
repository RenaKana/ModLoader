"""Explicit, copy-first model relocation between Windows and WSL."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import re
import hashlib
import os
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import uuid

from .assets import inspect_asset, IGNORED_MODEL_DIRECTORIES
from .models import LaunchContext, LaunchSpec, ModelAsset, RuntimeInstallation
from .platforms import (location_of, runtime_distribution, windows_to_wsl,
                        wsl_python, prepare_launch, require_wsl2, model_filesystem_path, windows_path)
from .runtime import ACTIVE_STATES
from .store import UNFINISHED_JOBS


STORAGE_BASE = '/opt/modloader/models'
STORAGE_MARKER = '.modloader-model-storage.json'
RESOURCE_FIELDS = ('mmproj', 'tokenizer', 'tokenizer_directory', 'chat_template',
                   'generation_config', 'conversion_source', 'draft_model',
                   'configuration', 'pack', 'expert_profile', '_prepared_artifact')

# Executed by the existing WSL supervisor. Its pipe-lifetime/process-group
# handling also applies to cancellation, application exit and a dead host.
COPY_CODE = r'''import hashlib,json,os,sys,time
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
base=Path('/opt/modloader/models')
root=Path(request['root'])
if root.parent!=base or root.resolve()!=root:
    raise ValueError('迁移目标所有权路径不匹配')
base.mkdir(parents=True,exist_ok=True)
if base.resolve()!=base or base.stat().st_dev!=Path('/').stat().st_dev:
    raise ValueError('迁移目标不是发行版原生文件系统')
root.mkdir(exist_ok=False)
with (root/'.modloader-model-storage.json').open('x',encoding='utf-8') as marker:
    json.dump(request['owner'],marker); marker.flush(); os.fsync(marker.fileno())
copied=completed=0
last=0
directories={base.parent,base,root}
def emit(phase,name):
    print(json.dumps(dict(phase=phase,message=name,downloaded=copied,total=request['total_bytes'],
        copied_bytes=copied,total_bytes=request['total_bytes'],completed_files=completed,
        total_files=len(request['files']),percent=int(copied*100/request['total_bytes']) if request['total_bytes'] else 100)),flush=True)
for item in request['files']:
    source=Path(item['source'])
    target=root/item['target']
    if target.resolve().is_relative_to(root) is False or target==root:
        raise ValueError('迁移清单包含无效目标路径')
    before=source.stat()
    if not source.is_file() or before.st_size!=item['size']:
        raise ValueError('迁移源文件发生变化：'+source.name)
    target.parent.mkdir(parents=True,exist_ok=True)
    directory=target.parent
    while directory.is_relative_to(root):
        directories.add(directory)
        if directory==root: break
        directory=directory.parent
    digest=hashlib.sha256() if request['verify_hashes'] else None
    count=0
    with source.open('rb') as incoming,target.open('xb') as outgoing:
        while chunk:=incoming.read(4*1024*1024):
            outgoing.write(chunk)
            if digest: digest.update(chunk)
            count+=len(chunk); copied+=len(chunk)
            if time.monotonic()-last>=.25:
                emit('copying',source.name); last=time.monotonic()
        outgoing.flush(); os.fsync(outgoing.fileno())
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or count!=item['size'] or target.stat().st_size!=count:
        raise ValueError('迁移期间源文件变化或复制不完整：'+source.name)
    os.utime(target,ns=(before.st_atime_ns,before.st_mtime_ns))
    if digest:
        emit('verifying',source.name)
        verification=hashlib.sha256()
        with target.open('rb') as stream:
            while chunk:=stream.read(4*1024*1024): verification.update(chunk)
        if verification.digest()!=digest.digest():
            raise ValueError('迁移目标完整性校验失败：'+source.name)
    completed+=1
    emit('copying',source.name)
for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True):
    descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
emit('copied','文件复制完成，正在检查模型与关联配置')
'''


def _file_stamp(path):
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _tree_files(root):
    """The selected model tree, not its arbitrary import/download parent."""
    files, pending, visited = [], [root], set()
    while pending:
        directory = pending.pop()
        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(root.resolve()) or resolved in visited:
            continue
        visited.add(resolved)
        for entry in directory.iterdir():
            if entry.is_dir():
                if entry.name.lower() not in IGNORED_MODEL_DIRECTORIES:
                    pending.append(entry)
            elif entry.is_file():
                files.append(entry)
    return files


def _model_payload(model):
    """Reinspect supported containers and include their local metadata resources."""
    if model['format'] not in {'safetensors', 'gguf', 'ninfer'}:
        raise ValueError('只能迁移已识别的完整 HF、GGUF 或 NInfer 模型')
    asset = inspect_asset(model['path'], name=model['name'], location=model.get('location'))
    asset.id = model['id']
    asset.source = deepcopy(model.get('source', {'kind': 'local'}))
    asset.managed = model.get('managed', False)
    if asset.format != model['format'] or asset.format not in {'safetensors', 'gguf', 'ninfer'}:
        raise ValueError('只能迁移已识别的完整 HF、GGUF 或 NInfer 模型')
    selected = model_filesystem_path(asset)
    root = selected if selected.is_dir() else selected.parent
    files = _tree_files(root) if selected.is_dir() else [model_filesystem_path(asset, value) for value in asset.files]
    if selected.is_file():
        # Embedded formats still have adjacent tokenizer/template/config resources.
        # Other model weights are not implicitly selected from the same directory.
        for path in root.iterdir():
            if path.is_file() and path.name != STORAGE_MARKER and path.suffix.lower() in {'.json', '.jinja', '.jinja2', '.model', '.txt', '.tiktoken'}:
                files.append(path)
    unique = {str(path.absolute()): path for path in files}
    return asset, root, sorted(unique.values(), key=lambda path: str(path).lower())


class ModelStorageMigration:
    def __init__(self, paths, store, runtime, environment_lock):
        self.paths, self.store, self.runtime = paths, store, runtime
        self.environment_lock = environment_lock
        self.plans = {}
        self.active = {}

    def assert_available(self, model_id, job_id=None):
        owner = self.active.get(model_id)
        if owner is not None and owner != job_id:
            raise ValueError('该模型正在迁移存储位置，请等待完成或取消任务')
        for job in self.store.jobs('model_migration', UNFINISHED_JOBS):
            if job['id'] != job_id and job.get('request', {}).get('model_id') == model_id:
                raise ValueError('该模型已有未完成的存储迁移任务')

    def _assert_stopped(self, model_id):
        if any(instance.model_id == model_id and instance.state in ACTIVE_STATES
               for instance in self.runtime.instances.values()):
            raise ValueError('请先卸载该模型的全部运行实例，再迁移存储位置')

    @staticmethod
    def _overlaps(left, right):
        left, right = Path(left).resolve(), Path(right).resolve()
        return left.is_relative_to(right) or right.is_relative_to(left)

    def _assert_no_downloads(self, source_root):
        for job in self.store.jobs('download', UNFINISHED_JOBS):
            destination = job.get('request', {}).get('destination') or job.get('destination')
            if destination and self._overlaps(source_root, destination):
                raise ValueError('源模型目录仍有未完成的下载任务；请先完成或取消该下载，再迁移')

    def assert_download_available(self, destination):
        """Called by download create/retry/resume before starting its writer."""
        model_ids = set(self.active)
        for job in self.store.jobs('model_migration', UNFINISHED_JOBS):
            model_ids.add(job.get('request', {}).get('model_id'))
        for identifier in model_ids:
            model = self.store.get('models', identifier)
            if model and location_of(model).platform == 'windows':
                source = Path(model['path'])
                root = source if model['format'] == 'safetensors' else source.parent
                if self._overlaps(root, destination):
                    raise ValueError('下载目录中的模型正在迁移，请先等待迁移完成或取消迁移')

    def _snapshot(self, model_id):
        with self.store.lock:
            model = self.store.get('models', model_id)
            if model is None:
                raise ValueError('模型登记不存在')
            profiles = [value for value in self.store.all('profiles') if value['model_id'] == model_id]
            ids = {value['runtime_id'] for value in profiles}
            runtimes = [value for value in self.store.all('runtimes') if value['id'] in ids]
        return model, profiles, runtimes

    def _check_snapshot(self, plan):
        self._assert_no_downloads(plan['_source_root'])
        if self._snapshot(plan['model_id']) != (plan['_model'], plan['_profiles'], plan['_runtimes']):
            raise ValueError('模型、加载配置或引擎已变化，请重新预览迁移')
        for original in plan['_preparations']:
            if self.store.get('ninfer_preparations', original['id']) != original:
                raise ValueError('NInfer 派生资源已变化，请重新预览迁移')

    def _disk(self):
        from .backends.installers import _verify_wsl_owner
        _verify_wsl_owner(self.paths)
        require_wsl2('ModLoader')
        code = """import json,os
from pathlib import Path
p=Path('/opt/modloader/models')
while not p.exists(): p=p.parent
if p.resolve()!=p or p.stat().st_dev!=Path('/').stat().st_dev:
    raise ValueError('迁移目标不是发行版原生文件系统')
s=os.statvfs(p)
print(json.dumps(dict(linux_free_bytes=s.f_bavail*s.f_frsize)))"""
        result = json.loads(wsl_python('ModLoader', code))
        backing = self.paths.runtimes / 'wsl-ModLoader'
        result.update(host_free_bytes=shutil.disk_usage(backing).free, host_path=str(backing))
        return result

    def _resource_target(self, value, root, target_root, distribution='ModLoader'):
        if not isinstance(value, str) or not value:
            return value, False
        if PureWindowsPath(value).is_absolute() or Path(value).is_absolute() and not value.startswith('/'):
            path = Path(value).resolve()
            if path.is_relative_to(root.resolve()):
                return str(PurePosixPath(target_root) / path.relative_to(root.resolve()).as_posix()), True
            return value, False
        if value.startswith('/'):
            guest_root = windows_to_wsl(str(root.resolve()), distribution)
            path = PurePosixPath(value)
            if path.is_relative_to(PurePosixPath(guest_root)):
                return str(PurePosixPath(target_root) / path.relative_to(guest_root)), True
        return value, False

    def _profile_replacement(self, profile, root, target_root, model_id):
        replacement = deepcopy(profile)
        replacement.update(model_id=model_id, revision=profile.get('revision', 1) + 1)
        external = []
        for key in RESOURCE_FIELDS:
            value = profile.get('parameters', {}).get(key)
            if not value:
                continue
            target, mapped = self._resource_target(value, root, target_root)
            replacement.setdefault('parameters', {})[key] = target
            if not mapped:
                external.append({'profile_id': profile['id'], 'field': key, 'path': value})
        # Arbitrary advanced arguments keep their meaning. A path override that
        # would continue using the old model must be resolved by its owner first.
        source_guest = windows_to_wsl(str(root.resolve()), 'ModLoader')
        for value in profile.get('extra_args', []):
            if str(root) in value or source_guest in value:
                raise ValueError('高级参数显式引用原模型目录，请先移至对应资源字段后迁移')
        return replacement, external

    def _delete_reasons(self, plan, *, after_switch=False):
        model = plan['_model']
        reasons = []
        source_root = Path(plan['_source_root'])
        source_location = location_of(model)
        files = [Path(item['source']) for item in plan['_main_files']]
        if any(not path.resolve().is_relative_to(source_root) or path.is_symlink() for path in files):
            reasons.append('源文件包含模型目录之外的文件或符号链接，不能自动删除')
        profiles = [value for value in self.store.all('profiles') if value['model_id'] == model['id']]
        switched = {value['id'] for value in plan['_replacements']}
        if any(value['id'] not in switched for value in profiles):
            reasons.append('仍有未迁移的加载配置使用源模型')
        candidates = {str(path.resolve()).casefold() for path in files}
        guest_roots = {}
        def guest_root(distribution):
            if source_location.platform == 'wsl2':
                return plan['_source_guest_root'] if distribution == source_location.distribution else None
            if distribution not in guest_roots:
                try:
                    guest_roots[distribution] = windows_to_wsl(str(source_root), distribution)
                except (OSError, ValueError):
                    reasons.append('无法确认 ' + distribution + ' 的源文件映射，原文件必须保留')
                    guest_roots[distribution] = None
            return guest_roots[distribution]
        def references(value, distribution=None):
            if not isinstance(value, str) or not value:
                return False
            if distribution and value.startswith('/'):
                if source_location.platform == 'wsl2' and distribution != source_location.distribution:
                    return False
                mapped = guest_root(distribution)
                if mapped is None:
                    return True
                root = PurePosixPath(mapped)
                selected = PurePosixPath(value)
                return selected.is_relative_to(root) or root.is_relative_to(selected)
            try:
                selected = Path(value).resolve()
                return (str(selected).casefold() in candidates
                        or any(path.resolve().is_relative_to(selected) for path in files))
            except (OSError, ValueError):
                return False
        for other in self.store.all('models'):
            if other['id'] in {model['id'], plan['id']}:
                continue
            location = location_of(other)
            distro = location.distribution if location.platform == 'wsl2' else None
            paths = [other['path'], *other.get('files', [])]
            # File containers can share implicit adjacent tokenizer/config files.
            if other.get('format') in {'gguf', 'ninfer'}:
                paths.append(str(PurePosixPath(other['path']).parent) if distro else str(Path(other['path']).parent))
            if any(references(value, distro) for value in paths):
                reasons.append('源文件被其他模型登记引用')
                break
        runtimes = {value['id']: RuntimeInstallation(**value) for value in self.store.all('runtimes')}
        for profile in self.store.all('profiles'):
            if not after_switch and profile['id'] in switched:
                profile = next(value for value in plan['_replacements'] if value['id'] == profile['id'])
            runtime = runtimes.get(profile['runtime_id'])
            distro = runtime_distribution(runtime) if runtime and runtime.platform == 'wsl2' else None
            parameters = profile.get('parameters', {})
            if any(references(parameters.get(key), distro) for key in RESOURCE_FIELDS):
                reasons.append('源目录仍被加载配置中的显式资源路径引用')
                break
            literals = list(profile.get('extra_args', []))
            if runtime:
                custom = runtime.metadata.get('custom', {})
                literals.extend(custom.get('args', []))
                if custom.get('cwd'):
                    literals.append(custom['cwd'])
                literals.extend(value for value in custom.get('env', {}).values() if isinstance(value, str))
            if any(any(str(path) in value for path in files) or str(source_root) in value
                   or (distro and (source_location.platform != 'wsl2' or distro == source_location.distribution)
                       and (guest_root(distro) is None or guest_root(distro) in value))
                   for value in literals):
                reasons.append('源目录仍被加载配置的高级参数或自定义引擎环境变量引用')
                break
        if source_location.platform == 'wsl2':
            try:
                self._check_guest_sources(plan)
            except (ValueError, OSError) as error:
                reasons.append(str(error))
        return list(dict.fromkeys(reasons))

    def _interrupted_targets(self, model_id):
        """Only job-recorded, marked, unregistered partials are retry cleanup candidates."""
        results = []
        for job in self.store.jobs('model_migration', ('interrupted', 'failed', 'cancelled')):
            request = job.get('request', {})
            identifier = request.get('plan_id')
            root = job.get('migration_root')
            if (request.get('model_id') != model_id or not identifier
                    or not root or PurePosixPath(root).parent != PurePosixPath(STORAGE_BASE)):
                continue
            if self._registered_target(root):
                continue
            owner = dict(version=1, model_id=identifier, source_model_id=model_id, job_id=job['id'])
            code = '''import json,sys
from pathlib import Path
p=Path(sys.argv[1]); expected=json.loads(sys.argv[2]); size=0
if p.exists():
    if p.resolve()!=p or json.loads((p/'.modloader-model-storage.json').read_text())!=expected:
        raise ValueError('中断任务目录所有权不同')
    size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
print(json.dumps(dict(size=size,exists=p.exists())))'''
            try:
                info = json.loads(wsl_python('ModLoader', code, [root, json.dumps(owner)]))
            except (OSError, ValueError):
                continue
            if info['exists']:
                results.append(dict(root=root, owner=owner, size_bytes=info['size']))
        return results

    def _registered_target(self, root):
        prefix = PurePosixPath(root)
        if any(location_of(model).platform == 'wsl2' and location_of(model).distribution == 'ModLoader'
               and PurePosixPath(model['path']).is_relative_to(prefix) for model in self.store.all('models')):
            return True
        for profile in self.store.all('profiles'):
            if any(isinstance(value, str) and value.startswith('/') and PurePosixPath(value).is_relative_to(prefix)
                   for value in profile.get('parameters', {}).values()):
                return True
            if any(root in value for value in profile.get('extra_args', [])):
                return True
        return any(str(record.get('artifact', '')).startswith(root + '/') for record in self.store.all('ninfer_preparations'))

    def _build_plan(self, model_id):
        from .engine_preparation import NativePreparation
        self.assert_available(model_id)
        self._assert_stopped(model_id)
        model, profiles, runtimes = self._snapshot(model_id)
        if location_of(model).platform != 'windows':
            raise ValueError('此操作用于将 Windows 模型复制到 ModLoader 原生 Linux 存储；当前模型已在 WSL 中')
        identifier = uuid.uuid4().hex
        storage_root = self._readable_root(model['name'])
        registered_path = Path(model['path'])
        self._assert_no_downloads(registered_path if model['format'] == 'safetensors' else registered_path.parent)
        asset, source_root, files = _model_payload(model)
        # Keep future .prepared outputs outside the HF source tree, including
        # when the original directory itself has a reserved or empty basename.
        target_root = str(PurePosixPath(storage_root) / 'model') if asset.format == 'safetensors' else storage_root
        target_path = target_root if asset.format == 'safetensors' else str(PurePosixPath(target_root) / Path(asset.path).name)
        manifest = [{'source': str(path), 'target': str(PurePosixPath(target_root).relative_to(storage_root) / path.relative_to(source_root).as_posix()), **_file_stamp(path)} for path in files]
        plan = dict(id=identifier, model_id=model_id, eligible=True, errors=[], warnings=[], source_path=model['path'],
                    target_path=target_path, target_directory=storage_root, source_platform='windows', target_platform='wsl2', distribution='ModLoader', profiles=[], preserved_profiles=[], external_resources=[],
                    _model=model, _profiles=profiles, _runtimes=runtimes, _preparations=[], _prepared_copies=[],
                    _target_location={'platform': 'wsl2', 'distribution': 'ModLoader'},
                    _source_root=str(source_root), _target_root=target_root, _storage_root=storage_root,
                    _main_files=deepcopy(manifest), _payload_sources={str(path) for path in files}, _files=manifest, _replacements=[])
        needs_preparation = False
        preparation = NativePreparation(self.paths, self.store)
        by_id = {value['id']: RuntimeInstallation(**value) for value in runtimes}
        for profile in profiles:
            runtime = by_id.get(profile['runtime_id'])
            reason = None
            if not runtime or runtime.platform != 'wsl2' or runtime_distribution(runtime) != 'ModLoader':
                reason = '保持原位置：此配置使用 Windows 或其他 WSL 发行版引擎'
            if not reason:
                adapter = self.runtime.adapters.get(profile['backend'])
                if adapter and adapter.inspect(asset, runtime).status == 'unsupported':
                    reason = '保持原位置：所选引擎不支持该模型格式'
            if not reason:
                try:
                    custom = runtime.metadata.get('custom', {})
                    literal_paths = [*custom.get('args', []), custom.get('cwd', ''),
                                     *(value for value in custom.get('env', {}).values() if isinstance(value, str))]
                    source_guest = windows_to_wsl(str(source_root), 'ModLoader')
                    if any(str(source_root) in value or source_guest in value for value in literal_paths):
                        raise ValueError('自定义引擎的参数、工作目录或环境变量固定引用原模型目录；请先调整共享引擎模板后重新预览')
                    replacement, external = self._profile_replacement(profile, source_root, target_root, identifier)
                    plan['external_resources'].extend(external)
                    if profile['backend'] == 'ninfer' and asset.format != 'ninfer':
                        from .models import LoadProfile
                        identity = preparation._identity(asset, LoadProfile(**profile), runtime)
                        matched = False
                        for record in self.store.all(preparation.KIND):
                            if record.get('identity') != identity:
                                continue
                            output = preparation._owned(record)
                            from .backends.native_engines import artifact_info
                            try:
                                prepared_asset = inspect_asset(str(output / 'model.ninfer'))
                                artifact_info(record['artifact'], runtime)
                            except (ValueError, OSError):
                                continue  # The preparation dispatcher also regenerates missing/incomplete artifacts.
                            if record['id'] not in {item['original']['id'] for item in plan['_prepared_copies']}:
                                prepared_id = uuid.uuid4().hex
                                relative = PurePosixPath('.prepared') / prepared_id
                                prepared_files = [dict(source=value, target=str(relative / Path(value).name), **_file_stamp(Path(value))) for value in prepared_asset.files]
                                copied = dict(original=record, id=prepared_id, relative=str(relative), profile_ids=[])
                                plan['_prepared_copies'].append(copied)
                                plan['_preparations'].append(record)
                                plan['_files'].extend(prepared_files)
                            copied = next(item for item in plan['_prepared_copies'] if item['original']['id'] == record['id'])
                            copied['profile_ids'].append(profile['id'])
                            matched = True
                            break
                        needs_preparation = needs_preparation or not matched
                    plan['_replacements'].append(replacement)
                    plan['profiles'].append({key: profile[key] for key in ('id', 'name', 'backend', 'runtime_id')})
                except (ValueError, OSError) as error:
                    reason = '保持原位置：' + str(error)
            if reason:
                plan['preserved_profiles'].append(dict(id=profile['id'], name=profile['name'], reason=reason))
        # Explicit in-root resource fields may include a projector or other known
        # model resource not included by the container's own file list.
        known = {item['source'] for item in manifest}
        for profile in plan['_replacements']:
            original = next(value for value in profiles if value['id'] == profile['id'])
            for key in RESOURCE_FIELDS:
                old = original.get('parameters', {}).get(key)
                new = profile.get('parameters', {}).get(key)
                if not old or old == new:
                    continue
                candidate = Path(old).resolve() if not str(old).startswith('/') else None
                if candidate is None:
                    guest_root = PurePosixPath(windows_to_wsl(str(source_root), 'ModLoader'))
                    candidate = source_root / str(PurePosixPath(old).relative_to(guest_root))
                if candidate.is_dir():
                    selected = _tree_files(candidate)
                elif candidate.suffix.lower() in {'.gguf', '.ninfer'}:
                    selected = [Path(value) for value in inspect_asset(str(candidate)).files]
                else:
                    selected = [candidate]
                for path in selected:
                    if str(path) not in known:
                        if not path.is_file():
                            raise ValueError('显式模型资源不存在：' + str(path))
                        row = dict(source=str(path), target=str(PurePosixPath(target_root).relative_to(storage_root) / path.relative_to(source_root).as_posix()), **_file_stamp(path))
                        manifest.append(row); plan['_main_files'].append(deepcopy(row)); known.add(str(path))
        plan.update(file_count=len(manifest), total_bytes=sum(item['size'] for item in manifest),
                    prepared_bytes=sum(item['size'] for item in manifest if item['target'].startswith('.prepared/')), prepared_count=len(plan['_prepared_copies']))
        plan['delete_source_bytes'] = sum(item['size'] for item in plan['_main_files'])
        plan['delete_source_reasons'] = self._delete_reasons(plan)
        plan['delete_source_allowed'] = not plan['delete_source_reasons']
        if plan['external_resources']:
            plan['warnings'].append('复制目录之外的显式分词器、模板、草稿等资源保持原路径；仅模型及列出的派生工件转为 Linux 原生存储。')
        if plan['_prepared_copies']:
            plan['warnings'].append('原 NInfer 派生工件与记录保留；删除源文件选项只清理本次复制的原模型文件。')
        if needs_preparation:
            plan['warnings'].append('部分 NInfer 配置尚无可复用派生工件；首次加载会在 Linux 模型目录转换，另需工件空间；迁移本身不会提前转换。')
        plan['disk'] = self._disk()
        plan['_partials'] = self._interrupted_targets(model_id)
        reclaimed = sum(value['size_bytes'] for value in plan['_partials'])
        plan['disk']['partial_reclaimable_bytes'] = reclaimed
        if plan['_partials']:
            plan['warnings'].append('重试会清理此模型此前中断任务的、带匹配所有权标记且尚未登记的部分副本；不会清理其他目录。')
        if plan['disk']['host_free_bytes'] < plan['total_bytes'] or plan['disk']['linux_free_bytes'] + reclaimed < plan['total_bytes']:
            plan['eligible'] = False
            plan['errors'].append('磁盘空间不足：WSL 可用空间和其宿主磁盘的实际可用空间都必须能容纳完整副本')
        return plan

    async def preview(self, model_id, options=None):
        options = options or {}
        target = options.get('target_platform', 'wsl2')
        try:
            if set(options) - {'target_platform', 'destination'}:
                raise ValueError('不支持的存储操作参数')
            if target not in {'windows', 'wsl2'}:
                raise ValueError('请选择 Windows 或 WSL 存储目标')
            if target == 'windows':
                plan = await asyncio.to_thread(self._build_windows_plan, model_id, options.get('destination'))
            else:
                plan = await asyncio.to_thread(self._build_plan, model_id)
        except (ValueError, OSError) as error:
            return dict(id='', model_id=model_id, eligible=False, errors=[str(error)], warnings=[], profiles=[],
                        preserved_profiles=[], file_count=0, total_bytes=0, delete_source_allowed=False,
                        delete_source_reasons=[str(error)], target_platform=target, distribution='ModLoader')
        self.plans[plan['id']] = plan
        return {key: deepcopy(value) for key, value in plan.items() if not key.startswith('_')}

    async def _copy(self, plan, job_id, progress, cancel, delete_source):
        if plan['target_platform'] == 'windows':
            return await self._copy_windows(plan, job_id, progress, cancel, delete_source)
        from .backends.installers import _command
        work = self.paths.work / ('model-migration-' + uuid.uuid4().hex)
        work.mkdir()
        try:
            request = dict(root=plan['_storage_root'], owner=dict(version=1, model_id=plan['id'], source_model_id=plan['model_id'], job_id=job_id),
                           total_bytes=plan['total_bytes'], verify_hashes=delete_source, files=[])
            roots = {}
            for item in plan['_files']:
                source = Path(item['source'])
                # Each prepared artifact can have its own Windows source root.
                parent = str(source.parent.resolve())
                if parent not in roots:
                    roots[parent] = await asyncio.to_thread(windows_to_wsl, parent, 'ModLoader')
                request['files'].append({**item, 'source': str(PurePosixPath(roots[parent]) / source.name)})
            request_file = work / 'migration.json'
            request_file.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            linux_request = await asyncio.to_thread(windows_to_wsl, str(request_file), 'ModLoader')
            spec = LaunchSpec(platform='wsl2', distribution='ModLoader', argv=['/usr/bin/python3', '-u', '-c', COPY_CODE, linux_request],
                              cwd='/opt/modloader', model_name='model-storage-migration')
            context = LaunchContext(port=0, instance_id='migration-' + plan['id'], work_dir=str(work), data_dir=str(self.paths.root))
            prepared = await asyncio.to_thread(prepare_launch, spec, context)
            async def update(event):
                try:
                    payload = json.loads(event.get('message', ''))
                except (ValueError, TypeError):
                    return
                if isinstance(payload, dict):
                    await progress({**payload, 'can_pause': False})
            # WSL may retain its host launch cwd after wsl.exe exits. Keep that
            # cwd at the stable volume root, never inside our disposable request
            # directory; guest worker cwd remains the explicit native Linux path.
            await _command(prepared.argv, update, cancel, cwd=Path(self.paths.work).anchor,
                           cleanup=prepared.cleanup_argv, timeout=None)
        finally:
            shutil.rmtree(work)

    def _cleanup(self, plan, job_id):
        owner = dict(version=1, model_id=plan['id'], source_model_id=plan['model_id'], job_id=job_id)
        if plan['target_platform'] == 'windows':
            self._cleanup_windows(plan, owner)
        else:
            self._cleanup_owned(plan['_storage_root'], owner)

    def _cleanup_owned(self, root, owner):
        from .backends.installers import _verify_wsl_owner
        _verify_wsl_owner(self.paths)
        code = """import json,shutil,sys
from pathlib import Path
p=Path(sys.argv[1]); expected=json.loads(sys.argv[2])
if not p.exists(): sys.exit(0)
if p.parent!=Path('/opt/modloader/models') or p.resolve()!=p:
    raise ValueError('拒绝清理迁移目录之外的路径')
if json.loads((p/'.modloader-model-storage.json').read_text())!=expected:
    raise ValueError('迁移目录所有权标记不同，未清理')
shutil.rmtree(p)"""
        if self._registered_target(root):
            raise ValueError('迁移目录已被模型登记使用，未清理')
        wsl_python('ModLoader', code, [root, json.dumps(owner)], timeout=120)

    def _source_unchanged(self, plan):
        _, _, current = _model_payload(plan['_model'])
        if {str(path) for path in current} != plan['_payload_sources']:
            raise ValueError('迁移期间源模型文件清单发生变化，未登记目标副本')
        for item in plan['_files']:
            if _file_stamp(Path(item['source'])) != {key: item[key] for key in ('size', 'mtime_ns')}:
                raise ValueError('迁移期间源文件发生变化，未登记目标副本')

    async def run(self, request, job_id, progress, cancel):
        plan = self.plans.get(request.get('plan_id'))
        if plan is None or plan['model_id'] != request.get('model_id'):
            raise ValueError('迁移预览已失效，请重新检查并确认')
        delete_source = request.get('delete_source', False)
        if type(delete_source) is not bool:
            raise ValueError('删除源文件选项必须为布尔值')
        if not plan['eligible']:
            raise ValueError('；'.join(plan['errors']))
        async with self.environment_lock:
            self.assert_available(plan['model_id'], job_id)
            self._assert_stopped(plan['model_id'])
            self._check_snapshot(plan)
            if delete_source:
                reasons = await asyncio.to_thread(self._delete_reasons, plan)
                if reasons:
                    raise ValueError('；'.join(reasons))
            self.active[plan['model_id']] = job_id
        committed = False
        result = dict(model_ids=[plan['id']], model_id=plan['id'], source_model_id=plan['model_id'],
                      switched_profile_ids=[value['id'] for value in plan['_replacements']], source_deleted=False,
                      warnings=list(plan['warnings']), target_platform=plan['target_platform'])
        try:
            if cancel.is_set():
                raise asyncio.CancelledError()
            await progress(dict(phase='checking', message='正在检查目标目录与可用空间', can_pause=False,
                                migration_root=plan['_storage_root'], total_bytes=plan['total_bytes'], total_files=plan['file_count']))
            async with self.environment_lock:
                for partial in plan['_partials']:
                    await asyncio.to_thread(self._cleanup_owned, partial['root'], partial['owner'])
            disk = await asyncio.to_thread(self._disk_windows, plan['_storage_root']) if plan['target_platform'] == 'windows' else await asyncio.to_thread(self._disk)
            if min(disk['host_free_bytes'], disk.get('linux_free_bytes', disk['host_free_bytes'])) < plan['total_bytes']:
                raise ValueError('磁盘空间不足，未开始复制模型')
            await self._copy(plan, job_id, progress, cancel, delete_source)
            if cancel.is_set():
                raise asyncio.CancelledError()
            await asyncio.to_thread(self._source_unchanged, plan)
            new_model = await asyncio.to_thread(inspect_asset, plan['target_path'], name=plan['_model']['name'],
                                              location=plan['_target_location'])
            new_model.id = plan['id']
            new_model.managed = True
            new_model.metadata = {**deepcopy(plan['_model'].get('metadata', {})), **new_model.metadata}
            if 'import_root' in new_model.metadata:
                new_model.metadata['import_root'] = plan['_target_root']
            new_model.source = dict(kind='storage_migration', managed_root=plan['_storage_root'], original_model_id=plan['model_id'],
                                    migration_id=plan['id'], original_source=deepcopy(plan['_model'].get('source', {})))
            from .engine_preparation import NativePreparation
            from .models import LoadProfile
            from .backends.native_engines import artifact_info
            preparation = NativePreparation(self.paths, self.store)
            runtimes = {value['id']: RuntimeInstallation(**value) for value in plan['_runtimes']}
            prepared_records = []
            for copied in plan['_prepared_copies']:
                profile = next(value for value in plan['_replacements'] if value['id'] in copied['profile_ids'])
                runtime = runtimes[profile['runtime_id']]
                output = str((Path if plan['target_platform'] == 'windows' else PurePosixPath)(plan['_storage_root']) / copied['relative'])
                artifact = str((Path if plan['target_platform'] == 'windows' else PurePosixPath)(output) / 'model.ninfer')
                # Container validation and the engine's own inspector both run on
                # the destination before any profile is switched.
                await asyncio.to_thread(inspect_asset, artifact, location=plan['_target_location'])
                runtime_artifact = await asyncio.to_thread(windows_to_wsl, artifact, runtime_distribution(runtime)) if plan['target_platform'] == 'windows' and runtime.platform == 'wsl2' else artifact
                summary = await asyncio.to_thread(artifact_info, runtime_artifact, runtime)
                identity = await asyncio.to_thread(preparation._identity, new_model, LoadProfile(**profile), runtime)
                prepared_records.append(dict(id=copied['id'], model_id=new_model.id, runtime_id=runtime.id,
                    identity=identity, output=output, artifact=runtime_artifact, artifact_summary=summary,
                    location=plan['_target_location'], storage_root=plan['_storage_root']))
            async with self.environment_lock:
                self._assert_stopped(plan['model_id'])
                self._check_snapshot(plan)
                if cancel.is_set():
                    raise asyncio.CancelledError()
                await progress(dict(phase='registering', message='模型检查完成，正在同时登记模型与切换配置', can_pause=False))
                self.store.apply_model_storage_migration(plan['_model'], plan['_profiles'], plan['_runtimes'],
                    new_model.model_dump(), plan['_replacements'], expected_preparations=plan['_preparations'], new_preparations=prepared_records)
                committed = True
                if delete_source:
                    try:
                        reasons = await asyncio.to_thread(self._delete_reasons, plan, after_switch=True)
                    except (OSError, ValueError) as error:
                        reasons = ['删除前的源文件引用检查未完成：' + str(error)]
                    if reasons:
                        result['warnings'].append('副本已登记，原文件保留：' + '；'.join(reasons))
                    else:
                        # The valid native copy is already registered. A cleanup
                        # failure cannot roll it back or remove its profiles.
                        try:
                            await asyncio.to_thread(self._source_unchanged, plan)
                            await asyncio.to_thread(self._delete_sources, plan)
                            self.store.remove('models', plan['model_id'])
                            result['source_deleted'] = True
                        except (OSError, ValueError) as error:
                            result['warnings'].append('副本已登记，原文件清理未全部完成：' + str(error))
            await progress(dict(phase='complete', message='模型存储位置已更新', percent=100,
                                copied_bytes=plan['total_bytes'], total_bytes=plan['total_bytes'], completed_files=plan['file_count'], total_files=plan['file_count'], can_pause=False))
            return result
        except asyncio.CancelledError:
            if committed:
                result['warnings'].append('迁移已登记；后续清理或进度更新已停止，原文件可能保留。')
                return result
            raise
        except OSError as error:
            if getattr(error, 'errno', None) == 28 or getattr(error, 'winerror', None) == 112:
                raise ValueError('磁盘空间不足，迁移未完成；原模型保持不变') from error
            raise
        except RuntimeError as error:
            if 'No space left on device' in str(error) or '[Errno 28]' in str(error):
                raise ValueError('磁盘空间不足，迁移未完成；原模型保持不变') from error
            raise
        finally:
            try:
                if not committed:
                    await asyncio.to_thread(self._cleanup, plan, job_id)
            finally:
                self.active.pop(plan['model_id'], None)
                self.plans.pop(plan['id'], None)

    @staticmethod
    def _directory_name(name):
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '-', name).strip().rstrip('. ')
        if not name or name in {'.', '..'}:
            name = 'model'
        if re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', name):
            name = '_' + name
        return name

    def _readable_root(self, name):
        # Reserve against previews too; mkdir(exist_ok=False) is the final arbiter.
        from .backends.installers import _verify_wsl_owner
        _verify_wsl_owner(self.paths)
        require_wsl2('ModLoader')
        code = '''import json,sys
from pathlib import Path
base=Path('/opt/modloader/models'); name=sys.argv[1]; reserved=set(json.loads(sys.argv[2])); index=1
while True:
    candidate=base/(name if index==1 else name+' ('+str(index)+')')
    if not candidate.exists() and not candidate.is_symlink() and str(candidate) not in reserved: break
    index+=1
print(json.dumps(str(candidate)))'''
        reserved = [plan['_storage_root'] for plan in self.plans.values() if plan['target_platform'] == 'wsl2']
        return json.loads(wsl_python('ModLoader', code, [self._directory_name(name), json.dumps(reserved)]))

    @staticmethod
    def _disk_windows(root):
        parent = Path(root).parent
        if not parent.is_dir() or parent.resolve() != parent:
            raise ValueError('Windows 目标文件夹已变化，请重新选择')
        return dict(host_free_bytes=shutil.disk_usage(parent).free, host_path=str(parent))

    def _build_windows_plan(self, model_id, destination):
        self.assert_available(model_id)
        self._assert_stopped(model_id)
        model, profiles, runtimes = self._snapshot(model_id)
        location = location_of(model)
        if location.platform != 'wsl2':
            raise ValueError('请选择 WSL 模型移到 Windows')
        if not isinstance(destination, str) or not PureWindowsPath(destination).is_absolute():
            raise ValueError('请选择 Windows 目标文件夹')
        parent = Path(destination).resolve(strict=True)
        if not parent.is_dir() or str(parent).lower().startswith(('\\\\wsl.localhost\\', '\\\\wsl$\\')):
            raise ValueError('目标必须是 Windows 文件夹，不能是 WSL 内部路径')
        asset, source_root, files = _model_payload(model)
        if self._overlaps(source_root, parent):
            raise ValueError('源目录与目标文件夹不能相同或互相包含')
        name = self._directory_name(model['name'])
        storage_root = parent / name
        index = 1
        reserved = {plan['_storage_root'] for plan in self.plans.values()}
        while storage_root.exists() or storage_root.is_symlink() or str(storage_root) in reserved:
            index += 1
            storage_root = parent / f'{name} ({index})'
        target_root = storage_root / 'model' if asset.format == 'safetensors' else storage_root
        selected = model_filesystem_path(asset)
        target_path = target_root if asset.format == 'safetensors' else target_root / selected.name
        guest_root = model['path'] if selected.is_dir() else str(PurePosixPath(model['path']).parent)
        destination_guest = windows_to_wsl(str(parent), location.distribution)
        overlap = json.loads(wsl_python(location.distribution, '''import json,sys
from pathlib import Path
source,target=(Path(value).resolve() for value in sys.argv[1:])
print(json.dumps(source.is_relative_to(target) or target.is_relative_to(source)))''', [guest_root, destination_guest]))
        if overlap:
            raise ValueError('源目录与目标文件夹不能相同或互相包含')
        identifier = uuid.uuid4().hex
        manifest = [dict(source=str(path), target=(target_root.relative_to(storage_root) / path.relative_to(source_root)).as_posix(), **_file_stamp(path)) for path in files]
        plan = dict(id=identifier, model_id=model_id, eligible=True, errors=[], warnings=[], source_path=model['path'],
            target_path=str(target_path), target_directory=str(storage_root), source_platform='wsl2', target_platform='windows',
            distribution=location.distribution, profiles=[], preserved_profiles=[], external_resources=[],
            _model=model, _profiles=profiles, _runtimes=runtimes, _preparations=[], _prepared_copies=[], _partials=[],
            _source_root=str(source_root), _source_guest_root=guest_root, _target_root=str(target_root), _storage_root=str(storage_root),
            _target_location={'platform': 'windows'}, _main_files=deepcopy(manifest), _payload_sources={str(path) for path in files},
            _files=manifest, _replacements=[])
        by_id = {value['id']: RuntimeInstallation(**value) for value in runtimes}
        for profile in profiles:
            runtime = by_id.get(profile['runtime_id'])
            try:
                if runtime is None or runtime.platform == 'wsl2' and runtime_distribution(runtime) != location.distribution:
                    raise ValueError('此配置使用其他发行版或缺失的引擎')
                adapter = self.runtime.adapters.get(profile['backend'])
                if adapter and adapter.inspect(asset, runtime).status == 'unsupported':
                    raise ValueError('所选引擎不支持该模型格式')
                replacement = deepcopy(profile)
                replacement.update(model_id=identifier, revision=profile.get('revision', 1) + 1)
                custom = runtime.metadata.get('custom', {})
                literals = [*profile.get('extra_args', []), *custom.get('args', []), custom.get('cwd', ''),
                            *(v for v in custom.get('env', {}).values() if isinstance(v, str))]
                if any(guest_root in value or str(source_root) in value for value in literals):
                    raise ValueError('高级参数或自定义引擎固定引用原模型目录')
                additions, external = [], []
                for key in RESOURCE_FIELDS:
                    value = profile.get('parameters', {}).get(key)
                    if not isinstance(value, str) or not value:
                        continue
                    relative = None
                    if value.startswith('/') and PurePosixPath(value).is_relative_to(guest_root):
                        relative = PurePosixPath(value).relative_to(guest_root)
                    elif PureWindowsPath(value).is_absolute() and Path(value).resolve().is_relative_to(source_root.resolve()):
                        relative = Path(value).resolve().relative_to(source_root.resolve())
                    if relative is None:
                        if runtime.platform != 'wsl2' and value.startswith('/'):
                            raise ValueError('此 Windows 配置仍引用目录之外的 Linux 资源')
                        external.append(dict(profile_id=profile['id'], field=key, path=value))
                        continue
                    resource = source_root / str(relative)
                    if resource.is_dir():
                        additions.extend(_tree_files(resource))
                    elif resource.suffix.lower() in {'.gguf', '.ninfer'}:
                        additions.extend(Path(path) for path in inspect_asset(str(resource)).files)
                    else:
                        if not resource.is_file():
                            raise ValueError('显式模型资源不存在：' + value)
                        additions.append(resource)
                    output = str(target_root / str(relative))
                    replacement.setdefault('parameters', {})[key] = windows_to_wsl(output, runtime_distribution(runtime)) if runtime.platform == 'wsl2' else output
                known = {item['source'] for item in manifest}
                for path in additions:
                    if str(path) not in known:
                        row = dict(source=str(path), target=(target_root.relative_to(storage_root) / path.relative_to(source_root)).as_posix(), **_file_stamp(path))
                        manifest.append(row); plan['_main_files'].append(deepcopy(row)); known.add(str(path))
                plan['_replacements'].append(replacement)
                plan['external_resources'].extend(external)
                plan['profiles'].append({key: profile[key] for key in ('id', 'name', 'backend', 'runtime_id')})
            except (ValueError, OSError) as error:
                plan['preserved_profiles'].append(dict(id=profile['id'], name=profile['name'], reason='保持原位置：' + str(error)))
        self._windows_preparations(plan, asset, by_id)
        if plan['_prepared_copies']:
            plan['warnings'].append('原 NInfer 派生工件与记录保留；删除源文件选项只清理本次复制的原模型文件。')
        plan.update(file_count=len(manifest), total_bytes=sum(item['size'] for item in manifest),
                    prepared_bytes=sum(item['size'] for item in manifest if item['target'].startswith('.prepared/')),
                    prepared_count=len(plan['_prepared_copies']))
        plan['delete_source_bytes'] = sum(item['size'] for item in plan['_main_files'])
        plan['delete_source_reasons'] = self._delete_reasons(plan)
        plan['delete_source_allowed'] = not plan['delete_source_reasons']
        plan['disk'] = self._disk_windows(storage_root)
        if plan['disk']['host_free_bytes'] < plan['total_bytes']:
            plan['eligible'] = False
            plan['errors'].append('Windows 目标磁盘空间不足，无法容纳完整副本')
        return plan

    def _windows_preparations(self, plan, asset, runtimes):
        from .engine_preparation import NativePreparation
        from .models import LoadProfile
        preparation = NativePreparation(self.paths, self.store)
        for replacement in plan['_replacements']:
            if replacement['backend'] != 'ninfer' or asset.format == 'ninfer':
                continue
            original = next(p for p in plan['_profiles'] if p['id'] == replacement['id'])
            runtime = runtimes[original['runtime_id']]
            identity = preparation._identity(asset, LoadProfile(**original), runtime)
            for record in self.store.all(preparation.KIND):
                if record.get('identity') != identity:
                    continue
                output = preparation._owned(record)
                try:
                    prepared = inspect_asset(str(output / 'model.ninfer'))
                except (ValueError, OSError):
                    continue
                copied = next((item for item in plan['_prepared_copies'] if item['original']['id'] == record['id']), None)
                if copied is None:
                    identifier = uuid.uuid4().hex
                    relative = PurePosixPath('.prepared') / identifier
                    copied = dict(original=record, id=identifier, relative=str(relative), profile_ids=[])
                    plan['_prepared_copies'].append(copied)
                    plan['_preparations'].append(record)
                    plan['_files'].extend(dict(source=value, target=str(relative / Path(value).name), **_file_stamp(Path(value))) for value in prepared.files)
                copied['profile_ids'].append(original['id'])
                if original.get('parameters', {}).get('_prepared_artifact'):
                    artifact = str(Path(plan['_storage_root']) / copied['relative'] / 'model.ninfer')
                    replacement['parameters']['_prepared_artifact'] = windows_to_wsl(artifact, runtime_distribution(runtime)) if runtime.platform == 'wsl2' else artifact
                break

    async def _copy_windows(self, plan, job_id, progress, cancel, verify):
        loop = asyncio.get_running_loop()
        def copy():
            root = Path(plan['_storage_root'])
            root.mkdir(exist_ok=False)
            owner = dict(version=1, model_id=plan['id'], source_model_id=plan['model_id'], job_id=job_id)
            with (root / STORAGE_MARKER).open('x', encoding='utf-8') as marker:
                json.dump(owner, marker); marker.flush(); os.fsync(marker.fileno())
            copied = completed = 0
            last = 0
            def emit(phase, name):
                asyncio.run_coroutine_threadsafe(progress(dict(phase=phase, message=name, copied_bytes=copied,
                    downloaded=copied, total=plan['total_bytes'], total_bytes=plan['total_bytes'], completed_files=completed,
                    total_files=plan['file_count'], percent=int(copied * 100 / plan['total_bytes']) if plan['total_bytes'] else 100,
                    can_pause=False)), loop).result()
            for item in plan['_files']:
                if cancel.is_set():
                    raise asyncio.CancelledError()
                source, target = Path(item['source']), root / item['target']
                if not target.resolve().is_relative_to(root.resolve()) or target == root:
                    raise ValueError('迁移清单包含无效目标路径')
                before = _file_stamp(source)
                if before != {key: item[key] for key in ('size', 'mtime_ns')}:
                    raise ValueError('迁移源文件发生变化：' + source.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256() if verify else None
                with source.open('rb') as incoming, target.open('xb') as outgoing:
                    while chunk := incoming.read(4 * 1024 * 1024):
                        if cancel.is_set():
                            raise asyncio.CancelledError()
                        outgoing.write(chunk); copied += len(chunk)
                        if digest:
                            digest.update(chunk)
                        if time.monotonic() - last >= .25:
                            emit('copying', source.name); last = time.monotonic()
                    outgoing.flush(); os.fsync(outgoing.fileno())
                if _file_stamp(source) != before or target.stat().st_size != before['size']:
                    raise ValueError('迁移期间源文件变化或复制不完整：' + source.name)
                shutil.copystat(source, target)
                if digest:
                    emit('verifying', source.name)
                    check = hashlib.sha256()
                    with target.open('rb') as stream:
                        while chunk := stream.read(4 * 1024 * 1024):
                            if cancel.is_set():
                                raise asyncio.CancelledError()
                            check.update(chunk)
                    if check.digest() != digest.digest():
                        raise ValueError('迁移目标完整性校验失败：' + source.name)
                completed += 1
                emit('copying', source.name)
            emit('copied', '文件复制完成，正在检查模型与关联配置')
        # Do not clean a target while its worker is still writing after task cancellation.
        worker = asyncio.create_task(asyncio.to_thread(copy))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel.set()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            raise

    def _cleanup_windows(self, plan, owner):
        root = Path(plan['_storage_root'])
        if not root.exists():
            return
        if root.resolve() != root or json.loads((root / STORAGE_MARKER).read_text(encoding='utf-8')) != owner:
            raise ValueError('迁移目录所有权标记不同，未清理')
        if any(location_of(model).platform == 'windows' and Path(model['path']).resolve().is_relative_to(root)
               for model in self.store.all('models')):
            raise ValueError('迁移目录已被模型登记使用，未清理')
        shutil.rmtree(root)

    def _check_guest_sources(self, plan, delete=False):
        location = location_of(plan['_model'])
        root = Path(plan['_source_root'])
        rows = [dict(path=str(PurePosixPath(plan['_source_guest_root']) / Path(item['source']).relative_to(root).as_posix()), size=item['size'])
                for item in plan['_main_files']]
        code = '''import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=json.loads(sys.argv[2])
for row in rows:
    p=Path(row['path'])
    if p==root or p.resolve()!=p or not p.is_relative_to(root) or not p.is_file() or p.stat().st_size!=row['size']:
        raise ValueError('源文件包含符号链接、目录外文件或已发生变化，不能自动删除')
if sys.argv[3]=='delete':
    for row in rows: Path(row['path']).unlink()
'''
        wsl_python(location.distribution, code, [plan['_source_guest_root'], json.dumps(rows), 'delete' if delete else 'check'])

    def _delete_sources(self, plan):
        if location_of(plan['_model']).platform == 'wsl2':
            self._check_guest_sources(plan, delete=True)
        else:
            for item in plan['_main_files']:
                Path(item['source']).unlink()
