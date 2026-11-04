"""Native NInfer artifacts and SparkInfer local targets, with truthful acceleration modes."""
from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path, PurePosixPath
import re
import uuid
from threading import RLock
from time import perf_counter

from ..models import Compatibility, LaunchSpec, ParameterField, RuntimeInstallation
from ..gpu_config import cuda_environment
from ..platforms import (distribution_name, linux_path, model_target_path, runtime_distribution,
                         validate_model_runtime, windows_to_wsl, wsl_prefix, _run)
from .adapters import BackendAdapter, OFFLINE_ENV, _caps, _extra, _mapped
from .source_builds import PINS
from .ninfer_capabilities import UINT32_MAX, ninfer_capacity_mode, uses_new_ninfer_source, validate_capacity_parameters
from .ninfer_tokenizer import RESOURCE_INSPECTION_CODE
from ..resource_inspection import resource_info as local_resource_info


_ARTIFACT_CACHE = OrderedDict()
_ARTIFACT_CACHE_LOCK = RLock()
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1

NINFER_GENERATION = ['temperature', 'top_p', 'max_tokens', 'top_k', 'seed']
SPARK_GENERATION = ['temperature', 'top_p', 'max_tokens', 'top_k', 'seed']
RECIPES = ['qwen3_6_27b', 'qwen3_6_27b_nvfp4', 'qwen3_8_27b', 'qwen3_8_27b_nvfp4', 'qwen3_6_35b_a3b', 'qwen3_8_flash_next']
QWEN_DENSE = {'Qwen3_5ForCausalLM', 'Qwen3_5ForConditionalGeneration'}
QWEN_MOE = {'Qwen3_5MoeForCausalLM', 'Qwen3_5MoeForConditionalGeneration'}
NINFER_PREPARATION_FIELDS = [
    ParameterField(key='conversion_recipe', label='官方转换配方', type='select', options=RECIPES),
    ParameterField(key='prepare_vision', label='准备视觉组件', type='boolean', default=False),
    ParameterField(key='prepare_mtp', label='准备 MTP 组件', type='boolean', default=False),
    ParameterField(key='prepare_proposal', label='准备优化提案头', type='boolean', default=True),
    ParameterField(key='conversion_source', label='配方的额外本地源目录', type='file', description='NVFP4 配方选择量化权重；Flash Next 选择 NVIDIA 权重。'),
    ParameterField(key='conversion_device', label='转换设备', type='text', default='cpu', description='cpu、cuda 或带设备索引，如 cuda:0。'),
    ParameterField(key='chat_template', label='本地聊天模板（.jinja）', type='file', description='留空使用源目录中的 chat_template.jinja。'),
    ParameterField(key='generation_config', label='本地生成配置（generation_config.json）', type='file', description='留空使用源目录中的 generation_config.json。'),
    ParameterField(key='tokenizer_directory', label='配套分词器目录', type='file', description='需含配套的 tokenizer.json 和 tokenizer_config.json；留空使用源目录。'),
]
NINFER_TEXT_RESOURCES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')
NINFER_VISION_RESOURCES = ('preprocessor_config.json', 'video_preprocessor_config.json')
# Pinned src/product/speculative_options.h rejects MTP draft counts outside 1..5.
NINFER_ACCEL_FIELDS = [ParameterField(key='draft_tokens', label='MTP 草稿 Token', type='number', default=3, minimum=1,
                                    maximum=5, integer=True, step=1)]
SPARK_ACCEL_FIELDS = [
    ParameterField(key='draft_model', label='本地 DFlash2 / DSpark 草稿目录', type='file',
                   description='仅用于 Qwen3.8-27B；选择配套 HF 草稿，不接受 MTP GGUF。'),
    ParameterField(key='draft_proposals', label='每轮草稿提案 Token', type='number', default=0, minimum=INT32_MIN,
                   maximum=INT32_MAX, integer=True, step=1,
                   description='非正数自适应；实际深度不超过 15，并由草稿 block 夹取。'),
]


def uses_pinned_ninfer_rules(runtime):
    """A display version or external --help does not establish source identity."""
    return (runtime.backend == 'ninfer'
            and runtime.metadata.get('source_commit') == PINS['ninfer']['commit']
            and runtime.metadata.get('version_basis') == 'verified_source_commit')


def import_native(backend, path, metadata=None, platform='wsl2'):
    if platform != 'wsl2':
        raise ValueError(f'{backend} 本次交付只支持 WSL2')
    meta = dict(metadata or {})
    distribution = distribution_name(meta.get('distribution', 'ModLoader'))
    path = linux_path(path)
    source = meta.get('source_root')
    if not source:
        expected = 'ninfer-serve' if backend == 'ninfer' else 'sparkinfer_server'
        if PurePosixPath(path).name == expected and PurePosixPath(path).parent.parent.name == 'build':
            source = str(PurePosixPath(path).parents[2])
    if source:
        meta['source_root'] = linux_path(source)
    meta['distribution'] = distribution
    version = str(meta.pop('version', 'unknown'))
    label = 'NInfer' if backend == 'ninfer' else 'SparkInfer'
    return RuntimeInstallation(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f'{backend}:wsl2:{distribution}:{path}')),
                               backend=backend, platform='wsl2', name=f'{label} · {version}', path=path, version=version, metadata=meta)


def resource_target(value, runtime):
    if not isinstance(value, str) or not value.strip() or any(character in value for character in '\0\r\n'):
        raise ValueError('请选择有效的本地资源路径')
    return linux_path(value) if value.startswith('/') else windows_to_wsl(str(Path(value).resolve()), runtime_distribution(runtime))


def resource_info(path, runtime, *, gdn=False):
    code = Path(__file__).parents[1].joinpath('resource_inspection.py').read_text(encoding='utf-8')
    code += "\nimport sys\nprint(json.dumps(resource_info(sys.argv[1],gdn=sys.argv[2]=='gdn')))"
    return json.loads(_run([*wsl_prefix(runtime_distribution(runtime)), '/usr/bin/python3', '-c', code, path, 'gdn' if gdn else '']))


def spark_launch_paths(model, profile, runtime):
    path = model_target_path(model, runtime)
    info = resource_info(path, runtime)
    tokenizer = (resource_target(profile.parameters['tokenizer'], runtime) if profile.parameters.get('tokenizer')
                 else str((PurePosixPath(path) if info.get('is_dir') else PurePosixPath(path).parent) / 'tokenizer.json'))
    return path, tokenizer


def check_spark_access(target, tokenizer, info, runtime):
    """Keep guest filesystem checks after parsing a Windows source locally."""
    code = '''import json,sys
from pathlib import Path
root=Path(sys.argv[1]); tokenizer=Path(sys.argv[2]); expected=json.loads(sys.argv[3])
if not root.exists(): raise ValueError('SparkInfer WSL 模型路径不可访问：'+str(root))
if expected['is_dir']:
    files={p.name for p in root.iterdir()}
    if 'config.json' not in files or not any(name.endswith('.safetensors') and (root/name).is_file() for name in files):
        raise ValueError('SparkInfer WSL 目录缺少 config.json 或 safetensors 权重')
    for name in expected['readable']:
        with (root/name).open('rb') as stream: stream.read(1)
    for name in expected['shards']:
        shard=root/name
        if not shard.resolve().is_relative_to(root.resolve()): raise ValueError('Shard path escaped selected model directory')
        if not shard.is_file(): raise ValueError('SparkInfer WSL 目录缺少分片：'+name)
elif not root.is_file(): raise ValueError('SparkInfer WSL 模型不是文件：'+str(root))
if not tokenizer.is_file(): raise ValueError('SparkInfer WSL tokenizer.json 不可访问：'+str(tokenizer))'''
    expected = dict(is_dir=info['is_dir'], shards=info.get('referenced_shards', []),
                    readable=[name for name in info.get('files', [])
                              if name == 'config.json' or name.endswith('.safetensors.index.json')])
    _run([*wsl_prefix(runtime_distribution(runtime)), '/usr/bin/python3', '-c', code,
          target, tokenizer, json.dumps(expected)])


def conversion_resources(target, profile, runtime):
    """Check the small required resources from pinned tools/convert/resources.py.

    Tensor/token-domain and component validation remain in the official tools.
    Explicit sidecars are read in place; the original HF directory is untouched.
    """
    roles = NINFER_TEXT_RESOURCES + (NINFER_VISION_RESOURCES if profile.parameters.get('prepare_vision') else ())
    selected = {role: str(PurePosixPath(target) / role) for role in roles}
    for key, role in [('chat_template', 'chat_template.jinja'), ('generation_config', 'generation_config.json')]:
        if profile.parameters.get(key):
            selected[role] = resource_target(profile.parameters[key], runtime)
    directory = profile.parameters.get('tokenizer_directory')
    if directory:
        directory = resource_target(directory, runtime)
        for role in ('tokenizer.json', 'tokenizer_config.json'):
            selected[role] = str(PurePosixPath(directory) / role)
    elif profile.parameters.get('tokenizer_json'):
        raise ValueError('分词器需选择包含 tokenizer.json + tokenizer_config.json 的“配套分词器目录”，不能只指定 tokenizer_json 单文件。')
    code = RESOURCE_INSPECTION_CODE + '''
import json,sys
from pathlib import Path
resources={}; errors=[]; parsed_resources={}; contract=None
for role,value in json.loads(sys.argv[1]).items():
    p=Path(value)
    try:
        if not p.is_file(): raise ValueError('文件不存在')
        data=p.read_bytes()
        if not data: raise ValueError('文件为空')
        text=data.decode('utf-8')
        if role.endswith('.json'):
            parsed=json.loads(text)
            if not isinstance(parsed,dict): raise ValueError('需要 JSON 对象')
            parsed_resources[role]=parsed
        s=p.stat(); resources[role]=dict(path=value,stamp=dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns))
    except (OSError,ValueError,UnicodeError) as error: errors.append(role+' ('+str(p)+'): '+str(error))
if sys.argv[2]=='pinned' and not errors:
    config=Path(sys.argv[3])/'config.json'
    config=json.loads(config.read_text()) if config.is_file() else {}
    config=config.get('text_config',config)
    contract=inspect_tokenizer_resources(parsed_resources['tokenizer.json'],parsed_resources['tokenizer_config.json'],parsed_resources['generation_config.json'],config.get('vocab_size'))
print(json.dumps(dict(resources=resources,errors=errors,tokenizer_contract=contract)))'''
    pinned = uses_pinned_ninfer_rules(runtime)
    result = json.loads(_run([*wsl_prefix(runtime_distribution(runtime)), '/usr/bin/python3', '-c', code,
                              json.dumps(selected), 'pinned' if pinned else '', target]))
    if result['errors']:
        raise ValueError('NInfer 转换所需资源缺失或无效：' + '；'.join(result['errors']) + '；请选择配套本地文件，不会自动下载或合成。')
    if pinned and result['tokenizer_contract'] is not None:
        from .ninfer_tokenizer import validate_resources
        validate_resources(result['tokenizer_contract'], selected['tokenizer.json'] + ' 配套')
    return result['resources']


def _inspect_artifact(path, runtime):
    source = runtime.metadata.get('source_root')
    if not source:
        raise ValueError('NInfer 运行时缺少官方 artifact 检查工具源目录；请导入完整构建，或提供 metadata.source_root')
    code = RESOURCE_INSPECTION_CODE + '''
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
def stamps(paths):
    rows=[]
    for name in paths:
        p=Path(name); s=p.stat()
        rows.append([str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_mode])
    return rows
source=Path(sys.argv[1])
if not (source/'tools/artifact/inspect.py').is_file(): raise ValueError('Missing official NInfer artifact inspector')
sys.path.insert(0,str(source))
from tools.artifact.reader import Artifact
from tools.artifact.inspect import artifact_summary
entry_before=stamps([sys.argv[2]])[0]
with Artifact(sys.argv[2]) as artifact:
    dependencies={str(artifact.path.parent/f.path) for f in artifact.directory.files[1:]}
    dependencies.update(str(m.__file__) for name,m in sys.modules.items() if name.startswith('tools.artifact') and getattr(m,'__file__',None))
    dependencies.add(sys.executable)
    dependencies.discard(str(artifact.path))
    files=[str(artifact.path),*sorted(dependencies)]
    before=stamps(files)
    for index in range(len(artifact.directory.files)): artifact._file(index)
    summary=artifact_summary(artifact)
    if sys.argv[3]=='gdn':
        from tools.artifact.schema import binding_parts
        summary['gdn_bindings']={}
        for name,binding in artifact.directory.bindings.items():
            if not name.startswith('text/') or not name.endswith(('/gdn/a_projection','/gdn/b_projection')): continue
            parents=[]
            for identifier,begin,end in binding_parts(binding,artifact.by_id):
                obj=artifact.object(identifier)
                parents.append(dict(id=identifier,shape=list(obj.shape),format=obj.format,range=[begin,end]))
            summary['gdn_bindings'][name]=parents
        text=artifact.directory.components.get('text',{}); refs=text.get('resources',{})
        parsed={}; missing=[]
        for role in ('tokenizer.json','tokenizer_config.json','generation_config.json','chat_template.jinja'):
            identifier=refs.get(role)
            if not identifier: missing.append('artifact 缺少 '+role); continue
            data=artifact.read_object(identifier)
            if not data: missing.append('artifact 内的 '+role+' 为空'); continue
            value=data.decode('utf-8')
            if role.endswith('.json'):
                value=json.loads(value)
                if not isinstance(value,dict): raise ValueError(role+' must be an object')
                parsed[role]=value
        contract=inspect_tokenizer_resources(parsed.get('tokenizer.json',{}),parsed.get('tokenizer_config.json',{}),parsed.get('generation_config.json',{}),text.get('config',{}).get('vocab_size'))
        contract['errors']=missing+contract['errors']; summary['tokenizer_contract']=contract
    after=stamps(files)
    summary['_inspection_stamps']=after if before==after and entry_before==before[0] else None
    print(json.dumps(summary))'''
    summary = json.loads(_run([*wsl_prefix(runtime_distribution(runtime)), runtime.metadata.get('python', '/usr/bin/python3'),
                               '-c', code, linux_path(source), path, 'gdn' if uses_pinned_ninfer_rules(runtime) else ''], timeout=90))
    validate_gdn_artifact(summary, runtime)
    contract = summary.pop('tokenizer_contract', None)
    if uses_pinned_ninfer_rules(runtime) and contract is not None:
        from .ninfer_tokenizer import validate_resources
        validate_resources(contract, path + ' 内的分词器/生成')
    # Parent records are a local validation input, not a second object manifest
    # carried into every preparation record and API refresh.
    summary.pop('gdn_bindings', None)
    stamps = summary.pop('_inspection_stamps', None)
    return summary, stamps


def _artifact_stamps(paths, runtime):
    code = '''import json,sys
from pathlib import Path
rows=[]
for name in json.loads(sys.argv[1]):
    p=Path(name); s=p.stat()
    rows.append([str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_mode])
print(json.dumps(rows))'''
    return json.loads(_run([*wsl_prefix(runtime_distribution(runtime)), '/usr/bin/python3', '-c', code, json.dumps(paths)]))


def artifact_info(path, runtime):
    """Reuse successful official inspections only while all dependencies match."""
    requested = perf_counter()
    key = (runtime.id, runtime_distribution(runtime), runtime.path, runtime.version, path,
           runtime.metadata.get('source_root'), runtime.metadata.get('python', '/usr/bin/python3'),
           runtime.metadata.get('source_commit'), runtime.metadata.get('version_basis'))
    with _ARTIFACT_CACHE_LOCK:
        entry = _ARTIFACT_CACHE.get(key)
        if entry is None:
            entry = {'lock': RLock(), 'summary': None, 'stamps': None, 'checked_at': 0}
            _ARTIFACT_CACHE[key] = entry
        _ARTIFACT_CACHE.move_to_end(key)
        while len(_ARTIFACT_CACHE) > 32:
            _ARTIFACT_CACHE.popitem(last=False)
    with entry['lock']:
        if entry['summary'] is not None:
            # Concurrent callers share the check that finished after they arrived.
            if entry['checked_at'] >= requested:
                return deepcopy(entry['summary'])
            current = _artifact_stamps([row[0] for row in entry['stamps']], runtime)
            if current == entry['stamps']:
                entry['checked_at'] = perf_counter()
                return deepcopy(entry['summary'])
        entry['summary'] = None
        summary, stamps = _inspect_artifact(path, runtime)
        if stamps is not None:
            entry.update(summary=summary, stamps=stamps, checked_at=perf_counter())
        return deepcopy(summary)


def mode(profile):
    return profile.parameters.get('acceleration_mode') or ('mtp' if profile.parameters.get('mtp_enabled') else 'none')


def _gdn_error(layer, topology, parents):
    details = '、'.join(f"{parent.get('format', parent.get('dtype'))}{parent.get('shape')}" for parent in parents)
    return ValueError(f"NInfer {PINS['ninfer']['version']} 的 GDN 算子不支持 {layer} 的{topology}父权重 {details}；"
                      '该 paired 路径要求每个父对象为 BF16[48, 5120]。这属于模型几何限制，关闭 MTP 也不能运行。'
                      '请使用与此固定版本兼容的权重。')


def validate_gdn_source(info, runtime):
    """Check actual HF control projections, not a model-family or size label.

    The converter may accept a source that the pinned runtime's GDN planner
    cannot execute. Only safetensors descriptors are read, never tensor data.
    """
    if not uses_pinned_ninfer_rules(runtime):
        return
    layers = {}
    for name, tensor in info.get('gdn_weights', {}).items():
        match = re.fullmatch(r'(.+\.linear_attn)\.in_proj_(a|b|ba)\.weight', name)
        if match:
            layers.setdefault(match[1], {})[match[2]] = tensor
    for layer, weights in layers.items():
        if 'a' in weights and 'b' in weights:
            parents = [weights['a'], weights['b']]
            if any(parent.get('shape') != [48, 5120] for parent in parents):
                raise _gdn_error(layer, '独立 A/B（paired）', parents)


def validate_gdn_artifact(summary, runtime):
    """Use the official inspector's binding parents; no container decoder here."""
    if not uses_pinned_ninfer_rules(runtime):
        return
    bindings = summary.get('gdn_bindings', {})
    for name, a in bindings.items():
        if not name.endswith('/gdn/a_projection'):
            continue
        b = bindings.get(name.removesuffix('a_projection') + 'b_projection')
        if not b or len(a) != 1 or len(b) != 1:
            continue  # Complex bindings remain subject to the official planner.
        parents = [a[0], b[0]]
        if a[0]['id'] == b[0]['id']:
            continue  # Shared/merged parents have a different planning route.
        if any(parent.get('format') != 'bf16' or parent.get('shape') != [48, 5120] for parent in parents):
            raise _gdn_error(name.removesuffix('/a_projection'), '独立 A/B（paired）', parents)


def validate_conversion_source(config, recipe=None, *, runtime=None):
    """Mirror the selected converter's architecture dispatch, not model-size names."""
    architectures = config.get('architectures')
    qwen4 = {'Qwen4ExpForCausalLM'}
    if uses_new_ninfer_source(runtime):
        # New tools/convert/qwen4_exp.py canonicalizes this alias before recipes.
        qwen4.add('Qwen4ExpForConditionalGeneration')
    if not isinstance(architectures, list) or len(architectures) != 1 or architectures[0] not in QWEN_DENSE | QWEN_MOE | qwen4:
        raise ValueError('此固定版本 NInfer 转换器不支持该源架构：' + str(architectures))
    if recipe is None:
        return
    expected = qwen4 if recipe == 'qwen3_8_flash_next' else QWEN_MOE if recipe == 'qwen3_6_35b_a3b' else QWEN_DENSE
    if architectures[0] not in expected:
        raise ValueError('所选官方配方与源模型的 Dense / MoE / Qwen4Exp 架构不匹配')


def acceleration(model, profile, runtime, summary=None):
    selected = mode(profile)
    ninfer = profile.backend == 'ninfer'
    allowed = {'none', 'mtp'} if ninfer else {'none', 'dflash2', 'dspark'}
    if selected not in allowed:
        raise ValueError('此适配器未提供所选加速模式')
    fields = NINFER_ACCEL_FIELDS if ninfer else SPARK_ACCEL_FIELDS
    result = dict(mode=selected, enabled=selected != 'none', status='disabled' if selected == 'none' else 'unverified',
                  kind='mtp' if ninfer else 'draft', label='原生 MTP' if ninfer else 'DFlash2 / DSpark', requires_draft=not ninfer,
                  fields=[field.model_dump() for field in fields], message='')
    if selected == 'none':
        return result
    if ninfer:
        if summary is not None:
            available = 'mtp' in summary.get('components', {})
            result.update(status='available' if available else 'unsupported', message='' if available else '该 artifact 未包含 MTP 组件；请重新准备并包含 MTP。')
        return result
    value = profile.parameters.get('draft_model')
    if not value:
        return {**result, 'status': 'unsupported', 'message': '请选择配套的本地 DFlash2 / DSpark 草稿目录。'}
    draft = resource_info(resource_target(value, runtime), runtime)
    config = draft.get('config', {})
    dflash2 = 'DFlash2DraftModel' in config.get('architectures', []) or (config.get('selector_rank', 0) > 0 and config.get('conv_kernel', 0) > 0)
    dspark = isinstance(config.get('dflash_config'), dict) and not dflash2
    if not draft.get('is_dir') or not any(name.endswith('.safetensors') for name in draft.get('files', [])):
        return {**result, 'status': 'unsupported', 'message': '草稿目录缺少 config.json 或 safetensors 权重。'}
    if draft.get('missing_shards'):
        return {**result, 'status': 'unsupported', 'message': '本地草稿缺少分片：' + '、'.join(draft['missing_shards'])}
    matches = dflash2 if selected == 'dflash2' else dspark
    if matches:
        nested = config.get('dflash_config')
        block = (nested if isinstance(nested, dict) else {}).get('block_size', config.get('block_size', 16))
        if type(block) is not int or block <= 0 or dflash2 and block not in {2, 4, 5, 6, 7, 8, 16}:
            return {**result, 'status': 'unsupported', 'message': '草稿 block_size 不属于此固定版本支持的宽度。'}
        # The runtime clamps the raw proposal count to its 15-token/block-row
        # capacity. That effective depth is not an input rejection boundary.
    return {**result, 'status': 'available' if matches else 'unsupported',
            'message': '' if matches else '本地草稿配置与所选 DFlash2 / DSpark 模式不匹配。'}


def ninfer_vision_enabled(profile, runtime, summary):
    """Resolve one startup choice for the flag, preflight and capabilities."""
    available = 'vision' in summary.get('components', {})
    selected = profile.parameters.get('vision_enabled')
    extra_vision = '--vision' in profile.extra_args
    if selected is False and extra_vision:
        raise ValueError('图像输入已关闭，请移除额外启动参数 --vision。')
    enabled = (available if selected is None else selected) or extra_vision
    if enabled and not available:
        raise ValueError('该 NInfer artifact 未包含视觉组件；请从原始权重重新准备并开启“准备视觉组件”。')
    known = runtime.metadata.get('supported_flags')
    if enabled and known and '--vision' not in known:
        raise ValueError('此 NInfer 运行环境未提供 --vision，请选择支持视觉输入的运行环境。')
    return bool(enabled)


def validate(model, profile, runtime):
    from ..parameter_rules import validate_number
    adapter = NinferAdapter() if profile.backend == 'ninfer' else SparkInferAdapter()
    inactive = {field.key for field in (NINFER_ACCEL_FIELDS if profile.backend == 'ninfer' else SPARK_ACCEL_FIELDS)} if mode(profile) == 'none' else set()
    if profile.backend == 'ninfer' and model.path.lower().endswith('.ninfer'):
        inactive.update(field.key for field in NINFER_PREPARATION_FIELDS)
    fields = adapter.parameters_for_runtime(runtime) if profile.backend == 'ninfer' else adapter.parameters()
    for field in fields:
        if field.key in inactive:
            continue
        value = profile.parameters.get(field.key)
        if field.type == 'number':
            validate_number(field, value)
        elif field.type == 'boolean' and value is not None and not isinstance(value, bool):
            raise ValueError(f'{field.label}必须为布尔值')
        elif field.type == 'select' and value not in (None, '') and value not in field.options:
            raise ValueError(f'{field.label}不是该引擎支持的选项')
        elif field.key == 'conversion_device' and value not in (None, ''):
            if not isinstance(value, str) or re.fullmatch(r'(?:cpu|cuda)(?::[0-9]+)?', value) is None:
                raise ValueError('转换设备必须为 cpu、cuda 或带非负索引的设备，如 cuda:0')
    if mode(profile) not in ({'none', 'mtp'} if profile.backend == 'ninfer' else {'none', 'dflash2', 'dspark'}):
        raise ValueError('加速模式与引擎不匹配')
    if profile.backend == 'ninfer':
        validate_capacity_parameters(profile, runtime)
        capacity = profile.parameters.get('kv_capacity')
        if ninfer_capacity_mode(runtime) == 'kv_tokens' and capacity not in (None, '', 'auto'):
            # strtol accepts a leading sign/ASCII whitespace, but requires the
            # final decimal digit to be the end of the input.
            number = capacity
            if isinstance(capacity, str) and re.fullmatch(r'[ \t\n\r\v\f]*[+-]?[0-9]+', capacity):
                number = int(capacity)
            if type(number) is not int or not 1 <= number <= INT32_MAX:
                raise ValueError(f'KV 容量必须为 auto 或 1..{INT32_MAX} 的整数（Token 数）')
            context = profile.parameters.get('context_length')
            if context is None or isinstance(context, str) and not context.strip():
                # Pinned src/serve/serve_options.h defaults max_context to 8192.
                context = 8192 if uses_pinned_ninfer_rules(runtime) else None
            if context is not None and number < context:
                raise ValueError('KV 容量不能小于最大上下文')
    elif mode(profile) != 'none' and profile.parameters.get('draft_model'):
        state = acceleration(model, profile, runtime)
        if state['status'] == 'unsupported':
            raise ValueError(state['message'])
    extra(profile, runtime)


def extra(profile, runtime):
    values = _extra(profile, runtime)
    protected = {'--model-path', '--model-id', '--model-name', '--draft-model', '--spec', '--draft-tokens', '--tokenizer', '--tokenizer-path'}
    for value in values:
        if value.split('=', 1)[0] in protected:
            raise ValueError('模型、分词器及加速绑定由加载配置管理，不能在高级参数中覆盖')
    return values


def map_generation(payload, backend):
    result = dict(payload)
    if 'repeat_penalty' in result or backend == 'sparkinfer' and 'repetition_penalty' in result:
        raise ValueError('此引擎没有所请求的重复惩罚采样入口；请移除该参数')
    return result


class NinferAdapter(BackendAdapter):
    id, label = 'ninfer', 'NInfer (WSL2)'

    def parameters(self):
        return [ParameterField(key='context_length', label='最大上下文', type='number', minimum=1, maximum=INT32_MAX, integer=True, step=1),
                ParameterField(key='cuda_visible_devices', label='CUDA 可见设备', type='text'),
                ParameterField(key='max_concurrency', label='最大并行请求', type='number', default=1, minimum=1, maximum=8, integer=True, step=1),
                ParameterField(key='kv_capacity', label='KV 容量（Token）', default='auto', description='auto 或不小于最大上下文的正整数 Token 数。'),
                ParameterField(key='kv_dtype', label='KV 数据类型', type='select', default='bf16', options=['bf16', 'int8', 'fp8', 'nvfp4', 'k8v4']),
                ParameterField(key='vision_enabled', label='图像输入', type='boolean'),
                ParameterField(key='acceleration_mode', label='推测解码', type='select', default='none', options=['none', 'mtp']),
                *[field.model_copy(deep=True) for field in NINFER_ACCEL_FIELDS + NINFER_PREPARATION_FIELDS]]

    def parameters_for_runtime(self, runtime):
        fields = self.parameters()
        if ninfer_capacity_mode(runtime) == 'context_bytes':
            for index, field in enumerate(fields):
                if field.key == 'kv_capacity':
                    fields[index] = ParameterField(key='context_capacity', label='KV / State 共享容量', default='auto',
                                                   description='auto 或字节数；支持 B、KiB、MiB、GiB。')
                elif field.key == 'context_length':
                    field.maximum = UINT32_MAX
        return fields

    def inspect(self, model, runtime):
        validate_model_runtime(model, runtime)
        if runtime.platform != 'wsl2':
            return Compatibility(status='unsupported', reasons=['NInfer 需要 Linux / WSL2 RTX 5090 sm_120a 环境'])
        return Compatibility(status='unverified',
                             capabilities={**_caps(model), 'generation_parameters': NINFER_GENERATION, 'stream_usage': True})

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        validate(model, profile, runtime)
        artifact = profile.parameters.get('_prepared_artifact')
        if not artifact and not model.path.lower().endswith('.ninfer'):
            raise ValueError('请先使用选定官方配方准备独立 NInfer artifact')
        artifact = artifact or model_target_path(model, runtime)
        summary = artifact_info(artifact, runtime)
        state = acceleration(model, profile, runtime, summary)
        if state['status'] == 'unsupported':
            raise ValueError(state['message'])
        vision = ninfer_vision_enabled(profile, runtime, summary)
        capacity_key, capacity_flag = ('context_capacity', '--context-capacity') if ninfer_capacity_mode(runtime) == 'context_bytes' else ('kv_capacity', '--kv-capacity')
        values = {'max_concurrency': 1, capacity_key: 'auto', 'kv_dtype': 'bf16', **profile.parameters}
        argv = [runtime.path, artifact, '--host', '127.0.0.1', '--port', str(context.port), '--model-id', profile.alias]
        argv += _mapped(values, {'context_length': '--max-context', 'max_concurrency': '--max-concurrency', capacity_key: capacity_flag, 'kv_dtype': '--kv-dtype'})
        if state['enabled']:
            argv += ['--spec', 'mtp', '--draft-tokens', str(values.get('draft_tokens', 3))]
        additional = extra(profile, runtime)
        if vision and '--vision' not in additional:
            argv.append('--vision')
        argv += additional
        caps = {**_caps(model, vision), 'generation_parameters': NINFER_GENERATION, 'stream_usage': True}
        return LaunchSpec(platform='wsl2', distribution=runtime_distribution(runtime), argv=argv, cwd=str(PurePosixPath(runtime.path).parent),
                          env={**runtime.metadata.get('build_environment', {}), **OFFLINE_ENV, **cuda_environment(profile, runtime)}, model_name=profile.alias,
                          health_path='/health', health_expected={'status': 'ok'}, capabilities=caps, startup_timeout=900)


class SparkInferAdapter(BackendAdapter):
    id, label = 'sparkinfer', 'SparkInfer (WSL2)'

    def parameters(self):
        return [ParameterField(key='context_length', label='上下文长度', type='number', minimum=INT32_MIN, maximum=INT32_MAX, integer=True, step=1,
                               description='非正数沿用模型配置。'),
                ParameterField(key='cuda_visible_devices', label='CUDA 可见设备', type='text'),
                # SparkInfer parses SPARKINFER_MAX_OUTPUT_TOKENS with atoi into a signed 32-bit int.
                ParameterField(key='server_max_output_tokens', label='服务最大输出 Token', type='number',
                               minimum=INT32_MIN, maximum=INT32_MAX, integer=True, step=1,
                               description='非正数使用引擎默认值。'),
                ParameterField(key='tokenizer', label='本地 tokenizer.json', type='file'),
                ParameterField(key='acceleration_mode', label='草稿加速', type='select', default='none', options=['none', 'dflash2', 'dspark']),
                *[field.model_copy(deep=True) for field in SPARK_ACCEL_FIELDS]]

    def inspect(self, model, runtime):
        validate_model_runtime(model, runtime)
        if runtime.platform != 'wsl2':
            return Compatibility(status='unsupported', reasons=['SparkInfer 本次使用 WSL2 原生构建'])
        return Compatibility(status='unverified',
                             capabilities={**_caps(model, 'image' in model.modalities), 'generation_parameters': SPARK_GENERATION, 'stream_usage': True})

    def build_launch(self, model, profile, runtime, context):
        self._validate(model, runtime, profile)
        validate(model, profile, runtime)
        state = acceleration(model, profile, runtime)
        if state['status'] == 'unsupported':
            raise ValueError(state['message'])
        path, tokenizer = spark_launch_paths(model, profile, runtime)
        if not resource_info(tokenizer, runtime).get('is_file'):
            raise ValueError('SparkInfer 缺少本地 tokenizer.json；请选择匹配文件，不会自动下载')
        argv = [runtime.path, '-m', path, '--tokenizer', tokenizer, '--host', '127.0.0.1', '--port', str(context.port), '--model-name', profile.alias]
        argv += _mapped(profile.parameters, {'context_length': '--ctx'})
        if state['enabled']:
            argv += ['--draft-model', resource_target(profile.parameters['draft_model'], runtime)]
        argv += extra(profile, runtime)
        env = {**runtime.metadata.get('build_environment', {}), **OFFLINE_ENV, 'SPARKINFER_DRAFT_MODEL': '',
               **cuda_environment(profile, runtime),
               'SPARKINFER_DFLASH_PROPOSALS': str(profile.parameters.get('draft_proposals') or 0) if state['enabled'] else '0'}
        output_limit = profile.parameters.get('server_max_output_tokens')
        if output_limit is not None and not (isinstance(output_limit, str) and not output_limit.strip()):
            env['SPARKINFER_MAX_OUTPUT_TOKENS'] = str(output_limit)
        return LaunchSpec(platform='wsl2', distribution=runtime_distribution(runtime), argv=argv, cwd=str(PurePosixPath(runtime.path).parent),
                          env=env,
                          model_name=profile.alias,
                          health_path='/health', health_expected={'status': 'ok'}, capabilities=self.inspect(model, runtime).capabilities, startup_timeout=900)
