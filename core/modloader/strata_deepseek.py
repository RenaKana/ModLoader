"""DeepSeek V4.1's distinct Strata engine and local pack contract."""
from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
import re

from .assets import GGUF_SHARD, gguf_metadata
from .models import ParameterField


def is_deepseek(model):
    return (model.architecture or model.metadata.get('general.architecture')) == 'deepseek41'


def engine_path(root: Path, boundary: Path | None = None) -> Path:
    root = Path(root)
    candidates = [root / 'engine-dsv41' / 'strata-dsv41.exe',
                  *(root / 'engine-dsv41').glob('*/strata-dsv41.exe'),
                  root / 'engine' / 'strata-dsv41.exe']
    def version(path):
        build = path.parent / 'BUILD.json'
        try:
            value = json.loads(build.read_text('utf-8-sig')).get('version', '') if build.is_file() else path.parent.name
        except (OSError, ValueError, AttributeError):
            value = path.parent.name
        match = re.match(r'(\d+(?:\.\d+)+)', str(value))
        return tuple(map(int, match[1].split('.'))) if match else ()
    present = [path for path in candidates if (boundary is None or path.resolve().is_relative_to(Path(boundary).resolve())) and path.is_file()]
    if not present:
        raise ValueError('所选 Strata 环境缺少 strata-dsv41.exe；请选择 DeepSeek V4.1 专用引擎环境')
    return max(present, key=version).resolve()


def fields():
    return [
        ParameterField(key='configuration', label='已有 Strata JSON', type='file'),
        ParameterField(key='gpu', label='GPU 索引', type='text'),
        ParameterField(key='context_length', label='上下文长度', type='number', default=8192, minimum=1, integer=True, step=1),
        ParameterField(key='vram_reserve_mib', label='显存预留 MiB', type='number', minimum=0, integer=True, step=1),
        ParameterField(key='cpu_threads', label='CPU 线程', type='number', minimum=0, integer=True, step=1),
        ParameterField(key='ssd_cache', label='专家 RAM 缓存 GB', type='text', default='auto'),
        ParameterField(key='dspark_mode', label='DSpark', type='select', default='auto', options=[
            {'value': 'auto', 'label': '自动'}, {'value': 'on', 'label': '开启'},
            {'value': 'off', 'label': '关闭推测（保留草稿加载）'}, {'value': 'none', 'label': '不加载草稿'}]),
        ParameterField(key='engram', label='Engram GGUF', type='file'),
        ParameterField(key='engram_meta', label='Engram 元数据 GGUF', type='file',
                       description='缺少哈希参数的量化使用同版本 UD-Q8_K_XL 第 1 分片。'),
        ParameterField(key='dspark', label='DSpark GGUF', type='file'),
        ParameterField(key='pack', label='已准备的 DeepSeek pack 目录', type='file', advanced=True),
        ParameterField(key='tokenizer', label='本地 Strata tokenizer 目录', type='file', advanced=True),
        ParameterField(key='expert_profile', label='专家访问配置', type='file', advanced=True),
    ]


def shards(model):
    source = Path(model.path).resolve()
    match = GGUF_SHARD.fullmatch(source.name)
    paths = [source] if not match else [source.with_name(
        f'{match[1]}-{i:05d}-of-{match[3]}{match[4]}') for i in range(1, int(match[3]) + 1)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f'模型分片不存在：{path}')
    return paths


def dependencies(model, profile):
    """Resolve only this model's directory/import root; never search arbitrary parents."""
    source = Path(model.path).resolve()
    roots = [source.parent]
    known = model.source.get('managed_root') or model.metadata.get('import_root')
    if known and source.is_relative_to(Path(known).resolve()):
        roots.append(Path(known).resolve())
    candidates = set()
    for root in roots:
        candidates.update(p.resolve() for p in root.glob('*.gguf') if p.is_file() and p.resolve().is_relative_to(root))
        for name in ('deepseek-v4.1-flash-shared', 'Engram'):
            candidates.update(p.resolve() for p in (root / name).glob('*.gguf') if p.is_file() and p.resolve().is_relative_to(root))
    result = {}
    patterns = {'engram': ('*engram*.gguf',), 'engram_meta': ('*UD-Q8_K_XL-00001-of-*.gguf',),
                'dspark': ('dspark-*.gguf',)}
    for key, patterns_for_key in patterns.items():
        explicit = profile.parameters.get(key)
        if explicit:
            chosen = Path(explicit).resolve()
            if not chosen.is_file():
                raise ValueError(f'{key} 文件不存在：{chosen}')
        else:
            found = [p for p in sorted(candidates) if any(fnmatchcase(p.name.lower(), pattern.lower()) for pattern in patterns_for_key)]
            if len(found) > 1:
                raise ValueError(f'发现多个 {key} 文件，请在加载配置中明确选择')
            chosen = found[0] if found else None
        if chosen:
            meta = gguf_metadata(chosen)
            arch = meta.get('general.architecture')
            valid = arch in ('engram', 'deepseek41-engram') if key == 'engram' else arch == 'deepseek41' if key == 'engram_meta' else arch == 'dflash'
            if not valid:
                raise ValueError(f'{key} 的 GGUF 架构不匹配：{arch}（{chosen}）')
            result[key] = str(chosen)
    return result


def preparation_inputs(model, profile, runtime):
    from .resource_metadata import read_metadata
    from .mtp import argument
    sources = shards(model)
    deps = dependencies(model, profile)
    descriptors = [read_metadata(path) for path in sources]
    names = {t['name'] for desc in descriptors for t in desc['tensors']}
    config = {key: value for desc in descriptors for key, value in desc['config'].items()}
    if not any(name == 'per_layer_token_embd.weight' or name.endswith('.engram_embd.weight') for name in names) and not deps.get('engram'):
        raise ValueError('此量化未内置 Engram 表；请选择配套 Engram GGUF')
    if not any(key.endswith(('.engram.token_map', '.ple.token_map')) for key in config) and 'engram_token_map.weight' not in names and not deps.get('engram_meta'):
        raise ValueError('模型缺少 Engram 哈希元数据；请选择同版本 UD-Q8_K_XL 第 1 分片')
    if (argument(profile.extra_args, '--dspark') or profile.parameters.get('dspark_mode')) == 'on' and not deps.get('dspark'):
        raise ValueError('开启 DSpark 需要配套 DSpark GGUF')
    profile_path = profile.parameters.get('expert_profile')
    if profile_path == 'none':
        profile_path = None
    if not profile_path:
        match = re.search(r'GSQ-RCO-(IQ1_M|IQ2_XXS|IQ2_XS)(?:-|\.)', sources[0].name, re.I)
        if match:
            name = 'profile-2.25.bin' if match[1].upper() == 'IQ2_XS' else 'profile-2.00.bin'
            candidate = Path(runtime.path) / 'data/dsv41' / name
            if candidate.is_file():
                profile_path = str(candidate.resolve())
    if profile_path:
        profile_path = str(Path(profile_path).resolve())
        if not Path(profile_path).is_file():
            raise ValueError(f'专家访问配置不存在：{profile_path}')
    return sources, deps, profile_path


def pack_description(pack):
    """Read the upstream ds4/tensor directory and referenced file bounds, not weights."""
    pack = Path(pack).resolve()
    geo = dict(line.rstrip('\r\n').split(' ', 1) for line in (pack / 'ds4.txt').read_text('utf-8').splitlines() if ' ' in line)
    if geo.get('format') != 'gguf' or not geo.get('model_dir'):
        raise ValueError('DeepSeek pack 缺少 GGUF model_dir')
    root = Path(geo['model_dir'])
    if not root.is_absolute():
        raise ValueError('DeepSeek pack 的 model_dir 必须为绝对路径')
    files, tensors = {}, []
    for line in (pack / 'tensors.txt').read_text('utf-8').splitlines():
        if line.startswith('file '):
            _, number, name = line.split(' ', 2)
            files[int(number)] = (root / name).resolve()
        elif line.startswith('t '):
            row = line.split()
            if len(row) < 7:
                raise ValueError('DeepSeek tensors.txt 张量描述不完整')
            index, offset, size = map(int, row[3:6])
            if offset < 0 or size <= 0:
                raise ValueError('DeepSeek tensors.txt 包含无效张量范围')
            tensors.append((row[1], index, offset + size))
    if not tensors:
        raise ValueError('DeepSeek pack 没有张量')
    sizes = {index: path.stat().st_size for index, path in files.items()}
    for name, index, end in tensors:
        if index not in sizes or sizes[index] < end:
            raise ValueError(f'DeepSeek pack 引用的权重缺失或截断：{name}')
    if not (pack / 'engram.bin').is_file() or not (pack / 'engram.bin').stat().st_size:
        raise ValueError('DeepSeek pack 缺少 engram.bin')
    if not (pack / 'tokenizer/chat_template.jinja').is_file():
        raise ValueError('DeepSeek pack 缺少模型 chat_template.jinja')
    return {'geometry': geo, 'files': list(files.values()), 'dspark': any(name.startswith('mtp.') for name, _, _ in tensors)}


def validate_pack(pack, model):
    result = pack_description(pack)
    if Path(model.path).resolve() not in result['files']:
        raise ValueError('DeepSeek pack 属于另一模型；请选择本模型的准备文件')
    return result
