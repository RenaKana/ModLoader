import json
import os
import re
import struct
import uuid
from pathlib import Path
from typing import Callable
from .models import ModelAsset


GGUF_SHARD = re.compile(r'^(.*)-(\d{5})-of-(\d{5})(\.gguf)$', re.IGNORECASE)
IGNORED_MODEL_DIRECTORIES = {'.git', '.cache'}
NINFER_HEADER = struct.Struct('<8sQ16s')
NINFER_MAGIC = b'NINFER\0\3'
NINFER_PART_MAGIC = b'NINPRT\0\3'
NINFER_ALIGNMENT = 4096


def _ninfer_continuation(path):
    with path.open('rb') as stream:
        return stream.read(len(NINFER_PART_MAGIC)) == NINFER_PART_MAGIC


def _canonical_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def projector_candidates(model: ModelAsset) -> list[str]:
    """Discover only within a known model/import/download root, never arbitrary parents."""
    if getattr(model, 'location', None):
        from .platforms import location_of, remote_asset_operation
        location = location_of(model)
        if location.platform == 'wsl2':
            return remote_asset_operation(location.distribution, 'projector_candidates', model.model_dump())
    main = Path(model.path).resolve()
    roots = [main.parent if main.is_file() else main]
    known_root = model.source.get('managed_root') or model.metadata.get('import_root')
    if known_root:
        root = Path(known_root).resolve()
        if main.is_relative_to(root):
            roots.append(root)
    candidates = [Path(p) for p in model.metadata.get('mmproj_candidates', []) if isinstance(p, str)]
    for root in roots:
        candidates.extend(root.glob('*mmproj*.gguf'))
    result = {}
    for path in candidates:
        resolved = path.resolve()
        if resolved.suffix.lower() != '.gguf' or not resolved.is_file() or not any(resolved.is_relative_to(root) for root in roots):
            continue
        match = GGUF_SHARD.match(resolved.name)
        if match and int(match.group(2)) != 1:
            continue
        result[_canonical_path(resolved)] = str(resolved)
    return sorted(result.values())


def _hf_files(root: Path) -> list[str]:
    """Include model files without following directory links outside the model."""
    files = []
    pending = [root]
    visited = set()
    while pending:
        directory = pending.pop().resolve(strict=True)
        key = _canonical_path(directory)
        if not directory.is_relative_to(root) or key in visited:
            continue
        visited.add(key)
        for entry in directory.iterdir():
            if entry.is_dir():
                if entry.name.lower() not in IGNORED_MODEL_DIRECTORIES:
                    pending.append(entry)
            elif entry.is_file():
                files.append(str(entry))
    return files


def gguf_metadata(path: Path, *, tensor_names=False):
    """Read metadata only. Skip tokenizer arrays without loading tensor payloads."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        position = 0
        def read(n):
            nonlocal position
            if n < 0 or position + n > size:
                raise ValueError('GGUF 文件截断或元数据损坏')
            value = stream.read(n)
            if len(value) != n:
                raise ValueError('无法完整读取 GGUF 元数据')
            position += n
            return value
        def skip(n, label):
            nonlocal position
            if n < 0 or position + n > size:
                raise ValueError(label + '超出文件边界')
            stream.seek(n, 1)
            position += n
        def integer():
            return struct.unpack('<Q', read(8))[0]
        def string(keep=True):
            length = integer()
            if keep:
                return read(length).decode('utf-8', errors='replace')
            skip(length, 'GGUF 字符串')
        scalar = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        def value(kind, keep):
            if kind in scalar:
                fmt = '<' + scalar[kind]
                val = struct.unpack(fmt, read(struct.calcsize(fmt)))[0]
                return val if keep else None
            if kind == 8:
                return string(keep)
            if kind == 9:
                item_kind = struct.unpack('<I', read(4))[0]
                count = integer()
                if item_kind in scalar:
                    length = struct.calcsize('<'+scalar[item_kind]) * count
                    skip(length, 'GGUF 数组')
                else:
                    for _ in range(count):
                        value(item_kind, False)
                return None
            raise ValueError(f'未知 GGUF 元数据类型 {kind}')
        if read(4) != b'GGUF':
            raise ValueError('文件不是 GGUF 模型')
        version = struct.unpack('<I', read(4))[0]
        if version not in (2, 3):
            raise ValueError(f'不支持的 GGUF 格式版本 {version}')
        tensor_count, count = integer(), integer()
        metadata = {'gguf_version': version, 'tensor_count': tensor_count}
        for _ in range(count):
            key = string()
            kind = struct.unpack('<I', read(4))[0]
            keep = key.startswith(('general.', 'split.')) or key.endswith(('.context_length', '.block_count', '.embedding_length', '.expert_count', '.nextn_predict_layers', '.mtp_layers'))
            item = value(kind, keep)
            if keep and item is not None:
                metadata[key] = item
        if tensor_names:
            names = []
            for _ in range(tensor_count):
                names.append(string())
                dimensions = struct.unpack('<I', read(4))[0]
                read(dimensions * 8 + 4 + 8)  # shape, type, offset; never read tensor payloads
            metadata['tensor_names'] = names
        return metadata


def ninfer_metadata(path: Path):
    """Identify the pinned v3 container, without decoding tensors or model bindings.

    Framing/files follow NInfer 64d10fbf's artifact/framing.py and reader.py.
    The runtime's official inspector remains responsible for full directory semantics.
    """
    def integer(value, label, *, positive=False):
        if type(value) is not int or not (1 if positive else 0) <= value < 1 << 64:
            raise ValueError(f'NInfer {label} 必须为有效 U64 整数')
        return value

    def identifier(value):
        if not isinstance(value, str) or not value or '\0' in value:
            raise ValueError('NInfer 目录包含无效标识符')
        try:
            value.encode('utf-8')
        except UnicodeEncodeError as error:
            raise ValueError('NInfer 目录包含无效 UTF-8 标识符') from error
        return value

    def members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('NInfer 目录 JSON 包含重复成员：' + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('NInfer 目录 JSON 包含非有限数值：' + value)

    size = path.stat().st_size
    with path.open('rb') as stream:
        raw = stream.read(NINFER_HEADER.size)
        if len(raw) != NINFER_HEADER.size:
            raise ValueError('NInfer 入口文件头截断')
        magic, json_bytes, artifact_id = NINFER_HEADER.unpack(raw)
        if magic == NINFER_PART_MAGIC:
            raise ValueError('这是 NInfer 续卷；请选择入口 .ninfer 文件')
        if magic != NINFER_MAGIC:
            raise ValueError('文件不是受支持的 NInfer v3 入口工件')
        if not 0 < json_bytes <= size - NINFER_HEADER.size:
            raise ValueError('NInfer 目录 JSON 长度超出入口文件边界')
        try:
            directory = json.loads(stream.read(json_bytes).decode('utf-8'), object_pairs_hook=members, parse_constant=nonfinite)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError('NInfer 目录 JSON 损坏') from error
    required = {'components', 'objects', 'bindings', 'uses', 'files'}
    if not isinstance(directory, dict) or not required <= directory.keys() or directory.keys() - required - {'metadata', 'provenance'}:
        raise ValueError('NInfer 目录缺少必要容器字段或包含未知字段')
    components, objects, records = (directory[key] for key in ('components', 'objects', 'files'))
    if not isinstance(components, dict) or 'text' not in components or any(
        not isinstance(value, dict) or not isinstance(value.get('config'), dict) for value in components.values()
    ):
        raise ValueError('NInfer 目录缺少 text 组件配置')
    for name in components:
        identifier(name)
    if (not isinstance(objects, list) or not objects or not isinstance(records, list) or not records
            or not isinstance(directory['bindings'], dict) or not isinstance(directory['uses'], list)
            or not isinstance(directory.get('metadata', {}), dict) or not isinstance(directory.get('provenance', {}), dict)):
        raise ValueError('NInfer 目录容器字段类型无效')
    files, payload_bytes, names = [], 0, {path.name}
    payload_offset = (NINFER_HEADER.size + json_bytes + NINFER_ALIGNMENT - 1) // NINFER_ALIGNMENT * NINFER_ALIGNMENT
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {'path', 'payload_bytes'}:
            raise ValueError('NInfer files 记录无效')
        length = integer(record['payload_bytes'], 'payload_bytes', positive=True)
        payload_bytes = integer(payload_bytes + length, '总 payload 长度', positive=True)
        name = record['path']
        if index == 0:
            if name is not None:
                raise ValueError('NInfer files[0].path 必须为 null')
            selected = path
        else:
            identifier(name)
            if name in {'.', '..'} or any(character in name for character in '/\\') or name in names or Path(name).name != name:
                raise ValueError('NInfer 续卷必须为不重复的同目录文件名')
            names.add(name)
            selected = path.parent / name
        expected = integer((payload_offset if index == 0 else NINFER_ALIGNMENT) + length, '文件长度', positive=True)
        if not selected.is_file():
            raise ValueError('NInfer 缺少续卷：' + selected.name)
        if selected.stat().st_size != expected:
            raise ValueError(f'NInfer 文件长度不匹配：{selected.name}（应为 {expected} B）')
        if index:
            with selected.open('rb') as stream:
                part = stream.read(NINFER_HEADER.size)
            if len(part) != NINFER_HEADER.size or NINFER_HEADER.unpack(part) != (NINFER_PART_MAGIC, index, artifact_id):
                raise ValueError('NInfer 续卷身份或编号不匹配：' + selected.name)
        files.append(str(selected))
    previous_end, object_ids, formats = 0, set(), set()
    for item in objects:
        if not isinstance(item, dict) or item.get('kind') not in {'tensor', 'resource'}:
            raise ValueError('NInfer 对象描述无效')
        name = identifier(item.get('id'))
        offset = integer(item.get('offset'), '对象 offset')
        length = integer(item.get('bytes'), '对象 bytes', positive=True)
        end = integer(offset + length, '对象范围末尾', positive=True)
        if name in object_ids or offset < previous_end or end > payload_bytes:
            raise ValueError('NInfer 对象范围重叠、重复或超出 payload')
        object_ids.add(name)
        previous_end = end
        if item['kind'] == 'tensor':
            formats.add(identifier(item.get('format')))
            identifier(item.get('layout'))
            shape = item.get('shape')
            if not isinstance(shape, list) or len(shape) > 16:
                raise ValueError('NInfer tensor shape 必须为 rank 0–16 的数组')
            for dimension in shape:
                integer(dimension, 'tensor shape', positive=True)
    if not formats:
        raise ValueError('NInfer 工件没有模型张量')
    config = components['text']['config']
    architectures = config.get('architectures')
    architecture = config.get('model_type') or (architectures[0] if isinstance(architectures, list) and architectures else None)
    metadata = {'ninfer_version': 3, 'ninfer_artifact_id': artifact_id.hex(), 'ninfer_validation': 'container',
                'ninfer_components': list(components), 'ninfer_tensor_formats': sorted(formats)}
    return metadata, files, architecture if isinstance(architecture, str) else None


def inspect_custom_asset(path: str, name=None, location=None) -> ModelAsset:
    """Register one explicitly selected local path without guessing its format."""
    if location is not None:
        from .models import ModelLocation
        location = ModelLocation.model_validate(location)
    if location and location.platform == 'wsl2':
        from .platforms import inspect_wsl_asset
        return inspect_wsl_asset('inspect_custom_asset', path, location, name=name)
    if not isinstance(path, str) or not path.strip() or any(character in path for character in ('\0', '\r', '\n')):
        raise ValueError('请选择存在的本机模型文件或目录')
    if path.startswith(('\\\\', '//')):
        raise ValueError('自定义模型必须是本机路径，不支持网络或 WSL 路径')
    selected = Path(path).expanduser().resolve(strict=True)
    if str(selected).startswith(('\\\\', '//')) or not (selected.is_file() or selected.is_dir()):
        raise ValueError('自定义模型必须是存在的本机文件或目录')
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError('模型名称不能为空')
    is_file = selected.is_file()
    return ModelAsset(id=uuid.uuid4().hex, name=name or (selected.stem if is_file else selected.name), path=str(selected), format='custom',
                      size_bytes=selected.stat().st_size if is_file else 0, files=[str(selected)] if is_file else [],
                      metadata={'custom_model': True, 'size_bytes_known': is_file}, managed=False,
                      warnings=['自定义格式与模型文件完整性由所选引擎确认。' + ('' if is_file else ' 目录未扫描，大小未统计。')])


def inspect_asset(path: str, name=None, source=None, managed=False, location=None) -> ModelAsset:
    if location is not None:
        from .models import ModelLocation
        location = ModelLocation.model_validate(location)
    if location and location.platform == 'wsl2':
        from .platforms import inspect_wsl_asset
        return inspect_wsl_asset('inspect_asset', path, location, name=name)
    selected = Path(path).expanduser().resolve(strict=True)
    warnings = []
    files = []
    modalities = ['text']
    if selected.is_dir():
        config_path = selected / 'config.json'
        if config_path.exists() and not any(selected.glob('*.safetensors')):
            # A native artifact embeds its config; adjacent HF metadata does not
            # turn a native-only export into an incomplete HF model.
            native = [p for p in selected.iterdir() if p.is_file() and p.suffix.lower() == '.ninfer' and not _ninfer_continuation(p)]
            if len(native) == 1:
                return inspect_asset(str(native[0]), name, source, managed)
            if native:
                raise ValueError('目录中有多个 NInfer 工件；请直接选择所需入口文件')
        if not config_path.exists():
            weights = sorted(p for p in selected.iterdir() if p.is_file() and p.suffix.lower() in {'.gguf', '.ninfer'})
            weights = [p for p in weights if p.suffix.lower() != '.gguf' or 'mmproj' not in p.name.lower()]
            weights = [p for p in weights if p.suffix.lower() != '.ninfer' or not _ninfer_continuation(p)]
            primary = [p for p in weights if not GGUF_SHARD.match(p.name) or '-00001-of-' in p.name]
            if len(primary) == 1:
                return inspect_asset(str(primary[0]), name, source, managed)
            raise ValueError('目录中没有 config.json；请直接选择所需 GGUF / NInfer 主文件')
        metadata = json.loads(config_path.read_text('utf-8'))
        weights = sorted(selected.glob('*.safetensors'))
        if not weights:
            raise ValueError('HF 模型目录缺少 safetensors 权重')
        shard_groups = {}
        for weight in weights:
            match = re.fullmatch(r'(.+)-(\d+)-of-(\d+)\.safetensors', weight.name, re.I)
            if match:
                prefix, number, count = match.groups()
                shard_groups.setdefault((prefix, int(count)), []).append(int(number))
        for (prefix, count), numbers in shard_groups.items():
            if count < 1 or len(numbers) != count or len(set(numbers)) != count or any(number < 1 or number > count for number in numbers):
                raise ValueError(f'HF 模型权重分片不完整：{prefix}（找到 {len(numbers)} / {count} 片）')
        for index in selected.glob('*.safetensors.index.json'):
            weight_map = json.loads(index.read_text('utf-8')).get('weight_map', {})
            for relative in set(weight_map.values()):
                target = (selected / relative).resolve()
                if not target.is_relative_to(selected) or not target.is_file():
                    raise ValueError(f'缺少或无效的模型分片：{relative}')
        files = _hf_files(selected)
        architecture = metadata.get('model_type') or next(iter(metadata.get('architectures', [])), None)
        quantization = (metadata.get('quantization_config') or {}).get('quant_method') or metadata.get('torch_dtype') or metadata.get('dtype')
        if 'vision_config' in metadata or any(x in str(architecture).lower() for x in ('vl', 'vision', 'llava', 'idefics')):
            modalities.append('image')
        if not any((selected / file).exists() for file in ('tokenizer.json', 'tokenizer.model', 'vocab.json')):
            warnings.append('未检测到常见分词器文件，加载前需确认该架构所需文件齐全')
        fmt = 'safetensors'
    elif selected.suffix.lower() == '.gguf':
        match = GGUF_SHARD.match(selected.name)
        if match:
            prefix, shard, count, extension = match.groups()
            if not 1 <= int(shard) <= int(count):
                raise ValueError('GGUF 分片编号或总数无效')
            weights = [selected.parent / f'{prefix}-{i:05d}-of-{count}{extension}' for i in range(1, int(count)+1)]
            missing = [p.name for p in weights if not p.is_file()]
            if missing:
                raise ValueError('GGUF 缺少分片：' + ', '.join(missing))
            selected = weights[0]
            files = [str(p) for p in weights]
        else:
            files = [str(selected)]
        metadata = gguf_metadata(selected)
        architecture = metadata.get('general.architecture')
        type_id = metadata.get('general.file_type')
        standard = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',7:'Q8_0',8:'Q5_0',9:'Q5_1',10:'Q2_K',11:'Q3_K_S',12:'Q3_K_M',13:'Q3_K_L',14:'Q4_K_S',15:'Q4_K_M',16:'Q5_K_S',17:'Q5_K_M',18:'Q6_K'}
        quantization = standard.get(type_id, f'GGUF 类型 {type_id}（待识别）' if type_id is not None else None)
        # Some engines store mixed tensor formats; a single aggregate file_type
        # is insufficient. Prefer the explicit conversion metadata when present.
        description = metadata.get('general.description','')
        if 'NVFP4' in metadata.get('general.finetune','').upper():
            formats = ['NVFP4'] + [q for q in ('Q8_0','BF16') if q in description]
            quantization = ' / '.join(formats)
        fmt = 'gguf'
        mmproj = list(selected.parent.glob('*mmproj*.gguf'))
        if mmproj:
            metadata['mmproj_candidates'] = [str(p) for p in mmproj]
            warnings.append('发现视觉投影文件；在加载配置中选择与模型匹配的文件后启用图片输入')
    elif selected.suffix.lower() == '.ninfer':
        metadata, files, architecture = ninfer_metadata(selected)
        quantization = ' / '.join(metadata['ninfer_tensor_formats'])
        if 'vision' in metadata['ninfer_components']:
            modalities.append('image')
        fmt = 'ninfer'
    else:
        raise ValueError('请选择 GGUF / NInfer 文件或含 config.json 的 HF 模型目录；未知格式请使用自定义模型路径')
    return ModelAsset(id=uuid.uuid4().hex, name=name or (selected.name if selected.is_dir() else selected.stem), path=str(selected), format=fmt, architecture=architecture, quantization=quantization, size_bytes=sum(Path(f).stat().st_size for f in files), files=files, modalities=modalities, metadata=metadata, source=source or {'kind':'local'}, managed=managed, warnings=warnings)


def scan_model_folder(path: str, location=None, progress: Callable[[dict], None] | None = None) -> dict:
    """Inspect models without writing files. Progress counts enumerated directories, with no known total."""
    if location is not None:
        from .models import ModelLocation
        location = ModelLocation.model_validate(location)
    if location and location.platform == 'wsl2':
        from .platforms import inspect_wsl_asset
        if progress:
            progress({'stage': 'scanning', 'message': '正在扫描 WSL 模型目录'})
        result = inspect_wsl_asset('scan_model_folder', path, location)
        if progress:
            progress({'stage': 'scanning', 'message': f'扫描完成 · 发现 {len(result["models"])} 个模型'})
        return result
    if not path or not path.strip():
        raise ValueError('请选择模型根目录')
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except RuntimeError as error:
        raise ValueError(f'模型根目录无效：{error}') from error
    if not root.is_dir():
        raise ValueError('模型根路径必须是目录')

    result = {'models': [], 'errors': [], 'skipped': []}
    pending = [(root, False)]
    seen_directories = set()
    seen_candidates = set()
    seen_models = set()
    inspection_errors = (OSError, ValueError, TypeError, AttributeError, RuntimeError, struct.error)

    def report(current=None):
        if progress:
            message = f'已扫描 {len(seen_directories)} 个目录 · 发现 {len(result["models"])} 个模型'
            if current:
                message += f' · 正在检查 {current}'
            progress({'stage': 'scanning', 'message': message, 'completed': len(seen_directories)})

    def add_model(model: ModelAsset):
        key = _canonical_path(Path(model.path))
        if key not in seen_models:
            model.metadata['import_root'] = str(root)
            candidates = projector_candidates(model)
            if candidates:
                model.metadata['mmproj_candidates'] = candidates
            seen_models.add(key)
            seen_candidates.add(key)
            result['models'].append(model)

    while pending:
        directory, gguf_only = pending.pop()
        report(directory.name)
        try:
            resolved = directory.resolve(strict=True)
            if not resolved.is_relative_to(root):
                result['skipped'].append({'path': str(directory), 'reason': '目录链接指向所选根目录之外'})
                continue
            key = _canonical_path(resolved)
            if key in seen_directories:
                result['skipped'].append({'path': str(directory), 'reason': '目录链接目标已扫描，避免重复或循环'})
                continue
            seen_directories.add(key)
            entries = sorted(resolved.iterdir(), key=lambda item: item.name.lower())
        except inspection_errors as error:
            result['errors'].append({'path': str(directory), 'error': str(error)})
            continue

        if not gguf_only:
            if any(entry.name.lower() == 'config.json' for entry in entries):
                report(resolved.name)
                try:
                    model = inspect_asset(str(resolved))
                    add_model(model)
                    gguf_only = model.format == 'safetensors'
                except inspection_errors as error:
                    result['errors'].append({'path': str(resolved), 'error': str(error)})
            elif any(entry.name.lower().endswith(('.safetensors', '.safetensors.index.json')) for entry in entries):
                result['errors'].append({'path': str(resolved), 'error': 'HF 模型目录缺少 config.json'})

        directories = []
        for entry in entries:
            if entry.suffix.lower() in {'.gguf', '.ninfer'}:
                report(entry.name)
            try:
                if entry.is_dir():
                    if entry.name.lower() not in IGNORED_MODEL_DIRECTORIES:
                        directories.append((entry, gguf_only))
                    continue
                if entry.suffix.lower() not in {'.gguf', '.ninfer'} or not entry.is_file():
                    continue
                selected = entry.resolve(strict=True)
                match = GGUF_SHARD.match(selected.name)
                if match:
                    prefix, _, count, extension = match.groups()
                    candidate_key = os.path.normcase(str(selected.parent / f'{prefix}-00001-of-{count}{extension.lower()}'))
                else:
                    candidate_key = _canonical_path(selected)
                if candidate_key in seen_candidates:
                    continue
                seen_candidates.add(candidate_key)
                if selected.suffix.lower() == '.ninfer' and _ninfer_continuation(selected):
                    result['skipped'].append({'path': str(entry), 'reason': 'NInfer 续卷需随入口工件导入，不能独立导入'})
                    continue
                if selected.suffix.lower() == '.gguf' and 'mmproj' in selected.name.lower():
                    result['skipped'].append({'path': str(entry), 'reason': '视觉投影器需与主模型配套，不能独立导入'})
                    continue
                if selected.suffix.lower() == '.gguf' and selected.name.lower().startswith('mtp'):
                    result['skipped'].append({'path': str(entry), 'reason': '附加权重需与主模型配套，不能独立导入'})
                    continue
                model = inspect_asset(str(selected))
                model_type = str(model.metadata.get('general.type', '')).lower()
                architecture = str(model.architecture or '').lower()
                if model.format == 'gguf' and (model_type in ('projector', 'mmproj') or architecture in ('clip', 'projector', 'mmproj')):
                    result['skipped'].append({'path': model.path, 'reason': '视觉投影器需与主模型配套，不能独立导入'})
                    continue
                if model.format == 'gguf' and model.metadata.get('tensor_count') == 0:
                    result['skipped'].append({'path': model.path, 'reason': '文件没有模型权重（仅词表），不能独立导入'})
                    continue
                if model.format == 'gguf' and model_type == 'adapter':
                    result['skipped'].append({'path': model.path, 'reason': '附加权重需与主模型配套，不能独立导入'})
                    continue
                add_model(model)
            except inspection_errors as error:
                result['errors'].append({'path': str(entry), 'error': str(error)})
        pending.extend(reversed(directories))
    report()
    return result
