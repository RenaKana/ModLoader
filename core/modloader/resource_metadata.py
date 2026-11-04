"""Small, standard-library-only weight descriptors for memory previews.

This file is also evaluated by the selected WSL distribution. Readers seek over
metadata they do not need and never read tensor payloads or import an engine.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import struct
from threading import RLock


_GGUF_LOCK = RLock()


def _json(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(str(path) + ' must contain a JSON object')
    return value


def _hf(path):
    root = path if path.is_dir() else path.parent
    result = dict(format='safetensors', config={}, tensors=[], errors=[])
    try:
        result['config'] = _json(root / 'config.json')
    except (OSError, ValueError) as error:
        result['errors'].append('config.json: ' + str(error))
    indices = sorted(root.glob('*.safetensors.index.json'))
    preferred = root / 'model.safetensors.index.json'
    if preferred in indices:
        indices = [preferred]
    if indices:
        names = set()
        for index in indices:
            mapping = _json(index).get('weight_map')
            if not isinstance(mapping, dict) or not mapping or any(not isinstance(v, str) for v in mapping.values()):
                raise ValueError(str(index) + ': invalid weight_map')
            names.update(mapping.values())
        files = sorted(root / name for name in names)
    else:
        files = sorted(root.glob('*.safetensors'))
    if not files:
        raise ValueError('没有 safetensors 张量元数据：' + str(root))
    seen = set()
    for file in files:
        try:
            if not file.resolve().is_relative_to(root.resolve()):
                raise ValueError('Shard path escaped selected model directory')
            size = file.stat().st_size
            with file.open('rb') as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError('Incomplete safetensors header')
                length = struct.unpack('<Q', raw)[0]
                if not 0 < length <= size - 8:
                    raise ValueError('Safetensors header exceeds file bounds')
                header = json.loads(stream.read(length))
            if not isinstance(header, dict):
                raise ValueError('Safetensors header must be an object')
            tensors = []
            for name, tensor in header.items():
                if name == '__metadata__':
                    continue
                if name in seen:
                    raise ValueError('Duplicate safetensors tensor: ' + name)
                shape, offsets = tensor['shape'], tensor['data_offsets']
                if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                        or not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(n) is not int for n in offsets)
                        or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length):
                    raise ValueError('Invalid safetensors tensor bounds: ' + name)
                tensors.append(dict(name=name, shape=shape, dtype=tensor['dtype'],
                                    bytes=offsets[1] - offsets[0]))
            seen.update(t['name'] for t in tensors)
            result['tensors'].extend(tensors)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors'].append(file.name + ': ' + str(error))
    return result


def _ninfer(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        raw = stream.read(32)
        if len(raw) != 32:
            raise ValueError('NInfer 入口文件头截断')
        magic, length, _ = struct.unpack('<8sQ16s', raw)
        if magic != b'NINFER\0\3' or not 0 < length <= size - 32:
            raise ValueError('文件不是有效 NInfer v3 入口工件')
        directory = json.loads(stream.read(length))
    # Full import/preflight validation remains with assets.py / the official
    # inspector. Here only trust descriptor ranges needed by the estimate.
    payload = sum(item['payload_bytes'] for item in directory['files'])
    objects = directory['objects']
    ids = set()
    previous = 0
    for item in objects:
        if (item['id'] in ids or type(item['bytes']) is not int or item['bytes'] <= 0
                or type(item['offset']) is not int or item['offset'] < previous
                or item['offset'] + item['bytes'] > payload):
            raise ValueError('NInfer 对象范围重叠、重复或超出 payload')
        ids.add(item['id'])
        previous = item['offset'] + item['bytes']
    errors = []
    offset = (32 + length + 4095) // 4096 * 4096
    for i, item in enumerate(directory['files']):
        name = item['path']
        if i and (not isinstance(name, str) or name in {'.', '..'} or '/' in name or '\\' in name):
            raise ValueError('NInfer 续卷必须为同目录文件名')
        file = path.parent / name if i else path
        if not file.is_file() or file.stat().st_size != (4096 if i else offset) + item['payload_bytes']:
            errors.append('NInfer 缺少或截断的分卷：' + str(file))
    return dict(format='ninfer', config=directory['components']['text']['config'],
                directory=directory, errors=errors)


def _gguf(path):
    """Read GGUF scalars and tensor shapes, not an approximation from file size."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        position = 0
        def read(n):
            nonlocal position
            if n < 0 or position + n > size:
                raise ValueError('GGUF 文件截断或元数据损坏')
            data = stream.read(n)
            if len(data) != n:
                raise ValueError('GGUF 元数据读取不完整')
            position += n
            return data

        def skip(n):
            nonlocal position
            if n < 0 or position + n > size:
                raise ValueError('GGUF 元数据超出文件边界')
            stream.seek(n, 1)
            position += n

        def unpack(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize('<' + fmt)))[0]

        def string(keep=True):
            n = unpack('Q')
            if keep:
                return read(n).decode('utf-8')
            skip(n)

        scalar = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

        def value(kind, keep):
            if kind in scalar:
                n = unpack(scalar[kind])
                return n if keep else None
            if kind == 8:
                return string(keep)
            if kind == 9:
                element, n = unpack('I'), unpack('Q')
                if not keep and element in scalar:
                    length = struct.calcsize('<' + scalar[element]) * n
                    skip(length)
                    return None
                if keep:
                    return [value(element, True) for _ in range(n)]
                for _ in range(n):
                    value(element, False)
                return None
            raise ValueError('未知 GGUF 元数据类型 ' + str(kind))

        if read(4) != b'GGUF' or unpack('I') not in (2, 3):
            raise ValueError('文件不是受支持的 GGUF')
        count, entries = unpack('Q'), unpack('Q')
        config = {}
        for _ in range(entries):
            name = string()
            v = value(unpack('I'), not name.startswith('tokenizer.'))
            if v is not None:
                config[name] = v
        tensors = []
        for _ in range(count):
            name, rank = string(), unpack('I')
            shape = [unpack('Q') for _ in range(rank)]
            dtype, offset = unpack('I'), unpack('Q')
            # Only the byte sizes defined for these formats are interpreted.
            block = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144),
                     14: (256, 210), 30: (1, 2)}.get(dtype)
            elements = math.prod(shape)
            length = elements // block[0] * block[1] if block and elements % block[0] == 0 else None
            tensors.append(dict(name=name, shape=shape, dtype=dtype, bytes=length, offset=offset))
        alignment = config.get('general.alignment', 32)
        if type(alignment) is not int or alignment <= 0:
            raise ValueError('无效 GGUF 对齐')
        start = (position + alignment - 1) // alignment * alignment
        for tensor in tensors:
            if tensor['bytes'] is not None and start + tensor['offset'] + tensor['bytes'] > size:
                raise ValueError('GGUF 张量超出文件边界：' + tensor['name'])
    return dict(format='gguf', config=config, tensors=tensors, errors=[])


@lru_cache(maxsize=32)
def _cached_gguf(path, stamp):
    return _gguf(Path(path))


def read_metadata(path):
    path = Path(path)
    if path.suffix.lower() == '.ninfer':
        return _ninfer(path)
    if path.suffix.lower() == '.gguf':
        path = path.resolve()
        info = path.stat()
        stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
        # Coalesce concurrent preview misses; failed reads never enter the LRU.
        with _GGUF_LOCK:
            metadata = _cached_gguf(str(path), stamp)
        # Strata annotates tensor dictionaries after reading them.
        return deepcopy(metadata)
    return _hf(path)
