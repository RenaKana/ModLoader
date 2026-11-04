"""The same stdlib-only file checks on the host and in a selected WSL runtime."""
import json
from pathlib import Path
import re
import struct


def resource_info(path, *, gdn=False):
    p = Path(path)
    result = dict(path=str(p), exists=p.exists(), is_file=p.is_file(), is_dir=p.is_dir())
    if p.is_file():
        s = p.stat()
        result['stamp'] = dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)
    if not p.is_dir():
        return result
    files = sorted((f for f in p.iterdir() if f.is_file()), key=lambda f: f.name)
    result['files'] = [f.name for f in files]
    result['file_stats'] = [dict(path=str(f), size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns) for f in files]
    config = p / 'config.json'
    if config.is_file():
        result['config'] = json.loads(config.read_text(encoding='utf-8'))
        if not isinstance(result['config'], dict):
            raise ValueError('config.json must be an object')
    missing, referenced = [], set()
    for index in files:
        if not index.name.endswith('.safetensors.index.json'):
            continue
        manifest = json.loads(index.read_text(encoding='utf-8'))
        mapping = manifest.get('weight_map') if isinstance(manifest, dict) else None
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(name, str) or not name for name in mapping.values()):
            raise ValueError('Invalid safetensors weight_map')
        for name in sorted(set(mapping.values())):
            referenced.add(name)
            shard = p / name
            if not shard.resolve().is_relative_to(p.resolve()):
                raise ValueError('Shard path escaped selected model directory')
            if not shard.is_file():
                missing.append(name)
    result['missing_shards'] = missing
    result['referenced_shards'] = sorted(referenced)
    if not gdn:
        return result
    result['gdn_weights'] = {}
    for weights in files:
        if not weights.name.endswith('.safetensors'):
            continue
        with weights.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('Incomplete safetensors header: ' + str(weights))
            length = struct.unpack('<Q', raw)[0]
            if not 0 < length <= weights.stat().st_size - 8:
                raise ValueError('Safetensors header exceeds file bounds: ' + str(weights))
            header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError('Safetensors header must be an object')
        for name, tensor in header.items():
            if not re.search(r'\.linear_attn\.in_proj_(a|b|ba)\.weight$', name):
                continue
            if name in result['gdn_weights']:
                raise ValueError('Duplicate GDN tensor: ' + name)
            if not isinstance(tensor, dict):
                raise ValueError('Invalid GDN tensor metadata: ' + name)
            result['gdn_weights'][name] = dict(dtype=tensor.get('dtype'), shape=tensor.get('shape'))
    return result
