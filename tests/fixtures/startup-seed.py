"""Seed an isolated, non-executable startup-editor UI scenario."""
import json
from pathlib import Path
import struct
import sys

from modloader.assets import gguf_metadata
from modloader.models import AppPaths, LoadProfile, ModelAsset, RuntimeInstallation
from modloader.store import Store


def write_gguf(target, architecture, metadata):
    def string(value):
        data = value.encode('utf-8')
        return struct.pack('<Q', len(data)) + data

    metadata = {'general.architecture': architecture, **metadata}
    content = b'GGUF' + struct.pack('<IQQ', 3, 0, len(metadata))
    for key, value in metadata.items():
        content += string(key) + struct.pack('<I', 8 if isinstance(value, str) else 4)
        content += string(value) if isinstance(value, str) else struct.pack('<I', value)
    target.write_bytes(content)


def seed(directory):
    directory = Path(directory).resolve()
    paths = AppPaths(directory / 'data')
    store = Store(paths.root / 'modloader.sqlite3')
    store.put('settings', {'id': 'main', 'api_port': 0, 'close_to_tray': False,
                           'proxy_mode': 'direct', 'api_history_enabled': False})
    weights = directory / 'fixture model.gguf'
    write_gguf(weights, 'qwen35', {'qwen35.block_count': 32, 'qwen35.embedding_length': 1024,
                                 'qwen35.context_length': 32768, 'general.name': 'Startup fixture'})
    executable = directory / 'llama-server.exe'
    executable.write_text('Startup editor fixture only. This is not an executable.', encoding='utf-8')
    model = ModelAsset(id='startup-model', name='启动参数集成 fixture', path=str(weights), format='gguf',
                       architecture='qwen35', metadata=gguf_metadata(weights), files=[str(weights)],
                       size_bytes=weights.stat().st_size)
    runtime = RuntimeInstallation(id='startup-llama', backend='llama_cpp', name='llama.cpp 参数 fixture',
                                  path=str(executable), version='startup-fixture',
                                  metadata={'supported_flags': ['--ctx-size', '--n-gpu-layers', '--parallel',
                                                               '--threads', '--cache-type-k', '--cache-type-v',
                                                               '--flash-attn', '--batch-size', '--temp', '--top-p',
                                                               '--top-k', '--seed', '--future-option']})
    profiles = [LoadProfile(id='startup-llama-profile', model_id=model.id, runtime_id=runtime.id,
                            name='llama 启动参数', alias='startup-llama', backend='llama_cpp')]
    strata_weights = directory / 'fixture strata.gguf'
    write_gguf(strata_weights, 'qwen4exp', {'qwen4exp.block_count': 48, 'qwen4exp.expert_count': 512,
                                          'qwen4exp.embedding_length': 2560})
    strata_model = ModelAsset(id='startup-strata-model', name='Strata 启动参数 fixture', path=str(strata_weights),
                             format='gguf', architecture='qwen4exp', metadata=gguf_metadata(strata_weights),
                             files=[str(strata_weights)], size_bytes=strata_weights.stat().st_size)
    strata_root = directory / 'strata'
    (strata_root / 'engine').mkdir(parents=True)
    (strata_root / 'engine' / 'strata.exe').write_text('Non-executable UI fixture.', encoding='utf-8')
    (strata_root / 'serve').mkdir()
    (strata_root / 'serve' / 'server.py').write_text('raise RuntimeError("UI fixture must never execute")\n', encoding='utf-8')
    expert_profile = strata_root / 'data' / 'expert-profile.bin'
    expert_profile.parent.mkdir()
    expert_profile.write_bytes(b'STRP' + struct.pack('<5I', 1, 48, 512, 1, 1) + struct.pack('<HH', 0, 0))
    pack = directory / 'strata-pack'
    pack.mkdir()
    (pack / 'dense.bin').write_bytes(b'dense')
    (pack / 'index.txt').write_text('# strata pack index v3\nweight ' + ' '.join(['0'] * 18) + '\n', encoding='utf-8')
    (pack / 'conversions.json').write_text('{"schema":1,"tensors":[]}', encoding='utf-8')
    (pack / 'native_experts.txt').write_text('# strata native experts v3\n0 17 20 0 10 0 1 2\n', encoding='utf-8')
    tokenizer = pack / 'tokenizer'
    tokenizer.mkdir()
    for name, value in [('vocab.json', {'test': 0}), ('token_type.json', [1]), ('tokenizer.json', {'model': 'gpt2'})]:
        (tokenizer / name).write_text(json.dumps(value), encoding='utf-8')
    (tokenizer / 'merges.txt').write_text('', encoding='utf-8')
    strata_runtime = RuntimeInstallation(id='startup-strata', backend='strata', name='Strata 参数 fixture',
                                         path=str(strata_root), version='startup-fixture',
                                         metadata={'python': str(directory / 'never-execute-python.exe'),
                                                   'supported_flags': ['--batch-size', '--dws', '--prefill']})
    profiles.append(LoadProfile(id='startup-strata-profile', model_id=strata_model.id, runtime_id=strata_runtime.id,
                                 name='Strata 启动参数', alias='startup-strata', backend='strata',
                                 parameters={'context_length': 8192, 'kv_cache': 'int8', 'strata_config': {
                                     'cwd': str(directory), 'exe': str(strata_root / 'engine' / 'strata.exe'),
                                     'tokenizer': str(tokenizer), 'args': ['--native', str(strata_weights), '--pack', str(pack),
                                                                          '--max-context', '8192', '--kv', 'int8']}}))
    for kind, items in [('models', [model, strata_model]), ('runtimes', [runtime, strata_runtime]), ('profiles', profiles)]:
        for item in items:
            store.put(kind, item.model_dump())
    store.close()
    return {'model_id': model.id, 'profile_id': profiles[0].id, 'runtime_id': runtime.id,
            'model_path': model.path, 'runtime_path': runtime.path, 'data_dir': str(paths.root),
            'strata_model_id': strata_model.id, 'strata_profile_id': profiles[1].id,
            'strata_executable': str(strata_root / 'engine' / 'strata.exe')}


if __name__ == '__main__':
    print(json.dumps(seed(sys.argv[1]), ensure_ascii=False))
