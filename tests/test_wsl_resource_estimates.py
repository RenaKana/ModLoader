"""Header-only regressions for Windows files and native WSL model locations."""
import asyncio
import builtins
import contextlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time

import pytest

from modloader.models import LaunchContext, LoadProfile, ModelAsset, ModelLocation, RuntimeInstallation
from modloader.resource_estimates import ResourceEstimator
from modloader.resource_metadata import read_metadata
from modloader import wsl_resource_estimates as estimates
from modloader.wsl_resource_estimates import cuda_devices_for as read_cuda_devices_for
from modloader import resource_metadata
from test_ninfer_assets import write_ninfer


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Memory previews must not start processes or import an inference engine')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    monkeypatch.setattr(estimates, 'cuda_devices_for', lambda profile, runtime: [])
    original_import = builtins.__import__
    def import_metadata(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split('.')[0] in {'torch', 'transformers', 'vllm', 'sglang', 'ninfer'}:
            forbidden()
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', import_metadata)


# Dimensions recorded from the actual Qwen3.8-27B NInfer failure; tensor data
# below are tiny synthetic fixtures, not a model or an engine execution.
CONFIG = dict(model_type='qwen3_5_text', architectures=['Qwen3_5ForCausalLM'], hidden_size=5120,
              num_hidden_layers=64, num_attention_heads=24, num_key_value_heads=4, head_dim=256,
              max_position_embeddings=262144, layer_types=['linear_attention'] * 3 + ['full_attention'])
CONFIG['layer_types'] *= 16
CONFIG.update(linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=48,
              linear_value_head_dim=128, linear_conv_kernel_dim=4)


def inputs(path, backend='ninfer', *, parameters=None):
    model = ModelAsset(id='model', name='header fixture', path=str(path), format=path.suffix[1:] or 'safetensors')
    runtime = RuntimeInstallation(id='runtime', backend=backend, name='WSL fixture', platform='wsl2',
                                  path='/opt/runtime/never-launch', metadata={'distribution': 'ModLoader'})
    profile = LoadProfile(id='profile', model_id='model', backend=backend, runtime_id='runtime', name='draft',
                          alias='never-serve', parameters=parameters or {})
    return model, profile, runtime


def artifact(tmp_path, *, architecture=None, bindings=None):
    objects = [('base', 2048), ('mtp', 1024), ('vision', 512), ('experts', 4096), ('shared', 256),
               ('ple', 8192), ('front', 32)]

    def change(d):
        config = {**CONFIG, **(architecture or {})}
        d['components'] = {'text': {'config': config, 'resources': {'tokenizer.json': 'front'}},
                           'mtp': {'config': {}}, 'vision': {'config': {}}}
        d['objects'] = []
        offset = 0
        for name, size in objects:
            d['objects'].append(dict(id=name, kind='resource' if name == 'front' else 'tensor',
                                     shape=[size // 2], format='bf16', offset=offset, bytes=size))
            offset += size
        d['bindings'] = bindings or {'text/token_embedding': {'object': 'base'},
                                      'text/output_head': {'object': 'base'},
                                      'mtp/projection': {'object': 'mtp'},
                                      'vision/patch': {'object': 'vision'}}
    path, _, directory = write_ninfer(tmp_path / 'model.ninfer', lengths=(sum(size for _, size in objects),), change=change)
    return path, directory


def safetensors(root, config=None, tensors=None):
    root.mkdir(exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config if config is not None else CONFIG))
    tensors = tensors or {'model.layers.0.weight': ('BF16', [8, 16])}
    descriptor, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * {'BF16': 2, 'F32': 4, 'F8_E4M3': 1, 'U8': 1}[dtype]
        descriptor[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + size])
        offset += size
    raw = json.dumps(descriptor).encode()
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(offset))
    return root


def part(resource, prefix):
    return next(p for p in resource['components'] if p['label'].startswith(prefix))


@pytest.mark.asyncio
@pytest.mark.parametrize('dtype,width', [('bf16', 1024), ('int8', 528), ('fp8', 516), ('nvfp4', 288), ('k8v4', 402)])
async def test_windows_ninfer_wsl_runtime_real_failure_dimensions_and_kv_scaling(tmp_path, monkeypatch, dtype, width):
    path, _ = artifact(tmp_path)
    model, profile, runtime = inputs(path, parameters=dict(context_length=128000, max_concurrency=1,
                                   kv_capacity=128000, kv_dtype=dtype, acceleration_mode='mtp', draft_tokens=3))
    async def forbidden(*args, **kwargs):
        pytest.fail('Metadata estimates must not start an engine/native fit process')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    gpu, ram = result['resources']
    assert result['source'] == 'ninfer-metadata' and gpu['quality'] == 'partial'
    assert part(gpu, 'GPU 权重')['bytes'] == 3072  # shared base parent counted once; MTP selected.
    assert part(gpu, 'KV 共享池')['bytes'] == (128000 * 16 + 128064) * 4 * width
    assert part(gpu, 'GDN')['bytes'] == 307908608
    assert gpu['bytes'] == 3072 + (128000 * 16 + 128064) * 4 * width + 307908608
    assert ram['bytes'] == 32 and ram['logical'] is False
    assert part(gpu, '工作区')['bytes'] is None
    assert gpu['bytes'] != path.stat().st_size


@pytest.mark.asyncio
async def test_ninfer_options_override_mtp_selection_and_auto_shared_capacity(tmp_path):
    path, _ = artifact(tmp_path)
    model, profile, runtime = inputs(path, parameters=dict(context_length=128000, kv_dtype='bf16',
                                                          max_concurrency=4, acceleration_mode='mtp'))
    profile.parameters['acceleration_mode'] = 'none'
    profile.extra_args = ['--max-context', '65', '--kv-dtype', 'nvfp4', '--device-state-slots', '0']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert part(gpu, 'GPU 权重')['bytes'] == 2048
    assert part(gpu, 'KV 最小')['bytes'] == 256 * 16 * 4 * 288  # max(ceil(65/64), concurrency) pages.
    assert part(gpu, 'GDN')['bytes'] == 307908608 * 2  # 4 active + 0 cache slots; default actual fixture has 2.
    profile.extra_args = []
    profile.parameters.update(kv_capacity=0, max_concurrency=0)
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert part(gpu, 'KV')['bytes'] is None and part(gpu, 'GDN')['bytes'] is None


@pytest.mark.asyncio
async def test_ninfer_new_runtime_counts_explicit_byte_pool_once_without_legacy_state(tmp_path):
    path, _ = artifact(tmp_path)
    model, profile, runtime = inputs(path, parameters=dict(context_length=128000, context_capacity='1.5GiB',
                                   max_concurrency=4, acceleration_mode='mtp', draft_tokens=3))
    runtime.metadata.update(source_commit='a754fcf3bf524494aa927c2f54e0cbfb58f569ab',
                            version_basis='verified_source_commit')
    result = await ResourceEstimator().estimate(model, profile, runtime)
    gpu = result['resources'][0]
    assert part(gpu, 'KV / State 共享池')['bytes'] == 1536 * 1024**2
    assert not any(item['label'].startswith(('GDN', 'KV 最小')) for item in gpu['components'])
    assert gpu['bytes'] == 3072 + 1536 * 1024**2


@pytest.mark.asyncio
async def test_ninfer_new_auto_pool_is_unknown_and_extra_byte_capacity_is_respected(tmp_path):
    path, _ = artifact(tmp_path)
    model, profile, runtime = inputs(path, parameters={'context_length': 128000, 'max_concurrency': 4})
    runtime.metadata['supported_flags'] = ['--context-capacity']
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert part(result['resources'][0], 'KV / State 共享池')['bytes'] is None
    profile.extra_args = ['--context-capacity=2GiB']
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert part(result['resources'][0], 'KV / State 共享池')['bytes'] == 2 * 1024**3
    profile.parameters['kv_capacity'] = 128000
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert result['resources'] == [] and result['source'] == 'ninfer-parameters'
    assert 'Token' in result['notes'][0] and 'context_capacity' in result['notes'][0]


@pytest.mark.asyncio
@pytest.mark.parametrize('capacity,expected', [('0B', 0), ('2GiB', 2 * 1024**3)])
async def test_ninfer_new_explicit_host_pool_uses_byte_budget_and_keeps_disabled_zero(tmp_path, capacity, expected):
    path, _ = artifact(tmp_path)
    model, profile, runtime = inputs(path)
    runtime.metadata['supported_flags'] = ['--context-capacity', '--host-cache-capacity']
    profile.extra_args = ['--host-cache-capacity', capacity]
    ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][1]
    assert part(ram, 'Host KV / State 共享池')['bytes'] == expected
    assert ram['bytes'] == 32 + expected


@pytest.mark.asyncio
async def test_ninfer_qwen4_main_experts_host_shared_experts_gpu_ple_disk_mtp_gpu(tmp_path):
    path, _ = artifact(tmp_path, architecture=dict(model_type='qwen4_exp', architectures=['Qwen4ExpForCausalLM']),
                       bindings={'text/embedding': {'object': 'base'},
                                 'text/layers/0/moe/experts/gate': {'object': 'experts'},
                                 'text/layers/0/moe/experts/up': {'object': 'experts'},
                                 'text/layers/0/moe/shared/gate': {'object': 'shared'},
                                 'text/layers/0/ple/table': {'object': 'ple'},
                                 'mtp/layers/0/moe/experts/gate': {'object': 'mtp'}})
    model, profile, runtime = inputs(path, parameters={'acceleration_mode': 'mtp'})
    gpu, ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert part(gpu, 'GPU 权重')['bytes'] == 2048 + 256 + 1024
    assert part(ram, 'Host 常驻')['bytes'] == 4096
    assert ram['bytes'] == 4096 + 32
    assert part(gpu, 'KV')['bytes'] is None  # Qwen4 QSA is not ordinary Qwen3.5 KV.


@pytest.mark.asyncio
async def test_ninfer_moe_unknown_expert_placement_keeps_known_allocations(tmp_path):
    path, _ = artifact(tmp_path, architecture=dict(model_type='qwen3_5_moe', architectures=['Qwen3_5MoeForCausalLM']),
                       bindings={'text/embedding': {'object': 'base'},
                                 'text/layers/0/moe/experts/0/gate': {'object': 'experts'},
                                 'text/layers/0/moe/shared/gate': {'object': 'shared'},
                                 'mtp/layers/0/moe/experts/0/gate': {'object': 'mtp'}})
    model, profile, runtime = inputs(path, parameters={'acceleration_mode': 'mtp'})
    gpu, ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert part(gpu, 'GPU 权重')['bytes'] == 2048 + 256 + 1024
    assert part(gpu, '专家')['bytes'] is None and part(ram, '专家')['bytes'] is None
    assert ram['bytes'] == 32


@pytest.mark.asyncio
async def test_native_wsl_root_location_runs_same_stdlib_reader_in_matching_distribution(tmp_path, monkeypatch):
    path, _ = artifact(tmp_path)
    model, profile, runtime = inputs(path)
    model.path = '/root/models/model.ninfer'
    model.location = ModelLocation(platform='wsl2', distribution='ModLoader')
    calls = []
    def remote(distribution, code, args, *, input, timeout):
        calls.append((distribution, args, timeout))
        assert args == ['/root/models/model.ninfer']
        output = io.StringIO()
        with monkeypatch.context() as patch, contextlib.redirect_stdout(output):
            patch.setattr(sys, 'stdin', io.StringIO(input))
            patch.setattr(sys, 'argv', ['metadata', str(path)])
            exec(code, {})
        return output.getvalue()
    monkeypatch.setattr(estimates, 'wsl_python', remote)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert result['resources'][0]['bytes'] > 2048
    assert calls == [('ModLoader', ['/root/models/model.ninfer'], 30)]
    runtime.metadata['distribution'] = 'Other'
    mismatch = await ResourceEstimator().estimate(model, profile, runtime)
    assert not mismatch['resources'] and '同一发行版' in mismatch['notes'][0]
    assert len(calls) == 1


def test_ninfer_reads_only_header_directory_and_file_stats(tmp_path, monkeypatch):
    path, directory = artifact(tmp_path)
    original = Path.open
    reads = []
    class HeaderOnly:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def read(self, size=-1):
            reads.append(size)
            assert size >= 0 and len(reads) <= 2
            return self.stream.read(size)
    monkeypatch.setattr(Path, 'open', lambda self, *a, **kw: HeaderOnly(original(self, *a, **kw)))
    result = read_metadata(str(path))
    assert reads == [32, len(json.dumps(directory, separators=(',', ':')).encode())]
    assert result['config']['hidden_size'] == 5120


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
async def test_hf_weights_dtype_kv_reference_and_budget_not_added_as_allocation(tmp_path, backend):
    path = safetensors(tmp_path / 'hf', tensors={'model.layers.0.weight': ('BF16', [8, 16]),
                       'model.layers.0.weight_scale': ('F32', [8])})
    model, profile, runtime = inputs(path, backend, parameters={'context_length': 128000, 'dtype': 'float32',
                                   'gpu_memory_utilization': .85, 'mem_fraction_static': .7})
    gpu, ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert gpu['bytes'] == 128 * 4 + 8 * 4
    assert gpu['budget_fraction'] == (.85 if backend == 'vllm' else .7)
    assert gpu['budget_basis'] == ('total' if backend == 'vllm' else 'available')
    reference = part(gpu, '单序列 KV')
    assert reference['bytes'] == 128000 * 16 * 4 * 512 * 4 and reference['reference']
    assert part(gpu, '实际 KV')['bytes'] is None
    assert ram['bytes'] is None  # Unestimated is not zero.
    assert 'GPU 总显存' not in str(gpu) if backend == 'sglang' else True


@pytest.mark.asyncio
async def test_hf_unknown_geometry_and_offload_do_not_erase_known_weights(tmp_path):
    path = safetensors(tmp_path / 'hf', config={'torch_dtype': 'bfloat16'})
    model, profile, runtime = inputs(path, 'vllm')
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] == 256 and part(gpu, '单序列 KV')['bytes'] is None
    profile.extra_args = ['--cpu-offload-gb', '4', '--tensor-parallel-size=2']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] is None and part(gpu, '权重')['reference']
    assert gpu['device'] == '多 GPU 合计'


@pytest.mark.asyncio
async def test_raw_qwen35_hf_config_uses_transformers_hybrid_default_not_all_layer_kv(tmp_path):
    config = dict(CONFIG)
    config.pop('layer_types')
    path = safetensors(tmp_path / 'hf', config=config)
    model, profile, runtime = inputs(path, 'sglang', parameters={'context_length': 32768})
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert part(gpu, '单序列 KV')['bytes'] == 32768 * 16 * 4 * 512 * 2


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
async def test_hf_missing_shard_and_bad_config_keep_readable_tensor_quantities(tmp_path, backend):
    path = safetensors(tmp_path / 'hf')
    (path / 'config.json').write_text('{')
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'good': 'model.safetensors', 'missing': 'absent.safetensors'}}))
    model, profile, runtime = inputs(path, backend)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert result['resources'][0]['bytes'] == 256
    assert 'absent.safetensors' in result['diagnostic'] and 'config.json' in result['diagnostic']


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['sglang', 'vllm'])
async def test_unreadable_metadata_keeps_known_gpu_budget_without_fabricating_allocations(tmp_path, backend):
    model, profile, runtime = inputs(tmp_path / 'absent', backend,
                                    parameters={'mem_fraction_static': .7, 'gpu_memory_utilization': .8})
    result = await ResourceEstimator().estimate(model, profile, runtime)
    gpu = result['resources'][0]
    assert gpu['bytes'] is None
    assert gpu['budget_fraction'] == (.7 if backend == 'sglang' else .8)
    assert part(gpu, '权重')['bytes'] is None and result['diagnostic']


@pytest.mark.asyncio
@pytest.mark.parametrize('context,int8', [(2048, False), (4096, True)])
async def test_spark_initial_hybrid_kv_pool_uses_scale_trap_and_table_not_full_layer_bf16(tmp_path, context, int8):
    path = safetensors(tmp_path / 'hf', config={'text_config': CONFIG}, tensors={
        'model.language_model.embed_tokens.weight': ('BF16', [8, 16]),
        'model.language_model.layers.0.mlp.gate_proj.weight': ('U8', [8, 8]),
        'model.language_model.layers.0.self_attn.q_proj.weight': ('F8_E4M3', [8, 16]),
        'mtp.weight': ('BF16', [64, 64]),
    })
    model, profile, runtime = inputs(path, 'sparkinfer', parameters={'context_length': context})
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    blocks = context // 16 + 8
    expected = blocks * 16 * 17 * 4 * (512 * (1 if int8 else 2) + (4 if int8 else 0)) + 256 * blocks * 4
    assert part(gpu, '初始 KV')['bytes'] == expected
    assert part(gpu, '已确认')['bytes'] == 256 + 64  # Refit FP8 attention and unused MTP are not raw GPU copies.
    assert gpu['bytes'] == expected + 320 and part(gpu, '其余')['bytes'] is None
    assert any('扩容' in note for note in gpu['notes'])
    profile.parameters.clear()
    default = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert part(default, '初始 KV')['bytes'] > 0
    assert part(default, '初始 KV')['bytes'] < 262144 * 16 * 4 * 512


@pytest.mark.asyncio
async def test_spark_qwen38_real_shape_targets_snapshot_pools_not_total_ram_or_mapped_weights(tmp_path):
    path = safetensors(tmp_path / 'hf', config={'text_config': CONFIG})
    model, profile, runtime = inputs(path, 'sparkinfer')
    runtime.version = '0.6.34'
    ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][1]
    # qwen35.cpp warm_snapshot_pool: 48 compact FP32 GDN states and 64 BF16
    # convolution states = 154,927,104 B; floor(4096 MiB / size) = 27 buffers.
    assert part(ram, '快照预热池（页锁定）')['bytes'] == 4_183_031_808
    assert part(ram, '快照预热池（可分页）')['bytes'] == 4_183_031_808
    assert ram['bytes'] is None and ram['budget_bytes'] == 8_366_063_616
    assert ram['quality'] == 'dynamic' and ram['logical'] is False
    assert ram['budget_component'] == '快照池预热'
    assert part(ram, '快照预热池（页锁定）')['reference']
    assert part(ram, '快照预热池（可分页）')['reference']
    assert part(ram, '加载峰值')['bytes'] is None
    assert part(ram, '按需前缀')['bytes'] is None
    assert ram['budget_bytes'] != (path / 'model.safetensors').stat().st_size


@pytest.mark.asyncio
@pytest.mark.parametrize('environment,expected', [
    ({'SPARKINFER_PREFIX_CACHE': '0disabled'}, None),
    ({'SPARKINFER_SNAPSHOT_POOL': '0'}, None),
    ({'SPARKINFER_DETERMINISTIC': '1'}, None),
    ({'SPARKINFER_DETERMINISTIC': '1', 'SPARKINFER_PREFIX_CACHE': '1'}, 8_366_063_616),
    ({'SPARKINFER_SNAPSHOT_MIGRATE': '0'}, 4_183_031_808),
    ({'SPARKINFER_SNAPSHOT_POOL_MB': '0'}, None),
    ({'SPARKINFER_SNAPSHOT_POOL_MB': '-1'}, None),
    ({'SPARKINFER_SNAPSHOT_POOL_MB': '147'}, None),
    ({'SPARKINFER_SNAPSHOT_POOL_MB': '148MiB'}, 2 * 154_927_104),
    ({'SPARKINFER_SNAPSHOT_POOL_MB': '100000'}, 2 * 64 * 154_927_104),
])
async def test_spark_snapshot_pool_follows_engine_switches_cap_and_migration(tmp_path, environment, expected):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf', config={'text_config': CONFIG}), 'sparkinfer')
    runtime.version = '0.6.34'
    runtime.metadata['build_environment'] = environment
    ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][1]
    assert ram['bytes'] is None and ram.get('budget_bytes') == expected
    assert all(p['bytes'] is None or p['bytes'] > 0 for p in ram['components'])


@pytest.mark.asyncio
@pytest.mark.parametrize('changes,version', [({}, 'unknown'), ({'linear_key_head_dim': None}, '0.6.34')])
async def test_spark_unknown_snapshot_layout_or_version_does_not_invent_ram(tmp_path, changes, version):
    config = {**CONFIG, **changes}
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf', config={'text_config': config}), 'sparkinfer')
    runtime.version = version
    ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][1]
    assert ram['bytes'] is None and part(ram, '快照预热池')['bytes'] is None
    assert 'budget_bytes' not in ram


@pytest.mark.asyncio
@pytest.mark.parametrize('interval', [0, 1])
async def test_spark_stack_without_linear_layers_does_not_prewarm_snapshots(tmp_path, interval):
    config = {**CONFIG, 'full_attention_interval': interval}
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf', config={'text_config': config}), 'sparkinfer')
    runtime.version = '0.6.34'
    ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][1]
    assert ram['bytes'] is None
    assert not any(p['label'].startswith('快照') for p in ram['components'])


@pytest.mark.asyncio
async def test_sglang_default_gpu_load_keeps_process_peak_and_page_cache_unknown(tmp_path):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), 'sglang')
    runtime.version = '0.5.21'
    ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][1]
    assert ram['bytes'] is None and ram['logical'] is False
    assert ram['components'] == [dict(label='进程与运行时', bytes=None), dict(label='加载峰值与文件缓存', bytes=None)]


@pytest.mark.asyncio
async def test_ninfer_source_can_show_kv_before_artifact_preparation(tmp_path):
    path = safetensors(tmp_path / 'hf')
    model, profile, runtime = inputs(path, parameters={'kv_dtype': 'nvfp4'})
    result = await ResourceEstimator().estimate(model, profile, runtime)
    gpu = result['resources'][0]
    assert part(gpu, '待转换')['bytes'] is None
    assert part(gpu, 'KV 最小')['bytes'] == 8192 * 16 * 4 * 288
    assert gpu['bytes'] > 0


@pytest.mark.asyncio
async def test_spark_gguf_uses_tensor_shapes_skips_mtp_and_hybrid_kv_not_all_65_layers(tmp_path):
    metadata = {'general.architecture': 'qwen35', 'qwen35.block_count': 65,
                'qwen35.nextn_predict_layers': 1, 'qwen35.embedding_length': 8,
                'qwen35.attention.head_count': 24, 'qwen35.attention.head_count_kv': 4,
                'qwen35.attention.key_length': 256, 'qwen35.context_length': 262144}
    tensors = [('token_embd.weight', [8, 16]), ('output_norm.weight', [8]),
               ('blk.0.ssm_in.weight', [8, 8]), ('blk.0.ffn_gate.weight', [8, 8]),
               ('blk.64.attn_norm.weight', [8])]
    string = lambda value: struct.pack('<Q', len(value.encode())) + value.encode()
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for name, value in metadata.items():
        header += string(name) + (struct.pack('<I', 8) + string(value) if isinstance(value, str) else struct.pack('<II', 4, value))
    offset = 0
    for name, shape in tensors:
        header += string(name) + struct.pack('<I', len(shape)) + b''.join(struct.pack('<Q', n) for n in shape)
        header += struct.pack('<IQ', 1, offset)
        offset += math.prod(shape) * 2
    header += bytes((-len(header)) % 32)
    path = tmp_path / 'model.gguf'
    path.write_bytes(header + bytes(offset))
    model, profile, runtime = inputs(path, 'sparkinfer', parameters={'context_length': 4096})
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert part(gpu, '已确认')['bytes'] == 256 + 16
    blocks = 4096 // 16 + 8
    expected_kv = blocks * 16 * 17 * 4 * (512 + 4) + 256 * blocks * 4
    assert part(gpu, '初始 KV')['bytes'] == expected_kv


@pytest.mark.asyncio
async def test_ninfer_blank_kv_capacity_matches_omitted_launch_flag_and_pinned_argv_syntax(tmp_path):
    from modloader.backends.source_builds import PINS
    path, _ = artifact(tmp_path)
    model, profile, runtime = inputs(path, parameters={'context_length': 65, 'kv_capacity': '', 'max_concurrency': 4})
    profile.extra_args = ['--max-context', '129']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert part(gpu, 'KV 共享池')['bytes'] == 192 * 16 * 4 * 1024
    runtime.metadata.update(source_commit=PINS['ninfer']['commit'], version_basis='verified_source_commit')
    profile.extra_args = ['--max-context=129']
    result = await ResourceEstimator().estimate(model, profile, runtime)
    assert result['resources'] == [] and result['source'] == 'ninfer-parameters'
    assert '两个 argv' in result['notes'][0]
    runtime.metadata.clear()  # Unknown external versions do not inherit this parser restriction.
    assert (await ResourceEstimator().estimate(model, profile, runtime))['resources']


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
async def test_hf_last_alias_override_matches_adapter_launch(tmp_path, monkeypatch, backend):
    from modloader.backends import adapters, get_adapters, sglang
    path = safetensors(tmp_path / 'hf')
    model, profile, runtime = inputs(path, backend, parameters={'context_length': 64, 'dtype': 'float32'})
    profile.extra_args = (['-tp', '2', '--max-model-len', '2k', '--dtype', 'float16'] if backend == 'vllm'
                          else ['--tensor-parallel-size', '2', '--context-length=128', '--dtype', 'bfloat16'])
    monkeypatch.setattr(adapters, 'model_target_path', lambda *args: '/mnt/d/hf')
    monkeypatch.setattr(sglang, 'model_target_path', lambda *args: '/mnt/d/hf')
    context = LaunchContext(port=12345, instance_id='fixture', work_dir=str(tmp_path), data_dir=str(tmp_path))
    launch = get_adapters()[backend].build_launch(model, profile, runtime, context)
    assert launch.argv[-len(profile.extra_args):] == profile.extra_args
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['device'] == '多 GPU 合计' and gpu['bytes'] == 256
    assert part(gpu, '单序列 KV')['bytes'] == (2000 if backend == 'vllm' else 128) * 16 * 4 * 512 * 2


@pytest.mark.asyncio
async def test_vllm_budget_default_is_versioned_and_explicit_kv_budget_overrides_fraction(tmp_path):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), 'vllm')
    for version, expected in [('0.31.0', .92), ('0.30.0', None), ('unknown', None)]:
        runtime.version = version
        gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
        assert gpu.get('budget_fraction') == expected and gpu['bytes'] == 256
    profile.parameters['gpu_memory_utilization'] = .7
    profile.extra_args = ['--gpu-memory-utilization=.8', '--kv-cache-memory-bytes', '2G']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert 'budget_fraction' not in gpu and gpu['bytes'] == 256
    assert part(gpu, 'KV 池预算') == dict(label='KV 池预算', bytes=2 * 1024**3, reference=True)
    profile.extra_args = ['--max-model-len=auto']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert part(gpu, '单序列 KV')['bytes'] is None and gpu['budget_fraction'] == .7


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
async def test_hf_quantization_and_model_overrides_are_not_fabricated_allocation_lower_bounds(tmp_path, backend):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), backend)
    profile.extra_args = ['--quantization', 'fp8']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] is None and part(gpu, '权重')['bytes'] == 256
    assert part(gpu, '权重')['reference'] and part(gpu, '单序列 KV')['bytes'] > 0
    flag = '--hf-overrides' if backend == 'vllm' else '--json-model-override-args'
    profile.extra_args = [flag, '{"num_hidden_layers": 8}']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] is None and part(gpu, '权重')['bytes'] == 256
    assert part(gpu, '单序列 KV')['bytes'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
async def test_hf_offload_numeric_zero_cpu_device_and_pipeline_placement(tmp_path, backend):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), backend)
    profile.extra_args = ['--cpu-offload-gb', '0.0']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] == 256 and not part(gpu, '权重').get('reference')
    profile.extra_args = ['--cpu-offload-gb', '.5']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] is None and part(gpu, '权重')['reference']
    profile.extra_args = ['--device', 'cpu']
    resources = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert len(resources) == 1 and resources[0]['kind'] == 'ram' and resources[0]['bytes'] == 256
    assert 'budget_fraction' not in resources[0]
    profile.extra_args = ['--pipeline-parallel-size', '2']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['device'] == '多 GPU 合计' and gpu['bytes'] == 256


@pytest.mark.asyncio
async def test_vllm_mtp_disabled_override_and_tied_output_do_not_double_count_weights(tmp_path):
    path = safetensors(tmp_path / 'hf', config={**CONFIG, 'tie_word_embeddings': True}, tensors={
        'model.embed_tokens.weight': ('BF16', [8, 16]),
        'lm_head.weight': ('BF16', [8, 16]),
        'model.mtp.weight': ('BF16', [8, 16]),
    })
    model, profile, runtime = inputs(path, 'vllm', parameters={'mtp_enabled': False})
    profile.extra_args = ['--speculative-config', '{"method":"mtp"}']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] == 256
    profile.parameters['mtp_enabled'] = True
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] == 512 and part(gpu, 'MTP')['bytes'] is None
    profile.parameters.clear()
    for method in ('eagle', 'mtp'):
        profile.extra_args = ['--speculative-config', json.dumps({'method': method, 'model': '/external/draft'})]
        gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
        assert gpu['bytes'] == 256 and part(gpu, 'MTP')['bytes'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('interval,slots', [(0, 64), (1, 64), (4, 17), (128, 64)])
async def test_spark_only_mixed_hybrid_layer_map_allocates_trap_slice(tmp_path, interval, slots):
    path = safetensors(tmp_path / 'hf', config={'text_config': {**CONFIG, 'full_attention_interval': interval}})
    model, profile, runtime = inputs(path, 'sparkinfer', parameters={'context_length': 2048})
    runtime.metadata['build_environment'] = {'SPARKINFER_KV_INT8': '0disabled', 'SPARKINFER_KV_GROW': '0disabled'}
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    blocks = 2048 // 16 + 8
    assert part(gpu, '初始 KV')['bytes'] == blocks * 16 * slots * 4 * 1024 + 256 * blocks * 4
    assert any('已关闭 KV 扩容' in note for note in gpu['notes'])


def spark_nvfp4_metadata():
    """Real 27B geometry and export layout; descriptors only, no weight payload."""
    config = {**CONFIG, 'full_attention_interval': 4, 'intermediate_size': 17408, 'vocab_size': 248320}
    tensors = []
    def tensor(name, shape, dtype='BF16'):
        tensors.append(dict(name=name, shape=shape, dtype=dtype,
                            bytes=math.prod(shape) * {'BF16': 2, 'U8': 1, 'F8_E4M3': 1, 'F32': 4}[dtype]))
    def projection(prefix, rows, cols):
        tensor(prefix + '.weight', [rows, cols // 2], 'U8')
        tensor(prefix + '.weight_scale', [rows, cols // 16], 'F8_E4M3')
        tensor(prefix + '.weight_scale_2', [], 'F32')
        tensor(prefix + '.input_scale', [], 'F32')  # Not an extra resident GPU tensor.
    tensor('model.language_model.embed_tokens.weight', [248320, 5120])
    tensor('model.language_model.norm.weight', [5120])
    projection('lm_head', 248320, 5120)
    for layer in range(64):
        p = f'model.language_model.layers.{layer}.'
        tensor(p + 'input_layernorm.weight', [5120])
        tensor(p + 'post_attention_layernorm.weight', [5120])
        if (layer + 1) % 4:
            for name, rows, cols in [('in_proj_qkv', 10240, 5120), ('in_proj_z', 6144, 5120), ('out_proj', 5120, 6144)]:
                projection(p + 'linear_attn.' + name, rows, cols)
            for name, shape in [('A_log', [48]), ('dt_bias', [48]), ('norm.weight', [128]),
                                ('conv1d.weight', [10240, 1, 4]), ('in_proj_a.weight', [48, 5120]),
                                ('in_proj_b.weight', [48, 5120])]:
                tensor(p + 'linear_attn.' + name, shape)
        else:
            for name, rows, cols in [('q_proj', 12288, 5120), ('k_proj', 1024, 5120),
                                      ('v_proj', 1024, 5120), ('o_proj', 5120, 6144)]:
                projection(p + 'self_attn.' + name, rows, cols)
            tensor(p + 'self_attn.q_norm.weight', [256])
            tensor(p + 'self_attn.k_norm.weight', [256])
        for name, rows, cols in [('gate_proj', 17408, 5120), ('up_proj', 17408, 5120), ('down_proj', 5120, 17408)]:
            projection(p + 'mlp.' + name, rows, cols)
    tensor('model.visual.patch_embed.proj.weight', [1152, 3, 2, 16, 16])
    tensor('model.visual.patch_embed.proj.bias', [1152])
    tensor('model.visual.pos_embed.weight', [2304, 1152])  # Host table, not GPU residency.
    for layer in range(27):
        for name, shape in [('norm1.weight', [1152]), ('norm1.bias', [1152]),
                            ('norm2.weight', [1152]), ('norm2.bias', [1152]),
                            ('attn.qkv.weight', [3456, 1152]), ('attn.qkv.bias', [3456]),
                            ('attn.proj.weight', [1152, 1152]), ('attn.proj.bias', [1152]),
                            ('mlp.linear_fc1.weight', [4304, 1152]), ('mlp.linear_fc1.bias', [4304]),
                            ('mlp.linear_fc2.weight', [1152, 4304]), ('mlp.linear_fc2.bias', [1152])]:
            tensor(f'model.visual.blocks.{layer}.' + name, shape)
    for name, shape in [('norm.weight', [1152]), ('norm.bias', [1152]),
                        ('linear_fc1.weight', [4608, 4608]), ('linear_fc1.bias', [4608]),
                        ('linear_fc2.weight', [5120, 4608]), ('linear_fc2.bias', [5120])]:
        tensor('model.visual.merger.' + name, shape)
    return dict(format='safetensors', config=dict(text_config=config, vision_config={'depth': 27},
                quantization_config={'quant_method': 'modelopt'}), tensors=tensors, errors=[])


@pytest.mark.asyncio
async def test_spark_real_nvfp4_residency_counts_scales_sfb_head_vision_and_state(tmp_path, monkeypatch):
    metadata = spark_nvfp4_metadata()
    monkeypatch.setattr(estimates, 'metadata_for', lambda *args: metadata)
    calls = []
    def identities(*args):
        calls.append(True)
        return [dict(device='CUDA0', gpu_id='GPU-fixture', compute_capability=[12, 0], memory_total=32607 * 1024**2)]
    monkeypatch.setattr(estimates, 'cuda_devices_for', identities)
    model, profile, runtime = inputs(tmp_path / 'metadata-only', 'sparkinfer', parameters={'context_length': 256000})
    runtime.version = '0.6.34'
    gpu, ram = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert part(gpu, 'BF16')['bytes'] == 2_595_273_728
    assert part(gpu, '原生量化')['bytes'] == 12_163_481_600
    assert part(gpu, '量化缩放')['bytes'] == 1_520_435_200
    assert part(gpu, 'NVFP4 对齐头')['bytes'] == 400 * 256
    assert part(gpu, 'CUTLASS')['bytes'] == 1_520_435_200
    assert part(gpu, 'Q4_K 转换')['bytes'] == 715_161_600  # Head, not a second 64-layer FFN copy.
    assert part(gpu, 'lm_head 原生')['bytes'] == 794_624_256
    assert part(gpu, '视觉常驻')['bytes'] == 916_151_776
    assert part(gpu, 'GDN 常驻')['bytes'] == 154_927_104
    assert part(gpu, 'Q4_K 预填充')['bytes'] == 15_204_352
    assert part(gpu, '解码与采样')['bytes'] == 15_750_958
    assert part(gpu, '初始 KV')['bytes'] == 9_003_411_456
    assert gpu['bytes'] == sum(p['bytes'] for p in gpu['components'] if p['bytes'] is not None)
    assert gpu['bytes'] == 29_414_959_630 and gpu['quality'] == 'estimate'
    assert any('不足以触发 KV 扩容' in note for note in gpu['notes'])
    assert gpu['gpu_id'] == 'GPU-fixture' and calls == [True]  # Same query for identity and SM120 routing.
    assert ram['budget_bytes'] == 8_366_063_616 and ram['bytes'] is None


@pytest.mark.parametrize('environment,removed,added_q4', [
    ({'SPARKINFER_Q38_HEAD_NVFP4': '0'}, 794_624_256, 0),
    ({'SPARKINFER_QWEN38_PREFILL_NVFP4': '0'}, 1_069_547_520 + 79_462_400, 0),
    ({'SPARKINFER_QWEN38_DECODE_NVFP4': '0'}, 0, 9_625_927_680),
    ({'SPARKINFER_QWEN38_DECODE_NVFP4': '0', 'SPARKINFER_QWEN38_PREFILL_NVFP4': '0'},
     9_625_927_680 + 192 * 256 + 1_069_547_520 + 79_462_400, 9_625_927_680),
])
def test_spark_native_decode_aliasing_and_requantization_follow_real_environment(tmp_path, environment, removed, added_q4):
    metadata = spark_nvfp4_metadata()
    _, profile, runtime = inputs(tmp_path / 'metadata-only', 'sparkinfer', parameters={'context_length': 256000})
    runtime.version = '0.6.34'
    devices = [dict(device='CUDA0', compute_capability=[12, 0])]
    baseline = estimates._spark(metadata, profile, runtime, {'--ctx': 256000}, devices)[0]
    runtime.metadata['build_environment'] = environment
    gpu = estimates._spark(metadata, profile, runtime, {'--ctx': 256000}, devices)[0]
    assert gpu['bytes'] == baseline['bytes'] - removed + added_q4
    assert part(gpu, 'Q4_K 转换')['bytes'] == 715_161_600 + added_q4


@pytest.mark.parametrize('capability', [None, [8, 9], [12, 1]])
def test_spark_sfb_requires_selected_sm120_and_unknown_device_keeps_known_weights(tmp_path, capability):
    _, profile, runtime = inputs(tmp_path / 'metadata-only', 'sparkinfer')
    runtime.version = '0.6.34'
    runtime.metadata['build_environment'] = {'SPARKINFER_KV_GROW': '0'}
    gpu = estimates._spark(spark_nvfp4_metadata(), profile, runtime, {},
                           [dict(device='CUDA0', compute_capability=capability)])[0]
    assert part(gpu, 'CUTLASS')['bytes'] == 0
    assert part(gpu, '原生量化')['bytes'] == 12_163_481_600
    assert gpu['quality'] == ('partial' if capability is None else 'estimate')


def test_spark_fp8_and_missing_companion_are_not_counted_as_raw_nvfp4(tmp_path):
    metadata = spark_nvfp4_metadata()
    prefix = 'model.language_model.layers.0.linear_attn.in_proj_qkv'
    for tensor in metadata['tensors']:
        if tensor['name'] == prefix + '.weight':
            tensor.update(dtype='F8_E4M3', shape=[10240, 5120], bytes=10240 * 5120)
        if tensor['name'] == prefix + '.weight_scale':
            tensor.update(dtype='BF16', shape=[10240, 1], bytes=10240 * 2)
    _, profile, runtime = inputs(tmp_path / 'metadata-only', 'sparkinfer')
    runtime.version = '0.6.34'
    runtime.metadata['build_environment'] = {'SPARKINFER_KV_GROW': '0'}
    devices = [dict(device='CUDA0', compute_capability=[12, 0])]
    gpu = estimates._spark(metadata, profile, runtime, {}, devices)[0]
    assert part(gpu, '原生量化')['bytes'] == 12_163_481_600 + 10240 * 5120 // 2
    assert part(gpu, '量化缩放')['bytes'] == 1_520_435_200 - 10240 * 5120 // 16 + 10240 * 2
    assert part(gpu, 'NVFP4 对齐头')['bytes'] == 399 * 256
    assert gpu['quality'] == 'estimate'
    metadata['tensors'] = [t for t in metadata['tensors'] if t['name'] != prefix + '.weight_scale']
    gpu = estimates._spark(metadata, profile, runtime, {}, devices)[0]
    assert gpu['quality'] == 'partial' and part(gpu, '未确认')['bytes'] is None
    assert part(gpu, '初始 KV')['bytes'] > 0 and part(gpu, '原生量化')['bytes'] > 0


@pytest.mark.parametrize('environment,reserve', [({}, 6 * 1024**3),
    ({'SPARKINFER_KV_HEADROOM_GIB': '4.5GiB'}, int(4.5 * 1024**3)),
    ({'SPARKINFER_KV_HEADROOM_GIB': '0'}, 0)])
def test_spark_small_context_uses_source_kv_growth_budget_not_a_fixed_residency_forecast(tmp_path, environment, reserve):
    _, profile, runtime = inputs(tmp_path / 'metadata-only', 'sparkinfer')
    runtime.version = '0.6.34'
    runtime.metadata['build_environment'] = environment
    devices = [dict(device='CUDA0', compute_capability=[12, 0], memory_total=32607 * 1024**2)]
    gpu = estimates._spark(spark_nvfp4_metadata(), profile, runtime, {'--ctx': 4096}, devices)[0]
    assert gpu['quality'] == 'dynamic' and gpu['reserve_bytes'] == reserve
    assert gpu['fixed_bytes'] == gpu['bytes'] == 29_414_959_630 - 9_003_411_456 + 264 * 562432
    assert 'budget_bytes' not in gpu  # UI projects the budget against current available memory.
    runtime.metadata['build_environment'] = {**environment, 'SPARKINFER_KV_GROW': '0'}
    fixed = estimates._spark(spark_nvfp4_metadata(), profile, runtime, {'--ctx': 4096}, devices)[0]
    assert fixed['quality'] == 'estimate' and 'reserve_bytes' not in fixed


def test_spark_compressed_tensors_export_names_and_unused_layers_match_modelopt(tmp_path):
    metadata = spark_nvfp4_metadata()
    _, profile, runtime = inputs(tmp_path / 'metadata-only', 'sparkinfer')
    runtime.version = '0.6.34'
    runtime.metadata['build_environment'] = {'SPARKINFER_KV_GROW': '0'}
    devices = [dict(device='CUDA0', compute_capability=[12, 0])]
    baseline = estimates._spark(metadata, profile, runtime, {}, devices)[0]
    for tensor in metadata['tensors']:
        if tensor['dtype'] == 'U8':
            tensor['name'] = tensor['name'].removesuffix('.weight') + '.weight_packed'
        if tensor['name'].endswith('.weight_scale_2'):
            tensor['name'] = tensor['name'].removesuffix('.weight_scale_2') + '.weight_global_scale'
    metadata['tensors'].extend([dict(name='model.visual.blocks.27.norm1.weight', shape=[1024], dtype='BF16', bytes=2048),
                               dict(name='model.language_model.layers.64.input_layernorm.weight', shape=[1024], dtype='BF16', bytes=2048)])
    converted = estimates._spark(metadata, profile, runtime, {}, devices)[0]
    assert converted['bytes'] == baseline['bytes'] and converted['quality'] == 'estimate'
    metadata['tensors'] = [t for t in metadata['tensors'] if t['name'] != 'model.visual.patch_embed.proj.weight']
    partial = estimates._spark(metadata, profile, runtime, {}, devices)[0]
    assert partial['quality'] == 'partial' and part(partial, '视觉常驻')['bytes'] > 0


@pytest.mark.asyncio
async def test_device_index_and_visibility_mapping_are_not_always_cuda0(tmp_path):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), 'sglang')
    runtime.metadata['build_environment'] = {'CUDA_VISIBLE_DEVICES': '3,5'}
    profile.extra_args = ['--base-gpu-id', '1']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['device'] == 'CUDA5'
    profile.extra_args = ['--device', 'xpu']
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['device'] == 'XPU' and gpu['bytes'] == 256


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
async def test_hf_parallel_budget_rows_use_target_uuid_without_dividing_weight_reference(tmp_path, monkeypatch, backend):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), backend, parameters={
        'tensor_parallel_size': 2, 'pipeline_parallel_size': 2, 'cuda_visible_devices': '5,3,1,7',
        'gpu_memory_utilization': .8, 'mem_fraction_static': .7})
    identities = [dict(device=f'CUDA{index}', gpu_id=f'GPU-physical-{physical}',
                       pci_bus_id=f'0000:0{index+1}:00.0') for index, physical in enumerate([5, 3, 1, 7])]
    monkeypatch.setattr(estimates, 'cuda_devices_for', lambda *args: identities)
    resources = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    aggregate, *budgets, ram = resources
    assert aggregate['aggregate'] and aggregate['device'] == '多 GPU 合计'
    assert aggregate['bytes'] == 256 and part(aggregate, '权重')['bytes'] == 256
    assert aggregate['identity_status'] == 'unknown' and 'gpu_id' not in aggregate
    assert [row['gpu_id'] for row in budgets] == [row['gpu_id'] for row in identities]
    assert [row['logical_device'] for row in budgets] == ['CUDA0', 'CUDA1', 'CUDA2', 'CUDA3']
    assert [row['device'] for row in budgets] == ['CUDA5', 'CUDA3', 'CUDA1', 'CUDA7']
    assert all(row['bytes'] is None and row['quality'] == 'dynamic' for row in budgets)
    assert all(row['budget_fraction'] == (.8 if backend == 'vllm' else .7) for row in budgets)
    assert all(row['budget_basis'] == ('total' if backend == 'vllm' else 'available') for row in budgets)
    assert ram['kind'] == 'ram'


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
@pytest.mark.parametrize('extra', [['--nnodes', '2'], ['--data-parallel-size', '2'],
                                  ['--node-rank', '1'], ['--distributed-executor-backend', 'ray']])
async def test_distributed_or_replicated_hf_keeps_reference_without_local_budget(tmp_path, monkeypatch, backend, extra):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), backend,
        parameters={'tensor_parallel_size': 2, 'gpu_memory_utilization': .8, 'mem_fraction_static': .7})
    profile.extra_args = extra
    monkeypatch.setattr(estimates, 'cuda_devices_for', lambda *args: [
        dict(device=f'CUDA{index}', gpu_id=f'GPU-{index}') for index in range(8)])
    resources = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert len(resources) == 2 and resources[0]['bytes'] == 256
    assert resources[0]['identity_status'] == 'unknown' and 'gpu_id' not in resources[0]


@pytest.mark.asyncio
async def test_target_identity_failure_does_not_guess_single_or_multi_gpu_capacity(tmp_path, monkeypatch):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), 'sglang',
        parameters={'mem_fraction_static': .7, 'cuda_visible_devices': 'GPU-selected'})
    gpu = (await ResourceEstimator().estimate(model, profile, runtime))['resources'][0]
    assert gpu['bytes'] == 256 and gpu['identity_status'] == 'unknown' and 'gpu_id' not in gpu
    profile.parameters['tensor_parallel_size'] = 2
    monkeypatch.setattr(estimates, 'cuda_devices_for', lambda *args: [dict(device='CUDA0', gpu_id='GPU-selected')])
    resources = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert len(resources) == 2 and resources[0]['aggregate'] and resources[0]['identity_status'] == 'unknown'


@pytest.mark.asyncio
async def test_sglang_base_stride_and_last_parallel_override_select_exact_target_cards(tmp_path, monkeypatch):
    model, profile, runtime = inputs(safetensors(tmp_path / 'hf'), 'sglang',
        parameters={'tensor_parallel_size': 8, 'pipeline_parallel_size': 8, 'mem_fraction_static': .7})
    profile.extra_args = ['--tensor-parallel-size', '2', '--pipeline-parallel-size', '2',
                          '--base-gpu-id', '1', '--gpu-id-step', '3']
    monkeypatch.setattr(estimates, 'cuda_devices_for', lambda *args: [
        dict(device=f'CUDA{index}', gpu_id=f'GPU-{index}') for index in range(7)])
    resources = (await ResourceEstimator().estimate(model, profile, runtime))['resources']
    assert [item['gpu_id'] for item in resources[1:-1]] == ['GPU-1', 'GPU-4', 'GPU-3', 'GPU-6']


@pytest.mark.parametrize('backend', ['vllm', 'sglang', 'ninfer', 'sparkinfer'])
def test_identity_enumeration_uses_selected_wsl_environment_only_on_call(tmp_path, monkeypatch, backend):
    _, profile, runtime = inputs(tmp_path / 'unused', backend, parameters={'cuda_visible_devices': 'GPU-override'})
    runtime.metadata['build_environment'] = {'CUDA_VISIBLE_DEVICES': '3,5', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                                             'HF_TOKEN': 'fixture-must-not-be-passed'}
    calls = []
    def read(distribution, code, args, **kwargs):
        calls.append((distribution, json.loads(args[0])))
        assert 'ctypes' in kwargs['input'] and 'cuDeviceGetCount' in kwargs['input']
        assert kwargs['timeout'] == 10 and 'os.environ.update' in code
        return '[{"device":"CUDA0","gpu_id":"GPU-override"}]'
    monkeypatch.setattr(estimates, 'wsl_python', read)
    assert calls == []
    assert read_cuda_devices_for(profile, runtime)[0]['gpu_id'] == 'GPU-override'
    expected = {'CUDA_VISIBLE_DEVICES': 'GPU-override'}
    if backend != 'vllm':
        expected['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    assert calls == [('ModLoader', expected)]
    def unavailable(*args, **kwargs):
        raise ValueError('distribution unavailable')
    monkeypatch.setattr(estimates, 'wsl_python', unavailable)
    assert read_cuda_devices_for(profile, runtime) == []


def gguf_metadata_fixture(path, layers=4):
    string = lambda value: struct.pack('<Q', len(value.encode())) + value.encode()
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 3)
    header += string('general.architecture') + struct.pack('<I', 8) + string('qwen4exp')
    header += string('qwen4exp.block_count') + struct.pack('<II', 4, layers)
    header += string('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 2) + string('first') + string('second')
    header += string('weight') + struct.pack('<IQIQ', 1, 1, 0, 0)
    path.write_bytes(header + bytes((-len(header)) % 32) + bytes(4))
    return path


def test_gguf_cache_coalesces_cold_reads_and_never_exposes_shared_mutable_results(tmp_path, monkeypatch):
    path = gguf_metadata_fixture(tmp_path / 'model.gguf')
    original = resource_metadata._gguf
    reads = []
    def counted(path):
        reads.append(str(path))
        time.sleep(.02)
        return original(path)
    monkeypatch.setattr(resource_metadata, '_gguf', counted)
    barrier = Barrier(3)
    def preview(_):
        barrier.wait(timeout=5)
        return read_metadata(path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(preview, range(3)))
    assert len(reads) == 1
    results[0]['config']['qwen4exp.block_count'] = 999
    results[0]['tensors'][0]['type'] = 'caller annotation'
    assert results[1]['config']['qwen4exp.block_count'] == 4
    assert 'type' not in results[1]['tensors'][0]
    assert read_metadata(path)['config']['qwen4exp.block_count'] == 4
    assert len(reads) == 1


def test_gguf_cache_invalidates_each_shard_and_does_not_cache_failures(tmp_path, monkeypatch):
    from modloader.strata_resource_estimates import _headers
    a = gguf_metadata_fixture(tmp_path / 'model-00001-of-00002.gguf')
    b = gguf_metadata_fixture(tmp_path / 'model-00002-of-00002.gguf')
    # Distinct tensor names make this a valid two-shard descriptor set.
    raw = b.read_bytes().replace(b'weight', b'second')
    b.write_bytes(raw)
    original = resource_metadata._gguf
    reads = []
    def counted(path):
        reads.append(str(path))
        return original(path)
    monkeypatch.setattr(resource_metadata, '_gguf', counted)
    metadata, _ = _headers(a)
    assert metadata['qwen4exp.block_count'] == 4 and len(reads) == 2
    assert 'type' not in read_metadata(a)['tensors'][0]  # _headers annotations cannot pollute the cache.
    stamp = b.stat()
    b.write_bytes(raw.replace(struct.pack('<II', 4, 4), struct.pack('<II', 4, 8), 1))
    os.utime(b, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    assert _headers(a)[0]['qwen4exp.block_count'] == 8 and len(reads) == 3
    b.write_bytes(b'bad header')
    for _ in range(2):
        with pytest.raises(ValueError):
            read_metadata(b)
    assert len(reads) == 5
    gguf_metadata_fixture(b, layers=6)
    assert read_metadata(b)['config']['qwen4exp.block_count'] == 6 and len(reads) == 6


@pytest.mark.parametrize('element', [4, 8])
def test_gguf_skip_fast_path_still_rejects_truncated_tokenizer_arrays(tmp_path, element):
    name = b'tokenizer.ggml.tokens'
    header = b'GGUF' + struct.pack('<IQQQ', 3, 0, 1, len(name)) + name
    header += struct.pack('<IIQ', 9, element, 1)
    header += struct.pack('<Q', 100) if element == 8 else b''
    path = tmp_path / 'truncated.gguf'
    path.write_bytes(header)
    with pytest.raises(ValueError, match='文件边界'):
        read_metadata(path)
