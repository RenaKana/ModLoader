"""Granite legacy packed experts, from the actual public checkpoint header."""
import json
from pathlib import Path

import pytest

from test_sglang_paged_experts import allocation, case, component, metadata


@pytest.fixture
def granite(case):
    info = json.loads((Path(__file__).parent / 'fixtures/sglang-granite-moe-legacy-header.json').read_text(encoding='utf-8'))
    model, _, profile, _ = case
    model.metadata = dict(info['config'])
    profile.parameters.update(paged_experts_num_resident=8, dtype='bfloat16', context_length=2048)
    return case, info


@pytest.mark.parametrize('names', ['legacy', 'current'])
def test_granite_checkpoint_expert_payload_is_split_into_k_gpu_slots_and_full_host(granite, names):
    case, info = granite
    assert len(info['tensors']) == 218
    assert sum(t['bytes'] for t in info['tensors']) == 2669256704
    if names == 'current':
        for tensor in info['tensors']:
            tensor['name'] = tensor['name'].replace('.block_sparse_moe.input_linear.weight', '.block_sparse_moe.experts.gate_up_proj')
            tensor['name'] = tensor['name'].replace('.block_sparse_moe.output_linear.weight', '.block_sparse_moe.experts.down_proj')
    gpu, ram = allocation(case, info)
    assert component(gpu, '非分页权重')['bytes'] == 253337600
    assert component(gpu, '分页专家 GPU')['bytes'] == 603979776
    assert component(ram, '分页专家锁页 Host')['bytes'] == 2415919104
    assert gpu['bytes'] == 857317376 and ram['bytes'] == 2415919104
    assert component(gpu, '实际 KV 池')['bytes'] is None
    assert gpu['quality'] == ram['quality'] == 'partial'


def test_granite_router_shared_and_dense_linear_weights_are_not_routed_experts(granite):
    case, info = granite
    assert any(t['name'].endswith('.block_sparse_moe.router.layer.weight') for t in info['tensors'])
    for name in ('model.layers.0.mlp.input_linear.weight',
                 'model.layers.0.block_sparse_moe.shared_experts.input_linear.weight'):
        info['tensors'].append(dict(name=name, dtype='BF16', shape=[32, 1024], bytes=65536))
    gpu, ram = allocation(case, info)
    assert component(gpu, '非分页权重')['bytes'] == 253337600 + 2 * 65536
    assert component(gpu, '分页专家 GPU')['bytes'] == 603979776
    assert component(ram, '分页专家锁页 Host')['bytes'] == 2415919104


@pytest.mark.parametrize('shape', [[1024, 32, 1024], [32, 1048576]])
def test_granite_legacy_experts_need_a_real_three_dimensional_e_first_axis(granite, shape):
    case, info = granite
    tensor = next(t for t in info['tensors'] if t['name'].endswith('.block_sparse_moe.input_linear.weight'))
    tensor['shape'] = shape  # Same byte count, but not the loader's [E, 2I, H].
    gpu, ram = allocation(case, info)
    assert component(gpu, '非分页权重')['bytes'] == 253337600
    assert component(gpu, '分页专家 GPU')['bytes'] is None
    assert component(ram, '分页专家锁页 Host')['bytes'] is None


def test_granite_explicit_extra_k_wins_without_mutating_the_profile(granite):
    case, info = granite
    profile = case[2]
    gpu, ram = allocation(case, info, args=['--paged-experts-num-resident=16'])
    assert component(gpu, '分页专家 GPU')['bytes'] == 1207959552
    assert component(ram, '分页专家锁页 Host')['bytes'] == 2415919104
    assert profile.parameters['paged_experts_num_resident'] == 8
    assert profile.extra_args == ['--paged-experts-num-resident=16']


@pytest.mark.parametrize('version', ['0.5.21', '0.5.23'])
def test_granite_paging_off_preserves_full_weight_preview_and_dormant_k(granite, version):
    case, info = granite
    _, runtime, profile, _ = case
    runtime.version = version
    profile.parameters.update(enable_paged_experts=False, paged_experts_num_resident='keep dormant value')
    gpu, ram = allocation(case, info)
    assert component(gpu, '权重张量参考')['bytes'] == 2669256704
    assert gpu['bytes'] == 2669256704 and ram['bytes'] is None
    assert not any('分页专家' in p['label'] for row in (gpu, ram) for p in row['components'])
    assert profile.parameters['paged_experts_num_resident'] == 'keep dormant value'


def test_existing_packed_quantized_bias_keeps_serialized_precision(case):
    info = metadata(packed=True, quant=True)
    case[2].parameters['dtype'] = 'float32'
    for layer in range(2):
        info['tensors'].append(dict(name=f'model.layers.{layer}.mlp.experts.gate_up_proj_bias',
                                    dtype='BF16', shape=[4, 2], bytes=16))
    gpu, ram = allocation(case, info)
    assert component(gpu, '分页专家 GPU')['bytes'] == 80
    assert component(ram, '分页专家锁页 Host')['bytes'] == 160
