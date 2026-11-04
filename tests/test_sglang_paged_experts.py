"""Paged Experts contracts, using only synthetic config and safetensors headers."""
import json
import subprocess

import pytest

from modloader.backends import sglang
from modloader.launch_parameters import edit_parameters, parameter_argv, validate_editor_profile
from modloader.parameter_rules import profile_fields, validate_profile
from modloader.resource_estimates import _options
from modloader import wsl_resource_estimates as estimates
from test_sglang_backend import fixture


@pytest.fixture
def case(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Paged Experts contracts must not start WSL or an engine')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(estimates, 'cuda_devices_for', lambda *args: [])
    model, runtime, profile, context = fixture(tmp_path, monkeypatch)
    model.metadata = dict(model_type='qwen3_moe', num_experts=4, num_experts_per_tok=2,
                          num_hidden_layers=2, torch_dtype='bfloat16')
    runtime.version = '0.5.23'
    profile.parameters.update(enable_paged_experts=True, paged_experts_num_resident=2)
    return model, runtime, profile, context


def test_runtime_aware_fields_and_router_lower_bound(case):
    model, runtime, profile, _ = case
    fields = {f.key: f for f in profile_fields(sglang.SglangAdapter(), model, runtime)}
    assert fields['enable_paged_experts'].type == 'boolean'
    assert fields['paged_experts_num_resident'].minimum == 1
    assert sglang.paged_experts(model, profile, runtime)['top_k'] == 2
    assert fields['paged_experts_num_resident'].maximum is None
    assert fields['moe_runner_backend'].type == 'text'  # Other runners remain legal outside paging.
    runtime.version = '0.5.21'
    assert 'enable_paged_experts' not in {f.key for f in profile_fields(sglang.SglangAdapter(), model, runtime)}
    with pytest.raises(ValueError, match='Paged Experts.*0.5.21'):
        sglang.validate(model, profile, runtime)
    runtime.metadata['supported_flags'] = ['--enable-paged-experts', '--paged-experts-num-resident',
                                            '--moe-runner-backend', '--disable-shared-experts-fusion', '--dtype', '--tp-size']
    assert 'enable_paged_experts' in {f.key for f in profile_fields(sglang.SglangAdapter(), model, runtime)}
    sglang.validate(model, profile, runtime)


def test_one_effective_command_for_launch_editor_and_estimate(case):
    model, runtime, profile, context = case
    profile.extra_args = ['--paged-experts-num-resident=3', '--moe-runner-backend', 'triton']
    expected = parameter_argv(model, profile, runtime)
    spec = sglang.SglangAdapter().build_launch(model, profile, runtime, context)
    assert spec.argv[-len(expected):] == expected
    options = sglang.effective_options(model, profile, runtime)
    assert options['--paged-experts-num-resident'] == '3'
    assert options['--moe-runner-backend'] == 'triton'
    assert '--disable-shared-experts-fusion' in options
    assert estimates._options(profile, runtime, _options, model=model) == options


@pytest.mark.parametrize('parameters,args,message', [
    ({'paged_experts_num_resident': None}, [], '驻留专家 K.*必填'),
    ({'paged_experts_num_resident': 1}, [], 'top.k.*2'),
    ({'paged_experts_num_resident': 2.5}, [], '驻留专家 K.*整数'),
    ({'tensor_parallel_size': 2}, [], 'tp-size.*1'),
    ({}, ['--pp', '2'], 'pp-size.*1'),
    ({}, ['--ep-size=2'], 'ep-size.*1'),
    ({}, ['--data-parallel-size', '2'], 'dp-size.*1'),
    ({'acceleration_mode': 'mtp'}, [], '推测解码'),
    ({}, ['--enable-eplb'], 'enable-eplb'),
    ({}, ['--enable-fused-moe-sum-all-reduce'], 'fused-moe-sum-all-reduce'),
    ({}, ['--load-format=dummy'], 'load-format.*dummy'),
    ({'cuda_graph_backend_decode': 'breakable'}, [], '解码.*full.*disabled'),
    ({'cuda_graph_backend_prefill': 'full'}, [], '预填充.*breakable.*disabled'),
    ({}, ['--cuda-graph-config', '{"decode":{"backend":"breakable"}}'], '解码.*full'),
    ({}, ['--enforce-shared-experts-fusion'], '共享专家融合'),
    ({'moe_runner_backend': 'flashinfer_cutlass'}, [], 'MoE.*triton.*marlin'),
    ({'moe_runner_backend': 'marlin'}, [], '未量化.*triton'),
])
def test_paged_hard_combinations(case, parameters, args, message):
    model, runtime, profile, _ = case
    model.metadata['n_shared_experts'] = 1
    profile.parameters.update(parameters)
    profile.extra_args = args
    with pytest.raises(ValueError, match=message):
        sglang.validate(model, profile, runtime)


def test_graph_json_wins_over_scalar_and_k_can_exceed_expert_count(case):
    model, runtime, profile, _ = case
    profile.parameters.update(paged_experts_num_resident=99, cuda_graph_backend_decode='breakable',
                              cuda_graph_backend_prefill='full')
    profile.extra_args = ['--cuda-graph-config', '{"decode":{"backend":"full"},"prefill":{"backend":"disabled"}}']
    sglang.validate(model, profile, runtime)
    state = sglang.paged_experts(model, profile, runtime)
    assert state['num_resident'] == 99 and state['num_experts'] == 4
    assert state['top_k'] == 2 and state['status'] == 'available'


@pytest.mark.parametrize('quant,runner,good', [
    ({'quant_method': 'gptq', 'bits': 4}, 'marlin', True),
    ({'quant_method': 'awq', 'bits': 4}, 'marlin', True),
    ({'quant_method': 'mxfp4'}, 'marlin', True),
    ({'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}, 'marlin', True),
    ({'quant_method': 'fp8', 'weight_block_size': [128, 128]}, 'triton', True),
    ({'quant_method': 'fp8'}, 'triton', False),
    ({'quant_method': 'gptq', 'bits': 3}, 'marlin', False),
    ({'quant_method': 'gptq', 'bits': 4}, 'triton', False),
    ({'quant_method': 'bitsandbytes'}, 'triton', False),
])
def test_known_quantization_rules_and_runner_recommendation(case, quant, runner, good):
    model, runtime, profile, _ = case
    model.metadata['quantization_config'] = quant
    profile.parameters['moe_runner_backend'] = runner
    state = sglang.paged_experts(model, profile, runtime)
    if good:
        sglang.validate(model, profile, runtime)
        assert state['recommended_runner'] == runner
    else:
        assert state['status'] == 'unsupported' and state['message']
        with pytest.raises(ValueError, match='Paged Experts'):
            sglang.validate(model, profile, runtime)


def test_unknown_metadata_allows_attempt_with_missing_state(case):
    model, runtime, profile, _ = case
    model.metadata = {}
    sglang.validate(model, profile, runtime)
    state = sglang.paged_experts(model, profile, runtime)
    assert state['status'] == 'unverified'
    assert state['top_k'] is None and state['num_experts'] is None
    assert state['message']


def test_disabled_feature_preserves_dormant_invalid_k_and_nonpaged_runner(case):
    model, runtime, profile, context = case
    profile.parameter_editor_version = 1
    profile.parameters.update(enable_paged_experts=False, paged_experts_num_resident='not a number',
                              moe_runner_backend='flashinfer_cutlass')
    validate_profile(model, profile, runtime, sglang.SglangAdapter())
    spec = sglang.SglangAdapter().build_launch(model, profile, runtime, context)
    assert '--enable-paged-experts' not in spec.argv and '--paged-experts-num-resident' not in spec.argv
    assert spec.argv[spec.argv.index('--moe-runner-backend') + 1] == 'flashinfer_cutlass'
    assert profile.parameters['paged_experts_num_resident'] == 'not a number'


def test_legacy_editor_extras_still_launch_and_normalize_without_validation(case):
    model, runtime, profile, _ = case
    profile.parameter_editor_version = 1
    profile.parameters = {'enable_paged_experts': False, 'acceleration_mode': 'mtp'}
    profile.extra_args = ['--enable-paged-experts', '--paged-experts-num-resident=broken',
                          '--moe-runner-backend', 'triton', '--attention-backend', 'flashinfer']
    before = profile.model_dump()
    state = sglang.paged_experts(model, profile, runtime)
    assert state['enabled'] and state['num_resident'] == 'broken'
    assert state['normalized_parameters']['paged_experts_num_resident'] == 'broken'
    assert state['normalized_extra_args'] == ['--attention-backend', 'flashinfer']
    assert profile.model_dump() == before
    fixed = profile.model_copy(update={'parameters': {**state['normalized_parameters'], 'enable_paged_experts': False},
                                       'extra_args': state['normalized_extra_args']})
    validate_editor_profile(model, fixed, runtime)  # Dormant bad K does not block fixing the switch.
    profile.parameters = {}
    profile.extra_args = ['--moe-runner-backend', 'flashinfer_cutlass', '--disable-shared-experts-fusion']
    validate_editor_profile(model, profile, runtime)  # New mappings must not invalidate old editor profiles.
    sglang.validate(model, profile, runtime)
    result = edit_parameters(model, profile, runtime, text='--enable-paged-experts --paged-experts-num-resident 2 --moe-runner-backend triton')
    assert not result['errors'] and result['parameters']['enable_paged_experts'] is True
    assert result['parameters']['paged_experts_num_resident'] == 2 and result['extra_args'] == []


def metadata(*, packed=False, quant=False):
    config = dict(model_type='qwen3_moe', num_experts=4, num_experts_per_tok=2,
                  torch_dtype='bfloat16', num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, hidden_size=8)
    tensors = [dict(name='model.embed_tokens.weight', dtype='BF16', shape=[4, 8], bytes=64),
               dict(name='model.layers.0.mlp.shared_expert.weight', dtype='BF16', shape=[2, 8], bytes=32)]
    for layer in range(2):
        if packed:
            tensors += [dict(name=f'model.layers.{layer}.mlp.experts.gate_up_proj_blocks', dtype='U8', shape=[4, 2, 4], bytes=32),
                        dict(name=f'model.layers.{layer}.mlp.experts.gate_up_proj_scales', dtype='F32', shape=[4, 2], bytes=32)]
        else:
            for expert in range(4):
                tensors.append(dict(name=f'model.layers.{layer}.mlp.experts.{expert}.up_proj.weight',
                                    dtype='BF16', shape=[2, 8], bytes=32))
    if quant:
        config['quantization_config'] = {'quant_method': 'mxfp4'}
    return dict(format='safetensors', config=config, tensors=tensors, errors=[])


def component(row, label):
    return next(p for p in row['components'] if p['label'].startswith(label))


def allocation(case, info=None, args=()):
    model, runtime, profile, _ = case
    profile.extra_args = list(args)
    options = sglang.effective_options(model, profile, runtime)
    return estimates._hf_engine(info or metadata(), profile, runtime, options)


@pytest.mark.parametrize('k,gpu_experts', [(2, 128), (3, 192), (99, 256)])
def test_gpu_k_slots_and_full_host_store_are_counted_once(case, k, gpu_experts):
    model, runtime, profile, _ = case
    profile.parameters['paged_experts_num_resident'] = k
    gpu, ram = allocation(case)
    assert component(gpu, '非分页权重')['bytes'] == 96
    assert component(gpu, '分页专家 GPU')['bytes'] == gpu_experts
    assert gpu['bytes'] == 96 + gpu_experts
    assert component(ram, '分页专家锁页 Host')['bytes'] == 256
    assert ram['bytes'] == 256  # All E on host, not only E-K, and not doubled.
    assert component(gpu, '单序列 KV')['reference'] is True
    assert gpu['quality'] == ram['quality'] == 'partial'


def test_packed_quantized_scale_layout_not_cast_to_activation_dtype(case):
    _, _, profile, _ = case
    profile.parameters['dtype'] = 'float32'
    gpu, ram = allocation(case, metadata(packed=True, quant=True))
    assert component(gpu, '非分页权重')['bytes'] == 192
    assert component(gpu, '分页专家 GPU')['bytes'] == 64
    assert component(ram, '分页专家锁页 Host')['bytes'] == 128
    assert any('重排' in note for note in gpu['notes'])
    assert any('锁页' in note and 'RAM' in note for note in ram['notes'])


@pytest.mark.parametrize('args', [('--cpu-offload-gb', '1'), ('--tp', '2')])
def test_other_placement_does_not_report_unsharded_weights_as_actual_gpu(case, args):
    gpu, ram = allocation(case, args=args)
    assert component(gpu, '非分页权重').get('reference') is True
    assert component(ram, '分页专家锁页 Host')['bytes'] == 256
    if args[0] == '--tp':
        assert component(ram, '分页专家锁页 Host')['reference'] is True
        assert ram['bytes'] is None  # Invalid parallel placement is not a single-store allocation.
    else:
        assert ram['bytes'] == 256


def test_missing_or_unknown_expert_layout_is_unknown_not_full_gpu_or_zero(case):
    info = metadata()
    info['tensors'] = [dict(name='model.layers.0.mlp.experts.opaque', dtype='U8', shape=[9], bytes=9)]
    gpu, ram = allocation(case, info)
    assert component(gpu, '分页专家 GPU')['bytes'] is None
    assert component(ram, '分页专家锁页 Host')['bytes'] is None
    assert gpu['bytes'] is None and ram['bytes'] is None


def test_unpaged_estimate_and_mtp_path_are_unchanged(case):
    model, runtime, profile, _ = case
    profile.parameters['enable_paged_experts'] = False
    gpu, ram = allocation(case)
    assert component(gpu, '权重张量参考')['bytes'] == 352
    assert gpu['bytes'] == 352 and ram['bytes'] is None
    assert not any('分页专家' in p['label'] for row in (gpu, ram) for p in row['components'])


def test_effective_model_override_does_not_apply_stale_router_minimum(case):
    model, runtime, profile, _ = case
    profile.parameters['paged_experts_num_resident'] = 1
    profile.extra_args = ['--json-model-override-args', '{"num_experts_per_tok":1}']
    validate_profile(model, profile, runtime, sglang.SglangAdapter())
    assert sglang.paged_experts(model, profile, runtime)['top_k'] == 1
    result = edit_parameters(model, profile, runtime,
                             text='--enable-paged-experts --paged-experts-num-resident 1 --json-model-override-args \'{"num_experts_per_tok":1}\'')
    assert not result['errors']


def test_generic_generation_top_k_is_not_a_router_constraint(case):
    model, runtime, profile, _ = case
    del model.metadata['num_experts_per_tok']
    model.metadata['top_k'] = 50
    sglang.validate(model, profile, runtime)
    assert sglang.paged_experts(model, profile, runtime)['top_k'] is None


def test_unknown_fp8_layout_and_partial_quantization_are_not_guessed(case):
    model, runtime, profile, _ = case
    model.metadata = {}
    profile.extra_args = ['--quantization', 'fp8']
    assert sglang.paged_experts(model, profile, runtime)['status'] == 'unverified'
    sglang.validate(model, profile, runtime)
    model.metadata = {'num_experts': 4, 'num_experts_per_tok': 2,
                      'quantization_config': {'quant_method': 'bitsandbytes', 'modules_to_not_convert': ['model.layers.*.mlp.experts']}}
    profile.extra_args = []
    profile.parameters['moe_runner_backend'] = 'triton'
    assert sglang.paged_experts(model, profile, runtime)['status'] == 'unverified'
    sglang.validate(model, profile, runtime)


def test_compressed_nvfp4_blackwell_condition_uses_supplied_identity_not_a_probe(case):
    model, runtime, profile, _ = case
    model.metadata['quantization_config'] = {
        'quant_method': 'compressed-tensors', 'config_groups': {'group_0': {
            'weights': {'type': 'float', 'num_bits': 4, 'strategy': 'tensor_group', 'group_size': 16},
            'input_activations': {'type': 'float', 'num_bits': 4, 'strategy': 'tensor_group', 'group_size': 16},
        }}}
    assert sglang.paged_experts(model, profile, runtime)['status'] == 'unverified'
    assert sglang.paged_experts(model, profile, runtime, devices=[{'compute_capability': [12, 0]}])['status'] == 'unsupported'
    assert sglang.paged_experts(model, profile, runtime, devices=[{'compute_capability': [8, 9]}])['status'] == 'available'


def test_normalization_of_missing_k_does_not_swallow_the_next_flag(case):
    model, runtime, profile, _ = case
    profile.parameters = {}
    profile.extra_args = ['--enable-paged-experts', '--paged-experts-num-resident', '--attention-backend', 'flashinfer']
    state = sglang.paged_experts(model, profile, runtime)
    assert state['normalized_parameters']['paged_experts_num_resident'] is None
    assert state['normalized_extra_args'] == ['--attention-backend', 'flashinfer']
    fixed = profile.model_copy(update={'parameters': {**state['normalized_parameters'], 'paged_experts_num_resident': 2},
                                       'extra_args': state['normalized_extra_args']})
    sglang.validate(model, fixed, runtime)


def test_serialized_quantization_scale_reference_and_missing_shards_are_explicit(case):
    info = metadata(packed=True, quant=True)
    gpu, ram = allocation(case, info)
    assert component(gpu, '分页专家 GPU')['reference'] is True
    assert component(ram, '分页专家锁页 Host')['reference'] is True
    assert component(gpu, '分页专家 GPU 重排')['bytes'] is None
    assert component(ram, '分页专家 Host 重排')['bytes'] is None
    info['errors'] = ['missing model-00002-of-00002.safetensors']
    gpu, ram = allocation(case, info)
    assert component(gpu, '分页专家 GPU')['bytes'] is None
    assert component(ram, '分页专家锁页 Host')['bytes'] is None


def test_explicit_auto_runner_is_preserved_and_not_claimed_resolved(case):
    model, runtime, profile, _ = case
    profile.parameters['moe_runner_backend'] = 'auto'
    state = sglang.paged_experts(model, profile, runtime)
    assert state['runner'] == 'auto' and state['recommended_runner'] == 'triton'
    assert state['status'] == 'unverified' and 'auto' in state['message']
    sglang.validate(model, profile, runtime)


def test_valid_integer_cli_spellings_and_cpu_hard_requirement(case):
    model, runtime, profile, _ = case
    profile.extra_args = ['--tp', '+1', '--pp-size', '01']
    sglang.validate(model, profile, runtime)
    profile.extra_args = ['--device', 'cpu']
    with pytest.raises(ValueError, match='GPU'):
        sglang.validate(model, profile, runtime)


@pytest.mark.asyncio
async def test_header_reader_to_resource_estimator_without_wsl(case, tmp_path, monkeypatch):
    from modloader.resource_estimates import ResourceEstimator
    from test_wsl_resource_estimates import safetensors
    model, runtime, profile, _ = case
    info = metadata()
    root = safetensors(tmp_path / 'paged headers', config=info['config'],
                       tensors={t['name']: (t['dtype'], t['shape']) for t in info['tensors']})
    model.path = str(root)
    def forbidden(*args, **kwargs):
        pytest.fail('Windows header preview must not start WSL')
    monkeypatch.setattr(estimates, 'wsl_python', forbidden)
    result = await ResourceEstimator().estimate(model, profile, runtime)
    gpu, ram = result['resources']
    assert result['source'] == 'sglang-metadata' and not result.get('diagnostic')
    assert gpu['bytes'] == 224 and ram['bytes'] == 256
    assert component(gpu, '分页专家 GPU')['bytes'] == 128
    assert component(ram, '分页专家锁页 Host')['bytes'] == 256


def test_dormant_bad_k_in_legacy_extra_is_preserved_but_not_emitted(case):
    model, runtime, profile, context = case
    profile.parameter_editor_version = 1
    profile.parameters = {}
    profile.extra_args = ['--paged-experts-num-resident', 'old-invalid-value',
                          '--moe-runner-backend', 'flashinfer_cutlass']
    before = profile.model_dump()
    validate_profile(model, profile, runtime, sglang.SglangAdapter())
    spec = sglang.SglangAdapter().build_launch(model, profile, runtime, context)
    assert '--paged-experts-num-resident' not in spec.argv
    assert sglang.paged_experts(model, profile, runtime)['num_resident'] == 'old-invalid-value'
    assert estimates._options(profile, runtime, _options, model=model)['--moe-runner-backend'] == 'flashinfer_cutlass'
    assert profile.model_dump() == before


def test_host_store_does_not_depend_on_k_being_configured(case):
    _, _, profile, _ = case
    profile.parameters.pop('paged_experts_num_resident')
    gpu, ram = allocation(case)
    assert component(gpu, '分页专家 GPU')['bytes'] is None
    assert component(ram, '分页专家锁页 Host')['bytes'] == 256


def test_external_paging_help_does_not_imply_optional_flags_exist(case):
    model, runtime, profile, _ = case
    runtime.version = 'external-backport'
    runtime.metadata['supported_flags'] = ['--enable-paged-experts', '--paged-experts-num-resident', '--dtype', '--tp-size']
    sglang.validate(model, profile, runtime)
    state = sglang.paged_experts(model, profile, runtime)
    assert state['status'] == 'unverified' and state['runner'] is None
    args = parameter_argv(model, profile, runtime)
    assert '--disable-shared-experts-fusion' not in args and '--moe-runner-backend' not in args
