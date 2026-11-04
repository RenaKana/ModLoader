"""Pinned native parameter domains, without starting an engine or invoking WSL."""
from pathlib import Path

import pytest

from modloader.backends import native_engines as native
from modloader.models import LoadProfile, ModelAsset, RuntimeInstallation
from modloader.mtp import token_field, validate_tokens

from test_native_engines import _spark_output_fixture, local


INT_MIN, INT_MAX = -(2**31), 2**31 - 1


@pytest.fixture
def native_profile():
    model = ModelAsset(id='model', name='Fixture', path='/fixture/source', format='safetensors')
    profile = LoadProfile(id='profile', name='Fixture', model_id=model.id, runtime_id='runtime',
                          backend='ninfer', alias='fixture')
    runtime = RuntimeInstallation(id='runtime', name='Fixture', backend='ninfer', platform='wsl2',
                                  path='/fixture/ninfer-serve')
    return model, profile, runtime


@pytest.mark.parametrize('value', [0, -1, INT_MIN])
def test_spark_nonpositive_context_reaches_native_auto_unchanged(local, value):
    model, runtime, profile, context = _spark_output_fixture(local)
    profile.parameters['context_length'] = value
    spec = native.SparkInferAdapter().build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--ctx') + 1] == str(value)
    assert profile.parameters['context_length'] == value


@pytest.mark.parametrize('value', [0, -1, INT_MIN])
def test_spark_nonpositive_service_output_overrides_inherited_env_not_request_limit(local, value):
    model, runtime, profile, context = _spark_output_fixture(local)
    runtime.metadata['build_environment'] = {'SPARKINFER_MAX_OUTPUT_TOKENS': '70000'}
    profile.parameters['server_max_output_tokens'] = value
    spec = native.SparkInferAdapter().build_launch(model, profile, runtime, context)
    assert spec.env['SPARKINFER_MAX_OUTPUT_TOKENS'] == str(value)
    assert profile.generation == {'max_tokens': 13}


@pytest.mark.parametrize('key', ['context_length', 'server_max_output_tokens', 'draft_proposals'])
@pytest.mark.parametrize('value', [INT_MIN - 1, INT_MAX + 1])
def test_spark_atoi_fields_reject_out_of_int32_range(native_profile, key, value):
    model, profile, runtime = native_profile
    profile.backend = runtime.backend = 'sparkinfer'
    profile.parameters = {'acceleration_mode': 'dflash2', key: value}
    with pytest.raises(ValueError):
        native.validate(model, profile, runtime)


@pytest.mark.parametrize('value', [9, INT_MAX])
def test_ninfer_rejects_concurrency_above_actual_eight_slot_limit(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters['max_concurrency'] = value
    with pytest.raises(ValueError, match='最大并行请求'):
        native.validate(model, profile, runtime)


@pytest.mark.parametrize('value', [1, 8])
def test_ninfer_concurrency_limit_includes_both_endpoints(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters['max_concurrency'] = value
    native.validate(model, profile, runtime)


def test_ninfer_context_uses_native_int_parser_limit(native_profile):
    model, profile, runtime = native_profile
    profile.parameters['context_length'] = INT_MAX + 1
    with pytest.raises(ValueError, match='最大上下文'):
        native.validate(model, profile, runtime)


@pytest.mark.parametrize('value', [4095, '4095', INT_MAX + 1, str(INT_MAX + 1)])
def test_ninfer_explicit_kv_capacity_respects_context_and_integer_domain(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters = {'context_length': 4096, 'kv_capacity': value}
    with pytest.raises(ValueError, match='KV 容量'):
        native.validate(model, profile, runtime)


@pytest.mark.parametrize('value', ['auto', 4096, '4096', INT_MAX, str(INT_MAX)])
def test_ninfer_auto_or_sufficient_kv_capacity_preserves_saved_value(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters = {'context_length': 4096, 'kv_capacity': value}
    native.validate(model, profile, runtime)
    assert profile.parameters['kv_capacity'] == value


@pytest.mark.parametrize('value', ['+4096', ' 4096', '\t+4096'])
def test_ninfer_kv_capacity_accepts_native_sign_and_leading_whitespace(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters = {'context_length': 4096, 'kv_capacity': value}
    native.validate(model, profile, runtime)
    assert profile.parameters['kv_capacity'] == value


@pytest.mark.parametrize('value', ['4_096', '4096.0', '4096 ', ' 4096 ', '+2147483648', ' 4095'])
def test_ninfer_kv_capacity_retains_native_format_range_and_context_checks(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters = {'context_length': 4096, 'kv_capacity': value}
    with pytest.raises(ValueError, match='KV 容量'):
        native.validate(model, profile, runtime)


@pytest.mark.parametrize('parameters', [{}, {'context_length': None}, {'context_length': ''}])
def test_pinned_ninfer_kv_capacity_uses_native_context_default_when_omitted(native_profile, parameters):
    model, profile, runtime = native_profile
    runtime.metadata.update(source_commit=native.PINS['ninfer']['commit'], version_basis='verified_source_commit')
    profile.parameters = {**parameters, 'kv_capacity': 4096}
    with pytest.raises(ValueError, match='KV 容量'):
        native.validate(model, profile, runtime)
    profile.parameters['kv_capacity'] = 8192
    native.validate(model, profile, runtime)
    profile.parameters.update(context_length=4096, kv_capacity=4096)
    native.validate(model, profile, runtime)


@pytest.mark.parametrize('metadata', [
    {},
    {'source_commit': native.PINS['ninfer']['commit']},
    {'source_commit': 'external-source', 'version_basis': 'verified_source_commit'},
])
def test_unverified_ninfer_kv_capacity_does_not_guess_native_context_default(native_profile, metadata):
    model, profile, runtime = native_profile
    runtime.metadata = metadata
    profile.parameters = {'kv_capacity': 4096}
    native.validate(model, profile, runtime)


@pytest.mark.parametrize('value', ['cpu', 'cuda', 'cuda:0', 'cuda:12', 'cpu:0'])
def test_ninfer_conversion_device_accepts_indexed_cpu_cuda(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters['conversion_device'] = value
    native.validate(model, profile, runtime)
    assert profile.parameters['conversion_device'] == value


@pytest.mark.parametrize('value', ['cuda:-1', 'cuda:x', 'other:0', True, 0, 'cuda:0\n'])
def test_ninfer_conversion_device_retains_real_device_syntax_checks(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters['conversion_device'] = value
    with pytest.raises(ValueError, match='转换设备'):
        native.validate(model, profile, runtime)


def test_ninfer_conversion_device_is_editable_instead_of_fixed_choice():
    field = next(item for item in native.NINFER_PREPARATION_FIELDS if item.key == 'conversion_device')
    assert field.type == 'text' and field.default == 'cpu'


@pytest.mark.parametrize('backend', ['ninfer', 'sparkinfer'])
def test_disabled_native_acceleration_keeps_dormant_zero(native_profile, backend):
    model, profile, runtime = native_profile
    profile.backend = runtime.backend = backend
    profile.parameters = {'acceleration_mode': 'none', 'draft_tokens': 0, 'draft_proposals': 0}
    native.validate(model, profile, runtime)
    assert profile.parameters == {'acceleration_mode': 'none', 'draft_tokens': 0, 'draft_proposals': 0}


@pytest.mark.parametrize('value', [0, 6])
def test_enabled_ninfer_mtp_retains_actual_one_to_five_limit(native_profile, value):
    model, profile, runtime = native_profile
    profile.parameters = {'acceleration_mode': 'mtp', 'draft_tokens': value}
    with pytest.raises(ValueError, match='MTP 草稿 Token'):
        native.validate(model, profile, runtime)


@pytest.mark.parametrize('value', [-1, 16, INT_MIN, INT_MAX])
def test_spark_raw_draft_proposals_are_not_rejected_or_clamped_by_app(local, value):
    model, runtime, profile, context = _spark_output_fixture(local)
    draft = Path(model.path).parent / 'draft'
    draft.mkdir()
    (draft / 'config.json').write_text('{"architectures":["DFlash2DraftModel"],"block_size":4}')
    (draft / 'model.safetensors').write_bytes(b'fixture')
    profile.parameters.update(acceleration_mode='dflash2', draft_model=draft.as_posix(), draft_proposals=value)
    spec = native.SparkInferAdapter().build_launch(model, profile, runtime, context)
    assert spec.env['SPARKINFER_DFLASH_PROPOSALS'] == str(value)
    assert profile.parameters['draft_proposals'] == value


def test_spark_dflash_kernel_block_width_remains_a_real_constraint(local):
    model, runtime, profile, _ = _spark_output_fixture(local)
    draft = Path(model.path).parent / 'invalid-draft'
    draft.mkdir()
    (draft / 'config.json').write_text('{"architectures":["DFlash2DraftModel"],"block_size":3}')
    (draft / 'model.safetensors').write_bytes(b'fixture')
    profile.parameters.update(acceleration_mode='dflash2', draft_model=draft.as_posix(), draft_proposals=1)
    state = native.acceleration(model, profile, runtime)
    assert state['status'] == 'unsupported' and 'block_size' in state['message']
    with pytest.raises(ValueError, match='block_size'):
        native.validate(model, profile, runtime)


@pytest.mark.parametrize('tokens', [0, INT_MAX])
def test_llama_mtp_maximum_allows_zero_and_native_int_endpoint(native_profile, tokens):
    _, profile, runtime = native_profile
    profile.backend = runtime.backend = 'llama_cpp'
    profile.parameters = {'mtp_enabled': True, 'mtp_tokens': tokens}
    validate_tokens(profile, runtime)
    field = token_field('llama_cpp')
    assert field.minimum == 0 and field.maximum == INT_MAX


@pytest.mark.parametrize('tokens', [-1, INT_MAX + 1])
def test_llama_mtp_maximum_retains_native_nonnegative_int_domain(native_profile, tokens):
    _, profile, runtime = native_profile
    profile.backend = runtime.backend = 'llama_cpp'
    profile.parameters = {'mtp_enabled': True, 'mtp_tokens': tokens}
    with pytest.raises(ValueError, match='最大预测 Token'):
        validate_tokens(profile, runtime)


def test_llama_native_draft_acceptance_minimum_may_exceed_generated_maximum(native_profile):
    _, profile, runtime = native_profile
    profile.backend = runtime.backend = 'llama_cpp'
    profile.parameters = {'mtp_enabled': True, 'mtp_tokens': 2}
    profile.extra_args = ['--spec-type', 'draft-mtp', '--spec-draft-n-min', '4']
    validate_tokens(profile, runtime)
    assert profile.extra_args == ['--spec-type', 'draft-mtp', '--spec-draft-n-min', '4']
