"""Regressions from the version-matched startup parameter audit."""
import json
from pathlib import Path
import sys

import pytest

from modloader.backends import get_adapters
from modloader.backends.adapters import _extra
from modloader.launch_parameters import edit_parameters, format_argv, validate_editor_profile
from modloader.models import ModelAsset, LoadProfile, RuntimeInstallation, LaunchContext
from modloader.parameter_rules import validate_generation, validate_profile
from test_mtp import base, strata


@pytest.mark.parametrize('backend,text,key,value', [
    ('llama_cpp', '--ubatch-size 0', 'ubatch_size', 0),
    ('llama_cpp', '--n-gpu-layers -2', 'gpu_layers', -2),
    ('llama_cpp', '--threads -2', 'threads', -2),
    ('llama_cpp', '--threads-batch -3', 'threads_batch', -3),
    *[('llama_cpp', '--flash-attn ' + alias, 'flash_attention', canonical)
      for alias, canonical in [('1','on'), ('true','on'), ('enabled','on'), ('0','off'), ('false','off'), ('disabled','off'), ('-1','auto')]],
    ('vllm', '--max-model-len auto', 'context_length', -1),
    ('vllm', '--max-model-len AUTO', 'context_length', -1),
    ('vllm', '--max-model-len -1', 'context_length', -1),
    ('vllm', '--max-model-len 32K', 'context_length', 32768),
    ('vllm', '--max-model-len 25.6k', 'context_length', 25600),
    ('sglang', '--context-length 32Ki', 'context_length', 32768),
    ('sglang', '--context-length 25.6k', 'context_length', 25600),
    ('sglang', '--context-length 32_768', 'context_length', 32768),
    ('transformers', '--dtype half', 'dtype', 'float16'),
    ('transformers', '--dtype float', 'dtype', 'float32'),
])
def test_legal_cli_values_roundtrip_to_settings(tmp_path, backend, text, key, value):
    model, profile, runtime, _ = base(tmp_path, backend)
    result = edit_parameters(model, profile, runtime, text=text)
    assert not result['errors'], result
    assert result['parameters'][key] == value
    assert not result['extra_args']
    profile.parameters = result['parameters']
    profile.parameter_editor_version = 1
    validate_editor_profile(model, profile, runtime)
    assert edit_parameters(model, profile, runtime)['parameters'][key] == value


@pytest.mark.parametrize('backend,text', [('vllm','--max-model-len 0'), ('sglang','--context-length 32K'),
    ('sglang','--context-length 1.5Ki'), ('vllm','--max-model-len 1.5K'), ('llama_cpp','--cache-type-k f1'), ('llama_cpp','--ctx-size 2147483648')])
def test_real_cli_boundaries_remain_errors(tmp_path, backend, text):
    model, profile, runtime, _ = base(tmp_path, backend)
    assert edit_parameters(model, profile, runtime, text=text)['errors']


@pytest.mark.parametrize('backend,values', [
    ('strata', {'max_tokens': 0}), ('strata', {'max_tokens': -1}), ('strata', {'top_k': 0}), ('strata', {'top_k': 65}),
    ('llama_cpp', {'max_tokens': 0, 'top_p': 2}), ('transformers', {'top_p': 0}),
    ('vllm', {'seed': -(2**63)}), ('vllm', {'seed': 2**63-1}),
    ('sglang', {'seed': 2**63-1}), ('transformers', {'seed': 2**64-1}),
    ('ninfer', {'seed': 2**64-1}), ('sparkinfer', {'seed': 2**63-1}),
])
def test_request_values_keep_engine_semantics(backend, values):
    validate_generation(values, backend)


@pytest.mark.parametrize('backend,values', [
    ('vllm', {'temperature': 3}), ('llama_cpp', {'repeat_penalty': 0}),
    ('llama_cpp', {'repeat_penalty': 1e-50}), ('llama_cpp', {'repeat_penalty': 1e50}),
    ('llama_cpp', {'max_tokens': 2**31}), ('ninfer', {'max_tokens': 2**31}), ('sparkinfer', {'max_tokens': 2**31}),
    ('vllm', {'seed': 2**63}), ('vllm', {'seed': -2**63-1}), ('sglang', {'seed': 2**63}),
    ('transformers', {'seed': 2**64}), ('ninfer', {'seed': 2**64}), ('sparkinfer', {'seed': 2**63}),
])
def test_request_hard_boundaries_reject_before_forwarding(backend, values):
    with pytest.raises(ValueError):
        validate_generation(values, backend)


@pytest.mark.parametrize('backend,key', [('sglang', 'speculative_num_steps'), ('sglang', 'speculative_eagle_topk'),
    ('sglang', 'speculative_num_draft_tokens'), ('ninfer', 'draft_tokens')])
def test_disabled_drafting_does_not_reject_saved_zero(tmp_path, backend, key):
    model, profile, runtime, _ = base(tmp_path, backend)
    profile.parameters = {'acceleration_mode': 'none', key: 0}
    profile.parameter_editor_version = 1
    validate_editor_profile(model, profile, runtime)
    assert key not in edit_parameters(model, profile, runtime)['text']


@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
def test_generated_wsl_binding_uses_actual_path_converter(tmp_path, monkeypatch, backend):
    from modloader import platforms
    from modloader.backends import adapters, sglang
    model, profile, runtime, context = base(tmp_path, backend)
    runtime.platform = 'wsl2'
    runtime.path = '/opt/bin/vllm' if backend == 'vllm' else '/opt/bin/python'
    target = '/custom-mount/d/models/model.gguf'
    monkeypatch.setattr(platforms, 'model_target_path', lambda *_: target)
    monkeypatch.setattr(adapters if backend == 'vllm' else sglang, 'model_target_path', lambda *_: target)
    adapter = get_adapters()[backend]
    monkeypatch.setattr(adapter, '_validate', lambda *_: None)
    launch = adapter.build_launch(model, profile, runtime, context)
    result = edit_parameters(model, profile, runtime, text=format_argv(launch.argv))
    assert not result['errors'], result
    assert '--model-path' not in result['extra_args']


@pytest.mark.parametrize('template', ['-- {{ messages[0].content }}', '--\n{{ messages[0].content }}'])
@pytest.mark.parametrize('inline', [False, True])
def test_literal_option_looking_template_is_not_a_flag(tmp_path, template, inline):
    model, profile, runtime, _ = base(tmp_path)
    runtime.metadata['supported_flags'] = ['--jinja', '--chat-template']
    profile.extra_args = ['--jinja', *(['--chat-template=' + template] if inline else ['--chat-template', template])]
    assert _extra(profile, runtime) == profile.extra_args
    result = edit_parameters(model, profile, runtime, text=format_argv(profile.extra_args))
    assert not result['errors'], result
    assert result['extra_args'] == profile.extra_args


@pytest.mark.parametrize('literal', ['--host', '--ctx-size'])
def test_template_value_arity_protects_literal_flags_without_allowing_real_overrides(tmp_path, literal):
    model, profile, runtime, _ = base(tmp_path)
    runtime.metadata['supported_flags'] = ['--jinja', '--chat-template']
    profile.extra_args = ['--jinja', '--chat-template', literal]
    assert _extra(profile, runtime) == profile.extra_args
    profile.parameter_editor_version = 1
    validate_editor_profile(model, profile, runtime)
    result = edit_parameters(model, profile, runtime, text=format_argv(profile.extra_args))
    assert not result['errors'], result
    assert result['extra_args'] == profile.extra_args
    profile.extra_args += ['--host', '0.0.0.0']
    with pytest.raises(ValueError, match='保留'):
        _extra(profile, runtime)


def test_deleting_a_setting_does_not_mistake_template_text_for_that_setting(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    profile.parameters = {'context_length': 8192}
    before = '--ctx-size 8192 --jinja --chat-template "--ctx-size"'
    after = '--jinja --chat-template "--ctx-size"'
    result = edit_parameters(model, profile, runtime, text=after, previous_text=before)
    assert not result['errors'], result
    assert result['parameters']['context_length'] is None
    assert result['extra_args'] == ['--jinja', '--chat-template', '--ctx-size']


def test_custom_directory_fragment_and_zero_context(tmp_path):
    model = ModelAsset(id='m', name='m', path=str(tmp_path), format='custom')
    runtime = RuntimeInstallation(id='r', name='r', backend='custom', path=sys.executable, metadata={'custom': {
        'args': ['--model', '{model_path}', '--asset-root', '{model_dir}', '--ctx-size', '{context_length}', '--port', '{port}']}})
    profile = LoadProfile(id='p', name='p', alias='p', backend='custom', model_id='m', runtime_id='r', parameters={'context_length': 0})
    context = LaunchContext(port=12345, instance_id='i', data_dir=str(tmp_path), work_dir=str(tmp_path))
    launch = get_adapters()['custom'].build_launch(model, profile, runtime, context)
    result = edit_parameters(model, profile, runtime, text=format_argv(launch.argv[1:]))
    assert not result['errors'], result
    assert result['parameters']['context_length'] == 0
    assert any(row['flag'] == '--ctx-size' for row in result['unchecked'])


@pytest.mark.parametrize('value', ['auto:16384', 'auto:32768'])
def test_strata_prefill_automatic_cap_reaches_native_launch(tmp_path, value):
    model, profile, runtime, context, _ = strata(tmp_path)
    profile.extra_args = ['--prefill', value]
    launch = get_adapters()['strata'].build_launch(model, profile, runtime, context)
    config = json.loads(Path(launch.argv[launch.argv.index('--config') + 1]).read_text('utf-8'))
    assert config['args'][config['args'].index('--prefill') + 1] == value


def test_strata_native_context_has_actual_integer_bound(tmp_path):
    model, profile, runtime, _ = base(tmp_path, 'strata')
    profile.parameters['context_length'] = 2**31-8
    with pytest.raises(ValueError):
        validate_profile(model, profile, runtime, get_adapters()['strata'])


def test_strata_expert_cache_rejects_native_negative_domain(tmp_path):
    model, profile, runtime, context, _ = strata(tmp_path)
    result = edit_parameters(model, profile, runtime, text='--expert-cache -2')
    assert result['errors']
    profile.parameters['expert_cache'] = '-2'
    with pytest.raises(ValueError, match='专家缓存'):
        validate_profile(model, profile, runtime, get_adapters()['strata'])
    profile.parameters.pop('expert_cache')
    profile.extra_args = ['--expert-cache', '-2']
    with pytest.raises(ValueError, match='专家缓存'):
        get_adapters()['strata'].build_launch(model, profile, runtime, context)


@pytest.mark.parametrize('top_k', [0, -2])
def test_sglang_greedy_ignores_top_k_domain_but_sampling_does_not(top_k):
    validate_generation({'temperature': 0, 'top_k': top_k}, 'sglang')
    with pytest.raises(ValueError):
        validate_generation({'temperature': .5, 'top_k': top_k}, 'sglang')


@pytest.mark.parametrize('backend,key,alias,canonical', [('transformers','dtype','half','float16'), ('llama_cpp','flash_attention','enabled','on')])
def test_existing_alias_profile_can_open_and_save_without_injecting_defaults(tmp_path, backend, key, alias, canonical):
    model, profile, runtime, _ = base(tmp_path, backend)
    profile.parameters = {key: alias}
    profile.parameter_editor_version = 1
    validate_editor_profile(model, profile, runtime)
    opened = edit_parameters(model, profile, runtime)
    assert not opened['errors']
    assert opened['parameters'] == {key: canonical}
    assert profile.parameters == {key: alias}
