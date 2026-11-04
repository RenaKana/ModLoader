import json
from pathlib import Path
import struct

import pytest

from modloader.assets import gguf_metadata
from modloader.backends import get_adapters
from modloader.models import LaunchContext, LoadProfile, ModelAsset, RuntimeInstallation
from modloader.mtp import apply_mtp, apply_resident_experts, inspect_mtp, options, weight_info
from modloader.resource_estimates import llama_estimate_args, strata_estimate
from test_strata_preparation import write_pack, write_expert_profile


def write_weights(path, *, mtp=True, architecture='qwen35', embedding=5120):
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    metadata = {'general.architecture': architecture, architecture + '.embedding_length': embedding,
                'general.name': 'Same name MTP (names do not establish support)'}
    names = ['blk.0.attn_q.weight'] + (['blk.64.nextn.eh_proj.weight'] if mtp else [])
    content = b'GGUF' + struct.pack('<IQQ', 3, len(names), len(metadata))
    for key, value in metadata.items():
        content += string(key) + struct.pack('<I', 8 if isinstance(value, str) else 4)
        content += string(value) if isinstance(value, str) else struct.pack('<I', value)
    for name in names:
        content += string(name) + struct.pack('<IQIQ', 1, 1, 0, 0)
    path.write_bytes(content)


@pytest.mark.parametrize('item_kind,payload', [
    (8, struct.pack('<Q', 100) + b'short'),
    (4, b'\0\0'),
])
def test_gguf_skipped_tokenizer_arrays_still_reject_truncation(tmp_path, item_kind, payload):
    path = tmp_path / 'truncated.gguf'
    key = b'tokenizer.ggml.tokens'
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
        + struct.pack('<Q', len(key)) + key + struct.pack('<IIQ', 9, item_kind, 1) + payload)
    with pytest.raises(ValueError, match='超出文件边界'):
        gguf_metadata(path, tensor_names=True)


def base(tmp_path, backend='llama_cpp', mtp=True):
    path = tmp_path / 'model.gguf'
    write_weights(path, mtp=mtp)
    model = ModelAsset(id='model', name='Same name', path=str(path), format='gguf', architecture='qwen35', metadata=gguf_metadata(path))
    runtime = RuntimeInstallation(id='runtime', name='Runtime', backend=backend, path=str(tmp_path / 'engine.exe'),
                                  metadata={'speculative_types': ['draft-mtp'], 'distribution': 'ModLoader'})
    profile = LoadProfile(id='profile', model_id=model.id, runtime_id=runtime.id, name='Original', alias='original', backend=backend)
    context = LaunchContext(port=18764, instance_id='test', work_dir=str(tmp_path / 'work'), data_dir=str(tmp_path))
    return model, profile, runtime, context


def strata(tmp_path):
    model, profile, runtime, context = base(tmp_path, 'strata')
    model.architecture = 'qwen4exp'
    write_weights(Path(model.path), architecture='qwen4exp', embedding=2560)
    model.metadata = {'qwen4exp.block_count': 48, 'qwen4exp.expert_count': 512, 'qwen4exp.embedding_length': 2560}
    root = tmp_path / 'strata'
    (root / 'engine').mkdir(parents=True)
    (root / 'engine/strata.exe').touch()
    write_expert_profile(root / 'data/expert-profile.bin')
    pack = write_pack(tmp_path / 'pack')
    mtp = tmp_path / 'mtp rt'
    mtp.mkdir()
    (mtp / 'dense.bin').write_bytes(b'weights')
    (mtp / 'experts.bin').write_bytes(b'experts')
    (mtp / 'dense.txt').write_text('fc_embedding.weight q8_0 2560 2560 0 7\n')
    runtime.path = str(root)
    runtime.metadata = {'python': 'python.exe'}
    profile.parameters = {'strata_config': {'cwd': str(tmp_path), 'tokenizer': str(pack / 'tokenizer'),
        'args': ['--native', model.path, '--pack', str(pack), '--mtp', str(mtp), '--resident-experts', '--spec', '4', '--spec-min-p', '0.5']}}
    return model, profile, runtime, context, mtp


def test_mtp_detection_reads_weights_not_filename_and_refreshes_changed_file(tmp_path):
    model, profile, runtime, _ = base(tmp_path, mtp=False)
    assert inspect_mtp(model, profile, runtime)['status'] == 'missing'
    write_weights(Path(model.path))
    assert weight_info(model.path)['has_mtp'] is True
    assert inspect_mtp(model, profile, runtime)['status'] == 'builtin'


def test_mtp_can_be_in_later_gguf_shard(tmp_path):
    first, last = tmp_path / 'model-00001-of-00002.gguf', tmp_path / 'model-00002-of-00002.gguf'
    write_weights(first, mtp=False)
    write_weights(last)
    assert weight_info(first)['has_mtp'] is True
    last.unlink()
    with pytest.raises(FileNotFoundError):
        weight_info(first)


def test_missing_gguf_reports_the_weight_path_not_hf_config(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    Path(model.path).unlink()
    with pytest.raises(FileNotFoundError) as error:
        weight_info(model.path)
    assert error.value.filename == model.path
    result = inspect_mtp(model, profile, runtime)
    assert result['status'] == 'unverified'
    assert result['message'] == '文件不存在，无法检查 MTP'
    assert result['diagnostic'] == model.path
    assert 'config.json' not in result['diagnostic']
    assert 'Errno' not in result['message']


def test_original_strata_configuration_and_disabled_overrides(tmp_path):
    model, profile, runtime, context, mtp = strata(tmp_path)
    original = profile.model_dump_json()
    status = inspect_mtp(model, profile, runtime)
    assert status['enabled'] and status['resident_experts'] and status['status'] == 'external'
    adapter = get_adapters()['strata']
    adapter.build_launch(model, profile, runtime, context)
    args = json.loads((Path(context.work_dir) / 'strata.json').read_text())['args']
    assert args[args.index('--mtp') + 1] == str(mtp)
    assert '--resident-experts' in args and args[args.index('--spec') + 1] == '4'
    assert profile.model_dump_json() == original
    profile.parameters.update(mtp_enabled=False, resident_experts=False)
    adapter.build_launch(model, profile, runtime, context)
    args = json.loads((Path(context.work_dir) / 'strata.json').read_text())['args']
    assert '--mtp' not in args and '--resident-experts' not in args
    assert args[args.index('--spec-min-p') + 1] == '0.5'
    assert '--mtp' in profile.parameters['strata_config']['args']  # stored import stays intact


def test_strata_candidates_use_model_id_not_names_or_quantization(tmp_path):
    model, original, runtime, _, mtp = strata(tmp_path)
    new = original.model_copy(deep=True, update={'id': 'new', 'parameters': {}})
    unrelated = original.model_copy(deep=True, update={'id': 'unrelated', 'model_id': 'another-model'})
    assert not inspect_mtp(model, new, runtime, [unrelated.model_dump()])['candidates']
    result = inspect_mtp(model, new, runtime, [original.model_dump(), unrelated.model_dump()])
    assert result['candidates'] == [{'label': original.name, 'path': str(mtp)}]
    assert not result['enabled']  # discovery does not silently enable MTP
    new.parameters.update(mtp_enabled=True, mtp_path=str(mtp), resident_experts=True)
    assert apply_mtp([], model, new, runtime) == ['--mtp', str(mtp)]
    assert apply_resident_experts([], new) == ['--resident-experts']


def test_strata_missing_files_and_incompatible_geometry_fail_before_start(tmp_path):
    model, profile, runtime, _, mtp = strata(tmp_path)
    profile.parameters['mtp_enabled'] = True
    (mtp / 'dense.txt').write_text('fc_embedding.weight q8_0 5120 5120 0 7\n')
    with pytest.raises(ValueError, match='维度'):
        apply_mtp([], model, profile, runtime)
    (mtp / 'experts.bin').unlink()
    with pytest.raises(ValueError, match='experts.bin'):
        apply_mtp([], model, profile, runtime)


def test_resource_preview_uses_effective_mtp_and_resident_flags(tmp_path):
    model, profile, runtime, _, _ = strata(tmp_path)
    profile.parameters.update(mtp_enabled=False, resident_experts=False)
    result = strata_estimate(model, profile, runtime)
    assert not any(item['kind'] == 'ram' for item in result['resources'])
    assert not any('MTP' in note for item in result['resources'] for note in item['notes'])
    profile.parameters.update(mtp_enabled=True, resident_experts=True)
    result = strata_estimate(model, profile, runtime)
    assert any(item['kind'] == 'ram' for item in result['resources'])
    assert any('MTP' in note for item in result['resources'] for note in item['notes'])


def test_llama_estimate_marks_mtp_overhead_as_unestimated(tmp_path):
    _, profile, runtime, _ = base(tmp_path)
    profile.parameters['mtp_enabled'] = True
    assert any('MTP' in note for note in llama_estimate_args(profile, runtime)[1])


def test_llama_switch_generates_real_argv_and_preserves_other_speculation(tmp_path):
    model, profile, runtime, context = base(tmp_path)
    profile.extra_args = ['--spec-type=ngram-simple', '--spec-draft-n-max', '4']
    profile.parameters['mtp_enabled'] = True
    spec = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--spec-type') + 1] == 'ngram-simple,draft-mtp'
    assert spec.argv[spec.argv.index('--spec-draft-n-max') + 1] == '4'
    assert '--spec-draft-model' not in spec.argv  # builtin MTP needs no separate model
    profile.extra_args = ['--spec-type', 'draft-mtp,ngram-simple', '--spec-draft-model', str(tmp_path / 'missing.gguf')]
    profile.parameters['mtp_enabled'] = False
    spec = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--spec-type') + 1] == 'ngram-simple'
    assert '--spec-draft-model' not in spec.argv


@pytest.mark.parametrize('backend', ['strata', 'llama_cpp', 'vllm'])
@pytest.mark.parametrize('explicitly_disabled', [False, True])
def test_disabled_mtp_keeps_dormant_tokens_without_restricting_load(tmp_path, backend, explicitly_disabled):
    from modloader.parameter_rules import validate_profile
    if backend == 'strata':
        model, profile, runtime, _, _ = strata(tmp_path)
        profile.parameters['strata_config']['args'] = ['--spec', '4']
    else:
        model, profile, runtime, _ = base(tmp_path, backend)
    profile.parameters['mtp_tokens'] = 'dormant-old-value'
    if explicitly_disabled:
        profile.parameters['mtp_enabled'] = False
    before = profile.model_dump()
    validate_profile(model, profile, runtime, get_adapters()[backend])
    args = apply_mtp([], model, profile, runtime)
    assert '--mtp' not in args and '--speculative-config' not in args
    assert '--spec-draft-n-max' not in args
    assert profile.model_dump() == before


def test_llama_legacy_options_are_untouched_and_inherited_in_editor(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    profile.extra_args = ['--spec-type=draft-mtp', '--spec-draft-n-max', '4']
    assert options(profile, runtime)['enabled'] is True
    assert apply_mtp(profile.extra_args, model, profile, runtime) == profile.extra_args
    runtime.metadata['speculative_types'] = ['draft-simple']
    assert inspect_mtp(model, profile, runtime)['status'] == 'unsupported'
    profile.parameters['mtp_enabled'] = True
    with pytest.raises(ValueError, match='draft-mtp'):
        apply_mtp(profile.extra_args, model, profile, runtime)


def test_external_llama_mtp_requires_compatible_local_weights(tmp_path):
    model, profile, runtime, _ = base(tmp_path, mtp=False)
    path = tmp_path / 'draft.gguf'
    write_weights(path, embedding=2560)
    profile.parameters.update(mtp_enabled=True, mtp_path=str(path))
    with pytest.raises(ValueError, match='维度'):
        apply_mtp([], model, profile, runtime)
    write_weights(path)
    args = apply_mtp([], model, profile, runtime)
    assert args[args.index('--spec-draft-model') + 1] == str(path)


def hf_model(tmp_path):
    path = tmp_path / 'hf model'
    path.mkdir()
    (path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_next', 'hidden_size': 2048}))
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'mtp.layers.0.eh_proj.weight': 'part.safetensors'}}))
    (path / 'part.safetensors').touch()
    return path


def test_vllm_mtp_preserves_depth_and_quantization_settings(tmp_path):
    model, profile, runtime, context = base(tmp_path, 'vllm')
    model.path, model.format = str(hf_model(tmp_path)), 'hf'
    runtime.path = '/opt/vllm/bin/vllm'
    runtime.metadata['supported_flags'] = ['--speculative-config']
    original = {'method': 'mtp', 'num_speculative_tokens': 5, 'draft_lm_head_quantization': 'fp8'}
    profile.extra_args = ['--speculative-config=' + json.dumps(original)]
    assert inspect_mtp(model, profile, runtime)['status'] == 'builtin'
    profile.parameters['mtp_enabled'] = True
    spec = get_adapters()['vllm'].build_launch(model, profile, runtime, context)
    assert json.loads(spec.argv[spec.argv.index('--speculative-config') + 1]) == original
    assert inspect_mtp(model, profile, runtime)['tokens'] == 5
    profile.parameters['mtp_tokens'] = 7
    spec = get_adapters()['vllm'].build_launch(model, profile, runtime, context)
    assert json.loads(spec.argv[spec.argv.index('--speculative-config') + 1]) == {**original, 'num_speculative_tokens': 7}
    profile.parameters['mtp_enabled'] = False
    spec = get_adapters()['vllm'].build_launch(model, profile, runtime, context)
    assert not any(a.startswith('--speculative-config') for a in spec.argv)


@pytest.mark.parametrize('tokens', [1, 7])
def test_strata_prediction_count_maps_to_verify_window(tmp_path, tokens):
    model, profile, runtime, context, _ = strata(tmp_path)
    before = json.loads(json.dumps(profile.parameters))
    assert inspect_mtp(model, profile, runtime)['tokens'] == 3
    profile.parameters['mtp_tokens'] = tokens  # Editing length also works for an inherited enabled switch.
    get_adapters()['strata'].build_launch(model, profile, runtime, context)
    args = json.loads((Path(context.work_dir) / 'strata.json').read_text())['args']
    assert args[args.index('--spec') + 1] == str(tokens + 1)
    assert args.count('--spec') == 1
    assert profile.parameters['strata_config'] == before['strata_config']
    profile.parameters.pop('mtp_tokens')
    assert options(profile, runtime)['tokens'] == 3


@pytest.mark.parametrize('tokens', [-1, 0, 8, 1.5, True, float('inf')])
def test_strata_rejects_invalid_prediction_count(tmp_path, tokens):
    model, profile, runtime, context, _ = strata(tmp_path)
    profile.parameters['mtp_tokens'] = tokens
    with pytest.raises(ValueError, match='最大预测 Token'):
        get_adapters()['strata'].build_launch(model, profile, runtime, context)
    assert not (Path(context.work_dir) / 'strata.json').exists()


def test_llama_prediction_count_replaces_inherited_flag_and_preserves_acceptance_minimum(tmp_path):
    model, profile, runtime, context = base(tmp_path)
    profile.extra_args = ['--spec-type=draft-mtp', '--spec-draft-n-max=4']
    assert options(profile, runtime)['tokens'] == 4
    profile.parameters['mtp_tokens'] = 12  # No fabricated Strata-style cap on llama.cpp.
    argv = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context).argv
    assert argv[argv.index('--spec-draft-n-max') + 1] == '12'
    assert '--spec-draft-n-max=4' not in argv
    profile.extra_args += ['--spec-draft-n-min', '13']
    argv = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context).argv
    assert argv[argv.index('--spec-draft-n-min') + 1] == '13'
    assert argv[argv.index('--spec-draft-n-max') + 1] == '12'


@pytest.mark.parametrize('backend,args', [('llama_cpp', ['--spec-type', 'draft-eagle3']), ('vllm', ['--speculative-config', '{"method":"eagle3"}'])])
def test_enabling_mtp_does_not_silently_replace_other_draft_methods(tmp_path, backend, args):
    model, profile, runtime, _ = base(tmp_path, backend)
    profile.parameters['mtp_enabled'] = True
    with pytest.raises(ValueError, match='另一种'):
        apply_mtp(args, model, profile, runtime)


def test_transformers_does_not_claim_generate_support_as_serve_support(tmp_path):
    model, profile, runtime, _ = base(tmp_path, 'transformers')
    result = inspect_mtp(model, profile, runtime)
    assert result['status'] == 'unsupported' and 'serve' in result['message']
    profile.parameters['mtp_enabled'] = True
    with pytest.raises(ValueError, match='serve'):
        apply_mtp([], model, profile, runtime)


async def test_runtime_validation_keeps_vllm_help_capabilities(tmp_path, monkeypatch):
    from modloader.backends import installers
    _, _, runtime, _ = base(tmp_path, 'vllm')
    runtime.path = '/opt/vllm/bin/vllm'
    async def command(argv, *args, **kwargs):
        return '--speculative-config --max-model-len' if '--help' in argv else '0.31.0'
    monkeypatch.setattr(installers, '_command', command)
    updated = await installers.validate_runtime(runtime)
    assert '--speculative-config' in updated.metadata['supported_flags']
