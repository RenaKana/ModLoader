import json
import pytest
from modloader.launch_parameters import edit_parameters, format_argv, validate_editor_profile
from modloader.backends import get_adapters
from test_mtp import base


def edit(tmp_path, text, backend='llama_cpp', parameters=None):
    model, profile, runtime, _ = base(tmp_path, backend)
    profile.parameters = parameters or {}
    return edit_parameters(model, profile, runtime, text=text)


@pytest.mark.parametrize('text', ['-c 0 -ngl auto -t -1', '--ctx-size=0 --n-gpu-layers=auto --threads=-1', '["-c","0","-ngl","auto","-t","-1"]'])
def test_aliases_sentinels_and_json(tmp_path, text):
    result = edit(tmp_path, text)
    assert not result['errors']
    assert result['parameters'] == dict(context_length=0, gpu_layers='auto', threads=-1)
    assert result['extra_args'] == []


def test_last_wins_and_settings_become_authoritative(tmp_path):
    result = edit(tmp_path, '--ctx-size 12 -c 64 --future-option "C:\\some path\\file"')
    assert result['parameters']['context_length'] == 64
    assert result['extra_args'] == ['--future-option', 'C:\\some path\\file']
    assert result['unchecked'][0]['flag'] == '--future-option'
    assert result['unchecked'][0]['message'] == '缺少校验规则'
    model, profile, runtime, _ = base(tmp_path)
    profile.parameters = {**result['parameters'], 'context_length': 128}
    profile.extra_args = result['extra_args']
    assert '--ctx-size 128' in edit_parameters(model, profile, runtime)['text']


@pytest.mark.parametrize('text', ['--cache-type-k f1', '--cache-type-v f1', '--flash-attn huh', '--ctx-size -1', '--threads 1.5', '--parallel 0', '--batch-size 0', '--top-p -0.1', '--temp NaN', '--ctx-size', '--cache-type-k='])
def test_invalid_known_values_are_atomic(tmp_path, text):
    result = edit(tmp_path, '--ctx-size 1024 ' + text, parameters={'context_length': 8192})
    assert result['errors']
    assert result['parameters'] == {'context_length': 8192}
    error = result['errors'][-1]
    assert ('--ctx-size 1024 ' + text)[error['start']:error['end']] == text


def test_utf16_diagnostics_and_continuation(tmp_path):
    text = '--future "😀" ^\n--cache-type-k f1'
    result = edit(tmp_path, text)
    error = result['errors'][0]
    assert text.encode('utf-16-le')[error['start'] * 2:error['end'] * 2].decode('utf-16-le') == '--cache-type-k f1'


@pytest.mark.parametrize('backend,text,key,value', [('strata','--kv k8v4 --spec 4','kv_cache','k8v4'), ('vllm','--tensor-parallel-size 3','tensor_parallel_size',3), ('transformers','--dtype bfloat16 --compile','compile',True), ('sglang','--tp-size 2','tensor_parallel_size',2), ('ninfer','--max-context 4096','context_length',4096), ('sparkinfer','--ctx 4096','context_length',4096)])
def test_backend_scoped_mapping(tmp_path, backend, text, key, value):
    result = edit(tmp_path, text, backend)
    assert not result['errors'], result
    assert result['parameters'][key] == value
    assert not result['extra_args']


@pytest.mark.parametrize('shared,key,flag,value', [
    (True, 'context_capacity', '--context-capacity', '8GiB'),
    (False, 'kv_capacity', '--kv-capacity', '200000'),
])
def test_ninfer_capacity_editor_follows_runtime_contract(tmp_path, shared, key, flag, value):
    model, profile, runtime, _ = base(tmp_path, 'ninfer')
    runtime.metadata['supported_flags'] = [flag]
    profile.parameters = {key: value}
    current = edit_parameters(model, profile, runtime)
    assert flag + ' ' + value in current['text']
    assert ('--kv-capacity' if shared else '--context-capacity') not in current['text']
    parsed = edit_parameters(model, profile, runtime, text=flag + ' auto')
    assert not parsed['errors'], parsed
    assert parsed['parameters'][key] == 'auto'
    assert parsed['extra_args'] == []
    profile.parameters = {**parsed['parameters'], key: value}
    assert flag + ' ' + value in edit_parameters(model, profile, runtime)['text']
    removed = edit_parameters(model, profile, runtime, text='--max-context 4096',
                              previous_text=edit_parameters(model, profile, runtime)['text'])
    assert removed['parameters'][key] is None
    assert removed['extra_args'] == []


@pytest.mark.parametrize('text', ['--kv f1', '--max-context 0', '--spec 9', '--spec 1', '--pool-workers -1', '--pool-workers abc', '--pool-workers 1.5'])
def test_strata_rules(tmp_path, text):
    assert edit(tmp_path, text, 'strata')['errors']


@pytest.mark.parametrize('workers', [0, 64])
async def test_strata_pool_workers_survive_editor_save_and_launch(tmp_path, workers):
    import httpx
    from pathlib import Path
    from modloader.app import Controller, build_app
    from modloader.models import LoadProfile
    from test_mtp import strata

    # Zero is automatic; 64 is valid even above the automatic planner's cap.
    model, profile, runtime, context, _ = strata(tmp_path)
    profile.parameters['strata_config']['args'] += ['--pool-workers', str(workers)]
    adapter = get_adapters()['strata']
    legacy = adapter.build_launch(model, profile, runtime, context)
    legacy_config = json.loads(Path(legacy.argv[legacy.argv.index('--config') + 1]).read_text('utf-8'))
    assert legacy_config['args'][legacy_config['args'].index('--pool-workers') + 1] == str(workers)
    controller = Controller(tmp_path / 'pool-worker-data', 'test-secret')
    for key, item in [('models', model), ('runtimes', runtime)]:
        controller.store.put(key, item.model_dump())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                    headers={'Authorization': 'Bearer test-secret'}) as client:
            opened = await client.post('/api/profiles/launch-parameters', json={**profile.model_dump(), 'preview': True})
            assert opened.status_code == 200, opened.text
            assert not opened.json()['errors'], opened.json()
            parsed = await client.post('/api/profiles/launch-parameters', json={**profile.model_dump(), 'text': opened.json()['text'], 'preview': True})
            result = parsed.json()
            assert parsed.status_code == 200 and not result['errors'], result
            assert not result.get('preview_error'), result
            assert result['parameters']['pool_workers'] == workers
            assert '--pool-workers' not in result['extra_args']
            saved = await client.post('/api/profiles', json={**profile.model_dump(),
                **{key: result[key] for key in ('parameters', 'generation', 'extra_args', 'parameter_editor_version')}})
            assert saved.status_code == 200, saved.text
            stored = LoadProfile(**saved.json())
            assert stored.parameters['pool_workers'] == workers
            assert not edit_parameters(model, stored, runtime)['errors']
            launch = adapter.build_launch(model, stored, runtime, context)
            config = json.loads(Path(launch.argv[launch.argv.index('--config') + 1]).read_text('utf-8'))
            assert config['args'].count('--pool-workers') == 1
            assert config['args'][config['args'].index('--pool-workers') + 1] == str(workers)
    finally:
        await controller.close()


def test_other_backend_does_not_claim_llama_flags(tmp_path):
    result = edit(tmp_path, '--ctx-size 0', 'transformers')
    assert not result['errors']
    assert result['extra_args'] == ['--ctx-size', '0']
    assert result['unchecked']


def test_windows_roundtrip_and_shell_rejection(tmp_path):
    args = ['--future', 'C:\\some path\\', '--json', '{"text":"a b"}', '--empty', '']
    result = edit(tmp_path, format_argv(args))
    assert result['extra_args'] == args
    assert edit(tmp_path, '--ctx-size 1024 && calc.exe')['errors']
    assert edit(tmp_path, '--ctx-size "broken')['errors']


def test_full_command_keeps_manager_identity(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    result = edit_parameters(model, profile, runtime, text=format_argv([runtime.path, '--model', model.path, '--host', '127.0.0.1', '--port', '18888', '--alias', profile.alias, '-c', '128']))
    assert not result['errors'], result
    assert result['extra_args'] == []
    assert edit_parameters(model, profile, runtime, text='--host 0.0.0.0')['errors']
    assert edit_parameters(model, profile, runtime, text='--model another.gguf')['errors']
    assert edit_parameters(model, profile, runtime, text='powershell.exe -c whoami')['errors']


@pytest.mark.parametrize('model_path,tokenizer,target,expected_tokenizer', [
    ('/models/fixture directory', None, '/models/fixture directory', '/models/fixture directory/tokenizer.json'),
    ('/models/fixture.gguf', None, '/models/fixture.gguf', '/models/tokenizer.json'),
    ('/models/fixture directory', '/tokenizers/selected.json', '/models/fixture directory', '/tokenizers/selected.json'),
    ('D:\\models\\fixture directory', None, '/custom-mount/models/fixture directory', '/custom-mount/models/fixture directory/tokenizer.json'),
    ('D:\\models\\fixture.gguf', 'E:\\tokenizers\\selected.json', '/custom-mount/models/fixture.gguf', '/other-mount/tokenizers/selected.json'),
])
def test_spark_generated_commands_roundtrip_without_managed_extras(tmp_path, monkeypatch, model_path, tokenizer, target, expected_tokenizer):
    from pathlib import Path
    from modloader import platforms
    from modloader.backends import adapters, native_engines as native
    from modloader.models import ModelLocation

    model, profile, runtime, context = base(tmp_path, 'sparkinfer')
    model.path = model_path
    model.format = 'gguf' if model_path.endswith('.gguf') else 'safetensors'
    if model_path.startswith('/'):
        model.location = ModelLocation(platform='wsl2', distribution='ModLoader')
    runtime.platform = 'wsl2'
    runtime.path = '/opt/spark/build/server/sparkinfer_server'
    profile.parameters = {'context_length': 4096}
    if tokenizer:
        profile.parameters['tokenizer'] = tokenizer
    original_parameters = dict(profile.parameters)
    conversions = {str(Path(model_path).resolve()): target}
    if tokenizer and not tokenizer.startswith('/'):
        conversions[str(Path(tokenizer).resolve())] = expected_tokenizer

    def wslpath(argv, **kwargs):
        assert argv[:7] == ['wsl.exe', '--distribution', 'ModLoader', '--exec', 'wslpath', '-a', '-u']
        return conversions[argv[-1]]

    def resource_info(path, runtime):
        if path == target:
            return {'exists': True, 'is_dir': model.format != 'gguf', 'is_file': model.format == 'gguf'}
        assert path == expected_tokenizer
        return {'exists': True, 'is_dir': False, 'is_file': True}

    monkeypatch.setattr(platforms, '_run', wslpath)
    monkeypatch.setattr(native, 'resource_info', resource_info)
    monkeypatch.setattr(adapters, 'model_exists', lambda model: True)
    adapter = get_adapters()['sparkinfer']
    expected = adapter.build_launch(model, profile, runtime, context)
    assert expected.argv[expected.argv.index('-m') + 1] == target
    tokenizer_index = expected.argv.index('--tokenizer') + 1
    assert expected.argv[tokenizer_index] == expected_tokenizer

    for _ in range(2):
        result = edit_parameters(model, profile, runtime, text=format_argv(expected.argv))
        assert not result['errors'], result
        assert result['parameters'] == original_parameters
        assert result['extra_args'] == []
        profile = profile.model_copy(update={key: result[key] for key in ('parameters', 'generation', 'extra_args', 'parameter_editor_version')})
        assert adapter.build_launch(model, profile, runtime, context) == expected
        assert edit_parameters(model, profile, runtime)['text'] == result['text']

    for flag, value in [('--tokenizer', expected_tokenizer + '.other'), ('-m', target + '.other')]:
        changed = list(expected.argv)
        changed[changed.index(flag) + 1] = value
        refused = edit_parameters(model, profile, runtime, text=format_argv(changed))
        assert refused['errors'][0]['flag'] == flag
        assert refused['parameters'] == original_parameters
        assert refused['extra_args'] == []


@pytest.mark.parametrize('include_managed_paths', [False, True])
def test_spark_parameter_fragment_keeps_editing_available_without_launch_resources(tmp_path, monkeypatch, include_managed_paths):
    from modloader.backends import native_engines as native
    model, profile, runtime, _ = base(tmp_path, 'sparkinfer')
    runtime.platform = 'wsl2'

    def unavailable(*args, **kwargs):
        raise AssertionError('A mapped parameter fragment must not probe launch resources')

    monkeypatch.setattr(native, 'model_target_path', unavailable)
    monkeypatch.setattr(native, 'resource_info', unavailable)
    argv = ['--ctx', '8192']
    if include_managed_paths:
        profile.parameters = {'tokenizer': 'E:\\tokenizers\\selected.json'}
        argv += ['-m', model.path, '--tokenizer', profile.parameters['tokenizer']]
    result = edit_parameters(model, profile, runtime, text=format_argv(argv))
    assert not result['errors']
    assert result['parameters'] == {**profile.parameters, 'context_length': 8192}


def test_generation_is_consumed_and_reaches_llama_launch(tmp_path):
    model, profile, runtime, context = base(tmp_path)
    result = edit_parameters(model, profile, runtime, text='--temp .7 --top-p .9 --top-k 40 --seed -1')
    assert not result['errors']
    profile = profile.model_copy(update={key:result[key] for key in ('parameters','generation','extra_args','parameter_editor_version')})
    spec = get_adapters()['llama_cpp'].build_launch(model, profile, runtime, context)
    assert spec.argv[spec.argv.index('--temp')+1] == '0.7'


def test_editor_validation_is_opt_in_and_catches_raw_known_errors(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    profile.extra_args = ['--cache-type-k', 'f1']
    validate_editor_profile(model, profile, runtime)
    profile.parameter_editor_version = 1
    with pytest.raises(ValueError, match='缓存'):
        validate_editor_profile(model, profile, runtime)


def test_custom_template_roundtrips_without_duplicating_fixed_argv(tmp_path):
    from test_custom_backend import engine, profile_for
    from modloader.models import LaunchContext
    runtime = engine(tmp_path, args=['server.py', '--weights', '{model_path}', '--bind', '{host}', '--port', '{port}', '--ctx', '{context_length}'])
    model, profile = profile_for(tmp_path, runtime, parameters={'context_length': 2048})
    spec = get_adapters()['custom'].build_launch(model, profile, runtime, LaunchContext(port=12345, instance_id='p', work_dir=str(tmp_path), data_dir=str(tmp_path)))
    result = edit_parameters(model, profile, runtime, text=format_argv(spec.argv))
    assert not result['errors'], result
    assert not result['extra_args']
    assert result['parameters']['context_length'] == 2048


@pytest.mark.parametrize('backend,text,expected', [('llama_cpp','--spec-type draft-mtp --spec-draft-n-max 3', {'mtp_enabled':True,'mtp_tokens':3}), ('strata','--spec 4', {'spec':4}), ('vllm','--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\'', {'mtp_enabled':True,'mtp_tokens':3}), ('sglang','--speculative-algorithm NEXTN --speculative-num-steps 3', {'acceleration_mode':'mtp','speculative_num_steps':3}), ('ninfer','--spec mtp --draft-tokens 3', {'acceleration_mode':'mtp','draft_tokens':3})])
def test_acceleration_consumes_managed_options(tmp_path, backend, text, expected):
    result = edit(tmp_path, text, backend)
    assert not result['errors'], result
    for key, value in expected.items():
        assert result['parameters'][key] == value
    assert not result['extra_args']


async def test_editor_api_roundtrip_and_save_rejection(tmp_path):
    import httpx
    from modloader.app import Controller, build_app
    model, profile, runtime, _ = base(tmp_path)
    controller = Controller(tmp_path / 'api-data', 'test-secret')
    for key, item in [('models',model),('runtimes',runtime)]:
        controller.store.put(key,item.model_dump())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),base_url='http://testserver',headers={'Authorization':'Bearer test-secret'}) as client:
            parsed = await client.post('/api/profiles/launch-parameters',json={**profile.model_dump(),'text':'--cache-type-k f1'})
            assert parsed.status_code == 200
            assert parsed.json()['errors']
            bad = await client.post('/api/profiles',json={**profile.model_dump(),'parameter_editor_version':1,'extra_args':['--cache-type-k','f1']})
            assert bad.status_code == 400
            good = await client.post('/api/profiles/launch-parameters',json={**profile.model_dump(),'text':'--temp .7 --threads 0','preview':True})
            assert good.status_code == 200, good.text
            result = good.json()
            assert result['argv'][result['argv'].index('--temp')+1] == '0.7'
            again = await client.post('/api/profiles/launch-parameters',json={**profile.model_dump(),**{key:result[key] for key in ('parameters','generation','extra_args')},'text':result['text']})
            assert not again.json()['errors']
            assert not again.json()['extra_args']
            unknown = await client.post('/api/profiles',json={**profile.model_dump(),'parameter_editor_version':1,'extra_args':['--future-version-flag','x']})
            assert unknown.status_code == 200
    finally:
        await controller.close()


async def test_strata_preview_returns_effective_config_before_temp_cleanup(tmp_path):
    from test_mtp import strata
    from modloader.runtime import RuntimeManager
    from modloader.models import AppPaths
    model, profile, runtime, _, _ = strata(tmp_path)
    paths = AppPaths(tmp_path / 'preview-data')
    manager = RuntimeManager(paths, get_adapters())
    result = await manager.preview(model, profile, runtime)
    assert result['config']['exe']
    assert '--native' in result['config']['args']
    from pathlib import Path
    assert not Path(result['argv'][result['argv'].index('--config')+1]).exists()
    parsed = edit_parameters(model,profile,runtime,text=format_argv([result['config']['exe'],*result['config']['args']]))
    assert not parsed['errors'], parsed


def test_missing_strata_source_does_not_block_parameter_editing(tmp_path):
    model, profile, runtime, _ = base(tmp_path, 'strata')
    profile.parameters = {'configuration': str(tmp_path/'missing.json'), 'context_length': 4096}
    result = edit_parameters(model,profile,runtime,text='--max-context 8192')
    assert not result['errors']
    assert result['parameters']['context_length'] == 8192
    assert result['preview_error']


def test_strata_spec_edit_updates_enabled_mtp_tokens(tmp_path):
    result = edit(tmp_path,'--spec 6','strata',{'mtp_enabled':True,'mtp_tokens':3})
    assert not result['errors']
    assert result['parameters']['mtp_tokens'] == 5
    assert '--spec 6' in result['text']


@pytest.mark.parametrize('text',['--compile=true','--compile false','--ctx-size --threads 2'])
def test_known_arity_does_not_silently_accept_extra_values(tmp_path,text):
    assert edit(tmp_path,text,'transformers' if 'compile' in text else 'llama_cpp')['errors']


def test_known_gpu_error_marks_its_own_flag_and_value(tmp_path):
    text = '--ctx-size 1024 --tensor-split -1,2 --threads 2'
    result = edit(tmp_path,text)
    assert result['errors']
    error = result['errors'][0]
    assert text[error['start']:error['end']] == '--tensor-split -1,2'


def test_new_editor_save_validates_mapped_values_and_cannot_drop_version(tmp_path):
    from modloader.parameter_rules import validate_profile
    model, profile, runtime, _ = base(tmp_path)
    profile.parameter_editor_version = 1
    profile.parameters = {'cache_type_k':'f1'}
    with pytest.raises(ValueError):
        validate_profile(model,profile,runtime,get_adapters()['llama_cpp'])


async def test_editor_version_cannot_be_cleared_to_bypass_existing_profile_validation(tmp_path):
    import httpx
    from modloader.app import Controller, build_app
    model, profile, runtime, _ = base(tmp_path)
    controller = Controller(tmp_path/'version-data','test-secret')
    for key,item in [('models',model),('runtimes',runtime)]:
        controller.store.put(key,item.model_dump())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),base_url='http://testserver',headers={'Authorization':'Bearer test-secret'}) as client:
            saved = await client.post('/api/profiles',json={**profile.model_dump(),'parameter_editor_version':1})
            assert saved.status_code == 200
            bad = await client.patch('/api/profiles/'+saved.json()['id'],json={'parameter_editor_version':None,'parameters':{'cache_type_k':'f1'}})
            assert bad.status_code == 400
    finally:
        await controller.close()


def test_editor_profile_cannot_keep_a_mapped_extra_that_overrides_settings(tmp_path):
    model, profile, runtime, _ = base(tmp_path)
    profile.parameter_editor_version=1
    profile.parameters={'context_length':4096}
    profile.extra_args=['--ctx-size','2048']
    with pytest.raises(ValueError):
        validate_editor_profile(model,profile,runtime)


def test_llama_editor_mtp_token_argument_is_present_even_without_enabling_method(tmp_path):
    model,profile,runtime,context=base(tmp_path)
    result=edit_parameters(model,profile,runtime,text='--spec-draft-n-max 6')
    profile=profile.model_copy(update={key:result[key] for key in ('parameters','generation','extra_args','parameter_editor_version')})
    argv=get_adapters()['llama_cpp'].build_launch(model,profile,runtime,context).argv
    assert argv[argv.index('--spec-draft-n-max')+1]=='6'


def test_custom_binding_flag_cannot_redirect_model_in_args_only_input(tmp_path):
    from test_custom_backend import engine, profile_for
    runtime=engine(tmp_path,args=['server.py','--weights','{model_path}','--host','{host}','--port','{port}'])
    model,profile=profile_for(tmp_path,runtime)
    result=edit_parameters(model,profile,runtime,text='--weights other.bin')
    assert result['errors']


def test_strata_editor_settings_override_all_inherited_mapped_duplicates(tmp_path):
    from test_mtp import strata
    model,profile,runtime,context,_=strata(tmp_path)
    profile.parameters['strata_config']['args'] += ['--max-context','1024','--max-context','2048']
    profile.parameters['context_length']=8192
    profile.parameter_editor_version=1
    spec=get_adapters()['strata'].build_launch(model,profile,runtime,context)
    from pathlib import Path
    cfg=json.loads(Path(spec.argv[spec.argv.index('--config')+1]).read_text('utf-8'))
    assert cfg['args'].count('--max-context')==1
    assert cfg['args'][cfg['args'].index('--max-context')+1]=='8192'


def test_runtime_flags_do_not_claim_value_validation_for_text_grammar(tmp_path):
    model,profile,runtime,_=base(tmp_path,'vllm')
    runtime.metadata['supported_flags']=['--dtype']
    result=edit_parameters(model,profile,runtime,text='--dtype future-dtype')
    assert not result['errors']
    assert result['parameters']['dtype']=='future-dtype'
    assert result['unchecked'][0]['flag']=='--dtype'
    assert result['unchecked'][0]['message']=='可用模式未校验'


def test_full_text_deletion_clears_only_previously_displayed_fields(tmp_path):
    model,profile,runtime,context=base(tmp_path)
    profile.parameters={'context_length':4096,'threads':4,'startup_timeout':90}
    profile.generation={'temperature':.7,'top_p':.8}
    result=edit_parameters(model,profile,runtime,text='--threads 4',previous_text='--ctx-size 4096 --threads 4')
    assert not result['errors']
    assert result['parameters']=={'context_length':None,'threads':4,'startup_timeout':90}
    assert result['generation']==profile.generation
    assert '--ctx-size' not in result['text']
    changed=profile.model_copy(update={key:result[key] for key in ('parameters','generation','extra_args','parameter_editor_version')})
    argv=get_adapters()['llama_cpp'].build_launch(model,changed,runtime,context).argv
    assert '--ctx-size' not in argv
    assert '--ctx-size' not in edit_parameters(model,changed,runtime)['text']


def test_fragment_paste_keeps_unmentioned_settings_and_bad_full_edit_is_atomic(tmp_path):
    model,profile,runtime,_=base(tmp_path)
    profile.parameters={'context_length':4096,'cache_type_k':'f16'}
    fragment=edit_parameters(model,profile,runtime,text='--threads 2')
    assert fragment['parameters']['context_length']==4096
    bad=edit_parameters(model,profile,runtime,text='--cache-type-k f1',previous_text='--ctx-size 4096 --cache-type-k f16')
    assert bad['errors']
    assert bad['parameters']==profile.parameters


def test_deleting_launch_generation_does_not_delete_other_generation_defaults(tmp_path):
    model,profile,runtime,_=base(tmp_path)
    profile.generation={'temperature':.7,'top_p':.9,'other_request_default':'retain'}
    result=edit_parameters(model,profile,runtime,text='',previous_text='--temp .7')
    assert result['generation']=={'top_p':.9,'other_request_default':'retain'}


@pytest.mark.parametrize('key,value',[('cache_type_k',False),('flash_attention',True),('threads',False)])
def test_editor_saved_values_do_not_hide_invalid_types_during_rendering(tmp_path,key,value):
    model,profile,runtime,_=base(tmp_path)
    profile.parameter_editor_version=1
    profile.parameters={key:value}
    with pytest.raises(ValueError):
        validate_editor_profile(model,profile,runtime)


@pytest.mark.parametrize('supported,allowed', [
    (['--spec-min-p', '--prefill'], False),
    (None, True),
    (['--spec-min-p', '--prefill', '--dws'], True),
])
async def test_editor_unknown_flags_match_existing_strata_launch_contract(tmp_path, supported, allowed):
    import httpx
    from modloader.app import Controller, build_app
    from test_mtp import strata
    model, profile, runtime, context, _ = strata(tmp_path)
    if supported is not None:
        runtime.metadata['supported_flags'] = supported
    text = '--dws 2'
    parsed = edit_parameters(model, profile, runtime, text=text)
    assert bool(parsed['errors']) is not allowed
    if allowed:
        assert parsed['extra_args'] == ['--dws', '2']
        assert parsed['unchecked'] == [{'flag': '--dws', 'message': '缺少校验规则'}]
    else:
        assert parsed['errors'][0]['message'] == '此运行时的 --help 未列出参数 --dws'
        error = parsed['errors'][0]
        assert text[error['start']:error['end']] == text
        assert not parsed['unchecked']
    profile.extra_args = ['--dws', '2']
    # Check the pre-existing launch contract before enabling editor validation.
    if allowed:
        spec = get_adapters()['strata'].build_launch(model, profile, runtime, context)
        from pathlib import Path
        config = json.loads(Path(spec.argv[spec.argv.index('--config') + 1]).read_text('utf-8'))
        assert config['args'][config['args'].index('--dws') + 1] == '2'
    else:
        with pytest.raises(ValueError, match='--help 未列出参数 --dws'):
            get_adapters()['strata'].build_launch(model, profile, runtime, context)
    editor_profile = profile.model_copy(update={'parameter_editor_version': 1})
    if allowed:
        assert get_adapters()['strata'].build_launch(model, editor_profile, runtime, context).argv
    else:
        with pytest.raises(ValueError, match='--help 未列出参数 --dws'):
            get_adapters()['strata'].build_launch(model, editor_profile, runtime, context)
    controller = Controller(tmp_path / 'flag-contract-data', 'test-secret')
    for key, item in [('models', model), ('runtimes', runtime)]:
        controller.store.put(key, item.model_dump())
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver', headers={'Authorization': 'Bearer test-secret'}) as client:
            response = await client.post('/api/profiles', json={**profile.model_dump(), 'parameter_editor_version': 1})
            assert response.status_code == (200 if allowed else 400), response.text
            # Legacy save contract remains unchanged, even though launch rejects it.
            if not allowed:
                legacy = await client.post('/api/profiles', json=profile.model_dump())
                assert legacy.status_code == 200, legacy.text
    finally:
        await controller.close()
