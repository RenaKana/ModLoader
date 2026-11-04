"""Reproduce the real Unsloth pipeline failure before conversion or GPU load."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.backends import get_adapters, native_engines as native
from modloader.backends.ninfer_tokenizer import RESOURCE_INSPECTION_CODE, validate_pipeline, validate_resources
from modloader import engine_preparation as preparation
from modloader.models import AppPaths
from test_native_engines import MemoryStore, local, tokenizer_documents, write_converter


artifact_info = native.artifact_info
TOKENIZERS = json.loads((Path(__file__).parent / 'fixtures/ninfer-qwen38-tokenizers.json').read_text(encoding='utf-8'))
namespace = {}
exec(RESOURCE_INSPECTION_CODE, namespace)
inspect_resources = namespace['inspect_tokenizer_resources']


def contract(pipeline, config=None, generation=None):
    tokenizer, defaults = tokenizer_documents(pipeline)
    return inspect_resources(tokenizer, defaults if config is None else config,
                             {'eos_token_id': [2, 3]} if generation is None else generation, 4)


def write_tokenizer(path, pipeline):
    tokenizer, _ = tokenizer_documents(pipeline)
    path.write_text(json.dumps(tokenizer))


@pytest.fixture
def pinned(local):
    runtime = local[1]
    runtime.metadata.update(source_commit=native.PINS['ninfer']['commit'], version_basis='verified_source_commit')
    return local


def test_real_tokenizer_pipelines_are_not_semantically_equivalent_exports():
    validate_pipeline(TOKENIZERS['official_pipeline'], 'same-base official tokenizer.json')
    with pytest.raises(ValueError, match='64d10fbf9f8e.*pre_tokenizer.Split'):
        validate_pipeline(TOKENIZERS['unsloth_pipeline'], 'actual Unsloth tokenizer.json')


@pytest.mark.parametrize('config', [TOKENIZERS['unsloth_config'], TOKENIZERS['spark_config']])
def test_actual_partial_configs_report_decoder_and_prefix_failures_together(config):
    with pytest.raises(ValueError) as error:
        validate_resources(contract(TOKENIZERS['official_pipeline'], config=config), 'paired resources')
    assert all(field in str(error.value) for field in ('added_tokens_decoder', 'add_bos_token'))


@pytest.mark.parametrize('case,expected', [
    ('decoder_flags', '布尔字段'), ('decoder_conflict', '标志冲突'),
    ('pad', 'pad_token'), ('eos_missing', 'eos_token_id'), ('eos_outside', 'public Token ID'),
    ('merges_missing', 'model.merges'), ('merges_reference', 'model.merges'),
    ('added_missing', 'added_tokens'), ('vocab_type', 'model.type'),
])
def test_complete_frontend_required_fields_are_checked_before_preparation(case, expected):
    tokenizer, config = tokenizer_documents(TOKENIZERS['official_pipeline'])
    generation = {'eos_token_id': [2, 3]}
    if case == 'decoder_flags': config['added_tokens_decoder']['2'].pop('normalized')
    elif case == 'decoder_conflict': config['added_tokens_decoder']['2']['special'] = False
    elif case == 'pad': config['pad_token'] = '<wrong>'
    elif case == 'eos_missing': generation = {}
    elif case == 'eos_outside': generation['eos_token_id'] = [5]
    elif case == 'merges_missing': tokenizer['model'].pop('merges')
    elif case == 'merges_reference': tokenizer['model']['merges'] = [['a', 'missing']]
    elif case == 'added_missing': tokenizer.pop('added_tokens')
    elif case == 'vocab_type': tokenizer['model']['type'] = 'WordPiece'
    with pytest.raises(ValueError, match=expected):
        validate_resources(inspect_resources(tokenizer, config, generation, 4), 'frontend resources')


def test_partial_bundle_and_old_single_file_selection_cannot_silently_mix_resources(pinned):
    model, runtime, profile, context, source = pinned
    bundle = source.parent / 'incomplete tokenizer'
    bundle.mkdir()
    write_tokenizer(bundle / 'tokenizer.json', TOKENIZERS['official_pipeline'])
    profile.parameters = {'conversion_recipe': 'qwen3_8_27b', 'tokenizer_directory': bundle.as_posix()}
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    assert 'tokenizer_config.json' in prep.preflight(model, profile, runtime)['message']
    profile.parameters = {'conversion_recipe': 'qwen3_8_27b', 'tokenizer_json': (bundle / 'tokenizer.json').as_posix()}
    assert '配套分词器目录' in prep.preflight(model, profile, runtime)['message']
    assert not prep.root.exists()


@pytest.mark.parametrize('field,value,error', [
    ('normalizer', None, 'normalizer'),
    ('pre_tokenizer', {'type': 'ByteLevel'}, 'pre_tokenizer'),
    ('decoder', {'type': 'WordPiece'}, 'decoder'),
    ('post_processor', {'type': 'TemplateProcessing'}, 'post_processor'),
    ('model', {'dropout': .1}, 'model.dropout'),
    ('model', {'ignore_merges': True}, 'model.ignore_merges'),
])
def test_pipeline_check_mirrors_explicit_pinned_contract(field, value, error):
    pipeline = deepcopy(TOKENIZERS['official_pipeline'])
    pipeline[field] = value
    with pytest.raises(ValueError, match=error):
        validate_pipeline(pipeline, 'tokenizer.json')


async def test_actual_default_split_failure_blocks_preflight_preview_start_before_conversion(pinned):
    model, runtime, profile, context, source = pinned
    tokenizer = Path(model.path) / 'tokenizer.json'
    write_tokenizer(tokenizer, TOKENIZERS['unsloth_pipeline'])
    before = tokenizer.read_bytes()
    profile.parameters = {'conversion_recipe': 'qwen3_8_27b'}
    controller = Controller(Path(context.data_dir) / 'tokenizer-api', 'fixture-secret')
    for kind, value in [('models', model), ('runtimes', runtime), ('profiles', profile)]:
        controller.store.put(kind, value.model_dump())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer fixture-secret'}) as manager:
        try:
            checked = await manager.post('/api/profiles/preflight', json=profile.model_dump())
            assert checked.status_code == 200 and checked.json()['status'] == 'error', checked.text
            assert 'pre_tokenizer.Split' in checked.json()['message']
            assert 'tokenizer_directory' in {field['key'] for field in checked.json()['preparation_fields']}
            preview = await manager.post('/api/profiles/' + profile.id + '/preview')
            assert preview.status_code == 200 and preview.json()['argv'] == [], preview.text
            started = await manager.post('/api/profiles/' + profile.id + '/start')
            assert started.status_code == 400 and 'pre_tokenizer.Split' in started.text, started.text
            assert not controller.runtime.instances and not controller.preparation.native.root.exists()
            assert not (source / 'pid').exists() and tokenizer.read_bytes() == before
        finally:
            await controller.close()


@pytest.mark.parametrize('selected', ['D:\\官方分词器', '/official resources'])
@pytest.mark.parametrize('changed_role', ['tokenizer.json', 'tokenizer_config.json'])
async def test_explicit_tokenizer_pair_maps_argv_and_stamps_without_rewriting_source(pinned, monkeypatch, selected, changed_role):
    model, runtime, profile, context, source = pinned
    original = Path(model.path) / 'tokenizer.json'
    write_tokenizer(original, TOKENIZERS['unsloth_pipeline'])
    before = original.read_bytes()
    bundle = source.parent / 'same-base official tokenizer'
    bundle.mkdir()
    tokenizer, config = tokenizer_documents(TOKENIZERS['official_pipeline'])
    (bundle / 'tokenizer.json').write_text(json.dumps(tokenizer))
    (bundle / 'tokenizer_config.json').write_text(json.dumps(config))
    mapped = []
    def target(value, owner):
        mapped.append((value, owner.metadata['distribution']))
        return bundle.as_posix() if value == selected else str(value)
    monkeypatch.setattr(native, 'resource_target', target)
    profile.parameters = {'conversion_recipe': 'qwen3_8_27b', 'tokenizer_directory': selected}
    store = MemoryStore()
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), store)
    script = write_converter(source)
    (source / 'release').touch()
    monkeypatch.setattr(preparation, 'prepare_launch', lambda spec, context: spec.model_copy(update={
        'platform': 'windows', 'argv': [sys.executable, '-u', str(script), *spec.argv[4:]], 'cwd': str(source)}))
    async def progress(event): pass
    resolved = await prep.ensure(model, profile, runtime, progress)
    command = json.loads((source / 'argv.json').read_text())
    for role in ('tokenizer.json', 'tokenizer_config.json'):
        assert role + '=' + (bundle / role).as_posix() in command['resource']
    assert (selected, 'ExternalUbuntu') in mapped and profile.parameters['tokenizer_directory'] == selected
    record = store.all(prep.KIND)[0]
    assert record['identity']['parameters']['tokenizer_directory'] == selected
    for role in ('tokenizer.json', 'tokenizer_config.json'):
        assert record['identity']['resources'][role]['path'].replace('\\', '/') == (bundle / role).as_posix()
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert Path(resolved.parameters['_prepared_artifact']).is_file() and original.read_bytes() == before
    # Identical pipeline but changed sidecar bytes must invalidate the old output.
    changed = bundle / changed_role
    changed.write_text(changed.read_text() + '\n')
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    config.pop('added_tokens_decoder')
    config.pop('add_bos_token')
    (bundle / 'tokenizer_config.json').write_text(json.dumps(config))
    assert all(key in prep.preflight(model, profile, runtime)['message'] for key in ('added_tokens_decoder', 'add_bos_token'))


def test_existing_artifact_checks_embedded_tokenizer_not_a_dormant_profile_override(pinned, monkeypatch):
    model, runtime, profile, context, _ = pinned
    model.path = (Path(model.path).parent / 'existing.ninfer').as_posix()
    Path(model.path).write_bytes(b'official-query-fixture')
    runtime.metadata['source_root'] = '/opt/pinned/source'
    profile.parameters = {'tokenizer_directory': '/missing/dormant/official', 'tokenizer_json': '/old/single.json'}
    monkeypatch.setattr(native, 'artifact_info', artifact_info)
    monkeypatch.setattr(native, '_run', lambda *args, **kwargs: json.dumps({
        'components': {'text': {}}, 'tokenizer_contract': contract(TOKENIZERS['unsloth_pipeline'])}))
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    state = prep.preflight(model, profile, runtime)
    assert state['status'] == 'error' and 'pre_tokenizer.Split' in state['message']
    assert state['preparation_fields'] == []
    with pytest.raises(ValueError, match='pre_tokenizer.Split'):
        get_adapters()['ninfer'].build_launch(model, profile, runtime, context)
    monkeypatch.setattr(native, '_run', lambda *args, **kwargs: json.dumps({
        'components': {'text': {}}, 'tokenizer_contract': contract(TOKENIZERS['official_pipeline'])}))
    assert prep.preflight(model, profile, runtime)['status'] == 'ready'
    assert profile.parameters['tokenizer_directory'] == '/missing/dormant/official'
    assert profile.parameters['tokenizer_json'] == '/old/single.json'
    assert artifact_info(model.path, runtime) == {'components': {'text': {}}}


@pytest.mark.parametrize('identity', [
    {},
    {'source_commit': 'f' * 40, 'version_basis': 'verified_source_commit'},
    {'source_commit': native.PINS['ninfer']['commit'], 'version_basis': 'external_unknown_no_version_flag'},
])
def test_tokenizer_pin_restrictions_do_not_leak_to_unknown_external_engines(local, monkeypatch, identity):
    model, runtime, profile, context, _ = local
    runtime.metadata.update(identity)
    write_tokenizer(Path(model.path) / 'tokenizer.json', TOKENIZERS['unsloth_pipeline'])
    profile.parameters = {'conversion_recipe': 'qwen3_8_27b'}
    prep = preparation.NativePreparation(AppPaths(Path(context.data_dir) / 'isolated'), MemoryStore())
    assert prep.preflight(model, profile, runtime)['status'] == 'needs_preparation'
    runtime.metadata['source_root'] = '/opt/external/source'
    def query(argv, **kwargs):
        assert argv[-1] == ''
        return json.dumps({'components': {'text': {}}, 'tokenizer_contract': contract(TOKENIZERS['unsloth_pipeline'])})
    monkeypatch.setattr(native, '_run', query)
    assert artifact_info('/existing.ninfer', runtime) == {'components': {'text': {}}}
