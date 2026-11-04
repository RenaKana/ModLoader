import json
from pathlib import Path
import struct
import subprocess
import sys
import asyncio
import inspect

import pytest

from modloader.backends.adapters import _strata_config
from test_strata_preparation import fixture, write_pack


def test_import_retains_safe_drafting_and_normalizes_user_paths(tmp_path):
    cfg = _strata_config({'drafting': 'dflash2', 'mtp': False,
                         'user_args': ['--draft-model=custom.pack', '--mtp-draft-vocab', 'vocab.bin',
                                       '--spec', '7', '--api-key', 'secret'],
                         'env': {'SECRET': 'secret'}, 'hooks': ['unsafe']}, tmp_path / 'strata.json')
    assert cfg['drafting'] == 'dflash2'
    assert cfg['mtp'] is False
    assert cfg['user_args'] == ['--draft-model', str(tmp_path / 'custom.pack'), '--mtp-draft-vocab',
                                str(tmp_path / 'vocab.bin'), '--spec', '7']
    assert 'env' not in cfg and 'hooks' not in cfg


def test_off_never_requires_new_runtime_or_drafter(fixture):
    from modloader.strata_drafting import inspect_drafting
    prep, model, profile, runtime = fixture
    profile.parameters = {'strata_drafting': 'off', 'mtp_enabled': True, 'mtp_path': 'missing'}
    assert inspect_drafting(model, profile, runtime)['effective_mode'] == 'off'


def test_explicit_dflash_does_not_block_an_unbenchmarked_qwen_finetune(fixture):
    from modloader.strata_drafting import inspect_drafting
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Swift-Qwen3.8-Flash-Next'
    model.architecture = 'qwen4exp'
    model.metadata['qwen4exp.embedding_length'] = 2560
    profile.parameters['strata_drafting'] = 'dflash2'
    assert inspect_drafting(model, profile, runtime)['effective_mode'] == 'dflash2'


def test_explicit_dflash_rejects_old_engine_before_preparing(fixture):
    from modloader.strata_drafting import inspect_drafting
    prep, model, profile, runtime = fixture
    profile.parameters['strata_drafting'] = 'dflash2'
    with pytest.raises(ValueError, match='DFlash2'):
        inspect_drafting(model, profile, runtime)


def test_legacy_profile_does_not_gain_auto(fixture):
    from modloader.strata_drafting import inspect_drafting
    prep, model, profile, runtime = fixture
    assert inspect_drafting(model, profile, runtime)['mode'] == 'legacy'


def test_dflash_preflight_is_separate_from_main_pack(fixture):
    prep, model, profile, runtime = fixture
    root = Path(runtime.path)
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters = {'strata_drafting': 'auto', 'pack': str(write_pack(root / 'pack'))}
    result = prep.preflight(model, profile, runtime)
    assert result['pack'] == str(root / 'pack')
    assert result['status'] == 'needs_preparation'
    assert result['drafting']['effective_mode'] == 'dflash2'
    assert result['drafting']['download_bytes'] == 1206141244


def write_dflash(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    index = b'cfg hidden 2560\nt fc.weight q4_0 1 32 0 18\nend\n'
    path.write_bytes(b'DF2P' + struct.pack('<II', 1, len(index)) + index + bytes(18))
    return path


def modern(runtime):
    root = Path(runtime.path)
    (root / 'strata_control').mkdir(exist_ok=True)
    (root / 'strata_control/drafting.py').write_text('''def engine_flags(*args): pass
def conf_for(*args): pass
def slots_of(*args): pass
def strip_mtp_flags(*args): pass
''')
    (root / 'strata_control/vram_fit.py').write_text('def fit_start(cfg, args, *a): return args, None, ""\n')
    (root / 'data').mkdir(exist_ok=True)
    (root / 'data/draft_vocab.bin').write_bytes(b'vocab')
    runtime.metadata.update(python=sys.executable, engine_backend='cuda',
                            supported_flags=['--draft', '--draft-model', '--draft-conf', '--draft-fallback', '--mtp-draft-vocab', '--spec', '--prefill'])


def test_custom_alias_external_pack_and_explicit_parameters_survive_launch(tmp_path):
    from test_mtp import strata
    from modloader.backends import get_adapters
    from modloader.mtp import argument
    model, profile, runtime, context, _ = strata(tmp_path)
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.alias = 'my-personal-alias'
    pack = write_dflash(tmp_path / 'external/arbitrary.pack')
    profile.parameters.update(strata_drafting='dflash2', dflash2_path=str(pack), spec=7, mtp_enabled=True, mtp_path='unused missing')
    profile.extra_args = ['--draft-conf=0.72']
    spec = get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    assert cfg['model_name'] == 'my-personal-alias'
    assert argument(cfg['args'], '--draft-model') == str(pack)
    assert argument(cfg['args'], '--spec') == '7'
    assert argument(cfg['args'], '--draft-conf') == '0.72'
    assert '--mtp' not in cfg['args']
    assert argument(cfg['user_args'], '--draft-conf') == '0.72'
    assert argument(cfg['user_args'], '--spec') == '7'
    assert '--draft-fallback' not in cfg['user_args']
    # Execute the generated child shim against a fake serve module. The fake
    # upstream policy would otherwise silently replace this external resource.
    root = Path(runtime.path)
    (root / 'serve').mkdir()
    (root / 'serve/server.py').write_text('''import json, sys
from strata_control import drafting
cfg = json.load(open(sys.argv[sys.argv.index('--config')+1], encoding='utf-8'))
decision = drafting.apply(cfg)
assert cfg['model_name'] == 'my-personal-alias'
assert '--mtp' not in cfg['args']
assert decision.drafter == 'dflash2'
print(json.dumps(cfg['args']))
''')
    (root / 'strata_control/drafting.py').write_text('''from dataclasses import dataclass
@dataclass
class Decision:
    asked: str
    drafter: str
    flags: tuple = ()
def apply(cfg, **kw):
    cfg['args'] = ['wrong automatic policy']
DRAFT_FLAGS = dict.fromkeys(('--draft', '--draft-model', '--draft-conf', '--draft-fallback'), True)
MTP_ONLY_FLAGS = {'--mtp': True, '--mtp-draft-vocab': True, '--spec-min-p': True}
def _flag_value(args, flag):
    return args[args.index(flag)+1] if flag in args else None
def _without_flag(args, flag):
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i+2 if i+1 < len(out) and not out[i+1].startswith('--') else i+1]
    return out
def strip_draft_flags(args):
    for flag in DRAFT_FLAGS:
        args = _without_flag(args, flag)
    return args
def strip_mtp_flags(args):
    for flag in MTP_ONLY_FLAGS:
        args = _without_flag(args, flag)
    return args
def engine_flags(path, vocab):
    return ('--draft', 'dflash2', '--draft-model', path, '--draft-fallback', 'none', '--mtp-draft-vocab', vocab, '--spec', '6', '--draft-conf', '0.35')
def conf_for(*args): return 0.35
def slots_of(*args): return 1
''')
    (root / 'strata_control/vram_fit.py').write_text('def fit_start(cfg, args, *a): return args, None, ""\n')
    (root / 'strata_control/user_args.py').write_text('''from .drafting import _without_flag
def merge(args, mine):
    for flag in mine:
        if flag.startswith('--'):
            args = _without_flag(args, flag)
    return args + mine
''')
    child = subprocess.run(spec.argv, cwd=runtime.path, text=True, capture_output=True, timeout=10)
    assert child.returncode == 0, child.stderr
    from collections import Counter
    assert Counter(json.loads(child.stdout)) == Counter(cfg['args'])
    from modloader.launch_parameters import edit_parameters, format_argv
    imported = edit_parameters(model, profile, runtime, text=format_argv(spec.argv))
    assert not imported['errors']


def test_off_strips_both_drafters_and_ignores_missing_resources(tmp_path):
    from test_mtp import strata
    from modloader.backends import get_adapters
    model, profile, runtime, context, _ = strata(tmp_path)
    modern(runtime)
    profile.parameters.update(strata_drafting='off', mtp_enabled=True, mtp_path='missing', dflash2_path='missing')
    profile.extra_args = ['--draft', 'dflash2', '--draft-model', 'also-missing']
    get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    assert cfg['drafting'] == 'off'
    assert '--draft' not in cfg['args'] and '--mtp' not in cfg['args']


def test_custom_raw_engine_is_selected_without_turning_on_prefetch(tmp_path):
    from test_strata_prefetch import runtime_fixture, sidecar_fixture, profile_fixture
    from modloader.strata_prefetch import select_engine
    runtime = runtime_fixture(tmp_path / 'runtime')
    runtime.metadata['supported_flags'] = ['--spec']
    _, profile = profile_fixture(tmp_path, runtime, enabled=False)
    profile.extra_args = ['--raw-token-continuation']
    with pytest.raises(ValueError, match='准备自定义引擎'):
        select_engine(profile, runtime)
    sidecar_fixture(Path(runtime.path) / 'engine', features=['startup_expert_prefetch', 'raw_token_continuation'])
    exe, prefetch = select_engine(profile, runtime)
    assert exe.endswith('strata-startup-prefetch.exe')
    assert prefetch is False


def test_server_patch_is_pinned_exact_and_repeatable(tmp_path):
    from modloader.strata_patches import apply_server_patch, CONTROL_REF
    import shutil
    source = Path(__file__).resolve().parents[1] / '.artifacts/codex/engine-compat-20261103/strata/upstream/serve/server.py'
    if not source.is_file():
        pytest.skip('isolated official Control source not available')
    (tmp_path / 'serve').mkdir()
    shutil.copyfile(source, tmp_path / 'serve/server.py')
    original = (tmp_path / 'serve/server.py').read_text('utf-8')
    assert apply_server_patch(tmp_path, 'another-release') is False
    assert (tmp_path / 'serve/server.py').read_text('utf-8') == original
    assert apply_server_patch(tmp_path, CONTROL_REF)
    once = (tmp_path / 'serve/server.py').read_text('utf-8')
    assert 'raw_token_continuation_enabled' in once
    assert apply_server_patch(tmp_path, CONTROL_REF)
    assert (tmp_path / 'serve/server.py').read_text('utf-8') == once


def test_builder_0156_selects_both_custom_features(tmp_path):
    from test_strata_prefetch_builder import plan
    result = plan(tmp_path, build={'version': '0.1.56', 'commit': '06698c674673'})
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['features'] == ['startup_expert_prefetch', 'raw_token_continuation']
    assert value['patch_revision'] == 'win-mapped-profile-raw-continuation-0.1.56-v1'


def fake_fetch(runtime, body=''):
    root = Path(runtime.path)
    (root / 'tools/dflash2_pack.py').write_text('# fixture only')
    (root / 'strata_control/drafting_fetch.py').write_text('''import os, json, struct, time, urllib.request
from pathlib import Path
''' + inspect.getsource(write_dflash) + '''
def ensure(directory, progress):
    root = Path(directory)
    (root / 'environment.json').write_text(json.dumps({key:os.environ.get(key) for key in
        ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','HF_HUB_OFFLINE','STRATA_NO_ACQUIRE')}))
    (root / 'proxy-policy.json').write_text(json.dumps(urllib.request.getproxies()))
''' + (body or '''    write_dflash(root / 'dflash2/qwen3.8-flash-next-dflash2.q4_0.pack')
    progress('fixture ready', 1, 1)
''') + '''
def convert_selector(directory):
    pass
''')


async def test_ensure_prepares_only_selected_drafter_and_keeps_proxy_route(fixture, monkeypatch):
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters.update(strata_drafting='auto', pack=str(write_pack(Path(runtime.path) / 'pack')))
    fake_fetch(runtime)
    monkeypatch.setenv('HTTP_PROXY', 'http://must-not-inherit:12')
    prep.proxy_provider = lambda: ''
    async def progress(update): pass
    resolved = await prep.ensure(model, profile, runtime, progress)
    assert Path(resolved.parameters['dflash2_path']).is_file()
    assert 'mtp_path' not in resolved.parameters
    root = prep._draft_root(model)
    assert not (root / 'mtp').exists()
    env = json.loads((root / 'environment.json').read_text())
    assert env == dict(HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', NO_PROXY='', HF_HUB_OFFLINE='0', STRATA_NO_ACQUIRE='0')
    assert json.loads((root / 'proxy-policy.json').read_text()) == {}
    assert prep.preflight(model, resolved, runtime)['status'] == 'ready'


async def test_manual_proxy_is_the_only_drafting_download_route(fixture, monkeypatch):
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters.update(strata_drafting='dflash2', pack=str(write_pack(Path(runtime.path) / 'pack')))
    fake_fetch(runtime)
    monkeypatch.setenv('ALL_PROXY', 'http://unused-environment-proxy:12')
    prep.proxy_provider = lambda: 'http://selected-app-proxy:17890'
    async def progress(update): pass
    await prep.ensure(model, profile, runtime, progress)
    assert json.loads((prep._draft_root(model) / 'proxy-policy.json').read_text()) == {
        'http': 'http://selected-app-proxy:17890', 'https': 'http://selected-app-proxy:17890'}


async def test_explicit_mtp_prepares_rt_not_dflash(fixture):
    from test_mtp import write_weights
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    write_weights(Path(model.path), architecture='qwen4exp', embedding=2560)
    profile.parameters.update(strata_drafting='mtp', pack=str(write_pack(Path(runtime.path) / 'pack')))
    root = Path(runtime.path)
    prefix = 'import sys\nfrom pathlib import Path\nout=Path(sys.argv[sys.argv.index("--out")+1])\n'
    (root / 'tools/mtp_fetch.py').write_text(prefix + 'out.mkdir(parents=True,exist_ok=True)\n(out/"fetched").touch()\n')
    (root / 'tools/mtp_pack.py').write_text(prefix + 'assert (out.parent/"fetched").is_file()\nout.write_bytes(b"fixture")\n')
    (root / 'tools/mtp_rt.py').write_text(prefix + '''assert Path(sys.argv[sys.argv.index("--gguf")+1]).is_file()
out.mkdir(parents=True,exist_ok=True)
(out/'dense.bin').write_bytes(b'dense')
(out/'experts.bin').write_bytes(b'experts')
(out/'dense.txt').write_text('fc_embedding.weight q8_0 2560 2560 0 7\\n')
''')
    async def progress(update): pass
    resolved = await prep.ensure(model, profile, runtime, progress)
    assert resolved.parameters['mtp_path'] == str(prep._draft_root(model) / 'mtp/rt')
    assert 'dflash2_path' not in resolved.parameters
    assert not (prep._draft_root(model) / 'dflash2').exists()


async def test_drafting_failure_reports_original_error_and_does_not_fallback(fixture):
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters.update(strata_drafting='dflash2', pack=str(write_pack(Path(runtime.path) / 'pack')))
    fake_fetch(runtime, "    raise RuntimeError('fixture proxy connection refused')\n")
    async def progress(update): pass
    with pytest.raises(ValueError, match='fixture proxy connection refused'):
        await prep.ensure(model, profile, runtime, progress)
    assert not (prep._draft_root(model) / 'mtp').exists()


async def test_cancel_drafting_stops_its_process(fixture):
    import psutil
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters.update(strata_drafting='dflash2', pack=str(write_pack(Path(runtime.path) / 'pack')))
    fake_fetch(runtime, "    (root / 'pid').write_text(str(os.getpid()))\n    progress('started', 0, 1)\n    time.sleep(90)\n")
    ready = asyncio.Event()
    async def progress(update):
        if update['message'] == 'started': ready.set()
    task = asyncio.create_task(prep.ensure(model, profile, runtime, progress))
    await asyncio.wait_for(ready.wait(), 10)
    pid = int((prep._draft_root(model) / 'pid').read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    # Windows job termination is asynchronous for the venv launcher's child.
    try:
        await asyncio.to_thread(psutil.Process(pid).wait, timeout=3)
    except psutil.NoSuchProcess:
        pass
    assert not psutil.pid_exists(pid)


def test_dflash_estimate_matches_selection_not_stale_mtp(tmp_path):
    from test_mtp import strata
    from modloader.resource_estimates import strata_estimate
    from modloader.strata_resource_estimates import MIB
    model, profile, runtime, _, _ = strata(tmp_path)
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters.update(strata_drafting='dflash2', dflash2_path=str(write_dflash(tmp_path / 'draft.pack')),
                              mtp_enabled=True, mtp_path='idle missing mtp')
    estimate = strata_estimate(model, profile, runtime)
    components = estimate['resources'][0]['components']
    assert {'label': 'DFlash2 推理预算', 'bytes': 890 * MIB} in components
    assert not any(item['label'].startswith('MTP') for item in components)


def test_relative_cwd_and_switch_user_override_do_not_consume_next_option(tmp_path):
    from modloader.strata_drafting import merge_args
    cfg = _strata_config({'cwd': 'relative', 'user_args': ['--draft-model', 'draft.pack', '--log-file', 'secret.log']},
                         tmp_path / 'config.json')
    assert cfg['cwd'] == str(tmp_path / 'relative')
    assert cfg['user_args'] == ['--draft-model', str(tmp_path / 'relative/draft.pack')]
    assert merge_args(['--no-prefill-borrow', '--spec', '4'], ['--no-prefill-borrow']) == ['--spec', '4', '--no-prefill-borrow']


@pytest.mark.parametrize('flag,value,effective', [
    ('--pipeline-windows', '1', 'mtp'), ('--pipeline-windows', '2', 'mtp'),
    ('--pipeline-windows', '0', 'dflash2'), ('--batch', '1', 'dflash2'), ('--batch', '2', 'mtp'),
    ('--slots', '1', 'dflash2'), ('--slots', '2', 'mtp'), ('--parallel', '1', 'dflash2'), ('--parallel', '2', 'mtp')])
def test_auto_observes_each_concurrency_flag_semantics(fixture, flag, value, effective):
    from modloader.strata_drafting import inspect_drafting
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters['strata_drafting'] = 'auto'
    profile.extra_args = [flag + '=' + value]
    assert inspect_drafting(model, profile, runtime)['effective_mode'] == effective
    if effective == 'mtp':
        profile.parameters['strata_drafting'] = 'dflash2'
        with pytest.raises(ValueError, match='单并发'):
            inspect_drafting(model, profile, runtime)


def test_auto_observes_imported_server_parallel(fixture):
    from modloader.strata_drafting import inspect_drafting
    prep, model, profile, runtime = fixture
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters.update(strata_drafting='auto', strata_config={'parallel': 2})
    assert inspect_drafting(model, profile, runtime)['effective_mode'] == 'mtp'


@pytest.mark.parametrize('mode', ['off', 'dflash2', 'auto'])
@pytest.mark.parametrize('old_tokens', [99, 'idle invalid value'])
def test_new_non_mtp_modes_ignore_idle_mtp_tokens_in_editor(tmp_path, mode, old_tokens):
    from test_mtp import strata
    from modloader.launch_parameters import edit_parameters
    model, profile, runtime, _, _ = strata(tmp_path)
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.parameters.update(strata_drafting=mode, mtp_enabled=True, mtp_tokens=old_tokens,
                              dflash2_path=str(write_dflash(tmp_path / 'draft.pack')))
    result = edit_parameters(model, profile, runtime)
    assert not result['errors']
    assert '--spec 100' not in result['text']
    assert result['parameters']['mtp_tokens'] == old_tokens


def test_long_lookup_spec_only_changes_new_supported_window(tmp_path):
    from test_mtp import strata
    from modloader.launch_parameters import edit_parameters
    from modloader.mtp import validate_tokens
    model, profile, runtime, _, _ = strata(tmp_path)
    runtime.metadata.update(engine_backend='cuda', supported_flags=['--lookup-max-t', '--spec', '--suffix-draft'])
    result = edit_parameters(model, profile, runtime, text='--lookup-max-t 12 --spec 12')
    assert not result['errors']
    assert result['parameters']['spec'] == 12
    assert result['parameters'].get('mtp_tokens') != 11
    profile.parameters['spec'] = 16
    profile.extra_args = ['--lookup-max-t', '12']
    validate_tokens(profile, runtime)
    profile.extra_args = ['--lookup-max-t', '12', '--suffix-draft', '0']
    with pytest.raises(ValueError): validate_tokens(profile, runtime)
    profile.extra_args = []
    assert edit_parameters(model, profile, runtime, text='--spec 12')['errors']
    runtime.metadata['supported_flags'] = ['--spec']
    profile.extra_args = ['--lookup-max-t', '12']
    with pytest.raises(ValueError): validate_tokens(profile, runtime)
