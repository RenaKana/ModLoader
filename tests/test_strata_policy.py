import json
from pathlib import Path
import subprocess
import sys

import pytest

from modloader.backends import adapters
from modloader.mtp import strata_spec_field
from test_mtp import strata
from test_strata_drafting import modern, write_dflash


def legacy_control(runtime):
    """Control 0.2.4: cfg-only planning, old drafting API, and no fit_start."""
    root = Path(runtime.path)
    control = root / 'strata_control'
    control.mkdir(exist_ok=True)
    (control / 'vram_fit.py').write_text('def plan(*args): return None\n')
    (control / 'user_args.py').write_text('''def groups(args):
    out = []
    for value in args:
        if value.startswith('--') or not out:
            out.append([value])
        else:
            out[-1].append(value)
    return out
def merge(args, mine):
    flags = {group[0] for group in groups(mine)}
    return [value for group in groups(args) if group[0] not in flags for value in group] + mine
''')
    (control / 'drafting.py').write_text('''from dataclasses import dataclass
@dataclass
class Decision:
    asked: str
    drafter: str
    note: str = ''
    flags: tuple = ()
def apply(cfg, *, klass, data_dir=None, family=None):
    asked = cfg.get('drafting', 'auto')
    effective = 'off' if asked == 'off' else 'mtp'
    return Decision(asked, effective)
''')
    serve = root / 'serve'
    serve.mkdir(exist_ok=True)
    (serve / 'server.py').write_text('''import json, sys
from strata_control import drafting, user_args
def engine_args(cfg):
    assert '_modloader_policy' not in cfg
    args = user_args.merge(cfg['args'], cfg.get('user_args', []))
    decision = drafting.apply(cfg, klass='other')
    if cfg.get('mtp') is False or decision.drafter == 'off':
        args = [value for group in user_args.groups(args) if group[0] not in
                {'--mtp', '--spec-min-p', '--batch-mtp'} for value in group]
    print('[strata] legacy planning')
    return args
if __name__ == '__main__':
    from contextlib import redirect_stdout
    cfg = json.load(open(sys.argv[sys.argv.index('--config') + 1], encoding='utf-8'))
    with redirect_stdout(sys.stderr):
        args = engine_args(cfg)
    print(json.dumps(args))
''')
    runtime.metadata['python'] = sys.executable
    runtime.version = '0.1.48'


@pytest.mark.parametrize('mode', ['legacy', 'legacy-off', 'mtp', 'auto', 'off'])
def test_old_control_keeps_native_start_and_cfg_only_plan(tmp_path, mode):
    from modloader.mtp import argument
    from modloader.strata_drafting import plan_args
    model, profile, runtime, context, mtp = strata(tmp_path)
    legacy_control(runtime)
    profile.parameters['strata_config']['user_args'] = ['--spec', '6', '--prefill', 'auto:768']
    profile.extra_args = ['--max-context', '65536', '--spec', '7', '--resident-budget-gib', '4']
    if mode == 'legacy-off':
        profile.parameters['strata_config']['mtp'] = False
    elif mode != 'legacy':
        profile.parameters['strata_drafting'] = mode
    launch = adapters.get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    assert launch.argv[1:3] == ['-m', 'serve.server']
    assert '_modloader_policy' not in cfg
    planned = plan_args(runtime, cfg)
    child = subprocess.run(launch.argv, cwd=runtime.path, text=True, capture_output=True, timeout=10)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == planned
    assert argument(planned, '--max-context') == '65536'
    assert argument(planned, '--spec') == '7'
    assert argument(planned, '--prefill') == 'auto:768'
    assert argument(planned, '--resident-budget-gib') == '4'
    if mode in ('legacy-off', 'off'):
        assert '--mtp' not in planned and '--spec-min-p' not in planned
    else:
        assert argument(planned, '--mtp') == str(mtp)
        assert argument(planned, '--spec-min-p') == '0.5'


def test_old_control_plan_uses_its_cfg_only_signature(tmp_path):
    from modloader.strata_drafting import plan_args
    _, _, runtime, _, _ = strata(tmp_path)
    legacy_control(runtime)
    cfg = {'args': ['--spec', '4'], 'user_args': ['--spec', '7']}
    assert plan_args(runtime, cfg) == ['--spec', '7']


def test_new_control_plan_passes_selected_executable_and_disables_announcements(tmp_path):
    from modloader.strata_drafting import plan_args
    _, _, runtime, _, _ = strata(tmp_path)
    legacy_control(runtime)
    (Path(runtime.path) / 'serve/server.py').write_text('''def engine_args(cfg, exe=None, *, announce=True):
    assert exe == cfg['exe']
    assert announce is False
    return cfg['args']
''')
    cfg = {'args': ['--spec', '7'], 'exe': 'selected-custom-engine.exe'}
    assert plan_args(runtime, cfg) == ['--spec', '7']


@pytest.mark.parametrize('signature', ['cfg', 'cfg, exe=None, *, announce=True'])
def test_native_plan_failure_is_reported_without_retrying_another_api(tmp_path, signature):
    from modloader.strata_drafting import plan_args
    _, _, runtime, _, _ = strata(tmp_path)
    legacy_control(runtime)
    (Path(runtime.path) / 'serve/server.py').write_text(
        f'def engine_args({signature}): raise TypeError("invalid native plan fixture")\n')
    with pytest.raises(ValueError, match='invalid native plan fixture'):
        plan_args(runtime, {'args': []})


def test_old_control_resource_preview_uses_native_plan(tmp_path):
    from test_strata_resource_estimates import fixture
    from modloader.resource_estimates import strata_estimate
    model, profile, runtime = fixture(tmp_path)
    legacy_control(runtime)
    profile.parameters['context_length'] = 65536
    profile.parameters['strata_config'] = {'user_args': ['--spec', '7', '--prefill', '64']}
    result = strata_estimate(model, profile, runtime)
    gpu = result['resources'][0]
    assert any('上下文 65536' in note for note in gpu['notes'])
    assert any(part['label'] == 'KV 缓存' for part in gpu['components'])


def test_dflash_capability_requires_serving_control_api_not_module_names(tmp_path):
    from modloader.strata_drafting import supports_dflash
    _, _, runtime, _, _ = strata(tmp_path)
    legacy_control(runtime)
    runtime.metadata['supported_flags'] = ['--draft', '--draft-fallback', '--mtp-draft-vocab']
    assert not supports_dflash(runtime)


def test_registered_native_capabilities_enable_long_lookup(tmp_path, monkeypatch):
    model, profile, runtime, _, _ = strata(tmp_path)
    root = Path(runtime.path)
    (root / 'serve').mkdir()
    (root / 'serve/server.py').touch()
    python = root / 'python.exe'
    python.touch()
    (root / 'engine/BUILD.json').write_text('{"version":"0.1.56","backend":"cuda"}')
    monkeypatch.setattr(adapters, '_run_readonly', lambda argv, **kw: '--spec --lookup-max-t --suffix-draft --draft --draft-fallback --mtp-draft-vocab')
    registered = adapters.import_runtime('strata', str(root), {'python': str(python)})
    profile.extra_args = ['--lookup-max-t', '12', '--spec', '12']
    assert strata_spec_field(profile, registered).maximum == 16
    assert '--draft' in registered.metadata['supported_flags']


@pytest.mark.parametrize('advertised', [False, True])
def test_registered_budget_reaches_editor_and_launch(tmp_path, monkeypatch, advertised):
    from modloader.launch_parameters import edit_parameters
    from modloader.mtp import argument
    model, profile, runtime, context, _ = strata(tmp_path)
    root = Path(runtime.path)
    (root / 'serve').mkdir()
    (root / 'serve/server.py').touch()
    python = root / 'python.exe'
    python.touch()
    (root / 'engine/BUILD.json').write_text('{"version":"0.1.56","backend":"cuda"}')
    # Official 0.1.56 parses budget, but its help only lists the RAM headroom.
    help_text = '--spec --prefill --ram-keep-gib' + (' --resident-budget-gib' if advertised else '')
    monkeypatch.setattr(adapters, '_run_readonly', lambda *a, **kw: help_text)
    registered = adapters.import_runtime('strata', str(root), {'python': str(python)})
    profile.parameters['strata_config']['args'] = profile.parameters['strata_config']['args'][:4]
    profile.extra_args = ['--resident-budget-gib', '4', '--ram-keep-gib', '16']
    edited = edit_parameters(model, profile, registered)
    assert edited['errors'] == []
    assert registered.metadata['supported_flags'].count('--resident-budget-gib') == 1
    adapters.get_adapters()['strata'].build_launch(model, profile, registered, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    assert argument(cfg['args'], '--resident-budget-gib') == '4'
    assert argument(cfg['args'], '--ram-keep-gib') == '16'


@pytest.mark.parametrize('version', ['0.1.31', '0.1.56', '0.1.57'])
def test_existing_budget_capabilities_are_normalized_without_mutating_registration(tmp_path, version):
    from modloader.launch_parameters import edit_parameters
    model, profile, runtime, _, _ = strata(tmp_path)
    runtime.version = version
    profile.parameters['strata_config']['args'] = profile.parameters['strata_config']['args'][:4]
    runtime.metadata['supported_flags'] = ['--spec', '--prefill']
    profile.extra_args = ['--resident-budget-gib=4']
    assert edit_parameters(model, profile, runtime)['errors'] == []
    assert adapters._extra(profile, runtime) == profile.extra_args
    assert runtime.metadata['supported_flags'] == ['--spec', '--prefill']
    profile.extra_args = ['--invented-option']
    with pytest.raises(ValueError, match='invented-option'):
        adapters._extra(profile, runtime)


@pytest.mark.parametrize('version,family', [('0.1.30', ''),
                                         ('unknown', ''), ('0.1.56', 'deepseek41')])
def test_budget_is_not_advertised_outside_verified_engine_contract(tmp_path, version, family):
    _, profile, runtime, _, _ = strata(tmp_path)
    runtime.version = version
    runtime.metadata.update(supported_flags=['--spec'], strata_family=family)
    profile.extra_args = ['--resident-budget-gib', '4']
    with pytest.raises(ValueError, match='resident-budget-gib'):
        adapters._extra(profile, runtime)


def test_old_file_only_runtime_keeps_its_original_argument_contract(tmp_path):
    _, profile, runtime, _, _ = strata(tmp_path)
    runtime.version = '0.1.43'
    profile.extra_args = ['--resident-budget-gib', '4', '--other-native-option']
    assert adapters._extra(profile, runtime) == profile.extra_args
    assert 'supported_flags' not in runtime.metadata


def test_auto_projects_explicit_budget_and_drafting_values_for_native_policy(tmp_path):
    from modloader.mtp import argument
    model, profile, runtime, context, _ = strata(tmp_path)
    modern(runtime)
    runtime.version = '0.1.56'
    model.name = 'Qwen3.8-Flash-Next'
    profile.parameters.update(strata_drafting='auto', dflash2_path=str(write_dflash(tmp_path / 'draft.pack')))
    profile.parameters['strata_config']['args'] = profile.parameters['strata_config']['args'][:4]
    profile.extra_args = ['--resident-budget-gib=4', '--spec', '7', '--draft-conf', '0.72']
    adapters.get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    assert cfg['drafting'] == 'auto'
    for flag, value in [('--resident-budget-gib', '4'), ('--spec', '7'), ('--draft-conf', '0.72')]:
        assert argument(cfg['args'], flag) == value
        assert argument(cfg['user_args'], flag) == value


def test_auto_retains_request_and_user_intent_for_engine_retry(tmp_path):
    model, profile, runtime, context, mtp = strata(tmp_path)
    modern(runtime)
    model.name = 'Qwen3.8-Flash-Next-IQ3_XXS'
    profile.alias = 'private-api-alias'
    profile.parameters['strata_config']['args'] = ['--native', model.path, '--pack', str(tmp_path / 'pack'), '--mtp', str(mtp), '--spec-min-p', '0.62']
    profile.parameters.update(strata_drafting='auto', dflash2_path=str(write_dflash(tmp_path / 'draft.pack')))
    adapters.get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    assert cfg['drafting'] == 'auto'
    assert cfg['model_name'] == 'private-api-alias'
    assert '--draft-conf' not in cfg['user_args']
    assert model.name in cfg['_modloader_policy']['model_name']
    assert cfg['_modloader_policy']['mtp_path'] == str(mtp)
    assert cfg['_modloader_policy']['mtp_args'] == ['--spec-min-p', '0.62']
    assert '--spec-min-p' not in cfg['user_args']


def test_known_new_engine_without_custom_binary_rejects_raw_before_launch(tmp_path):
    from modloader.strata_prefetch import select_engine
    _, profile, runtime, _, _ = strata(tmp_path)
    (Path(runtime.path) / 'engine/BUILD.json').write_text('{"version":"0.1.56","backend":"cuda"}')
    profile.extra_args = ['--raw-token-continuation']
    with pytest.raises(ValueError, match='准备自定义引擎'):
        select_engine(profile, runtime)


def test_new_registration_does_not_reuse_stale_flags_when_help_fails(tmp_path, monkeypatch):
    _, _, runtime, _, _ = strata(tmp_path)
    root = Path(runtime.path)
    (root / 'serve').mkdir()
    (root / 'serve/server.py').touch()
    python = root / 'python.exe'
    python.touch()
    (root / 'engine/BUILD.json').write_text('{"version":"0.1.56","backend":"cuda"}')
    monkeypatch.setattr(adapters, '_run_readonly', lambda *a, **kw: '')
    with pytest.raises(ValueError, match='能力'):
        adapters.import_runtime('strata', str(root), {'python': str(python), 'supported_flags': ['--old-flag']})


async def test_refresh_replaces_native_capabilities(tmp_path, monkeypatch):
    from modloader.backends.installers import validate_runtime
    _, _, runtime, _, _ = strata(tmp_path)
    root = Path(runtime.path)
    (root / 'serve').mkdir()
    (root / 'serve/server.py').touch()
    python = root / 'python.exe'
    python.touch()
    (root / 'engine/BUILD.json').write_text('{"version":"0.1.56","backend":"cuda"}')
    runtime.metadata.update(python=str(python), supported_flags=['--stale-flag'])
    monkeypatch.setattr(adapters, '_run_readonly', lambda *a, **kw: '--spec --lookup-max-t')
    updated = await validate_runtime(runtime)
    assert updated.metadata['supported_flags'] == ['--lookup-max-t', '--resident-budget-gib', '--spec']
    assert runtime.metadata['supported_flags'] == ['--stale-flag']


def test_legacy_user_args_do_not_require_new_control_modules(tmp_path):
    model, profile, runtime, context, _ = strata(tmp_path)
    profile.parameters['strata_config']['user_args'] = ['--spec', '7']
    launch = adapters.get_adapters()['strata'].build_launch(model, profile, runtime, context)
    assert launch.argv[1:3] == ['-m', 'serve.server']


def test_resource_preview_consumes_runtime_plan_with_real_identity(tmp_path, monkeypatch):
    from test_strata_resource_estimates import fixture
    from modloader.resource_estimates import strata_estimate
    model, profile, runtime = fixture(tmp_path)
    modern(runtime)
    (tmp_path / 'serve').mkdir()
    (tmp_path / 'serve/server.py').write_text('def engine_args(cfg, exe=None, *, announce=True): return cfg["args"]\n')
    model.name = 'Qwen3.8-Flash-Next'
    profile.alias = 'api-only-alias'
    profile.parameters['strata_drafting'] = 'off'
    profile.parameters.update(resident_experts=False, startup_expert_prefetch=False, vision_enabled=False,
                              strata_config={'args': ['--resident-experts', '--startup-expert-prefetch', '--vision', '--expert-profile', 'rank.bin']})
    profile.parameters.pop('expert_profile')
    def plan(runtime, cfg):
        from modloader.mtp import argument
        assert cfg['model_name'] == 'api-only-alias'
        assert model.name in cfg['_modloader_policy']['model_name']
        assert '--prefill' not in cfg['user_args']
        for flag in ('--resident-experts', '--startup-expert-prefetch', '--vision'):
            assert flag not in cfg['args'] and flag not in cfg['user_args']
        assert argument(cfg['user_args'], '--expert-profile') == str(tmp_path / 'rank.bin')
        return cfg['args'] + ['--kv-resident', '4096']
    monkeypatch.setattr('modloader.strata_drafting.plan_args', plan)
    result = strata_estimate(model, profile, runtime)
    gpu = result['resources'][0]
    assert any('4096' in note and '驻留窗口' in note for note in gpu['notes'])
    assert not any(part['label'] == 'KV 缓存' for part in gpu['components'])


def test_final_user_intent_excludes_disabled_switches_and_resolves_paths(tmp_path):
    from modloader.mtp import argument
    model, profile, runtime, context, _ = strata(tmp_path)
    modern(runtime)
    profile.parameters.update(strata_drafting='off', resident_experts=False,
                              startup_expert_prefetch=False, vision_enabled=False)
    profile.parameters['strata_config']['args'] += ['--startup-expert-prefetch', '--vision',
                                                   '--expert-profile', 'strata/data/expert-profile.bin']
    adapters.get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    for flag in ('--resident-experts', '--startup-expert-prefetch', '--vision'):
        assert flag not in cfg['args']
        assert flag not in cfg['user_args']
    expected = str(Path(runtime.path) / 'data/expert-profile.bin')
    assert argument(cfg['user_args'], '--expert-profile') == expected
    assert argument(cfg['user_args'], '--spec') == '4'


@pytest.mark.parametrize('enabled', [True, False])
def test_custom_prefetch_argument_uses_selected_engine_capabilities(tmp_path, enabled):
    from test_strata_prefetch import runtime_fixture, profile_fixture
    from modloader.models import LaunchContext
    runtime = runtime_fixture(tmp_path / 'runtime', optimized=True)
    runtime.metadata['supported_flags'] = ['--spec', '--prefill', '--expert-cache']
    model, profile = profile_fixture(tmp_path, runtime, enabled)
    profile.extra_args = ['--startup-expert-prefetch']
    context = LaunchContext(port=12345, instance_id='test', work_dir=str(tmp_path / 'work'), data_dir=str(tmp_path))
    adapters.get_adapters()['strata'].build_launch(model, profile, runtime, context)
    cfg = json.loads((Path(context.work_dir) / 'strata.json').read_text('utf-8'))
    assert ('--startup-expert-prefetch' in cfg['args']) == enabled
    assert Path(cfg['exe']).name == ('strata-startup-prefetch.exe' if enabled else 'strata.exe')
    profile.extra_args += ['--not-a-custom-feature']
    with pytest.raises(ValueError, match='not-a-custom-feature'):
        adapters.get_adapters()['strata'].build_launch(model, profile, runtime, context)
