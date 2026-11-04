"""The reproducible native builder must bind to the engine, not the server bundle."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/build-strata-prefetch.ps1'
POWERSHELL = shutil.which('powershell.exe') or shutil.which('pwsh')
pytestmark = pytest.mark.skipif(not POWERSHELL, reason='PowerShell is required')


def plan(tmp_path, *, build=None, component=None, extra_args=()):
    runtime = tmp_path / 'runtime'
    engine = runtime / 'engine'
    engine.mkdir(parents=True)
    original = {'version': '0.1.48', 'commit': '6f8800aefe27', 'cuda': '13.0',
                'archs': [75, 86, 89, 120], 'ptx': True}
    original.update(build or {})
    identity = {'version': original['version'], 'backend': 'cuda',
                'build_id': original['commit'] + '+cuda13.0', 'platform': 'windows-x86_64'}
    identity.update(component or {})
    (engine / 'BUILD.json').write_text(json.dumps(original))
    (engine / 'engine-component.json').write_text(json.dumps(identity))
    # Outer source belongs to the Python service and can have a different engine version.
    (runtime / 'CMakeLists.txt').write_text('project(strata VERSION 9.9.9)')
    (engine / 'strata.exe').write_bytes(b'original engine')
    before = {p.name: p.read_bytes() for p in engine.iterdir()}
    result = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
                             '-RuntimeRoot', str(runtime), '-OutputDirectory', str(tmp_path / 'output'),
                             '-WorkDirectory', str(tmp_path / 'work'), '-PlanOnly', *extra_args],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
    assert {p.name: p.read_bytes() for p in engine.iterdir()} == before
    assert not (tmp_path / 'output').exists()
    assert not (tmp_path / 'work').exists()
    return result


def test_builder_plan_uses_engine_commit_and_preserves_cuda_architectures(tmp_path):
    result = plan(tmp_path)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['engine_identity'] == {'version': '0.1.48', 'commit': '6f8800aefe27',
                                       'backend': 'cuda', 'build_id': '6f8800aefe27+cuda13.0'}
    assert value['source_ref'] == '6f8800aefe27'
    assert value['cuda_architectures'] == ['75-real', '86-real', '89-real', '120-real', '120-virtual']
    assert value['patch_revision']
    assert value['executable'] == 'strata-startup-prefetch.exe'


@pytest.mark.parametrize('build,component', [
    ({}, {'version': '0.1.49'}),
    ({}, {'build_id': 'aaaaaaaaaaaa+cuda13.0'}),
    ({'commit': '../other'}, {}),
    ({}, {'backend': 'hip'}),
    ({'archs': []}, {}),
])
def test_builder_rejects_unbound_or_unsupported_identity_before_writes(tmp_path, build, component):
    result = plan(tmp_path, build=build, component=component)
    assert result.returncode != 0
    assert 'PREFETCH_BUILD:' in result.stderr


def test_builder_new_engine_can_attempt_same_patch_without_version_allowlist(tmp_path):
    result = plan(tmp_path, build={'version': '0.1.50', 'commit': 'abcdef123456', 'ptx': False})
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['source_ref'] == 'abcdef123456'
    assert value['cuda_architectures'] == ['75-real', '86-real', '89-real', '120-real']


def test_packaged_core_carries_builder_and_versioned_patch():
    packaging = (ROOT / 'scripts/build-core.ps1').read_text(encoding='utf-8-sig')
    assert 'build-strata-prefetch.ps1' in packaging
    assert 'patches/strata-prefetch' in packaging


def test_plan_does_not_require_or_read_toolchain_configuration(tmp_path):
    config = tmp_path / 'invalid-toolchain.json'
    config.write_text('not json; plan must not inspect the compiler environment')
    result = plan(tmp_path, extra_args=['-ToolchainConfig', str(config)])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['source_ref'] == '6f8800aefe27'


def test_plan_keeps_nondefault_architectures_and_uses_highest_ptx_target(tmp_path):
    result = plan(tmp_path, build={'archs': [89, 75, 130]})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['cuda_architectures'] == [
        '89-real', '75-real', '130-real', '130-virtual']


def test_plan_rejects_non_windows_component_before_writes(tmp_path):
    result = plan(tmp_path, component={'platform': 'linux-x86_64'})
    assert result.returncode != 0
    assert 'PREFETCH_BUILD:' in result.stderr


@pytest.mark.parametrize('repository,directory,wrong_head', [
    ('Niko1221/Strata', 'src', False), ('ggml-org/llama.cpp', 'ggml', False), ('Niko1221/Strata', 'src', True)])
def test_git_sparse_source_inside_parent_repository_has_exact_independent_tree(tmp_path, repository, directory, wrong_head):
    git = shutil.which('git.exe') or shutil.which('git')
    assert git, 'Git is required to exercise the real nested-repository failure'
    remote = tmp_path / 'local-public-remote'
    remote.mkdir()

    def remote_git(*arguments):
        return subprocess.run([git, '-C', str(remote), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()

    remote_git('init')
    remote_git('config', 'uploadpack.allowFilter', 'true')
    (remote / directory).mkdir()
    (remote / directory / 'feature.cpp').write_text('before\n')
    if directory == 'src':
        (remote / 'bench').mkdir()
        (remote / 'bench/gdn_chunk_bench.cpp').write_text('required by CMake\n')
    (remote / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\n')
    (remote / 'docs').mkdir()
    (remote / 'docs/excluded.md').write_text('not materialized\n')
    remote_git('add', '.')
    remote_git('-c', 'user.name=Builder test', '-c', 'user.email=builder@example.invalid', 'commit', '-m', 'Fixture')
    commit = remote_git('rev-parse', 'HEAD')
    parent = tmp_path / 'parent-repository'
    parent.mkdir()
    subprocess.run([git, 'init', str(parent)], check=True, capture_output=True)
    work = parent / 'work'
    work.mkdir()
    source = work / 'source'
    patch = tmp_path / 'startup.patch'
    patch.write_text(f'diff --git a/{directory}/feature.cpp b/{directory}/feature.cpp\n'
                     f'--- a/{directory}/feature.cpp\n+++ b/{directory}/feature.cpp\n'
                     '@@ -1 +1 @@\n-before\n+after\n')
    calls = tmp_path / 'git-calls.txt'

    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    # Redirect only the public remote to a local Git fixture. Acquisition, sparse
    # checkout, commit validation and patch application still use the real Git.
    runner = tmp_path / 'acquire-git-source.ps1'
    runner.write_text(f"""$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ast = [Management.Automation.Language.Parser]::ParseFile({quote(SCRIPT)}, [ref]$null, [ref]$null)
foreach ($function in $ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }}, $false)) {{
    . ([scriptblock]::Create($function.Extent.Text))
}}
$work = {quote(work)}
$git = {quote(git)}
$env:HTTPS_PROXY = 'http://fixture-user:fixture-secret@127.0.0.1:17890'
$realRun = (Get-Command Run).ScriptBlock
function Run([string]$Program, [string[]]$Arguments, [string]$Directory, [switch]$Capture, [switch]$Quiet, [string]$Label) {{
    Add-Content -LiteralPath {quote(calls)} -Value ($Arguments -join ' ')
    $forward = @($Arguments)
    $remoteIndex = [array]::IndexOf($forward, 'https://github.com/{repository}.git')
    if ($remoteIndex -ge 0) {{ $forward[$remoteIndex] = {quote(remote)} }}
    if ($Arguments -contains 'fetch') {{
        if ($env:GIT_CONFIG_KEY_0 -ne 'http.proxy' -or $env:GIT_CONFIG_VALUE_0 -ne $env:HTTPS_PROXY) {{
            throw 'Selected proxy was not passed through process environment'
        }}
    }}
    if ({'$true' if wrong_head else '$false'} -and $Arguments -contains 'rev-parse') {{ return ('0' * 40) }}
    & $realRun -Program $Program -Arguments $forward -Directory $Directory -Capture:$Capture -Quiet:$Quiet -Label $Label
}}
function Download {{ throw 'NETWORK_NOT_ALLOWED' }}
try {{
Get-GitHubSource '{repository}' '{commit}' {quote(source)} 'fixture' | Out-Null
Run $git @('-C', {quote(source)}, 'apply', '--check', {quote(patch)}) $work
Run $git @('-C', {quote(source)}, 'apply', {quote(patch)}) $work
}} catch {{
    [Console]::Error.WriteLine('PREFETCH_BUILD: ' + $_.Exception.Message)
    exit 1
}}
""", encoding='utf-8-sig')
    result = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(runner)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
    if wrong_head:
        assert result.returncode != 0
        assert 'HEAD does not match' in result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert (source / directory / 'feature.cpp').read_text() == 'after\n', result.stdout
    assert (source / 'CMakeLists.txt').exists()
    if directory == 'src':
        assert (source / 'bench/gdn_chunk_bench.cpp').exists()
    assert not (source / 'docs').exists()
    top = subprocess.run([git, '-C', str(source), 'rev-parse', '--show-toplevel'],
                         check=True, capture_output=True, text=True).stdout.strip()
    assert Path(top).resolve() == source.resolve()
    assert not (parent / directory).exists()
    commands = calls.read_text(encoding='utf-8-sig')
    assert f'https://github.com/{repository}.git' in commands
    assert 'fetch --depth=1 --filter=blob:none' in commands
    assert 'http.version=HTTP/1.1' in commands
    assert 'http.lowSpeedLimit=1024' in commands and 'http.lowSpeedTime=30' in commands
    assert 'credential.helper=' in commands
    expected_paths = 'include src cmake third_party tests tools bench' if directory == 'src' else 'ggml'
    assert f'sparse-checkout set --cone -- {expected_paths}' in commands
    assert 'fixture-secret' not in result.stdout + result.stderr + commands


@pytest.mark.parametrize('dirty', [False, True])
def test_source_directory_snapshots_only_clean_materialized_sparse_files(tmp_path, dirty):
    git = shutil.which('git.exe') or shutil.which('git')
    assert git, 'Git is required to exercise sparse-checkout source snapshots'
    original = tmp_path / 'sparse-source'
    original.mkdir()

    def git_run(*arguments):
        return subprocess.run([git, '-C', str(original), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()

    git_run('init')
    (original / 'src').mkdir()
    (original / 'src/feature.cpp').write_text('tracked source\n')
    (original / 'docs').mkdir()
    (original / 'docs/unmaterialized.md').write_text('not a build input\n')
    git_run('add', '.')
    git_run('-c', 'user.name=Builder test', '-c', 'user.email=builder@example.invalid', 'commit', '-m', 'Fixture')
    commit = git_run('rev-parse', 'HEAD')
    git_run('sparse-checkout', 'set', '--cone', 'src')
    assert not (original / 'docs').exists()
    (original / 'src/untracked.cpp').write_text('do not copy\n')
    if dirty:
        (original / 'src/feature.cpp').write_text('locally modified\n')
    before_index = (original / '.git/index').read_bytes()
    before_source = (original / 'src/feature.cpp').read_bytes()
    work = tmp_path / 'work'
    work.mkdir()
    source = work / 'source'

    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    runner = tmp_path / 'snapshot.ps1'
    runner.write_text(f"""$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ast = [Management.Automation.Language.Parser]::ParseFile({quote(SCRIPT)}, [ref]$null, [ref]$null)
foreach ($function in $ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }}, $false)) {{
    . ([scriptblock]::Create($function.Extent.Text))
}}
$work = {quote(work)}
$git = {quote(git)}
try {{
    Copy-GitSourceSnapshot {quote(original)} {quote(source)} '{commit}'
}} catch {{
    [Console]::Error.WriteLine('PREFETCH_BUILD: ' + $_.Exception.Message)
    exit 1
}}
""", encoding='utf-8-sig')
    result = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(runner)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
    assert (original / '.git/index').read_bytes() == before_index
    assert (original / 'src/feature.cpp').read_bytes() == before_source
    assert not (original / 'docs').exists()
    if dirty:
        assert result.returncode != 0
        assert 'tracked' in result.stderr.lower()
        assert not source.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (source / 'src/feature.cpp').read_text() == 'tracked source\n'
        assert not (source / 'src/untracked.cpp').exists()
        assert not (source / 'docs').exists()
        top = subprocess.run([git, '-C', str(source), 'rev-parse', '--show-toplevel'],
                             check=True, capture_output=True, text=True).stdout.strip()
        assert Path(top).resolve() == source.resolve()


def run_builder_functions(tmp_path, body):
    script = str(SCRIPT).replace("'", "''")
    runner = tmp_path / 'builder-functions.ps1'
    runner.write_text(f"""$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$ast = [Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$null, [ref]$null)
foreach ($function in $ast.FindAll({{ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }}, $false)) {{
    . ([scriptblock]::Create($function.Extent.Text))
}}
try {{
{body}
}} catch {{
    [Console]::Error.WriteLine('PREFETCH_BUILD: ' + $_.Exception.Message)
    exit 1
}}
""", encoding='utf-8-sig')
    return subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(runner)],
                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)


@pytest.mark.parametrize('exit_code', [0, 22])
def test_download_has_hard_curl_timeouts_and_reports_nonzero_exit(tmp_path, exit_code):
    curl = tmp_path / 'fake curl.cmd'
    arguments = tmp_path / 'curl-arguments.txt'
    curl.write_text('@echo off\r\necho %* > "%FAKE_CURL_ARGUMENTS%"\r\n'
                    f'exit /b {exit_code}\r\n')
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    result = run_builder_functions(tmp_path, f"""
$work = {quote(tmp_path)}
$fakeCurl = {quote(curl)}
$env:FAKE_CURL_ARGUMENTS = {quote(arguments)}
$env:HTTPS_PROXY = 'http://proxy-user:fixture-secret@127.0.0.1:17890'
$env:HTTP_PROXY = 'http://must-not-override-https.invalid'
function Find-Tool {{ return $fakeCurl }}
function Invoke-WebRequest {{ throw 'NETWORK_NOT_ALLOWED' }}
Download 'https://codeload.github.com/example/project/tar.gz/abc' {quote(tmp_path / 'archive.tar.gz')}
""")
    if exit_code:
        assert result.returncode != 0
        assert 'PREFETCH_BUILD:' in result.stderr
        assert 'exit 22' in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
    value = arguments.read_text()
    assert '--connect-timeout 15' in value
    assert '--max-time 180' in value
    assert '--fail' in value and '--location' in value
    assert '--proxy http://proxy-user:fixture-secret@127.0.0.1:17890' in value
    assert '--retry' not in value
    assert 'fixture-secret' not in result.stdout + result.stderr


@pytest.mark.parametrize('engine_cuda,nvcc_release,compatible', [('13.0', '13.1', True), ('12.6', '13.0', False)])
def test_cuda_toolkit_must_match_engine_major_before_building(tmp_path, engine_cuda, nvcc_release, compatible):
    result = run_builder_functions(tmp_path,
        f"Assert-CudaMajor '{engine_cuda}' 'Cuda compilation tools, release {nvcc_release}, V{nvcc_release}.88'")
    if compatible:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert 'CUDA toolkit major mismatch' in result.stderr


@pytest.mark.parametrize('exit_code', [0, 7])
def test_msvc_batch_with_spaces_is_called_without_exposing_environment(tmp_path, exit_code):
    tools = tmp_path / 'VS & build tools'
    tools.mkdir()
    binary = tools / 'bin'
    binary.mkdir()
    (binary / 'cl.exe').write_bytes(b'fixture; not executed')
    vcvars = tools / 'vcvarsall.bat'
    vcvars.write_text('@echo off\r\nif /I not "%~1"=="x64" exit /b 9\r\n'
                      'set "MSVC_FIXTURE_SECRET=must-stay-in-memory"\r\n'
                      f'set "PATH={binary};%PATH%"\r\nexit /b {exit_code}\r\n')
    temporary = tmp_path / 'temp with spaces'
    temporary.mkdir()
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    result = run_builder_functions(tmp_path, f"""
$work = {quote(tmp_path)}
Import-MsvcEnvironment {quote(vcvars)} {quote(temporary)}
if ($env:MSVC_FIXTURE_SECRET -ne 'must-stay-in-memory') {{ throw 'Environment was not imported' }}
[Console]::Out.WriteLine('MSVC fixture ready')
""")
    if exit_code:
        assert result.returncode != 0
        assert 'exit 7' in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'MSVC fixture ready' in result.stdout
    assert 'must-stay-in-memory' not in result.stdout + result.stderr


@pytest.mark.parametrize('wrong_pin', [False, True])
def test_explicit_ggml_source_uses_clean_snapshot_bound_to_cmake_pin(tmp_path, wrong_pin):
    git = shutil.which('git.exe') or shutil.which('git')
    assert git
    original = tmp_path / 'readonly-ggml'
    original.mkdir()

    def original_git(*arguments):
        return subprocess.run([git, '-C', str(original), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()

    original_git('init')
    (original / 'ggml').mkdir()
    (original / 'ggml/CMakeLists.txt').write_text('project(ggml)\n')
    original_git('add', '.')
    original_git('-c', 'user.name=Builder test', '-c', 'user.email=builder@example.invalid', 'commit', '-m', 'Fixture')
    commit = original_git('rev-parse', 'HEAD')
    (original / 'ggml/untracked.cpp').write_text('not a build input\n')
    before_index = (original / '.git/index').read_bytes()
    before_source = (original / 'ggml/CMakeLists.txt').read_bytes()
    pin = '0' * 40 if wrong_pin else commit
    strata = tmp_path / 'strata-source'
    strata.mkdir()
    (strata / 'CMakeLists.txt').write_text(f'''FetchContent_Declare(strata_llamacpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG {pin}
        GIT_SHALLOW FALSE)
''')
    work = tmp_path / 'work'
    work.mkdir()
    output = tmp_path / 'output'
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    result = run_builder_functions(tmp_path, f"""
$work = {quote(work)}
$output = {quote(output)}
$git = {quote(git)}
function Get-GitHubSource {{ throw 'NETWORK_NOT_ALLOWED' }}
$resolved = Get-GgmlSource {quote(strata)} {quote(original)}
if ($resolved.commit -ne '{pin}') {{ throw 'GGML identity was not preserved' }}
if ($resolved.path -ne {quote(work / 'llama.cpp')}) {{ throw 'GGML is not isolated in WorkDirectory' }}
""")
    assert (original / '.git/index').read_bytes() == before_index
    assert (original / 'ggml/CMakeLists.txt').read_bytes() == before_source
    if wrong_pin:
        assert result.returncode != 0
        assert 'requested commit' in result.stderr
        assert not (work / 'llama.cpp').exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (work / 'llama.cpp/ggml/CMakeLists.txt').read_bytes() == before_source
        assert not (work / 'llama.cpp/ggml/untracked.cpp').exists()
