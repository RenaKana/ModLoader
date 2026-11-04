import asyncio
import sys

import pytest

from modloader.backends import installers
from modloader.backends.source_builds import PINS
from modloader.backends.sglang import toolchain_environment
from modloader.models import RuntimeInstallation


async def test_actual_command_reports_early_compile_failure_after_thousands_of_parallel_warnings():
    script = '''
import sys
print('FAILED: sample_ops.o')
print('nvcc -DSPARKINFER_BUILD_NVFP4=1 sample_ops.cu')
print('source/kernels/csrc/cuda/fused/sample_ops.cu:13:10: fatal error: curand_kernel.h: No such file or directory')
print('   13 | #include <curand_kernel.h>')
print('      |          ^~~~~~~~~~~~~~~~~')
for index in range(7000): print('parallel CUTLASS warning ' + str(index))
print('ninja: build stopped: subcommand failed.')
sys.exit(1)
'''
    messages = []
    with pytest.raises(RuntimeError) as error:
        await installers._command([sys.executable, '-u', '-c', script], lambda event: messages.append(event.get('message')),
                                  asyncio.Event(), timeout=20)
    summary = str(error.value)
    assert 'FAILED: sample_ops.o' in summary and '-DSPARKINFER_BUILD_NVFP4=1' in summary
    assert 'fatal error: curand_kernel.h: No such file or directory' in summary
    assert '#include <curand_kernel.h>' in summary and '^~~~~~~~~~~~~~~~~' in summary
    assert 'parallel CUTLASS warning 6999' in summary and 'ninja: build stopped' in summary
    assert 'parallel CUTLASS warning 1000' not in summary
    assert 'parallel CUTLASS warning 1000' in messages  # Full output still reaches the existing log sink.


async def test_command_failure_diagnostic_memory_and_exception_are_bounded():
    script = "import sys\nfor index in range(120): print('ERROR: failure-' + str(index) + ' ' + 'x'*3000)\nsys.exit(1)"
    with pytest.raises(RuntimeError) as error:
        await installers._command([sys.executable, '-u', '-c', script], None, asyncio.Event(), timeout=20)
    summary = str(error.value)
    assert 'ERROR: failure-0 ' in summary and 'ERROR: failure-119 ' in summary
    bound = (installers.COMMAND_DIAGNOSTIC_LINES + installers.COMMAND_TAIL_LINES) * (installers.COMMAND_DIAGNOSTIC_LINE_CHARS + 1) + 100
    assert len(summary) <= bound


async def test_successful_command_keeps_existing_untruncated_tail_contract():
    value = 'x' * 3000
    script = f"for index in range(17): print(str(index) + ' ' + {value!r})"
    result = await installers._command([sys.executable, '-u', '-c', script], None, asyncio.Event(), timeout=20)
    assert result.splitlines() == [str(index) + ' ' + value for index in range(2, 17)]


@pytest.mark.parametrize('backend,flags,version', [
    ('sglang', '--model-path --host --port --served-model-name --enable-metrics', '0.5.21'),
    ('ninfer', '--host --port --spec --draft-tokens', PINS['ninfer']['version']),
    ('sparkinfer', '--host --port --ctx --tokenizer --draft-model', PINS['sparkinfer']['version']),
    ('vllm', '--model --host --port --served-model-name', '0.31.0'),
    ('transformers', '--host --port --model', '5.19.0'),
])
async def test_validation_reads_flags_before_long_help_tail_using_actual_process(monkeypatch, backend, flags, version):
    command = installers._command
    calls, events = [], []
    async def local_command(argv, progress, cancel, **options):
        calls.append(argv)
        # Only translate the WSL/external entrypoint: retain actual process output reading.
        script = (f"print({flags!r})\nfor index in range(200): print('later help description ' + str(index))"
                  if argv[-1] == '--help' else "print('Cuda compilation tools, release 13.4, V13.4.92')"
                  if argv[-2].endswith('/nvcc') else f'print({version!r})')
        return await command([sys.executable, '-u', '-c', script], progress, cancel, **options)
    monkeypatch.setattr(installers, '_command', local_command)
    runtime = RuntimeInstallation(id='fixture', backend=backend, name='Installed fixture', platform='windows' if backend == 'transformers' else 'wsl2',
                                  path='/opt/fixture/bin/python', metadata={'distribution': 'FixtureUbuntu', 'requested_version': version,
                                  **({'source_commit': PINS[backend]['commit'], 'source_root': '/opt/fixture/source',
                                      'python': '/opt/fixture/venv/bin/python'} if backend in PINS else {}),
                                  **({'linux_root': '/opt/fixture', 'build_environment': toolchain_environment('/opt/fixture')}
                                     if backend == 'sglang' else {})})
    result = await installers.validate_runtime(runtime, _installing=True, progress=events.append)
    assert result.version == version and set(flags.split()) <= set(result.metadata['supported_flags'])
    assert result.metadata['validation'] == 'entrypoint'
    assert any(event.get('message') == 'later help description 199' for event in events)
    assert all(event['phase'] == 'verify' for event in events)
    assert len(calls) == (3 if backend in {'sglang', 'ninfer'} else 1 if backend in PINS else 2)
    if backend == 'sglang':
        assert calls[-1][-2:] == ['/opt/fixture/cuda/bin/nvcc', '--version']
        assert result.metadata['toolchain_validation'] == 'nvcc'
        assert all('CUDA_HOME=/opt/fixture/cuda' in argv for argv in calls)
    else:
        assert calls[-1][-1] == '--help'
    if backend == 'ninfer':
        assert result.metadata['preparation_validation'] == 'tool_entrypoints'
        assert [argv[-2] for argv in calls[1:]] == ['tools.artifact.inspect', 'tools.convert']
        assert all('--chdir=/opt/fixture/source' in argv for argv in calls[1:])


async def test_sglang_validation_still_rejects_actual_missing_required_flag(monkeypatch):
    command = installers._command
    async def local_command(argv, progress, cancel, **options):
        script = ("print('--model-path --host --port --served-model-name')\nfor index in range(200): print('help description')"
                  if argv[-1] == '--help' else "print('0.5.21')")
        return await command([sys.executable, '-u', '-c', script], progress, cancel, **options)
    monkeypatch.setattr(installers, '_command', local_command)
    runtime = RuntimeInstallation(id='fixture', backend='sglang', name='Fixture', platform='wsl2', path='/opt/fixture/bin/python',
                                  metadata={'distribution': 'FixtureUbuntu'})
    with pytest.raises(ValueError, match='--enable-metrics'):
        await installers.validate_runtime(runtime)
