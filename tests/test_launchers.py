import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows CMD launchers')


@pytest.fixture(scope='session')
def launch_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp('launcher-probe')
    source = directory / 'LaunchProbe.cs'
    source.write_text('''
using System;
using System.IO;
using System.Reflection;
class LaunchProbe {
    static void Main(string[] args) {
        File.WriteAllLines(Environment.GetEnvironmentVariable("MODLOADER_TEST_LAUNCH_RECORD"), new[] {
            Assembly.GetExecutingAssembly().Location,
            Environment.CurrentDirectory,
            Environment.GetEnvironmentVariable("MODLOADER_DATA_DIR") ?? "",
            Environment.GetEnvironmentVariable("MODLOADER_DEV_URL") ?? "",
            String.Join("\\t", args)
        });
    }
}
''', encoding='utf-8')
    executable = directory / 'LaunchProbe.exe'
    compiler = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    result = subprocess.run(
        [str(compiler), '/nologo', '/target:winexe', f'/out:{executable}', str(source)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.fixture
def project(tmp_path):
    directory = tmp_path / 'project with spaces'
    directory.mkdir()
    for name in ('Start-ModLoader.cmd', 'Start-ModLoader-Dev.cmd'):
        shutil.copyfile(ROOT / name, directory / name)
    return directory


def install_probe(project, relative_path, probe):
    target = project / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(probe, target)
    return target


def run_launcher(project, name, *args, data_dir=None):
    env = os.environ.copy()
    env.pop('MODLOADER_DATA_DIR', None)
    if data_dir is not None:
        env['MODLOADER_DATA_DIR'] = str(data_dir)
    env['MODLOADER_DEV_URL'] = 'http://127.0.0.1:1/stale-dev-server'
    env['MODLOADER_TEST_LAUNCH_RECORD'] = str(project / 'launch-record.txt')
    return subprocess.run(
        [env.get('COMSPEC', 'cmd.exe'), '/d', '/c', str(project / name), *args],
        cwd=project.parent, env=env, input='\n', capture_output=True, text=True, timeout=15,
    )


def read_launch_record(project):
    record = project / 'launch-record.txt'
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if record.exists():
            lines = record.read_text(encoding='utf-8-sig').splitlines()
            if len(lines) == 5:
                return lines
        time.sleep(0.02)
    pytest.fail('Launcher did not complete the isolated launch probe')


def prepare_source(project, probe, *, interface=True):
    executable = install_probe(project, 'node_modules/electron/dist/electron.exe', probe)
    python = project / '.venv/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    if interface:
        (project / 'dist').mkdir()
        (project / 'dist/index.html').write_text('fixture', encoding='utf-8')
    return executable


def test_release_without_package_stops_instead_of_launching_source(project, launch_probe):
    prepare_source(project, launch_probe)

    result = run_launcher(project, 'Start-ModLoader.cmd')

    assert result.returncode == 1
    assert r'release\win-unpacked\ModLoader.exe' in result.stdout
    assert 'npm run package' in result.stdout
    assert 'Start-ModLoader-Dev.cmd' in result.stdout
    assert not (project / 'launch-record.txt').exists()


def test_release_starts_packaged_executable_with_existing_project_data_default(project, launch_probe):
    executable = install_probe(project, 'release/win-unpacked/ModLoader.exe', launch_probe)
    prepare_source(project, launch_probe)

    result = run_launcher(project, 'Start-ModLoader.cmd', data_dir=project.parent / 'inherited-data')

    assert result.returncode == 0, result.stdout + result.stderr
    record = read_launch_record(project)
    assert Path(record[0]) == executable
    assert Path(record[1]) == project
    assert Path(record[2]) == project / 'data'
    assert record[3] == ''
    assert record[4] == ''


def test_release_local_override_runs_after_project_data_default(project, launch_probe):
    install_probe(project, 'release/win-unpacked/ModLoader.exe', launch_probe)
    (project / 'Start-ModLoader.local.cmd').write_text(
        '@echo off\n'
        '> "%~dp0local-input.txt" echo %MODLOADER_DATA_DIR%\n'
        'set "MODLOADER_DATA_DIR=%~dp0local-data"\n'
        'set "MODLOADER_DEV_URL=http://127.0.0.1:2/local-override"\n',
        encoding='utf-8',
    )

    result = run_launcher(project, 'Start-ModLoader.cmd', data_dir=project.parent / 'inherited-data')

    assert result.returncode == 0, result.stdout + result.stderr
    assert Path((project / 'local-input.txt').read_text().strip()) == project / 'data'
    record = read_launch_record(project)
    assert Path(record[2]) == project / 'local-data'
    assert record[3] == 'http://127.0.0.1:2/local-override'


@pytest.mark.parametrize('custom_data', [False, True])
def test_dev_starts_source_and_local_interface_not_packaged_executable(project, launch_probe, custom_data):
    executable = prepare_source(project, launch_probe)
    install_probe(project, 'release/win-unpacked/ModLoader.exe', launch_probe)
    data_dir = project.parent / 'isolated-data' if custom_data else None

    result = run_launcher(project, 'Start-ModLoader-Dev.cmd', data_dir=data_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    record = read_launch_record(project)
    assert Path(record[0]) == executable
    assert Path(record[1]) == project
    assert Path(record[2]) == (data_dir or project / 'data')
    assert record[3] == ''
    assert record[4] == '.'


def test_dev_without_built_interface_stops_with_rebuild_command(project, launch_probe):
    prepare_source(project, launch_probe, interface=False)

    result = run_launcher(project, 'Start-ModLoader-Dev.cmd')

    assert result.returncode == 1
    assert 'Start-ModLoader-Dev.cmd --build' in result.stdout
    assert not (project / 'launch-record.txt').exists()


@pytest.mark.parametrize('build_status', [0, 7])
def test_dev_build_launches_only_after_success(project, launch_probe, build_status):
    prepare_source(project, launch_probe)
    (project / 'scripts').mkdir()
    (project / 'scripts/build-cpu-sensors.ps1').write_text('exit 0\n', encoding='utf-8')
    (project / 'npm.cmd').write_text(
        '@echo off\n'
        '> "dist\\index.html" echo rebuilt\n'
        f'exit /b {build_status}\n',
        encoding='utf-8',
    )

    result = run_launcher(project, 'Start-ModLoader-Dev.cmd', '--build')

    assert (project / 'dist/index.html').read_text().strip() == 'rebuilt'
    if build_status:
        assert result.returncode == 1
        assert 'Build failed.' in result.stdout
        assert not (project / 'launch-record.txt').exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert Path(read_launch_record(project)[0]) == project / 'node_modules/electron/dist/electron.exe'
