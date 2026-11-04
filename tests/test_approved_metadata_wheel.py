import json
from pathlib import Path
import zipfile

import pytest

from modloader.backends import resolve_packages


def test_preconfirmed_nvidia_wheel_metadata_participates_in_full_resolution(tmp_path, monkeypatch):
    from pip._internal.cli import main as pip_cli
    from pip._internal.network.session import PipSession
    from pip._internal.operations.prepare import RequirementPreparer
    from pip._internal.req.constructors import install_req_from_line
    wheel = tmp_path / 'cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('cuda_tile-1.6.0rc5.dist-info/METADATA', 'Metadata-Version: 2.1\nName: cuda-tile\nVersion: 1.6.0rc5\nRequires-Dist: typing-extensions\n\n')
        archive.writestr('cuda_tile-1.6.0rc5.dist-info/WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-cp312-manylinux2014_x86_64\n')
    manifest = tmp_path / 'approved.json'
    manifest.write_text(json.dumps([{'path': str(wheel), 'url': resolve_packages.SGLANG_CUDA_TILE_URL,
                                    'digest': 'sha256:' + resolve_packages.SGLANG_CUDA_TILE_SHA256}]))
    namespace = {'__name__': 'bounded_resolver_test'}
    exec(compile(resolve_packages.SOURCE, 'resolver.py', 'exec'), namespace)
    # The destination helper is short-lived; restore its local pip patches here.
    for cls, name in [(PipSession, 'request'), (RequirementPreparer, 'prepare_linked_requirement'),
                      (RequirementPreparer, '_prepare_linked_requirement'), (RequirementPreparer, 'prepare_linked_requirements_more')]:
        monkeypatch.setattr(cls, name, getattr(cls, name))
    observed = []
    def resolver(argv):
        assert '--ignore-installed' in argv and '--only-binary=:all:' in argv
        request = install_req_from_line('cuda-tile @ ' + wheel.as_uri() + '#sha256=' + resolve_packages.SGLANG_CUDA_TILE_SHA256)
        distribution = RequirementPreparer.prepare_linked_requirement(object(), request)
        assert str(distribution.version) == '1.6.0rc5'
        assert list(map(str, distribution.iter_dependencies())) == ['typing-extensions']
        report = {'install': [{'metadata': {'name': 'cuda-tile', 'version': '1.6.0rc5'}, 'download_info': request.download_info.to_dict()}]}
        observed.extend(resolve_packages.package_files(report))
        return 0
    monkeypatch.setattr(pip_cli, 'main', resolver)
    assert namespace['main'](['--modloader-approved-wheels', str(manifest), 'sglang==0.5.21']) == 0
    assert observed[0]['url'] == resolve_packages.SGLANG_CUDA_TILE_URL
    assert observed[0]['digest'] == 'sha256:' + resolve_packages.SGLANG_CUDA_TILE_SHA256


def test_official_exception_does_not_allow_other_nvidia_payloads():
    report = {'install': [{'metadata': {'name': 'other', 'version': '1'}, 'download_info': {
        'url': 'https://pypi.nvidia.com/other/other.whl', 'archive_info': {'hashes': {'sha256': 'a' * 64}}}}]}
    with pytest.raises(ValueError, match='官方 wheel'):
        resolve_packages.package_files(report)
