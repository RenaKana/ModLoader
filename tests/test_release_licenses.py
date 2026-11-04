import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('release_licenses', Path(__file__).parents[1] / 'scripts/collect_release_licenses.py')
licenses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(licenses)


def package(root, name, version='1.0.0', notice=True):
    directory = root / 'node_modules' / name
    directory.mkdir(parents=True)
    (directory / 'package.json').write_text(json.dumps({'name': name, 'version': version, 'license': 'MIT'}))
    if notice:
        (directory / 'LICENSE').write_text('Copyright fixture author\nPermission is hereby granted.\n')
    return directory


def test_collects_production_and_electron_not_other_dev_packages(tmp_path):
    for name in ('library', 'electron', 'dev-only'):
        package(tmp_path, name)
    (tmp_path / 'package-lock.json').write_text(json.dumps({'packages': {
        '': {}, 'node_modules/library': {'version': '1.0.0'},
        'node_modules/electron': {'version': '1.0.0', 'dev': True},
        'node_modules/dev-only': {'version': '1.0.0', 'dev': True},
    }}))
    records = licenses.collect_npm(tmp_path, tmp_path / 'out')
    assert {r['name'] for r in records} == {'library', 'electron'}
    for record in records:
        assert record['license_files']
        assert (tmp_path / 'out' / record['license_files'][0]).read_text().startswith('Copyright fixture author')


def test_missing_license_is_reported_not_silently_omitted(tmp_path):
    package(tmp_path, 'unlicensed', notice=False)
    (tmp_path / 'package-lock.json').write_text(json.dumps({'packages': {
        '': {}, 'node_modules/unlicensed': {'version': '1.0.0'},
    }}))
    with pytest.raises(RuntimeError, match='unlicensed@1.0.0'):
        licenses.collect_npm(tmp_path, tmp_path / 'out')


def test_lock_version_mismatch_is_reported(tmp_path):
    package(tmp_path, 'library', version='2.0.0')
    (tmp_path / 'package-lock.json').write_text(json.dumps({'packages': {
        '': {}, 'node_modules/library': {'version': '1.0.0'},
    }}))
    with pytest.raises(RuntimeError, match='version mismatch'):
        licenses.collect_npm(tmp_path, tmp_path / 'out')


def test_version_bound_supplement_retains_upstream_notice_and_provenance(tmp_path):
    package(tmp_path, 'library', notice=False)
    (tmp_path / 'package-lock.json').write_text(json.dumps({'packages': {
        '': {}, 'node_modules/library': {'version': '1.0.0'},
    }}))
    third_party = tmp_path / 'third_party'
    third_party.mkdir()
    (third_party / 'upstream-LICENSE.txt').write_text('Copyright upstream author')
    (third_party / 'npm-license-supplements.json').write_text(json.dumps({'library@1.0.0': {
        'files': ['third_party/upstream-LICENSE.txt'], 'sources': [{'url': 'https://example.org/library/v1.0.0/LICENSE'}],
    }}))
    [record] = licenses.collect_npm(tmp_path, tmp_path / 'out')
    assert (tmp_path / 'out' / record['license_files'][0]).read_text() == 'Copyright upstream author'
    assert record['upstream_materials'][0]['url'].endswith('/v1.0.0/LICENSE')
