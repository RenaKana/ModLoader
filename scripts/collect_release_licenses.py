"""Collect installed dependency notices for a Windows desktop distribution."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import quote


NOTICE = re.compile(r'^(licen[cs]e|copying|notice|copyright|authors)(?:$|[._-])', re.I)


def notice_files(directory):
    for parent, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ('node_modules', '.git')]
        for name in files:
            if NOTICE.match(name):
                yield Path(parent) / name


def copy_notices(files, base, output, prefix):
    copied = []
    for source in sorted(set(files)):
        relative = Path(prefix) / source.relative_to(base)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(relative.as_posix())
    return copied


def collect_npm(project, output):
    lock = json.loads((project / 'package-lock.json').read_text(encoding='utf-8'))
    supplement_path = project / 'third_party/npm-license-supplements.json'
    supplements = json.loads(supplement_path.read_text(encoding='utf-8')) if supplement_path.exists() else {}
    records, seen, missing = [], set(), []
    for location, entry in sorted(lock['packages'].items()):
        if not location or (entry.get('dev') and location != 'node_modules/electron'):
            continue
        directory = project / location
        metadata = json.loads((directory / 'package.json').read_text(encoding='utf-8'))
        name, version = metadata['name'], metadata['version']
        if version != entry['version']:
            raise RuntimeError(f'version mismatch: {name}: installed {version}, locked {entry["version"]}')
        identity = f'{name}@{version}'
        if identity in seen:
            continue
        seen.add(identity)
        prefix = 'npm/' + identity.replace('/', '__')
        files = copy_notices(notice_files(directory), directory, output, prefix)
        supplement = supplements.get(identity, {})
        files += copy_notices([project / p for p in supplement.get('files', [])], project / 'third_party', output, prefix + '/upstream')
        if not files:
            missing.append(identity)
        repository = metadata.get('repository', '')
        if isinstance(repository, dict):
            repository = repository.get('url', '')
        records.append({'ecosystem': 'npm', 'name': name, 'version': version,
                        'license': metadata.get('license', entry.get('license', 'See license files')),
                        'repository': repository, 'package_url': f'https://www.npmjs.com/package/{name}/v/{version}',
                        'archive_url': entry.get('resolved', ''), 'upstream_materials': supplement.get('sources', []), 'license_files': files})
    if missing:
        raise RuntimeError('Missing npm license texts: ' + ', '.join(missing))
    return records


def collect_python(output):
    records, missing = [], []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name, version = distribution.metadata['Name'], distribution.version
        if name.lower() == 'modloader':
            continue
        base = Path(distribution.locate_file(''))
        files = [Path(distribution.locate_file(f)) for f in distribution.files or [] if NOTICE.match(Path(f).name)]
        # Wheel RECORD paths can point outside site-packages; only notice files
        # inside the distribution's installed package area are copied.
        files = [f for f in files if f.is_file() and f.is_relative_to(base) and '..' not in f.relative_to(base).parts]
        copied = copy_notices(files, base, output, 'python/' + quote(f'{name}@{version}', safe='@.-'))
        if not copied:
            missing.append(f'{name}@{version}')
        records.append({'ecosystem': 'python-build-environment', 'name': name, 'version': version,
                        'license': distribution.metadata.get('License-Expression') or distribution.metadata.get('License') or 'See license files',
                        'project_urls': distribution.metadata.get_all('Project-URL') or [],
                        'package_url': f'https://pypi.org/project/{name}/{version}/', 'license_files': copied})
    if missing:
        raise RuntimeError('Missing Python license texts: ' + ', '.join(missing))
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    files = copy_notices([python_license], python_license.parent, output, 'python/CPython')
    records.append({'ecosystem': 'runtime', 'name': 'CPython', 'version': sys.version.split()[0],
                    'source_url': f'https://www.python.org/downloads/release/python-{sys.version_info.major}{sys.version_info.minor}{sys.version_info.micro}/',
                    'license_files': files})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = collect_npm(args.project, args.output) + collect_python(args.output)
    index = {'schema_version': 1, 'scope': 'npm production dependencies plus Electron; all distributions in the Python build environment (a superset of frozen runtime dependencies); CPython. Native sensor notices are shipped alongside the helper.', 'components': records}
    (args.output / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Dependency licenses', '', index['scope'], '', 'Each entry links to the installed package\'s original license and notice files.', '']
    for record in records:
        lines.extend([f'## {record["name"]} {record["version"]}', '', record.get('package_url', record.get('source_url', '')), ''])
        lines.extend(f'- [{p}]({quote(p, safe="/@+._-")})' for p in record['license_files'])
        lines.append('')
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Collected notices for {len(records)} components into {args.output}')


if __name__ == '__main__':
    main()
