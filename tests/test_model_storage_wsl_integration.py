"""Opt-in real WSL transport integration; only uniquely named tiny fixtures are written."""
import asyncio
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import uuid

import pytest
import httpx

from modloader.app import Controller, build_app
from modloader.models import ModelAsset

from modloader.assets import inspect_asset
from modloader.model_storage import ModelStorageMigration, STORAGE_MARKER
from modloader.models import AppPaths, LoadProfile, RuntimeInstallation
from modloader.platforms import wsl_python, windows_to_wsl
from modloader.store import Store


@pytest.mark.skipif(os.environ.get('MODLOADER_STORAGE_WSL_INTEGRATION') != '1', reason='explicit real WSL fixture opt-in')
async def test_real_wsl_direct_import_round_trip_and_cancel(tmp_path):
    token = 'modloader-storage-fixture-' + uuid.uuid4().hex
    guest_source = '/tmp/' + token
    name = token + ' · 中文 空格'
    paths = AppPaths(tmp_path / 'isolated-data')
    # Read the installed distribution ownership marker; DB/work/model paths remain isolated.
    paths.runtimes = Path(__file__).resolve().parents[1] / 'data/runtimes'
    store = Store(paths.root / 'fixture.sqlite')
    service = ModelStorageMigration(paths, store, SimpleNamespace(instances={}, adapters={}), asyncio.Lock())
    owned_targets = []
    events = []
    async def progress(event):
        events.append(event)
    try:
        wsl_python('ModLoader', '''from pathlib import Path
import sys
p=Path(sys.argv[1]); p.mkdir(exist_ok=False)
(p/'config.json').write_text('{"model_type":"llama"}')
(p/'model.safetensors').write_bytes(b'tiny fixture, not inference weights')
(p/'tokenizer.json').write_text('{}')
''', [guest_source])
        controller = Controller(tmp_path / 'isolated-import-api', 'fixture-token')
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                        base_url='http://testserver', headers={'Authorization': 'Bearer fixture-token'}) as client:
                body = dict(path=guest_source, name=name, location={'platform': 'wsl2', 'distribution': 'ModLoader'})
                response = await client.post('/api/models/import', json=body)
                assert response.status_code == 200, response.text
                imported = ModelAsset(**response.json())
                assert imported.path == guest_source and len(imported.files) == 3
                duplicate = await client.post('/api/models/import', json=body)
                assert duplicate.status_code == 200 and duplicate.json()['id'] == imported.id
        finally:
            await controller.close()
        store.put('models', imported.model_dump())
        runtime = RuntimeInstallation(id='fixture-runtime', name='fixture', backend='custom', platform='wsl2',
                                      path='/usr/bin/python3', metadata={'distribution': 'ModLoader'})
        profile = LoadProfile(id='fixture-profile', name='fixture', model_id=imported.id, runtime_id=runtime.id,
                              backend='custom', alias='fixture', parameters={'tokenizer': guest_source + '/tokenizer.json'})
        store.put('runtimes', runtime.model_dump()); store.put('profiles', profile.model_dump())
        destination = tmp_path / 'Windows destination 中文'
        destination.mkdir()
        options = {'target_platform': 'windows', 'destination': str(destination)}
        preview = await service.preview(imported.id, options)
        assert preview['eligible'] and preview['delete_source_allowed'], preview
        result = await service.run(dict(model_id=imported.id, plan_id=preview['id'], delete_source=True),
                                   'real-wsl-to-windows', progress, asyncio.Event())
        assert result['source_deleted'], result
        windows_model = store.get('models', result['model_id'])
        assert (Path(windows_model['path']) / 'model.safetensors').read_bytes() == b'tiny fixture, not inference weights'
        assert store.get('profiles', profile.id)['parameters']['tokenizer'] == windows_to_wsl(str(Path(windows_model['path']) / 'tokenizer.json'), 'ModLoader')
        assert json.loads(wsl_python('ModLoader', 'import json,sys; from pathlib import Path; print(json.dumps(list(map(str,Path(sys.argv[1]).iterdir()))))', [guest_source])) == []
        # Exercise the unchanged guest supervisor copy path with the new readable Unicode directory.
        back = await service.preview(windows_model['id'])
        assert back['eligible'], back
        owned_targets.append((back['target_directory'], back['id']))
        native = await service.run(dict(model_id=windows_model['id'], plan_id=back['id'], delete_source=False),
                                   'real-windows-to-wsl', progress, asyncio.Event())
        assert PurePosixPath(back['target_directory']).name == name
        checked = inspect_asset(store.get('models', native['model_id'])['path'], location={'platform': 'wsl2', 'distribution': 'ModLoader'})
        assert checked.format == 'safetensors' and len(checked.files) == 3
        cancelled = await service.preview(native['model_id'], options)
        cancellation = asyncio.Event()
        async def cancel_progress(event):
            if event['phase'] == 'copying':
                cancellation.set()
        with pytest.raises(asyncio.CancelledError):
            await service.run(dict(model_id=native['model_id'], plan_id=cancelled['id']),
                              'real-cancel', cancel_progress, cancellation)
        assert not Path(cancelled['target_directory']).exists()
        assert store.get('models', native['model_id']) is not None
        print(json.dumps(dict(wsl_import=True, windows_move=True, native_copy=True, cancellation=True,
                              readable_name=name, progress_phases=sorted({row['phase'] for row in events})), ensure_ascii=False))
    finally:
        store.close()
        wsl_python('ModLoader', '''import json,shutil,sys
from pathlib import Path
source=Path(sys.argv[1]); token=sys.argv[2]
if source.parent!=Path('/tmp') or source.name!=token: raise ValueError('fixture source mismatch')
if source.exists(): shutil.rmtree(source)
for root,identifier in json.loads(sys.argv[3]):
    p=Path(root)
    if p.parent!=Path('/opt/modloader/models') or not p.name.startswith(token): raise ValueError('fixture target mismatch')
    if p.exists():
        if json.loads((p/'.modloader-model-storage.json').read_text())['model_id']!=identifier: raise ValueError('fixture owner mismatch')
        shutil.rmtree(p)
''', [guest_source, token, json.dumps(owned_targets)])
