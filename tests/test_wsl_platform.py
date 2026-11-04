"""Platform contracts with an isolated guest bridge; no installed WSL is started."""
import asyncio
import io
import json
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from modloader import platforms
from modloader.app import Controller, build_app
from modloader.assets import inspect_asset, scan_model_folder
from modloader.backends import get_adapters, import_runtime
from modloader.models import LaunchContext, LaunchSpec, ModelAsset, RuntimeInstallation


@pytest.fixture
def guest(tmp_path, monkeypatch):
    root = tmp_path / 'guest'
    root.mkdir()
    monkeypatch.setattr(platforms, 'require_wsl2', lambda distro: {'name': distro, 'version': 2})
    # Execute the production parser source with stdlib fixtures, translating only
    # the guest filesystem boundary. This does not claim real WSL execution.
    def python(distribution, code, args=(), *, timeout=60, input=None):
        assert distribution in ('External', 'Other')
        request = json.loads(input)
        if isinstance(request, list):
            request = [str(root if value == '/models' else root / value.removeprefix('/models/')) for value in request]
            result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], input=json.dumps(request),
                                    capture_output=True, encoding='utf-8', timeout=10)
            if result.returncode:
                raise ValueError(result.stderr)
            return result.stdout
        payload = request['payload']
        payload['path'] = str(root if payload['path'] == '/models' else root / payload['path'].removeprefix('/models/'))
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], input=json.dumps(request),
                                capture_output=True, encoding='utf-8', timeout=10)
        if result.returncode:
            raise ValueError(result.stderr)
        def native(value):
            if isinstance(value, str):
                return value.replace(str(root), '/models').replace('\\', '/')
            if isinstance(value, list): return [native(item) for item in value]
            if isinstance(value, dict): return {key: native(item) for key, item in value.items()}
            return value
        return json.dumps(native(json.loads(result.stdout)))
    monkeypatch.setattr(platforms, 'wsl_python', python)
    return root


def hf(root, name='HF'):
    path = root / name
    path.mkdir()
    (path / 'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':32,'torch_dtype':'float16'}))
    (path / 'tokenizer.json').write_text('{}')
    header = b'{"layer.mtp.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}'
    (path / 'model.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + b'00')
    return path


def test_legacy_fields_and_linux_identity_are_nonmutating():
    old = {'id':'r','backend':'vllm','name':'old','path':'/opt/vllm'}
    before = deepcopy(old)
    assert RuntimeInstallation(**old).platform == 'wsl2'
    assert old == before
    assert RuntimeInstallation(**{**old,'backend':'custom'}).platform == 'windows'
    def model(path, distribution):
        return ModelAsset(id='m',name='m',format='custom',path=path,location={'platform':'wsl2','distribution':distribution})
    assert platforms.model_identity(model('/models/Case','A')) != platforms.model_identity(model('/models/case','A'))
    assert platforms.model_identity(model('/models/Case','A')) != platforms.model_identity(model('/models/Case','B'))
    assert platforms.windows_path('A','/models/Case') == '\\\\wsl.localhost\\A\\models\\Case'


def test_list_never_starts_distribution_and_decodes_names(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0,stdout='  NAME       STATE     VERSION\n* 外部环境    Stopped   2\n  Old        Stopped   1\n'.encode('utf-16-le'))
    monkeypatch.setattr(platforms.subprocess, 'run', run)
    result = platforms.list_wsl()
    assert calls == [['wsl.exe','--list','--verbose']]
    assert result['distributions'][0]['name'] == '外部环境'
    with pytest.raises(ValueError, match='WSL2'):
        platforms.require_wsl2('Old')


def test_linux_hf_uses_real_metadata_and_rejects_incomplete_shards(guest):
    path = hf(guest)
    location = {'platform':'wsl2','distribution':'External'}
    model = inspect_asset('/models/HF', location=location)
    assert model.path == '/models/HF' and model.architecture == 'llama'
    assert model.quantization == 'float16' and model.size_bytes > 2
    assert model.files and all(value.startswith('/models/HF/') for value in model.files)
    from modloader.mtp import model_weight_info
    assert model_weight_info(model)['has_mtp'] is True
    (path / 'model.safetensors.index.json').write_text('{"weight_map":{"x":"missing.safetensors"}}')
    with pytest.raises(ValueError, match='分片'):
        inspect_asset('/models/HF', location=location)


def test_linux_gguf_inspection_reads_metadata_and_shards(guest):
    import struct
    def string(value):
        raw=value.encode(); return struct.pack('<Q',len(raw))+raw
    metadata = string('general.architecture')+struct.pack('<I',8)+string('llama')
    content = b'GGUF'+struct.pack('<IQQ',3,1,1)+metadata
    for name in ('part-00001-of-00002.gguf','part-00002-of-00002.gguf'):
        (guest/name).write_bytes(content)
    location={'platform':'wsl2','distribution':'External'}
    model=inspect_asset('/models/part-00002-of-00002.gguf',location=location)
    assert model.path.endswith('00001-of-00002.gguf') and model.architecture=='llama'
    assert len(model.files)==2 and model.size_bytes==len(content)*2
    (guest/'part-00002-of-00002.gguf').unlink()
    with pytest.raises(ValueError,match='分片'):
        inspect_asset('/models/part-00001-of-00002.gguf',location=location)


def test_linux_ninfer_import_and_scan_use_injected_container_parser(guest):
    from test_ninfer_assets import write_ninfer
    path, files, _ = write_ninfer(guest / 'native' / 'main.ninfer')
    location = {'platform': 'wsl2', 'distribution': 'External'}
    model = inspect_asset('/models/native/main.ninfer', location=location)
    assert model.format == 'ninfer' and model.path == '/models/native/main.ninfer'
    assert model.files == ['/models/native/main.ninfer', '/models/native/main.ninfer.part-0001']
    assert model.size_bytes == sum(part.stat().st_size for part in files)
    assert model.location.distribution == 'External' and not model.managed
    scanned = scan_model_folder('/models', location=location)
    assert not scanned['errors'] and len(scanned['models']) == 1
    assert scanned['models'][0].metadata['ninfer_validation'] == 'container'
    assert scanned['models'][0].metadata['import_root'] == '/models'
    files[1].write_bytes(files[1].read_bytes()[:-1])
    with pytest.raises(ValueError, match='长度不匹配'):
        inspect_asset('/models/native/main.ninfer', location=location)


async def test_normal_ninfer_import_api_preserves_linux_identity_and_duplicate_registration(guest, tmp_path):
    from test_ninfer_assets import write_ninfer
    write_ninfer(guest / 'native.ninfer')
    controller = Controller(tmp_path / 'isolated-ninfer-api', 'test-secret')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer test-secret'}) as client:
        try:
            location = {'platform': 'wsl2', 'distribution': 'External'}
            first = await client.post('/api/models/import', json={'path': '/models/native.ninfer', 'location': location})
            assert first.status_code == 200, first.text
            model = first.json()
            assert model['format'] == 'ninfer' and model['location'] == location
            again = await client.post('/api/models/import-folder', json={'path': '/models', 'location': location})
            assert again.status_code == 200 and again.json()['imported'] == [] and again.json()['existing'] == [model]
            other = await client.post('/api/models/import', json={'path': '/models/native.ninfer',
                                                               'location': {**location, 'distribution': 'Other'}})
            assert other.status_code == 200 and other.json()['id'] != model['id']
            assert len(controller.store.all('models')) == 2
        finally:
            await controller.close()


def test_wsl_template_save_is_syntax_only_and_platform_mismatch_is_rejected(monkeypatch):
    def forbid(*args, **kwargs): raise AssertionError('registration started WSL')
    monkeypatch.setattr(platforms.subprocess, 'run', forbid)
    runtime = import_runtime('custom','/env/bin/python', {'distribution':'External','custom':{
        'args':['-m','server','--model','{model_path}','--port','{port}'],'cwd':'/env'}}, platform='wsl2')
    assert runtime.platform == 'wsl2' and runtime.metadata['distribution']=='External'
    model=ModelAsset(id='m',name='m',path='/models/HF',format='hf',location={'platform':'wsl2','distribution':'Other'})
    with pytest.raises(ValueError,match='同一发行版'): platforms.validate_model_runtime(model,runtime)
    with pytest.raises(ValueError,match='Windows'): platforms.validate_model_runtime(model,runtime.model_copy(update={'platform':'windows'}))
    with pytest.raises(ValueError,match='shell'):
        import_runtime('custom','/bin/bash',runtime.metadata,platform='wsl2')


def test_platform_runner_preserves_native_argv_env_and_instance_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(platforms,'require_wsl2',lambda distro: {'version':2})
    monkeypatch.setattr(platforms,'windows_to_wsl',lambda path,distribution='ModLoader':'/mounted/'+Path(path).parent.name+'/'+Path(path).name)
    spec=LaunchSpec(platform='wsl2',distribution='External',argv=['/env/python','-m','server','/models/with space'],cwd='/env',env={'MODE':'a b'},model_name='m')
    wrapped=[]
    for name in ('one','two'):
        ctx=LaunchContext(port=1234,instance_id=name,work_dir=str(tmp_path/name),data_dir=str(tmp_path))
        wrapped.append(platforms.prepare_launch(spec,ctx))
    payload=json.loads(wrapped[0].argv[-1])
    assert payload=={'argv':spec.argv,'cwd':'/env','env':{'MODE':'a b'}}
    assert wrapped[0].cleanup_argv[-1]=='one' and wrapped[1].cleanup_argv[-1]=='two'
    assert wrapped[0].guest_record != wrapped[1].guest_record
    assert 'MODLOADER_INSTANCE_ID' in platforms.WSL_SUPERVISOR and 'saved_start is not None' in platforms.WSL_SUPERVISOR


@pytest.mark.asyncio
async def test_linux_import_preflight_launch_and_stop_with_isolated_bridge(guest, tmp_path, monkeypatch):
    hf(guest)
    import modloader.backends.custom as custom
    def stat(distribution,path):
        return {'is_file':path=='/env/python','is_dir':path!='/env/python','executable':True,'path':path,'size_bytes':0}
    monkeypatch.setattr(custom,'path_stat',stat)
    monkeypatch.setattr(platforms,'path_stat',stat)
    server=Path(__file__).parent/'fixtures/custom_engine_server.py'
    def bridge(spec,context):
        assert spec.platform=='wsl2' and spec.distribution=='External'
        argv=[sys.executable,str(server),*(str(guest/'HF') if item=='/models/HF' else item for item in spec.argv[2:])]
        return spec.model_copy(update={'argv':argv,'cwd':str(tmp_path),'cleanup_argv':[]})
    monkeypatch.setattr('modloader.runtime.prepare_launch',bridge)
    controller=Controller(tmp_path/'data','test-secret')
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),base_url='http://testserver',headers={'Authorization':'Bearer test-secret'}) as client:
            location={'platform':'wsl2','distribution':'External'}
            model_response=await client.post('/api/models/import',json={'path':'/models/HF','location':location})
            assert model_response.status_code==200,model_response.text
            model=model_response.json()
            again=await client.post('/api/models/import',json={'path':'/models/HF','location':location})
            assert again.json()['id']==model['id']
            runtime_response=await client.post('/api/runtimes/import',json={'backend':'custom','platform':'wsl2','path':'/env/python','metadata':{
                'distribution':'External','custom':{'args':['/bridge.py','--model','{model_path}','--host','{host}','--port','{port}','--api-model-name','{alias}'],'cwd':'/env'}}})
            assert runtime_response.status_code==200,runtime_response.text
            runtime=runtime_response.json()
            profile={'model_id':model['id'],'backend':'custom','runtime_id':runtime['id'],'name':'Linux','alias':'linux-test','parameters':{}}
            saved=await client.post('/api/profiles',json=profile)
            assert saved.status_code==200,saved.text
            preflight=await client.post('/api/profiles/preflight',json=profile)
            assert preflight.status_code==200 and preflight.json()['status']=='ready',preflight.text
            from modloader.models import LoadProfile
            instance=await controller.runtime.start(ModelAsset(**model),LoadProfile(**saved.json()),RuntimeInstallation(**runtime))
            async with asyncio.timeout(10):
                while instance.state=='starting': await asyncio.sleep(.05)
            assert instance.state=='ready',instance.error
            assert instance.platform=='wsl2' and instance.distribution=='External' and instance.pid
            blocked=await client.patch('/api/runtimes/'+runtime['id'],json={'platform':'windows'})
            assert blocked.status_code==409
            await controller.runtime.stop(instance.id)
            assert instance.state=='stopped' and controller.runtime.processes[instance.id].returncode is not None
    finally:
        await controller.close()


def test_list_uses_verified_ownership_not_reserved_name(tmp_path, monkeypatch):
    import modloader.backends.installers as installers
    monkeypatch.setattr(platforms.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=b'  ModLoader      Stopped      2\n'))
    def foreign(paths): raise ValueError('foreign registry path')
    monkeypatch.setattr(installers,'_verify_wsl_owner',foreign)
    assert platforms.list_wsl(tmp_path)['distributions'][0]['managed'] is False
    monkeypatch.setattr(installers,'_verify_wsl_owner',lambda paths:None)
    assert platforms.list_wsl(tmp_path)['distributions'][0]['managed'] is True
    assert platforms.list_wsl()['distributions'][0]['managed'] is False


def test_packaged_parser_resources_use_same_source(guest, tmp_path, monkeypatch):
    hf(guest)
    original=Path(platforms.__file__).parent
    package=tmp_path/'_internal'/'modloader'
    package.mkdir(parents=True)
    for name in ('assets.py','mtp.py'):
        (package/name).write_bytes((original/name).read_bytes())
    monkeypatch.setattr(platforms,'__file__',str(package/'platforms.py'))
    model=inspect_asset('/models/HF',location={'platform':'wsl2','distribution':'External'})
    from modloader.mtp import model_weight_info
    assert model.architecture=='llama' and model_weight_info(model)['has_mtp'] is True
    from test_ninfer_assets import write_ninfer
    write_ninfer(guest/'native.ninfer', lengths=(8,))
    native = inspect_asset('/models/native.ninfer', location={'platform':'wsl2','distribution':'External'})
    assert native.format == 'ninfer' and native.metadata['ninfer_version'] == 3


def test_runtime_resource_guard_distinguishes_distro_and_mapped_host_paths(tmp_path, monkeypatch):
    from modloader.runtime_management import protect_runtime_resources
    from modloader.store import Store
    store=Store(tmp_path/'guard.sqlite')
    controller=SimpleNamespace(store=store)
    runtime=RuntimeInstallation(id='r',backend='vllm',name='r',path='/opt/modloader/env/bin/vllm',metadata={
        'distribution':'ModLoader','linux_root':'/opt/modloader/env','managed_root':str(tmp_path/'env')})
    monkeypatch.setattr(platforms,'windows_to_wsl',lambda path,distribution='ModLoader':'/host/env')
    row=ModelAsset(id='m',name='kept',path='/opt/modloader/env/model',format='custom',location={'platform':'wsl2','distribution':'Other'})
    store.put('models',row.model_dump())
    protect_runtime_resources(controller,runtime,[])
    row.location.distribution='ModLoader'
    store.put('models',row.model_dump())
    with pytest.raises(ValueError,match='kept'): protect_runtime_resources(controller,runtime,[])
    row.location.distribution='Other'
    row.path='/host/env/model'
    store.put('models',row.model_dump())
    with pytest.raises(ValueError,match='kept'): protect_runtime_resources(controller,runtime,[])
    store.close()


def test_linux_folder_and_custom_import_use_native_paths(guest):
    from modloader.assets import inspect_custom_asset
    hf(guest)
    location={'platform':'wsl2','distribution':'External'}
    scanned=scan_model_folder('/models',location=location)
    assert not scanned['errors'] and len(scanned['models'])==1
    assert scanned['models'][0].path=='/models/HF'
    assert scanned['models'][0].metadata['import_root']=='/models'
    custom=inspect_custom_asset('/models/HF',location=location)
    assert custom.path=='/models/HF' and custom.format=='custom'
    assert custom.location.distribution=='External' and custom.metadata['size_bytes_known'] is False


async def test_wsl_import_rejects_files_removed_after_inspection(guest, tmp_path, monkeypatch):
    hf(guest)
    import modloader.app as app_module
    original = app_module.inspect_asset
    def inspected_then_removed(*args, **kwargs):
        model = original(*args, **kwargs)
        (guest / 'HF/model.safetensors').unlink()
        return model
    monkeypatch.setattr(app_module, 'inspect_asset', inspected_then_removed)
    controller = Controller(tmp_path / 'import-race', 'test-secret')
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                    headers={'Authorization': 'Bearer test-secret'}) as client:
            response = await client.post('/api/models/import', json={'path': '/models/HF', 'location': {'platform': 'wsl2', 'distribution': 'External'}})
            assert response.status_code == 400 and '移动或删除' in response.text
            assert controller.store.all('models') == []
    finally:
        await controller.close()
