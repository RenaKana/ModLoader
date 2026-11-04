"""Allocation arithmetic and header-only Strata preview contracts."""
import json
from pathlib import Path
import struct

import pytest

from modloader.models import LoadProfile, ModelAsset, RuntimeInstallation
from modloader.resource_estimates import strata_estimate
from modloader.strata_resource_estimates import _headers, _session


GEOMETRY = {'block_count': 48, 'full_attention_interval': 4, 'attention.head_count_kv': 2,
            'attention.key_length': 256, 'attention.indexer.key_length': 128,
            'rope.dimension_count': 64, 'ssm.state_size': 128, 'ssm.time_step_rank': 48,
            'ssm.group_count': 16, 'ssm.inner_size': 6144, 'ssm.conv_kernel': 4, 'expert_count': 512}


def gguf(path, metadata, tensors):
    def string(s):
        b = s.encode()
        return struct.pack('<Q', len(b)) + b
    data = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        data += string(key)
        data += struct.pack('<I', 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack('<I', value)
    offset = 0
    for name, shape, dtype, size in tensors:
        data += string(name) + struct.pack('<I', len(shape))
        data += struct.pack('<'+'Q'*len(shape), *shape) + struct.pack('<IQ', dtype, offset)
        offset += size
    data += bytes((-len(data)) % 32)
    with path.open('wb') as stream:
        stream.write(data)
        stream.truncate(len(data)+offset)


def fixture(tmp_path):
    path = tmp_path / 'model.gguf'
    metadata = {'general.architecture': 'qwen4exp', **{'qwen4exp.'+k: v for k, v in GEOMETRY.items()}}
    tensors = [('output.weight', [256, 32], 14, 210*32),
               ('token_embd.weight', [256, 32], 18, 98*32),
               ('blk.0.attn_qkv.weight', [256, 64], 18, 98*64),
               ('blk.1.ple_key.weight', [256, 64], 18, 98*64)]
    gguf(path, metadata, tensors)
    pack = tmp_path / 'pack'
    pack.mkdir()
    (pack/'dense.bin').write_bytes(bytes(20000))
    # name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits ...
    rows = []
    for name, shape, _, size in tensors:
        rows.append(f'{name} 0 0 0 {size} 0 {size*2} {shape[0]} {shape[1]} 8 0 32 0 0 0 0 0 0 0')
    rows.append('norm.weight 0 0 0 65 0 65 65 0 0 0 1 0 0 0 0 0 0 0')
    (pack/'index.txt').write_text('# align 64 pool 99999 tensors 5\n'+'\n'.join(rows))
    (pack/'native_experts.txt').write_text('\n'.join(f'{l} 18 18 0 {1000+l*10} 0 0 0' for l in range(48)))
    ranking = tmp_path/'rank.bin'
    ranking.write_bytes(b'STRP'+struct.pack('<5I', 1, 48, 512, 3, 3)+
                        b''.join(struct.pack('<HH', l, e) for l,e in [(0,0),(1,0),(2,0)]))
    model = ModelAsset(id='m', name='fixture', path=str(path), format='gguf')
    runtime = RuntimeInstallation(id='r', name='fixture', backend='strata', path=str(tmp_path), version='0.1.43')
    profile = LoadProfile(id='p', name='fixture', alias='fixture', model_id='m', runtime_id='r',
                          backend='strata', parameters={'pack': str(pack), 'expert_profile': str(ranking),
                                                       'context_length': 8192, 'kv_cache': 'q4_0'})
    return model, profile, runtime


def parts(result):
    return {x['label']: x['bytes'] for x in result['resources'][0]['components']}


def test_real_geometry_context_and_kv_storage():
    a = dict(_session(GEOMETRY, 8192, 'q4_0'))
    b = dict(_session(GEOMETRY, 131072, 'q4_0'))
    assert a['KV 缓存'] == 54*1024**2
    assert b['KV 缓存'] == 864*1024**2
    assert a['索引缓存'] == (8192//4+2)*128*4*12
    assert a['RoPE 表'] == 2*1024**2 and b['RoPE 表'] == 32*1024**2
    assert a['GDN 状态'] == b['GDN 状态'] == 117669888
    for kv, per_token in [('fp16', 2048), ('int8',1056), ('q4_0',576), ('k8v4',816)]:
        assert dict(_session(GEOMETRY, 8193, kv))['KV 缓存'] == 8196*per_token*12


def test_native_pack_skip_weights_and_dynamic_cache(tmp_path, monkeypatch):
    model, profile, runtime = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('No engine/model launch in preview')
    monkeypatch.setattr('subprocess.run', forbidden)
    monkeypatch.setattr(Path, 'write_text', forbidden)
    monkeypatch.setattr(Path, 'write_bytes', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    progress = []
    result = strata_estimate(model, profile, runtime, progress.append)
    assert [(event['completed'], event['total']) for event in progress if event['stage'] == 'model_metadata'] == [(0, 1), (1, 1)]
    assert progress[-1] == {'stage': 'allocation', 'message': '计算权重、上下文与专家缓存占用'}
    gpu = result['resources'][0]
    assert parts(result)['基础权重'] == 128
    assert parts(result)['原生投影权重'] == 2*98*64
    assert parts(result)['输出头'] == 210*32
    assert gpu['quality'] == 'dynamic' and gpu['bytes'] is None
    assert gpu['fixed_bytes'] == sum(parts(result).values())
    assert gpu['reserve_bytes'] == 700*1024**2
    assert not (tmp_path/'strata.json').exists()
    profile.extra_args = ['--keep-canonical']
    assert parts(strata_estimate(model, profile, runtime))['基础权重'] == 99999


def test_fixed_cache_budget_uses_profile_sizes_and_ram_complement(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    profile.parameters.update(expert_cache=2, resident_experts=True)
    result = strata_estimate(model, profile, runtime)
    gpu, ram = result['resources']
    # 2*max_blob=2940 budget holds two 1024-byte aligned profile entries.
    assert gpu['cache_budget_bytes'] == 2048
    assert gpu['budget_bytes'] == gpu['fixed_bytes']+2048
    assert gpu['bytes'] is None and gpu['quality'] == 'dynamic'
    total = sum(1000+l*10 for l in range(48))*512
    assert ram['expert_total_bytes'] == total
    assert ram['complement_bytes'] == total-2010  # excludes alignment padding
    assert ram['bytes'] is None
    assert 'budget_bytes' not in ram  # Hardware-dependent RAM cap is applied to the plan, not the raw complement.
    assert ram['resident_plan']['total_expert_bytes'] == total
    assert 'fixed_bytes' not in ram
    assert all(part.get('reference') for part in ram['components'])


def test_overrides_and_disabled_vision(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    cfg = tmp_path/'import.json'
    raw = {'cwd': str(tmp_path), 'args': ['--native', model.path, '--pack', 'missing', '--max-context', '4096',
           '--kv', 'fp16', '--expert-cache', 'auto'], 'vision': {'mmproj': 'missing'}}
    cfg.write_text(json.dumps(raw))
    profile.parameters.update(configuration=str(cfg), vision_enabled=False)
    profile.extra_args = ['--max-context','16384','--kv','int8','--expert-cache','2']
    result = strata_estimate(model, profile, runtime)
    assert parts(result)['KV 缓存'] == 16384*1056*12
    assert parts(result)['基础权重'] == 128  # explicit pack overrides imported one
    assert not any('视觉' in n for n in result['resources'][0]['notes'])
    assert json.loads(cfg.read_text()) == raw
    raw['args'][1] = str(tmp_path/'different.gguf')
    cfg.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='另一模型'):
        strata_estimate(model, profile, runtime)


def test_mtp_weights_kv_head_and_off(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    mtp = tmp_path/'mtp'
    mtp.mkdir()
    (mtp/'dense.bin').write_bytes(bytes(1024))
    with (mtp/'experts.bin').open('wb') as stream:
        stream.truncate(512*1382400)
    (mtp/'draft_vocab.bin').write_bytes(struct.pack('<4I',0,1,2,3))
    profile.parameters['strata_config'] = {'args': ['--mtp',str(mtp)]}
    result = strata_estimate(model, profile, runtime)
    p = parts(result)
    assert p['MTP 权重'] == 1024+512*1382400
    assert p['MTP KV'] == 8192*576
    assert p['MTP 草稿头'] == 4*210+16
    assert p['MTP logits'] == 6*32*4  # --spec 4 plus default suffix lookup 2
    profile.parameters['mtp_enabled'] = False
    assert not any(k.startswith('MTP') for k in parts(strata_estimate(model,profile,runtime)))


def test_missing_metadata_and_special_kv_keep_independent_components(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    profile.extra_args = ['--kv-resident','20480']
    result = strata_estimate(model, profile, runtime)
    assert 'KV 缓存' not in parts(result)
    assert 'GDN 状态' in parts(result) and '基础权重' in parts(result)
    model.path = str(tmp_path/'missing.gguf')
    gpu = strata_estimate(model, profile, runtime)['resources'][0]
    assert not gpu['components'] and gpu['quality'] == 'dynamic'
    assert 'fixed_bytes' not in gpu


def test_split_headers_read_all_shards_without_payload(tmp_path):
    first = tmp_path/'model-00001-of-00002.gguf'
    second = tmp_path/'model-00002-of-00002.gguf'
    gguf(first, {'general.architecture':'qwen4exp'}, [('a',[256,2],18,196)])
    gguf(second, {}, [('b',[256,3],18,294)])
    meta, tensors = _headers(first)
    assert set(tensors) == {'a','b'} and meta['general.architecture'] == 'qwen4exp'
    second.unlink()
    with pytest.raises(OSError):
        _headers(first)


def test_broken_pack_and_unknown_architecture_do_not_claim_weights(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    (Path(profile.parameters['pack'])/'dense.bin').unlink()
    result = strata_estimate(model, profile, runtime)
    assert '基础权重' not in parts(result) and 'KV 缓存' in parts(result)
    gguf(Path(model.path), {'general.architecture': 'unknown'}, [('output.weight',[256,32],14,6720)])
    result = strata_estimate(model, profile, runtime)
    assert not parts(result)


def test_pruned_target_does_not_prune_canonical_mtp(tmp_path):
    from modloader.strata_resource_estimates import allocation_preview, MTP_EXPERT_BYTES
    model, profile, runtime = fixture(tmp_path)
    gguf(Path(model.path), {'general.architecture': 'qwen4exp',
         **{'qwen4exp.'+k: v for k,v in {**GEOMETRY, 'expert_count': 256}.items()}},
         [('output.weight',[256,32],14,6720)])
    mtp = tmp_path/'mtp'
    mtp.mkdir()
    (mtp/'dense.bin').write_bytes(bytes(1024))
    with (mtp/'experts.bin').open('wb') as stream:
        stream.truncate(MTP_EXPERT_BYTES)
    (mtp/'draft_vocab.bin').write_bytes(struct.pack('<4I',0,1,2,3))
    result = allocation_preview(model, {'--mtp': str(mtp), '--max-context': '8192', '--kv':'q4_0'},
                                {'cwd': str(tmp_path)})
    p = {x['label']: x['bytes'] for x in result['components']}
    assert p['MTP 权重'] == 1024+707788800
    assert p['MTP KV'] == 8192*576
    # A truncated (target-pruned) MTP file must not be priced as a valid drafter.
    with (mtp/'experts.bin').open('wb') as stream:
        stream.truncate(MTP_EXPERT_BYTES//2)
    result = allocation_preview(model, {'--mtp': str(mtp)}, {'cwd': str(tmp_path)})
    assert 'MTP 权重' not in {x['label'] for x in result['components']}


def test_layer_split_never_attributes_whole_session_or_drafter_to_one_gpu(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    profile.extra_args = ['--layer-split','24','--mtp',str(tmp_path/'mtp')]
    gpu = strata_estimate(model, profile, runtime)['resources'][0]
    assert not gpu['components']
    assert any('最后一张 GPU' in n for n in gpu['notes'])


def test_streaming_kv_retains_full_index_and_rope(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    baseline = parts(strata_estimate(model, profile, runtime))
    profile.extra_args = ['--kv-resident','20480']
    streamed = parts(strata_estimate(model, profile, runtime))
    assert 'KV 缓存' not in streamed
    assert streamed['索引缓存'] == baseline['索引缓存']
    assert streamed['RoPE 表'] == baseline['RoPE 表']


def test_known_strata_parser_rejects_equals_but_keeps_launcher_profile_alias(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    profile.extra_args = ['--max-context=16384']
    with pytest.raises(ValueError, match='--max-context'):
        strata_estimate(model, profile, runtime)
    runtime.version = 'unknown'
    assert parts(strata_estimate(model, profile, runtime))['KV 缓存'] == 16384*576*12
    runtime.version = '0.1.43'
    profile.extra_args = ['--expert-profile='+profile.parameters['expert_profile']]
    assert 'KV 缓存' in parts(strata_estimate(model, profile, runtime))


def test_ram_modes_and_rank_descriptor_preserve_unranked_experts(tmp_path):
    model, profile, runtime = fixture(tmp_path)
    arena = strata_estimate(model,profile,runtime)['resources'][1]
    assert arena['logical'] is False and arena['resident_plan']['mode'] == 'arena'
    assert arena['bytes'] == arena['expert_total_bytes']
    profile.extra_args = ['--mmap-experts']
    mapped = strata_estimate(model,profile,runtime)['resources'][1]
    assert mapped['resident_plan']['mode'] == 'mmap' and mapped['bytes'] == 0
    profile.parameters['resident_experts'] = True
    resident = strata_estimate(model,profile,runtime)['resources'][1]
    plan = resident['resident_plan']
    assert plan['mode'] == 'resident' and plan['soft']
    assert plan['headroom_bytes'] == 4*2**30
    assert plan['rank_runs'] == [[0,1],[1,1],[2,1]]
    assert plan['gpu_rank_count'] == 3
    assert plan['unranked_counts'][:4] == [511,511,511,512]
    assert plan['exchange_bytes'] == 96*1470
    assert 'device' not in resident and plan['gpu_device'] == 'CUDA0'
    assert plan['prefill']['candidates'][-1][0] == 8192


def test_ram_budget_and_hard_mode_parameters(tmp_path,monkeypatch):
    model, profile, runtime = fixture(tmp_path)
    profile.extra_args = ['--mmap-experts','--resident-cpu-experts','--adapt-swaps','4']
    plan = strata_estimate(model,profile,runtime)['resources'][1]['resident_plan']
    assert not plan['soft'] and plan['budget_bytes'] is None
    assert plan['headroom_bytes'] == 8*2**30 and plan['exchange_bytes'] == 4*1470
    monkeypatch.setenv('STRATA_RESIDENT_HEADROOM_GIB','2.5')
    profile.extra_args += ['--resident-budget-gib','12','--adapt-async','1']
    plan = strata_estimate(model,profile,runtime)['resources'][1]['resident_plan']
    assert plan['budget_bytes'] == 12*2**30
    assert plan['headroom_bytes'] == int(2.5*2**30) and plan['exchange_bytes'] == 8*1470


def test_ple_host_cache_is_not_a_mapped_weight(tmp_path):
    from modloader.strata_resource_estimates import _host_buffers
    tensors = {'per_layer_token_embd.weight': {'shape':[160,320001536],'type':20}}
    entries = _host_buffers(tensors,{})
    assert entries == [{'label':'PLE 行缓存','bytes':98697216},{'label':'PLE 读取缓冲','bytes':2097152}]
    assert _host_buffers(tensors,{'--ple-io':'mmap'}) == []
    assert _host_buffers(tensors,{'--ple-row-cache':'0','--ple-inflight':'1'})[1]['bytes'] == 8192


def test_prefill_borrow_and_owned_counts_from_native_allocator(monkeypatch):
    from modloader.strata_resource_estimates import _prefill_plan
    for name in ('STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_GR_UNFUSED','STRATA_PF_HCDOWN',
                 'STRATA_HCD_EXACT','STRATA_PF_PAD','STRATA_PREFILL_MMQ','STRATA_PREFILL_RING',
                 'STRATA_PREFILL_STREAM_MIN','STRATA_PREFILL_LEND_PCT','STRATA_EMB_REUSE_ACCOUNT'):
        monkeypatch.delenv(name,raising=False)
    # Mixed IQ3_XXS pack: largest GU is IQ3_S, largest down is IQ4_NL.
    formats = [(21,20),(17,42)]
    opts = {'--prefill':'auto','--expert-profile':'rank.bin'}
    plan = _prefill_plan(opts,8192,512,2329600,formats,'resident')
    assert plan['ring_slots'] == 96 and plan['lend_percent'] == 85
    assert len(plan['candidates']) == 32
    assert plan['candidates'][-1] == [8192,3617237760,208331028]
    assert plan['fallback'] == [1024,838860800,58663188]
    assert all(a[1] < b[1] for a,b in zip(plan['candidates'],plan['candidates'][1:]))
    owned = _prefill_plan({**opts,'--no-prefill-borrow':None},8192,512,2329600,formats,'resident')
    assert owned['auto'] is False and owned['borrow'] is False and owned['fallback'][0] == 2048
