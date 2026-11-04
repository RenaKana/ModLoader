"""Header-only Strata allocation preview (0.1.43 native/pack layout).

No engine invocation, tensor payload reads or CUDA initialization. Components
are requested buffer bytes, not driver allocation granules or process RSS.
Source: Strata core/{weights,native_dense,native_head,layer,session,mtp}.cpp
and program/generate.cpp's expert-cache sizing and profile walk.
"""
from pathlib import Path
import os
import re
import struct

from .assets import GGUF_SHARD
from .resource_metadata import read_metadata

MIB = 1024 ** 2
# generate.cpp:4438-4457 gives the drafter canonical geometry even when the
# target is expert-pruned. expert.hpp:34-45 defines Q2_0 canonical expert blobs.
MTP_EXPERT_BYTES = 512 * 3 * (2560 * 640 * 18 // 64)
# native_mmvq.cu / iq_kernels.cu: GGML block elements, stored bytes.
BLOCKS = {2: (32, 18), 6: (32, 22), 7: (32, 24), 8: (32, 34),
          11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
          16: (256, 66), 17: (256, 74), 18: (256, 98), 20: (32, 18),
          21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 42: (64, 18)}
PROJECTIONS = ('.attn_qkv.weight', '.attn_gate.weight', '.ssm_out.weight',
               '.attn_q.weight', '.attn_k.weight', '.attn_v.weight', '.attn_output.weight',
               '.ffn_gate_shexp.weight', '.ffn_up_shexp.weight', '.ffn_down_shexp.weight')


def _align(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def _positive(value):
    n = int(value)
    if n <= 0:
        raise ValueError('非正的模型维度或上下文')
    return n


def _headers(path, progress=None):
    path = Path(path)
    match = GGUF_SHARD.match(path.name)
    paths = [path] if not match else [path.with_name(f'{match[1]}-{i:05d}-of-{int(match[3]):05d}{match[4]}')
                                    for i in range(1, int(match[3]) + 1)]
    metadata, tensors = {}, {}
    for index, shard in enumerate(paths):
        if progress:
            progress({'stage': 'model_metadata', 'message': f'读取分片信息 · {shard.name}', 'completed': index, 'total': len(paths)})
        item = read_metadata(shard)
        for source_tensor in item['tensors']:
            tensor = {**source_tensor, 'type': source_tensor['dtype']}
            if tensor['name'] in tensors:
                raise ValueError('GGUF 分片存在重复 tensor')
            tensors[tensor['name']] = tensor
        metadata.update(item['config'])
        if progress:
            progress({'stage': 'model_metadata', 'message': f'读取分片信息 · {shard.name}', 'completed': index + 1, 'total': len(paths)})
    return metadata, tensors


def _tensor_bytes(tensor):
    shape, kind = tensor['shape'], tensor['type']
    if len(shape) != 2 or any(n <= 0 for n in shape):
        raise ValueError('不支持的 tensor 形状')
    if kind in (0, 1, 30):
        return shape[0] * shape[1] * (4 if kind == 0 else 2)
    block, size = BLOCKS[kind]
    if shape[0] % block:
        raise ValueError('量化行未对齐')
    return shape[0] // block * size * shape[1]


def _pack(path):
    rows, alignment, pool = {}, None, None
    dense_size = (path / 'dense.bin').stat().st_size
    for line in (path / 'index.txt').read_text(encoding='utf-8').splitlines():
        header = re.match(r'# align (\d+) pool (\d+) tensors (\d+)', line)
        if header:
            alignment, pool = int(header[1]), int(header[2])
        elif line and not line.startswith('#'):
            fields = line.split()
            if len(fields) != 19:
                raise ValueError('Strata pack index 行格式不支持')
            row = [int(v) for v in fields[1:]]
            if row[0] == 0 and (row[2] < 0 or row[3] < 0 or row[2]+row[3] > dense_size):
                raise ValueError('Strata dense 文件被截断')
            rows[fields[0]] = row
    if not alignment or pool is None or not rows:
        raise ValueError('Strata pack index 缺少布局')
    return rows, alignment, pool


def _weights(tensors, pack, options):
    rows, alignment, pool = _pack(pack)
    served = set()
    native_bytes = 0
    for name, tensor in tensors.items():
        ple = name == 'blk.1.ple_key.weight' and tensor['type'] in (42, 18, 23, 8)
        eligible = name.startswith('blk.') and (name.endswith(PROJECTIONS) or ple)
        if eligible and tensor['type'] in BLOCKS and len(tensor['shape']) == 2:
            row = rows.get(name)
            if row is None:
                raise ValueError('native tensor 不在 pack 中')
            if ple and not row[8]:  # code_bits: canonical BF16 PLE stays in the arena
                continue
            if tensor['shape'] != row[6:8]:
                raise ValueError('pack 与 GGUF tensor 维度不匹配')
            served.add(name)
            native_bytes += _tensor_bytes(tensor)
    head = tensors.get('output.weight')
    if not head or head['type'] not in BLOCKS:
        raise ValueError('缺少支持的 native output head')
    skip = served | {'output.weight', 'token_embd.weight'}
    canonical = pool if '--keep-canonical' in options else sum(
        _align(row[5], alignment) for name, row in rows.items() if name not in skip)
    return [('基础权重', canonical), ('原生投影权重', native_bytes), ('输出头', _tensor_bytes(head))]


def _geometry(metadata):
    if metadata.get('general.architecture') != 'qwen4exp':
        raise ValueError('尚未支持该 Strata 架构')
    prefix = 'qwen4exp.'
    names = ('block_count', 'full_attention_interval', 'attention.head_count_kv',
             'attention.key_length', 'attention.indexer.key_length', 'rope.dimension_count',
             'ssm.state_size', 'ssm.time_step_rank', 'ssm.group_count', 'ssm.inner_size',
             'ssm.conv_kernel', 'expert_count')
    return {key: _positive(metadata[prefix + key]) for key in names}


def _session(g, context, kv):
    """Separately identifiable persistent state; scratch remains a named omission."""
    h, d = g['attention.head_count_kv'], g['attention.key_length']
    if d % 64:
        raise ValueError('KV head 维度未满足 Strata 量化分组')
    rows = {'fp16': h*d*4, 'int8': h*d*2 + h*(d//64)*4,
            'q4_0': h*(d//32)*18*2, 'k8v4': h*(d + d//64*2 + d//32*18)}
    per_cell = rows[kv]
    qsa = g['block_count'] // g['full_attention_interval']
    gdn = g['block_count'] - qsa
    channels = g['ssm.inner_size'] + 2*g['ssm.group_count']*g['ssm.state_size']
    recurrent = gdn * (g['ssm.state_size']**2*g['ssm.time_step_rank'] +
                       channels*(g['ssm.conv_kernel']-1))*4
    # layer.cpp KvPlan::pooled_rows includes TWO spare rows, not one.
    return [('KV 缓存', _align(context, 4)*per_cell*qsa),
            ('索引缓存', (context//4+2)*g['attention.indexer.key_length']*4*qsa),
            ('RoPE 表', context*g['rope.dimension_count']*4), ('GDN 状态', recurrent)]


def _experts(pack, g, profile, count, per_layer):
    blobs, formats = {}, {}
    for line in (pack / 'native_experts.txt').read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            if len(fields) not in (8, 9):
                raise ValueError('native experts 布局不支持')
            layer = int(fields[0])
            blobs[layer] = _positive(fields[4])
            formats[layer] = (int(fields[1]), int(fields[2]))
    nl, ne = g['block_count'], g['expert_count']
    if set(blobs) != set(range(nl)):
        raise ValueError('native experts 层数不匹配')
    total = sum(blobs.values()) * ne
    ranked_layers, seen = [], set()
    if profile:
        with profile.open('rb') as stream:
            header = stream.read(24)
            if len(header) != 24 or header[:4] != b'STRP':
                raise ValueError('expert profile 文件头无效')
            version, layers, experts, slots, ranked = struct.unpack('<5I', header[4:])
            if version != 1 or (layers, experts) != (nl, ne) or ranked > slots:
                raise ValueError('expert profile 布局不匹配')
            for _ in range(ranked):
                pair = stream.read(4)
                if len(pair) != 4:
                    raise ValueError('expert profile 被截断')
                layer, expert = struct.unpack('<HH', pair)
                if layer >= nl or expert >= ne or (layer, expert) in seen:
                    raise ValueError('expert profile 专家无效或重复')
                seen.add((layer, expert))
                ranked_layers.append(layer)
    runs, unranked = [], [ne]*nl
    for layer in ranked_layers:
        unranked[layer] -= 1
        if runs and runs[-1][0] == layer:
            runs[-1][1] += 1
        else:
            runs.append([layer, 1])
    descriptor = {'total_expert_bytes': total, 'max_blob_bytes': max(blobs.values()),
                  'layer_blob_bytes': [blobs[l] for l in range(nl)],
                  'rank_runs': runs, 'gpu_rank_count': len(ranked_layers),
                  'unranked_counts': unranked, 'experts_per_layer': ne,
                  'cache_slots': count, 'per_layer': per_layer}
    if count is None:
        return total, None, None, descriptor, list(formats.values())
    budget = count * max(blobs.values())
    if per_layer:
        quota = min(ne, budget // sum(_align(v, 256) for v in blobs.values()))
        # Sized slots are allocated uniformly, but only ranked pairs are filled.
        counts = [min(quota, ne-unranked[l]) for l in range(nl)]
        selected = sum(counts[l]*blobs[l] for l in range(nl))
        return total, quota*sum(_align(v,256) for v in blobs.values()), total-selected, descriptor, list(formats.values())
    if not profile:
        return total, budget, None, descriptor, list(formats.values())
    used, selected = 0, 0
    for layer in ranked_layers:
        size = _align(blobs[layer], 256)
        if used + size > budget:
            break
        used += size
        selected += blobs[layer]
    return total, used, total-selected, descriptor, list(formats.values())


def _prefill_plan(options, context, experts, max_blob, formats, mode):
    """CUDA native MMQ/default non-fused byte counter, no kernel initialization.

    Mirrors Prefill::bytes_needed_impl/set_bytes/Alloc and moe_mmq::q8_bytes.
    Resident copies are made AFTER pinned_share is sampled, so the mmap source
    uses the unpinned 96-slot ring and 85% lend ceiling at this sizing point.
    """
    value = str(options.get('--prefill', 'auto'))
    auto = value.startswith('auto')
    requested = int(value.split(':', 1)[1]) if ':' in value else 8192 if auto else int(value)
    if auto and ('--no-prefill-borrow' in options or not options.get('--expert-profile')):
        auto, requested = False, 2048  # generate.cpp:2430-2432
    ceiling = max(8192, min(requested, context)) if auto else requested
    # Fused kernels and alternate layouts require device/build capability checks.
    if any(os.environ.get(key) not in (None, '', '0') for key in
           ('STRATA_PF_FUSED', 'STRATA_PREFILL_BF16X2', 'STRATA_GR_UNFUSED',
            'STRATA_PF_HCDOWN', 'STRATA_HCD_EXACT', 'STRATA_PF_PAD')):
        raise ValueError('非默认 prefill kernel 布局')
    if os.environ.get('STRATA_PREFILL_MMQ') == '0':
        raise ValueError('MMQ 已关闭')
    supported = {16, 17, 18, 20, 21, 22, 23, 42, 8, 12, 13, 7}
    if not formats or any(a not in supported or b not in supported for a, b in formats):
        raise ValueError('prefill MMQ 专家格式未覆盖')
    def matrix(kind, rows, cols):
        block, size = BLOCKS[kind]
        return rows*(cols//block)*size
    gu = max(matrix(a,1280,2560) for a,b in formats)
    down = max(matrix(b,2560,640) for a,b in formats)
    ring = 96 if mode in ('resident','mmap') else 384
    ring = int(os.environ.get('STRATA_PREFILL_RING', ring))
    ring = 8 if ring == 8 else max(16, min(ring,512))
    stream_min = int(os.environ.get('STRATA_PREFILL_STREAM_MIN',1024))
    lend_percent = int(os.environ.get('STRATA_PREFILL_LEND_PCT',85 if mode in ('resident','mmap') else 90))
    def amount(chunk, owned=False):
        T = chunk
        def takes(values):
            return sum(_align(_align(v+256,256),2*MIB) if owned else _align(v+256,256) for v in values)
        # Region set counters always use 256-byte carving, even in owned mode.
        carve = lambda values: sum(_align(v+256,256) for v in values)
        gdn = carve([T*x*4 for x in (10240,6144,96,48,48,10240,6144)]+[T*6144*2])
        attn_scratch = 33*24*(256+2)+64  # ceil(2051/64), qsa_decode_attn.cpp
        qsa = carve([T*x*4 for x in (512,512,12288,6144,128,512,6144)]+
                    [T*6144*2,T*2051*4,256*(context//4+2)*4,32*attn_scratch*4])
        q8 = lambda rows, cols: rows*_align(cols,512)*144//128+128*144
        moe = carve([T*experts*4]+[T*10*4]*4+[T*10*1280*4,T*10*2560*4,
                    T*640*4,T*640*4,T*640*2,T*2560*4,T*4,
                    q8(T*10,2560),T*10*640*4,q8(T*10,640)])
        values = [64*MIB,32*MIB,T*2560*4,T*10240*4,T*4*4,
                  T*10240*2,T*320*4,T*320*2,T*10240*4,T*4*4,T*2560*4,
                  T*2560*2,T*2560*2]
        if os.environ.get('STRATA_EMB_REUSE_ACCOUNT') != '1':
            values.append(T*2560*4)
        values += [T*4*4,max(gdn,qsa,moe),1280*2560*2,2560*640*2,
                   1280*2560*2,2560*640*2,T*10*4,
                   2*(experts+experts//16+2)*4,16*gu+4096,16*down+4096]
        r = ring if T >= stream_min else 8
        values += [r*max_blob] if owned else [max_blob]*r
        values += [T*2560*4,10240*4]
        return takes(values)+8*MIB
    def host_after(chunk):
        # Two PLE host buffers+row IDs, token steps+upload, routing arrays and
        # their mapped grouping buffer. Stager has >=16 host blobs; cold file
        # staging can grow to 128, so this remains an estimate, not full RSS.
        T = chunk
        return (2*T*2560*4+2*T*16*4+2*T*4*4+3*T*10*4+
                (2*experts+1)*4+(3*T*10+2*(experts+experts//16+2))*4+16*max_blob)
    chunks = range(256,ceiling+1,256) if auto else []
    if not auto:
        chunks = []
        c = ceiling
        while c >= 256:
            chunks.append(c)
            c //= 2
        chunks.sort()
    fallback = 1024 if auto else max(1,requested)
    return {'auto': auto, 'lend_percent': lend_percent, 'ring_slots': ring,
            'borrow': '--no-prefill-borrow' not in options,
            'candidates': [[c,amount(c),host_after(c)] for c in chunks],
            'fallback': [fallback,amount(fallback,True),host_after(fallback)]}


def _host_buffers(tensors, options):
    """Known non-mapped host allocations, separate from embedding/PLE mappings."""
    result = []
    tensor = tensors.get('per_layer_token_embd.weight')
    if tensor and options.get('--ple-io','direct') == 'direct':
        rows = int(options.get('--ple-row-cache',1<<20))
        inflight = int(options.get('--ple-inflight',256))
        if rows < 0 or inflight <= 0:
            raise ValueError('PLE cache 参数无效')
        row_bytes = _tensor_bytes(tensor)//tensor['shape'][1]
        # RowCache::init: sets*8 keys/data and one next-way byte per set.
        result.append({'label':'PLE 行缓存','bytes':(rows//8)*(8*(row_bytes+4)+1)})
        result.append({'label':'PLE 读取缓冲','bytes':inflight*2*4096})
    return result


def allocation_preview(model, options, cfg, progress=None):
    """Augment existing resources; omissions only suppress affected components."""
    components, host, notes = [], [], []
    cwd = Path(cfg['cwd'])
    def path(value):
        return (cwd / str(value)).resolve()
    def add(label, size):
        components.append({'label': label, 'bytes': size})
    metadata, tensors = {}, {}
    try:
        metadata, tensors = _headers(path(options.get('--native', model.path)), progress)
    except (OSError, ValueError, KeyError, struct.error):
        notes.append('模型元数据不可读取，权重与上下文分项未计入。')
    if progress:
        progress({'stage': 'allocation', 'message': '计算权重、上下文与专家缓存占用'})
    pack = path(options['--pack']) if options.get('--pack') else None
    layer_split = '--layer-split' in options
    multi = layer_split or any(options.get(flag) not in (None, '0')
                for flag in ('--batch', '--batch-mtp', '--pipeline-windows'))
    remote_cache = any(options.get(flag) not in (None, '0') for flag in
                       ('--peer-device', '--expert-cache-device1', '--expert-cache-device2', '--expert-cache-device3'))
    alternate_weights = any(flag in options for flag in ('--native-dense-gguf', '--native-head-gguf', '--embd-gguf'))
    supported_model = metadata.get('general.architecture') == 'qwen4exp'
    if multi:
        notes.append('多卡分层或并发窗口的重复分配未计入。')
    if remote_cache:
        notes.append('辅助 GPU 专家缓存及其 RAM 补集未计入。')
    if pack and tensors and supported_model and not multi and not alternate_weights:
        try:
            for label, size in _weights(tensors, pack, options):
                add(label, size)
        except (OSError, ValueError, KeyError):
            notes.append('pack/native 权重布局不可读取，主模型权重未计入。')
    else:
        notes.append('主模型权重未计入：缺少 pack、架构未适配或使用了独立权重/多卡布局。')
    embedding = tensors.get('token_embd.weight')
    if embedding and '--embd-gguf' not in options:
        try:
            host.append({'label': '映射词嵌入', 'bytes': _tensor_bytes(embedding), 'reference': True})
        except (KeyError, ValueError):
            notes.append('词嵌入格式未计入。')
    g, context, kv = None, None, options.get('--kv', 'fp16')
    special_kv = '--kv-elastic' in options or options.get('--kv-resident') not in (None, '0')
    try:
        g = _geometry(metadata)
        context = _positive(options.get('--max-context', 4096))
        state = _session(g, context, kv)
        for label, size in state:
            if layer_split or (special_kv and label == 'KV 缓存'):
                continue
            add(label, size)
        if special_kv:
            notes.append('流式/弹性 KV 池未计入。')
    except (KeyError, ValueError, TypeError):
        notes.append('上下文或模型结构不可解析，KV/GDN 状态未计入。')
    if options.get('--draft') == 'dflash2':
        from .strata_drafting import pack_description
        try:
            draft = pack_description(path(options['--draft-model']))
            # Upstream 0.2.10 drafting.DRAFTER_MIB includes drafter state, not
            # just the pack bytes. Only the selected drafter is resident.
            add('DFlash2 推理预算', (715 if draft['selector_q4'] else 890) * MIB)
        except (OSError, ValueError, KeyError, TypeError):
            notes.append('DFlash2 pack 不可读取，草稿资源未计入。')
    if options.get('--mtp') and options.get('--draft') != 'dflash2' and layer_split:
        notes.append('分层 MTP 位于最后一张 GPU，未计入当前设备。')
    if options.get('--mtp') and options.get('--draft') != 'dflash2' and not layer_split:
        mtp = path(options['--mtp'])
        try:
            # mtp.cpp allocates all dense.bin, but only the canonical expert blobs.
            dense = (mtp / 'dense.bin').stat().st_size
            experts = MTP_EXPERT_BYTES
            if (mtp / 'experts.bin').stat().st_size < experts:
                raise ValueError('MTP experts 不完整')
            add('MTP 权重', dense + experts)
        except (OSError, ValueError, TypeError):
            notes.append('MTP 权重文件不可读取，未计入。')
        if g and context and not multi and not special_kv:
            try:
                draft_kv = 'int8' if kv == 'k8v4' else kv
                # Independent canonical draft geometry, NOT the target's MoE/attention.
                per_cell = {'fp16': 2048, 'int8': 1056, 'q4_0': 576}[draft_kv]
                draft = _align(context, 4)*per_cell
                add('MTP KV', draft)
                head = tensors['output.weight']
                vocab = head['shape'][1]
                spec = int(options.get('--spec', 0))
                mtp_max = int(options.get('--mtp-max-t', 0))
                if int(options.get('--suffix-draft', 3)) > 0 and spec >= 2 and not mtp_max:
                    mtp_max = spec
                    spec = min(spec+2, 8)
                chain = int(options.get('--lookup-chain', 0))
                if chain > 0 and spec >= 2:
                    spec = max(spec, min((mtp_max or spec)+chain, 8))
                if spec > 0:
                    add('MTP logits', spec*vocab*4)
                if '--native-head-gguf' in options:
                    notes.append('独立输出头对应的 MTP 草稿头未计入。')
                elif os.environ.get('STRATA_MTP_FULL_HEAD') != '1':
                    vocab_file = path(options['--mtp-draft-vocab']) if options.get('--mtp-draft-vocab') else mtp / 'draft_vocab.bin'
                    subset = vocab_file.stat().st_size
                    if subset < 4 or subset % 4:
                        raise ValueError('MTP vocab 损坏')
                    add('MTP 草稿头', subset//4*(_tensor_bytes(head)//vocab)+subset)
            except (OSError, ValueError, KeyError, ZeroDivisionError):
                notes.append('MTP 草稿头或 KV 布局未完全计入。')
        else:
            notes.append('MTP 流式 KV/并发缓冲未计入。')
        notes.append('MTP/验证计算缓冲未计入。')
        if '--mtp-q4' in options:
            notes.append('MTP Q4 附加副本未计入。')
    count = options.get('--expert-cache', '0')
    auto = count in ('auto', '-1') or (count == '0' and bool(options.get('--expert-profile')))
    expert_total = cache = complement = resident_plan = None
    nonmapped_host = []
    try:
        nonmapped_host = _host_buffers(tensors,options)
    except (KeyError,ValueError,TypeError):
        notes.append('PLE 主机缓冲布局未计入。')
    if pack and g and not multi and not remote_cache:
        try:
            expert_total, cache, complement, descriptor, formats = _experts(pack, g,
                path(options['--expert-profile']) if options.get('--expert-profile') else None,
                None if auto else max(0, int(count)), '--expert-cache-per-layer' in options)
            resident = any(flag in options for flag in ('--resident-experts','--resident-cpu-experts','--resident-budget-gib'))
            mode = 'resident' if resident else 'mmap' if '--mmap-experts' in options else 'arena'
            soft = '--resident-experts' in options
            requested = options.get('--resident-budget-gib')
            explicit_budget = int(float(requested)*2**30) if requested is not None else None
            if explicit_budget is not None and explicit_budget <= 0:
                raise ValueError('resident budget 必须为正值')
            headroom = 4 if soft or requested is not None else 8
            if soft or requested is not None:
                headroom = max(0,float(os.environ.get('STRATA_RESIDENT_HEADROOM_GIB',headroom)))
            exchange = 0
            if resident and int(options.get('--adapt-every',4)) > 0:
                exchange = max(0,min(int(options.get('--adapt-swaps',96)),96))*descriptor['max_blob_bytes']
                if int(options.get('--adapt-async',0)) > 0:
                    exchange *= 2
            device = cfg.get('gpu')
            resident_plan = {**descriptor,'mode':mode,
                'gpu_device':f'CUDA{device}' if type(device) is int else 'CUDA0',
                'headroom_bytes':int(headroom*2**30), 'budget_bytes':explicit_budget,
                'soft':soft,'host_bytes':sum(x['bytes'] for x in nonmapped_host),
                'exchange_bytes':exchange,'prefill':None}
            if context and not special_kv:
                try:
                    resident_plan['prefill'] = _prefill_plan(options,context,g['expert_count'],
                        descriptor['max_blob_bytes'],formats,mode)
                except (ValueError,KeyError,TypeError):
                    notes.append('非默认 prefill 布局的尾槽副本和主机缓冲未计入。')
            else:
                notes.append('流式 KV 的 prefill 主机缓冲与尾槽副本未计入。')
            if mode == 'resident' and not descriptor['gpu_rank_count']:
                notes.append('缺少有效专家排名，RAM 补集分配尚不可推演。')
        except (OSError, ValueError, TypeError, KeyError):
            notes.append('专家布局或排名不可读取，专家缓存预算未计入。')
    notes.append('计算临时缓冲和 CUDA graph 未计入；prefill 借用的专家槽位不重复累加。')
    if any(os.environ.get(key) == '1' for key in ('STRATA_Q8_PACKED', 'STRATA_Q6_PACKED', 'STRATA_HC_Q8')):
        notes.append('环境变量开启的附加 packed/Q8 权重副本未计入。')
    if cfg.get('vision') or '--vision' in options:
        notes.append('未计入视觉投影器和图像缓冲。')
    return {'components': components, 'host_components': host, 'notes': notes,
            'auto': auto, 'cache_bytes': cache, 'expert_total_bytes': expert_total,
            'complement_bytes': complement, 'nonmapped_host_components':nonmapped_host,
            'resident_plan':resident_plan}
