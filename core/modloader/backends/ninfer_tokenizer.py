"""The verified NInfer pin's explicit Qwen tokenizer pipeline contract.

This checks the fixed frontend's resource contract, not tokenization or Jinja
execution. The stdlib probe runs beside the selected files, so vocabulary and
merge payloads never travel through management state or IPC.
"""
from __future__ import annotations

QWEN_SPLIT_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# tokenizer.cpp:88-410, 798-847; frontend.cpp:185-198;
# frontend/resources.cpp:7-20. Like the existing WSL probes, this trusted source
# is passed to Python -c; it needs no package installed in the guest environment.
RESOURCE_INSPECTION_CODE = r'''
def inspect_tokenizer_resources(tokenizer, config, generation, embedding_rows=None):
    errors=[]
    def issue(message):
        if message not in errors: errors.append(message)
    def token_id(value, field):
        # The fixed C++ reader's kMaxTokenId, not an application model-size cap.
        if type(value) is not int or not 0<=value<=1000000:
            issue(field+' 必须为 0..1000000 的整数 Token ID'); return None
        return value
    model=tokenizer.get('model')
    if not isinstance(model,dict): issue('tokenizer.json 缺少 model 对象'); model={}
    if model.get('type')!='BPE': issue('tokenizer.json model.type 必须为 BPE')
    vocab=model.get('vocab')
    if not isinstance(vocab,dict) or not vocab:
        issue('tokenizer.json model.vocab 必须为非空对象'); vocab={}
    ids={}; contents={}
    for content,raw_id in vocab.items():
        identifier=token_id(raw_id,'model.vocab')
        if identifier is None: continue
        if identifier in ids: issue('model.vocab 包含重复 Token ID')
        ids[identifier]=content; contents[content]=identifier
    base_ids=set(ids); base_contents=set(contents)
    records={}; added_contents={}
    flags=('single_word','lstrip','rstrip','normalized','special')
    def added_record(item, identifier, field):
        if not isinstance(item,dict): issue(field+' 记录必须为对象'); return
        content=item.get('content')
        if not isinstance(content,str) or not content:
            issue(field+'.content 必须为非空字符串'); return
        if any(type(item.get(key)) is not bool for key in flags):
            issue(field+' 每条记录必须包含布尔字段 '+','.join(flags)); return
        if any(item[key] for key in flags[:-1]):
            issue(field+' 的 single_word/lstrip/rstrip/normalized 必须为 false')
        if identifier is None: return
        record=(content,*(item[key] for key in flags))
        if identifier in base_ids or content in base_contents:
            issue(field+' 与基础词表的 ID/内容重叠')
        if identifier in records and records[identifier]!=record:
            issue('tokenizer.json 与 tokenizer_config.json 的 added token ID/内容/标志冲突')
        if content in added_contents and added_contents[content]!=identifier:
            issue(field+' 包含重复内容映射')
        records[identifier]=record; added_contents[content]=identifier; ids[identifier]=content
    added=tokenizer.get('added_tokens')
    if not isinstance(added,list): issue('tokenizer.json 缺少 added_tokens 数组'); added=[]
    seen=set()
    for item in added:
        identifier=token_id(item.get('id') if isinstance(item,dict) else None,'added_tokens[].id')
        if identifier is not None and identifier in seen: issue('added_tokens 包含重复 ID')
        seen.add(identifier); added_record(item,identifier,'added_tokens')
    decoder=config.get('added_tokens_decoder')
    if not isinstance(decoder,dict):
        issue('tokenizer_config.json 缺少 added_tokens_decoder 对象'); decoder={}
    for key,item in decoder.items():
        try:
            identifier=int(key)
            if str(identifier)!=key: raise ValueError()
        except (TypeError,ValueError):
            issue('added_tokens_decoder 的 key 必须为规范非负整数 ID'); continue
        identifier=token_id(identifier,'added_tokens_decoder key')
        added_record(item,identifier,'added_tokens_decoder')
    for key in ('add_bos_token','add_prefix_space'):
        if config.get(key) is not False: issue('tokenizer_config.json.'+key+' 必须显式为 false')
    if config.get('pad_token')!='<|endoftext|>':
        issue('tokenizer_config.json.pad_token 必须为 <|endoftext|> 字符串')
    merges=model.get('merges')
    if not isinstance(merges,list): issue('tokenizer.json 缺少 model.merges 数组'); merges=[]
    pairs=set()
    for value in merges:
        pair=value.split(' ') if isinstance(value,str) else value
        if not isinstance(pair,list) or len(pair)!=2 or any(not isinstance(s,str) for s in pair):
            issue('model.merges 必须为有序字符串二元组'); continue
        left,right=pair
        if any(s not in vocab for s in (left,right,left+right)):
            issue('model.merges 引用了基础词表以外的符号')
        key=(left,right)
        if key in pairs: issue('model.merges 包含重复合并对')
        pairs.add(key)
    if not ids or min(ids)!=0 or max(ids)+1!=len(ids):
        issue('分词器必须提供连续的 public Token ID 域')
    if type(embedding_rows) is int and ids and max(ids)>=embedding_rows:
        issue('分词器 public Token ID 超出模型 embedding 行数')
    eos=generation.get('eos_token_id')
    if type(eos) is int: eos=[eos]
    if not isinstance(eos,list) or not eos:
        issue('generation_config.json.eos_token_id 必须为整数或非空整数数组'); eos=[]
    for value in eos:
        identifier=token_id(value,'generation_config.json.eos_token_id')
        if identifier is not None and identifier not in ids:
            issue('generation_config.json.eos_token_id 不在分词器 public Token ID 域中')
    pipeline={k:tokenizer[k] for k in ('normalizer','pre_tokenizer','decoder','post_processor') if k in tokenizer}
    pipeline['model']={k:model[k] for k in ('dropout','continuing_subword_prefix','end_of_word_suffix','ignore_merges') if k in model}
    return dict(pipeline=pipeline,errors=errors,public_token_count=len(ids))
'''


def validate_resources(contract, label):
    issues = list(contract['errors'])
    pipeline_error = None
    try:
        validate_pipeline(contract['pipeline'], label)
    except ValueError as error:
        pipeline_error = str(error)
    if issues or pipeline_error:
        prefix = pipeline_error or f'NInfer 64d10fbf9f8e 的 {label} 资源不兼容，请选择同一底模的配套分词器目录'
        raise ValueError('；'.join([prefix, *issues]))


def validate_pipeline(description, label):
    """Caller must establish the verified pin before applying this contract."""
    def typed(value, name):
        return isinstance(value, dict) and value.get('type') == name

    def require(valid, field):
        if not valid:
            raise ValueError(f'NInfer 64d10fbf9f8e 不支持 {label} 的 {field}；'
                             '请选择同一底模配套、符合此固定版本的官方分词器目录。'
                             '不会自动改写或删除分词流程；已有 artifact 需使用正确资源重新准备。')

    if 'normalizer' in description:
        require(typed(description['normalizer'], 'NFC'), 'normalizer')
    if 'pre_tokenizer' in description:
        pre = description['pre_tokenizer']
        require(typed(pre, 'Sequence') and isinstance(pre.get('pretokenizers'), list)
                and len(pre['pretokenizers']) == 2, 'pre_tokenizer')
        split, byte_level = pre['pretokenizers']
        require(typed(split, 'Split') and isinstance(split.get('pattern'), dict)
                and split['pattern'].get('Regex') == QWEN_SPLIT_PATTERN
                and split.get('behavior') == 'Isolated' and split.get('invert', False) is False,
                'pre_tokenizer.Split')
        require(typed(byte_level, 'ByteLevel') and byte_level.get('add_prefix_space', True) is False
                and byte_level.get('use_regex', True) is False, 'pre_tokenizer.ByteLevel')
    if 'decoder' in description:
        require(typed(description['decoder'], 'ByteLevel'), 'decoder')
    if description.get('post_processor') is not None:
        require(typed(description['post_processor'], 'ByteLevel'), 'post_processor')
    model = description.get('model', {})
    require(isinstance(model, dict), 'model')
    if model.get('dropout') is not None:
        require(type(model['dropout']) in (int, float) and model['dropout'] == 0, 'model.dropout')
    for field in ('continuing_subword_prefix', 'end_of_word_suffix'):
        require(model.get(field) in (None, ''), 'model.' + field)
    require(model.get('ignore_merges', False) is False, 'model.ignore_merges')
