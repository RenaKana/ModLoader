"""Management chat history, separate from the stateless inference gateway."""
from datetime import datetime, timezone
import json
import math
import uuid

from .chat_reasoning import validate_reasoning_effort
from .request_history import PREFILL_FIELDS, STATES, local_date, request_history_stats, _reported
from .request_metrics import USAGE_FIELDS, TIMING_FIELDS, _numbers, clean_measurements, clean_counters
from .web_search import validate_search_result


SUMMARY_KIND = 'conversations'
DETAIL_KIND = 'conversation_details'
STATE_KIND = 'conversation_state'
MESSAGE_PREFIX = 'conversation_messages:'
MESSAGE_STATES = ('completed', 'generating', 'stopped', 'failed', 'interrupted')
SEARCH_SETTINGS = {'webSearch': 'web_search', 'webSearchMode': 'web_search_mode', 'webSearchRounds': 'web_search_rounds'}
DETAIL_SETTINGS = {**SEARCH_SETTINGS, 'basicTools': 'basic_tools', 'autoCompact': 'auto_compact', 'toolDirectory': 'tool_directory', 'toolConcurrency': 'tool_concurrency'}


class ConversationNotFound(LookupError):
    pass


def _text(value, field, nonblank=False):
    if not isinstance(value, str) or (nonblank and not value.strip()):
        raise ValueError(f'{field} 必须为' + ('非空字符串' if nonblank else '字符串'))
    return value


def _identifier(value):
    _text(value, 'id', nonblank=True)
    try:
        uuid.UUID(value)
    except ValueError:
        raise ValueError('对话 id 必须为 UUID') from None
    return value


def _object(value, field, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - {*required, *optional}:
        raise ValueError(f'{field} 字段不完整或包含不支持的字段')


def _number(value, field):
    if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f'{field} 必须为有限数值')


def _images(value):
    if not isinstance(value, list):
        raise ValueError('images 必须为图片数组')
    for image in value:
        _object(image, '图片', ('id', 'name', 'size', 'url'))
        _text(image['id'], '图片 id', nonblank=True)
        _text(image['name'], '图片名称')
        _text(image['url'], '图片地址', nonblank=True)
        if type(image['size']) is not int or image['size'] < 0:
            raise ValueError('图片 size 必须为非负整数')


def _documents(value):
    if not isinstance(value, list):
        raise ValueError('documents 必须为文档数组')
    for document in value:
        _object(document, '文档', ('id', 'name', 'size', 'text'))
        _text(document['id'], '文档 id', nonblank=True)
        _text(document['name'], '文档名称', nonblank=True)
        _text(document['text'], '文档正文')
        if type(document['size']) is not int or document['size'] < 0:
            raise ValueError('文档 size 必须为非负整数')


def _selection(value):
    if value is not None:
        _object(value, 'selection', ('profile_id', 'model_id', 'alias'), ('instance_id',))
        for field in ('profile_id', 'model_id', 'alias'):
            _text(value[field], field, nonblank=True)
        if 'instance_id' in value:
            _text(value['instance_id'], 'instance_id', nonblank=True)


def _draft(value):
    _object(value, 'draft', ('text', 'images'), ('documents',))
    _text(value['text'], '草稿正文')
    _images(value['images'])
    if 'documents' in value:
        _documents(value['documents'])


def _settings(value):
    _object(value, 'settings', ('system', 'stream', 'toolsDraft'), ('temperature', 'maxTokens', 'presetId', 'reasoningEffort', *DETAIL_SETTINGS))
    _text(value['system'], 'system')
    _text(value['toolsDraft'], 'toolsDraft')
    if 'presetId' in value:
        _identifier(value['presetId'])
    if 'reasoningEffort' in value:
        validate_reasoning_effort(value['reasoningEffort'])
    if 'webSearch' in value and type(value['webSearch']) is not bool:
        raise ValueError('webSearch 必须为布尔值')
    for field in ('basicTools', 'autoCompact'):
        if field in value and type(value[field]) is not bool:
            raise ValueError(f'{field} 必须为布尔值')
    if 'toolDirectory' in value:
        _text(value['toolDirectory'], 'toolDirectory')
    if 'webSearchMode' in value and value['webSearchMode'] not in ('agent', 'retrieve'):
        raise ValueError('webSearchMode 必须为 agent 或 retrieve')
    if 'webSearchRounds' in value and (type(value['webSearchRounds']) is not int or value['webSearchRounds'] < 1):
        raise ValueError('webSearchRounds 必须为正整数')
    if 'toolConcurrency' in value and (type(value['toolConcurrency']) is not int or value['toolConcurrency'] < 1):
        raise ValueError('toolConcurrency 必须为正整数')
    if type(value['stream']) is not bool:
        raise ValueError('stream 必须为布尔值')
    for field in ('temperature', 'maxTokens'):
        if field in value:
            _number(value[field], field)


def _set_settings(detail, value):
    # Keep the stored settings compatible with older management clients. Their
    # strict settings validator does not know about preset/reasoning/search metadata.
    detail['settings'] = {field: item for field, item in value.items() if field not in ('presetId', 'reasoningEffort', *DETAIL_SETTINGS)}
    for field, stored_field in DETAIL_SETTINGS.items():
        if field in value:
            detail[stored_field] = value[field]
    if value.get('reasoningEffort') is not None:
        detail['reasoning_effort'] = value['reasoningEffort']
    else:
        detail.pop('reasoning_effort', None)
    if 'presetId' in value:
        detail['preset_id'] = value['presetId']
    else:
        detail.pop('preset_id', None)


def _tool(value):
    if not isinstance(value, dict) or not isinstance(value.get('function'), dict):
        raise ValueError('工具调用必须包含 function 对象')
    for field in ('name', 'arguments'):
        _text(value['function'].get(field), '工具 function.' + field)
    for field in ('id', 'type'):
        if field in value:
            _text(value[field], '工具 ' + field)


def _nonnegative(value, field, *, integer=False):
    _number(value, field)
    if value < 0 or (integer and type(value) is not int):
        raise ValueError(f'{field} 必须为非负' + ('整数' if integer else '数值'))


def _resources(value):
    if not isinstance(value, list):
        raise ValueError('resources 必须为附件引用数组')
    for resource in value:
        _object(resource, '附件引用', ('id', 'name', 'path', 'directory', 'size', 'mime', 'kind', 'can_open'))
        for field in ('id', 'name', 'path', 'directory', 'mime'):
            _text(resource[field], '附件 ' + field, nonblank=True)
        _nonnegative(resource['size'], '附件 size', integer=True)
        if resource['kind'] not in ('image', 'file') or type(resource['can_open']) is not bool:
            raise ValueError('附件类型或打开状态无效')


REQUEST_STATES = ('running', 'completed', 'cancelled', 'failed', 'incomplete')


def _request_records(value):
    if not isinstance(value, list):
        raise ValueError('requests 必须为模型请求数组')
    identifiers = set()
    for request in value:
        if not isinstance(request, dict):
            raise ValueError('模型请求必须为对象')
        identifier = _text(request.get('id'), '模型请求 id', nonblank=True)
        if identifier in identifiers:
            raise ValueError('模型请求 id 不可重复')
        identifiers.add(identifier)
        _nonnegative(request.get('index'), '模型请求 index', integer=True)
        if request.get('kind') not in ('generation', 'planning', 'compaction') or request.get('status') not in REQUEST_STATES:
            raise ValueError('模型请求类型或状态无效')
        _nonnegative(request.get('startedAt'), '模型请求 startedAt')
        for field in ('endedAt', 'elapsed', 'firstOutput'):
            if field in request:
                _nonnegative(request[field], '模型请求 ' + field)
        if 'requestKey' in request:
            _text(request['requestKey'], 'requestKey', nonblank=True)
        if 'firstOutputBasis' in request and request['firstOutputBasis'] not in ('gateway', 'client'):
            raise ValueError('首输出计时来源无效')
        if not isinstance(request.get('labels'), list) or not all(isinstance(label, str) for label in request['labels']):
            raise ValueError('模型请求 labels 必须为字符串数组')
        for field in ('usage', 'timings', 'prefill'):
            if field in request and not isinstance(request[field], dict):
                raise ValueError('模型请求 ' + field + ' 必须为对象')


def _turn_totals(value):
    if not isinstance(value, dict) or value.get('status') not in REQUEST_STATES:
        raise ValueError('turn 必须包含有效的整轮状态')
    for field in ('startedAt', 'elapsed'):
        _nonnegative(value.get(field), 'turn ' + field)
    if 'endedAt' in value:
        _nonnegative(value['endedAt'], 'turn endedAt')
    _nonnegative(value.get('requests'), 'turn requests', integer=True)
    for field in ('input', 'output', 'cached', 'reasoning', 'decode', 'prefill'):
        total = value.get(field)
        if not isinstance(total, dict) or 'value' not in total:
            raise ValueError('turn ' + field + ' 必须为统计对象')
        if total['value'] is not None:
            _nonnegative(total['value'], 'turn ' + field + '.value')
        _nonnegative(total.get('reported'), 'turn ' + field + '.reported', integer=True)
        if field in ('decode', 'prefill'):
            for key in ('tokens', 'seconds'):
                _nonnegative(total.get(key), 'turn ' + field + '.' + key)
    if value.get('e2e') is not None:
        _nonnegative(value['e2e'], 'turn e2e')


def _messages(value):
    if not isinstance(value, list):
        raise ValueError('messages 必须为消息数组')
    identifiers = set()
    for message in value:
        if not isinstance(message, dict) or not {'id', 'role', 'text', 'images', 'createdAt', 'status'} <= message.keys():
            raise ValueError('消息字段不完整')
        identifier = _text(message['id'], '消息 id', nonblank=True)
        if identifier in identifiers:
            raise ValueError('同一次更新的消息 id 不可重复')
        identifiers.add(identifier)
        if message['role'] not in ('user', 'assistant'):
            raise ValueError('消息 role 必须为 user 或 assistant')
        if message['status'] not in MESSAGE_STATES:
            raise ValueError('消息 status 无效')
        _text(message['text'], '消息正文')
        _images(message['images'])
        if 'documents' in message:
            _documents(message['documents'])
        _number(message['createdAt'], 'createdAt')
        if 'contextRevision' in message and (type(message['contextRevision']) is not int or message['contextRevision'] < 0):
            raise ValueError('contextRevision 必须为非负整数')
        for field in ('reasoning', 'reasoningHistory', 'model', 'profileId', 'modelId', 'error', 'finishReason'):
            if field in message:
                _text(message[field], field, nonblank=field in ('profileId', 'modelId'))
        for field in ('thinking', 'stopped'):
            if field in message and type(message[field]) is not bool:
                raise ValueError(f'{field} 必须为布尔值')
        for field in ('elapsed', 'firstOutput', 'responseAt', 'contextCapacity'):
            if field in message:
                _number(message[field], field)
        for field in ('usage', 'timings'):
            if field in message and not isinstance(message[field], dict):
                raise ValueError(f'{field} 必须为对象')
        if 'webSearch' in message:
            validate_search_result(message['webSearch'])
        if 'resources' in message:
            _resources(message['resources'])
        if 'requests' in message:
            _request_records(message['requests'])
        if 'turn' in message:
            _turn_totals(message['turn'])
        if 'tools' in message:
            if not isinstance(message['tools'], list):
                raise ValueError('tools 必须为工具调用对象数组')
            for tool in message['tools']:
                _tool(tool)
        if 'activity' in message:
            if not isinstance(message['activity'], list):
                raise ValueError('activity 必须为事件数组')
            event_ids = set()
            for event in message['activity']:
                if not isinstance(event, dict):
                    raise ValueError('activity 事件必须为对象')
                event_id = _text(event.get('id'), 'activity id', nonblank=True)
                if event_id in event_ids:
                    raise ValueError('activity id 不可重复')
                event_ids.add(event_id)
                if event.get('kind') in ('reasoning', 'text'):
                    _object(event, 'activity 文本', ('id', 'kind', 'text'))
                    _text(event['text'], 'activity text')
                elif event.get('kind') == 'tool':
                    _object(event, 'activity 工具', ('id', 'kind', 'call', 'status'), ('result', 'requestId', 'startedAt', 'endedAt', 'elapsed'))
                    _tool(event['call'])
                    if event['status'] not in ('pending', 'running', 'completed', 'failed', 'cancelled'):
                        raise ValueError('activity 工具状态无效')
                    if 'result' in event:
                        _text(event['result'], 'activity result')
                    if 'requestId' in event:
                        _text(event['requestId'], 'activity requestId', nonblank=True)
                    for field in ('startedAt', 'endedAt', 'elapsed'):
                        if field in event:
                            _nonnegative(event[field], 'activity ' + field)
                else:
                    raise ValueError('activity 事件种类无效')
        if 'toolResults' in message:
            if not isinstance(message['toolResults'], list):
                raise ValueError('toolResults 必须为工具结果数组')
            for result in message['toolResults']:
                if not isinstance(result, dict):
                    raise ValueError('工具结果必须为对象')
                _tool(result.get('call'))
                _text(result.get('content'), '工具结果 content')


def _compaction(value):
    if value is None:
        return
    _object(value, 'compaction', ('summary', 'throughMessageId', 'messageRevisions', 'createdAt'))
    _text(value['summary'], '摘要', nonblank=True)
    _text(value['throughMessageId'], '压缩消息边界', nonblank=True)
    _number(value['createdAt'], '摘要 createdAt')
    snapshots = value['messageRevisions']
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError('messageRevisions 必须为非空消息版本数组')
    identifiers = set()
    for snapshot in snapshots:
        _object(snapshot, '消息版本', ('id', 'revision'))
        identifier = _text(snapshot['id'], '消息版本 id', nonblank=True)
        if identifier in identifiers:
            raise ValueError('压缩消息 id 不可重复')
        identifiers.add(identifier)
        if type(snapshot['revision']) is not int or snapshot['revision'] < 0:
            raise ValueError('消息 revision 必须为非负整数')
    if snapshots[-1]['id'] != value['throughMessageId']:
        raise ValueError('压缩消息边界与版本数组不一致')


def _valid_compaction(value, messages):
    snapshots = value['messageRevisions']
    return len(messages) >= len(snapshots) and all(snapshot == {'id': message['id'], 'revision': message.get('contextRevision', 0)}
                                                  for snapshot, message in zip(snapshots, messages))


def _context_content(message):
    if message.get('error') or message.get('tools') or not (message['text'] or message['images'] or message.get('documents') or (message['role'] == 'assistant' and message.get('reasoning'))):
        return None
    return (message['role'], message['text'], [image['url'] for image in message['images']],
            (message.get('reasoning') or '') if message['role'] == 'assistant' else None,
            [(document['name'], document['text']) for document in message.get('documents', [])])


def _validate(body, create=False):
    fields = ('title', 'selection', 'draft', 'settings')
    if create:
        _object(body, '对话', ('id', *fields))
        _identifier(body['id'])
    else:
        _object(body, '对话更新', (), (*fields, 'messages', 'clear_messages', 'activate', 'pinned', 'move_before', 'compaction'))
    if 'title' in body:
        _text(body['title'], '对话标题', nonblank=True)
    for field, validator in (('selection', _selection), ('draft', _draft), ('settings', _settings), ('messages', _messages), ('compaction', _compaction)):
        if field in body:
            validator(body[field])
    for field in ('clear_messages', 'activate', 'pinned'):
        if field in body and type(body[field]) is not bool:
            raise ValueError(f'{field} 必须为布尔值')
    if 'move_before' in body and body['move_before'] is not None:
        _identifier(body['move_before'])
    # Keep arbitrary generation payload fields intact, while requiring actual JSON.
    json.dumps(body, ensure_ascii=False, allow_nan=False)


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _history_request(request):
    """Copy only request identity and metrics, never arbitrary saved fields."""
    clean = {key: request[key] for key in ('id', 'index', 'kind', 'status', 'startedAt', 'labels')}
    if isinstance(request.get('requestKey'), str):
        clean['requestKey'] = request['requestKey']
    clean.update(_numbers(request, ('endedAt', 'elapsed', 'firstOutput')))
    if request.get('firstOutputBasis') in ('gateway', 'client'):
        clean['firstOutputBasis'] = request['firstOutputBasis']
    usage = request.get('usage') or {}
    clean['usage'] = _numbers(usage, USAGE_FIELDS)
    for key, field in (('prompt_tokens_details', 'cached_tokens'), ('completion_tokens_details', 'reasoning_tokens')):
        details = _numbers(usage.get(key), (field,))
        if details:
            clean['usage'][key] = details
    clean['timings'] = _numbers(request.get('timings'), TIMING_FIELDS)
    if isinstance(request.get('measurements'), dict):
        clean['measurements'] = clean_measurements(request['measurements'])
    if isinstance(request.get('counters'), dict):
        clean['counters'] = clean_counters(request['counters'])
    prefill = request.get('prefill') or {}
    clean['prefill'] = _numbers(prefill, PREFILL_FIELDS)
    if prefill.get('prefill_average_basis') in ('backend_prefill', 'sampled_prefill'):
        clean['prefill']['prefill_average_basis'] = prefill['prefill_average_basis']
    live = request.get('live')
    if isinstance(live, dict):
        clean['live'] = _numbers(live, ('prompt_tokens', 'generated_tokens', 'average_tokens_per_second',
                                      'average_prefill_tokens_per_second', 'prefill_processed_tokens', 'prefill_cached_tokens'))
        clean['live']['source'] = live.get('source') if isinstance(live.get('source'), str) else ''
        for key, allowed in (('average_basis', ('backend_decode', 'sampled_decode')),
                             ('prefill_average_basis', ('backend_prefill', 'sampled_prefill'))):
            if live.get(key) in allowed:
                clean['live'][key] = live[key]
    started = clean['startedAt'] / 1000
    ended = clean.get('endedAt')
    duration = clean.get('elapsed', max(0, ended / 1000 - started) if ended is not None else 0)
    record = {'status': clean['status'], 'started_at': started, 'duration_s': duration, 'date': local_date(started),
              'first_output_s': clean.get('firstOutput'), 'first_output_basis': clean.get('firstOutputBasis', 'unknown'),
              **{key: clean[key] for key in ('usage', 'timings', 'prefill')}, 'live': clean.get('live', {})}
    if 'measurements' in clean:
        record['measurements'] = clean['measurements']
    if 'counters' in clean:
        record['counters'] = clean['counters']
    if ended is not None:
        record['ended_at'] = ended / 1000
    elif clean['status'] != 'running':
        record['record_incomplete'] = True
    return clean, record


class Conversations:
    def __init__(self, store):
        self.store = store
        # Only checkpoint metadata is cached. Ordinary tail streaming must not
        # repeatedly load all prior messages or their image payloads.
        self._compactions = {identifier: json.loads(body) for identifier, body in self.store.connection.execute(
            "SELECT id,json_extract(body,'$.compaction') FROM records WHERE kind=? AND json_type(body,'$.compaction')='object'",
            (DETAIL_KIND,))}

    def _put(self, kind, record):
        return self.store.connection.execute(
            'INSERT INTO records(kind,id,body) VALUES(?,?,?) '
            'ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body WHERE records.body<>excluded.body',
            (kind, record['id'], json.dumps(record, ensure_ascii=False, allow_nan=False))).rowcount > 0

    def _message_times(self, identifier):
        return self.store.connection.execute(
            "SELECT MAX(MAX(json_extract(body,'$.createdAt'), COALESCE(json_extract(body,'$.responseAt'), json_extract(body,'$.createdAt')))), MAX(json_extract(body,'$.createdAt')) FROM records WHERE kind=?",
            (MESSAGE_PREFIX + identifier,)).fetchone()

    def _with_message_time(self, summary):
        if 'last_message_at' not in summary or 'last_sent_at' not in summary:
            message_at, sent_at = self._message_times(summary['id'])
            summary = {**summary, 'last_message_at': message_at, 'last_sent_at': sent_at}
        return {**summary, 'pinned': summary.get('pinned', False), 'sort_order': summary.get('sort_order', summary['last_sent_at'] or 0)}

    def _require(self, identifier):
        summary = self.store.get(SUMMARY_KIND, identifier)
        if summary is None:
            raise ConversationNotFound('对话不存在')
        return self._with_message_time(summary)

    def list(self):
        with self.store.lock:
            # These records contain only summary metadata, never drafts or images.
            items = [json.loads(row[0]) for row in self.store.connection.execute(
                "SELECT body FROM records WHERE kind=? ORDER BY json_extract(body,'$.updated_at') DESC, rowid DESC",
                (SUMMARY_KIND,))]
            items = sorted((self._with_message_time(item) for item in items),
                           key=lambda item: (item['pinned'], item['sort_order'], item['created_at']), reverse=True)
            state = self.store.get(STATE_KIND, 'main')
            return {'items': items, 'active_id': state['active_id'] if state else None}

    def create(self, body):
        _validate(body, create=True)
        identifier = body['id']
        with self.store.lock, self.store.connection:
            existing = self.store.get(SUMMARY_KIND, identifier)
            if existing is not None:
                return self._with_message_time(existing)
            now = _timestamp()
            summary = {'id': identifier, 'title': body['title'], 'created_at': now, 'updated_at': now, 'last_message_at': None, 'last_sent_at': None, 'pinned': False, 'sort_order': 0, 'message_count': 0}
            self._put(SUMMARY_KIND, summary)
            detail = {field: body[field] for field in ('id', 'selection', 'draft')}
            _set_settings(detail, body['settings'])
            self._put(DETAIL_KIND, detail)
            return summary

    def get(self, identifier):
        _identifier(identifier)
        with self.store.lock:
            summary = self._require(identifier)
            detail = self.store.get(DETAIL_KIND, identifier)
            preset_id = detail.pop('preset_id', None)
            if preset_id is not None:
                detail['settings'] = {**detail['settings'], 'presetId': preset_id}
            effort = detail.pop('reasoning_effort', None)
            if effort is not None:
                detail['settings'] = {**detail['settings'], 'reasoningEffort': effort}
            for field, stored_field in DETAIL_SETTINGS.items():
                if stored_field in detail:
                    detail['settings'] = {**detail['settings'], field: detail.pop(stored_field)}
            return {**summary, **detail, 'messages': self.store.all(MESSAGE_PREFIX + identifier)}

    def _top_order(self, pinned):
        return max((item['sort_order'] for item in self.list()['items'] if item['pinned'] == pinned), default=0) + 1

    def patch(self, identifier, body):
        _identifier(identifier)
        _validate(body)
        with self.store.lock:
            previous_compaction = self._compactions.get(identifier)
            compaction = previous_compaction
            message_kind = MESSAGE_PREFIX + identifier
            # Check only covered edits; new tail tokens never load the old transcript.
            if compaction:
                covered = {item['id']: item['revision'] for item in compaction['messageRevisions']}
                if body.get('clear_messages') and not _valid_compaction(compaction, body.get('messages', [])):
                    compaction = None
                for message in body.get('messages', []):
                    if compaction and message['id'] in covered:
                        old = self.store.get(message_kind, message['id'])
                        if not old or covered[message['id']] != message.get('contextRevision', 0) or _context_content(old) != _context_content(message):
                            compaction = None
            with self.store.connection:
                summary = self._require(identifier)
                changed = False
                if 'pinned' in body and body['pinned'] != summary['pinned']:
                    summary['pinned'] = body['pinned']
                    summary['sort_order'] = self._top_order(summary['pinned'])
                    changed = True
                if 'move_before' in body and body['move_before'] is not None:
                    target = self._require(body['move_before'])
                    if target['id'] == identifier or target['pinned'] != summary['pinned']:
                        raise ValueError('只能在同一分组内调整会话顺序')
                if 'title' in body and body['title'] != summary['title']:
                    summary['title'] = body['title']
                    changed = True
                messages_changed = False
                if body.get('clear_messages'):
                    messages_changed = self.store.connection.execute('DELETE FROM records WHERE kind=?', (message_kind,)).rowcount > 0
                    from .local_resources import forget_conversation_resources
                    keep = [resource['id'] for message in body.get('messages', []) for resource in message.get('resources', [])]
                    forget_conversation_resources(self.store, identifier, keep_ids=keep)
                for message in body.get('messages', []):
                    messages_changed = self._put(message_kind, message) or messages_changed
                if 'compaction' in body:
                    compaction = body['compaction']
                    if compaction is not None and not _valid_compaction(compaction, self.store.all(message_kind)):
                        raise ValueError('摘要对应的消息已改变，请重新压缩')
                detail_fields = ('selection', 'draft', 'settings')
                if any(field in body for field in detail_fields) or 'compaction' in body or compaction != previous_compaction:
                    detail = self.store.get(DETAIL_KIND, identifier)
                    detail.update({field: body[field] for field in detail_fields if field in body and field != 'settings'})
                    if 'settings' in body:
                        _set_settings(detail, body['settings'])
                    if compaction is None:
                        detail.pop('compaction', None)
                    else:
                        detail['compaction'] = compaction
                    changed = self._put(DETAIL_KIND, detail) or changed
                if messages_changed:
                    previous_sent = summary['last_sent_at']
                    summary['last_message_at'], summary['last_sent_at'] = self._message_times(identifier)
                    if not summary['pinned'] and summary['last_sent_at'] is not None and (previous_sent is None or summary['last_sent_at'] > previous_sent):
                        summary['sort_order'] = self._top_order(False)
                    summary['message_count'] = self.store.connection.execute('SELECT COUNT(*) FROM records WHERE kind=?', (message_kind,)).fetchone()[0]
                if changed or messages_changed:
                    summary['updated_at'] = _timestamp()
                    self._put(SUMMARY_KIND, summary)
                if 'move_before' in body:
                    group = [item for item in self.list()['items'] if item['pinned'] == summary['pinned'] and item['id'] != identifier]
                    index = next((index for index, item in enumerate(group) if item['id'] == body['move_before']), len(group))
                    group.insert(index, summary)
                    for index, item in enumerate(group):
                        item['sort_order'] = len(group) - index
                        self._put(SUMMARY_KIND, item)
                if body.get('activate'):
                    self._put(STATE_KIND, {'id': 'main', 'active_id': identifier})
            if compaction is None:
                self._compactions.pop(identifier, None)
            else:
                self._compactions[identifier] = compaction
            return summary

    def delete(self, identifier):
        _identifier(identifier)
        with self.store.lock:
            with self.store.connection:
                self._require(identifier)
                from .local_resources import forget_conversation_resources
                forget_conversation_resources(self.store, identifier)
                self.store.connection.execute('DELETE FROM records WHERE (kind IN (?,?) AND id=?) OR kind=?',
                                              (SUMMARY_KIND, DETAIL_KIND, identifier, MESSAGE_PREFIX + identifier))
                state = self.store.get(STATE_KIND, 'main')
                if state and state['active_id'] == identifier:
                    self._put(STATE_KIND, {'id': 'main', 'active_id': None})
            self._compactions.pop(identifier, None)

    def request_history(self, date_from=None, date_to=None, instance_id=None, conversation_id=None, message_id=None):
        """Read retained per-request metrics without loading message content or attachments."""
        where, args = ["json_extract(body,'$.role')='assistant'"], []
        if conversation_id is not None:
            _identifier(conversation_id)
            where.append('kind=?')
            args.append(MESSAGE_PREFIX + conversation_id)
        else:
            where.append('kind GLOB ?')
            args.append(MESSAGE_PREFIX + '*')
        if instance_id is not None:
            where.append("json_extract(body,'$.instanceId')=?")
            args.append(instance_id)
        if message_id is not None:
            where.append('id=?')
            args.append(message_id)
            date_from = date_to = None
        with self.store.lock:
            rows = self.store.connection.execute(
                "SELECT kind,id,json_object('instance_id',json_extract(body,'$.instanceId'),"
                "'alias',json_extract(body,'$.model'),'started_at',json_extract(body,'$.createdAt'),"
                "'context_capacity',json_extract(body,'$.contextCapacity'),'status',json_extract(body,'$.turn.status'),"
                "'message_status',json_extract(body,'$.status'),"
                "'backend',json_extract(body,'$.backend'),'requests',json_extract(body,'$.requests'),"
                "'usage',json_extract(body,'$.usage'),'timings',json_extract(body,'$.timings')) "
                'FROM records WHERE ' + ' AND '.join(where) + ' ORDER BY rowid', args).fetchall()
        groups, all_records, days, legacy = [], [], {}, []
        coverage = {'legacy_message_count': 0, 'legacy_metrics_message_count': 0}
        for kind, identifier, body in rows:
            metadata = json.loads(body)
            requests = metadata['requests']
            if not isinstance(requests, list):
                day = local_date(metadata['started_at'] / 1000)
                if (date_from is None or day >= date_from) and (date_to is None or day <= date_to):
                    has_metrics = bool(_numbers(metadata['usage'], USAGE_FIELDS) or _numbers(metadata['timings'], TIMING_FIELDS))
                    coverage['legacy_message_count'] += 1
                    coverage['legacy_metrics_message_count'] += has_metrics
                    legacy.append({'date': day, 'instance_id': metadata['instance_id'],
                                   'conversation_id': kind[len(MESSAGE_PREFIX):], 'message_id': identifier,
                                   'has_metrics': has_metrics, 'active': metadata['message_status'] == 'generating'})
                continue
            pairs = [_history_request(request) for request in requests]
            matched_records = [record for _, record in pairs
                               if (date_from is None or record['date'] >= date_from) and (date_to is None or record['date'] <= date_to)]
            if not matched_records:
                continue
            pairs.sort(key=lambda pair: pair[0]['startedAt'])
            clean_requests, records = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            started = records[0]['started_at']
            status = metadata['status'] if metadata['status'] in STATES else records[-1]['status']
            ends = [record.get('ended_at') for record in records]
            ended = max(ends) if all(end is not None for end in ends) and status != 'running' else None
            elapsed_end = max(record.get('ended_at', record['started_at'] + record['duration_s']) for record in records)
            instance = metadata['instance_id'] if isinstance(metadata['instance_id'], str) and metadata['instance_id'] else None
            usage_active = (metadata['message_status'] == 'generating' or metadata['status'] in {'running', 'waiting'}
                            or any(record['status'] == 'running' for record in records))
            for request, record in pairs:
                record.update(source='internal', conversation_id=kind[len(MESSAGE_PREFIX):], message_id=identifier,
                              request_key=request['id'], instance_id=instance, turn_state=status, _usage_active=usage_active)
            group = {**request_history_stats(records), 'source': 'internal', 'conversation_id': kind[len(MESSAGE_PREFIX):],
                     'message_id': identifier, 'instance_id': instance, 'alias': metadata['alias'] or '',
                     'started_at': started, 'ended_at': ended, 'elapsed_s': max(0, elapsed_end - started),
                     'status': status, 'requests': clean_requests, 'matched_request_count': len(matched_records)}
            if isinstance(metadata['backend'], str) and metadata['backend']:
                group['backend'] = metadata['backend']
            capacity = _reported(metadata['context_capacity'])
            if capacity is not None:
                group['context_capacity'] = capacity
            groups.append(group)
            all_records.extend(matched_records)
            for record in matched_records:
                day = days.setdefault(record['date'], {'date': record['date'], 'request_count': 0, 'instances': {}})
                day['request_count'] += 1
                identity = (instance, group['conversation_id'] if instance is None else None)
                entry = {'source': 'internal', 'id': instance, 'alias': group['alias'],
                         'backend': group.get('backend', ''), 'started_at': started}
                if instance is None:
                    entry['conversation_id'] = group['conversation_id']
                day['instances'][identity] = entry
        return {'groups': groups, 'records': all_records, 'coverage': coverage, 'legacy': legacy,
                'dates': [{**day, 'instances': list(day['instances'].values())} for day in days.values()]}

    def interrupt_generating(self):
        with self.store.lock, self.store.connection:
            rows = self.store.connection.execute(
                "SELECT kind,body FROM records WHERE kind GLOB ? AND json_extract(body,'$.status')='generating'",
                (MESSAGE_PREFIX + '*',)).fetchall()
            for kind, body in rows:
                message = json.loads(body)
                message.update(status='interrupted', thinking=False)
                for request in message.get('requests', []):
                    if request.get('status') == 'running':
                        request['status'] = 'incomplete'
                if message.get('turn', {}).get('status') == 'running':
                    message['turn']['status'] = 'incomplete'
                for event in message.get('activity', []):
                    if event.get('kind') == 'tool' and event.get('status') in ('pending', 'running'):
                        event['status'] = 'cancelled'
                self._put(kind, message)
