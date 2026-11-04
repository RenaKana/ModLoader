"""Reusable chat settings, independent of model profiles."""
from datetime import datetime, timezone
import json
import uuid

from .chat_reasoning import validate_reasoning_effort


PRESET_KIND = 'chat_presets'
PRESET_STATE_KIND = 'chat_preset_state'


class ChatPresetNotFound(LookupError):
    pass


def _identifier(value):
    if not isinstance(value, str):
        raise ValueError('预设 id 必须为 UUID')
    try:
        uuid.UUID(value)
    except ValueError:
        raise ValueError('预设 id 必须为 UUID') from None
    return value


def _validate(body, create=False):
    required = {'id', 'name', 'system', 'toolsDraft'} if create else set()
    allowed = {'name', 'system', 'toolsDraft', 'make_default', 'reasoningEffort', 'toolConcurrency'} | required
    if not isinstance(body, dict) or not required <= body.keys() or body.keys() - allowed:
        raise ValueError('对话预设字段不完整或包含不支持的字段')
    if create:
        _identifier(body['id'])
    if 'make_default' in body and type(body['make_default']) is not bool:
        raise ValueError('make_default 必须为布尔值')
    if 'reasoningEffort' in body:
        validate_reasoning_effort(body['reasoningEffort'])
    if 'toolConcurrency' in body and (type(body['toolConcurrency']) is not int or body['toolConcurrency'] < 1):
        raise ValueError('toolConcurrency 必须为正整数')
    for field in ('name', 'system', 'toolsDraft'):
        if field in body and (not isinstance(body[field], str) or (field == 'name' and not body[field].strip())):
            raise ValueError('预设名称不能为空' if field == 'name' else f'{field} 必须为字符串')
    if body.get('toolsDraft', '').strip():
        def invalid_constant(value):
            raise ValueError('工具定义须为有效 JSON 数组')
        try:
            tools = json.loads(body['toolsDraft'], parse_constant=invalid_constant)
        except ValueError:
            raise ValueError('工具定义须为有效 JSON 数组') from None
        if not isinstance(tools, list):
            raise ValueError('工具定义须为 JSON 数组')


class ChatPresets:
    def __init__(self, store):
        self.store = store

    def _put(self, kind, record):
        self.store.connection.execute(
            'INSERT INTO records(kind,id,body) VALUES(?,?,?) '
            'ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body',
            (kind, record['id'], json.dumps(record, ensure_ascii=False, allow_nan=False)))

    def _require(self, identifier):
        _identifier(identifier)
        value = self.store.get(PRESET_KIND, identifier)
        if value is None:
            raise ChatPresetNotFound('对话预设不存在')
        return value

    def list(self):
        with self.store.lock:
            state = self.store.get(PRESET_STATE_KIND, 'main')
            return {'items': self.store.all(PRESET_KIND), 'default_id': state['default_id'] if state else None}

    def _apply_default_choice(self, identifier, choice):
        if choice is None:
            return
        if not choice:
            state = self.store.get(PRESET_STATE_KIND, 'main')
            if not state or state['default_id'] != identifier:
                return
        self._put(PRESET_STATE_KIND, {'id': 'main', 'default_id': identifier if choice else None})

    def create(self, body):
        _validate(body, create=True)
        with self.store.lock, self.store.connection:
            if self.store.get(PRESET_KIND, body['id']) is None:
                now = datetime.now(timezone.utc).isoformat()
                effort = {'reasoningEffort': body['reasoningEffort']} if body.get('reasoningEffort') is not None else {}
                concurrency = {'toolConcurrency': body['toolConcurrency']} if 'toolConcurrency' in body else {}
                self._put(PRESET_KIND, {**{field: body[field] for field in ('id', 'name', 'system', 'toolsDraft')}, **effort, **concurrency, 'created_at': now, 'updated_at': now})
            self._apply_default_choice(body['id'], body.get('make_default'))
            return self.list()

    def patch(self, identifier, body):
        _validate(body)
        with self.store.lock, self.store.connection:
            previous = self._require(identifier)
            fields = {field: value for field, value in body.items() if field != 'make_default'}
            if any(previous.get(field) != value for field, value in fields.items()):
                updated = {**previous, **fields, 'updated_at': datetime.now(timezone.utc).isoformat()}
                if updated.get('reasoningEffort') is None:
                    updated.pop('reasoningEffort', None)
                self._put(PRESET_KIND, updated)
            self._apply_default_choice(identifier, body.get('make_default'))
            return self.list()

    def set_default(self, body):
        if not isinstance(body, dict) or set(body) != {'default_id'}:
            raise ValueError('默认预设字段必须为 default_id')
        identifier = body['default_id']
        with self.store.lock, self.store.connection:
            if identifier is not None:
                self._require(identifier)
            self._put(PRESET_STATE_KIND, {'id': 'main', 'default_id': identifier})
            return self.list()
