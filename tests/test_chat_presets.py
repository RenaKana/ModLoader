"""Preset APIs use isolated databases; no inference process is started."""
import json
import sqlite3
import uuid

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.chat_presets import PRESET_KIND, PRESET_STATE_KIND
from modloader.conversations import DETAIL_KIND


def preset(**changes):
    return {'id': str(uuid.uuid4()), 'name': '通用预设', 'system': '保留系统提示词。',
            'toolsDraft': '[\n  {"type": "function", "function": {"name": "lookup"}}\n]', **changes}


@pytest.fixture
async def controller(tmp_path):
    value = Controller(tmp_path / 'data', 'preset-fixture')
    try:
        yield value
    finally:
        await value.close()


@pytest.fixture
async def client(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                base_url='http://testserver',
                                headers={'Authorization': 'Bearer preset-fixture'}) as value:
        yield value


async def test_create_update_and_default_are_separate_and_create_retry_is_idempotent(client):
    assert (await client.get('/api/chat-presets')).json() == {'items': [], 'default_id': None}
    body = preset()
    response = await client.post('/api/chat-presets', json=body)
    assert response.status_code == 200, response.text
    original = response.json()['items'][0]
    assert {field: original[field] for field in body} == body
    assert set(original) == {'id', 'name', 'system', 'toolsDraft', 'created_at', 'updated_at'}
    assert original['created_at'] == original['updated_at']
    retry = await client.post('/api/chat-presets', json={**body, 'name': '不得覆盖', 'system': ''})
    assert retry.json() == {'items': [original], 'default_id': None}
    default = await client.patch('/api/chat-presets/default', json={'default_id': body['id']})
    assert default.json() == {'items': [original], 'default_id': body['id']}
    changed = await client.patch('/api/chat-presets/' + body['id'],
                                 json={'name': '新名称', 'system': '新版提示', 'toolsDraft': '  []\n'})
    assert changed.status_code == 200, changed.text
    updated = changed.json()['items'][0]
    assert updated['created_at'] == original['created_at']
    assert updated['name'] == '新名称' and updated['system'] == '新版提示' and updated['toolsDraft'] == '  []\n'
    assert changed.json()['default_id'] == body['id']
    assert (await client.patch('/api/chat-presets/default', json={'default_id': None})).json() == {'items': [updated], 'default_id': None}


async def test_first_save_can_set_default_in_the_same_transaction(client, controller):
    body = preset(make_default=True)
    response = await client.post('/api/chat-presets', json=body)
    assert response.status_code == 200, response.text
    catalog = response.json()
    assert catalog['default_id'] == body['id']
    assert 'make_default' not in catalog['items'][0]
    before = controller.chat_presets.list()
    controller.store.connection.execute(
        f"CREATE TEMP TRIGGER fail_save_default BEFORE UPDATE ON records WHEN OLD.kind='{PRESET_STATE_KIND}' "
        "BEGIN SELECT RAISE(ABORT,'fixture save default failure'); END")
    failed = preset(make_default=True)
    with pytest.raises(sqlite3.IntegrityError, match='fixture save default failure'):
        controller.chat_presets.create(failed)
    assert controller.chat_presets.list() == before
    assert controller.store.get(PRESET_KIND, failed['id']) is None
    with pytest.raises(sqlite3.IntegrityError, match='fixture save default failure'):
        controller.chat_presets.patch(body['id'], {'name': '不得部分更新', 'make_default': False})
    assert controller.chat_presets.list() == before


async def test_saved_default_choice_can_set_unset_and_preserve_another_default(client):
    first = preset(make_default=True)
    await client.post('/api/chat-presets', json=first)
    second = preset(name='另存副本', make_default=False)
    copied = await client.post('/api/chat-presets', json=second)
    assert copied.json()['default_id'] == first['id']
    chosen = await client.patch('/api/chat-presets/' + second['id'], json={'name': '副本作为默认', 'make_default': True})
    assert chosen.json()['default_id'] == second['id']
    untouched = await client.patch('/api/chat-presets/' + first['id'], json={'system': '非默认预设的新内容', 'make_default': False})
    assert untouched.json()['default_id'] == second['id']
    cleared = await client.patch('/api/chat-presets/' + second['id'], json={'system': '取消默认时一起更新', 'make_default': False})
    assert cleared.status_code == 200 and cleared.json()['default_id'] is None
    assert cleared.json()['items'][1]['system'] == '取消默认时一起更新'
    assert all('make_default' not in item for item in cleared.json()['items'])


async def test_default_survives_reopen_without_rewriting_conversations(tmp_path):
    root = tmp_path / 'reopen'
    first = Controller(root, 'fixture')
    body = preset()
    saved = {'id': str(uuid.uuid4()), 'title': '既有对话',
             'selection': {'profile_id': 'original-profile', 'model_id': 'original-model', 'alias': '原模型'},
             'draft': {'text': '保留草稿', 'images': []},
             'settings': {'system': body['system'], 'toolsDraft': body['toolsDraft'], 'presetId': body['id'],
                          'temperature': 2.5, 'maxTokens': -1, 'stream': False}}
    try:
        first.chat_presets.create(body)
        first.chat_presets.set_default({'default_id': body['id']})
        first.conversations.create(saved)
        first.conversations.patch(saved['id'], {'activate': True})
        previous = first.conversations.get(saved['id'])
        first.chat_presets.patch(body['id'], {'system': '仅修改预设', 'toolsDraft': ''})
        assert first.conversations.get(saved['id']) == previous
        first.conversations.patch(saved['id'], {'selection': {'profile_id': 'new-profile', 'model_id': 'new-model', 'alias': '新模型'}})
        assert first.conversations.get(saved['id'])['settings'] == saved['settings']
        first.conversations.patch(saved['id'], {'settings': {**saved['settings'], 'system': '仅修改本对话'}})
        assert first.chat_presets.list()['items'][0]['system'] == '仅修改预设'
    finally:
        await first.close()
    reopened = Controller(root, 'fixture')
    try:
        catalog = reopened.chat_presets.list()
        assert catalog['default_id'] == body['id']
        assert catalog['items'][0]['system'] == '仅修改预设'
        assert reopened.conversations.get(saved['id'])['settings'] == {**saved['settings'], 'system': '仅修改本对话'}
    finally:
        await reopened.close()


async def test_persisted_settings_keep_legacy_contract_and_old_style_edits_preserve_content(client, controller):
    source = preset()
    await client.post('/api/chat-presets', json=source)
    legacy_settings = {'system': '旧版可编辑的系统词', 'toolsDraft': source['toolsDraft'],
                       'temperature': 1.7, 'maxTokens': -1, 'stream': False}
    body = {'id': str(uuid.uuid4()), 'title': '兼容会话', 'selection': None,
            'draft': {'text': '保留草稿', 'images': []},
            'settings': {**legacy_settings, 'presetId': source['id']}}
    response = await client.post('/api/conversations', json=body)
    assert response.status_code == 200, response.text
    stored = controller.store.get(DETAIL_KIND, body['id'])
    assert stored['settings'] == legacy_settings
    assert stored['preset_id'] == source['id']
    assert set(stored['settings']) == {'system', 'toolsDraft', 'temperature', 'maxTokens', 'stream'}
    assert (await client.get('/api/conversations/' + body['id'])).json()['settings'] == body['settings']
    changed = {**body['settings'], 'system': '新版更新后的快照'}
    assert (await client.patch('/api/conversations/' + body['id'], json={'settings': changed})).status_code == 200
    stored = controller.store.get(DETAIL_KIND, body['id'])
    assert stored['settings'] == {**legacy_settings, 'system': '新版更新后的快照'}
    assert stored['preset_id'] == source['id']
    # This is the old Conversations.patch operation: update the stored detail's
    # settings object and leave unknown top-level metadata untouched.
    stored.update(settings={**stored['settings'], 'system': '旧版编辑的新内容', 'toolsDraft': '  []\n'})
    controller.store.put(DETAIL_KIND, stored)
    restored = (await client.get('/api/conversations/' + body['id'])).json()
    assert restored['settings'] == {**legacy_settings, 'system': '旧版编辑的新内容', 'toolsDraft': '  []\n', 'presetId': source['id']}
    assert restored['draft'] == body['draft']
    # Detaching the source through the new client removes only its metadata.
    detached = {field: value for field, value in restored['settings'].items() if field != 'presetId'}
    assert (await client.patch('/api/conversations/' + body['id'], json={'settings': detached})).status_code == 200
    assert controller.store.get(DETAIL_KIND, body['id'])['settings'] == detached
    assert 'preset_id' not in controller.store.get(DETAIL_KIND, body['id'])


async def test_invalid_tools_and_generation_fields_cannot_change_saved_presets(client, controller):
    body = preset()
    await client.post('/api/chat-presets', json=body)
    await client.patch('/api/chat-presets/default', json={'default_id': body['id']})
    before = controller.chat_presets.list()
    invalid = [
        {'name': ''}, {'name': '  '}, {'name': 3}, {'system': None}, {'toolsDraft': []},
        {'toolsDraft': '{"type":"function"}'}, {'toolsDraft': '[}'}, {'toolsDraft': '[NaN]'},
        {'temperature': .7}, {'maxTokens': 4000}, {'stream': True}, {'model_id': 'not-a-model-preset'},
        {'id': str(uuid.uuid4())},
        {'make_default': 'true'},
    ]
    for fields in invalid:
        response = await client.patch('/api/chat-presets/' + body['id'], json={'name': '不得部分写入', **fields})
        assert response.status_code == 400, (fields, response.text)
        assert controller.chat_presets.list() == before
    missing = str(uuid.uuid4())
    assert (await client.patch('/api/chat-presets/' + missing, json={'system': ''})).status_code == 404
    assert (await client.patch('/api/chat-presets/default', json={'default_id': missing})).status_code == 404
    assert (await client.patch('/api/chat-presets/default', json={'default_id': 'not-a-uuid'})).status_code == 400
    assert (await client.patch('/api/chat-presets/default', json={'default_id': None, 'extra': 1})).status_code == 400
    assert (await client.post('/api/chat-presets', json=preset(id='not-a-uuid'))).status_code == 400
    assert controller.chat_presets.list() == before


async def test_large_names_and_tool_arrays_preserve_source_without_fixture_limits(client):
    name = '完整名称' * 1000
    tools = '[\n' + ',\n'.join(json.dumps({'name': f'tool-{index}', 'extra': '值' * 50}, ensure_ascii=False)
                              for index in range(600)) + '\n]'
    body = preset(name=name, system='系统提示\n' * 2000, toolsDraft=tools)
    response = await client.post('/api/chat-presets', json=body)
    assert response.status_code == 200, response.text
    assert all(response.json()['items'][0][field] == value for field, value in body.items())
    assert (await client.patch('/api/chat-presets/' + body['id'], json={'toolsDraft': ' \n\t'})).status_code == 200


async def test_tool_concurrency_is_optional_positive_and_persists_with_presets(client, controller):
    body = preset(toolConcurrency=3)
    response = await client.post('/api/chat-presets', json=body)
    assert response.status_code == 200, response.text
    assert response.json()['items'][0]['toolConcurrency'] == 3
    changed = await client.patch('/api/chat-presets/' + body['id'], json={'toolConcurrency': 10**30})
    assert changed.status_code == 200, changed.text
    assert changed.json()['items'][0]['toolConcurrency'] == 10**30
    before = controller.chat_presets.list()
    for value in (None, 0, -1, True, 1.5, '3'):
        response = await client.patch('/api/chat-presets/' + body['id'], json={'name': '不得部分保存', 'toolConcurrency': value})
        assert response.status_code == 400, (value, response.text)
        assert controller.chat_presets.list() == before
    # Older clients omit the new setting and must leave its saved value intact.
    response = await client.patch('/api/chat-presets/' + body['id'], json={'system': '旧客户端更新'})
    assert response.json()['items'][0]['toolConcurrency'] == 10**30
    old = preset()
    response = await client.post('/api/chat-presets', json=old)
    assert 'toolConcurrency' not in next(item for item in response.json()['items'] if item['id'] == old['id'])


async def test_default_write_failure_leaves_previous_choice_and_preset_intact(controller):
    first, second = preset(), preset()
    controller.chat_presets.create(first)
    controller.chat_presets.create(second)
    controller.chat_presets.set_default({'default_id': first['id']})
    before = controller.chat_presets.list()
    controller.store.connection.execute(
        f"CREATE TEMP TRIGGER fail_default BEFORE UPDATE ON records WHEN OLD.kind='{PRESET_STATE_KIND}' "
        "BEGIN SELECT RAISE(ABORT,'fixture default failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='fixture default failure'):
        controller.chat_presets.set_default({'default_id': second['id']})
    assert controller.chat_presets.list() == before
    assert len(controller.store.all(PRESET_KIND)) == 2


async def test_preset_api_is_management_only_and_requires_auth(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver') as client:
        assert (await client.get('/api/chat-presets')).status_code == 401
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller, public=True)),
                                base_url='http://testserver', headers={'Authorization': 'Bearer ' + controller.api_key}) as client:
        assert (await client.get('/api/chat-presets')).status_code == 404
        assert (await client.post('/api/chat-presets', json=preset())).status_code == 404
        assert (await client.patch('/api/chat-presets/default', json={'default_id': None})).status_code == 404
