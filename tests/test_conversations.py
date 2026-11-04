"""Chat history acceptance uses temporary databases and a mocked inference backend."""
import json
import sqlite3
import uuid

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.conversations import DETAIL_KIND, MESSAGE_PREFIX, SUMMARY_KIND
from modloader.models import LaunchSpec, RuntimeInstance


def conversation(**changes):
    return {'id': str(uuid.uuid4()), 'title': '新对话', 'selection': None,
            'draft': {'text': '', 'images': []},
            'settings': {'system': '', 'stream': True, 'toolsDraft': ''}, **changes}


def message(identifier='message', **changes):
    return {'id': identifier, 'role': 'assistant', 'text': '已收到', 'images': [],
            'createdAt': 1730000000123, 'status': 'completed', **changes}


def image():
    return {'id': 'picture', 'name': '截图.png', 'size': 100_000,
            'url': 'data:image/png;base64,' + 'a' * 100_000}


def document(**changes):
    return {'id': str(uuid.uuid4()), 'name': '模型参数.md', 'size': 512, 'text': '# 模型参数\n上下文 4096\n', **changes}


@pytest.fixture
async def controller(tmp_path):
    value = Controller(tmp_path / 'data', 'conversation-management-secret')
    try:
        yield value
    finally:
        await value.close()


@pytest.fixture
async def client(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                base_url='http://testserver',
                                headers={'Authorization': 'Bearer conversation-management-secret'}) as value:
        yield value


def stored_rows(controller):
    with controller.store.lock:
        return controller.store.connection.execute('SELECT kind,id,body FROM records ORDER BY kind,id').fetchall()


async def test_crud_isolates_messages_and_post_retry_does_not_overwrite(client, controller):
    first, second = conversation(title='第一段'), conversation(title='第二段')
    assert (await client.get('/api/conversations')).json() == {'items': [], 'active_id': None}
    summaries = []
    for body in (first, second):
        response = await client.post('/api/conversations', json=body)
        assert response.status_code == 200, response.text
        summaries.append(response.json())
    assert set(summaries[0]) == {'id', 'title', 'created_at', 'updated_at', 'last_message_at', 'last_sent_at', 'pinned', 'sort_order', 'message_count'}
    assert summaries[0]['created_at'] == summaries[0]['updated_at']
    assert summaries[0]['message_count'] == 0
    assert (await client.get('/api/conversations')).json()['items'] == list(reversed(summaries))
    for body, text in ((first, '第一段正文'), (second, '第二段正文')):
        response = await client.patch('/api/conversations/' + body['id'], json={'messages': [message(text=text)]})
        assert response.status_code == 200 and response.json()['message_count'] == 1
    original = (await client.get('/api/conversations/' + first['id'])).json()
    retry = await client.post('/api/conversations', json={**first, 'title': '不可覆盖', 'draft': {'text': '重试', 'images': []}})
    assert retry.status_code == 200 and retry.json()['title'] == '第一段'
    assert (await client.get('/api/conversations/' + first['id'])).json() == original
    changes = {'title': '重新命名', 'selection': {'profile_id': 'p', 'model_id': 'm', 'alias': '模型'},
               'draft': {'text': '未发送正文', 'images': []},
               'settings': {'system': '提示', 'temperature': .7, 'maxTokens': 4096, 'stream': False, 'toolsDraft': '[]'}}
    changed = await client.patch('/api/conversations/' + first['id'], json=changes)
    assert changed.status_code == 200 and changed.json()['message_count'] == 1
    detail = (await client.get('/api/conversations/' + first['id'])).json()
    assert all(detail[field] == value for field, value in changes.items())
    assert detail['created_at'] == original['created_at'] and detail['messages'] == original['messages']
    assert (await client.get('/api/conversations/' + second['id'])).json()['messages'] == [message(text='第二段正文')]
    absent = '/api/conversations/' + str(uuid.uuid4())
    assert (await client.get(absent)).status_code == 404
    assert (await client.patch(absent, json={'title': '不存在'})).status_code == 404
    assert (await client.delete(absent)).status_code == 404
    assert (await client.get('/api/conversations/not-a-uuid')).status_code == 400


async def test_reopened_database_keeps_rich_payload_and_interrupts_only_generating(tmp_path):
    root = tmp_path / 'reopen'
    first = Controller(root, 'fixture')
    body = conversation(selection={'profile_id': 'offline-profile', 'model_id': 'offline-model', 'alias': '离线模型'},
                        draft={'text': '保留草稿', 'images': [image()]},
                        settings={'system': '测试系统提示', 'temperature': 3.5, 'maxTokens': -1,
                                  'stream': False, 'toolsDraft': '[{"type":"function"}]'})
    rich = message('rich', text='正文' * 40_000, images=[image()], reasoning='思考\n完整保留', reasoningHistory='检索前思考\n\n阅读前思考', thinking=False,
                   model='原模型', profileId='offline-profile', modelId='offline-model', stopped=False, error='',
                   tools=[{'id': 'tool-1', 'type': 'function', 'function': {'name': '查询', 'arguments': '{"value":1}'},
                           'unknown_tool_value': {'provider': [None, True, '额外信息']}}],
                   usage={'prompt_tokens': 12, 'completion_tokens': 24, 'future_count': 2**70,
                          'future_details': {'nested': [1, False, None]}},
                   timings={'predicted_ms': 450, 'future_timing': {'units': 'ms'}},
                   elapsed=1.25, firstOutput=.125, responseAt=1730000000248, finishReason='tool_calls',
                   providerPayload={'arbitrary': ['未知字段', 2**75, {'keep': True}]})
    pending = message('pending', status='generating', text='半段正文', reasoning='尚未结束', reasoningHistory='此前轮次思考', thinking=True)
    stopped = message('stopped', status='stopped', stopped=True)
    failed = message('failed', status='failed', error='真实错误')
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(first)), base_url='http://testserver',
                                    headers={'Authorization': 'Bearer fixture'}) as api:
            assert (await api.post('/api/conversations', json=body)).status_code == 200
            response = await api.patch('/api/conversations/' + body['id'],
                                       json={'messages': [rich, pending, stopped, failed], 'activate': True})
            saved_summary = response.json()
            assert response.status_code == 200, response.text
    finally:
        await first.close()
    reopened = Controller(root, 'fixture')
    try:
        detail = reopened.conversations.get(body['id'])
        assert detail == {**body, **saved_summary, 'messages': [rich, {**pending, 'status': 'interrupted', 'thinking': False}, stopped, failed]}
        assert reopened.conversations.list() == {'items': [saved_summary], 'active_id': body['id']}
        assert not {'elapsed', 'firstOutput', 'usage', 'timings'} & detail['messages'][1].keys()
    finally:
        await reopened.close()


async def test_clear_preserves_details_and_delete_only_clears_its_active_id(client, controller):
    first, second = conversation(draft={'text': '草稿', 'images': [image()]}), conversation()
    for body in (first, second):
        await client.post('/api/conversations', json=body)
        await client.patch('/api/conversations/' + body['id'], json={'messages': [message()]})
    before = (await client.get('/api/conversations/' + first['id'])).json()
    activated = await client.patch('/api/conversations/' + first['id'], json={'activate': True})
    assert activated.json()['updated_at'] == before['updated_at']
    cleared = await client.patch('/api/conversations/' + first['id'], json={'clear_messages': True})
    assert cleared.json()['message_count'] == 0
    detail = (await client.get('/api/conversations/' + first['id'])).json()
    assert detail['messages'] == []
    assert all(detail[field] == before[field] for field in ('title', 'selection', 'draft', 'settings', 'created_at'))
    assert (await client.get('/api/conversations/' + second['id'])).json()['messages'] == [message()]
    second_before = (await client.get('/api/conversations/' + second['id'])).json()
    await client.patch('/api/conversations/' + second['id'], json={'activate': True})
    listed = (await client.get('/api/conversations')).json()
    assert listed['active_id'] == second['id']
    assert next(item for item in listed['items'] if item['id'] == second['id'])['updated_at'] == second_before['updated_at']
    deleted = await client.delete('/api/conversations/' + first['id'])
    assert deleted.status_code == 204 and deleted.content == b''
    assert (await client.get('/api/conversations')).json()['active_id'] == second['id']
    assert controller.store.get(DETAIL_KIND, first['id']) is None
    assert controller.store.all(MESSAGE_PREFIX + first['id']) == []
    assert (await client.delete('/api/conversations/' + second['id'])).status_code == 204
    assert (await client.get('/api/conversations')).json() == {'items': [], 'active_id': None}
    assert not [row for row in stored_rows(controller) if row[0] in (SUMMARY_KIND, DETAIL_KIND) or row[0].startswith(MESSAGE_PREFIX)]


async def test_incremental_upserts_keep_old_images_and_insertion_order_and_summary_reads_light(client, controller):
    body = conversation(draft={'text': '未发送图片', 'images': [image()]})
    await client.post('/api/conversations', json=body)
    user = message('old-picture', role='user', images=[image()])
    assistant = message('assistant', status='generating', text='一')
    url = '/api/conversations/' + body['id']
    assert (await client.patch(url, json={'messages': [user, assistant]})).status_code == 200
    database = controller.store.connection
    kind = MESSAGE_PREFIX + body['id']
    old = database.execute('SELECT rowid,body FROM records WHERE kind=? AND id=?', (kind, user['id'])).fetchone()
    # A rewrite of an old attachment is an actual SQLite failure, not a mock count.
    database.execute("CREATE TEMP TRIGGER protect_old_image BEFORE UPDATE ON records WHEN OLD.id='old-picture' BEGIN SELECT RAISE(FAIL,'old image was rewritten'); END")
    statements = []
    database.set_trace_callback(statements.append)
    try:
        assert (await client.patch(url, json={'messages': [{**assistant, 'text': '一二三'}]})).status_code == 200
        streaming_reads = [query for query in statements if query.lstrip().upper().startswith('SELECT')]
        assert not any(DETAIL_KIND in query or (kind in query and 'SELECT body' in query) for query in streaming_reads)
        # Also tolerate the same complete old message in an upsert batch without rewriting it.
        assert (await client.patch(url, json={'messages': [{**assistant, 'text': '一二三', 'status': 'completed'}, user]})).status_code == 200
        statements.clear()
        listed = (await client.get('/api/conversations')).json()
        assert listed['items'][0]['message_count'] == 2
        assert not any(DETAIL_KIND in query or MESSAGE_PREFIX in query for query in statements)
    finally:
        database.set_trace_callback(None)
    assert database.execute('SELECT rowid,body FROM records WHERE kind=? AND id=?', (kind, user['id'])).fetchone() == old
    detail = (await client.get(url)).json()
    assert detail['messages'] == [user, {**assistant, 'text': '一二三', 'status': 'completed'}]
    # Clear and append can be combined atomically without leaving the old messages behind.
    response = await client.patch(url, json={'clear_messages': True, 'messages': [message('new')]})
    assert response.json()['message_count'] == 1
    assert (await client.get(url)).json()['messages'] == [message('new')]


async def test_malformed_patches_reject_all_changes_atomically(client, controller):
    body, active = conversation(), conversation()
    for value in (body, active):
        await client.post('/api/conversations', json=value)
    await client.patch('/api/conversations/' + body['id'], json={'messages': [message()]})
    await client.patch('/api/conversations/' + active['id'], json={'activate': True})
    original = stored_rows(controller)
    malformed = [
        {'title': '  '}, {'selection': {'profile_id': 'p', 'model_id': 'm'}},
        {'selection': {'profile_id': '', 'model_id': 'm', 'alias': 'a'}},
        {'draft': {'text': 3, 'images': []}}, {'draft': {'text': '', 'images': [dict(image(), size=-1)]}},
        {'settings': {'system': '', 'stream': 'true', 'toolsDraft': ''}},
        {'settings': {'system': '', 'stream': True, 'toolsDraft': '', 'temperature': True}},
        {'messages': [message(), message('invalid', role='system')]},
        {'messages': [message(createdAt='today')]}, {'messages': [message(status='unknown')]},
        {'messages': [message(responseAt='today')]}, {'messages': [message(responseAt=float('inf'))]},
        {'messages': [message(images='not-images')]}, {'messages': [message(id=' ')]},
        {'messages': [message(thinking='yes')]}, {'messages': [message(usage=[])]},
        {'messages': [message(reasoningHistory=['not text'])]},
        {'messages': [message(), message()]}, {'messages': [message(providerPayload=float('nan'))]},
        {'messages': [message(createdAt=float('inf'))]}, {'clear_messages': 'true'}, {'activate': 1},
        {'unrecognized_patch_field': True},
    ]
    for invalid in malformed:
        payload = {'title': '不得写入', 'clear_messages': True, 'activate': True, **invalid}
        response = await client.patch('/api/conversations/' + body['id'], content=json.dumps(payload),
                                      headers={'Content-Type': 'application/json'})
        assert response.status_code == 400, (invalid, response.text)
        assert stored_rows(controller) == original
    for bad in (conversation(id='not-uuid'), conversation(title=''), conversation(draft=None)):
        assert (await client.post('/api/conversations', json=bad)).status_code == 400
        assert stored_rows(controller) == original


async def test_sqlite_failure_rolls_back_detail_messages_summary_and_active(controller):
    body = conversation()
    controller.conversations.create(body)
    controller.conversations.patch(body['id'], {'messages': [message()], 'activate': True})
    before = stored_rows(controller)
    controller.store.connection.execute("CREATE TEMP TRIGGER fail_message BEFORE INSERT ON records WHEN NEW.id='fail' BEGIN SELECT RAISE(ABORT,'fixture write failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='fixture write failure'):
        controller.conversations.patch(body['id'], {'title': '不得写入', 'draft': {'text': '不得写入', 'images': []},
                                                   'clear_messages': True, 'messages': [message('fail')]})
    assert stored_rows(controller) == before
    controller.store.connection.execute("CREATE TEMP TRIGGER fail_active_update BEFORE UPDATE ON records WHEN OLD.kind='conversation_state' BEGIN SELECT RAISE(ABORT,'fixture active failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='fixture active failure'):
        controller.conversations.delete(body['id'])
    assert stored_rows(controller) == before


async def test_malformed_tools_patch_preserves_all_original_records(client, controller):
    body, active = conversation(draft={'text': '保留草稿', 'images': []}), conversation()
    for value in (body, active):
        await client.post('/api/conversations', json=value)
    url = '/api/conversations/' + body['id']
    valid_function = {'name': 'lookup', 'arguments': '{}'}
    await client.patch(url, json={'messages': [message(tools=[{'function': valid_function}])]})
    await client.patch('/api/conversations/' + active['id'], json={'activate': True})
    before = stored_rows(controller)
    malformed = [{}, {'function': None}, {'function': {'name': 'lookup'}},
                 {'function': {'name': 5, 'arguments': '{}'}},
                 {'function': {'name': 'lookup', 'arguments': {}}},
                 {'id': 17, 'function': valid_function}, {'type': False, 'function': valid_function}]
    for tool in malformed:
        response = await client.patch(url, json={'title': '不得写入', 'draft': {'text': '不得写入', 'images': []},
                                                'clear_messages': True, 'activate': True,
                                                'messages': [message(tools=[tool])]})
        assert response.status_code == 400, (tool, response.text)
        assert stored_rows(controller) == before


async def test_management_auth_and_public_app_do_not_expose_history_or_resources(controller):
    app, public = build_app(controller), build_app(controller, public=True)
    identifier = str(uuid.uuid4())
    methods = [('GET', '/api/conversations'), ('POST', '/api/conversations'),
               ('GET', '/api/conversations/' + identifier), ('PATCH', '/api/conversations/' + identifier),
               ('DELETE', '/api/conversations/' + identifier), ('POST', '/api/chat-tools/attach'),
               ('GET', '/api/chat-tools/resources/' + identifier), ('GET', '/api/chat-tools/resources/' + identifier + '/content'),
               ('POST', '/api/chat-tools/resources/' + identifier + '/action')]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as api:
        for method, path in methods:
            assert (await api.request(method, path, json={})).status_code == 401
            assert (await api.request(method, path, json={}, headers={'Authorization': 'Bearer ' + controller.api_key})).status_code == 401
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=public), base_url='http://testserver',
                                headers={'Authorization': 'Bearer ' + controller.api_key}) as api:
        for method, path in methods:
            assert (await api.request(method, path, json={})).status_code == 404
        assert (await api.get('/v1/models')).status_code == 200
        assert (await api.post('/api/chat', json={})).status_code == 404


async def test_inference_gateway_does_not_create_or_modify_history(controller, tmp_path, monkeypatch):
    existing = conversation()
    controller.conversations.create(existing)
    controller.conversations.patch(existing['id'], {'messages': [message()]})
    controller.store.put('profiles', {'id': 'fixture-profile', 'generation': {}})
    instance = RuntimeInstance(id='fixture', model_id='fixture-model', profile_id='fixture-profile', profile_revision=1,
                               alias='fixture-alias', backend='llama_cpp', runtime_id='fixture-runtime', state='ready',
                               port=12345, started_at='', capabilities={'input': ['text']})
    controller.runtime.instances[instance.id] = instance
    controller.runtime.specs[instance.id] = LaunchSpec(argv=[], cwd=str(tmp_path), model_name='backend-name')
    controller.runtime.requests[instance.id] = set()
    original = httpx.AsyncClient
    requests = []

    def backend(request):
        payload = json.loads(request.content)
        requests.append(payload)
        chunk = {'model': 'backend-name', 'choices': [{'message': {'role': 'assistant', 'content': 'fixture reply'}, 'finish_reason': 'stop'}],
                 'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}
        if payload.get('stream'):
            return httpx.Response(200, content=('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode(),
                                  headers={'Content-Type': 'text/event-stream'})
        return httpx.Response(200, json=chunk)

    before = stored_rows(controller)
    try:
        async with original(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                            headers={'Authorization': 'Bearer conversation-management-secret'}) as management, \
                   original(transport=httpx.ASGITransport(app=build_app(controller, public=True)), base_url='http://testserver',
                            headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
            monkeypatch.setattr('modloader.app.httpx.AsyncClient', lambda **options: original(transport=httpx.MockTransport(backend), **options))
            for api, path, stream in ((management, '/api/chat', False), (management, '/api/chat', True),
                                      (public, '/v1/chat/completions', False), (public, '/v1/chat/completions', True)):
                response = await api.post(path, json={'model': 'fixture-alias', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream})
                assert response.status_code == 200, response.text
                if stream:
                    assert '[DONE]' in response.text and 'fixture-alias' in response.text
                else:
                    assert response.json()['model'] == 'fixture-alias'
                assert stored_rows(controller) == before
        assert len(requests) == 4 and all(request['model'] == 'backend-name' for request in requests)
        assert instance.active_requests == 0
    finally:
        controller.runtime.instances.clear()


async def test_last_message_time_ignores_rename_and_draft_and_handles_legacy(client, controller):
    body = conversation()
    url = '/api/conversations/' + body['id']
    created = (await client.post('/api/conversations', json=body)).json()
    assert created['last_message_at'] is None
    sent = message('user', role='user', createdAt=1700000000000)
    reply = message('assistant', createdAt=1700000001000, responseAt=1700000002000)
    assert (await client.patch(url, json={'messages': [sent, reply]})).json()['last_message_at'] == reply['responseAt']
    changed = (await client.patch(url, json={'title': '新名称', 'draft': {'text': '新草稿', 'images': []}})).json()
    assert changed['last_message_at'] == reply['responseAt']
    legacy = controller.store.get(SUMMARY_KIND, body['id'])
    del legacy['last_message_at']
    controller.store.put(SUMMARY_KIND, legacy)
    assert (await client.get('/api/conversations')).json()['items'][0]['last_message_at'] == reply['responseAt']
    assert (await client.get(url)).json()['last_message_at'] == reply['responseAt']
    shortened = (await client.patch(url, json={'clear_messages': True, 'messages': [sent]})).json()
    assert shortened['last_message_at'] == sent['createdAt']
    assert (await client.patch(url, json={'clear_messages': True})).json()['last_message_at'] is None


async def test_history_order_changes_only_when_sending_not_draft_rename_view_or_output(client):
    first, second = conversation(), conversation()
    for body in (first, second):
        await client.post('/api/conversations', json=body)
    a, b = '/api/conversations/' + first['id'], '/api/conversations/' + second['id']
    await client.patch(a, json={'messages': [message(createdAt=1000)]})
    await client.patch(b, json={'messages': [message(createdAt=2000)]})
    async def order():
        return [item['id'] for item in (await client.get('/api/conversations')).json()['items']]
    assert await order() == [second['id'], first['id']]
    await client.patch(a, json={'activate': True, 'title': '改名', 'draft': {'text': '草稿', 'images': []}})
    await client.patch(a, json={'messages': [message(createdAt=1000, responseAt=9000, text='迟到完成')]})
    assert await order() == [second['id'], first['id']]
    retried = await client.patch(a, json={'clear_messages': True, 'messages': [message('retry', createdAt=10000)]})
    assert retried.json()['last_sent_at'] == 10000
    assert await order() == [first['id'], second['id']]


async def test_pin_and_group_reorder_persist_without_changing_message_times(client, controller):
    bodies = [conversation(title=title) for title in ('A', 'B', 'C', 'D')]
    for index, body in enumerate(bodies):
        await client.post('/api/conversations', json=body)
        await client.patch('/api/conversations/' + body['id'], json={'messages': [message(createdAt=index + 1)]})
    async def order():
        return [item['title'] for item in (await client.get('/api/conversations')).json()['items']]
    url = lambda index: '/api/conversations/' + bodies[index]['id']
    assert await order() == ['D', 'C', 'B', 'A']
    await client.patch(url(0), json={'pinned': True})
    await client.patch(url(2), json={'pinned': True})
    assert await order() == ['C', 'A', 'D', 'B']
    await client.patch(url(0), json={'move_before': bodies[2]['id']})
    assert await order() == ['A', 'C', 'D', 'B']
    await client.patch(url(1), json={'move_before': bodies[3]['id']})
    assert await order() == ['A', 'C', 'B', 'D']
    before = stored_rows(controller)
    assert (await client.patch(url(1), json={'move_before': bodies[0]['id']})).status_code == 400
    assert stored_rows(controller) == before
    await client.patch(url(3), json={'messages': [message(createdAt=100)]})
    assert await order() == ['A', 'C', 'D', 'B']
    await client.patch(url(2), json={'messages': [message(createdAt=200)]})
    assert await order() == ['A', 'C', 'D', 'B']
    await client.patch(url(0), json={'move_before': None})
    assert await order() == ['C', 'A', 'D', 'B']
    assert (await client.get(url(0))).json()['last_message_at'] == 1
    await client.patch(url(0), json={'pinned': False})
    assert await order() == ['C', 'A', 'D', 'B']


def search_result():
    return {'query': '中文 + 日本語', 'elapsed_ms': 1217,
            'sources': [{'id': 1, 'title': '来源', 'url': 'https://source.test/中文', 'snippet': '完整摘要'},
                        {'id': 2, 'title': '', 'url': 'http://empty.test/', 'snippet': ''}]}


async def test_web_search_settings_remain_compatible_with_old_settings_writes(client, controller):
    body = conversation()
    assert (await client.post('/api/conversations', json=body)).status_code == 200
    url = '/api/conversations/' + body['id']
    assert 'webSearch' not in (await client.get(url)).json()['settings']
    for enabled in (True, False):
        response = await client.patch(url, json={'settings': {**body['settings'], 'webSearch': enabled}})
        assert response.status_code == 200, response.text
        stored = controller.store.get(DETAIL_KIND, body['id'])
        assert stored['web_search'] is enabled and stored['settings'] == body['settings']
        assert (await client.get(url)).json()['settings']['webSearch'] is enabled
        # An old client only knows the legacy settings object and omits webSearch.
        legacy = {**body['settings'], 'system': '旧客户端更新'}
        assert (await client.patch(url, json={'settings': legacy})).status_code == 200
        assert (await client.get(url)).json()['settings'] == {**legacy, 'webSearch': enabled}
        # The old server also rewrites just detail.settings, preserving unknown top-level fields.
        stored = controller.store.get(DETAIL_KIND, body['id'])
        assert set(stored['settings']) <= {'system', 'stream', 'toolsDraft', 'temperature', 'maxTokens'}
        stored['settings']['toolsDraft'] = '[]'
        controller.store.put(DETAIL_KIND, stored)
        assert (await client.get(url)).json()['settings']['webSearch'] is enabled
        assert controller.store.get(DETAIL_KIND, body['id'])['web_search'] is enabled


async def test_search_snapshot_and_provenance_survive_database_reopen(tmp_path):
    root = tmp_path / 'search-history'
    body = conversation(settings={'system': '词', 'stream': True, 'toolsDraft': '', 'reasoningEffort': 'high', 'webSearch': True,
                                  'webSearchMode': 'agent', 'webSearchRounds': 10**30, 'toolConcurrency': 10**30})
    provenance = {**search_result(), 'query': '', 'mode': 'agent', 'queries': ['图中的建筑', '图中的建筑 历史'],
                  'images': [{'source_id': 1, 'title': '建筑', 'url': 'https://images.test/building.png', 'thumbnail_url': 'https://images.test/thumb.webp'},
                             {'source_id': 1, 'title': '建筑侧面', 'url': 'https://images.test/side.png'}],
                  'pages': [{'source_id': 1, 'url': 'https://source.test/redirected', 'title': '完整页面',
                             'content': '段落一\n    code', 'start_line': 1, 'end_line': 2, 'total_lines': 4, 'elapsed_ms': 30},
                            {'source_id': 1, 'url': 'https://source.test/redirected', 'title': '完整页面',
                             'content': '段落二\n段落三', 'start_line': 3, 'end_line': 4, 'total_lines': 4, 'elapsed_ms': 10}]}
    saved = message(webSearch=provenance)
    first = Controller(root, 'fixture')
    try:
        first.conversations.create(body)
        first.conversations.patch(body['id'], {'messages': [saved]})
    finally:
        await first.close()
    reopened = Controller(root, 'fixture')
    try:
        detail = reopened.conversations.get(body['id'])
        assert detail['settings'] == body['settings']
        assert detail['messages'] == [saved]
        stored = reopened.store.get(DETAIL_KIND, body['id'])
        assert stored['web_search'] is True
        assert stored['web_search_mode'] == 'agent' and stored['web_search_rounds'] == 10**30
        assert stored['tool_concurrency'] == 10**30
        assert not {'webSearch', 'webSearchMode', 'webSearchRounds', 'reasoningEffort', 'toolConcurrency'} & stored['settings'].keys()
        legacy = {'system': '旧客户端更新', 'stream': False, 'toolsDraft': ''}
        reopened.conversations.patch(body['id'], {'settings': legacy})
        assert reopened.conversations.get(body['id'])['settings'] == {**legacy, 'webSearch': True,
                                                                     'webSearchMode': 'agent', 'webSearchRounds': 10**30, 'toolConcurrency': 10**30}
        reopened.conversations.patch(body['id'], {'settings': {**legacy, 'webSearchMode': 'retrieve', 'webSearchRounds': 1}})
        assert reopened.conversations.get(body['id'])['settings']['webSearchMode'] == 'retrieve'
    finally:
        await reopened.close()


async def test_invalid_search_settings_or_provenance_reject_entire_patch(client, controller):
    body = conversation()
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    await client.patch(url, json={'messages': [message(webSearch=search_result())]})
    before = stored_rows(controller)
    result = search_result()
    source = result['sources'][0]
    malformed = [None, False, {}, {**result, 'query': 1},
                 {**result, 'elapsed_ms': -1}, {**result, 'elapsed_ms': True}, {**result, 'elapsed_ms': float('inf')},
                 {**result, 'sources': {}}, {**result, 'extra': True},
                 *[{**result, 'sources': [invalid]} for invalid in (None, {}, {**source, 'id': True},
                    {**source, 'id': 2}, {**source, 'title': None}, {**source, 'snippet': []},
                    {**source, 'url': 'javascript:alert(1)'}, {**source, 'url': 'file:///private'},
                    {**source, 'url': 'https://bad.test:bad'}, {**source, 'url': 'https://bad.test/a\nb'})],
                 {**result, 'sources': [source, source]}]
    patches = [{'messages': [message(webSearch=value)]} for value in malformed]
    patches.extend({'settings': {**body['settings'], 'webSearch': value}} for value in (None, 0, 'true', [], {}))
    patches.extend({'settings': {**body['settings'], 'webSearchMode': value}} for value in (None, 'auto', [], {}))
    patches.extend({'settings': {**body['settings'], 'webSearchRounds': value}} for value in (None, 0, -1, True, 1.5, '6'))
    patches.extend({'settings': {**body['settings'], 'toolConcurrency': value}} for value in (None, 0, -1, True, 1.5, '3'))
    page = {'source_id': 1, 'url': 'https://source.test/', 'title': '页面', 'content': '正文',
            'start_line': 1, 'end_line': 1, 'total_lines': 3, 'elapsed_ms': 10}
    bad_pages = [None, {}, {**page, 'source_id': 0}, {**page, 'source_id': 3}, {**page, 'source_id': True},
                 {**page, 'url': 'file:///private'}, {**page, 'start_line': 0}, {**page, 'end_line': 4},
                 {**page, 'start_line': 2}, {**page, 'total_lines': 0}, {**page, 'content': None}, {**page, 'elapsed_ms': -1}]
    bad_results = [{**result, 'pages': [value]} for value in bad_pages]
    bad_results.extend([{**result, 'mode': 'auto'}, {**result, 'queries': 'query'}, {**result, 'queries': [1]}, {**result, 'pages': {}}])
    image = {'source_id': 1, 'title': '图片', 'url': 'https://images.test/image.png'}
    bad_images = [None, {}, {**image, 'source_id': 0}, {**image, 'source_id': True}, {**image, 'source_id': 3},
                  {**image, 'title': None}, {**image, 'url': 'file:///private'}, {**image, 'thumbnail_url': 'javascript:alert(1)'},
                  {**image, 'unexpected': True}]
    bad_results.extend([{**result, 'images': {}}, *[{**result, 'images': [value]} for value in bad_images]])
    patches.extend({'messages': [message(webSearch=value)]} for value in bad_results)
    for invalid in patches:
        # Deliberately exercise JSON's non-finite values as well as ordinary JSON inputs.
        response = await client.patch(url, content=json.dumps({'title': '不得写入', 'clear_messages': True, **invalid}),
                                      headers={'Content-Type': 'application/json'})
        assert response.status_code == 400, (invalid, response.text)
        assert stored_rows(controller) == before


def compaction(messages, summary='保留精确数值 42、路径和未完成任务', **changes):
    return {'summary': summary, 'throughMessageId': messages[-1]['id'],
            'messageRevisions': [{'id': value['id'], 'revision': value.get('contextRevision', 0)} for value in messages],
            'createdAt': 1730000000300, **changes}


async def test_compaction_settings_and_tool_results_survive_reopen_without_removing_history(tmp_path):
    root = tmp_path / 'compaction-reopen'
    body = conversation(settings={'system': '原系统提示', 'stream': True, 'toolsDraft': '', 'basicTools': True,
                                  'autoCompact': False, 'toolDirectory': 'D:\\工具资料'})
    values = [message('user', role='user', text='精确保留 42 项', images=[image()]),
              message('reply', text='已读取文件，第二步待完成', contextRevision=3,
                      toolResults=[{'call': {'id': 'call', 'function': {'name': 'read_file', 'arguments': '{"path":"例子.txt"}'},
                                            'providerExtra': True}, 'content': '原始工具结果', 'future': {'keep': True}}])]
    snapshot = compaction(values)
    first = Controller(root, 'fixture')
    try:
        first.conversations.create(body)
        first.conversations.patch(body['id'], {'messages': values, 'compaction': snapshot})
        saved = first.conversations.get(body['id'])
        assert saved['messages'] == values and saved['compaction'] == snapshot
        stored = first.store.get(DETAIL_KIND, body['id'])
        assert stored['settings'] == {'system': '原系统提示', 'stream': True, 'toolsDraft': ''}
        assert stored['basic_tools'] is True and stored['auto_compact'] is False
        assert stored['tool_directory'] == 'D:\\工具资料'
    finally:
        await first.close()
    reopened = Controller(root, 'fixture')
    try:
        assert reopened.conversations.get(body['id']) == saved
        # Legacy settings writes preserve new top-level settings and the checkpoint.
        legacy = {'system': '更新系统提示', 'stream': False, 'toolsDraft': '[]'}
        reopened.conversations.patch(body['id'], {'settings': legacy})
        restored = reopened.conversations.get(body['id'])
        assert restored['settings'] == {**legacy, 'basicTools': True, 'autoCompact': False, 'toolDirectory': 'D:\\工具资料'}
        assert restored['compaction'] == snapshot and restored['messages'] == values
        reopened.conversations.patch(body['id'], {'messages': [message('user', role='user', text='编辑已压缩正文')]})
        assert 'compaction' not in reopened.conversations.get(body['id'])
    finally:
        await reopened.close()


async def test_compacted_tail_streaming_stays_incremental_and_only_covered_edits_invalidate(client, controller):
    body = conversation()
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    prefix = [message('old-user', role='user', images=[image()]), message('old-reply')]
    snapshot = compaction(prefix)
    assert (await client.patch(url, json={'messages': prefix, 'compaction': snapshot})).status_code == 200
    statements = []
    controller.store.connection.set_trace_callback(statements.append)
    tail = message('tail', status='generating', text='流式内容')
    try:
        assert (await client.patch(url, json={'messages': [tail]})).status_code == 200
        assert (await client.patch(url, json={'messages': [{**tail, 'text': '流式内容增加', 'status': 'completed'}]})).status_code == 200
        reads = [query for query in statements if query.lstrip().upper().startswith('SELECT')]
        assert not any(DETAIL_KIND in query or (MESSAGE_PREFIX in query and 'SELECT body' in query) for query in reads)
    finally:
        controller.store.connection.set_trace_callback(None)
    assert (await client.get(url)).json()['compaction'] == snapshot
    # Display-only usage/timing changes must not lose an otherwise valid checkpoint.
    assert (await client.patch(url, json={'messages': [{**prefix[1], 'usage': {'prompt_tokens': 12}, 'elapsed': 2}]})).status_code == 200
    assert (await client.get(url)).json()['compaction'] == snapshot
    # Old clients do not know revisions; actual changed context still invalidates.
    assert (await client.patch(url, json={'messages': [{**prefix[0], 'text': '修改要求'}]})).status_code == 200
    detail = (await client.get(url)).json()
    assert 'compaction' not in detail
    assert detail['messages'][0]['images'] == prefix[0]['images']


async def test_retry_after_boundary_preserves_compaction_but_prefix_deletion_and_reorder_clear_it(client):
    body = conversation()
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    prefix = [message('user', role='user'), message('reply')]
    tail = [message('next-user', role='user'), message('next-reply')]
    snapshot = compaction(prefix)
    await client.patch(url, json={'messages': [*prefix, *tail], 'compaction': snapshot})
    # Replacement is the retry/edit persistence entry point; unchanged covered IDs
    # and contents remain valid even though the later branch is replaced.
    assert (await client.patch(url, json={'clear_messages': True, 'messages': [*prefix, *tail[:1], message('replacement')]})).status_code == 200
    assert (await client.get(url)).json()['compaction'] == snapshot
    await client.patch(url, json={'clear_messages': True, 'messages': prefix[:1]})
    assert 'compaction' not in (await client.get(url)).json()
    await client.patch(url, json={'clear_messages': True, 'messages': prefix, 'compaction': snapshot})
    await client.patch(url, json={'clear_messages': True, 'messages': list(reversed(prefix))})
    assert 'compaction' not in (await client.get(url)).json()
    await client.patch(url, json={'clear_messages': True, 'messages': prefix, 'compaction': snapshot})
    await client.patch(url, json={'clear_messages': True})
    assert (await client.get(url)).json()['messages'] == []
    assert 'compaction' not in (await client.get(url)).json()


async def test_second_compaction_and_explicit_clear_are_atomic_metadata_updates(client, controller):
    body = conversation()
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    values = [message('u1', role='user'), message('a1'), message('u2', role='user'), message('a2')]
    first, second = compaction(values[:2]), compaction(values, '新的合并摘要')
    await client.patch(url, json={'messages': values, 'compaction': first})
    before = (await client.get(url)).json()
    assert (await client.patch(url, json={'compaction': second})).status_code == 200
    saved = (await client.get(url)).json()
    assert saved['compaction'] == second and saved['messages'] == values
    assert saved['last_message_at'] == before['last_message_at'] and saved['message_count'] == 4
    await client.patch(url, json={'compaction': None})
    assert 'compaction' not in (await client.get(url)).json()
    assert controller.store.all(MESSAGE_PREFIX + body['id']) == values


async def test_invalid_compaction_revision_settings_and_tool_results_reject_every_change(client, controller):
    body = conversation()
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    values = [message('u', role='user'), message('a')]
    snapshot = compaction(values)
    await client.patch(url, json={'messages': values, 'compaction': snapshot, 'activate': True})
    before = stored_rows(controller)
    malformed = [{}, False, {'summary': ''}, {**snapshot, 'summary': '  '}, {**snapshot, 'createdAt': True},
                 {**snapshot, 'throughMessageId': 'missing'}, {**snapshot, 'messageRevisions': []},
                 {**snapshot, 'messageRevisions': [{'id': 'a', 'revision': 0}]},
                 {**snapshot, 'messageRevisions': [{'id': 'u', 'revision': 1}, {'id': 'a', 'revision': 0}]},
                 {**snapshot, 'messageRevisions': [{'id': 'u', 'revision': -1}, {'id': 'a', 'revision': 0}]},
                 {**snapshot, 'messageRevisions': [{'id': 'u', 'revision': True}, {'id': 'a', 'revision': 0}]}]
    patches = [{'compaction': value} for value in malformed]
    patches.extend({'messages': [message(contextRevision=value)]} for value in (-1, True, 1.5, '2'))
    for setting in ('basicTools', 'autoCompact'):
        patches.extend({'settings': {**body['settings'], setting: value}} for value in (None, 1, 'true', {}))
    patches.extend({'settings': {**body['settings'], 'toolDirectory': value}} for value in (None, 12, []))
    patches.extend({'messages': [message(toolResults=value)]} for value in ({}, [None], [{}], [{'call': {}, 'content': ''}],
                    [{'call': {'function': {'name': 'read_file', 'arguments': '{}'}}, 'content': 1}]))
    for invalid in patches:
        response = await client.patch(url, json={'title': '不得更新', **invalid})
        assert response.status_code == 400, (invalid, response.text)
        assert stored_rows(controller) == before


async def test_compaction_cache_follows_only_successful_transaction(controller):
    body = conversation()
    controller.conversations.create(body)
    values = [message('u', role='user'), message('a')]
    first = compaction(values)
    controller.conversations.patch(body['id'], {'messages': values, 'compaction': first})
    before = stored_rows(controller)
    controller.store.connection.execute("CREATE TEMP TRIGGER fail_checkpoint BEFORE UPDATE ON records WHEN OLD.kind='conversation_details' BEGIN SELECT RAISE(ABORT,'checkpoint write failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='checkpoint write failure'):
        controller.conversations.patch(body['id'], {'messages': [{**values[0], 'text': '应回滚'}]})
    assert stored_rows(controller) == before
    assert controller.conversations._compactions[body['id']] == first


async def test_activity_order_and_exact_instance_persist_and_interrupt(client, controller):
    body = conversation(selection={'profile_id': 'p', 'model_id': 'm', 'alias': 'model--2', 'instance_id': 'second'})
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    activity = [
        {'id': 'r1', 'kind': 'reasoning', 'text': '先读取'},
        {'id': 't1', 'kind': 'tool', 'call': {'id': 'read', 'function': {'name': 'read_file', 'arguments': '{"path":"a.txt"}'}}, 'status': 'completed', 'result': '{"content":"42"}'},
        {'id': 'r2', 'kind': 'reasoning', 'text': '再写入'},
        {'id': 't2', 'kind': 'tool', 'call': {'id': 'write', 'function': {'name': 'write_file', 'arguments': '{}'}}, 'status': 'running'},
    ]
    response = await client.patch(url, json={'messages': [message(activity=activity, status='generating')]})
    assert response.status_code == 200, response.text
    saved = (await client.get(url)).json()
    assert saved['messages'][0]['activity'] == activity
    assert saved['selection']['instance_id'] == 'second'
    controller.conversations.interrupt_generating()
    restored = (await client.get(url)).json()['messages'][0]
    assert restored['status'] == 'interrupted'
    assert restored['activity'] == [*activity[:3], {**activity[3], 'status': 'cancelled'}]
    for invalid in ([activity[0], activity[0]], [{**activity[1], 'status': 'mystery'}], [{**activity[1], 'result': {}}]):
        assert (await client.patch(url, json={'messages': [message(activity=invalid)]})).status_code == 400
    assert (await client.get(url)).json()['messages'][0] == restored


def turn_fixture():
    requests = [
        {'id': 'call-1', 'index': 1, 'kind': 'compaction', 'status': 'completed', 'startedAt': 1000,
         'endedAt': 1500, 'elapsed': .5, 'labels': ['自动压缩'], 'usage': {'prompt_tokens': 23, 'completion_tokens': 3},
         'timings': {'predicted_n': 3, 'predicted_ms': 300}},
        {'id': 'call-2', 'index': 2, 'kind': 'generation', 'status': 'running', 'startedAt': 1700,
         'labels': ['读取文件'], 'usage': {'prompt_tokens': 25, 'completion_tokens': 2}},
    ]
    turn = {'startedAt': 900, 'elapsed': 1.1, 'status': 'running', 'requests': 2,
            'input': {'value': 48, 'reported': 2}, 'output': {'value': 5, 'reported': 2},
            'cached': {'value': None, 'reported': 0}, 'reasoning': {'value': None, 'reported': 0},
            'decode': {'value': 10, 'tokens': 3, 'seconds': .3, 'reported': 1},
            'prefill': {'value': None, 'tokens': 0, 'seconds': 0, 'reported': 0}, 'e2e': 5 / 1.1}
    return requests, turn


async def test_turn_ledger_survives_history_without_replacing_context_usage(client, controller):
    body = conversation()
    assert (await client.post('/api/conversations', json=body)).status_code == 200
    requests, turn = turn_fixture()
    activity = [{'id': 'read-event', 'kind': 'tool', 'call': {'id': 'read', 'function': {'name': 'read_file', 'arguments': '{"path":"a.txt"}'}},
                 'status': 'running', 'requestId': 'call-2', 'startedAt': 1900, 'elapsed': .1}]
    rich = message(status='generating', usage={'prompt_tokens': 25, 'completion_tokens': 2}, requests=requests, turn=turn, activity=activity)
    url = '/api/conversations/' + body['id']
    result = await client.patch(url, json={'messages': [rich]})
    assert result.status_code == 200, result.text
    assert (await client.get(url)).json()['messages'] == [rich]
    controller.conversations.interrupt_generating()
    restored = (await client.get(url)).json()['messages'][0]
    assert restored['requests'][0] == requests[0]
    assert restored['requests'][1] == {**requests[1], 'status': 'incomplete'}
    assert restored['turn'] == {**turn, 'status': 'incomplete'}
    assert restored['usage'] == {'prompt_tokens': 25, 'completion_tokens': 2}
    assert restored['activity'][0] == {**activity[0], 'status': 'cancelled'}
    before = stored_rows(controller)
    for invalid in ({'requests': {}}, {'requests': [{**requests[0], 'labels': '推测意图'}]},
                    {'turn': {**turn, 'elapsed': -1}}, {'activity': [{**activity[0], 'elapsed': '0.1'}]},
                    {'resources': [{'id': 'incomplete'}]}):
        assert (await client.patch(url, json={'messages': [message(**invalid)]})).status_code == 400
        assert stored_rows(controller) == before


async def test_resource_history_replace_clear_and_delete_preserve_user_files(client, controller, tmp_path):
    from modloader.local_tools import LocalToolError
    directory = tmp_path / 'documents'
    directory.mkdir()
    (directory / 'keep.md').write_text('保留文档', encoding='utf-8')
    (directory / 'remove.md').write_text('移除引用，保留文件', encoding='utf-8')
    body = conversation(settings={'system': '', 'stream': True, 'toolsDraft': '', 'toolDirectory': str(directory)})
    assert (await client.post('/api/conversations', json=body)).status_code == 200
    service = controller.local_resources
    async def attach(name):
        response = await client.post('/api/chat-tools/attach', json={'conversation_id': body['id'], 'directory': str(directory), 'path': name})
        assert response.status_code == 200, response.text
        return response.json()['resource']
    keep, remove = await attach('keep.md'), await attach('remove.md')
    first, second = message('first', resources=[keep]), message('second', resources=[remove])
    url = '/api/conversations/' + body['id']
    assert (await client.patch(url, json={'messages': [first, second]})).status_code == 200
    assert (await client.get(url)).json()['messages'] == [first, second]
    # replaceMessages also powers selective deletion: retained messages keep references.
    assert (await client.patch(url, json={'clear_messages': True, 'messages': [first]})).status_code == 200
    assert service.metadata(keep['id']) == keep
    with pytest.raises(LocalToolError, match='不存在'):
        service.metadata(remove['id'])
    assert (await client.patch(url, json={'clear_messages': True, 'messages': [], 'title': '新会话'})).status_code == 200
    with pytest.raises(LocalToolError, match='不存在'):
        service.metadata(keep['id'])
    retained = (await client.get(url)).json()
    assert retained['title'] == '新会话' and retained['settings']['toolDirectory'] == str(directory)
    final = service.attach(body['id'], str(directory), 'keep.md')['resource']
    assert (await client.delete(url)).status_code == 204
    with pytest.raises(LocalToolError, match='不存在'):
        service.metadata(final['id'])
    assert (directory / 'keep.md').read_text(encoding='utf-8') == '保留文档'
    assert (directory / 'remove.md').read_text(encoding='utf-8') == '移除引用，保留文件'


async def test_document_draft_and_messages_survive_reopen_with_old_fields_unchanged(tmp_path):
    root = tmp_path / 'document-reopen'
    docs = [document(), document(name='empty.txt', size=0, text='')]
    body = conversation(draft={'text': '保留草稿', 'images': [image()], 'documents': docs})
    values = [message('document-only', role='user', text='', documents=docs), message('legacy')]
    first = Controller(root, 'fixture')
    try:
        first.conversations.create(body)
        first.conversations.patch(body['id'], {'messages': values})
        saved = first.conversations.get(body['id'])
        assert saved['draft'] == body['draft'] and saved['messages'] == values
        assert saved['settings'] == body['settings']
        assert 'documents' not in first.conversations.list()['items'][0]
        assert 'documents' not in saved['messages'][1]
    finally:
        await first.close()
    reopened = Controller(root, 'fixture')
    try:
        assert reopened.conversations.get(body['id']) == saved
        legacy = conversation()
        reopened.conversations.create(legacy)
        assert reopened.conversations.get(legacy['id'])['draft'] == {'text': '', 'images': []}
    finally:
        await reopened.close()


async def test_invalid_document_payloads_reject_atomic_create_and_patch(client, controller):
    body = conversation()
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    before = stored_rows(controller)
    invalid = [None, {}, [None], [{}], [document(id='')], [document(name=' ')], [document(text=None)],
               [document(size=-1)], [document(size=True)], [document(size=1.5)], [document(extra=True)]]
    for docs in invalid:
        draft = {'text': '不得写入', 'images': [], 'documents': docs}
        response = await client.post('/api/conversations', json=conversation(draft=draft))
        assert response.status_code == 400, response.text
        for changes in ({'draft': draft}, {'messages': [message(documents=docs)]}):
            response = await client.patch(url, json={'title': '不得写入', 'clear_messages': True, **changes})
            assert response.status_code == 400, response.text
            assert stored_rows(controller) == before


@pytest.mark.parametrize('change', ['name', 'text', 'remove', 'revision'])
async def test_compaction_tracks_document_content_but_not_display_metadata(client, change):
    body = conversation()
    await client.post('/api/conversations', json=body)
    url = '/api/conversations/' + body['id']
    doc = document()
    prefix = [message('user', role='user', text='', documents=[doc]), message('reply')]
    snapshot = compaction(prefix)
    assert (await client.patch(url, json={'messages': prefix, 'compaction': snapshot})).status_code == 200
    # ID and original byte size do not change the text sent to the model.
    metadata = {**prefix[0], 'documents': [{**doc, 'id': 'reimported', 'size': 1024}]}
    assert (await client.patch(url, json={'messages': [metadata]})).status_code == 200
    assert (await client.get(url)).json()['compaction'] == snapshot
    if change == 'remove':
        changed = {**metadata, 'documents': []}
    elif change == 'revision':
        changed = {**metadata, 'contextRevision': 1}
    else:
        changed = {**metadata, 'documents': [{**doc, change: '改变模型上下文'}]}
    # Legacy clients can omit contextRevision: document text/name still invalidate.
    assert (await client.patch(url, json={'messages': [changed]})).status_code == 200
    assert 'compaction' not in (await client.get(url)).json()
