"""Management-only SearXNG protocol and cancellation; no live data or model."""
import asyncio
import json
from urllib.parse import parse_qs

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.models import LaunchSpec, RuntimeInstance
from modloader.web_search import DEFAULT_SEARXNG_URL, SEARCH_TIMEOUT_S


def test_public_searxng_default_only_targets_the_local_machine():
    assert DEFAULT_SEARXNG_URL == 'http://127.0.0.1:8888'


@pytest.fixture
async def controller(tmp_path):
    value = Controller(tmp_path / 'data', 'web-search-fixture')
    value.settings['proxy_mode'] = 'direct'
    try:
        yield value
    finally:
        value.runtime.instances.clear()
        await value.close()


@pytest.fixture
async def client(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer web-search-fixture'}) as value:
        yield value


def mock_searxng(monkeypatch, handler):
    original = httpx.AsyncClient
    options = []

    def factory(**kwargs):
        options.append(kwargs)
        # The proxy option is asserted separately; a real proxy mount would bypass MockTransport.
        return original(transport=httpx.MockTransport(handler), **{key: value for key, value in kwargs.items() if key != 'proxy'})

    monkeypatch.setattr('modloader.web_search.httpx.AsyncClient', factory)
    return options


@pytest.mark.parametrize('streaming', [False, True])
async def test_form_search_preserves_all_safe_sources_and_unicode(client, controller, monkeypatch, streaming):
    controller.settings.update(searxng_url='http://searxng.test/prefix/', proxy_mode='manual', proxy_url='http://proxy.test:17890')
    calls = []
    long_text = '长摘要' * 3000

    async def backend(request):
        calls.append(request)
        assert request.method == 'POST' and str(request.url) == 'http://searxng.test/prefix/search'
        assert request.headers['content-type'].startswith('application/x-www-form-urlencoded')
        assert parse_qs(request.content.decode()) == {'q': ['今天的中文 + C++ & 日本語'], 'format': ['json'], 'categories': ['general']}
        return httpx.Response(200, json={'results': [
            {'url': 'javascript:alert(1)', 'title': '不得显示'}, {'url': 'data:text/html,unsafe'},
            {'url': ''}, {'url': 'ftp://files.test/item'}, {'url': 'https://bad.test:bad'},
            {'url': 'https://bad.test/a\nb'}, {'url': None}, None,
            {'url': ' https://example.test/中文 ', 'title': '<b>中文</b> &amp; 日本語',
             'content': '<p>第一段&nbsp;<em>摘要</em></p><script>不保留脚本</script><p>第二段\n 内容</p>'},
            {'url': 'https://example.test/中文#duplicate', 'title': '同一页'},
            {'url': 'http://empty.test/missing'},
            *[{'url': f'https://many.test/{index}', 'title': f'第 {index} 条', 'content': long_text} for index in range(35)],
        ]})

    options = mock_searxng(monkeypatch, backend)
    response = await client.post('/api/web-search', json={'query': '  今天的中文 + C++ & 日本語  '},
                                 headers={'Accept': 'application/x-ndjson'} if streaming else {})
    assert response.status_code == 200, response.text
    if streaming:
        assert response.headers['content-type'].startswith('application/x-ndjson')
        events = [json.loads(line) for line in response.text.splitlines()]
        assert any(event['type'] == 'progress' and event['stage'] == 'web_search' for event in events)
        assert events[-1]['type'] == 'result'
        result = events[-1]['value']
    else:
        result = response.json()
    assert result['query'] == '今天的中文 + C++ & 日本語'
    assert type(result['elapsed_ms']) is int and result['elapsed_ms'] >= 0
    assert len(result['sources']) == 37
    assert [source['id'] for source in result['sources']] == list(range(1, 38))
    assert result['sources'][0] == {'id': 1, 'url': 'https://example.test/中文', 'title': '中文 & 日本語', 'snippet': '第一段 摘要 第二段 内容'}
    assert result['sources'][1] == {'id': 2, 'url': 'http://empty.test/missing', 'title': '', 'snippet': ''}
    assert result['sources'][-1]['snippet'] == long_text
    assert len(calls) == 1
    assert options == [{'proxy': 'http://proxy.test:17890', 'trust_env': False, 'timeout': SEARCH_TIMEOUT_S}]


async def test_settings_url_is_validated_without_changing_legacy_patch_contract(client, controller):
    assert controller.safe_settings()['searxng_url'] == DEFAULT_SEARXNG_URL
    assert controller.store.get('settings', 'main') is None  # Defaults do not rewrite a shared database.
    saved = await client.patch('/api/settings', json={'searxng_url': 'https://search.test/searx', 'font_scale': 1.2})
    assert saved.status_code == 200 and saved.json()['searxng_url'] == 'https://search.test/searx'
    legacy = await client.patch('/api/settings', json={'close_to_tray': False, 'unsupported_old_field': 'ignored'})
    assert legacy.status_code == 200 and legacy.json()['searxng_url'] == 'https://search.test/searx'
    assert legacy.json()['close_to_tray'] is False
    before = controller.store.get('settings', 'main')
    for invalid in ('', 'ftp://search.test', '//search.test', 'http://', 'http://host.test:bad', 'http://host.test:65536',
                    'https://bad.test/a\nb', 'https://bad.test/a b', None, False, 123, [], {}):
        response = await client.patch('/api/settings', json={'searxng_url': invalid, 'font_scale': 2})
        assert response.status_code == 400, (invalid, response.text)
        assert controller.settings == before
        assert controller.store.get('settings', 'main') == before


@pytest.mark.parametrize('streaming', [False, True])
async def test_image_search_returns_original_urls_and_multiple_images_per_source(client, monkeypatch, streaming):
    def backend(request):
        assert parse_qs(request.content.decode()) == {'q': ['!bdi OpenAI logo'], 'format': ['json'], 'categories': ['images']}
        return httpx.Response(200, json={'results': [
            {'url': 'https://news.test/logo', 'title': '<b>标志</b>', 'img_src': 'https://cdn.test/first.png',
             'thumbnail_src': 'https://thumb.test/first.webp', 'content': '<p>原出处</p>'},
            {'url': 'https://news.test/logo#details', 'title': '第二张', 'img_src': 'https://cdn.test/second.png',
             'thumbnail_src': 'javascript:alert(1)'},
            {'url': 'https://news.test/logo', 'img_src': 'https://cdn.test/first.png'},
            {'url': 'https://news.test/no-image'},
            {'url': 'https://news.test/unsafe', 'img_src': 'file:///private.png'},
            {'url': 'javascript:alert(1)', 'img_src': 'https://cdn.test/unsafe.png'},
            {'url': 'https://news.test/bad', 'img_src': 10},
            *[{'url': f'https://news.test/{index}', 'img_src': f'https://cdn.test/{index}.jpg'} for index in range(35)],
        ]})

    mock_searxng(monkeypatch, backend)
    response = await client.post('/api/web-search', json={'query': ' OpenAI logo ', 'category': 'images'},
                                 headers={'Accept': 'application/x-ndjson'} if streaming else {})
    assert response.status_code == 200, response.text
    result = json.loads(response.text.splitlines()[-1])['value'] if streaming else response.json()
    assert result['query'] == 'OpenAI logo'
    assert len(result['sources']) == 36 and len(result['images']) == 37
    assert result['images'][:2] == [
        {'source_id': 1, 'title': '标志', 'url': 'https://cdn.test/first.png', 'thumbnail_url': 'https://thumb.test/first.webp'},
        {'source_id': 1, 'title': '第二张', 'url': 'https://cdn.test/second.png'},
    ]
    assert result['images'][-1]['source_id'] == 36


async def test_image_engine_setting_and_empty_default_are_explicit(client, controller, monkeypatch):
    assert controller.safe_settings()['searxng_image_engine'] == 'bdi'
    assert (await client.patch('/api/settings', json={'searxng_image_engine': ''})).status_code == 200
    forms = []
    def backend(request):
        forms.append(parse_qs(request.content.decode()))
        return httpx.Response(200, json={'results': [{'url': 'https://source.test/', 'img_src': 'https://image.test/1.png'}]})
    mock_searxng(monkeypatch, backend)
    assert (await client.post('/api/web-search', json={'query': '图片', 'category': 'images'})).status_code == 200
    assert forms[-1]['q'] == ['图片']
    assert (await client.patch('/api/settings', json={'searxng_image_engine': 'goi'})).status_code == 200
    assert (await client.post('/api/web-search', json={'query': '图片', 'category': 'images'})).status_code == 200
    assert forms[-1]['q'] == ['!goi 图片']
    for invalid in (None, False, [], 'bdi goi', '!bdi', 'bdi\n'):
        assert (await client.patch('/api/settings', json={'searxng_image_engine': invalid})).status_code == 400
    assert controller.settings['searxng_image_engine'] == 'goi'
    for invalid in ('video', None, [], 1):
        assert (await client.post('/api/web-search', json={'query': '图片', 'category': invalid})).status_code == 400
    assert len(forms) == 2


async def test_image_engine_failures_report_engine_error_without_retry(client, monkeypatch):
    calls = []
    def backend(request):
        calls.append(request)
        return httpx.Response(200, json={'results': [], 'unresponsive_engines': [['baidu images', '超时']]})
    mock_searxng(monkeypatch, backend)
    response = await client.post('/api/web-search', json={'query': '图片', 'category': 'images'})
    assert response.status_code == 502
    assert 'baidu images：超时' in response.json()['detail']
    assert len(calls) == 1


@pytest.mark.parametrize('body', [{}, {'query': ''}, {'query': '  '}, {'query': None}, {'query': False}, {'query': 123}, {'query': []}])
async def test_blank_or_nonstring_query_never_reaches_network(client, monkeypatch, body):
    def unexpected(request):
        pytest.fail('Invalid query reached SearXNG')
    mock_searxng(monkeypatch, unexpected)
    response = await client.post('/api/web-search', json=body)
    assert response.status_code == 400 and '关键词' in response.json()['detail']


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize(('failure', 'status', 'detail'), [
    ('forbidden', 502, 'search.formats'), ('upstream', 502, 'HTTP 500'), ('redirect', 502, 'HTTP 302'),
    ('html', 502, '有效 JSON'), ('wrong_object', 502, 'results 数组'), ('wrong_results', 502, 'results 数组'),
    ('empty', 422, '调整关键词'), ('unsafe', 422, '调整关键词'),
    ('timeout', 504, '超时'), ('connection', 502, '无法连接'),
])
async def test_search_failures_are_explicit_and_do_not_retry(client, monkeypatch, streaming, failure, status, detail):
    calls = []

    def backend(request):
        calls.append(request)
        if failure == 'timeout':
            raise httpx.ReadTimeout('fixture timeout', request=request)
        if failure == 'connection':
            raise httpx.ConnectError('fixture connection', request=request)
        if failure in ('forbidden', 'upstream', 'redirect'):
            return httpx.Response({'forbidden': 403, 'upstream': 500, 'redirect': 302}[failure],
                                  headers={'location': 'https://other.test/search'})
        if failure == 'html':
            return httpx.Response(200, text='<html>search</html>')
        return httpx.Response(200, json={'wrong_object': [], 'wrong_results': {'results': {}},
                                          'empty': {'results': []}, 'unsafe': {'results': [{'url': 'javascript:alert(1)'}]}}[failure])

    mock_searxng(monkeypatch, backend)
    response = await client.post('/api/web-search', json={'query': '查询'},
                                 headers={'Accept': 'application/x-ndjson'} if streaming else {})
    assert len(calls) == 1
    if streaming:
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['type'] == 'error' and events[-1]['status'] == status
        assert detail in events[-1]['detail']
        assert not any(event['type'] == 'result' for event in events)
    else:
        assert response.status_code == status and detail in response.json()['detail']


@pytest.mark.parametrize('streaming', [False, True])
async def test_disconnect_cancels_request_and_closes_search_client(controller, monkeypatch, streaming):
    started, cancelled, closed, disconnect = (asyncio.Event() for _ in range(4))

    class PendingSearch(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def aclose(self):
            closed.set()

    original = httpx.AsyncClient
    monkeypatch.setattr('modloader.web_search.httpx.AsyncClient', lambda **options: original(transport=PendingSearch(), **options))
    delivered = False
    sent = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': b'{"query":"pending"}', 'more_body': False}
        await disconnect.wait()
        return {'type': 'http.disconnect'}

    async def send(event):
        sent.append(event)

    headers = [(b'host', b'testserver'), (b'authorization', b'Bearer web-search-fixture'), (b'content-type', b'application/json')]
    if streaming:
        headers.append((b'accept', b'application/x-ndjson'))
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'http_version': '1.1',
             'method': 'POST', 'scheme': 'http', 'path': '/api/web-search', 'raw_path': b'/api/web-search',
             'query_string': b'', 'headers': headers, 'client': ('127.0.0.1', 1), 'server': ('testserver', 80)}
    task = asyncio.create_task(build_app(controller)(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), 2)
        disconnect.set()
        await asyncio.wait_for(task, 2)
        assert cancelled.is_set() and closed.is_set()
        assert not any(b'"type": "result"' in event.get('body', b'') for event in sent)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(('path', 'body'), [('/api/web-search', {'query': 'test'}), ('/api/web-fetch', {'url': 'http://public.test/'})])
async def test_search_requires_management_token_and_is_absent_from_public_gateway(controller, monkeypatch, path, body):
    original = httpx.AsyncClient
    app, public = build_app(controller), build_app(controller, public=True)

    def unexpected(request):
        pytest.fail('Unauthorized request reached SearXNG')

    async with original(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as management, \
               original(transport=httpx.ASGITransport(app=public), base_url='http://testserver',
                        headers={'Authorization': 'Bearer ' + controller.api_key}) as gateway:
        mock_searxng(monkeypatch, unexpected)
        for headers in ({}, {'Authorization': 'Bearer ' + controller.api_key}):
            assert (await management.post(path, json=body, headers=headers)).status_code == 401
        assert (await gateway.post(path, json=body)).status_code == 404


async def test_ordinary_chat_gateway_never_searches_even_with_search_enabled_history(controller, tmp_path, monkeypatch):
    import uuid
    controller.settings['web_search_provider'] = 'openai'
    controller.vault.set('search_openai', 'fixture-unused-key')
    controller.conversations.create({'id': str(uuid.uuid4()), 'title': '启用搜索的历史', 'selection': None,
                                     'draft': {'text': '', 'images': []},
                                     'settings': {'system': '', 'stream': True, 'toolsDraft': '', 'webSearch': True}})
    controller.store.put('profiles', {'id': 'profile', 'generation': {}})
    instance = RuntimeInstance(id='fixture', model_id='model', profile_id='profile', profile_revision=1, alias='fixture',
                               backend='llama_cpp', runtime_id='runtime', state='ready', port=12345, started_at='', capabilities={'input': ['text']})
    controller.runtime.instances[instance.id] = instance
    controller.runtime.specs[instance.id] = LaunchSpec(argv=[], cwd=str(tmp_path), model_name='backend-model')
    controller.runtime.requests[instance.id] = set()
    original = httpx.AsyncClient
    requests = []

    def backend(request):
        assert request.url.host == '127.0.0.1' and request.url.path == '/v1/chat/completions'
        payload = json.loads(request.content)
        requests.append(payload)
        result = {'choices': [{'message': {'content': '回答'}, 'finish_reason': 'stop'}]}
        if payload.get('stream'):
            return httpx.Response(200, content=('data: ' + json.dumps(result) + '\n\ndata: [DONE]\n\n').encode(),
                                  headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json=result)

    async with original(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                        headers={'Authorization': 'Bearer web-search-fixture'}) as management, \
               original(transport=httpx.ASGITransport(app=build_app(controller, public=True)), base_url='http://testserver',
                        headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
        monkeypatch.setattr('modloader.app.httpx.AsyncClient', lambda **options: original(transport=httpx.MockTransport(backend), **options))
        for api, path in ((management, '/api/chat'), (public, '/v1/chat/completions')):
            for stream in (False, True):
                payload = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream}
                response = await api.post(path, json=payload)
                assert response.status_code == 200, response.text
                assert requests[-1] == {**payload, 'model': 'backend-model',
                                       **({'stream_options': {'include_usage': True}, 'timings_per_token': True,
                                           'return_progress': True} if stream else {})}
    assert len(requests) == 4 and instance.active_requests == 0
