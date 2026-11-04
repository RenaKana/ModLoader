"""Provider wire contracts, independent selection, credentials and cancellation; no live API calls."""
import asyncio
import json
from urllib.parse import parse_qs

import httpx
import pytest

from modloader.app import Controller, build_app
from modloader.network_services import DEFAULT_NETWORK_SETTINGS, WEB_CREDENTIAL_KEYS
from modloader.secrets import Vault
from modloader.web_search import validate_search_result


@pytest.fixture
async def controller(tmp_path):
    value = Controller(tmp_path / 'data', 'network-services-fixture')
    value.settings['proxy_mode'] = 'direct'
    try:
        yield value
    finally:
        await value.close()


@pytest.fixture
async def client(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer network-services-fixture'}) as value:
        yield value


def mock_api(monkeypatch, handler):
    original = httpx.AsyncClient
    options = []

    def factory(**kwargs):
        options.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **{key: value for key, value in kwargs.items() if key != 'proxy'})

    monkeypatch.setattr('modloader.network_services.httpx.AsyncClient', factory)
    return options


def result_of(response, streaming=False):
    assert response.status_code == 200, response.text
    if not streaming:
        return response.json()
    events = [json.loads(line) for line in response.text.splitlines()]
    assert any(event['type'] == 'progress' and event['stage'] in ('web_search', 'web_fetch') for event in events)
    assert events[-1]['type'] == 'result', events
    return events[-1]['value']


async def test_settings_defaults_capabilities_and_encrypted_credentials(client, controller):
    safe = controller.safe_settings()
    assert {key: safe[key] for key in DEFAULT_NETWORK_SETTINGS} == DEFAULT_NETWORK_SETTINGS
    assert safe['web_credentials'] == dict.fromkeys(WEB_CREDENTIAL_KEYS, False)
    assert {item['id']: item['capabilities'] for item in safe['web_services']} == {
        'local': ['fetch'], 'searxng': ['search', 'images'], 'deepseek': ['search', 'fetch'],
        'tavily': ['search', 'images', 'fetch'], 'brave': ['search', 'images'], 'serper': ['search', 'images'],
        'openai': ['search', 'images', 'fetch'],
        'saucenao': ['reverse'],
    }
    assert controller.store.get('settings', 'main') is None
    secrets = {key: 'fixture-' + key for key in WEB_CREDENTIAL_KEYS}
    response = await client.patch('/api/settings', json={
        'web_search_provider': 'deepseek', 'web_image_provider': 'brave', 'web_fetch_provider': 'tavily',
        'deepseek_url': 'https://deepseek.test/api/v1/', 'tavily_url': 'https://tavily.test/api',
        'brave_search_url': 'https://brave.test/', 'serper_url': 'https://serper.test/', 'secrets': secrets,
    })
    assert response.status_code == 200, response.text
    assert all(response.json()['web_credentials'].values())
    assert response.json()['web_search_provider'] == 'deepseek'
    assert response.json()['web_image_provider'] == 'brave'
    assert response.json()['web_fetch_provider'] == 'tavily'
    stored = controller.store.get('settings', 'main')
    encrypted = controller.vault.path.read_text('utf-8')
    reopened = Vault(controller.vault.path)
    for key, value in secrets.items():
        assert reopened.get(key) == value
        assert value not in encrypted and value not in response.text and value not in json.dumps(stored)
    cleared = await client.patch('/api/settings', json={'secrets': dict.fromkeys(WEB_CREDENTIAL_KEYS, '')})
    assert cleared.status_code == 200 and not any(cleared.json()['web_credentials'].values())
    assert all(Vault(controller.vault.path).get(key) is None for key in WEB_CREDENTIAL_KEYS)
    assert cleared.json()['web_search_provider'] == 'deepseek'  # Missing credentials do not disable a selection.


@pytest.mark.parametrize('patch', [
    {'web_search_provider': 'local'}, {'web_image_provider': 'deepseek'}, {'web_fetch_provider': 'searxng'},
    {'web_search_provider': 'unknown'}, {'web_image_provider': []}, {'web_fetch_provider': None},
    {'deepseek_url': 'https://user:password@api.test'}, {'tavily_url': 'https://@api.test'},
    {'brave_search_url': 'https://bad.test:bad'}, {'serper_url': 'file:///private'},
    {'deepseek_url': None}, {'secrets': []}, {'secrets': {'search_deepseek': False}},
    {'secrets': {'search_tavily': 'bad\nkey'}}, {'secrets': {'search_brave': '中文'}},
    {'openai_search_url': 'https://user:password@api.test'}, {'openai_search_url': 'file:///private'},
    {'openai_search_model': ''}, {'openai_search_model': ' \n'}, {'openai_search_model': None},
    {'openai_search_model': 42}, {'openai_search_model': []}, {'secrets': {'search_openai': False}},
    {'openai_read_model': ''}, {'openai_read_model': None}, {'openai_read_model': '  '}, {'openai_read_model': 42},
    {'openai_image_model': ''}, {'openai_image_model': None}, {'openai_image_model': ' \n'},
    {'openai_image_model': 42}, {'openai_image_model': []},
])
async def test_invalid_settings_do_not_partially_save_or_return_secrets(client, controller, patch):
    before = dict(controller.settings)
    response = await client.patch('/api/settings', json={**patch, 'font_scale': 1.5})
    assert response.status_code == 400, response.text
    assert controller.settings == before and controller.store.get('settings', 'main') is None
    assert 'password' not in response.text and 'bad\nkey' not in response.text


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('provider', ['deepseek', 'tavily', 'brave', 'serper'])
async def test_search_contracts_and_selected_provider(client, controller, monkeypatch, provider, streaming):
    controller.settings.update(web_search_provider=provider, proxy_mode='manual', proxy_url='http://proxy.test:17890')
    controller.vault.set('search_' + provider, 'fixture-api-key')
    if provider == 'deepseek':
        controller.settings['deepseek_url'] = 'https://deepseek.test/prefix/v1/'
    calls = []
    summary = '中文 & 日本語 ' * 3000

    def backend(request):
        calls.append(request)
        query = '今日 中文 + C++'
        if provider == 'deepseek':
            assert request.method == 'POST' and str(request.url) == 'https://deepseek.test/prefix/v1/tools/search'
            assert request.headers['authorization'] == 'Bearer fixture-api-key'
            assert json.loads(request.content) == {'query': query, 'need_content': False}
            return httpx.Response(200, json={'results': [
                {'url': 'javascript:bad', 'title': 'unsafe'},
                {'url': 'https://source.test/1', 'title': '<b>来源</b>', 'summary': summary, 'snippet': 'not the preferred summary'},
                {'url': 'https://source.test/1#duplicate', 'summary': 'duplicate'},
                {'url': 'https://source.test/2', 'snippet': '<p>摘要</p>'},
                *[{'url': f'https://many.test/{index}', 'summary': summary} for index in range(55)],
            ]})
        if provider == 'tavily':
            assert request.method == 'POST' and str(request.url) == 'https://api.tavily.com/search'
            assert request.headers['authorization'] == 'Bearer fixture-api-key'
            assert json.loads(request.content) == {'query': query, 'include_images': False, 'include_image_descriptions': False}
            return httpx.Response(200, json={'results': [
                {'url': 'https://source.test/1', 'title': '<b>来源</b>', 'content': summary},
                {'url': 'https://source.test/1#duplicate', 'content': 'duplicate'},
                {'url': 'https://source.test/2', 'content': '<p>摘要</p>'},
            ]})
        if provider == 'brave':
            assert request.method == 'GET' and request.url.path == '/res/v1/web/search'
            assert request.headers['x-subscription-token'] == 'fixture-api-key'
            assert dict(request.url.params) == {'q': query}
            return httpx.Response(200, json={'web': {'results': [
                {'url': 'https://source.test/1', 'title': '<b>来源</b>', 'description': summary},
                {'url': 'https://source.test/1#duplicate', 'description': 'duplicate'},
                {'url': 'https://source.test/2', 'description': '<p>摘要</p>'},
            ]}})
        assert request.method == 'POST' and str(request.url) == 'https://google.serper.dev/search'
        assert request.headers['x-api-key'] == 'fixture-api-key'
        assert json.loads(request.content) == {'q': query}
        return httpx.Response(200, json={'organic': [
            {'link': 'https://source.test/1', 'title': '<b>来源</b>', 'snippet': summary},
            {'link': 'https://source.test/1#duplicate', 'snippet': 'duplicate'},
            {'link': 'https://source.test/2', 'snippet': '<p>摘要</p>'},
        ]})

    options = mock_api(monkeypatch, backend)
    response = await client.post('/api/web-search', json={'query': '  今日 中文 + C++  '},
                                 headers={'Accept': 'application/x-ndjson'} if streaming else {})
    result = result_of(response, streaming)
    validate_search_result(result)
    assert result['query'] == '今日 中文 + C++'
    assert result['sources'][0] == {'id': 1, 'url': 'https://source.test/1', 'title': '来源', 'snippet': summary.strip()}
    assert result['sources'][1]['snippet'] == '摘要'
    assert len(result['sources']) == (57 if provider == 'deepseek' else 2)
    assert len(calls) == 1
    assert options == [{'proxy': 'http://proxy.test:17890', 'trust_env': False, 'verify': True, 'follow_redirects': False, 'timeout': 30}]
    assert not controller.runtime.instances and controller.store.all('jobs') == []


@pytest.mark.parametrize('provider', ['tavily', 'brave', 'serper'])
async def test_image_contract_preserves_source_image_and_thumbnail(client, controller, monkeypatch, provider):
    controller.settings.update(web_search_provider='deepseek', web_image_provider=provider)
    controller.vault.set('search_' + provider, 'fixture-api-key')

    def backend(request):
        if provider == 'tavily':
            assert request.url.path == '/search'
            assert json.loads(request.content) == {'query': '图片', 'include_images': True, 'include_image_descriptions': True}
            return httpx.Response(200, json={'results': [{'url': 'https://source.test/page', 'title': '出处', 'content': '<b>摘要</b>',
                'images': [{'url': 'https://image.test/original.png', 'description': '<b>图片</b>'},
                           {'url': 'https://image.test/second.png', 'description': '另一张'}]}],
                'images': [{'url': 'https://image.test/original.png', 'description': 'duplicate'},
                           {'url': 'https://image.test/no-page.png', 'description': '仅图片链接'},
                           {'url': 'javascript:unsafe', 'description': '不得显示'}]})
        if provider == 'brave':
            assert request.method == 'GET' and request.url.path == '/res/v1/images/search' and dict(request.url.params) == {'q': '图片'}
            return httpx.Response(200, json={'results': [
                {'url': 'https://source.test/page', 'title': '<b>图片</b>', 'description': '摘要',
                 'properties': {'url': 'https://image.test/original.png'}, 'thumbnail': {'src': 'https://thumb.test/small.webp'}},
                {'url': 'https://source.test/page#same', 'title': '另一张', 'properties': {'url': 'https://image.test/second.png'},
                 'thumbnail': {'src': 'javascript:bad'}},
                {'url': 'file:///unsafe', 'properties': {'url': 'https://image.test/unsafe.png'}},
            ]})
        assert request.method == 'POST' and request.url.path == '/images' and json.loads(request.content) == {'q': '图片'}
        return httpx.Response(200, json={'images': [
            {'link': 'https://source.test/page', 'title': '<b>图片</b>', 'imageUrl': 'https://image.test/original.png', 'thumbnailUrl': 'https://thumb.test/small.webp'},
            {'link': 'https://source.test/page#same', 'title': '另一张', 'imageUrl': 'https://image.test/second.png', 'thumbnailUrl': 'javascript:bad'},
            {'link': 'javascript:unsafe', 'imageUrl': 'https://image.test/unsafe.png'},
        ]})

    mock_api(monkeypatch, backend)
    result = result_of(await client.post('/api/web-search', json={'query': '图片', 'category': 'images'}))
    validate_search_result(result)
    assert result['sources'][0]['url'] == 'https://source.test/page'
    assert result['images'][:2] == [
        {'source_id': 1, 'title': '图片', 'url': 'https://image.test/original.png',
         **({'thumbnail_url': 'https://thumb.test/small.webp'} if provider != 'tavily' else {})},
        {'source_id': 1, 'title': '另一张', 'url': 'https://image.test/second.png'},
    ]
    if provider == 'tavily':
        assert result['sources'][1]['url'] == 'https://image.test/no-page.png'
        assert result['images'][2] == {'source_id': 2, 'title': '仅图片链接', 'url': 'https://image.test/no-page.png'}
    else:
        assert len(result['images']) == 2 and len(result['sources']) == 1


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('provider', ['deepseek', 'tavily'])
async def test_fetch_markdown_default_and_continuation_windows(client, controller, monkeypatch, provider, streaming):
    controller.settings['web_fetch_provider'] = provider
    controller.vault.set('search_' + provider, 'fixture-api-key')
    lines = ['# 文档', '', '```python', '    print("中")', '```', *[f'第 {index} 行' for index in range(6, 121)]]
    content = '\n'.join(lines)
    calls = []

    def backend(request):
        calls.append(request)
        if provider == 'deepseek':
            assert request.url.path == '/v1/tools/fetch' and json.loads(request.content) == {'url': 'https://public.test/doc'}
            return httpx.Response(200, json={'title': '<b>中文文档</b>', 'content': content})
        assert request.url.path == '/extract' and json.loads(request.content) == {'urls': ['https://public.test/doc'], 'format': 'markdown'}
        return httpx.Response(200, json={'results': [{'url': 'https://public.test/doc', 'raw_content': content}], 'failed_results': []})

    mock_api(monkeypatch, backend)
    first = result_of(await client.post('/api/web-fetch', json={'url': 'https://public.test/doc#fragment'},
                                       headers={'Accept': 'application/x-ndjson'} if streaming else {}), streaming)
    assert first['start_line'] == 1 and first['end_line'] == 80 and first['total_lines'] == 120
    assert first['content'] == '\n'.join(lines[:80]) and first['url'] == 'https://public.test/doc'
    assert first['title'] == ('中文文档' if provider == 'deepseek' else '')
    continuation = result_of(await client.post('/api/web-fetch', json={'url': first['url'], 'start_line': 81, 'line_count': 10**30}))
    assert continuation['content'] == '\n'.join(lines[80:]) and continuation['end_line'] == 120
    assert len(calls) == 2
    beyond = await client.post('/api/web-fetch', json={'url': first['url'], 'start_line': 121})
    assert beyond.status_code == 400 and '120' in beyond.json()['detail']


async def test_explicit_service_test_does_not_change_settings_or_start_model(client, controller, monkeypatch):
    before = dict(controller.settings)
    controller.vault.set('search_deepseek', 'fixture-api-key')
    mock_api(monkeypatch, lambda request: httpx.Response(200, json={'results': [{'url': 'https://source.test/', 'summary': '摘要'}]}))
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'deepseek', 'capability': 'search', 'query': 'test'}))
    assert result['sources'][0]['snippet'] == '摘要'
    assert controller.settings == before and controller.store.get('settings', 'main') is None
    assert not controller.runtime.instances and controller.store.all('jobs') == []


@pytest.mark.parametrize('query', ['中' * 128, '🐼' * 128])
async def test_deepseek_query_boundary_counts_unicode_not_utf8_bytes(client, controller, monkeypatch, query):
    controller.vault.set('search_deepseek', 'fixture-api-key')

    def backend(request):
        assert json.loads(request.content)['query'] == query
        return httpx.Response(200, json={'results': [{'url': 'https://source.test/'}]})

    mock_api(monkeypatch, backend)
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'deepseek', 'capability': 'search', 'query': query}))
    assert result['query'] == query


async def test_tavily_image_url_strings_do_not_invent_page_provenance(client, controller, monkeypatch):
    controller.vault.set('search_tavily', 'fixture-api-key')
    mock_api(monkeypatch, lambda request: httpx.Response(200, json={'results': [{'url': 'https://unrelated.test/page', 'title': '网页'}],
        'images': ['https://image.test/1.png', 'https://image.test/2.png']}))
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'tavily', 'capability': 'images', 'query': 'test'}))
    validate_search_result(result)
    assert [source['url'] for source in result['sources']] == ['https://image.test/1.png', 'https://image.test/2.png']
    assert [image['source_id'] for image in result['images']] == [1, 2]


@pytest.mark.parametrize('payload', [{}, {'web': None}, {'web': {'results': []}}])
async def test_brave_absent_or_null_web_is_a_valid_empty_search(client, controller, monkeypatch, payload):
    controller.vault.set('search_brave', 'fixture-api-key')
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=payload))
    response = await client.post('/api/web-services/test', json={'provider': 'brave', 'capability': 'search', 'query': 'test'})
    assert response.status_code == 422 and '未找到' in response.json()['detail']


async def test_existing_searxng_and_local_test_entries_remain_available(client, controller, monkeypatch):
    controller.settings.update(web_search_provider='deepseek', web_image_provider='brave', web_fetch_provider='tavily')
    calls = []

    def backend(request):
        calls.append(parse_qs(request.content.decode()))
        return httpx.Response(200, json={'results': [{'url': 'https://source.test/page', 'img_src': 'https://image.test/1.png'}]})

    mock_api(monkeypatch, backend)
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'searxng', 'capability': 'images', 'query': 'test'}))
    assert result['images'][0]['url'] == 'https://image.test/1.png'
    assert calls == [{'q': ['!bdi test'], 'format': ['json'], 'categories': ['images']}]
    local_calls = []

    async def local(url, start_line, line_count, proxy, *, progress):
        local_calls.append((url, start_line, line_count, proxy))
        return {'url': url, 'content': 'local'}

    monkeypatch.setattr('modloader.network_services.fetch_page', local)
    assert result_of(await client.post('/api/web-services/test', json={'provider': 'local', 'capability': 'fetch', 'url': 'https://public.test/'}))['content'] == 'local'
    assert local_calls == [('https://public.test/', 1, 80, '')]


@pytest.mark.parametrize('body', [
    {'provider': 'deepseek', 'capability': 'images', 'query': 'test'},
    {'provider': 'brave', 'capability': 'fetch', 'url': 'https://public.test/'},
    {'provider': 'local', 'capability': 'search', 'query': 'test'},
    {'provider': 'missing', 'capability': 'search', 'query': 'test'},
    {'provider': [], 'capability': 'search', 'query': 'test'},
    {'provider': 'deepseek', 'capability': [], 'query': 'test'},
    {'provider': 'deepseek', 'capability': 'search', 'query': '中' * 129},
    {'provider': 'deepseek', 'capability': 'search', 'query': ' '},
    {'provider': 'deepseek', 'capability': 'fetch', 'url': 'https://user:secret@public.test/'},
    {'provider': 'deepseek', 'capability': 'fetch', 'url': 'https://public.test/', 'line_count': 0},
    {'provider': 'tavily', 'capability': 'fetch', 'url': 'https://public.test/', 'start_line': True},
])
async def test_invalid_test_arguments_fail_before_network(client, controller, monkeypatch, body):
    controller.vault.set('search_deepseek', 'fixture-api-key')
    controller.vault.set('search_tavily', 'fixture-api-key')
    mock_api(monkeypatch, lambda request: pytest.fail('Invalid arguments reached the network'))
    response = await client.post('/api/web-services/test', json=body)
    assert response.status_code == 400, response.text
    assert 'secret' not in response.text


@pytest.mark.parametrize('provider', ['deepseek', 'tavily', 'brave', 'serper', 'openai'])
async def test_unconfigured_key_is_explicit_and_does_not_fallback(client, controller, monkeypatch, provider):
    controller.settings['web_search_provider'] = provider
    mock_api(monkeypatch, lambda request: pytest.fail('Missing key reached a service or fallback'))
    response = await client.post('/api/web-search', json={'query': 'test'})
    assert response.status_code == 400 and 'API Key' in response.json()['detail']


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('failure', ['401', '403', '429', '500', '302', 'html', 'shape', 'empty', 'timeout', 'connection'])
async def test_errors_are_sanitized_without_retry_or_fallback(client, controller, monkeypatch, streaming, failure):
    controller.vault.set('search_deepseek', 'fixture-sensitive-key')
    calls = []

    def backend(request):
        calls.append(request)
        if failure == 'timeout':
            raise httpx.ReadTimeout('fixture-sensitive-key', request=request)
        if failure == 'connection':
            raise httpx.ConnectError('fixture-sensitive-key', request=request)
        if failure.isdecimal():
            return httpx.Response(int(failure), text='fixture-sensitive-key remote body', headers={'location': 'https://other.test/'})
        if failure == 'html':
            return httpx.Response(200, text='<html>fixture-sensitive-key</html>')
        return httpx.Response(200, json={'shape': {'results': 'fixture-sensitive-key'}, 'empty': {'results': []}}[failure])

    mock_api(monkeypatch, backend)
    response = await client.post('/api/web-services/test', json={'provider': 'deepseek', 'capability': 'search', 'query': 'test'},
                                 headers={'Accept': 'application/x-ndjson'} if streaming else {})
    expected = 504 if failure == 'timeout' else 422 if failure == 'empty' else 502
    if streaming:
        event = json.loads(response.text.splitlines()[-1])
        assert event['type'] == 'error' and event['status'] == expected
    else:
        assert response.status_code == expected, response.text
    assert 'fixture-sensitive-key' not in response.text and 'remote body' not in response.text
    assert len(calls) == 1


async def test_tavily_failed_url_does_not_return_remote_error_body(client, controller, monkeypatch):
    controller.vault.set('search_tavily', 'fixture-sensitive-key')
    mock_api(monkeypatch, lambda request: httpx.Response(200, json={'results': [], 'failed_results': [{'url': 'https://public.test/', 'error': 'fixture-sensitive-key'}]}))
    response = await client.post('/api/web-services/test', json={'provider': 'tavily', 'capability': 'fetch', 'url': 'https://public.test/'})
    assert response.status_code == 502 and '未能读取' in response.json()['detail'] and 'fixture-sensitive-key' not in response.text


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('routed', [False, True])
@pytest.mark.parametrize(('provider', 'capability'), [('deepseek', 'search'), ('openai', 'search'), ('openai', 'images'), ('openai', 'fetch')])
async def test_disconnect_cancels_authenticated_service_and_closes_client(controller, monkeypatch, streaming, provider, capability, routed):
    controller.vault.set('search_' + provider, 'fixture-api-key')
    prefix = {'search': 'web_search', 'images': 'web_image', 'fetch': 'web_fetch'}[capability]
    controller.settings[prefix + '_provider'] = provider
    controller.settings[prefix + '_fallbacks'] = ['openai' if provider == 'deepseek' else 'tavily']
    endpoint = ('/api/web-fetch' if capability == 'fetch' else '/api/web-search') if routed else '/api/web-services/test'
    body = {'provider': provider, 'capability': capability, 'query': 'pending', 'url': 'https://source.test/',
            'category': 'images' if capability == 'images' else 'general'}
    if routed:
        body.pop('provider')
    started, cancelled, closed, disconnect = (asyncio.Event() for _ in range(4))
    requests = []

    class PendingService(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            requests.append(request)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def aclose(self):
            closed.set()

    original = httpx.AsyncClient
    monkeypatch.setattr('modloader.network_services.httpx.AsyncClient', lambda **options: original(transport=PendingService(), **options))
    delivered = False
    sent = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
        await disconnect.wait()
        return {'type': 'http.disconnect'}

    async def send(event):
        sent.append(event)

    headers = [(b'host', b'testserver'), (b'authorization', b'Bearer network-services-fixture'), (b'content-type', b'application/json')]
    if streaming:
        headers.append((b'accept', b'application/x-ndjson'))
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'http_version': '1.1',
             'method': 'POST', 'scheme': 'http', 'path': endpoint, 'raw_path': endpoint.encode(),
             'query_string': b'', 'headers': headers, 'client': ('127.0.0.1', 1), 'server': ('testserver', 80)}
    task = asyncio.create_task(build_app(controller)(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), 2)
        disconnect.set()
        await asyncio.wait_for(task, 2)
        assert cancelled.is_set() and closed.is_set()
        assert len(requests) == 1
        assert not any(b'"type": "result"' in event.get('body', b'') for event in sent)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_service_test_requires_management_auth_and_is_not_public(controller, monkeypatch):
    original = httpx.AsyncClient
    mock_api(monkeypatch, lambda request: pytest.fail('Unauthorized request reached a service'))
    body = {'provider': 'deepseek', 'capability': 'search', 'query': 'test'}
    for public, token, status in [(False, '', 401), (False, 'wrong', 401), (True, controller.api_key, 404)]:
        async with original(transport=httpx.ASGITransport(app=build_app(controller, public=public)), base_url='http://testserver',
                            headers={'Authorization': 'Bearer ' + token}) as anonymous:
            response = await anonymous.post('/api/web-services/test', json=body)
        assert response.status_code == status, response.text
