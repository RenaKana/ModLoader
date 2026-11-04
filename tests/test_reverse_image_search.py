"""SauceNAO contracts use synthetic images and mocked HTTP, never live uploads."""
import asyncio
import base64
from email import policy
from email.parser import BytesParser
import json

import httpx
import pytest

from modloader.app import build_app
from modloader.secrets import Vault

from test_network_services import client, controller, mock_api


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
IMAGE = 'data:image/png;base64,' + base64.b64encode(PNG).decode('ascii')
BODY = {'image': IMAGE, 'filename': 'sample.png'}


def response_payload(*rows, header=None):
    return {'header': {'status': 0, 'short_remaining': 3, 'long_remaining': '149', **(header or {})},
            'results': list(rows)}


def match_row(similarity='12.34', **data):
    return {'header': {'similarity': similarity, 'thumbnail': 'https://img.saucenao.test/thumbnail.jpg',
                       'index_name': 'Pixiv Images'},
            'data': {'title': '<b>完整标题</b>', 'member_name': '画师',
                     'ext_urls': ['https://www.pixiv.net/artworks/42'], **data}}


def multipart(request):
    message = BytesParser(policy=policy.default).parsebytes(
        b'Content-Type: ' + request.headers['content-type'].encode('ascii') + b'\r\n\r\n' + request.content)
    return {part.get_param('name', header='content-disposition'): part for part in message.iter_parts()}


@pytest.mark.parametrize('anonymous', [True, False])
@pytest.mark.parametrize('service_test', [False, True])
async def test_uploads_only_supplied_image_bytes_with_optional_key_and_proxy(client, controller, monkeypatch, anonymous, service_test):
    controller.settings.update(saucenao_url='https://saucenao.test/prefix/', proxy_mode='manual', proxy_url='http://proxy.test:17890')
    if not anonymous:
        controller.vault.set('search_saucenao', 'fixture-sensitive-key')
    calls = []

    def backend(request):
        calls.append(request)
        assert request.method == 'POST' and str(request.url) == 'https://saucenao.test/prefix/search.php'
        assert not request.url.query and 'authorization' not in request.headers
        assert request.headers['accept'] == 'application/json'
        fields = multipart(request)
        assert set(fields) == {'file', 'output_type'} | (set() if anonymous else {'api_key'})
        assert fields['file'].get_filename() == 'sample.png'
        assert fields['file'].get_content_type() == 'image/png'
        assert fields['file'].get_payload(decode=True) == PNG
        assert fields['output_type'].get_payload(decode=True) == b'2'
        if not anonymous:
            assert fields['api_key'].get_payload(decode=True) == b'fixture-sensitive-key'
        return httpx.Response(200, json=response_payload(match_row()))

    options = mock_api(monkeypatch, backend)
    route = '/api/web-services/test' if service_test else '/api/web-reverse-image'
    body = {**BODY, **({'provider': 'saucenao', 'capability': 'reverse'} if service_test else {})}
    response = await client.post(route, json=body)
    assert response.status_code == 200, response.text
    value = response.json()
    assert set(value) == {'provider', 'elapsed_ms', 'matches', 'quota'}
    assert value['provider'] == 'saucenao' and value['elapsed_ms'] >= 0
    assert value['quota'] == {'short_remaining': 3, 'long_remaining': 149}
    assert value['matches'] == [{'similarity': 12.34, 'thumbnail': 'https://img.saucenao.test/thumbnail.jpg',
                                 'title': '完整标题', 'author': '画师', 'urls': ['https://www.pixiv.net/artworks/42'],
                                 'index_name': 'Pixiv Images'}]
    assert 'fixture-sensitive-key' not in response.text
    assert len(calls) == 1
    assert options == [{'proxy': 'http://proxy.test:17890', 'trust_env': False, 'verify': True,
                        'follow_redirects': False, 'timeout': 30}]


async def test_preserves_all_low_similarity_matches_and_safe_provenance(client, monkeypatch):
    rows = [match_row(str(index / 10), title=f'匹配 {index}') for index in range(65)]
    rows.append({'header': {'similarity': '0', 'thumbnail': 'javascript:bad', 'index_name': 'Anime'},
                 'data': {'source': '作品原名', 'creator': ['作者一', '作者二'],
                          'ext_urls': ['javascript:bad', 'file:///private', 'https://user:secret@source.test/',
                                       'https://source.test/42', 'https://source.test/42']}})
    rows.extend([None, {'header': {}, 'data': {}}, match_row('NaN'), match_row(True), match_row(101)])
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=response_payload(*rows, header={'short_remaining': 0, 'long_remaining': 0})))
    response = await client.post('/api/web-reverse-image', json=BODY)
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result['matches']) == 66
    assert [value['title'] for value in result['matches'][:65]] == [f'匹配 {index}' for index in range(65)]
    assert result['matches'][-1] == {'similarity': 0, 'thumbnail': '', 'title': '作品原名', 'author': '作者一, 作者二',
                                     'urls': ['https://source.test/42'], 'index_name': 'Anime'}
    # Zero quota is valid after the last successful query, not an error by itself.
    assert result['quota'] == {'short_remaining': 0, 'long_remaining': 0}


async def test_reverse_progress_stream_keeps_independent_result_shape(client, monkeypatch):
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=response_payload(match_row(), header={'long_remaining': None})))
    response = await client.post('/api/web-reverse-image', json=BODY, headers={'Accept': 'application/x-ndjson'})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert any(event.get('stage') == 'web_reverse' for event in events)
    assert events[-1]['type'] == 'result'
    result = events[-1]['value']
    assert 'sources' not in result and 'query' not in result and result['provider'] == 'saucenao'
    assert result['quota'] == {'short_remaining': 3}


@pytest.mark.parametrize('service_test', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
async def test_successful_search_without_matches_is_a_successful_result(client, monkeypatch, service_test, streaming):
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=response_payload()))
    route = '/api/web-services/test' if service_test else '/api/web-reverse-image'
    body = {**BODY, **({'provider': 'saucenao', 'capability': 'reverse'} if service_test else {})}
    response = await client.post(route, json=body, headers={'Accept': 'application/x-ndjson'} if streaming else {})
    assert response.status_code == 200, response.text
    if streaming:
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['type'] == 'result'
        assert not any(event['type'] == 'error' for event in events)
        result = events[-1]['value']
    else:
        result = response.json()
    assert result['provider'] == 'saucenao' and result['matches'] == [] and result['elapsed_ms'] >= 0
    assert result['quota'] == {'short_remaining': 3, 'long_remaining': 149}


@pytest.mark.parametrize('body', [
    {}, {'filename': 'a.png'}, {**BODY, 'image': None}, {**BODY, 'image': 'https://image.test/a.png'},
    {**BODY, 'image': 'C:\\private\\a.png'}, {**BODY, 'image': 'file:///private/a.png'},
    {**BODY, 'image': 'data:text/plain;base64,SGVsbG8='}, {**BODY, 'image': 'data:image/png;base64,'},
    {**BODY, 'image': 'data:image/png;base64,a==='}, {**BODY, 'image': 'data:image/png;base64,not base64!'},
    {**BODY, 'filename': ''}, {**BODY, 'filename': 42}, {**BODY, 'filename': 'bad\nname.png'},
    {**BODY, 'path': 'C:\\private\\a.png'}, {**BODY, 'url': 'https://image.test/a.png'},
])
async def test_invalid_input_never_reads_files_or_fetches_urls(client, monkeypatch, body):
    mock_api(monkeypatch, lambda request: pytest.fail('Invalid reverse-image input reached the network'))
    response = await client.post('/api/web-reverse-image', json=body)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('failure,status,detail', [
    ('401', 502, 'API Key'), ('403', 502, 'API Key'), ('429', 429, '额度或频率'), ('500', 502, 'HTTP 500'),
    ('302', 502, 'HTTP 302'), ('html', 502, '验证页面'), ('invalid-json', 502, '有效 JSON'),
    ('empty-body', 502, '有效 JSON'),
    ('challenge-403', 502, '验证页面'), ('server-html', 502, 'HTTP 500'),
    ('array', 502, 'header'), ('missing-header', 502, 'header'), ('missing-status', 502, '状态码'),
    ('api-key', 502, 'API Key'), ('daily-limit', 429, '当日搜索额度'), ('short-limit', 429, '搜索频率'),
    ('unknown-status', 502, '状态 -3'), ('invalid-results', 502, 'results 数组'),
    ('missing-results', 502, 'results 数组'), ('malformed-results', 502, '有效的匹配项'),
    ('timeout', 504, '请求超时'), ('connection', 502, '连接失败'),
])
async def test_upstream_failures_are_finite_explicit_and_redacted(client, controller, monkeypatch, streaming, failure, status, detail):
    controller.vault.set('search_saucenao', 'fixture-sensitive-key')
    calls = []

    def backend(request):
        calls.append(request)
        if failure == 'timeout':
            raise httpx.ReadTimeout('fixture-sensitive-key', request=request)
        if failure == 'connection':
            raise httpx.ConnectError('fixture-sensitive-key', request=request)
        if failure.isdecimal():
            return httpx.Response(int(failure), content=b'fixture-sensitive-key remote body', headers={'location': 'https://other.test/'})
        if failure in ('html', 'challenge-403', 'server-html'):
            return httpx.Response(403 if failure == 'challenge-403' else 500 if failure == 'server-html' else 200,
                                  text='<!DOCTYPE html><html>Cloudflare challenge fixture-sensitive-key</html>', headers={'Content-Type': 'text/html'})
        if failure in ('invalid-json', 'empty-body'):
            return httpx.Response(200, content=b'' if failure == 'empty-body' else b'invalid fixture-sensitive-key')
        payload = {
            'array': [], 'missing-header': {}, 'missing-status': {'header': {}, 'results': []},
            'api-key': response_payload(header={'status': -2, 'short_remaining': 0, 'long_remaining': 0,
                                                'message': 'Invalid API key fixture-sensitive-key'}),
            'daily-limit': response_payload(header={'status': -2, 'long_remaining': 0, 'message': 'fixture-sensitive-key'}),
            'short-limit': response_payload(header={'status': -1, 'short_remaining': '0', 'message': 'fixture-sensitive-key'}),
            'unknown-status': response_payload(header={'status': -3, 'message': 'fixture-sensitive-key'}),
            'invalid-results': {'header': {'status': 0}, 'results': 'fixture-sensitive-key'},
            'missing-results': {'header': {'status': 0}},
            'malformed-results': response_payload({'header': {'similarity': 'NaN'}, 'data': {}}),
        }[failure]
        return httpx.Response(200, json=payload)

    mock_api(monkeypatch, backend)
    response = await client.post('/api/web-reverse-image', json=BODY,
                                 headers={'Accept': 'application/x-ndjson'} if streaming else {})
    if streaming:
        event = json.loads(response.text.splitlines()[-1])
        assert event['type'] == 'error' and event['status'] == status
        assert detail in event['detail']
    else:
        assert response.status_code == status, response.text
        assert detail in response.json()['detail']
    assert len(calls) == 1
    assert 'fixture-sensitive-key' not in response.text and 'remote body' not in response.text


async def test_total_timeout_cancels_a_stalled_reverse_service(client, monkeypatch):
    cancelled = asyncio.Event()

    async def stalled(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr('modloader.reverse_image_search.SEARCH_TIMEOUT_S', .01)
    mock_api(monkeypatch, stalled)
    response = await client.post('/api/web-reverse-image', json=BODY)
    assert response.status_code == 504 and cancelled.is_set()


async def test_settings_share_existing_vault_without_changing_old_routes(client, controller):
    before = {field: value for field, value in controller.settings.items() if field != 'saucenao_url'}
    safe = controller.safe_settings()
    assert safe['saucenao_url'] == 'https://saucenao.com'
    assert 'web_reverse_provider' not in safe
    assert next(service for service in safe['web_services'] if service['id'] == 'saucenao') == {
        'id': 'saucenao', 'label': 'SauceNAO', 'capabilities': ['reverse'],
        'url_setting': 'saucenao_url', 'credential_key': 'search_saucenao'}
    response = await client.patch('/api/settings', json={'saucenao_url': 'https://reverse.test/api',
                                 'secrets': {'search_saucenao': 'fixture-sensitive-key'}})
    assert response.status_code == 200, response.text
    assert response.json()['saucenao_url'] == 'https://reverse.test/api'
    assert response.json()['web_credentials']['search_saucenao'] is True
    assert Vault(controller.vault.path).get('search_saucenao') == 'fixture-sensitive-key'
    assert 'fixture-sensitive-key' not in response.text
    assert 'fixture-sensitive-key' not in controller.vault.path.read_text('utf-8')
    assert 'fixture-sensitive-key' not in json.dumps(controller.store.get('settings', 'main'))
    assert {field: value for field, value in controller.settings.items() if field != 'saucenao_url'} == before
    for patch in ({'saucenao_url': 'file:///private'}, {'saucenao_url': 'https://user:password@reverse.test'},
                  {'secrets': {'search_saucenao': 'bad\nkey'}}, {'web_search_provider': 'saucenao'},
                  {'web_image_provider': 'saucenao'}, {'web_fetch_provider': 'saucenao'},
                  {'web_image_fallbacks': ['saucenao']}):
        response = await client.patch('/api/settings', json=patch)
        assert response.status_code == 400, response.text


async def test_reverse_requires_management_auth_and_is_not_public(controller, monkeypatch):
    original = httpx.AsyncClient
    mock_api(monkeypatch, lambda request: pytest.fail('Unauthorized reverse request reached the network'))
    for public, token, status in ((False, '', 401), (False, 'wrong', 401), (False, controller.api_key, 401),
                                   (True, controller.api_key, 404)):
        async with original(transport=httpx.ASGITransport(app=build_app(controller, public=public)), base_url='http://testserver',
                            headers={'Authorization': 'Bearer ' + token}) as anonymous:
            response = await anonymous.post('/api/web-reverse-image', json=BODY)
        assert response.status_code == status, response.text


@pytest.mark.parametrize('streaming', [False, True])
async def test_disconnect_cancels_reverse_request_and_closes_client(controller, monkeypatch, streaming):
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
    monkeypatch.setattr('modloader.reverse_image_search.httpx.AsyncClient', lambda **options: original(transport=PendingService(), **options))
    delivered, sent = False, []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': json.dumps(BODY).encode(), 'more_body': False}
        await disconnect.wait()
        return {'type': 'http.disconnect'}

    async def send(event):
        sent.append(event)

    route = '/api/web-reverse-image'
    headers = [(b'host', b'testserver'), (b'authorization', b'Bearer network-services-fixture'), (b'content-type', b'application/json')]
    if streaming:
        headers.append((b'accept', b'application/x-ndjson'))
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'http_version': '1.1',
             'method': 'POST', 'scheme': 'http', 'path': route, 'raw_path': route.encode(),
             'query_string': b'', 'headers': headers, 'client': ('127.0.0.1', 1), 'server': ('testserver', 80)}
    task = asyncio.create_task(build_app(controller)(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), 2)
        disconnect.set()
        await asyncio.wait_for(task, 2)
        assert cancelled.is_set() and closed.is_set() and len(requests) == 1
        assert not any(b'"type": "result"' in event.get('body', b'') for event in sent)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
