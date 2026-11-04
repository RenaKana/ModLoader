"""Public-page extraction, SSRF boundaries, TLS pinning and request cancellation."""
import asyncio
from datetime import datetime, timedelta, timezone
import gzip
import ipaddress
import json
import socket
import ssl

import httpcore
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from modloader.app import Controller, build_app
from modloader.web_search import FETCH_MAX_REDIRECTS, FETCH_TIMEOUT_S


PUBLIC_IP = '93.184.216.34'


@pytest.fixture
async def controller(tmp_path):
    value = Controller(tmp_path / 'data', 'web-fetch-fixture')
    value.settings['proxy_mode'] = 'direct'
    try:
        yield value
    finally:
        await value.close()


@pytest.fixture
async def client(controller):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                headers={'Authorization': 'Bearer web-fetch-fixture'}) as value:
        yield value


@pytest.fixture
async def dns(monkeypatch):
    calls = []

    async def resolve(host, port, **options):
        calls.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (PUBLIC_IP, port))]

    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    return calls


def mock_pages(monkeypatch, handler):
    original = httpx.AsyncClient
    options = []

    def factory(**kwargs):
        options.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **{key: value for key, value in kwargs.items() if key != 'proxy'})

    monkeypatch.setattr('modloader.web_search.httpx.AsyncClient', factory)
    return options


@pytest.mark.parametrize('streaming', [False, True])
async def test_html_article_extraction_pagination_and_pinned_request(client, controller, monkeypatch, dns, streaming):
    controller.settings.update(proxy_mode='manual', proxy_url='http://proxy.test:17890')
    requests = []

    def backend(request):
        requests.append(request)
        assert str(request.url) == f'https://{PUBLIC_IP}:8443/article?q=1'
        assert request.headers['host'] == 'public.test:8443'
        assert request.headers['accept-encoding'] == 'identity'
        return httpx.Response(200, stream=Chunks('''<html><head><title>Example &amp; 中文</title></head><body>
          <nav>导航</nav><p>外部侧栏</p><main><h1>Main heading</h1><header>不读页眉</header>
          <p>Hello <strong>bold</strong> world.</p><script>不读脚本</script><style>不读样式</style>
          <form>不读表单</form><div hidden>不读隐藏内容</div><pre><code>if True:
    print("中")

return 7</code></pre><p>尾段</p></main><footer>不读页脚</footer></body></html>'''.encode()),
                              headers={'content-type': 'text/html; charset=utf-8'})

    options = mock_pages(monkeypatch, backend)
    headers = {'Accept': 'application/x-ndjson'} if streaming else {}
    response = await client.post('/api/web-fetch', json={'url': 'https://public.test:8443/article?q=1#fragment', 'line_count': 3}, headers=headers)
    assert response.status_code == 200, response.text
    if streaming:
        events = [json.loads(line) for line in response.text.splitlines()]
        assert any(event['type'] == 'progress' and event['stage'] == 'web_fetch' for event in events)
        assert events[-1]['type'] == 'result'
        result = events[-1]['value']
    else:
        result = response.json()
    assert result == {'url': 'https://public.test:8443/article?q=1', 'title': 'Example & 中文',
                      'content': 'Main heading\nHello bold world.\nif True:', 'start_line': 1, 'end_line': 3,
                      'total_lines': 7, 'elapsed_ms': result['elapsed_ms']}
    continuation = await client.post('/api/web-fetch', json={'url': result['url'], 'start_line': 4, 'line_count': 10**30})
    assert continuation.status_code == 200, continuation.text
    assert continuation.json()['content'] == '    print("中")\n\nreturn 7\n尾段'
    assert continuation.json()['end_line'] == 7
    assert len(requests) == 2 and dns == [('public.test', 8443)] * 2
    assert all(item['proxy'] == 'http://proxy.test:17890' and item['trust_env'] is False
               and item['follow_redirects'] is False and item['timeout'] == FETCH_TIMEOUT_S
               and item['verify'].check_hostname and item['verify'].verify_mode == ssl.CERT_REQUIRED for item in options)


async def test_plain_text_has_80_line_default_window_not_a_hard_limit(client, monkeypatch, dns):
    body = '\n'.join(f'第 {index} 行' for index in range(1, 121))
    mock_pages(monkeypatch, lambda request: httpx.Response(200, stream=Chunks(body.encode('gb18030')), headers={'content-type': 'text/plain; charset=gb18030'}))
    first = await client.post('/api/web-fetch', json={'url': 'http://public.test/doc'})
    assert first.status_code == 200 and first.json()['title'] == ''
    assert first.json()['start_line'] == 1 and first.json()['end_line'] == 80 and first.json()['total_lines'] == 120
    all_lines = await client.post('/api/web-fetch', json={'url': 'http://public.test/doc', 'line_count': 121})
    assert all_lines.status_code == 200 and all_lines.json()['content'] == body
    beyond = await client.post('/api/web-fetch', json={'url': 'http://public.test/doc', 'start_line': 121})
    assert beyond.status_code == 400 and '120' in beyond.json()['detail']


@pytest.mark.parametrize('body', [{}, {'url': 'file:///private'}, {'url': 'https://user:password@public.test/'},
    {'url': 'http://@public.test/'}, {'url': 'https://public.test/', 'start_line': 0},
    {'url': 'https://public.test/', 'line_count': -1}, {'url': 'https://public.test/', 'line_count': True},
    {'url': 'https://public.test/', 'start_line': 1.0}, {'url': 'https://public.test/', 'line_count': '80'}])
async def test_invalid_arguments_never_reach_network(client, monkeypatch, body):
    mock_pages(monkeypatch, lambda request: pytest.fail('Invalid arguments reached a page server'))
    response = await client.post('/api/web-fetch', json=body)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize('host', ['127.0.0.1', '0.0.0.0', '10.0.0.1', '192.168.0.1', '169.254.169.254',
    '100.64.0.1', '192.0.2.1', '224.0.0.1', '255.255.255.255', '[::1]', '[::]', '[fc00::1]', '[fe80::1]',
    '[ff02::1]', '[::ffff:127.0.0.1]'])
async def test_nonpublic_literal_is_rejected_before_connect(client, monkeypatch, host):
    mock_pages(monkeypatch, lambda request: pytest.fail('Non-public literal was connected'))
    response = await client.post('/api/web-fetch', json={'url': f'http://{host}/private'})
    assert response.status_code == 400 and '公网' in response.json()['detail']
    assert str(ipaddress.ip_address(host.strip('[]'))) in response.json()['detail']
    assert '网页地址' in response.json()['detail'] and 'DNS' not in response.json()['detail']


@pytest.mark.parametrize('addresses', [['127.0.0.1'], [PUBLIC_IP, '10.0.0.2'], ['::1'], ['169.254.169.254'],
    ['198.18.0.45'], ['198.19.255.254'], [PUBLIC_IP, '198.18.0.46']])
async def test_nonpublic_dns_results_are_rejected_including_mixed_answers(client, monkeypatch, addresses):
    async def resolve(host, port, **options):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (address, port)) for address in addresses]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    mock_pages(monkeypatch, lambda request: pytest.fail('Private DNS result was connected'))
    response = await client.post('/api/web-fetch', json={'url': 'http://public.test/private'})
    assert response.status_code == 400 and '公网' in response.json()['detail']
    detail = response.json()['detail']
    assert 'DNS 解析 public.test' in detail
    for address in addresses:
        if address != PUBLIC_IP:
            assert address in detail
    fake_ip = any(ipaddress.ip_address(address) in ipaddress.ip_network('198.18.0.0/15') for address in addresses)
    assert ('Fake-IP，需真实公网 DNS 解析' in detail) is fake_ip


@pytest.mark.parametrize('location', ['http://127.0.0.1/private', 'http://192.168.1.1/', 'file:///private', 'https://user@public.test/'])
async def test_each_redirect_target_is_validated_before_following(client, monkeypatch, dns, location):
    requests = []
    def backend(request):
        requests.append(request)
        return httpx.Response(302, headers={'location': location})
    mock_pages(monkeypatch, backend)
    response = await client.post('/api/web-fetch', json={'url': 'https://public.test/start'})
    assert response.status_code == 400 and len(requests) == 1


async def test_dns_rebinding_is_rechecked_on_redirect_and_no_second_connect_occurs(client, monkeypatch):
    resolutions, requests = [], []
    async def resolve(host, port, **options):
        resolutions.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (PUBLIC_IP if len(resolutions) == 1 else '127.0.0.1', port))]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    def backend(request):
        requests.append(request)
        assert request.url.host == PUBLIC_IP
        return httpx.Response(302, headers={'location': '/rebound'})
    mock_pages(monkeypatch, backend)
    response = await client.post('/api/web-fetch', json={'url': 'http://public.test/start'})
    assert response.status_code == 400 and len(requests) == 1 and resolutions == ['public.test', 'public.test']


async def test_public_redirect_is_pinned_again_and_has_a_finite_budget(client, monkeypatch, dns):
    requests = []
    repeat = False
    def backend(request):
        requests.append(request)
        if repeat:
            return httpx.Response(302, headers={'location': '/again'})
        if len(requests) == 1:
            return httpx.Response(302, headers={'location': 'https://next.test/final'})
        assert request.headers['host'] == 'next.test' and request.url.host == PUBLIC_IP
        return httpx.Response(200, stream=Chunks(b'Final text'), headers={'content-type': 'text/plain'})
    mock_pages(monkeypatch, backend)
    response = await client.post('/api/web-fetch', json={'url': 'http://public.test/start'})
    assert response.status_code == 200 and response.json()['url'] == 'https://next.test/final'
    assert dns == [('public.test', 80), ('next.test', 443)]
    requests.clear()
    repeat = True
    response = await client.post('/api/web-fetch', json={'url': 'http://public.test/loop'})
    assert response.status_code == 502 and '重定向超过' in response.json()['detail']
    assert len(requests) == FETCH_MAX_REDIRECTS + 1


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *chunks):
        self.chunks = chunks
        self.closed = False
        self.read = False

    async def __aiter__(self):
        self.read = True
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize(('headers', 'chunks', 'status', 'detail'), [
    ({'content-type': 'application/pdf'}, [b'%PDF-1.7'], 415, 'PDF'),
    ({'content-type': 'image/png'}, [b'\x89PNG'], 415, 'HTML'),
    ({}, [b'unknown'], 415, 'HTML'),
    ({'content-type': 'text/plain'}, [b' \n\t'], 422, '正文'),
    ({'content-type': 'text/html'}, [b'<html><nav>menu</nav><script>hidden</script></html>'], 422, '正文'),
    ({'content-type': 'text/plain', 'content-length': '101'}, [b'a'], 413, '100'),
    ({'content-type': 'text/plain'}, [b'a' * 60, b'b' * 60], 413, '100'),
    ({'content-type': 'text/plain', 'content-encoding': 'gzip'}, [gzip.compress(b'a' * 10000)], 413, '100'),
    ({'content-type': 'text/plain', 'content-encoding': 'gzip'}, [b'invalid'], 502, '压缩'),
    ({'content-type': 'text/plain', 'content-encoding': 'gzip'}, [gzip.compress(b'text')[:-4]], 502, '不完整'),
])
async def test_content_type_empty_body_and_stream_budget_fail_explicitly_and_release(client, monkeypatch, dns, headers, chunks, status, detail):
    monkeypatch.setenv('MODLOADER_WEB_FETCH_MAX_BYTES', '100')
    stream = Chunks(*chunks)
    mock_pages(monkeypatch, lambda request: httpx.Response(200, headers=headers, stream=stream))
    response = await client.post('/api/web-fetch', json={'url': 'https://public.test/'})
    assert response.status_code == status and detail in response.json()['detail'], response.text
    assert stream.closed
    if status == 415 or headers.get('content-length'):
        assert not stream.read


async def test_exact_configured_byte_budget_and_gzip_text_remain_readable(client, monkeypatch, dns):
    monkeypatch.setenv('MODLOADER_WEB_FETCH_MAX_BYTES', '100')
    requests = []
    def backend(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, headers={'content-type': 'text/plain'}, stream=Chunks(b'a' * 100))
        return httpx.Response(200, headers={'content-type': 'text/plain', 'content-encoding': 'gzip'}, stream=Chunks(gzip.compress(b'b' * 100)))
    mock_pages(monkeypatch, backend)
    for expected in ('a', 'b'):
        response = await client.post('/api/web-fetch', json={'url': 'http://public.test/'})
        assert response.status_code == 200 and response.json()['content'] == expected * 100, response.text


async def test_total_deadline_includes_a_stalled_body_and_releases_it(client, monkeypatch, dns):
    from modloader import web_search
    monkeypatch.setattr(web_search, 'FETCH_TIMEOUT_S', .1)
    closed = asyncio.Event()
    class Pending(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b'unreachable'
        async def aclose(self):
            closed.set()
    mock_pages(monkeypatch, lambda request: httpx.Response(200, headers={'content-type': 'text/plain'}, stream=Pending()))
    response = await client.post('/api/web-fetch', json={'url': 'http://public.test/'})
    assert response.status_code == 504 and closed.is_set()


async def test_invalid_budget_is_reported_instead_of_ignored(client, monkeypatch):
    monkeypatch.setenv('MODLOADER_WEB_FETCH_MAX_BYTES', '0')
    mock_pages(monkeypatch, lambda request: pytest.fail('Invalid safety budget was ignored'))
    response = await client.post('/api/web-fetch', json={'url': 'http://public.test/'})
    assert response.status_code == 500 and 'MODLOADER_WEB_FETCH_MAX_BYTES' in response.json()['detail']


@pytest.mark.parametrize('failure', ['timeout', 'connect', 'dns', 'status'])
async def test_network_errors_are_explicit_without_retry(client, monkeypatch, dns, failure):
    requests = []
    if failure == 'dns':
        async def resolve(*args, **kwargs):
            raise socket.gaierror('fixture DNS error')
        monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    def backend(request):
        requests.append(request)
        if failure == 'timeout':
            raise httpx.ReadTimeout('fixture timeout', request=request)
        if failure == 'connect':
            raise httpx.ConnectError('fixture connect', request=request)
        return httpx.Response(404)
    mock_pages(monkeypatch, backend)
    response = await client.post('/api/web-fetch', json={'url': 'http://public.test/'})
    assert response.status_code == (504 if failure == 'timeout' else 502)
    assert len(requests) == (0 if failure == 'dns' else 1)


@pytest.mark.parametrize('streaming', [False, True])
async def test_disconnect_during_page_body_cancels_and_closes_response_and_client(controller, monkeypatch, dns, streaming):
    started, cancelled, response_closed, client_closed, disconnect = (asyncio.Event() for _ in range(5))
    class PendingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            try:
                await asyncio.Event().wait()
                yield b'unreachable'
            finally:
                cancelled.set()
        async def aclose(self):
            response_closed.set()
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, headers={'content-type': 'text/plain'}, stream=PendingBody())
        async def aclose(self):
            client_closed.set()
    original = httpx.AsyncClient
    monkeypatch.setattr('modloader.web_search.httpx.AsyncClient', lambda **options: original(transport=Transport(), **options))
    delivered = False
    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': b'{"url":"https://public.test/"}', 'more_body': False}
        await disconnect.wait()
        return {'type': 'http.disconnect'}
    async def send(event):
        pass
    headers = [(b'host', b'testserver'), (b'authorization', b'Bearer web-fetch-fixture'), (b'content-type', b'application/json')]
    if streaming:
        headers.append((b'accept', b'application/x-ndjson'))
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'http_version': '1.1',
             'method': 'POST', 'scheme': 'http', 'path': '/api/web-fetch', 'raw_path': b'/api/web-fetch',
             'query_string': b'', 'headers': headers, 'client': ('127.0.0.1', 1), 'server': ('testserver', 80)}
    task = asyncio.create_task(build_app(controller)(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), 2)
        disconnect.set()
        await asyncio.wait_for(task, 2)
        assert cancelled.is_set() and response_closed.is_set() and client_closed.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def tls_certificate(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'public.test')])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                   .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1))
                   .not_valid_after(now + timedelta(days=1))
                   .add_extension(x509.SubjectAlternativeName([x509.DNSName('public.test')]), critical=False)
                   .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                   .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / 'fixture-cert.pem', tmp_path / 'fixture-key.pem'
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return cert_path, key_path


@pytest.mark.parametrize('proxy_enabled', [False, True])
@pytest.mark.parametrize('certificate_case', ['valid', 'wrong-host', 'untrusted'])
async def test_real_httpcore_tls_uses_original_sni_and_certificate_verification_with_pinned_connect(client, controller, monkeypatch, tls_certificate, proxy_enabled, certificate_case):
    """Real local TLS sockets exercise the installed httpx/httpcore/AnyIO path, not MockTransport."""
    cert_path, key_path = tls_certificate
    requests, sni, tcp_targets, connect_targets = [], [], [], []
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    context.set_servername_callback(lambda ssl_socket, hostname, ssl_context: sni.append(hostname))

    async def page(reader, writer):
        try:
            requests.append(await reader.readuntil(b'\r\n\r\n'))
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 11\r\nConnection: close\r\n\r\nTLS content')
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(page, '127.0.0.1', 0, ssl=context)
    tls_port = server.sockets[0].getsockname()[1]
    proxy_tasks = set()

    async def proxy(reader, writer):
        task = asyncio.current_task()
        proxy_tasks.add(task)
        outgoing = None
        pumps = []
        try:
            connect_targets.append(await reader.readuntil(b'\r\n\r\n'))
            upstream, outgoing = await asyncio.open_connection('127.0.0.1', tls_port)
            writer.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
            await writer.drain()
            async def pump(source, destination):
                while data := await source.read(16384):
                    destination.write(data)
                    await destination.drain()
            pumps = [asyncio.create_task(pump(reader, outgoing)), asyncio.create_task(pump(upstream, writer))]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for pending in pumps:
                pending.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            writer.close()
            if outgoing:
                outgoing.close()
                await outgoing.wait_closed()
            await writer.wait_closed()
            proxy_tasks.discard(task)

    proxy_server = await asyncio.start_server(proxy, '127.0.0.1', 0) if proxy_enabled else None
    proxy_port = proxy_server.sockets[0].getsockname()[1] if proxy_server else None
    if proxy_server:
        controller.settings.update(proxy_mode='manual', proxy_url=f'http://127.0.0.1:{proxy_port}')
    original_connect = httpcore.AnyIOBackend.connect_tcp
    async def connect(backend, host, port, **kwargs):
        tcp_targets.append((host, port))
        if host == PUBLIC_IP:
            return await original_connect(backend, '127.0.0.1', tls_port, **kwargs)
        assert host == '127.0.0.1' and port == proxy_port
        return await original_connect(backend, host, port, **kwargs)
    monkeypatch.setattr(httpcore.AnyIOBackend, 'connect_tcp', connect)
    original_resolve = asyncio.get_running_loop().getaddrinfo
    async def resolve(host, port, **kwargs):
        if host in ('public.test', 'wrong.test'):
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (PUBLIC_IP, port))]
        assert host in ('127.0.0.1', b'127.0.0.1')
        return await original_resolve(host, port, **kwargs)
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    from modloader import web_search
    original_context = web_search._OriginTLSContext
    def trusted_context(hostname):
        value = original_context(hostname)
        if certificate_case != 'untrusted':
            value.load_verify_locations(cafile=cert_path)
        return value
    monkeypatch.setattr(web_search, '_OriginTLSContext', trusted_context)
    hostname = 'wrong.test' if certificate_case == 'wrong-host' else 'public.test'
    try:
        response = await client.post('/api/web-fetch', json={'url': f'https://{hostname}/page'})
        if certificate_case == 'valid':
            assert response.status_code == 200 and response.json()['content'] == 'TLS content', response.text
            assert b'Host: public.test\r\n' in requests[0]
        else:
            assert response.status_code == 502 and not requests
        assert sni == [hostname]
        assert tcp_targets == [('127.0.0.1', proxy_port)] if proxy_enabled else tcp_targets == [(PUBLIC_IP, 443)]
        if proxy_enabled:
            assert connect_targets[0].startswith(f'CONNECT {PUBLIC_IP}:443 HTTP/1.1\r\n'.encode())
            assert b'public.test' not in connect_targets[0] and b'wrong.test' not in connect_targets[0]
    finally:
        if proxy_server:
            proxy_server.close()
            await proxy_server.wait_closed()
        for task in list(proxy_tasks):
            task.cancel()
        await asyncio.gather(*proxy_tasks, return_exceptions=True)
        server.close()
        await server.wait_closed()


async def test_real_httpcore_http_proxy_receives_pinned_absolute_target_and_original_host(client, controller, monkeypatch):
    requests = []
    async def proxy(reader, writer):
        try:
            requests.append(await reader.readuntil(b'\r\n\r\n'))
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 9\r\nConnection: close\r\n\r\nVia proxy')
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    server = await asyncio.start_server(proxy, '127.0.0.1', 0)
    controller.settings.update(proxy_mode='manual', proxy_url=f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}')
    async def resolve(url, proxy=''):
        return httpx.URL(url), PUBLIC_IP
    monkeypatch.setattr('modloader.web_search._resolve_public_url', resolve)
    try:
        response = await client.post('/api/web-fetch', json={'url': 'http://public.test:8080/path?q=1'})
        assert response.status_code == 200 and response.json()['content'] == 'Via proxy', response.text
        assert requests[0].startswith(f'GET http://{PUBLIC_IP}:8080/path?q=1 HTTP/1.1\r\n'.encode())
        assert b'Host: public.test:8080\r\n' in requests[0]
    finally:
        server.close()
        await server.wait_closed()


def doh_reply(host, kind, addresses=None):
    number = 1 if kind == 'A' else 28
    canonical = 'dualstack.python.map.fastly.net.'
    if addresses is None:
        addresses = ['151.101.64.223', '151.101.192.223'] if kind == 'A' else ['2a04:4e42:600::223', '2a04:4e42:400::223']
    # The CNAME and address layout reproduces the supplied www.python.org Google responses.
    return {'Status': 0, 'TC': False, 'Question': [{'name': host + '.', 'type': number}],
            'Answer': [{'name': host.upper() + '.', 'type': 5, 'TTL': 100, 'data': canonical},
                       *[{'name': canonical, 'type': number, 'TTL': 60, 'data': address} for address in addresses],
                       {'name': canonical, 'type': 46, 'data': 'DNSSEC metadata is not an IP'}]}


@pytest.fixture
async def fake_dns(monkeypatch):
    calls = []
    async def resolve(host, port, **options):
        calls.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('198.18.0.45', port))]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    return calls


@pytest.mark.parametrize('ipv4', [True, False])
async def test_fake_ip_uses_google_a_and_aaaa_through_same_proxy_then_pins_public_ip(client, controller, monkeypatch, fake_dns, ipv4):
    proxy = 'http://proxy.test:17890'
    controller.settings.update(proxy_mode='manual', proxy_url=proxy)
    requests = []
    expected = '151.101.64.223' if ipv4 else '2a04:4e42:600::223'
    def backend(request):
        requests.append(request)
        if request.url.host == 'dns.google':
            assert request.method == 'GET' and request.url.scheme == 'https' and request.url.path == '/resolve'
            assert dict(request.url.params) == {'name': 'www.python.org', 'type': 'A' if len(requests) == 1 else 'AAAA',
                                                'edns_client_subnet': '0.0.0.0/0'}
            kind = request.url.params['type']
            return httpx.Response(200, json=doh_reply('www.python.org', kind, [] if kind == 'A' and not ipv4 else None))
        assert request.url.host == expected and request.headers['host'] == 'www.python.org'
        return httpx.Response(200, headers={'content-type': 'text/plain'}, stream=Chunks(b'Real public page'))
    options = mock_pages(monkeypatch, backend)
    response = await client.post('/api/web-fetch', json={'url': 'https://www.python.org/about/'})
    assert response.status_code == 200 and response.json()['content'] == 'Real public page', response.text
    assert response.json()['url'] == 'https://www.python.org/about/'
    assert fake_dns == [('www.python.org', 443)] and len(requests) == 3
    assert options[0] == {'proxy': proxy, 'trust_env': False, 'verify': True, 'follow_redirects': False, 'timeout': FETCH_TIMEOUT_S}
    assert options[1]['proxy'] == proxy and options[1]['trust_env'] is False
    assert options[1]['verify'].hostname == 'www.python.org' and options[1]['verify'].check_hostname


@pytest.mark.parametrize(('host', 'addresses', 'use_proxy', 'dns_failure'), [
    ('198.18.0.45', [], True, False), ('127.0.0.1', [], True, False),
    ('localhost', ['198.18.0.45'], True, False), ('LOCALHOST.', ['198.18.0.45'], True, False),
    ('child.localhost', ['198.18.0.45'], True, False),
    ('public.test', ['10.0.0.1'], True, False),
    ('public.test', ['198.18.0.45', '127.0.0.1'], True, False),
    ('public.test', [PUBLIC_IP, '198.18.0.45', '10.0.0.1'], True, False),
    ('public.test', ['198.18.0.45'], False, False),
    ('public.test', [], True, True), ('public.test', [], True, False),
])
async def test_fake_ip_doh_does_not_launder_literals_localhost_private_mixtures_or_dns_failure(client, controller, monkeypatch, host, addresses, use_proxy, dns_failure):
    if use_proxy:
        controller.settings.update(proxy_mode='manual', proxy_url='http://proxy.test:17890')
    async def resolve(host, port, **options):
        if dns_failure:
            raise socket.gaierror(socket.EAI_NONAME, 'NXDOMAIN fixture')
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (address, port)) for address in addresses]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    options = mock_pages(monkeypatch, lambda request: pytest.fail('Unsafe or failed system DNS triggered Google DoH'))
    response = await client.post('/api/web-fetch', json={'url': f'http://{host}/'})
    assert response.status_code == (502 if dns_failure else 400), response.text
    assert not options


@pytest.mark.parametrize('failure', ['http', 'redirect', 'json', 'nxdomain', 'truncated', 'question', 'invalid_ip',
    'wrong_family', 'mixed_private', 'private_aaaa', 'fake_answer', 'unrelated', 'cname_loop', 'cname_conflict', 'empty', 'timeout', 'connect'])
async def test_fake_ip_doh_failures_are_explicit_without_partial_results_or_another_resolver(client, controller, monkeypatch, fake_dns, failure):
    controller.settings.update(proxy_mode='manual', proxy_url='http://proxy.test:17890')
    requests = []
    def backend(request):
        requests.append(request)
        assert request.url.host == 'dns.google'
        kind = request.url.params['type']
        if failure == 'timeout':
            raise httpx.ReadTimeout('DoH timeout fixture', request=request)
        if failure == 'connect':
            raise httpx.ConnectError('DoH connection fixture', request=request)
        if failure in ('http', 'redirect'):
            return httpx.Response(500 if failure == 'http' else 302, headers={'location': 'https://other-resolver.test/'})
        if failure == 'json':
            return httpx.Response(200, text='<html>not DNS JSON</html>')
        payload = doh_reply('www.python.org', kind)
        if failure == 'nxdomain':
            payload['Status'] = 3
        elif failure == 'truncated':
            payload['TC'] = True
        elif failure == 'question':
            payload['Question'][0]['name'] = 'unrelated.test.'
        elif failure in ('invalid_ip', 'wrong_family', 'mixed_private', 'fake_answer'):
            data = {'invalid_ip': 'not an address', 'wrong_family': '2606:4700:4700::1111',
                    'mixed_private': '127.0.0.1', 'fake_answer': '198.18.0.46'}[failure]
            payload['Answer'].insert(1, {'name': 'dualstack.python.map.fastly.net.', 'type': 1, 'data': data})
        elif failure == 'private_aaaa' and kind == 'AAAA':
            payload['Answer'][1]['data'] = '::1'
        elif failure == 'unrelated':
            payload['Answer'][1]['name'] = 'unrelated.test.'
        elif failure == 'cname_loop':
            payload['Answer'].append({'name': 'dualstack.python.map.fastly.net.', 'type': 5, 'data': 'www.python.org.'})
        elif failure == 'cname_conflict':
            payload['Answer'].append({'name': 'www.python.org.', 'type': 5, 'data': 'conflicting.test.'})
        elif failure == 'empty':
            payload['Answer'] = []
        return httpx.Response(200, json=payload)
    mock_pages(monkeypatch, backend)
    response = await client.post('/api/web-fetch', json={'url': 'https://www.python.org/'})
    assert response.status_code == (504 if failure == 'timeout' else 502), response.text
    assert 'Google DoH' in response.json()['detail']
    assert len(requests) == (2 if failure in ('private_aaaa', 'empty') else 1)


async def test_fake_ip_doh_redirect_target_is_revalidated_and_private_dns_is_not_retried(client, controller, monkeypatch):
    controller.settings.update(proxy_mode='manual', proxy_url='http://proxy.test:17890')
    dns, requests = [], []
    async def resolve(host, port, **options):
        dns.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('198.18.0.45' if host == 'www.python.org' else '192.168.1.1', port))]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
    def backend(request):
        requests.append(request)
        if request.url.host == 'dns.google':
            assert request.url.params['name'] == 'www.python.org'
            return httpx.Response(200, json=doh_reply('www.python.org', request.url.params['type']))
        assert request.url.host == '151.101.64.223'
        return httpx.Response(302, headers={'location': 'http://private.test/'})
    mock_pages(monkeypatch, backend)
    response = await client.post('/api/web-fetch', json={'url': 'https://www.python.org/'})
    assert response.status_code == 400 and '192.168.1.1' in response.json()['detail']
    assert dns == ['www.python.org', 'private.test'] and len(requests) == 3


@pytest.mark.parametrize('streaming', [False, True])
async def test_fake_ip_doh_disconnect_cancels_lookup_and_closes_proxy_client(controller, monkeypatch, fake_dns, streaming):
    controller.settings.update(proxy_mode='manual', proxy_url='http://proxy.test:17890')
    started, cancelled, closed, disconnect = (asyncio.Event() for _ in range(4))
    class PendingLookup(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            assert request.url.host == 'dns.google'
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def aclose(self):
            closed.set()
    original = httpx.AsyncClient
    monkeypatch.setattr('modloader.web_search.httpx.AsyncClient',
                        lambda **options: original(transport=PendingLookup(), **{key: value for key, value in options.items() if key != 'proxy'}))
    delivered = False
    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': b'{"url":"https://www.python.org/"}', 'more_body': False}
        await disconnect.wait()
        return {'type': 'http.disconnect'}
    async def send(event):
        pass
    headers = [(b'host', b'testserver'), (b'authorization', b'Bearer web-fetch-fixture'), (b'content-type', b'application/json')]
    if streaming:
        headers.append((b'accept', b'application/x-ndjson'))
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'http_version': '1.1',
             'method': 'POST', 'scheme': 'http', 'path': '/api/web-fetch', 'raw_path': b'/api/web-fetch',
             'query_string': b'', 'headers': headers, 'client': ('127.0.0.1', 1), 'server': ('testserver', 80)}
    task = asyncio.create_task(build_app(controller)(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), 2)
        disconnect.set()
        await asyncio.wait_for(task, 2)
        assert cancelled.is_set() and closed.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_fake_ip_doh_is_inside_the_existing_total_fetch_deadline(client, controller, monkeypatch, fake_dns):
    from modloader import web_search
    controller.settings.update(proxy_mode='manual', proxy_url='http://proxy.test:17890')
    monkeypatch.setattr(web_search, 'FETCH_TIMEOUT_S', .1)
    cancelled, closed = asyncio.Event(), asyncio.Event()
    class PendingLookup(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            assert request.url.host == 'dns.google'
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def aclose(self):
            closed.set()
    original = httpx.AsyncClient
    monkeypatch.setattr('modloader.web_search.httpx.AsyncClient',
                        lambda **options: original(transport=PendingLookup(), **{key: value for key, value in options.items() if key != 'proxy'}))
    response = await client.post('/api/web-fetch', json={'url': 'https://www.python.org/'})
    assert response.status_code == 504 and cancelled.is_set() and closed.is_set()
