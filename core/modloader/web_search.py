"""Request-scoped web search and bounded public-page reading for management chat."""
from __future__ import annotations

import asyncio
import ipaddress
import math
import os
import re
import socket
import ssl
import time
import uuid
from urllib.parse import urlsplit, urlunsplit
import zlib

import certifi
import httpx
from bs4 import BeautifulSoup, Comment
from fastapi import HTTPException


DEFAULT_SEARXNG_URL = 'http://127.0.0.1:8888'
DEFAULT_IMAGE_ENGINE = 'bdi'
SEARCH_TIMEOUT_S = 30
FETCH_TIMEOUT_S = 30
FETCH_MAX_REDIRECTS = 5
FETCH_DEFAULT_MAX_BYTES = 4 * 1024 * 1024
FAKE_IP_NETWORK = ipaddress.ip_network('198.18.0.0/15')
GOOGLE_DOH_URL = 'https://dns.google/resolve'


def validate_http_url(value, field='地址'):
    if not isinstance(value, str) or not value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f'{field}需要有效的 HTTP(S) URL')
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in ('http', 'https') and bool(parsed.hostname)
        parsed.port  # Reject malformed or out-of-range ports before making a request.
        httpx.URL(value)
    except (ValueError, httpx.InvalidURL):
        valid = False
    if not valid:
        raise ValueError(f'{field}需要有效的 HTTP(S) URL')
    return value


def _plain_text(value):
    if not isinstance(value, str):
        return ''
    soup = BeautifulSoup(value, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    return ' '.join(soup.get_text(' ', strip=True).split())


def validate_image_engine(value):
    if not isinstance(value, str) or any(character.isspace() or ord(character) < 32 or character == '!' for character in value):
        raise ValueError('图片引擎需要 SearXNG 简称，不含 ! 或空白；留空使用服务默认引擎')
    return value


def _elapsed(value, field):
    if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)) or value < 0:
        raise ValueError(f'{field} 必须为非负有限数值')


def _positive_integer(value, field):
    if type(value) is not int or value < 1:
        raise ValueError(f'{field} 必须为正整数')


class PageRangeError(ValueError):
    """A continuation cannot switch readers after discovering its line range is invalid."""


def _validate_routing(value):
    if not isinstance(value, dict) or set(value) != {'provider', 'label', 'attempts'}:
        raise ValueError('联网路由字段不完整或包含不支持的字段')
    for field in ('provider', 'label'):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError('联网路由 provider 和 label 必须为非空字符串')
    attempts = value['attempts']
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('联网路由 attempts 必须为非空渠道数组')
    seen = set()
    for index, attempt in enumerate(attempts):
        fields = {'provider', 'label'} | ({'error'} if index < len(attempts) - 1 else set())
        if not isinstance(attempt, dict) or set(attempt) != fields:
            raise ValueError('联网路由尝试字段无效')
        for field in fields:
            if not isinstance(attempt[field], str) or not attempt[field].strip():
                raise ValueError('联网路由尝试字段必须为非空字符串')
        if attempt['provider'] in seen:
            raise ValueError('联网路由渠道不可重复尝试')
        seen.add(attempt['provider'])
    if attempts[-1]['provider'] != value['provider'] or attempts[-1]['label'] != value['label']:
        raise ValueError('联网路由结果与成功渠道不一致')


def validate_search_result(value):
    """Validate saved provenance without changing its citation IDs or content."""
    required = {'query', 'sources', 'elapsed_ms'}
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - {'mode', 'queries', 'pages', 'images', 'summaries', 'routing'}:
        raise ValueError('webSearch 字段不完整或包含不支持的字段')
    if not isinstance(value['query'], str):
        raise ValueError('webSearch.query 必须为字符串')
    _elapsed(value['elapsed_ms'], 'webSearch.elapsed_ms')
    if 'routing' in value:
        _validate_routing(value['routing'])
    if 'mode' in value and value['mode'] not in ('agent', 'retrieve'):
        raise ValueError('webSearch.mode 必须为 agent 或 retrieve')
    if 'queries' in value and (not isinstance(value['queries'], list) or any(not isinstance(query, str) for query in value['queries'])):
        raise ValueError('webSearch.queries 必须为字符串数组')
    if not isinstance(value['sources'], list):
        raise ValueError('webSearch.sources 必须为来源数组')
    for identifier, source in enumerate(value['sources'], 1):
        if not isinstance(source, dict) or set(source) != {'id', 'title', 'url', 'snippet'}:
            raise ValueError('搜索来源字段不完整或包含不支持的字段')
        if type(source['id']) is not int or source['id'] != identifier:
            raise ValueError('搜索来源 id 必须从 1 连续编号')
        for field in ('title', 'snippet'):
            if not isinstance(source[field], str):
                raise ValueError(f'搜索来源 {field} 必须为字符串')
        validate_http_url(source['url'], '搜索来源地址')
    if 'summaries' in value:
        if not isinstance(value['summaries'], list):
            raise ValueError('webSearch.summaries 必须为检索摘要数组')
        for summary in value['summaries']:
            if not isinstance(summary, dict) or not {'text', 'source_ids'} <= summary.keys() or summary.keys() - {'text', 'source_ids', 'url'}:
                raise ValueError('检索摘要字段不完整或包含不支持的字段')
            if not isinstance(summary['text'], str):
                raise ValueError('检索摘要 text 必须为字符串')
            if 'url' in summary:
                validate_http_url(summary['url'], '阅读摘要地址')
                if '@' in urlsplit(summary['url']).netloc:
                    raise ValueError('阅读摘要地址不可包含凭据')
            if not isinstance(summary['source_ids'], list):
                raise ValueError('检索摘要 source_ids 必须为来源编号数组')
            for identifier in summary['source_ids']:
                _positive_integer(identifier, '检索摘要 source_id')
                if identifier > len(value['sources']):
                    raise ValueError('检索摘要 source_id 未关联搜索来源')
    if 'images' in value:
        if not isinstance(value['images'], list):
            raise ValueError('webSearch.images 必须为图片数组')
        fields = {'source_id', 'title', 'url'}
        for image in value['images']:
            if not isinstance(image, dict) or not fields <= image.keys() or image.keys() - fields - {'thumbnail_url'}:
                raise ValueError('搜索图片字段不完整或包含不支持的字段')
            _positive_integer(image['source_id'], '图片 source_id')
            if image['source_id'] > len(value['sources']):
                raise ValueError('图片 source_id 未关联搜索来源')
            if not isinstance(image['title'], str):
                raise ValueError('图片 title 必须为字符串')
            validate_http_url(image['url'], '图片地址')
            if 'thumbnail_url' in image:
                validate_http_url(image['thumbnail_url'], '缩略图地址')
    if 'pages' in value:
        if not isinstance(value['pages'], list):
            raise ValueError('webSearch.pages 必须为已读网页数组')
        fields = {'source_id', 'url', 'title', 'content', 'start_line', 'end_line', 'total_lines', 'elapsed_ms'}
        for page in value['pages']:
            if not isinstance(page, dict) or set(page) != fields:
                raise ValueError('已读网页字段不完整或包含不支持的字段')
            _positive_integer(page['source_id'], '网页 source_id')
            if page['source_id'] > len(value['sources']):
                raise ValueError('已读网页 source_id 未关联搜索来源')
            validate_http_url(page['url'], '网页地址')
            for field in ('title', 'content'):
                if not isinstance(page[field], str):
                    raise ValueError(f'网页 {field} 必须为字符串')
            for field in ('start_line', 'end_line', 'total_lines'):
                _positive_integer(page[field], '网页 ' + field)
            if not page['start_line'] <= page['end_line'] <= page['total_lines']:
                raise ValueError('已读网页行范围无效')
            _elapsed(page['elapsed_ms'], '网页 elapsed_ms')


async def search_web(query, searxng_url, proxy='', *, category='general', image_engine=DEFAULT_IMAGE_ENGINE, progress=lambda event: None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('搜索关键词不能为空')
    query = query.strip()
    if category not in ('general', 'images'):
        raise ValueError('搜索类别必须为 general 或 images')
    if category == 'images':
        validate_image_engine(image_engine)
    upstream_query = f'!{image_engine} {query}' if category == 'images' and image_engine else query
    parsed = urlsplit(validate_http_url(searxng_url, 'SearXNG 地址'))
    path = parsed.path.rstrip('/')
    endpoint = urlunsplit(parsed._replace(path=path if path.endswith('/search') else path + '/search', fragment=''))
    started = time.monotonic()
    progress({'stage': 'web_search', 'message': '正在搜索图片' if category == 'images' else '正在搜索网页'})
    try:
        async with asyncio.timeout(SEARCH_TIMEOUT_S):
            async with httpx.AsyncClient(proxy=proxy or None, trust_env=False, timeout=SEARCH_TIMEOUT_S) as client:
                response = await client.post(endpoint, data={'q': upstream_query, 'format': 'json', 'categories': category})
                if response.status_code == 403:
                    raise HTTPException(502, 'SearXNG 拒绝搜索（HTTP 403），请检查 search.formats 是否启用 json')
                if not response.is_success:
                    raise HTTPException(502, f'SearXNG 搜索失败（HTTP {response.status_code}）')
                try:
                    payload = response.json()
                except ValueError as error:
                    raise HTTPException(502, 'SearXNG 未返回有效 JSON，请检查 search.formats 是否启用 json') from error
    except (httpx.TimeoutException, TimeoutError) as error:
        raise HTTPException(504, 'SearXNG 搜索超时，请重试或检查服务') from error
    except httpx.HTTPError as error:
        raise HTTPException(502, '无法连接 SearXNG，请检查服务地址和代理设置') from error
    if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
        raise HTTPException(502, 'SearXNG 响应缺少有效的 results 数组')
    sources, images = normalize_search_items(payload['results'], images=category == 'images')
    if not sources:
        if category == 'images':
            failures = payload.get('unresponsive_engines')
            if isinstance(failures, list) and failures:
                details = '；'.join(f'{_plain_text(item[0])}：{_plain_text(item[1])}' for item in failures
                                    if isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str) and isinstance(item[1], str))
                if details:
                    raise HTTPException(502, f'SearXNG 图片引擎未返回可用结果（{details}）')
            raise HTTPException(422, '未找到可用的图片结果，请调整关键词或检查 SearXNG 图片引擎')
        raise HTTPException(422, '未找到可用的搜索结果，请调整关键词后重试')
    return {'query': query, 'sources': sources, 'elapsed_ms': round((time.monotonic() - started) * 1000),
            **({'images': images} if category == 'images' else {})}


def normalize_search_items(items, *, images=False):
    """Normalize source/image rows without changing citation order or imposing a result cap."""
    image_search = images
    sources, source_ids, images, seen_images = [], {}, [], set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('url'), str):
            continue
        url = item['url'].strip()
        try:
            validate_http_url(url, '搜索来源地址')
            image_url = validate_http_url(item.get('img_src', '').strip(), '图片地址') if image_search and isinstance(item.get('img_src'), str) else ''
        except ValueError:
            continue
        if image_search and not image_url:
            continue
        key = url.split('#', 1)[0]
        if key not in source_ids:
            source_ids[key] = len(sources) + 1
            sources.append({'id': source_ids[key], 'title': _plain_text(item.get('title')),
                            'url': url, 'snippet': _plain_text(item.get('content'))})
        if image_url and image_url not in seen_images:
            seen_images.add(image_url)
            image = {'source_id': source_ids[key], 'title': _plain_text(item.get('title')), 'url': image_url}
            thumbnail = item.get('thumbnail_src')
            if isinstance(thumbnail, str):
                try:
                    image['thumbnail_url'] = validate_http_url(thumbnail.strip(), '缩略图地址')
                except ValueError:
                    pass  # A missing/unsafe thumbnail does not discard the original image.
            images.append(image)
    return sources, images


def _public_ip(value):
    address = ipaddress.ip_address(value)
    return (address.is_global and not (address.is_multicast or address.is_reserved or address.is_unspecified)
            and (not isinstance(address, ipaddress.IPv6Address) or not address.ipv4_mapped or _public_ip(address.ipv4_mapped)))


def _dns_name(value):
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise HTTPException(502, 'Google DoH 返回无效的 DNS 名称')
    return value.removesuffix('.').lower()


def _doh_addresses(payload, host, record_type):
    if not isinstance(payload, dict) or type(payload.get('Status')) is not int:
        raise HTTPException(502, 'Google DoH 返回无效的 DNS 响应')
    if payload['Status'] != 0:
        raise HTTPException(502, f'Google DoH 解析失败（DNS Status {payload["Status"]}）')
    if payload.get('TC') is not False:
        raise HTTPException(502, 'Google DoH 响应被截断或缺少 TC 状态')
    question = payload.get('Question')
    if (not isinstance(question, list) or len(question) != 1 or not isinstance(question[0], dict)
            or _dns_name(question[0].get('name')) != _dns_name(host) or type(question[0].get('type')) is not int
            or question[0]['type'] != record_type):
        raise HTTPException(502, 'Google DoH 响应与查询不匹配')
    answers = payload.get('Answer', [])
    if not isinstance(answers, list):
        raise HTTPException(502, 'Google DoH 返回无效的 Answer 数组')
    aliases, addresses = {}, []
    for answer in answers:
        if not isinstance(answer, dict) or type(answer.get('type')) is not int:
            raise HTTPException(502, 'Google DoH 返回无效的 DNS 记录')
        kind = answer['type']
        if kind not in (1, 5, 28):
            continue  # DNSSEC and other RR types are not address records.
        name = _dns_name(answer.get('name'))
        if kind == 5:
            target = _dns_name(answer.get('data'))
            if name in aliases and aliases[name] != target:
                raise HTTPException(502, 'Google DoH 返回冲突的 CNAME 记录')
            aliases[name] = target
            continue
        try:
            address = ipaddress.ip_address(answer.get('data')) if isinstance(answer.get('data'), str) else None
        except ValueError:
            address = None
        if address is None or address.version != (4 if kind == 1 else 6):
            raise HTTPException(502, 'Google DoH 返回无效的 A/AAAA 地址')
        if not _public_ip(address):
            raise HTTPException(502, f'Google DoH 返回非公网 IP：{address}，无法读取')
        addresses.append((name, kind, str(address)))
    canonical, seen = _dns_name(host), set()
    while canonical in aliases:
        if canonical in seen:
            raise HTTPException(502, 'Google DoH 返回循环 CNAME 记录')
        seen.add(canonical)
        canonical = aliases[canonical]
    if any(name != canonical for name, kind, address in addresses):
        raise HTTPException(502, 'Google DoH 地址记录不属于查询域名的 CNAME 链')
    return [address for name, kind, address in addresses if kind == record_type]


async def _resolve_fake_ip(host, proxy):
    # This path is only entered for system Fake-IP answers and an explicitly configured proxy.
    # Google documents separate A/AAAA queries; ANY need not contain either address family.
    addresses = []
    try:
        async with httpx.AsyncClient(proxy=proxy, trust_env=False, verify=True, follow_redirects=False,
                                     timeout=FETCH_TIMEOUT_S) as client:
            for kind, number in (('A', 1), ('AAAA', 28)):
                response = await client.get(GOOGLE_DOH_URL, params={'name': host, 'type': kind,
                                             'edns_client_subnet': '0.0.0.0/0'})
                if not response.is_success:
                    raise HTTPException(502, f'Google DoH 解析失败（HTTP {response.status_code}）')
                try:
                    payload = response.json()
                except ValueError as error:
                    raise HTTPException(502, 'Google DoH 未返回有效 JSON') from error
                addresses.extend(_doh_addresses(payload, host, number))
    except httpx.TimeoutException as error:
        raise HTTPException(504, 'Google DoH 解析超时，请检查当前代理') from error
    except httpx.HTTPError as error:
        raise HTTPException(502, 'Google DoH 连接失败，请检查当前代理') from error
    if not addresses:
        raise HTTPException(502, 'Google DoH 未返回可用的公网 A/AAAA 地址')
    return list(dict.fromkeys(addresses))


async def _resolve_public_url(value, proxy=''):
    validate_http_url(value, '网页地址')
    url = httpx.URL(value).copy_with(fragment=None)
    # Also reject an empty userinfo component (http://@host), not just nonempty passwords.
    if url.username or url.password or '@' in urlsplit(value).netloc:
        raise ValueError('网页地址不可包含凭据')
    host = url.raw_host.decode('ascii')
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [str(literal)]
    else:
        try:
            results = await asyncio.get_running_loop().getaddrinfo(host, url.port or (443 if url.scheme == 'https' else 80), type=socket.SOCK_STREAM)
        except OSError as error:
            raise HTTPException(502, '网页地址无法解析') from error
        addresses = list(dict.fromkeys(item[4][0] for item in results))
    rejected = [address for address in addresses if not _public_ip(address)]
    if not addresses:
        raise ValueError(f'DNS 解析 {host} 未返回地址，无法读取网页')
    if rejected:
        if literal is not None:
            raise ValueError(f'网页地址使用非公网 IP：{rejected[0]}，无法读取')
        localhost = host.removesuffix('.').lower() == 'localhost' or host.removesuffix('.').lower().endswith('.localhost')
        if proxy and not localhost and all(ipaddress.ip_address(address) in FAKE_IP_NETWORK for address in rejected):
            return url, (await _resolve_fake_ip(host, proxy))[0]
        detail = f'DNS 解析 {host} 返回非公网 IP：{", ".join(rejected)}，无法读取'
        if any(ipaddress.ip_address(address) in FAKE_IP_NETWORK for address in rejected):
            detail += '（Fake-IP，需真实公网 DNS 解析）'
        raise ValueError(detail)
    return url, addresses[0]


class _OriginTLSContext(ssl.SSLContext):
    """Verify the original hostname while connecting (or proxy-CONNECTing) to its pinned IP.

    httpcore's HTTPS proxy tunnel does not honor the sni_hostname extension.
    wrap_bio is the TLS boundary used by its AnyIO backend, including proxy tunnels.
    """
    def __new__(cls, hostname):
        return super().__new__(cls, ssl.PROTOCOL_TLS_CLIENT)

    def __init__(self, hostname):
        self.hostname = hostname
        self.load_verify_locations(cafile=certifi.where())

    def wrap_bio(self, incoming, outgoing, server_side=False, server_hostname=None, session=None):
        return super().wrap_bio(incoming, outgoing, server_side=server_side, server_hostname=self.hostname, session=session)


def _fetch_max_bytes():
    try:
        value = int(os.environ.get('MODLOADER_WEB_FETCH_MAX_BYTES', FETCH_DEFAULT_MAX_BYTES))
        if value < 1:
            raise ValueError()
        return value
    except ValueError as error:
        raise HTTPException(500, 'MODLOADER_WEB_FETCH_MAX_BYTES 必须为正整数') from error


async def _read_page_body(response, max_bytes):
    """Bound both transferred and expanded bytes, including compressed responses."""
    length = response.headers.get('content-length', '')
    if length.isdecimal() and int(length) > max_bytes:
        raise HTTPException(413, f'网页超过读取预算（{max_bytes} 字节）')
    encoding = response.headers.get('content-encoding', '').strip().lower()
    if encoding not in ('', 'identity', 'gzip', 'deflate'):
        raise HTTPException(415, f'不支持网页响应的内容编码：{encoding}')
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == 'gzip' else zlib.MAX_WBITS) if encoding in ('gzip', 'deflate') else None
    body, transferred = bytearray(), 0
    try:
        async for chunk in response.aiter_raw():
            transferred += len(chunk)
            if transferred > max_bytes:
                raise HTTPException(413, f'网页超过读取预算（{max_bytes} 字节）')
            decoded = decoder.decompress(chunk, max_bytes - len(body) + 1) if decoder else chunk
            body.extend(decoded)
            if len(body) > max_bytes:
                raise HTTPException(413, f'网页超过读取预算（{max_bytes} 字节）')
            if decoder and decoder.unused_data:
                raise HTTPException(502, '网页压缩响应包含多余数据')
        if decoder and not decoder.eof:
            raise HTTPException(502, '网页压缩响应不完整')
    except zlib.error as error:
        raise HTTPException(502, '网页压缩响应无效') from error
    return bytes(body)


def _document_lines(body, mime, encoding):
    if mime not in ('text/html', 'application/xhtml+xml'):
        return '', body.decode(encoding or 'utf-8', errors='replace').splitlines()
    soup = BeautifulSoup(body, 'html.parser', from_encoding=encoding)
    title = ' '.join(soup.title.stripped_strings) if soup.title else ''
    for tag in soup.select('head, script, style, noscript, nav, header, footer, form, iframe, svg, canvas, template, button, input, select, textarea, [hidden], [aria-hidden="true"]'):
        tag.decompose()
    root = soup.find('main') or soup.find('article') or soup.body or soup
    for node in list(root.find_all(string=True)):
        if isinstance(node, Comment):
            node.extract()
        elif not node.find_parent('pre'):
            node.replace_with(re.sub(r'\s+', ' ', str(node)))
    code_blocks = {}
    for pre in root.find_all('pre'):
        marker = uuid.uuid4().hex
        code_blocks[marker] = pre.get_text().splitlines()
        pre.replace_with('\n' + marker + '\n')
    for br in root.find_all('br'):
        br.replace_with('\n')
    for tag in root.find_all(('p', 'div', 'section', 'article', 'main', 'blockquote', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr')):
        tag.insert_before('\n')
        tag.insert_after('\n')
    for cell in root.find_all(('td', 'th')):
        cell.insert_after(' ')
    lines = []
    for line in root.get_text().splitlines():
        line = line.strip()
        if line in code_blocks:
            lines.extend(code_blocks[line])
        elif line:
            lines.append(line)
    return title, lines


async def fetch_page(url, start_line=1, line_count=80, proxy='', *, progress=lambda event: None):
    _positive_integer(start_line, 'start_line')
    _positive_integer(line_count, 'line_count')
    max_bytes = _fetch_max_bytes()
    started = time.monotonic()
    progress({'stage': 'web_fetch', 'message': '正在读取网页'})
    try:
        async with asyncio.timeout(FETCH_TIMEOUT_S):
            current = url
            for redirects in range(FETCH_MAX_REDIRECTS + 1):
                original, address = await _resolve_public_url(current, proxy)
                pinned = original.copy_with(host=address)
                context = _OriginTLSContext(original.raw_host.decode('ascii'))
                async with httpx.AsyncClient(proxy=proxy or None, trust_env=False, follow_redirects=False,
                                             timeout=FETCH_TIMEOUT_S, verify=context) as client:
                    async with client.stream('GET', pinned, headers={'Host': original.netloc.decode('ascii'),
                                             'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.9,text/*;q=0.8',
                                             'Accept-Encoding': 'identity'}) as response:
                        if response.is_redirect:
                            location = response.headers.get('location')
                            if not location:
                                raise HTTPException(502, '网页重定向缺少 Location')
                            if redirects == FETCH_MAX_REDIRECTS:
                                raise HTTPException(502, f'网页重定向超过 {FETCH_MAX_REDIRECTS} 次')
                            try:
                                current = str(original.join(location))
                            except httpx.InvalidURL as error:
                                raise HTTPException(502, '网页重定向地址无效') from error
                            continue
                        if not response.is_success:
                            raise HTTPException(502, f'网页读取失败（HTTP {response.status_code}）')
                        mime = response.headers.get('content-type', '').split(';', 1)[0].strip().lower()
                        if not mime.startswith('text/') and mime != 'application/xhtml+xml':
                            raise HTTPException(415, '网页读取仅支持 HTML 或文本，不支持 PDF 和二进制文件')
                        body = await _read_page_body(response, max_bytes)
                        # Let HTML's charset declaration participate when no HTTP charset is supplied.
                        encoding = response.encoding if response.charset_encoding else None
                        title, lines = await asyncio.to_thread(_document_lines, body, mime, encoding)
                        return page_window(str(original), title, lines, start_line, line_count, started)
    except (httpx.TimeoutException, TimeoutError) as error:
        raise HTTPException(504, '网页读取超时，请重试') from error
    except httpx.HTTPError as error:
        raise HTTPException(502, '网页连接失败，请检查网址、TLS 证书或代理设置') from error


def page_window(url, title, lines, start_line, line_count, started):
    if not lines or not any(line.strip() for line in lines):
        raise HTTPException(422, '网页没有可读取的正文')
    if start_line > len(lines):
        raise PageRangeError(f'start_line 超出全文范围（共 {len(lines)} 行）')
    end_line = min(len(lines), start_line + line_count - 1)
    return {'url': url, 'title': title, 'content': '\n'.join(lines[start_line - 1:end_line]),
            'start_line': start_line, 'end_line': end_line, 'total_lines': len(lines),
            'elapsed_ms': round((time.monotonic() - started) * 1000)}
