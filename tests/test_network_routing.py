"""Configured sequential failover and explicitly generated OpenAI reading summaries."""
import asyncio
import copy
import json
import uuid

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from modloader.app import Controller
from modloader.network_services import DEFAULT_NETWORK_SETTINGS, run_network_route
from modloader.operation_progress import operation_response
from modloader.web_search import validate_search_result
from test_network_services import client, controller, mock_api, result_of
from test_openai_search import citation, message, output_text, response, search_call


PREFIX = {'search': 'web_search', 'images': 'web_image', 'fetch': 'web_fetch'}


def route_result(capability):
    if capability == 'fetch':
        return {'kind': 'summary', 'url': 'https://source.test/', 'sources': [],
                'summaries': [{'text': '摘要', 'source_ids': [], 'url': 'https://source.test/'}], 'elapsed_ms': 1}
    return {'query': '关键词', 'sources': [{'id': 1, 'url': 'https://source.test/', 'title': '来源', 'snippet': ''}],
            'elapsed_ms': 1, **({'images': [{'source_id': 1, 'url': 'https://images.test/image.jpg', 'title': '图片'}]} if capability == 'images' else {})}


async def test_fallback_settings_preserve_order_allow_primary_and_persist(client, controller):
    assert {field: controller.settings[field] for field in ('web_search_fallbacks', 'web_image_fallbacks', 'web_fetch_fallbacks')} == {
        'web_search_fallbacks': [], 'web_image_fallbacks': [], 'web_fetch_fallbacks': []}
    patch = {'web_search_fallbacks': ['deepseek', 'openai', 'searxng'], 'web_image_fallbacks': ['openai', 'brave'],
             'web_fetch_fallbacks': ['tavily', 'local', 'openai'], 'openai_read_model': 'custom/阅读模型-' + 'x' * 1024}
    saved = await client.patch('/api/settings', json=patch)
    assert saved.status_code == 200, saved.text
    assert all(saved.json()[key] == value for key, value in patch.items())
    assert saved.json()['openai_search_model'] == 'gpt-4.1-mini'
    assert DEFAULT_NETWORK_SETTINGS['openai_read_model'] == 'gpt-5.4-mini'
    independent = await client.patch('/api/settings', json={'web_search_provider': 'openai', 'web_fetch_provider': 'openai'})
    assert independent.status_code == 200
    assert independent.json()['web_search_fallbacks'] == patch['web_search_fallbacks']
    assert independent.json()['web_fetch_fallbacks'] == patch['web_fetch_fallbacks']
    assert next(service for service in independent.json()['web_services'] if service['id'] == 'openai')['fetch_label'] == 'OpenAI 阅读摘要'
    reopened = Controller(controller.paths.root, 'reopened')
    try:
        assert all(reopened.settings[key] == value for key, value in patch.items())
    finally:
        await reopened.close()


@pytest.mark.parametrize('patch', [
    {'web_search_fallbacks': 'openai'}, {'web_search_fallbacks': None}, {'web_search_fallbacks': ['openai', 'openai']},
    {'web_search_fallbacks': ['local']}, {'web_image_fallbacks': ['deepseek']}, {'web_fetch_fallbacks': ['brave']},
    {'web_fetch_fallbacks': [None]}, {'web_image_fallbacks': [[]]}, {'web_search_fallbacks': ['unknown']},
    {'openai_read_model': ''}, {'openai_read_model': None}, {'openai_read_model': ' \n'},
])
async def test_invalid_fallback_settings_reject_entire_patch(client, controller, patch):
    before = copy.deepcopy(controller.settings)
    saved = await client.patch('/api/settings', json={**patch, 'font_scale': 1.5})
    assert saved.status_code == 400, saved.text
    assert controller.settings == before and controller.store.get('settings', 'main') is None


@pytest.mark.parametrize('capability', ['search', 'images', 'fetch'])
@pytest.mark.parametrize('streaming', [False, True])
async def test_each_route_tries_configured_order_stops_on_success_and_counts_total_time(client, controller, monkeypatch, capability, streaming):
    prefix = PREFIX[capability]
    primary = 'brave' if capability == 'images' else 'deepseek'
    controller.settings.update({prefix + '_provider': primary, prefix + '_fallbacks': [primary, 'openai', 'tavily']})
    calls = []

    async def service(provider, kind, settings, get_credential, proxy, **arguments):
        calls.append((provider, kind))
        if provider == primary:
            await asyncio.sleep(.02)
            raise HTTPException(504, '请求超时')
        assert provider == 'openai'
        await asyncio.sleep(.03)
        return route_result(capability)

    monkeypatch.setattr('modloader.network_services.run_network_service', service)
    endpoint = '/api/web-fetch' if capability == 'fetch' else '/api/web-search'
    body = {'url': 'https://source.test/', 'query': '关键词', 'category': 'images' if capability == 'images' else 'general'}
    raw = await client.post(endpoint, json=body, headers={'Accept': 'application/x-ndjson'} if streaming else {})
    if streaming:
        events = [json.loads(line) for line in raw.text.splitlines()]
        assert any(item.get('stage') == 'web_fallback' and '切换至' in item['message'] for item in events)
        assert events[-1]['type'] == 'result', events
        value = events[-1]['value']
    else:
        assert raw.status_code == 200, raw.text
        value = raw.json()
    assert calls == [(primary, capability), ('openai', capability)]
    assert value['elapsed_ms'] >= 45
    assert value['routing'] == {'provider': 'openai', 'label': 'OpenAI 阅读摘要' if capability == 'fetch' else 'OpenAI',
        'attempts': [{'provider': primary, 'label': 'Brave Search' if primary == 'brave' else 'DeepSeek', 'error': '请求超时'},
                     {'provider': 'openai', 'label': 'OpenAI 阅读摘要' if capability == 'fetch' else 'OpenAI'}]}
    if capability != 'fetch':
        validate_search_result(value)


@pytest.mark.parametrize('capability', ['search', 'images', 'fetch'])
async def test_operation_stream_keeps_fallback_reason_while_backup_is_pending(controller, monkeypatch, capability):
    prefix = PREFIX[capability]
    primary = 'brave' if capability == 'images' else 'deepseek'
    controller.settings.update({prefix + '_provider': primary, prefix + '_fallbacks': ['openai']})
    pending, next_stage, release, received = (asyncio.Event() for _ in range(4))
    events = []

    async def service(provider, kind, settings, get_credential, proxy, *, progress, **arguments):
        if provider == primary:
            raise HTTPException(503, '服务暂不可用（HTTP 503）')
        assert provider == 'openai'
        # These updates occur in the same tick as the route's transition event.
        stage = 'web_fetch' if kind == 'fetch' else 'web_search'
        progress({'stage': stage, 'message': '正在读取网页' if kind == 'fetch' else '正在搜索网页'})
        pending.set()
        await next_stage.wait()
        progress({'stage': stage, 'message': '正在处理结果'})
        await release.wait()
        return route_result(capability)

    monkeypatch.setattr('modloader.network_services.run_network_service', service)
    request = Request({'type': 'http', 'headers': [(b'accept', b'application/x-ndjson')]})

    async def perform(progress):
        return await run_network_route(capability, controller.settings, controller.vault.get,
                                       query='关键词', url='https://source.test/', progress=progress)

    stream = await operation_response(request, perform, cancel_on_disconnect=True)

    async def consume():
        async for chunk in stream.body_iterator:
            event = json.loads(chunk)
            events.append(event)
            if event['type'] == 'progress':
                received.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(pending.wait(), 2)
        await asyncio.wait_for(received.wait(), 2)
        first = events[-1]
        assert first['type'] == 'progress' and first['stage'] == 'web_fallback'
        assert '503' in first['message'] and '切换至 OpenAI' in first['message']
        assert not task.done() and not release.is_set()
        received.clear()
        next_stage.set()
        await asyncio.wait_for(received.wait(), 2)
        assert events[-1] == first  # Later backup progress retains the same reason and target.
        release.set()
        await asyncio.wait_for(task, 2)
        assert events[-1]['type'] == 'result' and events[-1]['value']['routing']['provider'] == 'openai'
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('capability', ['search', 'images', 'fetch'])
async def test_successful_primary_never_calls_backups_and_no_backups_keep_old_shape(controller, monkeypatch, capability):
    primary = 'local' if capability == 'fetch' else 'searxng'
    prefix = PREFIX[capability]
    calls = []

    async def service(provider, kind, settings, get_credential, proxy, **arguments):
        calls.append(provider)
        return route_result(capability)

    monkeypatch.setattr('modloader.network_services.run_network_service', service)
    arguments = {'url': 'https://source.test/'} if capability == 'fetch' else {'query': '关键词'}
    controller.settings.update({prefix + '_provider': primary, prefix + '_fallbacks': []})
    assert await run_network_route(capability, controller.settings, controller.vault.get, **arguments) == route_result(capability)
    controller.settings[prefix + '_fallbacks'] = [primary, 'openai']
    result = await run_network_route(capability, controller.settings, controller.vault.get, **arguments)
    assert result['routing']['attempts'] == [{'provider': primary, 'label': '本地读取' if primary == 'local' else 'SearXNG'}]
    assert calls == [primary, primary]


@pytest.mark.parametrize('failure', ['missing_key', 'query_limit', 'timeout'])
async def test_deepseek_configuration_limit_or_timeout_can_fall_back_to_openai(client, controller, monkeypatch, failure):
    controller.settings.update(web_search_provider='deepseek', web_search_fallbacks=['openai'])
    controller.vault.set('search_openai', 'fixture-api-key')
    if failure != 'missing_key':
        controller.vault.set('search_deepseek', 'fixture-api-key')
    calls = []

    def backend(request):
        calls.append(request.url.host)
        if request.url.host == 'api.deepseek.com':
            raise httpx.ReadTimeout('fixture-sensitive-key remote body', request=request)
        assert request.url.path == '/v1/responses'
        return httpx.Response(200, json=response(search_call([{'type': 'url', 'url': 'https://source.test/'}])))

    mock_api(monkeypatch, backend)
    value = result_of(await client.post('/api/web-search', json={'query': '中' * 129 if failure == 'query_limit' else 'test'}))
    assert calls == (['api.deepseek.com', 'api.openai.com'] if failure == 'timeout' else ['api.openai.com'])
    assert [attempt['provider'] for attempt in value['routing']['attempts']] == ['deepseek', 'openai']
    assert 'fixture-sensitive-key' not in json.dumps(value)
    assert not controller.runtime.instances and not controller.store.all('jobs')


async def test_route_snapshots_settings_and_does_not_add_implicit_backups(controller, monkeypatch):
    controller.settings.update(web_search_provider='deepseek', web_search_fallbacks=['openai'], openai_search_model='original-model')
    calls = []

    async def service(provider, capability, settings, get_credential, proxy, **arguments):
        calls.append(provider)
        if provider == 'deepseek':
            controller.settings.update(web_search_fallbacks=['tavily'], openai_search_model='edited-during-request')
            raise HTTPException(502, '连接失败')
        assert provider == 'openai' and settings['openai_search_model'] == 'original-model'
        return route_result('search')

    monkeypatch.setattr('modloader.network_services.run_network_service', service)
    result = await run_network_route('search', controller.settings, controller.vault.get, query='test')
    assert calls == ['deepseek', 'openai'] and result['routing']['provider'] == 'openai'


async def test_all_failures_are_aggregated_once_and_explicit_service_tests_do_not_fallback(client, controller, monkeypatch):
    controller.settings.update(web_search_provider='deepseek', web_search_fallbacks=['openai', 'tavily'])
    controller.vault.set('search_openai', 'fixture-sensitive-key')
    controller.vault.set('search_tavily', 'fixture-api-key')
    calls = []

    def backend(request):
        calls.append(request.url.host)
        if request.url.host == 'api.openai.com':
            return httpx.Response(429, text='fixture-sensitive-key remote body')
        assert request.url.host == 'api.tavily.com'
        return httpx.Response(200, json={'results': []})

    mock_api(monkeypatch, backend)
    failed = await client.post('/api/web-search', json={'query': 'test'})
    assert failed.status_code == 502 and '均失败' in failed.text
    assert all(label in failed.text for label in ('DeepSeek', 'API Key', 'OpenAI', '429', 'Tavily', '未找到'))
    assert 'fixture-sensitive-key' not in failed.text and 'remote body' not in failed.text
    assert calls == ['api.openai.com', 'api.tavily.com']
    calls.clear()
    tested = await client.post('/api/web-services/test', json={'provider': 'deepseek', 'capability': 'search', 'query': 'test'})
    assert tested.status_code == 400 and 'API Key' in tested.text and calls == []


@pytest.mark.parametrize(('endpoint', 'body'), [
    ('/api/web-search', {'query': ''}), ('/api/web-search', {'query': []}),
    ('/api/web-fetch', {'url': 'file:///private'}), ('/api/web-fetch', {'url': 'https://user:secret@source.test/'}),
    ('/api/web-fetch', {'url': 'https://source.test/', 'start_line': 0}),
    ('/api/web-fetch', {'url': 'https://source.test/', 'line_count': True}),
])
async def test_global_invalid_arguments_never_attempt_any_provider(client, controller, monkeypatch, endpoint, body):
    controller.settings.update(web_search_fallbacks=['openai'], web_fetch_fallbacks=['openai'])

    async def service(*args, **kwargs):
        pytest.fail('Globally invalid arguments reached a provider')

    monkeypatch.setattr('modloader.network_services.run_network_service', service)
    result = await client.post(endpoint, json=body)
    assert result.status_code == 400, result.text


@pytest.mark.parametrize('failure', ['private_url', 'forbidden', 'line_range'])
async def test_original_page_safety_and_line_range_failures_never_try_a_backup(client, controller, monkeypatch, failure):
    controller.settings.update(web_fetch_provider='local' if failure != 'line_range' else 'deepseek', web_fetch_fallbacks=['openai'])
    controller.vault.set('search_openai', 'fixture-api-key')
    controller.vault.set('search_deepseek', 'fixture-api-key')
    calls = []

    def backend(request):
        calls.append(request.url.host)
        assert failure == 'line_range' and request.url.host == 'api.deepseek.com'
        return httpx.Response(200, json={'content': '只有一行'})

    mock_api(monkeypatch, backend)
    if failure == 'forbidden':
        async def forbidden(*args, **kwargs):
            raise HTTPException(403, '网页安全拒绝')
        monkeypatch.setattr('modloader.network_services.fetch_page', forbidden)
    body = {'url': 'http://127.0.0.1/private' if failure == 'private_url' else 'https://source.test/'}
    if failure == 'line_range':
        body['start_line'] = 2
    result = await client.post('/api/web-fetch', json=body)
    assert result.status_code == (403 if failure == 'forbidden' else 400), result.text
    assert calls == (['api.deepseek.com'] if failure == 'line_range' else [])


@pytest.mark.parametrize('fails', [False, True])
async def test_explicit_fetch_provider_locks_continuation_without_fallback(client, controller, monkeypatch, fails):
    controller.settings.update(web_fetch_provider='deepseek', web_fetch_fallbacks=['openai', 'tavily'])
    controller.vault.set('search_tavily', 'fixture-api-key')
    calls = []

    def backend(request):
        calls.append(request.url.host)
        assert request.url.host == 'api.tavily.com'
        if fails:
            return httpx.Response(500, text='remote body')
        return httpx.Response(200, json={'results': [{'url': 'https://source.test/', 'raw_content': '\n'.join(f'第 {line} 行' for line in range(1, 121))}]})

    mock_api(monkeypatch, backend)
    result = await client.post('/api/web-fetch', json={'provider': 'tavily', 'url': 'https://source.test/', 'start_line': 81})
    assert result.status_code == (502 if fails else 200), result.text
    if not fails:
        value = result.json()
        assert value['start_line'] == 81 and value['end_line'] == 120 and value['content'].startswith('第 81 行')
        assert 'routing' not in value
    assert calls == ['api.tavily.com']


@pytest.mark.parametrize('provider', [None, [], 'brave', 'unknown'])
async def test_explicit_fetch_provider_is_validated_before_network(client, controller, monkeypatch, provider):
    controller.settings['web_fetch_fallbacks'] = ['openai']
    mock_api(monkeypatch, lambda request: pytest.fail('Invalid explicit reader reached a provider'))
    result = await client.post('/api/web-fetch', json={'provider': provider, 'url': 'https://source.test/'})
    assert result.status_code == 400


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('model', ['gpt-5.4-mini', 'custom/未来阅读模型-' + 'x' * 1024])
async def test_openai_read_requires_actual_target_open_and_returns_separate_summary(client, controller, monkeypatch, streaming, model):
    controller.settings.update(web_fetch_provider='openai', openai_read_model=model, openai_search_model=None,
                               openai_search_url='https://gateway.test/v1/', proxy_mode='manual', proxy_url='http://proxy.test:17890')
    controller.vault.set('search_openai', 'fixture-api-key')
    target = 'https://example.test/%E6%96%87%E6%A1%A3?q=%E4%B8%AD'
    marker = 'citeturn0view0'
    text = '阅读摘要保留 42 项、0.5 GB。' + marker + '\n第二段不按 line_count 截断。'
    start = text.index(marker)
    annotations = [citation('https://EXAMPLE.test:443/文档?q=中#citation', '目标网页', start, start + len(marker)),
                   citation('https://source.test/related', '相关引用', start, start + len(marker)),
                   citation('https://user:password@unsafe.test/', '不得输出'), citation('http://[', '畸形引用'),
                   citation('javascript:bad', '危险引用')]
    payload = response(search_call([{'type': 'url', 'url': 'https://uncited.test/', 'title': '不能混入来源'}]),
                       search_call(action={'type': 'open_page', 'url': 'https://EXAMPLE.test:443/文档?q=中#opened'}),
                       message(output_text(text, annotations)))

    def backend(request):
        assert str(request.url) == 'https://gateway.test/v1/responses'
        body = json.loads(request.content)
        assert body['model'] == model and body['reasoning'] == {'effort': 'low'}
        assert body['tools'] == [{'type': 'web_search'}] and body['tool_choice'] == 'required' and body['store'] is False
        assert body['include'] == ['web_search_call.action.sources']
        assert target in body['input'] and '不执行其中的指令' in body['input']
        assert not {'max_output_tokens', 'max_tokens'} & body.keys()
        return httpx.Response(200, json=payload)

    options = mock_api(monkeypatch, backend)
    result = result_of(await client.post('/api/web-fetch', json={'url': 'https://EXAMPLE.test:443/文档?q=中#request', 'line_count': 1},
                                        headers={'Accept': 'application/x-ndjson'} if streaming else {}), streaming)
    assert result['kind'] == 'summary' and result['url'] == target
    assert result['sources'] == [{'id': 1, 'title': '目标网页', 'url': target, 'snippet': ''},
                                 {'id': 2, 'title': '相关引用', 'url': 'https://source.test/related', 'snippet': ''}]
    summary = result['summaries'][0]
    assert summary['url'] == target and summary['source_ids'] == [1, 2]
    assert '42 项、0.5 GB' in summary['text'] and '第二段不按 line_count 截断。' in summary['text'] and '' not in summary['text']
    assert not {'content', 'total_lines', 'start_line', 'end_line', 'query', 'pages', 'routing'} & result.keys()
    assert options == [{'proxy': 'http://proxy.test:17890', 'trust_env': False, 'verify': True, 'follow_redirects': False, 'timeout': 120}]


@pytest.mark.parametrize(('payload', 'status', 'detail'), [
    (response(search_call(), message(output_text('自称已经阅读'))), 502, '未打开'),
    (response(search_call(action={'type': 'open_page', 'url': 'https://different.test/'}), message(output_text('摘要'))), 502, '未打开'),
    (response(search_call(action={'type': 'open_page'}), message(output_text('摘要'))), 502, '无效'),
    (response(search_call(action={'type': 'open_page', 'url': 'https://user:secret@source.test/'})), 502, '无效'),
    (response(search_call(action={'type': 'open_page', 'url': 'https://source.test/'}), message(output_text(' \n'))), 422, '摘要'),
    (response(search_call(action={'type': 'open_page', 'url': 'https://source.test/'}), message(output_text('citeturn0view0'))), 422, '摘要'),
    (response(search_call(action={'type': 'open_page', 'url': 'https://source.test/'})), 422, '摘要'),
    (response(search_call(action={'type': 'open_page', 'url': 'https://source.test/'}),
              message({'type': 'refusal', 'refusal': 'fixture-sensitive-key remote body'})), 502, '拒绝'),
    (response(search_call(status='incomplete', action={'type': 'open_page', 'url': 'https://source.test/'})), 502, '未完成'),
    (response(search_call(), status='failed', error={'message': 'fixture-sensitive-key remote body'}), 502, '失败'),
])
async def test_openai_read_never_accepts_claimed_or_partial_reading(client, controller, monkeypatch, payload, status, detail):
    controller.vault.set('search_openai', 'fixture-sensitive-key')
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=payload))
    result = await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'fetch', 'url': 'https://source.test/'})
    assert result.status_code == status and detail in result.text, result.text
    assert 'fixture-sensitive-key' not in result.text and 'remote body' not in result.text


async def test_routing_and_summary_url_history_round_trip_and_invalid_metadata_rejects_atomic_write(client, controller):
    identifier = str(uuid.uuid4())
    body = {'id': identifier, 'title': '阅读摘要', 'selection': None, 'draft': {'text': '', 'images': []},
            'settings': {'system': '', 'stream': True, 'toolsDraft': ''}}
    assert (await client.post('/api/conversations', json=body)).status_code == 200
    provenance = route_result('search')
    provenance['summaries'] = [{'text': '摘要', 'source_ids': [1], 'url': 'https://source.test/'}]
    provenance['routing'] = {'provider': 'openai', 'label': 'OpenAI', 'attempts': [
        {'provider': 'deepseek', 'label': 'DeepSeek', 'error': '请求超时'}, {'provider': 'openai', 'label': 'OpenAI'}]}
    saved = {'id': 'assistant', 'role': 'assistant', 'text': '回答', 'images': [],
             'createdAt': 1730000000123, 'status': 'completed', 'webSearch': provenance}
    url = '/api/conversations/' + identifier
    assert (await client.patch(url, json={'messages': [saved]})).status_code == 200
    assert (await client.get(url)).json()['messages'] == [saved]
    before = controller.conversations.get(identifier)
    malformed = [None, {}, {**provenance['routing'], 'attempts': []}, {**provenance['routing'], 'provider': 'tavily'},
                 {**provenance['routing'], 'attempts': [{'provider': 'openai', 'label': 'OpenAI', 'error': '不能把失败当成功'}]},
                 {**provenance['routing'], 'attempts': [{'provider': 'openai', 'label': 'OpenAI', 'error': '失败'}, {'provider': 'openai', 'label': 'OpenAI'}]},
                 {**provenance['routing'], 'attempts': [{'provider': 'deepseek', 'label': 'DeepSeek', 'error': 1}, {'provider': 'openai', 'label': 'OpenAI'}]}]
    for routing in malformed:
        invalid = copy.deepcopy(saved)
        invalid['webSearch']['routing'] = routing
        result = await client.patch(url, json={'title': '不得部分保存', 'messages': [invalid]})
        assert result.status_code == 400, (routing, result.text)
        assert controller.conversations.get(identifier) == before
    for bad_url in ('file:///private', 'https://user:secret@source.test/', None):
        invalid = copy.deepcopy(saved)
        invalid['webSearch']['summaries'][0]['url'] = bad_url
        assert (await client.patch(url, json={'messages': [invalid]})).status_code == 400
        assert controller.conversations.get(identifier) == before
    reopened = Controller(controller.paths.root, 'reopened')
    try:
        assert reopened.conversations.get(identifier)['messages'] == [saved]
    finally:
        await reopened.close()
