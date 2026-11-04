"""OpenAI Responses hosted-search contracts; all HTTP and credentials are fixtures."""
import copy
import json
import uuid

import httpx
import pytest

from modloader.app import Controller
from modloader.network_services import DEFAULT_NETWORK_SETTINGS, OPENAI_SEARCH_TIMEOUT_S
from modloader.web_search import validate_search_result
from test_network_services import client, controller, mock_api, result_of


def search_call(sources=None, **changes):
    return {'id': 'ws_fixture', 'type': 'web_search_call', 'status': 'completed',
            'action': {'type': 'search', 'queries': ['fixture'], 'sources': sources or []}, **changes}


def output_text(text='检索摘要', annotations=None):
    return {'type': 'output_text', 'text': text, 'annotations': annotations or []}


def image_call(results, **changes):
    return {'type': 'web_search_call', 'status': 'completed', 'results': results, **changes}


def image_result(image_url='https://images.test/original.png', source_website_url='https://source.test/page', **changes):
    return {'type': 'image_result', 'image_url': image_url, 'source_website_url': source_website_url, **changes}


def message(*contents, **changes):
    return {'id': 'msg_fixture', 'type': 'message', 'status': 'completed', 'role': 'assistant',
            'content': list(contents), **changes}


def response(*output, **changes):
    return {'id': 'resp_fixture', 'status': 'completed', 'error': None, 'output': list(output), **changes}


def citation(url, title='来源', start_index=0, end_index=1):
    return {'type': 'url_citation', 'url': url, 'title': title, 'start_index': start_index, 'end_index': end_index}


async def test_openai_model_and_url_save_without_changing_default_routes(client, controller):
    assert controller.settings['web_search_provider'] == 'searxng'
    assert controller.settings['web_image_provider'] == 'searxng'
    assert controller.settings['web_fetch_provider'] == 'local'
    assert controller.settings['openai_search_model'] == 'gpt-4.1-mini'
    assert controller.settings['openai_image_model'] == 'gpt-5.4-mini'
    custom_model = 'vendor/自定义模型-' + 'x' * 1024
    custom_image_model = 'vendor/自定义图片模型-' + 'y' * 1024
    patch = {'openai_search_url': 'https://gateway.test/prefix/v1/', 'openai_search_model': custom_model,
             'openai_image_model': custom_image_model,
             'secrets': {'search_openai': 'fixture-openai-secret'}}
    saved = await client.patch('/api/settings', json=patch)
    assert saved.status_code == 200, saved.text
    assert saved.json()['openai_search_model'] == custom_model
    assert saved.json()['openai_image_model'] == custom_image_model
    assert saved.json()['openai_read_model'] == 'gpt-5.4-mini'
    assert saved.json()['web_credentials']['search_openai'] is True
    assert saved.json()['web_search_provider'] == 'searxng'
    assert saved.json()['web_image_provider'] == 'searxng' and saved.json()['web_fetch_provider'] == 'local'
    assert 'fixture-openai-secret' not in saved.text
    reopened = Controller(controller.paths.root, 'fixture-reopened')
    try:
        assert reopened.settings['openai_search_model'] == custom_model
        assert reopened.settings['openai_image_model'] == custom_image_model
        assert reopened.settings['openai_search_url'] == patch['openai_search_url']
        assert reopened.vault.get('search_openai') == 'fixture-openai-secret'
    finally:
        await reopened.close()
    selected = await client.patch('/api/settings', json={'web_search_provider': 'openai'})
    assert selected.status_code == 200 and selected.json()['web_search_provider'] == 'openai'
    assert selected.json()['web_image_provider'] == 'searxng' and selected.json()['web_fetch_provider'] == 'local'
    images = await client.patch('/api/settings', json={'web_image_provider': 'openai'})
    assert images.status_code == 200 and images.json()['web_image_provider'] == 'openai'
    assert images.json()['web_search_provider'] == 'openai' and images.json()['web_fetch_provider'] == 'local'
    assert images.json()['openai_search_model'] == custom_model
    assert images.json()['openai_image_model'] == custom_image_model


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize(('base', 'path', 'model'), [
    (DEFAULT_NETWORK_SETTINGS['openai_search_url'], '/v1/responses', 'gpt-4.1-mini'),
    ('https://gateway.test', '/v1/responses', 'custom/model-2027'),
    ('https://gateway.test/prefix/v1/', '/prefix/v1/responses', '自定义-' + 'x' * 1024),
    ('https://gateway.test/prefix', '/prefix/v1/responses', 'gpt-4.1-mini'),
])
async def test_responses_request_endpoint_model_proxy_and_no_arbitrary_caps(client, controller, monkeypatch, streaming, base, path, model):
    controller.settings.update(web_search_provider='openai', openai_search_url=base, openai_search_model=model,
                               proxy_mode='manual', proxy_url='http://proxy.test:17890')
    controller.vault.set('search_openai', 'fixture-api-key')
    for field in ('openai_image_model', 'openai_read_model'):
        if streaming:
            controller.settings[field] = None  # Unrelated new capabilities do not constrain existing search callers.
        else:
            controller.settings.pop(field)
    query = '中文 + C++ 🐼 ' * 500
    calls = []

    def backend(request):
        calls.append(request)
        assert request.method == 'POST' and request.url.path == path
        assert request.headers['authorization'] == 'Bearer fixture-api-key'
        assert json.loads(request.content) == {
            'model': model, 'tools': [{'type': 'web_search'}], 'tool_choice': 'required',
            'include': ['web_search_call.action.sources'], 'store': False, 'input': query.strip(),
        }
        return httpx.Response(200, json=response(search_call([
            {'type': 'url', 'url': f'https://source.test/{index}'} for index in range(65)
        ]), message(output_text('生成摘要；未引用具体来源。'))))

    options = mock_api(monkeypatch, backend)
    before = dict(controller.settings)
    result = result_of(await client.post('/api/web-search', json={'query': query},
                                        headers={'Accept': 'application/x-ndjson'} if streaming else {}), streaming)
    validate_search_result(result)
    assert len(calls) == 1 and len(result['sources']) == 65
    assert result['summaries'] == [{'text': '生成摘要；未引用具体来源。', 'source_ids': []}]
    assert all(source['snippet'] == '' for source in result['sources'])
    assert options == [{'proxy': 'http://proxy.test:17890', 'trust_env': False, 'verify': True,
                        'follow_redirects': False, 'timeout': OPENAI_SEARCH_TIMEOUT_S}]
    assert controller.settings == before and controller.store.get('settings', 'main') is None
    assert not controller.runtime.instances and controller.store.all('jobs') == []


async def test_sources_citations_unicode_and_separate_generated_summaries(client, controller, monkeypatch):
    controller.vault.set('search_openai', 'fixture-api-key')
    marker = 'citeturn0search0turn0search1'
    text = '中文😀结论。' + marker + '\n第二段仍保留。'
    start = text.index(marker)
    annotations = [citation('https://source.test/第二篇', '第二篇', start, start + len(marker)),
                   citation('https://source.test/first#cite', '<b>第一篇</b>', start, start + len(marker)),
                   citation('https://source.test/first#cite', '第一篇', start, start + len(marker)),
                   citation('javascript:bad', '危险来源', start, start + len(marker)),
                   citation('https://source.test/uncited#bad fragment', '无效片段不得成为引用', start, start + len(marker))]
    payload = response(search_call([
        {'type': 'url', 'url': 'https://source.test/first'},
        {'type': 'url', 'url': 'https://source.test/first#duplicate', 'title': '重复'},
        {'type': 'url', 'url': 'file:///unsafe', 'title': '危险'},
        {'type': 'url', 'url': 'https://source.test/uncited', 'title': '仅检索到、未引用'},
    ]), {'type': 'reasoning', 'id': 'rs_fixture', 'summary': []},
        message(output_text(text, annotations), output_text('无引用段落。')))
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=payload))
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'search', 'query': '中文😀'}))
    validate_search_result(result)
    assert result['sources'] == [
        {'id': 1, 'url': 'https://source.test/first', 'title': '第一篇', 'snippet': ''},
        {'id': 2, 'url': 'https://source.test/uncited', 'title': '仅检索到、未引用', 'snippet': ''},
        {'id': 3, 'url': 'https://source.test/第二篇', 'title': '第二篇', 'snippet': ''},
    ]
    summary = result['summaries'][0]
    assert summary['source_ids'] == [3, 1]
    assert '中文😀结论。' in summary['text'] and '第二段仍保留。' in summary['text']
    assert 'https://source.test/%E7%AC%AC%E4%BA%8C%E7%AF%87' in summary['text']
    assert summary['text'].count('https://source.test/first#cite') == 1
    assert '' not in summary['text'] and 'javascript:' not in summary['text'] and 'uncited' not in summary['text']
    assert result['summaries'][1] == {'text': '无引用段落。', 'source_ids': []}
    assert not {'pages', 'content', 'images'} & result.keys()


async def test_bad_or_semantic_citation_positions_keep_text_and_real_source_links(client, controller, monkeypatch):
    controller.vault.set('search_openai', 'fixture-api-key')
    text = '真实语义内容。还有后文。citeturn0search0'
    annotations = [citation('https://source.test/first', '来源 [A]', 0, 6),
                   citation('https://source.test/second', '来源B', True, 10000)]
    payload = response(search_call(), message(output_text(text, annotations)))
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=payload))
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'search', 'query': '引用位置'}))
    summary = result['summaries'][0]
    assert summary['source_ids'] == [1, 2]
    assert '真实语义内容' in summary['text'] and '还有后文。' in summary['text']
    assert 'https://source.test/first' in summary['text'] and 'https://source.test/second' in summary['text']
    assert '' not in summary['text']


@pytest.mark.parametrize(('span', 'already_linked'), [
    ('[Python 文档](https://docs.python.org/3/library/pathlib.html)', True),
    ('[Python 文档](https://other.test/pathlib.html)', False),
    ('Python 文档', False),
    ('说明文字 [Python 文档](https://docs.python.org/3/library/pathlib.html)', False),
])
async def test_reading_fixture_keeps_existing_matching_markdown_link_without_duplicate(client, controller, monkeypatch, span, already_linked):
    controller.vault.set('search_openai', 'fixture-api-key')
    target = 'https://docs.python.org/3/library/pathlib.html'
    text = 'pathlib 提供面向对象的文件系统路径。\n\n- Path.read_text() 返回文本。\n- Path.read_bytes() 返回字节。\n\n' + span
    payload = response(search_call(action={'type': 'open_page', 'url': target}),
                       message(output_text(text, [citation(target, 'Python pathlib 文档', text.index(span), len(text))])))
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=payload))
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'fetch', 'url': target}))
    summary = result['summaries'][0]
    assert summary['source_ids'] == [1] and summary['url'] == target
    assert summary['text'] == (text if already_linked else text + ' [Python pathlib 文档](<' + target + '>)')


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize(('payload', 'status', 'detail'), [
    (response(message(output_text('只回答，没有搜索'))), 502, '未执行'),
    (response(search_call(status='incomplete')), 502, '未完成'),
    (response(search_call(status='failed')), 502, '未完成'),
    (response(search_call(), status='incomplete', incomplete_details={'reason': 'fixture-sensitive-key'}), 502, '未完成'),
    (response(search_call(), status='failed', error={'message': 'fixture-sensitive-key remote body'}), 502, '失败'),
    (response(search_call(), status='completed', error={'message': 'fixture-sensitive-key'}), 502, '失败'),
    (response(search_call(), message({'type': 'refusal', 'refusal': 'fixture-sensitive-key remote body'})), 502, '拒绝'),
    (response(search_call(), message(output_text(), status='incomplete')), 502, '未完成'),
    (response(search_call(), message(output_text('无实际来源'))), 422, '未找到'),
    (response(search_call([{'type': 'url', 'url': 'javascript:bad'}])), 422, '未找到'),
    ({}, 502, '格式'),
    (response(output='fixture-sensitive-key'), 502, '格式'),
    (response('fixture-sensitive-key'), 502, '无效'),
    (response(search_call(action=None)), 502, '搜索操作'),
    (response(search_call(action={'type': 'search', 'sources': {}})), 502, '来源'),
    (response(search_call(), message({'type': 'output_text', 'text': 'fixture-sensitive-key', 'annotations': None})), 502, '格式'),
    (response(search_call(), message(output_text(annotations=['fixture-sensitive-key']))), 502, '格式'),
])
async def test_incomplete_refused_malformed_or_unexecuted_search_is_not_success(client, controller, monkeypatch, streaming, payload, status, detail):
    controller.vault.set('search_openai', 'fixture-sensitive-key')
    calls = []

    def backend(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    mock_api(monkeypatch, backend)
    response_value = await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'search', 'query': 'test'},
                                       headers={'Accept': 'application/x-ndjson'} if streaming else {})
    if streaming:
        event = json.loads(response_value.text.splitlines()[-1])
        assert event['type'] == 'error' and event['status'] == status
    else:
        assert response_value.status_code == status, response_value.text
    assert detail in response_value.text and 'fixture-sensitive-key' not in response_value.text and 'remote body' not in response_value.text
    assert len(calls) == 1


@pytest.mark.parametrize('capability', ['search', 'images'])
@pytest.mark.parametrize(('failure', 'status'), [('401', 502), ('429', 502), ('500', 502), ('302', 502),
                                             ('html', 502), ('list', 502), ('timeout', 504), ('connection', 502)])
async def test_openai_transport_failures_are_sanitized_without_fallback(client, controller, monkeypatch, failure, status, capability):
    controller.settings['web_search_provider' if capability == 'search' else 'web_image_provider'] = 'openai'
    controller.vault.set('search_openai', 'fixture-sensitive-key')
    calls = []

    def backend(request):
        calls.append(request)
        if failure == 'timeout':
            raise httpx.ReadTimeout('fixture-sensitive-key', request=request)
        if failure == 'connection':
            raise httpx.ConnectError('fixture-sensitive-key', request=request)
        if failure.isdecimal():
            return httpx.Response(int(failure), text='fixture-sensitive-key remote body', headers={'location': 'https://other.test/'})
        return httpx.Response(200, text='fixture-sensitive-key') if failure == 'html' else httpx.Response(200, json=[])

    mock_api(monkeypatch, backend)
    result = await client.post('/api/web-search', json={'query': '失败', 'category': 'general' if capability == 'search' else 'images'})
    assert result.status_code == status, result.text
    assert 'fixture-sensitive-key' not in result.text and 'remote body' not in result.text
    assert len(calls) == 1


async def test_openai_read_summary_rejects_line_continuation_before_network(client, controller, monkeypatch):
    controller.vault.set('search_openai', 'fixture-api-key')
    mock_api(monkeypatch, lambda request: pytest.fail('Unsupported OpenAI capability reached the network'))
    result = await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'fetch', 'url': 'https://source.test/', 'start_line': 2})
    assert result.status_code == 400 and '按行续读' in result.json()['detail']


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('model', ['gpt-5.4-mini', 'custom/image-model-' + 'x' * 1024, 'gpt-4.1-mini'])
async def test_image_request_uses_its_model_shared_key_proxy_and_no_fixed_result_limit(client, controller, monkeypatch, streaming, model):
    controller.settings.update(web_image_provider='openai', openai_search_url='https://gateway.test/prefix/v1/',
                               openai_image_model=model, openai_search_model='unchanged-search-model',
                               proxy_mode='manual', proxy_url='http://proxy.test:17890')
    controller.vault.set('search_openai', 'fixture-api-key')
    query = '中文图片 🐼 ' * 500
    calls = []

    def backend(request):
        calls.append(request)
        assert request.method == 'POST' and str(request.url) == 'https://gateway.test/prefix/v1/responses'
        assert request.headers['authorization'] == 'Bearer fixture-api-key'
        assert json.loads(request.content) == {
            'model': model, 'tools': [{'type': 'web_search', 'search_content_types': ['image', 'text'],
                                                    'image_settings': {'caption': True}}], 'tool_choice': 'required',
            'include': ['web_search_call.action.sources', 'web_search_call.results'], 'store': False, 'input': query.strip(),
            'instructions': '使用 web_search 搜索与用户关键词相关的图片结果，而不仅是文字搜索。',
        }
        return httpx.Response(200, json=response(image_call([
            image_result(f'https://images.test/{index}.jpg', f'https://source.test/{index}') for index in range(65)
        ])))

    options = mock_api(monkeypatch, backend)
    result = result_of(await client.post('/api/web-search', json={'query': query, 'category': 'images'},
                                        headers={'Accept': 'application/x-ndjson'} if streaming else {}), streaming)
    validate_search_result(result)
    assert len(calls) == 1 and len(result['images']) == len(result['sources']) == 65
    assert result['images'][-1] == {'source_id': 65, 'title': '', 'url': 'https://images.test/64.jpg'}
    assert all(source['snippet'] == '' for source in result['sources'])
    assert 'summaries' not in result and 'pages' not in result
    assert options == [{'proxy': 'http://proxy.test:17890', 'trust_env': False, 'verify': True,
                        'follow_redirects': False, 'timeout': OPENAI_SEARCH_TIMEOUT_S}]
    assert controller.settings['web_search_provider'] == 'searxng' and controller.settings['web_fetch_provider'] == 'local'
    assert controller.settings['openai_search_model'] == 'unchanged-search-model'


@pytest.mark.parametrize('streaming', [False, True])
async def test_structured_image_results_preserve_real_provenance_captions_and_summaries(client, controller, monkeypatch, streaming):
    controller.vault.set('search_openai', 'fixture-api-key')
    marker = 'citeturn0search0'
    text = '模型综合说明😀。' + marker
    start = text.index(marker)
    payload = response(image_call([
        image_result(caption='<b>图片一</b>', thumbnail_url='https://images.test/thumb.webp'),
        image_result('https://images.test/second.jpg', 'https://source.test/page#second', caption='图片二', thumbnail_url='javascript:bad'),
        image_result(caption='重复图片'),
        image_result('javascript:bad', caption='不得显示'),
        image_result(source_website_url=None),
        image_result('https://images.test/unsafe.jpg', 'file:///unsafe'),
        {'type': 'text_result', 'url': 'https://not-an-image.test/', 'image_url': 'https://images.test/not-structured.jpg'},
    ]), image_call([image_result('https://images.test/third.jpg', 'https://source.test/second-page')],
                   action={'type': 'search', 'sources': [{'type': 'url', 'url': 'https://source.test/supporting', 'title': '辅助文字来源'}]}),
        message(output_text(text, [citation('https://source.test/page#cited', '图片出处', start, start + len(marker)),
                                   citation('https://source.test/supporting', '辅助文字来源', start, start + len(marker))])))
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=payload))
    result = result_of(await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'images', 'query': '图片'},
                                        headers={'Accept': 'application/x-ndjson'} if streaming else {}), streaming)
    validate_search_result(result)
    assert result['sources'] == [
        {'id': 1, 'title': '图片一', 'url': 'https://source.test/page', 'snippet': ''},
        {'id': 2, 'title': '', 'url': 'https://source.test/second-page', 'snippet': ''},
        {'id': 3, 'title': '辅助文字来源', 'url': 'https://source.test/supporting', 'snippet': ''},
    ]
    assert result['images'] == [
        {'source_id': 1, 'title': '图片一', 'url': 'https://images.test/original.png', 'thumbnail_url': 'https://images.test/thumb.webp'},
        {'source_id': 1, 'title': '图片二', 'url': 'https://images.test/second.jpg'},
        {'source_id': 2, 'title': '', 'url': 'https://images.test/third.jpg'},
    ]
    summary = result['summaries'][0]
    assert summary['source_ids'] == [1, 3]
    assert '模型综合说明😀。' in summary['text'] and 'https://source.test/page#cited' in summary['text']
    assert 'https://source.test/supporting' in summary['text'] and '' not in summary['text']
    assert 'pages' not in result and 'content' not in result


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize(('payload', 'status', 'detail'), [
    (response(message(output_text('![不可猜测的图片](https://images.test/guessed.jpg)'))), 502, '未执行'),
    (response(image_call([]), message(output_text('![不能替代真实结果](https://images.test/guessed.jpg)',
                                                 [citation('https://source.test/page')]))), 422, '图片'),
    (response(image_call([image_result(source_website_url=None)])), 422, '图片'),
    (response(image_call([image_result('javascript:bad')])), 422, '图片'),
    (response(image_call([image_result(source_website_url='file:///unsafe')])), 422, '图片'),
    (response(image_call([{'type': 'text_result', 'url': 'https://source.test/'}])), 422, '图片'),
    (response(image_call(None)), 502, 'results'),
    (response(image_call('fixture-sensitive-key remote body')), 502, 'results'),
    (response(image_call([image_result()], status='failed')), 502, '未完成'),
    (response(image_call([image_result()], status='incomplete')), 502, '未完成'),
    (response(image_call([image_result()]), status='incomplete'), 502, '未完成'),
    (response(image_call([image_result()]), status='failed', error={'message': 'fixture-sensitive-key remote body'}), 502, '失败'),
    (response(image_call([image_result()]), message({'type': 'refusal', 'refusal': 'fixture-sensitive-key remote body'})), 502, '拒绝'),
])
async def test_image_errors_empty_results_and_assistant_urls_cannot_fake_success(client, controller, monkeypatch, streaming, payload, status, detail):
    controller.vault.set('search_openai', 'fixture-sensitive-key')
    calls = []

    def backend(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    mock_api(monkeypatch, backend)
    result = await client.post('/api/web-services/test', json={'provider': 'openai', 'capability': 'images', 'query': '图片'},
                               headers={'Accept': 'application/x-ndjson'} if streaming else {})
    if streaming:
        event = json.loads(result.text.splitlines()[-1])
        assert event['type'] == 'error' and event['status'] == status
    else:
        assert result.status_code == status, result.text
    assert detail in result.text and 'fixture-sensitive-key' not in result.text and 'remote body' not in result.text
    assert len(calls) == 1


async def test_saved_summaries_round_trip_and_invalid_references_reject_whole_patch(client, controller):
    identifier = str(uuid.uuid4())
    body = {'id': identifier, 'title': 'OpenAI 检索', 'selection': None, 'draft': {'text': '', 'images': []},
            'settings': {'system': '', 'stream': True, 'toolsDraft': ''}}
    assert (await client.post('/api/conversations', json=body)).status_code == 200
    provenance = {'query': '中文', 'elapsed_ms': 123, 'sources': [
        {'id': 1, 'title': '来源', 'url': 'https://source.test/', 'snippet': ''}],
        'summaries': [{'text': '综合摘要 [来源](https://source.test/)', 'source_ids': [1]},
                      {'text': '没有引用的段落。', 'source_ids': []}]}
    saved = {'id': 'assistant', 'role': 'assistant', 'text': '回答', 'images': [],
             'createdAt': 1730000000123, 'status': 'completed', 'webSearch': provenance}
    url = '/api/conversations/' + identifier
    assert (await client.patch(url, json={'messages': [saved]})).status_code == 200
    assert (await client.get(url)).json()['messages'] == [saved]
    before = controller.conversations.get(identifier)
    malformed = [None, {}, [None], [{}], [{'text': 1, 'source_ids': [1]}], [{'text': '摘要', 'source_ids': 1}],
                 *[[{'text': '摘要', 'source_ids': [value]}] for value in (0, -1, 2, True, '1', 1.0)],
                 [{'text': '摘要', 'source_ids': [], 'pages': []}]]
    for value in malformed:
        invalid = copy.deepcopy(saved)
        invalid['webSearch']['summaries'] = value
        result = await client.patch(url, json={'title': '不得部分写入', 'messages': [invalid]})
        assert result.status_code == 400, (value, result.text)
        assert controller.conversations.get(identifier) == before
    reopened = Controller(controller.paths.root, 'reopened')
    try:
        assert reopened.conversations.get(identifier)['messages'] == [saved]
    finally:
        await reopened.close()
    legacy = copy.deepcopy(saved)
    del legacy['webSearch']['summaries']
    assert (await client.patch(url, json={'messages': [legacy]})).status_code == 200
    assert (await client.get(url)).json()['messages'] == [legacy]
