"""Gateway lifecycle, conservative turn inference and opt-in append-only archives."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from modloader.app import build_app
from modloader.request_history import RequestHistory
from modloader.request_metrics import RequestMetrics
from test_core import controller, gateway_client, GatewayStream, sse


@pytest.fixture
def ledger(tmp_path):
    value = RequestHistory(tmp_path / 'history', enabled=False)
    yield value
    value.archive.drain()


def instance(identifier='one'):
    return SimpleNamespace(id=identifier, alias='Model ' + identifier, backend='llama_cpp')


def assistant(ids=(), content=None):
    message = {'role': 'assistant', 'content': content}
    if ids:
        message['tool_calls'] = [{'id': key, 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"private":"arguments-sentinel"}'}} for key in ids]
    return message


def complete(ledger, messages, answer=None, ids=(), status='completed', streaming=False, target=None, started_at=None):
    metrics = RequestMetrics(streaming, backend='llama_cpp')
    if started_at is not None:
        metrics.started_at = started_at
    ledger.start(metrics, target or instance(), messages)
    message = assistant(ids, answer)
    chunk = {'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if ids else 'stop'}],
             'usage': {'prompt_tokens': 50, 'completion_tokens': 10, 'total_tokens': 60},
             'timings': {'prompt_n': 20, 'cache_n': 30, 'prompt_per_second': 200, 'predicted_per_second': 10}}
    metrics.observe(chunk)
    ledger.observe(metrics.token, chunk)
    metrics.finish(status)
    ledger.finish(metrics)
    return metrics, message


def follow(messages, message):
    return messages + [message] + [{'role': 'tool', 'tool_call_id': call['id'], 'content': 'result-sentinel'} for call in message.get('tool_calls', [])]


def test_standalone_and_twelve_request_chain_seventeen_tools(ledger):
    standalone, _ = complete(ledger, [{'role': 'user', 'content': 'standalone'}], 'answer')
    messages = [{'role': 'user', 'content': 'prompt-sentinel'}]
    tokens = []
    for index in range(12):
        ids = [f'call-{index}-{position}' for position in range(2 if index < 6 else 1)] if index < 11 else []
        metrics, message = complete(ledger, messages, 'reply-sentinel' if index == 11 else None, ids)
        tokens.append(metrics.token)
        messages = follow(messages, message)
        latest = ledger.latest('one')
        assert latest['state'] == ('completed' if index == 11 else 'waiting')
        assert len(latest['requests']) == index + 1
    assert len(ledger.records) == 13
    assert ledger.records[standalone.token]['aggregation'] == 'request'
    turns = {ledger.records[token]['turn_id'] for token in tokens}
    assert len(turns) == 1 and None not in turns
    assert all(ledger.records[token]['turn_state'] == 'completed' for token in tokens)
    assert sum(len(ledger.boundaries[token]['tool_ids']) for token in tokens) == 17
    view = ledger.page({}, [], turn_id=turns.pop())
    assert len(view['items']) == view['limit'] == 12
    assert view['items'][0]['request_key'] == tokens[0]
    assert view['items'][0]['revision'] > 2
    serialized = json.dumps(ledger.records) + repr(ledger.boundaries)
    assert not any(value in serialized for value in ('prompt-sentinel', 'reply-sentinel', 'result-sentinel', 'arguments-sentinel'))
    duration = ledger.latest('one')['requests'][-1]['duration_s']
    assert ledger.latest('one')['requests'][-1]['duration_s'] == duration


@pytest.mark.parametrize('change', ['new_user', 'trim', 'rewrite', 'duplicate_response', 'duplicate_id_other_prompt', 'repeat', 'other_instance'])
def test_no_false_grouping(ledger, change):
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'}]
    first, response = complete(ledger, messages, ids=['same-id'])
    continuation = follow(messages, response)
    target = instance()
    if change == 'new_user':
        continuation.append({'role': 'user', 'content': 'another'})
    elif change == 'trim':
        continuation = continuation[1:]
    elif change == 'rewrite':
        continuation[0] = {'role': 'system', 'content': 'rewritten'}
    elif change == 'duplicate_response':
        complete(ledger, messages, ids=['same-id'])
    elif change == 'duplicate_id_other_prompt':
        complete(ledger, [{'role': 'user', 'content': 'different'}], ids=['same-id'])
    elif change == 'repeat':
        complete(ledger, continuation, 'finished')
    elif change == 'concurrent':
        running = RequestMetrics(False)
        ledger.start(running, target, [{'role': 'user', 'content': 'concurrent'}])
    elif change == 'other_instance':
        target = instance('two')
    last, _ = complete(ledger, continuation, 'finished', target=target)
    assert ledger.records[last.token]['aggregation'] == 'request'
    assert ledger.records[last.token]['turn_id'] is None


def test_duplicate_ids_within_one_response_do_not_link(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    _, response = complete(ledger, messages, ids=['duplicate', 'duplicate'])
    last, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[last.token]['turn_id'] is None


def test_stream_fragmented_ids_by_choice_and_index(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    request = RequestMetrics(True)
    ledger.start(request, instance(), messages)
    for position, fragment in enumerate(('call-', '123')):
        chunk = {'choices': [{'index': 0, 'delta': {'tool_calls': [
            {'index': 1, 'id': fragment + ('b' if position else ''), 'function': {'name': 'look' if position == 0 else 'up', 'arguments': '{"private":' if position == 0 else '"arguments-sentinel"}'}},
            {'index': 0, 'id': fragment, 'function': {'name': 'look' if position == 0 else 'up', 'arguments': '{"private":' if position == 0 else '"arguments-sentinel"}'}}]}, 'finish_reason': None}]}
        request.observe(chunk)
        ledger.observe(request.token, chunk)
    ledger.observe(request.token, {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})
    request.finish('completed')
    ledger.finish(request)
    response = assistant(['call-123', 'call-123b'])
    final, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[final.token]['turn_id'] == ledger.records[request.token]['turn_id'] is not None
    assert len(ledger.latest('one')['requests']) == 2
    assert ledger.records[request.token]['tool_names'] == ['lookup']
    assert ledger.records[final.token]['tool_names'] == []


def test_multiple_choices_never_link(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    metrics = RequestMetrics(True, single_choice=False)
    ledger.start(metrics, instance(), messages)
    for index in (0, 1):
        ledger.observe(metrics.token, {'choices': [{'index': index, 'message': assistant(['call']), 'finish_reason': 'tool_calls'}]})
    metrics.finish('completed')
    ledger.finish(metrics)
    final, _ = complete(ledger, follow(messages, assistant(['call'])), 'answer')
    assert ledger.records[final.token]['turn_id'] is None


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'incomplete'])
def test_chain_terminal_and_finish_once(ledger, status):
    messages = [{'role': 'user', 'content': 'question'}]
    first, response = complete(ledger, messages, ids=['tool'])
    last, _ = complete(ledger, follow(messages, response), status=status)
    before = deepcopy(ledger.records)
    ledger.finish(last)
    assert ledger.records == before
    assert ledger.latest('one')['state'] == status
    assert ledger.records[first.token]['status'] == 'completed'
    assert ledger.records[first.token]['turn_state'] == status


def test_normalized_complete_sequence_fallback(ledger):
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'question'}]}]
    metrics = RequestMetrics(False)
    ledger.start(metrics, instance(), messages)
    ledger.observe(metrics.token, {'choices': [{'message': {'role': 'assistant', 'content': 'continue'}, 'finish_reason': 'tool_calls'}]})
    metrics.finish('completed')
    ledger.finish(metrics)
    last, _ = complete(ledger, [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'continue'}], 'answer')
    assert ledger.records[last.token]['turn_id'] is not None


def test_archive_backfill_off_destination_reopen_revisions_and_privacy(ledger, tmp_path):
    first, response = complete(ledger, [{'role': 'user', 'content': 'prompt-sentinel'}], ids=['call-sentinel'])
    directory = tmp_path / 'history'
    assert not directory.exists()
    ledger.configure(True, directory)
    assert ledger.archive.drain()
    last, _ = complete(ledger, follow([{'role': 'user', 'content': 'prompt-sentinel'}], response), 'reply-sentinel')
    assert ledger.archive.drain()
    saved, issues = ledger.archive.read()
    assert not issues
    assert saved[ledger.run_id]['records'][first.token]['turn_state'] == 'completed'
    assert saved[ledger.run_id]['records'][first.token]['revision'] == ledger.records[first.token]['revision']
    ledger.configure(False, directory)
    assert ledger.archive.drain() and ledger.archive.status()['state'] == 'disabled'
    complete(ledger, [{'role': 'user', 'content': 'not saved until enabled'}], 'answer')
    assert len(ledger.archive.read()[0][ledger.run_id]['records']) == 2
    other = tmp_path / 'new-history'
    ledger.configure(True, other)
    assert ledger.archive.drain()
    assert len(ledger.archive.read()[0][ledger.run_id]['records']) == 3
    assert directory.exists()
    restarted = RequestHistory(other, enabled=False)
    saved, issues = restarted.archive.read()
    runs = restarted.catalog(saved, issues)['runs']
    assert runs[0]['current'] and runs[0]['request_count'] == 0
    assert runs[1]['id'] == ledger.run_id and runs[1]['request_count'] == 3
    old = restarted.page(saved, issues, ledger.run_id, turn_id=ledger.records[first.token]['turn_id'])
    assert len(old['items']) == 2
    for file in [*directory.glob('*.jsonl'), *other.glob('*.jsonl')]:
        text = file.read_text(encoding='utf-8')
        assert not any(value in text for value in ('prompt-sentinel', 'reply-sentinel', 'arguments-sentinel', 'result-sentinel', 'call-sentinel'))
        assert 'observer' not in text
        for event in (json.loads(line) for line in text.splitlines()):
            record = event.get('record', {})
            assert not {'input', 'observer'}.intersection(record)  # Private boundary signatures.
            if 'counters' in record:
                assert set(record['counters']) == {'input', 'output', 'cached'}
                assert all(value is None or type(value) in (int, float) for value in record['counters'].values())


def test_archive_failure_partial_write_retry_preserves_latest(ledger, tmp_path, monkeypatch):
    original = ledger.archive._append
    failed = []
    def append(directory, data):
        if not failed:
            failed.append(True)
            target = tmp_path / 'history'
            original(directory, data)
            with (target / (json.loads(data)['date'] + '.jsonl')).open('ab') as stream:
                stream.write(data[:11])
            raise OSError('raw-secret-error')
        original(directory, data)
    monkeypatch.setattr(ledger.archive, '_append', append)
    ledger.configure(True, tmp_path / 'history')
    request, _ = complete(ledger, [{'role': 'user', 'content': 'question'}], 'answer')
    assert not ledger.archive.drain()
    status = ledger.archive.status()
    assert status['state'] == 'error' and 'raw-secret-error' not in status['error']
    assert ledger.archive.queue
    ledger.archive.retry()
    assert ledger.archive.drain()
    saved, issues = ledger.archive.read()
    assert len(issues) == 1
    assert len(saved[ledger.run_id]['records']) == 1
    assert saved[ledger.run_id]['records'][request.token]['status'] == 'completed'
    assert ledger.archive.status()['state'] == 'saved'


def test_archive_corruption_metadata_validation_and_missing_terminal(ledger, tmp_path):
    ledger.configure(True, tmp_path / 'history')
    metrics = RequestMetrics(True)
    ledger.start(metrics, instance(), [{'role': 'user', 'content': 'question'}])
    assert ledger.archive.drain()
    file = next((tmp_path / 'history').glob('*.jsonl'))
    with file.open('ab') as stream:
        stream.write(b'not JSON\n')
        bad = deepcopy(ledger.records[metrics.token])
        bad.update(revision=999, started_at='bad', usage={'private': 'secret'})
        stream.write(json.dumps({'schema': 1, 'type': 'request', 'record': bad}).encode() + b'\n')
        stream.write(b'{"schema":1')
    saved, issues = ledger.archive.read()
    assert len(issues) == 3
    record = saved[ledger.run_id]['records'][metrics.token]
    assert record['status'] == 'running' and record['record_incomplete'] is True
    assert record['revision'] == 1


async def test_gateway_fast_calls_management_public_internal_and_removed_instance(controller, tmp_path, monkeypatch):
    async def handler(request):
        return httpx.Response(200, json={'choices': [{'message': assistant(content='answer'), 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 3}})
    original_client = httpx.AsyncClient
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            before = (await client.get('/api/request-history/runs')).json()
            assert before['runs'][0]['request_count'] == 0
            for index in range(9):
                result = await client.post('/v1/chat/completions', json={'model': 'one', 'messages': [{'role': 'user', 'content': str(index)}]}, headers={'X-Modloader-Call-Id': 'not-internal'})
                assert result.status_code == 200 and 'modloader_request' not in result.json()
            async with original_client(transport=httpx.ASGITransport(app=build_app(controller, public=True)), base_url='http://testserver', headers={'Authorization': 'Bearer ' + controller.api_key}) as public:
                result = await public.post('/v1/chat/completions', json={'model': 'one', 'messages': []})
                assert result.status_code == 200 and 'modloader_request' not in result.json()
                assert (await public.get('/api/request-history')).status_code == 404
                assert (await public.get('/api/request-history/runs')).status_code == 404
                assert (await public.post('/api/request-history/retry')).status_code == 404
            internal = await client.post('/api/chat', json={'model': 'one', 'messages': []}, headers={'X-Modloader-Call-Id': 'internal-call'})
            assert internal.json()['modloader_request']['client_call_id'] == 'internal-call'
            assert len(controller.request_history.records) == 10
            instance = instances['one']
            instance.state = 'stopped'
            view = (await client.get('/api/instances/' + instance.id + '/telemetry')).json()['external_history']
            assert view['state'] == 'completed' and len(view['requests']) == 1
            controller.runtime.instances.pop(instance.id)
            result = (await client.get('/api/request-history?limit=3')).json()
            assert result['total'] == 10 and len(result['items']) == 3
            assert all(record['client_call_id'] is None for record in result['items'])
            assert (await client.get('/api/request-history?offset=-1')).status_code == 422
            assert (await client.get('/api/request-history?limit=0')).status_code == 422
            assert (await client.get('/api/request-history?run_id=missing')).status_code == 404
            assert (await client.get('/api/request-history/runs')).json()['runs'][0]['instances'][0]['id'] == instance.id
    finally:
        await controller.close()


@pytest.mark.parametrize('terminal', ['completed', 'failed', 'cancelled', 'incomplete'])
async def test_gateway_stream_terminal_once_no_private_events(controller, tmp_path, monkeypatch, terminal):
    from starlette.requests import Request
    chunks = [sse({'choices': [{'index': 0, 'delta': {'content': 'private-reply'}}]})]
    if terminal == 'completed':
        chunks.extend([sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}), sse('[DONE]')])
    if terminal == 'failed':
        chunks.append(httpx.ReadError('private-error'))
    stream = GatewayStream(chunks)
    async def handler(request):
        return httpx.Response(200, stream=stream)
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    response = None
    try:
        endpoint = next(route.endpoint for route in client._transport.app.routes if route.path == '/v1/chat/completions')
        async def receive():
            return {'type': 'http.request', 'body': json.dumps({'model': 'one', 'messages': [], 'stream': True}).encode(), 'more_body': False}
        response = await endpoint(Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions', 'headers': []}, receive))
        initial = controller.request_history.page({}, [])['items'][0]
        assert initial['status'] == 'running' and initial['revision'] == 1
        text = await response.body_iterator.__anext__()
        if terminal == 'cancelled':
            await response.body_iterator.aclose()
        else:
            async for piece in response.body_iterator:
                text += piece
        record = controller.request_history.page({}, [])['items'][0]
        assert record['status'] == terminal and record['revision'] == 2
        assert 'modloader_request' not in text and 'external_history' not in text
        assert 'private-reply' not in json.dumps(record)
    finally:
        if response is not None:
            await response.body_iterator.aclose()
        await client.aclose()
        await controller.close()


async def test_settings_history_validation_backfill_and_retry(controller, tmp_path, monkeypatch):
    async def handler(request):
        return httpx.Response(200, json={'choices': []})
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            assert controller.safe_settings()['api_history_enabled']
            assert controller.safe_settings()['api_history_retention_days'] == 0
            await client.patch('/api/settings', json={'api_history_enabled': False})
            assert await asyncio.to_thread(controller.request_history.archive.drain)
            assert controller.safe_settings()['api_history_status']['state'] == 'disabled'
            await client.post('/v1/chat/completions', json={'model': 'one', 'messages': []})
            for invalid in ({'api_history_enabled': 'yes'}, {'api_history_directory': ''}, {'api_history_directory': 12}):
                assert (await client.patch('/api/settings', json=invalid)).status_code == 400
            directory = tmp_path / 'chosen-history'
            result = await client.patch('/api/settings', json={'api_history_enabled': True, 'api_history_directory': str(directory)})
            assert result.status_code == 200 and result.json()['api_history_enabled']
            assert await asyncio.to_thread(controller.request_history.archive.drain)
            assert (await client.post('/api/request-history/retry')).json()['state'] == 'saved'
            assert len(list(directory.glob('*.jsonl'))) == 1
    finally:
        await controller.close()


def test_independent_concurrent_tool_chains_resume_without_cross_grouping(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    first, second = RequestMetrics(False), RequestMetrics(False)
    ledger.start(first, instance(), messages)
    ledger.start(second, instance(), [{'role': 'user', 'content': 'other'}])
    for metrics, key in ((first, 'first-call'), (second, 'second-call')):
        ledger.observe(metrics.token, {'choices': [{'message': assistant([key]), 'finish_reason': 'tool_calls'}]})
        metrics.finish('completed')
        ledger.finish(metrics)
    final, _ = complete(ledger, follow(messages, assistant(['first-call'])), 'answer')
    other, _ = complete(ledger, follow([{'role': 'user', 'content': 'other'}], assistant(['second-call'])), 'other answer')
    assert ledger.records[final.token]['turn_id'] == ledger.records[first.token]['turn_id'] is not None
    assert ledger.records[other.token]['turn_id'] == ledger.records[second.token]['turn_id'] is not None
    assert ledger.records[final.token]['turn_id'] != ledger.records[other.token]['turn_id']


def test_live_observations_are_attributable_and_persisted_at_terminal(ledger, tmp_path):
    ledger.configure(True, tmp_path / 'history')
    metrics = RequestMetrics(True)
    ledger.start(metrics, instance(), [])
    ledger.capture_live(metrics.token, {'request_key': 'different', 'scope': 'request', 'generated_tokens': 999})
    assert 'live' not in ledger.records[metrics.token]
    ledger.capture_live(metrics.token, {'request_key': metrics.token, 'scope': 'model', 'generated_tokens': 999})
    assert 'live' not in ledger.records[metrics.token]
    ledger.capture_live(metrics.token, {'request_key': metrics.token, 'scope': 'request', 'generated_tokens': 20,
        'phase': 'output', 'average_tokens_per_second': 10, 'average_basis': 'backend_decode', 'reason': 'private-error', 'private': 'private-text'})
    assert ledger.latest('one')['requests'][0]['live']['generated_tokens'] == 20
    metrics.finish('completed')
    ledger.finish(metrics)
    assert ledger.archive.drain()
    restored = ledger.archive.read()[0][ledger.run_id]['records'][metrics.token]
    assert restored['live']['average_basis'] == 'backend_decode'
    assert 'private' not in json.dumps(restored)
    lines = next((tmp_path / 'history').glob('*.jsonl')).read_text(encoding='utf-8').splitlines()
    assert len(lines) == 4  # Day header, run, start, terminal; no per-sample writes.


async def test_exact_request_filter_retains_pinned_old_request(controller, tmp_path, monkeypatch):
    async def handler(request):
        return httpx.Response(200, json={'choices': []})
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            for _ in range(55):
                await client.post('/v1/chat/completions', json={'model': 'one', 'messages': []})
            first = next(iter(controller.request_history.records))
            default = (await client.get('/api/request-history')).json()
            assert first not in {record['request_key'] for record in default['items']}
            exact = (await client.get('/api/request-history', params={'run_id': controller.request_history.run_id, 'instance_id': 'one-instance', 'request_key': first})).json()
            assert exact['total'] == 1 and exact['items'][0]['request_key'] == first
            mismatch = (await client.get('/api/request-history', params={'instance_id': 'different', 'request_key': first})).json()
            assert mismatch['total'] == 0
    finally:
        await controller.close()


async def test_gateway_twelve_request_chain_streamed_and_nonstream(controller, tmp_path, monkeypatch):
    calls = []
    async def handler(request):
        payload = json.loads(request.content)
        index = len(calls)
        calls.append(payload)
        ids = [f'round-{index}-{position}' for position in range(2 if index < 6 else 1)] if index < 11 else []
        message = assistant(ids, 'final answer' if not ids else None)
        if payload.get('stream'):
            chunks = []
            for fragment in (0, 1):
                tools = [{'index': position, 'id': key[:5] if fragment == 0 else key[5:],
                    'function': {'name': 'look' if fragment == 0 else 'up',
                    'arguments': '{"private":' if fragment == 0 else '"arguments-sentinel"}'}} for position, key in enumerate(ids)]
                chunks.append(sse({'choices': [{'index': 0, 'delta': {'tool_calls': tools} if ids else {'content': 'final ' if fragment == 0 else 'answer'}}]}))
            chunks.extend([sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if ids else 'stop'}], 'usage': {'completion_tokens': 3}}), sse('[DONE]')])
            return httpx.Response(200, stream=GatewayStream(chunks))
        return httpx.Response(200, json={'choices': [{'message': message, 'finish_reason': 'tool_calls' if ids else 'stop'}], 'usage': {'completion_tokens': 3}})
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            messages = [{'role': 'user', 'content': 'question'}]
            for index in range(12):
                response = await client.post('/v1/chat/completions', json={'model': 'one', 'messages': messages, 'stream': index % 2 == 1})
                assert response.status_code == 200
                ids = [f'round-{index}-{position}' for position in range(2 if index < 6 else 1)] if index < 11 else []
                messages = follow(messages, assistant(ids, 'final answer' if not ids else None))
            latest = controller.request_history.latest(instances['one'].id)
            assert latest['state'] == 'completed' and len(latest['requests']) == 1
            assert latest['aggregation'] == 'request' and latest['turn_id'] is None
            assert all(record['turn_id'] is None for record in controller.request_history.records.values())
            page = (await client.get('/api/request-history')).json()
            assert page['total'] == 12
            assert len(page['items']) == 12
            assert sum(record['usage']['completion_tokens'] for record in page['items']) == 36
            assert [record['tool_names'] for record in reversed(page['items'])] == [['lookup']] * 11 + [[]]
    finally:
        await controller.close()


def test_identical_clock_timestamps_preserve_start_order(ledger, monkeypatch):
    import modloader.request_metrics as module
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda: 1000, monotonic=lambda: 1))
    first, _ = complete(ledger, [], 'first')
    second, _ = complete(ledger, [], 'second')
    assert ledger.latest('one')['requests'][0]['request_key'] == second.token
    assert [record['request_key'] for record in ledger.page({}, [])['items']] == [second.token, first.token]


@pytest.mark.parametrize('streaming', [False, True])
def test_tool_arguments_json_reserialization_preserves_chain(ledger, streaming):
    messages = [{'role': 'user', 'content': 'question'}]
    original = {'path': 'C:\\fixture\\marker.txt', 'options': {'b': 2, 'a': '含 空格'}}
    args = json.dumps(original, ensure_ascii=True, indent=2)
    response = assistant(['call-json'])
    response['tool_calls'][0]['function']['arguments'] = args
    metrics = RequestMetrics(streaming)
    ledger.start(metrics, instance(), messages)
    if streaming:
        for index, fragment in enumerate((args[:9], args[9:23], args[23:])):
            function = {'arguments': fragment, **({'name': 'lookup'} if index == 0 else {})}
            call = {'index': 0, 'function': function, **({'id': 'call-json'} if index == 0 else {})}
            ledger.observe(metrics.token, {'choices': [{'index': 0, 'delta': {'tool_calls': [call]}}]})
        ledger.observe(metrics.token, {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})
    else:
        ledger.observe(metrics.token, {'choices': [{'message': response, 'finish_reason': 'tool_calls'}]})
    metrics.finish('completed')
    ledger.finish(metrics)
    response['tool_calls'][0]['function']['arguments'] = json.dumps({'options': {'a': '含 空格', 'b': 2}, 'path': original['path']}, ensure_ascii=False, separators=(',', ':'))
    final, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[final.token]['turn_id'] == ledger.records[metrics.token]['turn_id'] is not None
    assert 'arguments' not in repr(ledger.boundaries)


def test_meaningful_tool_argument_rewrite_does_not_link(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    _, response = complete(ledger, messages, ids=['call-json'])
    response['tool_calls'][0]['function']['arguments'] = '{"private":"different value"}'
    final, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[final.token]['turn_id'] is None


def test_id_emitted_by_ambiguous_multichoice_disqualifies_later_match(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    metrics = RequestMetrics(True)
    ledger.start(metrics, instance(), [{'role': 'user', 'content': 'other'}])
    ledger.observe(metrics.token, {'choices': [
        {'index': 0, 'message': assistant(['duplicate']), 'finish_reason': 'tool_calls'},
        {'index': 1, 'message': assistant(['other']), 'finish_reason': 'tool_calls'}]})
    metrics.finish('completed')
    ledger.finish(metrics)
    _, response = complete(ledger, messages, ids=['duplicate'])
    final, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[final.token]['turn_id'] is None


def test_interrupted_regroup_snapshot_marks_record_incomplete_without_changing_request_status(ledger, tmp_path):
    metrics, _ = complete(ledger, [], ids=['call'])
    record = ledger.records[metrics.token]
    record.update(aggregation='turn', turn_id='interrupted-turn', turn_state='running')
    ledger.configure(True, tmp_path / 'history')
    assert ledger.archive.drain()
    restored = ledger.archive.read()[0][ledger.run_id]['records'][metrics.token]
    assert restored['record_incomplete'] and restored['status'] == 'completed'


def test_unique_complete_stop_sequence_without_new_user_upgrades_turn(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    first, response = complete(ledger, messages, 'intermediate answer')
    assert ledger.latest('one')['state'] == 'completed'
    last, _ = complete(ledger, messages + [response], 'continued answer')
    assert ledger.records[first.token]['turn_id'] == ledger.records[last.token]['turn_id'] is not None
    assert ledger.latest('one')['state'] == 'completed'
    assert len(ledger.latest('one')['requests']) == 2
    fresh, _ = complete(ledger, messages + [response, {'role': 'user', 'content': 'new question'}], 'new answer')
    assert ledger.records[fresh.token]['turn_id'] is None


def test_ambiguous_complete_stop_sequence_does_not_link(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    _, response = complete(ledger, messages, 'same answer')
    complete(ledger, messages, 'same answer')
    last, _ = complete(ledger, messages + [response], 'continued answer')
    assert ledger.records[last.token]['turn_id'] is None


@pytest.mark.parametrize('failure', ['failed', 'cancelled', 'incomplete'])
def test_exact_serial_retry_remains_separate_request_in_evidenced_turn(ledger, failure):
    messages = [{'role': 'user', 'content': 'question'}]
    parent, response = complete(ledger, messages, ids=['retry-tool'])
    continuation = follow(messages, response)
    failed, _ = complete(ledger, continuation, 'partial', status=failure)
    assert ledger.latest('one')['state'] == failure
    retried, _ = complete(ledger, continuation, 'finished')
    records = [ledger.records[request.token] for request in (parent, failed, retried)]
    assert len({record['request_key'] for record in records}) == 3
    assert len({record['turn_id'] for record in records}) == 1
    assert records[1]['status'] == failure
    expected_state = 'recovered' if failure == 'failed' else 'completed'
    assert all(record['turn_state'] == expected_state for record in records)
    successful_fork, _ = complete(ledger, continuation, 'another success')
    assert ledger.records[successful_fork.token]['turn_id'] is None


@pytest.mark.parametrize('change', ['tool_result'])
def test_serial_retry_requires_exact_nonconcurrent_failed_child(ledger, change):
    messages = [{'role': 'user', 'content': 'question'}]
    _, response = complete(ledger, messages, ids=['retry-tool'])
    continuation = follow(messages, response)
    failed, _ = complete(ledger, continuation, status='failed')
    if change == 'tool_result':
        continuation[-1]['content'] = 'rewritten result'
    else:
        ledger.boundaries[failed.token]['concurrent'] = True
    final, _ = complete(ledger, continuation, 'answer')
    assert ledger.records[final.token]['turn_id'] is None


def test_archive_parses_only_selected_or_changed_files(tmp_path, monkeypatch):
    from datetime import datetime
    directory = tmp_path / 'archives'
    first, second = RequestHistory(directory, False), RequestHistory(directory, False)
    complete(first, [], 'first', started_at=datetime(2026, 10, 1, 12).timestamp())
    complete(second, [], 'second', started_at=datetime(2026, 10, 2, 12).timestamp())
    first.configure(True, directory)
    second.configure(True, directory)
    assert first.archive.drain() and second.archive.drain()
    reader = RequestHistory(directory, False).archive
    original = reader._read_file
    parsed = []
    def tracked(path):
        parsed.append(path.name)
        return original(path)
    monkeypatch.setattr(reader, '_read_file', tracked)
    saved, issues = reader.read(date_from='2026-10-01', date_to='2026-10-01')
    assert set(saved) == {first.run_id} and not issues
    assert parsed == ['2026-10-01.jsonl']
    reader.read(date_from='2026-10-01', date_to='2026-10-01')
    assert len(parsed) == 1
    reader.read()
    baseline = len(parsed)
    reader.read()
    assert len(parsed) == baseline
    complete(second, [], 'updated', started_at=datetime(2026, 10, 2, 13).timestamp())
    assert second.archive.drain()
    reader.read(date_from='2026-10-01', date_to='2026-10-01')
    assert len(parsed) == baseline
    saved, _ = reader.read()
    assert len(parsed) == baseline + 1 and parsed[-1] == '2026-10-02.jsonl'
    assert len(saved[second.run_id]['records']) == 2
    path = directory / '2026-10-01.jsonl'
    stat = path.stat()
    import os
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    reader.read(date_from='2026-10-01', date_to='2026-10-01')
    assert parsed[-1] == '2026-10-01.jsonl' and len(parsed) == baseline + 2
    assert reader.read('../outside')[0] == {}


@pytest.mark.parametrize('code', [13, 28])
def test_archive_os_error_contains_diagnostic_code_without_raw_payload(ledger, tmp_path, monkeypatch, code):
    import os
    def failed(directory, data):
        raise OSError(code, 'private-payload-sentinel')
    monkeypatch.setattr(ledger.archive, '_append', failed)
    ledger.configure(True, tmp_path / 'history')
    assert not ledger.archive.drain()
    message = ledger.archive.status()['error']
    assert f'errno={code}' in message and os.strerror(code) in message
    assert 'private-payload-sentinel' not in message


def test_reported_tool_names_survive_archive_without_tool_payloads(ledger, tmp_path):
    metrics = RequestMetrics(False)
    ledger.start(metrics, instance(), [{'role': 'user', 'content': 'prompt-sentinel'}])
    response = assistant(['first', 'second', 'third'], 'reply-sentinel')
    for call, name in zip(response['tool_calls'], ['read', 'web_search', 'read']):
        call['function']['name'] = name
        call['function']['description'] = 'description-sentinel'
    ledger.observe(metrics.token, {'choices': [{'message': response, 'finish_reason': 'tool_calls'}]})
    metrics.finish('completed')
    ledger.finish(metrics)
    assert ledger.records[metrics.token]['tool_names'] == ['read', 'web_search']
    ledger.configure(True, tmp_path / 'history')
    assert ledger.archive.drain()
    reader = RequestHistory(tmp_path / 'history', enabled=False)
    saved, issues = reader.archive.read()
    assert not issues
    record = reader.page(saved, issues, ledger.run_id)['items'][0]
    assert record['tool_names'] == ['read', 'web_search']
    text = next((tmp_path / 'history').glob('*.jsonl')).read_text(encoding='utf-8')
    assert not any(secret in text for secret in ('prompt-sentinel', 'reply-sentinel', 'arguments-sentinel', 'description-sentinel'))


@pytest.mark.parametrize('finish', [None, 'tool_calls'])
def test_interrupted_stream_never_labels_partial_tool_name(ledger, finish):
    metrics = RequestMetrics(True)
    ledger.start(metrics, instance(), [])
    ledger.observe(metrics.token, {'choices': [{'index': 0, 'delta': {'tool_calls': [
        {'index': 0, 'id': 'call', 'function': {'name': 'web_', 'arguments': ''}}]}}]})
    if finish:
        ledger.observe(metrics.token, {'choices': [{'index': 0, 'delta': {'tool_calls': [
            {'index': 0, 'function': {'name': 'search', 'arguments': '{}'}}]}, 'finish_reason': finish}]})
    metrics.finish('cancelled')
    ledger.finish(metrics)
    assert ledger.records[metrics.token]['tool_names'] == (['web_search'] if finish else [])
    assert ledger.records[metrics.token]['status'] == 'cancelled'


def test_archive_tool_names_are_optional_and_whitelisted(ledger, tmp_path):
    metrics, _ = complete(ledger, [], 'answer')
    record = deepcopy(ledger.records[metrics.token])
    record.pop('tool_names')
    record['tool_calls'] = [{'function': {'name': 'hidden-name', 'arguments': 'private-arguments'}}]
    path = tmp_path / 'history' / (ledger.run_id + '.jsonl')
    path.parent.mkdir()
    path.write_text(json.dumps({'schema': 1, 'type': 'run', 'id': ledger.run_id, 'started_at': ledger.started_at}) + '\n' + json.dumps({'schema': 1, 'type': 'request', 'record': record}) + '\n', encoding='utf-8')
    reader = RequestHistory(path.parent, enabled=False)
    saved, issues = reader.archive.read()
    assert not issues
    restored = saved[ledger.run_id]['records'][metrics.token]
    assert restored['tool_names'] == [] and 'tool_calls' not in restored
    record.update(revision=record['revision'] + 1, tool_names=['read', {'name': 'hidden-name'}, None, '', 'read', 'bash'])
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'schema': 1, 'type': 'request', 'record': record}) + '\n')
    saved, issues = reader.archive.read()
    assert not issues
    assert saved[ledger.run_id]['records'][metrics.token]['tool_names'] == ['read', 'bash']


async def test_summary_and_context_reduction_stay_unclassified_but_fully_metered(controller, tmp_path, monkeypatch):
    usages = [
        {'prompt_tokens': 172532, 'completion_tokens': 4067, 'total_tokens': 176599, 'prompt_tokens_details': {'cached_tokens': 171803}},
        {'prompt_tokens': 5000, 'completion_tokens': 100, 'total_tokens': 5100},
    ]
    replies = []
    async def handler(request):
        body = {'choices': [{'message': assistant(content='summary-sentinel'), 'finish_reason': 'stop'}], 'usage': usages[len(replies)]}
        replies.append(body)
        return httpx.Response(200, json=body)
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            controller.request_history.configure(True, tmp_path / 'history')
            for content in ('Compress the previous conversation into a summary.', 'Continue with the shortened context.'):
                response = await client.post('/v1/chat/completions', json={'model': 'one', 'messages': [{'role': 'user', 'content': content}]})
                assert response.status_code == 200 and response.json() == {**replies[-1], 'model': 'one'}
            page = (await client.get('/api/request-history')).json()
            assert page['total'] == 2
            records = list(reversed(page['items']))
            assert [record['usage'] for record in records] == usages
            assert all(record['status'] == 'completed' and record['aggregation'] == 'request'
                       and record['labels'] == [] and record['tool_names'] == [] for record in records)
            assert controller.request_history.archive.drain()
            saved, issues = RequestHistory(tmp_path / 'history', enabled=False).archive.read()
            assert not issues
            restored = saved[controller.request_history.run_id]['records']
            assert [restored[record['request_key']]['usage'] for record in records] == usages
            assert all('kind' not in record for record in restored.values())
    finally:
        await controller.close()


@pytest.mark.parametrize('streaming', [False, True])
def test_unique_tool_evidence_survives_omitted_reasoning_and_unrelated_overlap(ledger, streaming):
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'question'}]
    main = RequestMetrics(streaming)
    ledger.start(main, instance(), messages)
    side, _ = complete(ledger, [{'role': 'user', 'content': 'independent title'}], 'title')
    assert ledger.latest('one')['requests'][0]['request_key'] == main.token
    response = assistant(['reason-tool'], 'checking')
    response['reasoning_content'] = 'private-thought-sentinel'
    key = 'delta' if streaming else 'message'
    if streaming:
        response['tool_calls'][0]['index'] = 0
    ledger.observe(main.token, {'choices': [{'index': 0, key: response, 'finish_reason': 'tool_calls'}]})
    main.finish('completed')
    ledger.finish(main)
    response.pop('reasoning_content')
    final, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[main.token]['turn_id'] == ledger.records[final.token]['turn_id'] is not None
    assert ledger.records[side.token]['turn_id'] is None
    assert ledger.records[main.token]['tool_names'] == ['lookup']
    assert 'private-thought-sentinel' not in repr(ledger.boundaries) + json.dumps(ledger.records)


def test_tool_evidence_does_not_accept_body_rewrite_when_reasoning_is_omitted(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    main = RequestMetrics(False)
    ledger.start(main, instance(), messages)
    response = assistant(['body-tool'], 'original body')
    ledger.observe(main.token, {'choices': [{'message': {**response, 'reasoning_content': 'thought'}, 'finish_reason': 'tool_calls'}]})
    main.finish('completed')
    ledger.finish(main)
    response['content'] = 'rewritten body'
    final, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[final.token]['turn_id'] is None


def test_short_concurrent_auxiliary_does_not_replace_active_display_or_tool_retry(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    parent, response = complete(ledger, messages, ids=['retry-after-side'])
    child = RequestMetrics(False)
    ledger.start(child, instance(), follow(messages, response))
    side, _ = complete(ledger, [{'role': 'user', 'content': 'auxiliary'}], 'side')
    assert ledger.latest('one')['turn_id'] == ledger.records[parent.token]['turn_id']
    assert ledger.latest('one')['state'] == 'running'
    child.finish('failed')
    ledger.finish(child)
    retried, _ = complete(ledger, follow(messages, response), 'answer')
    assert ledger.records[retried.token]['turn_id'] == ledger.records[parent.token]['turn_id']
    assert ledger.records[side.token]['turn_id'] is None


def test_multiple_active_requests_choose_oldest_until_it_finishes(ledger):
    first, second = RequestMetrics(False), RequestMetrics(False)
    ledger.start(first, instance(), [{'role': 'user', 'content': 'first'}])
    ledger.start(second, instance(), [{'role': 'user', 'content': 'second'}])
    assert ledger.latest('one')['requests'][0]['request_key'] == first.token
    first.finish('completed')
    ledger.finish(first)
    assert ledger.latest('one')['requests'][0]['request_key'] == second.token
    second.finish('completed')
    ledger.finish(second)
    assert ledger.latest('one')['requests'][0]['request_key'] == second.token


def day_time(day, hour=12):
    from datetime import datetime
    return datetime.fromisoformat(day).replace(hour=hour).timestamp()


def test_daily_multiple_runs_midnight_revisions_and_full_turn_query(tmp_path):
    directory = tmp_path / 'daily'
    first, second = RequestHistory(directory, False), RequestHistory(directory, False)
    messages = [{'role': 'user', 'content': 'question'}]
    parent, response = complete(first, messages, ids=['midnight-tool'], started_at=day_time('2026-10-01', 23))
    child, _ = complete(first, follow(messages, response), 'answer', started_at=day_time('2026-10-02', 0))
    other, _ = complete(second, [], 'different run', started_at=day_time('2026-10-02', 1))
    first.configure(True, directory)
    second.configure(True, directory)
    assert first.archive.drain() and second.archive.drain()
    from modloader.request_history import ARCHIVE_ID
    for day in ('2026-10-01', '2026-10-02'):
        rows = [json.loads(line) for line in (directory / (day + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert rows[0] == {'application': ARCHIVE_ID, 'schema': 2, 'type': 'day', 'date': day}
        assert all(row['record']['date'] == day for row in rows if row['type'] == 'request')
    reopened = RequestHistory(directory, False)
    saved, issues = reopened.archive.read()
    assert not issues
    days = reopened.dates(saved, issues)['dates']
    assert [(item['date'], item['request_count']) for item in days] == [('2026-10-02', 2), ('2026-10-01', 1)]
    assert len(days[0]['instances']) == 2 and all('started_at' in item for item in days[0]['instances'])
    page = reopened.page(saved, issues, date_from='2026-10-02', date_to='2026-10-02')
    assert page['run_id'] is None and page['total'] == 2
    turn = reopened.page(saved, issues, first.run_id, turn_id=first.records[parent.token]['turn_id'], date_from='2026-10-02', date_to='2026-10-02')
    assert [item['request_key'] for item in turn['items']] == [parent.token, child.token]
    assert first.records[parent.token]['date'] == '2026-10-01'
    assert first.records[parent.token]['turn_state'] == 'completed'


@pytest.mark.parametrize('days,old,keep', [(7, '2026-10-23', '2026-10-24'), (30, '2026-09-30', '2026-10-01'), (0, '2020-01-01', '2026-10-30')])
def test_retention_inclusive_local_days_and_permanent_keeps_memory(tmp_path, monkeypatch, days, old, keep):
    import modloader.request_history as module
    real_date = module.local_date
    monkeypatch.setattr(module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    directory = tmp_path / 'daily'
    history = RequestHistory(directory, False)
    old_request, _ = complete(history, [], 'old', started_at=day_time(old))
    complete(history, [], 'keep', started_at=day_time(keep))
    history.configure(True, directory, days)
    assert history.archive.drain()
    assert (directory / (old + '.jsonl')).exists() == (days == 0)
    assert (directory / (keep + '.jsonl')).exists()
    assert old_request.token in history.records  # Retention never purges this run's memory.
    assert history.page({}, [])['total'] == 2


def test_retention_protects_active_requests_and_cross_boundary_turns(tmp_path, monkeypatch):
    import modloader.request_history as module
    real_date = module.local_date
    monkeypatch.setattr(module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    directory = tmp_path / 'daily'
    history = RequestHistory(directory, False)
    active = RequestMetrics(False)
    active.started_at = day_time('2026-10-20')
    history.start(active, instance('active'), [])
    messages = [{'role': 'user', 'content': 'question'}]
    parent, response = complete(history, messages, ids=['cross-day'], started_at=day_time('2026-10-21'))
    complete(history, follow(messages, response), 'answer', started_at=day_time('2026-10-29'))
    history.configure(True, directory, 7)
    assert history.archive.drain()
    assert (directory / '2026-10-20.jsonl').exists()
    assert (directory / '2026-10-21.jsonl').exists()
    active.finish('completed')
    history.finish(active)
    assert history.archive.drain()
    assert not (directory / '2026-10-20.jsonl').exists()
    assert (directory / '2026-10-21.jsonl').exists()


def legacy_file(directory, history, records, name=None):
    path = directory / ((name or history.run_id) + '.jsonl')
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{'schema': 1, 'type': 'run', 'id': history.run_id, 'started_at': history.started_at}]
    rows.extend({'schema': 1, 'type': 'request', 'record': record} for record in records)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return path


def test_retention_verifies_legacy_and_preserves_unknown_corrupt_and_mixed_dates(tmp_path, monkeypatch):
    import modloader.request_history as module
    real_date = module.local_date
    monkeypatch.setattr(module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    directory = tmp_path / 'daily'
    source = RequestHistory(tmp_path / 'source', False)
    old, _ = complete(source, [], 'old', started_at=day_time('2026-10-01'))
    fresh, _ = complete(source, [], 'fresh', started_at=day_time('2026-10-29'))
    old_record, fresh_record = source.records[old.token], source.records[fresh.token]
    clean_old = legacy_file(directory, source, [old_record])
    unknown = legacy_file(directory, source, [old_record], name='unknown')
    mixed_source = RequestHistory(tmp_path / 'mixed', False)
    mixed_records = [{**record, 'run_id': mixed_source.run_id} for record in (old_record, fresh_record)]
    mixed = legacy_file(directory, mixed_source, mixed_records)
    corrupt_source = RequestHistory(tmp_path / 'corrupt', False)
    corrupt = legacy_file(directory, corrupt_source, [{**old_record, 'run_id': corrupt_source.run_id}])
    with corrupt.open('ab') as stream:
        stream.write(b'broken line\n')
    reader = RequestHistory(directory, True, 7)
    assert reader.archive.drain()
    assert not clean_old.exists()
    assert unknown.exists() and mixed.exists() and corrupt.exists()
    saved, issues = reader.archive.read()
    assert mixed_source.run_id in saved and corrupt_source.run_id in saved
    assert any('unknown.jsonl' in issue for issue in issues)
    assert any(corrupt.name in issue for issue in issues)


def test_disable_and_directory_change_do_not_clean_original_directory(tmp_path, monkeypatch):
    import modloader.request_history as module
    real_date = module.local_date
    monkeypatch.setattr(module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    original, new = tmp_path / 'original', tmp_path / 'new'
    history = RequestHistory(original, False)
    complete(history, [], 'old', started_at=day_time('2026-10-01'))
    history.configure(True, original, 0)
    assert history.archive.drain()
    old_file = original / '2026-10-01.jsonl'
    history.configure(False, original, 7)
    assert history.archive.drain() and old_file.exists()
    history.configure(True, new, 7)
    assert history.archive.drain()
    assert old_file.exists() and not (new / '2026-10-01.jsonl').exists()


def test_cleanup_failure_is_separate_and_retry_does_not_stop_writes(tmp_path, monkeypatch):
    import modloader.request_history as module
    real_date = module.local_date
    monkeypatch.setattr(module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    history = RequestHistory(tmp_path / 'daily', False)
    complete(history, [], 'old', started_at=day_time('2026-10-01'))
    original = history.archive._cleanup
    attempts = []
    def failed_cleanup():
        attempts.append(True)
        raise PermissionError(13, 'private-path-sentinel')
    monkeypatch.setattr(history.archive, '_cleanup', failed_cleanup)
    history.configure(True, tmp_path / 'daily', 7)
    assert history.archive.drain()
    status = history.archive.status()
    assert status['state'] == 'saved' and status['error'] is None and 'errno=13' in status['cleanup_error']
    assert 'private-path-sentinel' not in status['cleanup_error']
    complete(history, [], 'new', started_at=day_time('2026-10-30'))
    assert history.archive.drain() and len(attempts) == 1
    monkeypatch.setattr(history.archive, '_cleanup', original)
    history.archive.retry()
    assert history.archive.drain()
    assert history.archive.status()['cleanup_error'] is None
    assert not (tmp_path / 'daily' / '2026-10-01.jsonl').exists()


def test_concurrent_duplicate_followups_do_not_make_later_retry_unambiguous(ledger):
    messages = [{'role': 'user', 'content': 'question'}]
    parent, response = complete(ledger, messages, ids=['same-parent'])
    continuation = follow(messages, response)
    child = RequestMetrics(False)
    ledger.start(child, instance(), continuation)
    independent, _ = complete(ledger, continuation, 'concurrent success')
    child.finish('failed')
    ledger.finish(child)
    assert ledger.records[independent.token]['turn_id'] is None
    last, _ = complete(ledger, continuation, 'retry')
    assert ledger.records[last.token]['turn_id'] is None
    assert ledger.records[parent.token]['turn_id'] == ledger.records[child.token]['turn_id']


def test_locked_expired_file_is_preserved_and_can_be_cleaned_after_release(tmp_path, monkeypatch):
    import modloader.request_history as module
    real_date = module.local_date
    monkeypatch.setattr(module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    directory = tmp_path / 'daily'
    history = RequestHistory(directory, False)
    complete(history, [], 'old', started_at=day_time('2026-10-01'))
    history.configure(True, directory)
    assert history.archive.drain()
    path = directory / '2026-10-01.jsonl'
    with path.open('r+b') as stream, module.exclusive_log(stream):
        history.configure(True, directory, 7)
        assert history.archive.drain()
        assert path.exists()
    history.archive.retry()
    assert history.archive.drain()
    assert not path.exists()


def test_retention_never_follows_log_symlink(tmp_path, monkeypatch):
    import modloader.request_history as module
    real_date = module.local_date
    monkeypatch.setattr(module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    outside = tmp_path / 'outside'
    source = RequestHistory(outside, False)
    complete(source, [], 'old', started_at=day_time('2026-10-01'))
    source.configure(True, outside)
    assert source.archive.drain()
    target = outside / '2026-10-01.jsonl'
    directory = tmp_path / 'daily'
    directory.mkdir()
    link = directory / target.name
    try:
        link.symlink_to(target)
    except OSError:
        # Windows permits directory junctions without symlink privilege. Keep
        # both endpoints inside this test's disposable artifact directory.
        import os
        import subprocess
        if os.name != 'nt':
            raise
        junction = tmp_path / 'linked-directory'
        created = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(outside)], capture_output=True)
        assert created.returncode == 0, created.stderr.decode(errors='replace')
        history = RequestHistory(junction, True, 7)
        assert history.archive.drain()
        assert target.exists()
        assert '链接' in history.archive.status()['cleanup_error']
    else:
        history = RequestHistory(directory, True, 7)
        assert history.archive.drain()
        assert link.is_symlink() and target.exists()
        assert any('链接' in issue for issue in history.archive.read()[1])


async def test_daily_endpoints_cross_run_filter_exact_query_and_settings(controller, tmp_path, monkeypatch):
    from modloader.request_history import local_date
    async def handler(request):
        return httpx.Response(200, json={'choices': []})
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            directory = tmp_path / 'daily-api'
            result = await client.patch('/api/settings', json={'api_history_directory': str(directory), 'api_history_retention_days': 0})
            assert result.json()['api_history_enabled'] and result.json()['api_history_retention_days'] == 0
            for invalid in (True, -1, 8, '7'):
                assert (await client.patch('/api/settings', json={'api_history_retention_days': invalid})).status_code == 400
            history = controller.request_history
            messages = [{'role': 'user', 'content': 'question'}]
            parent, response = complete(history, messages, ids=['api-midnight'], started_at=day_time('2026-10-01'))
            child, _ = complete(history, follow(messages, response), 'answer', started_at=day_time('2026-10-02'))
            # Legacy turn identifiers remain queryable without enabling inference for new HTTP calls.
            for token in (parent.token, child.token):
                history.records[token].update(aggregation='turn', turn_id='archived-midnight', turn_state='completed')
                history._publish(history.records[token])
            previous = RequestHistory(directory, False)
            old, _ = complete(previous, [], 'other run', started_at=day_time('2026-10-02'))
            previous.configure(True, directory)
            assert previous.archive.drain() and await asyncio.to_thread(history.archive.drain)
            dates = (await client.get('/api/request-history/dates')).json()
            assert dates['today'] == local_date()
            assert dates['dates'][0]['date'] == '2026-10-02' and dates['dates'][0]['request_count'] == 2
            page = (await client.get('/api/request-history', params={'date_from': '2026-10-02', 'date_to': '2026-10-02'})).json()
            assert page['run_id'] is None and page['total'] == 2
            turn = (await client.get('/api/request-history', params={'run_id': history.run_id,
                'turn_id': history.records[parent.token]['turn_id'], 'date_from': '2026-10-02'})).json()
            assert [record['request_key'] for record in turn['items']] == [parent.token, child.token]
            exact = (await client.get('/api/request-history', params={'run_id': previous.run_id,
                'request_key': old.token, 'date_from': '2026-10-03'})).json()
            assert exact['total'] == 1
            assert (await client.get('/api/request-history')).json()['run_id'] == history.run_id
            for params in ({'date_from': '20261002'}, {'date_to': '2026-02-30'}, {'date_from': '2026-10-03', 'date_to': '2026-10-02'}):
                assert (await client.get('/api/request-history', params=params)).status_code == 400
            public_paths = {route.path for route in build_app(controller, public=True).routes}
            assert '/api/request-history/dates' not in public_paths
    finally:
        await controller.close()


async def test_existing_explicit_disabled_setting_is_respected(tmp_path):
    from modloader.app import Controller
    from modloader.store import Store
    root = tmp_path / 'existing'
    root.mkdir()
    store = Store(root / 'modloader.sqlite3')
    store.put('settings', {'id': 'main', 'api_history_enabled': False})
    store.close()
    controller = Controller(root, 'isolated-history-test')
    try:
        settings = controller.safe_settings()
        assert settings['api_history_enabled'] is False
        assert settings['api_history_retention_days'] == 0
        assert settings['api_history_status']['state'] == 'disabled'
        assert not (root / 'logs' / 'api-requests').exists()
    finally:
        await controller.close()


def test_partial_initial_daily_header_retry_preserves_fragment_and_records(ledger, tmp_path, monkeypatch):
    import modloader.request_history as module
    original = ledger.archive._append
    failed = []
    def partial(directory, data):
        if not failed:
            failed.append(True)
            event = json.loads(data)
            folder = tmp_path / 'history'
            folder.mkdir()
            path = folder / (event['date'] + '.jsonl')
            header = module._json({'application': module.ARCHIVE_ID, 'schema': 2, 'type': 'day', 'date': event['date']}).encode()
            path.write_bytes(header[:12])
            ledger.archive.created_paths.add(path)
            raise OSError(28, 'private-error')
        original(directory, data)
    monkeypatch.setattr(ledger.archive, '_append', partial)
    ledger.configure(True, tmp_path / 'history')
    request, _ = complete(ledger, [], 'answer')
    assert not ledger.archive.drain()
    ledger.archive.retry()
    assert ledger.archive.drain()
    saved, issues = ledger.archive.read()
    assert saved[ledger.run_id]['records'][request.token]['status'] == 'completed'
    assert issues and next((tmp_path / 'history').glob('*.jsonl')).read_bytes().startswith(b'{"applicatio\n')


def test_existing_unknown_daily_file_is_not_appended_or_deleted(ledger, tmp_path):
    from modloader.request_history import local_date
    directory = tmp_path / 'history'
    directory.mkdir()
    path = directory / (local_date() + '.jsonl')
    original = b'{"personal":"keep"}\n'
    path.write_bytes(original)
    ledger.configure(True, directory, 7)
    complete(ledger, [], 'answer')
    assert not ledger.archive.drain()
    assert path.read_bytes() == original
    assert ledger.archive.status()['state'] == 'error'


@pytest.mark.parametrize('nested', [False, True])
async def test_settings_junction_path_survives_save_reload_and_blocks_only_cleanup(controller, tmp_path, monkeypatch, nested):
    import os
    import subprocess
    from pathlib import Path
    import modloader.request_history as history_module
    from modloader.app import Controller
    real_date = history_module.local_date
    monkeypatch.setattr(history_module, 'local_date', lambda timestamp=None: '2026-10-30' if timestamp is None else real_date(timestamp))
    outside = tmp_path / 'outside'
    outside.mkdir()
    destination = outside / 'nested' if nested else outside
    source = RequestHistory(destination, False)
    complete(source, [], 'old', started_at=day_time('2026-10-01'))
    source.configure(True, destination)
    assert source.archive.drain()
    old_file = destination / '2026-10-01.jsonl'
    chosen = tmp_path / 'chosen-junction'
    if os.name == 'nt':
        created = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(chosen), str(outside)], capture_output=True)
        assert created.returncode == 0, created.stderr.decode(errors='replace')
    else:
        chosen.symlink_to(outside, target_is_directory=True)
    selected = chosen / 'nested' if nested else chosen
    async def handler(request):
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'answer'}, 'finish_reason': 'stop'}]})
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    closed = False
    try:
        async with client:
            # Exercise absolute normalization without erasing the selected link.
            supplied = os.path.relpath(selected, Path.cwd()) if nested else str(selected)
            response = await client.patch('/api/settings', json={'api_history_directory': supplied,
                'api_history_enabled': True, 'api_history_retention_days': 7})
            assert response.status_code == 200
            expected = os.path.abspath(selected)
            assert response.json()['api_history_directory'] == expected
            assert controller.store.get('settings', 'main')['api_history_directory'] == expected
            assert await asyncio.to_thread(controller.request_history.archive.drain)
            status = controller.safe_settings()['api_history_status']
            assert status['state'] == 'saved' and status['error'] is None and '链接' in status['cleanup_error']
            assert old_file.exists()
            generated = await client.post('/v1/chat/completions', json={'model': 'one', 'messages': []})
            assert generated.status_code == 200
            assert await asyncio.to_thread(controller.request_history.archive.drain)
            saved, issues = controller.request_history.archive.read()
            assert not issues and saved[controller.request_history.run_id]['records']
            assert (destination / '2026-10-30.jsonl').exists() and old_file.exists()
        await controller.close()
        closed = True
        reloaded = Controller(controller.paths.root, 'isolated-reloaded')
        try:
            assert reloaded.settings['api_history_directory'] == expected
            assert await asyncio.to_thread(reloaded.request_history.archive.drain)
            status = reloaded.safe_settings()['api_history_status']
            assert status['error'] is None and '链接' in status['cleanup_error']
            assert old_file.exists()
        finally:
            await reloaded.close()
    finally:
        if not closed:
            await controller.close()


def aggregate_record(history, key, started_at, *, turn_id=None, instance_id='one',
                     status='completed', duration_s=2, usage=None, timings=None, **extra):
    from modloader.request_history import local_date
    return {'run_id': history.run_id, 'instance_id': instance_id, 'alias': 'Model ' + instance_id,
            'backend': 'llama_cpp', 'request_key': key, 'revision': 1, 'streaming': True,
            'aggregation': 'turn' if turn_id else 'request', 'turn_id': turn_id,
            'turn_state': status if turn_id else None, 'status': status,
            'started_at': started_at, 'ended_at': started_at + duration_s, 'duration_s': duration_s,
            'date': local_date(started_at), 'first_output_s': None,
            'usage': usage or {}, 'timings': timings or {}, 'prefill': {}, 'labels': [],
            'tool_names': [], **extra}


def test_grouped_page_has_single_request_groups_and_filter_wide_totals_before_paging(ledger):
    start = day_time('2026-10-30')
    for index, turn in enumerate(('first', 'first', 'first', 'second', 'second', None)):
        key = f'request-{index}'
        ledger.records[key] = aggregate_record(ledger, key, start + index * 10, turn_id=turn,
            usage={'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 150,
                   'prompt_tokens_details': {'cached_tokens': 40}},
            timings={'prompt_n': 60, 'prompt_ms': 600, 'predicted_n': 10, 'predicted_ms': 200})

    first = ledger.page({}, [], group_by='turn', limit=1)
    assert first['total'] == 6 and first['group_total'] == 3
    assert len(first['items']) == len(first['groups']) == 1
    single = first['groups'][0]
    assert single['aggregation'] == 'request' and single['turn_id'] is None
    assert single['request_key'] == first['items'][0]['request_key'] == 'request-5'
    assert single['request_count'] == 1 and single['input'] == {'value': 100, 'reported': 1}
    assert single['started_at'] == start + 50 and single['ended_at'] == start + 52
    assert single['elapsed_s'] == 2 and single['status'] == 'completed'
    summary = first['summary']
    assert summary['request_count'] == 6
    assert summary['input'] == {'value': 600, 'reported': 6}
    assert summary['output'] == {'value': 60, 'reported': 6}
    assert summary['total'] == {'value': 660, 'reported': 6}  # Cache is already included in input.
    assert summary['cached'] == {'value': 240, 'reported': 6}
    assert summary['cache_pair'] == {'input': 600, 'cached': 240, 'reported': 6}
    assert summary['cache_ratio'] == pytest.approx(0.4)
    assert summary['decode'] == {'value': pytest.approx(50), 'tokens': 60, 'seconds': pytest.approx(1.2), 'reported': 6}
    assert summary['prefill'] == {'value': pytest.approx(100), 'tokens': 360, 'seconds': pytest.approx(3.6), 'reported': 6}

    second = ledger.page({}, [], group_by='turn', offset=1, limit=1)
    third = ledger.page({}, [], group_by='turn', offset=2, limit=1)
    assert second['summary'] == third['summary'] == summary
    assert second['groups'][0]['request_count'] == 2
    assert second['groups'][0]['started_at'] == start + 30
    assert second['groups'][0]['elapsed_s'] == 12
    assert third['groups'][0]['request_count'] == 3
    assert [page['items'][0]['request_key'] for page in (first, second, third)] == ['request-5', 'request-4', 'request-2']
    exhausted = ledger.page({}, [], group_by='turn', offset=3, limit=1)
    assert exhausted['items'] == exhausted['groups'] == [] and exhausted['summary'] == summary


def test_grouped_summary_preserves_partial_usage_failure_and_paired_cache_coverage(ledger):
    start = day_time('2026-10-30')
    records = [
        aggregate_record(ledger, 'complete', start,
            usage={'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 999,
                   'ttft_ms': 200, 'prompt_tokens_details': {'cached_tokens': 30}},
            timings={'prompt_n': 7, 'prompt_ms': 1000, 'predicted_n': 10, 'predicted_ms': 200},
            first_output_s=0.9),
        aggregate_record(ledger, 'failed', start + 10, status='failed',
            usage={'prompt_tokens': 50}, timings={'cache_n': 20}, first_output_s=1.2),
        aggregate_record(ledger, 'cache-only', start + 20, status='failed',
            usage={'prompt_tokens_details': {'cached_tokens': 9}}),
        aggregate_record(ledger, 'output-only', start + 30, status='cancelled',
            usage={'completion_tokens': 4}),
        aggregate_record(ledger, 'lost-terminal', start + 40, status='incomplete', record_incomplete=True),
    ]
    ledger.records.update({record['request_key']: record for record in records})
    summary = ledger.page({}, [], group_by='turn', limit=1)['summary']
    assert summary['request_count'] == 5
    assert summary['input'] == {'value': 150, 'reported': 2}
    assert summary['output'] == {'value': 14, 'reported': 2}
    assert summary['total'] == {'value': 110, 'reported': 1}
    assert summary['cached'] == {'value': 59, 'reported': 3}
    assert summary['cache_pair'] == {'input': 150, 'cached': 50, 'reported': 2}
    assert summary['cache_ratio'] == pytest.approx(1 / 3)
    assert summary['prefill'] == {'value': 7, 'tokens': 7, 'seconds': 1, 'reported': 1}
    assert summary['ttft'] == {'value': pytest.approx(0.7), 'reported': 2, 'engine_reported': 1, 'gateway_reported': 1}
    assert summary['status_counts'] == {'running': 0, 'completed': 1, 'failed': 2, 'cancelled': 1, 'incomplete': 1}
    assert summary['record_incomplete_count'] == 1
    unknown = ledger.page({}, [], group_by='turn')['groups'][0]
    assert all(unknown[key] == {'value': None, 'reported': 0} for key in ('input', 'output', 'total', 'cached'))
    assert unknown['decode']['value'] is None and unknown['decode']['reported'] == 0


def test_grouped_empty_and_reported_zero_are_distinct(ledger):
    empty = ledger.page({}, [], group_by='turn')['summary']
    assert empty['request_count'] == 0
    assert empty['input'] == empty['output'] == empty['total'] == empty['cached'] == {'value': None, 'reported': 0}
    assert empty['cache_pair'] == {'input': None, 'cached': None, 'reported': 0}
    assert empty['cache_ratio'] is None and empty['ttft']['value'] is None
    ledger.records['zero'] = aggregate_record(ledger, 'zero', day_time('2026-10-30'),
        usage={'prompt_tokens': 0, 'completion_tokens': 0, 'prompt_tokens_details': {'cached_tokens': 0}},
        timings={'prompt_n': 0, 'prompt_ms': 0, 'predicted_n': 0, 'predicted_ms': 0})
    zero = ledger.page({}, [], group_by='turn')['summary']
    assert zero['input'] == zero['output'] == zero['total'] == zero['cached'] == {'value': 0, 'reported': 1}
    assert zero['cache_pair'] == {'input': 0, 'cached': 0, 'reported': 1} and zero['cache_ratio'] is None
    assert zero['prefill'] == zero['decode'] == {'value': None, 'tokens': 0, 'seconds': 0, 'reported': 1}


def test_grouped_rates_weight_native_work_and_elapsed_time_not_request_means(ledger):
    start = day_time('2026-10-30')
    records = [
        aggregate_record(ledger, 'timings', start,
            usage={'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 20}},
            timings={'prompt_n': 70, 'prompt_ms': 1000, 'predicted_n': 20, 'predicted_ms': 500,
                     'prompt_per_second': 700, 'predicted_per_second': 400}),
        aggregate_record(ledger, 'means', start + 10, usage={'completion_tokens': 30},
            timings={'predicted_per_second': 15},
            prefill={'prefill_processed_tokens': 30, 'average_prefill_tokens_per_second': 10,
                     'prefill_average_basis': 'backend_prefill'}),
        aggregate_record(ledger, 'live', start + 20,
            prefill={'prefill_processed_tokens': 50, 'average_prefill_tokens_per_second': 5000,
                     'prefill_average_basis': 'sampled_prefill'},
            live={'request_key': 'live', 'scope': 'request', 'generated_tokens': 5,
                  'average_tokens_per_second': 10, 'average_basis': 'backend_decode',
                  'prefill_processed_tokens': 20, 'average_prefill_tokens_per_second': 20,
                  'prefill_average_basis': 'backend_prefill'}),
    ]
    ledger.records.update({record['request_key']: record for record in records})
    summary = ledger.page({}, [], group_by='turn')['summary']
    assert summary['decode'] == {'value': pytest.approx(55 / 3), 'tokens': 55, 'seconds': 3, 'reported': 3}
    assert summary['prefill'] == {'value': 24, 'tokens': 120, 'seconds': 5, 'reported': 3}


def test_grouped_identity_includes_run_instance_and_ungrouped_request(ledger):
    start = day_time('2026-10-30')
    records = [aggregate_record(ledger, key, start + index, turn_id=turn, instance_id=target)
               for index, (key, turn, target) in enumerate([
                   ('one-a', 'same', 'one'), ('one-b', 'same', 'one'), ('two', 'same', 'two'),
                   ('unclassified-a', None, 'one'), ('unclassified-b', None, 'one')])]
    ledger.records.update({record['request_key']: record for record in records})
    old = {**aggregate_record(ledger, 'previous', start + 10, turn_id='same'), 'run_id': 'previous-run'}
    saved = {'previous-run': {'id': 'previous-run', 'started_at': start, 'records': {'previous': old}}}
    page = ledger.page(saved, [], group_by='turn', date_from='2026-10-30', date_to='2026-10-30')
    assert page['total'] == 6 and page['group_total'] == 5
    assert sorted(group['request_count'] for group in page['groups']) == [1, 1, 1, 1, 2]
    assert len({(group['run_id'], group['instance_id'], group['turn_id'],
                 group['request_key'] if group['turn_id'] is None else None) for group in page['groups']}) == 5
    filtered = ledger.page(saved, [], ledger.run_id, 'one', group_by='turn')
    assert filtered['group_total'] == 3 and filtered['summary']['request_count'] == 4


def test_grouped_dates_select_full_retained_turns_but_keep_filter_wide_totals_and_exact_details(ledger):
    first = aggregate_record(ledger, 'before-midnight', day_time('2026-10-29'), turn_id='cross-day')
    second = aggregate_record(ledger, 'after-midnight', day_time('2026-10-30'), turn_id='cross-day', status='failed')
    ledger.records.update({record['request_key']: record for record in (first, second)})
    filtered = ledger.page({}, [], group_by='turn', date_from='2026-10-30', date_to='2026-10-30')
    assert filtered['total'] == filtered['group_total'] == filtered['summary']['request_count'] == 1
    assert filtered['groups'][0]['request_count'] == 2 and filtered['groups'][0]['matched_request_count'] == 1
    assert filtered['groups'][0]['started_at'] == first['started_at']
    complete_turn = ledger.page({}, [], group_by='turn', date_from='2026-10-29', date_to='2026-10-30')
    assert complete_turn['total'] == 2 and complete_turn['group_total'] == 1
    assert complete_turn['groups'][0]['request_count'] == 2 and complete_turn['groups'][0]['status'] == 'failed'
    details = ledger.page({}, [], ledger.run_id, turn_id='cross-day', group_by='turn', date_from='2026-10-31')
    assert [record['request_key'] for record in details['items']] == ['before-midnight', 'after-midnight']
    assert not {'summary', 'groups', 'group_total'}.intersection(details)
    exact = ledger.page({}, [], ledger.run_id, request_key='before-midnight', group_by='turn', date_from='2026-10-31')
    assert exact['items'] == [first] and not {'summary', 'groups', 'group_total'}.intersection(exact)
    legacy = ledger.page({}, [], offset=1, limit=1)
    assert legacy['items'] == [first] and legacy['total'] == 2
    assert not {'summary', 'groups', 'group_total'}.intersection(legacy)


async def test_request_history_group_query_and_legacy_detail_routes(controller, tmp_path, monkeypatch):
    async def handler(request):
        return httpx.Response(200, json={'choices': []})
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        history = controller.request_history
        start = day_time('2026-10-30')
        for index, turn in enumerate(('chain', 'chain', None)):
            key = f'route-{index}'
            history.records[key] = aggregate_record(history, key, start + index, turn_id=turn,
                usage={'prompt_tokens': 100, 'completion_tokens': 10})
        async with client:
            page = await client.get('/api/request-history', params={'group_by': 'turn', 'limit': 1})
            assert page.status_code == 200
            body = page.json()
            assert body['group_total'] == 2 and body['total'] == 3 and body['summary']['input']['value'] == 300
            assert len(body['groups']) == 1 and body['groups'][0]['request_key'] == body['items'][0]['request_key']
            old = (await client.get('/api/request-history', params={'limit': 1})).json()
            assert old['total'] == 3 and not {'groups', 'group_total', 'summary'}.intersection(old)
            detail = (await client.get('/api/request-history', params={'turn_id': 'chain', 'group_by': 'turn'})).json()
            assert [record['request_key'] for record in detail['items']] == ['route-0', 'route-1']
            assert not {'groups', 'group_total', 'summary'}.intersection(detail)
            assert (await client.get('/api/request-history', params={'group_by': 'instance'})).status_code == 422
    finally:
        await controller.close()


def internal_metric_request(key, started_at, *, input_tokens=100, output_tokens=10, **extra):
    return {'id': key, 'requestKey': 'gateway-' + key, 'index': 1, 'kind': 'generation', 'status': 'completed',
            'startedAt': started_at * 1000, 'endedAt': (started_at + 2) * 1000, 'elapsed': 2,
            'firstOutput': 0.5, 'firstOutputBasis': 'client', 'labels': ['生成'],
            'usage': {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens,
                      'prompt_tokens_details': {'cached_tokens': 40}, 'private': 'private-usage-sentinel'},
            'timings': {'prompt_n': input_tokens - 40, 'prompt_ms': 500, 'predicted_n': output_tokens,
                        'predicted_ms': 1000}, 'private': 'private-request-sentinel', **extra}


def persist_internal_metrics(controller, records=None, *, conversation_id=None, message_id='internal-message', **extra):
    from test_conversations import conversation, message
    body = conversation(**({'id': conversation_id} if conversation_id else {}))
    controller.conversations.create(body)
    saved = message(message_id, text='private-message-sentinel', model='Internal model', instanceId='shared',
                    createdAt=day_time('2026-10-30') * 1000, contextCapacity=8192,
                    **({'requests': records} if records is not None else {}))
    saved.update(extra)
    controller.conversations.patch(body['id'], {'messages': [saved]})
    return body['id'], saved


async def test_unified_history_sorts_and_pages_external_and_retained_internal_groups(controller, tmp_path):
    start = day_time('2026-10-30')
    history = controller.request_history
    current = aggregate_record(history, 'external-middle', start + 40, instance_id='shared',
                               usage={'prompt_tokens': 50, 'completion_tokens': 5})
    history.records[current['request_key']] = current
    directory = tmp_path / 'unified-archive'
    history.configure(True, directory)
    previous = RequestHistory(directory, False)
    old = aggregate_record(previous, 'previous-run', start + 10, usage={'prompt_tokens': 25, 'completion_tokens': 2})
    previous.records[old['request_key']] = old
    previous.configure(True, directory)
    assert previous.archive.drain() and history.archive.drain()
    conversation_id, _ = persist_internal_metrics(controller, [
        internal_metric_request('internal-first', start + 30),
        internal_metric_request('internal-last', start + 50, input_tokens=200, output_tokens=20, index=2)])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            pages = [(await client.get('/api/request-history', params={'group_by': 'turn', 'offset': index, 'limit': 1})).json()
                     for index in range(3)]
            first, second, third = pages
            assert first['group_total'] == second['group_total'] == third['group_total'] == 3
            assert first['total'] == first['summary']['request_count'] == 4
            assert first['summary'] == second['summary'] == third['summary']
            assert first['summary']['input'] == {'value': 375, 'reported': 4}
            assert first['items'][0]['request_key'] == 'external-middle' and first['internal_groups'] == []
            assert second['items'] == second['groups'] == []
            group = second['internal_groups'][0]
            assert group['source'] == 'internal' and group['conversation_id'] == conversation_id
            assert group['message_id'] == 'internal-message' and group['instance_id'] == 'shared'
            assert group['request_count'] == 2 and group['started_at'] == start + 30
            assert group['elapsed_s'] == 22 and group['context_capacity'] == 8192
            assert [request['id'] for request in group['requests']] == ['internal-first', 'internal-last']
            assert group['ttft'] == {'value': 0.5, 'reported': 2, 'engine_reported': 0, 'gateway_reported': 0, 'client_reported': 2}
            assert third['items'][0]['run_id'] == previous.run_id
            assert 'private-' not in json.dumps(second)
            old_contract = (await client.get('/api/request-history')).json()
            assert old_contract['total'] == 1 and old_contract['run_id'] == history.run_id
            assert not {'internal_groups', 'summary', 'coverage'}.intersection(old_contract)
    finally:
        await controller.close()


async def test_unified_history_source_instance_dates_and_internal_detail_filter(controller):
    start = day_time('2026-10-30')
    history = controller.request_history
    external = aggregate_record(history, 'external', start, instance_id='shared')
    history.records['external'] = external
    conversation_id, _ = persist_internal_metrics(controller, [
        internal_metric_request('before-midnight', day_time('2026-10-29')),
        internal_metric_request('after-midnight', start, index=2)])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            params = {'group_by': 'turn', 'instance_id': 'shared', 'date_from': '2026-10-30', 'date_to': '2026-10-30'}
            unified = (await client.get('/api/request-history', params=params)).json()
            assert unified['total'] == unified['group_total'] == 2
            assert unified['internal_groups'][0]['request_count'] == 2
            assert unified['internal_groups'][0]['matched_request_count'] == 1
            internal = (await client.get('/api/request-history', params={**params, 'source': 'internal'})).json()
            assert internal['total'] == internal['group_total'] == 1 and internal['groups'] == []
            assert [request['id'] for request in internal['internal_groups'][0]['requests']] == ['before-midnight', 'after-midnight']
            external_only = (await client.get('/api/request-history', params={**params, 'source': 'external'})).json()
            assert external_only['total'] == 1 and external_only['internal_groups'] == []
            exact = (await client.get('/api/request-history', params={'conversation_id': conversation_id,
                'message_id': 'internal-message', 'date_from': '2026-10-31'})).json()
            assert exact['total'] == 2 and len(exact['internal_groups'][0]['requests']) == 2
            assert exact['items'] == []
            dates = (await client.get('/api/request-history/dates')).json()
            days = {day['date']: day for day in dates['dates']}
            assert days['2026-10-29']['request_count'] == 1 and days['2026-10-30']['request_count'] == 2
            sources = {item.get('source', 'external') for item in days['2026-10-30']['instances']}
            assert sources == {'internal', 'external'}
            assert any(item['source'] == 'internal' and item['id'] == 'shared' and 'run_id' not in item
                       for item in days['2026-10-30']['instances'])
            run_only = (await client.get('/api/request-history', params={'group_by': 'turn', 'run_id': history.run_id})).json()
            assert run_only['total'] == 1 and run_only['internal_groups'] == []
            assert (await client.get('/api/request-history', params={'message_id': 'internal-message'})).status_code == 400
            assert (await client.get('/api/request-history', params={'group_by': 'turn', 'source': 'invalid'})).status_code == 422
    finally:
        await controller.close()


async def test_retained_internal_updates_replace_same_request_and_deleted_conversations_leave_stats(controller):
    from modloader.conversations import MESSAGE_PREFIX
    start = day_time('2026-10-30')
    conversation_id, message = persist_internal_metrics(controller, [internal_metric_request('same-request', start)])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            params = {'group_by': 'turn', 'source': 'internal'}
            first = (await client.get('/api/request-history', params=params)).json()
            assert first['total'] == 1 and first['summary']['input']['value'] == 100
            controller.conversations.patch(conversation_id, {'messages': [{**message, 'requests': [
                internal_metric_request('same-request', start, input_tokens=200, output_tokens=20),
                internal_metric_request('follow-up', start + 10, input_tokens=50, output_tokens=5, index=2)]}]})
            rows_before_read = controller.store.connection.execute('SELECT kind,id,body FROM records ORDER BY kind,id').fetchall()
            second = (await client.get('/api/request-history', params=params)).json()
            assert second['total'] == 2 and second['group_total'] == 1
            assert second['summary']['input'] == {'value': 250, 'reported': 2}
            assert controller.store.connection.execute('SELECT kind,id,body FROM records ORDER BY kind,id').fetchall() == rows_before_read
            assert len(controller.store.all(MESSAGE_PREFIX + conversation_id)) == 1
            controller.conversations.delete(conversation_id)
            deleted = (await client.get('/api/request-history', params=params)).json()
            assert deleted['total'] == deleted['group_total'] == 0 and deleted['summary']['input']['value'] is None
            assert deleted['internal_groups'] == []
    finally:
        await controller.close()


async def test_internal_history_legacy_coverage_unknown_instance_and_private_management_only(controller):
    start = day_time('2026-10-30')
    persist_internal_metrics(controller, None, message_id='legacy', usage={'prompt_tokens': 800, 'completion_tokens': 8})
    conversation_id, _ = persist_internal_metrics(controller, [internal_metric_request('unknown-instance', start)], instanceId=None)
    try:
        app = build_app(controller)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver',
                                    headers={'Authorization': 'Bearer management-secret'}) as client:
            page = (await client.get('/api/request-history', params={'group_by': 'turn'})).json()
            assert page['total'] == 1 and page['summary']['input']['value'] == 100
            assert page['coverage'] == {'legacy_message_count': 1, 'legacy_metrics_message_count': 1}
            assert page['internal_groups'][0]['instance_id'] is None
            dates = (await client.get('/api/request-history/dates')).json()
            unknown = next(item for day in dates['dates'] for item in day['instances'] if item.get('source') == 'internal' and item['id'] is None)
            assert unknown['conversation_id'] == conversation_id and 'run_id' not in unknown
            chosen = (await client.get('/api/request-history', params={'group_by': 'turn', 'source': 'internal',
                'conversation_id': unknown['conversation_id']})).json()
            assert chosen['total'] == 1 and chosen['coverage']['legacy_message_count'] == 0
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as unauthenticated:
            assert (await unauthenticated.get('/api/request-history', params={'group_by': 'turn'})).status_code == 401
        public_paths = {route.path for route in build_app(controller, public=True).routes}
        assert '/api/request-history' not in public_paths and '/api/request-history/dates' not in public_paths
    finally:
        await controller.close()


async def test_retained_internal_optional_ttft_basis_and_untyped_live_fields_stay_unknown(controller):
    request = internal_metric_request('old-optional-fields', day_time('2026-10-30'))
    request.pop('firstOutputBasis')
    request['prefill'] = {'prefill_average_basis': ['invalid'], 'prefill_processed_tokens': 60}
    request['live'] = {'source': 'request counters', 'average_basis': {'invalid': True}, 'generated_tokens': 10,
                       'average_tokens_per_second': 100, 'prefill_average_basis': ['invalid']}
    persist_internal_metrics(controller, [request])
    try:
        result = controller.conversations.request_history()
        group = result['groups'][0]
        assert group['ttft'] == {'value': 0.5, 'reported': 1, 'engine_reported': 0,
                                 'gateway_reported': 0, 'unknown_reported': 1}
        assert 'average_basis' not in group['requests'][0]['live']
        assert 'prefill_average_basis' not in group['requests'][0]['prefill']
    finally:
        await controller.close()


def test_grouped_engine_average_weights_spark_usage_time_and_never_becomes_decode(ledger):
    start = day_time('2026-10-30')
    records = [
        aggregate_record(ledger, 'spark-first', start,
            usage={'completion_tokens': 20, 'generation_ms': 4000, 'decode_tps': 999}),
        aggregate_record(ledger, 'spark-second', start + 10,
            usage={'completion_tokens': 30, 'generation_ms': 1000, 'decode_tps': 30}),
        aggregate_record(ledger, 'spark-failed', start + 20, status='failed',
            usage={'completion_tokens': 99, 'generation_ms': 1000, 'decode_tps': 99}),
    ]
    ledger.records.update({record['request_key']: record for record in records})
    page = ledger.page({}, [], group_by='turn')
    assert page['summary']['engine'] == {'value': 10, 'tokens': 50, 'seconds': 5, 'reported': 2}
    assert page['summary']['decode']['value'] is None and page['summary']['decode']['reported'] == 0
    failed = next(group for group in page['groups'] if group['request_key'] == 'spark-failed')
    assert failed['engine'] == {'value': None, 'tokens': 0, 'seconds': 0, 'reported': 0}


async def test_retained_internal_engine_average_uses_native_time_or_paired_reported_mean(controller):
    start = day_time('2026-10-30')
    persist_internal_metrics(controller, [
        internal_metric_request('time-only', start, usage={'completion_tokens': 10, 'generation_ms': 2000}, timings={}),
        internal_metric_request('mean-only', start + 10, usage={'completion_tokens': 20, 'decode_tps': 10}, timings={}),
        internal_metric_request('unpaired-mean', start + 20, usage={'decode_tps': 50}, timings={}),
    ])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            page = (await client.get('/api/request-history', params={'group_by': 'turn', 'source': 'internal'})).json()
            expected = {'value': 7.5, 'tokens': 30, 'seconds': 4, 'reported': 2}
            assert page['summary']['engine'] == page['internal_groups'][0]['engine'] == expected
            assert page['summary']['decode']['value'] is None and page['summary']['decode']['reported'] == 0
    finally:
        await controller.close()


async def test_cross_midnight_archive_and_internal_parents_are_complete_on_first_group_page(controller, tmp_path):
    before, after = day_time('2026-10-29'), day_time('2026-10-30')
    directory = tmp_path / 'cross-midnight-retained'
    archived = RequestHistory(directory, False)
    archived.records.update({record['request_key']: record for record in (
        aggregate_record(archived, 'archived-before', before, turn_id='cross-day',
                         usage={'prompt_tokens': 101, 'completion_tokens': 10}),
        aggregate_record(archived, 'archived-after', after, turn_id='cross-day',
                         usage={'prompt_tokens': 202, 'completion_tokens': 20}))})
    archived.configure(True, directory)
    assert archived.archive.drain()
    controller.request_history.configure(True, directory)
    assert controller.request_history.archive.drain()
    persist_internal_metrics(controller, [
        internal_metric_request('internal-before', before, input_tokens=101, output_tokens=10),
        internal_metric_request('internal-after', after, input_tokens=202, output_tokens=20, index=2)])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            params = {'group_by': 'turn', 'date_from': '2026-10-30', 'date_to': '2026-10-30', 'limit': 1}
            pages = [(await client.get('/api/request-history', params={**params, 'offset': offset})).json() for offset in (0, 1)]
            assert pages[0]['summary'] == pages[1]['summary']
            assert all(page['total'] == 2 and page['group_total'] == 2 for page in pages)
            assert pages[0]['summary']['input'] == {'value': 404, 'reported': 2}
            assert pages[0]['summary']['output'] == {'value': 40, 'reported': 2}
            parents = [group for page in pages for group in page['groups'] + page['internal_groups']]
            assert len(parents) == 2
            for group in parents:
                assert group['request_count'] == 2 and group['matched_request_count'] == 1
                assert group['input'] == {'value': 303, 'reported': 2}
                assert group['output'] == {'value': 30, 'reported': 2}
                assert group['started_at'] == before and group['ended_at'] == after + 2
            external = next(page for page in pages if page['groups'])
            assert external['items'][0]['request_key'] == 'archived-after'
            internal = next(group for group in parents if group['source'] == 'internal')
            assert len(internal['requests']) == 2
            # A parent remains complete even when its latest request is outside the date filter.
            yesterday = (await client.get('/api/request-history', params={**params,
                'date_from': '2026-10-29', 'date_to': '2026-10-29', 'limit': 10})).json()
            assert yesterday['summary']['input'] == {'value': 202, 'reported': 2}
            assert yesterday['groups'][0]['input'] == yesterday['internal_groups'][0]['input'] == {'value': 303, 'reported': 2}
            assert yesterday['items'][0]['request_key'] == 'archived-after'
            legacy = (await client.get('/api/request-history', params={'run_id': archived.run_id,
                'date_from': '2026-10-30', 'date_to': '2026-10-30'})).json()
            assert legacy['total'] == 1 and legacy['items'][0]['request_key'] == 'archived-after'
            assert 'summary' not in legacy
    finally:
        await controller.close()


@pytest.mark.parametrize('minutes,expected', [(4, 1), (5, 5), (30, 5), (31, 15), (120, 15),
                                              (121, 30), (360, 30), (361, 60), (1440, 60), (2880, 60)])
def test_usage_trend_adapts_equal_width_buckets_to_retained_span(ledger, minutes, expected):
    start = day_time('2026-10-30')
    ledger.records['first'] = aggregate_record(ledger, 'first', start,
        usage={'prompt_tokens': 100, 'completion_tokens': 10})
    ledger.records['last'] = aggregate_record(ledger, 'last', start + minutes * 60,
        usage={'prompt_tokens': 200, 'completion_tokens': 20})
    result = ledger.usage({}, [], as_of=start + 4 * 86400 + 17)
    trend = result['trend']
    assert trend['bucket_seconds'] == expected * 60
    assert trend['end_at'] == (int((start + minutes * 60) // (expected * 60)) + 1) * expected * 60
    assert trend['end_at'] - trend['start_at'] <= 86400
    assert trend['next_refresh_at'] > result['as_of']
    assert result['summary']['request_count'] == 2
    for index, bucket in enumerate(trend['buckets']):
        assert bucket['start_at'] == trend['start_at'] + index * expected * 60
        assert bucket['end_at'] == bucket['start_at'] + expected * 60
    assert trend['buckets'][-1]['end_at'] == trend['end_at']
    if minutes >= 1440:
        assert sum(bucket['request_count'] for bucket in trend['buckets']) == 1
        assert result['summary']['input'] == {'value': 300, 'reported': 2}


def test_usage_trend_boundaries_cache_weighting_empty_and_missing_counters(ledger):
    start = day_time('2026-10-30')
    examples = [
        ('before', 299, {'prompt_tokens': 100, 'completion_tokens': 10, 'prompt_tokens_details': {'cached_tokens': 50}}),
        ('boundary', 300, {'prompt_tokens': 900, 'completion_tokens': 20, 'prompt_tokens_details': {'cached_tokens': 900}}),
        ('same-bucket', 301, {'prompt_tokens': 100, 'completion_tokens': 0, 'prompt_tokens_details': {'cached_tokens': 0}}),
        ('unknown', 1200, {}),
        ('invalid-pair', 1201, {'prompt_tokens': 10, 'prompt_tokens_details': {'cached_tokens': 11}}),
    ]
    for key, offset, usage in examples:
        ledger.records[key] = aggregate_record(ledger, key, start + offset, usage=usage)
    result = ledger.usage({}, [], as_of=start + 7200)
    trend = result['trend']
    assert trend['bucket_seconds'] == 300
    buckets = trend['buckets']
    assert [bucket['request_count'] for bucket in buckets] == [1, 2, 0, 0, 2]
    assert buckets[0]['total'] == {'value': 110, 'reported': 1}
    assert buckets[1]['cache_ratio'] == pytest.approx(0.9)
    assert buckets[1]['total'] == {'value': 1020, 'reported': 2}  # Input already contains cache.
    assert all(buckets[2][key] == {'value': 0, 'reported': 0} for key in ('input', 'output', 'total', 'cached'))
    assert buckets[2]['cache_ratio'] is None
    assert buckets[4]['input'] == {'value': 10, 'reported': 1}
    assert buckets[4]['output'] == buckets[4]['total'] == {'value': None, 'reported': 0}
    assert buckets[4]['cached'] == {'value': 11, 'reported': 1}
    assert buckets[4]['cache_pair']['reported'] == 0 and buckets[4]['cache_ratio'] is None
    assert result['summary'] == ledger.page({}, [], group_by='turn')['summary']


def test_usage_trend_only_running_requests_use_now_and_finished_tools_use_latest_request(ledger):
    start = day_time('2026-10-30')
    request, _ = complete(ledger, [], ids=['pending-tool'], started_at=start)
    waiting = ledger.usage({}, [], as_of=start + 1201)['trend']
    assert waiting['end_at'] == start + 60 and waiting['bucket_seconds'] == 60
    ledger.records[request.token]['labels'] = []
    ledger.boundaries[request.token]['waiting'] = False
    idle = ledger.usage({}, [], as_of=start + 1201)['trend']
    assert idle['end_at'] == start + 60 and idle['bucket_seconds'] == 60
    active = RequestMetrics(True, backend='llama_cpp')
    active.started_at = start + 10
    ledger.start(active, instance(), [])
    running = ledger.usage({}, [], as_of=start + 1201)['trend']
    assert running['end_at'] == start + 1500


def test_usage_empty_range_retains_summary_semantics_and_has_future_refresh(ledger):
    result = ledger.usage({}, [], as_of=1234.5)
    assert result['summary']['input'] == {'value': None, 'reported': 0}
    assert result['trend'] == {'start_at': None, 'end_at': None, 'bucket_seconds': 60,
                               'next_refresh_at': 1260, 'buckets': []}


def test_history_input_uses_known_full_prefill_total_without_adding_cache_twice(ledger):
    ledger.records['progress-only'] = aggregate_record(ledger, 'progress-only', day_time('2026-10-30'),
        prefill={'prefill_total_tokens': 100, 'prefill_cached_tokens': 80},
        usage={'completion_tokens': 10})
    summary = ledger.page({}, [], group_by='turn')['summary']
    assert summary['input'] == {'value': 100, 'reported': 1}
    assert summary['total'] == {'value': 110, 'reported': 1}
    assert summary['cache_ratio'] == 0.8


def test_usage_revision_ignores_stream_tokens_and_updates_at_each_external_request_terminal(ledger):
    revision = lambda: ledger.usage({}, [])['revision']
    initial = revision()
    metrics = RequestMetrics(True, backend='llama_cpp')
    ledger.start(metrics, instance(), [])
    assert revision() == initial
    metrics.observe({'usage': {'prompt_tokens': 100, 'completion_tokens': 1}})
    ledger.capture_live(metrics.token, {'request_key': metrics.token, 'scope': 'request', 'generated_tokens': 1})
    assert revision() == initial
    metrics.finish('completed')
    ledger.finish(metrics)
    standalone = revision()
    assert standalone != initial and revision() == standalone

    messages = [{'role': 'user', 'content': 'tools'}]
    parent, response = complete(ledger, messages, ids=['first-tool'])
    parent_revision = revision()
    assert parent_revision != standalone
    child, response2 = complete(ledger, follow(messages, response), ids=['second-tool'])
    child_revision = revision()
    assert child_revision != parent_revision
    final, _ = complete(ledger, follow(follow(messages, response), response2), 'answer')
    completed = revision()
    assert completed != child_revision and revision() == completed
    ledger.records[final.token]['duration_s'] += 20
    assert revision() == completed
    ledger.records[final.token]['usage']['completion_tokens'] += 1
    assert revision() == completed  # New records read counters, not stale provider diagnostics.
    ledger.records[final.token]['counters']['output'] += 1
    corrected = revision()
    assert corrected != completed
    ledger.records[parent.token]['counters']['input'] += 1
    earlier_child_corrected = revision()
    assert earlier_child_corrected != corrected
    del ledger.records[metrics.token]
    assert revision() != earlier_child_corrected
    # Legacy saved records still use their original usage/timings contract.
    legacy = aggregate_record(ledger, 'legacy-correction', day_time('2026-10-30'), usage={'completion_tokens': 10})
    ledger.records[legacy['request_key']] = legacy
    legacy_before = revision()
    legacy['usage']['completion_tokens'] += 1
    assert revision() != legacy_before


async def test_usage_endpoint_unifies_retained_sources_filters_and_has_no_page_limit(controller, tmp_path):
    start = day_time('2026-10-30')
    history = controller.request_history
    for index in range(55):
        key = f'external-{index}'
        history.records[key] = aggregate_record(history, key, start + index, instance_id='shared',
                                                usage={'prompt_tokens': 10, 'completion_tokens': 1})
    archived = RequestHistory(tmp_path / 'usage-archive', False)
    old = aggregate_record(archived, 'previous-run', start + 100, instance_id='shared',
                           usage={'prompt_tokens': 25, 'completion_tokens': 2})
    archived.records[old['request_key']] = old
    archived.configure(True, tmp_path / 'usage-archive')
    assert archived.archive.drain()
    history.configure(False, tmp_path / 'usage-archive')
    conversation_id, _ = persist_internal_metrics(controller, [
        internal_metric_request('before-midnight', day_time('2026-10-29')),
        internal_metric_request('internal-today', start + 30, index=2)])
    persist_internal_metrics(controller, None, message_id='legacy', usage={'prompt_tokens': 800})
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            params = {'date_from': '2026-10-30', 'date_to': '2026-10-30', 'instance_id': 'shared'}
            usage_response = await client.get('/api/request-history/usage', params=params)
            assert usage_response.status_code == 200, usage_response.text
            usage = usage_response.json()
            page = (await client.get('/api/request-history', params={**params, 'group_by': 'turn', 'limit': 1})).json()
            assert usage['summary'] == page['summary'] and usage['summary']['request_count'] == 57
            assert usage['summary']['input'] == {'value': 675, 'reported': 57}
            assert sum(bucket['request_count'] for bucket in usage['trend']['buckets']) == 57
            assert usage['coverage'] == page['coverage']
            assert not {'items', 'internal_groups', 'offset', 'limit'}.intersection(usage)
            dates = (await client.get('/api/request-history/dates')).json()
            assert dates['usage_revision'] == usage['revision']
            internal = (await client.get('/api/request-history/usage', params={**params, 'source': 'internal'})).json()
            assert internal['summary']['request_count'] == 1
            external = (await client.get('/api/request-history/usage', params={**params, 'source': 'external'})).json()
            assert external['summary']['request_count'] == 56
            previous = (await client.get('/api/request-history/usage', params={'run_id': archived.run_id})).json()
            assert previous['summary']['request_count'] == 1
            conversation = (await client.get('/api/request-history/usage', params={'conversation_id': conversation_id})).json()
            assert conversation['summary']['request_count'] == 2
            assert 'private-' not in json.dumps(usage)
            for invalid in ({'date_from': 'invalid'}, {'date_from': '2026-10-31', 'date_to': '2026-10-30'},
                            {'source': 'internal', 'run_id': history.run_id},
                            {'source': 'external', 'conversation_id': conversation_id}):
                assert (await client.get('/api/request-history/usage', params=invalid)).status_code == 400
            assert (await client.get('/api/request-history/usage', params={'source': 'other'})).status_code == 422
            assert (await client.get('/api/request-history/usage', params={'run_id': 'missing'})).status_code == 404
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver') as client:
            assert (await client.get('/api/request-history/usage')).status_code == 401
        assert '/api/request-history/usage' not in {route.path for route in build_app(controller, public=True).routes}
    finally:
        archived.archive.drain()
        await controller.close()


async def test_usage_revision_waits_for_persisted_internal_whole_turn_and_tracks_deletion(controller):
    start = day_time('2026-10-30')
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            async def revision():
                return (await client.get('/api/request-history/dates')).json()['usage_revision']
            initial = await revision()
            request = internal_metric_request('tool-child', start)
            conversation_id, saved = persist_internal_metrics(controller, [request], status='generating')
            assert await revision() == initial  # Completed child inside a generating assistant is not a completed turn.
            saved['requests'][0]['usage']['completion_tokens'] += 3
            saved['requests'].append(internal_metric_request('final-child', start + 10, index=2))
            controller.conversations.patch(conversation_id, {'messages': [saved]})
            assert await revision() == initial
            saved['status'] = 'completed'
            controller.conversations.patch(conversation_id, {'messages': [saved]})
            completed = await revision()
            assert completed != initial and await revision() == completed
            saved['text'] += ' another token is not a metric'
            saved['requests'][0]['elapsed'] += 30
            controller.conversations.patch(conversation_id, {'messages': [saved]})
            assert await revision() == completed
            saved['requests'][0]['usage']['completion_tokens'] += 1
            controller.conversations.patch(conversation_id, {'messages': [saved]})
            corrected = await revision()
            assert corrected != completed
            controller.conversations.delete(conversation_id)
            deleted = await revision()
            assert deleted != corrected
            legacy_id, _ = persist_internal_metrics(controller, None, message_id='legacy-revision', usage={'prompt_tokens': 1})
            with_legacy = await revision()
            assert with_legacy != deleted
            controller.conversations.delete(legacy_id)
            assert await revision() != with_legacy
    finally:
        await controller.close()
