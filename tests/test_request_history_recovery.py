"""Native session isolation, exact failed retries and truthful error metadata."""
from copy import deepcopy

import pytest
import httpx

from modloader.request_history import RequestHistory
from modloader.request_metrics import RequestMetrics
from test_request_history import ledger, instance, assistant, follow
from test_core import controller, gateway_client, GatewayStream, sse


CONTEXT = {'source': 'opencode', 'session_id': 'session-one'}
PROMPT = [{'role': 'user', 'content': 'private-prompt-sentinel'}]


def begin(history, messages=PROMPT, context=CONTEXT, target=None):
    metrics = RequestMetrics(False, backend='ninfer')
    history.start(metrics, target or instance(), messages, client_context=context)
    return metrics


def end(history, metrics, status='completed', calls=(), upstream=None):
    if upstream:
        metrics.observe_upstream(**upstream)
    if status == 'completed':
        chunk = {'choices': [{'index': 0, 'message': assistant(calls, 'private-answer-sentinel'),
                              'finish_reason': 'tool_calls' if calls else 'stop'}],
                 'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
        metrics.observe(chunk)
        history.observe(metrics.token, chunk)
    metrics.finish(status)
    history.finish(metrics)
    return history.records[metrics.token]


def test_native_session_exact_initial_retry_recovers_without_rewriting_failed_attempt(ledger):
    failed = end(ledger, begin(ledger), 'failed')
    final = end(ledger, begin(ledger))
    assert failed['status'] == 'failed'
    assert {record['turn_state'] for record in (failed, final)} == {'recovered'}
    assert failed['turn_id'] == final['turn_id'] is not None
    page = ledger.page({}, [], group_by='turn')
    assert page['group_total'] == 1
    group = page['groups'][0]
    assert group['status'] == 'recovered' and group['request_count'] == 2
    assert group['status_counts']['failed'] == 1 and group['status_counts']['completed'] == 1
    assert group['client_context'] == CONTEXT


@pytest.mark.parametrize('field', ['source', 'session_id', 'instance_id'])
def test_native_session_scope_isolates_content_matching(ledger, field):
    first = end(ledger, begin(ledger), calls=['same-call'])
    context, target = dict(CONTEXT), instance()
    if field == 'instance_id':
        target = instance('two')
    else:
        context[field] += '-different'
    continuation = follow(PROMPT, assistant(['same-call'], 'private-answer-sentinel'))
    second = end(ledger, begin(ledger, continuation, context, target))
    assert first['turn_id'] is None and second['turn_id'] is None
    assert ledger.page({}, [], group_by='turn')['group_total'] == 2


@pytest.mark.parametrize('parent_identified', [False, True])
def test_identified_and_legacy_requests_cannot_fallback_into_each_other(ledger, parent_identified):
    parent_context = CONTEXT if parent_identified else None
    child_context = None if parent_identified else CONTEXT
    parent = end(ledger, begin(ledger, context=parent_context), calls=['same-call'])
    continuation = follow(PROMPT, assistant(['same-call'], 'private-answer-sentinel'))
    child = end(ledger, begin(ledger, continuation, child_context))
    assert parent['turn_id'] is None and child['turn_id'] is None
    assert ledger.page({}, [], group_by='turn')['group_total'] == 2


def test_identified_tool_ids_do_not_poison_unidentified_content_matching(ledger):
    end(ledger, begin(ledger, context=CONTEXT), calls=['same-call'])
    parent = end(ledger, begin(ledger, context=None), calls=['same-call'])
    continuation = follow(PROMPT, assistant(['same-call'], 'private-answer-sentinel'))
    child = end(ledger, begin(ledger, continuation, None))
    assert child['turn_id'] == parent['turn_id'] is not None
    assert ledger.page({}, [], group_by='turn')['group_total'] == 2


def test_active_attempt_wins_and_late_older_failure_cannot_cover_newer_success(ledger):
    old = begin(ledger)
    newer = begin(ledger, [{'role': 'user', 'content': 'retry with changed context'}])
    # Exercise the valid overlapping-turn model directly. Session headers alone
    # are deliberately insufficient to infer this relationship from content.
    for metrics in (old, newer):
        ledger.records[metrics.token].update(aggregation='turn', turn_id='overlapping-turn', turn_state='running')
    success = end(ledger, newer)
    assert success['turn_state'] == ledger.records[old.token]['turn_state'] == 'running'
    failed = end(ledger, old, 'failed')
    assert success['turn_state'] == failed['turn_state'] == 'recovered'
    assert failed['status'] == 'failed' and success['status'] == 'completed'


@pytest.mark.parametrize('other_context', [{**CONTEXT, 'session_id': 'different'}, None])
def test_concurrent_sessions_and_legacy_clients_keep_independent_outcomes(ledger, other_context):
    first = begin(ledger)
    second = begin(ledger, context=other_context)
    failed = end(ledger, first, 'failed')
    assert failed['status'] == 'failed'
    assert ledger.records[second.token]['status'] == 'running'
    success = end(ledger, second)
    assert success['status'] == 'completed' and failed['status'] == 'failed'
    groups = ledger.page({}, [], group_by='turn')['groups']
    assert len(groups) == 2 and {group['status'] for group in groups} == {'completed', 'failed'}


def test_newer_failed_attempt_remains_terminal_failure(ledger):
    completed = end(ledger, begin(ledger), calls=['terminal-tool'])
    continuation = follow(PROMPT, assistant(['terminal-tool'], 'private-answer-sentinel'))
    failed = end(ledger, begin(ledger, continuation), 'failed')
    assert completed['turn_state'] == failed['turn_state'] == 'failed'
    assert ledger.page({}, [], group_by='turn')['groups'][0]['status'] == 'failed'


def test_tool_waiting_is_not_reported_as_recovered_before_final_answer(ledger):
    end(ledger, begin(ledger), 'failed')
    tool = end(ledger, begin(ledger), calls=['tool'])
    assert tool['turn_state'] == 'waiting'
    continuation = follow(PROMPT, assistant(['tool'], 'private-answer-sentinel'))
    final = end(ledger, begin(ledger, continuation))
    assert final['turn_state'] == 'recovered'


@pytest.mark.parametrize('change', ['new_user', 'rewrite', 'compact'])
def test_native_session_does_not_join_changed_input_to_failed_attempt(ledger, change):
    failed = end(ledger, begin(ledger), 'failed')
    messages = deepcopy(PROMPT)
    if change == 'new_user':
        messages.append({'role': 'user', 'content': 'new user message'})
    elif change == 'rewrite':
        messages[0]['content'] += ' changed'
    else:
        messages = [{'role': 'system', 'content': 'compacted conversation'}]
    success = end(ledger, begin(ledger, messages))
    assert failed['turn_id'] is None and success['turn_id'] is None
    assert ledger.page({}, [], group_by='turn')['group_total'] == 2


def test_native_session_never_reuses_failure_from_an_already_recovered_group(ledger):
    failed = end(ledger, begin(ledger), 'failed')
    recovered = end(ledger, begin(ledger))
    later = end(ledger, begin(ledger))
    assert failed['turn_id'] == recovered['turn_id'] is not None
    assert failed['turn_state'] == 'recovered'
    assert later['turn_id'] is None
    assert ledger.page({}, [], group_by='turn')['group_total'] == 2


def test_native_session_only_retries_the_most_recent_failed_input(ledger):
    failed = end(ledger, begin(ledger), 'failed')
    end(ledger, begin(ledger, [{'role': 'user', 'content': 'intervening request'}]))
    later = end(ledger, begin(ledger))
    assert failed['turn_id'] is None and later['turn_id'] is None
    assert ledger.page({}, [], group_by='turn')['group_total'] == 3


def test_concurrent_same_session_inputs_are_not_misclassified_as_serial_retries(ledger):
    first, second = begin(ledger), begin(ledger)
    end(ledger, first, 'failed')
    end(ledger, second, 'failed')
    success = end(ledger, begin(ledger))
    assert success['turn_id'] is None
    assert ledger.page({}, [], group_by='turn')['group_total'] == 3


def test_other_session_concurrency_does_not_prevent_an_exact_serial_retry(ledger):
    other = begin(ledger, context={**CONTEXT, 'session_id': 'other'})
    failed = end(ledger, begin(ledger), 'failed')
    success = end(ledger, begin(ledger))
    assert success['turn_id'] == failed['turn_id'] is not None
    assert success['turn_state'] == 'recovered'
    end(ledger, other)


@pytest.mark.parametrize('native_session', [False, True])
async def test_gateway_native_session_header_keeps_failed_retry_as_distinct_requests(controller, tmp_path, monkeypatch, native_session):
    requests = []
    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(400, json={'error': {'code': 'invalid_request_error'}})
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'answer'}, 'finish_reason': 'stop'}]})
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    headers = {'X-Opencode-Session-Id': CONTEXT['session_id']} if native_session else {}
    try:
        async with client:
            for _ in range(2):
                await client.post('/v1/chat/completions', json={'model': 'one', 'messages': PROMPT}, headers=headers)
            page = (await client.get('/api/request-history', params={'group_by': 'turn'})).json()
        assert page['group_total'] == 2
        assert [group['status'] for group in page['groups']] == ['completed', 'failed']
        if native_session:
            assert page['groups'][0]['client_context'] == CONTEXT
        assert all('x-opencode-session-id' not in request.headers for request in requests)
        import json
        assert all('client_context' not in json.loads(request.content) for request in requests)
    finally:
        await controller.close()


async def test_native_retry_recovery_preserves_generation_identity_and_terminal_clear(controller, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from starlette.requests import Request
    import modloader.generation_events as event_module
    from test_generation_events import counter, advance_data
    clock = [10.]
    monkeypatch.setattr(event_module, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]))
    stream = GatewayStream([sse(counter(10)), sse(counter(14)), sse('[DONE]')])
    async def handler(request):
        assert request.method == 'POST', 'generation events must not perform telemetry polls'
        return httpx.Response(200, stream=stream)
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    target = instances['one']
    target.backend = 'ninfer'
    failed = end(controller.request_history, begin(controller.request_history, target=target), 'failed')
    queue = controller.generation_events.subscribe(target.id)
    response = None
    try:
        endpoint = next(route.endpoint for route in client._transport.app.routes if route.path == '/v1/chat/completions')
        async def receive():
            return {'type': 'http.request', 'body': json.dumps({'model': 'one', 'messages': PROMPT, 'stream': True}).encode(), 'more_body': False}
        response = await endpoint(Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions',
            'headers': [(b'x-opencode-session-id', CONTEXT['session_id'].encode())]}, receive))
        waiting = queue.get_nowait()['live_generation']
        assert waiting['phase'] == 'waiting' and waiting['request_key'] != failed['request_key']
        assert controller.request_history.records[waiting['request_key']]['client_context'] == CONTEXT
        await advance_data(response)
        assert queue.get_nowait()['live_generation']['generated_tokens'] == 10
        clock[0] += .02
        await advance_data(response)
        assert queue.get_nowait()['live_generation']['tokens_per_second'] == pytest.approx(200)
        await advance_data(response)
        assert queue.get_nowait()['live_generation'] is None
        async for _ in response.body_iterator:
            pass
        assert failed['status'] == 'failed' and failed['turn_state'] is None
        assert controller.request_history.latest(target.id)['state'] == 'completed'
        assert target.active_requests == 0 and not controller.active_request_tokens[target.id]
        assert stream.closed
    finally:
        if response is not None:
            await response.body_iterator.aclose()
        controller.generation_events.unsubscribe(target.id, queue)
        await client.aclose()
        await controller.close()


def test_usage_revision_updates_on_each_external_terminal_not_start_or_stream_tokens(ledger):
    initial = ledger.usage_revision({})
    failed = begin(ledger)
    assert ledger.usage_revision({}) == initial
    end(ledger, failed, 'failed')
    terminal_failure = ledger.usage_revision({})
    assert terminal_failure != initial
    tool = begin(ledger)
    for count in (1, 2, 3):
        chunk = {'usage': {'completion_tokens': count}}
        tool.observe(chunk)
        ledger.observe(tool.token, chunk)
        assert ledger.usage_revision({}) == terminal_failure
    end(ledger, tool, calls=['tool'])
    tool_terminal = ledger.usage_revision({})
    assert tool_terminal != terminal_failure
    continuation = follow(PROMPT, assistant(['tool'], 'private-answer-sentinel'))
    final = begin(ledger, continuation)
    assert ledger.usage_revision({}) == tool_terminal
    end(ledger, final)
    recovered = ledger.usage_revision({})
    assert recovered != tool_terminal
    assert ledger.usage_revision({}) == recovered
    assert ledger.latest('one')['state'] == 'recovered'


def test_context_and_upstream_recovery_survive_reload_without_legacy_backfill(ledger, tmp_path):
    failed = end(ledger, begin(ledger), 'failed', upstream={
        'status': 400, 'error': {'error': {'code': 'context_length_exceeded',
            'message': 'private-prompt-sentinel Bearer private-key-sentinel',
            'request': deepcopy(PROMPT)}}})
    success = end(ledger, begin(ledger), upstream={'status': 200})
    legacy = end(ledger, begin(ledger, context=None))
    directory = tmp_path / 'recovery-archive'
    ledger.configure(True, directory)
    assert ledger.archive.drain()
    reader = RequestHistory(directory, enabled=False)
    saved, issues = reader.archive.read()
    assert not issues
    restored = saved[ledger.run_id]['records']
    assert restored[failed['request_key']]['status'] == 'failed'
    assert restored[failed['request_key']]['turn_state'] == 'recovered'
    assert restored[failed['request_key']]['client_context'] == CONTEXT
    assert restored[failed['request_key']]['upstream'] == {
        'status': 400, 'code': 'context_length_exceeded', 'reason': '请求超出后端上下文限制'}
    assert restored[success['request_key']]['upstream'] == {'status': 200}
    assert 'client_context' not in restored[legacy['request_key']]
    assert 'upstream' not in restored[legacy['request_key']]
    page = reader.page(saved, issues, run_id=ledger.run_id, group_by='turn')
    assert page['group_total'] == 2
    assert next(group for group in page['groups'] if group['turn_id'])['status'] == 'recovered'
    # A new gateway run does not adopt saved retry identities from its archive.
    current = end(reader, begin(reader))
    assert current['turn_id'] != failed['turn_id']
    text = '\n'.join(path.read_text(encoding='utf-8') for path in directory.glob('*.jsonl'))
    assert 'private-prompt-sentinel' not in text and 'private-key-sentinel' not in text


@pytest.mark.parametrize('error', [
    {'error': {'code': 'private-key-sentinel', 'message': 'private-prompt-sentinel'}},
    {'error': 'private-prompt-sentinel'},
    {'error': {'code': {'private': 'private-key-sentinel'}, 'detail': PROMPT}},
])
def test_untrusted_upstream_error_text_is_never_retained(error):
    metrics = RequestMetrics(False, backend='ninfer')
    metrics.observe_upstream(status=400, error=error)
    assert metrics.finish('failed')['upstream'] == {'status': 400, 'reason': '后端返回 HTTP 400'}


@pytest.mark.parametrize('code,expected', [
    (400, {'status': 200, 'code': 400, 'reason': '后端报告错误（400）'}),
    (-1, {'status': 200, 'code': 'stream_error', 'reason': '后端流报告错误'}),
])
def test_numeric_stream_error_codes_keep_a_safe_reason(code, expected):
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe_upstream(200, {'error': {'code': code}}, cause='stream_error')
    assert metrics.finish('failed')['upstream'] == expected


@pytest.mark.parametrize('failure,expected', [
    ('http', {'status': 400, 'code': 'context_length_exceeded', 'reason': '请求超出后端上下文限制'}),
    ('connect', {'code': 'ConnectError', 'reason': '无法连接模型后端'}),
    ('read', {'status': 200, 'code': 'ReadError', 'reason': '后端流读取中断'}),
    ('stream', {'status': 200, 'code': 'stream_error', 'reason': '后端流报告错误'}),
])
async def test_gateway_retains_safe_upstream_failure_without_error_body(controller, tmp_path, monkeypatch, failure, expected):
    async def handler(request):
        if failure == 'connect':
            raise httpx.ConnectError('Bearer private-key-sentinel', request=request)
        if failure == 'http':
            return httpx.Response(400, json={'error': {'code': 'context_length_exceeded',
                'message': 'private-prompt-sentinel Bearer private-key-sentinel'}})
        chunks = [sse({'choices': [{'delta': {'content': 'private-answer-sentinel'}}]})]
        chunks += [httpx.ReadError('private-key-sentinel')] if failure == 'read' else [
            sse({'error': {'message': 'private-prompt-sentinel private-key-sentinel'}}), sse('[DONE]')]
        return httpx.Response(200, stream=GatewayStream(chunks))
    client, _, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            await client.post('/v1/chat/completions', json={'model': 'one', 'messages': PROMPT,
                                                        'stream': failure in ('read', 'stream')})
        record = controller.request_history.page({}, [])['items'][0]
        assert record['status'] == 'failed'
        assert record['upstream'] == expected
        assert 'private-' not in repr(record)
    finally:
        await controller.close()
