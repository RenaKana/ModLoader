"""Request-scoped live counters pushed independently of the telemetry poller."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from modloader.app import build_app
from modloader.generation_events import GenerationEvents
from modloader.request_metrics import RequestMetrics
from test_core import controller, gateway_client, GatewayStream, sse


def counter(count, phase='content'):
    return {'choices': [{'delta': {phase: 'private text'}}],
            'timings': {'predicted_n': count, 'predicted_per_second': 999},
            'usage': {'prompt_tokens': 100}}


def observe(events, metrics, chunk, at, identifier='one'):
    metrics.observe(chunk)
    events.publish(identifier, metrics, chunk, sampled_at=at, observed_at=1000 + at)


def test_event_rates_follow_real_counter_frames_not_polls_or_backend_mean():
    events = GenerationEvents('run-one')
    metrics = RequestMetrics(True, backend='ninfer', client_request_id='message', client_call_id='call')
    queue = events.subscribe('one', metrics, sampled_at=10, observed_at=1010)
    waiting = queue.get_nowait()
    assert waiting['run_id'] == 'run-one' and waiting['instance_id'] == 'one'
    assert waiting['live_generation']['request_key'] == metrics.token
    assert waiting['live_generation']['client_request_id'] == 'message'
    assert waiting['live_generation']['client_call_id'] == 'call'
    assert waiting['live_generation']['phase'] == 'waiting'
    observe(events, metrics, counter(10), 10)
    assert queue.get_nowait()['live_generation']['tokens_per_second'] is None
    observe(events, metrics, counter(11), 10.02)
    second = queue.get_nowait()
    assert second['observed_at'] == pytest.approx(1010.02)
    assert second['live_generation']['tokens_per_second'] == pytest.approx(50)
    assert second['live_generation']['window_s'] == pytest.approx(.02)
    observe(events, metrics, counter(15), 10.04)
    third = queue.get_nowait()['live_generation']
    assert third['tokens_per_second'] == pytest.approx(125)
    assert third['tokens_per_second'] != third['average_tokens_per_second']
    assert third['source'] == 'SSE timings.predicted_n' and third['available_for_chat']
    # A content chunk with no counter is not another token or a stalled counter sample.
    observe(events, metrics, {'choices': [{'delta': {'content': 'more private text'}}]}, 10.05)
    assert queue.empty()
    observe(events, metrics, None, 10.06)
    assert queue.empty()
    observe(events, metrics, counter(20), 10.08)
    assert queue.get_nowait()['live_generation']['tokens_per_second'] == pytest.approx(125)
    observe(events, metrics, counter(25), 14)
    assert queue.get_nowait()['live_generation']['tokens_per_second'] is None
    observe(events, metrics, counter(29), 14.02)
    assert queue.get_nowait()['live_generation']['tokens_per_second'] == pytest.approx(200)
    events.unsubscribe('one', queue)


def test_subscribing_mid_request_uses_current_counter_as_baseline_and_latest_only_buffer():
    events = GenerationEvents('run')
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe(counter(100))
    first = events.subscribe('one', metrics, sampled_at=10, observed_at=1010)
    baseline = first.get_nowait()['live_generation']
    assert baseline['generated_tokens'] == 100 and baseline['tokens_per_second'] is None
    observe(events, metrics, counter(105), 10.02)
    observe(events, metrics, counter(110), 10.04)
    assert first.qsize() == 1
    assert first.get_nowait()['live_generation']['tokens_per_second'] == pytest.approx(250)
    second = events.subscribe('one', metrics)
    assert second.get_nowait()['live_generation']['generated_tokens'] == 110
    events.unsubscribe('one', first)
    assert events.subscribed('one')
    observe(events, metrics, counter(120), 10.08)
    assert second.get_nowait()['live_generation']['tokens_per_second'] == pytest.approx(250)
    events.unsubscribe('one', second)
    assert not events.subscribed('one')
    fresh = events.subscribe('one', metrics, sampled_at=11, observed_at=1011)
    assert fresh.get_nowait()['live_generation']['tokens_per_second'] is None
    events.unsubscribe('one', fresh)


def test_events_do_no_counter_work_without_subscribers_and_clear_request_identity(monkeypatch):
    events = GenerationEvents('run')
    metrics = RequestMetrics(True, backend='ninfer')
    with monkeypatch.context() as patch:
        patch.setattr(metrics, 'live_generation', lambda *args: pytest.fail('unobserved request calculated a rate'))
        assert not events.publish('one', metrics, counter(10))
    queue = events.subscribe('one', metrics)
    queue.get_nowait()
    observe(events, metrics, counter(10), 10)
    observe(events, metrics, counter(20), 10.1)
    assert queue.get_nowait()['live_generation']['tokens_per_second'] == pytest.approx(100)
    events.publish('one', None, reset=True)
    assert queue.get_nowait()['live_generation'] is None
    next_request = RequestMetrics(True, backend='ninfer')
    events.publish('one', next_request, reset=True, sampled_at=11)
    new = queue.get_nowait()['live_generation']
    assert new['request_key'] == next_request.token and new['phase'] == 'waiting'
    assert new['tokens_per_second'] is None and new['generated_tokens'] is None
    events.unsubscribe('one', queue)


@pytest.mark.parametrize('backend,single_choice', [('strata', True), ('ninfer', False), ('sparkinfer', True)])
def test_missing_or_unattributable_stream_counts_do_not_invent_realtime_speed(backend, single_choice):
    events = GenerationEvents('run')
    metrics = RequestMetrics(True, backend=backend, single_choice=single_choice)
    queue = events.subscribe('one', metrics)
    queue.get_nowait()
    observe(events, metrics, {'choices': [{'delta': {'content': 'one potentially many-token chunk'}}]}, 10)
    event = queue.get_nowait()['live_generation']
    assert event['request_key'] == metrics.token
    assert event['tokens_per_second'] is None and not event['available_for_chat'] and event['reason']
    assert event['generated_tokens'] is None
    observe(events, metrics, {'choices': [{'delta': {'content': 'another text chunk'}}]}, 10.01)
    assert queue.empty()
    if not single_choice or backend == 'sparkinfer':
        observe(events, metrics, {'choices': [], 'usage': {'completion_tokens': 100}}, 10.02)
        assert queue.empty() or queue.get_nowait()['live_generation']['tokens_per_second'] is None
    events.unsubscribe('one', queue)


def test_prefill_event_windows_use_processed_work_without_cached_prefix():
    events = GenerationEvents('run')
    metrics = RequestMetrics(True, backend='ninfer')
    queue = events.subscribe('one', metrics, sampled_at=10)
    queue.get_nowait()
    for at, processed in ((10, 60), (10.1, 70)):
        observe(events, metrics, {'choices': [{'delta': {}}],
            'prompt_progress': {'total': 100, 'cache': 60, 'processed': processed}}, at)
        live = queue.get_nowait()['live_generation']
    assert live['prefill_cached_tokens'] == 60 and live['prefill_processed_tokens'] == 10
    assert live['prefill_tokens_per_second'] == pytest.approx(100)
    assert live['tokens_per_second'] is None
    observe(events, metrics, counter(5, 'reasoning_content'), 10.2)
    thinking = queue.get_nowait()['live_generation']
    assert thinking['phase'] == 'thinking' and thinking['prefill_tokens_per_second'] is None
    events.unsubscribe('one', queue)


async def response_event(response):
    while True:
        data = await asyncio.wait_for(response.body_iterator.__anext__(), 2)
        for line in data.splitlines():
            if line.startswith('data:'):
                return json.loads(line[5:])


async def advance_data(response):
    while True:
        data = await asyncio.wait_for(response.body_iterator.__anext__(), 2)
        if data.startswith('data:'):
            return data


async def open_gateway(client, path='/v1/chat/completions'):
    endpoint = next(route.endpoint for route in client._transport.app.routes if route.path == path)
    async def receive():
        return {'type': 'http.request', 'body': json.dumps({'model': 'one', 'messages': [], 'stream': True}).encode(), 'more_body': False}
    return await endpoint(Request({'type': 'http', 'method': 'POST', 'path': path,
        'headers': [(b'x-modloader-request-id', b'assistant-message'), (b'x-modloader-call-id', b'assistant-call')]}, receive))


@pytest.mark.parametrize('path', ['/v1/chat/completions', '/api/chat'])
async def test_gateway_pushes_ninfer_counters_before_any_telemetry_poll(controller, tmp_path, monkeypatch, path):
    import modloader.generation_events as event_module
    clock = [10.0]
    monkeypatch.setattr(event_module, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]))
    stream = GatewayStream([sse(counter(10, 'reasoning_content')), sse(counter(14)), sse('[DONE]')])
    async def handler(request):
        assert request.method == 'POST', 'event stream must not poll backend telemetry'
        assert json.loads(request.content)['timings_per_token'] is True
        return httpx.Response(200, stream=stream)
    client, instances, backend_clients = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    instance.backend = 'ninfer'
    response = events = None
    try:
        async with client:
            response = await open_gateway(client, path)
            endpoint = next(route.endpoint for route in client._transport.app.routes if route.path.endswith('/generation-events'))
            events = await endpoint(instance.id)
            waiting = await response_event(events)
            live = waiting['live_generation']
            assert waiting['run_id'] == controller.request_history.run_id and live['phase'] == 'waiting'
            assert live['client_request_id'] == ('assistant-message' if path == '/api/chat' else None)
            assert live['client_call_id'] == ('assistant-call' if path == '/api/chat' else None)
            untouched = {'token': 'poll-owned-sentinel'}
            controller.generation_samples[instance.id] = untouched
            first_bytes = await advance_data(response)
            first = await response_event(events)
            assert first['live_generation']['generated_tokens'] == 10
            clock[0] = 10.02
            second_bytes = await advance_data(response)
            second = await response_event(events)
            assert second['live_generation']['tokens_per_second'] == pytest.approx(200)
            assert second['live_generation']['request_key'] == live['request_key']
            assert controller.generation_samples[instance.id] is untouched
            assert json.loads(first_bytes[5:])['timings']['predicted_n'] == 10
            assert json.loads(second_bytes[5:])['timings']['predicted_n'] == 14
            await advance_data(response)
            assert (await response_event(events))['live_generation'] is None
            async for _ in response.body_iterator:
                pass
            assert instance.active_requests == 0 and not controller.active_request_tokens[instance.id]
            assert stream.closed and all(client.is_closed for client in backend_clients)
    finally:
        if events is not None:
            await events.body_iterator.aclose()
        if response is not None:
            await response.body_iterator.aclose()
        assert not controller.generation_events.subscribed(instance.id)
        await controller.close()


async def test_concurrent_gateway_requests_clear_fast_identity_and_restart_after_ambiguity(controller, tmp_path, monkeypatch):
    streams = []
    async def handler(request):
        stream = GatewayStream([sse(counter(10)), sse(counter(20)), sse('[DONE]')])
        streams.append(stream)
        return httpx.Response(200, stream=stream)
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    instance.backend = 'ninfer'
    first = second = None
    queue = None
    try:
        async with client:
            first = await open_gateway(client)
            metrics = controller.active_request_metrics[next(iter(controller.active_request_tokens[instance.id]))]
            queue = controller.generation_events.subscribe(instance.id, metrics)
            queue.get_nowait()
            second = await open_gateway(client)
            assert queue.get_nowait()['live_generation'] is None
            await advance_data(first)
            await advance_data(second)
            assert queue.empty()
            await first.body_iterator.aclose()
            restored = queue.get_nowait()
            unique = restored['live_generation']
            assert unique['request_key'] != metrics.token and unique['tokens_per_second'] is None
            assert restored['accounting']['request']['request_key'] == unique['request_key']
            assert restored['finalized'][0]['request']['request_key'] == metrics.token
            assert restored['finalized'][0]['request']['status'] == 'cancelled'
            assert instance.active_requests == 1
            await second.body_iterator.aclose()
            terminal = queue.get_nowait()
            assert terminal['live_generation'] is terminal['accounting'] is None
            assert terminal['finalized'][0]['request']['request_key'] == unique['request_key']
            assert instance.active_requests == 0 and all(stream.closed for stream in streams)
    finally:
        if queue is not None:
            controller.generation_events.unsubscribe(instance.id, queue)
        if first is not None:
            await first.body_iterator.aclose()
        if second is not None:
            await second.body_iterator.aclose()
        await controller.close()


async def test_generation_event_endpoint_auth_idle_heartbeat_and_disconnect_cleanup(controller, tmp_path, monkeypatch):
    async def handler(request):
        pytest.fail('idle generation event subscription called backend')
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    app = client._transport.app
    path = '/api/instances/' + instance.id + '/generation-events'
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    await incoming.put({'type': 'http.request', 'body': b'', 'more_body': False})
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'http_version': '1.1',
             'method': 'GET', 'scheme': 'http', 'path': path, 'raw_path': path.encode(), 'query_string': b'',
             'headers': [(b'host', b'testserver'), (b'authorization', b'Bearer management-secret')],
             'server': ('testserver', 80), 'client': ('127.0.0.1', 12345)}
    task = None
    try:
        async with client:
            assert (await client.get(path, headers={'Authorization': 'wrong'})).status_code == 401
            assert (await client.get('/api/instances/missing/generation-events')).status_code == 404
            assert not any(route.path.endswith('/generation-events') for route in build_app(controller, public=True).routes)
            task = asyncio.create_task(app(scope, incoming.get, outgoing.put))
            start = await asyncio.wait_for(outgoing.get(), 2)
            assert start['type'] == 'http.response.start' and start['status'] == 200
            body = await asyncio.wait_for(outgoing.get(), 2)
            assert body['body'].startswith(b':') and b'data:' not in body['body']
            assert controller.generation_events.subscribed(instance.id)
            await incoming.put({'type': 'http.disconnect'})
            await asyncio.wait_for(task, 2)
            assert not controller.generation_events.subscribed(instance.id)
            assert not controller.telemetry_samples and not controller.generation_samples
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await controller.close()


async def test_upstream_cancel_clears_subscribed_identity_and_releases_inference_resources(controller, tmp_path, monkeypatch):
    waiting = asyncio.Event()
    stream = GatewayStream([sse(counter(10))], waiting)
    async def handler(request):
        return httpx.Response(200, stream=stream)
    client, instances, backend_clients = gateway_client(controller, tmp_path, monkeypatch, handler)
    instance = instances['one']
    instance.backend = 'ninfer'
    queue = task = None
    try:
        async with client:
            task = asyncio.create_task(client.post('/v1/chat/completions', json={'model': 'one', 'messages': [], 'stream': True}))
            await asyncio.wait_for(waiting.wait(), 2)
            current = controller.generation_event_metrics(instance.id)
            queue = controller.generation_events.subscribe(instance.id, current)
            assert queue.get_nowait()['live_generation']['generated_tokens'] == 10
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert queue.get_nowait()['live_generation'] is None
            assert controller.request_metrics[instance.id]['status'] == 'cancelled'
            assert not controller.active_request_tokens[instance.id] and instance.active_requests == 0
            assert stream.closed and all(client.is_closed for client in backend_clients)
            assert controller.generation_events.subscribed(instance.id)  # Only the request ended, not its monitor.
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if queue is not None:
            controller.generation_events.unsubscribe(instance.id, queue)
        await controller.close()
