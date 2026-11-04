"""Native phase pairs and lossless accounting on the coalesced fast lane."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from modloader.conversations import _history_request
from modloader.generation_events import GenerationEvents
from modloader.request_history import _clean_record, request_history_stats, HistoryArchive
from modloader.request_metrics import RequestMetrics, generation_window, request_metrics_group
from test_core import controller, gateway_client, GatewayStream, sse
from test_generation_events import advance_data, observe
from test_request_history import ledger, instance, assistant, follow, persist_internal_metrics


def native(count, milliseconds, **extra):
    return {'timings': {'predicted_n': count, 'predicted_ms': milliseconds, **extra}}


def test_native_phase_pairs_and_means_use_phase_time_not_gateway_or_sampled_age():
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.started -= 100
    metrics.observe({'timings': {'prompt_n': 30, 'prompt_ms': 50, 'cache_n': 70,
                                 'predicted_n': 20, 'predicted_ms': 500}})
    record = metrics.snapshot()
    decode, prefill = (record['measurements'][phase] for phase in ('decode', 'prefill'))
    assert decode == {'tokens': 20, 'seconds': .5, 'value': 40, 'basis': 'backend_decode',
                      'reported': 1, 'coverage': 'complete'}
    assert prefill['tokens'] == 30 and prefill['seconds'] == .05 and prefill['value'] == 600
    _, live = generation_window(None, metrics.token, metrics.live_generation(), 10)
    assert live['average_tokens_per_second'] == 40 and live['average_basis'] == 'backend_decode'
    assert record['duration_s'] >= 100 and decode['seconds'] != record['duration_s']


def test_normalized_fresh_counters_unify_live_history_and_recent_request_without_sticky_usage():
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe({'usage': {'prompt_tokens': 200, 'completion_tokens': 100,
                              'prompt_tokens_details': {'cached_tokens': 50}}})
    metrics.observe({'timings': {'predicted_n': 150, 'predicted_ms': 3000, 'prompt_n': 170, 'cache_n': 60}})
    snapshot = metrics.snapshot()
    assert snapshot['usage']['completion_tokens'] == 100  # Keep original provider fields truthful.
    assert snapshot['counters'] == {'input': 200, 'output': 150, 'cached': 60}
    live = metrics.live_generation()['_overview']
    assert live['generated_tokens'] == 150 and live['prompt_tokens'] == 200 and live['prefill_cached_tokens'] == 60
    assert request_history_stats([snapshot])['output'] == {'value': 150, 'reported': 1}
    metrics.observe({'usage': {'completion_tokens': 160, 'prompt_tokens_details': {'cached_tokens': 55}}})
    assert metrics.snapshot()['counters'] == {'input': 200, 'output': 160, 'cached': 55}
    assert metrics.live_generation()['_overview']['generated_tokens'] == 160
    assert metrics.live_generation()['_overview']['prefill_cached_tokens'] == 55
    assert metrics.snapshot()['measurements']['decode']['tokens'] == 150
    assert metrics.snapshot()['measurements']['decode']['coverage'] == 'partial'
    group = request_metrics_group(metrics.finish('completed'))
    values = {metric['id']: metric['value'] for metric in group['metrics']}
    assert values['prompt_tokens'] == 200 and values['completion_tokens'] == 160


def test_new_counter_unknown_and_zero_never_fall_back_to_sticky_raw_fields():
    metrics = RequestMetrics(True, backend='ninfer')
    assert metrics.snapshot()['counters'] == {'input': None, 'output': None, 'cached': None}
    record = {'status': 'completed', 'duration_s': 1, 'usage': {'prompt_tokens': 100, 'completion_tokens': 50},
              'timings': {'cache_n': 40}, 'counters': {'input': None, 'output': 0, 'cached': None}}
    stats = request_history_stats([record])
    assert stats['input'] == stats['cached'] == {'value': None, 'reported': 0}
    assert stats['output'] == {'value': 0, 'reported': 1}
    legacy = {key: value for key, value in record.items() if key != 'counters'}
    assert request_history_stats([legacy])['input']['value'] == 100


@pytest.mark.parametrize('backend,delta,phase', [
    ('llama_cpp', {'content': 'answer'}, 'output'),
    ('llama_cpp', {'reasoning_content': 'thinking'}, 'thinking'),
    ('ninfer', {'tool_calls': [{'index': 0, 'function': {'name': 'lookup'}}]}, 'generating'),
])
def test_prefill_zero_yields_to_attributable_decode_samples_after_first_output(backend, delta, phase):
    metrics = RequestMetrics(True, backend=backend)
    metrics.observe({'prompt_progress': {'total': 1024, 'processed': 950, 'cache': 900}})
    sample = {'backend_running': 1, 'backend_queued': 0, 'source': '/slots next_token.n_decoded',
              '_generated_tokens': 7, '_overview': {'phase': 'generating'}}
    # An older backend sample must not turn current prefill into generation.
    prefill = metrics.live_generation(sample)
    assert prefill['_overview']['generated_tokens'] == 0
    assert prefill['_overview']['phase'] == 'prefill'

    metrics.observe({'choices': [{'delta': delta}]})
    first = metrics.live_generation(sample)
    window, view = generation_window(None, metrics.token, first, 20)
    assert first['source'] == sample['source'] and view['generated_tokens'] == 7
    assert view['phase'] == phase and view['prefill_processed_tokens'] == 50
    assert view['tokens_per_second'] is None  # One decode sample cannot define speed.
    second = metrics.live_generation({**sample, '_generated_tokens': 9})
    _, view = generation_window(window, metrics.token, second, 20.5)
    assert view['tokens_per_second'] == 4 and view['average_basis'] == 'sampled_decode'

    # Real SSE counts take ownership; neither old polls nor delayed prefill can undo them.
    metrics.observe({'usage': {'completion_tokens': 10}})
    metrics.observe({'prompt_progress': {'processed': 974}})
    final = metrics.live_generation(sample)
    assert final['source'] == 'SSE usage.completion_tokens'
    assert final['_overview']['generated_tokens'] == 10
    assert final['_overview']['phase'] == phase
    assert metrics.snapshot()['counters']['output'] == 10


@pytest.mark.parametrize('report,source', [
    ({'usage': {'completion_tokens': 0}}, 'SSE usage.completion_tokens'),
    ({'timings': {'predicted_n': 0}}, 'SSE timings.predicted_n'),
])
def test_actual_sse_zero_is_not_cleared_with_the_provisional_prefill_zero(report, source):
    metrics = RequestMetrics(True, backend='llama_cpp')
    metrics.observe({'prompt_progress': {'total': 100, 'processed': 20, 'cache': 0}})
    metrics.observe({**report, 'choices': [{'delta': {'content': 'output'}}]})
    live = metrics.live_generation({'backend_running': 1, 'backend_queued': 0,
        'source': '/slots next_token.n_decoded', '_generated_tokens': 7})
    assert live['source'] == source and live['_overview']['generated_tokens'] == 0


@pytest.mark.parametrize('sample', [
    None,
    {'backend_running': 2, 'backend_queued': 0},
    {'backend_running': 1, 'backend_queued': 1},
])
def test_first_output_with_no_attributable_counter_is_unknown_not_prefill_zero(sample):
    metrics = RequestMetrics(True, backend='llama_cpp')
    metrics.observe({'prompt_progress': {'total': 100, 'processed': 20, 'cache': 0}})
    metrics.observe({'choices': [{'delta': {'content': 'output'}}]})
    backend = {**sample, 'source': '/slots next_token.n_decoded', '_generated_tokens': 99} if sample else None
    live = metrics.live_generation(backend)
    _, view = generation_window(None, metrics.token, live, 20)
    assert metrics.snapshot()['counters']['output'] is None
    assert view['phase'] == 'output'
    assert view['generated_tokens'] is None and view['tokens_per_second'] is None


def test_multichoice_native_branch_count_cannot_replace_reported_aggregate_usage_or_phase_pair():
    metrics = RequestMetrics(True, backend='ninfer', single_choice=False)
    metrics.observe({**native(10, 100, predicted_per_second=100, prompt_n=20, prompt_ms=50, prompt_per_second=400),
                     'usage': {'completion_tokens': 100}})
    live = metrics.live_generation({'backend_running': 1, 'backend_queued': 0, 'source': 'backend sample',
        '_generated_tokens': 10, '_overview': {'generated_tokens': 10, 'average_tokens_per_second': 100,
                                              'average_prefill_tokens_per_second': 400}})
    snapshot = metrics.finish('completed')
    assert snapshot['counters']['output'] == 100
    assert snapshot['measurements']['decode']['reported'] == 0
    assert live['_generated_tokens'] is None and live['_overview'].get('average_tokens_per_second') is None
    assert live['_overview'].get('average_prefill_tokens_per_second') is None
    assert not any(key.startswith(('predicted_', 'prompt_')) for key in snapshot['timings'])
    assert 'average_prefill_tokens_per_second' not in snapshot['prefill']
    assert metrics.timings['predicted_n'] == 10  # Original upstream observation is not rewritten.
    speed = next(metric for metric in request_metrics_group(snapshot)['metrics'] if metric['id'] == 'output_speed')
    assert speed['rate_basis'] != 'decode'


@pytest.mark.parametrize('partial', [native(30, None), {'timings': {'predicted_ms': 700}},
                                     {'timings': {'predicted_per_second': 25}}])
def test_unmatched_native_updates_preserve_only_real_reported_prefix(partial):
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe(native(20, 500))
    metrics.observe(partial)
    measured = metrics.snapshot()['measurements']['decode']
    assert measured['tokens'] == 20 and measured['seconds'] == .5
    assert measured['reported'] == 1 and measured['coverage'] == 'partial'
    stats = request_history_stats([metrics.snapshot()])
    assert stats['decode'] == {'value': 40, 'tokens': 20, 'seconds': .5,
                               'reported': 1, 'partial_reported': 1}
    metrics.observe(native(40, 1000))
    assert metrics.snapshot()['measurements']['decode']['coverage'] == 'complete'


def test_separate_counter_and_native_mean_do_not_make_a_whole_request_pair():
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe({'timings': {'predicted_n': 10}})
    metrics.observe({'timings': {'predicted_per_second': 100}})
    measurement = metrics.snapshot()['measurements']['decode']
    assert measurement['reported'] == 0 and measurement['seconds'] is None
    assert measurement['coverage'] == 'unknown'
    assert metrics.live_generation()['_overview']['average_tokens_per_second'] == 100
    assert request_history_stats([metrics.snapshot()])['decode']['reported'] == 0
    # An explicit counter reset cannot retain a prefix from the old sequence.
    metrics.observe(native(30, 300))
    metrics.observe({'timings': {'predicted_n': 2}})
    assert metrics.snapshot()['measurements']['decode']['reported'] == 0


@pytest.mark.parametrize('phase,prefix', [('decode', 'predicted'), ('prefill', 'prompt')])
@pytest.mark.parametrize('rate_fields', [{'_per_second': 25, '_per_token_ms': 40}, {'_per_token_ms': 40}])
def test_fresh_matching_pair_restores_coverage_and_replaces_older_native_mean(phase, prefix, rate_fields):
    metrics = RequestMetrics(True, backend='ninfer')
    pair = {'timings': {prefix + '_n': 20, prefix + '_ms': 500}}
    metrics.observe(pair)
    metrics.observe({'timings': {prefix + suffix: value for suffix, value in rate_fields.items()}})
    partial = metrics.snapshot()
    assert partial['measurements'][phase]['coverage'] == 'partial'
    assert all(partial['timings'][prefix + suffix] == value for suffix, value in rate_fields.items())
    live_mean = 'average_tokens_per_second' if phase == 'decode' else 'average_prefill_tokens_per_second'
    assert metrics.live_generation()['_overview'][live_mean] == 25  # Real independent mean-only report.
    partial_group = request_metrics_group({**partial, 'status': 'completed',
        'ended_at': partial['started_at'] + partial['duration_s']})
    speed_id = 'output_speed' if phase == 'decode' else 'prefill_speed'
    assert next(metric['value'] for metric in partial_group['metrics'] if metric['id'] == speed_id) == 25
    metrics.observe(pair)  # Same numeric pair is fresh evidence after the unmatched report.
    full = metrics.finish('completed')
    assert full['measurements'][phase]['coverage'] == 'complete'
    assert full['measurements'][phase]['value'] == 40
    assert prefix + '_per_second' not in full['timings']
    assert prefix + '_per_token_ms' not in full['timings']
    assert all(metrics.timings[prefix + suffix] == value for suffix, value in rate_fields.items())
    group = request_metrics_group(full)
    assert next(metric['value'] for metric in group['metrics'] if metric['id'] == speed_id) == 40


def test_true_zero_pair_and_unknown_pair_are_distinct_without_invented_backend_metrics():
    measured = RequestMetrics(True, backend='ninfer')
    measured.observe(native(0, 0))
    assert measured.snapshot()['measurements']['decode']['reported'] == 1
    assert measured.snapshot()['measurements']['decode']['value'] is None
    unknown = RequestMetrics(True, backend='strata')
    unknown.observe({'choices': [{'delta': {'content': 'unmeasured'}}]})
    assert unknown.snapshot()['measurements']['decode']['reported'] == 0
    assert unknown.snapshot()['measurements']['decode']['seconds'] is None


def test_existing_native_backend_overview_pairs_remain_available_without_sse_counters():
    metrics = RequestMetrics(True, backend='strata')
    metrics.live_generation({'backend_running': 1, 'backend_queued': 0, 'source': '/metrics live.generated',
        '_generated_tokens': 20, '_overview': {'phase': 'generating', 'generated_tokens': 20,
                                              'average_tokens_per_second': 40}})
    assert metrics.snapshot()['measurements']['decode']['seconds'] == .5
    assert request_history_stats([metrics.snapshot()])['decode']['tokens'] == 20
    other = RequestMetrics(True, backend='strata')
    other.live_generation({'backend_running': 2, 'backend_queued': 0, 'source': '/metrics live.generated',
        '_generated_tokens': 999, '_overview': {'generated_tokens': 999, 'average_tokens_per_second': 1}})
    assert other.snapshot()['measurements']['decode']['reported'] == 0


def test_native_counter_only_tail_does_not_get_a_fabricated_rate_in_recent_request_group():
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe(native(20, 500))
    metrics.observe({'timings': {'predicted_n': 30}})
    group = request_metrics_group(metrics.finish('completed'))
    speed = next(metric for metric in group['metrics'] if metric['id'] == 'output_speed')
    assert speed['value'] == 40 and speed.get('coverage') == 'partial'


def test_fast_metric_only_frames_publish_without_resampling_or_losing_real_window():
    events, metrics = GenerationEvents('run'), RequestMetrics(True, backend='ninfer')
    queue = events.subscribe('one', metrics, sampled_at=10)
    queue.get_nowait()
    observe(events, metrics, native(10, 100), 10)
    queue.get_nowait()
    observe(events, metrics, native(20, 500), 10.1)
    previous = queue.get_nowait()
    window = deepcopy(events._windows['one'])
    # Native average/time, usage and TTFT can all advance without another token.
    observe(events, metrics, {'timings': {'predicted_n': 20, 'predicted_ms': 700,
                                        'predicted_per_second': 20 / .7},
                             'usage': {'prompt_tokens': 120, 'ttft_ms': 25,
                                       'prompt_tokens_details': {'cached_tokens': 80}}}, 10.2)
    updated = queue.get_nowait()
    assert events._windows['one'] == window
    assert updated['live_generation']['tokens_per_second'] == previous['live_generation']['tokens_per_second']
    assert updated['live_generation']['average_tokens_per_second'] == pytest.approx(20 / .7)
    request = updated['accounting']['request']
    assert request['measurements']['decode']['seconds'] == .7
    assert request['usage']['prompt_tokens_details']['cached_tokens'] == 80
    observe(events, metrics, {'timings': {'predicted_per_second': 25}}, 10.3)
    mean = queue.get_nowait()
    assert mean['live_generation']['average_tokens_per_second'] == 25
    assert mean['accounting']['request']['measurements']['decode']['coverage'] == 'partial'
    assert events._windows['one'] == window
    observe(events, metrics, {'usage': {'prompt_tokens': 140}}, 10.31)
    assert queue.get_nowait()['accounting']['request']['usage']['prompt_tokens'] == 140
    observe(events, metrics, {'usage': {'prompt_tokens_details': {'cached_tokens': 90}}}, 10.32)
    assert queue.get_nowait()['accounting']['request']['usage']['prompt_tokens_details']['cached_tokens'] == 90
    observe(events, metrics, {'usage': {'ttft_ms': 30}}, 10.33)
    assert queue.get_nowait()['accounting']['request']['usage']['ttft_ms'] == 30
    assert events._windows['one'] == window
    observe(events, metrics, {'choices': [{'delta': {'content': 'not a counter'}}]}, 10.4)
    # Initial content changes phase, subsequent same-phase text is not a metric.
    if not queue.empty():
        queue.get_nowait()
    observe(events, metrics, {'choices': [{'delta': {'content': 'still not a counter'}}]}, 10.5)
    assert queue.empty() and events._windows['one'] == window
    events.unsubscribe('one', queue)


def test_fresh_progress_wins_over_retained_prefill_timings_and_uses_no_cached_work():
    events, metrics = GenerationEvents('run'), RequestMetrics(True, backend='ninfer')
    queue = events.subscribe('one', metrics, sampled_at=10)
    queue.get_nowait()
    observe(events, metrics, {'timings': {'prompt_n': 10, 'prompt_ms': 20, 'cache_n': 60},
                             'prompt_progress': {'total': 100, 'processed': 70, 'cache': 60}}, 10)
    queue.get_nowait()
    observe(events, metrics, {'prompt_progress': {'processed': 80}}, 10.1)
    live = queue.get_nowait()['live_generation']
    assert live['prefill_processed_tokens'] == 20 and live['prefill_cached_tokens'] == 60
    assert metrics.snapshot()['measurements']['prefill']['tokens'] == 10
    assert metrics.snapshot()['measurements']['prefill']['coverage'] == 'partial'
    observe(events, metrics, {'prompt_progress': {'processed': 90}}, 10.2)
    assert queue.get_nowait()['live_generation']['prefill_tokens_per_second'] == pytest.approx(100)


def test_complete_prompt_target_wins_over_partial_work_and_delayed_progress_cannot_undo_prefill():
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe({'timings': {'prompt_n': 30, 'prompt_ms': 50, 'cache_n': 70},
                     'prompt_progress': {'total': 120, 'cache': 70, 'processed': 100}})
    assert request_history_stats([metrics.snapshot()])['input']['value'] == 120
    metrics.observe({'timings': {'prompt_n': 50, 'prompt_ms': 100, 'cache_n': 70,
                                 'predicted_n': 1, 'predicted_ms': 5}})
    metrics.observe({'prompt_progress': {'processed': 100, 'cache': 60}})
    assert metrics.phase in ('generating', 'thinking', 'output')
    assert metrics.snapshot()['prefill']['prefill_processed_tokens'] == 50
    assert metrics.snapshot()['prefill']['prefill_cached_tokens'] == 70
    assert metrics.snapshot()['measurements']['prefill']['coverage'] == 'complete'


def test_terminal_accounting_survives_null_next_request_and_slow_subscribers():
    events, metrics = GenerationEvents('run'), RequestMetrics(True, backend='ninfer')
    first = events.subscribe('one', metrics)
    second = events.subscribe('one', metrics)
    first.get_nowait()
    second.get_nowait()
    metrics.observe(native(20, 500))
    metrics.finish('cancelled')
    final = {'request': metrics.result, 'turn': {'key': 'request:' + metrics.token}}
    events.publish('one', None, reset=True, finalized=[final])
    assert first.get_nowait()['finalized'] == [final]
    events.publish('one', None, reset=True)
    newer = RequestMetrics(True, backend='ninfer')
    events.publish('one', newer, reset=True)
    latest = second.get_nowait()
    assert latest['live_generation']['request_key'] == newer.token
    assert latest['finalized'] == [final] and latest['finalized'][0]['request']['status'] == 'cancelled'
    events.publish('one', None, finalized=[final, final], reset=True)
    assert len(first.get_nowait()['finalized']) == 1
    events.unsubscribe('one', first)
    events.unsubscribe('one', second)
    assert not events._windows and not events._latest and not events._subscribers


@pytest.mark.parametrize('status', ['failed', 'cancelled'])
async def test_prefix_measurements_survive_archive_and_internal_history_cleaning(ledger, status, controller, tmp_path):
    metrics = RequestMetrics(True, backend='ninfer')
    ledger.start(metrics, instance(), [])
    metrics.observe(native(20, 500))
    metrics.observe({'timings': {'predicted_n': 30}})
    metrics.finish(status)
    ledger.finish(metrics)
    clean = _clean_record(ledger.records[metrics.token])
    assert clean['measurements']['decode']['coverage'] == 'partial'
    assert request_history_stats([clean])['decode']['seconds'] == .5
    directory = tmp_path / 'paired-archive'
    ledger.configure(True, directory)
    assert ledger.archive.drain()
    restored = HistoryArchive(directory).read()[0][ledger.run_id]['records'][metrics.token]
    assert restored['measurements'] == clean['measurements']
    assert restored['counters'] == clean['counters'] == metrics.snapshot()['counters']
    assert request_history_stats([restored])['decode']['seconds'] == .5
    request = {'id': 'call', 'index': 1, 'kind': 'generation', 'status': status, 'startedAt': 1000,
               'endedAt': 2000, 'elapsed': 1, 'labels': [], 'usage': {}, 'timings': dict(metrics.timings),
               'measurements': metrics.snapshot()['measurements']}
    request['counters'] = metrics.snapshot()['counters']
    internal, record = _history_request(request)
    assert internal['measurements'] == clean['measurements'] == record['measurements']
    assert request_history_stats([record])['decode']['tokens'] == 20
    persist_internal_metrics(controller, [request])
    reloaded = controller.conversations.request_history()['records'][0]
    assert reloaded['measurements'] == clean['measurements']
    assert reloaded['counters'] == clean['counters']
    assert request_history_stats([reloaded])['decode']['seconds'] == .5
    # Older saved requests without normalized pairs retain their old contract.
    legacy = {key: value for key, value in record.items() if key != 'measurements'}
    assert request_history_stats([legacy])['decode']['tokens'] == 30


async def open_request(client, messages, path='/v1/chat/completions', *, with_ids=True):
    endpoint = next(route.endpoint for route in client._transport.app.routes if route.path == path)
    async def receive():
        return {'type': 'http.request', 'body': json.dumps({'model': 'one', 'messages': messages,
                                                          'stream': True}).encode(), 'more_body': False}
    headers = [(b'x-modloader-request-id', b'assistant-message'), (b'x-modloader-call-id', b'assistant-call')] if with_ids else []
    return await endpoint(Request({'type': 'http', 'method': 'POST', 'path': path, 'headers': headers}, receive))


@pytest.mark.parametrize('with_ids', [True, False])
async def test_internal_accounting_keeps_entrypoint_source_call_identity_and_final_report(controller, tmp_path, monkeypatch, with_ids):
    stream = GatewayStream([sse(native(10, 100)), sse('[DONE]')])
    async def handler(request):
        return httpx.Response(200, stream=stream)
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    target = instances['one']
    target.backend = 'ninfer'
    queue = controller.generation_events.subscribe(target.id)
    response = None
    try:
        async with client:
            response = await open_request(client, [], '/api/chat', with_ids=with_ids)
            waiting = queue.get_nowait()
            assert waiting['accounting']['turn']['source'] == 'internal'
            assert waiting['accounting']['turn']['prior'] is None
            expected_id = 'assistant-message' if with_ids else None
            assert waiting['accounting']['request']['client_request_id'] == expected_id
            assert waiting['accounting']['turn']['client_request_id'] == expected_id
            chunks = [data async for data in response.body_iterator]
            terminal = queue.get_nowait()
            assert terminal['live_generation'] is terminal['accounting'] is None
            request = terminal['finalized'][0]['request']
            assert request['measurements']['decode']['seconds'] == .1
            assert request['client_call_id'] == ('assistant-call' if with_ids else None)
            reports = [json.loads(line[5:])['modloader_request'] for data in chunks for line in data.splitlines()
                       if line.startswith('data:') and 'modloader_request' in line]
            assert reports and reports[-1]['measurements'] == request['measurements']
            assert not controller.request_history.records
    finally:
        controller.generation_events.unsubscribe(target.id, queue)
        if response is not None:
            await response.body_iterator.aclose()
        await controller.close()


async def test_gateway_produces_replaceable_request_basis_and_lossless_terminal_fixture(controller, tmp_path, monkeypatch):
    import modloader.generation_events as event_module
    clock = [10.0]
    monkeypatch.setattr(event_module, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]))
    message = assistant(['fixture-call'])
    tool = {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, **message['tool_calls'][0]}]},
                         'finish_reason': 'tool_calls'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
            'timings': {'prompt_n': 40, 'prompt_ms': 200, 'cache_n': 60, 'predicted_n': 20, 'predicted_ms': 1000}}
    streams = [GatewayStream([sse(tool), sse('[DONE]')]), GatewayStream([
        sse({'prompt_progress': {'total': 120, 'cache': 70, 'processed': 100},
             'timings': {'prompt_n': 30, 'prompt_ms': 50, 'cache_n': 70}}),
        sse({**native(10, 100), 'usage': {'completion_tokens': 10}}), sse(native(20, 500)),
        sse({'timings': {'predicted_ms': 600, 'predicted_per_second': 20 / .6}}),
        sse(native(24, 800)), sse('[DONE]')]), GatewayStream([sse('[DONE]')])]
    async def handler(request):
        assert request.method == 'POST', 'fast accounting must not call telemetry'
        return httpx.Response(200, stream=streams.pop(0))
    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    target = instances['one']
    target.backend = 'ninfer'
    queue = controller.generation_events.subscribe(target.id)
    responses = []
    try:
        async with client:
            first = await open_request(client, [])
            responses.append(first)
            queue.get_nowait()
            async for _ in first.body_iterator:
                pass
            sibling = queue.get_nowait()['finalized'][0]['request']
            assert sibling['measurements']['decode']['tokens'] == 20
            second = await open_request(client, follow([], message))
            responses.append(second)
            waiting = queue.get_nowait()
            baseline = controller.request_history.latest(target.id)
            current_key = waiting['accounting']['request']['request_key']
            turn = waiting['accounting']['turn']
            assert current_key != sibling['request_key']
            assert turn == {'source': 'external', 'key': 'request:' + current_key,
                            'turn_id': None, 'client_request_id': None,
                            'request_keys': [current_key], 'prior': None}
            revision = controller.request_history.usage_revision({})
            payloads = [waiting]
            for index in range(4):
                clock[0] += .1
                await advance_data(second)
                payloads.append(queue.get_nowait())
                assert controller.request_history.usage_revision({}) == revision
            assert payloads[2]['accounting']['request']['measurements']['decode']['value'] == 100
            assert payloads[3]['accounting']['request']['measurements']['decode']['value'] == 40
            assert payloads[3]['accounting']['request']['usage']['completion_tokens'] == 10
            assert payloads[3]['accounting']['request']['counters']['output'] == 20
            assert payloads[4]['live_generation']['average_tokens_per_second'] == pytest.approx(20 / .6)
            assert payloads[4]['accounting']['request']['measurements']['decode']['coverage'] == 'partial'
            assert payloads[4]['live_generation']['tokens_per_second'] == payloads[3]['live_generation']['tokens_per_second']
            async for _ in second.body_iterator:
                pass
            # No consumer read between final native counters, terminal and next request.
            third = await open_request(client, [{'role': 'user', 'content': 'different user turn'}])
            responses.append(third)
            combined = queue.get_nowait()
            assert combined['accounting']['request']['request_key'] != current_key
            final = next(item for item in combined['finalized'] if item['request']['request_key'] == current_key)
            assert final['request']['measurements']['decode']['tokens'] == 24
            assert final['request']['measurements']['decode']['seconds'] == .8
            assert final['request']['usage']['completion_tokens'] == 10
            assert final['request']['counters']['output'] == 24
            assert 'predicted_per_second' not in final['request']['timings']
            assert next(metric['value'] for metric in request_metrics_group(final['request'])['metrics']
                        if metric['id'] == 'output_speed') == 30
            assert final['turn'] == turn
            payloads.append(combined)
            assert controller.request_history.usage_revision({}) != revision
            # Reconnection rebuilds the finished request from retained history, not event memory.
            controller.generation_events.unsubscribe(target.id, queue)
            queue = controller.generation_events.subscribe(target.id)
            recovered = controller.request_history.page({}, [], request_key=current_key)
            assert len(recovered['items']) == 1
            assert recovered['items'][0]['request_key'] == current_key
            assert request_history_stats(recovered['items'])['decode']['value'] == 30
            completed_stats = request_history_stats([sibling, recovered['items'][0]])
            assert completed_stats['request_count'] == 2
            assert completed_stats['decode']['value'] == pytest.approx(44 / 1.8)
            fixture = {'run_id': controller.request_history.run_id, 'instance_id': target.id,
                       'baseline': {'external_history': baseline}, 'events': payloads}
            destination = tmp_path / 'producer-fixture.json'
            serialized = json.dumps(fixture, ensure_ascii=False, allow_nan=False, indent=2)
            assert 'arguments-sentinel' not in serialized and 'different user turn' not in serialized
            pending = destination.with_suffix('.pending.json')
            pending.write_text(serialized, encoding='utf-8')
            pending.replace(destination)
    finally:
        controller.generation_events.unsubscribe(target.id, queue)
        for response in responses:
            await response.body_iterator.aclose()
        await controller.close()
