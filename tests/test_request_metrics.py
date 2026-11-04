"""Fresh backend samples and retained request accounting are separate contracts."""
import pytest

from modloader.request_metrics import RequestMetrics, generation_window, request_metrics_group


def strata_sample(count=140, mean=40, **overview):
    return {'scope': 'model', 'source': '/metrics live.generated', 'backend_running': 1,
            'backend_queued': 0, '_generated_tokens': count,
            '_overview': {'phase': 'generating', 'prompt_tokens': 1000, 'generated_tokens': count,
                          'average_tokens_per_second': mean, **overview}}


@pytest.mark.parametrize('missing', [
    None,
    {'scope': 'model', 'source': '/metrics live.generated', 'backend_running': 1, 'backend_queued': 0},
    {'scope': 'model', 'source': '/metrics live.generated', 'backend_running': 2, 'backend_queued': 0,
     '_generated_tokens': 999},
])
@pytest.mark.parametrize('status', ['completed', 'cancelled'])
def test_native_counter_requires_fresh_attributable_sample_but_accounting_keeps_last_observation(missing, status):
    metrics = RequestMetrics(True, backend='strata')
    metrics.live_generation(strata_sample())
    live = metrics.live_generation(missing)
    _, view = generation_window(None, metrics.token, live, 10)
    assert live['_generated_tokens'] is None
    assert view['generated_tokens'] is None and view['context_tokens'] is None
    assert view['tokens_per_second'] is None and view['average_tokens_per_second'] is None
    assert view['request_key'] == metrics.token
    record = metrics.finish(status)
    assert record['counters']['output'] == 140 and record['counters']['input'] == 1000
    assert record['measurements']['decode']['value'] == 40
    group = {metric['id']: metric for metric in request_metrics_group(record)['metrics']}
    assert group['completion_tokens']['value'] == 140
    assert group['output_speed']['value'] == (40 if status == 'completed' else None)
    assert group['output_speed']['rate_basis'] == ('decode' if status == 'completed' else None)


def test_strata_mean_is_decode_time_not_request_age_or_end_to_end_rate():
    metrics = RequestMetrics(True, backend='strata')
    metrics.started -= 100
    _, view = generation_window(None, metrics.token, metrics.live_generation(strata_sample(20, 40)), 10)
    assert view['average_tokens_per_second'] == 40 and view['average_basis'] == 'backend_decode'
    assert metrics.snapshot()['measurements']['decode']['seconds'] == .5


def test_end_to_end_backend_rate_cannot_be_recorded_as_a_decode_measurement():
    other = RequestMetrics(True, backend='strata')
    _, end_to_end = generation_window(None, other.token, other.live_generation(
        strata_sample(20, 4, average_basis='end_to_end')), 10)
    assert end_to_end['average_tokens_per_second'] is None and end_to_end['average_basis'] is None
    assert other.snapshot()['measurements']['decode']['reported'] == 0


def test_missing_backend_sample_does_not_clear_independent_exact_sse_count_or_mean():
    metrics = RequestMetrics(True, backend='ninfer')
    metrics.observe({'usage': {'prompt_tokens': 1000},
                     'timings': {'predicted_n': 20, 'predicted_ms': 500}})
    live = metrics.live_generation(None)
    _, view = generation_window(None, metrics.token, live, 10)
    assert live['source'] == 'SSE timings.predicted_n'
    assert view['generated_tokens'] == 20 and view['context_tokens'] == 1020
    assert view['average_tokens_per_second'] == 40 and view['average_basis'] == 'backend_decode'


def test_unchanged_fresh_backend_counter_is_a_real_zero_rate_not_a_missing_sample():
    metrics = RequestMetrics(True, backend='strata')
    window, _ = generation_window(None, metrics.token, metrics.live_generation(strata_sample()), 10)
    _, view = generation_window(window, metrics.token, metrics.live_generation(strata_sample()), 11)
    assert view['tokens_per_second'] == 0 and view['window_s'] == 1
    assert view['generated_tokens'] == 140 and view['available_for_chat']


def observe_sse_count(metrics, phase, count):
    if phase == 'decode':
        metrics.observe({'usage': {'completion_tokens': count}})
    else:
        metrics.observe({'prompt_progress': {'total': 1000, 'processed': count, 'cache': 0}})


@pytest.mark.parametrize('phase, rate, span', [
    ('decode', 'tokens_per_second', 'window_s'),
    ('prefill', 'prefill_tokens_per_second', 'prefill_window_s'),
])
def test_sse_poll_requires_a_new_counter_observation_and_restarts_after_missing_frames(phase, rate, span):
    metrics = RequestMetrics(True, backend='ninfer')
    observe_sse_count(metrics, phase, 100)
    window, _ = generation_window(None, metrics.token, metrics.live_generation(None), 10)
    observe_sse_count(metrics, phase, 140)
    window, view = generation_window(window, metrics.token, metrics.live_generation(None), 11)
    assert view[rate] == 40 and view[span] == 1

    window, missing = generation_window(window, metrics.token, metrics.live_generation(None), 12)
    assert missing[rate] is None and missing[span] is None
    metrics.observe({'choices': [{'delta': {'role': 'assistant'}}]})
    window, no_counter = generation_window(window, metrics.token, metrics.live_generation(None), 13)
    assert no_counter[rate] is None and no_counter[span] is None

    observe_sse_count(metrics, phase, 180)
    window, recovering = generation_window(window, metrics.token, metrics.live_generation(None), 14)
    assert recovering[rate] is None and recovering[span] is None
    observe_sse_count(metrics, phase, 200)
    _, recovered = generation_window(window, metrics.token, metrics.live_generation(None), 15)
    assert recovered[rate] == 20 and recovered[span] == 1


@pytest.mark.parametrize('phase, rate', [
    ('decode', 'tokens_per_second'), ('prefill', 'prefill_tokens_per_second'),
])
def test_new_sse_counter_observation_with_the_same_value_is_real_zero_progress(phase, rate):
    metrics = RequestMetrics(True, backend='ninfer')
    observe_sse_count(metrics, phase, 140)
    window, _ = generation_window(None, metrics.token, metrics.live_generation(None), 10)
    observe_sse_count(metrics, phase, 140)
    _, view = generation_window(window, metrics.token, metrics.live_generation(None), 11)
    assert view[rate] == 0
