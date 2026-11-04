"""Gateway request metadata and live token-counter samples; no conversation storage."""
from __future__ import annotations

import math
import time
import uuid


USAGE_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'ttft_ms', 'generation_ms', 'decode_tps')
TIMING_FIELDS = ('cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second',
                 'predicted_n', 'predicted_ms', 'predicted_per_token_ms', 'predicted_per_second')
SNAPSHOT_FIELDS = ('request_key', 'phase', 'prompt_tokens', 'generated_tokens', 'elapsed_s',
                   'average_tokens_per_second', 'average_basis', 'context_tokens', 'context_basis', 'first_output_s', 'request_elapsed_s',
                   'prefill_tokens_per_second', 'prefill_window_s', 'average_prefill_tokens_per_second',
                   'prefill_average_basis', 'prefill_processed_tokens', 'prefill_total_tokens', 'prefill_cached_tokens')
DECODE_PHASES = {'generating', 'thinking', 'output'}
GENERATION_WINDOW_S = 3.0

# Store classifications, never an upstream message: error text can echo prompts,
# tool arguments or credentials. Unknown opaque codes are not safe diagnostics.
UPSTREAM_ERROR_REASONS = {
    'context_length_exceeded': '请求超出后端上下文限制',
    'context_size_exceeded': '请求超出后端上下文限制',
    'invalid_request_error': '后端拒绝请求参数',
    'model_not_found': '后端未找到模型',
    'rate_limit_exceeded': '后端请求限流',
    'rate_limit_error': '后端请求限流',
    'server_error': '后端内部错误',
    'internal_server_error': '后端内部错误',
    'ConnectError': '无法连接模型后端',
    'ConnectTimeout': '连接模型后端超时',
    'ReadError': '后端流读取中断',
    'StreamClosed': '后端响应流已关闭',
    'ReadTimeout': '等待后端响应超时',
    'WriteError': '向后端发送请求失败',
    'WriteTimeout': '向后端发送请求超时',
    'PoolTimeout': '等待后端连接超时',
    'RemoteProtocolError': '后端响应协议异常',
    'LocalProtocolError': '请求协议异常',
    'HTTPError': '后端连接异常',
    'stream_error': '后端流报告错误',
    'response_error': '后端报告错误',
    'internal_error': '处理后端响应失败',
}


def clean_upstream(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    status = value.get('status')
    if type(status) is int and 100 <= status <= 599:
        result['status'] = status
    code = value.get('code')
    if isinstance(code, str) and code in UPSTREAM_ERROR_REASONS:
        result.update(code=code, reason=UPSTREAM_ERROR_REASONS[code])
    elif type(code) is int and 100 <= code <= 599:
        result.update(code=code)
    if 'reason' not in result and result.get('status', 0) >= 400:
        result['reason'] = f"后端返回 HTTP {result['status']}"
    elif 'reason' not in result and 'code' in result:
        result['reason'] = f"后端报告错误（{result['code']}）"
    return result


def _numbers(source, fields):
    if not isinstance(source, dict):
        return {}
    return {key: source[key] for key in fields
            if type(source.get(key)) in (int, float) and math.isfinite(source[key]) and source[key] >= 0}


def clean_measurements(value):
    """Native matching phase pairs; never reconstruct a pair from sticky fields."""
    result = {}
    for phase in ('decode', 'prefill'):
        raw = value.get(phase, {}) if isinstance(value, dict) else {}
        numbers = _numbers(raw, ('tokens', 'seconds'))
        tokens, seconds = numbers.get('tokens'), numbers.get('seconds')
        basis = 'backend_' + phase
        paired = (isinstance(raw, dict) and raw.get('reported') == 1 and raw.get('basis') == basis
                  and tokens is not None and seconds is not None and (seconds > 0 or tokens == 0))
        result[phase] = {'tokens': tokens, 'seconds': seconds if paired else None,
                         'value': tokens / seconds if paired and seconds > 0 else None,
                         'basis': basis if paired else None, 'reported': int(paired),
                         'coverage': ('partial' if raw.get('coverage') == 'partial' else 'complete') if paired else 'unknown'}
    return result


def clean_counters(value):
    numbers = _numbers(value, ('input', 'output', 'cached'))
    return {field: numbers.get(field) for field in ('input', 'output', 'cached')}


class RequestMetrics:
    def __init__(self, streaming, *, backend=None, single_choice=True, client_request_id=None, client_call_id=None, source=None):
        self.token = uuid.uuid4().hex
        self.client_request_id = client_request_id
        self.client_call_id = client_call_id
        self.source = source or ('internal' if client_request_id else 'external')
        self.streaming = bool(streaming)
        self.backend, self.single_choice = backend, single_choice
        self.started = time.monotonic()
        self.started_at = time.time()
        self.first_output_s = None
        self.usage = {}
        self.timings = {}
        self.prompt_progress = {}
        self.prefill = {}
        self.measurements = clean_measurements({})
        self.counters = clean_counters({})
        self._total_sources = {field: None for field in self.counters}
        self._full_input = False
        self._phase_means = {'decode': None, 'prefill': None}
        self._phase_rate_fields = {'decode': set(), 'prefill': set()}
        self._phase_counts = {'decode': None, 'prefill': None}
        self._phase_observations = {'decode': 0, 'prefill': 0}
        self._counter_sources = {'decode': None, 'prefill': None}
        self._prefill_cached = None
        self.phase = 'waiting'
        self.result = None
        self.upstream = {}

    def observe_upstream(self, status=None, error=None, *, cause=None):
        value = dict(self.upstream)
        if status is not None:
            value['status'] = status
        if cause is not None:
            value['code'] = cause if cause in UPSTREAM_ERROR_REASONS else 'HTTPError'
        if isinstance(error, dict):
            details = error.get('error', error)
            if isinstance(details, dict):
                code = details.get('code')
                if ((isinstance(code, str) and code in UPSTREAM_ERROR_REASONS)
                        or (type(code) is int and 100 <= code <= 599)):
                    value['code'] = code
                elif isinstance(details.get('type'), str) and details['type'] in UPSTREAM_ERROR_REASONS:
                    value['code'] = details['type']
        self.upstream = clean_upstream(value)

    def observe(self, chunk):
        if not isinstance(chunk, dict):
            return
        fresh_usage = _numbers(chunk.get('usage'), USAGE_FIELDS)
        self.usage.update(fresh_usage)
        usage = chunk.get('usage')
        if isinstance(usage, dict):
            for field, keys in (('prompt_tokens_details', ('cached_tokens',)), ('completion_tokens_details', ('reasoning_tokens',))):
                details = _numbers(usage.get(field), keys)
                if details:
                    self.usage[field] = {**self.usage.get(field, {}), **details}
        fresh_timings = _numbers(chunk.get('timings'), TIMING_FIELDS)
        self.timings.update(fresh_timings)
        progress = {}
        if self.backend in {'ninfer', 'llama_cpp', 'strata'}:
            progress = _numbers(chunk.get('prompt_progress'), ('total', 'processed', 'cache'))
            self.prompt_progress.update(progress)
            if 'processed' in progress and self.phase not in DECODE_PHASES:
                self.phase = 'prefill'
        cached_usage = _numbers(usage.get('prompt_tokens_details'), ('cached_tokens',)) if isinstance(usage, dict) else {}
        if 'cache_n' in fresh_timings:
            self._set_counter('cached', fresh_timings['cache_n'], 'SSE timings.cache_n')
        elif 'cached_tokens' in cached_usage:
            self._set_counter('cached', cached_usage['cached_tokens'], 'SSE usage.prompt_tokens_details.cached_tokens')
        elif 'cache' in progress and self.phase not in DECODE_PHASES:
            self._set_counter('cached', progress['cache'], 'SSE prompt_progress.cache')
        self._prefill_cached = self.counters['cached']
        native_decode = fresh_timings.get('predicted_n') if self.single_choice else None
        decode_count = native_decode if native_decode is not None else fresh_usage.get('completion_tokens')
        decode_source = 'SSE timings.predicted_n' if native_decode is not None else 'SSE usage.completion_tokens'
        if decode_count is not None:
            self._set_counter('output', decode_count, decode_source)
        self._observe_phase('decode', decode_count, fresh_timings.get('predicted_ms'),
                            fresh_timings.get('predicted_per_second'), decode_source, fresh_timings)
        processed, cached = progress.get('processed'), self._prefill_cached
        prefill_count = fresh_timings.get('prompt_n',
            processed - cached if processed is not None and cached is not None and processed >= cached
            and self.phase not in DECODE_PHASES else None)
        prefill_source = 'SSE timings.prompt_n' if 'prompt_n' in fresh_timings else 'SSE prompt_progress.processed - cache'
        self._observe_phase('prefill', prefill_count, fresh_timings.get('prompt_ms'),
                            fresh_timings.get('prompt_per_second'), prefill_source, fresh_timings)
        if 'prompt_tokens' in fresh_usage:
            self._set_counter('input', fresh_usage['prompt_tokens'], 'SSE usage.prompt_tokens')
            self._full_input = True
        elif 'total' in progress:
            self._set_counter('input', progress['total'], 'SSE prompt_progress.total')
            self._full_input = True
        elif not self._full_input and self._prefill_cached is not None:
            work = prefill_count if prefill_count is not None else self._phase_counts['prefill']
            if work is not None:
                self._set_counter('input', work + self._prefill_cached, 'SSE computed + cached')
        if self.backend in {'ninfer', 'llama_cpp'} and self.phase == 'prefill' and self.counters['output'] is None:
            # This native phase precedes decode. It establishes zero generated
            # tokens, not a positive counter sample or a measurable decode time.
            self._set_counter('output', 0, 'SSE prompt_progress')
        self.prefill.update(self._native_prefill())
        if self.usage.get('completion_tokens', 0) > 0 or self.timings.get('predicted_n', 0) > 0:
            if self.phase not in DECODE_PHASES:
                self.phase = 'generating'
        if self.streaming:
            choices = chunk.get('choices')
            if not isinstance(choices, list):
                return
            for choice in choices:
                delta = choice.get('delta') if isinstance(choice, dict) else None
                if not isinstance(delta, dict):
                    continue
                if delta.get('content'):
                    self.phase = 'output'
                elif delta.get('reasoning_content') or delta.get('reasoning'):
                    self.phase = 'thinking'
                elif delta.get('tool_calls') or delta.get('function_call'):
                    self.phase = 'generating'
                else:
                    continue
                if self._total_sources['output'] == 'SSE prompt_progress':
                    # Prefill establishes zero only until output starts. It is
                    # not an SSE decode measurement: let an attributable backend
                    # counter take over, or leave the count unknown without one.
                    self._set_counter('output', None, None)
                if self.first_output_s is None:
                    self.first_output_s = max(0, time.monotonic() - self.started)

    def _set_counter(self, field, value, source):
        self.counters[field], self._total_sources[field] = value, source

    def _observe_phase(self, phase, count, milliseconds, rate, source, native_fields=None):
        if not self.single_choice:
            return
        prefix = 'predicted' if phase == 'decode' else 'prompt'
        native_rates = _numbers(native_fields, (prefix + '_per_second', prefix + '_per_token_ms'))
        token_ms = native_rates.get(prefix + '_per_token_ms')
        if token_ms == 0:
            native_rates.pop(prefix + '_per_token_ms')  # Zero cannot define a finite mean.
        elif rate is None and token_ms is not None:
            rate = 1000 / token_ms
        if count is None and milliseconds is None and rate is None:
            return
        if count is not None:
            self._phase_observations[phase] += 1
        previous = self.measurements[phase]
        seconds = (milliseconds / 1000 if count is not None and milliseconds is not None
                   and (milliseconds > 0 or count == 0) else
                   count / rate if count is not None and rate is not None and rate > 0 else None)
        if (seconds is None and (count is None or count == self._phase_counts[phase])
                and (milliseconds is None or (previous['reported'] and milliseconds / 1000 == previous['seconds']))
                and (rate is None or rate == self._phase_means[phase])):
            if rate is not None:
                self._phase_rate_fields[phase] = set(native_rates)
            return
        if count is not None:
            self._phase_counts[phase], self._counter_sources[phase] = count, source
        self._phase_rate_fields[phase] = set(native_rates)
        self._phase_means[phase] = rate if rate is not None else count / seconds if seconds else None
        if phase == 'prefill' and self._phase_means[phase] is None:
            self.prefill.pop('average_prefill_tokens_per_second', None)
            self.prefill.pop('prefill_average_basis', None)
        if seconds is not None:
            self.measurements[phase] = {'tokens': count, 'seconds': seconds,
                'value': count / seconds if seconds > 0 else None, 'basis': 'backend_' + phase,
                'reported': 1, 'coverage': 'complete'}
        elif (previous['reported'] and (count is None or count >= previous['tokens'])
              and (milliseconds is None or milliseconds / 1000 >= previous['seconds'])):
            # A known prefix is still real work on cancellation/failure. Its
            # unmeasured tail must not inherit the earlier phase duration/rate.
            self.measurements[phase] = {**previous, 'coverage': 'partial'}
        else:
            self.measurements[phase] = {'tokens': self._phase_counts[phase], 'seconds': None,
                'value': None, 'basis': None, 'reported': 0, 'coverage': 'unknown'}

    def _native_prefill(self):
        if self.backend not in {'ninfer', 'llama_cpp', 'strata'}:
            return {}
        progress = self.prompt_progress
        prefill = {target: progress[key] for key, target in
                   (('total', 'prefill_total_tokens'), ('cache', 'prefill_cached_tokens')) if key in progress}
        # All three protocols report a prompt position including the cached
        # prefix. Missing cache does not establish how much work was computed.
        if self._phase_counts['prefill'] is not None:
            prefill['prefill_processed_tokens'] = self._phase_counts['prefill']
        if self._prefill_cached is not None:
            prefill['prefill_cached_tokens'] = self._prefill_cached
        mean = self._phase_means['prefill']
        if mean is not None:
            prefill.update(average_prefill_tokens_per_second=mean, prefill_average_basis='backend_prefill')
        return prefill

    def live_generation(self, backend_live=None):
        """Request identity with exact SSE or uniquely attributable backend counters.

        Unlike model-global counters, these remain attributable when the backend
        has other work. A multi-choice stream's usage may describe each choice,
        so do not interpret it as a single monotonically increasing counter.
        """
        base = backend_live or {'backend_running': None, 'backend_queued': None}
        attributable = base.get('scope') == 'request' or (base.get('backend_running') == 1 and base.get('backend_queued') == 0)
        raw_overview = base.get('_overview', {})
        overview = dict(raw_overview) if attributable else {
            key: raw_overview[key] for key in ('context_capacity', 'context_capacity_basis') if key in raw_overview}
        decode_mean = raw_overview.get('average_basis') in (None, 'backend_decode')
        if not decode_mean:
            overview.pop('average_tokens_per_second', None)
            overview.pop('average_basis', None)
        fresh_counters, fresh_means = set(), set()
        if attributable:
            # A native backend sample is only a fallback until request-scoped
            # SSE establishes this counter. Never let an older awaited poll
            # overwrite the gateway's newer observation.
            for field, key in (('input', 'prompt_tokens'), ('output', 'generated_tokens'), ('cached', 'prefill_cached_tokens')):
                number = _numbers(raw_overview, (key,)).get(key)
                if field == 'output' and number is None:
                    number = _numbers(base, ('_generated_tokens',)).get('_generated_tokens')
                if number is not None:
                    fresh_counters.add(field)
                if (number is not None and (field != 'output' or self.single_choice)
                        and not (self._total_sources[field] or '').startswith('SSE ')):
                    self._set_counter(field, number, base.get('source'))
                    if field == 'input':
                        self._full_input = True
                    elif field == 'cached':
                        self._prefill_cached = number
            # Preserve existing native /metrics measurements at the established
            # poll cadence. Both values must belong to this one backend sample;
            # SSE counters, when present, remain authoritative for their phase.
            for phase, count_key, mean_key in (
                    ('decode', 'generated_tokens', 'average_tokens_per_second'),
                    ('prefill', 'prefill_processed_tokens', 'average_prefill_tokens_per_second')):
                numbers = _numbers(raw_overview, (count_key, mean_key))
                if count_key in numbers and mean_key in numbers and (phase != 'decode' or decode_mean):
                    fresh_means.add(phase)
                    if not (self._counter_sources[phase] or '').startswith('SSE '):
                        self._observe_phase(phase, numbers[count_key], None, numbers[mean_key], base.get('source'))
            self.prefill.update({key: value for key, value in overview.items()
                                 if key.startswith('prefill_') or key == 'average_prefill_tokens_per_second'
                                 if value is not None})
            backend_phase = overview.get('phase')
            if self.phase not in DECODE_PHASES and not (self.phase == 'prefill' and self.prompt_progress):
                if backend_phase in DECODE_PHASES or (base.get('_generated_tokens') or 0) > 0:
                    self.phase = 'generating'
                elif backend_phase in {'prefill', 'queued'}:
                    self.phase = backend_phase
        # A poll may have started before a newer SSE frame. Never let its
        # /slots counters overwrite this request's exact progress or timings.
        self.prefill.update(self._native_prefill())
        # Poll counters remain in request accounting, but only a fresh backend
        # sample can make them live again. SSE observations are independently
        # attributable even while the backend metrics endpoint is unavailable.
        live_counters = {field: value if field in fresh_counters or (self._total_sources[field] or '').startswith('SSE ')
                         else None for field, value in self.counters.items()}
        exact = self.streaming and self.single_choice
        # SparkInfer 0.6.34 sends usage only after generation, not a live counter.
        generated = live_counters['output'] if exact and self.backend != 'sparkinfer' else None
        prompt = live_counters['input'] if exact else None
        source = self._total_sources['output'] if generated is not None else None
        mean = None
        if (exact and self.backend in {'ninfer', 'llama_cpp', 'strata'}
                and ('decode' in fresh_means or (self._counter_sources['decode'] or '').startswith('SSE '))):
            mean = self._phase_means['decode']
        if prompt is not None:
            overview['prompt_tokens'] = prompt
        if source is not None:
            overview.update(prompt_tokens=prompt, generated_tokens=generated,
                            context_tokens=prompt + generated if prompt is not None and generated is not None else None,
                            context_basis='输入 + 已生成', average_tokens_per_second=mean)
            slot_id = None
        elif attributable:
            generated, source, slot_id = base.get('_generated_tokens'), base.get('source'), base.get('_slot_id')
        else:
            generated, source, slot_id = None, '统一 API 请求', None
        overview.update(self.prefill)
        overview['prefill_cached_tokens'] = live_counters['cached']
        overview['phase'] = self.phase
        fresh_prefill = attributable and 'prefill_processed_tokens' in _numbers(raw_overview, ('prefill_processed_tokens',))
        prefill_source = base.get('_prefill_source') if fresh_prefill else None
        if exact and self.backend in {'ninfer', 'llama_cpp', 'strata'}:
            if self._counter_sources['prefill'] and ((self._counter_sources['prefill'] or '').startswith('SSE ') or fresh_prefill):
                prefill_source = self._counter_sources['prefill']
        if not self.single_choice:
            generated, source, slot_id, prefill_source = None, '统一 API 请求', None, None
            overview.update(prompt_tokens=self.counters['input'], generated_tokens=self.counters['output'], context_tokens=None,
                            average_tokens_per_second=None, average_basis=None,
                            average_prefill_tokens_per_second=None, prefill_average_basis=None,
                            prefill_tokens_per_second=None, prefill_window_s=None, prefill_processed_tokens=None)
        return {**base, 'scope': 'request', 'source': source or '统一 API 请求',
                '_generated_tokens': generated, '_elapsed_s': max(0, time.monotonic()-self.started),
                '_slot_id': slot_id, '_prefill_source': prefill_source,
                '_decode_observation': self._phase_observations['decode'] if (source or '').startswith('SSE ') else None,
                '_prefill_observation': self._phase_observations['prefill'] if (prefill_source or '').startswith('SSE ') else None,
                '_overview': overview}

    def stream_sample(self, chunk):
        """Use only counters actually carried by this frame, not retained older ones."""
        live = self.live_generation()
        chunk = chunk if isinstance(chunk, dict) else {}
        usage = _numbers(chunk.get('usage'), ('completion_tokens',))
        timings = _numbers(chunk.get('timings'), ('predicted_n', 'prompt_n'))
        progress = _numbers(chunk.get('prompt_progress'), ('processed',))
        decode = ((live['source'] == 'SSE timings.predicted_n' and 'predicted_n' in timings)
                  or (live['source'] == 'SSE usage.completion_tokens' and 'completion_tokens' in usage))
        prefill = (live['_overview'].get('phase') == 'prefill'
                   and ((live.get('_prefill_source') == 'SSE timings.prompt_n' and 'prompt_n' in timings)
                        or (live.get('_prefill_source') == 'SSE prompt_progress.processed - cache' and 'processed' in progress)))
        if not decode:
            live['_generated_tokens'] = None
        if not prefill:
            live['_prefill_source'] = None
        return live, decode or prefill

    def snapshot(self, status='running'):
        # Native phase fields can describe one choice of an n>1 response. Do
        # not expose them as a request mean later through a raw-field fallback.
        timings = dict(self.timings) if self.single_choice else _numbers(self.timings, ('cache_n',))
        prefill = dict(self.prefill) if self.single_choice else _numbers(self.prefill, ('prefill_total_tokens', 'prefill_cached_tokens'))
        if self.single_choice:
            for phase, prefix in (('decode', 'predicted'), ('prefill', 'prompt')):
                # A newer N/ms pair or unmatched counter supersedes an earlier
                # explicit mean. Do not resurrect it after current live clears.
                for field in (prefix + '_per_second', prefix + '_per_token_ms'):
                    if field not in self._phase_rate_fields[phase]:
                        timings.pop(field, None)
        return {'status': status, 'started_at': self.started_at,
                **({'ended_at': time.time()} if status != 'running' else {}),
                'request_key': self.token, 'client_request_id': self.client_request_id, 'client_call_id': self.client_call_id,
                'duration_s': max(0, time.monotonic() - self.started),
                'streaming': self.streaming, 'first_output_s': self.first_output_s,
                'usage': dict(self.usage), 'timings': timings, 'prefill': prefill,
                'measurements': {phase: dict(value) for phase, value in self.measurements.items()},
                'counters': dict(self.counters),
                **({'upstream': dict(self.upstream)} if self.upstream else {})}

    def finish(self, status):
        if self.result is None:
            self.result = self.snapshot(status)
        return self.result


def _counter_window(previous, request_token, source, slot, count, elapsed, sampled_at, observation=None):
    """Sample one cumulative counter without interpolating or using request age."""
    current = {'token': request_token, 'slot': slot, 'source': source,
               'count': count, 'time': sampled_at, 'elapsed': elapsed, 'observation': observation,
               'start_count': count, 'start_time': sampled_at, 'samples': [(sampled_at, count)]}
    same = previous and previous.get('count') is not None and all(previous[key] == current[key] for key in ('token', 'slot', 'source'))
    reset = same and (count < previous['count'] or sampled_at <= previous['time'] or
                     (elapsed is not None and previous['elapsed'] is not None and elapsed < previous['elapsed']))
    if not same or reset:
        return current, None, None, None, '等待同一请求的两次连续 token 采样'
    if observation is not None and previous.get('observation') == observation:
        # Polling a retained SSE counter is not another token observation.
        # Preserve its accounting/mean origin, but require two fresh samples.
        return {**previous, 'time': sampled_at, 'elapsed': elapsed, 'samples': []}, None, None, None, '等待新的 token 计数观测'
    current.update(start_count=previous['start_count'], start_time=previous['start_time'])
    mean = (count - current['start_count']) / (sampled_at - current['start_time'])
    mean = mean if math.isfinite(mean) else None
    if sampled_at - previous['time'] > GENERATION_WINDOW_S or not previous['samples']:
        return current, None, None, mean, '采样中断，等待同一请求的连续 token 采样'
    samples = previous['samples'] + current['samples']
    cutoff = sampled_at - GENERATION_WINDOW_S
    # Use the real endpoint nearest the cutoff, retaining its neighbour while
    # locating the boundary. Never interpolate token counts or pad elapsed time.
    while len(samples) > 1 and samples[1][0] <= cutoff:
        samples.pop(0)
    if len(samples) > 2 and cutoff - samples[0][0] > samples[1][0] - cutoff:
        samples.pop(0)
    current['samples'] = samples
    span = sampled_at - samples[0][0]
    speed = (count - samples[0][1]) / span
    if not math.isfinite(speed):
        return current, None, None, mean, '当前 token 采样无法计算有限速度'
    return current, speed, span, mean, None


def generation_window(previous, request_token, live, sampled_at):
    """Independent prefill/decode windows for one attributable request."""
    result = {key: value for key, value in live.items() if not key.startswith('_')}
    result.update(tokens_per_second=None, window_s=None, available_for_chat=False)
    result.update({key: None for key in SNAPSHOT_FIELDS})
    overview = live.get('_overview', {})
    # Capacity belongs to the loaded engine, not the lifetime of a request.
    result['context_capacity'] = _numbers(overview, ('context_capacity',)).get('context_capacity')
    result['context_capacity_basis'] = overview.get('context_capacity_basis')
    if request_token is None:
        result['reason'] = '统一 API 当前没有唯一的在途请求，模型速度无法归属此对话'
        return None, result
    if live.get('scope') != 'request' and (live.get('backend_running') != 1 or live.get('backend_queued') != 0):
        result['reason'] = '后端存在并发、排队或缺少请求数，模型速度无法归属此对话'
        return None, result
    result.update({key: value for key, value in overview.items() if key in SNAPSHOT_FIELDS})
    if _numbers(result, ('average_tokens_per_second',)):
        result['average_basis'] = 'backend_decode'
    if _numbers(result, ('average_prefill_tokens_per_second',)):
        result['prefill_average_basis'] = 'backend_prefill'
    result.update(request_key=request_token, elapsed_s=_numbers(live, ('_elapsed_s',)).get('_elapsed_s'))
    previous = previous if previous and previous['token'] == request_token else None
    if previous and previous.get('phase') in DECODE_PHASES and result.get('phase') not in DECODE_PHASES:
        result['phase'] = previous['phase']
    numbers = _numbers(live, ('_generated_tokens', '_elapsed_s'))
    count, elapsed = numbers.get('_generated_tokens'), numbers.get('_elapsed_s')
    current = None
    if count is not None and count > 0:
        current, speed, span, mean, reason = _counter_window(previous, request_token, live['source'], live.get('_slot_id'), count, elapsed, sampled_at,
            live.get('_decode_observation'))
        result.update(tokens_per_second=speed, window_s=span, available_for_chat=speed is not None, reason=reason)
        if not _numbers(result, ('average_tokens_per_second',)) and mean is not None:
            result.update(average_tokens_per_second=mean, average_basis='sampled_decode')
    else:
        result['reason'] = '等待模型开始生成 token' if count == 0 else '后端未提供当前请求的实时 token 计数'
    prefill = None
    processed = _numbers(overview, ('prefill_processed_tokens',)).get('prefill_processed_tokens')
    if result.get('phase') == 'prefill' and processed is not None and live.get('_prefill_source'):
        prefill, speed, span, mean, _ = _counter_window(previous.get('prefill') if previous else None,
            request_token, live['_prefill_source'], live.get('_slot_id'), processed, elapsed, sampled_at,
            live.get('_prefill_observation'))
        result.update(prefill_tokens_per_second=speed, prefill_window_s=span)
        if not _numbers(result, ('average_prefill_tokens_per_second',)) and mean is not None:
            result.update(average_prefill_tokens_per_second=mean, prefill_average_basis='sampled_prefill')
    if result['average_prefill_tokens_per_second'] is None and previous and (prefill is None or not prefill['samples']):
        result.update(average_prefill_tokens_per_second=previous.get('prefill_average'),
                      prefill_average_basis=previous.get('prefill_average_basis'))
    if prefill is None and result.get('phase') == 'prefill' and not live.get('_prefill_source') and previous and previous.get('prefill'):
        prefill = {**previous['prefill'], 'samples': []}
    if current is None and count is None and previous and previous.get('count') is not None:
        current = {**previous, 'time': sampled_at, 'elapsed': elapsed, 'samples': []}
    if current is None:
        current = {'token': request_token, 'source': live['source'], 'slot': live.get('_slot_id'),
                   'count': None, 'time': sampled_at, 'elapsed': elapsed, 'samples': []}
    current.update(phase=result['phase'], prefill=prefill,
                   prefill_average=result['average_prefill_tokens_per_second'], prefill_average_basis=result['prefill_average_basis'])
    return current, result


def generation_metrics_group(live):
    description = live['source'] + f'；约 {GENERATION_WINDOW_S:g} 秒内累计 token 增量 / 实际时间'
    if live.get('reason'):
        description += '；' + live['reason']
    return {'id': 'model_generation', 'title': '当前实时生成', 'metrics': [
        {'id': 'sampled_tok_s', 'label': '当前解码速度（滚动）', 'value': live['tokens_per_second'], 'unit': 'token/s', 'kind': 'gauge', 'description': description},
        {'id': 'sampled_window_s', 'label': '实际滚动窗口', 'value': live['window_s'], 'unit': 's', 'kind': 'gauge', 'description': f'初始不足 {GENERATION_WINDOW_S:g} 秒时使用已有时间；保留边界采样，窗口随采样间隔略有变化'},
    ]}


def request_metrics_group(record):
    usage, timings = record['usage'], record['timings']
    prefill = record.get('prefill', {})
    speed, speed_label = None, '生成速度'
    rate_basis = None
    speed_description = '后端未返回可用的 usage/timings；不按字符或分块估算 token'
    measured = clean_measurements(record['measurements'])['decode'] if isinstance(record.get('measurements'), dict) else None
    counters = clean_counters(record['counters']) if isinstance(record.get('counters'), dict) else None
    output = counters['output'] if counters is not None else usage.get('completion_tokens')
    if record['status'] != 'completed':
        speed_description = '请求未成功完整结束，不将已返回的部分用量换算为完整请求速度'
    elif 'predicted_per_second' in timings:
        speed = timings['predicted_per_second']
        speed_label = '后端解码速度'
        rate_basis = 'decode'
        speed_description = '来源响应 timings.predicted_per_second；后端报告的解码速度'
    elif timings.get('predicted_per_token_ms', 0) > 0:
        speed = 1000 / timings['predicted_per_token_ms']
        speed_label = '后端解码速度'
        rate_basis = 'decode'
        speed_description = '来源响应 timings.predicted_per_token_ms；后端报告的解码速度'
    elif measured is not None and measured['reported']:
        speed = measured['value']
        speed_label = '后端解码速度'
        rate_basis = 'decode'
        speed_description = '来源原生 token 与对应解码耗时' + ('；已报告前缀' if measured['coverage'] == 'partial' else '')
    elif measured is None and 'predicted_n' in timings and timings.get('predicted_ms', 0) > 0:
        speed = timings['predicted_n'] * 1000 / timings['predicted_ms']
        speed_label = '后端解码速度'
        rate_basis = 'decode'
        speed_description = '来源响应 timings.predicted_n / predicted_ms；后端报告的解码耗时'
    elif 'decode_tps' in usage:
        speed = usage['decode_tps']
        speed_label = '后端平均速度'
        rate_basis = 'end_to_end'
        speed_description = '来源响应 usage.decode_tps；SparkInfer 0.6.34 使用输出 token / 引擎完整生成耗时（含首 token 等待）'
    elif record['status'] == 'completed' and output is not None and record['duration_s'] > 0:
        speed = output / record['duration_s']
        speed_label = '端到端平均速度'
        rate_basis = 'end_to_end'
        speed_description = '来源响应 usage.completion_tokens / 完整请求耗时；含排队、预填充与传输'
    states = {'completed': '已完成', 'failed': '失败', 'cancelled': '已中止', 'incomplete': '未完整结束'}
    metrics = [
        {'id': 'status', 'label': '请求状态', 'value': states[record['status']], 'kind': 'text'},
        {'id': 'ended_at', 'label': '结束时间', 'value': record['ended_at'], 'kind': 'timestamp'},
        {'id': 'duration_s', 'label': '请求耗时', 'value': record['duration_s'], 'unit': 's', 'kind': 'gauge'},
        {'id': 'prompt_tokens', 'label': '输入 token', 'value': counters['input'] if counters is not None else usage.get('prompt_tokens'), 'unit': 'token', 'kind': 'gauge', 'description': '来源请求最新已观测计数' if counters is not None else '来源响应 usage.prompt_tokens'},
        {'id': 'completion_tokens', 'label': '输出 token', 'value': output if counters is not None else usage.get('completion_tokens',timings.get('predicted_n')), 'unit': 'token', 'kind': 'gauge', 'description': '来源请求最新已观测计数' if counters is not None else '来源响应 usage.completion_tokens；包含后端计入用量的思考 token' if 'completion_tokens' in usage else '来源响应 timings.predicted_n；后端报告的解码 token'},
        {'id': 'output_speed', 'label': speed_label, 'value': speed, 'unit': 'token/s', 'kind': 'gauge', 'description': speed_description, 'rate_basis': rate_basis,
         **({'coverage': measured['coverage']} if measured is not None and rate_basis == 'decode' else {})},
    ]
    if record['streaming']:
        metrics.insert(3, {'id': 'first_output_s', 'label': '首输出等待', 'value': record['first_output_s'], 'unit': 's', 'kind': 'gauge', 'description': '从统一 API 转发开始到收到首个正文、思考或工具输出；不将 role/usage 块视为输出'})
    for field, label, description in (
            ('ttft_ms', '后端首 token 等待', '引擎提交到首 token；多分支取最小值'),
            ('generation_ms', '后端生成耗时', '引擎提交到结束，含首 token 等待；多分支取最大值')):
        if field in usage:
            metrics.append({'id': 'backend_' + field, 'label': label, 'value': usage[field], 'unit': 'ms',
                            'kind': 'gauge', 'basis': 'usage.' + field, 'description': description})
    if 'decode_tps' in usage:
        metrics.append({'id': 'backend_average_tps', 'label': '后端平均速度',
                        'value': usage['decode_tps'] if record['status'] == 'completed' else None,
                        'unit': 'token/s', 'kind': 'gauge', 'rate_basis': 'end_to_end', 'basis': 'usage.decode_tps',
                        'description': '输出 token / 引擎完整生成耗时（含首 token 等待）'})
    if prefill.get('average_prefill_tokens_per_second') is not None:
        metrics.append({'id': 'prefill_speed', 'label': '平均预填充速度', 'value': prefill['average_prefill_tokens_per_second'],
                        'unit': 'token/s', 'kind': 'gauge', 'rate_basis': 'prefill'})
    for key, label in (('prefill_processed_tokens', '预填充已处理 token'), ('prefill_total_tokens', '预填充目标 token'),
                       ('prefill_cached_tokens', '缓存复用 token')):
        if key in prefill:
            metrics.append({'id': key, 'label': label, 'value': prefill[key], 'unit': 'token', 'kind': 'gauge'})
    return {'id': 'api_request', 'title': '最近一次 API 请求', 'metrics': metrics,
            'request_status': record['status'], 'request_key': record.get('request_key'),
            'client_request_id': record.get('client_request_id'), 'client_call_id': record.get('client_call_id')}
