"""External request metrics and reported tool names; message bodies stay in memory as digests."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
import uuid

from .request_metrics import USAGE_FIELDS, TIMING_FIELDS, SNAPSHOT_FIELDS, _numbers, clean_upstream, clean_measurements, clean_counters

STATES = {'running', 'completed', 'cancelled', 'failed', 'incomplete'}
TURN_STATES = STATES | {'waiting', 'recovered'}
ARCHIVE_ID = 'modloader.request-history'
ARCHIVE_IO_LOCK = threading.RLock()


@contextmanager
def exclusive_log(stream):
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def local_date(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp).date().isoformat()


def validate_date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('日期必须为 YYYY-MM-DD')
    return value
PREFILL_FIELDS = ('prefill_processed_tokens', 'prefill_total_tokens', 'prefill_cached_tokens',
                  'average_prefill_tokens_per_second', 'prefill_tokens_per_second', 'prefill_window_s')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def clean_client_context(value):
    if (isinstance(value, dict) and value.get('source') == 'opencode'
            and isinstance(value.get('session_id'), str) and value['session_id'].strip()):
        return {'source': 'opencode', 'session_id': value['session_id']}
    return None


def _clean_record(value):
    """Validate the archive and copy only identity, metrics and reported tool names."""
    if not isinstance(value, dict):
        raise ValueError('record')
    for key in ('run_id', 'instance_id', 'alias', 'backend', 'request_key'):
        if not isinstance(value.get(key), str):
            raise ValueError(key)
    if type(value.get('revision')) is not int or value['revision'] < 1:
        raise ValueError('revision')
    if value.get('status') not in STATES or value.get('aggregation') not in {'turn', 'request'}:
        raise ValueError('state')
    if value.get('turn_state') is not None and value['turn_state'] not in TURN_STATES:
        raise ValueError('turn_state')
    if value.get('turn_id') is not None and not isinstance(value['turn_id'], str):
        raise ValueError('turn_id')
    if (value['aggregation'] == 'turn') != isinstance(value.get('turn_id'), str):
        raise ValueError('aggregation')
    for key in ('started_at', 'duration_s'):
        if type(value.get(key)) not in (float, int) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError(key)
    if type(value.get('streaming')) is not bool:
        raise ValueError('streaming')
    result = {key: value[key] for key in ('run_id', 'instance_id', 'alias', 'backend', 'request_key',
              'revision', 'status', 'aggregation', 'started_at', 'duration_s', 'streaming')}
    result['date'] = validate_date(value['date']) if 'date' in value else local_date(value['started_at'])
    result.update(turn_id=value.get('turn_id'), turn_state=value.get('turn_state'),
                  labels=[label for label in value.get('labels', []) if label in ('tool_calls',)] if isinstance(value.get('labels'), list) else [])
    names = value.get('tool_names')
    result['tool_names'] = list(dict.fromkeys(name for name in names if isinstance(name, str) and name)) if isinstance(names, list) else []
    result.update(_numbers(value, ('ended_at', 'first_output_s')))
    result.setdefault('first_output_s', None)
    # External calls never adopt internal conversation/call identifiers.
    result.update(client_request_id=None, client_call_id=None)
    context = clean_client_context(value.get('client_context'))
    if context is not None:
        result['client_context'] = context
    upstream = clean_upstream(value.get('upstream'))
    if upstream:
        result['upstream'] = upstream
    result['usage'] = _numbers(value.get('usage'), USAGE_FIELDS)
    usage = value.get('usage', {})
    if isinstance(usage, dict):
        for key, fields in (('prompt_tokens_details', ('cached_tokens',)), ('completion_tokens_details', ('reasoning_tokens',))):
            details = _numbers(usage.get(key), fields)
            if details:
                result['usage'][key] = details
    result['timings'] = _numbers(value.get('timings'), TIMING_FIELDS)
    if isinstance(value.get('measurements'), dict):
        result['measurements'] = clean_measurements(value['measurements'])
    if isinstance(value.get('counters'), dict):
        result['counters'] = clean_counters(value['counters'])
    result['prefill'] = _numbers(value.get('prefill'), PREFILL_FIELDS)
    prefill = value.get('prefill')
    if isinstance(prefill, dict) and prefill.get('prefill_average_basis') in {'backend_prefill', 'sampled_prefill'}:
        result['prefill']['prefill_average_basis'] = prefill['prefill_average_basis']
    live = value.get('live')
    if isinstance(live, dict) and live.get('request_key') == result['request_key']:
        result['live'] = _clean_live(live)
    return result


def _clean_live(live):
    result = _numbers(live, tuple(key for key in SNAPSHOT_FIELDS if key not in ('request_key', 'phase', 'average_basis', 'context_basis', 'prefill_average_basis')) + ('tokens_per_second', 'window_s'))
    result.update(request_key=live['request_key'], scope='request')
    for key, allowed in (('phase', {'waiting', 'queued', 'prefill', 'generating', 'thinking', 'output'}),
                         ('average_basis', {'backend_decode', 'sampled_decode'}),
                         ('prefill_average_basis', {'backend_prefill', 'sampled_prefill'}),
                         ('context_basis', {'输入 + 已生成'})):
        if isinstance(live.get(key), str) and live[key] in allowed:
            result[key] = live[key]
    return result


def _reported(*values):
    return next((value for value in values
                 if type(value) in (int, float) and math.isfinite(value) and value >= 0), None)


def _history_rate(record, phase):
    if isinstance(record.get('measurements'), dict):
        measured = clean_measurements(record['measurements'])[phase]
        return (measured['tokens'], measured['seconds']) if measured['reported'] else None
    timings, usage, prefill, live = (record.get(key) or {} for key in ('timings', 'usage', 'prefill', 'live'))
    decode = phase == 'decode'
    tokens = _reported(timings.get('predicted_n' if decode else 'prompt_n'),
                       usage.get('completion_tokens') if decode else prefill.get('prefill_processed_tokens'))
    milliseconds = _reported(timings.get('predicted_ms' if decode else 'prompt_ms'))
    mean = _reported(timings.get('predicted_per_second' if decode else 'prompt_per_second'),
                     prefill.get('average_prefill_tokens_per_second')
                     if not decode and prefill.get('prefill_average_basis') != 'sampled_prefill' else None)
    if tokens is not None:
        if milliseconds is not None and milliseconds > 0:
            return tokens, milliseconds / 1000
        if tokens == 0 and milliseconds == 0:
            return 0, 0
        if mean is not None and mean > 0:
            return tokens, tokens / mean
    basis = live.get('average_basis' if decode else 'prefill_average_basis')
    if basis == ('backend_decode' if decode else 'backend_prefill'):
        mean = _reported(live.get('average_tokens_per_second' if decode else 'average_prefill_tokens_per_second'))
        tokens = _reported(live.get('generated_tokens' if decode else 'prefill_processed_tokens'))
        if tokens is not None and mean is not None and mean > 0:
            return tokens, tokens / mean
    return None


def _history_engine_rate(record):
    if record['status'] != 'completed':
        return None
    usage = record.get('usage') or {}
    tokens, milliseconds, mean = (_reported(usage.get(key)) for key in ('completion_tokens', 'generation_ms', 'decode_tps'))
    seconds = milliseconds / 1000 if milliseconds is not None else tokens / mean if tokens is not None and mean is not None and mean > 0 else None
    value = mean if mean is not None else tokens / seconds if tokens is not None and seconds is not None and seconds > 0 else None
    return (tokens, seconds) if value is not None and tokens is not None and seconds is not None else None


def _history_values(record):
    usage, timings, prefill, live = (record.get(key) or {} for key in ('usage', 'timings', 'prefill', 'live'))
    cached = _reported((usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                       timings.get('cache_n'), prefill.get('prefill_cached_tokens'), live.get('prefill_cached_tokens'))
    processed = _reported(timings.get('prompt_n'))
    input_tokens = _reported(usage.get('prompt_tokens'),
                            prefill.get('prefill_total_tokens'),
                             processed + cached if processed is not None and cached is not None else None,
                            live.get('prompt_tokens'))
    output = _reported(usage.get('completion_tokens'), timings.get('predicted_n'), live.get('generated_tokens'))
    if isinstance(record.get('counters'), dict):
        counters = clean_counters(record['counters'])
        input_tokens, output, cached = (counters[key] for key in ('input', 'output', 'cached'))
    native_ttft = _reported(usage.get('ttft_ms'))
    return {'input': input_tokens, 'output': output, 'cached': cached,
            'total': input_tokens + output if input_tokens is not None and output is not None else None,
            'decode': _history_rate(record, 'decode'), 'prefill': _history_rate(record, 'prefill'),
            'engine': _history_engine_rate(record),
            'ttft': native_ttft / 1000 if native_ttft is not None else _reported(record.get('first_output_s')),
            'ttft_engine': native_ttft is not None,
            'ttft_client': native_ttft is None and record.get('first_output_basis') == 'client',
            'ttft_unknown': native_ttft is None and record.get('first_output_basis') == 'unknown'}


def request_history_stats(records):
    """Aggregate available request metrics, retaining each metric's coverage."""
    values = [_history_values(record) for record in records]

    def total(field):
        numbers = [value[field] for value in values if value[field] is not None]
        return {'value': sum(numbers) if numbers else None, 'reported': len(numbers)}

    def rate(field):
        pairs = [value[field] for value in values if value[field] is not None]
        tokens, seconds = sum(pair[0] for pair in pairs), sum(pair[1] for pair in pairs)
        partial = sum(value[field] is not None and (record.get('measurements', {}).get(field) or {}).get('coverage') == 'partial'
                      for record, value in zip(records, values))
        return {'value': tokens / seconds if seconds > 0 else None,
                'tokens': tokens, 'seconds': seconds, 'reported': len(pairs),
                **({'partial_reported': partial} if partial else {})}

    cache_pairs = [value for value in values if value['input'] is not None and value['cached'] is not None
                   and value['cached'] <= value['input']]
    cache_input = sum(value['input'] for value in cache_pairs) if cache_pairs else None
    cache_tokens = sum(value['cached'] for value in cache_pairs) if cache_pairs else None
    ttft = total('ttft')
    engine_ttft = sum(value['ttft'] is not None and value['ttft_engine'] for value in values)
    client_ttft = sum(value['ttft'] is not None and value['ttft_client'] for value in values)
    unknown_ttft = sum(value['ttft'] is not None and value['ttft_unknown'] for value in values)
    return {'request_count': len(records), **{field: total(field) for field in ('input', 'output', 'total', 'cached')},
            'cache_pair': {'input': cache_input, 'cached': cache_tokens, 'reported': len(cache_pairs)},
            'cache_ratio': cache_tokens / cache_input if cache_input is not None and cache_input > 0 else None,
            'decode': rate('decode'), 'prefill': rate('prefill'), 'engine': rate('engine'),
            'ttft': {'value': ttft['value'] / ttft['reported'] if ttft['reported'] else None,
                     'reported': ttft['reported'], 'engine_reported': engine_ttft,
                     'gateway_reported': ttft['reported'] - engine_ttft - client_ttft - unknown_ttft,
                     **({'client_reported': client_ttft} if client_ttft else {}),
                     **({'unknown_reported': unknown_ttft} if unknown_ttft else {})},
            'status_counts': {state: sum(record['status'] == state for record in records)
                              for state in ('running', 'completed', 'failed', 'cancelled', 'incomplete')},
            'record_incomplete_count': sum(bool(record.get('record_incomplete')) for record in records)}


def _usage_active(record):
    if '_usage_active' in record:
        return record['_usage_active']
    return record['status'] == 'running'


def request_history_trend(records, as_of):
    """One retained request belongs wholly to the bucket containing its start."""
    if not records:
        return {'start_at': None, 'end_at': None, 'bucket_seconds': 60,
                'next_refresh_at': (math.floor(as_of / 60) + 1) * 60, 'buckets': []}
    earliest = min(record['started_at'] for record in records)
    anchor = as_of if any(_usage_active(record) for record in records) else max(record['started_at'] for record in records)
    span = min(86400, max(60, anchor - earliest))
    seconds = 60 if span < 300 else 300 if span <= 1800 else 900 if span <= 7200 else 1800 if span <= 21600 else 3600
    end = (math.floor(anchor / seconds) + 1) * seconds
    start = max(math.floor(earliest / seconds) * seconds, end - 86400)
    slots = [[] for _ in range(round((end - start) / seconds))]
    for record in records:
        if start <= record['started_at'] < end:
            slots[math.floor((record['started_at'] - start) / seconds)].append(record)
    buckets = []
    for index, slot in enumerate(slots):
        stats = request_history_stats(slot)
        if not slot:
            # A measured empty interval differs from requests with missing counters.
            for field in ('input', 'output', 'total', 'cached'):
                stats[field]['value'] = 0
        buckets.append({'start_at': start + index * seconds, 'end_at': start + (index + 1) * seconds, **stats})
    return {'start_at': start, 'end_at': end, 'bucket_seconds': seconds,
            'next_refresh_at': (math.floor(as_of / seconds) + 1) * seconds, 'buckets': buckets}


def _history_group(records, *, per_request=False):
    # The query has already sorted requests newest first; the same record is
    # returned in items so callers can use the existing exact-detail endpoint.
    representative = records[0]
    started = min(record['started_at'] for record in records)
    status = representative['status'] if per_request else representative.get('turn_state') or representative['status']
    ends = [_reported(record.get('ended_at')) for record in records]
    ended = max(ends) if all(end is not None for end in ends) and status not in {'running', 'waiting'} else None
    elapsed_end = max(record.get('ended_at', record['started_at'] + record['duration_s']) for record in records)
    return {**request_history_stats(records), 'source': 'external',
            **{key: representative[key] for key in ('run_id', 'instance_id', 'turn_id', 'request_key', 'aggregation')},
            **({'client_context': representative['client_context']} if representative.get('client_context') else {}),
            'started_at': started, 'ended_at': ended, 'elapsed_s': max(0, elapsed_end - started), 'status': status}


class ResponseSignature:
    """Hash response text incrementally; hold tool arguments only until terminal normalization."""
    def __init__(self, signatures):
        self.signatures = signatures
        self.choices = {}
        self.valid = True

    def observe(self, chunk):
        if not isinstance(chunk, dict) or not isinstance(chunk.get('choices'), list):
            return
        for ordinal, choice in enumerate(chunk['choices']):
            if not isinstance(choice, dict):
                self.valid = False
                continue
            index = choice.get('index', ordinal)
            if type(index) is not int:
                self.valid = False
                continue
            state = self.choices.setdefault(index, {'fields': {}, 'tools': {}, 'names': {}, 'finish': None})
            message = choice.get('message')
            full = isinstance(message, dict)
            if full:
                state['full'] = True
            delta = message if full else choice.get('delta')
            if isinstance(delta, dict):
                for field in ('content', 'reasoning_content', 'reasoning', 'refusal'):
                    part = delta.get(field)
                    if part is None:
                        continue
                    if field == 'content':
                        part = self.signatures.text_content(part)
                    if not isinstance(part, str):
                        self.valid = False
                        continue
                    if full:
                        state['fields'][field] = self.signatures.hasher()
                    state['fields'].setdefault(field, self.signatures.hasher()).update(part.encode('utf-8'))
                calls = delta.get('tool_calls')
                if calls is not None and not isinstance(calls, list):
                    self.valid = False
                for position, call in enumerate(calls or []):
                    if not isinstance(call, dict):
                        self.valid = False
                        continue
                    tool_index = position if full else call.get('index')
                    if type(tool_index) is not int:
                        self.valid = False
                        continue
                    tool = state['tools'].setdefault(tool_index, {})
                    function = call.get('function') or {}
                    if not isinstance(function, dict):
                        self.valid = False
                        continue
                    for field, part in (('id', call.get('id')), ('name', function.get('name')), ('arguments', function.get('arguments'))):
                        if part is None:
                            continue
                        if not isinstance(part, str):
                            self.valid = False
                            continue
                        if field == 'name':
                            if full:
                                state['names'][tool_index] = []
                            state['names'].setdefault(tool_index, []).append(part)
                        if field == 'arguments':
                            if full:
                                tool[field] = []
                            tool.setdefault(field, []).append(part)
                        else:
                            if full:
                                tool[field] = self.signatures.hasher()
                            tool.setdefault(field, self.signatures.hasher()).update(part.encode('utf-8'))
                    if call.get('type', 'function') != 'function':
                        self.valid = False
                if delta.get('function_call'):
                    self.valid = False  # Legacy protocol has no safely matchable tool result ID.
            if choice.get('finish_reason') is not None:
                state['finish'] = choice['finish_reason']

    def tool_names(self):
        # A cancelled stream can end inside a function-name fragment. Only full
        # messages or choices with a terminal marker establish the reported name.
        names = (''.join(parts) for _, state in sorted(self.choices.items())
                 if state.get('full') or state['finish'] is not None
                 for _, parts in sorted(state['names'].items()))
        return list(dict.fromkeys(name for name in names if name))

    def finish(self):
        all_ids = tuple(tool['id'].hexdigest() for state in self.choices.values() for tool in state['tools'].values() if 'id' in tool)
        if not self.valid or len(self.choices) != 1:
            return None, all_ids, any(state['finish'] == 'tool_calls' for state in self.choices.values())
        state = next(iter(self.choices.values()))
        empty = self.signatures.digest('')
        fields = {key: value.hexdigest() for key, value in state['fields'].items() if value.hexdigest() != empty}
        fields.setdefault('content', empty)
        calls = [{key: self.signatures.arguments(''.join(value)) if key == 'arguments' else value.hexdigest()
                  for key, value in tool.items()} for _, tool in sorted(state['tools'].items())]
        ids = tuple(call.get('id') for call in calls)
        # Missing IDs permit only a complete sequence match; duplicate IDs never link.
        duplicate = len([item for item in ids if item]) != len(set(item for item in ids if item))
        if duplicate:
            return None, ids, state['finish'] == 'tool_calls'
        message = {'role': 'assistant', **fields}
        if calls:
            message['tool_calls'] = calls
        self.body_signature = self.signatures.digest({key: value for key, value in message.items() if key not in ('reasoning_content', 'reasoning')})
        return self.signatures.digest(message), ids, state['finish'] == 'tool_calls'


class Signatures:
    def __init__(self):
        self.key = secrets.token_bytes(32)

    def hasher(self):
        return hashlib.blake2s(key=self.key)

    def digest(self, value):
        digest = self.hasher()
        digest.update((value if isinstance(value, str) else _json(value)).encode('utf-8'))
        return digest.hexdigest()

    def arguments(self, value):
        if isinstance(value, str):
            try:
                return self.digest(_json(json.loads(value)))
            except (ValueError, TypeError):
                pass
        return self.digest(value)

    @staticmethod
    def text_content(content):
        if isinstance(content, list) and all(isinstance(part, dict) and part.get('type') == 'text' and isinstance(part.get('text'), str) for part in content):
            return ''.join(part['text'] for part in content)
        return content

    def message(self, message, *, omit_reasoning=False):
        if not isinstance(message, dict):
            return None
        normalized = {'role': message.get('role'), 'content': self.digest(self.text_content(message.get('content') or ''))}
        for field in ('reasoning_content', 'reasoning', 'refusal', 'name', 'tool_call_id'):
            if message.get(field) and not (omit_reasoning and field in ('reasoning_content', 'reasoning')):
                normalized[field] = self.digest(message[field])
        if message.get('tool_calls'):
            calls = message['tool_calls']
            if not isinstance(calls, list):
                return None
            normalized['tool_calls'] = []
            for call in calls:
                if not isinstance(call, dict) or call.get('type', 'function') != 'function' or not isinstance(call.get('function'), dict):
                    return None
                normalized['tool_calls'].append({key: self.arguments(value) if key == 'arguments' else self.digest(value) for key, value in
                    (('id', call.get('id')), ('name', call['function'].get('name')), ('arguments', call['function'].get('arguments'))) if value is not None})
        if message.get('function_call'):
            normalized['function_call'] = self.digest(message['function_call'])
        return self.digest(normalized)


class HistoryArchive:
    """Serialized daily append log, cached reads, and explicitly selected retention."""
    def __init__(self, directory):
        self.directory = str(directory)
        self.enabled = False
        self.retention_days = 0
        self.error = None
        self.cleanup_error = None
        self.cleanup_requested = False
        self.protected_dates = set()
        self.run_starts = {}
        self.queue = deque()
        self.condition = threading.Condition()
        self.thread = None
        self.writing = False
        self.read_lock = threading.RLock()
        self.file_cache = {}
        self.created_paths = set()

    def status(self):
        with self.condition:
            state = 'error' if self.error else 'saving' if self.queue or self.writing else 'saved' if self.enabled else 'disabled'
            return {'state': state, 'error': self.error, 'cleanup_error': self.cleanup_error, 'directory': self.directory}

    @staticmethod
    def _error_detail(error):
        detail = type(error).__name__
        if isinstance(error, OSError):
            winerror, code = getattr(error, 'winerror', None), error.errno
            if type(winerror) is int:
                import ctypes
                detail += f'，winerror={winerror}：' + ctypes.FormatError(winerror).strip()
            elif type(code) is int:
                detail += f'，errno={code}：' + os.strerror(code)
        elif isinstance(error, ValueError):
            detail += '：保存文件的身份或格式不符'
        return detail

    def configure(self, enabled, directory, header, records, retention_days=0):
        with self.condition:
            changed = str(directory) != self.directory
            activating = enabled and (not self.enabled or changed)
            policy_changed = retention_days != self.retention_days
            self.enabled, self.directory, self.retention_days = enabled, str(directory), retention_days
            self.run_starts[header['id']] = header['started_at']
            if changed:
                self.queue = deque((self.directory, item) for _, item in self.queue)
                self.error = None
            if changed or policy_changed:
                self.cleanup_error = None
            if activating:
                self._enqueue({**header, 'schema': 2, 'application': ARCHIVE_ID, 'date': local_date()})
                for record in records:
                    self._enqueue_record(record)
            self.cleanup_requested = enabled and bool(retention_days)
            self._wake()

    def protect(self, dates):
        with self.condition:
            self.protected_dates = set(dates)

    def _enqueue(self, event):
        self.queue.append((self.directory, _json(event).encode('utf-8') + b'\n'))

    def _enqueue_record(self, record):
        clean = _clean_record(record)
        self._enqueue({'schema': 2, 'application': ARCHIVE_ID, 'type': 'request', 'date': clean['date'],
                       'run_started_at': self.run_starts[clean['run_id']], 'record': clean})

    def enqueue(self, record):
        with self.condition:
            if self.enabled:
                self._enqueue_record(record)
                self.cleanup_requested = bool(self.retention_days)
                self._wake()

    def _wake(self):
        cleanup = self.enabled and self.retention_days and self.cleanup_requested and not self.cleanup_error
        if ((self.queue and not self.error) or cleanup) and (self.thread is None or not self.thread.is_alive()):
            self.thread = threading.Thread(target=self._work, name='api-request-history', daemon=True)
            self.thread.start()
        self.condition.notify_all()

    def retry(self):
        with self.condition:
            self.error = self.cleanup_error = None
            self.cleanup_requested = self.enabled and bool(self.retention_days)
            self._wake()
        return self.status()

    def _append(self, directory, data):
        event = json.loads(data)
        day = validate_date(event['date'])
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / (day + '.jsonl')
        header = {'application': ARCHIVE_ID, 'schema': 2, 'type': 'day', 'date': day}
        with ARCHIVE_IO_LOCK:
            if self._linked(path):
                raise ValueError('linked daily log')
            is_new = not path.exists()
            with path.open('a+b') as stream, exclusive_log(stream):
                if is_new:
                    self.created_paths.add(path)
                stream.seek(0, 2)
                size = stream.tell()
                recover_header = False
                if size:
                    stream.seek(0)
                    raw_header = stream.readline()
                    try:
                        first = json.loads(raw_header)
                    except (ValueError, UnicodeError):
                        first = None
                    if first != header:
                        header_bytes = _json(header).encode('utf-8')
                        if path not in self.created_paths or not header_bytes.startswith(raw_header.rstrip(b'\n')):
                            raise ValueError('unrecognized daily log')
                        # Preserve a header fragment from this writer's failed
                        # creation; append one intact header without overwriting it.
                        recover_header = stream.readline().rstrip(b'\n') != header_bytes
                if not size:
                    stream.seek(0, 2)
                    stream.write(_json(header).encode('utf-8') + b'\n')
                else:
                    stream.seek(-1, 2)
                    if stream.read(1) != b'\n':
                        stream.seek(0, 2)
                        stream.write(b'\n')
                stream.seek(0, 2)
                if recover_header:
                    stream.write(_json(header).encode('utf-8') + b'\n')
                stream.write(data)
                stream.flush()

    def _work(self):
        while True:
            with self.condition:
                entry = self.queue[0] if self.queue and not self.error else None
                cleanup = not self.queue and self.enabled and self.retention_days and self.cleanup_requested and not self.cleanup_error
                if entry is None and not cleanup:
                    self.thread = None
                    self.writing = False
                    self.condition.notify_all()
                    return
                self.writing = True
                if cleanup:
                    self.cleanup_requested = False
            if entry is not None:
                try:
                    self._append(*entry)
                except Exception as error:
                    with self.condition:
                        if self.queue and self.queue[0] != entry:
                            continue
                        self.error = '请求历史写入失败（' + self._error_detail(error) + '），检查保存目录后重试'
                else:
                    with self.condition:
                        if self.queue and self.queue[0] == entry:
                            self.queue.popleft()
            else:
                try:
                    self._cleanup()
                except OSError as error:
                    with self.condition:
                        self.cleanup_error = '过期历史清理失败（' + self._error_detail(error) + '），检查保存目录后重试'
            with self.condition:
                self.writing = False
                self.condition.notify_all()

    def drain(self, timeout=5):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (self.queue or self.writing or (self.enabled and self.retention_days and self.cleanup_requested and not self.cleanup_error)) and not self.error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return not self.queue and not self.error

    @staticmethod
    def _linked(path):
        try:
            info = path.lstat()
            return path.is_symlink() or bool(getattr(info, 'st_file_attributes', 0) & 0x400)
        except FileNotFoundError:
            return False

    def _read_file(self, path):
        runs, issues, owner = {}, [], True
        day_header, legacy_header, schema_seen = None, None, None
        try:
            with path.open('rb') as stream:
                for number, raw in enumerate(stream, 1):
                    try:
                        event = json.loads(raw.decode('utf-8'))
                        schema = event.get('schema') if isinstance(event, dict) else None
                        if schema not in (1, 2) or (schema_seen is not None and schema != schema_seen):
                            raise ValueError('schema')
                        schema_seen = schema
                        if schema == 2:
                            if event.get('application') != ARCHIVE_ID:
                                raise ValueError('application')
                            day = validate_date(event.get('date'))
                            if day != path.stem:
                                owner = False
                            if event.get('type') == 'day':
                                if number != 1 or day_header is not None:
                                    owner = False
                                day_header = day
                                if not raw.endswith(b'\n'):
                                    raise ValueError('trailing line')
                                continue
                        if event.get('type') == 'run':
                            identifier, started = event.get('id'), event.get('started_at')
                            if not isinstance(identifier, str) or type(started) not in (float, int) or not math.isfinite(started):
                                raise ValueError('run')
                            if schema == 1:
                                legacy_header = identifier if legacy_header in (None, identifier) else False
                            run = runs.setdefault(identifier, {'id': identifier, 'started_at': started, 'records': {}})
                            run['started_at'] = started
                        elif event.get('type') == 'request':
                            record = _clean_record(event.get('record'))
                            started = event.get('run_started_at', record['started_at'])
                            if type(started) not in (float, int) or not math.isfinite(started):
                                raise ValueError('run_started_at')
                            if schema == 2 and record['date'] != day:
                                raise ValueError('date')
                            run = runs.setdefault(record['run_id'], {'id': record['run_id'], 'started_at': started, 'records': {}})
                            run['started_at'] = min(run['started_at'], started)
                            old = run['records'].get(record['request_key'])
                            if old is None or record['revision'] > old['revision']:
                                run['records'][record['request_key']] = record
                        else:
                            raise ValueError('event')
                        if not raw.endswith(b'\n'):
                            issues.append(f'{path.name}:{number} 行尾不完整')
                    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, OSError):
                        issues.append(f'{path.name}:{number} 无效历史记录')
        except OSError:
            issues.append(f'{path.name} 无法读取')
        if schema_seen == 2:
            owner = owner and day_header == path.stem
        elif schema_seen == 1:
            try:
                valid_uuid = uuid.UUID(path.stem).hex == path.stem.replace('-', '').lower()
            except (ValueError, AttributeError):
                valid_uuid = False
            owner = owner and valid_uuid and legacy_header == path.stem and set(runs) == {legacy_header}
        else:
            owner = False
        if not owner:
            issues.append(f'{path.name} 身份或文件名不匹配，保留原文件')
        owner = owner and not issues
        for run in runs.values():
            for record in run['records'].values():
                if record['status'] == 'running' or record.get('turn_state') == 'running' or 'ended_at' not in record:
                    record['record_incomplete'] = True
        return runs, issues, owner

    def _files(self, directory, date_from=None, date_to=None):
        files = sorted(directory.glob('*.jsonl'))
        if date_from is None and date_to is None:
            return files
        selected = []
        for path in files:
            try:
                day = validate_date(path.stem)
            except ValueError:
                selected.append(path)  # A legacy run can contain several dates.
            else:
                if (date_from is None or day >= date_from) and (date_to is None or day <= date_to):
                    selected.append(path)
        return selected

    def _cached_file(self, path):
        if self._linked(path):
            return {}, [f'{path.name} 是链接，保留原文件'], False, None
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        cached = self.file_cache.get(path)
        if cached is None or cached[0] != stamp:
            runs, issues, owner = self._read_file(path)
            cached = (stamp, runs, issues, owner)
            if f'{path.name} 无法读取' in issues:
                self.file_cache.pop(path, None)  # A released OS lock need not change mtime.
            else:
                self.file_cache[path] = cached
        return cached[1], cached[2], cached[3], stamp

    def read(self, run_id=None, date_from=None, date_to=None):
        directory = Path(self.directory)
        with self.read_lock:
            try:
                files = self._files(directory, date_from, date_to)
            except OSError:
                return {}, ['无法读取请求历史目录']
            runs, issues = {}, []
            for path in files:
                # A per-run legacy filename can be excluded before parsing.
                if run_id is not None and path.stem != run_id:
                    try:
                        validate_date(path.stem)
                    except ValueError:
                        continue
                try:
                    loaded_runs, errors, _, _ = self._cached_file(path)
                except OSError:
                    issues.append(f'{path.name} 无法读取')
                    continue
                issues.extend(errors)
                for identifier, loaded in loaded_runs.items():
                    if run_id is not None and identifier != run_id:
                        continue
                    run = runs.setdefault(identifier, {'id': identifier, 'started_at': loaded['started_at'], 'records': {}})
                    run['started_at'] = min(run['started_at'], loaded['started_at'])
                    for key, record in loaded['records'].items():
                        if (date_from is not None and record['date'] < date_from) or (date_to is not None and record['date'] > date_to):
                            continue
                        previous = run['records'].get(key)
                        if previous is None or record['revision'] > previous['revision']:
                            run['records'][key] = deepcopy(record)
            return runs, issues

    @staticmethod
    def _delete_owned_file(path, stamp):
        if os.name == 'nt':
            # An exclusive DELETE handle both rejects another open writer and
            # marks deletion while still holding the handle; there is no
            # close-then-unlink window in which a new append could be lost.
            import ctypes
            from ctypes import wintypes
            import msvcrt
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            create = kernel.CreateFileW
            create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                               wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
            create.restype = wintypes.HANDLE
            handle = create(str(path), 0x80010000, 0, None, 3, 0x00200000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            except BaseException:
                kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
                kernel.CloseHandle(handle)
                raise
            with os.fdopen(descriptor, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if (info.st_size, info.st_mtime_ns) != stamp or getattr(info, 'st_file_attributes', 0) & 0x400:
                    return False
                dispose = kernel.SetFileInformationByHandle
                dispose.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
                dispose.restype = wintypes.BOOL
                flag = ctypes.c_ubyte(1)  # FILE_DISPOSITION_INFO.DeleteFile (BOOLEAN)
                if not dispose(handle, 4, ctypes.byref(flag), ctypes.sizeof(flag)):
                    raise ctypes.WinError(ctypes.get_last_error())
            return True
        with path.open('r+b') as stream, exclusive_log(stream):
            info = os.fstat(stream.fileno())
            if (info.st_size, info.st_mtime_ns) != stamp:
                return False
            path.unlink()
        return True

    def _cleanup(self):
        with self.condition:
            directory, days = Path(self.directory), self.retention_days
            if not self.enabled or not days:
                return
        # Never follow a directory junction/symlink when removing product data.
        if any(self._linked(path) for path in (directory, *directory.parents)):
            with self.condition:
                self.cleanup_error = '保存目录包含链接，未清理过期历史'
            return
        cutoff = (date.fromisoformat(local_date()) - timedelta(days=days - 1)).isoformat()
        with self.read_lock:
            files = self._files(directory)
            parsed, all_records = {}, {}
            for path in files:
                runs, issues, owner, stamp = self._cached_file(path)
                records = [record for run in runs.values() for record in run['records'].values()]
                parsed[path] = (records, owner, stamp)
                for record in records:
                    key = (record['run_id'], record['request_key'])
                    if key not in all_records or record['revision'] > all_records[key]['revision']:
                        all_records[key] = record
            protected_turns = set()
            for record in all_records.values():
                if record.get('turn_id') and (record['date'] >= cutoff or record['status'] == 'running' or record.get('turn_state') in {'running', 'waiting'}):
                    protected_turns.add((record['run_id'], record['instance_id'], record['turn_id']))
            for path, (records, owner, stamp) in parsed.items():
                if not owner:
                    continue
                dates = {record['date'] for record in records}
                if not dates:
                    try:
                        dates = {validate_date(path.stem)}
                    except ValueError:
                        continue
                if any(day >= cutoff for day in dates):
                    continue
                if any(record['status'] == 'running' or record.get('turn_state') == 'running' or
                       (record['run_id'], record['instance_id'], record.get('turn_id')) in protected_turns for record in records):
                    continue
                with self.condition:
                    if not self.enabled or self.directory != str(directory) or self.retention_days != days:
                        return
                    if dates.intersection(self.protected_dates):
                        continue
                    pending_days = {json.loads(data)['date'] for target, data in self.queue if target == str(directory)}
                    if dates.intersection(pending_days) or self._linked(path):
                        continue
                    current = path.stat()
                    if (current.st_size, current.st_mtime_ns) != stamp:
                        continue
                    with ARCHIVE_IO_LOCK:
                        if not self._delete_owned_file(path, stamp):
                            continue
                    self.file_cache.pop(path, None)
                    self.created_paths.discard(path)


class RequestHistory:
    def __init__(self, directory, enabled=True, retention_days=0, *, infer_turns=True):
        self.run_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.records = {}
        self.active = {}
        self.boundaries = {}
        self.signatures = Signatures()
        self.infer_turns = infer_turns
        self._usage_version = 0
        self._usage_finished = {}
        self._usage_known = set()
        self._generation_bases = {}
        self.archive = HistoryArchive(directory)
        self.configure(enabled, directory, retention_days)

    def configure(self, enabled, directory, retention_days=0):
        self.archive.configure(enabled, directory,
            {'schema': 2, 'type': 'run', 'id': self.run_id, 'started_at': self.started_at}, self.records.values(), retention_days)

    def _publish(self, record):
        self._generation_bases.clear()
        protected_turns = {item['turn_id'] for item in self.records.values()
                           if item['turn_id'] and (item['status'] == 'running' or item['turn_state'] in {'running', 'waiting'})}
        self.archive.protect({item['date'] for item in self.records.values()
                              if item['status'] == 'running' or item['turn_id'] in protected_turns
                              or self.boundaries.get(item['request_key'], {}).get('waiting')})
        record['revision'] += 1
        self.archive.enqueue(record)

    def mark_concurrent(self, instance_id):
        for token, record in self.active.items():
            if record['instance_id'] == instance_id:
                self.boundaries[token]['concurrent'] = True

    def capture_live(self, token, live):
        if token in self.active and live.get('request_key') == token and live.get('scope') == 'request':
            self.records[token]['live'] = _clean_live(live)

    def start(self, metrics, instance, messages, *, client_context=None):
        client_context = clean_client_context(client_context)
        concurrent = getattr(instance, 'active_requests', 0) > 1 or any(record['instance_id'] == instance.id for record in self.active.values())
        if concurrent:
            self.mark_concurrent(instance.id)
        same_session = [token for token, record in self.active.items()
                        if client_context is not None and record['instance_id'] == instance.id
                        and record.get('client_context') == client_context]
        for token in same_session:
            self.boundaries[token]['session_concurrent'] = True
        sequence = [self.signatures.message(message) for message in messages] if self.infer_turns else []
        body_sequence = [self.signatures.message(message, omit_reasoning=True) for message in messages] if self.infer_turns else []
        input_signature = self.signatures.digest(sequence)
        body_input = self.signatures.digest(body_sequence)
        candidate = None
        matches = []
        if self.infer_turns and all(sequence):
            for token, boundary in self.boundaries.items():
                record = self.records[token]
                size = boundary['size']
                if record.get('client_context') != client_context or record['instance_id'] != instance.id or not boundary.get('response') or len(sequence) < size + 1:
                    continue
                tail = messages[size + 1:]
                if any(not isinstance(message, dict) or message.get('role') != 'tool' for message in tail):
                    continue
                ids = [self.signatures.digest(message['tool_call_id']) for message in tail if isinstance(message.get('tool_call_id'), str) and message['tool_call_id']]
                expected = boundary['tool_ids']
                identified = bool(expected) and bool(tail) and None not in expected
                if identified:
                    if len(ids) != len(tail) or len(ids) != len(set(ids)) or set(ids) != set(expected):
                        continue
                    duplicate = any(set(ids).intersection(other['tool_ids']) for key, other in self.boundaries.items()
                                    if key != token and self.records[key]['instance_id'] == instance.id
                                    and self.records[key].get('client_context') == client_context)
                    if duplicate:
                        continue
                    # Optional reasoning need not be echoed by clients. Body,
                    # arguments, results and the complete history boundary must match.
                    if self.signatures.digest(body_sequence[:size]) != boundary['body_input'] or body_sequence[size] != boundary.get('body_response'):
                        continue
                else:
                    if expected or tail or concurrent or boundary.get('concurrent'):
                        continue
                    if self.signatures.digest(sequence[:size]) != boundary['input'] or sequence[size] != boundary['response']:
                        continue
                matches.append((token, identified))
            if len(matches) == 1:
                token, identified = matches[0]
                boundary = self.boundaries[token]
                children = boundary.get('children', [])
                retry = bool(children) and all(
                    self.records[child]['status'] in {'failed', 'cancelled', 'incomplete'}
                    and (identified or not self.boundaries[child].get('concurrent'))
                    and self.boundaries[child]['body_input' if identified else 'input'] == (body_input if identified else input_signature)
                    and self.boundaries[child]['size'] == len(sequence)
                    for child in children)
                signature_key = 'body_input' if identified else 'input'
                outsiders = any(key not in children and other['size'] == len(sequence)
                                and other[signature_key] == (body_input if identified else input_signature)
                                and self.records[key]['instance_id'] == instance.id
                                and self.records[key].get('client_context') == client_context
                                for key, other in self.boundaries.items())
                if not outsiders and (not children or retry) and self.records[token]['status'] == 'completed':
                    candidate = token
        if self.infer_turns and candidate is None and client_context is not None and not same_session and all(sequence):
            # A native session is an isolation boundary, not a user-turn ID.
            # Only its latest unambiguous failed attempt can be an exact retry;
            # changed/compacted inputs and older recovered groups never qualify.
            previous = next((record for record in reversed(self.records.values())
                             if record['instance_id'] == instance.id and record.get('client_context') == client_context), None)
            if previous is not None:
                boundary = self.boundaries[previous['request_key']]
                if (previous['status'] == 'failed' and previous.get('turn_state') not in {'completed', 'recovered'}
                        and not boundary.get('session_concurrent') and boundary['size'] == len(sequence)
                        and boundary['input'] == input_signature):
                    candidate = previous['request_key']
        record = {**metrics.snapshot(), 'run_id': self.run_id, 'instance_id': instance.id,
                  'alias': instance.alias, 'backend': instance.backend, 'revision': 0, 'date': local_date(metrics.started_at),
                  'aggregation': 'request', 'turn_id': None, 'turn_state': None, 'labels': [], 'tool_names': []}
        if client_context is not None:
            record['client_context'] = client_context
        self.records[metrics.token] = record
        self.active[metrics.token] = record
        self.boundaries[metrics.token] = {'input': input_signature, 'body_input': body_input, 'size': len(sequence),
            'response': None, 'tool_ids': (), 'observer': ResponseSignature(self.signatures), 'metrics': metrics,
            'concurrent': concurrent, 'session_concurrent': bool(same_session)}
        turn = None
        if candidate:
            previous = self.records[candidate]
            turn = previous['turn_id'] or uuid.uuid4().hex
            self.boundaries[candidate].setdefault('children', []).append(metrics.token)
        if turn:
            for item in self.records.values():
                if item['request_key'] in (candidate, metrics.token) or item['turn_id'] == turn:
                    item.update(aggregation='turn', turn_id=turn, turn_state='running')
                    if item is not record:
                        self._publish(item)
            record.update(aggregation='turn', turn_id=turn, turn_state='running')
        self._publish(record)

    def _turn_state(self, turn_id):
        # Insertion order is request-start order, not response-arrival order.
        # An old concurrent attempt finishing late cannot replace a newer result.
        members = [record for record in self.records.values() if record['turn_id'] == turn_id]
        if any(record['request_key'] in self.active for record in members):
            return 'running'
        latest = members[-1]
        if self.boundaries[latest['request_key']].get('waiting'):
            return 'waiting'
        if latest['status'] == 'completed' and any(record['status'] == 'failed' for record in members):
            return 'recovered'
        return latest['status']

    def observe(self, token, chunk):
        boundary = self.boundaries.get(token)
        if boundary and boundary.get('observer'):
            boundary['observer'].observe(chunk)

    def finish(self, metrics):
        token = metrics.token
        if token not in self.active:
            return
        record = self.active.pop(token)
        record.update(metrics.result or metrics.finish('incomplete'))
        boundary = self.boundaries[token]
        observer = boundary.pop('observer')
        response, ids, waiting = observer.finish()
        record['tool_names'] = observer.tool_names()
        boundary.pop('metrics', None)
        boundary.update(response=response, body_response=getattr(observer, 'body_signature', None), tool_ids=ids,
                        waiting=self.infer_turns and waiting and record['status'] == 'completed')
        if waiting:
            record['labels'] = ['tool_calls']
        if record['turn_id']:
            state = self._turn_state(record['turn_id'])
            for other in self.records.values():
                if other['turn_id'] == record['turn_id']:
                    other['turn_state'] = state
                    if other is not record:
                        self._publish(other)
        self._publish(record)

    def _snapshot(self, record):
        value = deepcopy(record)
        boundary = self.boundaries.get(record['request_key'])
        if boundary and boundary.get('metrics'):
            value.update(boundary['metrics'].snapshot())
        return value

    def latest(self, instance_id):
        matching = [record for record in self.records.values() if record['instance_id'] == instance_id]
        if not matching:
            return None
        latest = next((record for record in matching if record['request_key'] in self.active), matching[-1])  # Oldest active request stays selected until it ends.
        turn = latest['turn_id'] if self.infer_turns else None
        records = [record for record in matching if record['turn_id'] == turn] if turn else [latest]
        state = (latest['turn_state'] or ('waiting' if self.boundaries[latest['request_key']].get('waiting') else latest['status'])) if self.infer_turns else latest['status']
        return {'run_id': self.run_id, 'aggregation': latest['aggregation'] if self.infer_turns else 'request', 'turn_id': turn,
                'state': state, 'requests': [self._snapshot(record) for record in records]}

    def generation_basis(self, token):
        """Same-turn terminal base, invalidated only by history lifecycle writes."""
        if token in self._generation_bases:
            return self._generation_bases[token]
        record = self.records.get(token)
        if record is None:
            return None
        if not self.infer_turns:
            return {'source': 'external', 'key': 'request:' + token, 'turn_id': None, 'client_request_id': None,
                    'request_keys': [token], 'prior': None}
        turn = record['turn_id']
        members = [item for item in self.records.values()
                   if item['instance_id'] == record['instance_id'] and (item['turn_id'] == turn if turn else item is record)]
        prior = [item for item in members if item['request_key'] != token and item['status'] != 'running']
        result = {'source': 'external', 'key': 'turn:' + turn if turn else 'request:' + token,
                  'turn_id': turn, 'client_request_id': None,
                  'request_keys': [item['request_key'] for item in members], 'prior': request_history_stats(prior)}
        self._generation_bases[token] = result
        return result

    def catalog(self, saved, issues):
        runs = [{**run, 'current': False} for key, run in saved.items() if key != self.run_id]
        runs.sort(key=lambda run: run['started_at'], reverse=True)
        runs.insert(0, {'id': self.run_id, 'started_at': self.started_at, 'current': True, 'records': self.records})
        result = []
        for run in runs:
            instances = {record['instance_id']: {'id': record['instance_id'], 'alias': record['alias'], 'backend': record['backend']} for record in run['records'].values()}
            result.append({key: run[key] for key in ('id', 'started_at', 'current')} | {'request_count': len(run['records']), 'instances': list(instances.values())})
        return {'current_run_id': self.run_id, 'runs': result, 'issues': issues}

    def _all_records(self, saved):
        merged = {(record['run_id'], record['request_key']): deepcopy(record)
                  for run in saved.values() for record in run['records'].values()}
        merged.update({(self.run_id, record['request_key']): self._snapshot(record) for record in self.records.values()})
        return list(merged.values())

    def usage_revision(self, saved, internal=None):
        """Compare retained terminal metrics, never streaming text or duration ticks."""
        records = self._all_records(saved) + (internal or {}).get('records', [])
        known, finished = set(), {}
        for record in records:
            identity = (('internal', record['conversation_id'], record['message_id'], record['request_key'])
                        if record.get('source') == 'internal' else ('external', record['run_id'], record['request_key']))
            known.add(identity)
            if not _usage_active(record):
                finished[identity] = (record.get('instance_id'), record['date'], record['started_at'],
                                      record.get('turn_state') if record.get('source') == 'internal' else None,
                                      request_history_stats([record]))
            elif identity in self._usage_finished:
                # Linking a finished call to a new running turn is not a usage update.
                finished[identity] = self._usage_finished[identity]
        for legacy in (internal or {}).get('legacy', []):
            identity = ('legacy', legacy['conversation_id'], legacy['message_id'])
            known.add(identity)
            if not legacy['active']:
                finished[identity] = (legacy['instance_id'], legacy['date'], legacy['has_metrics'])
            elif identity in self._usage_finished:
                finished[identity] = self._usage_finished[identity]
        if finished != self._usage_finished or self._usage_known - known:
            self._usage_version += 1
        self._usage_finished, self._usage_known = finished, known
        return f'{self.run_id}:{self._usage_version}'

    def usage(self, saved, issues, *, run_id=None, instance_id=None, date_from=None, date_to=None,
              source=None, conversation_id=None, internal=None, as_of=None):
        for value in (date_from, date_to):
            if value is not None:
                validate_date(value)
        if date_from and date_to and date_from > date_to:
            raise ValueError('开始日期不得晚于结束日期')
        if conversation_id is not None:
            if source == 'external' or run_id is not None:
                raise ValueError('内置对话与外部请求身份不可混用')
            source = 'internal'
        if source == 'internal' and run_id is not None:
            raise ValueError('内置对话不使用外部运行或请求身份')
        if run_id is not None and run_id != self.run_id and run_id not in saved:
            raise KeyError(run_id)
        revision = self.usage_revision(saved, internal)
        records = []
        if source != 'internal':
            records.extend(record for record in self._all_records(saved)
                           if run_id is None or record['run_id'] == run_id)
        if source != 'external' and run_id is None:
            records.extend(record for record in (internal or {}).get('records', [])
                           if conversation_id is None or record['conversation_id'] == conversation_id)
        records = [record for record in records
                   if (instance_id is None or record.get('instance_id') == instance_id)
                   and (date_from is None or record['date'] >= date_from)
                   and (date_to is None or record['date'] <= date_to)]
        coverage = {'legacy_message_count': 0, 'legacy_metrics_message_count': 0}
        if source != 'external' and run_id is None:
            for legacy in (internal or {}).get('legacy', []):
                if ((instance_id is None or legacy['instance_id'] == instance_id)
                        and (conversation_id is None or legacy['conversation_id'] == conversation_id)
                        and (date_from is None or legacy['date'] >= date_from)
                        and (date_to is None or legacy['date'] <= date_to)):
                    coverage['legacy_message_count'] += 1
                    coverage['legacy_metrics_message_count'] += legacy['has_metrics']
        as_of = time.time() if as_of is None else as_of
        return {'revision': revision, 'as_of': as_of, 'summary': request_history_stats(records),
                'coverage': coverage, 'issues': issues, 'trend': request_history_trend(records, as_of)}

    def dates(self, saved, issues, internal=None):
        days = {}
        for record in self._all_records(saved):
            day = record['date']
            group = days.setdefault(day, {'date': day, 'request_count': 0, 'instances': {}})
            group['request_count'] += 1
            run_start = self.started_at if record['run_id'] == self.run_id else saved[record['run_id']]['started_at']
            group['instances'][('external', record['run_id'], record['instance_id'])] = {
                'source': 'external', 'run_id': record['run_id'], 'id': record['instance_id'], 'alias': record['alias'],
                'backend': record['backend'], 'started_at': run_start}
        for day in (internal or {}).get('dates', []):
            group = days.setdefault(day['date'], {'date': day['date'], 'request_count': 0, 'instances': {}})
            group['request_count'] += day['request_count']
            for item in day['instances']:
                group['instances'][('internal', item['id'], item.get('conversation_id'))] = item
        return {'current_run_id': self.run_id, 'today': local_date(), 'dates': [
            {**days[day], 'instances': list(days[day]['instances'].values())} for day in sorted(days, reverse=True)],
            'issues': issues, 'usage_revision': self.usage_revision(saved, internal),
            **({'coverage': internal['coverage']} if internal is not None else {})}

    def page(self, saved, issues, run_id=None, instance_id=None, turn_id=None, offset=0, limit=None, request_key=None,
             date_from=None, date_to=None, group_by=None, internal=None, include_external=True, external_group_by=None):
        if group_by not in (None, 'turn'):
            raise ValueError('分组方式必须为 turn')
        if external_group_by not in (None, 'request'):
            raise ValueError('外部分组方式必须为 request')
        if date_from is not None:
            validate_date(date_from)
        if date_to is not None:
            validate_date(date_to)
        if date_from and date_to and date_from > date_to:
            raise ValueError('开始日期不得晚于结束日期')
        if run_id and (turn_id or request_key):
            date_from = date_to = None
        dated = date_from is not None or date_to is not None
        grouped = group_by == 'turn' and turn_id is None and request_key is None
        if not dated and not grouped:
            run_id = run_id or self.run_id
        if run_id == self.run_id:
            records = [self._snapshot(record) for record in self.records.values()]
        elif run_id is None:
            records = self._all_records(saved)
        elif run_id in saved:
            records = list(saved[run_id]['records'].values())
        else:
            raise KeyError(run_id)
        if not include_external:
            records = []
        records = [record for record in records
                   if (instance_id is None or record['instance_id'] == instance_id)
                   and (turn_id is None or record['turn_id'] == turn_id)
                   and (request_key is None or record['request_key'] == request_key)]
        if turn_id is None:
            records.reverse()
        records.sort(key=lambda record: record['started_at'], reverse=turn_id is None)
        def matches_date(record):
            return (date_from is None or record['date'] >= date_from) and (date_to is None or record['date'] <= date_to)
        matched_records = [record for record in records if matches_date(record)]
        limit = (len(matched_records) if turn_id is not None else 50) if limit is None else limit
        if grouped:
            per_request = external_group_by == 'request'
            groups = {}
            for record in records:
                identity = (record['run_id'], record['instance_id'],
                            'turn' if not per_request and record['turn_id'] is not None else 'request',
                            record['turn_id'] if not per_request and record['turn_id'] is not None else record['request_key'])
                groups.setdefault(identity, []).append(record)
            combined = []
            for group in groups.values():
                matched_count = sum(matches_date(record) for record in group)
                if matched_count:
                    combined.append(({**_history_group(group, per_request=per_request), 'matched_request_count': matched_count}, group[0]))
            combined.extend((group, None) for group in (internal or {}).get('groups', []))
            combined.sort(key=lambda item: item[0]['started_at'], reverse=True)
            page = combined[offset:offset + limit]
            all_records = matched_records + (internal or {}).get('records', [])
            return {'run_id': run_id, 'items': [record for _, record in page if record is not None],
                    'groups': [group for group, record in page if record is not None],
                    'internal_groups': [group for group, record in page if record is None],
                    'total': len(all_records), 'group_total': len(combined), 'summary': request_history_stats(all_records),
                    'coverage': (internal or {}).get('coverage', {'legacy_message_count': 0, 'legacy_metrics_message_count': 0}),
                    'offset': offset, 'limit': limit, 'issues': issues,
                    **({'external_group_by': 'request'} if per_request else {})}
        return {'run_id': run_id, 'items': matched_records[offset:offset + limit], 'total': len(matched_records), 'offset': offset, 'limit': limit, 'issues': issues}
