"""Coalesced management samples with retained terminal accounting; no polling."""
from __future__ import annotations

import asyncio
import time

from .request_metrics import DECODE_PHASES, generation_window


def _stable(payload):
    """Request age is not a new measurement or an excuse to emit text chunks."""
    live = payload.get('live_generation')
    accounting = payload.get('accounting')
    return ({key: value for key, value in live.items() if key not in ('elapsed_s', 'request_elapsed_s')} if live else None,
            {'request': {key: value for key, value in accounting['request'].items() if key != 'duration_s'},
             'turn': accounting['turn']} if accounting else None)


def request_accounting(metrics, turn=None):
    if turn is None or metrics.source == 'external':
        client_id = metrics.client_request_id if metrics.source == 'internal' else None
        turn = {'source': metrics.source,
                'key': 'chat:' + client_id if client_id else 'request:' + metrics.token,
                'turn_id': None, 'client_request_id': client_id, 'request_keys': [metrics.token], 'prior': None}
    return {'request': metrics.result or metrics.snapshot(), 'turn': turn}


class GenerationEvents:
    def __init__(self, run_id):
        self.run_id = run_id
        self._subscribers = {}
        self._windows = {}
        self._latest = {}

    def subscribed(self, identifier):
        return bool(self._subscribers.get(identifier))

    def subscribe(self, identifier, current=None, *, active=False, turn=None, sampled_at=None, observed_at=None):
        queue = asyncio.Queue(maxsize=1)
        existing = self.subscribed(identifier)
        self._subscribers.setdefault(identifier, set()).add(queue)
        if current is not None or active:
            if existing and identifier in self._latest:
                queue.put_nowait(self._latest[identifier])
            else:
                self.publish(identifier, current, reset=True, turn=turn, sampled_at=sampled_at, observed_at=observed_at)
        return queue

    def unsubscribe(self, identifier, queue):
        subscribers = self._subscribers.get(identifier)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(identifier, None)
                self._windows.pop(identifier, None)
                self._latest.pop(identifier, None)

    def publish(self, identifier, current, chunk=None, *, reset=False, turn=None, finalized=(), sampled_at=None, observed_at=None):
        subscribers = self._subscribers.get(identifier)
        if not subscribers:
            return False
        last_payload = self._latest.get(identifier) or {}
        last = last_payload.get('live_generation')
        if reset or current is None:
            self._windows.pop(identifier, None)
        if current is None:
            if last_payload and last is None and not finalized:
                return False
            result, accounting = None, None
        else:
            # Lifecycle snapshots seed a real baseline. A text/null frame, or a
            # time-only update to the same count, must not advance that baseline.
            live, fresh = (current.live_generation(), True) if reset else current.stream_sample(chunk)
            same = last is not None and last.get('request_key') == current.token
            overview = live.get('_overview', {})
            counted = reset or (fresh and (not same
                or (live.get('_generated_tokens') is not None and live['_generated_tokens'] != last.get('generated_tokens'))
                or (live.get('_prefill_source') and overview.get('prefill_processed_tokens') != last.get('prefill_processed_tokens'))))
            if not counted:
                live = {**live, '_generated_tokens': None, '_prefill_source': None}
            previous = self._windows.get(identifier)
            sampled_at = time.monotonic() if sampled_at is None else sampled_at
            window, result = generation_window(previous, current.token, live, sampled_at)
            if counted:
                self._windows[identifier] = window
            elif same and (last.get('phase') == current.phase or last.get('phase') in DECODE_PHASES and current.phase in DECODE_PHASES):
                for key in ('tokens_per_second', 'window_s', 'available_for_chat', 'reason',
                            'prefill_tokens_per_second', 'prefill_window_s'):
                    result[key] = last.get(key)
                for mean, basis in (('average_tokens_per_second', 'average_basis'),
                                    ('average_prefill_tokens_per_second', 'prefill_average_basis')):
                    if result.get(mean) is None and last.get(basis) in ('sampled_decode', 'sampled_prefill'):
                        result[mean], result[basis] = last.get(mean), last.get(basis)
            result.update(client_request_id=current.client_request_id, client_call_id=current.client_call_id,
                          phase=current.phase, first_output_s=current.first_output_s,
                          request_elapsed_s=live.get('_elapsed_s'))
            if current.phase in DECODE_PHASES:
                result.update(prefill_tokens_per_second=None, prefill_window_s=None)
            accounting = request_accounting(current, turn)
        payload = {'run_id': self.run_id, 'instance_id': identifier,
                   'observed_at': time.time() if observed_at is None else observed_at,
                   'live_generation': result, 'accounting': accounting, 'finalized': list(finalized)}
        if not reset and not finalized and last_payload and _stable(payload) == _stable(last_payload):
            return False
        self._latest[identifier] = payload
        for queue in subscribers:
            pending = queue.get_nowait().get('finalized', []) if queue.full() else []
            # Latest fast samples can replace one another; terminal contributions
            # are request-keyed replacements, never event deltas or lossy clears.
            closed = {item['request']['request_key']: item for item in (*pending, *finalized)}
            queue.put_nowait({**payload, 'finalized': list(closed.values())})
        return True
