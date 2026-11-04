"""External HTTP requests stay independent while legacy archive queries remain readable."""
import json

import httpx

from modloader.app import build_app
from modloader.generation_events import request_accounting
from modloader.request_metrics import RequestMetrics
from modloader.request_history import request_history_trend
from test_core import controller, gateway_client
from test_request_history import aggregate_record, day_time, internal_metric_request, persist_internal_metrics


async def test_external_gateway_tool_chain_and_retry_are_independent_requests(controller, tmp_path, monkeypatch):
    tool = {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'image-1', 'type': 'function', 'function': {'name': 'view_image', 'arguments': '{}'}}]}
    replies = [(200, {'choices': [{'message': tool, 'finish_reason': 'tool_calls'}]}),
               (503, {'error': {'message': 'unavailable'}}),
               (200, {'choices': [{'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}]})]

    async def handler(request):
        status, body = replies.pop(0)
        return httpx.Response(status, json=body)

    client, instances, _ = gateway_client(controller, tmp_path, monkeypatch, handler)
    try:
        async with client:
            messages = [{'role': 'user', 'content': 'look'}]
            for index in range(3):
                response = await client.post('/v1/chat/completions', headers={'X-Opencode-Session-Id': 'same-session'},
                                             json={'model': 'one', 'messages': messages})
                assert response.status_code == (503 if index == 1 else 200)
                if index == 0:
                    messages += [tool, {'role': 'tool', 'tool_call_id': 'image-1', 'content': 'image'}]
                latest = controller.request_history.latest(instances['one'].id)
                assert latest['aggregation'] == 'request' and latest['turn_id'] is None
                assert latest['state'] == ('failed' if index == 1 else 'completed')
                assert len(latest['requests']) == 1
                record = latest['requests'][0]
                assert record['aggregation'] == 'request' and record['turn_id'] is None
                assert record['turn_state'] is None
                basis = controller.request_history.generation_basis(record['request_key'])
                assert basis['key'] == 'request:' + record['request_key']
                assert basis['request_keys'] == [record['request_key']] and basis['prior'] is None
            records = list(controller.request_history.records.values())
            assert len(records) == 3
            assert records[0]['tool_names'] == ['view_image']
            assert [record['status'] for record in records] == ['completed', 'failed', 'completed']
    finally:
        await controller.close()


def test_external_event_accounting_never_reuses_legacy_turn_totals():
    metrics = RequestMetrics(False, source='external')
    legacy = {'source': 'external', 'key': 'turn:old', 'turn_id': 'old', 'client_request_id': None,
              'request_keys': ['prior', metrics.token], 'prior': {'request_count': 1}}
    accounting = request_accounting(metrics, legacy)
    assert accounting['turn'] == {'source': 'external', 'key': 'request:' + metrics.token,
                                 'turn_id': None, 'client_request_id': None,
                                 'request_keys': [metrics.token], 'prior': None}
    internal = RequestMetrics(False, source='internal', client_request_id='message')
    basis = {**legacy, 'source': 'internal', 'key': 'chat:message'}
    assert request_accounting(internal, basis)['turn'] is basis


def test_legacy_external_waiting_metadata_does_not_keep_usage_active(controller):
    history = controller.request_history
    start = day_time('2026-11-02')
    record = aggregate_record(history, 'old-tool', start, turn_id='old-turn', turn_state='waiting', labels=['tool_calls'])
    history.records[record['request_key']] = record
    assert request_history_trend([record], start + 1201)['end_at'] == start + 60
    first = history.usage_revision({})
    record['turn_state'] = 'running'
    assert history.usage_revision({}) == first
    record['usage']['completion_tokens'] = 30
    assert history.usage_revision({}) != first


async def test_request_mode_pages_legacy_external_requests_and_keeps_internal_whole_turns(controller):
    history = controller.request_history
    start = day_time('2026-11-02')
    for index, status in enumerate(('failed', 'completed', 'completed')):
        record = aggregate_record(history, f'old-{index}', start + index * 10, turn_id='old-turn',
                                  status=status, turn_state='waiting', usage={'prompt_tokens': (index + 1) * 100})
        history.records[record['request_key']] = record
    persist_internal_metrics(controller, [internal_metric_request('inside-one', start + 5),
                                          internal_metric_request('inside-two', start + 7, index=2)])
    before = json.dumps(history.records, sort_keys=True)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),
                                    base_url='http://testserver', headers={'Authorization': 'Bearer management-secret'}) as client:
            pages = [(await client.get('/api/request-history', params={
                'group_by': 'turn', 'external_group_by': 'request', 'offset': index, 'limit': 1})).json()
                     for index in range(4)]
            assert all(page['external_group_by'] == 'request' for page in pages)
            assert all(page['group_total'] == 4 and page['total'] == 5 for page in pages)
            assert [pages[index]['items'][0]['request_key'] for index in (0, 1, 3)] == ['old-2', 'old-1', 'old-0']
            assert pages[2]['internal_groups'][0]['request_count'] == 2
            assert [pages[index]['groups'][0]['status'] for index in (0, 1, 3)] == ['completed', 'completed', 'failed']
            assert all(pages[index]['groups'][0]['request_count'] == 1 for index in (0, 1, 3))
            legacy = (await client.get('/api/request-history', params={'group_by': 'turn'})).json()
            assert legacy['group_total'] == 2
            exact = (await client.get('/api/request-history', params={'turn_id': 'old-turn'})).json()
            assert len(exact['items']) == 3
            assert json.dumps(history.records, sort_keys=True) == before
    finally:
        await controller.close()
