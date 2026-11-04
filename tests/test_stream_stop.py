"""Owned response-close races exercise the gateway, not a live model."""
import asyncio
import json

import httpx
import pytest

from test_core import controller, gateway_client, sse


class ClosingStream(httpx.AsyncByteStream):
    def __init__(self, chunks, failure=None):
        self.chunks = chunks
        self.failure = failure or httpx.StreamClosed()
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.close_count = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            yield sse(chunk)
        self.waiting.set()
        await self.release.wait()
        raise self.failure

    async def aclose(self):
        self.close_count += 1
        self.release.set()


def chunks_of(response):
    return [json.loads(line[5:]) for line in response.text.splitlines()
            if line.startswith('data:') and line[5:].strip() != '[DONE]']


def assert_released(controller, instance, stream, clients):
    assert instance.active_requests == 0
    assert not controller.runtime.requests[instance.id]
    assert not controller.active_request_tokens[instance.id]
    assert not controller.active_request_metrics
    assert stream.close_count == 1
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize('stage,expected', [('before_first', 'cancelled'), ('partial', 'cancelled'),
                                          ('upstream_error', 'failed'), ('completed', 'completed')])
async def test_force_stop_preserves_request_terminal_state(controller, tmp_path, monkeypatch, stage, expected):
    chunks = [] if stage == 'before_first' else [{'model': 'fixture', 'choices': [{'delta': {'content': '保留部分输出'}}]}]
    if stage == 'upstream_error':
        chunks.append({'error': {'message': '明确的上游错误', 'code': 'server_error'}})
    if stage == 'completed':
        chunks.append('[DONE]')
    stream = ClosingStream(chunks)
    async def backend(request):
        return httpx.Response(200, stream=stream)
    client, instances, clients = gateway_client(controller, tmp_path, monkeypatch, backend)
    instance = instances['one']
    try:
        async with client:
            task = asyncio.create_task(client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': True}))
            await asyncio.wait_for(stream.waiting.wait(), 5)
            await controller.runtime.stop(instance.id, force=True)
            response = await asyncio.wait_for(task, 5)
            assert response.status_code == 200
            received = chunks_of(response)
            reports = [chunk['modloader_request'] for chunk in received if 'modloader_request' in chunk]
            assert len(reports) == 1
            assert reports[0]['status'] == expected
            assert controller.request_metrics[instance.id]['status'] == expected
            assert ('[DONE]' in response.text) is (stage == 'completed')
            if stage == 'completed':
                assert not any('error' in chunk for chunk in received)
            elif stage == 'upstream_error':
                assert [chunk['error']['message'] for chunk in received if 'error' in chunk] == ['明确的上游错误']
            else:
                assert any('error' in chunk for chunk in received)
            if stage != 'before_first':
                assert '保留部分输出' in response.text
            assert_released(controller, instance, stream, clients)
            await controller.runtime.stop(instance.id, force=True)
            assert_released(controller, instance, stream, clients)
    finally:
        await controller.close()


async def test_unowned_stream_close_is_failed_not_cancelled(controller, tmp_path, monkeypatch):
    stream = ClosingStream([])
    async def backend(request):
        return httpx.Response(200, stream=stream)
    client, instances, clients = gateway_client(controller, tmp_path, monkeypatch, backend)
    instance = instances['one']
    try:
        async with client:
            task = asyncio.create_task(client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': True}))
            await asyncio.wait_for(stream.waiting.wait(), 5)
            stream.release.set()  # No instance stop owns this closure.
            response = await asyncio.wait_for(task, 5)
            assert instance.state == 'ready'
            assert controller.request_metrics[instance.id]['status'] == 'failed'
            assert controller.request_metrics[instance.id]['upstream']['code'] == 'StreamClosed'
            assert chunks_of(response)[-1]['error']['message'] == '后端连接已结束'
            assert_released(controller, instance, stream, clients)
    finally:
        await controller.close()


async def test_client_cancel_and_instance_stop_release_once(controller, tmp_path, monkeypatch):
    stream = ClosingStream([{'choices': [{'delta': {'content': '部分正文'}}]}])
    async def backend(request):
        return httpx.Response(200, stream=stream)
    client, instances, clients = gateway_client(controller, tmp_path, monkeypatch, backend)
    instance = instances['one']
    try:
        async with client:
            task = asyncio.create_task(client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': True}))
            await asyncio.wait_for(stream.waiting.wait(), 5)
            task.cancel()
            await controller.runtime.stop(instance.id, force=True)
            with pytest.raises(asyncio.CancelledError):
                await task
            assert controller.request_metrics[instance.id]['status'] == 'cancelled'
            assert_released(controller, instance, stream, clients)
    finally:
        await controller.close()


async def test_unrelated_runtime_error_is_not_swallowed(controller, tmp_path, monkeypatch):
    stream = ClosingStream([], RuntimeError('fixture implementation defect'))
    stream.release.set()
    async def backend(request):
        return httpx.Response(200, stream=stream)
    client, instances, clients = gateway_client(controller, tmp_path, monkeypatch, backend)
    try:
        async with client:
            with pytest.raises(RuntimeError, match='fixture implementation defect'):
                await client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': True})
            assert_released(controller, instances['one'], stream, clients)
    finally:
        await controller.close()


@pytest.mark.parametrize('owned_stop,expected', [(True, 'cancelled'), (False, 'failed')])
async def test_nonstream_closed_response_keeps_http_error_contract(controller, tmp_path, monkeypatch, owned_stop, expected):
    stream = ClosingStream([])
    async def backend(request):
        return httpx.Response(200, stream=stream)
    client, instances, clients = gateway_client(controller, tmp_path, monkeypatch, backend)
    instance = instances['one']
    try:
        async with client:
            task = asyncio.create_task(client.post('/api/chat', json={'model': 'one', 'messages': [], 'stream': False}))
            await asyncio.wait_for(stream.waiting.wait(), 5)
            if owned_stop:
                await controller.runtime.stop(instance.id, force=True)
            else:
                stream.release.set()
            response = await asyncio.wait_for(task, 5)
            assert response.status_code == 502
            assert response.json() == {'detail': '无法连接模型后端，请检查实例状态'}
            assert controller.request_metrics[instance.id]['status'] == expected
            assert controller.request_metrics[instance.id]['upstream']['code'] == 'StreamClosed'
            assert instance.state == ('stopped' if owned_stop else 'ready')
            assert_released(controller, instance, stream, clients)
    finally:
        await controller.close()
