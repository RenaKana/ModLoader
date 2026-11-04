import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

from modloader.operation_progress import operation_response


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
async def test_progress_negotiation_preserves_json_result_and_error(streaming):
    app = FastAPI()

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({'detail': str(error)}, status_code=400)

    @app.get('/operation')
    async def run(request: Request, fail: bool = False):
        async def work(progress):
            progress({'stage': 'metadata', 'message': '读取模型信息', 'completed': 0, 'total': 2})
            await asyncio.sleep(.01)
            progress({'stage': 'metadata', 'message': '读取模型信息', 'completed': 1, 'total': 2})
            await asyncio.sleep(.01)
            if fail:
                raise ValueError('模型分片不存在')
            return {'bytes': 123, 'notes': []}
        return await operation_response(request, work)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture') as client:
        headers = {'Accept': 'application/x-ndjson'} if streaming else {}
        result = await client.get('/operation', headers=headers)
        failed = await client.get('/operation?fail=true', headers=headers)
    if not streaming:
        assert result.json() == {'bytes': 123, 'notes': []}
        assert failed.status_code == 400 and failed.json() == {'detail': '模型分片不存在'}
    else:
        events = [json.loads(line) for line in result.text.splitlines()]
        assert [event['completed'] for event in events if event['type'] == 'progress'] == [0, 1]
        assert events[-1] == {'type': 'result', 'value': {'bytes': 123, 'notes': []}}
        failure = [json.loads(line) for line in failed.text.splitlines()]
        assert failure[-1] == {'type': 'error', 'status': 400, 'detail': '模型分片不存在'}
        assert not any(event['type'] == 'result' for event in failure)


@pytest.mark.asyncio
async def test_closing_progress_stream_cancels_pending_operation():
    request = StarletteRequest({'type': 'http', 'headers': [(b'accept', b'application/x-ndjson')]})
    cancelled = asyncio.Event()

    async def operation(progress):
        try:
            await asyncio.to_thread(progress, {'stage': 'scanning', 'message': '扫描目录'})
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    response = await operation_response(request, operation)
    first = json.loads(await anext(response.body_iterator))
    assert first['type'] == 'progress' and first['stage'] == 'scanning'
    await response.body_iterator.aclose()
    await asyncio.wait_for(cancelled.wait(), 1)


@pytest.mark.asyncio
async def test_repository_search_stream_preserves_query_cursor_and_result(tmp_path, monkeypatch):
    from modloader.app import Controller, build_app
    controller = Controller(tmp_path / 'data', 'fixture-token')
    calls = []
    expected = {'items': [{'id': 'fixture/model'}], 'next_cursor': 'next'}
    async def search(source, query, cursor):
        calls.append((source, query, cursor))
        await asyncio.sleep(.01)
        return expected
    monkeypatch.setattr(controller.sources, 'search', search)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url='http://testserver',
                                    headers={'Authorization': 'Bearer fixture-token', 'Accept': 'application/x-ndjson'}) as client:
            response = await client.get('/api/sources/hf/search', params={'q': '模型 q4', 'cursor': 'page-2'})
        events = [json.loads(line) for line in response.text.splitlines()]
        assert response.status_code == 200
        assert calls == [('hf', '模型 q4', 'page-2')]
        assert events[0]['type'] == 'progress' and events[0]['stage'] == 'repository_search'
        assert events[-1] == {'type': 'result', 'value': expected}
    finally:
        await controller.close()
