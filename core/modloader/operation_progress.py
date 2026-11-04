"""Request-scoped progress for the UI; ordinary JSON callers keep their contract."""
from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from anyio import CancelScope


class _OperationStream(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Disconnect can interrupt send() while the generator is suspended at yield.
            # Close it explicitly so its operation task is cancelled before returning.
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


async def operation_response(request, operation, *, cancel_on_disconnect=False):
    if 'application/x-ndjson' not in request.headers.get('accept', ''):
        if not cancel_on_disconnect:
            return await operation(lambda event: None)
        task = asyncio.create_task(operation(lambda event: None))
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=.25)
                if not task.done() and await request.is_disconnected():
                    raise HTTPException(499, '操作已取消')
            return await task
        finally:
            if not task.done():
                task.cancel()
                with CancelScope(shield=True):
                    await asyncio.gather(task, return_exceptions=True)

    async def stream():
        # Keep the latest update, not a backlog of every scanned path.
        updates = asyncio.Queue(maxsize=1)
        loop = asyncio.get_running_loop()
        active = True

        def enqueue(event):
            if not active:
                return
            if updates.full():
                updates.get_nowait()
            updates.put_nowait(event)

        def progress(event):
            if active:
                loop.call_soon_threadsafe(enqueue, dict(event))

        def encode(event):
            return json.dumps(jsonable_encoder(event), ensure_ascii=False) + '\n'

        task = asyncio.create_task(operation(progress))
        update = asyncio.create_task(updates.get())
        try:
            while True:
                await asyncio.wait({task, update}, return_when=asyncio.FIRST_COMPLETED)
                if update.done():
                    yield encode({'type': 'progress', **update.result()})
                    update = asyncio.create_task(updates.get())
                if task.done():
                    if not updates.empty():
                        yield encode({'type': 'progress', **updates.get_nowait()})
                    yield encode({'type': 'result', 'value': await task})
                    return
        except Exception as error:
            from .runtime import scrub
            status = error.status_code if isinstance(error, HTTPException) else 400 if isinstance(error, (ValueError, FileNotFoundError)) else 403 if isinstance(error, PermissionError) else 500
            detail = error.detail if isinstance(error, HTTPException) else '文件或目录不存在，请重新选择' if isinstance(error, FileNotFoundError) else '无法访问文件或目录，请检查权限' if isinstance(error, PermissionError) else str(error)
            yield encode({'type': 'error', 'status': status, 'detail': scrub(detail) if isinstance(detail, str) else detail})
        finally:
            active = False
            task.cancel()
            update.cancel()
            with CancelScope(shield=True):
                await asyncio.gather(task, update, return_exceptions=True)

    return _OperationStream(stream(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store'})
