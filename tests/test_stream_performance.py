"""Real loopback SSE must keep flowing while a telemetry client initializes."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time

import httpx
import uvicorn

from modloader.app import Controller, build_app
from modloader.models import LaunchSpec, RuntimeInstance
from modloader.runtime import free_port


async def test_secondly_telemetry_does_not_block_gateway_sse(tmp_path, monkeypatch):
    class Backend(BaseHTTPRequestHandler):
        generated = 0

        def log_message(self, *_):
            pass

        def do_GET(self):
            data = {'status': 'ok', 'loaded': True} if self.path == '/health' else {
                'engine': {}, 'live': {'state': 'generating', 'running': 1, 'queued': 0,
                                      'generated': Backend.generated, 'elapsed_s': Backend.generated * .02},
                'totals': {}, 'requests': [],
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for index in range(135):
                Backend.generated = index
                data = {'choices': [{'delta': {'reasoning_content': f'分析 {index}。'}}]}
                self.wfile.write(('data: ' + json.dumps(data) + '\n\n').encode())
                self.wfile.flush()
                time.sleep(.02)
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    controller = Controller(tmp_path, 'performance-test-secret')
    controller.store.put('profiles', {'id': 'stream', 'generation': {}})
    instance = RuntimeInstance(id='stream', model_id='stream', profile_id='stream', profile_revision=1,
                               alias='stream', backend='strata', runtime_id='fixture', state='ready',
                               port=backend.server_port, started_at='', pid=os.getpid(), capabilities={'input': ['text']})
    controller.runtime.instances[instance.id] = instance
    controller.runtime.specs[instance.id] = LaunchSpec(argv=[], cwd=str(tmp_path), model_name='stream')
    controller.runtime.requests[instance.id] = set()
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(build_app(controller), host='127.0.0.1', port=port, log_level='error'))
    server_task = asyncio.create_task(server.serve())
    arrivals = []
    frames = []
    reader_task = None
    original_client = httpx.AsyncClient

    # The OS trust-store delay varies across machines. A synchronous 200ms
    # constructor makes the regression repeatable without replacing HTTP/SSE.
    def slow_client(*args, **kwargs):
        # Both gateway and polling clients are HTTP-loopback-only. A default
        # TLS context here costs a substantial part of a short generation.
        assert kwargs.get('verify') is False
        assert kwargs.get('follow_redirects') is False
        time.sleep(.2)
        return original_client(*args, **kwargs)

    def read_stream():
        with httpx.Client(trust_env=False, timeout=20) as client:
            with client.stream('POST', f'http://127.0.0.1:{port}/api/chat',
                               headers={'Authorization': 'Bearer performance-test-secret'},
                               json={'model': 'stream', 'messages': [], 'stream': True}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: '):
                        data = line[6:]
                        frame = data if data == '[DONE]' else json.loads(data)
                        frames.append(frame)
                        if frame != '[DONE]' and 'modloader_request' not in frame:
                            arrivals.append(time.perf_counter())

    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(.01)
        async with original_client(trust_env=False, headers={'Authorization': 'Bearer performance-test-secret'}) as client:
            monkeypatch.setattr('modloader.backends.telemetry.httpx.AsyncClient', slow_client)
            reader_task = asyncio.create_task(asyncio.to_thread(read_stream))
            async with asyncio.timeout(10):
                while not arrivals:
                    if reader_task.done():
                        await reader_task
                    await asyncio.sleep(.01)
            first = arrivals[0]
            sampled = []
            for index in (1, 2):
                await asyncio.sleep(max(0, first + index - time.perf_counter()))
                response = await client.get(f'http://127.0.0.1:{port}/api/instances/stream/telemetry')
                assert response.status_code == 200
                sampled.append(response.json())
            await reader_task
        assert len(arrivals) == 135
        assert frames[:-2] == [
            {'choices': [{'delta': {'reasoning_content': f'分析 {index}。'}}]}
            for index in range(135)
        ]
        assert set(frames[-2]) == {'modloader_request'}
        assert frames[-2]['modloader_request']['status'] == 'completed'
        assert frames[-1] == '[DONE]'
        assert max(after - before for before, after in zip(arrivals, arrivals[1:])) < .15
        assert sampled[1]['updated_at'] > sampled[0]['updated_at']
        assert sampled[1]['live_generation']['available_for_chat']
        assert sampled[1]['live_generation']['tokens_per_second'] > 0
    finally:
        if reader_task:
            await asyncio.gather(reader_task, return_exceptions=True)
        server.should_exit = True
        await server_task
        await asyncio.to_thread(backend.shutdown)
        backend.server_close()
        controller.runtime.instances.clear()
        await controller.close()
