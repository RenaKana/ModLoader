import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sys
import uvicorn


class LocalServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        yield


def listener(port):
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.bind(('127.0.0.1',port))
    sock.listen(128)
    sock.setblocking(False)
    return sock


async def run(args):
    from .app import Controller,build_app
    token=os.environ.get('MODLOADER_MANAGEMENT_TOKEN')
    if not token:
        raise RuntimeError('需要设置 MODLOADER_MANAGEMENT_TOKEN；请从桌面入口启动')
    controller=Controller(args.data_dir,token)
    sock=listener(args.management_port)
    manager=LocalServer(uvicorn.Config(build_app(controller),access_log=False,log_level='warning',timeout_graceful_shutdown=10))
    try:
        public_socket=listener(controller.settings['api_port'])
        controller.public_server=LocalServer(uvicorn.Config(build_app(controller,public=True),access_log=False,log_level='warning',timeout_graceful_shutdown=10))
        controller.public_task=asyncio.create_task(controller.public_server.serve(sockets=[public_socket]))
    except OSError:
        controller.api_error=f"API 端口 {controller.settings['api_port']} 已被占用；请在设置中修改端口并重启"
    controller.metrics_task=asyncio.create_task(controller.metrics())
    task=asyncio.create_task(manager.serve(sockets=[sock]))
    while not manager.started:
        if task.done(): await task
        await asyncio.sleep(.05)
    print(json.dumps({'event':'ready','port':sock.getsockname()[1],'api_port':controller.settings['api_port']}),flush=True)
    async def parent():
        await asyncio.to_thread(sys.stdin.buffer.read)
        manager.should_exit=True
    parent_task=asyncio.create_task(parent()) if args.parent_pipe else None
    try:
        await task
    finally:
        if parent_task: parent_task.cancel()
        await controller.close()


def main():
    if '--source-worker' in sys.argv:
        from modloader.sources.worker import main as worker_main
        sys.exit(worker_main())
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default=str(Path.cwd()/'data'))
    parser.add_argument('--management-port',type=int,default=0)
    parser.add_argument('--parent-pipe',action='store_true')
    args=parser.parse_args()
    asyncio.run(run(args))


if __name__=='__main__':
    main()
