"""Explicit real-model smoke through the same controller and OpenAI HTTP gateway."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import secrets
import sys
import time
import httpx
from modloader.app import Controller,build_app
from modloader.models import ModelAsset,LoadProfile,RuntimeInstallation
from modloader.__main__ import listener,LocalServer
import uvicorn


async def main(args):
    root=Path(__file__).resolve().parents[1]
    controller=Controller(root/'work'/'gpu-acceptance',secrets.token_urlsafe(32))
    management=httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)),base_url='http://testserver',headers={'Authorization':'Bearer '+controller.token},timeout=120)
    server=None
    try:
        if args.config:
            response=await management.post('/api/config/import',json={'path':args.config})
            response.raise_for_status()
            imported=response.json()
            profile=imported['profile']
        else:
            from modloader.store import Store
            store=Store(root/'data'/'modloader.sqlite3')
            model=next(m for m in store.all('models') if m.get('source',{}).get('repo_id')==args.repo)
            runtime=next(r for r in store.all('runtimes') if r['backend']==args.backend)
            controller.store.put('models',model)
            controller.store.put('runtimes',runtime)
            store.close()
            parameters={}
            if args.backend=='vllm': parameters={'context_length':4096,'gpu_memory_utilization':.35,'enforce_eager':True,'max_num_seqs':2}
            response=await management.post('/api/profiles',json={'model_id':model['id'],'runtime_id':runtime['id'],'backend':args.backend,'alias':args.backend+'-'+str(int(time.time())),'name':'实际验收','parameters':parameters})
            response.raise_for_status()
            profile=response.json()
        started=time.monotonic()
        response=await management.post('/api/profiles/'+profile['id']+'/start')
        response.raise_for_status()
        identifier=response.json()['id']
        instance=controller.runtime.instances[identifier]
        last=''
        while instance.state=='starting':
            log=next(reversed(controller.runtime.logs[identifier]),'')
            if log!=last:
                print(json.dumps({'phase':'loading','backend':instance.backend,'log':log},ensure_ascii=False),flush=True)
                last=log
            await asyncio.sleep(2)
        if instance.state!='ready':
            print('\n'.join(controller.runtime.logs[identifier]),flush=True)
            raise RuntimeError(instance.error)
        print(json.dumps({'phase':'ready','backend':instance.backend,'load_seconds':round(time.monotonic()-started,1),'capabilities':instance.capabilities}),flush=True)
        sock=listener(0)
        server=LocalServer(uvicorn.Config(build_app(controller,public=True),access_log=False,log_level='warning'))
        task=asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started: await asyncio.sleep(.05)
        base_url=f'http://127.0.0.1:{sock.getsockname()[1]}/v1'
        from openai import AsyncOpenAI
        async with AsyncOpenAI(base_url=base_url,api_key=controller.api_key,timeout=180) as client:
            content='请只回答 2+3 的计算结果，不要解释。'
            if args.image:
                encoded=base64.b64encode(Path(args.image).read_bytes()).decode()
                content=[{'type':'text','text':'Read the four digits shown in this image. Answer only the digits.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}}]
            body={'model':profile['alias'],'messages':[{'role':'user','content':content}],'max_tokens':256,'temperature':0,'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}}
            output=await client.chat.completions.create(**body)
            text=output.choices[0].message.content
            print(json.dumps({'phase':'generation','backend':instance.backend,'image':bool(args.image),'text':text,'usage':output.usage.model_dump() if output.usage else None},ensure_ascii=False),flush=True)
            assert text and ('4729' in text if args.image else '5' in text), '实际模型结果未符合受控输入'
            if not args.image:
                stream=await client.chat.completions.create(**body,stream=True)
                chunks=[]
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content: chunks.append(chunk.choices[0].delta.content)
                assert '5' in ''.join(chunks), '流式输出未返回受控输入答案'
                print(json.dumps({'phase':'stream','backend':instance.backend,'content_chunks':len(chunks)}),flush=True)
        await controller.runtime.stop(identifier)
        print(json.dumps({'phase':'stopped','backend':instance.backend,'state':instance.state}),flush=True)
        server.should_exit=True
        await task
        server=None
    finally:
        if server: server.should_exit=True
        await management.aclose()
        await controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--backend')
    parser.add_argument('--repo')
    parser.add_argument('--image')
    asyncio.run(main(parser.parse_args()))
