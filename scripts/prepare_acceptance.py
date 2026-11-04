"""Prepare the two agreed small HF fixtures and isolated backends via real app jobs."""
import asyncio
import json
from pathlib import Path
import secrets
from modloader.app import Controller
from modloader.models import DownloadRequest


async def main():
    controller=Controller(Path(__file__).resolve().parents[1]/'data',secrets.token_urlsafe(32))
    jobs=[]
    try:
        for backend in ('transformers','vllm'):
            existing=next((r for r in controller.store.all('runtimes') if r['backend']==backend and r['managed']),None)
            if not existing:
                jobs.append(controller.new_job('install',{'backend':backend,'options':{}}))
        for repo in ('Qwen/Qwen3-0.6B','Qwen/Qwen2.5-VL-3B-Instruct'):
            existing=next((m for m in controller.store.all('models') if m.get('source',{}).get('repo_id')==repo),None)
            if existing: continue
            metadata=await controller.sources.resolve('hf',repo)
            indices=[f.path for f in metadata.files if f.path.endswith('.safetensors.index.json')]
            weights=indices[:1] or [f.path for f in metadata.files if f.path.endswith('.safetensors')]
            request=DownloadRequest(source='hf',repo_id=repo,revision=metadata.revision,files=weights,destination=str(controller.paths.models/repo.replace('/','--')))
            jobs.append(controller.new_job('download',request.model_dump()))
        last={}
        while any(not controller.job_tasks[j['id']].done() for j in jobs):
            for original in jobs:
                job=controller.store.get('jobs',original['id'])
                signature=(job['state'],job.get('phase'),job.get('percent'),job.get('message'))
                if signature!=last.get(job['id']):
                    last[job['id']]=signature
                    print(json.dumps({k:job.get(k) for k in ('id','kind','backend','repo_id','state','phase','message','percent','downloaded_bytes','total_bytes','error')},ensure_ascii=False),flush=True)
            await asyncio.sleep(10)
        await asyncio.gather(*(controller.job_tasks[j['id']] for j in jobs))
        for original in jobs:
            job=controller.store.get('jobs',original['id'])
            print(json.dumps({'final':job['id'],'backend':job.get('backend'),'repo':job.get('repo_id'),'state':job['state'],'error':job.get('error')},ensure_ascii=False),flush=True)
    finally:
        await controller.close()

if __name__=='__main__': asyncio.run(main())
