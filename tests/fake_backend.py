"""Local CPU HTTP fixture for supervision, gateway and cancellation integration tests."""
import argparse
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app=FastAPI()
@app.get('/v1/models')
async def models(): return {'data':[{'id':'fixture'}]}
@app.post('/v1/chat/completions')
async def chat(body:dict):
    if body.get('stream'):
        async def stream():
            for word in ('hello',' ','world'):
                yield 'data: '+json.dumps({'model':'fixture','choices':[{'index':0,'delta':{'content':word}}]})+'\n\n'
                await asyncio.sleep(.15)
            yield 'data: [DONE]\n\n'
        return StreamingResponse(stream(),media_type='text/event-stream')
    return {'id':'fixture','model':'fixture','choices':[{'message':{'role':'assistant','content':'hello world'},'finish_reason':'stop'}]}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int)
    args=parser.parse_args()
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
