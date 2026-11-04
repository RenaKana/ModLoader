"""No-model OpenAI-compatible process fixture for custom-engine wiring checks."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--context-length', type=int)
    parser.add_argument('--api-model-name')
    parser.add_argument('--health-path', default='/v1/models')
    parser.add_argument('--chat-path', default='/v1/chat/completions')
    parser.add_argument('--exit-immediately', action='store_true')
    parser.add_argument('--never-ready', action='store_true')
    parser.add_argument('--reject-stream-usage', action='store_true')
    args = parser.parse_args()
    if args.exit_immediately:
        print('Fixture requested immediate startup failure', flush=True)
        return 9
    if not Path(args.model).exists() or args.host != '127.0.0.1':
        parser.error('fixture requires an existing selected model path and loopback binding')
    if args.context_length is not None and args.context_length <= 0:
        parser.error('context length must be positive')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != args.health_path:
                return self.send_json(404, {'error': {'message': 'unknown fixture path'}})
            if args.never_ready:
                return self.send_json(503, {'error': {'message': 'fixture is not ready'}})
            return self.send_json(200, {'object': 'list', 'data': [{'id': args.api_model_name or 'fixture-custom'}]})

        def do_POST(self):
            if self.path != args.chat_path:
                return self.send_json(404, {'error': {'message': 'unknown fixture path'}})
            payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            if args.api_model_name and payload.get('model') != args.api_model_name:
                return self.send_json(400, {'error': {'message': 'upstream model name was not expanded'}})
            if args.reject_stream_usage and (payload.get('stream_options') or {}).get('include_usage'):
                return self.send_json(400, {'error': {'message': 'fixture does not support include_usage'}})
            if payload.get('fixture_error'):
                return self.send_json(503, {'error': {'message': 'fixture request failure'}})
            model = payload['model']
            usage = {'prompt_tokens': 4, 'completion_tokens': 3, 'total_tokens': 7}
            if not payload.get('stream'):
                return self.send_json(200, {'id': 'fixture-chat', 'object': 'chat.completion', 'model': model,
                                           'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '自定义引擎接线成功。'}, 'finish_reason': 'stop'}],
                                           'usage': usage, 'received': payload})
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            try:
                for text in ('自定义引擎', '接线成功。'):
                    time.sleep(float(payload.get('fixture_delay', .05)))
                    chunk = {'id': 'fixture-chat', 'model': model, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}
                    self.wfile.write(('data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n').encode('utf-8'))
                    self.wfile.flush()
                chunk = {'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode('utf-8'))
                if (payload.get('stream_options') or {}).get('include_usage'):
                    self.wfile.write(('data: ' + json.dumps({'model': model, 'choices': [], 'usage': usage}) + '\n\n').encode('utf-8'))
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    print('Custom-engine no-model fixture listening on loopback', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    raise SystemExit(main())
