"""Serve the local AGM 6 chat and run the selected model."""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'models/lite_runtime'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data/_agm_runtime'))
from agm_6_lite_runner import LiteChat
from agm_6_runner import AGM6
from agm_6_tools import ToolRouter


ROOT = Path(__file__).resolve().parent
ASSETS = {'/': 'index.html', '/index.html': 'index.html', '/banner.png': 'banner.png'}
MODELS = {'agm-6': 'AGM 6', 'agm-6-lite': 'AGM 6 Lite'}


class Engine:
    def __init__(self, device):
        self.device = device
        self.lock = threading.Lock()
        self.model = None
        self.name = None
        self.router = None

    def reply(self, name, messages, tools):
        with self.lock:
            if name != self.name:
                self.router = None
                self.model, self.name = None, None
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                if name == 'agm-6':
                    self.model = AGM6(ROOT / 'models/agm-6', self.device)
                else:
                    self.model = LiteChat(ROOT / 'models/agm-6-lite.pt', self.device)
                self.name = name
            if name == 'agm-6':
                if tools:
                    if self.router is None:
                        self.router = ToolRouter(self.model, ROOT / 'models/agm-6/tool-lora.pt')
                    reply = self.router.reply(messages[-1]['content'])
                    if reply is not None:
                        return reply
                return self.model.generate(messages, 192)
            return self.model.generate(messages, 160, tools)


def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object.')
    model, messages = data.get('model'), data.get('messages')
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError('Choose an available model.')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 41 or len(messages) % 2 == 0:
        raise ValueError('Expected a conversation ending in a user message.')
    for index, message in enumerate(messages):
        role = 'assistant' if index % 2 else 'user'
        if (not isinstance(message, dict) or message.get('role') != role
                or not isinstance(message.get('content'), str)
                or not 1 <= len(message['content'].strip()) <= 8000):
            raise ValueError('Invalid message.')
    return model, messages, data.get('tools') is True


class Handler(BaseHTTPRequestHandler):
    def send_payload(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, payload):
        self.send_payload(status, json.dumps(payload).encode(), 'application/json')

    def do_GET(self):
        name = ASSETS.get(self.path)
        if name is None:
            self.send_error(404)
            return
        content_type = 'image/png' if name.endswith('.png') else 'text/html; charset=utf-8'
        self.send_payload(200, (ROOT / 'public' / name).read_bytes(), content_type)

    def do_POST(self):
        if self.path != '/api/chat':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 384_000 or self.headers.get_content_type() != 'application/json':
                raise ValueError('Expected a JSON chat request.')
            model, messages, tools = validate_request(json.loads(self.rfile.read(size)))
        except (ValueError, UnicodeError) as error:
            self.send_json(400, {'error': str(error)})
            return
        try:
            response = self.server.engine.reply(model, messages, tools)
        except (RuntimeError, ValueError, OSError) as error:
            self.log_error('%s', error)
            self.send_json(500, {'error': 'The local model could not generate a reply. See the server terminal.'})
        else:
            self.send_json(200, {'reply': response or 'The model returned an empty reply.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8785)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(4)
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.engine = Engine(device)
        url = f'http://127.0.0.1:{server.server_port}/'
        print(f'AGM 6 Chat · {url} · {device}', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
