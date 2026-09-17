"""Local development dashboard; binds only to loopback."""
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from agent import assess
from advisor import explain

ROOT = Path(__file__).parent
LIMIT = 5 * 1024 * 1024

class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, kind='application/json'):
        if kind == 'application/json':
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get('Host') in (f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host():
            return self.send(403, {'error': 'Invalid host'})
        routes = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'application/javascript'), '/style.css': ('style.css', 'text/css'), '/sample': ('examples/degraded.json', 'application/json')}
        if self.path == '/health':
            return self.send(200, {'status': 'ok', 'ai_enabled': bool(os.getenv('OLLAMA_MODEL'))})
        if self.path not in routes:
            return self.send(404, {'error': 'Not found'})
        file, kind = routes[self.path]
        data = (ROOT / file).read_bytes()
        if kind == 'application/json':
            return self.send(200, json.loads(data))
        self.send(200, data, kind)

    def do_POST(self):
        if not self.valid_host():
            return self.send(403, {'error': 'Invalid host'})
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://localhost:{self.server.server_port}', f'http://127.0.0.1:{self.server.server_port}'):
            return self.send(403, {'error': 'Invalid origin'})
        if self.headers.get_content_type() != 'application/json':
            return self.send(415, {'error': 'Use application/json'})
        if self.path not in ('/assess', '/ask'):
            return self.send(404, {'error': 'Not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= LIMIT:
                return self.send(413, {'error': 'Upload must be between 1 byte and 5 MB'})
            body = json.loads(self.rfile.read(length))
            if self.path == '/assess':
                return self.send(200, assess(body))
            if not isinstance(body, dict):
                raise ValueError('Expected an object')
            report = assess(body.get('document'))
            self.send(200, explain(report, body.get('question'), os.getenv('OLLAMA_MODEL')))
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            self.send(400, {'error': str(error)})
        except Exception:
            self.send(502, {'error': 'Local AI provider unavailable. Check Ollama and the configured model.'})

if __name__ == '__main__':
    print('Open http://127.0.0.1:8081 — local demo, press Ctrl+C to stop')
    ThreadingHTTPServer(('127.0.0.1', 8081), Handler).serve_forever()
