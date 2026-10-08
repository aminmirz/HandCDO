"""Serve the website and a loopback-only generation_v2 Blender API.

python -B website_assets/scripts/serve_generator.py --port 8080
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse
from generator_config import ROOT, validate
from build_v2_library import find_blender, generate


class GeneratorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, blender):
        super().__init__(('127.0.0.1', port), Handler)
        self.blender = blender
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.lock = threading.Lock()
        self.origin = f'http://127.0.0.1:{self.server_port}'
        self.allowed_hosts = {f'127.0.0.1:{self.server_port}', f'localhost:{self.server_port}'}
        digest = hashlib.sha256()
        sources = list((ROOT.parent / 'generation_v2').rglob('*.py')) + list((ROOT.parent / 'generation_v2').rglob('*.blend'))
        sources += [ROOT / 'scripts/export_v2_model.py', ROOT / 'scripts/generator_config.py']
        for source in sorted(sources):
            digest.update(source.read_bytes())
        self.source_version = digest.hexdigest()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def reply(self, status, data, content_type='application/json'):
        body = (json.dumps(data) if content_type == 'application/json' else data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        if self.headers.get('Host') not in self.server.allowed_hosts:
            self.reply(403, {'error': 'This generator accepts loopback requests only.'})
            return False
        return True

    def do_GET(self):
        if not self.valid_host():
            return
        path = urlparse(self.path).path
        if path == '/site/generator-config.js':
            self.reply(200, 'window.HandCDOGenerator = ' + json.dumps({'endpoint': '/api/generate', 'token': self.server.token}) + ';', 'text/javascript')
            return
        if path.startswith('/api/jobs/'):
            job_id = path.rsplit('/', 1)[-1]
            job = self.server.jobs.get(job_id)
            self.reply(200 if job else 404, job or {'error': 'Unknown generation job.'})
            return
        if path.startswith('/generated/'):
            parts = path.strip('/').split('/')
            if len(parts) != 3 or len(parts[1]) != 24 or any(c not in '0123456789abcdef' for c in parts[1]) or parts[2] not in ('model.js', 'metadata.json', 'parameters.json', 'configuration.zip'):
                self.reply(404, {'error': 'Unknown generated asset.'})
                return
            file = ROOT / '.runtime/models' / parts[1] / parts[2]
            if not file.is_file():
                self.reply(404, {'error': 'Model is not available.'})
                return
            self.send_response(200)
            self.send_header('Content-Type', self.guess_type(str(file)))
            self.send_header('Content-Length', str(file.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with file.open('rb') as source:
                self.copyfile(source, self.wfile)
            return
        if path.startswith('/.runtime'):
            self.reply(404, {'error': 'Not found.'})
            return
        super().do_GET()

    def do_POST(self):
        if not self.valid_host():
            return
        origin = self.headers.get('Origin', '')
        if origin != 'http://' + self.headers['Host'] or not secrets.compare_digest(self.headers.get('X-HandCDO-Token', ''), self.server.token):
            self.reply(403, {'error': 'Use the generator from this local website.'})
            return
        if self.path != '/api/generate':
            self.reply(404, {'error': 'Unknown endpoint.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2048:
                raise ValueError('Invalid request size.')
            values = validate(json.loads(self.rfile.read(length)))
        except (ValueError, TypeError, UnicodeError) as error:
            self.reply(400, {'error': str(error)})
            return
        key = hashlib.sha256((self.server.source_version + json.dumps(values, sort_keys=True)).encode()).hexdigest()[:24]
        output = ROOT / '.runtime/models' / key
        base = f'generated/{key}'
        if (output / 'configuration.zip').is_file() and (output / 'model.js').is_file():
            self.server.jobs[key] = {'state': 'complete', 'id': key, 'base': base, 'controls': values}
            self.reply(200, dict(self.server.jobs[key], job=key))
            return
        if not self.server.lock.acquire(blocking=False):
            self.reply(409, {'error': 'Blender is generating another hand. Please wait for it to finish.'})
            return
        self.server.jobs[key] = {'state': 'running'}

        def work():
            try:
                generate(self.server.blender, key, values, output)
                self.server.jobs[key] = {'state': 'complete', 'id': key, 'base': base, 'controls': values}
            except Exception as error:
                print(str(error), file=sys.stderr, flush=True)
                self.server.jobs[key] = {'state': 'failed', 'error': 'The original generator could not build this configuration. Adjust the parameters and retry. See the local server log for details.'}
            finally:
                self.server.lock.release()
        threading.Thread(target=work, daemon=True).start()
        self.reply(202, {'job': key, 'state': 'running'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--blender')
    args = parser.parse_args()
    server = GeneratorServer(args.port, find_blender(args.blender))
    print('HandCDO generation_v2: ' + server.origin, flush=True)
    print('Press Ctrl+C to stop. All generated files remain inside website_assets.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
