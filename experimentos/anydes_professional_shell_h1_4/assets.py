"""Read-only allowlisted loopback assets; no core endpoints or file browsing."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote
import mimetypes
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / 'experimentos/anydes_presence_lab_v001'
PUBLIC = {'app.js', 'motion.mjs', 'speech-motion.mjs', 'runtime-bridge.mjs',
          'style.css', 'vendor/lucide.min.js'}


def asset_path(url):
    relative = unquote(urlsplit(url).path).lstrip('/')
    path = (LAB / relative).resolve()
    if not path.is_relative_to(LAB.resolve()):
        return None
    library = any(path.is_relative_to((LAB / prefix).resolve()) for prefix in
                  ('node_modules/three/build', 'node_modules/three/examples/jsm'))
    return path if relative in PUBLIC or library else None


def shell_document():
    page = (LAB / 'index.html').read_text(encoding='utf-8')
    css = '''<style>
    html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#09090f}
    .app{height:100vh;min-height:0!important;display:block!important}
    header,footer,.console,.motion-lab,.scene-label,.state-readout,.scene-tools,
    .signal-visual,#motion-debug{display:none!important}
    main#stage{position:absolute;inset:0;min-height:0!important;height:100%!important;
      width:100%;margin:0;flex:none}#scene{width:100%;height:100%}
    </style>'''
    return page.replace('</head>', css + '</head>').encode('utf-8')


class ShellAssetServer:
    def __init__(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.headers.get('Host') != f'127.0.0.1:{owner.port}':
                    self.send_error(403)
                    return
                if urlsplit(self.path).path == '/shell':
                    body, mime = shell_document(), 'text/html; charset=utf-8'
                else:
                    path = asset_path(self.path)
                    if path is None or not path.is_file():
                        self.send_error(404)
                        return
                    body = path.read_bytes()
                    mime = 'text/javascript' if path.suffix in {'.js', '.mjs'} else mimetypes.guess_type(path.name)[0]
                self.send_response(200)
                self.send_header('Content-Type', mime or 'application/octet-stream')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self' data: blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self' http://127.0.0.1:8781; img-src 'self' data: blob:; object-src 'none'; frame-src 'none'")
                self.end_headers()
                self.wfile.write(body)

        self.server = None
        for port in range(8766, 8777):
            try:
                self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
                break
            except OSError:
                continue
        if self.server is None:
            raise OSError('presence_asset_ports_unavailable')
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .1},
                                       name='ShellAssets', daemon=True)
        self.thread.start()
        self.closed = False

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}/shell?mode=runtime'

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
