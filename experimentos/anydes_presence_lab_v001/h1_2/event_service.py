"""Loopback SSE. Read-only endpoints; bounded observer work off core threads."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import json
from pathlib import Path
import queue
import threading
import time

from state_adapter import PresenceStateAdapter, RuntimeEvent


class PresenceService:
    def __init__(self, evidence, port=8781, motion_provider=None):
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.adapter = PresenceStateAdapter()
        self.port = port
        self.pending = queue.Queue(maxsize=512)
        self.overflow = threading.Event()
        self.stopped = threading.Event()
        self.clients = set()
        self.lock = threading.RLock()
        self.latest = self.adapter.snapshot()
        self.available = False
        self.errors = 0
        self.motion_provider = motion_provider

    def submit(self, event):
        # Producers never wait for HTTP, a browser or disk I/O.
        try:
            self.pending.put_nowait(event)
        except queue.Full:
            self.overflow.set()

    def _log(self, kind, value):
        try:
            with (self.evidence/'presence_events.jsonl').open('a', encoding='utf-8') as out:
                out.write(json.dumps(dict(kind=kind, monotonic=time.monotonic(), **value))+'\n')
        except OSError:
            self.errors += 1

    def _publish(self, frame):
        if frame is None:
            return
        self._log('presence', frame)
        with self.lock:
            self.latest = frame
            for client in tuple(self.clients):
                try:
                    client.put_nowait(frame)
                except queue.Full:
                    # A slow subscriber must reconnect, never retain lost activity.
                    self.clients.discard(client)
                    while not client.empty():
                        try: client.get_nowait()
                        except queue.Empty: break
                    client.put_nowait(None)

    def _run(self):
        while not self.stopped.is_set():
            if self.overflow.is_set():
                self.overflow.clear()
                while not self.pending.empty():
                    try: self.pending.get_nowait()
                    except queue.Empty: break
                self._publish(self.adapter.disconnect())
                self._log('observer_overflow', {})
            try:
                item = self.pending.get(timeout=0.025)
                if item == 'READY': self._publish(self.adapter.ready())
                elif item == 'DISCONNECT': self._publish(self.adapter.disconnect())
                elif isinstance(item, RuntimeEvent):
                    self._log('activity', dict(domain=item.domain.value, phase=item.phase,
                                              activity_id=item.activity_id, task_id=item.task_id))
                    self._publish(self.adapter.apply(item))
            except queue.Empty:
                pass
            self._publish(self.adapter.advance())
        self._publish(self.adapter.disconnect())

    def start(self):
        owner = self
        def current_frame():
            with owner.lock: frame = dict(owner.latest)
            frame['timestamp'] = owner.adapter.snapshot()['timestamp']
            if not owner.worker.is_alive():
                frame.update(state='IDLE',source='runtime.disconnected',task_id=None,connected=False)
            return frame
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args): pass

            def do_GET(self):
                origin = self.headers.get('Origin')
                try:
                    parsed = urlparse(origin or '')
                    allowed = (parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost'}
                               and parsed.port in range(8766, 8777) and not parsed.username and parsed.path in {'', '/'})
                except ValueError:
                    allowed = False
                if self.headers.get('Host') not in {f'127.0.0.1:{owner.port}', f'localhost:{owner.port}'} or (origin and not allowed):
                    self.send_error(403); return
                if self.path not in {'/events', '/health'}:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                if origin:
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')
                self.send_header('Content-Type', 'text/event-stream' if self.path == '/events' else 'application/json')
                self.send_header('Connection', 'close')
                if self.path == '/health':
                    data = json.dumps(current_frame()).encode()
                    self.send_header('Content-Length', str(len(data))); self.end_headers()
                    self.wfile.write(data); return
                self.end_headers()
                client = queue.Queue(maxsize=32)
                with owner.lock:
                    owner.clients.add(client)
                    client.put_nowait(current_frame())
                try:
                    self.wfile.write(b'retry: 1500\n\n')
                    heartbeat = 0.0
                    while not owner.stopped.is_set():
                        try: frame = client.get(timeout=0.04 if owner.motion_provider else 1)
                        except queue.Empty:
                            frame = current_frame() if time.monotonic()-heartbeat >= 1 else False
                        if frame is None: break
                        if frame is not False:
                            self.wfile.write(('data: '+json.dumps(frame)+'\n\n').encode())
                            heartbeat = time.monotonic()
                        if owner.motion_provider:
                            motion = owner.motion_provider(current_frame())
                            self.wfile.write(('event: motion\ndata: '+json.dumps(motion)+'\n\n').encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with owner.lock: owner.clients.discard(client)

        self.server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.worker = threading.Thread(target=self._run, name='PresenceProjection')
        self.http = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.1), name='PresenceSSE')
        self.worker.start(); self.http.start(); self.available = True
        return self

    def close(self):
        if not self.available: return
        self.stopped.set(); self.worker.join(2)
        self.server.shutdown(); self.server.server_close(); self.http.join(2)
        self._log('observer_shutdown', dict(projection_alive=self.worker.is_alive(), http_alive=self.http.is_alive(), errors=self.errors))
        self.available = False
