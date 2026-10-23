import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';

const root = path.dirname(fileURLToPath(import.meta.url));
let port = Number(process.env.PORT || 8766);
let retries = 0;
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' };
const server = http.createServer(async (request, response) => {
  try {
    if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405, { Allow: 'GET, HEAD' }).end(); return; }
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const target = path.resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!target.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    const publicPath = '/' + path.relative(root, target).split(path.sep).join('/');
    const publicFiles = ['/index.html', '/app.js', '/style.css', '/motion.mjs', '/speech-motion.mjs', '/runtime-bridge.mjs', '/vendor/lucide.min.js'];
    const libraryAsset = publicPath.startsWith('/node_modules/three/build/') || publicPath.startsWith('/node_modules/three/examples/jsm/');
    if (!publicFiles.includes(publicPath) && !libraryAsset) { response.writeHead(404).end('Not found'); return; }
    const data = await readFile(target);
    response.writeHead(200, { 'Content-Type': types[path.extname(target)] || 'application/octet-stream', 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff' });
    response.end(data);
  } catch { response.writeHead(404).end('Not found'); }
});
server.on('error', error => {
  if (error.code === 'EADDRINUSE' && retries++ < 10) { port++; server.listen(port, '127.0.0.1'); }
  else { console.error(error.message); process.exitCode = 1; }
});
server.on('listening', () => {
  const url = `http://127.0.0.1:${server.address().port}`;
  console.log(`ANYDES presence lab: ${url}`);
  if (process.argv.includes('--open') && process.platform === 'win32') {
    execFile('rundll32.exe', ['url.dll,FileProtocolHandler', url], { windowsHide: true }, error => { if (error) console.log(`Abre ${url}`); });
  }
});
server.listen(port, '127.0.0.1');
