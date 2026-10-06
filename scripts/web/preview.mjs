// Static files only. Node 18+; no dependency installation or Python required.
import http from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, realpath } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
const rootAlias = resolve(process.argv[2] || 'web/out');
await realpath(rootAlias); // Refuse a missing initial generation before listening.
const port = Number(process.env.PORT || 8008);
const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2',
  '.whl': 'application/zip',
  '.zip': 'application/zip',
  '.npz': 'application/octet-stream',
  '.csv': 'text/csv; charset=utf-8',
};
http
  .createServer(async (req, res) => {
    try {
      // Resolve a generation pointer once per request. Atomic pointer publication
      // cannot change the tree backing a response already being consumed.
      const root = await realpath(rootAlias);
      if (!['GET', 'HEAD'].includes(req.method)) {
        res.writeHead(405, { Allow: 'GET, HEAD' }).end();
        return;
      }
      const requestUrl = new URL(req.url, 'http://localhost');
      let path = resolve(root, '.' + decodeURIComponent(requestUrl.pathname));
      path = await realpath(path);
      if (path !== root && !path.startsWith(root + sep)) {
        res.writeHead(403).end();
        return;
      }
      let info = await stat(path);
      if (info.isDirectory()) {
        // Relative app assets must resolve within the assignment directory.
        if (!requestUrl.pathname.endsWith('/')) {
          res
            .writeHead(308, {
              Location: requestUrl.pathname + '/' + requestUrl.search,
              'Cache-Control': 'no-cache',
              'Content-Length': 0,
            })
            .end();
          return;
        }
        path = await realpath(resolve(path, 'index.html'));
        info = await stat(path);
      }
      if (!path.startsWith(root + sep)) {
        res.writeHead(403).end();
        return;
      }
      let start = 0,
        end = info.size - 1,
        status = 200;
      if (req.headers.range) {
        const match = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range);
        if (
          !match ||
          (start = Number(match[1])) >= info.size ||
          (end = match[2] ? Math.min(Number(match[2]), end) : end) < start
        ) {
          res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }).end();
          return;
        }
        status = 206;
      }
      const headers = {
        'Content-Type': mime[extname(path)] || 'application/octet-stream',
        'Content-Length': Math.max(0, end - start + 1),
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-cache',
        'X-Content-Type-Options': 'nosniff',
      };
      if (status === 206)
        headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
      res.writeHead(status, headers);
      if (req.method === 'HEAD' || !info.size) res.end();
      else {
        const stream = createReadStream(path, { start, end });
        stream.on('error', () => res.destroy());
        stream.pipe(res);
      }
    } catch (error) {
      res.writeHead(error.code === 'ENOENT' ? 404 : 400).end();
    }
  })
  .listen(port, '127.0.0.1', () =>
    console.log(
      `Static preview: http://127.0.0.1:${port}/unit1/ (through /unit7/)`,
    ),
  );
