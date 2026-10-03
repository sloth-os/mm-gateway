// Portable Caddy handle_path equivalent for browser tests without Caddy installed.
import http from "node:http";

const prefix = "/nested/gateway";
http
  .createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1:8766");
    if (url.pathname === prefix) {
      response.writeHead(308, { Location: `${prefix}/admin/` }).end();
      return;
    }
    if (!url.pathname.startsWith(`${prefix}/`)) {
      response.writeHead(404).end("Not found");
      return;
    }
    const upstream = http.request(
      {
        hostname: "127.0.0.1",
        port: 8767,
        path: request.url.slice(prefix.length),
        method: request.method,
        headers: request.headers,
      },
      (result) => {
        response.writeHead(result.statusCode, result.headers);
        result.pipe(response);
      },
    );
    upstream.on("error", () => response.writeHead(502).end());
    request.pipe(upstream);
  })
  .listen(8766, "127.0.0.1");
