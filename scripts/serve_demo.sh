#!/usr/bin/env bash
# Serve the demo (simulator) alone: http://localhost:${PORT:-8000}/  (also at /sim/, same as the console)
# The page fetches data.json, so it needs an HTTP server; opening index.html as a file won't work.
set -euo pipefail
cd "$(dirname "$0")/../demo"
PORT=${PORT:-8000}
echo "demo: http://localhost:${PORT}/  (remote server: ssh -L ${PORT}:localhost:${PORT} <host>)"
exec python3 - "$PORT" "${BIND:-127.0.0.1}" <<'PY'
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/sim":
            self.send_response(301); self.send_header("Location", "sim/"); self.end_headers(); return
        if self.path.startswith("/sim/"):
            self.path = self.path[4:]
        return super().do_GET()
    def log_message(self, *a): pass
ThreadingHTTPServer((sys.argv[2], int(sys.argv[1])), H).serve_forever()
PY
