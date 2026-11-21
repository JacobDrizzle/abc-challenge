# Serve the k8_3d folder (now ~/Dev/projects/ABC-Challenge/6-visualisation/k8_3d) on 127.0.0.1:8791 with caching off (Safari otherwise keeps stale copies).
# Run from anywhere: python3 tactical/proto/serve.py   ->  http://127.0.0.1:8791/tactical/proto/dive.html
import functools, http.server, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]   # .../k8_3d

class NoStore(http.server.SimpleHTTPRequestHandler):
    # On Windows Python takes file types from the registry, and some PCs list .mjs as text/plain there. Browsers will not run a
    # module script sent as text/plain, so viewer.mjs (the 3D view) never started. This map is checked before the registry.
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript'}

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

http.server.ThreadingHTTPServer(('127.0.0.1', 8791), functools.partial(NoStore, directory=str(ROOT))).serve_forever()
