"""Read-only browser for saved predictions: python -m benchmark.vis --dataset dataset --engine hough.

The viewer never imports an engine or runs inference. Counts, detections, and timings come
from prediction records written by benchmark.compute. Grayscale and blur stages are display
reconstructions derived with Pillow from the saved manifest parameters.
"""
import argparse
import html
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

from PIL import Image, ImageDraw, ImageFilter

from .cache import file_hash, identity
from .dataset import samples

ROOT = Path(__file__).resolve().parents[1]


class Viewer:
    """Resolves dataset images to saved prediction records and renders display stages."""

    def __init__(self, dataset, engine, results, split=None):
        self.dataset = Path(dataset)
        self.engine = engine
        self.results = Path(results)
        self.split = split
        self.items = samples(self.dataset, split)
        self.dataset_id = identity(str(self.dataset.resolve()))[:12]
        self.selection = split or "all"
        self.configurations = self._configurations()
        self._hashes = {}

    def _configurations(self):
        folder = self.results / self.engine
        found = {}
        if folder.is_dir():
            for manifest in sorted(folder.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    found[manifest.parent.name] = json.loads(manifest.read_text())
                except (OSError, ValueError):
                    continue
        return found

    def default_configuration(self):
        return next(iter(self.configurations), None)

    def index_of(self, image):
        name = Path(image).name
        for index, (path, _) in enumerate(self.items):
            if path.name == name:
                return index
        raise ValueError(f"{name} is not in the browsed dataset selection")

    def image_hash(self, path):
        if path not in self._hashes:
            self._hashes[path] = file_hash(path)
        return self._hashes[path]

    def record(self, index, configuration):
        if configuration not in self.configurations:
            return None
        path, _ = self.items[index]
        base = self.results / self.engine / configuration / self.dataset_id
        name = f"{path.name}-{self.image_hash(path)[:12]}.json"
        candidates = [base / self.selection / name] + sorted(base.glob(f"*/{name}"))
        for candidate in candidates:
            try:
                return json.loads(candidate.read_text())
            except (OSError, ValueError):
                continue
        return None

    def stage_names(self, configuration, record):
        names = ["original"]
        if self.engine == "hough":
            names += ["grayscale", "blur"]
        if record is None or record.get("detections") is not None or self.engine == "hough":
            names.append("detections")
        return names

    def item(self, index, configuration=None):
        configuration = configuration or self.default_configuration()
        path, actual = self.items[index]
        record = self.record(index, configuration) if configuration else None
        messages = []
        if not self.configurations:
            messages.append(f"No saved results for engine '{self.engine}' under {self.results}. "
                            f"Run benchmark.compute first; the viewer never runs inference.")
        elif configuration not in self.configurations:
            messages.append(f"Unknown configuration '{configuration}'.")
        elif record is None:
            messages.append(f"No saved prediction for {path.name} in configuration {configuration}.")
        predicted = record["count"] if record else None
        return {
            "index": index, "total": len(self.items), "filename": path.name,
            "configuration": configuration, "configurations": list(self.configurations),
            "actual": actual, "predicted": predicted,
            "error": None if predicted is None or actual is None else predicted - actual,
            "seconds": record.get("seconds") if record else None,
            "detections": record.get("detections") if record else None,
            "stages": self.stage_names(configuration, record), "messages": messages,
            "parameters": self.configurations.get(configuration, {}).get("parameters", {}),
        }

    def stage(self, index, name, configuration=None):
        """Return PNG bytes, or a message string when the stage cannot be shown."""
        configuration = configuration or self.default_configuration()
        path, _ = self.items[index]
        image = Image.open(path).convert("RGB")
        if name == "original":
            return _png(image)
        parameters = self.configurations.get(configuration, {}).get("parameters", {})
        if name == "grayscale":
            return _png(image.convert("L"))
        if name == "blur":
            size = parameters.get("blur_size")
            if type(size) is not int or size < 3 or size % 2 == 0:
                return "Blur stage unavailable: blur_size is not recorded in the manifest."
            return _png(image.convert("L").filter(ImageFilter.MedianFilter(size)))
        if name == "detections":
            record = self.record(index, configuration) if configuration else None
            if record is None:
                return "Detections unavailable: no saved prediction for this image."
            detections = record.get("detections")
            if detections is None:
                return "Detections were not recorded by this engine."
            return _png(_overlay(image, detections))
        return f"Unknown stage '{name}'."


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _overlay(image, detections):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, round(min(canvas.size) / 160))
    for detection in detections:
        if all(k in detection for k in ("x", "y", "radius")):
            x, y, r = detection["x"], detection["y"], detection["radius"]
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 40, 40), width=width)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 40, 40))
        elif all(k in detection for k in ("x1", "y1", "x2", "y2")):
            draw.rectangle([detection["x1"], detection["y1"], detection["x2"], detection["y2"]],
                           outline=(255, 40, 40), width=width)
    return canvas


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CoinCounter results viewer</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;padding:16px 24px;background:#f5f5f4;color:#1c1917}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;margin-bottom:12px}
h1{font-size:18px;margin:0}
button{padding:6px 14px;font-size:14px}
button:disabled{opacity:.4}
.facts{display:flex;flex-wrap:wrap;gap:8px 24px;margin:8px 0 16px}
.facts b{display:block;font-size:11px;text-transform:uppercase;color:#78716c}
.stages{display:flex;flex-wrap:wrap;gap:16px}
figure{margin:0;width:320px}
figure img{width:100%;image-rendering:auto;background:#e7e5e4;display:block}
figcaption{font-size:13px;margin-top:4px;color:#44403c}
.missing{width:100%;aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#e7e5e4;color:#78716c;font-size:13px;padding:12px;box-sizing:border-box;text-align:center}
.message{background:#fef3c7;border:1px solid #f59e0b;padding:8px 12px;margin:8px 0;font-size:14px}
.err-pos{color:#b91c1c}.err-neg{color:#1d4ed8}.err-zero{color:#15803d}
</style></head><body>
<header>
<h1>CoinCounter · <span id="engine"></span></h1>
<span><button id="prev">&larr; Previous</button> <span id="position"></span> <button id="next">Next &rarr;</button></span>
<label id="configlabel" hidden>Configuration <select id="config"></select></label>
</header>
<div id="messages"></div>
<div class="facts">
<span><b>Filename</b><span id="filename"></span></span>
<span><b>Ground truth</b><span id="actual"></span></span>
<span><b>Predicted</b><span id="predicted"></span></span>
<span><b>Error</b><span id="error"></span></span>
<span><b>Recorded inference</b><span id="seconds"></span></span>
</div>
<div class="stages" id="stages"></div>
<script>
const state = {index: __INDEX__, total: __TOTAL__, config: __CONFIG__};
document.getElementById("engine").textContent = __ENGINE__;
const $ = id => document.getElementById(id);
async function load() {
  const q = new URLSearchParams({index: state.index});
  if (state.config) q.set("config", state.config);
  const item = await (await fetch("/api/item?" + q)).json();
  state.total = item.total; state.config = item.configuration;
  $("position").textContent = `${item.index + 1} / ${item.total}`;
  $("prev").disabled = item.index === 0;
  $("next").disabled = item.index >= item.total - 1;
  $("filename").textContent = item.filename;
  $("actual").textContent = item.actual ?? "—";
  $("predicted").textContent = item.predicted ?? "—";
  const e = $("error");
  e.textContent = item.error == null ? "—" : (item.error > 0 ? "+" : "") + item.error;
  e.className = item.error == null ? "" : item.error > 0 ? "err-pos" : item.error < 0 ? "err-neg" : "err-zero";
  $("seconds").textContent = item.seconds == null ? "—" : (item.seconds * 1000).toFixed(3) + " ms";
  $("messages").innerHTML = item.messages.map(m => `<div class="message">${m}</div>`).join("");
  const select = $("config");
  if (item.configurations.length > 1) {
    $("configlabel").hidden = false;
    select.innerHTML = item.configurations.map(c => `<option ${c === item.configuration ? "selected" : ""}>${c}</option>`).join("");
  }
  const stages = $("stages"); stages.innerHTML = "";
  for (const name of item.stages) {
    const fig = document.createElement("figure");
    const sq = new URLSearchParams({index: item.index});
    if (item.configuration) sq.set("config", item.configuration);
    const url = `/stage/${name}?` + sq;
    const r = await fetch(url);
    if (r.headers.get("content-type").startsWith("image/")) {
      const img = document.createElement("img"); img.src = URL.createObjectURL(await r.blob()); fig.appendChild(img);
    } else {
      const div = document.createElement("div"); div.className = "missing"; div.textContent = await r.text(); fig.appendChild(div);
    }
    const cap = document.createElement("figcaption");
    cap.textContent = name === "detections" && item.detections ? `detections (${item.detections.length} circles)` : name;
    fig.appendChild(cap); stages.appendChild(fig);
  }
}
function go(delta) { const n = state.index + delta; if (n < 0 || n >= state.total) return; state.index = n; load(); }
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
$("config").onchange = ev => { state.config = ev.target.value; load(); };
document.addEventListener("keydown", ev => { if (ev.key === "ArrowLeft") go(-1); if (ev.key === "ArrowRight") go(1); });
load();
</script></body></html>
"""


def make_handler(viewer, start):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                index = int(query.get("index", start))
                if not 0 <= index < len(viewer.items):
                    raise ValueError("index out of range")
                configuration = query.get("config") or None
                if url.path == "/":
                    page = (PAGE.replace("__INDEX__", str(index)).replace("__TOTAL__", str(len(viewer.items)))
                            .replace("__CONFIG__", json.dumps(configuration or viewer.default_configuration()))
                            .replace("__ENGINE__", json.dumps(viewer.engine)))
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/item":
                    self._send(200, json.dumps(viewer.item(index, configuration)).encode(), "application/json")
                elif url.path.startswith("/stage/"):
                    result = viewer.stage(index, url.path[len("/stage/"):], configuration)
                    if isinstance(result, bytes):
                        self._send(200, result, "image/png")
                    else:
                        self._send(200, result.encode(), "text/plain; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            except (ValueError, IndexError) as exc:
                self._send(400, html.escape(str(exc)).encode(), "text/plain; charset=utf-8")
    return Handler


def serve(viewer, start, host="127.0.0.1", port=0):
    server = ThreadingHTTPServer((host, port), make_handler(viewer, start))
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--image", type=Path, help="Start at this image (must belong to the browsed selection)")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args(argv)
    try:
        viewer = Viewer(args.dataset, args.engine, args.results, args.split)
        start = viewer.index_of(args.image) if args.image else 0
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if not viewer.configurations:
        print(f"No saved results for '{args.engine}' under {args.results}; the page will show missing-result messages.",
              file=sys.stderr)
    server = serve(viewer, start, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/?index={start}"
    print(f"Viewing {len(viewer.items)} images at {url} (Ctrl+C to stop)", file=sys.stderr)
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
