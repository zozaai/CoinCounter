#!/usr/bin/env python3
"""Browser-based coin-counting labeler (stdlib only).

Usage:
    python3 tools/label_tool.py [--images DIR] [--labels FILE] [--port 8765]

Shows one image at a time; type the number of coins and press Enter.
Labels are saved to labels.json after every entry, so you can stop and resume.
"""
import argparse
import json
import os
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HTML = r"""<!doctype html>
<meta charset="utf-8"><title>Coin labeler</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;font:14px -apple-system,system-ui,sans-serif;background:#14161a;color:#e8eaed;
      display:flex;flex-direction:column;align-items:center;gap:14px;padding:18px}
 #img{width:480px;height:480px;object-fit:contain;background:#000;border-radius:8px}
 #bar{display:flex;gap:10px;align-items:center}
 input{width:110px;font-size:26px;text-align:center;padding:6px;border-radius:8px;
       border:1px solid #3a3f47;background:#1d2026;color:#fff}
 button{font-size:14px;padding:8px 14px;border-radius:8px;border:1px solid #3a3f47;
        background:#252932;color:#e8eaed;cursor:pointer}
 button:hover{background:#2e333d}
 #meta{color:#9aa0a6}
 #prog{width:480px;height:6px;background:#252932;border-radius:3px;overflow:hidden}
 #progf{height:100%;background:#4c8bf5;width:0}
 kbd{background:#252932;border:1px solid #3a3f47;border-radius:4px;padding:1px 5px}
</style>
<div id="meta"></div>
<div id="prog"><div id="progf"></div></div>
<img id="img">
<div id="bar">
  <button onclick="go(-1)">&larr; Prev</button>
  <input id="val" type="number" min="0" step="1" placeholder="coins" autofocus>
  <button onclick="save()">Save &amp; next &rarr;</button>
  <button onclick="go(1)">Skip</button>
</div>
<div id="meta2" style="color:#9aa0a6">
  <kbd>0-9</kbd> type count &nbsp; <kbd>Enter</kbd> save+next &nbsp;
  <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move &nbsp; <kbd>u</kbd> unlabel
</div>
<script>
let S={files:[],labels:{},i:0};
async function load(){
  S = Object.assign(S, await (await fetch('/api/state')).json());
  const first = S.files.findIndex(f => !(f in S.labels));
  S.i = first === -1 ? 0 : first;
  render();
}
function render(){
  const f = S.files[S.i];
  img.src = '/img/' + encodeURIComponent(f);
  const done = Object.keys(S.labels).length;
  meta.textContent = `${S.i+1} / ${S.files.length} — ${f} — labeled ${done}/${S.files.length}`;
  progf.style.width = (100*done/S.files.length) + '%';
  val.value = (f in S.labels) ? S.labels[f] : '';
  val.focus(); val.select();
}
function go(d){ S.i = (S.i + d + S.files.length) % S.files.length; render(); }
async function save(){
  const v = val.value.trim();
  if(v === '' || isNaN(v) || Number(v) < 0){ val.focus(); return; }
  const f = S.files[S.i];
  S.labels[f] = parseInt(v,10);
  await fetch('/api/label',{method:'POST',body:JSON.stringify({file:f,count:S.labels[f]})});
  go(1);
}
async function unlabel(){
  const f = S.files[S.i];
  delete S.labels[f];
  await fetch('/api/label',{method:'POST',body:JSON.stringify({file:f,count:null})});
  render();
}
document.addEventListener('keydown', e=>{
  if(e.key === 'Enter'){ e.preventDefault(); save(); }
  else if(e.key === 'ArrowRight'){ e.preventDefault(); go(1); }
  else if(e.key === 'ArrowLeft'){ e.preventDefault(); go(-1); }
  else if(e.key === 'u' && document.activeElement !== val){ unlabel(); }
});
load();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    images_dir = ""
    labels_path = ""
    files = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, json.dumps({"files": self.files, "labels": read_labels(self.labels_path)}).encode())
        elif path.startswith("/img/"):
            from urllib.parse import unquote
            name = os.path.basename(unquote(path[len("/img/"):]))
            fp = os.path.join(self.images_dir, name)
            if not os.path.isfile(fp):
                self._send(404, b"not found", "text/plain")
                return
            with open(fp, "rb") as fh:
                self._send(200, fh.read(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        labels = read_labels(self.labels_path)
        if req.get("count") is None:
            labels.pop(req["file"], None)
        else:
            labels[req["file"]] = int(req["count"])
        write_labels(self.labels_path, labels)
        print(f"  labeled {len(labels)}/{len(self.files)}", end="\r", flush=True)
        self._send(200, b'{"ok":true}')


def read_labels(path):
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def write_labels(path, labels):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(labels, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(here, "dataset", "images"))
    ap.add_argument("--labels", default=os.path.join(here, "dataset", "labels.json"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.images_dir = args.images
    Handler.labels_path = args.labels
    Handler.files = sorted(
        (f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".jpeg"))),
        key=natural_key,
    )
    if not Handler.files:
        raise SystemExit(f"No .jpg images in {args.images}")

    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(Handler.files)} images | labels -> {args.labels}")
    print(f"Open {url}  (Ctrl+C to stop; progress is saved as you go)")
    webbrowser.open(url)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Labels saved.")


if __name__ == "__main__":
    main()
