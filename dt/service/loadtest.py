"""Measurements behind docs/SERVICE.md (run from the repo root; each mode starts / stops its own service
under a private DT_HOME, so nothing touches a service you already run).

    python -m dt.service.loadtest determinism [--workers 2]
    python -m dt.service.loadtest throughput --clients 6 --renders 40 --modes local,service:3,service:5
    python -m dt.service.loadtest cache [--stages render] [--bench-workers 3]

* determinism: every corpus doc (synth + Material Web gt IR) via render_html, every Material Web corpus
  page via render_url, an icon batch page (Material Symbols atlas layout) and font batch pages
  (every self-hosted family) — service bytes vs local bytes.
* throughput: N client processes x M ``render_doc`` calls of mixed corpus docs, started together.
  ``local`` = each process renders with its own Chrome (today's behaviour); ``service:K`` = a K-worker
  service (cache off, so it measures rendering, not the cache; ``service:K+cache`` turns it on). Reports renders/s, p50/p95 latency,
  peak Chrome processes (descendants of this process, sampled every 0.2 s), failures, local fallbacks.
* cache: the same bench pass twice through a service with a cold cache; hit rate and wall-time speedup.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPORA = [os.path.join(ROOT, "fixtures", "corpus", "synth"), os.path.join(ROOT, "fixtures", "corpus", "mwc")]


def _corpus_docs() -> list[str]:
    out = []
    for d in CORPORA:
        out += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".gt.json"))
    return out


def _load() -> str:
    return " ".join(f"{x:.2f}" for x in os.getloadavg())


def _pcts(v: list[float]) -> dict:
    if not v:
        return {}
    s = sorted(v)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return {"p50_ms": round(q(0.5) * 1000, 1), "p95_ms": round(q(0.95) * 1000, 1), "max_ms": round(s[-1] * 1000, 1),
            "mean_ms": round(sum(s) / len(s) * 1000, 1)}


# --------------------------------------------------------------------------- process sampling
def chrome_count(root_pid: int) -> int:
    out = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    kids: dict[int, list[int]] = {}
    cmd: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        kids.setdefault(ppid, []).append(pid)
        cmd[pid] = parts[2]
    n, stack, seen = 0, [root_pid], set()
    while stack:
        for c in kids.get(stack.pop(), []):
            if c in seen:
                continue
            seen.add(c)
            stack.append(c)
            low = cmd.get(c, "").lower()
            if "chrome" in low or "chromium" in low:
                n += 1
    return n


class Sampler:
    def __init__(self, period: float = 0.2):
        self.peak = 0
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.period = period

    def _run(self) -> None:
        while not self._stop.is_set():
            c = chrome_count(os.getpid())
            self.samples.append(c)
            self.peak = max(self.peak, c)
            self._stop.wait(self.period)

    def __enter__(self) -> "Sampler":
        self._t.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        self._t.join(5)


# --------------------------------------------------------------------------- service process
class ServiceProc:
    def __init__(self, workers: int, cache: bool, home: Optional[str] = None):
        self._own_home = home is None
        self.home = home or tempfile.mkdtemp(prefix="dtsvc_")
        self.workers, self.cache = workers, cache
        self.proc: Optional[subprocess.Popen] = None
        self.env: dict[str, str] = {}

    def __enter__(self) -> "ServiceProc":
        env = dict(os.environ, DT_HOME=self.home)
        env.pop("DT_EVAL_SERVICE", None)
        argv = [sys.executable, "-m", "dt.cli", "serve", "--workers", str(self.workers), "--port", "0"] + ([] if self.cache else ["--no-cache"])
        self.log = open(os.path.join(self.home, "service.log"), "ab")
        self.proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=self.log, stderr=self.log)
        state = os.path.join(self.home, "service.json")
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"service exited {self.proc.returncode}; log {self.log.name}")
            try:
                st = json.load(open(state))
                if st.get("pid") == self.proc.pid:
                    self.env = {"DT_HOME": self.home, "DT_EVAL_SERVICE": st["url"], "DT_EVAL_SERVICE_TOKEN": st["token"]}
                    return self
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        raise RuntimeError("service did not start")

    def stats(self) -> dict:
        from dt.service import client
        return client.stats(client.parse_url(self.env["DT_EVAL_SERVICE"], self.env["DT_EVAL_SERVICE_TOKEN"]))

    def __exit__(self, *a: Any) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()
        if self._own_home:
            import shutil
            shutil.rmtree(self.home, ignore_errors=True)


# --------------------------------------------------------------------------- throughput
def _client(idx: int, paths: list[str], barrier, q) -> None:
    from dt.ir import Document
    from dt.render import screenshot as s
    docs = [Document.load(p) for p in paths]
    lat: list[float] = []
    errors: list[str] = []
    barrier.wait()
    t0 = time.time()
    for d in docs:
        t = time.perf_counter()
        try:
            s.render_doc(d)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}"[:200])
        lat.append(time.perf_counter() - t)
    t1 = time.time()
    q.put({"idx": idx, "lat": lat, "errors": errors, "t0": t0, "t1": t1, "local_browser": s._st()["browser"] is not None})
    s.shutdown()


def run_clients(n_clients: int, n_renders: int, env: dict, seed: int = 0) -> dict:
    pool = _corpus_docs()
    ctx = multiprocessing.get_context("spawn")
    barrier, q = ctx.Barrier(n_clients + 1), ctx.Queue()
    old = {k: os.environ.get(k) for k in ("DT_EVAL_SERVICE", "DT_EVAL_SERVICE_TOKEN", "DT_HOME")}
    for k in old:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        procs = []
        for i in range(n_clients):
            rng = random.Random(seed * 1000 + i)
            procs.append(ctx.Process(target=_client, args=(i, [rng.choice(pool) for _ in range(n_renders)], barrier, q)))
        for p in procs:
            p.start()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    load0 = _load()
    with Sampler() as smp:
        barrier.wait(timeout=600)
        res = [q.get(timeout=1800) for _ in procs]
    for p in procs:
        p.join(60)
    lat = [x for r in res for x in r["lat"]]
    wall = max(r["t1"] for r in res) - min(r["t0"] for r in res)
    fails = sum(len(r["errors"]) for r in res)
    return {"clients": n_clients, "renders": len(lat), "wall_s": round(wall, 2), "renders_per_s": round(len(lat) / wall, 2),
            **_pcts(lat), "peak_chrome": smp.peak, "failures": fails, "errors": [e for r in res for e in r["errors"]][:5],
            "clients_with_local_browser": sum(r["local_browser"] for r in res), "load_start": load0, "load_end": _load()}


def cmd_throughput(a: argparse.Namespace) -> dict:
    out: dict[str, Any] = {}
    for mode in a.modes.split(","):
        if mode == "local":
            r = run_clients(a.clients, a.renders, {}, a.seed)
        else:
            k = int(mode.split(":")[1].split("+")[0])
            with ServiceProc(k, cache=mode.endswith("+cache")) as sp:
                r = run_clients(a.clients, a.renders, sp.env, a.seed)
                st = sp.stats()
                r["service"] = {"workers": k, "counters": st["pool"]["counters"], "job_latency": st.get("latency")}
        out[mode] = r
        print(f"{mode:10s} {json.dumps({k: v for k, v in r.items() if k not in ('errors', 'service')})}", file=sys.stderr, flush=True)
    return out


# --------------------------------------------------------------------------- determinism
def cmd_determinism(a: argparse.Namespace) -> dict:
    from dt.ir import Document
    from dt.render import screenshot as s
    from dt.render.html import FONTS_DIR, _font_face_css, render_html
    from dt.service import client
    from dt.perceive.icons import VARIABLE_FONT, _read_names
    res: dict[str, Any] = {}
    with ServiceProc(a.workers, cache=False) as sp:
        addr = client.parse_url(sp.env["DT_EVAL_SERVICE"], sp.env["DT_EVAL_SERVICE_TOKEN"])

        def svc_html(html, w, h):
            return client.call("render_html", {"html": html, "width": w, "height": h, "tmp_dir": s._TMP_DIR}, addr=addr)[1]

        def svc_url(url, w, h, wait_ms, script):
            hdr, png = client.call("render_url", {"url": url, "width": w, "height": h, "wait_ms": wait_ms, "script": script,
                                                  "wait_until": "load"}, addr=addr)
            return png, hdr.get("result")
        docs = _corpus_docs()
        same = 0
        diff: list[str] = []
        for p in docs:
            d = Document.load(p)
            html = render_html(d)
            if svc_html(html, d.width, d.height) == s._html_to_png_bytes(html, d.width, d.height):
                same += 1
            else:
                diff.append(os.path.basename(p))
        res["corpus_docs"] = {"n": len(docs), "identical": same, "different": diff}
        mwc = sorted(os.path.join(CORPORA[1], f) for f in os.listdir(CORPORA[1]) if f.endswith(".html"))
        same, diff = 0, []
        script = "(() => [document.querySelectorAll('*').length, document.fonts.status])()"
        for p in mwc:
            d = Document.load(p.replace(".html", ".gt.json"))
            a1 = svc_url("file://" + p, d.width, d.height, 600, script)
            b1 = s._render_url_bytes("file://" + p, d.width, d.height, 600, script, "load")
            if a1[0] == b1[0] and a1[1] == b1[1]:
                same += 1
            else:
                diff.append(os.path.basename(p))
        res["material_web_pages"] = {"n": len(mwc), "identical": same, "different": diff}
        # icon batch pages: the atlas layout of dt.perceive.icons.build_atlas, 400 glyphs per page
        names = _read_names()
        cols, cell = 40, 48
        same, n = 0, 0
        for fill in (0, 1):
            for start in (0, 400, 800):
                chunk = names[start:start + 400]
                rows = (len(chunk) + cols - 1) // cols
                spans = "".join(f'<span class=g style="left:{(i % cols) * cell}px;top:{(i // cols) * cell}px">{nm}</span>'
                                for i, nm in enumerate(chunk))
                html = (f"<!doctype html><meta charset=utf-8><style>@font-face{{font-family:'MSOV';src:url('file://{VARIABLE_FONT}') "
                        f"format('woff2');}}html,body{{margin:0;background:#fff}}.g{{position:absolute;width:{cell}px;height:{cell}px;"
                        "display:flex;align-items:center;justify-content:center;font-family:'MSOV';font-size:32px;line-height:1;"
                        f"color:#000;font-variation-settings:'FILL' {fill},'wght' 400,'GRAD' 0,'opsz' 24;}}</style>{spans}")
                tmp = s._tmp_html(html)
                ok = svc_url("file://" + tmp, cols * cell, rows * cell, 400, None)[0] == \
                    s._render_url_bytes("file://" + tmp, cols * cell, rows * cell, 400, None, "load")[0]
                same += ok
                n += 1
        res["icon_batch_pages"] = {"n": n, "identical": same}
        # font batch pages: every family declared in the self-hosted CSS, 4 weights x 3 sizes
        import re
        fams = sorted(set(re.findall(r"font-family:\s*'([^']+)'", _font_face_css())))
        same, n = 0, 0
        for fam in fams:
            divs, y = [], 4
            for w in (400, 500, 700, 300):
                for px in (12, 16, 24):
                    divs.append(f"<div style=\"position:absolute;left:6px;top:{y}px;font:{w} {px}px '{fam}';white-space:nowrap\">"
                                "Sphinx of black quartz, judge my vow 0123456789 ÄÖÜ</div>")
                    y += int(px * 1.6) + 4
            html = f"<!doctype html><meta charset=utf-8><style>{_font_face_css()} body{{margin:0;background:#fff}}</style>{''.join(divs)}"
            same += svc_html(html, 640, y + 4) == s._html_to_png_bytes(html, 640, y + 4)
            n += 1
        res["font_batch_pages"] = {"n": n, "identical": same, "families": fams}
        _ = FONTS_DIR
    s.shutdown()
    return res


# --------------------------------------------------------------------------- cache
def cmd_cache(a: argparse.Namespace) -> dict:
    from dt.selftest import bench
    stages = tuple(a.stages.split(","))
    out: dict[str, Any] = {"stages": list(stages), "bench_workers": a.bench_workers}
    t = time.perf_counter()
    rep = bench.run(stages=stages, workers=a.bench_workers, out_root=None)
    out["local"] = {"wall_s": round(time.perf_counter() - t, 2), "composite": rep["composite"], "load": _load()}
    with ServiceProc(a.workers, cache=True) as sp:
        old = {k: os.environ.get(k) for k in sp.env}
        os.environ.update(sp.env)
        try:
            for name in ("cold", "warm"):
                s0 = sp.stats()["cache"]
                t = time.perf_counter()
                rep = bench.run(stages=stages, workers=a.bench_workers, out_root=None)
                wall = time.perf_counter() - t
                s1 = sp.stats()["cache"]
                hits, misses = s1["hits"] - s0["hits"], s1["misses"] - s0["misses"]
                out[name] = {"wall_s": round(wall, 2), "composite": rep["composite"], "render_lookups": hits + misses, "hits": hits,
                             "hit_rate": round(hits / max(1, hits + misses), 4), "render_s_sum": round(sum(
                                 r["timing"].get("render_s", 0) for r in rep["cases"]), 2), "load": _load()}
                print(f"{name}: {out[name]}", file=sys.stderr, flush=True)
            out["cache_stats"] = sp.stats()["cache"]
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    out["speedup_warm_vs_cold"] = round(out["cold"]["wall_s"] / max(1e-9, out["warm"]["wall_s"]), 2)
    out["speedup_warm_vs_local"] = round(out["local"]["wall_s"] / max(1e-9, out["warm"]["wall_s"]), 2)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dt.service.loadtest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("throughput")
    p.add_argument("--clients", type=int, default=6); p.add_argument("--renders", type=int, default=40)
    p.add_argument("--modes", default="local,service:3,service:5"); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("determinism")
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("cache")
    p.add_argument("--stages", default="render"); p.add_argument("--bench-workers", type=int, default=3)
    p.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    t = time.time()
    res = {"throughput": cmd_throughput, "determinism": cmd_determinism, "cache": cmd_cache}[a.cmd](a)
    res = {"cmd": a.cmd, "elapsed_s": round(time.time() - t, 1), "cpu": os.cpu_count(), "load_after": _load(), "result": res}
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
