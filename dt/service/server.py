"""The evaluation service: a localhost-only HTTP front over a fixed pool of worker processes.

    from dt.service.server import EvalService
    svc = EvalService(workers=4, port=47615)      # or socket_path="~/.rsdesign/eval.sock"
    svc.start(); svc.serve_forever()              # `dt serve` does exactly this

Process model
-------------
* Each worker is a ``spawn`` process that owns one Playwright browser through
  :mod:`dt.render.screenshot` (the same code local renders use, so outputs are byte-identical). It
  drops ``DT_EVAL_SERVICE`` from its environment first, so nothing it runs (perceive's font/icon
  renders, bench cases) ever calls back into the pool it occupies.
* Admission: at most ``workers + service.queue_max`` jobs are in the service at once; the rest get
  HTTP 503 (clients back off and retry). An admitted job waits up to ``service.queue_wait_s`` for an
  idle worker.
* A job that exceeds its timeout (``service.job_timeout_s``) or whose worker dies is retried once on
  a freshly spawned worker; the stuck worker is killed together with its whole process tree
  (driver + Chrome). A failed retry is reported as a worker error.
* A worker is recycled (graceful stop, fresh process + browser) after
  ``service.max_jobs_per_worker`` jobs. Workers exit on their own when the service process dies.

Jobs (``POST /v1/job``, framed: ``u32 len | JSON header | bytes``, see :mod:`dt.service.client`)
---------------------------------------------------------------------------------------------
``render_html(html, width, height, dpr, wait_fonts, full_page, tmp_dir)`` → PNG (render cache)
``render_doc(doc, mode, extra_css)`` → PNG (HTML built here with the service's html.py; cached)
``render_url(url, width, height, wait_ms, script, wait_until)`` → PNG + ``result``
``capture_url(url, width, height, device_scale_factor, wait_ms, wait_until)`` → PNG
``perceive(<png bytes> | path, dpr, overrides)`` → ``doc`` (IR dict)
``run_case(corpus_dir, case_id, stages, overrides, out_dir)`` → ``row`` (dt.selftest.bench row)

Render jobs are refused (409) when the client's ``dt/render/screenshot.py`` (``render_src``) differs
from the service's; ``render_doc`` also needs the same ``dt/render/html.py``; ``perceive`` /
``run_case`` need the same code fingerprint (all of dt/ + fixtures it reads), because the workers run
the service checkout's code. Clients then fall back to local work.

Other endpoints: ``GET /v1/status``, ``GET /v1/stats`` (status + cache + latency), ``POST /v1/stop``.
Every request must carry ``X-DT-Token`` (random per start, in ``$DT_HOME/service.json``, mode 0600).
"""
from __future__ import annotations

import http.server
import json
import multiprocessing
import os
import queue
import secrets
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Optional

from dt.service import API, P, ROOT, cache_dir, dt_home, state_path
from dt.service.cache import RenderCache, asset_fingerprint, code_fingerprint, referenced_files_sig, render_key, _file_sha
from dt.service.client import FRAME_TYPE, pack, unpack

RENDER_JOBS = ("render_html", "render_doc", "render_url", "capture_url")
CODE_JOBS = ("perceive", "run_case")
JOBS = RENDER_JOBS + CODE_JOBS


def _log(msg: str) -> None:
    print(f"[dt serve {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# =========================================================================== worker process
def _execute(job: str, args: dict, blob: bytes) -> tuple[dict, bytes]:
    from dt.render import screenshot as s
    if job == "render_html":
        tmp_dir = args.get("tmp_dir")
        if tmp_dir:
            try:
                os.makedirs(tmp_dir, exist_ok=True)
            except OSError:
                tmp_dir = None
        return {}, s._html_to_png_bytes(args["html"], int(args["width"]), int(args["height"]), float(args.get("dpr", 1.0)),
                                        bool(args.get("wait_fonts", True)), bool(args.get("full_page", False)), tmp_dir)
    if job == "render_url":
        data, result = s._render_url_bytes(args["url"], int(args["width"]), int(args["height"]), int(args.get("wait_ms", 500)),
                                           args.get("script"), args.get("wait_until", "networkidle"))
        json.dumps(result)  # must survive the wire; otherwise a job error -> the client renders locally
        return {"result": result}, data
    if job == "capture_url":
        return {}, s._capture_url_bytes(args["url"], int(args["width"]), int(args["height"]),
                                        float(args.get("device_scale_factor", 1.0)), int(args.get("wait_ms", 300)),
                                        args.get("wait_until", "load"))
    if job == "perceive":
        from dt.perceive import perceive
        from dt.selftest.bench import param_overrides
        src: Any = args.get("path")
        if not src:
            from io import BytesIO
            import numpy as np
            from PIL import Image
            src = np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.uint8).copy()
        with param_overrides(args.get("overrides") or None):
            doc = perceive(src, dpr=float(args.get("dpr", 1.0)))
        return {"doc": doc.to_dict()}, b""
    if job == "run_case":
        from dt.selftest.bench import run_case
        row = run_case(args["corpus_dir"], args["case_id"], tuple(args.get("stages") or ("perceive", "map", "render")),
                       args.get("out_dir"), args.get("overrides") or None)
        return {"row": row}, b""
    raise ValueError(f"unknown job {job!r}")


def _worker_main(conn, params: dict) -> None:
    """Worker process body: own browser, run jobs from the pipe until told to stop or orphaned."""
    os.environ.pop("DT_EVAL_SERVICE", None)  # never delegate back into the pool we are part of
    parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from dt.params import P as _P
    for k, v in (params or {}).items():
        _P.set(k, v)
    from dt.render import screenshot as s
    try:
        s.html_to_png("<style>html,body{margin:0;background:#fff}</style>", 8, 8)
        conn.send(("ready", {"pid": os.getpid(), "browser": s.BROWSER_USED, "version": s._brw().version,
                             "pref": (os.environ.get("DT_BROWSER") or "auto").strip()}))
    except BaseException as e:  # noqa: BLE001
        try:
            conn.send(("dead", f"{type(e).__name__}: {e}"))
        finally:
            s.shutdown()
        return
    try:
        while True:
            if not conn.poll(1.0):
                if os.getppid() != parent:  # service died: exit and take the browser with us
                    break
                continue
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            if msg is None:
                break
            job, args, blob = msg
            t0 = time.perf_counter()
            try:
                hdr, out = _execute(job, args, blob)
                conn.send(("ok", hdr, out, time.perf_counter() - t0))
            except Exception as e:  # noqa: BLE001 - a job error is reported, the worker lives on
                if not _browser_alive(s):
                    # Chrome died under us (crash, OOM kill): every later job would fail the same way. Exit
                    # without replying: the pool sees a crash and retries the job on a fresh worker.
                    print(f"[dt eval worker {os.getpid()}] browser gone ({type(e).__name__}); exiting", file=sys.stderr, flush=True)
                    break
                conn.send(("err", {"type": type(e).__name__, "error": str(e)[:4000],
                                   "traceback": traceback.format_exc()[-6000:]}, b"", time.perf_counter() - t0))
    finally:
        try:
            conn.close()  # the pool sees EOF now, even if closing a dead browser below is slow
        except OSError:
            pass
        s.shutdown()


def _browser_alive(s) -> bool:
    try:
        b = s._brw()
        return b is not None and b.is_connected()
    except Exception:  # noqa: BLE001
        return False


def _descendants(pid: int) -> list[int]:
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    kids: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, stack = [], [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in seen:
                seen.append(c)
                stack.append(c)
    return seen


def kill_tree(pid: int) -> None:
    """SIGKILL a process and every descendant (Playwright launches Chrome in its own process group,
    so a process-group kill would miss it)."""
    for p in [pid] + _descendants(pid):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


class WorkerFailure(Exception):
    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind


class Worker:
    def __init__(self, ctx, params: dict):
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_worker_main, args=(child, params), name="dt-eval-worker")
        self.proc.start()
        child.close()
        self.pid = self.proc.pid
        self.jobs = 0
        self.info: dict = {}
        self.started = time.time()
        self.state = "starting"

    def wait_ready(self, timeout: float) -> dict:
        if not self.conn.poll(timeout):
            self.kill()
            raise WorkerFailure("spawn", f"worker {self.pid} not ready within {timeout:.0f}s")
        try:
            msg = self.conn.recv()
        except (EOFError, OSError):
            self.kill()
            raise WorkerFailure("spawn", f"worker {self.pid} died while starting")
        if msg[0] != "ready":
            self.kill()
            raise WorkerFailure("spawn", f"worker {self.pid} failed to start: {msg[1]}")
        self.info = msg[1]
        self.state = "idle"
        return self.info

    def run(self, job: str, args: dict, blob: bytes, timeout: float) -> tuple[bool, dict, bytes, float]:
        self.state = "busy"
        try:
            self.conn.send((job, args, blob))
        except (OSError, ValueError) as e:
            raise WorkerFailure("crash", f"worker {self.pid} pipe broken: {e}")
        if not self.conn.poll(timeout):
            raise WorkerFailure("timeout", f"{job} exceeded {timeout:.0f}s on worker {self.pid}")
        try:
            msg = self.conn.recv()
        except (EOFError, OSError):
            raise WorkerFailure("crash", f"worker {self.pid} died during {job} (exit {self.proc.exitcode})")
        self.state = "idle"
        return msg[0] == "ok", msg[1], msg[2], float(msg[3])

    def stop(self, timeout: float = 10.0) -> None:
        self.state = "stopping"
        try:
            self.conn.send(None)
        except (OSError, ValueError):
            pass
        self.proc.join(timeout)
        if self.proc.is_alive():
            self.kill()
        try:
            self.conn.close()
        except OSError:
            pass

    def kill(self) -> None:
        self.state = "killed"
        kill_tree(self.pid)
        self.proc.join(5)


# =========================================================================== pool
class Pool:
    def __init__(self, n: int, max_jobs: int, job_timeout: float, queue_max: int, queue_wait: float,
                 spawn_timeout: float, params: Optional[dict] = None):
        self.n, self.max_jobs, self.job_timeout = int(n), int(max_jobs), float(job_timeout)
        self.queue_wait, self.spawn_timeout = float(queue_wait), float(spawn_timeout)
        self.capacity = self.n + int(queue_max)
        self.params = dict(params or {})
        self.ctx = multiprocessing.get_context("spawn")
        self._idle: "queue.Queue[Worker]" = queue.Queue()
        self._admit = threading.BoundedSemaphore(self.capacity)
        self._lock = threading.Lock()
        self.workers: dict[int, Worker] = {}
        self.closing = False
        self.c = {"jobs": 0, "ok": 0, "job_errors": 0, "worker_errors": 0, "timeouts": 0, "crashes": 0, "retries": 0,
                  "recycled": 0, "spawned": 0, "spawn_failures": 0, "rejected_busy": 0}
        self.in_flight = 0
        self.waiting = 0
        self.browser: dict = {}

    # ----------------------------------------------------------------- lifecycle
    def _spawn(self) -> Worker:
        w = Worker(self.ctx, self.params)
        with self._lock:
            self.workers[w.pid] = w
        try:
            info = w.wait_ready(self.spawn_timeout)
        except WorkerFailure:
            with self._lock:
                self.workers.pop(w.pid, None)
                self.c["spawn_failures"] += 1
            raise
        with self._lock:
            self.c["spawned"] += 1
            self.browser = {"name": info.get("browser"), "version": info.get("version"), "pref": info.get("pref")}
        return w

    def start(self) -> None:
        errs: list[str] = []

        def one():
            try:
                self._idle.put(self._spawn())
            except WorkerFailure as e:
                errs.append(str(e))
        ts = [threading.Thread(target=one) for _ in range(self.n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs and self._idle.qsize() == 0:
            raise RuntimeError("no eval worker could start: " + "; ".join(errs))
        for e in errs:
            _log(f"worker failed to start: {e}")
            self._replace_async()

    def _discard(self, w: Worker, kill: bool = True) -> None:
        with self._lock:
            self.workers.pop(w.pid, None)
        if kill:
            w.kill()
        else:
            w.stop()

    def _replace_async(self) -> None:
        def run():
            attempt = 0
            while not self.closing:  # never give up: a pool that silently shrinks to 0 workers stalls every client
                try:
                    w = self._spawn()
                    if self.closing:
                        self._discard(w, kill=False)
                    else:
                        self._idle.put(w)
                    return
                except WorkerFailure as e:
                    attempt += 1
                    _log(f"replacement worker failed (attempt {attempt}): {e}")
                    time.sleep(min(30.0, 2.0 * attempt))
        threading.Thread(target=run, daemon=True, name="dt-eval-respawn").start()

    def _has_workers(self) -> bool:
        with self._lock:
            return bool(self.workers)

    def _release(self, w: Worker) -> None:
        w.jobs += 1
        if self.closing:
            self._discard(w, kill=False)
        elif w.jobs >= self.max_jobs:
            with self._lock:
                self.c["recycled"] += 1
            threading.Thread(target=self._discard, args=(w, False), daemon=True).start()
            self._replace_async()
        else:
            self._idle.put(w)

    def close(self) -> None:
        self.closing = True
        with self._lock:
            ws = list(self.workers.values())
        threads = [threading.Thread(target=w.stop, args=(10.0,)) for w in ws]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
        for w in ws:
            if w.proc.is_alive():
                w.kill()
        with self._lock:
            self.workers.clear()

    # ----------------------------------------------------------------- jobs
    def submit(self, job: str, args: dict, blob: bytes, timeout: Optional[float] = None) -> tuple[bool, dict, bytes, dict]:
        """Run a job; returns (ok, header, bytes, meta). Raises queue.Full when saturated,
        WorkerFailure when the job failed twice for infrastructure reasons."""
        if self.closing or not self._admit.acquire(blocking=False):
            with self._lock:
                self.c["rejected_busy"] += 1
            raise queue.Full("eval service saturated" if not self.closing else "eval service is stopping")
        timeout = float(timeout or self.job_timeout)
        t0 = time.perf_counter()
        try:
            with self._lock:
                self.waiting += 1
            try:
                deadline = time.monotonic() + self.queue_wait
                while True:
                    try:
                        w = self._idle.get(timeout=max(0.01, min(1.0, deadline - time.monotonic())))
                        break
                    except queue.Empty:
                        if not self._has_workers():  # every worker died and no replacement is starting: fail fast
                            with self._lock:
                                self.c["worker_errors"] += 1
                            raise WorkerFailure("spawn", "no live eval worker (replacements are failing; see the service log)")
                        if time.monotonic() >= deadline:
                            with self._lock:
                                self.c["rejected_busy"] += 1
                            raise queue.Full(f"no idle worker within {self.queue_wait:.0f}s")
            finally:
                with self._lock:
                    self.waiting -= 1
            queued = time.perf_counter() - t0
            with self._lock:
                self.in_flight += 1
                self.c["jobs"] += 1
            retried = False
            try:
                try:
                    ok, hdr, out, secs = w.run(job, args, blob, timeout)
                except WorkerFailure as e:
                    with self._lock:
                        self.c["timeouts" if e.kind == "timeout" else "crashes"] += 1
                        self.c["retries"] += 1
                    _log(f"{e}; retrying on a fresh worker")
                    self._discard(w, kill=True)
                    retried = True
                    try:
                        w = self._spawn()  # takes the dead worker's slot
                    except WorkerFailure:
                        with self._lock:
                            self.c["worker_errors"] += 1
                        self._replace_async()  # keep the pool at size; the job is reported failed
                        raise
                    try:
                        ok, hdr, out, secs = w.run(job, args, blob, timeout)
                    except WorkerFailure as e2:
                        with self._lock:
                            self.c["timeouts" if e2.kind == "timeout" else "crashes"] += 1
                            self.c["worker_errors"] += 1
                        self._discard(w, kill=True)
                        self._replace_async()
                        raise
            finally:
                with self._lock:
                    self.in_flight -= 1
            with self._lock:
                self.c["ok" if ok else "job_errors"] += 1
            meta = {"worker": w.pid, "job_s": round(secs, 4), "queue_s": round(queued, 4), "retried": retried}
            self._release(w)
            return ok, hdr, out, meta
        finally:
            self._admit.release()

    def status(self) -> dict:
        with self._lock:
            ws = [{"pid": w.pid, "jobs": w.jobs, "state": w.state, "age_s": round(time.time() - w.started, 1)}
                  for w in self.workers.values()]
            return {"size": self.n, "alive": len(ws), "idle": self._idle.qsize(), "in_flight": self.in_flight,
                    "waiting": self.waiting, "capacity": self.capacity, "workers": ws, "counters": dict(self.c)}


# =========================================================================== HTTP front
class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dt-eval/1"
    svc: "EvalService"  # set on the per-service subclass

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: the service logs its own events
        pass

    def address_string(self) -> str:  # unix sockets have no (host, port)
        return self.client_address[0] if isinstance(self.client_address, tuple) and self.client_address else "unix"

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, default=str).encode())

    def _authorized(self) -> bool:
        tok = self.svc.token
        if tok and not secrets.compare_digest(self.headers.get("X-DT-Token", ""), tok):
            self._json(401, {"ok": False, "error": "missing or wrong X-DT-Token (see $DT_HOME/service.json)"})
            return False
        return True

    def _body(self) -> Optional[bytes]:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n < 0:
                raise ValueError(n)
        except ValueError:
            self.close_connection = True
            self._json(400, {"ok": False, "error": "bad Content-Length"})
            return None
        if n > int(P["service.max_body_mb"]) * (1 << 20):
            self.close_connection = True
            self._json(413, {"ok": False, "error": "request body too large"})
            return None
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path == f"/{API}/status":
            self._json(200, self.svc.status())
        elif self.path == f"/{API}/stats":
            self._json(200, self.svc.stats())
        else:
            self._json(404, {"ok": False, "error": f"no such endpoint {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self.close_connection = True  # the unread body would otherwise be parsed as the next request
            return
        body = self._body()
        if body is None:
            return
        if self.path == f"/{API}/stop":
            self._json(200, {"ok": True, "stopping": True, "pid": os.getpid()})
            threading.Thread(target=self.svc.shutdown, daemon=True).start()
            return
        if self.path != f"/{API}/job":
            self._json(404, {"ok": False, "error": f"no such endpoint {self.path}"})
            return
        try:
            req, blob = unpack(body)
            if not isinstance(req, dict):
                raise ValueError("header is not a JSON object")
        except Exception as e:  # noqa: BLE001
            self._send(400, pack({"ok": False, "error_kind": "bad_request", "error": f"bad frame: {e}"}), FRAME_TYPE)
            return
        try:
            code, hdr, out = self.svc.handle_job(req, blob)
        except Exception as e:  # noqa: BLE001 - never drop the connection without an answer
            _log(f"internal error on {req.get('job')!r}: {type(e).__name__}: {e}")
            code, hdr, out = 500, {"ok": False, "error_kind": "internal", "error": f"{type(e).__name__}: {e}"}, b""
        self._send(code, pack(hdr, out), FRAME_TYPE)


class _TCPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 128


# =========================================================================== service
class EvalService:
    """The service: worker pool + render cache + HTTP front (TCP on 127.0.0.1, or a unix socket)."""

    def __init__(self, workers: Optional[int] = None, port: Optional[int] = None, socket_path: Optional[str] = None,
                 cache: Optional[bool] = None, cache_root: Optional[str] = None, write_state: bool = True,
                 token: Optional[str] = None, worker_params: Optional[dict] = None, **pool_kw: Any):
        self.n = int(workers or P["service.workers"])
        self.port = int(P["service.port"] if port is None else port)
        self.socket_path = os.path.abspath(os.path.expanduser(socket_path)) if socket_path else None
        self.cache_enabled = bool(P["service.cache.enabled"] if cache is None else cache)
        self.cache = RenderCache(cache_root or cache_dir(), int(P["service.cache.max_mb"]) * (1 << 20)) if self.cache_enabled else None
        self.write_state = write_state
        self.token = token if token is not None else secrets.token_hex(16)
        kw = dict(max_jobs=P["service.max_jobs_per_worker"], job_timeout=P["service.job_timeout_s"],
                  queue_max=P["service.queue_max"], queue_wait=P["service.queue_wait_s"],
                  spawn_timeout=P["service.spawn_timeout_s"])
        kw.update(pool_kw)
        self.pool = Pool(self.n, params=worker_params, **kw)
        self.httpd: Any = None
        self.started = time.time()
        self.render_src = _file_sha(os.path.join(ROOT, "dt", "render", "screenshot.py")) or ""
        self.html_src = _file_sha(os.path.join(ROOT, "dt", "render", "html.py")) or ""
        self.code_fp = ""
        self._stopped = threading.Event()
        self._closed = threading.Event()
        self._lat: dict[str, list[float]] = {}
        self._lat_lock = threading.Lock()

    # ----------------------------------------------------------------- lifecycle
    @property
    def url(self) -> str:
        if self.socket_path:
            return "unix:" + self.socket_path
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "EvalService":
        self.code_fp = code_fingerprint(ROOT)
        handler = type("Handler", (_Handler,), {"svc": self})
        if self.socket_path:
            os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
            if os.path.exists(self.socket_path):
                try:  # a live service owns it; a stale file from a crash does not
                    s = socket.socket(socket.AF_UNIX)
                    s.connect(self.socket_path)
                    s.close()
                    raise RuntimeError(f"another service is listening on {self.socket_path}")
                except (ConnectionRefusedError, FileNotFoundError):
                    os.remove(self.socket_path)
            self.httpd = _UnixServer(self.socket_path, handler)
            os.chmod(self.socket_path, 0o600)
        else:
            self.httpd = _TCPServer(("127.0.0.1", self.port), handler)
            self.port = int(self.httpd.server_address[1])
        t0 = time.perf_counter()
        try:
            self.pool.start()
        except Exception:
            self.httpd.server_close()
            raise
        _log(f"{self.n} workers ready in {time.perf_counter() - t0:.1f}s ({self.pool.browser.get('name')} "
             f"{self.pool.browser.get('version')}); listening on {self.url}; cache "
             f"{self.cache.root if self.cache else 'off'}")
        if self.write_state:
            self._write_state()
        return self

    def _write_state(self) -> None:
        os.makedirs(dt_home(), exist_ok=True)
        p = state_path()
        tmp = p + f".{os.getpid()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "url": self.url, "token": self.token, "workers": self.n, "root": ROOT,
                       "started": self.started}, f, indent=2)
        os.replace(tmp, p)

    def _clear_state(self) -> None:
        try:
            with open(state_path()) as f:
                st = json.load(f)
            if st.get("pid") == os.getpid():
                os.remove(state_path())
        except (OSError, ValueError):
            pass

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever(poll_interval=0.5)
        finally:
            self.shutdown()

    def start_background(self) -> "EvalService":
        self.start()
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True,
                         name="dt-eval-http").start()
        return self

    def shutdown(self) -> None:
        """Stop accepting, stop every worker (and its browser), remove the socket and state file.
        Safe to call from any thread and more than once (later calls wait for the first to finish)."""
        if self._stopped.is_set():
            self._closed.wait(60)
            return
        self._stopped.set()
        _log("stopping")
        self.pool.closing = True
        if self.httpd is not None:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()
        self.pool.close()
        if self.httpd is not None:
            try:
                self.httpd.server_close()
            except OSError:
                pass
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError:
                pass
        if self.write_state:
            self._clear_state()
        _log("stopped")
        self._closed.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the service has fully stopped."""
        return self._closed.wait(timeout)

    # ----------------------------------------------------------------- status
    def _current(self) -> dict:
        render_now = _file_sha(os.path.join(ROOT, "dt", "render", "screenshot.py")) or ""
        html_now = _file_sha(os.path.join(ROOT, "dt", "render", "html.py")) or ""
        return {"render_src": render_now, "html_src": html_now}

    def status(self) -> dict:
        cur = self._current()
        stale = cur["render_src"] != self.render_src or cur["html_src"] != self.html_src
        return {"ok": True, "api": API, "pid": os.getpid(), "url": self.url, "root": ROOT, "uptime_s": round(time.time() - self.started, 1),
                "browser": self.pool.browser, "render_src": self.render_src, "html_src": self.html_src,
                "code_fp": self.code_fp, "stale": stale, "cache_enabled": self.cache_enabled,
                "pool": self.pool.status(), "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}

    def stats(self) -> dict:
        st = self.status()
        st["cache"] = self.cache.stats() if self.cache else None
        with self._lat_lock:
            lat = {k: list(v) for k, v in self._lat.items()}
        st["latency"] = {k: _pcts(v) for k, v in lat.items()}
        return st

    def _record(self, job: str, secs: float) -> None:
        with self._lat_lock:
            v = self._lat.setdefault(job, [])
            v.append(secs)
            if len(v) > 5000:
                del v[:1000]

    # ----------------------------------------------------------------- jobs
    def _check(self, job: str, client: dict) -> Optional[str]:
        cur = self._current()
        if job in RENDER_JOBS + CODE_JOBS:
            if client.get("render_src") and not (client["render_src"] == self.render_src == cur["render_src"]):
                return "dt/render/screenshot.py differs from the service's (or changed since it started)"
            want = client.get("browser_pref", "auto")
            if want != "auto" and want not in (self.pool.browser.get("pref"), self.pool.browser.get("name")):
                return f"client wants DT_BROWSER={want}; service renders with {self.pool.browser.get('name')}"
        if job == "render_doc" and not (client.get("html_src") == self.html_src == cur["html_src"]):
            return "dt/render/html.py differs from the service's: build the HTML locally and send render_html"
        if job in CODE_JOBS:
            now = code_fingerprint(ROOT)
            if not client.get("code_fp") or not (client["code_fp"] == self.code_fp == now):
                return ("code fingerprint differs from the service's checkout (workers run the service's code); "
                        "run this job locally, or restart the service from this checkout")
        return None

    def handle_job(self, req: dict, blob: bytes) -> tuple[int, dict, bytes]:
        t0 = time.perf_counter()
        job = req.get("job")
        if job not in JOBS:
            return 400, {"ok": False, "error_kind": "bad_request", "error": f"unknown job {job!r}; jobs: {list(JOBS)}"}, b""
        try:
            args, client = _validate(job, req)
        except (KeyError, TypeError, ValueError) as e:
            return 400, {"ok": False, "error_kind": "bad_request", "error": f"bad {job} request: {type(e).__name__}: {e}"}, b""
        why = self._check(job, client)
        if why:
            return 409, {"ok": False, "error_kind": "incompatible", "error": why}, b""
        if job == "render_doc":
            from dt.ir import Document
            from dt.render.html import render_html
            try:
                doc = Document.from_dict(args["doc"])
                html = render_html(doc, mode=args.get("mode", "absolute"), extra_css=args.get("extra_css", ""))
            except Exception as e:  # noqa: BLE001 - a bad IR is the caller's error, reported as such
                return 400, {"ok": False, "error_kind": "bad_request", "error": f"bad render_doc IR: {type(e).__name__}: {e}"}, b""
            args = {"html": html, "width": int(doc.width), "height": int(doc.height), "dpr": 1.0, "wait_fonts": True,
                    "full_page": False, "cache": args.get("cache", True)}
            job = "render_html"
        key = None
        cache_state = "off"
        if job == "render_html" and self.cache is not None and args.get("cache", True):
            root = client.get("root") or ROOT
            fonts = [os.path.join(root, "fixtures", "fonts"), os.path.join(root, "fixtures", "icons")]
            fp = "|".join([str(self.pool.browser.get("name")), str(self.pool.browser.get("version")), asset_fingerprint(root),
                           referenced_files_sig(args["html"], fonts), str(args.get("tmp_dir") or "")])
            key = render_key(args["html"], args["width"], args["height"], args.get("dpr", 1.0),
                             {"wait_fonts": bool(args.get("wait_fonts", True)), "full_page": bool(args.get("full_page", False))}, fp)
            hit = self.cache.get(key)
            if hit is not None:
                secs = time.perf_counter() - t0
                self._record("render_html:hit", secs)
                return 200, {"ok": True, "meta": {"cache": "hit", "total_s": round(secs, 4)}}, hit
            cache_state = "miss"
        args.pop("cache", None)
        timeout = None
        if req.get("timeout_s"):  # a client may ask for less (or up to 4x more) than service.job_timeout_s (validated)
            timeout = max(0.1, min(float(req["timeout_s"]), 4 * self.pool.job_timeout))
        try:
            ok, hdr, out, meta = self.pool.submit(job, args, blob, timeout)
        except queue.Full as e:
            return 503, {"ok": False, "error_kind": "busy", "error": str(e)}, b""
        except WorkerFailure as e:
            return 200, {"ok": False, "error_kind": "worker", "error": str(e)}, b""
        meta["cache"] = cache_state
        meta["total_s"] = round(time.perf_counter() - t0, 4)
        if not ok:
            _log(f"job error {job} (worker {meta['worker']}): {hdr.get('type')}: {str(hdr.get('error'))[:300]}")
            return 200, {"ok": False, "error_kind": "job", "error": f"{hdr.get('type')}: {hdr.get('error')}",
                         "type": hdr.get("type"), "traceback": hdr.get("traceback"), "meta": meta}, b""
        if key is not None:
            try:
                self.cache.put(key, out, meta["job_s"])
            except OSError as e:
                _log(f"cache write failed: {e}")
        self._record(job, meta["total_s"])
        hdr = dict(hdr)
        hdr.update(ok=True, meta=meta)
        return 200, hdr, out


_REQUIRED = {"render_html": ("html", "width", "height"), "render_doc": ("doc",), "render_url": ("url", "width", "height"),
             "capture_url": ("url", "width", "height"), "perceive": (), "run_case": ("corpus_dir", "case_id")}


def _validate(job: str, req: dict) -> tuple[dict, dict]:
    """(args, client) of a job request; raises KeyError / TypeError / ValueError on a malformed one (-> 400)."""
    args, client = req.get("args") or {}, req.get("client") or {}
    if not isinstance(args, dict) or not isinstance(client, dict):
        raise TypeError("args and client must be JSON objects")
    for k in _REQUIRED[job]:
        if args.get(k) is None:
            raise KeyError(k)
    for k in ("width", "height"):
        if k in args and int(args[k]) <= 0:
            raise ValueError(f"{k} must be positive")
    for k in ("dpr", "device_scale_factor"):
        if k in args and not float(args[k]) > 0:
            raise ValueError(f"{k} must be positive")
    if job in ("render_html", "render_url", "capture_url") and not isinstance(args.get("html", args.get("url")), str):
        raise TypeError("html / url must be a string")
    if job == "render_doc" and not isinstance(args["doc"], dict):
        raise TypeError("doc must be an IR object")
    if req.get("timeout_s") is not None:
        float(req["timeout_s"])
    return dict(args), dict(client)


def _pcts(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    s = sorted(v)

    def q(p: float) -> float:
        return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 4)
    return {"n": len(s), "p50": q(0.5), "p95": q(0.95), "max": round(s[-1], 4), "mean": round(sum(s) / len(s), 4)}
