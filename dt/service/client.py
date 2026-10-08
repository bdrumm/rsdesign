"""Client for the shared evaluation service (``dt serve``).

    from dt.service import client
    client.available()                       # service reachable (probe memoised for service.client.probe_ttl_s)
    hdr, png = client.call("render_html", {"html": html, "width": 400, "height": 300})
    row = client.run_case("fixtures/corpus/synth", "synth_1_000")          # bench row (same code only)
    doc = client.perceive(open("shot.png", "rb").read(), dpr=1.0)            # IR dict (same code only)

Where the service is: ``DT_EVAL_SERVICE`` = ``1`` / ``auto`` (read ``$DT_HOME/service.json``, written by
``dt serve``) | ``http://127.0.0.1:PORT`` | ``unix:/path/to.sock``; unset / ``0`` / ``off`` = disabled.
The auth token comes from ``DT_EVAL_SERVICE_TOKEN`` or ``$DT_HOME/service.json``.

:func:`delegate` is what :mod:`dt.render.screenshot` uses: it returns ``None`` (caller renders
locally) whenever the service cannot serve the job (down, saturated past
``service.client.busy_retry_s``, incompatible renderer, worker failure, or a job error — the local
path then raises the genuine exception), and logs each kind of fallback once per process.

Wire format (both directions, ``POST /v1/job``): ``u32 big-endian header length | JSON header | raw
bytes`` (PNG in, PNG out), so images are never base64-encoded.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Any, Optional

from dt.service import API, P, ROOT, state_path

FRAME_TYPE = "application/x-dt-frame"
_OFF = ("", "0", "off", "false", "no", "none")
_AUTO = ("1", "on", "true", "yes", "auto")


# --------------------------------------------------------------------------- errors
class ServiceError(RuntimeError):
    kind = "service"


class Unavailable(ServiceError):
    kind = "unavailable"


class Busy(ServiceError):
    kind = "busy"


class Incompatible(ServiceError):
    kind = "incompatible"


class WorkerFailed(ServiceError):
    kind = "worker"


class JobError(ServiceError):
    kind = "job"


# --------------------------------------------------------------------------- framing
def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, bytes):
        return o.decode("latin-1")
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def pack(header: dict, blob: bytes = b"") -> bytes:
    h = json.dumps(header, default=_json_default).encode("utf-8")
    return struct.pack(">I", len(h)) + h + (blob or b"")


def unpack(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 4:
        raise ValueError("short frame")
    n = struct.unpack(">I", data[:4])[0]
    return json.loads(data[4:4 + n].decode("utf-8")), data[4 + n:]


# --------------------------------------------------------------------------- address
def address() -> Optional[dict]:
    """Resolved service address ``{kind: tcp|unix, host, port, path, url, token}`` or None when disabled."""
    v = (os.environ.get("DT_EVAL_SERVICE") or "").strip()
    if v.lower() in _OFF:
        return None
    state: dict = {}
    try:
        with open(state_path()) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    url = state.get("url", "") if v.lower() in _AUTO else v
    if not url:
        return None
    tok = os.environ.get("DT_EVAL_SERVICE_TOKEN") or state.get("token")
    return parse_url(url, tok)


def parse_url(url: str, token: Optional[str] = None) -> dict:
    if url.startswith("unix:") or url.startswith("/"):
        path = url[5:] if url.startswith("unix:") else url
        return {"kind": "unix", "path": path, "url": "unix:" + path, "token": token}
    rest = url.split("://", 1)[-1].rstrip("/")
    host, _, port = rest.partition(":")
    return {"kind": "tcp", "host": host or "127.0.0.1", "port": int(port or P["service.port"]),
            "url": f"http://{host or '127.0.0.1'}:{int(port or P['service.port'])}", "token": token}


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: Optional[float] = None):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


_tls = threading.local()


def _conn(addr: dict, timeout: Optional[float]) -> http.client.HTTPConnection:
    """One persistent (keep-alive) connection per thread and address."""
    pool = getattr(_tls, "conns", None)
    if pool is None:
        pool = _tls.conns = {}
    c = pool.get(addr["url"])
    if c is None:
        c = _UnixConnection(addr["path"], timeout) if addr["kind"] == "unix" else \
            http.client.HTTPConnection(addr["host"], addr["port"], timeout=timeout)
        pool[addr["url"]] = c
    c.timeout = timeout
    if c.sock is not None:
        c.sock.settimeout(timeout)
    return c


def _drop_conn(addr: dict) -> None:
    pool = getattr(_tls, "conns", None) or {}
    c = pool.pop(addr["url"], None)
    if c is not None:
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass


def request(addr: dict, method: str, path: str, body: bytes = b"", ctype: str = FRAME_TYPE,
            timeout: Optional[float] = None) -> tuple[int, str, bytes]:
    """One HTTP request (reconnects once if a kept-alive connection went stale). Raises Unavailable."""
    headers = {"Content-Type": ctype, "Content-Length": str(len(body))}
    if addr.get("token"):
        headers["X-DT-Token"] = addr["token"]
    last: Optional[BaseException] = None
    for attempt in range(2):
        c = _conn(addr, timeout)
        try:
            c.request(method, path, body=body, headers=headers)
            r = c.getresponse()
            data = r.read()
            if r.getheader("Connection", "").lower() == "close":
                _drop_conn(addr)
            return r.status, r.getheader("Content-Type", ""), data
        except (ConnectionError, http.client.HTTPException, socket.timeout, OSError) as e:
            _drop_conn(addr)
            last = e
            if isinstance(e, socket.timeout):
                break
    raise Unavailable(f"eval service {addr['url']} unreachable: {type(last).__name__}: {last}")


# --------------------------------------------------------------------------- probing
_probe_lock = threading.Lock()
_probe: dict = {"key": None, "at": -1e9, "status": None}
_warned: set[str] = set()


def _warn_once(kind: str, msg: str) -> None:
    if kind in _warned:
        return
    _warned.add(kind)
    print(f"[dt.service] {msg}", file=sys.stderr, flush=True)


def status(addr: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
    """GET /v1/status (raises Unavailable / ServiceError)."""
    addr = addr or address()
    if addr is None:
        raise Unavailable("DT_EVAL_SERVICE is not set")
    code, _ct, data = request(addr, "GET", f"/{API}/status", timeout=timeout or P["service.client.connect_timeout_s"])
    if code == 401:
        raise Incompatible(f"eval service {addr['url']} rejected the token (set DT_HOME or DT_EVAL_SERVICE_TOKEN)")
    if code != 200:
        raise Unavailable(f"eval service {addr['url']} status HTTP {code}")
    return json.loads(data)


def stats(addr: Optional[dict] = None) -> dict:
    addr = addr or address()
    if addr is None:
        raise Unavailable("DT_EVAL_SERVICE is not set")
    code, _ct, data = request(addr, "GET", f"/{API}/stats", timeout=P["service.client.connect_timeout_s"] * 10)
    if code != 200:
        raise ServiceError(f"stats HTTP {code}")
    return json.loads(data)


def stop(addr: Optional[dict] = None) -> dict:
    addr = addr or address()
    if addr is None:
        raise Unavailable("DT_EVAL_SERVICE is not set")
    code, _ct, data = request(addr, "POST", f"/{API}/stop", ctype="application/json", timeout=10)
    if code != 200:
        raise ServiceError(f"stop HTTP {code}")
    return json.loads(data)


def probe(refresh: bool = False) -> Optional[dict]:
    """Service status if reachable (memoised for service.client.probe_ttl_s, failures too), else None."""
    addr = address()
    if addr is None:
        return None
    key = (addr["url"], addr.get("token"))
    now = time.monotonic()
    with _probe_lock:
        if not refresh and _probe["key"] == key and now - _probe["at"] < P["service.client.probe_ttl_s"]:
            return _probe["status"]
    try:
        st = status(addr)
    except ServiceError as e:
        st = None
        _warn_once("unavailable", f"{e}; rendering locally (logged once)")
    with _probe_lock:
        _probe.update(key=key, at=time.monotonic(), status=st)
    return st


def _mark_down() -> None:
    with _probe_lock:
        _probe.update(at=time.monotonic(), status=None)


def available(refresh: bool = False) -> bool:
    return probe(refresh) is not None


# --------------------------------------------------------------------------- compatibility
def _src_sha(rel: str) -> str:
    from dt.service.cache import _file_sha
    return _file_sha(os.path.join(ROOT, rel)) or ""


def client_info(code: bool = False) -> dict:
    from dt.render import screenshot as s
    info = {"root": ROOT, "tmp_dir": s._TMP_DIR, "render_src": _src_sha("dt/render/screenshot.py"),
            "html_src": _src_sha("dt/render/html.py"), "browser_pref": (os.environ.get("DT_BROWSER") or "auto").strip(),
            "pid": os.getpid()}
    if code:
        from dt.service.cache import code_fingerprint
        info["code_fp"] = code_fingerprint(ROOT)
    return info


def incompatibility(st: dict, info: Optional[dict] = None) -> Optional[str]:
    """Why renders from this service would not be byte-identical to this process's local renders (None = fine)."""
    info = info or client_info()
    want = info["browser_pref"]
    b = st.get("browser") or {}
    if want != "auto" and want not in (b.get("pref"), b.get("name")):
        return f"DT_BROWSER={want} but the service renders with {b.get('name')} (pref {b.get('pref')})"
    if st.get("render_src") and st["render_src"] != info["render_src"]:
        return "dt/render/screenshot.py differs between this checkout and the service's"
    return None


def usable() -> Optional[dict]:
    """Status of a reachable service that renders exactly like this process would, else None."""
    st = probe()
    if st is None:
        return None
    why = incompatibility(st)
    if why:
        _warn_once("incompatible", f"eval service not used: {why}; rendering locally (logged once)")
        return None
    return st


# --------------------------------------------------------------------------- jobs
def call(job: str, args: dict, blob: bytes = b"", addr: Optional[dict] = None, code: bool = False,
         busy_retry_s: Optional[float] = None, timeout_s: Optional[float] = None) -> tuple[dict, bytes]:
    """Run one job on the service; returns (result header, result bytes). Raises :class:`ServiceError`
    subclasses: Unavailable, Busy (still saturated after ``busy_retry_s``), Incompatible, WorkerFailed
    (timed out / crashed twice), JobError (the job itself raised; ``.info`` has type/traceback)."""
    addr = addr or address()
    if addr is None:
        raise Unavailable("DT_EVAL_SERVICE is not set")
    req = {"job": job, "args": args, "client": client_info(code=code)}
    if timeout_s:
        req["timeout_s"] = float(timeout_s)
    body = pack(req, blob)
    timeout = float(P["service.queue_wait_s"]) + 2 * float(P["service.job_timeout_s"]) + 60
    deadline = time.monotonic() + float(P["service.client.busy_retry_s"] if busy_retry_s is None else busy_retry_s)
    delay = 0.05
    while True:
        code_, _ct, data = request(addr, "POST", f"/{API}/job", body, timeout=timeout)
        if code_ == 503 and time.monotonic() < deadline:
            time.sleep(delay)
            delay = min(1.0, delay * 2)
            continue
        break
    try:
        hdr, out = unpack(data)
    except (ValueError, json.JSONDecodeError):
        hdr, out = {"ok": False, "error": data[:300].decode("utf-8", "replace")}, b""
    if code_ == 200 and hdr.get("ok"):
        return hdr, out
    msg = hdr.get("error") or f"HTTP {code_}"
    if code_ == 503:
        raise Busy(f"eval service saturated: {msg}")
    if code_ in (401, 409):
        raise Incompatible(msg)
    if code_ == 200 and hdr.get("error_kind") == "job":
        e = JobError(msg)
        e.info = hdr  # type: ignore[attr-defined]
        raise e
    if code_ == 200:
        raise WorkerFailed(msg)
    raise ServiceError(f"HTTP {code_}: {msg}")


def delegate(job: str, args: dict) -> Optional[tuple[dict, bytes]]:
    """Run a render job on the service if it is usable; None means "render locally"."""
    if usable() is None:
        return None
    try:
        return call(job, args)
    except Unavailable as e:
        _mark_down()
        _warn_once("unavailable", f"{e}; rendering locally (logged once)")
    except ServiceError as e:
        _warn_once(e.kind, f"eval service {e.kind} on {job}: {str(e).splitlines()[0][:300]}; rendering locally (logged once)")
    return None


def browser_name() -> Optional[str]:
    st = _probe.get("status") or {}
    return (st.get("browser") or {}).get("name")


# --------------------------------------------------------------------------- typed helpers
def render_html(html: str, width: int, height: int, dpr: float = 1.0, wait_fonts: bool = True,
                full_page: bool = False, cache: bool = True) -> bytes:
    from dt.render import screenshot as s
    _hdr, png = call("render_html", {"html": html, "width": int(width), "height": int(height), "dpr": float(dpr),
                                     "wait_fonts": bool(wait_fonts), "full_page": bool(full_page), "cache": bool(cache),
                                     "tmp_dir": s._TMP_DIR})
    return png


def render_doc(doc_dict: dict, mode: str = "absolute", extra_css: str = "", cache: bool = True) -> bytes:
    """IR (dict) -> PNG rendered with the *service's* dt/render/html.py (refused if it differs from ours)."""
    _hdr, png = call("render_doc", {"doc": doc_dict, "mode": mode, "extra_css": extra_css, "cache": bool(cache)})
    return png


def render_url(url: str, width: int, height: int, wait_ms: int = 500, script: Optional[str] = None,
               wait_until: str = "networkidle") -> tuple[bytes, Any]:
    hdr, png = call("render_url", {"url": url, "width": int(width), "height": int(height), "wait_ms": int(wait_ms),
                                   "script": script, "wait_until": wait_until})
    return png, hdr.get("result")


def capture_url(url: str, width: int, height: int, device_scale_factor: float = 1.0, wait_ms: int = 300,
                wait_until: str = "load") -> bytes:
    _hdr, png = call("capture_url", {"url": url, "width": int(width), "height": int(height),
                                     "device_scale_factor": float(device_scale_factor), "wait_ms": int(wait_ms),
                                     "wait_until": wait_until})
    return png


def perceive(png: bytes, dpr: float = 1.0, overrides: Optional[dict] = None) -> dict:
    """PNG bytes -> perceived IR dict. Only served when this checkout's code equals the service's."""
    hdr, _ = call("perceive", {"dpr": float(dpr), "overrides": overrides or {}}, png, code=True)
    return hdr["doc"]


def run_case(corpus_dir: str, case_id: str, stages: Any = ("perceive", "map", "render"),
             overrides: Optional[dict] = None, out_dir: Optional[str] = None) -> dict:
    """dt.selftest.bench.run_case on the service (same code only). Returns the bench row."""
    hdr, _ = call("run_case", {"corpus_dir": os.path.abspath(corpus_dir), "case_id": case_id, "stages": list(stages),
                               "overrides": overrides or {}, "out_dir": os.path.abspath(out_dir) if out_dir else None},
                  code=True)
    return hdr["row"]
