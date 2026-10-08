"""Shared evaluation service (dt/service): byte-identical renders vs local, render cache, delegation from
dt.render.screenshot with fallback, code-dependent jobs, timeout + retry on a fresh worker (with the
stuck browser killed), backpressure, worker recycling, and the CLI. All renders use the real browser."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from dt.ir import Document
from dt.render import screenshot as s
from dt.render.html import _font_face_css, render_html
from dt.service import client
from dt.service.cache import RenderCache
from dt.service.server import EvalService, _descendants

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTH = os.path.join(ROOT, "fixtures", "corpus", "synth")
MWC = os.path.join(ROOT, "fixtures", "corpus", "mwc")


def _reset_client() -> None:
    client._probe.update(key=None, at=-1e9, status=None)
    client._warned.clear()


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    return str(tmp_path_factory.mktemp("dthome"))


@pytest.fixture(scope="module")
def svc(home):
    mp = pytest.MonkeyPatch()
    mp.setenv("DT_HOME", home)
    service = EvalService(workers=2, port=0).start_background()
    mp.setenv("DT_EVAL_SERVICE", service.url)
    _reset_client()
    yield service
    client.stop(client.address())
    assert service.wait(60)
    mp.undo()
    _reset_client()


@pytest.fixture
def env(svc, monkeypatch):
    monkeypatch.setenv("DT_EVAL_SERVICE", svc.url)
    _reset_client()
    return svc


def _icon_batch_html() -> str:
    from dt.perceive.icons import VARIABLE_FONT
    names = ["home", "search", "settings", "favorite", "menu", "close", "add", "delete", "star", "mail", "person", "info"]
    spans = "".join(f'<span class=g style="left:{(i % 6) * 40}px;top:{(i // 6) * 40}px">{n}</span>' for i, n in enumerate(names))
    return (f"<!doctype html><meta charset=utf-8><style>@font-face{{font-family:'MSOV';src:url('file://{VARIABLE_FONT}') format('woff2');}}"
            "html,body{margin:0;background:#fff}.g{position:absolute;width:40px;height:40px;display:flex;align-items:center;"
            "justify-content:center;font-family:'MSOV';font-size:24px;line-height:1;color:#000;"
            "font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24}</style>" + spans)


def _font_batch_html() -> str:
    rows = [("Roboto", 400, 14), ("Roboto", 500, 22), ("Google Sans", 400, 16), ("Google Sans", 700, 28),
            ("Material Symbols Outlined", 400, 24)]
    divs = "".join(f"<div style=\"position:absolute;left:8px;top:{8 + i * 40}px;font:{w} {px}px '{f}';white-space:nowrap\">"
                   f"{'home search' if f.startswith('Material') else 'Quick brown fox 0123 — ÄÖÜ'}</div>" for i, (f, w, px) in enumerate(rows))
    return f"<!doctype html><meta charset=utf-8><style>{_font_face_css()} body{{margin:0;background:#fff}}</style>{divs}"


# --------------------------------------------------------------------------- determinism
def test_service_renders_are_byte_identical_to_local(env):
    """Corpus docs (synth + Material Web gt IR), a Material Web page, and icon / font batch pages."""
    pages = []
    for cid in ("synth_1_000", "synth_1_003"):
        d = Document.load(os.path.join(SYNTH, f"{cid}.gt.json"))
        pages.append((render_html(d), d.width, d.height))
    d = Document.load(os.path.join(MWC, "mwc_1_001.gt.json"))
    pages.append((render_html(d), d.width, d.height))
    pages.append((_font_batch_html(), 420, 220))
    for html, w, h in pages:
        local = s._html_to_png_bytes(html, w, h)
        remote = client.render_html(html, w, h, cache=False)
        assert remote == local
    # Material Web page and an icon batch page through render_url (script result included)
    mwc_url = "file://" + os.path.join(MWC, "mwc_1_001.html")
    script = "(() => document.querySelectorAll('*').length)()"
    lp, lr = s._render_url_bytes(mwc_url, d.width, d.height, 600, script, "load")
    rp, rr = client.render_url(mwc_url, d.width, d.height, wait_ms=600, script=script, wait_until="load")
    assert rp == lp and rr == lr and isinstance(rr, int)
    tmp = s._tmp_html(_icon_batch_html())
    lp, _ = s._render_url_bytes("file://" + tmp, 240, 80, 0, None, "load")
    rp, _ = client.render_url("file://" + tmp, 240, 80, wait_ms=0, wait_until="load")
    assert rp == lp
    assert client.capture_url(mwc_url, 300, 200, device_scale_factor=2.0, wait_ms=0) == \
        s._capture_url_bytes(mwc_url, 300, 200, 2.0, 0, "load")


def test_delegation_from_screenshot_module_and_cache(env, tmp_path):
    d = Document.load(os.path.join(SYNTH, "synth_1_001.gt.json"))
    before = client.stats()
    out = str(tmp_path / "r.png")
    a = s.render_doc(d, out)            # delegated: miss
    b = s.render_doc(d)                 # delegated: hit
    after = client.stats()
    assert after["pool"]["counters"]["jobs"] >= before["pool"]["counters"]["jobs"] + 1
    assert after["cache"]["hits"] >= before["cache"]["hits"] + 1
    assert np.array_equal(a, b) and os.path.getsize(out) > 0
    from io import BytesIO
    from PIL import Image
    local = np.asarray(Image.open(BytesIO(s._html_to_png_bytes(render_html(d), d.width, d.height))).convert("RGB"))
    assert np.array_equal(a, local)


def test_fallback_to_local_when_service_is_down(monkeypatch, capsys):
    import socket
    sk = socket.socket()
    sk.bind(("127.0.0.1", 0))
    port = sk.getsockname()[1]
    sk.close()  # nothing listens here now
    monkeypatch.setenv("DT_EVAL_SERVICE", f"http://127.0.0.1:{port}")
    _reset_client()
    html = "<style>html,body{margin:0;background:#123456}</style>"
    a = s.html_to_png(html, 12, 8)
    b = s.html_to_png(html, 12, 8)
    assert a.shape == (8, 12, 3) and tuple(a[0, 0]) == (0x12, 0x34, 0x56) and np.array_equal(a, b)
    err = capsys.readouterr().err
    assert err.count("rendering locally") == 1
    _reset_client()


def test_incompatible_clients_render_locally():
    st = {"browser": {"name": "chrome", "pref": "auto"}, "render_src": "x"}
    info = {"browser_pref": "auto", "render_src": "x"}
    assert client.incompatibility(st, info) is None
    assert "DT_BROWSER" in client.incompatibility(st, dict(info, browser_pref="/opt/other-chromium"))
    assert client.incompatibility(st, dict(info, browser_pref="chrome")) is None
    assert "screenshot.py" in client.incompatibility(st, dict(info, render_src="y"))


# --------------------------------------------------------------------------- code-dependent jobs
def _no_ids(o):
    if isinstance(o, dict):
        return {k: _no_ids(v) for k, v in o.items() if k not in ("id", "node_id")}
    if isinstance(o, list):
        return [_no_ids(v) for v in o]
    return o


def test_run_case_and_perceive_jobs_match_local(env):
    from dt.perceive import perceive
    from dt.selftest.bench import run_case
    row = client.run_case(SYNTH, "synth_1_002", stages=("render",))
    ref = run_case(SYNTH, "synth_1_002", ("render",))
    assert row["error"] is None and row["composite"] == pytest.approx(ref["composite"], abs=1e-9)
    assert row["metrics"]["mean_de"] == pytest.approx(ref["metrics"]["mean_de"], abs=1e-9)
    png = os.path.join(SYNTH, "synth_1_002.png")
    doc = client.perceive(open(png, "rb").read())
    loc = perceive(png).to_dict()
    assert _no_ids(doc["root"]) == _no_ids(json.loads(json.dumps(loc["root"])))  # node ids are random (uuid4)
    # a client whose code differs is refused (409) instead of being scored with the service's code
    addr = client.address()
    body = client.pack({"job": "run_case", "args": {"corpus_dir": SYNTH, "case_id": "synth_1_002"},
                        "client": dict(client.client_info(), code_fp="not-this-code")})
    code, _ct, data = client.request(addr, "POST", "/v1/job", body, timeout=30)
    assert code == 409 and "fingerprint" in client.unpack(data)[0]["error"]


# --------------------------------------------------------------------------- failure handling
@pytest.fixture
def small(home, monkeypatch):
    """One-worker service: no queue slack, recycle every 3 jobs."""
    service = EvalService(workers=1, port=0, write_state=False, token="t", queue_max=0, max_jobs=3,
                          cache=False).start_background()
    monkeypatch.setenv("DT_EVAL_SERVICE", service.url)
    monkeypatch.setenv("DT_EVAL_SERVICE_TOKEN", "t")
    _reset_client()
    yield service
    service.shutdown()
    _reset_client()


def test_timeout_retries_on_fresh_worker_and_kills_the_stuck_browser(small):
    hang = "new Promise(() => {})"
    w0 = small.pool.status()["workers"][0]["pid"]
    tree0 = _descendants(w0)
    assert tree0, "worker should own a browser process tree"
    t0 = time.time()
    with pytest.raises(client.WorkerFailed, match="exceeded"):
        client.call("render_url", {"url": "about:blank", "width": 50, "height": 50, "wait_ms": 0, "script": hang,
                                   "wait_until": "load"}, timeout_s=1.5)
    assert time.time() - t0 < 60
    c = small.pool.status()["counters"]
    assert c["timeouts"] == 2 and c["retries"] == 1 and c["worker_errors"] == 1
    time.sleep(0.5)
    alive = []
    for p in [w0] + tree0:
        try:
            os.kill(p, 0)
            alive.append(p)
        except OSError:
            pass
    assert not alive, f"stuck worker tree survived: {alive}"
    # the pool heals (replacement spawned in the background) and serves again
    png = client.render_html("<b>ok</b>", 20, 10)
    assert png[:4] == b"\x89PNG"


def test_backpressure_and_recycling(small):
    hang = "new Promise(() => {})"
    t = threading.Thread(target=lambda: pytest.raises(client.ServiceError, client.call, "render_url",
                                                      {"url": "about:blank", "width": 20, "height": 20, "wait_ms": 0,
                                                       "script": hang, "wait_until": "load"}, timeout_s=3))
    t.start()
    time.sleep(0.5)
    with pytest.raises(client.Busy):
        client.call("render_html", {"html": "<b>x</b>", "width": 10, "height": 10}, busy_retry_s=0)
    t.join(60)
    deadline = time.time() + 60
    while small.pool.status()["idle"] < 1 and time.time() < deadline:
        time.sleep(0.2)
    pids = set()
    for i in range(7):
        hdr, png = client.call("render_html", {"html": f"<b>{i}</b>", "width": 10, "height": 10})
        pids.add(hdr["meta"]["worker"])
    assert small.pool.status()["counters"]["recycled"] >= 2 and len(pids) >= 3


# --------------------------------------------------------------------------- cache unit
def test_render_cache_lru(tmp_path):
    c = RenderCache(str(tmp_path / "c"), max_bytes=1000)
    for i in range(3):
        c.put(f"{i:064x}", bytes([i]) * 300)
    assert c.get(f"{0:064x}") == bytes([0]) * 300  # 0 is now most recently used
    c.put(f"{3:064x}", b"z" * 300)                 # over the cap -> evict LRU (1) down to 90 %
    assert c.get(f"{1:064x}") is None and c.get(f"{0:064x}") is not None
    st = c.stats()
    assert st["evictions"] >= 1 and st["bytes"] <= 900 and st["hits"] == 2 and st["misses"] == 1
    c2 = RenderCache(str(tmp_path / "c"), max_bytes=1000)  # index rebuilt from disk
    assert c2.stats()["entries"] == st["entries"]


# --------------------------------------------------------------------------- CLI end to end
def test_cli_serve_background_status_stop(tmp_path):
    env = dict(os.environ, DT_HOME=str(tmp_path / "h"))
    env.pop("DT_EVAL_SERVICE", None)
    run = lambda *a: subprocess.run([sys.executable, "-m", "dt.cli", *a], cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
    r = run("serve", "--background", "--workers", "1", "--port", "0", "--json")
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    try:
        st = run("service", "status", "--json")
        assert st.returncode == 0 and json.loads(st.stdout)["pool"]["size"] == 1
        assert run("serve", "--workers", "1", "--port", "0").returncode == 1  # one per DT_HOME
        stats = run("service", "stats", "--json")
        assert stats.returncode == 0 and "cache" in json.loads(stats.stdout)
    finally:
        r = run("service", "stop", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["stopped"], r.stderr
    with pytest.raises(OSError):
        os.kill(info["pid"], 0)
    assert run("service", "status").returncode == 1


# --------------------------------------------------------------------------- verifier regressions
def _browser_pids(worker_pid: int) -> list[int]:
    """The worker's browser main process(es): Playwright launches Chrome with --remote-debugging-pipe."""
    kids = _descendants(worker_pid)
    out = subprocess.run(["ps", "-ww", "-o", "pid=,command=", "-p", ",".join(map(str, kids))], capture_output=True,
                         text=True).stdout if kids else ""
    return [int(ln.split(None, 1)[0]) for ln in out.splitlines() if "--remote-debugging-pipe" in ln and "--type=" not in ln]


def test_worker_whose_browser_died_is_replaced(small):
    """A worker whose Chrome crashed (OOM kill, GPU crash) must not keep answering every job with
    TargetClosedError: the job is retried on a fresh worker and the pool heals."""
    w0 = small.pool.status()["workers"][0]["pid"]
    bpids = _browser_pids(w0)
    assert bpids, "worker should own a Chrome process"
    for p in bpids:
        os.kill(p, 9)
    time.sleep(0.5)
    for i in range(2):
        hdr, png = client.call("render_html", {"html": f"<b>{i}</b>", "width": 10, "height": 10})
        assert png[:4] == b"\x89PNG" and hdr["meta"]["worker"] != w0
    assert small.pool.status()["counters"]["job_errors"] == 0


def test_malformed_jobs_get_400_and_the_service_keeps_serving(small):
    addr = client.address()
    bad = [{"job": "render_html", "args": {"width": 10, "height": 10}},                 # no html
           {"job": "render_html", "args": {"html": "x", "width": "wide", "height": 10}},
           {"job": "render_html", "args": [1, 2]},
           {"job": "render_html", "args": {"html": "x", "width": 5, "height": 5}, "timeout_s": "soon"},
           {"job": "render_doc", "args": {"doc": {"nope": 1}}, "client": client.client_info()}]
    for req in bad:
        code, _ct, data = client.request(addr, "POST", "/v1/job", client.pack(req), timeout=30)
        assert code == 400, (req, code, data[:200])
        assert client.unpack(data)[0]["ok"] is False
    code, _ct, data = client.request(addr, "POST", "/v1/job", b"\x00\x00\x00\x02[]", timeout=30)
    assert code == 400
    with pytest.raises(client.ServiceError) as ei:
        client.render_doc({"nope": 1})
    assert not isinstance(ei.value, client.Unavailable)  # a bad request must not mark the service down
    assert client.render_html("<b>ok</b>", 20, 10)[:4] == b"\x89PNG"


def test_pool_that_lost_every_worker_fails_fast_and_heals(home, monkeypatch):
    """Browser unlaunchable for a while (e.g. Chrome mid-update): jobs must fail fast (clients render
    locally) instead of waiting service.queue_wait_s, and the pool must come back once Chrome does."""
    service = EvalService(workers=1, port=0, write_state=False, token="t", cache=False).start_background()
    try:
        monkeypatch.setenv("DT_EVAL_SERVICE", service.url)
        monkeypatch.setenv("DT_EVAL_SERVICE_TOKEN", "t")
        _reset_client()
        info = client.client_info
        monkeypatch.setattr(client, "client_info", lambda code=False: dict(info(code), browser_pref="auto"))
        monkeypatch.setenv("DT_BROWSER", "/nonexistent/chrome-for-dt-test")  # inherited by respawned workers only
        os.kill(service.pool.status()["workers"][0]["pid"], 9)
        with pytest.raises(client.WorkerFailed):
            client.call("render_html", {"html": "<b>a</b>", "width": 10, "height": 10}, busy_retry_s=0)
        time.sleep(8)  # several replacement attempts fail
        for _ in range(2):
            t0 = time.time()
            with pytest.raises(client.WorkerFailed):
                client.call("render_html", {"html": "<b>b</b>", "width": 10, "height": 10}, busy_retry_s=0)
            assert time.time() - t0 < 20, "job waited for a worker that will not come"
        monkeypatch.delenv("DT_BROWSER")
        deadline = time.time() + 90
        while service.pool.status()["idle"] < 1 and time.time() < deadline:
            time.sleep(0.5)
        hdr, png = client.call("render_html", {"html": "<b>c</b>", "width": 10, "height": 10}, busy_retry_s=0)
        assert png[:4] == b"\x89PNG"
    finally:
        service.shutdown()
        _reset_client()
