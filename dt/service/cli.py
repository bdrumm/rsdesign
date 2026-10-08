"""``dt serve`` and ``dt service status|stats|stop`` (registered by dt/cli.py's subsystem hook).

    dt serve                         # foreground; Ctrl-C / SIGTERM / `dt service stop` stops it
    dt serve --background            # detach, log to $DT_HOME/service.log, return once ready
    dt serve --workers 3 --port 47615 | --socket ~/.rsdesign/eval.sock
    dt service status [--json]       # exit 1 when no service is reachable
    dt service stats  [--json]       # + cache hit rate, per-job latency percentiles
    dt service stop
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

EXIT_OK, EXIT_FAIL = 0, 1


def _addr():
    """The service this command talks to: DT_EVAL_SERVICE if set, else $DT_HOME/service.json."""
    from dt.service import client
    a = client.address()
    if a is None and (os.environ.get("DT_EVAL_SERVICE") or "").strip().lower() in ("", "0", "off", "false", "no", "none"):
        old = os.environ.get("DT_EVAL_SERVICE")
        os.environ["DT_EVAL_SERVICE"] = "auto"
        try:
            a = client.address()
        finally:
            if old is None:
                os.environ.pop("DT_EVAL_SERVICE", None)
            else:
                os.environ["DT_EVAL_SERVICE"] = old
    return a


def _running():
    from dt.service import client
    a = _addr()
    if a is None:
        return None, None
    try:
        return a, client.status(a)
    except client.ServiceError:
        return a, None


def _human_status(st: dict) -> str:
    pool = st.get("pool", {})
    c = pool.get("counters", {})
    lines = [f"eval service {st['url']}  pid {st['pid']}  up {st['uptime_s']:.0f}s  {st['browser'].get('name')} {st['browser'].get('version')}"
             + ("  STALE (renderer sources changed: restart)" if st.get("stale") else ""),
             f"workers {pool.get('alive')}/{pool.get('size')}  idle {pool.get('idle')}  in flight {pool.get('in_flight')}  "
             f"waiting {pool.get('waiting')}  capacity {pool.get('capacity')}",
             f"jobs {c.get('jobs')}  ok {c.get('ok')}  job errors {c.get('job_errors')}  worker errors {c.get('worker_errors')}  "
             f"timeouts {c.get('timeouts')}  crashes {c.get('crashes')}  retries {c.get('retries')}  recycled {c.get('recycled')}  "
             f"rejected(busy) {c.get('rejected_busy')}"]
    cache = st.get("cache")
    if cache:
        lines.append(f"cache {cache['entries']} entries {cache['bytes'] / 1e6:.1f}/{cache['max_bytes'] / 1e6:.0f} MB  hits {cache['hits']}  "
                     f"misses {cache['misses']}  hit rate {cache['hit_rate']}  evictions {cache['evictions']}  saved {cache['saved_s']}s")
    for k, v in (st.get("latency") or {}).items():
        if v.get("n"):
            lines.append(f"  {k:18s} n={v['n']:6d}  p50 {v['p50'] * 1000:7.1f} ms  p95 {v['p95'] * 1000:7.1f} ms  max {v['max'] * 1000:7.1f} ms")
    return "\n".join(lines)


def cmd_serve(a: argparse.Namespace) -> int:
    from dt.service import ROOT, dt_home, state_path
    addr, st = _running()
    if st is not None:
        print(f"an eval service is already running at {st['url']} (pid {st['pid']}); `dt service stop` first", file=sys.stderr)
        return EXIT_FAIL
    if a.background:
        os.makedirs(dt_home(), exist_ok=True)
        log = os.path.join(dt_home(), "service.log")
        argv = [sys.executable, "-m", "dt.cli", "serve"] + _forward(a)
        with open(log, "ab") as lf:
            proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=lf, stderr=lf, start_new_session=True)
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"eval service exited with {proc.returncode}; see {log}", file=sys.stderr)
                return EXIT_FAIL
            try:
                with open(state_path()) as f:
                    pid = json.load(f).get("pid")
            except (OSError, ValueError):
                pid = None
            if pid == proc.pid:
                _a, st = _running()
                if st is not None:
                    res = {"ok": True, "pid": proc.pid, "url": st["url"], "workers": st["pool"]["size"], "log": log}
                    print(json.dumps(res) if a.json else f"eval service ready at {st['url']} (pid {proc.pid}, "
                          f"{st['pool']['size']} workers); export DT_EVAL_SERVICE=1; log {log}")
                    return EXIT_OK
            time.sleep(0.3)
        print(f"eval service did not become ready; see {log}", file=sys.stderr)
        return EXIT_FAIL

    from dt.service.server import EvalService
    from dt.params import P
    if a.max_jobs_per_worker:
        P.set("service.max_jobs_per_worker", a.max_jobs_per_worker)
    if a.job_timeout:
        P.set("service.job_timeout_s", a.job_timeout)
    svc = EvalService(workers=a.workers, port=a.port, socket_path=a.socket, cache=False if a.no_cache else None)
    svc.start()

    def _stop(*_: Any) -> None:
        threading.Thread(target=svc.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if a.json:
        print(json.dumps({"ok": True, "pid": os.getpid(), "url": svc.url, "workers": svc.n}), flush=True)
    try:
        svc.httpd.serve_forever(poll_interval=0.5)
    finally:
        svc.shutdown()
        svc.wait(60)
    return EXIT_OK


def _forward(a: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for flag, v in (("--workers", a.workers), ("--port", a.port), ("--socket", a.socket),
                    ("--max-jobs-per-worker", a.max_jobs_per_worker), ("--job-timeout", a.job_timeout)):
        if v is not None:
            out += [flag, str(v)]
    if a.no_cache:
        out.append("--no-cache")
    return out


def cmd_service(a: argparse.Namespace) -> int:
    from dt.service import client, state_path
    addr, st = _running()
    if a.action in ("status", "stats"):
        if st is None:
            res = {"ok": False, "running": False, "url": addr["url"] if addr else None}
            print(json.dumps(res) if a.json else f"no eval service reachable{' at ' + addr['url'] if addr else ''} "
                  "(start one with `dt serve --background`)")
            return EXIT_FAIL
        if a.action == "stats":
            st = client.stats(addr)
        st["running"] = True
        print(json.dumps(st, indent=2, default=str) if a.json else _human_status(st))
        return EXIT_OK
    # stop
    pid = None
    if st is not None:
        pid = st["pid"]
        try:
            client.stop(addr)
        except client.ServiceError:
            pass
    else:
        try:
            with open(state_path()) as f:
                pid = json.load(f).get("pid")
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ValueError, TypeError):
            pid = None
    if pid is None:
        print(json.dumps({"ok": True, "stopped": False}) if a.json else "no eval service running")
        return EXIT_OK
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            os.kill(int(pid), 0)
        except OSError:
            break
        time.sleep(0.2)
    else:
        print(f"eval service pid {pid} still alive after 60s", file=sys.stderr)
        return EXIT_FAIL
    print(json.dumps({"ok": True, "stopped": True, "pid": pid}) if a.json else f"eval service (pid {pid}) stopped")
    return EXIT_OK


def register(add: Callable[..., argparse.ArgumentParser], sub: Any) -> None:
    p = add("serve", cmd_serve, "run the shared evaluation service (one per machine): a fixed pool of browser "
                                "workers that every process with DT_EVAL_SERVICE set renders through")
    p.add_argument("--workers", type=int, default=None, help="worker processes (default service.workers)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--port", type=int, default=None, help="TCP port on 127.0.0.1 (default service.port)")
    g.add_argument("--socket", default=None, help="serve on this unix socket instead of TCP")
    p.add_argument("--no-cache", action="store_true", help="disable the content-addressed render cache")
    p.add_argument("--max-jobs-per-worker", type=int, default=None)
    p.add_argument("--job-timeout", type=float, default=None, help="seconds (default service.job_timeout_s)")
    p.add_argument("--background", action="store_true", help="detach (log: $DT_HOME/service.log) and return when ready")

    p = add("service", cmd_service, "eval service control: status | stats | stop")
    p.add_argument("action", choices=("status", "stats", "stop"))
