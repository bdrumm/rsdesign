"""Shared local evaluation service: one fixed pool of browser-owning worker processes that every agent,
worktree and process on the machine submits render / perceive / bench jobs to, instead of each
process launching its own Chrome.

    dt serve [--workers N] [--port P | --socket PATH] [--background]   # once per machine
    export DT_EVAL_SERVICE=1                                           # every agent / worktree
    dt service status|stats|stop

With ``DT_EVAL_SERVICE`` set and the service reachable, :mod:`dt.render.screenshot` delegates
``html_to_png`` / ``render_doc`` / ``render_url`` / ``capture_url`` to the service (renders are
byte-identical to local ones); when it is unset or the service is down, rendering is local exactly
as before (the client logs the fallback once). See docs/SERVICE.md.

Modules: :mod:`.server` (HTTP front + worker pool), :mod:`.client` (``available()``, ``call()``),
:mod:`.cache` (content-addressed render cache + renderer/code fingerprints), :mod:`.cli`
(``dt serve`` / ``dt service``), :mod:`.loadtest` (the measurements in docs/SERVICE.md).
"""
from __future__ import annotations

import os

from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
API = "v1"


def performance_cores() -> int:
    """Physical performance cores (Apple silicon: hw.perflevel0.physicalcpu), else physical cores, else
    half the logical CPUs."""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        for name in (b"hw.perflevel0.physicalcpu", b"hw.physicalcpu"):
            v, n = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
            if libc.sysctlbyname(name, ctypes.byref(v), ctypes.byref(n), None, ctypes.c_size_t(0)) == 0 and v.value > 0:
                return int(v.value)
    except Exception:  # noqa: BLE001 - not macOS / no sysctlbyname
        pass
    return max(1, (os.cpu_count() or 2) // 2)


register("service.workers", max(2, performance_cores() - 1),
         "eval service: worker processes (each owns one browser); default max(2, performance cores - 1)")
register("service.max_jobs_per_worker", 500, "eval service: recycle a worker (fresh process + browser) after this many jobs")
register("service.job_timeout_s", 180.0, "eval service: per-job timeout; a timed-out job is retried once on a fresh worker")
register("service.queue_max", 64, "eval service: jobs that may wait for a worker beyond the busy ones; more get HTTP 503 (backpressure)")
register("service.queue_wait_s", 900.0, "eval service: longest an admitted job waits for an idle worker before 503")
register("service.spawn_timeout_s", 90.0, "eval service: a new worker must import + launch its browser within this")
register("service.port", 47615, "eval service: default localhost TCP port")
register("service.max_body_mb", 256, "eval service: largest accepted request body")
register("service.cache.enabled", True, "eval service: content-addressed render cache on/off")
register("service.cache.max_mb", 2048, "eval service: render cache size cap (LRU eviction to 90% of it)")
register("service.client.probe_ttl_s", 5.0, "eval client: how long a reachability probe (positive or negative) is trusted")
register("service.client.connect_timeout_s", 1.0, "eval client: connect / status probe timeout")
register("service.client.busy_retry_s", 900.0, "eval client: keep retrying a 503 (pool saturated) this long before rendering locally")


def dt_home() -> str:
    """Per-user state directory (``DT_HOME``, default ``~/.rsdesign``): service.json, service.log, cache/."""
    return os.path.abspath(os.path.expanduser(os.environ.get("DT_HOME") or "~/.rsdesign"))


def state_path() -> str:
    return os.path.join(dt_home(), "service.json")


def cache_dir() -> str:
    return os.path.join(dt_home(), "cache", "render")


__all__ = ["API", "P", "ROOT", "cache_dir", "dt_home", "performance_cores", "state_path"]
