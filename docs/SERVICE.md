# Shared evaluation service (`dt serve`)

Many agents / worktrees / processes on one machine each launching their own Chrome does not scale:
we saw 84 Chrome processes, load 12 on 15 cores and 30 s browser timeouts. The evaluation service is
one fixed pool of browser-owning worker processes that everybody submits render (and, optionally,
perceive / bench-case) jobs to.

## Use it

```bash
dt serve --background            # ONCE per machine (per DT_HOME); returns when the workers are ready
export DT_EVAL_SERVICE=1         # in EVERY agent / worktree / shell that renders
dt service status                # pool, counters, browser; exit 1 if none is reachable
dt service stats                 # + render cache hit rate and per-job latency p50/p95
dt service stop
```

* `DT_EVAL_SERVICE`: `1`/`auto` reads `$DT_HOME/service.json` (written by `dt serve`, mode 0600, holds
  the URL and a per-start auth token); or an explicit `http://127.0.0.1:PORT` / `unix:/path.sock`
  (token from `DT_EVAL_SERVICE_TOKEN` or the state file). Unset / `0` / `off`: nothing changes.
* `DT_HOME` (default `~/.rsdesign`): `service.json`, `service.log`, `cache/render/`.
* `dt serve [--workers N] [--port P | --socket PATH] [--no-cache] [--max-jobs-per-worker K]
  [--job-timeout S] [--background]`. Localhost only (TCP bound to 127.0.0.1, or a 0600 unix socket),
  every request needs the token. One service per `DT_HOME`: a second `dt serve` refuses to start.

With `DT_EVAL_SERVICE` set, `dt.render.screenshot.html_to_png / render_doc / render_url /
capture_url` get their PNG bytes from the service; decoding and `out_path` handling are the same code
as before, so callers (bench, perceive's font/icon renders, refine, translate, tests) need no change.
The HTML is still built in the calling process with *its* `dt/render/html.py`, so a worktree that
edits the HTML generator is rendered correctly by a service started from `main`.

**Fallback (never breaks a workflow):** the client renders locally, logging the reason once per
process, when the service is unreachable, saturated for longer than `service.client.busy_retry_s`,
incompatible (the client's `dt/render/screenshot.py` differs from the service's, or `DT_BROWSER`
asks for another browser), a job's worker timed out twice, or the job itself raised (the local path
then raises the genuine exception type). Reachability is re-probed every `service.client.probe_ttl_s`
(5 s), so a restarted service is picked up without restarting clients.

## How it works

| piece | what |
|---|---|
| `dt/service/server.py` | `ThreadingHTTPServer` (or unix socket) front; `Pool` of `spawn` worker processes, each owning one Playwright browser through `dt.render.screenshot` (the code local renders use). Workers drop `DT_EVAL_SERVICE`, so nothing they run can call back into their own pool. |
| admission / backpressure | at most `workers + service.queue_max` (64) jobs in the service; more get HTTP 503 and the client backs off (50 ms → 1 s) and retries. Admitted jobs wait ≤ `service.queue_wait_s` for an idle worker. |
| timeout + retry | a job over `service.job_timeout_s` (180 s; a client may ask for less) or whose worker died is retried once on a freshly spawned worker; the stuck worker is SIGKILLed with its whole process tree (Playwright starts Chrome in its own process group, so the tree is walked, not the group). |
| recycling | a worker is replaced (fresh process + browser) after `service.max_jobs_per_worker` (500) jobs; workers exit on their own when the service dies. |
| dead browser | a worker whose Chrome died (crash, OOM kill) exits instead of answering every later job with `TargetClosedError`; the pool retries the job on a fresh worker. Replacement spawns retry until they succeed (backoff ≤ 30 s), and while no worker is alive or starting, jobs fail fast (clients render locally) instead of waiting `queue_wait_s`. |
| malformed requests | bad frames / missing or mistyped args / an unparsable IR get HTTP 400 (never a dropped connection, which a client would read as "service down"); the token is checked before a body is read. |
| `dt/service/cache.py` | content-addressed render cache `DT_HOME/cache/render/<k[:2]>/<k>.png`, key = sha256(html bytes, w, h, dpr, wait_fonts/full_page, renderer fingerprint = browser name + version + sha of the client checkout's `fixtures/fonts` + `fixtures/icons` font files + `dt/render/html.py` + `dt/render/screenshot.py`, + size/mtime of any other `file://` resource the HTML references). LRU by last use with a size cap (`service.cache.max_mb`, 2048; trims to 90 %). `render_url` / `capture_url` are never cached (pages can change). |
| `dt/service/client.py` | `available()`, `call(job, args, blob)`, typed helpers `render_html / render_doc / render_url / capture_url / perceive / run_case`, `delegate()` (used by screenshot.py), `status() / stats() / stop()`. Wire format: `u32 len | JSON header | raw bytes`, keep-alive connections, no base64. |
| `dt/service/cli.py` | `dt serve`, `dt service status|stats|stop` (registered through dt/cli.py's subsystem hook). |
| `dt/service/loadtest.py` | the measurements below (`python -m dt.service.loadtest determinism|throughput|cache`). |

Jobs: `render_html(html, w, h, dpr)` → PNG · `render_doc(IR json)` → PNG (HTML built with the
service's html.py; refused if it differs from the client's) · `render_url(url, w, h, wait, script)` →
PNG + script result · `capture_url(url, w, h, device_scale_factor)` → PNG · `perceive(png, dpr,
overrides)` → IR json · `run_case(corpus_dir, case_id, stages, overrides)` → bench row (for a future
`dt.selftest.bench` that submits cases; bench.py is unchanged). `perceive` / `run_case` run the
*service checkout's* pipeline code, so they are only served when the client's code fingerprint (all
of `dt/`, `dt/params.json`, design systems, fonts, icon atlases) equals the service's; otherwise
HTTP 409 and the caller works locally.

Params (all `service.*`, not tuned): `workers` (default max(2, performance cores − 1) = 4 on this
machine), `max_jobs_per_worker`, `job_timeout_s`, `queue_max`, `queue_wait_s`, `spawn_timeout_s`,
`port` (47615), `max_body_mb`, `cache.enabled`, `cache.max_mb`, `client.probe_ttl_s`,
`client.connect_timeout_s`, `client.busy_retry_s`.

## Measurements

All on this machine (Apple silicon, 15 cores = 5 performance + 10 efficiency, 24 GB, system Chrome
154.0.8037.98), **shared with another workflow of ~7 agents**: 1-minute load average was 8–20 throughout
(given per row), so absolute times are pessimistic and noisy; ratios within one run are the signal.
Reproduce with `python -m dt.service.loadtest determinism|throughput|cache`.

### 1. Determinism: service renders are byte-identical to local renders

`python -m dt.service.loadtest determinism` (2 workers, cache off; PNG bytes compared, not pixels):

| set | job | identical |
|---|---|---|
| every corpus doc (12 synth + 12 Material Web gt IR) | `render_html` | 24 / 24 |
| every Material Web corpus page (`fixtures/corpus/mwc/*.html`, 600 ms wait, + a DOM script whose result must match too) | `render_url` | 12 / 12 |
| icon batch pages (atlas layout of `dt.perceive.icons.build_atlas`, 400 glyphs each, FILL 0 and 1) | `render_url` | 6 / 6 |
| font batch pages (every self-hosted family x 4 weights x 3 sizes) | `render_html` | 5 / 5 |

`tests/test_service.py` asserts the same on a sample, plus `capture_url` at DPR 2, a bench `run_case`
row and a `perceive` IR (identical up to the random uuid node ids) against local runs.

### 2. Throughput: 6 client processes x 40 `render_doc` of mixed corpus docs

`python -m dt.service.loadtest throughput --clients 6 --renders 40` — all six processes start
rendering at a barrier; wall = first start to last finish; latency = one `render_doc` call as the
caller sees it (incl. HTML build, transfer, decode); Chrome = processes under the test's process tree
(browser + GPU + network + renderer helpers), sampled every 0.2 s. Service runs: cache **off**.
Two runs, the second in reverse order:

| mode | renders/s | p50 ms | p95 ms | max ms | peak Chrome procs | failures | load (1 min) |
|---|---|---|---|---|---|---|---|
| local (each process its own Chrome) — run 1 | 7.94 | 695 | 962 | 2597 | **42** | 0 | 12.1 → 16.3 |
| service, 3 workers — run 1 | 6.48 | 853 | 1376 | 2186 | **21** | 0 | 16.3 → 17.9 |
| service, 5 workers — run 1 | 7.84 | 731 | 1096 | 1477 | **35** | 0 | 18.2 → 20.6 |
| service, 5 workers — run 2 | 9.33 | 622 | 811 | 1220 | 35 | 0 | 10.6 → 12.2 |
| service, 3 workers — run 2 | 7.43 | 804 | 876 | 1192 | 21 | 0 | 13.3 → 15.1 |
| local — run 2 | 8.89 | 616 | 889 | 2225 | 42 | 0 | 15.1 → 18.4 |
| service, 3 workers, **cache on** (24 distinct docs, 240 requests) | **48.4** | **5.8** | 783 | 841 | 21 | 0 | 16.0 → 15.5 |

Reading it: on a CPU-saturated machine a render costs ~300 ms of CPU even in a single warm process,
so a pool cannot add raw throughput — what it does is **cap the browsers**: 5 workers match six
private browsers' throughput (0.99x / 1.05x) with 17 % fewer Chrome processes and a lower worst case
(no per-process Chrome launch: max 1.2–1.5 s vs 2.2–2.6 s); 3 workers give 0.82x / 0.84x with half
the Chrome processes. The local count grows with every agent x bench worker (7 Chrome processes
each; 12 agents x 3 bench workers would be ≈ 250), the service's stays at 7 x workers, which bounds the browser
count behind the load spikes and 30 s browser timeouts we saw. Repeated renders are where it gets faster: with the cache,
6 clients re-rendering corpus docs ran 6x faster (p50 5.8 ms).

### 3. Render cache on a repeated bench pass

`python -m dt.service.loadtest cache --stages render --bench-workers 3` (bench on gt IR, render
stage only; service 3 workers, cold cache; load 16):

| pass | wall s | sum of `render_s` over cases | cache hit rate | composite |
|---|---|---|---|---|
| local (no service) | 8.94 | — | — | 0.99323 |
| service, cold | 9.03 | 13.58 | 0 / 24 | 0.99323 |
| service, warm | **3.50** (2.6x) | **0.35** (39x) | **24 / 24** | 0.99323 |

Full default bench through the service (`DT_EVAL_SERVICE=1 dt bench --workers 3 --gate --quiet`,
perceive + map + render; perceive's font-candidate and icon renders also go to the pool; load 13–16):

| run | wall s | client-side CPU (user) | renders looked up | hit rate | gate |
|---|---|---|---|---|---|
| local, no service | 79.0 | 206 s | — | — | PASS (0.8931) |
| service, cold cache | 73.1 | 62 s | 190 (+44 render_url) | 12 % | PASS (0.8931) |
| service, warm cache | **41.9** (1.75x) | 59 s | 190 (+44 render_url) | **75 %** | PASS (0.8931) |

The warm-pass misses are perceive's font-candidate pages, whose HTML embeds per-run random node ids
(`dt.ir.new_id` is uuid4), and `render_url` icon pages (never cached).

### 4. Test suite with and without the service

| run | result | wall | client CPU (user) | load at start |
|---|---|---|---|---|
| `python -m pytest -q`, `DT_EVAL_SERVICE` unset | 380 passed, 4 skipped | 8:20 | 513 s | 12.7 |
| same with `DT_EVAL_SERVICE=1` (3-worker service) | 380 passed, 4 skipped | 5:59 | 114 s | 14.1 |

With the service, the suite sent 674 jobs (670 ok; 4 jobs raised inside the worker, so those calls
fell back to the local path as designed and the tests still passed; job errors are now logged to
`service.log`) and hit the cache on 298 of 850
`render_html` lookups (35 %), with zero timeouts, crashes or 503s.

## Operational notes

* Restart the service after pulling changes to `dt/render/screenshot.py` or `dt/render/html.py`
  (`dt service status` shows `STALE`); until then clients whose copy differs render locally.
* The service is per user and per machine; it binds 127.0.0.1 only and checks a random token, but
  any process of the same user can submit `render_url` jobs for arbitrary URLs (as it could run
  Chrome itself).
* `dt.selftest.bench --workers N` still spawns N case processes; with the service on, their renders
  (and perceive's font/icon candidate renders) go to the pool, so the number of browsers on the
  machine is the service's worker count, not N per agent.
