"""HTML -> PNG via Playwright on the system Chrome (channel='chrome', no browser download);
falls back to Playwright's bundled Chromium when Chrome is missing (``DT_BROWSER`` overrides).

A single shared browser is kept alive per process for speed (~30ms/screenshot after warmup).
All renders are at deviceScaleFactor=1 unless `dpr` is passed, so pixels == IR units.
"""
from __future__ import annotations

import atexit
import os
import threading
from typing import Optional

import numpy as np

from dt.common.image import load_rgb
from dt.ir import Document
from dt.params import P, register
from dt.render.html import render_html

register("render.persistent_page", True, "render_doc swaps documents into one long-lived page per thread (fonts stay loaded); ~5x faster, pixel-identical")
register("render.persistent_mode", "rewrite", "persistent page update: 'rewrite' (document.write a fresh document: full raster, exact) | 'swap' (style+body swap: faster, can differ by a few AA pixels)")
register("render.font_wait_ms", 5000, "upper bound (ms) on waiting for web fonts before a capture", (500, 60000))
register("render.frame_wait_ms", 2000, "upper bound (ms) on waiting for two animation frames before a capture", (100, 30000))
register("render.persistent_recycle", 500, "recycle the long-lived page after this many renders", (10, 100000))

_lock = threading.RLock()
# Playwright's sync objects are bound to the thread that created them (greenlets), so each thread gets
# its own driver/browser/context. Single-threaded use (CLI, tests) still has exactly one browser; servers
# that run work on worker threads (the MCP server) get one per worker thread instead of a crash.
_tls = threading.local()
_all_states: list = []  # every thread's state, for shutdown()


def _st():
    s = getattr(_tls, "state", None)
    if s is None:
        s = {"pw": None, "browser": None, "context": None, "page": None}
        _tls.state = s
        _all_states.append(s)
    return s


_LAUNCH_ARGS = ["--font-render-hinting=none", "--disable-lcd-text"]
BROWSER_USED: Optional[str] = None  # which browser the shared instance runs on (for `dt doctor`)
BROWSER_HELP = ("install Google Chrome, or run `python -m playwright install chromium` "
                "(Linux: also `python -m playwright install-deps chromium`, needs sudo), "
                "or point DT_BROWSER at a Chromium executable")


def browser_candidates(want: Optional[str] = None) -> list[str]:
    """Browsers to try, in order. ``DT_BROWSER``: auto (system Chrome, then Playwright's bundled
    Chromium) | chrome | chromium | msedge | chrome-beta | ... | /path/to/chromium-executable."""
    want = (want or os.environ.get("DT_BROWSER") or "auto").strip()
    return ["chrome", "chromium"] if want == "auto" else [want]


def launch_browser(pw, candidates: Optional[list[str]] = None):
    """Launch the first working browser of `candidates`; returns (browser, name). Raises RuntimeError
    with an actionable message when none starts. Renders are only comparable on the same browser
    build: the shipped baselines were made with system Chrome."""
    errors: list[str] = []
    for c in candidates or browser_candidates():
        try:
            if os.sep in c or c.startswith("."):
                b = pw.chromium.launch(executable_path=c, headless=True, args=_LAUNCH_ARGS)
            elif c == "chromium":
                b = pw.chromium.launch(headless=True, args=_LAUNCH_ARGS)
            else:
                b = pw.chromium.launch(channel=c, headless=True, args=_LAUNCH_ARGS)
            return b, c
        except Exception as e:  # missing channel / executable / system libraries
            errors.append(f"{c}: {str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__}")
    raise RuntimeError("no browser could be launched for rendering. Fix: " + BROWSER_HELP + ". Tried: " + "; ".join(errors))


def _ensure_browser():
    global BROWSER_USED
    st = _st()
    if st["browser"] is not None:
        return
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    try:
        browser, BROWSER_USED = launch_browser(pw)
    except Exception:
        pw.stop()  # never leave a started driver behind: the next call starts a fresh one
        raise
    st["pw"], st["browser"] = pw, browser
    st["context"] = browser.new_context(viewport={"width": 800, "height": 600}, device_scale_factor=1)


def _ctx():
    return _st()["context"]


def _brw():
    return _st()["browser"]


def shutdown() -> None:
    """Close the calling thread's browser (and, at exit, any thread's browser that can still be closed)."""
    for st in [_st()] + [s for s in _all_states if s is not _st()]:
        try:
            if st["context"]:
                st["context"].close()
            if st["browser"]:
                st["browser"].close()
            if st["pw"]:
                st["pw"].stop()
        except Exception:
            pass
        st["pw"] = st["browser"] = st["context"] = None
        st["page"] = None


atexit.register(shutdown)


_TMP_DIR = os.environ.get("DT_TMP", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out", ".render"))
_tmp_counter = 0


def _tmp_html(html: str, tmp_dir: Optional[str] = None) -> str:
    global _tmp_counter
    tmp_dir = tmp_dir or _TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)
    _tmp_counter += 1
    path = os.path.join(tmp_dir, f"page_{os.getpid()}_{_tmp_counter % 64}.html")
    with open(path, "w") as f:
        f.write(html)
    return path


# --------------------------------------------------------------------------- opt-in eval service
# With DT_EVAL_SERVICE set and the shared service (`dt serve`, dt/service) reachable and compatible,
# the four public functions below get their PNG bytes from the service's browser pool instead of this
# process's browser; post-processing (decode, out_path) is the same code either way. Unset, or service
# down, everything below runs exactly as before.
def _service_render(job: str, args: dict):
    """(header, png bytes) from the eval service, or None to render locally."""
    if not os.environ.get("DT_EVAL_SERVICE"):
        return None
    try:
        from dt.service import client
    except Exception:  # noqa: BLE001 - a broken optional subsystem must never break rendering
        return None
    res = client.delegate(job, args)
    if res is not None:
        global BROWSER_USED
        BROWSER_USED = BROWSER_USED or client.browser_name()
    return res


def html_to_png(html: str, width: int, height: int, out_path: Optional[str] = None, dpr: float = 1.0,
                wait_fonts: bool = True, full_page: bool = False) -> np.ndarray:
    """Render HTML string to PNG. Returns RGB numpy array (and writes to out_path if given)."""
    res = _service_render("render_html", {"html": html, "width": int(width), "height": int(height), "dpr": float(dpr),
                                          "wait_fonts": bool(wait_fonts), "full_page": bool(full_page), "tmp_dir": _TMP_DIR})
    data = res[1] if res is not None else _html_to_png_bytes(html, width, height, dpr, wait_fonts, full_page)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        return load_rgb(out_path)
    from io import BytesIO
    from PIL import Image
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()


def _html_to_png_bytes(html: str, width: int, height: int, dpr: float = 1.0, wait_fonts: bool = True,
                       full_page: bool = False, tmp_dir: Optional[str] = None) -> bytes:
    """The local render behind :func:`html_to_png` (also what eval-service workers run): PNG bytes."""
    if bool(P["render.persistent_page"]) and dpr == 1.0 and wait_fonts and not full_page and _split_doc_html(html):
        with _lock:
            _ensure_browser()
            try:
                data = _render_persistent(html, width, height)
                if data is not None:
                    return data
            except Exception:  # fall through to a fresh page
                st = _st()
                try:
                    if st.get("page") is not None:
                        st["page"].close()
                except Exception:
                    pass
                st["page"] = None
    with _lock:
        _ensure_browser()
        page = _ctx().new_page()
        try:
            page.set_viewport_size({"width": int(width), "height": int(height)})
            # NOTE: set_content() runs at about:blank and cannot load file:// fonts; write + goto instead.
            tmp = _tmp_html(html, tmp_dir)
            page.goto("file://" + tmp, wait_until="load")
            if wait_fonts:
                # fonts.ready can resolve before a lazily-requested face starts loading: load every declared
                # face that the page uses, then wait two frames so layout reflects the loaded fonts
                page.evaluate("""async () => {
                    await Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, %d))]);
                    const used = new Set();
                    for (const el of document.querySelectorAll('body *')) {
                        const cs = getComputedStyle(el); used.add(cs.fontWeight + ' 16px ' + cs.fontFamily.split(',')[0]);
                    }
                    const cap = (p, ms) => Promise.race([p, new Promise(r => setTimeout(r, ms))]);
                    await cap(Promise.all([...used].map(f => document.fonts.load(f).catch(() => null))), %d);
                    await cap(new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))), %d);
                    return true;
                }""" % (int(P["render.font_wait_ms"]), int(P["render.font_wait_ms"]), int(P["render.frame_wait_ms"])))
            if dpr != 1.0:
                # emulate DPR by CSS zoom; screenshot scale handled by caller downscale
                page.evaluate(f"document.body.style.zoom='{dpr}'")
            data = page.screenshot(type="png", full_page=full_page, animations="disabled", caret="hide")
        finally:
            page.close()
    return data


_FONT_WAIT_JS = """async ([fontMs, frameMs]) => {
    // every wait is bounded: a page Chrome treats as hidden pauses requestAnimationFrame, and an unbounded
    // await then blocks the whole renderer (seen once as a silent hang of a scenario gate)
    const cap = (p, ms) => Promise.race([p, new Promise(r => setTimeout(r, ms))]);
    await cap(document.fonts.ready, fontMs);
    const used = new Set();
    for (const el of document.querySelectorAll('body *')) {
        const cs = getComputedStyle(el); used.add(cs.fontWeight + ' 16px ' + cs.fontFamily.split(',')[0]);
    }
    await cap(Promise.all([...used].map(f => document.fonts.load(f).catch(() => null))), fontMs);
    await cap(new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))), frameMs);
    return true;
}"""


def _split_doc_html(html: str) -> Optional[tuple[str, str]]:
    """(css, body) of a page produced by dt.render.html.render_html (marked with <meta name="dt-render">), else
    None: arbitrary pages may carry scripts or stylesheet links in <head> that a body swap would drop."""
    if '<meta name="dt-render" content="1">' not in html[:2000]:
        return None
    i, j = html.find("<style>"), html.find("</style>")
    b0, b1 = html.find("<body>"), html.rfind("</body>")
    if min(i, j, b0, b1) < 0:
        return None
    return html[i + len("<style>"):j], html[b0 + len("<body>"):b1]


def _render_persistent(html: str, width: int, height: int) -> Optional[bytes]:
    """Render a render_html() page into ONE long-lived page per thread whose fonts are already in memory: no
    navigation and no temp file. Measured ~2x faster than a fresh page at low load (82 vs 161 ms) and up to
    ~5x under heavy load; pixel-identical to a fresh page (corpus, back-to-back same-size variants, the
    adversary benchmark). Returns None to request the normal path."""
    parts = _split_doc_html(html)
    if parts is None:
        return None
    css, body = parts
    st = _st()
    page = st.get("page")
    if page is None or page.is_closed():
        page = _ctx().new_page()
        page.set_viewport_size({"width": int(width), "height": int(height)})
        page.goto("file://" + _tmp_html(html), wait_until="load")
        st["page"] = page
    else:
        try:
            page.bring_to_front()
        except Exception:
            pass
        # a viewport change forces a full re-raster: without it, back-to-back renders of the same size reuse
        # raster tiles and only repaint invalidated regions, whose anti-aliasing can differ from a fresh page
        page.set_viewport_size({"width": int(width) + 1, "height": int(height)})
        page.set_viewport_size({"width": int(width), "height": int(height)})
        if str(P["render.persistent_mode"]) == "swap":
            page.evaluate("""([css, body]) => { document.querySelector('style').textContent = css;
                                                 document.body.innerHTML = body; }""", [css, body])
        else:
            # replace the whole DOCUMENT in place: a fresh document gets a full raster (a body swap only
            # re-rasterises invalidated regions, and Chrome's anti-aliasing depends on raster-tile placement,
            # which made back-to-back same-size renders differ from a fresh page by a few pixels)
            page.evaluate("""(html) => { document.open(); document.write(html); document.close(); }""", html)
    st["page_renders"] = st.get("page_renders", 0) + 1
    page.evaluate(_FONT_WAIT_JS, [int(P["render.font_wait_ms"]), int(P["render.frame_wait_ms"])])
    data = page.screenshot(type="png", animations="disabled", caret="hide")
    if st["page_renders"] >= int(P["render.persistent_recycle"]):  # bound any growth of a long-lived page
        page.close()
        st["page"], st["page_renders"] = None, 0
    return data


def render_doc(doc: Document, out_path: Optional[str] = None, mode: str = "absolute", extra_css: str = "") -> np.ndarray:
    """IR Document -> RGB array (optionally also written to out_path)."""
    html = render_html(doc, mode=mode, extra_css=extra_css)
    if os.environ.get("DT_EVAL_SERVICE"):
        # the shared service owns the browsers (and its workers use the persistent page themselves);
        # html_to_png delegates and falls back to local rendering when the service is unavailable
        res = _service_render("render_html", {"html": html, "width": int(doc.width), "height": int(doc.height), "dpr": 1.0,
                                              "wait_fonts": True, "full_page": False, "tmp_dir": _TMP_DIR})
        if res is not None:
            data = res[1]
            if out_path:
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(data)
            from io import BytesIO
            from PIL import Image
            return np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()
    if bool(P["render.persistent_page"]):
        data = None
        with _lock:
            _ensure_browser()
            try:
                data = _render_persistent(html, doc.width, doc.height)
            except Exception:  # never fail a render because of the fast path: drop the page, use the normal one
                st = _st()
                try:
                    if st.get("page") is not None:
                        st["page"].close()
                except Exception:
                    pass
                st["page"], data = None, None
        if data is not None:
            if out_path:
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(data)
            from io import BytesIO
            from PIL import Image
            return np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()
    return html_to_png(html, doc.width, doc.height, out_path=out_path)


def render_url(url: str, width: int, height: int, out_path: Optional[str] = None, wait_ms: int = 500,
               script: Optional[str] = None, wait_until: str = "networkidle") -> tuple[np.ndarray, object]:
    """Screenshot a URL (or file://) and optionally evaluate `script` in the page; returns (png, script_result)."""
    res = _service_render("render_url", {"url": url, "width": int(width), "height": int(height), "wait_ms": int(wait_ms),
                                         "script": script, "wait_until": wait_until})
    if res is not None:
        data, result = res[1], res[0].get("result")
    else:
        data, result = _render_url_bytes(url, width, height, wait_ms, script, wait_until)
    from io import BytesIO
    from PIL import Image
    arr = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
    return arr, result


def _render_url_bytes(url: str, width: int, height: int, wait_ms: int = 500, script: Optional[str] = None,
                      wait_until: str = "networkidle") -> tuple[bytes, object]:
    """The local render behind :func:`render_url`: (PNG bytes, script result)."""
    with _lock:
        _ensure_browser()
        page = _ctx().new_page()
        try:
            page.set_viewport_size({"width": int(width), "height": int(height)})
            page.goto(url, wait_until=wait_until)  # "load" for local pages: networkidle costs >= 500ms
            page.evaluate("Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 5000))]).then(() => true)")
            if wait_ms:
                page.wait_for_timeout(wait_ms)
            result = page.evaluate(script) if script else None
            data = page.screenshot(type="png", animations="disabled", caret="hide")
        finally:
            page.close()
    return data, result


def capture_url(url: str, width: int, height: int, device_scale_factor: float = 1.0, out_path: Optional[str] = None,
                wait_ms: int = 300, wait_until: str = "load") -> np.ndarray:
    """Screenshot a URL at a device pixel ratio (e.g. a real @2x/@3x capture) using the shared browser.
    The image is width*dpr x height*dpr device pixels, rendered natively at that density (not upscaled)."""
    res = _service_render("capture_url", {"url": url, "width": int(width), "height": int(height),
                                          "device_scale_factor": float(device_scale_factor), "wait_ms": int(wait_ms),
                                          "wait_until": wait_until})
    data = res[1] if res is not None else _capture_url_bytes(url, width, height, device_scale_factor, wait_ms, wait_until)
    from io import BytesIO
    from PIL import Image
    arr = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
    return arr


def _capture_url_bytes(url: str, width: int, height: int, device_scale_factor: float = 1.0, wait_ms: int = 300,
                       wait_until: str = "load") -> bytes:
    """The local render behind :func:`capture_url`: PNG bytes."""
    with _lock:
        _ensure_browser()
        ctx = _brw().new_context(viewport={"width": int(width), "height": int(height)}, device_scale_factor=float(device_scale_factor))
        try:
            page = ctx.new_page()
            page.goto(url, wait_until=wait_until)
            page.evaluate("Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 5000))]).then(() => true)")
            if wait_ms:
                page.wait_for_timeout(wait_ms)
            data = page.screenshot(type="png", animations="disabled", caret="hide")
        finally:
            ctx.close()
    return data
