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
from dt.render.html import render_html

_lock = threading.RLock()
# Playwright's sync objects are bound to the thread that created them (greenlets), so each thread gets
# its own driver/browser/context. Single-threaded use (CLI, tests) still has exactly one browser; servers
# that run work on worker threads (the MCP server) get one per worker thread instead of a crash.
_tls = threading.local()
_all_states: list = []  # every thread's state, for shutdown()


def _st():
    s = getattr(_tls, "state", None)
    if s is None:
        s = {"pw": None, "browser": None, "context": None}
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


atexit.register(shutdown)


_TMP_DIR = os.environ.get("DT_TMP", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out", ".render"))
_tmp_counter = 0


def _tmp_html(html: str) -> str:
    global _tmp_counter
    os.makedirs(_TMP_DIR, exist_ok=True)
    _tmp_counter += 1
    path = os.path.join(_TMP_DIR, f"page_{os.getpid()}_{_tmp_counter % 64}.html")
    with open(path, "w") as f:
        f.write(html)
    return path


def html_to_png(html: str, width: int, height: int, out_path: Optional[str] = None, dpr: float = 1.0,
                wait_fonts: bool = True, full_page: bool = False) -> np.ndarray:
    """Render HTML string to PNG. Returns RGB numpy array (and writes to out_path if given)."""
    with _lock:
        _ensure_browser()
        page = _ctx().new_page()
        try:
            page.set_viewport_size({"width": int(width), "height": int(height)})
            # NOTE: set_content() runs at about:blank and cannot load file:// fonts; write + goto instead.
            tmp = _tmp_html(html)
            page.goto("file://" + tmp, wait_until="load")
            if wait_fonts:
                # fonts.ready can resolve before a lazily-requested face starts loading: load every declared
                # face that the page uses, then wait two frames so layout reflects the loaded fonts
                page.evaluate("""async () => {
                    await document.fonts.ready;
                    const used = new Set();
                    for (const el of document.querySelectorAll('body *')) {
                        const cs = getComputedStyle(el); used.add(cs.fontWeight + ' 16px ' + cs.fontFamily.split(',')[0]);
                    }
                    await Promise.all([...used].map(f => document.fonts.load(f).catch(() => null)));
                    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
                    return true;
                }""")
            if dpr != 1.0:
                # emulate DPR by CSS zoom; screenshot scale handled by caller downscale
                page.evaluate(f"document.body.style.zoom='{dpr}'")
            data = page.screenshot(type="png", full_page=full_page, animations="disabled", caret="hide")
        finally:
            page.close()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        return load_rgb(out_path)
    from io import BytesIO
    from PIL import Image
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()


def render_doc(doc: Document, out_path: Optional[str] = None, mode: str = "absolute", extra_css: str = "") -> np.ndarray:
    """IR Document -> RGB array (optionally also written to out_path)."""
    html = render_html(doc, mode=mode, extra_css=extra_css)
    return html_to_png(html, doc.width, doc.height, out_path=out_path)


def render_url(url: str, width: int, height: int, out_path: Optional[str] = None, wait_ms: int = 500,
               script: Optional[str] = None, wait_until: str = "networkidle") -> tuple[np.ndarray, object]:
    """Screenshot a URL (or file://) and optionally evaluate `script` in the page; returns (png, script_result)."""
    with _lock:
        _ensure_browser()
        page = _ctx().new_page()
        try:
            page.set_viewport_size({"width": int(width), "height": int(height)})
            page.goto(url, wait_until=wait_until)  # "load" for local pages: networkidle costs >= 500ms
            page.evaluate("document.fonts.ready.then(() => true)")
            if wait_ms:
                page.wait_for_timeout(wait_ms)
            result = page.evaluate(script) if script else None
            data = page.screenshot(type="png", animations="disabled", caret="hide")
        finally:
            page.close()
    from io import BytesIO
    from PIL import Image
    arr = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
    return arr, result


def capture_url(url: str, width: int, height: int, device_scale_factor: float = 1.0, out_path: Optional[str] = None,
                wait_ms: int = 300, wait_until: str = "load") -> np.ndarray:
    """Screenshot a URL at a device pixel ratio (e.g. a real @2x/@3x capture) using the shared browser.
    The image is width*dpr x height*dpr device pixels, rendered natively at that density (not upscaled)."""
    with _lock:
        _ensure_browser()
        ctx = _brw().new_context(viewport={"width": int(width), "height": int(height)}, device_scale_factor=float(device_scale_factor))
        try:
            page = ctx.new_page()
            page.goto(url, wait_until=wait_until)
            page.evaluate("document.fonts.ready.then(() => true)")
            if wait_ms:
                page.wait_for_timeout(wait_ms)
            data = page.screenshot(type="png", animations="disabled", caret="hide")
        finally:
            ctx.close()
    from io import BytesIO
    from PIL import Image
    arr = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
    return arr
