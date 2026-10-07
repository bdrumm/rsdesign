"""Reference screenshot corpus: public Material Design / Google product pages captured programmatically.

This is the "working content" for the pipeline's self-tests on *real* UI: screenshots of the
official Material 3 docs (m3.material.io), the Material Web component demos (material-web.dev)
and public Google product pages that show real app UI. Every screenshot is taken with the same
renderer the rest of the pipeline uses (`dt.render.screenshot.render_url`, Playwright on system
Chrome) at DPR 1, so pixels == IR units.

Layout of the corpus (``fixtures/screens/``)::

    fixtures/screens/<name>.png     one capture per Source (only when it succeeded)
    fixtures/screens/manifest.json  {"version", "captured_at", "dpr", "screens": [entry, ...]}

A manifest *entry* records ``name, url, file, width, height, size, tags, ok, error, page``
where ``size`` is the measured ``[w, h]`` of the PNG, ``ok`` says whether the capture succeeded
and ``error`` holds the failure reason otherwise (sources that fail are still listed so the
run is reproducible and failures are visible).

Capturing needs network access; it is NOT run by the test-suite. Re-capture with::

    python -m dt.selftest.reference_screens            # all sources -> fixtures/screens/
    python -m dt.selftest.reference_screens --only mw_button_types m3_buttons_overview

Each Source may describe a few deterministic page actions executed *inside the page* before the
screenshot (JS evaluated by `render_url`): dismiss/hide overlays, click one element, scroll a
selector or heading into view, or scroll by a pixel offset (this also scrolls inner scroll
containers, which both doc sites use). Overlays are only dismissed when they obstruct content
and always via the most privacy-preserving control (close / reject).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np

from dt.params import P, register

# --------------------------------------------------------------------------- params
register("selftest.screens.wait_ms", 800,
         "Extra wait after network idle before acting on a reference page (ms).", (200, 3000))
register("selftest.screens.settle_ms", 1200,
         "Wait after in-page actions (scroll/click) so lazy content renders before capture (ms).",
         (200, 4000))
register("selftest.screens.min_content_frac", 0.01,
         "Minimum fraction of pixels that differ from the dominant colour for a capture to count "
         "as non-blank; below this the capture is recorded as a failure.", (0.001, 0.2))
register("selftest.screens.blank_quant", 16,
         "Colour quantisation step (per channel) used by the blank-capture check.", (4, 64))
register("selftest.screens.retries", 1,
         "Extra attempts for a source whose navigation times out or errors (network flakiness).", (0, 3))
register("selftest.screens.target_poll_ms", 3000,
         "Max time to poll for a scroll target (selector/heading) that a SPA renders lazily (ms).",
         (0, 8000))
register("selftest.screens.image_poll_ms", 5000,
         "Max time to wait for images inside the viewport to finish loading after scrolling (ms); "
         "doc sites swap blurred placeholders for the real image lazily.", (0, 10000))
register("selftest.screens.mobile_width", 412, "Mobile viewport width for responsive sources (px).")
register("selftest.screens.mobile_height", 915, "Mobile viewport height for responsive sources (px).")
register("selftest.screens.desktop_width", 1280, "Desktop viewport width for reference sources (px).")
register("selftest.screens.desktop_height", 800, "Desktop viewport height for reference sources (px).")

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "fixtures", "screens")

# Overlay selectors that are safe to *hide* (not consent banners; chat widgets / promos that float
# over page content). Consent banners are handled by `dismiss` (click reject/close) per source.
_CHAT_OVERLAYS: tuple[str, ...] = (
    'iframe[title*="chat" i]', '[id*="chat-widget" i]', '[class*="chat-widget" i]',
    '[id*="livechat" i]', '[class*="livechat" i]', '[aria-label*="chat" i][role="dialog"]',
)
# Privacy-preserving consent controls: reject / close only, never "accept".
_CONSENT_REJECT: tuple[str, ...] = (
    'button[aria-label*="reject" i]', 'button[aria-label*="decline" i]',
    '[class*="cookie" i] button[aria-label*="close" i]', '[id*="cookie" i] button[aria-label*="close" i]',
    '[class*="consent" i] button[aria-label*="close" i]',
)


# --------------------------------------------------------------------------- sources
@dataclass(frozen=True)
class Source:
    """One reference capture: a URL, a viewport and optional deterministic in-page actions.

    Actions run in order: ``dismiss`` -> ``hide`` -> ``click`` -> scroll (``scroll_to`` selector,
    else the first ``scroll_text`` heading prefix found (str or ordered tuple of candidates),
    else ``scroll_y`` pixels) -> wait ``settle_ms``. Scroll targets are polled for up to
    ``selftest.screens.target_poll_ms`` because both doc sites render sections lazily.
    ``wait_ms``/``settle_ms`` of ``None`` fall back to the registered params.
    """

    name: str
    url: str
    width: int = 1280
    height: int = 800
    wait_ms: Optional[int] = None
    tags: tuple[str, ...] = ()
    scroll_to: Optional[str] = None
    scroll_text: Optional[str | tuple[str, ...]] = None
    scroll_y: Optional[int] = None
    click: Optional[str] = None
    dismiss: tuple[str, ...] = _CONSENT_REJECT
    hide: tuple[str, ...] = _CHAT_OVERLAYS
    settle_ms: Optional[int] = None

    @property
    def file(self) -> str:
        return f"{self.name}.png"

    def has_actions(self) -> bool:
        return bool(self.click or self.scroll_to or self.scroll_text or self.scroll_y is not None)

    @property
    def scroll_texts(self) -> tuple[str, ...]:
        if self.scroll_text is None:
            return ()
        return (self.scroll_text,) if isinstance(self.scroll_text, str) else tuple(self.scroll_text)


SPEC_HEADINGS: tuple[str, ...] = ("Anatomy", "Measurements", "Tokens & specs")
"""Headings tried in order on m3 'specs' pages: the anatomy diagram when the page has one."""


def _m3(slug: str, page: str, tags: Iterable[str], **kw: Any) -> Source:
    """m3.material.io component page at desktop size; 'specs' pages scroll to the spec diagrams."""
    kw.setdefault("width", P["selftest.screens.desktop_width"])
    kw.setdefault("height", P["selftest.screens.desktop_height"])
    if page == "specs":
        kw.setdefault("scroll_text", SPEC_HEADINGS)
    name = kw.pop("name", f"m3_{slug.replace('-', '_')}_{page}")
    return Source(name=name, url=f"https://m3.material.io/components/{slug}/{page}",
                  tags=("m3", "docs", slug, *tags), **kw)


def _mw(slug: str, tags: Iterable[str], anchor: str = "#types", **kw: Any) -> Source:
    """material-web.dev demo page scrolled to the section that renders live components."""
    kw.setdefault("width", P["selftest.screens.desktop_width"])
    kw.setdefault("height", P["selftest.screens.desktop_height"])
    name = kw.pop("name", f"mw_{slug.replace('-', '_')}_{anchor.lstrip('#').replace('-', '_')}")
    return Source(name=name, url=f"https://material-web.dev/components/{slug}/", scroll_to=anchor,
                  tags=("material-web", "demo", slug, *tags), **kw)


def _mobile(src: Source, suffix: str = "_mobile") -> Source:
    """Same source at the mobile viewport (responsive pages only)."""
    d = asdict(src)
    d.update(name=src.name + suffix, width=P["selftest.screens.mobile_width"],
             height=P["selftest.screens.mobile_height"], tags=tuple(t for t in src.tags if t != "desktop") + ("mobile",))
    d["dismiss"], d["hide"] = tuple(d["dismiss"]), tuple(d["hide"])
    return Source(**d)


def _google(name: str, url: str, tags: Iterable[str], **kw: Any) -> Source:
    kw.setdefault("width", P["selftest.screens.desktop_width"])
    kw.setdefault("height", P["selftest.screens.desktop_height"])
    return Source(name=name, url=url, tags=("google", "product", *tags), **kw)


def default_sources() -> list[Source]:
    """The curated public reference set (desktop 1280x800; a few responsive pages also at 412x915)."""
    m3 = [
        _m3("buttons", "overview", ["buttons", "nav-rail"]),
        _m3("buttons", "specs", ["buttons", "anatomy"]),
        _m3("cards", "specs", ["cards"], scroll_text=("Elevated card", *SPEC_HEADINGS)),
        _m3("chips", "specs", ["chips"]),
        _m3("text-fields", "specs", ["text-fields"]),
        _m3("navigation-bar", "overview", ["navigation-bar"]),
        _m3("navigation-bar", "specs", ["navigation-bar", "anatomy"]),
        _m3("top-app-bar", "overview", ["top-app-bar", "app-bars"]),
        _m3("lists", "specs", ["lists", "anatomy"]),
        _m3("dialogs", "specs", ["dialogs"]),
        _m3("floating-action-button", "specs", ["fab", "anatomy"], name="m3_fab_specs"),
        _m3("switch", "overview", ["switch"]),
        _m3("checkbox", "specs", ["checkbox"], scroll_text=("Measurements", "Tokens & specs")),
        _m3("tabs", "specs", ["tabs"]),
    ]
    m3_mobile = [
        _mobile(_m3("buttons", "overview", ["buttons"], scroll_y=700, name="m3_buttons_overview_types")),
        _mobile(_m3("lists", "overview", ["lists"])),
    ]
    layout = [
        Source("m3_layout_understanding", "https://m3.material.io/foundations/layout/understanding-layout/overview",
               tags=("m3", "docs", "foundations", "layout")),
        Source("m3_layout_window_size_classes",
               "https://m3.material.io/foundations/layout/applying-layout/window-size-classes",
               tags=("m3", "docs", "foundations", "layout")),
    ]
    mw = [
        _mw("button", ["buttons"]),
        _mw("chip", ["chips"]),
        _mw("list", ["lists"], anchor="#usage"),
        _mw("text-field", ["text-fields"]),
        _mw("switch", ["switch"], anchor="#usage"),
        _mw("tabs", ["tabs"]),
        # the only rendered dialog lives in the collapsed <details> interactive demo
        _mw("dialog", ["dialogs"], anchor="#interactive-demo", click="details summary", settle_ms=3000),
        _mw("fab", ["fab"]),
        _mw("icon-button", ["icon-buttons"]),
        _mw("select", ["select", "menus"], anchor="#usage"),
        _mw("slider", ["slider"], anchor="#usage"),
    ]
    mw_mobile = [_mobile(_mw("button", ["buttons"])), _mobile(_mw("text-field", ["text-fields"]))]
    google = [
        _google("gws_gmail", "https://workspace.google.com/products/gmail/", ["gmail", "app-ui"]),
        _google("gws_calendar", "https://workspace.google.com/products/calendar/", ["calendar", "app-ui"]),
        _google("gws_drive", "https://workspace.google.com/products/drive/", ["drive", "app-ui"]),
        _google("gws_keep", "https://workspace.google.com/products/keep/", ["keep", "app-ui"]),
        _google("about_google", "https://about.google/", ["about", "nav"]),
        _google("google_store", "https://store.google.com/", ["store", "nav"]),
        _google("material_blog", "https://material.io/blog", ["blog", "cards"]),
    ]
    google_mobile = [_mobile(google[0]), _mobile(google[4])]
    return [*m3, *m3_mobile, *layout, *mw, *mw_mobile, *google, *google_mobile]


SOURCES: list[Source] = default_sources()


# --------------------------------------------------------------------------- in-page script
def build_script(src: Source) -> str:
    """JS (async IIFE) that performs `src`'s page actions and returns a small evidence dict.

    The returned object has ``title, doc_h, dismissed, hidden, clicked, scrolled, target, target_top,
    images_pending`` (images in view still loading when the capture was taken) and
    is stored verbatim in the manifest entry under ``page`` so a reader can see what happened.
    """
    settle = src.settle_ms if src.settle_ms is not None else P["selftest.screens.settle_ms"]
    return """
(async () => {
  const cfg = %s;
  const all = (root, sel, acc) => {
    for (const e of root.querySelectorAll(sel)) acc.push(e);
    for (const e of root.querySelectorAll('*')) if (e.shadowRoot) all(e.shadowRoot, sel, acc);
    return acc;
  };
  const visible = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const out = {title: document.title, doc_h: document.documentElement.scrollHeight,
               dismissed: 0, hidden: 0, clicked: false, scrolled: null, target_top: null};
  for (const sel of cfg.dismiss) for (const e of all(document, sel, [])) if (visible(e)) { e.click(); out.dismissed++; }
  for (const sel of cfg.hide) for (const e of all(document, sel, [])) if (visible(e)) { e.style.setProperty('display', 'none', 'important'); out.hidden++; }
  if (cfg.click) { const e = document.querySelector(cfg.click); if (e) { e.click(); out.clicked = true; } }
  const findTarget = () => {
    if (cfg.scroll_to) { const e = document.querySelector(cfg.scroll_to); if (e) return e; }
    const heads = Array.from(document.querySelectorAll('h1,h2,h3'));
    for (const t of cfg.scroll_texts) { const e = heads.find(h => h.textContent.trim().startsWith(t)); if (e) return e; }
    return null;
  };
  let target = null;
  if (cfg.scroll_to || cfg.scroll_texts.length) {
    const t0 = performance.now();
    while (!(target = findTarget()) && performance.now() - t0 < cfg.poll_ms) await new Promise(r => setTimeout(r, 100));
  }
  if (target) { target.scrollIntoView({block: 'start', behavior: 'instant'}); out.scrolled = 'target'; }
  else if (cfg.scroll_y != null) {
    window.scrollTo(0, cfg.scroll_y);
    for (const e of document.querySelectorAll('*'))
      if (e.scrollHeight > e.clientHeight + 50 && /(auto|scroll)/.test(getComputedStyle(e).overflowY)) e.scrollTop = cfg.scroll_y;
    out.scrolled = 'offset';
  }
  if (cfg.settle_ms > 0) await new Promise(r => setTimeout(r, cfg.settle_ms));
  const inView = e => { const r = e.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; };
  const pending = () => all(document, 'img', []).filter(i => inView(i) && !(i.complete && i.naturalWidth > 0));
  const t1 = performance.now();
  while (pending().length && performance.now() - t1 < cfg.image_poll_ms) await new Promise(r => setTimeout(r, 100));
  out.images_pending = pending().length;
  if (target) { out.target_top = Math.round(target.getBoundingClientRect().top); out.target = (target.id ? '#' + target.id : target.tagName.toLowerCase() + ':' + target.textContent.trim().slice(0, 40)); }
  return out;
})()
""" % json.dumps({"dismiss": list(src.dismiss), "hide": list(src.hide), "click": src.click,
                   "scroll_to": src.scroll_to, "scroll_texts": list(src.scroll_texts), "scroll_y": src.scroll_y,
                   "settle_ms": int(settle), "poll_ms": int(P["selftest.screens.target_poll_ms"]),
                   "image_poll_ms": int(P["selftest.screens.image_poll_ms"])})


# --------------------------------------------------------------------------- capture
def content_fraction(rgb: np.ndarray, quant: Optional[int] = None) -> float:
    """Fraction of pixels that are not the dominant (quantised) colour; ~0 for a blank page."""
    if rgb.size == 0:
        return 0.0
    q = int(quant if quant is not None else P["selftest.screens.blank_quant"])
    px = (rgb[..., :3].reshape(-1, 3) // q).astype(np.int64)
    keys = px[:, 0] * 1_000_000 + px[:, 1] * 1000 + px[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    return float(1.0 - counts.max() / keys.size)


def capture_source(src: Source, out_dir: str) -> dict[str, Any]:
    """Screenshot one Source into ``out_dir/<name>.png``; returns its manifest entry (never raises)."""
    from dt.render.screenshot import render_url

    entry: dict[str, Any] = {
        "name": src.name, "url": src.url, "file": src.file, "width": src.width, "height": src.height,
        "size": None, "dpr": 1, "tags": list(src.tags), "ok": False, "error": None, "page": None,
        "content_frac": None,
    }
    path = os.path.join(out_dir, src.file)
    wait = src.wait_ms if src.wait_ms is not None else P["selftest.screens.wait_ms"]
    attempts = 1 + int(P["selftest.screens.retries"])
    for attempt in range(attempts):
        try:
            rgb, page = render_url(src.url, src.width, src.height, out_path=None, wait_ms=int(wait),
                                   script=build_script(src))
            break
        except Exception as e:  # network / timeout / navigation errors are data, not crashes
            entry["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]} (attempt {attempt + 1}/{attempts})"
    else:
        return entry
    entry["error"] = None
    entry["page"] = page
    frac = content_fraction(rgb)
    entry["content_frac"] = round(frac, 4)
    if frac < P["selftest.screens.min_content_frac"]:
        entry["error"] = f"blank capture (content fraction {frac:.4f})"
        return entry
    from dt.common.image import save_rgb
    save_rgb(rgb, path)
    entry["size"] = [int(rgb.shape[1]), int(rgb.shape[0])]
    entry["ok"] = True
    return entry


def capture(out_dir: str = DEFAULT_DIR, captured_at: Optional[str] = None,
            sources: Optional[Iterable[Source]] = None, names: Optional[Iterable[str]] = None,
            verbose: bool = True, prune: bool = True) -> dict[str, Any]:
    """Capture every source (or just `names`) into `out_dir` and write ``manifest.json``.

    `captured_at` is an ISO-8601 timestamp supplied by the caller (defaults to now, UTC) so a
    re-run can be labelled by the driver. Returns the manifest dict. When capturing a subset,
    entries for untouched sources are kept from the existing manifest; on a full capture, PNGs in
    `out_dir` that no longer belong to a successful entry are removed (`prune`).
    """
    os.makedirs(out_dir, exist_ok=True)
    srcs = list(sources) if sources is not None else list(SOURCES)
    if names is not None:
        want = set(names)
        unknown = want - {s.name for s in srcs}
        if unknown:
            raise KeyError(f"unknown source names: {sorted(unknown)}")
        srcs = [s for s in srcs if s.name in want]
    previous = {}
    old_path = os.path.join(out_dir, MANIFEST_NAME)
    if names is not None and os.path.exists(old_path):
        previous = {e["name"]: e for e in load_manifest(old_path)["screens"]}
    entries: list[dict[str, Any]] = []
    for s in srcs:
        e = capture_source(s, out_dir)
        if verbose:
            status = "ok  " if e["ok"] else "FAIL"
            print(f"[{status}] {s.name:36s} {s.width}x{s.height}  {e['error'] or ''}", file=sys.stderr, flush=True)
        entries.append(e)
    done = {e["name"] for e in entries}
    entries.extend(v for k, v in previous.items() if k not in done)
    if names is None and prune:
        keep = {e["file"] for e in entries if e["ok"]}
        for fn in os.listdir(out_dir):
            if fn.endswith(".png") and fn not in keep:
                os.remove(os.path.join(out_dir, fn))
    manifest = {
        "version": MANIFEST_VERSION,
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dpr": 1,
        "n_ok": sum(1 for e in entries if e["ok"]),
        "n_failed": sum(1 for e in entries if not e["ok"]),
        "screens": sorted(entries, key=lambda e: e["name"]),
    }
    with open(old_path, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def load_manifest(path: str = os.path.join(DEFAULT_DIR, MANIFEST_NAME)) -> dict[str, Any]:
    """Load a manifest written by `capture` (validates the top-level shape)."""
    with open(path) as f:
        m = json.load(f)
    if not isinstance(m.get("screens"), list) or "captured_at" not in m:
        raise ValueError(f"not a reference-screens manifest: {path}")
    return m


def ok_screens(manifest: dict[str, Any], out_dir: str = DEFAULT_DIR) -> list[tuple[str, dict[str, Any]]]:
    """[(absolute png path, entry)] for successful captures whose file exists."""
    res = []
    for e in manifest["screens"]:
        p = os.path.join(out_dir, e["file"])
        if e.get("ok") and os.path.exists(p):
            res.append((p, e))
    return res


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Capture public reference screenshots for self-tests.")
    ap.add_argument("--out", default=DEFAULT_DIR)
    ap.add_argument("--only", nargs="*", default=None, help="source names to (re)capture")
    ap.add_argument("--captured-at", default=None)
    ap.add_argument("--list", action="store_true", help="print the source list and exit")
    a = ap.parse_args(argv)
    if a.list:
        for s in SOURCES:
            print(f"{s.name:36s} {s.width}x{s.height}  {s.url}")
        return 0
    try:
        m = capture(a.out, captured_at=a.captured_at, names=a.only)
    except Exception:
        traceback.print_exc()
        return 2
    print(f"{m['n_ok']} ok, {m['n_failed']} failed -> {os.path.join(a.out, MANIFEST_NAME)}")
    return 0 if m["n_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
