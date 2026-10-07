#!/usr/bin/env python
"""(Re)generate the Material Web reference page used by the tests: out/mwc_test.html + out/mwc_test.png.

`out/` is git-ignored, so a fresh clone has neither file and every test that uses them is skipped.
This writes the page with file:// URLs into *this* checkout (fonts + fixtures/mwc/mwc.bundle.js) and
screenshots it with the project renderer at 600x180 (the size the tests were written against).

    python scripts/make_reference_page.py [--force]
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "out")
HTML = os.path.join(OUT, "mwc_test.html")
PNG = os.path.join(OUT, "mwc_test.png")
W, H = 600, 180

BODY = """<md-filled-button>Filled</md-filled-button>
<md-outlined-button>Outlined</md-outlined-button>
<md-text-button>Text</md-text-button>
<md-filled-tonal-button><md-icon slot="icon">add</md-icon>Tonal</md-filled-tonal-button>
<md-fab><md-icon slot="icon">edit</md-icon></md-fab>
<md-switch selected></md-switch>
<md-checkbox checked></md-checkbox>
<md-outlined-text-field label="Email" value="a@b.com"></md-outlined-text-field>
<md-chip-set><md-assist-chip label="Assist"></md-assist-chip><md-filter-chip label="Filter" selected></md-filter-chip></md-chip-set>
"""


def page_html(root: str = ROOT) -> str:
    fonts = os.path.join(root, "fixtures", "fonts")
    bundle = os.path.join(root, "fixtures", "mwc", "mwc.bundle.js")
    return (
        '<!doctype html><html><head><meta charset="utf-8">\n'
        f'<link rel="stylesheet" href="file://{fonts}/roboto.local.css">\n'
        f'<link rel="stylesheet" href="file://{fonts}/material-symbols.local.css">\n'
        f'<script src="file://{bundle}"></script>\n'
        "<style>body{margin:0;padding:16px;font-family:Roboto;background:#fef7ff;display:flex;gap:12px;"
        "align-items:center;flex-wrap:wrap;width:600px}</style>\n"
        "</head><body>\n" + BODY + "</body></html>\n"
    )


def make(force: bool = False) -> tuple[str, str]:
    """Write the page and its screenshot unless both exist and point at this checkout."""
    os.makedirs(OUT, exist_ok=True)
    html = page_html()
    fresh = os.path.exists(HTML) and open(HTML).read() == html and os.path.exists(PNG)
    if fresh and not force:
        return HTML, PNG
    bundle = os.path.join(ROOT, "fixtures", "mwc", "mwc.bundle.js")
    if not os.path.exists(bundle):
        raise SystemExit(f"missing {bundle}: run `npm install && npm run build:mwc` first")
    with open(HTML, "w") as f:
        f.write(html)
    from dt.render.screenshot import render_url

    script = ("new Promise(r => { const t0 = Date.now(); (function w() { if (window.__mwcReady || Date.now() - t0 > 5000) "
              "r(!!window.__mwcReady); else setTimeout(w, 20); })(); })")
    _, ok = render_url("file://" + HTML, W, H, out_path=None, wait_ms=0, wait_until="load", script=script)
    if not ok:
        raise SystemExit("Material Web bundle did not load (window.__mwcReady never set)")
    render_url("file://" + HTML, W, H, out_path=PNG, wait_ms=500, wait_until="load",
               script="document.fonts.ready.then(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))))")
    return HTML, PNG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="regenerate even when up to date")
    a = ap.parse_args()
    h, p = make(a.force)
    print(f"reference page: {h}\nscreenshot:     {p}")
