"""Portability: a fresh clone must install and run on macOS and Linux without hand-holding.

Covers the packaging metadata (macOS-only deps behind markers, the non-Vision OCR extra), the
Material Web bundle build, the browser launch fallback, the RapidOCR adapter's space restoration
(on text rendered by the real renderer, with per-character boxes measured in the browser), the
reference-page generator and `dt doctor`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from dt.ir import Box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- packaging
def _pyproject() -> dict:
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_macos_only_dependencies_have_platform_markers():
    """pyobjc has no Linux/Windows wheels and its sdist refuses to build there: unmarked, `pip install -e .` fails."""
    deps = _pyproject()["project"]["dependencies"]
    for d in deps:
        if d.lower().startswith("pyobjc"):
            assert "sys_platform == 'darwin'" in d.replace('"', "'"), d


def test_non_vision_ocr_extra_is_declared():
    extras = _pyproject()["project"].get("optional-dependencies", {})
    for name in ("linux", "ocr"):
        assert any(r.startswith("rapidocr-onnxruntime") for r in extras.get(name, [])), name


def test_every_third_party_import_is_declared():
    """Top-level imports of dt/ that are not stdlib must be declared (or be an optional OCR backend)."""
    import ast

    declared = {"numpy", "cv2", "PIL", "skimage", "scipy", "playwright", "requests", "pytest"}
    optional = {"Vision", "Quartz", "Foundation", "rapidocr_onnxruntime", "pytesseract",
                "mcp", "mlx"}  # mcp: the [mcp] extra; mlx: Apple-Silicon-only dependency (dt.accel falls back to numpy) (dt/mcp_server.py only); pyproject declares it under optional-dependencies
    found: set[str] = set()
    for dp, _, fs in os.walk(os.path.join(ROOT, "dt")):
        for fn in fs:
            if fn.endswith(".py"):
                tree = ast.parse(open(os.path.join(dp, fn)).read())
                for n in ast.walk(tree):
                    if isinstance(n, ast.Import):
                        found |= {a.name.split(".")[0] for a in n.names}
                    elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                        found.add(n.module.split(".")[0])
    third = {m for m in found if m not in sys.stdlib_module_names and m != "dt"}
    assert third <= declared | optional, sorted(third - declared - optional)


def test_mwc_bundle_build_script_reproduces_the_tracked_bundle(tmp_path):
    """fixtures/mwc/mwc.bundle.js is generated: `npm run build:mwc` must exist and rebuild it byte-for-byte."""
    pkg = json.load(open(os.path.join(ROOT, "package.json")))
    cmd = pkg.get("scripts", {}).get("build:mwc")
    assert cmd and "fixtures/mwc/entry.js" in cmd and "fixtures/mwc/mwc.bundle.js" in cmd
    esbuild = os.path.join(ROOT, "node_modules", ".bin", "esbuild")
    if not os.path.exists(esbuild):
        pytest.skip("node_modules missing (npm ci)")
    out = tmp_path / "mwc.bundle.js"
    args = cmd.split()[1:]
    args = [f"--outfile={out}" if a.startswith("--outfile=") else a for a in args]
    subprocess.run([esbuild, *args], cwd=ROOT, check=True, capture_output=True)
    assert out.read_bytes() == open(os.path.join(ROOT, "fixtures", "mwc", "mwc.bundle.js"), "rb").read()


def test_setup_script_is_valid_bash():
    sh = os.path.join(ROOT, "scripts", "setup.sh")
    assert os.access(sh, os.X_OK)
    subprocess.run(["bash", "-n", sh], check=True)


# --------------------------------------------------------------------------- browser launch
def test_launch_falls_back_past_a_missing_browser(monkeypatch):
    from dt.render import screenshot as s

    monkeypatch.delenv("DT_EVAL_SERVICE", raising=False)  # this test inspects the in-process driver
    s.html_to_png("<b>ok</b>", 20, 10)  # make sure the shared driver is up
    b, used = s.launch_browser(s._st()["pw"], ["/nonexistent/chrome-for-dt-test"] + s.browser_candidates("auto"))
    try:
        assert used in ("chrome", "chromium")
        assert b.is_connected()
    finally:
        b.close()


def test_no_browser_error_is_actionable_and_recoverable(monkeypatch):
    """When no browser starts the error says how to fix it, and the driver is torn down so the next
    call (after the fix) works in the same process instead of failing on a half-started driver."""
    from dt.render import screenshot as s

    s.shutdown()
    monkeypatch.setenv("DT_BROWSER", "/nonexistent/chrome-for-dt-test")
    with pytest.raises(RuntimeError, match="playwright install chromium"):
        s.html_to_png("<b>x</b>", 20, 10)
    assert s._st()["pw"] is None and s._st()["browser"] is None
    monkeypatch.delenv("DT_BROWSER")
    img = s.html_to_png("<style>html,body{margin:0;background:#000}</style>", 20, 10)
    assert img.shape == (10, 20, 3) and int(img.max()) < 10


# --------------------------------------------------------------------------- RapidOCR adapter: word spaces
def _render_lines(items, family: str, size: int, dark: bool):
    """Render one line per item with the real renderer; return (rgb, [(line box, [per-char (x0,x1)])])."""
    from dt.render.html import _font_face_css
    from dt.render.screenshot import _tmp_html, render_url

    bg, fg = ("#1f1f1f", "#e3e3e3") if dark else ("#ffffff", "#1f1f1f")
    divs, y = [], 6
    for i, (t, ls) in enumerate(items):
        lh = int(size * 1.6)
        divs.append(f"<div id=l{i} style=\"position:absolute;left:12px;top:{y}px;font:{size}px '{family}';"
                    f"letter-spacing:{ls}px;color:{fg};white-space:nowrap;line-height:{lh}px\">{t}</div>")
        y += lh + 10
    html = f"<!doctype html><meta charset=utf-8><style>{_font_face_css()} body{{margin:0;background:{bg}}}</style>" + "".join(divs)
    script = """(() => Array.from(document.querySelectorAll('div[id^=l]')).map(d => {
      const t = d.firstChild, r = document.createRange(), cs = [], b = d.getBoundingClientRect();
      for (let k = 0; k < t.length; k++) { r.setStart(t, k); r.setEnd(t, k + 1); const q = r.getBoundingClientRect(); cs.push([q.left, q.right]); }
      return {box: [b.left - 4, b.top, b.width + 8, b.height], cs}; }))()"""
    rgb, info = render_url("file://" + _tmp_html(html), 900, y + 6, wait_ms=0, wait_until="load", script=script)
    return rgb, [(Box(*d["box"]), [tuple(c) for c in d["cs"]]) for d in info]


SPACED = ["Google Workspace", "Try Calendar for work", "AI-powered, secure, and easy to use",
          "billions of people and businesses.", "Join with Google Meet", "Out of office"]


def _gst_installed() -> bool:
    from dt.perceive.fonts import _family_installed
    return _family_installed("Google Sans Text")


# Google Sans Text is local-only (license unconfirmed, not distributed): skip its cases on clones without it
_needs_gst = pytest.mark.skipif(not _gst_installed(), reason="Google Sans Text not installed (local-only font)")


@pytest.mark.parametrize("family,size,dark", [("Roboto", 16, False), ("Roboto", 22, True), ("Google Sans", 32, False), pytest.param("Google Sans Text", 14, True, marks=_needs_gst)])
def test_restore_spaces_recovers_dropped_spaces(family, size, dark):
    """The PP-OCR recogniser returns 'GoogleWorkspace'; with its per-character boxes the adapter must put the
    spaces back from the ink gaps of the real render. Contract: never a space where the text has none
    (precision first: the recogniser is usually right), and most dropped spaces recovered. Gaps narrowed
    by overhanging glyphs ('of people', 'for work') can stay joined: measured over 30 font/size/theme
    combinations recall is 0.82 with 0 false spaces (weakest: 12px light-on-dark)."""
    from dt.perceive.ocr import restore_spaces

    def space_slots(s: str) -> set[int]:
        out, n = set(), 0
        for ch in s:
            if ch == " ":
                out.add(n)
            else:
                n += 1
        return out

    rgb, lines = _render_lines([(t, 0) for t in SPACED], family, size, dark)
    want = hit = 0
    for t, (box, cs) in zip(SPACED, lines):
        keep = [k for k, ch in enumerate(t) if ch != " "]
        got, words = restore_spaces(rgb, t.replace(" ", ""), box, [cs[k] for k in keep])
        assert got.replace(" ", "") == t.replace(" ", "")
        assert space_slots(got) <= space_slots(t), (family, size, dark, got)  # no false spaces
        want += len(space_slots(t))
        hit += len(space_slots(got))
        assert [w for w, _ in words] == got.split(" ")
        for (w, wb), nxt in zip(words, words[1:]):
            assert wb.x2 <= nxt[1].x + 0.5  # word boxes in reading order, not overlapping
    assert hit / want >= 0.8, (family, size, dark, hit, want)


@pytest.mark.parametrize("family,size,dark", [("Roboto", 11, False), ("Roboto", 14, True), ("Google Sans", 12, True), pytest.param("Google Sans Text", 22, False, marks=_needs_gst)])
def test_restore_spaces_never_corrupts_correct_text(family, size, dark):
    """Text that already has its spaces, and letter-spaced caps, must come back unchanged (an earlier
    version re-assigned the existing word gaps to letter boundaries: 'Googl e Workspace', 'FRI DAY')."""
    from dt.perceive.ocr import restore_spaces

    items = [(t, 0) for t in SPACED + ["Appointment booking", "illuminating", "will fill it"]] + \
            [(t, 0.1 * size) for t in ("FRIDAY", "SETTINGS", "LABEL TEXT", "OVERLINE")]
    rgb, lines = _render_lines(items, family, size, dark)
    for (t, _), (box, cs) in zip(items, lines):
        got, _ = restore_spaces(rgb, t, box, cs)
        assert got == t, (family, size, dark, got)


def test_restore_spaces_degenerate_inputs():
    from dt.perceive.ocr import restore_spaces

    rgb = np.full((20, 40, 3), 255, np.uint8)
    assert restore_spaces(rgb, "  ", Box(0, 0, 40, 20)) == ("", [])
    assert restore_spaces(rgb, "ab", Box(0, 0, 40, 20))[0] == "ab"  # blank crop: nothing to measure
    assert restore_spaces(rgb, "ab", Box(50, 50, 10, 10))[0] == "ab"  # box outside the image


def test_rapidocr_backend_reads_rendered_text():
    """End to end through RapidOCR when it is installed (the `ocr`/`linux` extra)."""
    pytest.importorskip("rapidocr_onnxruntime")
    from dt.perceive import ocr
    from dt.compare import text_cer

    be = ocr.get_backend("rapidocr")
    rgb, lines = _render_lines([(t, 0) for t in SPACED], "Roboto", 22, True)
    engine_raw, _ = be._engine(np.ascontiguousarray(rgb[..., ::-1]))  # what PP-OCR itself returns
    raws = be.recognize(rgb)
    c_raw, c_ad = [], []

    def near(cy, items):
        return " ".join(t for y, t in items if abs(y - cy) < 10)
    eng = [((min(p[1] for p in q) + max(p[1] for p in q)) / 2, str(t).strip()) for q, t, *_ in engine_raw or []]
    ada = [(r.box.y + r.box.h / 2, r.text) for r in raws]
    for t, (box, _) in zip(SPACED, lines):
        cy = box.y + box.h / 2
        c_raw.append(text_cer(near(cy, eng).replace("Al-", "AI-"), t))  # I/l homoglyph is the recogniser's
        c_ad.append(text_cer(near(cy, ada).replace("Al-", "AI-"), t))
        assert c_ad[-1] <= c_raw[-1] + 1e-9, (t, near(cy, eng), near(cy, ada))
    for r in raws:
        assert r.text == r.text.strip() and "  " not in r.text
        assert all(w and " " not in w for w, _ in r.words)
    # measured: engine 0.1075 ('GoogleWorkspace', 'JoinwithGoogleMeet'), adapter 0.0048
    assert float(np.mean(c_ad)) <= 0.02 and float(np.mean(c_ad)) < float(np.mean(c_raw)), (c_raw, c_ad)


def test_vision_backend_is_never_selected_off_macos(monkeypatch):
    from dt.perceive import ocr

    monkeypatch.setattr(ocr.sys, "platform", "linux")
    monkeypatch.setattr(ocr, "_backend_cache", {})
    assert "vision" not in ocr.available_backends()
    with pytest.raises((RuntimeError, ValueError)) as ei:
        ocr.get_backend("vision")
    assert "macOS" in str(ei.value) or "no OCR backend" in str(ei.value)


# --------------------------------------------------------------------------- reference page + doctor
def test_reference_page_points_at_this_checkout():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import make_reference_page as m
    finally:
        sys.path.pop(0)
    html = m.page_html(ROOT)
    for rel in ("fixtures/fonts/roboto.local.css", "fixtures/fonts/material-symbols.local.css", "fixtures/mwc/mwc.bundle.js"):
        p = os.path.join(ROOT, rel)
        assert f"file://{p}" in html and os.path.exists(p)


def test_doctor_reports_and_fixes(monkeypatch, tmp_path):
    from dt import doctor

    res = doctor.run()
    names = [r.name for r in res]
    for want in ("browser", "font render", "ocr", "icon atlas", "mwc bundle"):
        assert want in names
    assert all(r.status in ("ok", "warn", "fail") for r in res)
    assert all(r.fix for r in res if r.status == "fail")
    text = doctor.format_human(res)
    assert ("READY" in text) and all(r.name in text for r in res)
    # an empty checkout: missing bundle fails with the npm fix, missing reference page warns with the script
    monkeypatch.setattr(doctor, "ROOT", str(tmp_path))
    b = doctor.check_bundle()
    assert b.status == "fail" and "build:mwc" in b.fix
    rp = doctor.check_reference_page()
    assert rp.status == "warn" and "make_reference_page.py" in rp.fix


def test_doctor_cli_json():
    r = subprocess.run([sys.executable, "-m", "dt.cli", "doctor", "--json"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    d = json.loads(r.stdout)
    assert r.returncode == (0 if d["ok"] else 1)
    assert {c["name"] for c in d["checks"]} >= {"python", "browser", "ocr"}
