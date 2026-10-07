"""Tests for dt.selftest.reference_screens.

Two layers:
  * corpus checks on the committed fixtures/screens/ (manifest loads, >=15 PNGs open, sizes match);
  * behavioural checks of the capture machinery against the REAL renderer on local file:// pages
    (no network): in-page actions (scroll-to-heading, hide overlay), blank detection, failure
    recording and manifest merging.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from dt.params import P
from dt.selftest import reference_screens as rs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENS = os.path.join(ROOT, "fixtures", "screens")
REQUIRED_KEYS = {"name", "url", "file", "width", "height", "size", "dpr", "tags", "ok", "error", "page"}


# --------------------------------------------------------------------------- committed corpus
def test_manifest_loads_and_is_well_formed():
    m = rs.load_manifest(os.path.join(SCREENS, rs.MANIFEST_NAME))
    assert m["version"] == rs.MANIFEST_VERSION
    assert m["dpr"] == 1
    datetime.fromisoformat(m["captured_at"].replace("Z", "+00:00"))
    names = [e["name"] for e in m["screens"]]
    assert len(names) == len(set(names)), "duplicate names in manifest"
    for e in m["screens"]:
        assert REQUIRED_KEYS <= set(e), e["name"]
        assert e["url"].startswith("https://"), e["name"]
        assert e["dpr"] == 1
        if e["ok"]:
            assert e["error"] is None and e["size"] is not None
        else:
            assert isinstance(e["error"], str) and e["error"]
    assert m["n_ok"] == sum(1 for e in m["screens"] if e["ok"])
    assert m["n_failed"] == sum(1 for e in m["screens"] if not e["ok"])


def test_corpus_pngs_exist_open_and_match_manifest():
    m = rs.load_manifest(os.path.join(SCREENS, rs.MANIFEST_NAME))
    shots = rs.ok_screens(m, SCREENS)
    if not shots and not any(f.endswith(".png") for f in os.listdir(SCREENS)):
        # the PNGs are git-ignored (.gitignore: fixtures/screens/*.png): a fresh clone has none at all.
        # A partial set still fails below; this only skips the "never captured here" state.
        pytest.skip("fixtures/screens/*.png not captured in this checkout: dt corpus build --kind screens (network)")
    assert len(shots) >= 15, f"only {len(shots)} successful reference screenshots"
    widths = set()
    for path, e in shots:
        with Image.open(path) as im:
            im.load()
            assert list(im.size) == e["size"], e["name"]
            assert im.size == (e["width"], e["height"]), e["name"]
        widths.add(e["width"])
        assert e["content_frac"] >= P["selftest.screens.min_content_frac"], e["name"]
    assert {P["selftest.screens.mobile_width"], P["selftest.screens.desktop_width"]} <= widths


def test_corpus_covers_required_sites():
    m = rs.load_manifest(os.path.join(SCREENS, rs.MANIFEST_NAME))
    ok_urls = [e["url"] for e in m["screens"] if e["ok"]]
    for host in ("m3.material.io", "material-web.dev", "workspace.google.com"):
        assert any(host in u for u in ok_urls), host


def test_source_list_is_consistent():
    names = [s.name for s in rs.SOURCES]
    assert len(names) == len(set(names))
    assert 25 <= len(names) <= 40
    for s in rs.SOURCES:
        assert s.url.startswith("https://")
        assert s.width in (P["selftest.screens.mobile_width"], P["selftest.screens.desktop_width"])
        assert s.file == f"{s.name}.png"
        assert s.name.replace("_", "").isalnum()


# --------------------------------------------------------------------------- behaviour on local pages
_PAGE = """<!doctype html><html><head><meta charset=utf-8><style>
 body{margin:0;font-family:sans-serif;background:#fff}
 .filler{height:%(gap)dpx;background:#fff}
 #anatomy{margin:0;height:60px;background:#6750a4;color:#fff;font-size:32px;line-height:60px;padding-left:16px}
 .chat-widget{position:fixed;right:16px;bottom:16px;width:200px;height:120px;background:#ff0000}
 .consent{position:fixed;left:0;bottom:0;width:100%%;height:80px;background:#00ff00}
</style></head><body>
<h1>Top heading</h1>
<div class="filler"></div>
<h2 id="anatomy">Anatomy</h2>
<div class="filler"></div>
<div class="chat-widget"></div>
<div class="consent" id="cookie-bar"><button aria-label="Close" onclick="this.parentNode.remove()">x</button></div>
</body></html>"""


@pytest.fixture(scope="module")
def local_page(tmp_path_factory):
    p = tmp_path_factory.mktemp("page") / "page.html"
    p.write_text(_PAGE % {"gap": 1500})
    return "file://" + str(p)


def _local_source(url: str, **kw) -> rs.Source:
    kw.setdefault("name", "local_page")
    kw.setdefault("width", 600)
    kw.setdefault("height", 400)
    kw.setdefault("settle_ms", 50)
    kw.setdefault("wait_ms", 50)
    return rs.Source(url=url, **kw)


def test_actions_scroll_hide_and_dismiss_with_real_renderer(local_page, tmp_path):
    src = _local_source(local_page, scroll_text="Anatomy", hide=(".chat-widget",),
                        dismiss=('[id*="cookie" i] button[aria-label*="close" i]',), tags=("local",))
    e = rs.capture_source(src, str(tmp_path))
    assert e["ok"], e["error"]
    assert e["page"]["scrolled"] == "target" and abs(e["page"]["target_top"]) <= 1
    assert e["page"]["hidden"] == 1 and e["page"]["dismissed"] == 1
    rgb = np.asarray(Image.open(tmp_path / "local_page.png").convert("RGB"))
    assert rgb.shape == (400, 600, 3) and e["size"] == [600, 400]
    purple = np.array([0x67, 0x50, 0xA4])
    assert np.abs(rgb[30, 300].astype(int) - purple).max() <= 2, "heading not scrolled to top"
    assert not (rgb[..., 0] > 200).__and__(rgb[..., 1] < 60).any(), "chat overlay still visible"
    assert not (rgb[..., 1] > 200).__and__(rgb[..., 0] < 60).any(), "consent bar still visible"


def test_scroll_by_offset_and_selector(local_page, tmp_path):
    e = rs.capture_source(_local_source(local_page, name="sel", scroll_to="#anatomy"), str(tmp_path))
    assert e["ok"] and e["page"]["scrolled"] == "target"
    e = rs.capture_source(_local_source(local_page, name="off", scroll_y=1300), str(tmp_path))
    assert e["ok"], e["error"]
    assert e["page"]["scrolled"] == "offset"
    rgb = np.asarray(Image.open(tmp_path / "off.png").convert("RGB"))
    # heading sits at y = (h1 block + 1500px filler) - 1300 -> inside the viewport, below the top
    rows = np.where(np.abs(rgb[:, 300].astype(int) - [0x67, 0x50, 0xA4]).max(axis=1) <= 2)[0]
    assert len(rows) >= 50 and rows.min() > 150


def test_content_fraction_and_blank_rejection(tmp_path):
    assert rs.content_fraction(np.full((50, 50, 3), 255, np.uint8)) == 0.0
    ref = np.asarray(Image.open(os.path.join(ROOT, "out", "mwc_test.png")).convert("RGB"))
    assert rs.content_fraction(ref) > P["selftest.screens.min_content_frac"]
    blank = tmp_path / "blank.html"
    blank.write_text("<!doctype html><html><body style='background:#fff'></body></html>")
    e = rs.capture_source(_local_source("file://" + str(blank), name="blank"), str(tmp_path))
    assert not e["ok"] and "blank" in e["error"]
    assert not (tmp_path / "blank.png").exists()


def test_failure_is_recorded_not_raised(tmp_path):
    e = rs.capture_source(_local_source("file:///definitely/not/here.html", name="missing"), str(tmp_path))
    assert e["ok"] is False and e["error"] and e["size"] is None
    assert not (tmp_path / "missing.png").exists()


def test_capture_writes_manifest_and_merges_subsets(local_page, tmp_path):
    good = _local_source(local_page, name="good", tags=("a",))
    bad = _local_source("file:///nope.html", name="bad")
    m = rs.capture(str(tmp_path), captured_at="2026-01-01T00:00:00+00:00", sources=[good, bad], verbose=False)
    assert m["captured_at"] == "2026-01-01T00:00:00+00:00"
    assert m["n_ok"] == 1 and m["n_failed"] == 1
    on_disk = rs.load_manifest(str(tmp_path / rs.MANIFEST_NAME))
    assert on_disk == json.loads(json.dumps(m))
    # re-capturing a subset keeps the other entries
    m2 = rs.capture(str(tmp_path), sources=[good, bad], names=["good"], verbose=False)
    assert {e["name"] for e in m2["screens"]} == {"good", "bad"}
    with pytest.raises(KeyError):
        rs.capture(str(tmp_path), sources=[good], names=["unknown"], verbose=False)


def test_params_registered_with_docs():
    docs = P.docs()
    for k in ("selftest.screens.wait_ms", "selftest.screens.settle_ms", "selftest.screens.min_content_frac",
              "selftest.screens.retries", "selftest.screens.target_poll_ms"):
        assert k in docs and docs[k]
        P[k]
