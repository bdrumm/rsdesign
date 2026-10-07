"""DPR detection: true 1x/2x/3x renders of the same Material Web page are detected correctly, and
translate(dpr='auto') downscales a 2x capture to CSS pixels."""
import os

import pytest

from dt.common.image import load_rgb
from dt.perceive.dpr import detect_dpr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAGE = os.path.join(ROOT, "fixtures", "corpus", "mwc", "mwc_1_001.html")


@pytest.fixture(scope="module")
def captures(tmp_path_factory):
    from dt.render.screenshot import capture_url
    d = tmp_path_factory.mktemp("dpr")
    out = {}
    for dpr in (1, 2, 3):
        p = str(d / f"cap{dpr}.png")
        capture_url("file://" + PAGE, 412, 730, device_scale_factor=dpr, out_path=p)
        out[dpr] = p
    return out


@pytest.mark.parametrize("dpr", [1, 2, 3])
def test_detects_capture_dpr(captures, dpr):
    got, ev = detect_dpr(load_rgb(captures[dpr]))
    assert got == float(dpr), ev


def test_translate_auto_dpr(captures, tmp_path):
    from dt.pipeline import translate
    m = translate(captures[2], ds="material3", out_dir=str(tmp_path / "run"), refine_iters=0, dpr="auto")
    assert m["dpr"] == 2.0 and m["size"] == {"w": 412, "h": 730}
