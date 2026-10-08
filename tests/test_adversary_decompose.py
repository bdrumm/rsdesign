"""Diff-decomposition adversary: noise silence, the layer each cause lands in, IR typing, params, benchmark smoke."""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np
import pytest

from dt.adversary import decompose as DC
from dt.adversary.taxonomy import Finding, is_known, is_noise
from dt.ir import Box, Color, Document, Fill, Node
from dt.params import P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(ROOT, "out", "adversary", "bench")


# --------------------------------------------------------------------------- synthetic pages
def _page(shapes, w=320, h=240, bg=(250, 250, 252)) -> np.ndarray:
    """Flat UI-like page: filled rects ``(x, y, w, h, rgb)``; a 1 px dark outline for ``rgb=None``."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bg
    for x, y, ww, hh, rgb in shapes:
        if rgb is None:
            cv2.rectangle(img, (x, y), (x + ww - 1, y + hh - 1), (120, 116, 126), 1)
        else:
            img[y:y + hh, x:x + ww] = rgb
    return img


BASE = [(20, 20, 120, 40, (103, 80, 164)), (20, 80, 280, 1, (200, 196, 208)), (40, 110, 60, 60, (230, 222, 248)),
        (160, 110, 120, 60, None), (180, 190, 24, 24, (33, 31, 38))]


def _errors(fs):
    return [f for f in fs if not is_noise(f.type)]


def _node(i, box, rgb):
    return Node(id=f"n{i}", type="rect", box=Box(*box), fills=[Fill.solid(Color(*rgb))])


def _doc(shapes, w=320, h=240):
    kids = [_node(i, s[:4], s[4]) for i, s in enumerate(shapes) if s[4] is not None]
    return Document(width=w, height=h, root=Node(id="root", type="frame", box=Box(0, 0, w, h), children=kids))


# --------------------------------------------------------------------------- silence on renderer noise
def test_identical_images_draw_no_finding():
    t = _page(BASE)
    assert DC.find(t, t.copy()) == []


@pytest.mark.parametrize("kind", ["subpixel", "blur", "jpeg"])
def test_renderer_noise_is_not_an_error(kind):
    c = _page(BASE)
    if kind == "subpixel":
        M = np.float32([[1, 0, 0.4], [0, 1, -0.35]])
        t = cv2.warpAffine(c, M, (c.shape[1], c.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    elif kind == "blur":
        t = cv2.GaussianBlur(c, (0, 0), 0.5)
    else:
        from dt.adversary.perturb import jpeg
        t = jpeg(c, 80)
    assert _errors(DC.find(t, c)) == []


# --------------------------------------------------------------------------- one cause per layer
def test_global_tint_is_one_finding_with_its_magnitude():
    c = _page(BASE)
    v = np.array([-3.0, 4.0, 3.0])
    t = DC.apply_tint(c, -v)  # target = candidate shifted the other way
    fs = _errors(DC.find(t, c))
    assert [f.type for f in fs] == ["color.tint_global"]
    assert 2.0 <= fs[0].magnitude <= 8.0


def test_moved_rectangle_is_a_shift_covering_both_footprints():
    t = _page(BASE)
    moved = list(BASE)
    moved[2] = (52, 116, 60, 60, (230, 222, 248))
    moved[0] = (20, 20, 120, 40, (103, 80, 164))
    c = _page(moved)
    ir = _doc(moved)
    fs = _errors(DC.find(t, c, ir))
    shifts = [f for f in fs if f.type == "geometry.shift"]
    assert shifts, [f.type for f in fs]
    f = shifts[0]
    assert abs(f.magnitude - math.hypot(12, 6)) <= 1.5
    assert f.box.contains(Box(40, 110, 60, 60), tol=3) and f.box.contains(Box(52, 116, 60, 60), tol=3)


def test_missing_and_extra_are_attributed_by_ink_ownership():
    t = _page(BASE + [(200, 30, 80, 24, (60, 60, 70))])
    c = _page(BASE + [(60, 200, 50, 20, (60, 60, 70))])
    fs = _errors(DC.find(t, c))
    types = {f.type for f in fs}
    assert "structure.missing" in types and "structure.extra" in types
    miss = next(f for f in fs if f.type == "structure.missing")
    extra = next(f for f in fs if f.type == "structure.extra")
    assert miss.box.iou(Box(200, 30, 80, 24)) > 0.5 and extra.box.iou(Box(60, 200, 50, 20)) > 0.5


def test_recoloured_fill_is_a_colour_finding_with_its_delta_e():
    t = _page(BASE)
    recol = list(BASE)
    recol[0] = (20, 20, 120, 40, (123, 96, 150))
    c = _page(recol)
    fs = _errors(DC.find(t, c, _doc(recol)))
    assert [f.type for f in fs] == ["color.fill"]
    want = Color(103, 80, 164).delta_e(Color(123, 96, 150))
    assert fs[0].magnitude == pytest.approx(want, rel=0.35)
    assert fs[0].box.iou(Box(20, 20, 120, 40)) > 0.6


def test_findings_are_typed_json_and_deterministic():
    t = _page(BASE)
    c = _page([(26, 20, 120, 40, (103, 80, 164))] + BASE[1:])
    a = DC.find(t, c, _doc(BASE))
    b = DC.find(t, c, _doc(BASE))
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    for f in a:
        assert is_known(f.type) and 0.0 < f.confidence <= 1.0
        g = Finding.from_dict(json.loads(json.dumps(f.to_dict())))
        assert g.type == f.type and g.box == f.box


def test_decomposition_reports_its_layers_and_noise_model():
    t = _page(BASE)
    c = _page([(26, 20, 120, 40, (103, 80, 164))] + BASE[1:])
    dec = DC.decompose(t, c)
    s = dec.summary()
    assert s["regions"] >= 1 and s["noise"]["tau_flat"] > 0 and not s["noisy"]
    assert set(s["layers"]) <= {"geometry", "colour", "effect", "structure", "noise"}


def test_size_mismatch_is_tolerated():
    t = _page(BASE)
    c = _page(BASE, w=300, h=220)
    fs = DC.find(t, c)
    assert all(f.box.x2 <= t.shape[1] + 1 and f.box.y2 <= t.shape[0] + 1 for f in fs)


# --------------------------------------------------------------------------- params
def test_every_threshold_is_a_registered_param_with_doc_and_range():
    keys = [k for k in P.defaults() if k.startswith("adversary.decompose.")]
    assert len(keys) >= 30
    docs, ranges = P.docs(), P.ranges()
    for k in keys:
        assert docs.get(k), k
        assert k in ranges, k


# --------------------------------------------------------------------------- benchmark smoke (uses the cached benchmark when present)
def _cached_bench():
    from dt.adversary import benchmark as BM
    if not os.path.isdir(BENCH_DIR):
        return None
    for key in sorted(os.listdir(BENCH_DIR)):
        if os.path.exists(os.path.join(BENCH_DIR, key, "manifest.json")):
            b = BM.Benchmark.load(os.path.join(BENCH_DIR, key))
            if len(b.split("calibration")) >= 24:
                return b
    return None


def test_beats_the_floor_on_calibration_cases():
    from dt.adversary import benchmark as BM
    bench = _cached_bench()
    if bench is None:
        pytest.skip("no cached adversary benchmark under out/adversary/bench")
    cases = [c for c in bench.split("calibration")][:16]
    ours = BM.score(DC.find, cases, name="decompose")
    floor = BM.score(BM.baseline_residual, cases, name="baseline_residual")
    assert ours["headline"] > floor["headline"] + 0.2
    assert ours["noise"]["false_case_rate"] <= floor["noise"]["false_case_rate"]
    assert ours["type_accuracy"]["parent"] > 0.5
