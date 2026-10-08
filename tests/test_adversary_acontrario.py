"""A-contrario adversary: nuisance registration, null model / NFA control, typing tree, end to end."""
from __future__ import annotations

import copy
import math
import os
import random

import cv2
import numpy as np
import pytest

from dt.adversary import acontrario as AC
from dt.adversary import perturb as PT
from dt.adversary.taxonomy import TYPES, Finding, is_known
from dt.ir import Box, Document

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTH = os.path.join(ROOT, "fixtures", "corpus", "synth", "synth_1_002.gt.json")


def _page(seed: int = 0, h: int = 160, w: int = 240) -> np.ndarray:
    """A small synthetic UI-like page: flat background, cards, bars and dark 'text' strokes."""
    rng = np.random.RandomState(seed)
    im = np.full((h, w, 3), 250, np.uint8)
    for _ in range(6):
        x, y = rng.randint(0, w - 60), rng.randint(0, h - 30)
        c = tuple(int(v) for v in rng.randint(60, 230, 3))
        cv2.rectangle(im, (x, y), (x + rng.randint(20, 60), y + rng.randint(10, 30)), c, -1)
    for k in range(5):
        cv2.putText(im, "Lorem ipsum", (8, 20 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return im


# --------------------------------------------------------------------------- tiles and statistics
def test_tile_grid_covers_the_image_and_box_sums_are_exact():
    m = np.random.RandomState(1).rand(37, 53).astype(np.float32)
    for s in (8, 16, 32):
        t = AC.tile_grid(37, 53, s)
        cover = np.zeros((37, 53), bool)
        for x0, y0, x1, y1 in t:
            cover[y0:y1, x0:x1] = True
        assert cover.all()
        sums = AC._box_sums(m, t)
        for i in (0, len(t) // 2, len(t) - 1):
            x0, y0, x1, y1 = t[i]
            assert math.isclose(sums[i], float(m[y0:y1, x0:x1].sum()), rel_tol=1e-5)


def test_identical_images_give_zero_statistics_and_no_region():
    im = _page()
    maps = AC.pixel_maps(im, im)
    st, content = AC.tile_stats(maps, AC.tile_grid(*im.shape[:2], 16))
    assert np.allclose(st, 0, atol=1e-6) and (content >= 0).all()
    an = AC.analyse(im, im, None)
    null = _synthetic_null()
    assert AC.detect_regions(an, null) == []


# --------------------------------------------------------------------------- nuisance registration (post-raster)
def test_post_raster_nuisance_is_registered_exactly():
    im = _page(2)
    blurred = cv2.GaussianBlur(im, (0, 0), 0.46)
    fitted, sigma, q, sc = AC.fit_image_level(blurred, im)
    assert sigma is not None and abs(sigma - 0.46) <= 0.01 and q is None
    assert AC.unexplained(blurred, fitted) <= 0.002 * im.shape[0] * im.shape[1]
    jp = AC._jpeg(im, 77)
    fitted, sigma, q, sc = AC.fit_image_level(jp, im)
    assert q is not None and abs(q - 77) <= 2 and sc == 0


def test_ls_offset_recovers_a_subpixel_shift_direction():
    im = _page(3)
    M = np.float32([[1, 0, 0.4], [0, 1, 0]])
    sh = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dx, dy = AC.ls_offset(sh, im)
    assert dx > 0.2 and abs(dy) < 0.1


# --------------------------------------------------------------------------- null model and NFA
def _synthetic_null(n_pages: int = 6) -> AC.NullModel:
    """H0 = the same synthetic page under random blur misfit / JPEG, compared with its registration."""
    null, pix = AC.NullModel(), AC._PixelAcc()
    rng = random.Random(0)
    for k in range(n_pages):
        im = _page(10 + k)
        noisy = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.35, 0.6)) if k % 2 else AC._jpeg(im, rng.randint(70, 95))
        an = AC.analyse(noisy, im, None)
        AC.accumulate_h0(null, pix, an, [])
        an0 = AC.analyse(im, im, None)
        AC.accumulate_h0(null, pix, an0, [])
    return AC.finalize_null(null, pix, {"test": True})


def test_null_model_pvalues_are_monotone_zero_inflated_and_roundtrip():
    null = _synthetic_null(4)
    key = next(k for k in null.strata if k.startswith("de_mean|16|"))
    st = null.strata[key]
    x = np.array([0.0, 1e-4, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4])
    p = null.survival(st, x)
    assert p[0] == 1.0 and np.all(np.diff(p) <= 1e-12) and p[-1] > 0
    back = AC.NullModel.from_dict(null.to_dict())
    assert np.allclose(back.survival(back.strata[key], x), p)
    u, beta, xi = AC.NullModel.gpd(np.sort(np.random.RandomState(0).exponential(1.0, 400)))
    assert beta > 0 and 0.1 <= xi <= 0.5


def test_false_alarms_are_controlled_on_held_out_nuisance_and_a_real_change_is_found():
    null = _synthetic_null(6)
    rng = random.Random(5)
    fa = 0
    for k in range(4):
        im = _page(100 + k)
        noisy = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.35, 0.6))
        an = AC.analyse(noisy, im, None)
        fa += len(AC.detect_regions(an, null))
    assert fa <= 2  # eps = 1 expected false alarm per page is an upper bound; observed is far below
    im = _page(200)
    cand = im.copy()
    cv2.rectangle(cand, (150, 100), (190, 130), (200, 40, 40), -1)  # an extra element in the candidate
    an = AC.analyse(im, cand, None)
    regs = AC.detect_regions(an, null)
    assert any(r.box.iou(Box(150, 100, 41, 31)) > 0.5 for r in regs)
    assert all(r.log_nfa < math.log10(1.0) for r in regs)


# --------------------------------------------------------------------------- typing tree
def test_tree_learns_rules_and_serialises():
    rng = np.random.RandomState(0)
    X = rng.rand(300, 3)
    y = ["color.fill" if a > 0.6 else "structure.missing" if b > 0.5 else AC.NONE_CLASS for a, b, _ in X]
    t = AC.Tree(4, 3).fit(X, y, ["chroma_frac", "ink_ratio", "noise"])
    acc = np.mean([t.classes[int(np.argmax(t.predict_proba(x)))] == v for x, v in zip(X, y)])
    assert acc > 0.97
    assert any("chroma_frac" in r for r in t.rules()) and "noise" not in t.importance()
    t2 = AC.Tree.from_dict(t.to_dict())
    assert np.allclose(t2.predict_proba(X[0]), t.predict_proba(X[0]))


# --------------------------------------------------------------------------- end to end with the shipped model (real renderer)
@pytest.mark.skipif(not os.path.exists(AC.MODEL_PATH), reason="calibrated model not built")
def test_end_to_end_find_on_a_rendered_pair_is_silent_on_noise_and_finds_a_perturbation():
    from dt.render.screenshot import render_doc
    doc = Document.load(SYNTH)
    doc.source_image = None
    clean = render_doc(doc)
    cand_ir, _ = PT.sanitize(copy.deepcopy(doc))
    # pure nuisance: a sub-pixel page offset, registered in the renderer -> silent
    target = PT.render_target(doc, PT.NoiseRecipe(translate=(0.375, 0.0)), clean)
    fs = AC.find(target, clean, cand_ir)
    assert [f for f in fs if TYPES.get(f.type) and TYPES[f.type].is_error] == []
    # one real error: a fill recolour
    cand, gt = PT.perturb(doc, [("color.fill", 0.8)], random.Random(3))
    cand_rgb = render_doc(cand)
    fs = AC.find(clean, cand_rgb, cand)
    errs = [f for f in fs if TYPES[f.type].is_error]
    assert errs and all(is_known(f.type) for f in fs)
    assert any(e.box.iou(gt[0].box) >= 0.3 or gt[0].box.contains_point(e.box.cx, e.box.cy) for e in errs)
    for f in errs:
        assert 0.0 <= f.confidence <= 1.0 and f.evidence["log10_nfa"] < 0 and Finding.from_dict(f.to_dict()).type == f.type
