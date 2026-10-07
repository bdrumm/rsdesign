"""Non-text segmentation: background, flat regions, bordered boxes, pills/circles, dividers,
icons, images and soft shadows.

Algorithm (recursive, deterministic):

    1. ``estimate_background``: mode of the border ring vs dominant colour of the image.
    2. ``mask_text``: paint every OCR text box with its local background so glyphs never form
       components of their own.
    3. ``_find_in(box, bg)``: pixels farther than ``bg_dist`` from ``bg`` form 8-connected
       components. Each component is classified by ``classify_component``:
         * divider  - thin and long
         * icon     - small, glyph-like (low solidity), high contrast
         * flat     - a fill-consistent core covers most of the hole-filled mask -> rect / pill /
                      ellipse (radius from the corner profile, see ``estimate_radius``), optional
                      stroke ring whose width = (outer box - core box) / 2
         * bordered - a thin closed ring around background-coloured interior
         * image    - many distinct colours
       Flat regions are then scanned again with ``bg = fill`` (inside their eroded core) so
       nested elements are found against the right background.
         * bordered - a thin (<= stroke_max) ring around background-coloured interior -> fill=None +
                      stroke; an outline whose edge is broken by a floating label is closed by
                      its convex hull (``_open_ring_fill``); radius from 50%-coverage ring pixels
         * mixed    - a large solid single-colour core (``_large_flat_core``) is split off and the
                      rest re-scanned (a dialog surface touching the darkened cards it overlaps)
    4. ``detect_shadow``: probes strips outside each region for a dark halo fading with distance.
    5. ``detect_scrim``: a uniform darkening of the whole page (dialog scrim). The page bg is then
       the true colour (``estimate_background``), a full-page translucent scrim Region is added,
       regions are tagged ``under_scrim`` (true colours in meta / on the region with
       ``perceive.seg.scrim_unblend``) or ``scrim_clear``; ``apply_scrim(root)`` re-stacks a built
       tree so under-scrim nodes paint below the scrim and the dialog above it.

Everything returns ``Region`` objects in absolute pixel coordinates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

from dt.common.image import dominant_color
from dt.ir import Box, Color, Shadow
from dt.params import P, register

# --------------------------------------------------------------------------- params
register("perceive.seg.bg_dist", 12.0, "RGB distance from the (local) background above which a pixel is 'ink'", (6.0, 30.0))
register("perceive.seg.fill_tol", 6.0, "RGB distance within which pixels count as the same flat fill (tight: shadows are gradients)", (3.0, 20.0))
register("perceive.seg.ring_consistency", 0.7, "fraction of a candidate stroke ring that must match its dominant colour", (0.4, 0.95))
register("perceive.seg.fragment_dist", 2.0, "regions with fill within x bg_dist of the bg lying in a shadow halo are dropped", (1.0, 4.0))
register("perceive.seg.min_area", 6, "min component area (px) to consider", (2, 40))
register("perceive.seg.text_pad", 2, "px padding around text boxes when masking them out", (0, 6))
register("perceive.seg.fill_band", 6, "px band inside a component's outline from which its fill colour is read", (3, 12))
register("perceive.seg.solid_frac", 0.6, "core/filled area ratio above which a component is a flat region", (0.4, 0.9))
register("perceive.seg.hollow_frac", 0.55, "ink/filled area ratio below which a component is a bordered (hollow) box", (0.2, 0.8))
register("perceive.seg.divider_max_t", 3, "max thickness (px) of a divider", (1, 6))
register("perceive.seg.divider_min_len", 24, "min length (px) of a divider", (8, 80))
register("perceive.seg.divider_ratio", 8.0, "min length/thickness ratio of a divider", (3.0, 20.0))
register("perceive.seg.icon_max", 48, "max side (px) of an icon blob", (16, 96))
register("perceive.seg.icon_min", 5, "min side (px) of an icon blob", (2, 12))
register("perceive.seg.icon_aspect", 2.2, "max aspect ratio (long/short side) of an icon blob", (1.2, 4.0))
register("perceive.seg.icon_solidity", 0.9, "ink/bbox ratio below which a small blob is a glyph-like icon", (0.5, 0.98))
register("perceive.seg.circle_aspect_tol", 2, "max |w-h| (px) for a fully rounded shape to be a circle", (0, 4))
register("perceive.seg.pill_frac", 0.42, "radius/min(w,h) above which a rect is a pill", (0.3, 0.5))
register("perceive.seg.image_colors", 40, "distinct quantised colours above which a region is an image", (10, 200))
register("perceive.seg.image_min_side", 16, "min side (px) for an image region", (8, 64))
register("perceive.seg.recurse_min_side", 8, "min side (px) of a flat region to scan its inside", (4, 32))
register("perceive.seg.recurse_inset", 2, "px eroded from a flat region's core before scanning its inside", (1, 4))
register("perceive.seg.max_depth", 6, "max nesting depth of the recursive scan", (1, 10))
register("perceive.seg.stroke_max", 6, "max stroke width (px) considered for a flat region's border ring", (1, 12))
register("perceive.seg.stroke_min_de", 8.0, "min RGB distance of a stroke from both fill and background", (4.0, 40.0))
register("perceive.seg.shadow_probe", 14, "px probed outside a region for a shadow halo", (4, 32))
register("perceive.seg.shadow_min", 2.0, "min darkening (mean RGB) at 1px outside for a shadow", (0.5, 10.0))
register("perceive.seg.shadow_fade", 0.5, "halo must fall below this fraction of its peak within the probe", (0.2, 0.9))
register("perceive.seg.glyph_color_rel", 0.97, "icon colour = median of pixels >= rel x max distance from bg (thin strokes rarely cover a pixel fully)", (0.8, 1.0))
register("perceive.seg.quant", 4, "colour quantisation step for dominant-colour estimates", (2, 16))
register("perceive.seg.split_frac", 0.3, "core/filled ratio above which a non-flat component is split into its flat core + remainder (overlapping siblings)", (0.15, 0.6))
register("perceive.seg.dedupe_iou", 0.95, "box IoU above which two detected regions are the same element (keep the deeper one)", (0.8, 1.0))
# hollow (outline-only) containers
register("perceive.seg.hollow_bg_de", 1.0, "Lab dE below which a ring's eroded interior counts as the parent background (hollow region: fill=None); M3 surface-container-low vs surface is only 2.0", (0.3, 3.0))
register("perceive.seg.ring_min_side", 12, "min side (px) of a component tested as an open outline ring (label gap in the stroke)", (6, 40))
register("perceive.seg.ring_band_frac", 0.97, "fraction of an open ring's ink that must lie within stroke_max px of its bbox edges", (0.85, 1.0))
# scrim (translucent full-page overlay under a dialog / modal sheet)
register("perceive.seg.scrim_enable", 1, "1 = detect a uniform full-page darkening (dialog scrim) and model it", (0, 1))
register("perceive.seg.scrim_alpha_min", 0.2, "min blend factor of a scrim (below: a darker surface role such as surface-dim, not an overlay)", (0.08, 0.4))
register("perceive.seg.scrim_alpha_max", 0.8, "max blend factor of a scrim", (0.5, 0.95))
register("perceive.seg.scrim_fit_tol", 3.0, "max per-channel RGB residual when the observed page bg is explained as a known surface blended toward the scrim colour", (1.0, 8.0))
register("perceive.seg.scrim_known_de", 3.0, "Lab dE within which an un-blended sample colour matches a known design-system colour or a colour seen un-darkened on the page", (1.0, 8.0))
register("perceive.seg.scrim_gamut_tol", 6.0, "RGB margin beyond [0,255] after un-blending above which a colour cannot lie under the scrim (it is on the clear dialog)", (2.0, 20.0))
register("perceive.seg.scrim_min_samples", 2, "min distinct page colours (besides the bg) explained as darkened known colours", (1, 6))
register("perceive.seg.scrim_sample_frac", 0.004, "min image fraction of a colour to be sampled as a surface for the scrim fit", (0.001, 0.05))
register("perceive.seg.scrim_min_bg_frac", 0.15, "min image fraction covered by the (darkened) page background", (0.05, 0.6))
register("perceive.seg.scrim_clear_frac", 0.02, "min image fraction of colours that cannot be darkened (the dialog surface above the scrim)", (0.005, 0.2))
register("perceive.seg.scrim_unblend", 0, "1 = under-scrim regions carry their un-blended (true) fill and need dt.perceive.segment.apply_scrim on the tree; 0 = keep the observed fill (pixel-exact with plain containment nesting) and put the true colour in meta.true_fill", (0, 1))
register("perceive.seg.split_min_area", 4000, "px area of a solid flat core above which a mixed component is split even when the core is a small part of it", (500, 40000))
register("perceive.seg.split_colors", 6, "most frequent ink colours searched for a large flat core in a mixed component", (2, 16))
register("perceive.seg.split_core_solid", 0.85, "min core area / core bbox area for the large-core split", (0.6, 0.98))
register("perceive.seg.radius_fit_rms", 2.0, "px RMS residual above which a corner profile is not an arc (occluded corner) and is left out of the radius estimate", (0.5, 6.0))
register("perceive.seg.ring_side_cov", 0.5, "min fraction of each bbox side covered by ink for an open ring (gaps such as a floating label allowed)", (0.2, 0.9))


# --------------------------------------------------------------------------- data
@dataclass
class Region:
    """A non-text visual region in absolute pixel coordinates.

    kind: rect | pill | ellipse | divider | icon | image. ``fill`` None = transparent (bordered
    box showing the parent background). ``stroke`` = (color, width). ``mask`` is the component
    mask cropped to ``box`` (bool). ``bg`` is the background it was detected against.
    """
    box: Box
    kind: str
    fill: Optional[Color] = None
    stroke: Optional[tuple[Color, float]] = None
    radius: float = 0.0
    has_shadow: bool = False
    conf: float = 1.0
    mask: Optional[np.ndarray] = None
    shadow: Optional[Shadow] = None
    bg: Color = field(default_factory=lambda: Color(255, 255, 255))
    depth: int = 0
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers
def _dist(a: np.ndarray, c: Color) -> np.ndarray:
    d = a[..., :3].astype(np.float32) - np.array([c.r, c.g, c.b], np.float32)
    return np.sqrt((d * d).sum(axis=-1))


def _ring(a: np.ndarray, t: int = 1) -> np.ndarray:
    h, w = a.shape[:2]
    t = max(1, min(t, max(1, h // 2), max(1, w // 2)))
    return np.concatenate([a[:t].reshape(-1, 3), a[-t:].reshape(-1, 3), a[:, :t].reshape(-1, 3), a[:, -t:].reshape(-1, 3)])


def _dominant(px: np.ndarray, quant: Optional[int] = None) -> Optional[Color]:
    if px.size == 0:
        return None
    return dominant_color(px.reshape(-1, 1, 3), quant=quant or int(P["perceive.seg.quant"]))


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _largest_component(mask: np.ndarray, seed: Optional[np.ndarray] = None) -> np.ndarray:
    """Largest 8-connected component of mask (or the one overlapping `seed` most)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    if seed is not None and seed.any():
        counts = np.bincount(lab[seed], minlength=n)
        counts[0] = 0
        best = int(np.argmax(counts)) if counts.max() > 0 else int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    else:
        best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return lab == best


def _seen_background(rgb: np.ndarray) -> Color:
    """Observed page background: mode of the 2px border ring (dominant colour as fallback)."""
    ring = _dominant(_ring(rgb, 2))
    if ring is None:
        return _dominant(rgb[::2, ::2]) or Color(255, 255, 255)
    return ring


def estimate_background(rgb: np.ndarray) -> Color:
    """Page background: mode of the 2px border ring. When the page is under a dialog scrim
    (``detect_scrim``) this is the TRUE page colour; ``find_regions`` then emits the scrim as a
    translucent full-page region, so root fill + scrim reproduce the observed colour."""
    seen = _seen_background(rgb)
    sc = detect_scrim(rgb, seen)
    return sc.page_bg if sc is not None else seen


def mask_text(rgb: np.ndarray, text_boxes: list[Box], pad: Optional[int] = None) -> np.ndarray:
    """Copy of rgb with each text box painted in its local background colour."""
    out = rgb.copy()
    H, W = rgb.shape[:2]
    pad = int(P["perceive.seg.text_pad"]) if pad is None else pad
    for b in text_boxes:
        x0, y0, x1, y1 = b.expand(pad).as_int()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        ox0, oy0, ox1, oy1 = max(0, x0 - 1), max(0, y0 - 1), min(W, x1 + 1), min(H, y1 + 1)
        outer = rgb[oy0:oy1, ox0:ox1]
        ringpx = _ring(outer, 1)
        c = _dominant(ringpx) or Color(255, 255, 255)
        out[y0:y1, x0:x1] = (c.r, c.g, c.b)
    return out


# --------------------------------------------------------------------------- radius
def _corner_profile(mask: np.ndarray, corner: str, n: int) -> np.ndarray:
    """Inset (px) of the mask's edge from the bounding rect for the first n rows of a corner."""
    h, w = mask.shape
    m = mask
    if corner in ("tr", "br"):
        m = m[:, ::-1]
    if corner in ("bl", "br"):
        m = m[::-1, :]
    prof = np.zeros(n, np.float32)
    for y in range(n):
        row = np.where(m[y])[0]
        prof[y] = float(row[0]) if row.size else float(w)
    return prof


def _arc_insets(r: int, n: int) -> np.ndarray:
    """Predicted first-on-pixel index per row for a corner arc of radius r (pixel-centre sampling)."""
    pred = np.zeros(n, np.float32)
    if r <= 0:
        return pred
    ys = np.arange(n, dtype=np.float32) + 0.5
    inside = ys < r
    pred[inside] = np.ceil(r - 0.5 - np.sqrt(np.maximum(0.0, r * r - (r - ys[inside]) ** 2)))
    return np.maximum(0.0, pred)


def estimate_radius(mask: np.ndarray) -> tuple[float, tuple[float, float, float, float]]:
    """Corner radius of a (filled, 50%-coverage) rounded-rect mask by fitting a circular arc to
    each corner's edge-inset profile. Returns (radius, per-corner radii tl,tr,br,bl).

    The radius is the median over the corners whose profile actually is an arc (RMS residual <=
    ``radius_fit_rms`` px): a corner hidden under an overlapping element (a translucent card, a
    badge) leaves a step-shaped notch that fits no arc and would otherwise drag the median of four
    corners to the notch size. The per-corner values are the raw fits (evidence)."""
    h, w = mask.shape
    rmax = int(min(w, h) // 2)
    if rmax < 1:
        return 0.0, (0.0, 0.0, 0.0, 0.0)
    out: list[float] = []
    good: list[float] = []
    tol = float(P["perceive.seg.radius_fit_rms"])
    for corner in ("tl", "tr", "br", "bl"):
        prof = _corner_profile(mask, corner, rmax)
        if prof[0] <= 0.0:
            out.append(0.0)
            good.append(0.0)
            continue
        best_r, best_err = 0.0, float("inf")
        for r in range(0, rmax + 1):
            err = float(((prof - _arc_insets(r, rmax)) ** 2).sum())
            if err < best_err:
                best_r, best_err = float(r), err
        out.append(best_r)
        if (best_err / rmax) ** 0.5 <= tol:
            good.append(best_r)
    tl, tr, br, bl = out
    return float(np.median(good if good else out)), (tl, tr, br, bl)


# --------------------------------------------------------------------------- classification
def _refine_box(crop: np.ndarray, core: np.ndarray, fill: Color, bg: Color) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """50%-coverage mask and bbox: pixels closer to `fill` than to `bg`, connected to core."""
    d_fill, d_bg = _dist(crop, fill), _dist(crop, bg)
    span = float(_dist(np.array([[[fill.r, fill.g, fill.b]]], np.uint8), bg)[0, 0])
    # a boundary pixel is a blend of fill and bg: closer to fill than bg, and no farther from fill
    # than bg is (shadow pixels next to the edge are darker than both and fail this)
    near = (d_fill <= d_bg) & (d_bg > 0.5 * float(P["perceive.seg.bg_dist"])) & (d_fill <= span + float(P["perceive.seg.fill_tol"]))
    grown = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    m = _largest_component(near & grown, seed=core) | core
    m = ndi.binary_fill_holes(m)
    return (_bbox(m) or _bbox(core)), m  # type: ignore[return-value]


def _strip_color(sub: np.ndarray, side: str, bb: tuple[int, int, int, int], d: int) -> Optional[np.ndarray]:
    """Median colour of the 1px strip at distance d outside bbox bb on one side (corners excluded)."""
    x0, y0, x1, y1 = bb
    h, w = sub.shape[:2]
    inset = max(1, min(x1 - x0, y1 - y0) // 4)
    if side == "top":
        y = y0 - d
        px = sub[y, x0 + inset:x1 - inset] if y >= 0 else None
    elif side == "bottom":
        y = y1 + d - 1
        px = sub[y, x0 + inset:x1 - inset] if y < h else None
    elif side == "left":
        x = x0 - d
        px = sub[y0 + inset:y1 - inset, x] if x >= 0 else None
    else:
        x = x1 + d - 1
        px = sub[y0 + inset:y1 - inset, x] if x < w else None
    if px is None or px.size == 0:
        return None
    return np.median(px[:, :3].astype(np.float32), axis=0)


def _stroke_from_strips(sub: np.ndarray, core_bb: tuple[int, int, int, int], fill: Color, bg: Color) -> Optional[tuple[Color, int]]:
    """Stroke of a flat region: constant-colour 1px strips right outside its fill-consistent core.

    All four sides at distance 1 must agree (a shadow with dy != 0 is asymmetric and fails),
    the colour must differ from both fill and background, and the width is the number of
    consecutive strips matching that colour (a fading shadow stops matching immediately).
    """
    tol = 1.5 * float(P["perceive.seg.fill_tol"])
    min_de = float(P["perceive.seg.stroke_min_de"])
    sides = ("top", "bottom", "left", "right")
    first = {sd: _strip_color(sub, sd, core_bb, 1) for sd in sides}
    first = {sd: v for sd, v in first.items() if v is not None}
    if len(first) < 2:
        return None
    ref = np.median(np.stack(list(first.values())), axis=0)
    if any(np.linalg.norm(v - ref) > tol for v in first.values()):
        return None
    color = Color.from_rgb(*ref)
    fillv = np.array([fill.r, fill.g, fill.b], np.float32)
    bgv = np.array([bg.r, bg.g, bg.b], np.float32)
    if np.linalg.norm(ref - fillv) < min_de or np.linalg.norm(ref - bgv) < min_de:
        return None
    width = 1
    for d in range(2, int(P["perceive.seg.stroke_max"]) + 1):
        cols = [_strip_color(sub, sd, core_bb, d) for sd in first]
        cols = [c for c in cols if c is not None]
        if not cols or any(np.linalg.norm(c - ref) > tol for c in cols):
            break
        width = d
    return color, width


def _ring_width(sub: np.ndarray, cmask: np.ndarray, stroke: Color, bg: Color) -> float:
    """Thickness of a hollow ring: median coverage mass of the first ink run from each side's midpoint.

    Mass (sum of min(1, dist_to_bg / dist_stroke_to_bg)) is robust to borders at fractional
    positions, which antialiasing spreads over two half-covered pixels.
    """
    h, w = cmask.shape
    full = max(1.0, float(_dist(np.array([[[stroke.r, stroke.g, stroke.b]]], np.uint8), bg)[0, 0]))
    cov = np.minimum(1.0, _dist(sub, bg) / full)
    runs: list[float] = []
    for line, cline in ((cmask[h // 2, :], cov[h // 2, :]), (cmask[h // 2, ::-1], cov[h // 2, ::-1]),
                        (cmask[:, w // 2], cov[:, w // 2]), (cmask[::-1, w // 2], cov[::-1, w // 2])):
        on = np.where(line)[0]
        if on.size == 0:
            continue
        start = on[0]
        k = 0
        while start + k < line.size and line[start + k]:
            k += 1
        runs.append(float(cline[start:start + k].sum()))
    return float(np.median(runs)) if runs else 1.0


def _hull_fill(m: np.ndarray) -> np.ndarray:
    """Filled convex hull of a mask (bridges gaps in a straight edge of an outline ring)."""
    ys, xs = np.nonzero(m)
    out = np.zeros(m.shape, np.uint8)
    if xs.size >= 3:
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
        cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool) | m


def _open_ring_fill(m: np.ndarray) -> Optional[np.ndarray]:
    """Hole-filled shape of an outline ring that is open on one edge (a floating label breaks the
    stroke of an outlined text field and OCR did not mask the whole gap): nearly all ink lies within
    ``stroke_max`` px of the bbox edges and every side is mostly covered -> its convex hull.
    Returns None for anything that is not such a ring."""
    h, w = m.shape
    t = int(P["perceive.seg.stroke_max"])
    if min(h, w) < max(int(P["perceive.seg.ring_min_side"]), 2 * t + 3):
        return None
    total = float(m.sum())
    band = np.zeros_like(m)
    band[:t] = band[-t:] = True
    band[:, :t] = band[:, -t:] = True
    if total <= 0 or float((m & band).sum()) < float(P["perceive.seg.ring_band_frac"]) * total:
        return None
    sides = (m[:t].any(0).mean(), m[-t:].any(0).mean(), m[:, :t].any(1).mean(), m[:, -t:].any(1).mean())
    if min(sides) < float(P["perceive.seg.ring_side_cov"]):
        return None
    return _hull_fill(m)


def _ring_shape(sub: np.ndarray, m: np.ndarray, stroke: Color, bg: Color) -> tuple[str, float, tuple[float, float, float, float]]:
    """rect / pill / ellipse + radius of an outline ring, measured on its 50%-coverage pixels (the
    faint antialiased fringe of a 1px stroke would otherwise square the corners off)."""
    full = max(1.0, float(_dist(np.array([[[stroke.r, stroke.g, stroke.b]]], np.uint8), bg)[0, 0]))
    m50 = m & (_dist(sub, bg) >= 0.5 * full)
    if m50.sum() < 4:
        m50 = m
    shape = ndi.binary_fill_holes(m50)
    if shape.sum() <= 1.05 * m50.sum():
        shape = _hull_fill(m50)
    bb = _bbox(shape)
    if bb is None:
        return _shape_kind(m, m.shape[1], m.shape[0])
    s = shape[bb[1]:bb[3], bb[0]:bb[2]]
    return _shape_kind(s, bb[2] - bb[0], bb[3] - bb[1])


def _solidity(mask: np.ndarray) -> float:
    """Mask area / its bbox area."""
    bb = _bbox(mask)
    if bb is None:
        return 0.0
    return float(mask.sum()) / max(1.0, float((bb[2] - bb[0]) * (bb[3] - bb[1])))


def _large_flat_core(sub: np.ndarray, m: np.ndarray, filled: np.ndarray, bg: Color, tol: float):
    """Largest solid single-colour blob inside a mixed component: (colour, core, core_filled) or None.

    Looks at the most frequent ink colours; a blob qualifies when its hole-filled area is at least
    ``split_min_area`` and fills ``split_core_solid`` of its bbox (a surface, not a stroke web)."""
    px = sub[m][:, :3]
    if px.shape[0] < float(P["perceive.seg.split_min_area"]):
        return None
    q = int(P["perceive.seg.quant"])
    keys = (px // q).astype(np.int32)
    keys = keys[:, 0] * 65536 + keys[:, 1] * 256 + keys[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    best = None
    for k in uniq[np.argsort(-counts)][: int(P["perceive.seg.split_colors"])]:
        sel = px[keys == k]
        c = Color.from_rgb(*np.median(sel, axis=0))
        if float(_dist(np.array([[[c.r, c.g, c.b]]], np.uint8), bg)[0, 0]) <= 0.5 * float(P["perceive.seg.bg_dist"]):
            continue
        cm = (_dist(sub, c) <= tol) & m  # ink only: hole pixels near the colour would make no progress on the remainder
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cm.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.5 * float(P["perceive.seg.split_min_area"]):
                continue
            core = lab == i
            c = _dominant(sub[core]) or c
            cf = ndi.binary_fill_holes(core) & filled
            area = float(cf.sum())
            if area < float(P["perceive.seg.split_min_area"]) or _solidity(cf) < float(P["perceive.seg.split_core_solid"]):
                continue
            if best is None or area > best[3]:
                best = (c, core, cf, area)
    return None if best is None else best[:3]


def _lab_de(a: Color, b: Color) -> float:
    return float(Color(a.r, a.g, a.b).delta_e(Color(b.r, b.g, b.b)))


def _shape_kind(mask: np.ndarray, w: int, h: int) -> tuple[str, float, tuple[float, float, float, float]]:
    """rect / pill / ellipse from the hole-filled mask, plus the radius estimate."""
    r, corners = estimate_radius(mask)
    if r >= float(P["perceive.seg.pill_frac"]) * min(w, h) and min(w, h) >= 6:
        r = min(w, h) / 2.0
        kind = "ellipse" if abs(w - h) <= int(P["perceive.seg.circle_aspect_tol"]) else "pill"
        return kind, r, (r, r, r, r)
    return "rect", r, corners


def _glyph_color(sub: np.ndarray, m: np.ndarray, bg: Color) -> Color:
    """Colour of a thin antialiased glyph: median of the pixels farthest from the background."""
    px = sub[m][:, :3]
    if px.size == 0:
        return bg
    d = _dist(px.reshape(-1, 1, 3), bg).ravel()
    top = px[d >= float(P["perceive.seg.glyph_color_rel"]) * d.max()]
    return Color.from_rgb(*np.median(top, axis=0))


def _is_divider(w: int, h: int) -> bool:
    t, L = min(w, h), max(w, h)
    return t <= int(P["perceive.seg.divider_max_t"]) and L >= int(P["perceive.seg.divider_min_len"]) and L / max(1, t) >= float(P["perceive.seg.divider_ratio"])


def _is_image(crop: np.ndarray, cmask: np.ndarray, w: int, h: int) -> bool:
    if min(w, h) < int(P["perceive.seg.image_min_side"]):
        return False
    px = (crop[cmask][:, :3] // 16).astype(np.int32)
    keys = px[:, 0] * 256 + px[:, 1] * 16 + px[:, 2]
    return len(np.unique(keys)) >= int(P["perceive.seg.image_colors"])


def _fill_candidates(sub: np.ndarray, filled: np.ndarray, w: int, h: int) -> list[tuple[str, np.ndarray]]:
    """Pixel sets whose dominant colour may be the component's fill, most specific first:
    a band just inside the outline (containers whose children outnumber their own pixels),
    then the eroded interior (shadowed elements, whose outline is the halo's edge)."""
    fb = int(P["perceive.seg.fill_band"])
    out: list[tuple[str, np.ndarray]] = []
    if min(w, h) > 2 * fb:
        out.append(("band", ndi.binary_erosion(filled, iterations=1) & ~ndi.binary_erosion(filled, iterations=fb)))
    if min(w, h) > 6:
        out.append(("interior", ndi.binary_erosion(filled, iterations=2)))
    if min(w, h) > 2 * fb + 4:
        out.append(("deep", ndi.binary_erosion(filled, iterations=fb + 2)))
    out.append(("all", filled))
    return [(n, m) for n, m in out if m.sum() >= 4]


def _best_fill(sub: np.ndarray, filled: np.ndarray, w: int, h: int, ink_px: np.ndarray, bg: Color, tol: float):
    """Fill colour (+ core masks) = the candidate whose colour-consistent, hole-filled core covers
    the largest part of the component; stops at the first candidate that is solid enough."""
    best = None
    tried: set[str] = set()
    for name, m in _fill_candidates(sub, filled, w, h):
        c = _dominant(sub[m])
        if c is None or c.hex() in tried:
            continue
        tried.add(c.hex())
        core = _largest_component((_dist(sub, c) <= tol) & filled, seed=m)
        core_filled = ndi.binary_fill_holes(core) & filled
        frac = float(core_filled.sum()) / max(1.0, float(filled.sum()))
        if best is None or frac > best[3]:
            best = (c, core, core_filled, frac)
        if frac >= float(P["perceive.seg.solid_frac"]):
            break
    if best is None:
        c = _dominant(ink_px) or bg
        return c, filled, filled, 0.0
    return best


def classify_component(crop: np.ndarray, cmask: np.ndarray, bg: Color, text_mask: Optional[np.ndarray] = None) -> Optional[Region]:
    """Classify one connected component (crop-local coords; caller offsets the box)."""
    bb = _bbox(cmask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    w, h = x1 - x0, y1 - y0
    sub = crop[y0:y1, x0:x1]
    m = cmask[y0:y1, x0:x1]
    ink_px = sub[m]
    # 1. divider
    if _is_divider(w, h):
        c = _dominant(ink_px) or bg
        return Region(Box(x0, y0, w, h), "divider", fill=c, mask=m.copy(), bg=bg, meta={"area": int(m.sum())})
    # hole-filled shape (text boxes count as ink so labels straddling a border keep the hole closed)
    closed = m.copy()
    if text_mask is not None:
        closed |= text_mask[y0:y1, x0:x1]
    filled = ndi.binary_fill_holes(closed)
    if filled.sum() <= 1.05 * closed.sum():  # no hole: maybe an outline whose edge has a label gap
        ring = _open_ring_fill(m)
        if ring is not None:
            filled = ring | closed
    ink_ratio = float(m.sum()) / max(1.0, float(filled.sum()))
    solidity = float(m.sum()) / max(1.0, float(w * h))
    # 2. icon: small glyph-like blob
    if max(w, h) <= int(P["perceive.seg.icon_max"]) and min(w, h) >= int(P["perceive.seg.icon_min"]):
        aspect = max(w, h) / max(1, min(w, h))
        if aspect <= float(P["perceive.seg.icon_aspect"]) and solidity < float(P["perceive.seg.icon_solidity"]):
            interior = ndi.binary_erosion(filled, iterations=1)
            flat_small = interior.sum() >= 0.5 * filled.sum() and ink_ratio > 0.9
            if not flat_small:
                c = _glyph_color(sub, m, bg)
                return Region(Box(x0, y0, w, h), "icon", fill=c, mask=m.copy(), bg=bg, conf=0.8,
                              meta={"solidity": round(solidity, 3), "ink_ratio": round(ink_ratio, 3)})
    # 3. flat region: the fill is read from a band just inside the outline (children may
    #    outnumber the container's own pixels), the core = pixels matching it, holes = children
    fill_tol = float(P["perceive.seg.fill_tol"])
    fill, core, core_filled, core_frac = _best_fill(sub, filled, w, h, ink_px, bg, fill_tol)
    if core_frac >= float(P["perceive.seg.solid_frac"]) and _dist(np.array([[[fill.r, fill.g, fill.b]]], np.uint8), bg)[0, 0] > 0.5 * float(P["perceive.seg.bg_dist"]):
        cb = _bbox(core_filled)
        stroke = _stroke_from_strips(sub, cb, fill, bg) if cb is not None else None
        if stroke is None:
            rb, m50 = _refine_box(sub, core_filled, fill, bg)
            box_mask = m50[rb[1]:rb[3], rb[0]:rb[2]]
            rw, rh = rb[2] - rb[0], rb[3] - rb[1]
            kind, radius, corners = _shape_kind(box_mask, rw, rh)
        else:
            # shape from the interior (50% against the stroke colour, immune to the shadow halo),
            # then grown by the stroke width; outer radius = inner radius + width
            sw = stroke[1]
            ib, m50 = _refine_box(sub, core_filled, fill, stroke[0])
            kind, radius, corners = _shape_kind(m50[ib[1]:ib[3], ib[0]:ib[2]], ib[2] - ib[0], ib[3] - ib[1])
            rb = (max(0, ib[0] - sw), max(0, ib[1] - sw), min(w, ib[2] + sw), min(h, ib[3] + sw))
            rw, rh = rb[2] - rb[0], rb[3] - rb[1]
            grown = ndi.binary_dilation(m50, iterations=sw)
            box_mask = grown[rb[1]:rb[3], rb[0]:rb[2]]
            if kind == "rect":
                radius = radius + sw if radius > 0 else 0.0
                corners = tuple(c + sw if c > 0 else 0.0 for c in corners)
            else:
                radius = min(rw, rh) / 2.0
        return Region(Box(x0 + rb[0], y0 + rb[1], rw, rh), kind, fill=fill, stroke=stroke, radius=radius,
                      mask=box_mask.copy(), bg=bg, meta={"core_frac": round(core_frac, 3), "corners": corners, "area": int(m.sum())})
    # 4. bordered (hollow) box: box = ink extent (antialiased 1px borders straddle two pixels)
    #    A ring wider than stroke_max is not an outline (e.g. overlapping elements that touch
    #    around a lighter one) and falls through to the split / textured paths below.
    if ink_ratio < float(P["perceive.seg.hollow_frac"]) and filled.sum() > m.sum():
        stroke_c = _dominant(ink_px) or bg
        width = _ring_width(sub, m, stroke_c, bg)
        if width <= float(P["perceive.seg.stroke_max"]):
            kind, radius, corners = _ring_shape(sub, m, stroke_c, bg)
            inner = sub[ndi.binary_erosion(filled, iterations=max(1, int(round(width)) + 1))]
            inner_c = _dominant(inner) if inner.size else None
            hollow = inner_c is None or _lab_de(inner_c, bg) < float(P["perceive.seg.hollow_bg_de"]) \
                or float(_dist(np.array([[[inner_c.r, inner_c.g, inner_c.b]]], np.uint8), bg)[0, 0]) <= fill_tol
            return Region(Box(x0, y0, w, h), kind, fill=None if hollow else inner_c, stroke=(stroke_c, max(1.0, round(width))),
                          radius=radius, mask=filled.copy(), bg=bg, conf=0.9,
                          meta={"ink_ratio": round(ink_ratio, 3), "corners": corners, "area": int(m.sum()), "hollow": bool(hollow)})
    # 5. image or textured rect. A component holding a large, solid flat core (a dialog surface
    #    touching the darkened cards it overlaps) is split first: photos have no such core.
    big_core = float(core_filled.sum()) >= float(P["perceive.seg.split_min_area"]) \
        and _solidity(core_filled) >= float(P["perceive.seg.split_core_solid"]) \
        and float(_dist(np.array([[[fill.r, fill.g, fill.b]]], np.uint8), bg)[0, 0]) > 0.5 * float(P["perceive.seg.bg_dist"])
    if not big_core:
        lc = _large_flat_core(sub, m, filled, bg, fill_tol)
        if lc is not None:
            fill, core, core_filled = lc
            core_frac = float(core_filled.sum()) / max(1.0, float(filled.sum()))
            big_core = True
    if _is_image(sub, m, w, h) and not big_core:
        mean = sub[m][:, :3].mean(axis=0)
        return Region(Box(x0, y0, w, h), "image", fill=Color.from_rgb(*mean), mask=filled.copy(), bg=bg, conf=0.7,
                      meta={"area": int(m.sum())})
    if max(w, h) <= int(P["perceive.seg.icon_max"]) and min(w, h) >= int(P["perceive.seg.icon_min"]):
        return Region(Box(x0, y0, w, h), "icon", fill=_glyph_color(sub, m, bg), mask=m.copy(), bg=bg, conf=0.5, meta={"fallback": True})
    # 5b. partly flat (overlapping siblings): emit the flat core as a region and hand the rest of
    #     the component back to the caller (meta['remainder']) to be scanned as separate components
    if (core_frac >= float(P["perceive.seg.split_frac"]) or big_core) and _dist(np.array([[[fill.r, fill.g, fill.b]]], np.uint8), bg)[0, 0] > 0.5 * float(P["perceive.seg.bg_dist"]):
        rb, m50 = _refine_box(sub, core_filled, fill, bg)
        rw, rh = rb[2] - rb[0], rb[3] - rb[1]
        box_mask = m50[rb[1]:rb[3], rb[0]:rb[2]]
        kind, radius, corners = _shape_kind(box_mask, rw, rh)
        rem = np.zeros_like(cmask)
        rem[y0:y1, x0:x1] = m & ~m50
        return Region(Box(x0 + rb[0], y0 + rb[1], rw, rh), kind, fill=fill, radius=radius, mask=box_mask.copy(), bg=bg, conf=0.6,
                      meta={"split": True, "core_frac": round(core_frac, 3), "corners": corners, "area": int(m.sum()), "remainder": rem})
    mean = sub[m][:, :3].mean(axis=0)
    kind, radius, corners = _shape_kind(filled, w, h)
    return Region(Box(x0, y0, w, h), kind, fill=Color.from_rgb(*mean), radius=radius, mask=filled.copy(), bg=bg, conf=0.4,
                  meta={"textured": True, "core_frac": round(core_frac, 3), "area": int(m.sum())})


# --------------------------------------------------------------------------- shadows
def _strip_mean(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Optional[np.ndarray]:
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return np.median(rgb[y0:y1, x0:x1, :3].reshape(-1, 3).astype(np.float32), axis=0)


def detect_shadow(rgb: np.ndarray, box: Box, bg: Color) -> Optional[Shadow]:
    """Soft dark halo just outside `box` that fades with distance -> Shadow, else None.

    Probes 1..shadow_probe px strips on each side (corners excluded); a shadow needs darkening
    >= shadow_min at 1px on the bottom side that decays below shadow_fade x peak within the probe.
    """
    D = int(P["perceive.seg.shadow_probe"])
    x0, y0, x1, y1 = box.as_int()
    bgv = np.array([bg.r, bg.g, bg.b], np.float32)
    inset = max(2, int(min(box.w, box.h) * 0.2))
    profiles: dict[str, list[float]] = {"top": [], "bottom": [], "left": [], "right": []}
    for d in range(1, D + 1):
        strips = {
            "top": _strip_mean(rgb, x0 + inset, y0 - d, x1 - inset, y0 - d + 1),
            "bottom": _strip_mean(rgb, x0 + inset, y1 + d - 1, x1 - inset, y1 + d),
            "left": _strip_mean(rgb, x0 - d, y0 + inset, x0 - d + 1, y1 - inset),
            "right": _strip_mean(rgb, x1 + d - 1, y0 + inset, x1 + d, y1 - inset),
        }
        for k, v in strips.items():
            profiles[k].append(float((bgv - v).mean()) if v is not None else 0.0)
    bottom = profiles["bottom"]
    peak = max(bottom[:3]) if bottom else 0.0
    if peak < float(P["perceive.seg.shadow_min"]):
        return None
    extent = {k: next((i for i, v in enumerate(p) if v < float(P["perceive.seg.shadow_fade"]) * peak), D) for k, p in profiles.items()}
    if extent["bottom"] >= D:  # never fades: an adjacent element, not a halo
        return None
    if any(v > peak * 1.5 for v in bottom):  # darkening grows away from the edge: neighbour element
        return None
    blur = float(max(1, extent["bottom"] + extent["top"]))
    dy = (extent["bottom"] - extent["top"]) / 2.0
    dx = (extent["right"] - extent["left"]) / 2.0
    alpha = min(0.6, max(0.05, peak / max(1.0, float(bgv.mean()))))
    return Shadow(color=Color(0, 0, 0, round(alpha, 3)), dx=round(dx, 1), dy=round(dy, 1), blur=round(blur, 1))


# --------------------------------------------------------------------------- scrim
_DS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fixtures", "design_systems", "material3.json")
_FALLBACK_COLORS = {"md.sys.color.surface": "#fef7ff", "md.sys.color.background": "#fef7ff", "md.sys.color.scrim": "#000000",
                    "md.sys.color.surface-container-lowest": "#ffffff", "md.sys.color.surface-container-low": "#f7f2fa",
                    "md.sys.color.surface-container": "#f3edf7", "md.sys.color.surface-container-high": "#ece6f0",
                    "md.sys.color.surface-container-highest": "#e6e0e9"}
_KNOWN: Optional[dict[str, Color]] = None


def known_colors() -> dict[str, Color]:
    """Design-system colour roles (``md.sys.color.*`` of the shipped Material 3 baseline) used as
    priors for un-blending a scrim; a small built-in surface set when the fixture is missing."""
    global _KNOWN
    if _KNOWN is None:
        vals = dict(_FALLBACK_COLORS)
        try:
            with open(_DS_PATH) as f:
                for t in json.load(f).get("tokens", []):
                    if t.get("category") == "color" and str(t.get("name", "")).startswith("md.sys.color.") \
                            and isinstance(t.get("value"), str) and t["value"].startswith("#"):
                        vals[t["name"]] = t["value"]
        except (OSError, ValueError):
            pass
        _KNOWN = {k: Color.from_hex(v) for k, v in vals.items()}
    return _KNOWN


def _surface_roles() -> list[Color]:
    """Colours a page background can be: surface / background / surface-container* roles."""
    out = []
    for k, c in known_colors().items():
        role = k.rsplit(".", 1)[-1]
        if (role.startswith("surface") and role not in ("surface-tint", "surface-variant")) or role == "background":
            out.append(c)
    return out


@dataclass
class Scrim:
    """A uniform translucent overlay: observed = (1 - alpha) * true + alpha * color."""
    color: Color
    alpha: float
    seen_bg: Color
    page_bg: Color
    clear: list[Color] = field(default_factory=list)
    explained: list[tuple[str, str]] = field(default_factory=list)  # (observed hex, true hex)

    def unblend_raw(self, c: Color) -> np.ndarray:
        s = np.array([self.color.r, self.color.g, self.color.b], np.float32)
        return (np.array([c.r, c.g, c.b], np.float32) - self.alpha * s) / max(1e-3, 1.0 - self.alpha)

    def unblend(self, c: Color) -> Color:
        v = np.clip(self.unblend_raw(c), 0, 255)
        return Color.from_rgb(*v, a=c.a)

    def in_gamut(self, c: Color) -> bool:
        """Can `c` be a colour darkened by this scrim (its un-blended value is a valid colour)?"""
        v = self.unblend_raw(c)
        tol = float(P["perceive.seg.scrim_gamut_tol"])
        return bool((v <= 255.0 + tol).all() and (v >= -tol).all())

    def overlay(self) -> Color:
        return Color(self.color.r, self.color.g, self.color.b, round(float(self.alpha), 3))


def _fit_alpha(seen: np.ndarray, true: np.ndarray, scrim: np.ndarray) -> Optional[tuple[float, float]]:
    """Blend factor a with seen = (1-a) true + a scrim (least squares over channels) and the max
    per-channel residual; None when the channels cannot determine it."""
    d = true - scrim
    ok = np.abs(d) > 20.0
    if ok.sum() < 2:
        return None
    a = float(((true - seen)[ok] * d[ok]).sum() / (d[ok] * d[ok]).sum())
    res = float(np.abs((1 - a) * true + a * scrim - seen).max())
    return a, res


def _color_samples(img: np.ndarray) -> list[tuple[Color, float]]:
    """Frequent colours of an image (quantised, subsampled) as (median colour, image fraction)."""
    sub = img[::2, ::2, :3].reshape(-1, 3)
    q = int(P["perceive.seg.quant"])
    k = (sub // q).astype(np.int32)
    keys = k[:, 0] * 65536 + k[:, 1] * 256 + k[:, 2]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    frac_min = float(P["perceive.seg.scrim_sample_frac"])
    n = float(sub.shape[0])
    out: list[tuple[Color, float]] = []
    for i in np.argsort(-counts):
        if counts[i] / n < frac_min:
            break
        px = sub[inv == i]
        out.append((Color.from_rgb(*np.median(px.astype(np.float32), axis=0)), float(counts[i]) / n))
    return out


def detect_scrim(img: np.ndarray, seen_bg: Optional[Color] = None) -> Optional[Scrim]:
    """Detect a dialog scrim: the page background (and other frequent colours) are one known
    colour each, darkened by the same blend factor toward the design system's scrim colour, while
    some colours (the dialog surface) cannot be such a darkening.

    Evidence required (all thresholds are ``perceive.seg.scrim_*`` params):
      * the border-ring background is a surface role blended with alpha in [alpha_min, alpha_max]
        (residual <= fit_tol) and covers >= min_bg_frac of the image (large, viewport-touching);
      * >= min_samples other frequent colours un-blend to a known colour role, or to a colour seen
        un-darkened on the page, within known_de;
      * colours that would un-blend above white cover >= clear_frac of the image (the clear surface).
    """
    if not int(P["perceive.seg.scrim_enable"]):
        return None
    seen_bg = seen_bg or _dominant(_ring(img, 2))
    if seen_bg is None:
        return None
    samples = _color_samples(img)
    bg_frac = sum(f for c, f in samples if c.delta_e(seen_bg) < 1.5)
    if bg_frac < float(P["perceive.seg.scrim_min_bg_frac"]):
        return None
    known = known_colors()
    scol = known.get("md.sys.color.scrim", Color(0, 0, 0))
    sv = np.array([scol.r, scol.g, scol.b], np.float32)
    bv = np.array([seen_bg.r, seen_bg.g, seen_bg.b], np.float32)
    amin, amax = float(P["perceive.seg.scrim_alpha_min"]), float(P["perceive.seg.scrim_alpha_max"])
    kde = float(P["perceive.seg.scrim_known_de"])
    best: Optional[tuple[float, Scrim]] = None
    tried: set[float] = set()
    for surf in _surface_roles():
        fit = _fit_alpha(bv, np.array([surf.r, surf.g, surf.b], np.float32), sv)
        if fit is None:
            continue
        a, res = fit
        if not (amin <= a <= amax) or res > float(P["perceive.seg.scrim_fit_tol"]) or round(a, 3) in tried:
            continue
        tried.add(round(a, 3))
        sc = Scrim(color=scol, alpha=a, seen_bg=seen_bg, page_bg=surf)
        clear = [(c, f) for c, f in samples if not sc.in_gamut(c)]
        clear_frac = sum(f for _, f in clear)
        if clear_frac < float(P["perceive.seg.scrim_clear_frac"]):
            continue
        refs = list(known.values()) + [c for c, _ in clear]
        expl, area = [], 0.0
        for c, f in samples:
            if c.delta_e(seen_bg) < 1.5 or not sc.in_gamut(c):
                continue
            u = sc.unblend(c)
            if min(u.delta_e(r) for r in refs) <= kde:
                expl.append((c.hex(), u.hex()))
                area += f
        if len(expl) < int(P["perceive.seg.scrim_min_samples"]):
            continue
        sc.clear = [c for c, _ in clear]
        sc.explained = expl
        score = len(expl) + area - res / 10.0
        if best is None or score > best[0]:
            best = (score, sc)
    return None if best is None else best[1]


def _rgb_close(a: Color, b: Color, tol: float) -> bool:
    return abs(a.r - b.r) <= tol and abs(a.g - b.g) <= tol and abs(a.b - b.b) <= tol


def _annotate_scrim(regions: list[Region], scrim: Scrim, W: int, H: int) -> list[Region]:
    """Split regions into clear (on top of the scrim: the dialog and everything inside it) and
    under-scrim ones (true colours recovered by un-blending), and add the full-page scrim region."""
    tol = float(P["perceive.seg.fill_tol"])
    seen = scrim.seen_bg
    clear_boxes: list[Box] = []
    for r in sorted(regions, key=lambda r: -r.box.area):
        paint = r.fill if r.fill is not None else (r.stroke[0] if r.stroke else None)
        inside = any(r.box.intersect(cb).area >= 0.9 * max(1.0, r.box.area) for cb in clear_boxes)
        on_clear = not _rgb_close(r.bg, seen, tol) and not scrim.in_gamut(r.bg)
        if inside or on_clear or (paint is not None and not scrim.in_gamut(paint)):
            r.meta["scrim_clear"] = True
            if r.kind in ("rect", "pill", "ellipse") and r.fill is not None and not inside:
                clear_boxes.append(r.box)
            continue
        r.meta["under_scrim"] = True
        r.meta["true_bg"] = scrim.unblend(r.bg).hex()
        if r.fill is not None:
            r.meta["true_fill"] = scrim.unblend(r.fill).hex()
        if r.stroke is not None:
            r.meta["true_stroke"] = scrim.unblend(r.stroke[0]).hex()
        if int(P["perceive.seg.scrim_unblend"]):
            if r.fill is not None:
                r.meta["blended_fill"] = r.fill.hex()
                r.fill = scrim.unblend(r.fill)
            if r.stroke is not None:
                r.stroke = (scrim.unblend(r.stroke[0]), r.stroke[1])
            r.bg = scrim.unblend(r.bg)
    sreg = Region(Box(0, 0, W, H), "rect", fill=scrim.overlay(), bg=scrim.page_bg, conf=0.8, depth=0,
                  meta={"scrim": True, "alpha": round(float(scrim.alpha), 3), "scrim_color": scrim.color.hex(),
                        "page_bg": scrim.page_bg.hex(), "seen_bg": scrim.seen_bg.hex(),
                        "explained": scrim.explained[:8], "unblended": bool(int(P["perceive.seg.scrim_unblend"]))})
    return [sreg] + regions


def apply_scrim(root) -> bool:
    """Re-stack a perceived tree (``dt.perceive.hierarchy.build_tree`` output) around its scrim node.

    Containment nesting puts every node inside the full-page scrim, i.e. painted above it. This
    moves the under-scrim subtrees out of it, below it, with their true (un-blended) fills, strokes
    and text colours, keeps the clear nodes (dialog) above it, and sets the root fill to the true
    page background. Returns False when the tree has no scrim node. Pixel output is unchanged
    (blend(true) == observed); the IR becomes faithful to the source design."""
    sn = next((c for c in root.children if c.meta.get("scrim")), None)
    if sn is None:
        return False
    from dt.ir import Fill, Stroke  # local: keep the module's import surface small
    sc = Scrim(color=Color.from_hex(sn.meta.get("scrim_color", "#000000")), alpha=float(sn.meta["alpha"]),
               seen_bg=Color.from_hex(sn.meta.get("seen_bg", "#000000")), page_bg=Color.from_hex(sn.meta["page_bg"]))
    pre_unblended = bool(sn.meta.get("unblended"))
    clear_nodes, under_nodes = [], []

    def is_clear(n) -> bool:
        return bool(n.meta.get("scrim_clear"))

    def lift_clear(n) -> None:  # clear nodes nested inside an under-scrim node move above the scrim
        keep = []
        for c in n.children:
            (clear_nodes if is_clear(c) else keep).append(c)
        n.children = keep
        for c in keep:
            lift_clear(c)

    def unblend(n) -> None:
        n.meta["under_scrim"] = True
        if not pre_unblended:
            if n.fills and n.meta.get("true_fill"):
                n.fills = [Fill.solid(Color.from_hex(n.meta["true_fill"]))] + n.fills[1:]
            if n.strokes and n.meta.get("true_stroke"):
                s0 = n.strokes[0]
                n.strokes = [Stroke(color=Color.from_hex(n.meta["true_stroke"]), width=s0.width, align=s0.align)] + n.strokes[1:]
        if n.type == "text" and n.text_style is not None and not n.meta.get("true_text_color"):
            n.meta["true_text_color"] = sc.unblend(n.text_style.color).hex()
            n.meta["seen_text_color"] = n.text_style.color.hex()
            n.text_style.color = sc.unblend(n.text_style.color)
        for c in n.children:
            unblend(c)

    for c in sn.children:
        (clear_nodes if is_clear(c) else under_nodes).append(c)
    for n in under_nodes:
        lift_clear(n)
        unblend(n)
    sn.children = []
    sn.type = "rect"
    sn.layout = None
    root.fills = [Fill.solid(sc.page_bg)]
    others = [c for c in root.children if c is not sn]
    root.children = others + under_nodes + [sn] + clear_nodes
    return True


# --------------------------------------------------------------------------- driver
def _find_in(img: np.ndarray, allowed: Optional[np.ndarray], origin: tuple[int, int], bg: Color, depth: int,
             text_mask: np.ndarray, out: list[Region]) -> None:
    """Scan `img` (a crop whose top-left is `origin`) for components differing from `bg`."""
    ink = _dist(img, bg) > float(P["perceive.seg.bg_dist"])
    if allowed is not None:
        ink &= allowed
    if not ink.any():
        return
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    min_area = int(P["perceive.seg.min_area"])
    ox, oy = origin
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        cmask = lab[y0:y1, x0:x1] == i
        reg = classify_component(img[y0:y1, x0:x1], cmask, bg, text_mask[y0:y1, x0:x1])
        if reg is None:
            continue
        reg.box = reg.box.translate(ox + x0, oy + y0)
        reg.depth = depth
        out.append(reg)
        rem = reg.meta.pop("remainder", None)
        if rem is not None and rem.any() and rem.sum() < cmask.sum():  # the non-flat rest of a split component: same bg, same depth
            _find_in(img[y0:y1, x0:x1], rem, (ox + x0, oy + y0), bg, depth, text_mask[y0:y1, x0:x1], out)
        # textured regions have no real fill: scanning their inside against the mean colour only
        # produces concentric "onion" layers
        if reg.kind in ("rect", "pill", "ellipse") and reg.fill is not None and not reg.meta.get("textured") \
                and depth < int(P["perceive.seg.max_depth"]) \
                and min(reg.box.w, reg.box.h) >= int(P["perceive.seg.recurse_min_side"]) and reg.mask is not None:
            inset = int(P["perceive.seg.recurse_inset"]) + (int(reg.stroke[1]) if reg.stroke else 0)
            inner = ndi.binary_erosion(reg.mask, iterations=inset) if inset > 0 else reg.mask
            if inner.any():
                bx, by, bw, bh = (int(v) for v in (reg.box.x - ox, reg.box.y - oy, reg.box.w, reg.box.h))
                sub = img[by:by + bh, bx:bx + bw]
                _find_in(sub, inner[:sub.shape[0], :sub.shape[1]], (ox + bx, oy + by), reg.fill, depth + 1,
                         text_mask[by:by + bh, bx:bx + bw], out)


def find_regions(rgb: np.ndarray, text_boxes: list[Box], bg: Optional[Color] = None) -> list[Region]:
    """Segment the non-text visual regions of an RGB screenshot.

    `text_boxes` (from OCR) are painted out first so glyphs never become regions. Returns
    regions in detection order (outer before inner), with shadows probed on the original image.
    """
    H, W = rgb.shape[:2]
    img = mask_text(rgb, text_boxes)
    seen = _seen_background(img)
    scrim = detect_scrim(img, seen)
    # under a scrim everything is segmented against the observed (darkened) background
    bg = seen if scrim is not None else (bg or seen)
    text_mask = np.zeros((H, W), bool)
    for b in text_boxes:
        x0, y0, x1, y1 = b.expand(int(P["perceive.seg.text_pad"])).as_int()
        text_mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    out: list[Region] = []
    _find_in(img, None, (0, 0), bg, 0, text_mask, out)
    out = _dedupe(out)
    for r in out:
        if r.kind in ("rect", "pill", "ellipse", "image") and min(r.box.w, r.box.h) >= 8:
            sh = detect_shadow(rgb, r.box, r.bg)
            if sh is not None:
                r.has_shadow, r.shadow = True, sh
    out = _drop_shadow_fragments(out)
    if scrim is not None:
        out = _annotate_scrim(out, scrim, W, H)
    return out


def _dedupe(regions: list[Region]) -> list[Region]:
    """Collapse regions found twice (a container whose fill is within bg_dist of the page bg is
    not 'ink', so its children are found at the top level AND again inside it); the deeper one
    was measured against the right background."""
    thr = float(P["perceive.seg.dedupe_iou"])
    keep: list[Region] = []
    for r in regions:
        j = next((i for i, k in enumerate(keep) if abs(k.box.y - r.box.y) <= 2 and k.box.iou(r.box) >= thr), None)
        if j is None:
            keep.append(r)
        elif r.depth >= keep[j].depth:
            keep[j] = r
    return keep


def _drop_shadow_fragments(regions: list[Region]) -> list[Region]:
    """Remove low-contrast crumbs lying in the halo zone of a shadowed region."""
    probe = int(P["perceive.seg.shadow_probe"])
    lim = float(P["perceive.seg.fragment_dist"]) * float(P["perceive.seg.bg_dist"])
    halos = [r.box.expand(probe) for r in regions if r.has_shadow]
    keep: list[Region] = []
    for r in regions:
        low = r.fill is not None and _dist(np.array([[[r.fill.r, r.fill.g, r.fill.b]]], np.uint8), r.bg)[0, 0] <= lim
        crumb = r.kind == "divider" or r.meta.get("fallback") or r.meta.get("textured")
        if low and crumb and any(h.contains(r.box) for h in halos) and not r.has_shadow:
            continue
        keep.append(r)
    return keep
