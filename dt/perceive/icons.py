"""Icon identification by template matching against a rendered Material Symbols atlas.

Programmatic, no ML: every one of the ~4300 Material Symbols is rendered once (per FILL axis
value) with the real font, its ink mask is normalised into a fixed canvas, and a query glyph
(the ink inside a perceived icon box) is scored against all of them by blurred-mask IoU.

    atlas = load_atlas()                       # cached npz under fixtures/icons/
    hits  = identify(rgb_crop, bg_color)       # [(name, score, fill)] best first
    em    = em_box_for(name, ink_box, fill)    # expand the tight ink box to the 24px-style em box

Build (one-off, ~20 s): `python -m dt.perceive.icons build`
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from dt.ir import Box, Color
from dt.params import P, register

ICON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "icons"))
CODEPOINTS = os.path.join(ICON_DIR, "MaterialSymbolsOutlined.codepoints")
VARIABLE_FONT = os.path.join(ICON_DIR, "MaterialSymbolsOutlined-variable.woff2")
CANVAS = 32  # normalised mask size

register("perceive.icons.opsz", 24, "optical-size axis used for the atlas, re-ranking and rendering (Google Fonts' static Material Symbols == opsz 24)", (20, 48))
register("perceive.icons.render_px", 64, "font-size used to render atlas glyphs", (32, 128))
register("perceive.icons.ink_thr", 0.5, "fraction of max contrast counted as ink when binarising a query", (0.3, 0.7))
register("perceive.icons.blur", 1.0, "gaussian sigma (canvas px) applied to masks before IoU", (0.0, 2.5))
register("perceive.icons.aspect_tol", 0.35, "max |log aspect ratio diff| for a candidate", (0.1, 1.0))
register("perceive.icons.min_score", 0.55, "minimum IoU score to accept an identification", (0.3, 0.9))
register("perceive.icons.w_sdf", 0.5, "weight of the distance-field similarity vs mask IoU", (0.0, 1.0))
register("perceive.icons.prior_boost", 0.04, "score boost for icons in fixtures/icons/common.txt", (0.0, 0.15))
register("perceive.icons.rerank_k", 16, "unique glyph names re-ranked by rendering (stage 2)", (2, 40))
register("perceive.icons.w_rerank", 0.65, "weight of render ΔE similarity vs atlas score in the final ranking", (0.0, 1.0))
register("perceive.icons.standard_sizes", [18, 20, 24, 36, 40, 48], "icon em sizes preferred when the estimate is within snap_tol")
register("perceive.icons.snap_tol", 1.25, "max |em - standard size| (px) to snap to a standard icon size", (0.0, 3.0))
register("perceive.icons.pen_uncommon", 0.4, "ΔE penalty for a glyph not in fixtures/icons/common.txt (breaks near-ties toward everyday icons)", (0.0, 2.0))
register("perceive.icons.pen_nonstandard", 0.3, "ΔE penalty for an em size not in standard_sizes", (0.0, 2.0))
register("perceive.icons.good_de", 0.0, "skip the fine pose search when the first render already reproduces the crop this well (mean ΔE)", (0.0, 3.0))
register("perceive.icons.pose_names", 2, "names carried into the pose search (stage 3)", (1, 5))
register("perceive.icons.pose_size_reach", 2.0, "standard sizes within this many px of the estimate are tried", (0.0, 4.0))
register("perceive.icons.accept_de", 2.5, "accept an identification when its best pose reproduces the crop with mean ΔE2000 <= this", (0.5, 6.0))
register("perceive.icons.crop_pad", 4, "px of context around an icon box used for identification", (1, 8))
register("perceive.icons.mis_cov", 0.6, "a glyph pixel is mismatched when target and render ink coverage (0..1 of the fg/bg contrast) differ by more than this (anti-aliasing differences stay below it)", (0.3, 0.9))
register("perceive.icons.ink_rel", 0.5, "coverage counted as glyph ink when normalising the mismatch (target or render)", (0.1, 0.9))
register("perceive.icons.accept_mis", 0.09, "accept an identification only when its glyph mismatch per outline pixel (glyph_mismatch, mis_e) is at most this: mean crop ΔE alone is diluted by background and passed a hexagon as brightness_1 (0.10) and 7px bullets as circle_notifications (0.105); correct real-screen icons measured <= 0.068", (0.0, 1.0))
register("perceive.icons.min_ink_px", 6, "min ink bbox side (px) for identification to be attempted", (3, 12))


# ----------------------------------------------------------------------------- normalisation
def ink_mask_from_crop(crop: np.ndarray, bg: Optional[Color] = None) -> tuple[np.ndarray, Box]:
    """Soft ink coverage of a glyph crop against its background. Returns (float mask in [0,1], ink bbox in crop coords)."""
    if crop.size == 0:
        return np.zeros((0, 0), np.float32), Box()
    px = crop[..., :3].astype(np.float32)
    if bg is None:
        # background = most common colour along the border
        border = np.concatenate([px[0], px[-1], px[:, 0], px[:, -1]], axis=0)
        q = (border // 8).astype(np.int32)
        keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
        u, c = np.unique(keys, return_counts=True)
        bgv = border[keys == u[np.argmax(c)]].mean(axis=0)
    else:
        bgv = np.array([bg.r, bg.g, bg.b], np.float32)
    dist = np.sqrt(((px - bgv) ** 2).sum(axis=-1))
    mx = float(dist.max())
    if mx < 24:  # nothing drawn
        return np.zeros(crop.shape[:2], np.float32), Box()
    # soft coverage in [0,1]: keeps 1-2px anti-aliased strokes that a hard threshold would erase
    cov = np.clip(dist / mx, 0.0, 1.0).astype(np.float32)
    ink = cov > float(P["perceive.icons.ink_thr"]) * 0.5
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return cov, Box()
    cov[~ink & (cov < 0.15)] = 0.0
    return cov, Box(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def normalise(mask: np.ndarray, bbox: Box, canvas: int = CANVAS) -> np.ndarray:
    """Crop the ink bbox, scale so the larger side == canvas, centre on a canvas x canvas float image."""
    if bbox.w <= 0 or bbox.h <= 0:
        return np.zeros((canvas, canvas), np.float32)
    x0, y0, x1, y1 = bbox.as_int()
    sub = mask[y0:y1, x0:x1].astype(np.float32)
    h, w = sub.shape
    s = canvas / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((canvas, canvas), np.float32)
    ox, oy = (canvas - nw) // 2, (canvas - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


def _sdf(m: np.ndarray) -> np.ndarray:
    """Signed-ish distance field of a soft mask: distance to ink outside, negative distance to background inside."""
    ink = (m > 0.5).astype(np.uint8)
    if ink.sum() == 0:
        return np.full(m.shape, float(CANVAS), np.float32)
    d_out = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    return (d_out - d_in).astype(np.float32)


def _blur(m: np.ndarray) -> np.ndarray:
    s = float(P["perceive.icons.blur"])
    if s <= 0:
        return m
    return cv2.GaussianBlur(m, (0, 0), s)


# ----------------------------------------------------------------------------- atlas
@dataclass
class Atlas:
    names: list[str]
    fills: np.ndarray  # (N,) int8 FILL axis value per entry
    masks: np.ndarray  # (N, CANVAS, CANVAS) float32 normalised (unblurred)
    aspect: np.ndarray  # (N,) ink w/h
    ink_rel: np.ndarray  # (N, 4) ink bbox relative to the em box: x/em, y/em, w/em, h/em
    render_px: int

    _blurred: Optional[np.ndarray] = None
    _sdf: Optional[np.ndarray] = None

    def blurred(self) -> np.ndarray:
        if self._blurred is None:
            self._blurred = np.stack([_blur(m) for m in self.masks]).astype(np.float32)
        return self._blurred

    def sdf(self) -> np.ndarray:
        if self._sdf is None:
            self._sdf = np.stack([_sdf(m) for m in self.masks]).astype(np.float32)
        return self._sdf


def _read_names() -> list[str]:
    names = []
    with open(CODEPOINTS) as f:
        for line in f:
            parts = line.split()
            if parts:
                names.append(parts[0])
    return names


def atlas_path(fill: int) -> str:
    return os.path.join(ICON_DIR, f"atlas_fill{fill}_{CANVAS}.npz")


def build_atlas(fill: int = 0, weight: int = 400, out_path: Optional[str] = None, names: Optional[list[str]] = None) -> Atlas:
    """Render every glyph with the variable font via Chrome and store normalised masks."""
    from dt.render.screenshot import render_url

    names = names or _read_names()
    px = int(P["perceive.icons.render_px"])
    cell = px + 16
    cols = 40
    rows = (len(names) + cols - 1) // cols
    font_url = "file://" + VARIABLE_FONT
    spans = "".join(
        f'<span class=g id="g{i}" style="left:{(i % cols) * cell}px;top:{(i // cols) * cell}px">{n}</span>'
        for i, n in enumerate(names)
    )
    html = f"""<!doctype html><meta charset=utf-8><style>
@font-face{{font-family:'MSOV';src:url('{font_url}') format('woff2');}}
html,body{{margin:0;background:#fff;width:{cols * cell}px;height:{rows * cell}px}}
.g{{position:absolute;width:{cell}px;height:{cell}px;display:flex;align-items:center;justify-content:center;
 font-family:'MSOV';font-size:{px}px;line-height:1;color:#000;-webkit-font-smoothing:antialiased;
 font-variation-settings:'FILL' {fill},'wght' {weight},'GRAD' 0,'opsz' {int(P["perceive.icons.opsz"])};}}
</style>{spans}"""
    tmp = os.path.join(ICON_DIR, f"_atlas_{fill}.html")
    with open(tmp, "w") as f:
        f.write(html)
    script = """(() => { const out=[]; document.fonts.ready; for (const el of document.querySelectorAll('.g')) { const r=el.getBoundingClientRect(); out.push([r.x, r.y, r.width, r.height]); } return {ok: document.fonts.check("64px 'MSOV'"), boxes: out}; })()"""
    masks, aspects, rels = [], [], []
    W, H = cols * cell, rows * cell
    # chunk vertically to keep screenshots manageable
    chunk_rows = 50
    for r0 in range(0, rows, chunk_rows):
        r1 = min(rows, r0 + chunk_rows)
        sub_names = names[r0 * cols:r1 * cols]
        sub_spans = "".join(
            f'<span class=g style="left:{(i % cols) * cell}px;top:{(i // cols) * cell}px">{n}</span>' for i, n in enumerate(sub_names)
        )
        sub_rows = r1 - r0
        sub_html = html.replace(spans, sub_spans).replace(f"height:{rows * cell}px", f"height:{sub_rows * cell}px")
        with open(tmp, "w") as f:
            f.write(sub_html)
        img, res = render_url("file://" + tmp, W, sub_rows * cell, wait_ms=400, script=script, wait_until="load")
        if not res or not res.get("ok"):
            raise RuntimeError("Material Symbols variable font did not load for atlas build")
        for i, (x, y, w, h) in enumerate(res["boxes"]):
            x0, y0 = int(round(x)), int(round(y))
            crop = img[y0:y0 + int(h), x0:x0 + int(w)]
            mask, bb = ink_mask_from_crop(crop, Color(255, 255, 255))
            masks.append(normalise(mask, bb))
            aspects.append((bb.w / bb.h) if bb.h > 0 else 1.0)
            # em box = the px square centred in the cell
            ex, ey = (w - px) / 2, (h - px) / 2
            rels.append([(bb.x - ex) / px, (bb.y - ey) / px, bb.w / px, bb.h / px] if bb.w > 0 else [0, 0, 0, 0])
    try:
        os.remove(tmp)
    except OSError:
        pass
    atlas = Atlas(names=list(names), fills=np.full(len(names), fill, np.int8), masks=np.stack(masks).astype(np.float32),
                  aspect=np.array(aspects, np.float32), ink_rel=np.array(rels, np.float32), render_px=px)
    out_path = out_path or atlas_path(fill)
    np.savez_compressed(out_path, names=np.array(atlas.names), fills=atlas.fills, masks=(atlas.masks * 255).astype(np.uint8),
                        aspect=atlas.aspect, ink_rel=atlas.ink_rel, render_px=np.array([px]))
    return atlas


_ATLAS_CACHE: dict[str, Atlas] = {}
_COMMON: Optional[set] = None


def common_icons() -> set:
    global _COMMON
    if _COMMON is None:
        _COMMON = set()
        p = os.path.join(ICON_DIR, "common.txt")
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("#"):
                    continue
                _COMMON.update(line.split())
    return _COMMON


def load_atlas(fills: tuple[int, ...] = (0, 1)) -> Atlas:
    """Load (and concatenate) cached atlases for the given FILL values; builds missing ones."""
    key = ",".join(map(str, fills))
    if key in _ATLAS_CACHE:
        return _ATLAS_CACHE[key]
    parts = []
    for fill in fills:
        p = atlas_path(fill)
        if not os.path.exists(p):
            build_atlas(fill)
        d = np.load(p, allow_pickle=False)
        parts.append(Atlas(names=[str(n) for n in d["names"]], fills=d["fills"], masks=d["masks"].astype(np.float32) / 255.0,
                           aspect=d["aspect"], ink_rel=d["ink_rel"], render_px=int(d["render_px"][0])))
    atlas = Atlas(
        names=sum((a.names for a in parts), []),
        fills=np.concatenate([a.fills for a in parts]),
        masks=np.concatenate([a.masks for a in parts]),
        aspect=np.concatenate([a.aspect for a in parts]),
        ink_rel=np.concatenate([a.ink_rel for a in parts]),
        render_px=parts[0].render_px,
    )
    _ATLAS_CACHE[key] = atlas
    return atlas


# ----------------------------------------------------------------------------- matching
@dataclass
class IconHit:
    name: str
    score: float
    fill: int
    aspect: float


def match_mask(mask: np.ndarray, bbox: Box, atlas: Optional[Atlas] = None, top_k: int = 5) -> list[IconHit]:
    """Score a binarised query glyph against the atlas. Returns top_k hits (best first)."""
    atlas = atlas or load_atlas()
    if bbox.w < P["perceive.icons.min_ink_px"] or bbox.h < P["perceive.icons.min_ink_px"]:
        return []
    q = _blur(normalise(mask, bbox))
    qa = bbox.w / bbox.h
    tol = float(P["perceive.icons.aspect_tol"])
    cand = np.where(np.abs(np.log(np.maximum(atlas.aspect, 1e-3) / qa)) <= tol)[0]
    if len(cand) == 0:
        cand = np.arange(len(atlas.names))
    A = atlas.blurred()[cand]  # (C, H, W)
    inter = np.minimum(A, q[None]).sum(axis=(1, 2))
    union = np.maximum(A, q[None]).sum(axis=(1, 2)) + 1e-6
    iou = inter / union
    # distance-field similarity: robust for thin strokes where mask overlap is tiny
    qd = _sdf(normalise(mask, bbox))
    D = atlas.sdf()[cand]
    sdf_sim = 1.0 - np.abs(D - qd[None]).mean(axis=(1, 2)) / (CANVAS * 0.5)
    w = float(P["perceive.icons.w_sdf"])
    score = (1 - w) * iou + w * np.clip(sdf_sim, 0, 1)
    boost = float(P["perceive.icons.prior_boost"])
    if boost > 0:
        com = common_icons()
        score = score + boost * np.array([1.0 if atlas.names[c] in com else 0.0 for c in cand], np.float32)
    order = np.argsort(-score)[:top_k]
    return [IconHit(atlas.names[cand[i]], float(score[i]), int(atlas.fills[cand[i]]), float(atlas.aspect[cand[i]])) for i in order]


def identify(crop: np.ndarray, bg: Optional[Color] = None, atlas: Optional[Atlas] = None, top_k: int = 5) -> list[IconHit]:
    """Identify the icon drawn in `crop` (RGB array of the icon region, a little padding is fine)."""
    mask, bb = ink_mask_from_crop(crop, bg)
    if bb.w <= 0:
        return []
    return match_mask(mask, bb, atlas, top_k)


def best(crop: np.ndarray, bg: Optional[Color] = None, atlas: Optional[Atlas] = None) -> Optional[IconHit]:
    hits = identify(crop, bg, atlas, top_k=1)
    if hits and hits[0].score >= P["perceive.icons.min_score"]:
        return hits[0]
    return None


def em_box_for(name: str, ink_box: Box, fill: int = 0, atlas: Optional[Atlas] = None) -> Box:
    """Given the identified glyph and its tight ink box (absolute coords), return the em box the
    renderer needs (so an `icon` node's box reproduces the glyph at the same place and size)."""
    atlas = atlas or load_atlas()
    try:
        idx = next(i for i, (n, f) in enumerate(zip(atlas.names, atlas.fills)) if n == name and int(f) == fill)
    except StopIteration:
        return ink_box
    rx, ry, rw, rh = (float(v) for v in atlas.ink_rel[idx])
    if rw <= 0 or rh <= 0:
        return ink_box
    em = (ink_box.w / rw + ink_box.h / rh) / 2.0
    return snap_em(Box(ink_box.x - rx * em, ink_box.y - ry * em, em, em), rx, ry, ink_box)


def snap_em(em_box: Box, rx: float, ry: float, ink_box: Box) -> Box:
    """Icons are drawn at whole-pixel sizes on whole-pixel positions; a fractional em changes the glyph's
    anti-aliasing. Snap the size (to a standard M3 size when close, else the nearest integer), re-derive
    the origin from the ink box at that size, and round it."""
    em = em_box.w
    std = [float(v) for v in P["perceive.icons.standard_sizes"]]
    near = min(std, key=lambda v: abs(v - em)) if std else em
    size = near if abs(near - em) <= float(P["perceive.icons.snap_tol"]) else float(round(em))
    size = max(1.0, size)
    # keep the ink centred where it was observed
    cx = ink_box.x - rx * size
    cy = ink_box.y - ry * size
    return Box(float(round(cx)), float(round(cy)), size, size)


# ----------------------------------------------------------------------------- analysis by synthesis
@dataclass
class IconQuery:
    crop: np.ndarray  # RGB crop containing the glyph (any padding)
    bg: Color
    fg: Color
    hits: list[IconHit]  # atlas candidates (from match_mask)
    ink_box: Optional[Box] = None  # ink bbox in crop coords (computed if None)


@dataclass
class Pose:
    """A fully specified candidate rendering: glyph + em box in CROP coordinates."""
    name: str
    fill: int
    box: Box
    de: float = float("inf")  # mean ΔE2000 of the rendered cell vs the crop (lower is better)
    mis: float = 1.0  # glyph mismatch (see glyph_mismatch): disagreeing pixels per glyph-ink pixel
    mis_e: float = 1.0  # disagreeing pixels per glyph-outline pixel (stricter for solid shapes)


def render_poses(queries: list[IconQuery], poses: list[list[Pose]], weight: int = 400) -> None:
    """Render every pose of every query on ONE page (each cell = the query's crop size, filled with its
    background) and set pose.de = mean ΔE2000 between the cell and the crop. The renderer is the judge."""
    from dt.render.screenshot import render_url
    cells = []  # (qi, pi, x, y, w, h)
    cur_y, max_w = 0, 0
    for qi, q in enumerate(queries):
        h, w = q.crop.shape[:2]
        cur_x = 0
        for pi, _ in enumerate(poses[qi]):
            cells.append((qi, pi, cur_x, cur_y, w, h))
            cur_x += w + 2
        if poses[qi]:
            max_w = max(max_w, cur_x)
            cur_y += h + 2
    if not cells:
        return
    font_url = "file://" + VARIABLE_FONT
    opsz = int(P["perceive.icons.opsz"])
    divs = []
    for (qi, pi, x, y, w, h) in cells:
        q, ps = queries[qi], poses[qi][pi]
        b = ps.box
        divs.append(
            f'<div class=c style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;background:{q.bg.css()}">'
            f'<span style="left:{b.x:.2f}px;top:{b.y:.2f}px;width:{b.w:.2f}px;height:{b.h:.2f}px;font-size:{min(b.w, b.h):.2f}px;'
            f"color:{q.fg.css()};font-variation-settings:'FILL' {ps.fill},'wght' {weight},'GRAD' 0,'opsz' {opsz}\">{ps.name}</span></div>"
        )
    html = f"""<!doctype html><meta charset=utf-8><style>
@font-face{{font-family:'MSOV';src:url('{font_url}') format('woff2');}}
html,body{{margin:0;background:#fff;width:{max_w}px;height:{cur_y}px}}
.c{{position:absolute;overflow:hidden}}
.c span{{position:absolute;display:flex;align-items:center;justify-content:center;font-family:'MSOV';line-height:1;-webkit-font-smoothing:antialiased}}
</style>{''.join(divs)}"""
    # a private temp file, not the tracked fixtures dir (a crash used to leave _poses_<pid>.html in the repo)
    fd, tmp = tempfile.mkstemp(prefix="dt_poses_", suffix=".html")
    with os.fdopen(fd, "w") as f:
        f.write(html)
    try:
        img, _ = render_url("file://" + tmp, max(1, max_w), max(1, cur_y), wait_ms=0, wait_until="load",
                            script="Promise.race([document.fonts.load(\"24px 'MSOV'\", 'home'), new Promise(r => setTimeout(r, 5000))])"
                                   ".then(() => Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 5000))]))"
                                   ".then(() => Promise.race([new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))), new Promise(r => setTimeout(r, 2000))]))")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    from dt import accel  # one batched CIEDE2000 call for every cell (GPU via MLX on Apple Silicon)
    covs = [_coverage(q.crop, q.bg, q.fg) for q in queries]
    tgt, ren, owners = [], [], []
    for (qi, pi, x, y, w, h) in cells:
        reg = img[y:y + h, x:x + w, :3]
        if reg.shape[:2] != queries[qi].crop.shape[:2]:
            continue
        tgt.append(queries[qi].crop[..., :3].reshape(-1, 3))
        ren.append(reg.reshape(-1, 3))
        owners.append((qi, pi, x, y, w, h))
        poses[qi][pi].mis, poses[qi][pi].mis_e = glyph_mismatch(covs[qi], _coverage(reg, queries[qi].bg, queries[qi].fg))
    if owners:
        de = accel.delta_e2000(np.concatenate(tgt)[None], np.concatenate(ren)[None])[0].astype(np.float64)
        off = 0
        for (qi, pi, x, y, w, h), t in zip(owners, tgt):
            n = len(t)
            poses[qi][pi].de = float(de[off:off + n].mean())
            off += n


def _coverage(rgb: np.ndarray, bg: Color, fg: Color) -> np.ndarray:
    """Per-pixel glyph coverage in [0, 1]: distance from the background relative to the fg/bg contrast."""
    px = rgb[..., :3].astype(np.float32)
    bgv = np.array([bg.r, bg.g, bg.b], np.float32)
    d = np.sqrt(((px - bgv) ** 2).sum(-1))
    contrast = float(np.sqrt(((np.array([fg.r, fg.g, fg.b], np.float32) - bgv) ** 2).sum()))
    if contrast < 24:  # fg unknown / equal to bg: fall back to the crop's own maximum contrast
        contrast = float(d.max()) if d.size else 0.0
    return np.clip(d / max(contrast, 1.0), 0.0, 1.0)


def glyph_mismatch(cov_a: np.ndarray, cov_b: np.ndarray) -> tuple[float, float]:
    """Shape error between two glyph coverage maps: pixels where one image has ink and the other has none
    (coverage differs by > mis_cov; anti-aliasing / resampling of the same glyph stays below it), counted
    (1) per glyph-ink pixel and (2) per glyph-outline pixel. Both are contrast- and background-independent;
    the outline form does not dilute a wrong silhouette of a solid shape (hexagon vs circle) by its area."""
    if cov_a.shape != cov_b.shape:
        return 1.0, 1.0
    thr = float(P["perceive.icons.ink_rel"])
    ink_a, ink_b = cov_a > thr, cov_b > thr
    ink = ink_a | ink_b
    n = int(ink.sum())
    if n == 0:
        return 0.0, 0.0
    bad = int((np.abs(cov_a - cov_b) > float(P["perceive.icons.mis_cov"])).sum())
    k = np.ones((3, 3), np.uint8)
    edge = 0
    for m in (ink_a, ink_b):
        m8 = m.astype(np.uint8)
        edge += int((m8 - cv2.erode(m8, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)).sum())
    return float(bad / n), float(bad / max(1, edge / 2))


def _size_variants(name: str, fill: int, base: Box) -> list[Pose]:
    """The estimate and +-1px sizes, each centred where the estimate was (no offset jitter)."""
    out = []
    for s in (base.w - 1, base.w, base.w + 1):
        if s >= 4:
            out.append(Pose(name, fill, Box(round(base.cx - s / 2), round(base.cy - s / 2), s, s)))
    return out


def _pose_variants(name: str, fill: int, base: Box) -> list[Pose]:
    """Integer sizes around the estimate (incl. standard sizes within reach) x integer offsets in [-1, 1]^2."""
    est = base.w
    sizes = {float(round(est)), float(round(est)) - 1, float(round(est)) + 1}
    for v in P["perceive.icons.standard_sizes"]:
        if abs(float(v) - est) <= float(P["perceive.icons.pose_size_reach"]):
            sizes.add(float(v))
    out = []
    for s in sorted(x for x in sizes if x >= 4):
        # keep the em box centred where the estimate put it, then jitter by whole pixels
        cx, cy = base.cx, base.cy
        x0, y0 = round(cx - s / 2), round(cy - s / 2)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out.append(Pose(name, fill, Box(x0 + dx, y0 + dy, s, s)))
    return out


def identify_many(crops: list[tuple[np.ndarray, Color, Color]], atlas: Optional[Atlas] = None, top_k: int = 5,
                  rerank: bool = True) -> list[list[Pose]]:
    """Identify several icons (crop, bg, fg). Returns, per crop, poses sorted best first (crop coords).

    Stage 1 (atlas): shape match against all glyphs -> top `rerank_k` names.
    Stage 2 (render): each name rendered at its estimated em box -> keep `pose_names` best.
    Stage 3 (render): best names x integer sizes x 1px offsets -> best pose by ΔE.
    Without `rerank`, poses carry de=inf and are ordered by atlas score only."""
    atlas = atlas or load_atlas()
    k = int(P["perceive.icons.rerank_k"])
    qs: list[IconQuery] = []
    for crop_, bg, fg in crops:
        mask, bb = ink_mask_from_crop(crop_, bg)
        raw = match_mask(mask, bb, atlas, top_k=3 * max(top_k, k)) if bb.w > 0 else []
        hits, seen = [], set()
        for h in raw:  # unique names: outlined/filled twins would otherwise waste half the shortlist
            if h.name not in seen:
                seen.add(h.name)
                hits.append(h)
        qs.append(IconQuery(crop_, bg, fg, hits[:max(top_k, k)], bb))
    # every shortlisted name x both FILL styles x the size estimate +-1px (anti-aliased fringe biases the
    # ink-based size by about a pixel), centred on the observed ink
    stage2 = [[p for h in q.hits[:k] for f in (0, 1) for p in _size_variants(h.name, f, em_box_for(h.name, q.ink_box, f, atlas))]
              if q.hits else [] for q in qs]
    if not rerank:
        return [p[:top_k] for p in stage2]
    render_poses(qs, stage2)
    stage3 = []
    n_names = int(P["perceive.icons.pose_names"])
    good = float(P["perceive.icons.good_de"])
    for ps in stage2:
        ps.sort(key=lambda p: p.de)  # finalists by pixels alone; priors only break final ties
        if ps and ps[0].de <= good:  # already reproduces the crop: no fine search needed
            stage3.append([])
            continue
        seen, best_names = set(), []
        for p in ps:
            if p.name not in seen:
                seen.add(p.name)
                best_names.append(p)
            if len(best_names) >= n_names:
                break
        # finalists are tried in both FILL styles (stage 1 kept one style per name)
        stage3.append([v for p in best_names for f in (0, 1) for v in _pose_variants(p.name, f, p.box)])
    render_poses(qs, stage3)
    out = []
    for ps2, ps3 in zip(stage2, stage3):
        allp = ps2 + ps3
        # priors only break ties among poses that already reproduce the crop; never let a prior
        # push an acceptable pose behind an unacceptable one
        ok = sorted((p for p in allp if _acceptable(p)), key=_pose_cost)
        rest = sorted((p for p in allp if not _acceptable(p)), key=lambda p: p.de)
        out.append((ok + rest)[:top_k])
    return out


def _acceptable(p: Pose) -> bool:
    """The renderer accepts a pose when it reproduces the crop (mean ΔE) AND the glyph itself (ink-relative
    mismatch): on a mostly-background crop a wrong glyph of similar mass can pass the mean alone."""
    return p.de <= float(P["perceive.icons.accept_de"]) and p.mis_e <= float(P["perceive.icons.accept_mis"])


def _pose_cost(p: Pose) -> float:
    """ΔE plus small priors: everyday glyphs and standard sizes win near-ties (aliases, check vs check_small)."""
    std = {float(v) for v in P["perceive.icons.standard_sizes"]}
    pen = 0.0 if p.name in common_icons() else float(P["perceive.icons.pen_uncommon"])
    pen += 0.0 if float(p.box.w) in std else float(P["perceive.icons.pen_nonstandard"])
    return p.de + pen


# ----------------------------------------------------------------------------- cli
if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        fills = [int(a) for a in sys.argv[2:]] or [0, 1]
        for f in fills:
            a = build_atlas(f)
            print(f"built atlas fill={f}: {len(a.names)} glyphs -> {atlas_path(f)}")
    else:
        print(__doc__)


# ----------------------------------------------------------------------------- perceive integration
def annotate_icons(doc, rgb: np.ndarray, atlas: Optional[Atlas] = None, rerank: bool = True) -> dict:
    """Identify every `icon` node in `doc` (in place): set icon_name, the em box the renderer needs,
    and meta['icon_fill'] / meta['icon_de'] / meta['icon_candidates']. Returns a summary dict.

    Acceptance is decided by the renderer: the best pose must reproduce the crop with mean ΔE2000
    <= perceive.icons.accept_de AND reproduce the glyph's silhouette (glyph_mismatch per outline pixel
    <= perceive.icons.accept_mis). Otherwise icon_name stays None and the candidates are recorded in
    meta (the decision queue surfaces them)."""
    atlas = atlas or load_atlas()
    nodes = [n for n in doc.walk() if n.type == "icon"]
    if not nodes:
        return {"icons": 0, "named": 0}
    H, W = rgb.shape[:2]
    pad = int(P["perceive.icons.crop_pad"])
    crops, origins = [], []
    for n in nodes:
        x0, y0, x1, y1 = n.box.expand(pad).as_int()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        crop_ = np.ascontiguousarray(rgb[y0:y1, x0:x1, :3])
        bg = n.meta.get("bg")
        bgc = Color.from_dict(bg) if isinstance(bg, dict) else (bg if isinstance(bg, Color) else None)
        if bgc is None and crop_.size:
            border = np.concatenate([crop_[0], crop_[-1], crop_[:, 0], crop_[:, -1]])
            bgc = Color.from_rgb(*np.median(border.reshape(-1, 3), axis=0))
        crops.append((crop_, bgc or Color(255, 255, 255), n.fill_color or Color(0, 0, 0)))
        origins.append((x0, y0))
    ranked = identify_many(crops, atlas, top_k=5, rerank=rerank)
    named = 0
    for n, (x0, y0), poses in zip(nodes, origins, ranked):
        n.meta["icon_candidates"] = [{"name": p.name, "fill": p.fill, "de": round(p.de, 3) if p.de != float("inf") else None,
                                      "mis": round(p.mis, 3), "mis_e": round(p.mis_e, 3)} for p in poses[:3]]
        if not poses or not _acceptable(poses[0]):
            continue
        best_pose = poses[0]
        n.meta["ink_box"] = n.box.to_dict()
        n.icon_name = best_pose.name
        n.meta["icon_fill"] = int(best_pose.fill)
        n.meta["icon_de"] = round(best_pose.de, 3)
        n.meta["icon_mis"] = round(best_pose.mis, 3)
        n.meta["icon_mis_e"] = round(best_pose.mis_e, 3)
        n.box = best_pose.box.translate(x0, y0)
        named += 1
    return {"icons": len(nodes), "named": named}
