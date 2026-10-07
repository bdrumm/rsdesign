"""Independent visual-fidelity validation: is the render 1:1 with the target screenshot?

Deliberately self-contained (does NOT reuse dt.compare's loss) so the final verdict is not
produced by the same function the optimizer minimised. See docs/VALIDATION.md.

Public API:
    report = validate(target_rgb, render_rgb, doc=None, out_dir=None)   -> FidelityReport
    report.to_dict(), report.gates -> {"pixel-exact": bool, ...}, report.markdown()
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np
from skimage import color as skcolor
from skimage.metrics import structural_similarity

from dt.common.image import save_rgb
from dt.ir import Box, Document
from dt.params import P, register

# ----------------------------------------------------------------------------- params
register("validate.jnd_de", 2.3, "ΔE2000 just-noticeable-difference threshold", (1.5, 4.0))
register("validate.edge.canny_low", 40, "Canny low threshold for edge maps", (10, 100))
register("validate.edge.canny_high", 120, "Canny high threshold for edge maps", (60, 250))
register("validate.region.quant", 12, "colour quantisation step for flat-region segmentation", (4, 32))
register("validate.region.min_area", 64, "min area (px) of a flat region to be scored", (16, 400))
register("validate.region.max_regions", 400, "cap on regions scored (largest first)", (50, 2000))
register("validate.edge.tile", 32, "tile size (px) for local (max) chamfer", (16, 96))
register("validate.gate.exact.chamfer_tile", 1.0, "pixel-exact: max tile-wise chamfer (px)")
register("validate.gate.ident.chamfer_tile", 2.0, "visually-identical: max tile-wise chamfer (px)")
register("validate.text.match_dist", 12.0, "max centre distance (px) to match OCR lines", (4, 40))
# gates
register("validate.gate.exact.jnd_frac", 0.005, "pixel-exact: max non-text JND fraction")
register("validate.gate.exact.chamfer", 0.5, "pixel-exact: max symmetric chamfer (px)")
register("validate.gate.exact.edge_within1", 0.99, "pixel-exact: min fraction of target edges within 1px")
register("validate.gate.exact.text_dpos", 1.0, "pixel-exact: max text position error (px)")
register("validate.gate.ident.jnd_frac", 0.02, "visually-identical: max non-text JND fraction")
register("validate.gate.ident.chamfer", 1.0, "visually-identical: max chamfer (px)")
register("validate.gate.ident.edge_within1", 0.97, "visually-identical: min edges within 1px")
register("validate.gate.ident.text_dpos", 2.0, "visually-identical: max text position error (px)")
register("validate.gate.ident.worst_region_de", 5.0, "visually-identical: max worst-region ΔE")
register("validate.gate.struct.edge_f1", 0.85,
         "structurally-faithful: min symmetric edge F1 at validate.gate.struct.edge_tol px (target edges with a "
         "render edge nearby x render edges with a target edge nearby). Replaces exact-pixel edge IoU, which "
         "anti-aliasing drives to ~0.56 for the gt IR itself and which ranked a 1px-shifted perfect doc (0.08) "
         "below a 6px colour mosaic (0.11)", (0.5, 0.99))
register("validate.gate.struct.edge_tol", 1.0, "structurally-faithful: edge match tolerance (px) of edge_f1", (0.5, 3.0))
register("validate.gate.struct.text_cer", 0.05, "structurally-faithful: max text CER")
register("validate.gate.struct.region_de", 10.0, "structurally-faithful: max any-region ΔE")
# editability (every tier): the output must be a design, not a picture of the target
register("validate.gate.raster_frac", 0.35,
         "every tier: max fraction of the screen covered by raster image nodes (image_ref crops); matches the "
         "critic's own budget refine.raster.budget_frac, so a full-page screenshot can never pass a gate", (0.0, 1.0))
register("validate.raster.text_cover", 0.5,
         "a target OCR line counts as rasterised text when at least this fraction of its box lies under a "
         "(non-logo) raster image node", (0.2, 1.0))
register("validate.raster.text_min_h", 9.0,
         "min target OCR line box height (px) for rasterised text to fail the gates; Vision boxes are ~0.72x the "
         "font size, so 9 px ~ 12 px text (the critic keeps text >= refine.raster.max_text_px editable)", (4.0, 24.0))


# ----------------------------------------------------------------------------- data
@dataclass
class RegionScore:
    box: dict
    area: int
    target_color: str
    render_color: str
    delta_e: float


@dataclass
class TextMatch:
    target_text: str
    render_text: str
    cer: float
    dx: float
    dy: float
    dsize: float
    target_box: dict
    matched: bool


@dataclass
class FidelityReport:
    width: int
    height: int
    # pixel
    mean_de: float = 0.0
    max_de: float = 0.0
    jnd_frac: float = 0.0
    de5_frac: float = 0.0
    de10_frac: float = 0.0
    ssim: float = 0.0
    ms_ssim: float = 0.0
    # split by text
    text_area_frac: float = 0.0
    jnd_frac_nontext: float = 0.0
    jnd_frac_text: float = 0.0
    mean_de_nontext: float = 0.0
    mean_de_text: float = 0.0
    # geometry
    edge_iou: float = 0.0  # exact pixel overlap (reported; too brittle to gate on)
    edge_f1: float = 0.0  # symmetric edge agreement within validate.gate.struct.edge_tol px (gated)
    chamfer: float = 0.0
    chamfer_tile_max: float = 0.0
    chamfer_tile_argmax: dict = field(default_factory=dict)
    edge_within1: float = 0.0
    n_target_edges: int = 0
    n_render_edges: int = 0
    # colour regions
    regions_scored: int = 0
    region_de_mean: float = 0.0
    region_de_worst: float = 0.0
    worst_regions: list[RegionScore] = field(default_factory=list)
    regions_over10: int = 0
    # text
    text_available: bool = False
    text_lines_target: int = 0
    text_lines_matched: int = 0
    text_cer: float = 0.0
    text_dpos_mean: float = 0.0
    text_dpos_max: float = 0.0
    text_dsize_mean: float = 0.0
    text_matches: list[TextMatch] = field(default_factory=list)
    # editability (from the IR): how much of the output is pixels rather than design
    raster_frac: float = 0.0
    raster_count: int = 0
    text_under_raster: int = 0
    text_rasterised: int = 0  # target OCR lines (>= validate.raster.text_min_h) painted by a non-logo raster
    # frame: everything is measured on the TARGET frame; a render of another size fails every gate
    render_width: int = 0
    render_height: int = 0
    size_match: bool = True
    # gates
    gates: dict[str, bool] = field(default_factory=dict)
    gate_failures: dict[str, list[str]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def markdown(self) -> str:
        g = self.gates
        rows = [
            "| gate | result |", "|---|---|",
            *[f"| {k} | {'PASS' if v else 'FAIL — ' + '; '.join(self.gate_failures.get(k, []))} |" for k, v in g.items()],
            "",
            "| measure | value |", "|---|---|",
            f"| mean ΔE2000 | {self.mean_de:.2f} |",
            f"| JND fraction (all / non-text / text) | {self.jnd_frac:.4f} / {self.jnd_frac_nontext:.4f} / {self.jnd_frac_text:.4f} |",
            f"| ΔE>5 / ΔE>10 fraction | {self.de5_frac:.4f} / {self.de10_frac:.4f} |",
            f"| max ΔE | {self.max_de:.1f} |",
            f"| SSIM / MS-SSIM | {self.ssim:.4f} / {self.ms_ssim:.4f} |",
            f"| edge IoU / edge F1 @{P['validate.gate.struct.edge_tol']:g}px | {self.edge_iou:.3f} / {self.edge_f1:.3f} |",
            f"| chamfer mean / tile-max (px) | {self.chamfer:.3f} / {self.chamfer_tile_max:.2f} @ {self.chamfer_tile_argmax} |",
            f"| target edges within 1px | {self.edge_within1:.4f} |",
            f"| regions scored / mean ΔE / worst ΔE / >10 | {self.regions_scored} / {self.region_de_mean:.2f} / {self.region_de_worst:.2f} / {self.regions_over10} |",
            f"| text lines (target/matched) | {self.text_lines_target} / {self.text_lines_matched} |",
            f"| text CER | {self.text_cer:.4f} |",
            f"| text Δpos mean / max (px) | {self.text_dpos_mean:.2f} / {self.text_dpos_max:.2f} |",
            f"| text Δsize mean (px) | {self.text_dsize_mean:.2f} |",
            f"| rasterised area / crops / text under a crop / target lines painted by a crop | {self.raster_frac:.1%} / "
            f"{self.raster_count} / {self.text_under_raster} / {self.text_rasterised} |",
            f"| render size (target {self.width}x{self.height}) | {self.render_width}x{self.render_height}"
            f"{'' if self.size_match else ' MISMATCH'} |",
        ]
        if self.worst_regions:
            rows += ["", "Worst regions (target vs render):", "| box | area | target | render | ΔE |", "|---|---|---|---|---|"]
            for r in self.worst_regions[:8]:
                b = r.box
                rows.append(f"| {b['x']:.0f},{b['y']:.0f} {b['w']:.0f}x{b['h']:.0f} | {r.area} | {r.target_color} | {r.render_color} | {r.delta_e:.1f} |")
        if self.artifacts:
            rows += ["", "Artifacts: " + ", ".join(f"`{v}`" for v in self.artifacts.values())]
        return "\n".join(rows)


# ----------------------------------------------------------------------------- pieces
def delta_e2000_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la = skcolor.rgb2lab(a[..., :3].astype(np.float32) / 255.0)
    lb = skcolor.rgb2lab(b[..., :3].astype(np.float32) / 255.0)
    return skcolor.deltaE_ciede2000(la, lb).astype(np.float32)


def _gray(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2GRAY)


def edge_map(a: np.ndarray) -> np.ndarray:
    return cv2.Canny(_gray(a), int(P["validate.edge.canny_low"]), int(P["validate.edge.canny_high"])) > 0


def edge_metrics(ta: np.ndarray, ra: np.ndarray) -> tuple[float, float, float, int, int, float, dict]:
    """edge IoU, symmetric chamfer (px), fraction of target edges within 1px of a render edge,
    tile-wise max chamfer (catches a single displaced element the global mean would dilute) + its tile."""
    et, er = edge_map(ta), edge_map(ra)
    nt, nr = int(et.sum()), int(er.sum())
    if nt == 0 and nr == 0:
        return 1.0, 0.0, 1.0, 0, 0, 0.0, {}
    inter = int((et & er).sum())
    union = int((et | er).sum())
    iou = inter / union if union else 0.0
    # distance transforms: distance from every pixel to nearest edge pixel
    dt_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3) if nr else np.full(et.shape, 1e3, np.float32)
    dt_t = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 3) if nt else np.full(et.shape, 1e3, np.float32)
    d_t2r = dt_r[et] if nt else np.array([0.0])
    d_r2t = dt_t[er] if nr else np.array([0.0])
    chamfer = 0.5 * (float(d_t2r.mean()) + float(d_r2t.mean()))
    within1 = float((d_t2r <= 1.0).mean()) if nt else 1.0
    # tile-wise symmetric chamfer
    T = int(P["validate.edge.tile"])
    H, W = et.shape
    tile_max, arg = 0.0, {}
    for y in range(0, H, T):
        for x in range(0, W, T):
            st, sr = et[y:y + T, x:x + T], er[y:y + T, x:x + T]
            a = dt_r[y:y + T, x:x + T][st]
            b = dt_t[y:y + T, x:x + T][sr]
            if a.size == 0 and b.size == 0:
                continue
            va = float(a.mean()) if a.size else float(b.max())
            vb = float(b.mean()) if b.size else float(a.max())
            v = 0.5 * (va + vb)
            if v > tile_max:
                tile_max, arg = v, {"x": x, "y": y, "w": min(T, W - x), "h": min(T, H - y)}
    return iou, chamfer, within1, nt, nr, tile_max, arg


def edge_f1(ta: np.ndarray, ra: np.ndarray, tol: Optional[float] = None) -> float:
    """Symmetric edge agreement: F1 of (target edges within ``tol`` px of a render edge) and
    (render edges within ``tol`` px of a target edge); ``tol`` defaults to
    ``validate.gate.struct.edge_tol``. 1 when neither image has edges, 0 when only one has."""
    tol = float(P["validate.gate.struct.edge_tol"] if tol is None else tol)
    et, er = edge_map(ta), edge_map(ra)
    nt, nr = int(et.sum()), int(er.sum())
    if nt == 0 or nr == 0:
        return 1.0 if nt == nr else 0.0
    dt_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    dt_t = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 3)
    rec, prec = float((dt_r[et] <= tol).mean()), float((dt_t[er] <= tol).mean())
    return 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0


def ms_ssim(ga: np.ndarray, gb: np.ndarray, levels: int = 4) -> float:
    vals = []
    a, b = ga.astype(np.float32), gb.astype(np.float32)
    for _ in range(levels):
        if min(a.shape) < 16:
            break
        vals.append(structural_similarity(a, b, data_range=255.0))
        a = cv2.pyrDown(a)
        b = cv2.pyrDown(b)
    return float(np.mean(vals)) if vals else 0.0


def flat_regions(target: np.ndarray) -> list[tuple[Box, np.ndarray]]:
    """Segment the TARGET into flat-colour regions (quantised connected components)."""
    q = int(P["validate.region.quant"])
    min_area = int(P["validate.region.min_area"])
    qimg = (target[..., :3] // q).astype(np.int32)
    key = qimg[..., 0] * 1_000_000 + qimg[..., 1] * 1000 + qimg[..., 2]
    out: list[tuple[Box, np.ndarray]] = []
    # label per colour key: process unique keys by frequency (bounded)
    uniq, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts)
    for idx in order:
        if counts[idx] < min_area:
            break
        mask = (key == uniq[idx]).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            out.append((Box(x, y, w, h), labels == i))
        if len(out) >= int(P["validate.region.max_regions"]):
            break
    out.sort(key=lambda t: -int(t[1].sum()))
    return out[: int(P["validate.region.max_regions"])]


def region_scores(target: np.ndarray, render: np.ndarray, regions: list[tuple[Box, np.ndarray]]) -> list[RegionScore]:
    from dt.ir import Color

    scores = []
    for box, mask in regions:
        tpx = target[mask][:, :3].astype(np.float32)
        rpx = render[mask][:, :3].astype(np.float32)
        tc = Color.from_rgb(*tpx.mean(axis=0))
        rc = Color.from_rgb(*rpx.mean(axis=0))
        # ΔE between mean colours plus mean per-pixel ΔE (captures texture mismatch)
        lab_t = skcolor.rgb2lab(tpx.reshape(1, -1, 3) / 255.0)
        lab_r = skcolor.rgb2lab(rpx.reshape(1, -1, 3) / 255.0)
        de = float(skcolor.deltaE_ciede2000(lab_t, lab_r).mean())
        scores.append(RegionScore(box.to_dict(), int(mask.sum()), tc.hex(), rc.hex(), de))
    return scores


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ocr_lines(rgb: np.ndarray) -> Optional[list[tuple[str, Box]]]:
    """[(text, box)] via dt.perceive.ocr if available, else None."""
    try:
        from dt.perceive import ocr as _ocr  # built by module A
    except Exception:
        return None
    try:
        lines = _ocr.ocr_lines(rgb)
    except Exception:
        return None
    out = []
    for ln in lines:
        text = getattr(ln, "text", None)
        box = getattr(ln, "box", None)
        if text is None or box is None:
            continue
        if not isinstance(box, Box):
            box = Box.from_dict(box) if isinstance(box, dict) else Box(*box)
        out.append((text, box))
    return out


def text_fidelity(target: np.ndarray, render: np.ndarray) -> tuple[bool, list[TextMatch]]:
    tl = _ocr_lines(target)
    if tl is None:
        return False, []
    rl = _ocr_lines(render) or []
    maxd = float(P["validate.text.match_dist"])
    used = set()
    matches: list[TextMatch] = []
    for t_text, tb in tl:
        best, best_d = None, 1e9
        for j, (r_text, rb) in enumerate(rl):
            if j in used:
                continue
            d = ((tb.cx - rb.cx) ** 2 + (tb.cy - rb.cy) ** 2) ** 0.5
            # prefer same text when distances tie-ish
            d_eff = d - (3.0 if r_text.strip() == t_text.strip() else 0.0)
            if d_eff < best_d and d <= maxd + tb.h:
                best, best_d = j, d_eff
        if best is None:
            matches.append(TextMatch(t_text, "", 1.0, 0.0, 0.0, 0.0, tb.to_dict(), False))
            continue
        used.add(best)
        r_text, rb = rl[best]
        cer = _levenshtein(t_text.strip(), r_text.strip()) / max(1, len(t_text.strip()))
        matches.append(TextMatch(t_text, r_text, cer, rb.x - tb.x, rb.y - tb.y, rb.h - tb.h, tb.to_dict(), True))
    return True, matches


def text_mask_from_lines(shape: tuple[int, int], lines: list[TextMatch] | list[tuple[str, Box]], pad: int = 2) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for item in lines:
        b = Box.from_dict(item.target_box) if isinstance(item, TextMatch) else item[1]
        x0, y0, x1, y1 = b.expand(pad).as_int()
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m


def text_mask_from_doc(doc: Document) -> np.ndarray:
    m = np.zeros((doc.height, doc.width), dtype=bool)
    for n in doc.walk():
        if n.type == "text" and n.text:
            x0, y0, x1, y1 = n.box.expand(2).as_int()
            m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m


# ----------------------------------------------------------------------------- artefacts
def save_artifacts(out_dir: str, target: np.ndarray, render: np.ndarray, de: np.ndarray, worst: list[RegionScore]) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    arts = {}
    h, w = de.shape
    # heatmap
    heat = np.clip(de / 20.0, 0, 1)
    heat_rgb = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    arts["heatmap"] = save_rgb(heat_rgb, os.path.join(out_dir, "heatmap.png"))
    # side by side with heatmap
    gap = np.full((h, 8, 3), 128, np.uint8)
    sbs = np.concatenate([target[..., :3], gap, render[..., :3], gap, heat_rgb], axis=1)
    arts["side_by_side"] = save_rgb(sbs, os.path.join(out_dir, "side_by_side.png"))
    # worst region crops
    if worst:
        tiles = []
        for r in worst[:8]:
            b = Box.from_dict(r.box).expand(6)
            x0, y0, x1, y1 = b.as_int()
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            tc, rc = target[y0:y1, x0:x1, :3], render[y0:y1, x0:x1, :3]
            tile = np.concatenate([tc, np.full((y1 - y0, 4, 3), 255, np.uint8), rc], axis=1)
            tiles.append(tile)
        if tiles:
            tw = max(t.shape[1] for t in tiles)
            padded = [np.pad(t, ((0, 6), (0, tw - t.shape[1]), (0, 0)), constant_values=200) for t in tiles]
            arts["worst_regions"] = save_rgb(np.concatenate(padded, axis=0), os.path.join(out_dir, "worst_regions.png"))
    # blink comparator
    save_rgb(target, os.path.join(out_dir, "target.png"))
    save_rgb(render, os.path.join(out_dir, "render.png"))
    html = f"""<!doctype html><meta charset=utf-8><title>blink</title>
<style>body{{margin:0;background:#222;color:#eee;font:12px monospace}} .wrap{{position:relative;width:{w}px;height:{h}px}}
img{{position:absolute;left:0;top:0}} #r{{animation:blink 1s steps(1) infinite}} @keyframes blink{{50%{{opacity:0}}}}
.bar{{padding:6px}}</style>
<div class=bar>blink: target ↔ render (1 Hz). <label><input type=checkbox id=stop> pause</label> <label><input type=range id=op min=0 max=1 step=0.01 value=1> render opacity</label></div>
<div class=wrap><img src="target.png"><img id=r src="render.png"></div>
<script>stop.onchange=e=>r.style.animationPlayState=e.target.checked?'paused':'running';op.oninput=e=>{{r.style.animation='none';r.style.opacity=e.target.value}}</script>"""
    p = os.path.join(out_dir, "blink.html")
    open(p, "w").write(html)
    arts["blink"] = p
    return arts


# ----------------------------------------------------------------------------- frame + editability
def conform_to_target(target: np.ndarray, render: np.ndarray) -> tuple[np.ndarray, bool]:
    """``(render', size_match)``: ``render`` cropped / padded to the target's frame (RGB).

    Measurements must cover the whole target: cropping both images to their common size (as
    ``dt.common.image.same_size`` does) lets a 48x48 render of a 600x900 page score as
    pixel-exact. Missing render pixels are filled with one flat colour opposite to the target
    there (black over a light target, white over a dark one) so they count as wrong in the pixel
    and region measures, and -- being flat -- match none of the target's edges. (A per-pixel
    inverse would reproduce the target's edges exactly.)"""
    th, tw = target.shape[:2]
    rh, rw = render.shape[:2]
    if (rh, rw) == (th, tw):
        return render[..., :3], True
    h, w = min(th, rh), min(tw, rw)
    missing = np.ones((th, tw), dtype=bool)
    missing[:h, :w] = False
    t3 = target[..., :3][missing].astype(np.float32)
    lum = float((0.299 * t3[:, 0] + 0.587 * t3[:, 1] + 0.114 * t3[:, 2]).mean()) if t3.size else 255.0
    out = np.full((th, tw, 3), 0 if lum > 127.5 else 255, dtype=np.uint8)
    out[:h, :w] = render[:h, :w, :3]
    return out, False


def painted_nodes(doc: Document) -> list:
    """Nodes that can paint: visible with opacity > 0 and no hidden ancestor."""
    out: list = []

    def rec(n) -> None:
        if not n.visible or n.opacity <= 0.0:
            return
        out.append(n)
        for c in n.children:
            rec(c)
    rec(doc.root)
    return out


def raster_nodes(doc: Document) -> list:
    """Painted image nodes that carry pixels (``image_ref``) -- flagged by the critic
    (``meta.rasterised``) or not: an unflagged screenshot crop is still not a design."""
    return [n for n in painted_nodes(doc) if n.type == "image" and (n.image_ref or n.meta.get("rasterised"))]


def raster_mask(nodes: list, shape: tuple[int, int]) -> np.ndarray:
    """Union mask of the node boxes clipped to the frame (overlapping crops count once)."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    for n in nodes:
        x0, y0, x1, y1 = n.box.as_int()
        m[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = True
    return m


# ----------------------------------------------------------------------------- main
def validate(target: np.ndarray, render: np.ndarray, doc: Optional[Document] = None, out_dir: Optional[str] = None,
             with_text: bool = True) -> FidelityReport:
    rh, rw = render.shape[:2]
    render, size_match = conform_to_target(target, render)
    target = target[..., :3]
    h, w = target.shape[:2]
    rep = FidelityReport(width=w, height=h, render_width=rw, render_height=rh, size_match=size_match)
    de = delta_e2000_map(target, render)
    jnd = float(P["validate.jnd_de"])
    rep.mean_de = float(de.mean())
    rep.max_de = float(de.max())
    rep.jnd_frac = float((de > jnd).mean())
    rep.de5_frac = float((de > 5).mean())
    rep.de10_frac = float((de > 10).mean())
    gt, gr = _gray(target), _gray(render)
    rep.ssim = float(structural_similarity(gt, gr, data_range=255)) if min(h, w) >= 7 else 1.0
    rep.ms_ssim = ms_ssim(gt, gr)
    # geometry
    (rep.edge_iou, rep.chamfer, rep.edge_within1, rep.n_target_edges, rep.n_render_edges,
     rep.chamfer_tile_max, rep.chamfer_tile_argmax) = edge_metrics(target, render)
    rep.edge_f1 = edge_f1(target, render)
    # colour regions
    regions = flat_regions(target)
    scores = region_scores(target, render, regions)
    rep.regions_scored = len(scores)
    if scores:
        des = np.array([s.delta_e for s in scores])
        rep.region_de_mean = float(des.mean())
        rep.region_de_worst = float(des.max())
        rep.regions_over10 = int((des > 10).sum())
        rep.worst_regions = sorted(scores, key=lambda s: -s.delta_e)[:12]
    # text
    text_mask = None
    if with_text:
        rep.text_available, matches = text_fidelity(target, render)
        if rep.text_available:
            rep.text_matches = matches
            rep.text_lines_target = len(matches)
            m_ok = [m for m in matches if m.matched]
            rep.text_lines_matched = len(m_ok)
            if matches:
                rep.text_cer = float(np.mean([m.cer for m in matches]))
            if m_ok:
                dpos = np.array([(m.dx ** 2 + m.dy ** 2) ** 0.5 for m in m_ok])
                rep.text_dpos_mean, rep.text_dpos_max = float(dpos.mean()), float(dpos.max())
                rep.text_dsize_mean = float(np.mean([abs(m.dsize) for m in m_ok]))
            text_mask = text_mask_from_lines((h, w), matches)
    if text_mask is None and doc is not None:
        text_mask = text_mask_from_doc(doc)
        if text_mask.shape != (h, w):
            text_mask = text_mask[:h, :w]
    if text_mask is not None and text_mask.any():
        rep.text_area_frac = float(text_mask.mean())
        rep.jnd_frac_text = float((de[text_mask] > jnd).mean())
        rep.mean_de_text = float(de[text_mask].mean())
        nt = ~text_mask
        rep.jnd_frac_nontext = float((de[nt] > jnd).mean()) if nt.any() else 0.0
        rep.mean_de_nontext = float(de[nt].mean()) if nt.any() else 0.0
    else:
        rep.jnd_frac_nontext, rep.mean_de_nontext = rep.jnd_frac, rep.mean_de
    # editability
    if doc is not None:
        rasters = raster_nodes(doc)
        rep.raster_count = len(rasters)
        rep.raster_frac = float(raster_mask(rasters, (h, w)).mean()) if rasters else 0.0
        rep.text_under_raster = sum(1 for t in painted_nodes(doc) if t.type == "text" and t.text and
                                    any(r.box.intersect(t.box).area > 0.3 * max(1.0, t.box.area) for r in rasters))
        # text *replaced* by a crop leaves no text node behind: find it on the target (OCR lines)
        non_logo = [r for r in rasters if not r.meta.get("logo")]
        if non_logo and rep.text_matches:
            m = raster_mask(non_logo, (h, w))
            cover, min_h = float(P["validate.raster.text_cover"]), float(P["validate.raster.text_min_h"])
            for tm in rep.text_matches:
                b = Box.from_dict(tm.target_box)
                x0, y0, x1, y1 = b.as_int()
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if b.h >= min_h and x1 > x0 and y1 > y0 and float(m[y0:y1, x0:x1].mean()) >= cover:
                    rep.text_rasterised += 1
    # gates
    rep.gates, rep.gate_failures = _gates(rep)
    if out_dir:
        rep.artifacts = save_artifacts(out_dir, target, render, de, rep.worst_regions)
        rep.save(os.path.join(out_dir, "validation.json"))
        open(os.path.join(out_dir, "validation.md"), "w").write(rep.markdown())
    return rep


def _gates(r: FidelityReport) -> tuple[dict[str, bool], dict[str, list[str]]]:
    fails: dict[str, list[str]] = {"pixel-exact": [], "visually-identical": [], "structurally-faithful": []}
    text_ok = (not r.text_available) or (r.text_cer == 0.0)

    def chk(tier: str, cond: bool, msg: str) -> None:
        if not cond:
            fails[tier].append(msg)

    chk("pixel-exact", r.jnd_frac_nontext < P["validate.gate.exact.jnd_frac"], f"non-text JND frac {r.jnd_frac_nontext:.4f}")
    chk("pixel-exact", r.chamfer < P["validate.gate.exact.chamfer"], f"chamfer {r.chamfer:.2f}px")
    chk("pixel-exact", r.chamfer_tile_max < P["validate.gate.exact.chamfer_tile"], f"tile chamfer {r.chamfer_tile_max:.2f}px @ {r.chamfer_tile_argmax}")
    chk("pixel-exact", r.edge_within1 > P["validate.gate.exact.edge_within1"], f"edges within 1px {r.edge_within1:.3f}")
    chk("pixel-exact", text_ok, f"text CER {r.text_cer:.3f}")
    chk("pixel-exact", (not r.text_available) or r.text_dpos_max <= P["validate.gate.exact.text_dpos"], f"text Δpos max {r.text_dpos_max:.1f}px")

    chk("visually-identical", r.jnd_frac_nontext < P["validate.gate.ident.jnd_frac"], f"non-text JND frac {r.jnd_frac_nontext:.4f}")
    chk("visually-identical", r.chamfer < P["validate.gate.ident.chamfer"], f"chamfer {r.chamfer:.2f}px")
    chk("visually-identical", r.chamfer_tile_max < P["validate.gate.ident.chamfer_tile"], f"tile chamfer {r.chamfer_tile_max:.2f}px @ {r.chamfer_tile_argmax}")
    chk("visually-identical", r.edge_within1 > P["validate.gate.ident.edge_within1"], f"edges within 1px {r.edge_within1:.3f}")
    chk("visually-identical", text_ok, f"text CER {r.text_cer:.3f}")
    chk("visually-identical", (not r.text_available) or r.text_dpos_max <= P["validate.gate.ident.text_dpos"], f"text Δpos max {r.text_dpos_max:.1f}px")
    chk("visually-identical", r.region_de_worst < P["validate.gate.ident.worst_region_de"], f"worst region ΔE {r.region_de_worst:.1f}")

    chk("structurally-faithful", r.edge_f1 > P["validate.gate.struct.edge_f1"],
        f"edge F1 @{P['validate.gate.struct.edge_tol']:g}px {r.edge_f1:.3f}")
    for tier in ("pixel-exact", "visually-identical", "structurally-faithful"):
        # everything is measured on the target frame; another canvas size is never a 1:1 translation
        chk(tier, r.size_match, f"render size {r.render_width}x{r.render_height} != target {r.width}x{r.height}")
        # a crop over text is never faithful, nor is a picture of the screen instead of a design
        chk(tier, r.text_under_raster == 0, f"{r.text_under_raster} text nodes under a raster")
        chk(tier, r.text_rasterised == 0, f"{r.text_rasterised} target text lines painted by a raster crop")
        chk(tier, r.raster_frac <= P["validate.gate.raster_frac"], f"rasterised area {r.raster_frac:.1%}")
    chk("structurally-faithful", (not r.text_available) or r.text_cer < P["validate.gate.struct.text_cer"], f"text CER {r.text_cer:.3f}")
    chk("structurally-faithful", r.region_de_worst < P["validate.gate.struct.region_de"], f"worst region ΔE {r.region_de_worst:.1f}")
    return {k: not v for k, v in fails.items()}, {k: v for k, v in fails.items() if v}
