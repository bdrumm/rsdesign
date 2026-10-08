"""Diff decomposition adversary: explain the difference image as a sum of interpretable causes.

``find(target_rgb, candidate_rgb, candidate_ir=None) -> list[Finding]``

Rather than thresholding a residual and guessing what it is, this adversary *decomposes* the
target-vs-candidate difference into layers, each with an explicit generative model, and attributes
every changed pixel to the first layer that explains it:

0. **Global colour layer** (``color.tint_global``): the candidate's flat palette is matched to the
   target's colours under it; a coherent Lab offset shared by most of the page's colour mass is a
   global tint. It is fitted as an affine Lab colour transfer and *removed* before anything else, so
   a theme error does not masquerade as hundreds of local findings.
1. **Renderer noise layer** (explicit, never reported): both images are blurred by a small kernel
   (anti-aliasing, resampling, JPEG ringing), the residual is the minimum over a half-pixel
   compensation grid (sub-pixel capture offsets), and a page-level noise model sets a per-pixel
   tolerance ``tau(x) = tau_flat + k * |grad(x)|``. ``tau_flat`` comes from the residual in flat areas
   and ``k`` from the residual-per-unit-gradient over edge-rich tiles: renderer noise is homogeneous
   over the page, a real error is a local outlier. Only the *excess* over ``tau`` is evidence.
2. **Geometry layer** (flow): dense optical flow (OpenCV DIS, coarse-to-fine) between target and
   candidate plus template search proposes integer displacements for every changed region; a
   displacement that removes most of the region's excess residual is a geometric cause. Flow below
   ``flow_min`` (sub-pixel) is renderer noise. Which IR node edges move decides between
   ``geometry.shift`` / ``geometry.size`` / ``text.position`` / ``layout.spacing`` (siblings moving by
   multiples of one step) / text ``structure.split`` and ``structure.merge``.
3. **Colour layer**: a per-region affine colour transfer (fitted both ways, so it must be invertible
   and edge-preserving) explains recolourings: ``color.fill``, ``color.stroke`` (the region hugs a
   node border), ``effect.opacity`` (the transfer is a blend towards the backdrop). Soft, one-signed
   luminance bands outside a node box are ``effect.shadow``; changes confined to node corners are
   ``geometry.radius``.
4. **Structure layer** (the residual): whatever no transform explains is attributed by *ink
   ownership* (which image paints the pixels: target only -> missing, candidate only -> extra, both
   -> content changed) and by shape statistics (text-like stroke density / row shape, icon-like
   compactness), or by the IR node type when the candidate IR is given: ``structure.missing`` /
   ``icon.missing`` / ``structure.extra`` / ``text.content`` / ``text.style`` / ``icon.glyph`` /
   shape ``structure.split`` / ``structure.merge``.

The IR is optional; when present it is only used to *snap* and *type* image evidence (which node a
region belongs to, its type and box edges), never as evidence that something changed.

``decompose(...)`` returns the full layered explanation (noise model, tint, per-region hypotheses);
``find`` returns only the findings. All thresholds are registered in ``dt.params``
(``adversary.decompose.*``) and were tuned on the CALIBRATION split only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import lab2rgb, rgb2lab

from dt.adversary.taxonomy import Finding
from dt.ir import Box, Document, Node
from dt.params import P, register

# --------------------------------------------------------------------------- params
_K = "adversary.decompose."
register(_K + "blur_sigma", 2.0, "noise layer: coarse scale -- Gaussian sigma (px) applied to both images before differencing (AA / font smoothing / JPEG)", (0.8, 4.0))
register(_K + "fine_mult", 1.25, "noise layer: the fine scale's tolerance is this multiple of its page noise model (it sees more renderer noise)", (1.0, 4.0))
register(_K + "cover_ratio", 0.02, "noise layer: an edge tile carries residual when its fine-scale residual per unit gradient exceeds this", (0.005, 0.2))
register(_K + "noisy_cover", 0.38, "noise layer: a capture is noisy when this share of edge tiles carries residual (switches off fine-detail layers)", (0.1, 0.9))
register(_K + "sigma_fine", 0.8, "noise layer: fine scale sigma (px): thin strokes, 1 px geometry, small recolourings", (0.0, 1.5))
register(_K + "subpixel", 0.5, "noise layer: half-width (px) of the sub-pixel compensation grid (renderer/capture offsets below 1 px)", (0.0, 0.75))
register(_K + "tau_min", 3.0, "noise layer: minimum ΔE tolerance in flat areas", (1.0, 10.0))
register(_K + "flat_grad", 1.0, "noise layer: |grad| (ΔE/px) under which a pixel is flat (for the flat-area noise estimate)", (0.2, 5.0))
register(_K + "flat_q", 0.9, "noise layer: per mostly-flat tile, this quantile of its flat-pixel residual", (0.5, 0.999))
register(_K + "tile_q", 0.75, "noise layer: quantile over tiles of the per-tile flat level taken as the page flat noise level", (0.5, 0.95))
register(_K + "tau_max", 12.0, "noise layer: cap on the flat tolerance", (3.0, 30.0))
register(_K + "edge_k_max", 1.5, "noise layer: cap on the page edge-noise level k", (0.2, 5.0))
register(_K + "flat_mult", 2.0, "noise layer: tau_flat = max(tau_min, flat_mult x flat noise level)", (1.0, 5.0))
register(_K + "tile", 24, "noise layer: tile size (px) for the residual-per-gradient page statistic", (8, 64))
register(_K + "tile_min_grad", 200.0, "noise layer: tiles with less summed |grad| are ignored by the page statistic", (20.0, 2000.0))
register(_K + "edge_q", 0.75, "noise layer: quantile over edge tiles of residual/|grad| taken as the page edge-noise level k", (0.5, 0.95))
register(_K + "edge_mult", 2.0, "noise layer: per-pixel tolerance adds edge_mult x k x |grad|", (0.5, 6.0))
register(_K + "edge_k_min", 0.05, "noise layer: floor of the page edge-noise level k", (0.0, 0.5))
register(_K + "dilate", 3, "regions: dilation (px) that merges fragments of one cause into one region", (0, 10))
register(_K + "min_px", 6, "regions: minimum number of changed pixels", (1, 100))
register(_K + "min_mass", 80.0, "regions: minimum excess ΔE mass (sum over pixels of residual above tolerance)", (5.0, 2000.0))
register(_K + "max_regions", 80, "regions: analyse at most this many regions (largest excess mass first)", (8, 400))
register(_K + "flow_min", 0.75, "geometry: displacements shorter than this (px) are sub-pixel renderer noise", (0.3, 1.5))
register(_K + "search", 28, "geometry: template search radius (px) for displacement proposals", (4, 64))
register(_K + "explain_min", 0.6, "a hypothesis explains a region when it removes at least this fraction of its excess mass", (0.3, 0.95))
register(_K + "colour_min", 0.8, "colour: the per-region affine transfer must explain this much in both directions", (0.5, 0.99))
register(_K + "text_shift_min", 0.95, "geometry: a text run counts as moved only when the displacement explains this much (else its glyphs changed)", (0.5, 0.99))
register(_K + "tint_min_de", 1.5, "tint: a palette colour counts as shifted at this ΔE76", (0.5, 5.0))
register(_K + "tint_min_cover", 0.2, "tint: shifted colours must cover this fraction of the flat palette mass", (0.05, 0.95))
register(_K + "tint_max_resid", 0.4, "tint: the one-offset model must leave at most this fraction of the palette error", (0.1, 0.8))
register(_K + "tint_min_colors", 2, "tint: at least this many distinct palette colours must shift", (1, 6))
register(_K + "palette_min_frac", 0.0005, "tint: a flat candidate colour joins the palette when it covers this fraction of the page", (0.0001, 0.05))
register(_K + "ink_de", 12.0, "structure: ΔE from the local background above which a pixel is ink", (4.0, 40.0))
register(_K + "own_frac", 0.75, "structure: one image owns a region when it paints this share of its ink-differing pixels", (0.5, 0.95))
register(_K + "merge_max_strip", 32.0, "merge: the filled strip is at most this thick (px)", (8.0, 96.0))
register(_K + "merge_backdrop_de", 20.0, "merge: the target inside the filled strip must be within this ΔE of the backdrop around the shape", (5.0, 60.0))
register(_K + "group_gap", 10.0, "grouping: regions with the same explanation closer than this (px) merge", (0.0, 64.0))
register(_K + "style_dh", 1.5, "text: ink height change (px) that marks a text.style change", (0.5, 4.0))
register(_K + "style_density", 0.2, "text: |log ink-mass ratio| (at an unchanged run width) that marks a weight change", (0.05, 0.6))
register(_K + "style_cols", 0.75, "text: a style change touches at least this share of the run's columns", (0.4, 0.99))
register(_K + "conf_mass", 400.0, "confidence = explanation quality x (1 - exp(-excess mass / conf_mass))", (20.0, 5000.0))
register(_K + "noisy_min_mass", 300.0, "systematic noise: on a noisy capture, regions below this coarse-scale excess mass are renderer noise", (0.0, 3000.0))
register(_K + "pop_min", 8, "systematic noise: a weak-evidence population this large is a page-wide renderer difference", (3, 40))
register(_K + "pop_mult", 5.0, "systematic noise: members below pop_mult x the population's median excess mass are renderer noise", (1.5, 12.0))
register(_K + "seg_noise_q", 0.5, "segments: quantile of the page's segment offsets taken as its segment noise level", (0.5, 0.95))
register(_K + "seg_noise_mult", 1.6, "segments: threshold = max(seg_de, seg_noise_mult x segment noise level)", (1.0, 8.0))
register(_K + "seg_margin", 4.0, "segments: segments within this distance (px) of a region found by the pixel layers belong to it", (0.0, 16.0))
register(_K + "seg_min_px", 20, "segments: candidate flat-colour segments smaller than this are not tested", (4, 100))
register(_K + "seg_de", 3.0, "segments: median Lab offset (ΔE76) that marks a recoloured segment", (1.0, 10.0))
register(_K + "seg_consistency", 0.6, "segments: share of the segment's pixels that must move along the median offset", (0.3, 0.95))
register(_K + "corner_reach", 12.0, "radius probe: corner squares extend this far (px) beyond the candidate radius", (4.0, 30.0))
register(_K + "corner_min_each", 4, "radius probe: minimum changed pixels in each corner", (1, 30))
register(_K + "corner_min_px", 24, "radius probe: minimum changed pixels over the corners", (3, 100))
register(_K + "max_node_frac", 0.35, "IR: nodes larger than this fraction of the page are never a region's owner", (0.05, 1.0))


def _p(name: str) -> float:
    return float(P[_K + name])


# --------------------------------------------------------------------------- image helpers
def _lab(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return rgb2lab(np.clip(a, 0.0, 1.0)).astype(np.float32)


def _blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    a = rgb.astype(np.float32)
    return cv2.GaussianBlur(a, (0, 0), sigma) if sigma > 0 else a


def _grad(L: np.ndarray) -> np.ndarray:
    g = np.zeros(L.shape[:2], np.float32)
    for k in range(3):
        gx = cv2.Sobel(L[..., k], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(L[..., k], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        g += gx * gx + gy * gy
    return np.sqrt(g)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """``out(x) = img(x + d)`` (bilinear, edge-replicated)."""
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``img`` at float coordinates (edge-replicated); returns (n, C)."""
    n = len(xs)
    cols = 1024
    rows = max(1, -(-n // cols))
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n], my[:n] = xs, ys
    out = cv2.remap(img, mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n]


def _grid(h: float) -> list[tuple[float, float]]:
    if h <= 0:
        return [(0.0, 0.0)]
    return [(sx, sy) for sy in (-h, 0.0, h) for sx in (-h, 0.0, h)]


def _tolerant_de(LT: np.ndarray, LC: np.ndarray, h: float) -> np.ndarray:
    """min over the sub-pixel grid of ΔE(T(x), C(x + s)) -- the noise-compensated residual."""
    best = None
    for sx, sy in _grid(h):
        s = LC if sx == 0 and sy == 0 else _shift(LC, sx, sy)
        d = np.sqrt(np.einsum("ijk,ijk->ij", LT - s, LT - s))
        best = d if best is None else np.minimum(best, d)
    return best.astype(np.float32)


def _box_of(x0: int, y0: int, x1: int, y1: int) -> Box:
    return Box(x0, y0, x1 - x0, y1 - y0)


def _clip(b: Box, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = b.as_int()
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def _ring_median(L: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    top, bot = L[y0, x0:x1], L[y1 - 1, x0:x1]
    lef, rig = L[y0:y1, x0], L[y0:y1, x1 - 1]
    ring = np.concatenate([top, bot, lef, rig], 0)
    return np.median(ring, axis=0)


# --------------------------------------------------------------------------- data
@dataclass
class NoiseModel:
    offset: tuple[float, float]
    tau_flat: float
    k_edge: float
    flat_level: float
    tiles: int

    def to_dict(self) -> dict:
        return {"offset": [round(self.offset[0], 3), round(self.offset[1], 3)], "tau_flat": round(self.tau_flat, 3),
                "k_edge": round(self.k_edge, 4), "flat_level": round(self.flat_level, 3), "tiles": self.tiles}


@dataclass
class Region:
    """One connected changed region and the hypotheses tested on it."""
    id: int
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray            # bool, window [y0:y1, x0:x1]
    px: int
    mass: float                 # excess ΔE mass
    layer: str = "structure"    # geometry | colour | effect | structure
    type: str = "structure.missing"
    magnitude: Optional[float] = None
    quality: float = 0.0        # fraction of the excess the winning hypothesis explains (structure: ownership share)
    evidence: dict = field(default_factory=dict)
    node_id: Optional[str] = None
    box: Optional[Box] = None   # reported box (defaults to the region bbox)
    scale: int = 1              # analysis scale (0 fine, 1 coarse) where the region's evidence is strongest
    members: list[int] = field(default_factory=list)

    @property
    def bbox(self) -> Box:
        return _box_of(self.x0, self.y0, self.x1, self.y1)


@dataclass
class Decomposition:
    findings: list[Finding]
    regions: list[Region]
    noise: NoiseModel
    tint: Optional[dict]
    noise_cover: float
    noisy: bool
    seconds: float
    timings: dict

    def summary(self) -> dict:
        layers: dict[str, int] = {}
        for r in self.regions:
            layers[r.layer] = layers.get(r.layer, 0) + 1
        return {"findings": len(self.findings), "regions": len(self.regions), "layers": layers,
                "noise": self.noise.to_dict(), "noise_cover": round(self.noise_cover, 3), "noisy": self.noisy, "tint": self.tint, "seconds": round(self.seconds, 3), "timings": self.timings}


# --------------------------------------------------------------------------- layer 0: global tint
def _palette(C: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Flat candidate colours (pixel and its 4-neighbours identical): (colours uint8 (n,3), counts, index arrays)."""
    packed = (C[..., 0].astype(np.int32) << 16) | (C[..., 1].astype(np.int32) << 8) | C[..., 2].astype(np.int32)
    flat = np.ones(packed.shape, bool)
    flat[1:, :] &= packed[1:, :] == packed[:-1, :]
    flat[:-1, :] &= packed[:-1, :] == packed[1:, :]
    flat[:, 1:] &= packed[:, 1:] == packed[:, :-1]
    flat[:, :-1] &= packed[:, :-1] == packed[:, 1:]
    idx = np.flatnonzero(flat)
    vals = packed.reshape(-1)[idx]
    uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    keep = np.flatnonzero(counts >= min_count)
    keep = keep[np.argsort(-counts[keep])][:24]
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)])
    groups = [idx[order[starts[k]:starts[k + 1]]] for k in keep]
    cols = np.stack([(uniq[keep] >> 16) & 255, (uniq[keep] >> 8) & 255, uniq[keep] & 255], 1).astype(np.uint8) if len(keep) else np.zeros((0, 3), np.uint8)
    return cols, counts[keep], groups


def _clip_lab(L: np.ndarray) -> np.ndarray:
    """Lab -> displayable sRGB (clipped, 8-bit) -> Lab: what a renderer shows for a requested colour."""
    rgb = np.clip(lab2rgb(np.asarray(L, np.float64).reshape(1, -1, 3)), 0.0, 1.0)
    return rgb2lab(np.round(rgb * 255.0) / 255.0).reshape(-1, 3)


def detect_tint(T: np.ndarray, C: np.ndarray) -> Optional[dict]:
    """Page-level colour transfer: a constant Lab offset ``v`` applied to every target colour (then
    gamut-clipped, as a renderer would) that maps the target colours onto the candidate's flat palette
    above them (the forward direction: clipping is not invertible). Reported when the offset is visible, at least ``tint_min_colors`` palette colours move, and
    the one-offset model explains most of the palette error (``tint_max_resid``)."""
    from scipy.optimize import minimize
    H, W = C.shape[:2]
    cols, counts, groups = _palette(C, max(30, int(_p("palette_min_frac") * H * W)))
    if len(cols) < 2:
        return None
    LTf = _lab(T).reshape(-1, 3)
    LCc = _lab(cols.reshape(1, -1, 3)).reshape(-1, 3).astype(np.float64)
    tgt = np.stack([np.median(LTf[g], axis=0) for g in groups]).astype(np.float64)
    v0 = tgt - LCc
    de = np.linalg.norm(v0, axis=1)
    w = counts.astype(np.float64) / counts.sum()
    shifted = de >= _p("tint_min_de")
    n_shift = int(shifted.sum())
    before = float((w * de).sum())
    if n_shift < int(P[_K + "tint_min_colors"]) or before < _p("tint_min_de") * 0.5:
        return None
    if float(w[shifted].sum()) < _p("tint_min_cover"):
        return None

    def cost(v: np.ndarray) -> float:
        return float((w * np.linalg.norm(_clip_lab(tgt + v) - LCc, axis=1)).sum())

    starts = [-np.median(v0[shifted], axis=0), -(v0[shifted] * w[shifted, None]).sum(0) / w[shifted].sum(),
              -v0[shifted][np.argmax(de[shifted])]]
    best = min((minimize(cost, s0, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-3, "maxiter": 300}) for s0 in starts),
               key=lambda r: r.fun)
    v = np.asarray(best.x, np.float64)
    after = float(best.fun)
    if np.linalg.norm(v) < _p("tint_min_de") or after > _p("tint_max_resid") * before:
        return None
    shown = np.linalg.norm(_clip_lab(tgt + v) - tgt, axis=1)
    return {"magnitude": float(np.mean(shown)), "offset": [round(float(x), 3) for x in v], "cover": round(float(w[shifted].sum()), 3),
            "colors": int(len(cols)), "shifted_colors": n_shift, "palette_error": [round(before, 3), round(after, 3)], "v": v}


def apply_tint(C: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply a global tint: every pixel's Lab colour + ``v``, gamut-clipped (used on the target, so the
    later layers compare like with like)."""
    L = _lab(C).astype(np.float64) + np.asarray(v, np.float64)
    return np.clip(np.round(lab2rgb(L) * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- layer 1: noise model
class _Scale:
    """One analysis scale: both images blurred by ``sigma``, the sub-pixel-compensated residual, its page
    noise model and the per-pixel excess over the tolerance."""

    def __init__(self, T: np.ndarray, C: np.ndarray, sigma: float, h: float, mult: float = 1.0):
        self.sigma, self.h = sigma, h
        self.H, self.W = T.shape[:2]
        self.LTb, self.LCb = _lab(np.clip(_blur(T, sigma), 0, 255) / 255.0), _lab(np.clip(_blur(C, sigma), 0, 255) / 255.0)
        self.D = _tolerant_de(self.LTb, self.LCb, h)
        self.G = np.maximum(_grad(self.LTb), _grad(self.LCb))
        self.noise = self._noise_model()
        # tau0: the page noise model; tau: the detection tolerance (a multiple of it at the fine scale).
        # Hypotheses are judged against tau0 (explained = residual removed beyond the noise model).
        self.tau0 = (self.noise.tau_flat + _p("edge_mult") * self.noise.k_edge * self.G).astype(np.float32)
        self.tau = (mult * self.tau0).astype(np.float32)
        self.excess = np.maximum(0.0, self.D - self.tau)
        self.excess0 = np.maximum(0.0, self.D - self.tau0)

    def _noise_model(self) -> NoiseModel:
        """Page-level renderer-noise model from tiles, robust to the few local errors a page carries:
        flat level = a high quantile of the flat-pixel residual inside each mostly-flat tile, then the
        ``tile_q`` quantile over tiles; edge level ``k`` = residual per unit gradient over edge-rich tiles."""
        D, G = self.D, self.G
        t = int(P[_K + "tile"])
        Hc, Wc = (self.H // t) * t, (self.W // t) * t
        fl, k, n = 0.0, 0.0, 0
        if Hc and Wc:
            Dt = D[:Hc, :Wc].reshape(Hc // t, t, Wc // t, t).transpose(0, 2, 1, 3).reshape(-1, t * t)
            Gt = G[:Hc, :Wc].reshape(Hc // t, t, Wc // t, t).transpose(0, 2, 1, 3).reshape(-1, t * t)
            flat = Gt < _p("flat_grad")
            okf = flat.mean(1) >= 0.25
            if okf.sum() >= 4:
                Df = np.where(flat[okf], Dt[okf], np.nan)
                per = np.nanquantile(Df, _p("flat_q"), axis=1)
                fl = float(np.quantile(per, _p("tile_q")))
            Gs, Ds = Gt.sum(1), Dt.sum(1)
            ok = Gs >= _p("tile_min_grad")
            n = int(ok.sum())
            if n >= 4:
                k = float(np.quantile(Ds[ok] / Gs[ok], _p("edge_q")))
        tau_flat = min(_p("tau_max"), max(_p("tau_min"), _p("flat_mult") * fl))
        return NoiseModel((0.0, 0.0), tau_flat, min(_p("edge_k_max"), max(_p("edge_k_min"), k)), fl, n)

    def cover(self) -> float:
        """Share of edge-rich tiles whose residual per unit gradient exceeds ``cover_ratio``."""
        t = int(P[_K + "tile"])
        Hc, Wc = (self.H // t) * t, (self.W // t) * t
        if not Hc or not Wc:
            return 0.0
        Ds = self.D[:Hc, :Wc].reshape(Hc // t, t, Wc // t, t).sum((1, 3))
        Gs = self.G[:Hc, :Wc].reshape(Hc // t, t, Wc // t, t).sum((1, 3))
        ok = Gs >= _p("tile_min_grad")
        return float((Ds[ok] / Gs[ok] > _p("cover_ratio")).mean()) if ok.sum() >= 4 else 0.0

    def shifted_excess(self, xs: np.ndarray, ys: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Excess of T(x) vs C(x + d) at the given pixels, compensated over the sub-pixel grid."""
        t = self.LTb[ys, xs]
        best = None
        for sx, sy in _grid(self.h):
            c = _sample(self.LCb, xs + dx + sx, ys + dy + sy)
            d = np.linalg.norm(t - c, axis=1)
            best = d if best is None else np.minimum(best, d)
        return np.maximum(0.0, best - self.tau0[ys, xs])


class _Page:
    """Everything computed once per (target, candidate) pair: two analysis scales (fine: thin strokes,
    1 px geometry, small recolourings; coarse: robust to anti-aliasing / font smoothing / JPEG), the
    combined excess, and lazily the dense flow."""

    def __init__(self, T: np.ndarray, C: np.ndarray):
        self.T, self.C = T, C
        self.H, self.W = T.shape[:2]
        self.LT, self.LC = _lab(T), _lab(C)
        h = _p("subpixel")
        self.scales = [_Scale(T, C, _p("sigma_fine"), h, _p("fine_mult")), _Scale(T, C, _p("blur_sigma"), h)]
        # renderer noise is page-wide, errors are local: the share of edge-rich tiles that carry any
        # fine-scale residual tells a noisy capture from a clean one. On a noisy capture the fine scale,
        # the segment colour layer and the corner probe (all built for subtle changes) are switched off.
        self.noise_cover = self.scales[0].cover()
        self.noisy = self.noise_cover >= _p("noisy_cover")
        if self.noisy:
            self.scales[0].excess = np.zeros_like(self.scales[0].excess)
        self.excess = np.maximum(self.scales[0].excess, self.scales[1].excess)
        self.noise = self.scales[1].noise
        self._flow: Optional[np.ndarray] = None
        self.grayT = cv2.cvtColor(T, cv2.COLOR_RGB2GRAY)
        self.grayC = cv2.cvtColor(C, cv2.COLOR_RGB2GRAY)
        self.Gc = _grad(self.LC)

    @property
    def flow(self) -> np.ndarray:
        if self._flow is None:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self._flow = dis.calc(self.grayT, self.grayC, None)
        return self._flow

    def view(self, k: int) -> "_View":
        return _View(self, self.scales[k])


class _View:
    """A page seen at one scale (what the per-region hypotheses work on)."""

    def __init__(self, pg: _Page, sc: _Scale):
        self.pg, self.sc = pg, sc
        self.H, self.W, self.LT, self.LC = pg.H, pg.W, pg.LT, pg.LC
        self.grayT, self.grayC = pg.grayT, pg.grayC
        self.LTb, self.LCb, self.tau, self.excess, self.h = sc.LTb, sc.LCb, sc.tau0, sc.excess0, sc.h
        self.all_excess = pg.excess

    @property
    def flow(self) -> np.ndarray:
        return self.pg.flow

    def shifted_excess(self, xs, ys, dx, dy):
        return self.sc.shifted_excess(xs, ys, dx, dy)


# --------------------------------------------------------------------------- regions
def _regions(pg: _Page) -> list[Region]:
    mask = pg.excess > 0
    if not mask.any():
        return []
    dil = int(P[_K + "dilate"])
    grown = ndimage.binary_dilation(mask, iterations=dil) if dil > 0 else mask
    lab, n = ndimage.label(grown)
    if n == 0:
        return []
    lab_m = np.where(mask, lab, 0)
    idx = np.arange(1, n + 1)
    pxs = ndimage.sum(mask, lab_m, idx)
    mass = ndimage.sum(pg.excess, lab_m, idx)
    sl = ndimage.find_objects(lab_m)
    out = []
    for i, s in enumerate(sl):
        if s is None or pxs[i] < int(P[_K + "min_px"]) or mass[i] < _p("min_mass"):
            continue
        ys, xs = s
        m = lab_m[s] == i + 1
        r = Region(i, xs.start, ys.start, xs.stop, ys.stop, m, int(pxs[i]), float(mass[i]))
        per = [float(sc.excess[s][m].sum()) for sc in pg.scales]
        r.scale = int(np.argmax(per))
        r.evidence["scale_mass"] = [round(v, 1) for v in per]
        out.append(r)
    out.sort(key=lambda r: -r.mass)
    return out[: int(P[_K + "max_regions"])]


def _pixels(r: Region) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(r.mask)
    return xs + r.x0, ys + r.y0


# --------------------------------------------------------------------------- IR helpers
class _IR:
    def __init__(self, doc: Optional[Document], W: int, H: int):
        self.doc = doc
        self.nodes: list[Node] = []
        self.parent: dict[str, Node] = {}
        self.max_area = _p("max_node_frac") * W * H
        if doc is not None:
            for n in doc.walk():
                for c in n.children:
                    self.parent[c.id] = n
                if n.id != doc.root.id and n.visible:
                    self.nodes.append(n)

    def __bool__(self) -> bool:
        return self.doc is not None

    def owner(self, b: Box, frac: float = 0.6, pad: float = 2.0) -> Optional[Node]:
        """Smallest node whose (padded) box covers ``frac`` of ``b``."""
        best = None
        for n in self.nodes:
            nb = n.box.expand(pad)
            if n.box.area > self.max_area or n.box.area <= 0:
                continue
            if nb.intersect(b).area >= frac * max(b.area, 1.0):
                if best is None or n.box.area < best.box.area:
                    best = n
        return best

    def overlapping(self, b: Box) -> list[Node]:
        return [n for n in self.nodes if n.box.intersect(b).area > 0 and n.box.area <= self.max_area]

    def lca(self, a_id: str, b_id: str) -> Optional[Node]:
        """Lowest common ancestor (inclusive) of two nodes."""
        chain, cur = [], self.doc.find(a_id) if self.doc else None
        while cur is not None:
            chain.append(cur.id)
            cur = self.parent.get(cur.id)
        cur = self.doc.find(b_id) if self.doc else None
        while cur is not None:
            if cur.id in chain:
                return cur
            cur = self.parent.get(cur.id)
        return None

    def slot(self, b: Box) -> Optional[Box]:
        """The empty place of a missing sibling: a row/column of equally pitched children with one pitch
        doubled; returns the gap's box when it contains the centre of ``b``."""
        for par in [self.doc.root] + self.nodes if self.doc else []:
            kids = [c for c in par.children if c.visible and c.box.area > 0]
            if len(kids) < 3:
                continue
            for axis in (1, 0):
                ks = sorted(kids, key=lambda c: (c.box.y, c.box.x) if axis else (c.box.x, c.box.y))
                lo = [c.box.y if axis else c.box.x for c in ks]
                ext = [c.box.h if axis else c.box.w for c in ks]
                cross = [(c.box.x, c.box.w) if axis else (c.box.y, c.box.h) for c in ks]
                if max(abs(a[0] - cross[0][0]) for a in cross) > 2 or max(abs(a[1] - cross[0][1]) for a in cross) > 2:
                    continue
                pitch = [b2 - a2 for a2, b2 in zip(lo, lo[1:])]
                if len(pitch) < 2:
                    continue
                med = float(np.median(pitch))
                if med <= 2:
                    continue
                for i, pv in enumerate(pitch):
                    if abs(pv - 2 * med) <= 0.25 * med:
                        size = float(np.median(ext))
                        st = lo[i] + med
                        box = Box(cross[0][0], st, cross[0][1], size) if axis else Box(st, cross[0][0], size, cross[0][1])
                        if box.expand(2).contains_point(b.cx, b.cy):
                            return box
        return None

    def ancestors(self, n: Node) -> list[Node]:
        out, cur = [], self.parent.get(n.id)
        while cur is not None:
            out.append(cur)
            cur = self.parent.get(cur.id)
        return out


def _paints_fill(n: Node) -> bool:
    return any(f.kind == "solid" and f.color is not None and f.color.a > 0.02 for f in n.fills)


# --------------------------------------------------------------------------- layer 2: geometry
def _flow_proposals(pg: _Page, r: Region, xs: np.ndarray, ys: np.ndarray) -> list[tuple[int, int]]:
    F = pg.flow[ys, xs]
    mag = np.hypot(F[:, 0], F[:, 1])
    sel = F[mag >= _p("flow_min")]
    out: list[tuple[int, int]] = []
    if len(sel):
        rv = np.round(sel).astype(int)
        keys, cnt = np.unique(rv, axis=0, return_counts=True)
        for k in np.argsort(-cnt)[:3]:
            out.append((int(keys[k][0]), int(keys[k][1])))
    return out


def _template_proposals(pg: _Page, r: Region) -> list[tuple[int, int]]:
    S = int(P[_K + "search"])
    x0, y0, x1, y1 = max(0, r.x0 - 2), max(0, r.y0 - 2), min(pg.W, r.x1 + 2), min(pg.H, r.y1 + 2)
    out = []
    for src, dst, sign in ((pg.grayT, pg.grayC, 1), (pg.grayC, pg.grayT, -1)):
        patch = src[y0:y1, x0:x1]
        if patch.size < 16 or float(patch.std()) < 4.0:
            continue
        X0, Y0, X1, Y1 = max(0, x0 - S), max(0, y0 - S), min(pg.W, x1 + S), min(pg.H, y1 + S)
        area = dst[Y0:Y1, X0:X1]
        if area.shape[0] < patch.shape[0] or area.shape[1] < patch.shape[1]:
            continue
        res = cv2.matchTemplate(area.astype(np.float32), patch.astype(np.float32), cv2.TM_SQDIFF)
        _, _, loc, _ = cv2.minMaxLoc(res)
        dx, dy = loc[0] + X0 - x0, loc[1] + Y0 - y0
        if (dx, dy) != (0, 0):
            out.append((sign * dx, sign * dy))
    return out


def _best_shift(pg: _Page, r: Region) -> tuple[Optional[tuple[int, int]], float]:
    xs, ys = _pixels(r)
    if not len(xs):
        return None, 0.0
    cands = set(_flow_proposals(pg, r, xs, ys)) | set(_template_proposals(pg, r))
    more = set()
    for dx, dy in cands:
        for ex in (-1, 0, 1):
            for ey in (-1, 0, 1):
                more.add((dx + ex, dy + ey))
    cands = {c for c in cands | more if c != (0, 0) and math.hypot(*c) <= int(P[_K + "search"]) + 2}
    base = pg.excess[ys, xs].sum()
    best, best_e = None, 0.0
    for dx, dy in sorted(cands):
        rest = pg.shifted_excess(xs, ys, dx, dy).sum()
        e = 1.0 - rest / max(base, 1e-6)
        if e > best_e + 1e-9 or (abs(e - best_e) <= 1e-9 and best is not None and math.hypot(dx, dy) < math.hypot(*best)):
            best, best_e = (dx, dy), e
    return best, best_e


# --------------------------------------------------------------------------- layer 3: colour
def _affine_explained(src: np.ndarray, dst: np.ndarray, base: np.ndarray, tau: np.ndarray, fit_src: np.ndarray, fit_dst: np.ndarray) -> float:
    X = np.hstack([fit_src, np.ones((len(fit_src), 1), np.float32)])
    sol, *_ = np.linalg.lstsq(X, fit_dst, rcond=None)
    pred = np.hstack([src, np.ones((len(src), 1), np.float32)]) @ sol
    rest = np.maximum(0.0, np.linalg.norm(pred - dst, axis=1) - tau).sum()
    return 1.0 - rest / max(base.sum(), 1e-6)


def _colour_explained(pg: _Page, r: Region) -> tuple[float, dict]:
    """Per-region affine colour transfer, fitted target<-candidate and candidate<-target on the region's
    window (changed pixels plus their unchanged surround). A recolouring is invertible and keeps edges,
    so both directions must explain the region."""
    x0, y0, x1, y1 = max(0, r.x0 - 3), max(0, r.y0 - 3), min(pg.W, r.x1 + 3), min(pg.H, r.y1 + 3)
    T = pg.LTb[y0:y1, x0:x1].reshape(-1, 3)
    C = pg.LCb[y0:y1, x0:x1].reshape(-1, 3)
    if len(T) > 40000:
        sel = np.random.default_rng(0).choice(len(T), 40000, replace=False)
        Tf, Cf = T[sel], C[sel]
    else:
        Tf, Cf = T, C
    xs, ys = _pixels(r)
    t, c = pg.LTb[ys, xs], pg.LCb[ys, xs]
    base = pg.excess[ys, xs]
    tau = pg.tau[ys, xs]
    fwd = _affine_explained(c, t, base, tau, Cf, Tf)
    bwd = _affine_explained(t, c, base, tau, Tf, Cf)
    return min(fwd, bwd), {"fwd": round(float(fwd), 3), "bwd": round(float(bwd), 3)}


def _colour_magnitude(page: "_Page", r: Region) -> float:
    """ΔE76 of a recolouring: the median Lab offset over the clearly-changed half of the region's pixels
    (unblurred, so flat fills and glyph cores give the colour difference itself)."""
    xs, ys = _pixels(r)
    d = page.LT[ys, xs] - page.LC[ys, xs]
    n = np.linalg.norm(d, axis=1)
    sel = n >= np.quantile(n, 0.5) if len(n) else n > 0
    return float(np.linalg.norm(np.median(d[sel], axis=0))) if sel.any() else 0.0


def _blend_alpha(pg: _Page, r: Region) -> tuple[float, float]:
    """Fit candidate = alpha * target + (1 - alpha) * backdrop on the region; returns (alpha, explained)."""
    x0, y0, x1, y1 = max(0, r.x0 - 3), max(0, r.y0 - 3), min(pg.W, r.x1 + 3), min(pg.H, r.y1 + 3)
    bg = _ring_median(pg.LTb, x0, y0, x1, y1)
    xs, ys = _pixels(r)
    t, c = pg.LTb[ys, xs] - bg, pg.LCb[ys, xs] - bg
    den = float((t * t).sum())
    if den <= 1e-6:
        return 1.0, 0.0
    a = float((t * c).sum() / den)
    rest = np.maximum(0.0, np.linalg.norm(a * t - c, axis=1) - pg.tau[ys, xs]).sum()
    return a, 1.0 - rest / max(pg.excess[ys, xs].sum(), 1e-6)


# --------------------------------------------------------------------------- layer 4: ink ownership + shape
def _ink(pg: _Page, r: Region, pad: int = 3) -> dict:
    x0, y0, x1, y1 = max(0, r.x0 - pad), max(0, r.y0 - pad), min(pg.W, r.x1 + pad), min(pg.H, r.y1 + pad)
    LT, LC = pg.LT[y0:y1, x0:x1], pg.LC[y0:y1, x0:x1]
    bgT, bgC = _ring_median(pg.LT, x0, y0, x1, y1), _ring_median(pg.LC, x0, y0, x1, y1)
    thr = _p("ink_de")
    inkT = np.linalg.norm(LT - bgT, axis=2) > thr
    inkC = np.linalg.norm(LC - bgC, axis=2) > thr
    m = np.zeros_like(inkT)
    m[r.y0 - y0:r.y1 - y0, r.x0 - x0:r.x1 - x0] = r.mask
    m = ndimage.binary_dilation(m, iterations=1)
    t_own = int((inkT & ~inkC & m).sum())
    c_own = int((inkC & ~inkT & m).sum())
    both = int((inkT & inkC & m).sum())
    inter = int((inkT & inkC).sum())
    union = int((inkT | inkC).sum())
    # do the candidate's edges exist (however faint) in the target? a recoloured element keeps its outline,
    # an extra element has none in the target
    gT, gC = _grad(LT), _grad(LC)
    ce = (gC > 4.0) & ndimage.binary_dilation(m, iterations=2)
    edge_agree = float(ndimage.binary_dilation(gT > 1.0, iterations=1)[ce].mean()) if ce.any() else 0.0
    def stats(ink: np.ndarray) -> dict:
        sub = ink & m
        if not sub.any():
            return {"px": 0}
        ys, xs = np.nonzero(sub)
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        _, ncc = ndimage.label(sub)
        return {"px": int(sub.sum()), "w": bw, "h": bh, "density": float(sub.sum() / (bw * bh)), "ncc": int(ncc),
                "box": [int(xs.min() + x0), int(ys.min() + y0), bw, bh]}
    return {"t_own": t_own, "c_own": c_own, "both": both, "iou": inter / union if union else 1.0, "edge_agree": edge_agree,
            "inkT": stats(inkT), "inkC": stats(inkC), "inkU": stats(inkT | inkC), "bg_de": float(np.linalg.norm(bgT - bgC))}


def _shape_kind(st: dict) -> str:
    """text / icon / shape from ink statistics (row-like glyph runs vs compact glyph vs solid)."""
    if not st or st.get("px", 0) == 0:
        return "none"
    w, h, dens, ncc = st["w"], st["h"], st["density"], st["ncc"]
    if 8 <= w <= 56 and 8 <= h <= 56 and 0.5 <= w / h <= 2.0 and dens < 0.75 and ncc <= 4:
        return "icon"
    if h <= 48 and dens < 0.65 and (ncc >= 3 or w / max(h, 1) >= 2.5):
        return "text"
    return "shape"


# --------------------------------------------------------------------------- typing with the IR
def _edge_changes(pg: _Page, b: Box, band: int = 2) -> dict[str, float]:
    """Fraction of each box edge (a ``band``-px strip straddling it) that carries excess residual."""
    H, W = pg.H, pg.W
    x0, y0, x1, y1 = (int(round(v)) for v in (b.x, b.y, b.x2, b.y2))
    out = {}
    ex = pg.all_excess > 0
    def frac(xa, ya, xb, yb) -> float:
        xa, ya, xb, yb = max(0, xa), max(0, ya), min(W, xb), min(H, yb)
        if xb <= xa or yb <= ya:
            return 0.0
        sub = ex[ya:yb, xa:xb]
        # per position along the edge: any changed pixel across the band
        along = sub.any(axis=0) if (yb - ya) <= 2 * band + 1 and (xb - xa) > (yb - ya) else sub.any(axis=1)
        return float(along.mean())
    out["left"] = frac(x0 - band, y0, x0 + band, y1)
    out["right"] = frac(x1 - band, y0, x1 + band, y1)
    out["top"] = frac(x0, y0 - band, x1, y0 + band)
    out["bottom"] = frac(x0, y1 - band, x1, y1 + band)
    return out


def _corner_region(n: Node, b: Box, pad: float = 1.5) -> bool:
    """Is region box ``b`` confined to one corner square of node ``n``'s box?"""
    nb = n.box
    if not nb.expand(pad).contains(b):
        return False
    side = max(max(n.radius) + 2.0, 6.0) + 20.0
    side = min(side, min(nb.w, nb.h) / 2 + 2)
    for cx, cy in ((nb.x, nb.y), (nb.x2, nb.y), (nb.x, nb.y2), (nb.x2, nb.y2)):
        if abs(b.cx - cx) <= side and abs(b.cy - cy) <= side and max(b.w, b.h) <= side + 2:
            near_x = min(abs(b.x - cx), abs(b.x2 - cx)) <= pad + 1
            near_y = min(abs(b.y - cy), abs(b.y2 - cy)) <= pad + 1
            if near_x or near_y:
                return True
    return False


def _border_share(n: Node, xs: np.ndarray, ys: np.ndarray, width: float, tol: float = 1.0) -> float:
    b = n.box
    dx = np.minimum(np.abs(xs + 0.5 - b.x), np.abs(xs + 0.5 - b.x2))
    dy = np.minimum(np.abs(ys + 0.5 - b.y), np.abs(ys + 0.5 - b.y2))
    inside = (xs + 0.5 >= b.x - tol) & (xs + 0.5 <= b.x2 + tol) & (ys + 0.5 >= b.y - tol) & (ys + 0.5 <= b.y2 + tol)
    on = inside & (np.minimum(dx, dy) <= width + tol)
    return float(on.mean()) if len(xs) else 0.0


def _ring_shape(r: Region) -> bool:
    """Image-only stroke test: changed pixels hug the region's bbox border, interior untouched."""
    h, w = r.mask.shape
    if min(h, w) < 8:
        return False
    ys, xs = np.nonzero(r.mask)
    d = np.minimum.reduce([xs, w - 1 - xs, ys, h - 1 - ys])
    return float((d <= 3).mean()) >= 0.85 and r.px <= 0.5 * h * w


def _text_sibling(ir: _IR, n: Node, side: str) -> Optional[Node]:
    par = ir.parent.get(n.id)
    if par is None or n.text_style is None:
        return None
    for s in par.children:
        if s is n or s.type != "text" or s.text_style is None:
            continue
        if abs(s.box.y - n.box.y) > 2 or abs(s.text_style.size - n.text_style.size) > 0.1:
            continue
        gap = (n.box.x - s.box.x2) if side == "left" else (s.box.x - n.box.x2)
        if -2 <= gap <= 2.5 * n.text_style.size:
            return s
    return None


def _analyse(page: _Page, ir: _IR, r: Region) -> None:
    """Pass 1: evidence for every hypothesis on one region (no decision yet)."""
    pg = page.view(r.scale)
    b = r.bbox
    ink = _ink(pg, r)
    r.evidence["ink"] = {k: ink[k] for k in ("t_own", "c_own", "both", "iou")}
    r.evidence["_ink"] = ink
    if ir:
        tn = [n for n in ir.overlapping(b) if n.type == "text"]
        cov = sum(n.box.expand(2).intersect(b).area for n in tn)
        r.evidence["text_like"] = bool(tn) and cov >= 0.6 * max(b.area, 1.0)
    else:
        # image only: a run of small glyph-like components in either image
        u = ink["inkU"]
        r.evidence["text_like"] = _shape_kind(u) == "text" or (u.get("px", 0) > 0 and u["h"] <= 24 and u["density"] < 0.6 and u["ncc"] >= 2)
    d, e_shift = _best_shift(pg, r)
    e_col, col_ev = _colour_explained(pg, r)
    r.evidence["explained"] = {"shift": round(float(e_shift), 3), "colour": round(float(e_col), 3), **col_ev}
    if d is not None:
        r.evidence["d"] = list(d)


def _one_sided(ink: dict) -> Optional[str]:
    """'t' / 'c' when one image owns (almost) all of a region's differing ink, None when both do."""
    t, c = ink["t_own"], ink["c_own"]
    tot = t + c
    if tot < 10:
        return None
    if c < 0.15 * tot:
        return "t"
    if t < 0.15 * tot:
        return "c"
    return None


def _partners(regs: list[Region]) -> None:
    """A moved element whose old and new footprints are separate regions: a target-owned region and a
    candidate-owned region related by the same displacement."""
    for r in regs:
        d = r.evidence.get("d")
        side = _one_sided(r.evidence["_ink"])
        if not d or side is None:
            continue
        for q in regs:
            if q is r or _one_sided(q.evidence["_ink"]) in (None, side):
                continue
            # target-owned (old) region + d = candidate-owned (new) region
            moved = r.bbox.translate(d[0], d[1]) if side == "t" else r.bbox.translate(-d[0], -d[1])
            if moved.expand(2).iou(q.bbox.expand(2)) >= 0.4:
                r.evidence["partner"] = q.id
                q.evidence["partner"] = r.id
                q.evidence["d"] = list(d)
                r.evidence["partner_box"] = [q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0]
                q.evidence["partner_box"] = [r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0]
                q.evidence["explained"]["shift"] = max(q.evidence["explained"]["shift"], r.evidence["explained"]["shift"])
                break


def _classify(page: _Page, ir: _IR, r: Region) -> None:
    """Pass 2: pick the explanation (layer order: corner/seam structure with the IR, shadow, geometry,
    colour, residual structure) and type it."""
    pg = page.view(r.scale)
    xs, ys = _pixels(r)
    b = r.bbox
    em = _p("explain_min")
    ink = r.evidence["_ink"]
    owner = ir.owner(b) if ir else None
    d = tuple(r.evidence["d"]) if r.evidence.get("d") else None
    e_shift, e_col = r.evidence["explained"]["shift"], r.evidence["explained"]["colour"]

    # ---- changes confined to one corner of a shape: radius
    if ir:
        for n in ir.overlapping(b):
            if n.type in ("frame", "rect", "instance", "image") and min(n.box.w, n.box.h) >= 12 and _corner_region(n, b):
                if _paints_fill(n) or n.strokes:
                    r.layer, r.type, r.node_id, r.quality = "geometry", "geometry.radius", n.id, 0.8
                    r.box = n.box
                    r.evidence["corner_of"] = n.id
                    return
        # ---- a seam inside / between shapes: split and merge
        sp = None if r.evidence.get("text_like") else _seam(pg, ir, sorted(ir.overlapping(b), key=lambda n: n.box.area), b, r)
        if sp is not None:
            r.layer = "structure"
            r.type, r.magnitude, r.node_id, r.box = sp
            r.quality = 0.8
            return

    # ---- shadow: soft one-signed luminance band outside a node box
    if ir:
        dl = pg.LTb[ys, xs, 0] - pg.LCb[ys, xs, 0]
        sign_share = max(float((dl > 0).mean()), float((dl < 0).mean()))
        for n in sorted(ir.overlapping(b.expand(2)), key=lambda n: n.box.area):
            if n.type not in ("frame", "rect", "instance", "ellipse") or min(n.box.w, n.box.h) < 8:
                continue
            nb = n.box
            outside = ~((xs + 0.5 >= nb.x) & (xs + 0.5 <= nb.x2) & (ys + 0.5 >= nb.y) & (ys + 0.5 <= nb.y2))
            near = (xs >= nb.x - 26) & (xs <= nb.x2 + 26) & (ys >= nb.y - 26) & (ys <= nb.y2 + 26)
            if outside.mean() >= 0.6 and near.mean() >= 0.95 and sign_share >= 0.85 and nb.expand(26).contains(b) \
                    and (b.w >= 0.6 * nb.w or b.h >= 0.6 * nb.h) and e_shift < 0.9:
                r.layer, r.type, r.node_id, r.quality = "effect", "effect.shadow", n.id, float(outside.mean())
                r.box = b.union(nb)
                return

    # ---- geometry: a displacement explains the region, and it moved ink (both footprints, or a partner)
    shift_min = _p("text_shift_min") if r.evidence.get("text_like") else em
    moved_ink = _one_sided(ink) is None or "partner" in r.evidence
    fwd, bwd = r.evidence["explained"].get("fwd", e_col), r.evidence["explained"].get("bwd", e_col)
    inked = ink["inkT"].get("px", 0) + ink["inkC"].get("px", 0) > 0
    # an element painted by one image only is structure, never a recolouring
    single = _one_sided(ink) is not None and ink["both"] < 0.05 * max(ink["t_own"] + ink["c_own"], 1) and ink["edge_agree"] < 0.5
    colour_ok = not single and (e_col >= _p("colour_min") or (max(fwd, bwd) >= em and inked and ink["iou"] >= 0.8))
    if colour_ok and r.evidence.get("text_like") and ink["iou"] < 0.7:
        colour_ok = False  # re-weighted / re-sized glyphs also look like an intensity change
    if ir and d is not None and e_shift >= em and moved_ink and not colour_ok and _spacing_ir(pg, ir, r, d):
        r.layer, r.quality = "geometry", float(e_shift)
        return
    if d is not None and e_shift >= shift_min and e_shift >= e_col and moved_ink:
        r.layer, r.quality = "geometry", float(e_shift)
        r.magnitude = float(math.hypot(*d))
        _type_geometry(pg, ir, r, d, owner)
        return

    # ---- colour: invertible per-region transfer (both directions; or one direction with unchanged ink shape)
    if colour_ok:
        r.layer, r.quality = "colour", float(e_col)
        r.magnitude = _colour_magnitude(page, r)
        alpha, e_blend = _blend_alpha(pg, r)
        r.evidence["alpha"] = round(alpha, 3)
        r.evidence["blend_explained"] = round(float(e_blend), 3)
        fading = None
        if ir:
            cov = [n for n in ir.overlapping(b) if n.box.expand(2).contains(b)]
            fading = next((n for n in sorted(cov, key=lambda n: n.box.area) if n.opacity < 0.99), None)
            stroked = [n for n in ir.overlapping(b) if n.strokes and n.box.expand(2).contains(b) and n.box.iou(b) >= 0.4]
            for n in sorted(stroked, key=lambda n: n.box.area):
                if _border_share(n, xs, ys, max(s.width for s in n.strokes), 1.0 + 1.5 * page.scales[r.scale].sigma) >= 0.7:
                    r.type, r.node_id = "color.stroke", n.id
                    r.box = b.union(n.box)
                    return
        elif _ring_shape(r):
            r.type = "color.stroke"
            return
        if fading is not None and e_blend >= em and 0.0 < alpha < 0.97:
            r.type, r.node_id, r.magnitude = "effect.opacity", fading.id, float(1.0 - alpha)
            r.box = b.union(fading.box) if fading.box.area <= 4 * max(b.area, 1) else b
            return
        if not ir and e_blend >= em and 0.0 < alpha < 0.9 and ink["inkT"].get("ncc", 0) >= 3 and ink["iou"] >= 0.8:
            r.type, r.magnitude = "effect.opacity", float(1.0 - alpha)
            return
        if ir and r.evidence.get("text_like"):
            tn = next((n for n in sorted(ir.overlapping(b), key=lambda n: n.box.area) if n.type == "text"), None)
            if tn is not None and _text_change(pg, tn, ink, colour=True) == "text.style":
                r.layer, r.type, r.node_id, r.magnitude = "structure", "text.style", tn.id, None
                return
        r.type = "color.fill"
        if owner is not None:
            r.node_id = owner.id
        return

    # ---- structure: the residual, by ink ownership and shape
    r.layer = "structure"
    _type_structure(pg, ir, r, ink, owner)


def _node_disp(pg, box: Box, axis: int, S: int) -> Optional[int]:
    """Displacement (px, along ``axis``) of the candidate content in ``box`` relative to the target: the
    candidate patch is matched in the target within +-S along the axis (+-1 across it)."""
    x0, y0, x1, y1 = _clip(box.expand(1), pg.W, pg.H)
    patch = pg.grayC[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 9 or float(patch.std()) < 3.0:
        return None
    ex, ey = (S, 1) if axis == 0 else (1, S)
    X0, Y0, X1, Y1 = max(0, x0 - ex), max(0, y0 - ey), min(pg.W, x1 + ex), min(pg.H, y1 + ey)
    area = pg.grayT[Y0:Y1, X0:X1].astype(np.float32)
    if area.shape[0] < patch.shape[0] or area.shape[1] < patch.shape[1]:
        return None
    res = cv2.matchTemplate(area, patch, cv2.TM_SQDIFF)
    _, _, loc, _ = cv2.minMaxLoc(res)
    # candidate content at x0 matches the target at X0 + loc -> it moved by x0 - (X0 + loc)
    return int((x0 - (X0 + loc[0])) if axis == 0 else (y0 - (Y0 + loc[1])))


def _spacing_ir(pg, ir: _IR, r: Region, d: tuple[int, int]) -> bool:
    """layout.spacing: the region's nodes are siblings of one row/column whose later members moved by
    increasing multiples of one step (each sibling's displacement measured by template matching)."""
    axis = 0 if abs(d[0]) >= abs(d[1]) else 1
    S = int(P[_K + "search"])
    seen: set[str] = set()
    for n in sorted(ir.overlapping(r.bbox), key=lambda n: n.box.area):
        par = ir.parent.get(n.id)
        if par is None or par.id in seen:
            continue
        seen.add(par.id)
        kids = sorted([c for c in par.children if c.visible and c.box.area > 0], key=lambda c: (c.box.x, c.box.y)[axis])
        if len(kids) < 3:
            continue
        lo = lambda c: c.box.x if axis == 0 else c.box.y
        hi = lambda c: c.box.x2 if axis == 0 else c.box.y2
        if not all(hi(a) <= lo(b) + 0.5 for a, b in zip(kids, kids[1:])):
            continue
        hit = [c for c in kids if c.box.expand(abs(d[axis]) + 2).intersect(r.bbox).area > 0]
        if not hit:
            continue
        disp = [_node_disp(pg, c.box, axis, S) for c in kids]
        nz = [(c, v) for c, v in zip(kids, disp) if v]
        if len(nz) < 2 or len({abs(v) for _, v in nz}) < 2:
            continue
        signs = {v > 0 for _, v in nz}
        vals = [v for _, v in nz]
        mono = all(abs(b) >= abs(a) for a, b in zip(vals, vals[1:]))
        if len(signs) != 1 or not mono:
            continue
        step = min(abs(v) for v in vals)
        box = None
        for c, v in nz:
            e = c.box.union(c.box.translate(-v if axis == 0 else 0, -v if axis == 1 else 0))
            box = e if box is None else box.union(e)
        r.type, r.magnitude, r.node_id, r.box = "layout.spacing", float(step), par.id, box
        r.evidence["sibling_disp"] = [v for _, v in nz]
        return True
    return False


def _type_geometry(pg, ir: _IR, r: Region, d: tuple[int, int], owner: Optional[Node]) -> None:
    """Type a displacement: which node moved, and which of its edges carry residual."""
    b = r.bbox
    if r.evidence.get("partner_box"):
        b = b.union(Box(*r.evidence["partner_box"]))
    dx, dy = d
    if ir and _spacing_ir(pg, ir, r, d):
        return
    if ir:
        # the node that moved: the largest node lying inside the region (its old + new footprints), else
        # the smallest node mostly inside it
        inside = [n for n in ir.overlapping(b) if n.box.area >= 4 and n.box.intersect(b.expand(3)).area >= 0.9 * n.box.area]
        moved = max(inside, key=lambda n: n.box.area) if inside else None
        if moved is None:
            for n in sorted(ir.overlapping(b), key=lambda n: n.box.area):
                if n.box.area >= 4 and n.box.intersect(b).area >= 0.5 * n.box.area:
                    moved = n
                    break
        if moved is None:
            moved = owner
        if moved is not None:
            r.node_id = moved.id
            nb = moved.box
            if moved.type == "text":
                left = _text_sibling(ir, moved, "left")
                if left is not None and abs(dy) <= 1 and dx != 0:
                    r.type, r.magnitude = "structure.split", float(abs(dx))
                    r.box = b.union(left.box)
                    return
                # only the right part of the run moved: one candidate run stands for two target runs
                if abs(dy) <= 1 and dx != 0:
                    x0, y0, x1, y1 = _clip(nb, pg.W, pg.H)
                    cols = np.flatnonzero((np.linalg.norm(pg.LC[y0:y1, x0:x1] - _ring_median(pg.LC, max(0, x0 - 1), max(0, y0 - 1),
                                                                                              min(pg.W, x1 + 1), min(pg.H, y1 + 1)), axis=2) > _p("ink_de")).any(0))
                    if len(cols) and b.x > x0 + cols[0] + 0.15 * (cols[-1] - cols[0] + 1):
                        r.type, r.magnitude = "structure.merge", float(abs(dx))
                        r.box = b.union(nb)
                        return
                # the run's painted container moved too (its border carries residual): the container shifted
                par = ir.parent.get(moved.id)
                while par is not None and not (_paints_fill(par) or par.strokes) and par.box.area <= ir.max_area:
                    par = ir.parent.get(par.id)
                if par is not None and (_paints_fill(par) or par.strokes) and par.box.area <= ir.max_area:
                    ec = _edge_changes(pg, par.box)
                    if sum(v >= 0.3 for v in ec.values()) >= 2:
                        r.type, r.node_id = "geometry.shift", par.id
                        r.box = b.union(par.box).union(par.box.translate(-dx, -dy))
                        return
                r.type = "text.position"
                return
            # size vs shift: which edges of the moved node carry residual. Only a node that paints its own
            # box (fill / stroke / shadow) has edges that can move independently; icons and bare containers shift
            if moved.type not in SHAPE_TYPES or not (_paints_fill(moved) or moved.strokes or moved.effects):
                r.type = "geometry.shift"
                r.box = b.union(nb).union(nb.translate(-dx, -dy)) if nb.area <= 4 * max(b.area, 1) else b
                return
            ec = _edge_changes(pg, nb)
            r.evidence["edges"] = {k: round(v, 2) for k, v in ec.items()}
            hit = {k: v >= 0.5 for k, v in ec.items()}
            if dx != 0 and hit["left"] != hit["right"] and not (dy != 0 and hit["top"] and hit["bottom"]):
                r.type, r.magnitude = "geometry.size", float(max(abs(dx), abs(dy)))
                r.box = b.union(nb)
                return
            if dy != 0 and hit["top"] != hit["bottom"] and not (dx != 0 and hit["left"] and hit["right"]):
                r.type, r.magnitude = "geometry.size", float(max(abs(dx), abs(dy)))
                r.box = b.union(nb)
                return
            r.type = "geometry.shift"
            r.box = b.union(nb).union(nb.translate(-dx, -dy)) if nb.area <= 4 * max(b.area, 1) else b
            return
    # image only: text-like ink -> text.position; a thin strip along one side -> size; else shift
    ink = _ink(pg, r)
    kind = _shape_kind(ink["inkT"] if ink["inkT"].get("px", 0) >= ink["inkC"].get("px", 0) else ink["inkC"])
    if kind == "text":
        r.type = "text.position"
        return
    thin = min(b.w, b.h) <= max(abs(dx), abs(dy)) + 3
    r.type = "geometry.size" if thin and (ink["t_own"] == 0 or ink["c_own"] == 0) else "geometry.shift"


def _type_structure(pg: _Page, ir: _IR, r: Region, ink: dict, owner: Optional[Node]) -> None:
    b = r.bbox
    t_own, c_own = ink["t_own"], ink["c_own"]
    tot = t_own + c_own
    of = _p("own_frac")
    r.quality = max(t_own, c_own) / tot if tot else 0.5
    st = ink["inkT"] if t_own >= c_own else ink["inkC"]
    kind = _shape_kind(st)
    r.evidence["shape"] = kind
    xs, ys = _pixels(r)
    if ir:
        cands = sorted(ir.overlapping(b), key=lambda n: n.box.area)
        pxT, pxC = ink["inkT"].get("px", 0), ink["inkC"].get("px", 0)
        # text / icon nodes whose ink changed (both images paint there)
        tn = next((n for n in cands if n.type in ("text", "icon") and n.box.expand(3).intersect(b).area >= 0.4 * b.area), None)
        if tn is not None and not (tot and t_own >= of * tot and tn.box.expand(3).intersect(b).area < 0.6 * b.area) \
                and pxT >= 0.25 * max(pxC, 1):
            r.node_id = tn.id
            if tn.type == "icon":
                if tot and t_own >= of * tot and pxC < 0.2 * max(pxT, 1):
                    r.type, r.node_id = "icon.missing", None
                else:
                    r.type = "icon.glyph"
                    r.box = b.union(tn.box)
                return
            r.type = _text_change(pg, tn, ink)
            return
        # candidate paints an element the target lacks: a node that sits on the region
        if tot and c_own >= of * tot:
            n = next((n for n in cands if n.box.expand(2).iou(b) >= 0.3 or (b.expand(2).contains(n.box) and n.box.area >= 0.3 * b.area)
                      or (n.box.expand(2).intersect(b).area >= 0.7 * b.area and n.box.area <= 8 * max(b.area, 1))), None)
            if n is not None:
                r.type, r.node_id = "structure.extra", n.id
                r.box = b.union(n.box) if n.box.area <= 8 * max(b.area, 1) else b
                r.magnitude = float(n.box.area)
                return
        if tot and t_own >= of * tot:
            r.type = "icon.missing" if kind == "icon" else "structure.missing"
            r.magnitude = float(b.area)
            sl = ir.slot(b)
            if sl is not None and sl.area >= b.area:
                # the target element sits in the candidate's gap between equally pitched siblings
                r.type, r.box, r.magnitude = "structure.missing", sl.union(b), float(sl.area)
                r.evidence["slot"] = [round(v, 1) for v in (sl.x, sl.y, sl.w, sl.h)]
            return
        if owner is not None and owner.type in SHAPE_TYPES:
            ec = _edge_changes(pg, owner.box)
            hit = [k for k, v in ec.items() if v >= 0.5]
            if 1 <= len(hit) <= 2 and owner.box.area > 2 * b.area:
                r.type, r.node_id = "geometry.size", owner.id
                r.box = b.union(owner.box)
                return
        r.type = "structure.extra" if c_own > t_own else "structure.missing"
        return
    # image only
    if tot and t_own >= of * tot:
        r.type = "icon.missing" if kind == "icon" else "structure.missing"
    elif tot and c_own >= of * tot:
        r.type = "structure.extra"
    elif kind == "text":
        r.type = _text_change(pg, None, ink)
    elif kind == "icon":
        r.type = "icon.glyph"
    else:
        r.type = "structure.extra" if c_own > t_own else "structure.missing"


SHAPE_TYPES = ("frame", "rect", "instance", "ellipse", "line", "image", "vector")


def _text_mixed(n: Node, ink: dict) -> bool:
    return n.type == "text" and ink["inkT"].get("px", 0) > 0.3 * max(ink["inkC"].get("px", 1), 1)


def _text_change(pg, n: Optional[Node], ink: dict, colour: bool = False) -> str:
    """text.style vs text.content. A style change (size or weight) touches every glyph of the run and
    changes its ink coherently: scaled (ink height and width move the same way) or re-weighted (same
    width, different ink mass). A content change leaves part of the run intact or changes its ink
    incoherently. Measured over the whole text node when the IR gives it, else over the region."""
    if n is not None:
        b = n.box.expand(2)
        x0, y0, x1, y1 = _clip(b, pg.W, pg.H)
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            bgT = _ring_median(pg.LT, x0, y0, x1, y1)
            bgC = _ring_median(pg.LC, x0, y0, x1, y1)
            thr = _p("ink_de")
            iT = np.linalg.norm(pg.LT[y0:y1, x0:x1] - bgT, axis=2) > thr
            iC = np.linalg.norm(pg.LC[y0:y1, x0:x1] - bgC, axis=2) > thr
            ch = pg.all_excess[y0:y1, x0:x1] > 0
            def ext(m: np.ndarray) -> dict:
                if not m.any():
                    return {"px": 0}
                ys, xs = np.nonzero(m)
                return {"px": int(m.sum()), "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1)}
            a, c = ext(iT), ext(iC)
            run = np.flatnonzero((iT | iC).any(0))
            colfrac = float(ch.any(0)[run[0]:run[-1] + 1].mean()) if len(run) else 0.0
        else:
            a, c, colfrac = ink["inkT"], ink["inkC"], 1.0
    else:
        a, c = ink["inkT"], ink["inkC"]
        colfrac = 1.0
    if not a.get("px") or not c.get("px"):
        return "text.content"
    rh, rw = a["h"] / max(c["h"], 1), a["w"] / max(c["w"], 1)
    scaled = (rh >= 1.05 and rw >= 1.03) or (rh <= 0.95 and rw <= 0.97)
    weighted = abs(rw - 1.0) <= 0.1 and abs(math.log(max(a["px"], 1) / max(c["px"], 1))) >= _p("style_density")
    if colour:  # a recoloured run keeps its ink: only a coherent re-weighting / re-sizing counts as style
        return "text.style" if colfrac >= _p("style_cols") and (scaled or weighted) else "text.content"
    if colfrac >= _p("style_cols") and (scaled or weighted) or colfrac >= 0.97:
        return "text.style"
    return "text.content"


def _seam(pg, ir: _IR, cands: list[Node], b: Box, r: Optional["Region"] = None) -> Optional[tuple[str, float, Optional[str], Box]]:
    """Shape split: two aligned same-type siblings with a thin gap at the region. Shape merge: the
    region is a strip across the interior of one candidate shape, and the target shows the backdrop
    (the colour just outside that shape) inside the strip -- the gap between two elements was filled."""
    for n in cands:
        if n.type not in ("rect", "frame", "instance", "line") or not (_paints_fill(n) or n.strokes):
            continue
        par = ir.parent.get(n.id)
        if par is None:
            continue
        for s in par.children:
            if s is n or s.type != n.type or not s.visible:
                continue
            if abs(s.box.y - n.box.y) <= 1 and abs(s.box.h - n.box.h) <= 1 and s.box.x >= n.box.x2 - 0.5:
                gap = s.box.x - n.box.x2
                seam = Box(n.box.x2, n.box.y, max(gap, 0.5), n.box.h)
            elif abs(s.box.x - n.box.x) <= 1 and abs(s.box.w - n.box.w) <= 1 and s.box.y >= n.box.y2 - 0.5:
                gap = s.box.y - n.box.y2
                seam = Box(n.box.x, n.box.y2, n.box.w, max(gap, 0.5))
            else:
                continue
            both = n.box.union(s.box)
            reach = max([abs(e.dx) + abs(e.dy) + e.blur + max(0.0, e.spread) for e in n.effects if not e.inner] or [0.0])
            across = b.w if seam.h >= seam.w else b.h
            if 0.5 <= gap <= 16 and seam.expand(3).intersect(b).area > 0 and both.expand(4 + reach).contains(b) \
                    and across <= gap + 10 + 2 * reach:
                if r is not None:
                    xs, ys = _pixels(r)
                    sb = seam.expand(1.5 + reach)
                    share = float(((xs + 0.5 >= sb.x) & (xs + 0.5 <= sb.x2) & (ys + 0.5 >= sb.y) & (ys + 0.5 <= sb.y2)).mean())
                    if share < 0.6:
                        continue
                return "structure.split", float(gap), n.id, both
    for n in cands:
        if n.type not in ("rect", "frame", "instance") or not (_paints_fill(n) or n.strokes):
            continue
        nb = n.box
        if nb.expand(3).intersect(b).area < 0.9 * b.area or nb.area > ir.max_area:
            continue
        thick = _p("merge_max_strip")
        vert = b.h >= 0.85 * nb.h and b.w <= min(0.4 * nb.w, thick) and b.x > nb.x + 2 and b.x2 < nb.x2 - 2
        horiz = b.w >= 0.85 * nb.w and b.h <= min(0.4 * nb.h, thick) and b.y > nb.y + 2 and b.y2 < nb.y2 - 2
        if not (vert or horiz):
            continue
        x0, y0, x1, y1 = _clip(nb.expand(3), pg.W, pg.H)
        X0, Y0, X1, Y1 = _clip(nb.expand(1), pg.W, pg.H)
        ring = np.ones((y1 - y0, x1 - x0), bool)
        ring[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] = False
        if not ring.any():
            continue
        backdrop = np.median(pg.LC[y0:y1, x0:x1][ring], axis=0)
        if vert:
            cx = int(round(b.cx))
            core = pg.LT[int(b.y + 0.2 * b.h):int(b.y2 - 0.2 * b.h) + 1, max(0, cx - 1):cx + 1].reshape(-1, 3)
        else:
            cy = int(round(b.cy))
            core = pg.LT[max(0, cy - 1):cy + 1, int(b.x + 0.2 * b.w):int(b.x2 - 0.2 * b.w) + 1].reshape(-1, 3)
        if len(core) and float(np.linalg.norm(np.median(core, axis=0) - backdrop)) < _p("merge_backdrop_de"):
            return "structure.merge", float(min(b.w, b.h)), n.id, nb
    return None


# --------------------------------------------------------------------------- layer 3b: colour on candidate segments
def _estimate_blur(T: np.ndarray, C: np.ndarray) -> float:
    """Renderer-noise model: the extra Gaussian blur of the target relative to the candidate (resampling /
    anti-aliasing differences), fitted on edge pixels of the whole page (robust to a few local errors)."""
    gT = cv2.cvtColor(T, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gC = cv2.cvtColor(C, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.Sobel(gC, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gC, cv2.CV_32F, 0, 1) ** 2
    m = g > np.quantile(g, 0.9) if g.size else g > 0
    best, best_s = None, 0.0
    for sg in (0.0, 0.35, 0.5, 0.65, 0.8):
        b = cv2.GaussianBlur(gC, (0, 0), sg) if sg > 0 else gC
        e = float(np.median(np.abs(gT[m] - b[m]))) if m.any() else 0.0
        if best is None or e < best - 0.05:
            best, best_s = e, sg
    return best_s


def _segment_regions(page: _Page, occupied: np.ndarray, ir: "_IR") -> list[Region]:
    """Per-segment colour transfer on the candidate's flat-colour segments (4-connected runs of one exact
    render colour, so 1 px borders and icon/glyph cores are segments of their own). Each segment is
    registered over the sub-pixel grid against the target (with the page's estimated extra blur applied
    to the candidate) and its median Lab offset is measured; a consistent offset >= ``seg_de`` is a
    recolouring the pixel-level layers were too coarse to see. Segments already inside a region are left
    to it."""
    from skimage.measure import label
    C, T = page.C, page.T
    H, W = C.shape[:2]
    packed = (C[..., 0].astype(np.int32) << 16) | (C[..., 1].astype(np.int32) << 8) | C[..., 2].astype(np.int32)
    lab = label(packed, connectivity=1, background=-1)
    counts = np.bincount(lab.ravel())
    lo, hi = int(P[_K + "seg_min_px"]), _p("max_node_frac") * H * W
    ids = np.flatnonzero((counts >= lo) & (counts <= hi))
    if not len(ids):
        return []
    sig = _estimate_blur(T, C)
    LCt = _lab(np.clip(_blur(C, sig), 0, 255) / 255.0) if sig > 0 else page.LC
    sl = ndimage.find_objects(lab)
    text_boxes = [n.box.expand(1) for n in ir.nodes if n.type == "text"] if ir else []
    cands: list[tuple[tuple, np.ndarray, Optional[str]]] = []
    for i in ids:
        s = sl[i - 1]
        if s is None:
            continue
        bb = _box_of(s[1].start, s[0].start, s[1].stop, s[0].stop)
        if any(tb.contains(bb) for tb in text_boxes):
            continue  # glyph cores are pooled per run below
        cands.append((s, lab[s] == i, None))
    # glyph runs and icons: pool the pixels of the node's exact ink colour (AA leaves only small cores per glyph)
    for n in (ir.nodes if ir else []):
        if n.type not in ("text", "icon"):
            continue
        x0, y0, x1, y1 = _clip(n.box, W, H)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        win = packed[y0:y1, x0:x1]
        vals, cnt = np.unique(win, return_counts=True)
        bgv = vals[np.argmax(cnt)]
        keep = vals != bgv
        if not keep.any():
            continue
        ink = vals[keep][np.argmax(cnt[keep])]
        m = win == ink
        if m.sum() >= lo:
            cands.append(((slice(y0, y1), slice(x0, x1)), m, n.id))
    tested: list[tuple[float, float, np.ndarray, np.ndarray, int, tuple]] = []
    for s, m, nid in cands:
        ys, xs = np.nonzero(m)
        ys, xs = ys + s[0].start, xs + s[1].start
        t = page.LT[ys, xs]
        best = None
        for sx, sy in _grid(_p("subpixel")):
            o = t - _sample(LCt, xs + sx, ys + sy)
            med = np.median(o, axis=0)
            mag = float(np.linalg.norm(med))
            if best is None or mag < best[0]:
                best = (mag, med, o)
        mag, med, o = best
        cons = float(((o @ med) / max(mag * mag, 1e-9) >= 0.5).mean()) if mag > 0 else 0.0
        # the target shows its backdrop where the candidate paints: that is missing/moved ink, not a recolouring
        X0, Y0, X1, Y1 = max(0, s[1].start - 2), max(0, s[0].start - 2), min(W, s[1].stop + 2), min(H, s[0].stop + 2)
        backdrop = _ring_median(page.LT, X0, Y0, X1, Y1)
        shows_backdrop = float(np.linalg.norm(np.median(t, axis=0) - backdrop)) < 0.5 * _p("ink_de") \
            and float(np.linalg.norm(np.median(page.LC[ys, xs], axis=0) - backdrop)) >= _p("ink_de")
        occ = bool(occupied[ys, xs].mean() > 0.2) or shows_backdrop
        tested.append((mag, cons, med, m, int(len(xs)), (s, occ, nid)))
    if not tested:
        return []
    # the page's own segment-offset distribution is the noise model (JPEG chroma, sub-pixel snapping,
    # resampling blur), per class: thin segments (borders, dividers) are far more exposed than solid ones
    def thin(m: np.ndarray) -> bool:
        return m.sum() / float(m.shape[0] + m.shape[1]) <= 2.5
    thr = {}
    for cls in (True, False):
        mags = [t[0] for t in tested if thin(t[3]) == cls]
        q = float(np.quantile(mags, _p("seg_noise_q"))) if len(mags) >= 4 else 0.0
        thr[cls] = max(_p("seg_de"), _p("seg_noise_mult") * q)
    out: list[Region] = []
    for mag, cons, med, m, n, (s, occ, nid) in tested:
        if occ or mag < thr[thin(m)] or cons < _p("seg_consistency"):
            continue
        r = Region(-1 - len(out), s[1].start, s[0].start, s[1].stop, s[0].stop, m, n, float(mag * n))
        if nid is not None:
            r.evidence["pool_of"] = nid
        r.scale = 0
        r.layer, r.magnitude, r.quality = "colour", mag, cons
        r.evidence.update({"segment": True, "offset_lab": [round(float(v), 2) for v in med], "target_blur": sig,
                           "seg_threshold": round(thr[thin(m)], 2), "scale_mass": [r.mass, r.mass]})
        out.append(r)
    return out


def _type_segment(page: _Page, ir: "_IR", r: Region) -> None:
    """stroke when the segment is a node's border (IR) or a thin ring (image); opacity when a covering node
    is translucent; else fill."""
    xs, ys = _pixels(r)
    b = r.bbox
    r.type = "color.fill"
    if ir and r.evidence.get("pool_of"):
        n = ir.doc.find(r.evidence["pool_of"])
        r.evidence["text_like"] = n is not None and n.type == "text"
        fading = next((a for a in [n] + ir.ancestors(n) if a.opacity < 0.99 and a.box.area <= ir.max_area), None) if n is not None else None
        r.type, r.node_id = ("effect.opacity", fading.id) if fading is not None else ("color.fill", r.evidence["pool_of"])
        r.box = (fading or n).box if n is not None else b
        if fading is not None:
            r.magnitude = None
        return
    if ir:
        tn = [n for n in ir.overlapping(b) if n.type == "text"]
        r.evidence["text_like"] = bool(tn) and sum(n.box.expand(2).intersect(b).area for n in tn) >= 0.6 * max(b.area, 1.0)
        for n in sorted([n for n in ir.overlapping(b) if n.strokes], key=lambda n: n.box.area):
            if n.box.expand(2).contains(b) and _border_share(n, xs, ys, max(st.width for st in n.strokes)) >= 0.8:
                r.type, r.node_id, r.box = "color.stroke", n.id, n.box
                return
        cov = sorted([n for n in ir.overlapping(b) if n.box.expand(2).contains(b)], key=lambda n: n.box.area)
        fading = next((n for n in cov if n.opacity < 0.99), None)
        if fading is not None:
            r.type, r.node_id, r.box = "effect.opacity", fading.id, fading.box
            r.magnitude = None
            return
        if cov:
            r.node_id = cov[0].id
            if cov[0].box.area <= 4 * max(b.area, 1):
                r.box = b.union(cov[0].box)
        return
    r.evidence["text_like"] = False
    h, w = r.mask.shape
    if min(h, w) >= 6 and r.px <= 0.35 * h * w:
        ys0, xs0 = np.nonzero(r.mask)
        dd = np.minimum.reduce([xs0, w - 1 - xs0, ys0, h - 1 - ys0])
        if float((dd <= 2).mean()) >= 0.85:
            r.type = "color.stroke"


# --------------------------------------------------------------------------- geometry 2b: corner probe (radius)
def _corner_regions(page: _Page, ir: "_IR", regs: list[Region]) -> list[Region]:
    """geometry.radius: residual confined to the four corners of a shape (inside its box), with the straight
    parts of its border clean. Shapes come from the candidate IR when given, else from the candidate's
    large flat segments."""
    ex = page.scales[0].excess0 > 0
    H, W = page.H, page.W
    boxes: list[tuple[Box, float, Optional[str]]] = []
    if ir:
        for n in ir.nodes:
            if n.type in ("frame", "rect", "instance", "image") and (_paints_fill(n) or n.strokes) and min(n.box.w, n.box.h) >= 12 \
                    and n.box.area <= ir.max_area:
                boxes.append((n.box, max(n.radius), n.id))
    out: list[Region] = []
    taken: list[Box] = []
    for nb, rad, nid in sorted(boxes, key=lambda t: t[0].area):
        x0, y0, x1, y1 = (int(round(v)) for v in (nb.x, nb.y, nb.x2, nb.y2))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 12 or y1 - y0 < 12 or any(t.iou(nb) > 0.8 for t in taken):
            continue
        sq = int(min(max(rad, 4.0) + _p("corner_reach"), (x1 - x0) / 2, (y1 - y0) / 2))
        sub = ex[y0:y1, x0:x1]
        cs = [int(sub[:sq, :sq].sum()), int(sub[:sq, -sq:].sum()), int(sub[-sq:, :sq].sum()), int(sub[-sq:, -sq:].sum())]
        band = np.zeros_like(sub)
        band[:2, :], band[-2:, :], band[:, :2], band[:, -2:] = True, True, True, True
        band[:sq, :sq] = band[:sq, -sq:] = band[-sq:, :sq] = band[-sq:, -sq:] = False
        mid = int((sub & band).sum())
        tot = sum(cs)
        inner_area = max(0, sub.shape[0] - 2 * sq) * max(0, sub.shape[1] - 2 * sq)
        inner_den = float(sub[sq:-sq, sq:-sq].sum()) / inner_area if inner_area else 0.0
        hh, ww = sub.shape
        centre = sub[hh // 4: hh - hh // 4, ww // 4: ww - ww // 4]
        centre_den = float(centre.mean()) if centre.size else 0.0
        if min(cs) < _p("corner_min_each") or min(cs) < 0.4 * max(cs) or tot < int(P[_K + "corner_min_px"]) \
                or mid > 0.15 * tot or inner_den > 0.2 * tot / (4.0 * sq * sq) or centre_den > 0.02:
            continue
        # one cause for all four corners: the target is consistently more (or less) filled than the candidate
        LTs, LCs = page.LT[y0:y1, x0:x1], page.LC[y0:y1, x0:x1]
        cy, cx = sub.shape[0] // 2, sub.shape[1] // 2
        fill = np.median(LCs[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3].reshape(-1, 3), axis=0)
        signs = []
        for ys_, xs_ in ((slice(0, sq), slice(0, sq)), (slice(0, sq), slice(-sq, None)), (slice(-sq, None), slice(0, sq)), (slice(-sq, None), slice(-sq, None))):
            k = sub[ys_, xs_]
            dd = np.linalg.norm(LCs[ys_, xs_][k] - fill, axis=1) - np.linalg.norm(LTs[ys_, xs_][k] - fill, axis=1)
            signs.append(float(np.mean(np.sign(dd))) if len(dd) else 0.0)
        # filled shapes: every corner gains (or loses) fill; outlines: the arcs move alike at every corner
        if max(signs) - min(signs) > 0.6:
            continue
        mask = np.zeros_like(sub)
        for ys_, xs_ in ((slice(0, sq), slice(0, sq)), (slice(0, sq), slice(-sq, None)), (slice(-sq, None), slice(0, sq)), (slice(-sq, None), slice(-sq, None))):
            mask[ys_, xs_] = sub[ys_, xs_]
        r = Region(-1000 - len(out), x0, y0, x1, y1, mask, int(mask.sum()),
                   float(page.scales[0].excess0[y0:y1, x0:x1][mask].sum()))
        r.scale = 0
        r.layer, r.type, r.node_id, r.box, r.quality = "geometry", "geometry.radius", nid, nb, 0.8
        # area between two quarter-circle corners of radii r1, r2 is (1 - pi/4) |r1^2 - r2^2|; the target is
        # rounder (less filled at the corners) when the corner sign is negative
        a = tot / 4.0 / (1.0 - math.pi / 4.0)
        rt = math.sqrt(max(0.0, rad * rad + (a if np.mean(signs) < 0 else -a)))
        r.magnitude = float(abs(rt - rad)) if rad > 0 or np.mean(signs) < 0 else None
        r.evidence.update({"corners_px": cs, "edge_px": mid, "corner_sign": [round(v, 2) for v in signs],
                           "scale_mass": [r.mass, r.mass], "text_like": False})
        out.append(r)
        taken.append(nb)
    return out


# --------------------------------------------------------------------------- systematic (page-wide) noise
def _systematic(regs: list[Region], noisy: bool = False) -> tuple[list[Region], list[Finding]]:
    """Renderer differences are page-wide and homogeneous (every text run is re-rasterised, every edge
    moves by the same sub-pixel snap); an error is a local outlier. When at least ``pop_min`` regions are
    weak-evidence candidates -- text-like, or explained by a 1 px displacement -- the population's median
    excess mass sets the bar: members below ``pop_mult`` x median are attributed to the renderer and
    reported once as an explicit ``noise.*`` abstention."""
    pop = [r for r in regs if r.evidence.get("text_like") or (r.layer == "geometry" and r.evidence.get("d")
                                                             and max(abs(v) for v in r.evidence["d"]) <= 1)]
    cm = lambda r: r.evidence.get("scale_mass", [r.mass, r.mass])[-1]
    weak: set[int] = set()
    bar = 0.0
    if len(pop) >= int(P[_K + "pop_min"]):
        bar = _p("pop_mult") * float(np.median([cm(r) for r in pop]))
        weak = {id(r) for r in pop if cm(r) < bar}
    if noisy:
        # on a noisy capture (page-wide residual) a weak region is indistinguishable from the renderer
        weak |= {id(r) for r in regs if cm(r) < _p("noisy_min_mass") and "corners_px" not in r.evidence}
    if not weak:
        return regs, []
    gone = [r for r in regs if id(r) in weak]
    n_geo = sum(1 for r in gone if r.layer == "geometry")
    kind = "noise.subpixel" if n_geo > len(gone) / 2 else "noise.antialias"
    box = gone[0].bbox
    for r in gone[1:]:
        box = box.union(r.bbox)
    ev = {"layer": "noise", "regions": len(gone), "population": len(pop), "bar_mass": round(bar, 1),
          "mass": round(sum(r.mass for r in gone), 1)}
    for r in gone:
        r.layer, r.type = "noise", kind
    return [r for r in regs if id(r) not in weak], [Finding(kind, box, None, 1.0, ev)]


# --------------------------------------------------------------------------- grouping
def _near(a: Box, b: Box, gap: float) -> bool:
    return a.expand(gap / 2).intersect(b.expand(gap / 2)).area > 0


def _empty_between(page: Optional["_Page"], a: Box, b: Box) -> bool:
    """Same row band, and the candidate paints nothing between the two boxes (a removed row / bar whose
    pieces show up as separate target-only fragments)."""
    if page is None:
        return False
    top, bot = max(a.y, b.y), min(a.y2, b.y2)
    if bot - top < 0.6 * min(a.h, b.h):
        return False
    left, right = (a, b) if a.x <= b.x else (b, a)
    x0, x1 = int(math.ceil(left.x2)), int(math.floor(right.x))
    if x1 - x0 < 1:
        return True
    band = page.Gc[int(top):int(math.ceil(bot)), x0:x1]
    return band.size > 0 and float((band > 4.0).mean()) < 0.02


def _group(regions: list[Region], ir: _IR, page: Optional["_Page"] = None) -> list[Region]:
    """Merge regions that one cause explains: same node and same type (radius corners, a faded subtree's
    glyphs), the same displacement close together (a moved element's leading/trailing strips and its
    moved children), or the same colour transform close together; siblings moving by multiples of one
    step become one layout.spacing."""
    gap = _p("group_gap")
    out: list[Region] = []
    for r in sorted(regions, key=lambda r: (r.type != "layout.spacing", -r.mass)):
        merged = False
        for g in out:
            same_node = r.node_id is not None and r.node_id == g.node_id and r.type == g.type
            same_cause = r.type == g.type and r.layer in ("colour", "effect") and _near(r.box or r.bbox, g.box or g.bbox, gap)
            rd, gd = r.evidence.get("d"), g.evidence.get("d")
            if r.layer == g.layer == "geometry" and rd and gd and _same_d(r, g) \
                    and r.type != "layout.spacing" and g.type != "layout.spacing" \
                    and _near(r.box or r.bbox, g.box or g.bbox, gap + max(map(abs, rd + gd))):
                same_cause = True
            if r.evidence.get("slot") and r.evidence.get("slot") == g.evidence.get("slot"):
                same_cause = True
            # fragments of one moved subtree (the same displacement inside their lowest common ancestor)
            if ir and not same_cause and r.layer == g.layer == "geometry" and rd and gd and _same_d(r, g) \
                    and r.node_id and g.node_id and "layout.spacing" not in (r.type, g.type):
                a = ir.lca(r.node_id, g.node_id)
                dd = max(map(abs, rd + gd)) + 3
                if a is not None and a.box.area <= ir.max_area and a.box.expand(dd).contains(r.bbox) and a.box.expand(dd).contains(g.bbox):
                    same_cause = True
                    if a.id not in (r.node_id, g.node_id):
                        g.node_id = a.id
                        g.type = "geometry.shift" if a.type != "text" else g.type
                        g.box = (g.box or g.bbox).union(a.box).union(a.box.translate(-gd[0], -gd[1]))
            # one text run, one finding: a content edit also shifts the glyphs after it
            if r.node_id is not None and r.node_id == g.node_id and r.type in TEXT_FAMILY and g.type in TEXT_FAMILY:
                same_cause = True
                for t in ("text.content", "text.style"):
                    if t in (r.type, g.type):
                        g.type = t
                        break
            if r.layer == g.layer == "geometry" and (r.evidence.get("partner") == g.id or g.evidence.get("partner") == r.id):
                same_cause = True
            if r.type == g.type and r.type in ("text.content", "text.style", "structure.missing", "structure.extra") and _near(r.bbox, g.bbox, gap / 2) and r.node_id == g.node_id:
                same_cause = True
            # a fragment inside a larger region of the same family (an app bar removed with its icons)
            fam = {"structure.missing": "m", "icon.missing": "m", "structure.extra": "e"}
            if fam.get(r.type) and fam.get(r.type) == fam.get(g.type) and (g.box or g.bbox).expand(2).contains(r.bbox):
                same_cause = True
            if fam.get(r.type) == fam.get(g.type) == "m" and _empty_between(page, r.box or r.bbox, g.box or g.bbox):
                same_cause = True
                g.type = "structure.missing"
            if g.type == "layout.spacing" and r.layer == "geometry" and (g.box or g.bbox).expand(gap).contains(r.bbox):
                same_cause = True
            if same_node or same_cause:
                _absorb(g, r)
                merged = True
                break
        if not merged:
            r.members = [r.id]
            out.append(r)
    # a lone corner blob is not a radius change (the four corners of a shape change together)
    out = [g for g in out if not (g.type == "geometry.radius" and "corners_px" not in g.evidence and len(g.members) < 2)]
    return _spacing(out, ir)


TEXT_FAMILY = ("text.content", "text.style", "text.position")


def _same_d(r: Region, g: Region) -> bool:
    """Displacements agree within 1 px on every axis the regions constrain (the aperture problem: a long
    horizontal edge band only measures the vertical component, and vice versa)."""
    rd, gd = r.evidence["d"], g.evidence["d"]
    def free(x: Region) -> tuple[bool, bool]:
        w, h = x.x1 - x.x0, x.y1 - x.y0
        return (w >= 4 * h, h >= 4 * w)  # (dx unconstrained, dy unconstrained)
    fr, fg = free(r), free(g)
    okx = fr[0] or fg[0] or abs(rd[0] - gd[0]) <= 1
    oky = fr[1] or fg[1] or abs(rd[1] - gd[1]) <= 1
    return okx and oky


def _absorb(g: Region, r: Region) -> None:
    gb = (g.box or g.bbox).union(r.box or r.bbox)
    g.box = gb
    g.mass += r.mass
    g.px += r.px
    g.quality = max(g.quality, r.quality)
    g.members.append(r.id)


def _spacing(regs: list[Region], ir: _IR) -> list[Region]:
    geo = [r for r in regs if r.layer == "geometry" and r.type in ("geometry.shift", "text.position", "geometry.size") and r.evidence.get("d")]
    if len(geo) < 2:
        return regs
    used: set[int] = set()
    out_extra: list[Region] = []
    for axis in (0, 1):
        cand = [r for r in geo if r.evidence["d"][1 - axis] == 0 and r.evidence["d"][axis] != 0 and id(r) not in used]
        # cluster by parent (IR) or alignment (image)
        groups: dict = {}
        for r in cand:
            if ir and r.node_id is not None:
                par = ir.parent.get(r.node_id)
                key = par.id if par is not None else None
            else:
                key = round((r.box or r.bbox).cy / 8) if axis == 0 else round((r.box or r.bbox).cx / 8)
            groups.setdefault(key, []).append(r)
        for key, rs in groups.items():
            ds = sorted({abs(r.evidence["d"][axis]) for r in rs})
            if len(rs) < 2 or len(ds) < 2 or key is None:
                continue
            base = ds[0]
            if not all(abs(v / base - round(v / base)) < 0.2 for v in ds):
                continue
            g = rs[0]
            for r in rs[1:]:
                _absorb(g, r)
                used.add(id(r))
            g.type, g.magnitude, g.layer = "layout.spacing", float(base), "geometry"
            if ir and key is not None and isinstance(key, str):
                g.node_id = key
            used.add(id(g))
    return [r for r in regs if id(r) not in used or r.type == "layout.spacing"]


# --------------------------------------------------------------------------- entry points
def _same_size(T: np.ndarray, C: np.ndarray) -> np.ndarray:
    if T.shape == C.shape:
        return C
    H, W = T.shape[:2]
    out = np.zeros_like(T)
    out[...] = C.reshape(-1, C.shape[-1]).mean(0).astype(np.uint8) if C.size else 255
    h, w = min(H, C.shape[0]), min(W, C.shape[1])
    out[:h, :w] = C[:h, :w, :3]
    return out


def decompose(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> Decomposition:
    """Layered explanation of the target-vs-candidate difference (see the module docstring)."""
    t0 = time.perf_counter()
    T = np.ascontiguousarray(target_rgb[..., :3], dtype=np.uint8)
    C = _same_size(T, np.ascontiguousarray(candidate_rgb[..., :3], dtype=np.uint8))
    tm: dict[str, float] = {}
    findings: list[Finding] = []
    tint = detect_tint(T, C)
    tm["tint"] = time.perf_counter() - t0
    LT0 = None
    if tint is not None:
        LT0 = _lab(T)
        T = apply_tint(T, tint["v"])
        H, W = T.shape[:2]
        findings.append(Finding("color.tint_global", Box(0, 0, W, H), tint["magnitude"], min(1.0, 0.5 + tint["cover"] / 2),
                                {"layer": "colour", "offset_lab": tint["offset"], "cover": tint["cover"], "palette_error": tint["palette_error"]}))
    pg = _Page(T, C)
    tm["noise"] = time.perf_counter() - t0
    regs = _regions(pg)
    tm["regions"] = time.perf_counter() - t0
    ir = _IR(candidate_ir, pg.W, pg.H)
    for r in regs:
        _analyse(pg, ir, r)
    _partners(regs)
    for r in regs:
        _classify(pg, ir, r)
    for r in regs:
        r.evidence.pop("_ink", None)
    tm["classify"] = time.perf_counter() - t0
    # corner probe: radius changes are small and split into four blobs; it claims the regions inside its corners
    corners = [] if pg.noisy else _corner_regions(pg, ir, regs)
    for c in corners:
        for r in regs:
            if r.layer != "noise" and c.box.expand(2).contains(r.bbox) and r.type != "geometry.radius" and r.mass < 2 * c.mass + 200:
                r.layer, r.type, r.node_id = "geometry", "geometry.radius", c.node_id
                r.box = c.box
    regs.extend(corners)
    # colour on candidate segments, outside every region found so far
    occ = np.zeros((pg.H, pg.W), bool)
    for r in regs:
        x0, y0, x1, y1 = _clip(r.bbox.expand(_p("seg_margin")), pg.W, pg.H)
        occ[y0:y1, x0:x1] = True
    segs = [] if pg.noisy else _segment_regions(pg, occ, ir)
    for r in segs:
        _type_segment(pg, ir, r)
    regs.extend(segs)
    tm["segments"] = time.perf_counter() - t0
    if tint is not None:
        # colour residue of the global tint (colours the constant-offset model clips differently): a region
        # whose own target->candidate offset runs along the tint is the tint, not a local recolouring
        v = np.asarray(tint["v"], np.float64)
        nv = float(np.linalg.norm(v))
        absorbed = 0
        for r in regs:
            if r.layer not in ("colour", "effect") or nv <= 0:
                continue
            xs, ys = _pixels(r)
            o = np.median(pg.LC[ys, xs] - LT0[ys, xs], axis=0)
            no = float(np.linalg.norm(o))
            # aligned with the tint, or barely changed at all (the compensation itself made the difference,
            # e.g. translucent shadows the theme offset does not move)
            if no <= 0.5 * nv or (float(o @ v) / (no * nv) >= 0.8 and no <= 2.0 * nv):
                r.layer, r.type = "noise", "color.tint_global"
                absorbed += 1
        regs = [r for r in regs if r.layer != "noise"]
        findings[0].evidence["absorbed_regions"] = absorbed
    live, noise_findings = _systematic(regs, pg.noisy)
    groups = _group(live, ir, pg)
    cm = _p("conf_mass")
    for g in groups:
        conf = float(np.clip((0.4 + 0.6 * g.quality) * (1.0 - math.exp(-g.mass / cm)), 0.01, 1.0))
        ev = {"layer": g.layer, "mass": round(g.mass, 1), "px": g.px, "regions": len(g.members) or 1, **g.evidence}
        findings.append(Finding(g.type, (g.box or g.bbox), g.magnitude, conf, _jsonable(ev), g.node_id))
    findings.extend(noise_findings)
    tm["total"] = time.perf_counter() - t0
    return Decomposition(findings, regs, pg.noise, None if tint is None else {k: v for k, v in tint.items() if k != "v"},
                         pg.noise_cover, pg.noisy, time.perf_counter() - t0, {k: round(v, 3) for k, v in tm.items()})


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def find(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> list[Finding]:
    """The decomposition adversary (``dt.adversary.benchmark.AdversaryFn``)."""
    return decompose(target_rgb, candidate_rgb, candidate_ir).findings


find.__name__ = "decompose"


# --------------------------------------------------------------------------- real pages + report CLI
PARENT_RGB = {"geometry": (220, 40, 40), "color": (40, 90, 230), "structure": (200, 0, 200), "text": (240, 140, 0),
              "icon": (0, 160, 60), "effect": (0, 170, 190), "layout": (190, 170, 0), "noise": (150, 150, 150)}


def overlay(target: np.ndarray, candidate: np.ndarray, findings: list[Finding], max_w: int = 2400) -> np.ndarray:
    """Evidence panel: target | candidate | residual heatmap, findings boxed in their parent's colour."""
    from dt.compare.pixel import diff_map
    from dt.compare.visualize import draw_boxes, heatmap
    C = _same_size(target, candidate)
    panels = []
    for img in (target, C, heatmap(diff_map(target, C))):
        for f in findings:
            img = draw_boxes(img, [f.box], PARENT_RGB.get(f.parent, (255, 0, 0)), 1 if f.parent == "noise" else 2)
        panels.append(img)
    sep = np.full((target.shape[0], 6, 3), 255, np.uint8)
    im = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    s = min(1.0, max_w / im.shape[1])
    if s < 1.0:
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return im


def _validator_agreement(findings: list[Finding], val: dict, top: int = 8) -> dict:
    """How many of the independent validator's worst colour regions one of our (error) findings localises."""
    worst = [Box.from_dict(w["box"]) for w in (val.get("worst_regions") or [])[:top]]
    errs = [f for f in findings if f.parent != "noise"]
    hit = sum(1 for w in worst if any(f.box.expand(2).intersect(w).area > 0 for f in errs))
    return {"worst_regions": len(worst), "localised": hit}


def run_real(root: str, out_dir: Optional[str] = None) -> list[dict]:
    """Decompose every translated page under ``root`` (``<page>/validation/target.png`` vs ``render.png`` with
    ``ir.mapped.json``); optionally write overlays and per-page findings to ``out_dir``."""
    import glob
    import json
    import os
    from dt.common.image import load_rgb, save_rgb
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        tp, cp, ip = os.path.join(d, "validation", "target.png"), os.path.join(d, "render.png"), os.path.join(d, "ir.mapped.json")
        if not (os.path.exists(tp) and os.path.exists(cp)):
            continue
        name = os.path.basename(os.path.normpath(d))
        T, C = load_rgb(tp), load_rgb(cp)
        ir = Document.load(ip) if os.path.exists(ip) else None
        dec = decompose(T, C, ir)
        val = {}
        vp = os.path.join(d, "validation", "validation.json")
        if os.path.exists(vp):
            with open(vp) as f:
                val = json.load(f)
        counts: dict[str, int] = {}
        for f in dec.findings:
            counts[f.type] = counts.get(f.type, 0) + 1
        row = {"page": name, "size": [int(T.shape[1]), int(T.shape[0])], "seconds": round(dec.seconds, 3),
               "noisy": dec.noisy, "noise_cover": round(dec.noise_cover, 3), "tint": dec.tint, "counts": dict(sorted(counts.items())),
               "agreement": _validator_agreement(dec.findings, val), "validator": {k: val.get(k) for k in ("text_cer", "region_de_worst", "chamfer", "jnd_frac_nontext")},
               "findings": [f.to_dict() for f in sorted(dec.findings, key=lambda f: -f.confidence)]}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            save_rgb(overlay(T, C, dec.findings), os.path.join(out_dir, f"{name}.png"))
            with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
                json.dump(_jsonable(row), f, indent=1)
            row["overlay"] = os.path.join(out_dir, f"{name}.png")
        rows.append(row)
    return rows


def find_image_only(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> list[Finding]:
    """The same decomposition without the candidate IR (pure image evidence)."""
    return decompose(target_rgb, candidate_rgb, None).findings


def _layer_usage(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        for p in r["preds"]:
            k = p.evidence.get("layer", "?") + ("/segment" if p.evidence.get("segment") else "") + ("/probe" if "corners_px" in p.evidence else "")
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json
    import os
    from dt.adversary import benchmark as BM
    ap = argparse.ArgumentParser(description="Score the diff-decomposition adversary and run it on real translated pages.")
    ap.add_argument("--out", default=BM.DEFAULT_OUT)
    ap.add_argument("--real", default=os.path.join(BM.ROOT, "out", "real_r3m"))
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--extra-calibration", nargs="*", default=None,
                    help="extra calibration-document benchmarks (other seeds) to report the tuning objective on")
    a = ap.parse_args(argv)
    bench = BM.build(BM.BenchConfig(), os.path.join(a.out, "bench"))
    reports, layer_use = [], {}
    for fn, split, use_ir, name in ((find, "calibration", True, "decompose"), (find, "evaluation", True, "decompose"),
                                    (find_image_only, "evaluation", False, "decompose (image only)"),
                                    (BM.baseline_residual, "evaluation", True, "baseline_residual")):
        rows = BM.run(fn, bench.split(split), use_ir)
        reports.append(BM.aggregate(rows, name, split))
        layer_use[f"{name}/{split}"] = _layer_usage(rows)
    extra = []
    for d in (a.extra_calibration or []):
        eb = BM.Benchmark.load(d)
        rows = BM.run(find, eb.split("calibration"), True)
        extra.append((os.path.basename(os.path.normpath(d)), eb.config.seed, BM.aggregate(rows, "decompose", "calibration")))
    md = BM.format_report(reports, bench, "Adversary approach: diff decomposition")
    if extra:
        md += ("\n## Extended calibration (same calibration documents, other perturbation/noise seeds)\n\n"
               "| benchmark key | seed | cases | headline | det F1 | macro leaf F1 | noise false-case rate |\n|---|---|---|---|---|---|---|\n")
        for key, seed, r in extra:
            md += (f"| `{key}` | {seed} | {r['cases']} | {r['headline']:.3f} | {100 * r['detection']['f1']:.1f} | "
                   f"{100 * r['macro_leaf']['f1']:.1f} | {100 * r['noise']['false_case_rate']:.1f} |\n")
    md += "\n## Which layer produced the findings\n\n" + "\n".join(f"* {k}: " + ", ".join(f"{l} {n}" for l, n in v.items()) for k, v in layer_use.items()) + "\n"
    ex_dir = os.path.join(a.out, "decompose", "examples")
    ex = BM.save_examples(bench, find, ex_dir, n=6)
    md += "\n## Benchmark evidence\n\n" + "\n".join(f"* `{os.path.relpath(p, BM.ROOT)}` (green = ground truth, red = decomposition findings)" for p in ex) + "\n"
    real = [] if a.no_real else run_real(a.real, os.path.join(a.out, "decompose", "real"))
    if real:
        md += "\n## Real translated pages (out/real_r3m)\n\n| page | size | noisy (edge-tile cover) | tint | findings (error types) | validator worst regions localised | s |\n|---|---|---|---|---|---|---|\n"
        for r in real:
            errs = {k: v for k, v in r["counts"].items() if not k.startswith("noise.")}
            md += (f"| {r['page']} | {r['size'][0]}x{r['size'][1]} | {r['noisy']} ({r['noise_cover']:.2f}) | "
                   f"{'%.1f ΔE' % r['tint']['magnitude'] if r['tint'] else '–'} | " + ", ".join(f"`{k}` {v}" for k, v in errs.items())
                   + f" | {r['agreement']['localised']}/{r['agreement']['worst_regions']} | {r['seconds']:.2f} |\n")
        md += "\nOverlays: " + ", ".join(f"`{os.path.relpath(r['overlay'], BM.ROOT)}`" for r in real if r.get("overlay")) + "\n"
    with open(os.path.join(a.out, "decompose.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out, "decompose", "decompose.json"), "w") as f:
        json.dump(_jsonable({"reports": reports, "layer_use": layer_use, "real": real}), f, indent=1, default=str)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
