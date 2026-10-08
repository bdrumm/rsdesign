"""A-contrario adversary: statistical change detection with a controlled number of false alarms.

``find(target_rgb, candidate_rgb, candidate_ir=None) -> list[Finding]`` (see ``dt.adversary.taxonomy``).

1. **Nuisance registration** (``fit_nuisance``). The nuisance between a target screenshot and our render
   is mostly *global*: a sub-pixel page offset, the font-smoothing mode, the browser build, resampling
   blur, lossy compression, a page-wide colour offset. It is registered by re-rendering the candidate IR
   in the real renderer under those transformations (LayoutUnit-exact offsets, smoothing on/off, the
   alternate browser build) and fitting the post-raster part (blur sigma, JPEG quality) on the pixels.
   The fitted render explains the target exactly when only nuisance separates them.
2. **Local statistics** (``pixel_maps``/``tile_stats``) of target vs fitted render on a multi-scale tile
   family (8-64 px, stride 1/2): mean ΔE2000, the fraction above a JND (a ΔE quantile), GMSD at two
   scales, the three SSIM components and an edge-orientation-histogram distance. Each statistic is
   self-normalised by its own page quantile per content stratum (removes the page-level misfit).
3. **Null model** (``NullModel``): the empirical H0 law of every normalised statistic, stratified by the
   nuisance model that explains the page (exact registration / post-raster fit / no IR / unexplained /
   colour-offset compensated) and by tile content (gradient energy), zero-inflated, with a conservative
   generalised-Pareto tail. Calibrated on the CALIBRATION documents only (``calibrate``).
4. **A-contrario decision**: NFA(tile) = N_tests * min_j p_j; a tile is meaningful when NFA < eps, so the
   expected number of false detections per page is <= eps under H0. A page-wide colour offset is a single
   global test. Meaningful tiles -> changed-pixel support -> regions, joined by interpretable relations
   (motion, layout drift, shared corners, pieces of one missing subtree).
5. **Typing** (``region_features`` + ``Forest``): interpretable features (which statistic fired, block-
   matching flow and its reverse consistency, chroma vs luminance, ink presence in target vs candidate,
   text / icon / shape anchoring in the candidate IR, ring / interior / corner coverage) -> a bagged forest
   of CART trees (a single surrogate tree gives printable rules). Regions typed ``_none`` (nuisance-like
   or fragments) are returned as ``noise.antialias`` abstentions.

CLI: ``python -m dt.adversary.acontrario calibrate|evaluate|all``; the report is
``out/adversary/a-contrario.md``.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from dt.ir import Document
from dt.params import P, register

register("adversary.acontrario.fit.score_tol", 3, "nuisance fit: a pixel counts as unexplained when max |ΔRGB| > this", (1, 16))
register("adversary.acontrario.fit.translate_step", 1.0 / 64, "nuisance fit: sub-pixel translate search step (px; Chrome LayoutUnit = 1/64)", (1 / 128, 0.25))
register("adversary.acontrario.fit.translate_max", 0.5, "nuisance fit: largest sub-pixel page offset searched (px)", (0.1, 1.0))
register("adversary.acontrario.fit.translate_y_probes", [-0.25, 0.25], "nuisance fit: vertical page offsets probed (px); Chrome snaps vertical text placement, so one probe per direction covers (-0.5, 0) and (0, 0.5)")
register("adversary.acontrario.fit.ls_bias", 1.27, "nuisance fit: least-squares offset estimate / true offset (gradient LS over-estimates on hinted glyph edges)", (0.8, 2.0))
register("adversary.acontrario.fit.min_gain", 0.02, "nuisance fit: a nuisance is adopted only when it removes at least this fraction of unexplained pixels", (0.0, 0.5))
register("adversary.acontrario.fit.blur_grid", [0.25, 0.75, 0.025], "nuisance fit: Gaussian resampling-blur sigma grid (lo, hi, step)")
register("adversary.acontrario.fit.blur_fine", 0.002, "nuisance fit: blur sigma refinement step", (0.0005, 0.02))
register("adversary.acontrario.fit.trim_cap", 8.0, "nuisance fit: per-pixel cap (RGB levels) of the trimmed-L1 refinement criterion (errors saturate, nuisance does not)", (1.0, 64.0))
register("adversary.acontrario.fit.jpeg_grid", [50, 100], "nuisance fit: JPEG quality range (every quality is tried: the criterion is sharp)")


def _css(translate: Optional[tuple[float, float]], smoothing: Optional[str]) -> str:
    css = ""
    if translate and (translate[0] or translate[1]):
        css += f"body {{ position: relative; left: {translate[0]:.6f}px; top: {translate[1]:.6f}px; }}\n"
    if smoothing:
        css += f"* {{ -webkit-font-smoothing: {smoothing} !important; }}\n"
    return css


def unexplained(a: np.ndarray, b: np.ndarray, tol: Optional[int] = None) -> int:
    """Robust fit criterion: number of pixels whose largest channel difference exceeds ``tol``."""
    t = int(P["adversary.acontrario.fit.score_tol"]) if tol is None else tol
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    d = np.abs(a[:h, :w].astype(np.int16) - b[:h, :w].astype(np.int16)).max(-1)
    return int((d > t).sum())


def ls_offset(target: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
    """Robust (IRLS) least-squares page offset of ``target`` w.r.t. ``cand`` from image gradients."""
    t = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3) / 8
    d = t - c
    sel = (np.abs(gx) + np.abs(gy)) > 1
    if sel.sum() < 50:
        return 0.0, 0.0
    A = np.stack([gx[sel], gy[sel]], 1).astype(np.float64)
    r0 = -d[sel].astype(np.float64)
    w = np.ones(len(r0))
    p = np.zeros(2)
    for _ in range(6):
        M = (A * w[:, None]).T @ A
        p = np.linalg.solve(M + 1e-3 * np.eye(2), (A * w[:, None]).T @ r0)
        res = r0 - A @ p
        s = np.median(np.abs(res)) + 1e-3
        w = 1.0 / np.maximum(1.0, np.abs(res) / (3 * s))
    return float(p[0]), float(p[1])


@dataclass
class NuisanceFit:
    """The nuisance that best explains ``target`` as a transformation of the candidate IR's render."""
    browser: str = "main"            # "main" | "alt"
    smoothing: Optional[str] = None
    translate: tuple[float, float] = (0.0, 0.0)
    blur_sigma: Optional[float] = None
    jpeg_quality: Optional[int] = None
    unexplained_before: int = 0
    unexplained_after: int = 0
    renders: int = 0
    seconds: float = 0.0
    trace: list = field(default_factory=list)
    render_img: Optional[np.ndarray] = field(default=None, repr=False)
    registered: bool = False         # render-level knobs were fitted in the renderer (an IR was available)
    flaky: bool = False              # the registration render was not reproducible (renderer nondeterminism)

    def kinds(self) -> list[str]:
        k = []
        if self.browser != "main":
            k.append("browser")
        if self.smoothing:
            k.append("smoothing")
        if self.translate != (0.0, 0.0):
            k.append("subpixel")
        if self.blur_sigma:
            k.append("blur")
        if self.jpeg_quality:
            k.append("jpeg")
        return k

    def to_dict(self) -> dict:
        return {"browser": self.browser, "smoothing": self.smoothing, "translate": [round(v, 4) for v in self.translate],
                "blur_sigma": self.blur_sigma, "jpeg_quality": self.jpeg_quality, "kinds": self.kinds(),
                "unexplained_before": self.unexplained_before, "unexplained_after": self.unexplained_after,
                "renders": self.renders, "seconds": round(self.seconds, 3), "registered": self.registered, "flaky": self.flaky}


def _jpeg(rgb: np.ndarray, q: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _blur(rgb: np.ndarray, s: float) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), float(s))


def trimmed_l1(target: np.ndarray, im: np.ndarray) -> float:
    """Continuous fit criterion: sum over pixels of min(max channel |difference|, trim_cap). Errors saturate at
    the cap; nuisance (blur, compression, sub-pixel offsets) produces many small differences it does count."""
    cap = float(P["adversary.acontrario.fit.trim_cap"])
    return float(np.minimum(np.abs(im.astype(np.int16) - target.astype(np.int16)).max(-1), cap).sum())


def fit_image_level(target: np.ndarray, base: np.ndarray, fast: bool = False,
                    q_hint: Optional[int] = None) -> tuple[np.ndarray, Optional[float], Optional[int], int]:
    """Post-raster nuisance: resampling blur (sigma grid, then a fine refinement) and JPEG (quality grid, then
    unit steps), each adopted only when it removes a ``min_gain`` share of the trimmed-L1 residual.
    Returns (image, sigma, quality, unexplained pixel count)."""
    gain = float(P["adversary.acontrario.fit.min_gain"])
    best, cost = base, trimmed_l1(target, base)
    sigma = quality = None
    lo, hi, st = (float(v) for v in P["adversary.acontrario.fit.blur_grid"])
    st = st * 2 if fast else st
    grid = [(trimmed_l1(target, _blur(base, sg)), round(float(sg), 4)) for sg in np.arange(lo, hi + 1e-9, st)]
    c, sg = min(grid)
    if c < cost * (1 - gain):
        sigma, cost = sg, c
        if not fast:
            fine = float(P["adversary.acontrario.fit.blur_fine"])
            ref = [(trimmed_l1(target, _blur(base, v)), round(float(v), 4)) for v in np.arange(max(fine, sg - 1.5 * st), sg + 1.5 * st + 1e-9, fine)]
            cost, sigma = min(ref + [(c, sg)])
            ref = [(trimmed_l1(target, _blur(base, v)), round(float(v), 5)) for v in np.arange(max(fine / 4, sigma - fine), sigma + fine + 1e-9, fine / 4)]
            cost, sigma = min(ref + [(cost, sigma)])
        best = _blur(base, sigma)
    # JPEG: the criterion is sharp (only the exact quality reproduces the quantisation), so every quality of
    # the range is tried; fast mode (probe scoring) only re-tries the hinted quality
    qlo, qhi = (int(v) for v in P["adversary.acontrario.fit.jpeg_grid"][:2])
    qs = ([q_hint] if q_hint else []) if fast else list(range(qlo, qhi + 1))
    grid = [(trimmed_l1(target, _jpeg(best, q)), q) for q in qs]
    if grid:
        c, q0 = min(grid)
        if c < cost * (1 - gain):
            quality, cost = q0, c
            best = _jpeg(best, quality)
    if sigma is not None and quality is not None and not fast:
        # the blur was estimated through the compression: one joint coordinate-descent round
        fine = float(P["adversary.acontrario.fit.blur_fine"])
        ref = [(trimmed_l1(target, _jpeg(_blur(base, v), quality)), round(float(v), 5))
               for v in np.arange(max(fine, sigma - 10 * fine), sigma + 10 * fine + 1e-9, fine / 2)]
        c, sg = min(ref)
        if c < cost:
            sigma, cost = sg, c
        ref = [(trimmed_l1(target, _jpeg(_blur(base, sigma), q)), q) for q in range(max(1, quality - 2), min(100, quality + 2) + 1)]
        cost, quality = min(ref + [(cost, quality)])
        best = _jpeg(_blur(base, sigma), quality)
    return best, sigma, quality, unexplained(target, best)


def fit_nuisance(target: np.ndarray, candidate: np.ndarray, ir: Optional[Document] = None,
                 use_alt: bool = True, post=None) -> tuple[np.ndarray, NuisanceFit]:
    """Register the nuisance: find the render-level (browser build, font smoothing, sub-pixel page
    offset) and post-raster (blur, JPEG) transformation of the candidate that best explains the
    target, by rendering the candidate IR in the real renderer. Without an IR only the post-raster
    part is fitted. Returns (fitted candidate image, fit)."""
    t0 = time.time()
    fit = NuisanceFit()
    gain = float(P["adversary.acontrario.fit.min_gain"])
    s0 = unexplained(target, candidate)
    fit.unexplained_before = s0
    if target.shape != candidate.shape or s0 == 0:
        fit.registered = ir is not None and s0 == 0
        fit.render_img = candidate
        fit.unexplained_after = s0
        fit.seconds = time.time() - t0
        return candidate, fit
    cache: dict = {}

    def render(browser: str, smoothing: Optional[str], tr: tuple[float, float]) -> Optional[np.ndarray]:
        key = (browser, smoothing, round(tr[0] * 4096), round(tr[1] * 4096))
        if key in cache:
            return cache[key]
        if browser == "main" and not smoothing and tr == (0.0, 0.0):
            cache[key] = candidate
            return candidate
        try:
            css = _css(tr, smoothing)
            if browser == "alt":
                from dt.adversary.perturb import alt_browser
                ab = alt_browser()
                if not ab.available:
                    cache[key] = None
                    return None
                im = ab.render(ir, css)
            else:
                from dt.render.screenshot import render_doc
                im = render_doc(ir, extra_css=css)
            fit.renders += 1
        except Exception:
            im = None
        if im is not None and im.shape != target.shape:
            im = None
        if im is not None and post is not None:
            im = post(im)
        cache[key] = im
        return im

    best_img, best_s = candidate, s0
    fit.registered = ir is not None
    if ir is not None:
        # every probe render is scored after its own (fast) post-raster fit, so that render-level knobs are
        # compared on equal terms when nuisances combine (browser build + JPEG, smoothing + blur, ...)
        _, _, q_hint, _ = fit_image_level(target, candidate)

        def score(im: np.ndarray) -> float:
            return trimmed_l1(target, fit_image_level(target, im, fast=True, q_hint=q_hint)[0])

        s0 = score(candidate)
        step = float(P["adversary.acontrario.fit.translate_step"])
        kmax = int(round(float(P["adversary.acontrario.fit.translate_max"]) / step))
        bias = float(P["adversary.acontrario.fit.ls_bias"])

        def search_x(mode: tuple[str, Optional[str]], s_at0: int) -> tuple[float, int]:
            """Local integer search of the horizontal page offset on the LayoutUnit grid, seeded by LS."""
            im0 = render(mode[0], mode[1], (0.0, 0.0))
            if im0 is None:
                return 0.0, 1 << 60
            lx, _ = ls_offset(target, im0)
            scores: dict[int, float] = {0: s_at0}
            sgn = 1 if lx > 0 else -1

            def f(k: int) -> int:
                if k not in scores:
                    im = render(mode[0], mode[1], (sgn * k * step, 0.0))
                    scores[k] = score(im) if im is not None else 1 << 60
                return scores[k]

            if abs(lx) / bias >= step * 2:
                k = max(1, min(kmax, int(round(abs(lx) / bias / step))))
            else:
                # no gradient evidence (e.g. aliased text does not move sub-pixel): probe half the range both ways
                k = max(1, kmax // 2)
                probe = {}
                for sg in (1, -1):
                    sgn = sg
                    scores.clear()
                    scores[0] = s_at0
                    probe[sg] = f(k)
                sgn = min(probe, key=probe.get)
                if probe[sgn] >= s_at0 * (1 - gain):
                    return 0.0, s_at0
                scores.clear()
                scores.update({0: s_at0, k: probe[sgn]})
            for d in (4, 2, 1):
                while True:
                    nb = min([kk for kk in (k - d, k + d) if 0 <= kk <= kmax], key=f)
                    if f(nb) < f(k):
                        k = nb
                    else:
                        break
            fit.trace.append({"mode": list(mode), "translate_x": {str(kk * sgn): v for kk, v in sorted(scores.items())}})
            return sgn * k * step, f(k)

        mode_scores = {("main", None): s0}
        for m in [("main", "none")] + ([("alt", None)] if use_alt else []):
            im = render(m[0], m[1], (0.0, 0.0))
            if im is not None:
                mode_scores[m] = score(im)
        # the two render-level knobs combine only when each one explains something on its own
        if use_alt and all(mode_scores.get(m, s0) < s0 * (1 - gain) for m in (("main", "none"), ("alt", None))):
            im = render("alt", "none", (0.0, 0.0))
            if im is not None:
                mode_scores[("alt", "none")] = score(im)
        fit.trace.append({"modes": {f"{m[0]}/{m[1]}": v for m, v in mode_scores.items()}})
        ranked = sorted(mode_scores, key=lambda m: mode_scores[m])
        # the offset is searched in the default mode and in the best other mode (nuisances combine)
        options = []
        search_modes = [("main", None)] + ([ranked[0]] if ranked[0] != ("main", None) and mode_scores[ranked[0]] < s0 * (1 - gain) else [])
        for m in search_modes:
            tx, sc = search_x(m, mode_scores[m])
            options.append((sc, m, tx))
        for m in ranked:
            options.append((mode_scores[m], m, 0.0))
        options.sort(key=lambda o: (o[0], o[1] != ("main", None), abs(o[2])))
        sc, mode, tx = options[0]
        if not (sc < s0 * (1 - gain)):
            sc, mode, tx = s0, ("main", None), 0.0
        best_img, best_s = render(mode[0], mode[1], (tx, 0.0)), sc
        ty = 0.0
        _, ly = ls_offset(target, best_img)
        if best_s > 0 and (tx != 0.0 or abs(ly) >= 0.02):
            # vertical page offsets snap to the pixel grid per text run: one probe per direction suffices
            for yv in tuple(P["adversary.acontrario.fit.translate_y_probes"]):
                im = render(mode[0], mode[1], (tx, float(yv)))
                if im is None:
                    continue
                s2 = score(im)
                fit.trace.append({"translate_y": yv, "score": s2})
                if s2 < best_s * (1 - gain):
                    best_img, best_s, ty = im, s2, float(yv)
        fit.browser, fit.smoothing, fit.translate = mode[0], mode[1], (tx, ty)
        if best_img is not candidate:
            # the null assumes the registration render is reproducible: render it again (bypassing the cache);
            # on disagreement take a third render and keep the majority, else flag the page as unexplained
            key = (mode[0], mode[1], round(tx * 4096), round(ty * 4096))
            cache.pop(key, None)
            again = render(mode[0], mode[1], (tx, ty))
            if again is None or not np.array_equal(again, best_img):
                cache.pop(key, None)
                third = render(mode[0], mode[1], (tx, ty))
                if third is not None and again is not None and np.array_equal(third, again):
                    best_img = again
                elif not (third is not None and np.array_equal(third, best_img)):
                    fit.flaky = True
                fit.trace.append({"render_reproducible": False, "flaky": fit.flaky})
    fit.render_img = best_img
    img, sigma, q, sc = fit_image_level(target, best_img)
    fit.blur_sigma, fit.jpeg_quality, fit.unexplained_after = sigma, q, sc
    fit.seconds = time.time() - t0
    return img, fit


# =========================================================================== local statistics
register("adversary.acontrario.scales", [8, 16, 32, 64], "tile sizes (px) of the multi-scale test family; stride = size/2")
register("adversary.acontrario.jnd", 2.3, "ΔE2000 just-noticeable difference (the quantile statistic counts pixels above it)", (1.0, 5.0))
register("adversary.acontrario.gms_c", 170.0, "GMSD stability constant (gray 0..255, Prewitt/3 gradients; Xue et al. 2014)", (10.0, 1000.0))
register("adversary.acontrario.eoh_bins", 8, "edge-orientation histogram bins over [0, pi)", (4, 16))
register("adversary.acontrario.eoh_kappa", 2.0, "edge-orientation histogram distance regulariser (gradient units per pixel)", (0.1, 20.0))
register("adversary.acontrario.content_edges", [1.0, 4.0, 12.0], "tile content strata: mean candidate gradient magnitude cut points")
register("adversary.acontrario.page.sub_de", 0.3, "page strata: ΔE2000 above which an edge pixel is 'not exactly explained' by the nuisance fit", (0.05, 2.0))
register("adversary.acontrario.page.unexplained_sub", 0.4,
         "page strata: an exactly registered page (no post-raster fit) whose fraction of candidate edge pixels above page.sub_de "
         "reaches this is 'unexplained' (calibration pages with exact registration stay below 0.33 even with four large errors)", (0.1, 0.95))
register("adversary.acontrario.page.unexplained_frac", 0.3,
         "page strata: a page whose fraction of candidate edge pixels still above a JND after the fit reaches this is 'unexplained'", (0.05, 0.9))

register("adversary.acontrario.norm_q", 0.9,
         "self-normalisation: each tile statistic is divided by (this quantile of the same statistic over the page's tiles of the same "
         "scale and content stratum + a floor), removing the page-level nuisance misfit (errors touch few tiles)", (0.5, 0.99))
register("adversary.acontrario.norm_floor", [0.05, 0.005, 0.002, 0.002, 0.0001, 0.0001, 0.001, 0.005],
         "self-normalisation floors per statistic (STATS order): the smallest scale a statistic is measured against")
register("adversary.acontrario.norm_min_tiles", 20, "self-normalisation: a content stratum with fewer tiles uses the whole level's quantile", (5, 500))

STATS = ("de_mean", "de_jnd", "gmsd", "gmsd2", "ssim_l", "ssim_c", "ssim_s", "eoh")
STAT_DOC = {
    "de_mean": "tile mean ΔE2000 (colour / luminance shift over an area)",
    "de_jnd": "fraction of tile pixels with ΔE2000 > JND (a ΔE quantile: q(1-f) > JND)",
    "gmsd": "gradient-magnitude-similarity deviation at full resolution (edges moved / added / removed)",
    "gmsd2": "gradient-magnitude-similarity deviation at half resolution (coarser structure)",
    "ssim_l": "1 - SSIM luminance component (mean intensity changed)",
    "ssim_c": "1 - SSIM contrast component (local contrast changed: ink added / removed, blur)",
    "ssim_s": "1 - SSIM structure component (pattern changed at equal contrast: glyph / shape swap)",
    "eoh": "edge-orientation histogram L1 distance (edge directions changed: shape, glyph, radius)",
}


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _prewitt_mag(g: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32) / 3.0
    gx = cv2.filter2D(g, cv2.CV_32F, kx, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(g, cv2.CV_32F, kx.T, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(gx * gx + gy * gy)


def _gms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = float(P["adversary.acontrario.gms_c"])
    ma, mb = _prewitt_mag(a), _prewitt_mag(b)
    return (2 * ma * mb + c) / (ma * ma + mb * mb + c)


def pixel_maps(target: np.ndarray, fitted: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel maps from which every tile statistic is a box sum (all float32, H x W [x k])."""
    from skimage.color import deltaE_ciede2000
    from dt.compare.pixel import to_lab
    lt, lr = to_lab(target), to_lab(fitted)
    de = deltaE_ciede2000(lt, lr).astype(np.float32)
    gt, gr = _gray(target), _gray(fitted)
    out: dict[str, np.ndarray] = {"de": de, "jnd": (de > float(P["adversary.acontrario.jnd"])).astype(np.float32)}
    # GMSD on the dissimilarity 1 - GMS: exactly zero where the images agree (no cancellation error)
    g1 = 1.0 - _gms(gt, gr)
    out["gms"], out["gms_sq"] = g1, g1 * g1
    h, w = gt.shape
    t2 = cv2.resize(gt, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(gr, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    g2 = cv2.resize(1.0 - _gms(t2, r2), (w, h), interpolation=cv2.INTER_NEAREST)
    out["gms2"], out["gms2_sq"] = g2, g2 * g2
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    C3 = C2 / 2
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5, borderType=cv2.BORDER_REPLICATE)  # noqa: E731
    mt, mr = blur(gt), blur(gr)
    vt = np.maximum(blur(gt * gt) - mt * mt, 0)
    vr = np.maximum(blur(gr * gr) - mr * mr, 0)
    cov = blur(gt * gr) - mt * mr
    st, sr = np.sqrt(vt), np.sqrt(vr)
    differs = blur(np.abs(gt - gr)) > 1e-4  # windows that are pixel-identical have components exactly 1
    out["ssim_l"] = np.where(differs, 1 - (2 * mt * mr + C1) / (mt * mt + mr * mr + C1), 0).astype(np.float32)
    out["ssim_c"] = np.where(differs, 1 - (2 * st * sr + C2) / (vt + vr + C2), 0).astype(np.float32)
    out["ssim_s"] = np.where(differs, np.clip(1 - (cov + C3) / (st * sr + C3), 0, 2), 0).astype(np.float32)
    nb = int(P["adversary.acontrario.eoh_bins"])
    for key, g in (("eoh_t", gt), ("eoh_r", gr)):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)
        b = np.minimum((ang / np.pi * nb).astype(np.int32), nb - 1)
        hist = np.zeros(g.shape + (nb,), np.float32)
        np.put_along_axis(hist, b[..., None], mag[..., None], axis=2)
        out[key] = hist
        if key == "eoh_r":
            out["grad_r"] = mag
        else:
            out["grad_t"] = mag
    return out


def tile_grid(h: int, w: int, size: int) -> np.ndarray:
    """(n, 4) int array of tiles x0, y0, x1, y1 with stride size/2 covering the image (edge tiles clipped)."""
    st = max(1, size // 2)
    ys = list(range(0, max(1, h - size) + 1, st))
    xs = list(range(0, max(1, w - size) + 1, st))
    if ys[-1] + size < h:
        ys.append(max(0, h - size))
    if xs[-1] + size < w:
        xs.append(max(0, w - size))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    x0, y0 = xx.ravel(), yy.ravel()
    return np.stack([x0, y0, np.minimum(x0 + size, w), np.minimum(y0 + size, h)], 1).astype(np.int64)


def _box_sums(m: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    I = cv2.integral(m.astype(np.float64)) if m.ndim == 2 else None
    if I is None:
        return np.stack([_box_sums(m[..., k], tiles) for k in range(m.shape[2])], 1)
    x0, y0, x1, y1 = tiles.T
    return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]


def tile_stats(maps: dict[str, np.ndarray], tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Statistics (n, len(STATS)) and content measure (n,) = mean candidate gradient magnitude."""
    area = ((tiles[:, 2] - tiles[:, 0]) * (tiles[:, 3] - tiles[:, 1])).astype(np.float64)
    S = lambda k: _box_sums(maps[k], tiles) / area  # noqa: E731
    g1, g1s = S("gms"), S("gms_sq")
    g2, g2s = S("gms2"), S("gms2_sq")
    ht, hr = _box_sums(maps["eoh_t"], tiles), _box_sums(maps["eoh_r"], tiles)
    kappa = float(P["adversary.acontrario.eoh_kappa"])
    eoh = np.abs(ht - hr).sum(1) / (ht.sum(1) + hr.sum(1) + kappa * area)
    stats = np.stack([S("de"), S("jnd"), np.sqrt(np.maximum(g1s - g1 * g1, 0)), np.sqrt(np.maximum(g2s - g2 * g2, 0)),
                      S("ssim_l"), S("ssim_c"), S("ssim_s"), eoh], 1)
    content = S("grad_r")
    return stats.astype(np.float64), content


def page_quantiles(stats: np.ndarray, cbin: np.ndarray) -> np.ndarray:
    """(n_cbins + 1, n_stats): the page's own ``norm_q`` quantile of every statistic per content stratum
    (last row: the whole level)."""
    nb = len(P["adversary.acontrario.content_edges"]) + 1
    qq = float(P["adversary.acontrario.norm_q"])
    out = np.zeros((nb + 1, stats.shape[1]))
    out[nb] = np.quantile(stats, qq, axis=0) if len(stats) else 0.0
    for b in range(nb):
        sel = cbin == b
        out[b] = np.quantile(stats[sel], qq, axis=0) if sel.sum() >= int(P["adversary.acontrario.norm_min_tiles"]) else out[nb]
    return out


def normalise(stats: np.ndarray, cbin: np.ndarray, q: np.ndarray) -> np.ndarray:
    floor = np.asarray(P["adversary.acontrario.norm_floor"], float)[None, :]
    return stats / (q[cbin] + floor)


def content_bin(content: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(P["adversary.acontrario.content_edges"], float), content, side="right")


def page_noise(maps: dict[str, np.ndarray]) -> tuple[float, float]:
    """Robust page-level residual after the nuisance fit, over candidate edge pixels (where every nuisance
    shows): fraction above ``page.sub_de`` and fraction above the JND. Errors are sparse, so these
    fractions measure the nuisance misfit, not the errors."""
    sel = maps["grad_r"] > 2.0
    if sel.sum() < 100:
        return 0.0, 0.0
    de = maps["de"][sel]
    return float((de > float(P["adversary.acontrario.page.sub_de"])).mean()), float((de > float(P["adversary.acontrario.jnd"])).mean())


TINT_BIN = 4  # page stratum of a page whose global colour offset was detected and compensated
N_PBINS = 5


def pbin_order(pbin: int, cbin: Optional[int]) -> list[tuple[Optional[int], Optional[int]]]:
    """Back-off order of null strata: own stratum, then noisier ones, then the pooled one."""
    noisier = [3] if pbin == TINT_BIN else list(range(pbin + 1, 4))
    out = [(pbin, cbin), (pbin, None)]
    for b in noisier:
        out += [(b, cbin), (b, None)]
    return out + [(None, cbin), (None, None)]


def page_bin(nu: tuple[float, float], fit: Optional["NuisanceFit"] = None) -> int:
    """Page stratum = the nuisance model that explains the page (never the errors, which are page-specific):
    0 exact registration in the renderer (none / sub-pixel / smoothing / browser build: deterministic, H0
    residual is exactly zero), 1 a continuous post-raster nuisance was fitted (blur, JPEG: sub-JND misfit),
    2 no render-level registration (no IR), 3 unexplained (a large share of edge pixels still differs by more
    than a JND after the fit, e.g. a real screenshot), 4 page-wide colour offset compensated (set by the caller)."""
    cut = float(P["adversary.acontrario.page.unexplained_frac"])
    if nu[1] >= cut or (fit is not None and fit.flaky):
        return 3
    if fit is None or not fit.registered:
        return 2
    if fit.blur_sigma or fit.jpeg_quality:
        return 1
    # an 'exact' registration must leave (almost) every edge pixel exact; a post-raster fit legitimately does not
    if nu[0] >= float(P["adversary.acontrario.page.unexplained_sub"]):
        return 3
    return 0


# =========================================================================== null model (H0 = nuisance only)
register("adversary.acontrario.eps", 1.0, "a-contrario threshold: a tile is meaningful when NFA = N_tests * p < eps (expected false alarms per page <= eps)", (1e-4, 100.0))
register("adversary.acontrario.null.top_k", 400, "null model: largest H0 samples kept per stratum for the exponential (POT) tail", (50, 5000))
register("adversary.acontrario.null.min_n", 3000, "null model: a stratum with fewer H0 samples backs off to the pooled stratum", (100, 100000))
register("adversary.acontrario.null.zero", 1e-7, "null model: statistics at or below this are exact agreement (p = 1)", (0.0, 1e-3))
register("adversary.acontrario.null.min_pos", 200, "null model: positive H0 samples needed for a stratum to supply its own conditional tail (else the nearest noisier stratum's)", (10, 10000))
register("adversary.acontrario.null.xi_min", 0.1, "null model: smallest GPD tail shape (heavier = more conservative far-tail p-values)", (0.0, 0.5))
register("adversary.acontrario.null.beta_ci", 2.0, "null model: tail scale inflated by (1 + this / sqrt(top_k)) (one-sided CI, keeps the tail conservative)", (0.0, 5.0))

_LOG_EDGES = np.concatenate([[0.0], np.logspace(-7, 2, 541)])


class NullModel:
    """Empirical H0 distribution of every tile statistic, stratified by (statistic, scale, page-noise
    bin, content bin), with an exponential peaks-over-threshold tail beyond the top-k samples.

    ``pvalue`` returns P_H0(S >= s): exact counts in the body, the fitted tail beyond. Strata with too
    few samples back off to the content-pooled and then the fully pooled stratum."""

    def __init__(self) -> None:
        self.strata: dict[str, dict] = {}
        self.meta: dict = {}

    @staticmethod
    def key(stat: int, scale: int, pbin: Optional[int], cbin: Optional[int]) -> str:
        return f"{STATS[stat]}|{scale}|{'*' if pbin is None else pbin}|{'*' if cbin is None else cbin}"

    # ---- accumulation
    def add(self, stat: int, scale: int, pbin: int, cbin: np.ndarray, values: np.ndarray) -> None:
        k = int(P["adversary.acontrario.null.top_k"])
        for pb in (pbin, None):
            for cb in sorted(set(cbin.tolist())) + [None]:
                v = values if cb is None else values[cbin == cb]
                if not len(v):
                    continue
                key = self.key(stat, scale, pb, cb)
                st = self.strata.setdefault(key, {"n": 0, "hist": np.zeros(len(_LOG_EDGES), np.int64), "top": np.zeros(0)})
                st["n"] += int(len(v))
                idx = np.searchsorted(_LOG_EDGES, v, side="right") - 1
                st["hist"] += np.bincount(np.clip(idx, 0, len(_LOG_EDGES) - 1), minlength=len(_LOG_EDGES))
                top = np.concatenate([st["top"], v[v > float(P["adversary.acontrario.null.zero"])]])
                if len(top) > k:
                    top = np.partition(top, len(top) - k)[-k:]
                st["top"] = np.sort(top)
            if pb is None:
                break

    # ---- query
    def _get(self, stat: int, scale: int, pb: Optional[int], cb: Optional[int]) -> Optional[dict]:
        return self.strata.get(self.key(stat, scale, pb, cb))

    def _stratum(self, stat: int, scale: int, pbin: int, cbin: int) -> Optional[dict]:
        mn = int(P["adversary.acontrario.null.min_n"])
        for pb, cb in pbin_order(pbin, cbin):
            st = self._get(stat, scale, pb, cb)
            if st is not None and st["n"] >= mn:
                return st
        return self._get(stat, scale, None, None)

    def _positive(self, stat: int, scale: int, pbin: int, cbin: int) -> Optional[dict]:
        """Stratum that supplies the conditional law of positive values: the own stratum when it has enough
        positive samples, else the nearest noisier one, else the pooled one (shape shared across strata)."""
        mp = int(P["adversary.acontrario.null.min_pos"])
        for pb, cb in pbin_order(pbin, cbin):
            st = self._get(stat, scale, pb, cb)
            if st is not None and st["n"] - int(st["hist"][0]) >= mp:
                return st
        return None

    def _tail(self, st: dict) -> Optional[tuple[float, float, float, int]]:
        if "tail" in st:
            return st["tail"]
        top = st["top"]
        if len(top) < 10:
            return None
        u, beta, xi = self.gpd(top)
        return (u, beta, xi, len(top))

    @staticmethod
    def gpd(top: np.ndarray) -> tuple[float, float, float]:
        """Generalised Pareto tail over the top-k sample: (threshold u, scale beta, shape xi), method of
        moments, xi clipped to [null.xi_min, 0.5] (never lighter than a conservative heavy tail)."""
        u = float(top[0])
        y = top - u
        m = float(np.mean(y))
        v = float(np.var(y)) or 1e-12
        xi = 0.5 * (1 - m * m / v)
        xi = min(0.5, max(float(P["adversary.acontrario.null.xi_min"]), xi))
        beta = max(m * (1 - xi), 1e-12) * (1 + float(P["adversary.acontrario.null.beta_ci"]) / math.sqrt(max(1, len(top))))
        return u, beta, xi

    def survival(self, st: dict, x: np.ndarray, pos: Optional[dict] = None) -> np.ndarray:
        """Zero-inflated survival P(S >= x) = P(S > 0) * P(S >= x | S > 0). P(S > 0) comes from ``st`` (upper
        estimate (k+1)/(n+1)); the conditional law from ``pos`` (body: exact counts, tail: GPD)."""
        zero = float(P["adversary.acontrario.null.zero"])
        n = max(1, st["n"])
        n_pos = n - int(st["hist"][0])
        pi = (n_pos + 1) / (n + 1)
        pos = pos if pos is not None else st
        npos = max(1, pos["n"] - int(pos["hist"][0]))
        hist = pos["hist"].copy()
        hist[0] = 0
        cum = np.cumsum(hist[::-1])[::-1]
        idx = np.clip(np.searchsorted(_LOG_EDGES, x, side="right") - 1, 1, len(_LOG_EDGES) - 1)
        cond = np.maximum(cum[idx], 1) / (npos + 1)
        tail = self._tail(pos)
        if tail is not None:
            u, beta, xi, k = tail
            z = np.maximum(x - u, 0) / beta
            tail = (k / (npos + 1)) * np.power(1 + xi * z, -1.0 / xi)
            cond = np.where(x > u, tail, cond)
        p = pi * np.minimum(cond, 1.0)
        return np.where(x <= zero, 1.0, np.clip(p, 1e-300, 1.0))

    def pvalues(self, stats: np.ndarray, scale: int, pbin: int, cbin: np.ndarray) -> np.ndarray:
        out = np.ones_like(stats)
        for j in range(stats.shape[1]):
            for cb in np.unique(cbin):
                sel = cbin == cb
                st = self._stratum(j, scale, pbin, int(cb))
                if st is None:
                    continue
                out[sel, j] = self.survival(st, stats[sel, j], self._positive(j, scale, pbin, int(cb)))
        return out

    # ---- persistence
    def to_dict(self) -> dict:
        """Compact form: the body histogram as (first bin, counts) and the fitted GPD tail (u, beta, xi, k)."""
        out = {}
        for k, v in self.strata.items():
            nz = np.nonzero(v["hist"])[0]
            lo, hi = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)
            e = {"n": int(v["n"]), "h0": lo, "h": [int(c) for c in v["hist"][lo:hi]]}
            t = self._tail(v)
            if t is not None:
                e["tail"] = [float(t[0]), float(t[1]), float(t[2]), int(t[3])]
            out[k] = e
        return {"meta": self.meta, "strata": out}

    @staticmethod
    def from_dict(d: dict) -> "NullModel":
        m = NullModel()
        m.meta = d.get("meta", {})
        for k, v in d["strata"].items():
            h = np.zeros(len(_LOG_EDGES), np.int64)
            h[v["h0"]:v["h0"] + len(v["h"])] = v["h"]
            st = {"n": int(v["n"]), "hist": h, "top": np.zeros(0)}
            if "tail" in v:
                u, beta, xi, kk = v["tail"]
                st["tail"] = (float(u), float(beta), float(xi), int(kk))
            m.strata[k] = st
        return m


# =========================================================================== analysis of one (target, candidate) pair
@dataclass
class Analysis:
    target: np.ndarray
    candidate: np.ndarray
    fitted: np.ndarray
    fit: NuisanceFit
    maps: dict
    nu: tuple
    pbin: int
    levels: list            # per scale: {"scale", "tiles", "stats", "content", "cbin"}
    ir: Optional[Document] = None


def analyse(target: np.ndarray, candidate: np.ndarray, ir: Optional[Document] = None, fitted: Optional[np.ndarray] = None,
            fit: Optional[NuisanceFit] = None, use_alt: bool = True) -> Analysis:
    from dt.common.image import same_size
    if target.shape != candidate.shape:
        # judged on the target frame: pad the candidate with a colour opposite to the target (never crop into agreement)
        h, w = target.shape[:2]
        pad = np.empty_like(target)
        pad[:] = 255 - np.median(target.reshape(-1, 3), 0).astype(np.uint8)
        hh, ww = min(h, candidate.shape[0]), min(w, candidate.shape[1])
        pad[:hh, :ww] = candidate[:hh, :ww]
        candidate = pad
        ir = None
    if fitted is None:
        fitted, fit = fit_nuisance(target, candidate, ir, use_alt=use_alt)
    maps = pixel_maps(target, fitted)
    nu = page_noise(maps)
    h, w = target.shape[:2]
    levels = []
    for s in P["adversary.acontrario.scales"]:
        s = int(s)
        if s > min(h, w):
            continue
        tiles = tile_grid(h, w, s)
        stats, content = tile_stats(maps, tiles)
        cbin = content_bin(content)
        q = page_quantiles(stats, cbin)
        levels.append({"scale": s, "tiles": tiles, "stats": stats, "content": content, "cbin": cbin, "q": q,
                       "u": normalise(stats, cbin, q)})
    fit = fit or NuisanceFit()
    return Analysis(target, candidate, fitted, fit, maps, nu, page_bin(nu, fit), levels, ir)


def h0_masks(an: Analysis, gt_boxes: list, pad: float = 4.0) -> list[np.ndarray]:
    """Per level: tiles that do not touch any ground-truth box (grown by ``pad``) -- pure-nuisance samples."""
    out = []
    for lv in an.levels:
        t = lv["tiles"]
        keep = np.ones(len(t), bool)
        for b in gt_boxes:
            x0, y0, x1, y1 = b.x - pad, b.y - pad, b.x + b.w + pad, b.y + b.h + pad
            keep &= ~((t[:, 0] < x1) & (t[:, 2] > x0) & (t[:, 1] < y1) & (t[:, 3] > y0))
        out.append(keep)
    return out


def fit_cached(case, cache_dir: str, use_ir: bool = True) -> tuple[np.ndarray, NuisanceFit]:
    """Development/calibration helper: the nuisance fit of a benchmark case, cached on disk. The render-level
    image is cached separately, so a change to the post-raster fit only re-runs that cheap part."""
    import json
    import os
    from dt.common.image import load_rgb, save_rgb
    tag = "ir" if use_ir else "img"
    os.makedirs(cache_dir, exist_ok=True)
    pj, pr = os.path.join(cache_dir, f"{case.id}.{tag}.fit.json"), os.path.join(cache_dir, f"{case.id}.{tag}.render.png")
    if os.path.exists(pj) and os.path.exists(pr):
        with open(pj) as f:
            d = json.load(f)
        fit = NuisanceFit(d["browser"], d["smoothing"], tuple(d["translate"]), None, None,
                          d["unexplained_before"], 0, d["renders"], d["seconds"])
        fit.registered = bool(d.get("registered", use_ir))
        target = case.target_rgb
        img, fit.blur_sigma, fit.jpeg_quality, fit.unexplained_after = fit_image_level(target, load_rgb(pr))
        if not fit.kinds() and fit.unexplained_before == 0:
            img = case.candidate_rgb
        return img, fit
    fitted, fit = fit_nuisance(case.target_rgb, case.candidate_rgb, case.candidate_ir if use_ir else None)
    save_rgb(fit.render_img if fit.render_img is not None else case.candidate_rgb, pr)
    with open(pj, "w") as f:
        json.dump(fit.to_dict(), f)
    return fitted, fit


# =========================================================================== a-contrario detection
register("adversary.acontrario.px_floor", 0.6, "region support: a pixel of a meaningful tile belongs to the change when its ΔE2000 exceeds max(this, the H0 pixel quantile of the page stratum)", (0.1, 5.0))
register("adversary.acontrario.px_quantile", 0.99,
         "region support: H0 pixel ΔE2000 quantile (per page stratum) a changed pixel of a meaningful tile must exceed. The tile test "
         "already decided; the support only delineates the change, so it must not re-test at a stricter level", (0.9, 0.99999))
register("adversary.acontrario.group_px", 3, "region grouping: dilation radius (px) joining changed pixels into one region", (0, 12))
register("adversary.acontrario.min_region_px", 3, "regions with fewer changed pixels are dropped", (1, 64))
register("adversary.acontrario.conf_span", 30.0, "confidence = clip((log10 eps - log10 NFA) / this, 0.02, 1)", (1.0, 200.0))


def n_tests(an: Analysis) -> int:
    return int(sum(len(lv["tiles"]) for lv in an.levels) * len(STATS))


def tile_nfa(an: Analysis, null: NullModel) -> list[dict]:
    """Per level: p-values (n, n_stats), log10 NFA of each tile (min over statistics) and the statistic
    that fired."""
    N = n_tests(an)
    out = []
    for lv in an.levels:
        p = null.pvalues(lv["u"], lv["scale"], an.pbin, lv["cbin"])
        j = np.argmin(p, 1)
        pmin = p[np.arange(len(p)), j]
        out.append({"p": p, "log_nfa": np.log10(N) + np.log10(np.maximum(pmin, 1e-300)), "fired": j})
    return out


@dataclass
class Region:
    box: "object"                 # dt.ir.Box
    mask: np.ndarray              # bool mask of changed pixels, cropped to box (y0:y1, x0:x1)
    log_nfa: float
    fired: str
    tile_scale: int
    stat_logp: dict
    features: dict = field(default_factory=dict)
    type: Optional[str] = None
    type_prob: float = 0.0
    magnitude: Optional[float] = None
    node_id: Optional[str] = None


def detect_regions(an: Analysis, null: NullModel, eps: Optional[float] = None) -> list[Region]:
    """Meaningful tiles (NFA < eps) -> changed-pixel support inside them -> connected regions."""
    from scipy import ndimage
    from dt.ir import Box
    eps = float(P["adversary.acontrario.eps"]) if eps is None else eps
    le = math.log10(eps)
    h, w = an.target.shape[:2]
    nfa = tile_nfa(an, null)
    best = np.full((h, w), np.inf, np.float64)     # best log NFA covering each pixel
    fired = np.full((h, w), -1, np.int16)
    scale_at = np.zeros((h, w), np.int16)
    for lv, nf in zip(an.levels, nfa):
        sel = np.nonzero(nf["log_nfa"] < le)[0]
        # weakest first so that the strongest evidence is written last
        for i in sel[np.argsort(-nf["log_nfa"][sel])]:
            x0, y0, x1, y1 = lv["tiles"][i]
            v = nf["log_nfa"][i]
            sub = best[y0:y1, x0:x1]
            upd = v < sub
            sub[upd] = v
            fired[y0:y1, x0:x1][upd] = nf["fired"][i]
            scale_at[y0:y1, x0:x1][upd] = lv["scale"]
    meaningful = np.isfinite(best)
    if not meaningful.any():
        return []
    thr = max(float(P["adversary.acontrario.px_floor"]), pixel_threshold(null, an.pbin))
    de = an.maps["de"]
    support = meaningful & (de > thr)
    # meaningful tiles whose change is sub-threshold everywhere (soft shadows, faint tints): keep their
    # strongest quarter of pixels so the evidence is not lost
    lab, n = ndimage.label(meaningful)
    if n:
        has = ndimage.maximum(support, lab, index=np.arange(1, n + 1))
        for k in np.nonzero(~np.asarray(has, bool))[0]:
            m = lab == (k + 1)
            vals = de[m]
            if vals.max() > float(P["adversary.acontrario.null.zero"]):
                support |= m & (de >= np.quantile(vals, 0.75)) & (de > 0)
    r = int(P["adversary.acontrario.group_px"])
    grown = cv2.dilate(support.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0 if r > 0 else support
    lab, n = ndimage.label(grown)
    regions = []
    objs = ndimage.find_objects(lab)
    for k, sl in enumerate(objs):
        if sl is None:
            continue
        m = (lab[sl] == k + 1) & support[sl]
        if m.sum() < int(P["adversary.acontrario.min_region_px"]):
            continue
        ys, xs = np.nonzero(m)
        y0, x0 = sl[0].start + ys.min(), sl[1].start + xs.min()
        y1, x1 = sl[0].start + ys.max() + 1, sl[1].start + xs.max() + 1
        mm = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        bsub = best[y0:y1, x0:x1][mm]
        i = int(np.argmin(bsub))
        fj = int(fired[y0:y1, x0:x1][mm][i])
        regions.append(Region(Box(float(x0), float(y0), float(x1 - x0), float(y1 - y0)), mm, float(bsub[i]),
                              STATS[fj] if fj >= 0 else "?", int(scale_at[y0:y1, x0:x1][mm][i]), {}))
    # per-statistic evidence of each region (min log10 p over the tiles that overlap it)
    for rg in regions:
        b = rg.box
        for lv, nf in zip(an.levels, nfa):
            t = lv["tiles"]
            ov = (t[:, 0] < b.x + b.w) & (t[:, 2] > b.x) & (t[:, 1] < b.y + b.h) & (t[:, 3] > b.y)
            if ov.any():
                lp = np.log10(np.maximum(nf["p"][ov].min(0), 1e-300))
                for j, sname in enumerate(STATS):
                    rg.stat_logp[sname] = min(rg.stat_logp.get(sname, 0.0), float(lp[j]))
    return regions


def pixel_threshold(null: NullModel, pbin: int) -> float:
    q = null.meta.get("px_quantile", {})
    return float(q.get(str(pbin), q.get("*", 0.0)))


# =========================================================================== calibration of the null model
class _PixelAcc:
    def __init__(self) -> None:
        self.h: dict[str, np.ndarray] = {}

    def add(self, pbin: int, values: np.ndarray) -> None:
        idx = np.clip(np.searchsorted(_LOG_EDGES, values, side="right") - 1, 0, len(_LOG_EDGES) - 1)
        c = np.bincount(idx, minlength=len(_LOG_EDGES))
        for k in (str(pbin), "*"):
            self.h[k] = self.h.get(k, np.zeros(len(_LOG_EDGES), np.int64)) + c

    def quantiles(self, q: float) -> dict[str, float]:
        out = {}
        for k, h in self.h.items():
            cum = np.cumsum(h) / max(1, h.sum())
            i = int(np.searchsorted(cum, q))
            out[k] = float(_LOG_EDGES[min(i + 1, len(_LOG_EDGES) - 1)])
        return out


def h0_pixel_mask(shape: tuple, gt_boxes: list, pad: float = 4.0) -> np.ndarray:
    m = np.ones(shape[:2], bool)
    for b in gt_boxes:
        x0, y0, x1, y1 = (int(math.floor(b.x - pad)), int(math.floor(b.y - pad)), int(math.ceil(b.x + b.w + pad)), int(math.ceil(b.y + b.h + pad)))
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    return m


def accumulate_h0(null: NullModel, pix: _PixelAcc, an: Analysis, gt_boxes: list) -> None:
    """Add the pure-nuisance tiles (and pixels) of one analysed pair to the null model. With a JPEG nuisance
    an error also re-quantises the 16 px macro-blocks it touches, so the exclusion zone grows accordingly."""
    pad = 4.0 + (16.0 if an.fit.jpeg_quality else 0.0)
    for lv, keep in zip(an.levels, h0_masks(an, gt_boxes, pad)):
        if not keep.any():
            continue
        for j in range(len(STATS)):
            null.add(j, lv["scale"], an.pbin, lv["cbin"][keep], lv["u"][keep, j])
    pm = h0_pixel_mask(an.target.shape, gt_boxes, pad)
    pix.add(an.pbin, an.maps["de"][pm])


def finalize_null(null: NullModel, pix: _PixelAcc, sources: dict) -> NullModel:
    q = float(P["adversary.acontrario.px_quantile"])
    null.meta.update({"px_quantile": pix.quantiles(q), "px_q": q, "sources": sources, "stats": list(STATS),
                      "scales": list(P["adversary.acontrario.scales"]),
                      "page_unexplained_frac": P["adversary.acontrario.page.unexplained_frac"],
                      "content_edges": list(P["adversary.acontrario.content_edges"])})
    return null


# =========================================================================== interpretable region features
register("adversary.acontrario.flow_radius", 24, "flow feature: largest displacement (px) searched by block matching", (4, 64))
register("adversary.acontrario.ink_de", 8.0, "ink feature: a pixel is ink when its ΔE76 to the local background exceeds this", (2.0, 30.0))
register("adversary.acontrario.big_node_frac", 0.35, "IR features: nodes covering more than this page fraction are backgrounds, not anchors", (0.05, 1.0))


def _ir_nodes(ir: Optional[Document]) -> list[dict]:
    """Painting candidate nodes with their geometry, parents and sibling index (for IR-aware features)."""
    if ir is None:
        return []
    from dt.adversary import perturb as PT
    page = float(ir.width * ir.height) or 1.0
    big = float(P["adversary.acontrario.big_node_frac"])
    out = []

    def walk(n, parent, depth):
        if not n.visible:
            return
        if PT.paints(n) and n.box.w > 0 and n.box.h > 0:
            kind = "text" if n.type == "text" else "icon" if n.type == "icon" else "shape"
            out.append({"node": n, "id": n.id, "kind": kind, "box": n.box, "ext": PT.extent(n, subtree=False),
                        "parent": parent.id if parent is not None else None, "big": n.box.area > big * page, "depth": depth,
                        "stroke": bool(n.strokes), "radius": float(max(n.radius)) if n.radius else 0.0,
                        "shadow": bool(n.effects), "opacity": float(n.opacity)})
        for ch in n.children:
            walk(ch, n, depth + 1)

    walk(ir.root, None, 0)
    return out


def _bg_color(img_lab: np.ndarray, box, pad: int = 3) -> np.ndarray:
    h, w = img_lab.shape[:2]
    x0, y0, x1, y1 = int(box.x), int(box.y), int(box.x + box.w), int(box.y + box.h)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    ring = np.ones((Y1 - Y0, X1 - X0), bool)
    ring[max(0, y0 - Y0):max(0, y1 - Y0), max(0, x0 - X0):max(0, x1 - X0)] = False
    vals = img_lab[Y0:Y1, X0:X1][ring]
    if not len(vals):
        vals = img_lab[Y0:Y1, X0:X1].reshape(-1, 3)
    return np.median(vals, 0)


def _flow(t_gray: np.ndarray, r_gray: np.ndarray, box, radius: int) -> tuple[float, float, float]:
    """Block matching: the target content of ``box`` is found in the candidate at ``box + d``.
    Returns (dx, dy, gain) with gain = 1 - SSD(d) / SSD(0)."""
    h, w = t_gray.shape
    x0, y0 = max(0, int(box.x) - 2), max(0, int(box.y) - 2)
    x1, y1 = min(w, int(box.x + box.w) + 2), min(h, int(box.y + box.h) + 2)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0, 0.0, 0.0
    sx0, sy0, sx1, sy1 = max(0, x0 - radius), max(0, y0 - radius), min(w, x1 + radius), min(h, y1 + radius)
    tpl = t_gray[y0:y1, x0:x1]
    src = r_gray[sy0:sy1, sx0:sx1]
    if tpl.size > 250000:
        return 0.0, 0.0, 0.0
    res = cv2.matchTemplate(src, tpl, cv2.TM_SQDIFF)
    e0 = float(res[y0 - sy0, x0 - sx0])
    _, _, mn, _ = cv2.minMaxLoc(res)
    e1 = float(res[mn[1], mn[0]])
    dx, dy = (mn[0] - (x0 - sx0)), (mn[1] - (y0 - sy0))
    gain = 1.0 - e1 / e0 if e0 > 1e-6 else 0.0
    return float(dx), float(dy), float(gain)


FEATURES = (["log_nfa", "log_px", "log_box", "page_frac", "aspect", "fill_ratio", "thin",
             "de_mean", "de_max", "dL", "abs_dL", "dab", "chroma_frac", "smooth",
             "ink_t", "ink_r", "ink_ratio", "ink_h_ratio", "edge_ratio", "flow_mag", "flow_gain", "flow_axis", "rev_gain", "flow_consistent",
             "global_frac"]
            + [f"lp_{s}" for s in STATS] + [f"fired_{s}" for s in STATS]
            + ["ir", "text_frac", "icon_frac", "shape_frac", "none_frac", "anchor_iou", "anchor_text", "anchor_icon",
               "anchor_shape", "anchor_stroke", "anchor_radius", "anchor_shadow", "anchor_opacity", "inside_frac",
               "border_frac", "outside_frac", "corner_frac", "box_vs_anchor", "n_nodes", "n_siblings", "sib_axis", "n_parts",
               "anchor_cov", "ring_cov", "edge_sides"])


def region_features(rg: Region, ctx: dict) -> dict:
    """Interpretable features of one region (see FEATURES); IR features are -1 without an IR."""
    b = rg.box
    x0, y0, x1, y1 = int(b.x), int(b.y), int(b.x + b.w), int(b.y + b.h)
    M = rg.mask
    n_px = float(M.sum())
    lt, lr = ctx["lab_t"][y0:y1, x0:x1], ctx["lab_r"][y0:y1, x0:x1]
    d = (lt - lr)[M]
    de = ctx["de"][y0:y1, x0:x1][M]
    dL = float(d[:, 0].mean()) if len(d) else 0.0
    dab = float(np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2).mean()) if len(d) else 0.0
    f = {"log_nfa": rg.log_nfa, "log_px": math.log10(n_px + 1), "log_box": math.log10(b.w * b.h + 1),
         "page_frac": b.w * b.h / ctx["page_area"], "aspect": max(b.w, b.h) / max(1.0, min(b.w, b.h)),
         "fill_ratio": n_px / max(1.0, b.w * b.h), "thin": min(b.w, b.h),
         "de_mean": float(de.mean()) if len(de) else 0.0, "de_max": float(de.max()) if len(de) else 0.0,
         "dL": dL, "abs_dL": float(np.abs(d[:, 0]).mean()) if len(d) else 0.0, "dab": dab,
         "chroma_frac": dab / (abs(dL) + dab + 1e-3)}
    gde = ctx["grad_de"][y0:y1, x0:x1][M]
    f["smooth"] = float(gde.mean() / (f["de_mean"] + 1e-3)) if len(gde) else 0.0
    ink = float(P["adversary.acontrario.ink_de"])
    bt, br = _bg_color(ctx["lab_t"], b), _bg_color(ctx["lab_r"], b)
    it = float((np.sqrt(((lt - bt) ** 2).sum(-1)) > ink).mean())
    ir_ = float((np.sqrt(((lr - br) ** 2).sum(-1)) > ink).mean())
    f.update(ink_t=it, ink_r=ir_, ink_ratio=(it - ir_) / (it + ir_ + 0.01))
    mt = (np.sqrt(((lt - bt) ** 2).sum(-1)) > ink).any(1)
    mr = (np.sqrt(((lr - br) ** 2).sum(-1)) > ink).any(1)
    ht = float(np.ptp(np.nonzero(mt)[0]) + 1) if mt.any() else 0.0
    hr = float(np.ptp(np.nonzero(mr)[0]) + 1) if mr.any() else 0.0
    f["ink_h_ratio"] = (ht - hr) / (ht + hr + 1.0)
    gt_, gr_ = float(ctx["grad_t"][y0:y1, x0:x1].sum()), float(ctx["grad_r"][y0:y1, x0:x1].sum())
    f["edge_ratio"] = (gt_ - gr_) / (gt_ + gr_ + 1.0)
    dx, dy, gain = _flow(ctx["gray_t"], ctx["gray_r"], b, int(P["adversary.acontrario.flow_radius"]))
    f.update(flow_mag=math.hypot(dx, dy), flow_gain=gain, flow_axis=(abs(dx) - abs(dy)) / (abs(dx) + abs(dy) + 1e-6), _dx=dx, _dy=dy)
    # reverse flow (candidate content searched in the target): a genuine move is consistent both ways
    rx, ry, rgain = _flow(ctx["gray_r"], ctx["gray_t"], b, int(P["adversary.acontrario.flow_radius"]))
    f["rev_gain"] = rgain
    f["flow_consistent"] = float(gain > 0.3 and rgain > 0.3 and abs(dx + rx) + abs(dy + ry) <= 1.0 and math.hypot(dx, dy) >= 1)
    f["global_frac"] = ctx.get("global_frac", 0.0)
    for s in STATS:
        f[f"lp_{s}"] = min(300.0, -rg.stat_logp.get(s, 0.0))
        f[f"fired_{s}"] = float(rg.fired == s)
    nodes = ctx["nodes"]
    if ctx.get("ir") is None:
        for k in FEATURES[FEATURES.index("ir"):]:
            f[k] = -1.0
        f["ir"] = 0.0
        return f
    f["ir"] = 1.0
    ys, xs = np.nonzero(M)
    px, py = xs + x0 + 0.5, ys + y0 + 0.5
    inside = {"text": np.zeros(len(xs), bool), "icon": np.zeros(len(xs), bool), "shape": np.zeros(len(xs), bool)}
    best, best_iou, overl = None, 0.0, []
    for nd in nodes:
        e = nd["ext"]
        if e.x > b.x + b.w or e.x + e.w < b.x or e.y > b.y + b.h or e.y + e.h < b.y:
            continue
        if not nd["big"]:
            nb = nd["box"].expand(1)
            inside[nd["kind"]] |= (px >= nb.x) & (px <= nb.x + nb.w) & (py >= nb.y) & (py <= nb.y + nb.h)
            overl.append(nd)
            iou = b.iou(e)
            if iou > best_iou:
                best, best_iou = nd, iou
    n = max(1, len(xs))
    f.update(text_frac=inside["text"].sum() / n, icon_frac=inside["icon"].sum() / n, shape_frac=inside["shape"].sum() / n,
             none_frac=(~(inside["text"] | inside["icon"] | inside["shape"])).sum() / n, anchor_iou=best_iou,
             n_nodes=math.log1p(len(overl)))
    if best is None:
        for k in ("anchor_cov", "ring_cov", "edge_sides", "anchor_text", "anchor_icon", "anchor_shape", "anchor_stroke", "anchor_radius", "anchor_shadow",
                  "anchor_opacity", "inside_frac", "border_frac", "outside_frac", "corner_frac", "box_vs_anchor", "n_siblings", "sib_axis"):
            f[k] = 0.0
        f["outside_frac"] = 1.0
        return f
    ab = best["box"]
    rg.node_id = best["id"]
    f["_parent"] = best["parent"]
    f.update(anchor_text=float(best["kind"] == "text"), anchor_icon=float(best["kind"] == "icon"), anchor_shape=float(best["kind"] == "shape"),
             anchor_stroke=float(best["stroke"]), anchor_radius=best["radius"], anchor_shadow=float(best["shadow"]),
             anchor_opacity=best["opacity"])
    ex = np.minimum.reduce([px - ab.x, ab.x + ab.w - px, py - ab.y, ab.y + ab.h - py])  # >0 inside
    f["inside_frac"] = float((ex > 3).sum() / n)
    f["border_frac"] = float((np.abs(ex) <= 3).sum() / n)
    f["outside_frac"] = float((ex < -3).sum() / n)
    c = max(4.0, best["radius"] + 2.0)
    cx = np.minimum(px - ab.x, ab.x + ab.w - px) < c
    cy = np.minimum(py - ab.y, ab.y + ab.h - py) < c
    f["corner_frac"] = float((cx & cy & (ex > -3)).sum() / n)
    f["box_vs_anchor"] = math.log10((b.w * b.h + 1) / (ab.w * ab.h + 1))
    # how the change sits on the anchor: interior coverage, border-ring coverage and how many sides changed
    ch = ctx["changed"]
    H, W = ch.shape
    ax0, ay0 = max(0, int(round(ab.x))), max(0, int(round(ab.y)))
    ax1, ay1 = min(W, int(round(ab.x + ab.w))), min(H, int(round(ab.y + ab.h)))
    if ax1 - ax0 > 6 and ay1 - ay0 > 6:
        inner = ch[ay0 + 3:ay1 - 3, ax0 + 3:ax1 - 3]
        f["anchor_cov"] = float(inner.mean()) if inner.size else 0.0
        sides = [ch[max(0, ay0 - 2):ay0 + 3, ax0:ax1], ch[ay1 - 3:min(H, ay1 + 2), ax0:ax1],
                 ch[ay0:ay1, max(0, ax0 - 2):ax0 + 3], ch[ay0:ay1, ax1 - 3:min(W, ax1 + 2)]]
        cov = [float(sd.mean()) if sd.size else 0.0 for sd in sides]
        f["ring_cov"] = float(np.mean(cov))
        f["edge_sides"] = float(sum(c_ > 0.3 for c_ in cov))
    else:
        f["anchor_cov"] = f["ring_cov"] = f["edge_sides"] = -1.0
    sib = [o for o in overl if o["parent"] == best["parent"]]
    f["n_siblings"] = math.log1p(len(sib))
    if len(sib) > 1:
        xsib = np.ptp([o["box"].cx for o in sib])
        ysib = np.ptp([o["box"].cy for o in sib])
        f["sib_axis"] = float((xsib - ysib) / (xsib + ysib + 1e-6))
    else:
        f["sib_axis"] = 0.0
    return f


def feature_context(an: Analysis, ir: Optional[Document]) -> dict:
    from dt.compare.pixel import to_lab
    de = an.maps["de"]
    gx = cv2.Sobel(de, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(de, cv2.CV_32F, 0, 1, ksize=3) / 8
    h, w = an.target.shape[:2]
    ancestors: dict = {}
    if ir is not None:
        def walk(n, chain):
            ancestors[n.id] = chain
            for ch in n.children:
                walk(ch, [n.id] + chain)
        walk(ir.root, [])
    return {"changed": de > float(P["adversary.acontrario.px_floor"]), "ancestors": ancestors, "lab_t": to_lab(an.target), "lab_r": to_lab(an.fitted), "de": de, "grad_de": np.sqrt(gx * gx + gy * gy),
            "grad_t": an.maps["grad_t"], "grad_r": an.maps["grad_r"], "gray_t": _gray(an.target), "gray_r": _gray(an.fitted),
            "page_area": float(h * w), "nodes": _ir_nodes(ir), "ir": ir}


# =========================================================================== global test: page-wide colour offset
register("adversary.acontrario.global_eps", 0.01, "a-contrario budget of the single page-wide test (expected false tint findings per page)", (1e-5, 1.0))
register("adversary.acontrario.global_min_de", 0.75, "page-wide colour offset smaller than this ΔE76 is never reported (effect-size floor)", (0.0, 5.0))


def global_offset(an: Analysis) -> tuple[np.ndarray, float, float]:
    """Median Lab offset target - fitted over all pixels, its norm, and the fraction of pixels whose own
    offset agrees with it (within half its norm)."""
    from dt.compare.pixel import to_lab
    lt, lr = to_lab(an.target).reshape(-1, 3), to_lab(an.fitted).reshape(-1, 3)
    d = lt - lr
    m = np.median(d, 0)
    g = float(np.linalg.norm(m))
    agree = float((np.linalg.norm(d - m, axis=1) < max(0.5, g / 2)).mean()) if g > 0 else 0.0
    return m, g, agree


def compensate(fitted: np.ndarray, offset: np.ndarray) -> np.ndarray:
    from skimage.color import lab2rgb
    from dt.compare.pixel import to_lab
    lab = to_lab(fitted) + offset[None, None, :].astype(np.float32)
    return np.clip(np.round(lab2rgb(lab) * 255), 0, 255).astype(np.uint8)


def global_pvalue(g: float, null_g: list) -> float:
    """Empirical H0 survival of the page-offset norm with an exponential tail (same POT rule as tiles)."""
    v = np.sort(np.asarray(null_g, float))
    n = len(v)
    if n == 0:
        return 1.0
    zero = float(P["adversary.acontrario.null.zero"])
    if g <= zero:
        return 1.0
    body = (np.sum(v >= g) + 1) / (n + 1)
    k = max(10, n // 10)
    top = v[-k:]
    u = top[0]
    beta = max(float(np.mean(top - u)), 1e-3) * (1 + float(P["adversary.acontrario.null.beta_ci"]) / math.sqrt(k))
    tail = (k / (n + 1)) * math.exp(-(g - u) / beta) if g > u else 1.0
    return float(min(body, tail))


# =========================================================================== a tiny CART (interpretable classifier)
class Tree:
    """Gini decision tree (CART) with class weights; small, deterministic and printable as rules."""

    def __init__(self, max_depth: int = 6, min_leaf: int = 4, max_thresholds: int = 48, max_features: Optional[int] = None,
                 seed: int = 0) -> None:
        self.max_depth, self.min_leaf, self.max_thr = max_depth, min_leaf, max_thresholds
        self.max_features = max_features
        self._rng = np.random.RandomState(seed)
        self.classes: list[str] = []
        self.features: list[str] = []
        self.root: dict = {}

    def fit(self, X: np.ndarray, y: list[str], features: list[str], weights: Optional[np.ndarray] = None) -> "Tree":
        self.classes = sorted(set(y))
        self.features = list(features)
        yi = np.array([self.classes.index(v) for v in y])
        w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
        self.root = self._grow(X, yi, w, 0)
        return self

    def _dist(self, yi: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.bincount(yi, weights=w, minlength=len(self.classes))

    def _grow(self, X, yi, w, depth) -> dict:
        dist = self._dist(yi, w)
        node = {"dist": (dist / max(dist.sum(), 1e-12)).tolist(), "n": int(len(yi))}
        if depth >= self.max_depth or len(yi) < 2 * self.min_leaf or (dist > 0).sum() <= 1:
            return node
        tot = dist.sum()
        gini0 = 1 - ((dist / tot) ** 2).sum()
        best = (0.0, None, None)
        cols = range(X.shape[1]) if not self.max_features else sorted(self._rng.choice(X.shape[1], self.max_features, replace=False))
        for j in cols:
            xj = X[:, j]
            u = np.unique(xj)
            if len(u) < 2:
                continue
            cand = (u[:-1] + u[1:]) / 2
            if len(cand) > self.max_thr:
                cand = np.unique(np.quantile(xj, np.linspace(0, 1, self.max_thr + 2)[1:-1]))
            order = np.argsort(xj, kind="stable")
            xs, ys, ws = xj[order], yi[order], w[order]
            onehot = np.zeros((len(ys), len(self.classes)))
            onehot[np.arange(len(ys)), ys] = ws
            cum = np.cumsum(onehot, 0)
            cnt = np.arange(1, len(ys) + 1)
            for t in cand:
                i = int(np.searchsorted(xs, t, side="right"))
                if i < self.min_leaf or len(ys) - i < self.min_leaf:
                    continue
                L = cum[i - 1]
                R = cum[-1] - L
                lt, rt = L.sum(), R.sum()
                if lt <= 0 or rt <= 0:
                    continue
                g = gini0 - (lt / tot) * (1 - ((L / lt) ** 2).sum()) - (rt / tot) * (1 - ((R / rt) ** 2).sum())
                if g > best[0] + 1e-12:
                    best = (g, j, float(t))
            del cnt
        if best[1] is None or best[0] < 1e-4:
            return node
        j, t = best[1], best[2]
        m = X[:, j] <= t
        node.update(feature=self.features[j], fi=int(j), thr=float(t), gain=float(best[0]),
                    left=self._grow(X[m], yi[m], w[m], depth + 1), right=self._grow(X[~m], yi[~m], w[~m], depth + 1))
        return node

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        nd = self.root
        while "feature" in nd:
            nd = nd["left"] if x[self.features.index(nd["feature"])] <= nd["thr"] else nd["right"]
        return np.asarray(nd["dist"])

    def rules(self) -> list[str]:
        out: list[str] = []

        def rec(nd, conds):
            if "feature" not in nd:
                d = np.asarray(nd["dist"])
                k = int(np.argmax(d))
                out.append(f"IF {' AND '.join(conds) or 'always'} THEN {self.classes[k]} (p={d[k]:.2f}, n={nd['n']})")
                return
            rec(nd["left"], conds + [f"{nd['feature']} <= {nd['thr']:.3g}"])
            rec(nd["right"], conds + [f"{nd['feature']} > {nd['thr']:.3g}"])

        rec(self.root, [])
        return out

    def importance(self) -> dict[str, float]:
        imp: dict[str, float] = {}

        def rec(nd):
            if "feature" in nd:
                imp[nd["feature"]] = imp.get(nd["feature"], 0.0) + nd["gain"] * nd["n"]
                rec(nd["left"])
                rec(nd["right"])

        rec(self.root)
        s = sum(imp.values()) or 1.0
        return {k: v / s for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}

    def to_dict(self) -> dict:
        def pack(nd: dict) -> dict:
            out = {"n": nd["n"], "d": {str(i): round(float(v), 4) for i, v in enumerate(nd["dist"]) if v > 1e-4}}
            if "feature" in nd:
                out.update(f=nd["feature"], t=nd["thr"], g=nd["gain"], l=pack(nd["left"]), r=pack(nd["right"]))
            return out
        return {"classes": self.classes, "features": self.features, "root": pack(self.root),
                "max_depth": self.max_depth, "min_leaf": self.min_leaf}

    @staticmethod
    def from_dict(d: dict) -> "Tree":
        t = Tree(d.get("max_depth", 6), d.get("min_leaf", 4))
        t.classes, t.features = d["classes"], d["features"]

        def unpack(nd: dict) -> dict:
            if "dist" in nd:  # legacy layout
                return nd
            dist = [0.0] * len(t.classes)
            for i, v in nd["d"].items():
                dist[int(i)] = v
            out = {"n": nd["n"], "dist": dist}
            if "f" in nd:
                out.update(feature=nd["f"], thr=nd["t"], gain=nd["g"], left=unpack(nd["l"]), right=unpack(nd["r"]))
            return out
        t.root = unpack(d["root"])
        return t


# =========================================================================== the adversary
import json as _json  # noqa: E402
import os as _os  # noqa: E402

MODEL_PATH = _os.environ.get("DT_ACONTRARIO_MODEL", _os.path.join(_os.path.dirname(__file__), "acontrario_model.json"))
NONE_CLASS = "_none"
register("adversary.acontrario.merge_anchor", 1, "1 = regions sharing the same IR anchor node are one finding (an element's fragments)", (0, 1))
register("adversary.acontrario.min_type_prob", 0.0, "regions whose best error-type probability is below this are dropped", (0.0, 1.0))
_MODEL: Optional[dict] = None


def load_model(path: Optional[str] = None) -> dict:
    global _MODEL
    if path is None and _MODEL is not None:
        return _MODEL
    with open(path or MODEL_PATH) as f:
        d = _json.load(f)
    typer = None
    if d.get("tree"):
        typer = Forest.from_dict(d["tree"]) if d["tree"].get("kind") == "forest" else Tree.from_dict(d["tree"])
    m = {"null": NullModel.from_dict(d["null"]), "tree": typer,
         "global_null": d.get("global_null", []), "meta": d.get("meta", {})}
    if path is None:
        _MODEL = m
    return m


def save_model(path: str, null: NullModel, tree: Optional[Tree], global_null: list, meta: dict) -> str:
    def rnd(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return float(f"{float(o):.5g}")
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rnd(v) for v in o]
        return o
    with open(path, "w") as f:
        _json.dump(rnd({"null": null.to_dict(), "tree": tree.to_dict() if tree else None,
                        "global_null": sorted(global_null), "meta": meta}), f, separators=(",", ":"))
    return path


def _merge(regions: list[Region]) -> Region:
    from dt.ir import Box
    x0 = min(r.box.x for r in regions)
    y0 = min(r.box.y for r in regions)
    x1 = max(r.box.x + r.box.w for r in regions)
    y1 = max(r.box.y + r.box.h for r in regions)
    m = np.zeros((int(y1 - y0), int(x1 - x0)), bool)
    for r in regions:
        oy, ox = int(r.box.y - y0), int(r.box.x - x0)
        m[oy:oy + r.mask.shape[0], ox:ox + r.mask.shape[1]] |= r.mask
    best = min(regions, key=lambda r: r.log_nfa)
    lp = {}
    for r in regions:
        for k, v in r.stat_logp.items():
            lp[k] = min(lp.get(k, 0.0), v)
    return Region(Box(x0, y0, x1 - x0, y1 - y0), m, best.log_nfa, best.fired, best.tile_scale, lp)


register("adversary.acontrario.group.flow_gain", 0.5, "grouping: a region 'moved' when block matching explains at least this share of its residual", (0.1, 0.95))
register("adversary.acontrario.group.drift_gap", 24, "grouping: moved regions of one row/column closer than this (px) along it are one layout drift", (0, 96))
register("adversary.acontrario.group.orphan_gap", 24, "grouping: target-only regions (no candidate anchor) closer than this (px) are one missing element", (0, 64))


def _gap(a, b) -> float:
    return max(a.x - (b.x + b.w), b.x - (a.x + a.w), a.y - (b.y + b.h), b.y - (a.y + a.h), 0.0)


def group_regions(regions: list[Region], ctx: dict) -> list[Region]:
    """Join the fragments of one element-level change (union-find over three interpretable relations):
    (1) motion: the block-matching displacement of one region lands on another (the vacated and the new
    position of a moved element); (2) layout drift: regions anchored on siblings of one parent that all moved
    along the same axis (a gap change displaces later siblings cumulatively); (3) absence: target-only
    regions with no candidate anchor within ``orphan_gap`` px (pieces of one missing subtree)."""
    n = len(regions)
    if n < 2:
        return regions
    par = list(range(n))

    def find_(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    def union(i, j):
        par[find_(i)] = find_(j)

    g0 = float(P["adversary.acontrario.group.flow_gain"])
    moved = [r.features.get("flow_gain", 0) >= g0 and r.features.get("flow_mag", 0) >= 1 for r in regions]
    for i, r in enumerate(regions):
        if not moved[i]:
            continue
        tb = r.box.translate(r.features["_dx"], r.features["_dy"])
        for j, q in enumerate(regions):
            if j != i and tb.intersect(q.box).area > 0.25 * min(tb.area, q.box.area):
                union(i, j)
    # layout drift: moved regions in one row/column (same axis, overlapping across it, close along it)
    dgap = float(P["adversary.acontrario.group.drift_gap"])
    for i, r in enumerate(regions):
        if not moved[i]:
            continue
        ax = 0 if abs(r.features["_dx"]) >= abs(r.features["_dy"]) else 1
        for j in range(i + 1, n):
            q = regions[j]
            if not moved[j] or (0 if abs(q.features["_dx"]) >= abs(q.features["_dy"]) else 1) != ax:
                continue
            a0, a1 = (r.box.y, r.box.y + r.box.h) if ax == 0 else (r.box.x, r.box.x + r.box.w)
            b0, b1 = (q.box.y, q.box.y + q.box.h) if ax == 0 else (q.box.x, q.box.x + q.box.w)
            across = max(0.0, min(a1, b1) - max(a0, b0)) / max(1.0, min(a1 - a0, b1 - b0))
            same_d = abs(r.features["_dx"] - q.features["_dx"]) + abs(r.features["_dy"] - q.features["_dy"]) <= 1.0
            if (across >= 0.5 or same_d) and _gap(r.box, q.box) <= dgap:
                union(i, j)
    if ctx.get("ir") is not None:
        # layout drift along one container: moved regions whose anchors share a parent or grandparent
        anc = ctx.get("ancestors", {})
        by_anc: dict = {}
        for i, r in enumerate(regions):
            if moved[i] and r.node_id is not None:
                ax = "x" if abs(r.features["_dx"]) >= abs(r.features["_dy"]) else "y"
                for a in anc.get(r.node_id, [])[:2]:
                    by_anc.setdefault((a, ax), []).append(i)
        for (a, ax), idx in by_anc.items():
            for j in idx[1:]:
                r, q = regions[idx[0]], regions[j]
                a0, a1 = (r.box.y, r.box.y + r.box.h) if ax == "x" else (r.box.x, r.box.x + r.box.w)
                b0, b1 = (q.box.y, q.box.y + q.box.h) if ax == "x" else (q.box.x, q.box.x + q.box.w)
                if max(0.0, min(a1, b1) - max(a0, b0)) >= 0.5 * max(1.0, min(a1 - a0, b1 - b0)):
                    union(idx[0], j)
        # corner changes of one shape (radius): regions sitting in two or more corners of the same node box
        corner_of: dict = {}
        for i, r in enumerate(regions):
            for nd in ctx["nodes"]:
                if nd["big"] or nd["kind"] != "shape":
                    continue
                b = nd["box"]
                zone = max(4.0, nd["radius"]) + 4.0
                if r.box.w > 2 * zone + 2 or r.box.h > 2 * zone + 2 and r.box.w > zone * 2:
                    continue
                for cx, cy in ((b.x, b.y), (b.x + b.w, b.y), (b.x, b.y + b.h), (b.x + b.w, b.y + b.h)):
                    if abs(r.box.cx - cx) <= zone and abs(r.box.cy - cy) <= zone:
                        corner_of.setdefault(nd["id"], set()).add(i)
        for idx in corner_of.values():
            idx = sorted(idx)
            if len(idx) >= 2:
                for j in idx[1:]:
                    union(idx[0], j)
        gap = float(P["adversary.acontrario.group.orphan_gap"])
        orphans = [i for i, r in enumerate(regions) if r.features.get("anchor_iou", 0) <= 0 and r.features.get("ink_ratio", 0) > 0.5]
        for a in range(len(orphans)):
            for b in range(a + 1, len(orphans)):
                if _gap(regions[orphans[a]].box, regions[orphans[b]].box) <= gap:
                    union(orphans[a], orphans[b])
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find_(i), []).append(regions[i])
    out = []
    for rs in groups.values():
        if len(rs) == 1:
            out.append(rs[0])
            continue
        mg = _merge(rs)
        mg.features = region_features(mg, ctx)
        mg.features["n_parts"] = float(len(rs))
        out.append(mg)
    for r in out:
        r.features.setdefault("n_parts", 1.0)
    return out


def estimate_magnitude(t: str, f: dict, d_lab: float) -> Optional[float]:
    if t in ("geometry.shift", "text.position"):
        return f["flow_mag"] if f["flow_mag"] > 0 else None
    if t in ("color.fill", "color.stroke"):
        return d_lab
    if t in ("geometry.size", "structure.split", "structure.merge", "layout.spacing"):
        return f["flow_mag"] if f["flow_mag"] > 0 else f["thin"]
    return None


def analyse_regions(an: Analysis, ir: Optional[Document], model: dict) -> tuple[list[Region], Optional[dict], Analysis]:
    """Global test, local a-contrario detection, grouping and features (no typing)."""
    null = model["null"]
    tint = None
    off, g, agree = global_offset(an)
    pg = global_pvalue(g, model["global_null"])
    if g >= float(P["adversary.acontrario.global_min_de"]) and pg < float(P["adversary.acontrario.global_eps"]):
        tint = {"offset": off.tolist(), "de": g, "agree": agree, "p": pg}
        # the page-wide offset is compensated; what it leaves (gamut clipping, blends) is a nuisance with its own
        # calibrated stratum
        an = analyse(an.target, an.candidate, None, fitted=compensate(an.fitted, off), fit=an.fit)
        an.pbin = TINT_BIN
    regions = detect_regions(an, null)
    ctx = feature_context(an, ir)
    ctx["global_frac"] = float(an.maps["jnd"].mean())
    for rg in regions:
        rg.features = region_features(rg, ctx)
    regions = group_regions(regions, ctx)
    if ir is not None and int(P["adversary.acontrario.merge_anchor"]):
        groups: dict = {}
        out = []
        for rg in regions:
            if rg.node_id is not None and rg.features.get("anchor_iou", 0) > 0:
                groups.setdefault(rg.node_id, []).append(rg)
            else:
                out.append(rg)
        for nid, rs in groups.items():
            if len(rs) == 1:
                out.append(rs[0])
                continue
            mg = _merge(rs)
            mg.features = region_features(mg, ctx)
            mg.features["n_parts"] = float(sum(r.features.get("n_parts", 1.0) for r in rs))
            out.append(mg)
        regions = out
    for rg in regions:
        b = rg.box
        x0, y0 = int(b.x), int(b.y)
        d = (ctx["lab_t"] - ctx["lab_r"])[y0:y0 + rg.mask.shape[0], x0:x0 + rg.mask.shape[1]][rg.mask]
        rg.features["_d_lab"] = float(np.linalg.norm(d.mean(0))) if len(d) else 0.0
    return regions, tint, an


def type_regions(regions: list[Region], tree: Optional[Tree]) -> None:
    from dt.adversary.taxonomy import ERROR_TYPES
    for rg in regions:
        if tree is None:
            rg.type, rg.type_prob = "structure.missing", 1.0
            continue
        x = np.array([rg.features.get(k, -1.0) for k in tree.features], float)
        pr = tree.predict_proba(x)
        order = np.argsort(-pr)
        k = int(order[0])
        rg.type, rg.type_prob = tree.classes[k], float(pr[k])
        if rg.type == NONE_CLASS:
            rg.features["_p_none"] = float(pr[k])
        errs = [i for i in order if tree.classes[i] in ERROR_TYPES]
        rg.features["_best_error"] = tree.classes[errs[0]] if errs else None
        rg.features["_best_error_p"] = float(pr[errs[0]]) if errs else 0.0
        rg.features["_p_none"] = float(pr[tree.classes.index(NONE_CLASS)]) if NONE_CLASS in tree.classes else 0.0


def to_findings(regions: list[Region], tint: Optional[dict], page_box, fit: NuisanceFit) -> list:
    from dt.adversary.taxonomy import Finding
    eps = float(P["adversary.acontrario.eps"])
    span = float(P["adversary.acontrario.conf_span"])
    out = []
    if tint is not None:
        out.append(Finding("color.tint_global", page_box, float(tint["de"]), 1.0,
                           {"method": "a-contrario", "p": tint["p"], "offset_lab": [round(v, 3) for v in tint["offset"]],
                            "agree": round(tint["agree"], 3), "nuisance": fit.to_dict()}))
    for rg in regions:
        t = rg.type
        if t == NONE_CLASS:
            # explicit abstention: a meaningful change that looks like nuisance / a fragment
            out.append(Finding("noise.antialias", rg.box, None, 0.0, {"log10_nfa": round(rg.log_nfa, 2), "fired": rg.fired}))
            continue
        if rg.type_prob < float(P["adversary.acontrario.min_type_prob"]):
            continue
        det = min(1.0, max(0.02, (math.log10(eps) - rg.log_nfa) / span))
        conf = det * (1.0 - 0.9 * rg.features.get("_p_none", 0.0))
        out.append(Finding(t, rg.box, estimate_magnitude(t, rg.features, rg.features.get("_d_lab", 0.0)), float(conf),
                           {"method": "a-contrario", "log10_nfa": round(rg.log_nfa, 2), "fired": rg.fired,
                            "tile_scale": rg.tile_scale, "type_prob": round(rg.type_prob, 3),
                            "features": {k: round(float(v), 4) for k, v in rg.features.items() if not k.startswith("_") and k in
                                         ("de_mean", "chroma_frac", "ink_ratio", "flow_mag", "flow_gain", "text_frac", "icon_frac")}},
                           rg.node_id))
    return out


def find(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> list:
    """The a-contrario adversary: nuisance registration in the real renderer, multi-scale tile tests
    against the calibrated H0 (NFA < eps), interpretable typing of the meaningful regions."""
    from dt.ir import Box
    model = load_model()
    ir = copy.deepcopy(candidate_ir) if candidate_ir is not None and target_rgb.shape == candidate_rgb.shape else None
    an = analyse(target_rgb, candidate_rgb, ir)
    regions, tint, _ = analyse_regions(an, ir, model)
    type_regions(regions, model["tree"])
    h, w = target_rgb.shape[:2]
    return to_findings(regions, tint, Box(0, 0, w, h), an.fit)


find.__name__ = "a_contrario"


class Forest:
    """Bagged CART trees with per-split feature subsampling over the same interpretable features. Every
    member is a printable :class:`Tree`; the forest averages their class distributions."""

    def __init__(self, n_trees: int = 31, max_depth: int = 10, min_leaf: int = 2, seed: int = 0) -> None:
        self.n_trees, self.max_depth, self.min_leaf, self.seed = n_trees, max_depth, min_leaf, seed
        self.trees: list[Tree] = []
        self.classes: list[str] = []
        self.features: list[str] = []

    def fit(self, X: np.ndarray, y: list[str], features: list[str], weights: Optional[np.ndarray] = None) -> "Forest":
        rng = np.random.RandomState(self.seed)
        self.classes, self.features = sorted(set(y)), list(features)
        w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
        mf = max(1, int(round(math.sqrt(X.shape[1]))))
        self.trees = []
        for k in range(self.n_trees):
            idx = rng.randint(0, len(y), len(y))
            t = Tree(self.max_depth, self.min_leaf, max_features=mf, seed=int(rng.randint(1 << 30)))
            t.fit(X[idx], [y[i] for i in idx], features, w[idx])
            self.trees.append(t)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.classes))
        for t in self.trees:
            pr = t.predict_proba(x)
            for c, v in zip(t.classes, pr):
                out[self.classes.index(c)] += v
        return out / max(1, len(self.trees))

    def importance(self) -> dict[str, float]:
        imp: dict[str, float] = {}
        for t in self.trees:
            for k, v in t.importance().items():
                imp[k] = imp.get(k, 0.0) + v / len(self.trees)
        return dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict:
        return {"kind": "forest", "classes": self.classes, "features": self.features, "trees": [t.to_dict() for t in self.trees],
                "n_trees": self.n_trees, "max_depth": self.max_depth, "min_leaf": self.min_leaf, "seed": self.seed}

    @staticmethod
    def from_dict(d: dict) -> "Forest":
        f = Forest(d["n_trees"], d["max_depth"], d["min_leaf"], d.get("seed", 0))
        f.classes, f.features = d["classes"], d["features"]
        f.trees = [Tree.from_dict(t) for t in d["trees"]]
        return f


# =========================================================================== calibration driver (CALIBRATION documents only)
register("adversary.acontrario.train.seed", 7, "seed of the extra calibration cases drawn from the CALIBRATION documents (never evaluation ones)", (0, 10 ** 6))
register("adversary.acontrario.train.cases_per_doc", 30, "extra calibration cases per calibration document", (0, 64))
register("adversary.acontrario.tree.depth", 7, "typing tree: maximum depth", (2, 12))
register("adversary.acontrario.tree.min_leaf", 3, "typing tree: minimum (unweighted) samples per leaf", (1, 50))
register("adversary.acontrario.forest.trees", 61, "typing forest: number of bagged trees", (1, 501))
register("adversary.acontrario.forest.depth", 12, "typing forest: maximum depth of each tree", (2, 32))
register("adversary.acontrario.forest.min_leaf", 2, "typing forest: minimum samples per leaf", (1, 50))
register("adversary.acontrario.forest.seed", 0, "typing forest: bootstrap / feature-subsampling seed", (0, 10 ** 6))
register("adversary.acontrario.tree.none_weight", 0.5, "typing tree: weight of the 'no error type' class relative to a balanced error class", (0.0, 4.0))

ROOT_DIR = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK = _os.path.join(ROOT_DIR, "out", "adversary", "a-contrario")


def calibration_cases(bench=None) -> list[tuple[object, str]]:
    """(case, fit-cache dir) for every calibration case: the benchmark's CALIBRATION split plus extra cases
    drawn (seed ``train.seed``) from the same calibration documents."""
    from dt.adversary import benchmark as B
    bench = bench or B.build(out_dir=_os.path.join(ROOT_DIR, "out", "adversary", "bench"))
    items = [(c, _os.path.join(WORK, "cache", "bench")) for c in bench.split("calibration")]
    n = int(P["adversary.acontrario.train.cases_per_doc"])
    if n:
        cfg = B.BenchConfig(seed=int(P["adversary.acontrario.train.seed"]), cases_per_doc=n, calibration_docs=B.DEFAULT_CALIBRATION, evaluation_docs=())
        extra = B.build(cfg, out_dir=_os.path.join(WORK, "train_bench"))
        items += [(c, _os.path.join(WORK, "cache", "train_" + extra.meta.get("key", "x"))) for c in extra.split("calibration")]
    return items


def _analyses(items, use_ir: bool = True, progress=None):
    for k, (c, cd) in enumerate(items):
        fitted, fit = fit_cached(c, cd, use_ir)
        an = analyse(c.target_rgb, c.candidate_rgb, None, fitted=fitted, fit=fit)
        if progress:
            progress(f"[{k + 1}/{len(items)}] {c.id} ir={use_ir} pbin={an.pbin} fit={fit.kinds()}")
        yield c, an


def build_null(analyses: list) -> tuple[NullModel, list]:
    null, pix, gl = NullModel(), _PixelAcc(), []
    for c, an in analyses:
        tint = [g for g in c.gt if g.type == "color.tint_global"]
        if tint:
            # after compensation every remaining difference of a tint page is nuisance (tint is page-wide)
            off = global_offset(an)[0]
            ca = analyse(an.target, an.candidate, None, fitted=compensate(an.fitted, off), fit=an.fit)
            ca.pbin = TINT_BIN
            accumulate_h0(null, pix, ca, [g.box for g in c.gt if g.type != "color.tint_global"])
            continue
        accumulate_h0(null, pix, an, [g.box for g in c.gt])
        gl.append(global_offset(an)[1])
    finalize_null(null, pix, {"pages": len(analyses)})
    return null, gl


def labelled_regions(c, an: Analysis, model: dict) -> list[tuple[Region, str]]:
    """Detected regions of a calibration case with their training label (matched ground-truth type or _none)."""
    from dt.adversary import benchmark as B
    from dt.adversary.taxonomy import Finding
    ir = c.candidate_ir
    regions, tint, _ = analyse_regions(an, ir, model)
    preds = [Finding("structure.missing", r.box) for r in regions]
    gts = [g for g in c.gt if g.type != "color.tint_global"]
    pairs = dict(B.match(preds, gts, c.page))
    return [(r, gts[pairs[i]].type if i in pairs else NONE_CLASS) for i, r in enumerate(regions)]


def train_tree(samples: list[tuple[dict, str]], depth: Optional[int] = None, min_leaf: Optional[int] = None, kind: str = "tree"):
    feats = list(FEATURES)
    X = np.array([[f.get(k, -1.0) for k in feats] for f, _ in samples], float)
    y = [lab for _, lab in samples]
    cnt: dict[str, int] = {}
    for lab in y:
        cnt[lab] = cnt.get(lab, 0) + 1
    err = [k for k in cnt if k != NONE_CLASS]
    mean_err = np.mean([cnt[k] for k in err]) if err else 1.0
    w = np.array([(float(P["adversary.acontrario.tree.none_weight"]) * mean_err / cnt[lab]) if lab == NONE_CLASS else mean_err / cnt[lab]
                  for lab in y])
    if kind == "forest":
        f = Forest(int(P["adversary.acontrario.forest.trees"]), int(depth or P["adversary.acontrario.forest.depth"]),
                   int(min_leaf or P["adversary.acontrario.forest.min_leaf"]), int(P["adversary.acontrario.forest.seed"]))
        return f.fit(X, y, feats, w)
    t = Tree(int(depth or P["adversary.acontrario.tree.depth"]), int(min_leaf or P["adversary.acontrario.tree.min_leaf"]))
    return t.fit(X, y, feats, w)


def calibrate(out_path: Optional[str] = None, progress=print, folds: int = 2) -> dict:
    """Fit the null model and the typing tree on CALIBRATION documents only; write the model JSON.

    The tree is trained on regions detected with a cross-fitted null (null from the other fold of
    documents), so its 'no error' class sees the false alarms a fresh page produces. Returns a report of
    the cross-validated false-alarm rates and typing accuracy."""
    t0 = time.time()
    items = calibration_cases()
    ans = list(_analyses(items, True, progress))
    ans_img = list(_analyses(items, False, progress))
    docs = sorted({c.doc for c, _ in ans})
    fold_of = {d: i % folds for i, d in enumerate(docs)}
    samples: list[tuple[dict, str]] = []
    sample_fold: list[int] = []
    cv = {"pages": 0, "fa_regions": 0, "noise_pages": 0, "noise_pages_fa": 0, "gt": 0, "loc": 0, "by_pbin": {}}
    for k in range(folds):
        null, gl = build_null([(c, a) for c, a in ans + ans_img if fold_of[c.doc] != k])
        model = {"null": null, "tree": None, "global_null": gl}
        for c, an in ans:
            if fold_of[c.doc] != k:
                continue
            lab = labelled_regions(c, an, model)
            samples += [(r.features, y) for r, y in lab]
            sample_fold += [k] * len(lab)
            fa = sum(1 for r, y in lab if y == NONE_CLASS and not _touches(r.box, c.gt))
            cv["pages"] += 1
            cv["fa_regions"] += fa
            pb = cv["by_pbin"].setdefault(str(an.pbin), [0, 0])
            pb[0] += fa
            pb[1] += 1
            if c.kind == "noise":
                cv["noise_pages"] += 1
                cv["noise_pages_fa"] += fa > 0
            cv["gt"] += len([g for g in c.gt if g.type != "color.tint_global"])
            cv["loc"] += sum(1 for _, y in lab if y != NONE_CLASS)
    progress(f"cross-fitted detection: {cv}")
    cv["typing"] = cv_typing(samples, sample_fold, folds, kind="forest")
    cv["typing_single_tree"] = cv_typing(samples, sample_fold, folds, kind="tree")
    progress(f"cross-validated typing: forest {cv['typing']} single tree {cv['typing_single_tree']}")
    tree = train_tree(samples, kind="forest")
    surrogate = train_tree(samples, kind="tree")
    null, gl = build_null(ans + ans_img)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "calibration_pages": len(ans), "docs": docs,
            "cv": cv, "samples": len(samples), "label_counts": {k: sum(1 for _, y in samples if y == k) for k in sorted({y for _, y in samples})},
            "seconds": round(time.time() - t0, 1), "params": {k: v for k, v in P.all().items() if k.startswith("adversary.acontrario.")}}
    meta["surrogate_tree"] = surrogate.to_dict()
    save_model(out_path or MODEL_PATH, null, tree, gl, meta)
    global _MODEL
    _MODEL = None
    import pickle
    _os.makedirs(WORK, exist_ok=True)
    with open(_os.path.join(WORK, "train_samples.pkl"), "wb") as f:
        pickle.dump({"samples": samples, "fold": sample_fold}, f)
    return meta


def cv_typing(samples: list, fold: list, folds: int, depth: Optional[int] = None, min_leaf: Optional[int] = None, kind: str = "forest") -> dict:
    """Document-fold cross-validation of the typing tree on detected regions: leaf accuracy on regions that
    match a ground truth, macro F1 over error types, and how often a matched region is wrongly abstained."""
    from dt.adversary.taxonomy import ERROR_TYPES
    preds, labels = [], []
    for k in range(folds):
        tr = [smp for smp, f in zip(samples, fold) if f != k]
        te = [smp for smp, f in zip(samples, fold) if f == k]
        if not tr or not te:
            continue
        t = train_tree(tr, depth, min_leaf, kind)
        for feats, y in te:
            x = np.array([feats.get(n, -1.0) for n in t.features], float)
            preds.append(t.classes[int(np.argmax(t.predict_proba(x)))])
            labels.append(y)
    f1s = []
    for ty in ERROR_TYPES:
        tp = sum(1 for p_, y in zip(preds, labels) if p_ == y == ty)
        npred = sum(1 for p_ in preds if p_ == ty)
        ngt = sum(1 for y in labels if y == ty)
        if ngt:
            pr, rc = (tp / npred if npred else 0.0), tp / ngt
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    matched = [(p_, y) for p_, y in zip(preds, labels) if y != NONE_CLASS]
    none = [(p_, y) for p_, y in zip(preds, labels) if y == NONE_CLASS]
    return {"n": len(labels), "leaf_acc_matched": round(sum(p_ == y for p_, y in matched) / max(1, len(matched)), 3),
            "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 3),
            "matched_abstained": round(sum(p_ == NONE_CLASS for p_, _ in matched) / max(1, len(matched)), 3),
            "none_recalled": round(sum(p_ == NONE_CLASS for p_, _ in none) / max(1, len(none)), 3)}


def _touches(box, gts: list, pad: float = 4.0) -> bool:
    for g in gts:
        b = g.box.expand(pad)
        if box.x < b.x + b.w and box.x + box.w > b.x and box.y < b.y + b.h and box.y + box.h > b.y:
            return True
    return False


# =========================================================================== evaluation helpers and CLI
register("adversary.acontrario.report.far_pad", 24.0, "report: H0 tiles 'far' from every ground-truth box (beyond JPEG macro-block / blur spill of an error)", (0.0, 128.0))


def h0_false_alarms(c, an: Analysis, model: dict, eps_list: list[float]) -> dict:
    """Meaningful tiles and regions that touch no ground-truth box, per eps (the calibrated false-alarm rate)."""
    null = model["null"]
    nfa = tile_nfa(an, null)
    masks = h0_masks(an, [g.box for g in c.gt])
    far = h0_masks(an, [g.box for g in c.gt], float(P["adversary.acontrario.report.far_pad"]))
    out = {}
    for e in eps_list:
        le = math.log10(e)
        tiles = int(sum(int(((nf["log_nfa"] < le) & k).sum()) for nf, k in zip(nfa, masks)))
        tiles_far = int(sum(int(((nf["log_nfa"] < le) & k).sum()) for nf, k in zip(nfa, far)))
        with _override("adversary.acontrario.eps", e):
            regs = detect_regions(an, null)
        out[e] = {"tiles": tiles, "tiles_far": tiles_far, "regions": sum(1 for r in regs if not _touches(r.box, c.gt))}
    return out


class _override:
    def __init__(self, key: str, value) -> None:
        self.key, self.value = key, value

    def __enter__(self):
        from dt import params as _p
        self.had = self.key in _p._OVERRIDES
        self.old = _p._OVERRIDES.get(self.key)
        _p._OVERRIDES[self.key] = self.value

    def __exit__(self, *a):
        from dt import params as _p
        if self.had:
            _p._OVERRIDES[self.key] = self.old
        else:
            _p._OVERRIDES.pop(self.key, None)


def real_pages(root: Optional[str] = None) -> list[dict]:
    """Run the adversary on translated real pages (``out/real_r3m/*``: validation/target.png vs render.png,
    IR = ir.mapped.json)."""
    import glob
    from dt.common.image import load_rgb
    root = root or _os.path.join(ROOT_DIR, "out", "real_r3m")
    model = load_model()
    out = []
    for d in sorted(glob.glob(_os.path.join(root, "*", ""))):
        tp, rp, ip = _os.path.join(d, "validation", "target.png"), _os.path.join(d, "render.png"), _os.path.join(d, "ir.mapped.json")
        if not (_os.path.exists(tp) and _os.path.exists(rp)):
            continue
        ir = Document.load(ip) if _os.path.exists(ip) else None
        t, r = load_rgb(tp), load_rgb(rp)
        t0 = time.time()
        an = analyse(t, r, copy.deepcopy(ir) if ir is not None else None)
        regions, tint, an2 = analyse_regions(an, ir if t.shape == r.shape else None, model)
        type_regions(regions, model["tree"])
        from dt.ir import Box
        fs = to_findings(regions, tint, Box(0, 0, t.shape[1], t.shape[0]), an.fit)
        secs = time.time() - t0
        n_tiles = int(sum(int((nf["log_nfa"] < math.log10(float(P["adversary.acontrario.eps"]))).sum()) for nf in tile_nfa(an2, model["null"])))
        # cross-check against the independent validator (dt.validate): its worst flat regions and OCR text lines
        # with a character error -- how many does a meaningful finding cover?
        xval = {}
        vp = _os.path.join(d, "validation", "validation.json")
        if _os.path.exists(vp):
            with open(vp) as f:
                v = _json.load(f)
            errs = [x for x in fs if not x.type.startswith("noise.")]

            def covered(b) -> bool:
                return any(e.box.intersect(b).area > 0 for e in errs)

            wr = [Box.from_dict(w["box"]) for w in v.get("worst_regions", []) if w.get("delta_e", 0) > 10]
            tl = [Box.from_dict(m["target_box"]) for m in v.get("text_matches", []) if m.get("cer", 0) > 0 and m.get("target_box")]
            xval = {"worst_regions_de10": len(wr), "worst_regions_covered": sum(covered(b) for b in wr),
                    "text_lines_cer": len(tl), "text_lines_covered": sum(covered(b) for b in tl),
                    "findings_on_validator_evidence": sum(1 for e in errs if any(e.box.intersect(b).area > 0 for b in wr + tl)),
                    "gates": v.get("gates")}
        out.append({"name": _os.path.basename(_os.path.dirname(d)), "size": [t.shape[1], t.shape[0]], "fit": an.fit.to_dict(), "xval": xval,
                    "nu": [round(v, 4) for v in an.nu], "pbin": an.pbin, "tint": tint, "meaningful_tiles": n_tiles,
                    "findings": [f.to_dict() for f in fs], "seconds": round(secs, 2), "dir": d,
                    "fired": {s: sum(1 for rg in regions if rg.fired == s) for s in STATS}})
    return out


def overlay(target: np.ndarray, candidate: np.ndarray, findings: list, gt: Optional[list] = None, path: Optional[str] = None) -> np.ndarray:
    """Evidence panel: target | candidate | ΔE heatmap; ground truth green, findings red (abstentions grey)."""
    from dt.compare.pixel import diff_map
    from dt.compare.visualize import draw_boxes, heatmap
    from dt.common.image import save_rgb
    from dt.adversary.taxonomy import is_noise
    hm = heatmap(diff_map(target, candidate))
    panels = []
    for img in (target, candidate, hm):
        if gt:
            img = draw_boxes(img, [g.box for g in gt], (0, 170, 0), 2)
        img = draw_boxes(img, [f.box for f in findings if is_noise(f.type)], (150, 150, 150), 1)
        img = draw_boxes(img, [f.box for f in findings if not is_noise(f.type)], (220, 0, 0), 1)
        panels.append(img)
    sep = np.full((target.shape[0], 6, 3), 255, np.uint8)
    im = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    scale = min(1.0, 2400 / im.shape[1])
    if scale < 1.0:
        im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if path:
        save_rgb(im, path)
    return im


EPS_SWEEP = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def evaluate(out_dir: Optional[str] = None, splits: tuple = ("calibration", "evaluation"), progress=print) -> dict:
    """Score with ``benchmark.score`` (live nuisance fit), sweep eps for the false-alarm calibration curve,
    run the real pages, save evidence panels; writes ``<out>/a-contrario.json`` (the markdown report is
    written by :func:`write_report`)."""
    from dt.adversary import benchmark as B
    out_dir = out_dir or _os.path.join(ROOT_DIR, "out", "adversary")
    bench = B.build(out_dir=_os.path.join(ROOT_DIR, "out", "adversary", "bench"))
    model = load_model()
    res: dict = {"model_meta": model["meta"], "reports": {}, "sweep": {}, "fits": {}}
    for split in splits:
        progress(f"scoring {split} ...")
        cases = bench.split(split)
        rows = B.run(find, cases, True)
        res["reports"][split] = B.aggregate(rows, "a_contrario", split)
        res["fits"][split] = [{"id": r["case"].id, "noise": r["case"].noise, "seconds": r["seconds"]} for r in rows]
        res.setdefault("rows", {})[split] = [{"id": r["case"].id, "kind": r["case"].kind, "family": r["case"].family,
                                             "noise": r["case"].noise, "gt": [g.to_dict() for g in r["case"].gt],
                                             "preds": [p.to_dict() for p in r["preds"]], "pairs": r["pairs"], "seconds": r["seconds"]}
                                            for r in rows]
        # eps sweep on the H0 part of every case (fits cached; same procedure as find)
        sw = {str(e): {"tiles": 0, "tiles_far": 0, "regions": 0} for e in EPS_SWEEP}
        sw_noise = {str(e): {"pages_with_fa": 0, "tiles": 0, "regions": 0} for e in EPS_SWEEP}
        pages = noise_pages = 0
        for c in cases:
            fitted, fit = fit_cached(c, _os.path.join(WORK, "cache", "bench"), True)
            an = analyse(c.target_rgb, c.candidate_rgb, None, fitted=fitted, fit=fit)
            fa = h0_false_alarms(c, an, model, list(EPS_SWEEP))
            pages += 1
            noise_pages += c.kind == "noise"
            for e, v in fa.items():
                sw[str(e)]["tiles"] += v["tiles"]
                sw[str(e)]["tiles_far"] += v["tiles_far"]
                sw[str(e)]["regions"] += v["regions"]
                if c.kind == "noise":
                    sw_noise[str(e)]["pages_with_fa"] += v["regions"] > 0
                    sw_noise[str(e)]["tiles"] += v["tiles"]
                    sw_noise[str(e)]["regions"] += v["regions"]
        res["sweep"][split] = {"pages": pages, "noise_pages": noise_pages,
                               "per_eps": {e: {"tiles_per_page": v["tiles"] / pages, "tiles_far_per_page": v["tiles_far"] / pages,
                                               "regions_per_page": v["regions"] / pages,
                                               "noise_tiles_per_page": sw_noise[e]["tiles"] / max(1, noise_pages),
                                               "noise_regions_per_page": sw_noise[e]["regions"] / max(1, noise_pages),
                                               "noise_pages_with_fa": sw_noise[e]["pages_with_fa"]} for e, v in sw.items()}}
    progress("real pages ...")
    res["real"] = real_pages()
    from dt.common.image import load_rgb
    _os.makedirs(_os.path.join(WORK, "real"), exist_ok=True)
    for r in res["real"]:
        t = load_rgb(_os.path.join(r["dir"], "validation", "target.png"))
        k = load_rgb(_os.path.join(r["dir"], "render.png"))
        overlay(t, k, [Finding_from(f) for f in r["findings"]], None, _os.path.join(WORK, "real", r["name"] + ".png"))
    progress("examples ...")
    ex = []
    _os.makedirs(_os.path.join(WORK, "examples"), exist_ok=True)
    rows = {r["id"]: r for r in res["rows"]["evaluation"]}
    picks = [c for c in bench.split("evaluation") if c.kind == "perturbed"][::7][:6] + [c for c in bench.split("evaluation") if c.kind == "noise"][:2]
    for c in picks:
        fs = [Finding_from(f) for f in rows[c.id]["preds"]]
        pth = _os.path.join(WORK, "examples", c.id + ".png")
        overlay(c.target_rgb, c.candidate_rgb, fs, c.gt, pth)
        ex.append(_os.path.relpath(pth, ROOT_DIR))
    res["examples"] = ex
    with open(_os.path.join(out_dir, "a-contrario.json"), "w") as f:
        _json.dump(res, f, indent=1, default=str)
    return res


def _pct(v: float) -> str:
    return f"{100 * v:.1f}"


def write_report(res: dict, path: str, baseline_json: Optional[str] = None, narrative: str = "") -> str:
    """Markdown report of an :func:`evaluate` result (tables only; ``narrative`` is prepended verbatim)."""
    from dt.adversary.taxonomy import ERROR_TYPES
    baseline_json = baseline_json or _os.path.join(ROOT_DIR, "out", "adversary", "benchmark_baseline.json")
    base = []
    if _os.path.exists(baseline_json):
        with open(baseline_json) as f:
            base = _json.load(f).get("reports", [])
    reps = base + [res["reports"][k] for k in ("calibration", "evaluation") if k in res["reports"]]
    L = [narrative.rstrip(), "", "## Score table", "",
         "| adversary | split | headline | det P | det R | det F1 | AP | macro leaf F1 | macro parent F1 | type acc leaf / parent | noise false-case rate | spurious / perturbed case (clean / noisy target) | s/case (mean / p95) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        d, n = r["detection"], r["noise"]
        L.append(f"| {r['adversary']} | {r['split']} | {r['headline']:.3f} | {_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} | {_pct(r['ap'])} | "
                 f"{_pct(r['macro_leaf']['f1'])} | {_pct(r['macro_parent']['f1'])} | {_pct(r['type_accuracy']['leaf'])} / {_pct(r['type_accuracy']['parent'])} | "
                 f"{_pct(n['false_case_rate'])} | {n['spurious_per_perturbed_case_clean_target']:.2f} / {n['spurious_per_perturbed_case_noisy_target']:.2f} | "
                 f"{r['runtime']['mean_s']:.2f} / {r['runtime']['p95_s']:.2f} |")
    L += ["", "headline = 0.4·detection F1 + 0.3·macro leaf F1 + 0.3·(1 − noise false-case rate)·detection recall (benchmark definition). "
          "Percentages unless noted. Baseline rows are copied from `out/adversary/benchmark_baseline.json` (same benchmark key).", ""]
    for split in ("evaluation", "calibration"):
        r = res["reports"].get(split)
        if not r:
            continue
        L += [f"### Per type ({split})", "", "| type | gt | pred | typed P | typed R | typed F1 | loc recall | magnitude MAE (n) | median rel. err |", "|---|---|---|---|---|---|---|---|---|"]
        for t, d in r["per_type"].items():
            mae = f"{d['mag_mae']:.2f} ({d['mag_n']})" if "mag_mae" in d else "–"
            rel = f"{d['mag_rel_median']:.2f}" if "mag_rel_median" in d else "–"
            L.append(f"| `{t}` | {d['gt']} | {d['pred']} | {_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} | {_pct(d['loc_recall'])} | {mae} | {rel} |")
        L += ["", "Recall (localised, any type) by magnitude bin: " + ", ".join(f"{b} {_pct(v)}" for b, v in r["recall_by_bin"].items()) + ". "
              "By family: " + ", ".join(f"{f} recall {_pct(v['recall'])} ({v['gt']} gt, {v['preds']} preds)" for f, v in r["by_family"].items()) + ".", ""]
        if split == "evaluation":
            cols = sorted({k for row in r["confusion"].values() for k in row}, key=lambda k: (k.startswith("("), k))
            L += ["### Confusion matrix (evaluation; rows: ground truth, columns: matched prediction type)", "",
                  "| gt \\ pred | " + " | ".join(f"`{c.split('.')[-1] if not c.startswith('(') else c}`" for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
            for g in [t for t in ERROR_TYPES if t in r["confusion"]] + (["(spurious)"] if "(spurious)" in r["confusion"] else []):
                L.append(f"| `{g}` | " + " | ".join(str(r["confusion"][g].get(c, "")) for c in cols) + " |")
            L += ["", "Column headers are leaf names (`fill` = `color.fill`, …)."]
            L.append("")
    L += ["## Calibrated false-alarm rates", "",
          "A tile is *meaningful* when NFA = N_tests · p_H0 < eps (N_tests = tiles × statistics on the page, ~10^5–10^6). "
          "The a-contrario bound is E[#meaningful tiles | H0] ≤ eps per page. Measured on the pure-nuisance part of every page "
          "(tiles / regions touching no ground-truth box), with the nuisance fitted exactly as `find` does.", "",
          "| split | eps | pure-noise pages: meaningful tiles / page | pure-noise pages: false regions / page | pure-noise pages with a false region | all pages: H0 tiles > far_pad from any error / page | all pages: H0 tiles (pad 4) / page | all pages: false regions / page |",
          "|---|---|---|---|---|---|---|---|"]
    cvm = res.get("model_meta", {}).get("cv", {})
    for split, sw in res.get("sweep", {}).items():
        for e, v in sw["per_eps"].items():
            L.append(f"| {split} | {float(e):g} | {v.get('noise_tiles_per_page', float('nan')):.3f} | {v.get('noise_regions_per_page', float('nan')):.3f} | "
                     f"{v['noise_pages_with_fa']}/{sw['noise_pages']} | {v.get('tiles_far_per_page', float('nan')):.3f} | {v['tiles_per_page']:.3f} | {v['regions_per_page']:.3f} |")
    if cvm:
        L += ["", f"Document-level 2-fold cross-fitting on the calibration documents (null model from the other fold, eps = "
              f"{P['adversary.acontrario.eps']}): {cvm.get('fa_regions', 0)} false regions on {cvm.get('pages', 0)} pages "
              f"({cvm.get('fa_regions', 0) / max(1, cvm.get('pages', 1)):.3f} / page); pure-noise pages with a false region "
              f"{cvm.get('noise_pages_fa', 0)}/{cvm.get('noise_pages', 0)}; per page stratum (false regions, pages): {cvm.get('by_pbin')}."]
    L.append("")
    try:
        model = load_model()
    except Exception:
        model = None
    if model is not None and model.get("tree") is not None:
        imp = model["tree"].importance()
        L += ["## Typing: interpretable features", "",
              f"Typer: bagged forest of {len(getattr(model['tree'], 'trees', [model['tree']]))} CART trees over {len(FEATURES)} interpretable features. "
              "Cross-validated on calibration documents (2 document folds): forest "
              f"{cvm.get('typing', {})}; a single tree {cvm.get('typing_single_tree', {})}.", "",
              "Feature importance (mean Gini gain share):", "",
              "| feature | share |", "|---|---|"] + [f"| `{k}` | {v:.3f} |" for k, v in list(imp.items())[:20]] + [""]
        sur = model.get("meta", {}).get("surrogate_tree")
        if sur:
            rules = Tree.from_dict(sur).rules()
            L += ["Surrogate single tree (same features, trained on the same calibration regions; its leaves are the printable rules "
                  f"that approximate the forest), {len(rules)} leaves; the 20 largest:", "", "```"]
            L += sorted(rules, key=lambda r: -int(r.rsplit("n=", 1)[1].rstrip(")")))[:20] + ["```", ""]
    return "\n".join(L)


def Finding_from(d: dict):
    from dt.adversary.taxonomy import Finding
    return Finding.from_dict(d)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="A-contrario adversary: calibrate (calibration docs only), evaluate, real pages, report.")
    ap.add_argument("cmd", choices=["calibrate", "evaluate", "all"])
    a = ap.parse_args(argv)
    if a.cmd in ("calibrate", "all"):
        meta = calibrate()
        print(_json.dumps(meta["cv"], indent=1))
    if a.cmd in ("evaluate", "all"):
        evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
