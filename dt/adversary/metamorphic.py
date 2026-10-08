"""Metamorphic and robustness testing of the translator: adversarial comparison without ground truth.

A correct translator ``T`` (screenshot -> IR) commutes with simple, exactly-known edits of its input:
for an input transform ``t`` with a known effect ``t*`` on the IR, ``T(t(x)) == t*(T(x))``. Neither
side needs ground truth; a violation (beyond tolerance) *proves* that ``T`` is wrong on ``x`` or on
``t(x)`` -- it cannot be a false alarm of a pixel comparison, because no pixels are compared.

Relations (``RELATIONS``; each one is also a *scenario family*: a violation is regenerated from its
spec by :func:`regenerate`, so a failure becomes a reproducible test case):

==================  ===========================================  ===============================================
relation            input transform ``t``                        expected IR transform ``t*``
==================  ===========================================  ===============================================
``translate``       pad top/left by (dx, dy) with the page bg    every node box translated by (dx, dy)
``recolor_tint``    uniform Lab offset on every pixel            every colour offset the same way; geometry fixed
``recolor_swap``    RGB -> BGR channel permutation               every colour permuted; geometry fixed
``dpr2``            2x capture (native for IR inputs) + dpr=auto identical to the 1x translation
``row_add``         duplicate one row band of a repeated list    one more item (copies of the row), rest shifted
``row_remove``      delete one row band of a repeated list       one item fewer, rest shifted
``crop``            keep the page below a gap line               nodes inside kept (translated), none invented
``mirror``          RTL mirror (glyphs kept readable)            boxes mirrored, radii/align mirrored
``jpeg``            JPEG re-encoding (nuisance)                  identical translation
==================  ===========================================  ===============================================

The comparison of two IRs is :func:`ir_diff`, a typed differ: nodes are paired (Hungarian over
overlap, distance, size and text similarity) and every disagreement becomes a
:class:`~dt.adversary.taxonomy.Finding` typed by the shared taxonomy (shift/size/radius, fill/stroke,
missing/extra/split/merge, text content/style/position, icon glyph/missing, shadow, opacity,
spacing, global tint), plus two out-of-taxonomy aspects that only an IR comparison can see:
``component.identity`` (matched component/variant differs) and ``geometry.size`` with
``aspect="dpr"`` (wrong device-pixel-ratio detection).

The same differ turns the pairwise setting into a differential test, :func:`find`: translate the
target and the candidate with the same ``T`` and diff the two IRs (the identity relation
"pixel-identical regions translate identically"). Disagreements are kept where the pixels actually
differ (``adversary.meta.gate_*``), so translator noise cancels. This is the benchmark adversary.

Robustness envelopes (minimum contrast / size / spacing at which perception still finds each
component type) live in :mod:`dt.adversary.fuzz`.

CLI: ``python -m dt.adversary.metamorphic [--calibrate] [--score] [--real] [--synthetic]``
writes ``out/adversary/metamorphic.md`` and ``out/adversary/metamorphic/*.json``.
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from dt.adversary.taxonomy import TYPES, Finding
from dt.ir import Box, Color, Document, Node, rgb_to_lab
from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "out", "adversary", "metamorphic")
CACHE_DIR = os.environ.get("DT_META_CACHE", os.path.join(OUT, "cache"))
META_VERSION = "1"
OUTSIDE_TAXONOMY = ("component.identity",)

# --------------------------------------------------------------------------- params
# Defaults marked (cal) were chosen on the CALIBRATION split only (see out/adversary/metamorphic.md).
register("adversary.meta.pos_tol", 1.0, "px: box-edge displacement tolerated between the expected and the actual translation", (0.0, 8.0))
register("adversary.meta.size_tol", 1.5, "px: width/height change tolerated before a moved box counts as resized", (0.0, 8.0))
register("adversary.meta.text_slack", 2.0, "px: extra edge tolerance for text runs (OCR line boxes jitter by a pixel or two)", (0.0, 8.0))
register("adversary.meta.color_tol", 6.0, "ΔE76 tolerated between expected and actual node colours (perceived colours jitter)", (1.0, 30.0))
register("adversary.meta.radius_tol", 2.0, "px: corner-radius difference tolerated", (0.0, 12.0))
register("adversary.meta.shadow_tol", 2.0, "px: shadow reach difference tolerated", (0.0, 12.0))
register("adversary.meta.text_size_tol", 1.0, "px: text size difference tolerated", (0.0, 6.0))
register("adversary.meta.match_dist", 40.0, "px: max centre distance for pairing two nodes that do not overlap", (4.0, 200.0))
register("adversary.meta.match_min", 0.35, "min pairing score (IoU + proximity + size and text similarity)", (0.0, 2.0))
register("adversary.meta.split_cover", 0.6, "split/merge: IoU of the two parts' union with the whole", (0.3, 0.95))
register("adversary.meta.tint_frac", 0.6, "global tint: min fraction of compared colours that changed", (0.2, 1.0))
register("adversary.meta.tint_spread", 0.4, "global tint: max relative spread of the Lab offsets of changed colours", (0.05, 1.0))
register("adversary.meta.tint_min_pairs", 4, "global tint: min compared colours", (2, 50))
register("adversary.meta.opacity_resid", 0.15, "opacity: max relative residual for reading a colour change as a fade towards the backdrop", (0.02, 0.5))
register("adversary.meta.gate_de", 20.0, "differential adversary: ΔE76 (target vs candidate) for a pixel to count as evidence (cal)", (2.0, 40.0))
register("adversary.meta.gate_px", 16, "differential adversary: min evidence pixels inside a finding box (cal)", (1, 400))
register("adversary.meta.find_color_tol", 10.0, "differential adversary: ΔE76 colour tolerance of the IR diff (relations use color_tol) (cal)", (1.0, 30.0))
register("adversary.meta.gate_open", 2, "differential adversary: morphological opening (k x k) of the evidence mask; removes 1 px edge seams of nuisance (cal)", (0, 5))
register("adversary.meta.gate_blur", 0.8, "differential adversary: Gaussian sigma applied to both images before the evidence ΔE (cal)", (0.0, 2.0))
register("adversary.meta.fallback", False, "differential adversary: also emit residual regions no IR finding explains, as low-confidence structure.missing (cal)")
register("adversary.meta.fallback_conf", 0.15, "confidence of residual fallback findings (ranked below every IR-typed finding)", (0.0, 1.0))
register("adversary.meta.translate", [3, 5], "translate relation: (dx, dy) px, deliberately off the 4 px grid")
register("adversary.meta.tint_de", 8.0, "recolor_tint relation: ΔE76 of the uniform Lab offset", (1.0, 30.0))
register("adversary.meta.jpeg_quality", 85, "jpeg relation: JPEG quality", (40, 100))
register("adversary.meta.min_rows", 3, "row relations: min repeated siblings that form a list", (2, 10))
register("adversary.meta.row_tol", 3.0, "row relations: px tolerance on row height and width", (0.0, 12.0))
register("adversary.meta.row_ratio", 1.6, "row relations: max height ratio of two rows of one list (1- and 2-line items mix)", (1.0, 3.0))
register("adversary.meta.crop_frac", 0.4, "crop relation: keep the page below the gap line nearest this fraction of the height", (0.1, 0.9))
register("adversary.meta.blind_jnd", 0.02, "validator emulation: a region is flagged when its non-text JND fraction exceeds this (visually-identical gate)", (0.0, 1.0))
register("adversary.meta.blind_region_de", 5.0, "validator emulation: a flat target region with ΔE2000 >= this is flagged (worst-region gate)", (1.0, 30.0))
register("adversary.meta.blind_dpos", 2.0, "validator emulation: a text line moved by more than this (px) is flagged", (0.0, 10.0))


@contextmanager
def overrides(**kv):
    """Temporarily set ``adversary.meta.<k>`` params (calibration, tests)."""
    try:
        for k, v in kv.items():
            P.set(f"adversary.meta.{k}", v)
        yield
    finally:
        for k in kv:
            P.reset(f"adversary.meta.{k}")


# --------------------------------------------------------------------------- translator under test (+ cache)
_SRC: dict[str, str] = {}
_DS = None


def translator_version(stage: str = "perceive") -> str:
    """Hash of the translator source + its params: cached translations are reused only while it holds."""
    if stage not in _SRC:
        import dt.perceive  # noqa: F401  (registers perceive.* params)
        dirs = ["perceive"] + (["mapping"] if stage == "map" else [])
        h = hashlib.sha1(META_VERSION.encode())
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(ROOT, "dt", d, "*.py"))):
                h.update(open(f, "rb").read())
        if stage == "map":
            import dt.mapping  # noqa: F401
        prefixes = ("perceive.", "map.") if stage == "map" else ("perceive.",)
        h.update(json.dumps({k: v for k, v in sorted(P.all().items()) if k.startswith(prefixes)}, default=str).encode())
        _SRC[stage] = h.hexdigest()[:12]
    return _SRC[stage]


def image_key(rgb: np.ndarray) -> str:
    a = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    return hashlib.sha1(str(a.shape).encode() + a.tobytes()).hexdigest()[:16]


def _design_system():
    global _DS
    if _DS is None:
        from dt.mapping.material3 import material3
        _DS = material3()
    return _DS


def cache_path(rgb: np.ndarray, stage: str = "perceive", dpr: "float | str" = 1.0) -> str:
    key = hashlib.sha1(f"{image_key(rgb)}|{stage}|{dpr}|{translator_version(stage)}".encode()).hexdigest()[:20]
    return os.path.join(CACHE_DIR, key + ".json")


def translate_ir(rgb: np.ndarray, stage: str = "perceive", dpr: "float | str" = 1.0, cache: bool = True) -> Document:
    """The translator under test: ``perceive`` (and ``map`` onto Material 3 when ``stage == "map"``).

    ``dpr="auto"`` runs the pipeline's DPR detection first (``dt.perceive.dpr.detect_dpr``). Ids are
    renumbered in pre-order (perceive assigns random ids, which would make every diff noisy).
    Results are cached on disk by image hash + :func:`translator_version`."""
    path = cache_path(rgb, stage, dpr)
    key = os.path.basename(path)[:-5]
    if cache and os.path.exists(path):
        try:
            return Document.load(path)
        except Exception:
            pass
    from dt.common.image import downscale_dpr
    from dt.ir import assign_ids
    from dt.perceive import perceive
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    used = 1.0
    if isinstance(dpr, str):
        from dt.perceive.dpr import detect_dpr
        used, _ev = detect_dpr(src)
        used = float(used)
    else:
        used = float(dpr)
    if used != 1.0:
        src = downscale_dpr(src, used)
    doc = perceive(src)
    if stage == "map":
        from dt.mapping import map_document
        doc = map_document(doc, _design_system())
    assign_ids(doc.root)
    doc.source_image = None
    doc.meta = {"metamorphic": {"stage": stage, "dpr": used, "dpr_spec": str(dpr), "key": key, "translator": translator_version(stage)}}
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        doc.save(tmp)
        os.replace(tmp, path)
        return Document.load(path)  # identical whether fresh or cached (JSON round trip)
    return doc


def _job(args: tuple) -> str:
    img, stage, dpr = args
    if isinstance(img, str):
        from dt.common.image import load_rgb
        img = load_rgb(img)
    translate_ir(img, stage, dpr)
    return "ok"


def precompute(jobs: list[tuple], workers: int = 3, progress: Optional[Callable[[str], None]] = None) -> None:
    """Translate ``[(rgb | path, stage, dpr), ...]`` into the cache with ``workers`` processes (each has
    its own browser). Already-cached jobs are skipped by the workers themselves."""
    if workers <= 1 or len(jobs) <= 1:
        for k, j in enumerate(jobs):
            _job(j)
            if progress and k % 10 == 0:
                progress(f"translated {k + 1}/{len(jobs)}")
        return
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for k, _ in enumerate(ex.map(_job, jobs, chunksize=1)):
            if progress and k % 10 == 0:
                progress(f"translated {k + 1}/{len(jobs)}")


# --------------------------------------------------------------------------- node helpers
def _solid(n: Node) -> Optional[Color]:
    for f in n.fills:
        if f.kind == "solid" and f.color is not None and f.color.a > 0.02 and f.opacity > 0.02:
            return f.color
    return None


def paints(n: Node) -> bool:
    if not n.visible or n.opacity <= 0.02:
        return False
    if n.type == "text":
        return bool((n.text or "").strip())
    if n.type in ("icon", "image", "vector"):
        return True
    return _solid(n) is not None or bool(n.strokes) or bool(n.effects)


def geom(n: Node) -> str:
    t = n.type
    if t == "instance" and isinstance(n.meta, dict):
        t = n.meta.get("orig_type") or t
    return t


def kind(n: Node) -> str:
    t = geom(n)
    return "text" if t == "text" else ("icon" if t in ("icon", "vector") else "shape")


def reach(n: Node) -> float:
    return max([abs(e.dx) + abs(e.dy) + e.blur + max(0.0, e.spread) for e in n.effects if not e.inner] or [0.0])


def ext(n: Node) -> Box:
    r = reach(n)
    return n.box.expand(r) if r else n.box


def node_color(n: Node) -> Optional[Color]:
    if n.type == "text":
        return n.text_style.color if n.text_style is not None else None
    return _solid(n)


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").split())


def _cer(a: Optional[str], ref: Optional[str]) -> float:
    from dt.compare.structural import levenshtein
    x, r = _norm(a), _norm(ref)
    if not r:
        return 0.0 if not x else 1.0
    return min(1.0, levenshtein(x, r) / len(r))


def _mean_radius(n: Node) -> float:
    cap = min(n.box.w, n.box.h) / 2
    return float(np.mean([min(r, cap) for r in n.radius]))


def _rgb(c: Color) -> np.ndarray:
    return np.array([c.r, c.g, c.b], dtype=np.float64)


@dataclass
class _Item:
    node: Node
    anc: tuple[str, ...]          # ancestor ids, root first

    @property
    def parent(self) -> Optional[str]:
        return self.anc[-1] if self.anc else None


def _items(doc: Document, include: Optional[Callable[[Node], bool]], painted_only: bool) -> list[_Item]:
    out: list[_Item] = []

    def rec(n: Node, anc: tuple[str, ...]) -> None:
        for c in n.children:
            if not c.visible:
                continue
            if (not painted_only or paints(c)) and (include is None or include(c)):
                out.append(_Item(c, anc))
            rec(c, anc + (c.id,))

    rec(doc.root, (doc.root.id,))
    return out


def _backdrop(doc: Document, item: _Item) -> Color:
    """Nearest painted ancestor fill (what a fading node blends towards)."""
    by_id = {n.id: n for n in doc.walk()}
    for aid in reversed(item.anc):
        a = by_id.get(aid)
        if a is not None and _solid(a) is not None and a is not item.node:
            return _solid(a)
    return Color(255, 255, 255)


# --------------------------------------------------------------------------- pairing
def _pair(E: list[_Item], A: list[_Item]) -> list[tuple[int, int, float]]:
    if not E or not A:
        return []
    from scipy.optimize import linear_sum_assignment
    maxd = float(P["adversary.meta.match_dist"])
    S = np.full((len(E), len(A)), -1e6)
    for i, ei in enumerate(E):
        e = ei.node
        ke = kind(e)
        for j, aj in enumerate(A):
            a = aj.node
            ka = kind(a)
            if ke != ka and not ({ke, ka} == {"icon", "shape"}):
                continue
            iou = e.box.iou(a.box)
            d = math.hypot(e.box.cx - a.box.cx, e.box.cy - a.box.cy)
            if iou < 0.05 and d > maxd:
                continue
            sw = min(e.box.w, a.box.w) / max(e.box.w, a.box.w, 1e-6)
            sh = min(e.box.h, a.box.h) / max(e.box.h, a.box.h, 1e-6)
            ss = sw * sh
            if ss < 0.2 and iou < 0.3:
                continue
            s = iou + 0.5 * math.exp(-d / 8.0) + 0.3 * ss + 0.2 * (geom(e) == geom(a)) - 0.3 * (ke != ka)
            if ke == "text":
                ts = 1.0 - _cer(a.text, e.text)
                if ts < 0.3 and iou < 0.5:
                    continue
                s += 0.6 * ts
            S[i, j] = s
    r, c = linear_sum_assignment(-S)
    lo = float(P["adversary.meta.match_min"])
    return [(int(i), int(j), float(S[i, j])) for i, j in zip(r, c) if S[i, j] >= lo]


# --------------------------------------------------------------------------- the typed IR differ
ALL_ASPECTS = ("geometry", "text", "color", "radius", "effect", "icon", "layout", "component")


def _f(type_: str, box: Box, mag: Optional[float], conf: float, aspect: str, e: Optional[_Item], a: Optional[_Item], **ev) -> Finding:
    ev.update(aspect=aspect, e_id=e.node.id if e else None, a_id=a.node.id if a else None)
    if e is not None:
        ev["e_box"] = e.node.box.to_dict()
    if a is not None:
        ev["a_box"] = a.node.box.to_dict()
    return Finding(type_, box, None if mag is None else float(mag), float(conf), ev, a.node.id if a else None)


def _pair_findings(e: _Item, a: _Item, aspects: tuple[str, ...]) -> list[Finding]:
    en, an = e.node, a.node
    k = kind(en)
    out: list[Finding] = []
    box = ext(en).union(ext(an))
    pos_tol = float(P["adversary.meta.pos_tol"])
    size_tol = float(P["adversary.meta.size_tol"])
    ctol = float(P["adversary.meta.color_tol"])
    dl, dt_, dr, db = an.box.x - en.box.x, an.box.y - en.box.y, an.box.x2 - en.box.x2, an.box.y2 - en.box.y2
    dw, dh = an.box.w - en.box.w, an.box.h - en.box.h
    dx, dy = (dl + dr) / 2, (dt_ + db) / 2
    edge = max(abs(dl), abs(dt_), abs(dr), abs(db))
    text_done = False
    if k == "text" and "text" in aspects and kind(an) == "text":
        if _norm(en.text) != _norm(an.text):
            out.append(_f("text.content", box.expand(1), _cer(an.text, en.text), 0.9, "text", e, a, expected=en.text, actual=an.text))
            text_done = True
        elif en.text_style is not None and an.text_style is not None and (
                abs(en.text_style.size - an.text_style.size) > float(P["adversary.meta.text_size_tol"])
                or abs(en.text_style.weight - an.text_style.weight) >= 100 or en.text_style.family != an.text_style.family):
            mag = abs(en.text_style.size - an.text_style.size) + abs(en.text_style.weight - an.text_style.weight) / 100.0
            asp = "text" if mag > 0 else "font_family"
            out.append(_f("text.style", box.expand(1), mag if mag > 0 else None, 0.8 if mag > 0 else 0.5, asp, e, a,
                          size=[en.text_style.size, an.text_style.size], weight=[en.text_style.weight, an.text_style.weight],
                          family=[en.text_style.family, an.text_style.family]))
            text_done = True
    if "geometry" in aspects and not text_done:
        slack = float(P["adversary.meta.text_slack"]) if k == "text" else 0.0
        if edge > pos_tol + slack:
            if abs(dw) <= size_tol + slack and abs(dh) <= size_tol + slack:
                t = "text.position" if k == "text" else "geometry.shift"
                out.append(_f(t, box.expand(1 if k == "text" else 0), math.hypot(dx, dy), 0.8, "geometry", e, a, dx=round(dx, 2), dy=round(dy, 2)))
            else:
                t = "text.style" if k == "text" else "geometry.size"
                out.append(_f(t, box, edge, 0.7, "geometry", e, a, dw=round(dw, 2), dh=round(dh, 2),
                              edges=[round(v, 2) for v in (dl, dt_, dr, db)]))
    if "color" in aspects:
        ce, ca = node_color(en), node_color(an)
        if ce is not None and ca is not None:
            de = ce.delta_e(ca)
            if de > ctol:
                out.append(_f("color.fill", box, de, 0.8, "color", e, a, expected=ce.hex(), actual=ca.hex(), **{"from": ce.hex(), "to": ca.hex()}))
        elif (ce is None) != (ca is None) and k == "shape" and kind(an) == "shape":
            out.append(_f("color.fill", box, None, 0.5, "fill_presence", e, a, expected=ce.hex() if ce else None, actual=ca.hex() if ca else None))
        se = en.strokes[0].color if en.strokes and en.strokes[0].width > 0 else None
        sa = an.strokes[0].color if an.strokes and an.strokes[0].width > 0 else None
        if se is not None and sa is not None:
            de = se.delta_e(sa)
            if de > ctol:
                out.append(_f("color.stroke", box, de, 0.8, "color", e, a, expected=se.hex(), actual=sa.hex()))
        elif (se is None) != (sa is None) and k == "shape":
            out.append(_f("color.stroke", box, None, 0.5, "stroke_presence", e, a, expected=se.hex() if se else None, actual=sa.hex() if sa else None))
    if "radius" in aspects and k == "shape" and kind(an) == "shape" and geom(en) not in ("line", "text") and min(en.box.w, en.box.h) >= 8:
        d = abs(_mean_radius(en) - _mean_radius(an))
        if d > float(P["adversary.meta.radius_tol"]):
            out.append(_f("geometry.radius", box, d, 0.7, "radius", e, a, radius=[_mean_radius(en), _mean_radius(an)]))
    if "effect" in aspects and abs(en.opacity - an.opacity) > 0.05:
        out.append(_f("effect.opacity", ext(en).union(ext(an)), abs(en.opacity - an.opacity), 0.8, "opacity", e, a, opacity=[en.opacity, an.opacity]))
    if "effect" in aspects and k == "shape":
        re_, ra = reach(en), reach(an)
        if (bool(en.effects) != bool(an.effects) and max(re_, ra) > float(P["adversary.meta.shadow_tol"])) or abs(re_ - ra) > float(P["adversary.meta.shadow_tol"]) + 2:
            out.append(_f("effect.shadow", box.expand(max(re_, ra, 4.0)), abs(re_ - ra), 0.6, "effect", e, a, reach=[re_, ra]))
    if "icon" in aspects and kind(en) == "icon" and kind(an) == "icon":
        if en.icon_name and an.icon_name and en.icon_name != an.icon_name:
            if same_glyph(en.icon_name, an.icon_name):
                out.append(_f("icon.glyph", box, None, 0.2, "icon_alias", e, a, expected=en.icon_name, actual=an.icon_name))
            else:
                out.append(_f("icon.glyph", box, None, 0.8, "icon", e, a, expected=en.icon_name, actual=an.icon_name))
        elif bool(en.icon_name) != bool(an.icon_name):
            out.append(_f("icon.glyph", box, None, 0.4, "icon_naming", e, a, expected=en.icon_name, actual=an.icon_name))
    if "layout" in aspects and en.layout is not None and an.layout is not None and en.children and an.children:
        if en.layout.mode != an.layout.mode and "none" not in (en.layout.mode, an.layout.mode):
            out.append(_f("layout.spacing", box, None, 0.5, "layout_mode", e, a, expected=en.layout.mode, actual=an.layout.mode))
        elif en.layout.mode == an.layout.mode != "none" and abs(en.layout.gap - an.layout.gap) > 2 * pos_tol + 1:
            out.append(_f("layout.spacing", box, abs(en.layout.gap - an.layout.gap), 0.5, "layout_gap", e, a, gap=[en.layout.gap, an.layout.gap]))
    if "component" in aspects and (en.component is not None or an.component is not None):
        ce_ = (en.component.name, tuple(sorted(en.component.variant.items()))) if en.component else None
        ca_ = (an.component.name, tuple(sorted(an.component.variant.items()))) if an.component else None
        if ce_ != ca_:
            out.append(_f("component.identity", box, None, 0.6, "component", e, a,
                          expected=None if ce_ is None else {"name": ce_[0], "variant": dict(ce_[1])},
                          actual=None if ca_ is None else {"name": ca_[0], "variant": dict(ca_[1])}))
    return out


_GLYPHS: dict = {}


def same_glyph(a: str, b: str) -> bool:
    """Two Material Symbols names that draw the same glyph (aliases such as mail/email, image/photo)."""
    if not _GLYPHS:
        try:
            from dt.perceive.icons import load_atlas
            at = load_atlas((0, 1))
            for k, (nm, fl) in enumerate(zip(at.names, at.fills)):
                _GLYPHS[(nm, int(fl))] = at.masks[k]
        except Exception:
            _GLYPHS[("", -1)] = None
    for fl in (0, 1):
        ma, mb = _GLYPHS.get((a, fl)), _GLYPHS.get((b, fl))
        if ma is not None and mb is not None and float(np.abs(ma - mb).mean()) < 0.01:
            return True
    return False


def _gap(a: Box, b: Box) -> float:
    gx = max(a.x, b.x) - min(a.x2, b.x2)
    gy = max(a.y, b.y) - min(a.y2, b.y2)
    return float(max(0.0, gx, gy))


def ir_diff(expected: Document, actual: Document, *, painted_only: bool = False, aspects: Iterable[str] = ALL_ASPECTS,
            include_e: Optional[Callable[[Node], bool]] = None, include_a: Optional[Callable[[Node], bool]] = None) -> list[Finding]:
    """Typed differences of ``actual`` w.r.t. ``expected`` (two IRs of the same frame).

    Boxes are in the documents' frame (union of the expected and actual extents). Evidence carries
    ``aspect`` and the node ids on both sides. Post-processing turns groups of raw differences into
    the taxonomy's composite types: split/merge (one node vs two), effect.opacity (colours faded
    towards the backdrop by one alpha), color.tint_global (most colours offset alike), layout.spacing
    (siblings shifted by k * d), one geometry.shift per moved subtree, one missing/extra per subtree.
    """
    aspects = tuple(aspects)
    E = _items(expected, include_e, painted_only)
    A = _items(actual, include_a, painted_only)
    pairs = _pair(E, A)
    pe = {i: j for i, j, _ in pairs}
    pa = {j: i for i, j, _ in pairs}
    raw: dict[tuple[int, int], list[Finding]] = {}
    for i, j, _s in pairs:
        raw[(i, j)] = _pair_findings(E[i], A[j], aspects)
    un_e = [i for i in range(len(E)) if i not in pe]
    un_a = [j for j in range(len(A)) if j not in pa]
    extra: list[Finding] = []
    cover = float(P["adversary.meta.split_cover"])

    # ---- split: one expected node <-> two actual nodes
    used_a: set[int] = set()
    for j2 in un_a:
        a2 = A[j2].node
        best = None
        for i, j, _s in pairs:
            e, a = E[i].node, A[j].node
            if kind(e) != kind(a2) or kind(a) != kind(a2):
                continue
            if a2.box.intersect(e.box).area < 0.6 * max(a2.box.area, 1e-6) or a.box.iou(e.box) > 0.92:
                continue
            u = a.box.union(a2.box)
            if u.iou(e.box) < cover:
                continue
            if kind(e) == "text":
                l, r = (a, a2) if a.box.x <= a2.box.x else (a2, a)
                if _cer(f"{l.text} {r.text}", e.text) > 0.2:
                    continue
            g = _gap(a.box, a2.box)
            if best is None or u.iou(e.box) > best[0]:
                best = (u.iou(e.box), i, j, g)
        if best is not None:
            _, i, j, g = best
            used_a.add(j2)
            raw[(i, j)] = [_f("structure.split", ext(E[i].node).union(ext(A[j].node)).union(ext(a2)).expand(1), g, 0.8, "structure", E[i], A[j],
                              parts=[A[j].node.id, a2.id])]
    un_a = [j for j in un_a if j not in used_a]

    # ---- merge: two expected nodes <-> one actual node
    used_e: set[int] = set()
    for i2 in un_e:
        e2 = E[i2].node
        best = None
        for i, j, _s in pairs:
            e, a = E[i].node, A[j].node
            if kind(e) != kind(e2) or kind(a) != kind(e2):
                continue
            if e2.box.intersect(a.box).area < 0.6 * max(e2.box.area, 1e-6) or e.box.iou(a.box) > 0.92:
                continue
            u = e.box.union(e2.box)
            if u.iou(a.box) < cover:
                continue
            if kind(e) == "text":
                l, r = (e, e2) if e.box.x <= e2.box.x else (e2, e)
                if _cer(a.text, f"{l.text} {r.text}") > 0.2:
                    continue
            g = _gap(e.box, e2.box)
            if best is None or u.iou(a.box) > best[0]:
                best = (u.iou(a.box), i, j, g)
        if best is not None:
            _, i, j, g = best
            used_e.add(i2)
            raw[(i, j)] = [_f("structure.merge", ext(E[i].node).union(ext(e2)).union(ext(A[j].node)).expand(1), g, 0.8, "structure", E[i], A[j],
                              merged_from=[E[i].node.id, e2.id])]
    un_e = [i for i in un_e if i not in used_e]

    # ---- opacity: colour changes that are one fade towards the backdrop
    rel = float(P["adversary.meta.opacity_resid"])
    fades: dict[tuple[int, int], float] = {}
    for (i, j), fs in raw.items():
        if not any(f.type == "color.fill" and f.evidence.get("aspect") == "color" for f in fs):
            continue
        ce, ca = node_color(E[i].node), node_color(A[j].node)
        bg = _rgb(_backdrop(expected, E[i]))
        v = _rgb(ce) - bg
        nv = float(np.dot(v, v))
        if nv < 20.0 ** 2:
            continue
        alpha = float(np.dot(_rgb(ca) - bg, v) / nv)
        resid = float(np.linalg.norm(_rgb(ca) - (bg + alpha * v))) / math.sqrt(nv)
        if 0.05 <= alpha <= 0.95 and resid <= rel:
            fades[(i, j)] = alpha
    if fades:
        idx_of = {E[i].node.id: (i, j) for (i, j) in fades}
        groups: dict[tuple, list[tuple[int, int]]] = {}
        for (i, j), al in fades.items():
            anc = next((idx_of[x] for x in E[i].anc if x in idx_of and abs(fades[idx_of[x]] - al) <= 0.2), None)
            if anc is not None:
                groups.setdefault(("anc",) + anc, []).append((i, j))
                continue
            groups.setdefault(("par", E[i].parent, round(al * 5)), []).append((i, j))
        merged: dict[tuple, list[tuple[int, int]]] = {}
        for key, members in groups.items():
            if key[0] == "anc":
                merged.setdefault(("root",) + key[1:], []).extend(members + [key[1:]])
            else:
                merged.setdefault(key, []).extend(members)
        for key, members in merged.items():
            members = list(dict.fromkeys(members))
            box = None
            for (i, j) in members:
                b = ext(E[i].node).union(ext(A[j].node))
                box = b if box is None else box.union(b)
                raw[(i, j)] = [f for f in raw[(i, j)] if f.type != "color.fill"]
            al = float(np.mean([fades[m] for m in members]))
            i0, j0 = members[0]
            extra.append(_f("effect.opacity", box, 1.0 - al, 0.7, "effect", E[i0], A[j0], alpha=round(al, 3), nodes=len(members)))

    # ---- global tint: most colours offset the same way
    comps = []
    for (i, j) in raw:
        for get in (node_color, lambda n: n.strokes[0].color if n.strokes else None):
            ce, ca = get(E[i].node), get(A[j].node)
            if ce is not None and ca is not None:
                comps.append((np.array(ce.lab()), np.array(ca.lab()), ce.delta_e(ca), kind(E[i].node)))
    if len(comps) >= int(P["adversary.meta.tint_min_pairs"]):
        ctol = float(P["adversary.meta.color_tol"])
        changed = [c for c in comps if c[2] > ctol / 2]
        shapes = [c for c in comps if c[3] == "shape"]
        shapes_changed = [c for c in shapes if c[2] > ctol / 2]
        # a global tint moves surfaces too; glyph-only colour drift (text antialiasing) is not a tint
        if len(changed) >= float(P["adversary.meta.tint_frac"]) * len(comps) and shapes_changed and len(shapes_changed) >= 0.5 * len(shapes):
            offs = np.array([c[1] - c[0] for c in changed])
            m = offs.mean(axis=0)
            spread = float(np.mean(np.linalg.norm(offs - m, axis=1))) / max(float(np.linalg.norm(m)), 1e-6)
            if spread <= float(P["adversary.meta.tint_spread"]) and np.linalg.norm(m) > ctol / 2:
                for key in raw:
                    raw[key] = [f for f in raw[key] if not f.type.startswith("color.")]
                extra = [f for f in extra if f.type != "effect.opacity"]
                extra.append(Finding("color.tint_global", Box(0, 0, expected.width, expected.height), float(np.mean([c[2] for c in changed])), 0.8,
                                     {"aspect": "color", "offset_lab": [round(float(x), 2) for x in m], "pairs": len(comps), "changed": len(changed)}))

    # ---- moved subtrees and spacing
    findings: list[Finding] = [f for fs in raw.values() for f in fs]
    moves = [f for f in findings if f.type in ("geometry.shift", "text.position")]
    eidx = {E[i].node.id: i for i in range(len(E))}
    move_of = {f.evidence["e_id"]: f for f in moves}
    drop: set[int] = set()
    tol = float(P["adversary.meta.pos_tol"]) + 1.0
    for f in moves:  # a child that moved with its moved ancestor is not a separate error
        anc = E[eidx[f.evidence["e_id"]]].anc
        for x in anc:
            g = move_of.get(x)
            if g is not None and abs(g.evidence["dx"] - f.evidence["dx"]) <= tol and abs(g.evidence["dy"] - f.evidence["dy"]) <= tol:
                drop.add(id(f))
                break
    moves = [f for f in moves if id(f) not in drop]
    anc_of = {id(f): E[eidx[f.evidence["e_id"]]].anc for f in moves}
    # spacing: under the smallest common container, moved nodes whose offsets along one axis are k * d
    containers = sorted({(k, a) for f in moves for k, a in enumerate(anc_of[id(f)])}, key=lambda t: -t[0])
    for _depth, cid in containers:
        fs = [f for f in moves if id(f) not in drop and cid in anc_of[id(f)]]
        for axis in ("dx", "dy"):
            other = "dy" if axis == "dx" else "dx"
            al = [f for f in fs if abs(f.evidence[other]) <= tol and abs(f.evidence[axis]) > tol - 1 and id(f) not in drop]
            if len(al) < 2:
                continue
            if cid == expected.root.id and any(anc_of[id(f)][-1] != cid for f in al):
                continue  # unrelated moves anywhere on the page are not one spacing change
            pos = "x" if axis == "dx" else "y"
            seq = [abs(f.evidence[axis]) for f in sorted(al, key=lambda f: f.evidence["e_box"][pos])]
            if any(b < a - tol for a, b in zip(seq, seq[1:])):
                continue  # spacing drift accumulates along the axis
            offs = sorted({round(abs(f.evidence[axis])) for f in al})
            d0 = offs[0]
            if len(offs) >= 2 and d0 > 0 and all(abs(o / d0 - round(o / d0)) <= 0.25 for o in offs):
                box = None
                for f in al:
                    box = f.box if box is None else box.union(f.box)
                    drop.add(id(f))
                extra.append(Finding("layout.spacing", box, float(d0), 0.7, {"aspect": "spacing", "axis": "x" if axis == "dx" else "y",
                                                                              "offsets": offs, "parent": cid, "moved": [f.evidence["e_id"] for f in al]}))
    by_parent: dict[Optional[str], list[Finding]] = {}
    for f in moves:
        by_parent.setdefault(E[eidx[f.evidence["e_id"]]].parent, []).append(f)
    for par, fs in by_parent.items():
        rest = [f for f in fs if id(f) not in drop]
        same: dict[tuple, list[Finding]] = {}
        for f in rest:
            same.setdefault((f.type, round(f.evidence["dx"]), round(f.evidence["dy"])), []).append(f)
        for key, grp in same.items():
            if len(grp) < 2:
                continue
            grp.sort(key=lambda f: -f.box.area)
            keep = grp[0]
            for f in grp[1:]:
                if _gap(keep.box, f.box) <= 8:
                    keep.box = keep.box.union(f.box)
                    keep.evidence["nodes"] = keep.evidence.get("nodes", 1) + 1
                    drop.add(id(f))
    findings = [f for f in findings if id(f) not in drop]

    # ---- missing / extra: one finding per absent subtree
    un_e_ids = {E[i].node.id for i in un_e}
    for i in un_e:
        if any(x in un_e_ids for x in E[i].anc):
            continue
        n = E[i].node
        t = "icon.missing" if kind(n) == "icon" else "structure.missing"
        extra.append(_f(t, ext(n), n.box.area, 0.8 if paints(n) else 0.4, "structure", E[i], None, missing_type=geom(n), painted=paints(n)))
    un_a_ids = {A[j].node.id for j in un_a}
    for j in un_a:
        if any(x in un_a_ids for x in A[j].anc):
            continue
        n = A[j].node
        extra.append(_f("structure.extra", ext(n), n.box.area, 0.8 if paints(n) else 0.4, "structure", None, A[j], extra_type=geom(n), painted=paints(n)))
    return findings + extra


# --------------------------------------------------------------------------- differential adversary (benchmark)
def _evidence_px(bad: np.ndarray, box: Box) -> int:
    H, W = bad.shape
    x0, y0, x1, y1 = box.expand(1).as_int()
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(bad[y0:y1, x0:x1].sum())


def evidence_mask(target_rgb: np.ndarray, candidate_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ΔE76 map, boolean evidence mask) for the differential gate: optional blur of both images, ΔE
    threshold, optional morphological opening (1 px seams from anti-aliasing / sub-pixel offsets / JPEG
    ringing do not survive a 2x2 opening; a real edit's area does)."""
    import cv2
    from dt.compare.pixel import diff_map
    s = float(P["adversary.meta.gate_blur"])
    t, c = target_rgb[..., :3], candidate_rgb[..., :3]
    if s > 0:
        t, c = cv2.GaussianBlur(t, (0, 0), s), cv2.GaussianBlur(c, (0, 0), s)
    d = diff_map(t, c)
    bad = d > float(P["adversary.meta.gate_de"])
    k = int(P["adversary.meta.gate_open"])
    if k > 1:
        bad = cv2.morphologyEx(bad.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool)
    return d, bad


def _covers(f: Finding, b: Box) -> bool:
    return f.box.iou(b) >= 0.1 or f.box.contains_point(b.cx, b.cy) or b.contains_point(f.box.cx, f.box.cy)


def find(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None, *,
         stage: str = "perceive") -> list[Finding]:
    """Differential metamorphic adversary: translate target and candidate with the same translator,
    diff the two IRs (:func:`ir_diff`, visible nodes only), keep a finding where the pixels differ.

    The identity relation "identical pixels translate identically" makes translator errors common to
    both sides cancel; what is left is typed by the IR. Optional residual fallback
    (``adversary.meta.fallback``) reports pixel residuals no IR finding explains, at low confidence."""
    from dt.compare.pixel import residual_regions
    e_doc = translate_ir(target_rgb, stage)
    a_doc = translate_ir(candidate_rgb, stage)
    with overrides(color_tol=float(P["adversary.meta.find_color_tol"])):
        raw = ir_diff(e_doc, a_doc, painted_only=True, aspects=("geometry", "text", "color", "radius", "effect", "icon"))
    d, bad = evidence_mask(target_rgb, candidate_rgb)
    need = int(P["adversary.meta.gate_px"])
    page = Box(0, 0, target_rgb.shape[1], target_rgb.shape[0])
    out: list[Finding] = []
    for f in raw:
        f.box = f.box.intersect(page)
        if f.box.area <= 0:
            continue
        n = _evidence_px(bad, f.box)
        if f.type == "color.tint_global":
            n = int(bad.sum())
        if n < need:
            continue
        f.confidence = float(min(1.0, f.confidence * (0.5 + 0.5 * min(1.0, n / 200.0))))
        f.evidence["evidence_px"] = n
        out.append(f)
    if bool(P["adversary.meta.fallback"]):
        for b in residual_regions(d):
            if any(_covers(f, b) for f in out):
                continue
            if _evidence_px(bad, b) < need:
                continue
            x0, y0, x1, y1 = b.as_int()
            m = float(d[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
            out.append(Finding("structure.missing", b, None, float(P["adversary.meta.fallback_conf"]) * min(1.0, m / 30.0),
                               {"aspect": "residual", "mean_de": m}))
    if candidate_ir is not None:
        nodes = [n for n in candidate_ir.walk() if n.id != candidate_ir.root.id and paints(n)]
        for f in out:
            if f.evidence.get("a_box"):
                ab = Box.from_dict(f.evidence["a_box"])
                best = max(nodes, key=lambda n: n.box.iou(ab), default=None)
                f.node_id = best.id if best is not None and best.box.iou(ab) > 0.3 else None
            else:
                f.node_id = None
    return out


# --------------------------------------------------------------------------- relations
@dataclass
class Instance:
    """One follow-up input of a relation with its expected translation (``expected`` = t*(T(x)))."""
    relation: str
    params: dict
    image: np.ndarray
    expected: Document
    to_source: Callable[[Box], Box]
    dpr: "float | str" = 1.0
    exact: bool = True
    include_e: Optional[Callable[[Node], bool]] = None
    include_a: Optional[Callable[[Node], bool]] = None
    aspects: tuple[str, ...] = ALL_ASPECTS
    extra_check: Optional[Callable[["Instance", Document], list[Finding]]] = None
    note: str = ""


@dataclass
class Relation:
    name: str
    description: str
    expected_types: tuple[str, ...]   # taxonomy types its violations typically map onto
    make: Callable[..., list[Instance]]

    @property
    def family(self) -> str:
        return f"meta.{self.name}"


def _border_color(rgb: np.ndarray) -> tuple[int, int, int]:
    b = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).reshape(-1, 3)
    vals, counts = np.unique(b, axis=0, return_counts=True)
    c = vals[int(np.argmax(counts))]
    return int(c[0]), int(c[1]), int(c[2])


def _map_boxes(doc: Document, fn: Callable[[Node], None]) -> Document:
    out = copy.deepcopy(doc)
    for n in out.root.walk():
        if n is not out.root:
            fn(n)
    return out


def _map_colors(doc: Document, f: Callable[[Color], Color]) -> Document:
    out = copy.deepcopy(doc)
    for n in out.walk():
        for fl in n.fills:
            if fl.color is not None:
                c = f(fl.color)
                fl.color = Color(c.r, c.g, c.b, fl.color.a)
            for s in fl.stops:
                c = f(s.color)
                s.color = Color(c.r, c.g, c.b, s.color.a)
        for s in n.strokes:
            c = f(s.color)
            s.color = Color(c.r, c.g, c.b, s.color.a)
        if n.text_style is not None:
            c = f(n.text_style.color)
            n.text_style.color = Color(c.r, c.g, c.b, n.text_style.color.a)
        for e in n.effects:
            pass  # shadows are black with alpha; recolouring keeps them shadows
    return out


def rel_translate(rgb: np.ndarray, base: Document, gt: Optional[Document] = None) -> list[Instance]:
    dx, dy = (int(v) for v in P["adversary.meta.translate"])
    H, W = rgb.shape[:2]
    img = np.empty((H + dy, W + dx, 3), np.uint8)
    img[:] = np.array(_border_color(rgb), np.uint8)
    img[dy:, dx:] = rgb[..., :3]
    exp = _map_boxes(base, lambda n: setattr(n, "box", n.box.translate(dx, dy)))
    exp.width, exp.height = W + dx, H + dy
    exp.root.box = Box(0, 0, W + dx, H + dy)
    return [Instance("translate", {"dx": dx, "dy": dy}, img, exp, lambda b: b.translate(-dx, -dy))]


def _lab_lut(rgb: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    flat = rgb[..., :3].reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    return fn(uniq)[inv.reshape(-1)].reshape(rgb.shape[0], rgb.shape[1], 3)


def tint_fn(de: float) -> Callable[[np.ndarray], np.ndarray]:
    """Uniform Lab offset of magnitude ``de`` (fixed direction) on an (N, 3) uint8 array."""
    from skimage.color import lab2rgb, rgb2lab
    v = np.array([0.35, 0.55, -0.76])
    v = v / np.linalg.norm(v) * de

    def f(u: np.ndarray) -> np.ndarray:
        lab = rgb2lab(u.reshape(1, -1, 3).astype(np.float64) / 255.0) + v
        out = np.clip(lab2rgb(lab), 0, 1) * 255.0
        return np.round(out).astype(np.uint8).reshape(-1, 3)
    return f


def _color_via(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[Color], Color]:
    def g(c: Color) -> Color:
        r, gg, b = fn(np.array([[c.r, c.g, c.b]], np.uint8))[0]
        return Color(int(r), int(gg), int(b), c.a)
    return g


def _token_check(inst: Instance, actual: Document) -> list[Finding]:
    """Uniform recolouring must map tokens consistently: nodes that shared a colour token (or a colour)
    before must still share one after."""
    E = _items(inst.expected, None, True)
    A = _items(actual, None, True)
    pairs = _pair(E, A)
    groups: dict[str, list[tuple[_Item, _Item]]] = {}
    for i, j, _ in pairs:
        e, a = E[i].node, A[j].node
        te = e.tokens.get("fill") or e.tokens.get("color")
        if te:
            groups.setdefault(te, []).append((E[i], A[j]))
    out = []
    for te, members in groups.items():
        if len(members) < 2:
            continue
        got: dict[str, list] = {}
        for e, a in members:
            ta = a.node.tokens.get("fill") or a.node.tokens.get("color") or "(none)"
            got.setdefault(ta, []).append((e, a))
        if len(got) <= 1:
            continue
        major = max(got, key=lambda k: len(got[k]))
        for ta, mem in got.items():
            if ta == major:
                continue
            for e, a in mem:
                out.append(_f("color.fill", ext(e.node).union(ext(a.node)), None, 0.6, "token_consistency", e, a,
                              expected_token=te, majority=major, actual_token=ta))
    return out


def rel_recolor(rgb: np.ndarray, base: Document, gt: Optional[Document] = None) -> list[Instance]:
    out = []
    fn = tint_fn(float(P["adversary.meta.tint_de"]))
    img = _lab_lut(rgb, fn)
    out.append(Instance("recolor_tint", {"de": float(P["adversary.meta.tint_de"])}, img, _map_colors(base, _color_via(fn)), lambda b: b,
                        extra_check=_token_check))
    swap = rgb[..., :3][..., ::-1].copy()
    out.append(Instance("recolor_swap", {"perm": "bgr"}, swap, _map_colors(base, lambda c: Color(c.b, c.g, c.r, c.a)), lambda b: b,
                        extra_check=_token_check))
    return out


def render_2x(doc: Document) -> np.ndarray:
    """A native device-pixel-ratio-2 capture of an IR document (same HTML and fonts as ``dt.render``)."""
    from dt.render.html import render_html
    from dt.render.screenshot import _tmp_html, capture_url
    path = _tmp_html(render_html(doc))
    return capture_url("file://" + path, doc.width, doc.height, device_scale_factor=2.0, wait_ms=100)


def _dpr_check(inst: Instance, actual: Document) -> list[Finding]:
    d = float(actual.meta.get("metamorphic", {}).get("dpr", 1.0))
    if abs(d - 2.0) < 1e-6:
        return []
    return [Finding("geometry.size", Box(0, 0, inst.expected.width, inst.expected.height), abs(2.0 / d - 1.0) * 100.0, 1.0,
                    {"aspect": "dpr", "expected_dpr": 2.0, "detected_dpr": d, "note": "every box is scaled by 2/d"})]


def rel_dpr2(rgb: np.ndarray, base: Document, gt: Optional[Document] = None) -> list[Instance]:
    if gt is not None:
        img = render_2x(gt)
        note, exact = "native 2x capture of the ground-truth IR", True
    else:
        from PIL import Image
        H, W = rgb.shape[:2]
        img = np.asarray(Image.fromarray(rgb[..., :3]).resize((2 * W, 2 * H), Image.LANCZOS), dtype=np.uint8).copy()
        note, exact = "Lanczos 2x upsample (approximates a 2x capture)", False
    return [Instance("dpr2", {"scale": 2, "source": "native" if gt is not None else "upsample"}, img, copy.deepcopy(base), lambda b: b,
                     dpr="auto", exact=exact, extra_check=_dpr_check, note=note)]


def _row_runs(doc: Document) -> list[tuple[Node, list[Node]]]:
    """Repeated sibling sequences stacked vertically: same kind, same left edge and width, heights within
    ``row_ratio`` (1- and 2-line items mix), consecutive gaps no taller than a row (dividers may sit in
    between)."""
    tol = float(P["adversary.meta.row_tol"])
    ratio = float(P["adversary.meta.row_ratio"])
    kmin = int(P["adversary.meta.min_rows"])
    runs, seen = [], set()
    for par in doc.walk():
        kids = [c for c in par.children if c.visible and c.box.h >= 8]
        for seed in kids:
            if seed.id in seen:
                continue
            same = sorted([c for c in kids if kind(c) == kind(seed) and abs(c.box.x - seed.box.x) <= tol
                           and abs(c.box.w - seed.box.w) <= max(tol, 0.1 * seed.box.w)
                           and max(c.box.h, seed.box.h) / max(1e-6, min(c.box.h, seed.box.h)) <= ratio], key=lambda c: c.box.y)
            chain: list[Node] = []
            for c in same:
                if chain and (c.box.y < chain[-1].box.y2 - 0.5 or c.box.y - chain[-1].box.y2 > max(chain[-1].box.h, c.box.h)):
                    if len(chain) >= kmin:
                        break
                    chain = []
                chain.append(c)
            if len(chain) >= kmin:
                runs.append((par, chain))
                seen.update(c.id for c in chain)
    runs.sort(key=lambda t: (-len(t[1]), -t[1][0].box.area))
    return runs


def _band_ok(doc: Document, c0: float, c1: float) -> bool:
    for n in doc.root.walk():
        if n is doc.root or not n.visible:
            continue
        b = ext(n)
        if b.y2 <= c0 + 0.5 or b.y >= c1 - 0.5:
            continue
        if b.y >= c0 - 0.5 and b.y2 <= c1 + 0.5:
            continue
        if n.box.y < c0 and n.box.y2 > c1:
            continue  # spans the band: grows/shrinks with it
        return False
    return True


def _row_band(base: Document) -> Optional[tuple[int, int, list[Node]]]:
    for par, run in _row_runs(base):
        for k in sorted(range(1, len(run) - 1), key=lambda k: abs(k - len(run) / 2)):
            c0 = int(round((run[k - 1].box.y2 + run[k].box.y) / 2))
            c1 = int(round((run[k].box.y2 + run[k + 1].box.y) / 2))
            if c1 - c0 >= 8 and _band_ok(base, c0, c1):
                return c0, c1, run
    return None


def _in_band(b: Box, c0: float, c1: float) -> bool:
    return b.y >= c0 - 0.5 and b.y2 <= c1 + 0.5


def rel_rows(rgb: np.ndarray, base: Document, gt: Optional[Document] = None) -> list[Instance]:
    band = _row_band(base)
    if band is None:
        return []
    c0, c1, run = band
    p = c1 - c0
    H, W = rgb.shape[:2]
    out = []
    # add: duplicate the band right after itself
    img = np.concatenate([rgb[:c1], rgb[c0:c1], rgb[c1:]], axis=0)[..., :3].copy()
    exp = copy.deepcopy(base)
    exp.height = H + p
    exp.root.box = Box(0, 0, W, H + p)

    def add(par: Node) -> None:
        new_children = []
        for c in par.children:
            b = c.box
            if b.y >= c1 - 0.5:
                for m in c.walk():
                    m.box = m.box.translate(0, p)
                new_children.append(c)
            elif _in_band(ext(c), c0, c1):
                dup = copy.deepcopy(c)
                for m in dup.walk():
                    m.box = m.box.translate(0, p)
                    m.id = m.id + "_dup"
                new_children += [c, dup]
            else:
                if b.y < c0 and b.y2 > c1:
                    c.box = Box(b.x, b.y, b.w, b.h + p)
                    add(c)
                new_children.append(c)
        par.children = new_children

    add(exp.root)

    def back_add(b: Box) -> Box:
        y0 = b.y - p if b.y >= c1 - 0.5 else b.y
        y1 = b.y2 - p if b.y2 > c1 + 0.5 else b.y2
        return Box(b.x, y0, b.w, max(1.0, y1 - y0))
    params = {"band": [c0, c1], "pitch": p, "rows": len(run)}
    out.append(Instance("row_add", params, img, exp, back_add, note=f"band y {c0}..{c1}"))
    # remove: delete the band
    img2 = np.concatenate([rgb[:c0], rgb[c1:]], axis=0)[..., :3].copy()
    exp2 = copy.deepcopy(base)
    exp2.height = H - p
    exp2.root.box = Box(0, 0, W, H - p)

    def rem(par: Node) -> None:
        kept = []
        for c in par.children:
            b = c.box
            if _in_band(ext(c), c0, c1):
                continue
            if b.y >= c1 - 0.5:
                for m in c.walk():
                    m.box = m.box.translate(0, -p)
            elif b.y < c0 and b.y2 > c1:
                c.box = Box(b.x, b.y, b.w, b.h - p)
                rem(c)
            kept.append(c)
        par.children = kept

    rem(exp2.root)

    def back_rem(b: Box) -> Box:
        y0 = b.y + p if b.y >= c0 - 0.5 else b.y
        y1 = b.y2 + p if b.y2 > c0 + 0.5 else b.y2
        return Box(b.x, y0, b.w, max(1.0, y1 - y0))
    out.append(Instance("row_remove", params, img2, exp2, back_rem, note=f"band y {c0}..{c1}"))
    return out


def _crop_line(base: Document) -> Optional[int]:
    H = base.height
    leaves = [ext(n) for n in base.root.walk() if n is not base.root and n.visible and (not n.children or paints(n) and n.type != "frame")]
    target = float(P["adversary.meta.crop_frac"]) * H
    best = None
    for y in range(8, H - 8):
        if any(b.y - 2 < y < b.y2 + 2 for b in leaves):
            continue
        if best is None or abs(y - target) < abs(best - target):
            best = y
    return best


def rel_crop(rgb: np.ndarray, base: Document, gt: Optional[Document] = None) -> list[Instance]:
    c = _crop_line(base)
    if c is None:
        return []
    H, W = rgb.shape[:2]
    img = rgb[c:, :, :3].copy()
    exp = _map_boxes(base, lambda n: setattr(n, "box", n.box.translate(0, -c)))
    exp.height = H - c
    exp.root.box = Box(0, 0, W, H - c)
    inside = lambda n: n.box.y >= -0.5  # noqa: E731  (expected: fully inside the crop)
    not_cut = lambda n: n.box.y > 1.5   # noqa: E731  (actual: ignore containers clipped at the cut)
    return [Instance("crop", {"y": c}, img, exp, lambda b: b.translate(0, c), include_e=inside, include_a=not_cut,
                     aspects=tuple(a for a in ALL_ASPECTS if a != "layout"), note=f"rows {c}..{H}")]


def mirror_doc(doc: Document) -> Document:
    """RTL mirror of an IR: boxes mirrored about the vertical axis, radii/align/shadow dx mirrored,
    glyphs (text, icons) unchanged."""
    W = doc.width
    out = copy.deepcopy(doc)
    for n in out.root.walk():
        if n is not out.root:
            n.box = Box(W - n.box.x2, n.box.y, n.box.w, n.box.h)
        tl, tr, br, bl = n.radius
        n.radius = (tr, tl, bl, br)
        for e in n.effects:
            e.dx = -e.dx
        if n.text_style is not None and n.text_style.align in ("left", "right"):
            n.text_style.align = "right" if n.text_style.align == "left" else "left"
        if n.layout is not None and n.layout.mode == "row":
            j = n.layout.justify
            n.layout.justify = {"start": "end", "end": "start"}.get(j, j)
            n.children = list(reversed(n.children))
    return out


def rel_mirror(rgb: np.ndarray, base: Document, gt: Optional[Document] = None) -> list[Instance]:
    H, W = rgb.shape[:2]
    # Pixel mirror for every input (also IR inputs: a rendered mirror of the ground truth re-anchors text
    # inside its asymmetric line box, so it is not exact w.r.t. T(x)). Glyphs stay readable: the
    # unflipped glyph pixels are pasted at the mirrored box, so a text/icon box moves with its ink.
    img = rgb[:, ::-1, :3].copy()
    for n in base.walk():
        if n.type in ("text", "icon") and n.visible:
            x0, y0, x1, y1 = n.box.expand(2).as_int()
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 > x0 and y1 > y0:
                img[y0:y1, W - x1:W - x0] = rgb[y0:y1, x0:x1, :3]
    note = "pixel mirror with glyph patches kept unflipped"
    exp = mirror_doc(base)
    for n in exp.walk():
        if n.layout is not None:
            n.layout = copy.deepcopy(n.layout)
    return [Instance("mirror", {"axis": "x", "source": "pixels"}, img, exp,
                     lambda b: Box(W - b.x2, b.y, b.w, b.h), exact=True,
                     aspects=tuple(a for a in ALL_ASPECTS if a != "layout"), note=note)]


def rel_jpeg(rgb: np.ndarray, base: Document, gt: Optional[Document] = None) -> list[Instance]:
    from dt.adversary.perturb import jpeg
    q = int(P["adversary.meta.jpeg_quality"])
    return [Instance("jpeg", {"quality": q}, jpeg(rgb[..., :3], q), copy.deepcopy(base), lambda b: b, exact=False,
                     note="nuisance: the translation must not change")]


RELATIONS: dict[str, Relation] = {r.name: r for r in [
    Relation("translate", "pad top/left by (dx, dy): every node translates by (dx, dy)",
             ("geometry.shift", "geometry.size", "structure.missing", "structure.extra"), rel_translate),
    Relation("recolor", "uniform recolouring (Lab tint, channel swap): colours map consistently, geometry fixed",
             ("color.fill", "color.stroke", "color.tint_global", "structure.missing", "structure.extra"), rel_recolor),
    Relation("dpr2", "2x capture translated with dpr=auto equals the 1x translation",
             ("geometry.size", "geometry.shift", "text.style", "structure.missing", "structure.extra"), rel_dpr2),
    Relation("rows", "add/remove one row of a repeated list: exactly one item frame more/less, the rest shifted",
             ("structure.missing", "structure.extra", "structure.merge", "structure.split", "layout.spacing"), rel_rows),
    Relation("crop", "crop below a gap line: nodes inside are kept and translated, none invented",
             ("structure.missing", "structure.extra", "geometry.size", "color.fill"), rel_crop),
    Relation("mirror", "RTL mirror: boxes mirror, layout order reverses, text and icons unchanged",
             ("geometry.shift", "geometry.size", "text.content", "icon.glyph", "structure.missing"), rel_mirror),
    Relation("jpeg", "JPEG re-encoding (nuisance) leaves the translation unchanged",
             ("color.fill", "geometry.size", "text.content", "structure.missing", "structure.extra"), rel_jpeg),
]}
DEFAULT_RELATIONS = tuple(RELATIONS)
# instance name -> relation (scenario family ``meta.<instance>``)
INSTANCE_RELATION = {"translate": "translate", "recolor_tint": "recolor", "recolor_swap": "recolor", "dpr2": "dpr2", "row_add": "rows",
                     "row_remove": "rows", "crop": "crop", "mirror": "mirror", "jpeg": "jpeg"}
RELATIONS_ORDER = tuple(INSTANCE_RELATION)


def instances(rgb: np.ndarray, base: Document, relations: Iterable[str] = DEFAULT_RELATIONS, gt: Optional[Document] = None) -> list[Instance]:
    out = []
    for r in relations:
        out.extend(RELATIONS[r].make(rgb, base, gt))
    return out


def check(inst: Instance, actual: Document, src_page: Box) -> list[Finding]:
    """Violations of one relation instance, boxes mapped back to the source frame."""
    if inst.extra_check is _dpr_check:
        g = _dpr_check(inst, actual)
        if g:  # a wrong DPR scales every box: one global violation, not hundreds
            fs = g
        else:
            fs = ir_diff(inst.expected, actual, include_e=inst.include_e, include_a=inst.include_a, aspects=inst.aspects)
    else:
        fs = ir_diff(inst.expected, actual, include_e=inst.include_e, include_a=inst.include_a, aspects=inst.aspects)
        if inst.extra_check is not None:
            fs += inst.extra_check(inst, actual)
    for f in fs:
        f.evidence["follow_box"] = f.box.to_dict()
        b = inst.to_source(f.box).intersect(src_page)
        f.box = b if b.area > 0 else inst.to_source(f.box)
        f.evidence.update(relation=inst.relation, params=inst.params, exact=inst.exact)
        if not inst.exact:
            f.confidence *= 0.7
    return fs


# --------------------------------------------------------------------------- validator emulation (blind spots)
class RegionJudge:
    """What the independent pairwise validator (``dt.validate``) would say about one region of a
    (target, render) pair: non-text JND fraction (visually-identical gate), flat-region ΔE (worst
    region gate) and OCR text lines (CER / Δpos gates)."""

    def __init__(self, target: np.ndarray, render: np.ndarray, doc: Optional[Document] = None, ocr: bool = True) -> None:
        from dt.validate import fidelity as V
        render, _ = V.conform_to_target(target, render)
        self.de = V.delta_e2000_map(target[..., :3], render)
        self.tmask = V.text_mask_from_doc(doc) if doc is not None and (doc.height, doc.width) == self.de.shape else np.zeros(self.de.shape, bool)
        self.matches = []
        if ocr:
            ok, ms = V.text_fidelity(target[..., :3], render)
            self.matches = ms if ok else []
            if ok:
                self.tmask |= V.text_mask_from_lines(self.de.shape, ms)
        self.regions = V.region_scores(target[..., :3], render, V.flat_regions(target[..., :3]))

    def flags(self, box: Box) -> dict:
        H, W = self.de.shape
        x0, y0, x1, y1 = box.as_int()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        out = {"jnd_nontext": 0.0, "region_de": 0.0, "text_bad": False}
        if x1 <= x0 or y1 <= y0:
            out["flagged"] = False
            return out
        de = self.de[y0:y1, x0:x1]
        nt = ~self.tmask[y0:y1, x0:x1]
        if nt.sum() >= 4:
            out["jnd_nontext"] = float((de[nt] > 2.3).mean())
        for r in self.regions:
            rb = Box.from_dict(r.box)
            if rb.intersect(box).area > 0.5 * min(rb.area, box.area):
                out["region_de"] = max(out["region_de"], float(r.delta_e))
        for m in self.matches:
            tb = Box.from_dict(m.target_box)
            if tb.intersect(box).area > 0.3 * tb.area:
                if not m.matched or m.cer > 0 or math.hypot(m.dx, m.dy) > float(P["adversary.meta.blind_dpos"]):
                    out["text_bad"] = True
        out["flagged"] = bool(out["jnd_nontext"] > float(P["adversary.meta.blind_jnd"]) or out["region_de"] >= float(P["adversary.meta.blind_region_de"])
                              or out["text_bad"])
        return out


# --------------------------------------------------------------------------- running relations on an input
@dataclass
class PageResult:
    name: str
    size: tuple[int, int]
    stage: str
    base_nodes: int
    runs: list[dict] = field(default_factory=list)          # one per instance
    violations: list[Finding] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "size": list(self.size), "stage": self.stage, "base_nodes": self.base_nodes, "runs": self.runs,
                "violations": [f.to_dict() for f in self.violations], "seconds": self.seconds}


def run_page(name: str, rgb: np.ndarray, *, stage: str = "perceive", relations: Iterable[str] = DEFAULT_RELATIONS,
             gt: Optional[Document] = None, workers: int = 1, judge: bool = True, base_render: Optional[np.ndarray] = None,
             log: Optional[Callable[[str], None]] = None, save_dir: Optional[str] = None) -> PageResult:
    """Run every relation on one input. With ``judge`` each violation is also checked against the
    pairwise validator's view of the base and the follow-up translations (``evidence.validator``)."""
    from dt.render.screenshot import render_doc
    t0 = time.time()
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    base = translate_ir(rgb, stage)
    insts = instances(rgb, base, relations, gt)
    if workers > 1:
        precompute([(i.image, stage, i.dpr) for i in insts], workers)
    page = Box(0, 0, rgb.shape[1], rgb.shape[0])
    res = PageResult(name, (rgb.shape[1], rgb.shape[0]), stage, base.root.count())
    base_judge = None
    if judge:
        base_judge = RegionJudge(rgb, base_render if base_render is not None else render_doc(base), base)
    for inst in insts:
        t1 = time.time()
        act = translate_ir(inst.image, stage, inst.dpr)
        fs = check(inst, act, page)
        if judge and fs:
            fj = None
            if not any(f.evidence.get("aspect") == "dpr" for f in fs):
                act_r = render_doc(act)
                if act_r.shape[:2] == inst.image.shape[:2]:
                    fj = RegionJudge(inst.image, act_r, act)
            for f in fs:
                b = base_judge.flags(f.box) if base_judge else {"flagged": None}
                fo = fj.flags(Box.from_dict(f.evidence["follow_box"])) if fj is not None else {"flagged": None}
                f.evidence["validator"] = {"base": b, "follow": fo,
                                           "blind": not b.get("flagged") and not fo.get("flagged") if fo.get("flagged") is not None else not b.get("flagged")}
        if save_dir:
            from dt.common.image import save_rgb
            os.makedirs(save_dir, exist_ok=True)
            save_rgb(inst.image, os.path.join(save_dir, f"{name}.{inst.relation}.png"))
        res.runs.append({"relation": inst.relation, "params": inst.params, "exact": inst.exact, "note": inst.note,
                         "expected_nodes": inst.expected.root.count(), "actual_nodes": act.root.count(),
                         "violations": len(fs), "by_type": _count([f.type for f in fs]), "seconds": round(time.time() - t1, 2),
                         "detected_dpr": act.meta.get("metamorphic", {}).get("dpr")})
        res.violations.extend(fs)
        if log:
            log(f"  {name} {inst.relation}: {len(fs)} violations")
    res.seconds = round(time.time() - t0, 1)
    return res


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- scenario families (a violation becomes a generator)
def scenario_spec(source: dict, inst_relation: str, params: dict, violations: list[Finding], stage: str = "perceive") -> dict:
    """A reproducible scenario: the relation, its parameters and the source; :func:`regenerate` rebuilds
    the follow-up input and its expected translation, so the failure is a standing test case."""
    return {"family": f"meta.{inst_relation}", "relation": INSTANCE_RELATION.get(inst_relation, inst_relation), "params": params, "source": source,
            "oracle": "T(t(x)) == t*(T(x))", "violations": _count([f.type for f in violations]), "n": len(violations),
            "stage": stage, "translator": translator_version(stage), "meta_version": META_VERSION}


def regenerate(spec: dict, stage: str = "perceive") -> Instance:
    """Rebuild the follow-up input + expected IR of a scenario spec (source ``{"image": path}`` or
    ``{"doc": "<benchmark doc spec>"}``)."""
    from dt.common.image import load_rgb
    gt = None
    src = spec["source"]
    if "doc" in src:
        from dt.adversary.benchmark import load_doc
        from dt.render.screenshot import render_doc
        gt = load_doc(src["doc"])
        rgb = render_doc(gt)
    else:
        rgb = load_rgb(os.path.join(ROOT, src["image"]) if not os.path.isabs(src["image"]) else src["image"])
    stage = spec.get("stage", stage)
    base = translate_ir(rgb, stage)
    for inst in RELATIONS[spec["relation"]].make(rgb, base, gt):
        if inst.relation == spec["family"].split(".", 1)[1]:
            return inst
    raise ValueError(f"relation {spec['relation']} no longer applies to {src}")


def ledger_entry(scenarios: list[dict], before: dict, gates: dict) -> dict:
    """A ``knowledge/ledger.jsonl`` record (kind ``scenario_baseline``) for these scenario families; the
    caller (the lead) decides whether to append it."""
    blob = json.dumps(scenarios, sort_keys=True, default=str)
    return {"id": "scenario_baseline-" + hashlib.sha1(blob.encode()).hexdigest()[:8],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": "scenario_baseline", "scope": "global",
            "source": {"type": "scenario", "ids": [s["family"] + ":" + json.dumps(s.get("source", s.get("params")), sort_keys=True) for s in scenarios]},
            "change": {"scenarios_added": len(scenarios), "families": sorted({s["family"] for s in scenarios})},
            "evidence": {"before": before, "after": before, "gates": gates}, "accepted": False, "reverts": None}


# --------------------------------------------------------------------------- benchmark-side statistics
def noise_sensitivity(cases: list, stage: str = "perceive") -> dict:
    """Pure-noise benchmark cases test the nuisance relation directly: T(noisy target) vs T(clean),
    before any pixel gating. Returns, per noise type, the share of cases whose translation changed and
    the mean number of raw (ungated) differences."""
    by: dict[str, list[int]] = {}
    for c in cases:
        if c.kind != "noise":
            continue
        fs = ir_diff(translate_ir(c.target_rgb, stage), translate_ir(c.candidate_rgb, stage), painted_only=True,
                     aspects=("geometry", "text", "color", "radius", "effect", "icon"))
        from dt.adversary.perturb import NoiseRecipe
        rec = NoiseRecipe(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.noise.items()})
        for t in rec.types() or ["(none)"]:
            by.setdefault(t, []).append(len(fs))
    return {t: {"cases": len(v), "changed_frac": float(np.mean([x > 0 for x in v])), "mean_diffs": float(np.mean(v))} for t, v in sorted(by.items())}


def bench_jobs(cases: list, stage: str = "perceive") -> list[tuple]:
    seen, jobs = set(), []
    for c in cases:
        for p in (c.paths.get("target"), c.paths.get("candidate")):
            if p and p not in seen:
                seen.add(p)
                jobs.append((p, stage, 1.0))
    return jobs


def load_bench(out_dir: Optional[str] = None):
    """The shared benchmark: the newest cached build under ``out/adversary/bench`` (never rebuilt here)."""
    from dt.adversary import benchmark as BM
    base = out_dir or os.path.join(ROOT, "out", "adversary", "bench")
    mans = sorted(glob.glob(os.path.join(base, "*", "manifest.json")), key=os.path.getmtime)
    if not mans:
        return BM.build(BM.BenchConfig(), base)
    return BM.Benchmark.load(os.path.dirname(mans[-1]))

