"""Module A: screenshot -> IR Document (OCR + segmentation + hierarchy + layout).

    from dt.perceive import perceive
    doc = perceive("shot.png")            # or perceive(rgb_array)
    sub = perceive_region(rgb, Box(...))  # re-perceive one area (used by the refine critic)

Everything is deterministic CV/OCR; thresholds live in ``dt.params`` under ``perceive.*``.
"""
from __future__ import annotations

import os
from typing import Union

import numpy as np

from dt.params import P, register

from dt.common.image import crop as crop_img, downscale_dpr, load_rgb
from dt.ir import Box, Color, Document, Node
from dt.perceive.hierarchy import build_tree
from dt.perceive.layout import infer_layout
from dt.perceive.ocr import TextLine, get_backend, group_paragraphs, ocr_lines, paragraph_node
from dt.perceive.segment import Region, apply_scrim, estimate_background, find_regions, mask_text

__all__ = ["perceive", "perceive_region", "perceive_parts", "Region", "TextLine", "get_backend"]


def _as_rgb(src: Union[str, "os.PathLike[str]", np.ndarray]) -> tuple[np.ndarray, str | None]:
    if isinstance(src, (str, os.PathLike)):
        src = os.fspath(src)
        return load_rgb(src), src
    arr = np.asarray(src)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("expected an (H, W, 3) RGB uint8 array")
    return np.ascontiguousarray(arr[..., :3]).astype(np.uint8), None


def perceive_parts(rgb: np.ndarray) -> tuple[list[TextLine], list[Node], list[Region], Color]:
    """Run OCR, paragraph grouping and segmentation; returns (lines, text_nodes, regions, bg)."""
    lines = ocr_lines(rgb)
    text_boxes = [l.box for l in lines if l.text]
    icon_boxes = [Box.from_dict(d) for l in lines for d in l.meta.get("icon_words", [])]
    texts = [paragraph_node(p) for p in group_paragraphs(rgb, lines)]
    bg = estimate_background(mask_text(rgb, text_boxes))
    regions = find_regions(rgb, text_boxes, bg)
    for r in regions:  # icon glyphs that OCR merged into text: label them so mapping can use them
        if r.kind == "icon" and any(r.box.intersect(ib).area > 0.5 * r.box.area for ib in icon_boxes):
            r.meta["ocr_icon_word"] = True
    return lines, texts, regions, bg


def perceive(src: Union[str, np.ndarray], dpr: float = 1.0) -> Document:
    """Screenshot (path or RGB array) -> IR Document.

    The root frame covers the image with the estimated background; every node box lies inside
    the image; text nodes carry a TextStyle; frames with >= 2 children carry an inferred Layout.
    """
    rgb, path = _as_rgb(src)
    if dpr != 1.0:
        rgb = downscale_dpr(rgb, dpr)
    H, W = rgb.shape[:2]
    lines, texts, regions, bg = perceive_parts(rgb)
    root = build_tree(regions, texts, W, H, bg)
    scrim = bool(P["perceive.scrim.restack"]) and apply_scrim(root)  # under-scrim content below the overlay, true colours
    doc = Document(width=W, height=H, root=root, dpr=1.0, source_image=path, fonts=["Roboto"])
    icon_summary = _identify_icons(doc, rgb)
    font_summary = _identify_fonts(doc, rgb)
    _clamp_to_image(root, W, H)  # font/icon passes may push a box a fraction of a pixel past the edge
    _refit_group_frames(root)  # icon em boxes / font-fitted text boxes can outgrow the frames grouping drew around them
    infer_layout(root)  # after icon and text boxes are render-verified, so gaps/padding use the real geometry
    doc.palette = _palette(root)
    doc.meta = {"perceive": {"n_lines": len(lines), "n_regions": len(regions), "ocr": get_backend().name,
                             "icons": icon_summary, "fonts": font_summary, "scrim": scrim}}
    return doc


register("perceive.scrim.restack", True, "re-stack a perceived tree around a detected dialog scrim (segment.apply_scrim)")
register("perceive.icons.enabled", True, "identify icon glyphs against the Material Symbols atlas (render-verified)")


def _identify_icons(doc: Document, rgb: np.ndarray) -> dict:
    """Name icons and give them renderable em boxes; never lets a box leave the image."""
    if not bool(P["perceive.icons.enabled"]):
        return {"enabled": False}
    try:
        from dt.perceive.icons import annotate_icons
    except Exception as e:  # atlas or font missing: perception still works, icons stay unnamed
        return {"error": f"{type(e).__name__}: {e}"}
    before = {n.id: n.box for n in doc.walk() if n.type == "icon"}
    try:
        summary = annotate_icons(doc, rgb)
    except Exception as e:  # e.g. a browser timeout under load: keep the unnamed icons, never lose the screen
        for n in doc.walk():
            if n.id in before:
                n.box = before[n.id]
        return {"error": f"{type(e).__name__}: {e}"}
    img = Box(0, 0, doc.width, doc.height)
    for n in doc.walk():
        if n.id in before and not img.contains(n.box):
            n.box, n.icon_name = before[n.id], None
            n.meta["icon_rejected"] = "em box outside image"
    return summary


def perceive_region(rgb: np.ndarray, box: Box) -> Node:
    """Perceive only the area `box` of `rgb`; returns a frame at `box` (absolute coords) whose
    children are the elements found inside. Used by the refine critic for 'missing' regions."""
    x0, y0, x1, y1 = box.as_int()
    x0, y0 = max(0, x0), max(0, y0)  # the crop is clipped to the image: translate by the real origin
    sub = crop_img(rgb, box)
    if sub.size == 0:
        return Node(type="frame", name="region", box=box)
    lines, texts, regions, bg = perceive_parts(np.ascontiguousarray(sub))
    root = build_tree(regions, texts, sub.shape[1], sub.shape[0], bg)
    infer_layout(root)
    for n in root.walk():
        n.box = n.box.translate(x0, y0)
    root.name = "region"
    return root


def _palette(root: Node) -> list[Color]:
    seen: list[Color] = []
    for n in root.walk():
        cands = [f.color for f in n.fills if f.color is not None] + [s.color for s in n.strokes]
        if n.text_style is not None:
            cands.append(n.text_style.color)
        for c in cands:
            if all(c.delta_e(s) > 3 for s in seen):
                seen.append(c)
    return seen


def _identify_fonts(doc: Document, rgb: np.ndarray) -> dict:
    """Render-verified font family/size/weight per text node (dt.perceive.fonts); never fatal."""
    try:
        from dt.perceive.fonts import annotate_fonts
        return annotate_fonts(doc, rgb)
    except Exception as e:  # perception still works with the Roboto calibration
        return {"error": f"{type(e).__name__}: {e}"}


def _refit_group_frames(root: Node) -> None:
    """Inferred (transparent) container frames must enclose their children after later passes changed
    child boxes; grow them to the children's union, clipped to their parent. Painted nodes never move."""
    def visit(n: Node, parent: Node | None) -> None:
        for c in n.children:
            visit(c, n)
        if n.meta.get("src") == "group" and n.children and not n.fills and not n.strokes:
            u = n.box
            for c in n.children:
                u = u.union(c.box)
            if parent is not None:
                u = u.intersect(parent.box) if parent.box.intersect(u).area > 0 else u
            n.box = u
    visit(root, None)


def _clamp_to_image(root: Node, W: int, H: int) -> None:
    """perceive()'s contract: every node box lies inside the image."""
    img = Box(0, 0, W, H)
    for n in root.walk():
        if not img.contains(n.box, tol=0.0):
            c = n.box.intersect(img)
            if c.area > 0:
                n.box = c
