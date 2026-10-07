"""Pure-IR synthetic corpus (module E, phase 1).

``generate(n, out_dir, seed)`` builds random Material-3-flavoured ``dt.ir.Document`` trees
straight from the token tables in :mod:`dt.selftest.grammar` (cards with headline/supporting
text, button rows, lists with dividers, icons, chips, text-field outlines, app bars, navigation
bars, FABs), renders each with the real renderer (``dt.render.screenshot.render_doc``) and saves
``<id>.png`` + ``<id>.gt.json``. The ground truth **is** the IR that was rendered, so every
perception stage can be scored against exact boxes, colours, radii, text and styles.

Nodes that mimic a Material component carry a ``ComponentRef`` in the shared variant vocabulary
(``dt.mapping.material3.VARIANTS``, docs/VARIANTS.md; same keys as the Material Web corpus and the
matcher), grouping frames that are not catalog components carry ``meta.part``, and fills carry
``tokens`` (``md.sys.color.*``) so the mapping stage can be scored too.

Text boxes are tightened to the real glyph extents with one measurement pass in the renderer
(``measure_text_widths``), keeping the alignment anchor (left edge / centre / right edge).
"""
from __future__ import annotations

import html as _html
import json
import os
import random
from typing import Optional

from dt.ir import Box, ComponentRef, Document, Fill, Node, Stroke, TextStyle
from dt.mapping.material3 import PARTS as M3_PARTS, VARIANTS as M3_VARIANTS, component_ref as m3_component_ref
from dt.params import P, register
from dt.render.screenshot import render_doc, render_url
from dt.selftest import grammar as G

register("selftest.synth.char_width_ratio", 0.52, "estimated average glyph advance / font size, used before text boxes are measured", (0.4, 0.7))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FONTS_DIR = os.path.join(ROOT, "fixtures", "fonts")
WIDTHS = (412, 600, 840, 1200)
HEIGHT_RANGE = (600, 1000)
PAD = 16  # M3 screen edge padding (4px grid)


# --------------------------------------------------------------------------- node factories
def _fill(role: str) -> tuple[list[Fill], dict[str, str]]:
    return [Fill.solid(G.color(role))], {"fill": G.token_name(role)}


def frame(name: str, box: Box, fill: Optional[str] = None, radius: float = 0, stroke: Optional[str] = None, elevation: int = 0) -> Node:
    """A rectangle frame painted with M3 colour roles (``fill``/``stroke`` are role names)."""
    n = Node(type="frame", name=name, box=box, radius=(radius, radius, radius, radius))
    if fill:
        n.fills, n.tokens = _fill(fill)
    if stroke:
        n.strokes = [Stroke(color=G.color(stroke), width=1.0, align="inside")]
        n.tokens["stroke"] = G.token_name(stroke)
    if elevation:
        n.effects = [s for s in G.M3_ELEVATION[elevation]]
        n.tokens["elevation"] = f"md.sys.elevation.level{elevation}"
    return n


def text(s: str, x: float, y: float, role: str, color: str = "on-surface", align: str = "left", max_w: Optional[float] = None) -> Node:
    """A single-line text node; width is estimated here and tightened by ``fit_text_boxes``."""
    tr = G.M3_TYPESCALE[role]
    est = len(s) * tr.size * float(P["selftest.synth.char_width_ratio"]) + len(s) * tr.tracking
    w = est if max_w is None else min(est, max_w)
    if align == "center":
        x = x - w / 2
    elif align == "right":
        x = x - w
    n = Node(type="text", name=f"text:{s[:24]}", box=Box(x, y, w, tr.line_height), text=s, text_style=tr.style(color, align))
    n.tokens = {"text": f"md.sys.typescale.{role}", "color": G.token_name(color)}
    n.meta = {"anchor": align}
    if max_w is not None:
        n.meta["max_w"] = float(max_w)
    return n


def icon(name: str, x: float, y: float, size: float = 24, color: str = "on-surface-variant") -> Node:
    n = Node(type="icon", name=f"icon:{name}", box=Box(x, y, size, size), icon_name=name)
    n.fills, n.tokens = _fill(color)
    return n


def ref(name: str, variant: Optional[dict] = None, **props: object) -> ComponentRef:
    """Ground-truth ref in the shared vocabulary; ``variant`` is validated by ``material3.variant``."""
    return m3_component_ref(name, props=dict(props), evidence={"src": "synth"}, **(variant or {}))


# --------------------------------------------------------------------------- blocks
class _Screen:
    """Stacks blocks top-to-bottom on a blank document; all randomness via ``rng``."""

    def __init__(self, rng: random.Random, width: int, height: int):
        self.rng = rng
        self.w = width
        self.h = height
        self.doc = Document.blank(width, height, G.color("surface"))
        self.doc.root.tokens = {"fill": G.token_name("surface")}
        self.y = 0.0

    def add(self, n: Node) -> Node:
        self.doc.root.children.append(n)
        return n

    # ---- blocks (each returns the height consumed)
    def app_bar(self) -> float:
        rng = self.rng
        bar = self.add(frame("TopAppBar", Box(0, self.y, self.w, 64), fill=rng.choice(["surface", "surface-container"])))
        center = rng.random() < 0.4
        bar.component = ref("TopAppBar", {"size": "center" if center else "small"})
        bar.children.append(icon(rng.choice(["menu", "arrow_back"]), 12, self.y + 20))
        title = rng.choice(G.APP_TITLES + G.SECTION_TITLES)
        if center:
            bar.children.append(text(title, self.w / 2, self.y + 18, "title-large", align="center"))
        else:
            bar.children.append(text(title, 60, self.y + 18, "title-large"))
        x = self.w - 12 - 24
        for ic in G.pick(rng, G.ACTION_ICONS, rng.randint(1, 3)):
            bar.children.append(icon(ic, x, self.y + 20))
            x -= 48
        bar.component.props["label"] = title
        return 64

    def headline(self) -> float:
        rng = self.rng
        h = rng.choice(G.HEADLINES)
        self.add(text(h, PAD, self.y + 16, "headline-small"))
        self.add(text(rng.choice(G.SNIPPETS), PAD, self.y + 56, "body-medium", "on-surface-variant", max_w=self.w - 2 * PAD))
        return 88

    def button(self, label: str, x: float, y: float, kind: str, right_edge: bool = False) -> Node:
        tr = G.M3_TYPESCALE["label-large"]
        w = len(label) * tr.size * float(P["selftest.synth.char_width_ratio"]) + 48
        if right_edge:
            x = x - w
        spec = {
            "filled": ("primary", None, 0, "on-primary"), "tonal": ("secondary-container", None, 0, "on-secondary-container"),
            "outlined": (None, "outline", 0, "primary"), "text": (None, None, 0, "primary"), "elevated": ("surface-container-low", None, 1, "primary"),
        }[kind]
        b = frame(f"Button/{kind}", Box(x, y, w, 40), fill=spec[0], radius=20, stroke=spec[1], elevation=spec[2])
        b.component = ref("Button", {"style": kind}, label=label)
        b.children.append(text(label, x + w / 2, y + 10, "label-large", spec[3], align="center"))
        b.meta = {"fit_label": True, "right_edge": right_edge}
        return b

    def button_row(self) -> float:
        rng = self.rng
        kinds = G.pick(rng, ["filled", "tonal", "outlined", "text", "elevated"], rng.randint(2, 3))
        x = float(PAD)
        for k in kinds:
            b = self.add(self.button(rng.choice(G.ACTIONS), x, self.y + 8, k))
            x = b.box.x2 + 8
        return 56

    def card_row(self) -> float:
        rng = self.rng
        cols = 1 if self.w < 600 else (2 if self.w < 1000 else 3)
        gap = 12
        cw = (self.w - 2 * PAD - gap * (cols - 1)) / cols
        ch = rng.choice([120, 148, 180])
        for c in range(cols):
            kind = rng.choice(["elevated", "filled", "outlined"])
            x = PAD + c * (cw + gap)
            spec = {"elevated": ("surface-container-low", None, 1), "filled": ("surface-container-highest", None, 0), "outlined": ("surface", "outline-variant", 0)}[kind]
            card = self.add(frame(f"Card/{kind}", Box(x, self.y + 8, cw, ch), fill=spec[0], radius=12, stroke=spec[1], elevation=spec[2]))
            hl = rng.choice(G.HEADLINES + G.SUBJECTS)
            card.component = ref("Card", {"style": kind}, label=hl)
            card.children.append(text(hl, x + 16, self.y + 24, "title-medium", max_w=cw - 32))
            card.children.append(text(rng.choice(G.SNIPPETS), x + 16, self.y + 52, "body-medium", "on-surface-variant", max_w=cw - 32))
            if ch >= 148 and rng.random() < 0.7:
                b = self.button(rng.choice(G.ACTIONS), x + cw - 16, self.y + 8 + ch - 56, rng.choice(["text", "filled", "tonal"]), right_edge=True)
                card.children.append(b)
            elif rng.random() < 0.5:
                card.children.append(icon(rng.choice(G.ACTION_ICONS), x + cw - 40, self.y + 24))
        return ch + 16

    def list_block(self) -> float:
        rng = self.rng
        n = rng.randint(3, 5)
        two = rng.random() < 0.6
        ih = 72 if two else 56
        y0 = self.y
        if rng.random() < 0.5:
            self.add(text(rng.choice(G.SECTION_TITLES), PAD, self.y + 16, "title-small", "on-surface-variant"))
            self.y += 44
        lst = self.add(frame("List", Box(0, self.y, self.w, n * ih + (n - 1))))
        lst.meta["part"] = "List"  # grouping container, not a catalog component
        for i in range(n):
            y = self.y + i * (ih + 1)
            item = frame("ListItem", Box(0, y, self.w, ih))
            item.component = ref("ListItem", {"lines": 2 if two else 1})
            lead = rng.random() < 0.7
            tx = 56 if lead else 16
            if lead:
                item.children.append(icon(rng.choice(G.LEADING_ICONS), 16, y + (ih - 24) / 2))
            hl = rng.choice(G.PEOPLE + G.FILES + G.SETTINGS)
            item.component.props["label"] = hl
            if two:
                item.children.append(text(hl, tx, y + 14, "body-large", max_w=self.w - tx - 80))
                item.children.append(text(rng.choice(G.SNIPPETS), tx, y + 38, "body-medium", "on-surface-variant", max_w=self.w - tx - 80))
            else:
                item.children.append(text(hl, tx, y + 16, "body-large", max_w=self.w - tx - 80))
            trail = rng.choice(["time", "icon", "switch", "none"])
            if trail == "time":
                item.children.append(text(rng.choice(G.TIMES), self.w - 16, y + 16 if not two else y + 14, "label-small", "on-surface-variant", align="right"))
            elif trail == "icon":
                item.children.append(icon(rng.choice(["more_vert", "chevron_right", "star"]), self.w - 40, y + (ih - 24) / 2))
            elif trail == "switch":
                on = rng.random() < 0.5
                sw = frame("Switch", Box(self.w - 16 - 52, y + (ih - 32) / 2, 52, 32), fill="primary" if on else "surface-container-highest", radius=16, stroke=None if on else "outline")
                sw.component = ref("Switch", {"selected": on})
                hx = sw.box.x2 - 4 - 24 if on else sw.box.x + 8
                hy = sw.box.y + 4 if on else sw.box.y + 8
                handle = Node(type="ellipse", name="handle", box=Box(hx, hy, 24 if on else 16, 24 if on else 16))
                handle.fills, handle.tokens = _fill("on-primary" if on else "outline")
                sw.children.append(handle)
                item.children.append(sw)
            lst.children.append(item)
            if i < n - 1:
                div = Node(type="line", name="divider", box=Box(16 if lead else 0, y + ih, self.w - (16 if lead else 0), 1))
                div.fills, div.tokens = _fill("outline-variant")
                div.component = ref("Divider")
                lst.children.append(div)
        self.y = lst.box.y2
        return self.y - y0

    def chip_row(self) -> float:
        rng = self.rng
        x = float(PAD)
        kind = rng.choice(["assist", "filter", "suggestion"])
        for lab in G.pick(rng, G.CHIP_LABELS, rng.randint(3, 4)):
            tr = G.M3_TYPESCALE["label-large"]
            w = len(lab) * tr.size * float(P["selftest.synth.char_width_ratio"]) + 32
            sel = kind == "filter" and rng.random() < 0.4
            chip = frame(f"Chip/{kind}", Box(x, self.y + 8, w, 32), fill="secondary-container" if sel else None, radius=8, stroke=None if sel else "outline")
            chip.component = ref("Chip", {"type": kind, "selected": sel}, label=lab)
            chip.children.append(text(lab, x + w / 2, self.y + 14, "label-large", "on-secondary-container" if sel else "on-surface", align="center"))
            chip.meta = {"fit_label": True}
            self.add(chip)
            x = chip.box.x2 + 8
        return 48

    def text_fields(self) -> float:
        rng = self.rng
        n = rng.randint(1, 2)
        y0 = self.y
        for i in G.pick(rng, list(range(len(G.FIELD_LABELS))), n):
            kind = rng.choice(["outlined", "filled"])
            lab, val = G.FIELD_LABELS[i], G.FIELD_VALUES[i]
            if kind == "outlined":
                f = self.add(frame("TextField/outlined", Box(PAD, self.y + 12, self.w - 2 * PAD, 56), radius=4, stroke="outline"))
                lab_node = text(lab, PAD + 16, self.y + 12 - 8, "body-small", "on-surface-variant")
                lab_bg = frame("label-bg", lab_node.box.expand(2), fill="surface")
                lab_bg.children.append(lab_node)
                lab_bg.meta = {"fit_label": True}
                self.add(lab_bg)
            else:
                f = self.add(frame("TextField/filled", Box(PAD, self.y + 12, self.w - 2 * PAD, 56), fill="surface-container-highest", radius=4))
                f.radius = (4, 4, 0, 0)
                line = Node(type="line", name="indicator", box=Box(PAD, self.y + 12 + 55, self.w - 2 * PAD, 1))
                line.fills, line.tokens = _fill("on-surface-variant")
                f.children.append(line)
                f.children.append(text(lab, PAD + 16, self.y + 12 + 8, "body-small", "on-surface-variant"))
            f.component = ref("TextField", {"style": kind}, label=lab, value=val)
            if val:
                f.children.append(text(val, PAD + 16, self.y + 12 + (16 if kind == "outlined" else 24), "body-large"))
            self.y += 72
        return self.y - y0 + 8

    def fab(self, bottom: float) -> None:
        rng = self.rng
        size = rng.choice([40, 56, 56, 96])
        variant = rng.choice(["surface", "primary", "secondary", "tertiary"])
        fill = {"surface": "surface-container-high", "primary": "primary-container", "secondary": "secondary-container", "tertiary": "tertiary-container"}[variant]
        fg = {"surface": "primary", "primary": "on-primary-container", "secondary": "on-secondary-container", "tertiary": "on-tertiary-container"}[variant]
        r = {40: 12, 56: 16, 96: 28}[size]
        f = self.add(frame("FAB", Box(self.w - PAD - size, self.h - bottom - size, size, size), fill=fill, radius=r, elevation=3))
        f.component = ref("FAB", {"size": {40: "small", 56: "regular", 96: "large"}[size], "extended": False})
        f.meta["fab_color"] = variant  # colour family is a token choice, not a variant
        isz = 36 if size == 96 else 24
        f.children.append(icon(rng.choice(G.FAB_ICONS), f.box.x + (size - isz) / 2, f.box.y + (size - isz) / 2, isz, fg))

    def nav_bar(self) -> None:
        rng = self.rng
        bar = self.add(frame("NavigationBar", Box(0, self.h - 80, self.w, 80), fill="surface-container"))
        dests = G.pick(rng, G.NAV_DESTS, rng.randint(3, 5))
        bar.component = ref("NavigationBar", destinations=len(dests))
        active = rng.randrange(len(dests))
        slot = self.w / len(dests)
        for i, (ic, lab) in enumerate(dests):
            cx = slot * i + slot / 2
            if i == active:
                pill = frame("indicator", Box(cx - 32, self.h - 80 + 12, 64, 32), fill="secondary-container", radius=16)
                bar.children.append(pill)
            bar.children.append(icon(ic, cx - 12, self.h - 80 + 16, 24, "on-surface" if i == active else "on-surface-variant"))
            bar.children.append(text(lab, cx, self.h - 80 + 48, "label-medium", "on-surface" if i == active else "on-surface-variant", align="center"))

    # ---- composition
    def compose(self) -> Document:
        rng = self.rng
        bottom_reserved = 0.0
        has_nav = self.w < 600 and rng.random() < 0.7
        if has_nav:
            bottom_reserved = 80
        if rng.random() < 0.9:
            self.y += self.app_bar()
        pool = [self.list_block, self.card_row, self.button_row, self.chip_row, self.text_fields, self.headline]
        weights = [4, 3, 2, 2, 2, 2]
        limit = self.h - bottom_reserved - 8
        for _ in range(rng.randint(3, 7)):
            fn = rng.choices(pool, weights)[0]
            start, before = self.y, len(self.doc.root.children)
            consumed = fn()
            self.y = max(self.y, start + consumed)
            if self.y > limit:
                # drop the block that ran past the reserved bottom strip (keeps GT fully visible)
                del self.doc.root.children[before:]
                self.y = start
                break
        if has_nav:
            self.nav_bar()
        if rng.random() < 0.5:
            self.fab(bottom_reserved + 16)
        return self.doc


# --------------------------------------------------------------------------- text measurement
_MEASURE_JS = """
(() => Array.from(document.querySelectorAll('span[data-i]')).map(s => s.getBoundingClientRect().width))()
"""


def measure_text_widths(items: list[tuple[str, TextStyle]], work_dir: str) -> list[float]:
    """Measure rendered single-line widths (px) for ``(text, style)`` pairs with the real renderer."""
    if not items:
        return []
    spans = []
    for i, (s, ts) in enumerate(items):
        css = (f"font-family:'{ts.family}';font-size:{ts.size}px;font-weight:{ts.weight};letter-spacing:{ts.letter_spacing}px;"
               f"font-style:{'italic' if ts.italic else 'normal'};white-space:nowrap;position:absolute;left:0;top:{i * 40}px")
        spans.append(f'<span data-i="{i}" style="{css}">{_html.escape(s)}</span>')
    html = (f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="file://{FONTS_DIR}/roboto.local.css">'
            f'<link rel="stylesheet" href="file://{FONTS_DIR}/material-symbols.local.css"><style>body{{margin:0}} * {{-webkit-font-smoothing:antialiased;text-rendering:geometricPrecision}}</style></head><body>'
            + "".join(spans) + "</body></html>")
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, "_measure.html")
    with open(path, "w") as f:
        f.write(html)
    _rgb, widths = render_url("file://" + os.path.abspath(path), 800, 100, wait_ms=50, script=_MEASURE_JS)
    os.remove(path)
    return [float(w) for w in widths]


def _truncate_words(s: str, ratio: float) -> str:
    """Drop trailing words until roughly ``ratio`` (<1) of the string length remains."""
    words = s.split()
    target = max(1, int(len(s) * ratio))
    out: list[str] = []
    for w in words:
        if len(" ".join(out + [w])) > target and out:
            break
        out.append(w)
    return " ".join(out)


def _set_width(n: Node, w: float) -> None:
    anchor = n.meta.get("anchor", "left")
    if anchor == "center":
        n.box = Box(n.box.cx - w / 2, n.box.y, w, n.box.h)
    elif anchor == "right":
        n.box = Box(n.box.x2 - w, n.box.y, w, n.box.h)
    else:
        n.box = Box(n.box.x, n.box.y, w, n.box.h)


def fit_text_boxes(doc: Document, work_dir: str, max_passes: int = 3) -> None:
    """Tighten every text node's width to its measured glyph width, keeping its alignment anchor.

    Texts that exceed their ``meta.max_w`` are shortened word-wise and re-measured (a few passes),
    so no ground-truth text ever wraps or overflows its container in the render.
    """
    pending = [n for n in doc.walk() if n.type == "text" and n.text and n.text_style]
    for _ in range(max_passes):
        widths = measure_text_widths([(n.text or "", n.text_style) for n in pending], work_dir)  # type: ignore[list-item]
        again = []
        for n, w in zip(pending, widths):
            w = float(round(w + 0.5))
            max_w = n.meta.get("max_w")
            if max_w is not None and w > max_w and n.text and " " in n.text.strip():
                n.text = _truncate_words(n.text, max_w / w * 0.95)
                n.name = f"text:{n.text[:24]}"
                again.append(n)
                continue
            _set_width(n, min(w, max_w) if max_w is not None else w)
        if not again:
            break
        pending = again
    for n in doc.walk():
        if n.type == "text":
            n.meta.pop("anchor", None)
            n.meta.pop("max_w", None)
    # containers that were sized from estimated labels (buttons, chips, outlined label plates)
    for n in doc.walk():
        if n.meta.pop("fit_label", False) and n.children:
            label = n.children[-1]
            pad = 2 if n.name == "label-bg" else (16 if n.component and n.component.name == "Chip" else 24)
            new_w = label.box.w + 2 * pad
            if n.component and n.component.name == "Button" and n.meta.pop("right_edge", False):
                n.box = Box(n.box.x2 - new_w, n.box.y, new_w, n.box.h)
            else:
                n.box = Box(n.box.x, n.box.y, new_w, n.box.h)
            label.box = Box(n.box.cx - label.box.w / 2, label.box.y, label.box.w, label.box.h) if n.name != "label-bg" else Box(n.box.x + pad, label.box.y, label.box.w, label.box.h)
    _reflow_rows(doc)


def _reflow_rows(doc: Document) -> None:
    """Re-space horizontally packed siblings (button/chip rows) after their widths changed."""
    rows: dict[float, list[Node]] = {}
    for c in doc.root.children:
        if c.component and c.component.name in ("Button", "Chip"):
            rows.setdefault(c.box.y, []).append(c)
    for row in rows.values():
        row.sort(key=lambda n: n.box.x)
        x = row[0].box.x
        for n in row:
            dx = x - n.box.x
            if abs(dx) > 1e-6:
                for m in n.walk():
                    m.box = m.box.translate(dx, 0)
            x = n.box.x2 + 8


# --------------------------------------------------------------------------- corpus
def make_document(rng: random.Random, width: int, height: int, work_dir: str) -> Document:
    """One random synthetic screen, text boxes measured; not yet rendered."""
    doc = _Screen(rng, width, height).compose()
    fit_text_boxes(doc, work_dir)
    from dt.ir import assign_ids
    assign_ids(doc.root)
    doc.design_system = "material3"
    doc.fonts = ["Roboto", "Material Symbols Outlined"]
    return doc


def generate(n: int, out_dir: str, seed: int = 1, widths: tuple[int, ...] = WIDTHS, height_range: tuple[int, int] = HEIGHT_RANGE) -> dict:
    """Generate ``n`` synthetic IR documents + renders under ``out_dir``; returns the manifest."""
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    cases = []
    comp_hist: dict[str, int] = {}
    for i in range(n):
        width = rng.choice(widths)
        height = rng.randint(height_range[0], height_range[1])
        cid = f"synth_{seed}_{i:03d}"
        doc = make_document(random.Random(rng.random()), width, height, out_dir)
        png_path = os.path.join(out_dir, cid + ".png")
        render_doc(doc, png_path)
        doc.source_image = cid + ".png"  # relative to out_dir; load_case() makes it absolute
        doc.meta = {"corpus": "synth", "id": cid, "seed": seed, "index": i}
        doc.save(os.path.join(out_dir, cid + ".gt.json"))
        hist: dict[str, int] = {}
        types: dict[str, int] = {}
        for nd in doc.walk():
            types[nd.type] = types.get(nd.type, 0) + 1
            if nd.component:
                hist[nd.component.key] = hist.get(nd.component.key, 0) + 1
                comp_hist[nd.component.key] = comp_hist.get(nd.component.key, 0) + 1
        cases.append({"id": cid, "width": width, "height": height, "nodes": doc.root.count(), "components": dict(sorted(hist.items())), "types": dict(sorted(types.items()))})
    manifest = {"kind": "synth", "seed": seed, "n": n, "widths": list(widths), "height_range": list(height_range), "ids": [c["id"] for c in cases], "cases": cases, "component_histogram": dict(sorted(comp_hist.items()))}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_case(out_dir: str, cid: str) -> tuple[Document, str]:
    """Return ``(gt_document, png_path)`` for a corpus case id (``source_image`` made absolute)."""
    doc = Document.load(os.path.join(out_dir, cid + ".gt.json"))
    png = os.path.join(out_dir, cid + ".png")
    doc.source_image = png
    return doc, png
