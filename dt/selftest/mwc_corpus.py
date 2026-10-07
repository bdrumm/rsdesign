"""Material Web ground-truth corpus (module E, phase 1).

``generate(n, out_dir, seed)`` composes random Google-app-like Material 3 screens out of real
Material Web elements (``md-filled-button``, ``md-list-item``, ``md-switch`` ...) plus hand-styled
M3 surfaces (top app bar, navigation bar/rail, cards, snackbar, badge), renders each page with
the real renderer (``dt.render.screenshot.render_url``) and extracts the **ground-truth IR from
the DOM** (light DOM + every shadow root, computed styles, per-line text rects).

Per page it writes ``<id>.html``, ``<id>.png`` and ``<id>.gt.json`` (a ``dt.ir.Document``), and
the corpus gets a ``manifest.json`` (ids, seed, sizes, component histogram). Everything is
deterministic from ``seed``.

Ground truth conventions
------------------------
* Every Material Web host element authored in the light DOM (and every hand-styled surface with
  a ``data-component`` attribute) becomes a ``frame`` node with
  ``component=ComponentRef(key='m3.<name>:<k=v,...>', library='material3', ...)`` whose
  ``variant`` uses exactly the shared vocabulary of ``dt.mapping.material3.VARIANTS``
  (docs/VARIANTS.md). Sub-parts and grouping containers that are not catalog components
  (md-list, md-chip-set, md-*-tab, md-menu-item, md-select-option) get ``meta.part`` instead.
* Paint lifting: Material Web paints most surfaces on an inner element, so the host is an
  unpainted square box while its child carries the pill. When an md-* host is unpainted and one
  child covers >= ``selftest.mwc.lift_cover`` of it, that child's fill/stroke/radius/shadow move
  onto the host, the child is removed (its children re-parented to the host) and
  ``meta.lifted_from`` records it. ``md-dialog``'s component moves to its surface container.
* Inline ``<svg>`` vectors take the colour of the first visible shape (outside mask/clipPath/
  defs) whose fill or stroke differs from the backdrop behind the svg; ``md-icon`` keeps its 24px
  font box as ``box`` and records the tight glyph bounds in ``meta.ink_box`` ([x, y, w, h]).
* Painted boxes (background / border / box-shadow, including ``::before``/``::after``
  pseudo-elements) become ``frame``/``rect``/``ellipse`` nodes; text becomes one ``text`` node
  per rendered line; ``md-icon`` becomes an ``icon`` node; ``md-divider``/``hr`` a ``line``;
  inline ``<svg>`` a ``vector``; ``<img>`` an ``image``; ``<input>`` values become text nodes.
* A text node's box is its **line box** (height == ``text_style.line_height``), the same
  contract as synth: the DOM's content-area rect is widened by the half-leading on both sides.
* Tree pre-order is paint order: children are sorted by stacking layer (``meta.z``) and a node
  that must paint above content outside its DOM parent (open menu, top layer, an outline over a
  snackbar) is re-parented to the root (``meta.reparented_from``), see ``_restack``.
* Nesting follows DOM ancestry; a node whose box pokes out of its DOM parent is re-parented to
  the nearest ancestor that contains it so the IR invariant "box encloses children" holds
  (``meta.reparented_from`` records the original parent).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Optional

from dt.ir import Box, Color, ComponentRef, Document, Fill, Node, Shadow, Stroke, TextStyle
from dt.mapping.material3 import PARTS as M3_PARTS, VARIANTS as M3_VARIANTS, component_ref as m3_component_ref
from dt.params import P, register
from dt.render.screenshot import render_url
from dt.selftest import grammar as G

register("selftest.mwc.wait_ms", 600, "ms to wait after load before extracting ground truth (lets Material Web upgrade/layout)", (200, 2000))
register("selftest.mwc.min_size", 1.0, "px; boxes smaller than this on either axis are not emitted as nodes", (0.5, 4.0))
register("selftest.mwc.min_alpha", 0.02, "effective alpha below which a paint (fill/stroke/shadow/text) counts as invisible", (0.0, 0.2))
register("selftest.mwc.settle_rounds", 3, "rounds of awaiting every Lit element's updateComplete (+2 frames) before extracting", (1, 6))
register("selftest.mwc.contain_tol", 0.75, "px tolerance when deciding whether a child box is inside its parent (re-parent otherwise)", (0.0, 4.0))
register("selftest.mwc.lift_cover", 0.9, "fraction of an unpainted md-* host's area a painted child must cover for its paint to be lifted onto the host", (0.5, 1.0))
register("selftest.mwc.restack_overlap", 1.0, "px^2 of overlap from which two painted nodes drawn in the wrong order count as a paint-order conflict (see _restack)", (0.0, 50.0))
register("selftest.mwc.backdrop_de", 3.0, "ΔE below which an svg shape's colour counts as equal to its backdrop (skipped when picking the vector colour)", (0.5, 10.0))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FONTS_DIR = os.path.join(ROOT, "fixtures", "fonts")
BUNDLE = os.path.join(ROOT, "fixtures", "mwc", "mwc.bundle.js")

WIDTHS = (412, 600, 840, 1200)
HEIGHT_RANGE = (600, 1000)

# --------------------------------------------------------------------------- component table
#: tag -> (component name, {variant key: value}) for the tags whose variant is the tag itself.
_STYLE_TAGS: dict[str, tuple[str, dict[str, str]]] = {
    "md-filled-button": ("Button", {"style": "filled"}), "md-outlined-button": ("Button", {"style": "outlined"}),
    "md-text-button": ("Button", {"style": "text"}), "md-elevated-button": ("Button", {"style": "elevated"}),
    "md-filled-tonal-button": ("Button", {"style": "tonal"}),
    "md-icon-button": ("IconButton", {"style": "standard"}), "md-filled-icon-button": ("IconButton", {"style": "filled"}),
    "md-filled-tonal-icon-button": ("IconButton", {"style": "tonal"}), "md-outlined-icon-button": ("IconButton", {"style": "outlined"}),
    "md-outlined-text-field": ("TextField", {"style": "outlined"}), "md-filled-text-field": ("TextField", {"style": "filled"}),
    "md-outlined-select": ("Select", {"style": "outlined"}), "md-filled-select": ("Select", {"style": "filled"}),
    "md-assist-chip": ("Chip", {"type": "assist"}), "md-filter-chip": ("Chip", {"type": "filter"}),
    "md-input-chip": ("Chip", {"type": "input"}), "md-suggestion-chip": ("Chip", {"type": "suggestion"}),
    "md-switch": ("Switch", {}), "md-checkbox": ("Checkbox", {}), "md-radio": ("Radio", {}), "md-slider": ("Slider", {}),
    "md-divider": ("Divider", {}), "md-dialog": ("Dialog", {}), "md-menu": ("Menu", {}),
    "md-linear-progress": ("Progress", {"type": "linear"}), "md-circular-progress": ("Progress", {"type": "circular"}),
}
#: md-fab ``size`` attribute -> vocabulary size
_FAB_SIZE = {"small": "small", "medium": "regular", "large": "large"}


def _has(attrs: dict[str, str], name: str) -> bool:
    return name in attrs and attrs[name] != "false"


def part_for(tag: str) -> Optional[str]:
    """Name of a Material Web sub-part / grouping container (``meta.part``), or None."""
    return M3_PARTS.get(tag)


def component_for(tag: str, attrs: dict[str, str], info: dict[str, Any]) -> Optional[ComponentRef]:
    """Map a Material Web tag (+attributes, +extraction info) to a ``ComponentRef`` whose variant
    is exactly the shared vocabulary (``dt.mapping.material3.VARIANTS``, docs/VARIANTS.md).

    ``info`` carries what the JS side computed: ``label``, ``icon``, ``value``, ``height``,
    ``child_tags`` (slotted light-DOM children) and ``data`` (``data-*`` attributes). Hand-styled
    surfaces declare ``data-component="<Name>"`` plus ``data-variant-<key>="<value>"`` (or a single
    ``data-variant`` for a component with exactly one variant key). Returns None for parts
    (:func:`part_for`) and unknown tags.
    """
    data = info.get("data", {})
    name: Optional[str] = None
    vd: dict[str, Any] = {}
    if tag in _STYLE_TAGS:
        name, base = _STYLE_TAGS[tag]
        vd = dict(base)
        if name == "Chip":
            vd["selected"] = _has(attrs, "selected")
        elif name == "Switch":
            vd["selected"] = _has(attrs, "selected")
        elif name in ("Checkbox", "Radio"):
            # an indeterminate checkbox paints the filled container like a checked one
            vd["checked"] = _has(attrs, "checked") or _has(attrs, "indeterminate")
    elif tag == "md-fab":
        name = "FAB"
        vd = {"size": _FAB_SIZE.get(attrs.get("size", "medium"), "regular"), "extended": bool(attrs.get("label"))}
    elif tag == "md-list-item":
        name = "ListItem"
        h = float(info.get("height", 56))
        vd = {"lines": 1 if h <= 60 else (2 if h <= 80 else 3)}
    elif tag == "md-tabs":
        name = "Tabs"
        vd = {"type": "secondary" if "md-secondary-tab" in info.get("child_tags", {}).get("", []) else "primary"}
    elif data.get("component"):
        name = str(data["component"])
        vocab = M3_VARIANTS.get(name, {})
        for k, v in data.items():
            # DOMStringMap camel-cases data-variant-size -> variantSize; accept both spellings
            key = k[len("variant-"):] if k.startswith("variant-") else (k[7:8].lower() + k[8:] if k.startswith("variant") and len(k) > 7 else None)
            if key in vocab:
                vd[key] = v
        if data.get("variant") and len(vocab) == 1:
            vd.setdefault(next(iter(vocab)), data["variant"])
    if name is None or name not in M3_VARIANTS:
        return None
    props: dict[str, object] = {}
    for key in ("label", "icon", "value", "supporting_text", "trailing_text", "headline"):
        if info.get(key) not in (None, ""):
            props[key] = info[key]
    return m3_component_ref(name, props=props, evidence={"src": "dom", "tag": tag}, **vd)


# --------------------------------------------------------------------------- in-page extractor
#: Evaluated in the page by ``render_url``. Returns ``{width, height, bg, nodes:[...]}`` where
#: each node is flat with a ``parent`` index (pre-order, flat-tree walk through shadow roots and
#: slots so emission order == paint order).
EXTRACT_JS = r"""
(async () => {
  // settle: fonts + every Lit element's pending update (also inside shadow roots), a few frames
  await document.fonts.ready;
  for (let round = 0; round < __SETTLE_ROUNDS__; round++) {
    const pending = [];
    const collect = root => { for (const e of root.querySelectorAll('*')) { if (e.updateComplete) pending.push(e.updateComplete); if (e.shadowRoot) collect(e.shadowRoot); } };
    collect(document);
    await Promise.all(pending);
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  }
  const MIN = __MIN_SIZE__, MIN_ALPHA = __MIN_ALPHA__, BACKDROP_DE = __BACKDROP_DE__;
  const VW = window.innerWidth, VH = window.innerHeight;
  const out = [];
  const TEXT_INPUTS = new Set(['text', 'email', 'password', 'search', 'tel', 'url', 'number', 'textarea']);
  const SKIP = new Set(['md-ripple', 'md-focus-ring', 'script', 'style', 'link', 'meta', 'template', 'noscript']);
  const R = v => Math.round(v * 100) / 100;
  const px = s => { const v = parseFloat(s); return isNaN(v) ? 0 : v; };
  const parseColor = s => {
    const m = s && s.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[\s,\/]+/).filter(Boolean).map(parseFloat);
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };
  const splitTop = s => { // split on commas outside parentheses
    const parts = []; let depth = 0, cur = '';
    for (const ch of s) { if (ch === '(') depth++; if (ch === ')') depth--; if (ch === ',' && depth === 0) { parts.push(cur.trim()); cur = ''; } else cur += ch; }
    if (cur.trim()) parts.push(cur.trim());
    return parts;
  };
  const parseShadows = (s, op) => {
    if (!s || s === 'none') return [];
    const res = [];
    for (const part of splitTop(s)) {
      const c = parseColor(part); if (!c) continue;
      const nums = part.replace(/rgba?\([^)]*\)/, '').trim().split(/\s+/);
      const inner = nums.includes('inset');
      const v = nums.filter(x => x !== 'inset').map(px);
      c.a = c.a * op;
      if (c.a <= MIN_ALPHA) continue;
      res.push({ color: c, dx: v[0] || 0, dy: v[1] || 0, blur: v[2] || 0, spread: v[3] || 0, inner });
    }
    return res;
  };
  const radiusOf = (cs, w, h) => {
    const r = [cs.borderTopLeftRadius, cs.borderTopRightRadius, cs.borderBottomRightRadius, cs.borderBottomLeftRadius].map(v => px(v.split(' ')[0]));
    const lim = Math.min(w, h) / 2;
    return r.map(v => Math.min(v, lim));
  };
  const strokesOf = (cs, op) => {
    const sides = ['Top', 'Right', 'Bottom', 'Left'].map(s => ({ w: cs['border' + s + 'Style'] === 'none' ? 0 : px(cs['border' + s + 'Width']), c: parseColor(cs['border' + s + 'Color']) }));
    const vis = sides.filter(s => s.w > 0 && s.c && s.c.a * op > MIN_ALPHA);
    if (!vis.length) return [];
    const c = { ...vis[0].c, a: vis[0].c.a * op };
    const ws = sides.map(s => (s.c && s.c.a * op > MIN_ALPHA) ? s.w : 0);
    if (ws.every(w => w === ws[0])) return [{ color: c, width: ws[0] }];
    return [{ color: c, width: Math.max(...ws), sides: ws }];
  };
  const inter = (a, b) => {
    if (!b) return a;
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y), x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
    return { x: x1, y: y1, w: Math.max(0, x2 - x1), h: Math.max(0, y2 - y1) };
  };
  const VIEW = { x: 0, y: 0, w: VW, h: VH };
  const boxOf = (r, clip) => inter(inter({ x: r.left, y: r.top, w: r.width, h: r.height }, VIEW), clip);
  const big = b => b.w >= MIN && b.h >= MIN;
  const emit = n => { n.box = { x: R(n.box.x), y: R(n.box.y), w: R(n.box.w), h: R(n.box.h) }; out.push(n); return out.length - 1; };
  // a gradient approximated by one solid colour; null when any stop is (near-)transparent: that is a
  // pattern with partial coverage (md-slider tick dots are a repeating radial-gradient), never a fill
  const gradientColor = s => {
    const cols = (s.match(/rgba?\([^)]*\)/g) || []).map(parseColor).filter(Boolean);
    if (!cols.length || cols.some(c => c.a <= MIN_ALPHA)) return null;
    return cols[0];
  };
  const paintOf = (cs, op, w, h) => {
    const p = { fill: null, strokes: strokesOf(cs, op), effects: parseShadows(cs.boxShadow, op), radius: radiusOf(cs, w, h), image: null };
    const bg = parseColor(cs.backgroundColor);
    if (bg && bg.a * op > MIN_ALPHA) p.fill = { ...bg, a: bg.a * op };
    if (cs.backgroundImage && cs.backgroundImage !== 'none') {
      const m = cs.backgroundImage.match(/url\(["']?([^"')]+)/);
      if (m) p.image = m[1]; else if (!p.fill) { const c = gradientColor(cs.backgroundImage); if (c && c.a * op > MIN_ALPHA) { p.fill = { ...c, a: c.a * op }; p.gradient = cs.backgroundImage; } }
    }
    return p;
  };
  const painted = p => p.fill || p.image || p.strokes.length || p.effects.length;
  const firstFamily = f => (f || '').split(',')[0].replace(/["']/g, '').trim();
  const textStyleOf = (cs, op) => {
    const c = parseColor(cs.color) || { r: 0, g: 0, b: 0, a: 1 };
    const lh = cs.lineHeight === 'normal' ? null : px(cs.lineHeight);
    let align = cs.textAlign; if (align === 'start' || align === 'justify') align = 'left'; if (align === 'end') align = 'right'; if (!['left', 'center', 'right'].includes(align)) align = 'left';
    return { family: firstFamily(cs.fontFamily), size: px(cs.fontSize), weight: parseInt(cs.fontWeight) || 400, line_height: lh, letter_spacing: cs.letterSpacing === 'normal' ? 0 : px(cs.letterSpacing), color: { ...c, a: c.a * op }, align, italic: cs.fontStyle === 'italic', decoration: (cs.textDecorationLine || 'none').split(' ')[0], transform: cs.textTransform || 'none' };
  };
  const canvas = document.createElement('canvas').getContext('2d');
  const measure = (cs, text) => { canvas.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`; canvas.letterSpacing = cs.letterSpacing === 'normal' ? '0px' : cs.letterSpacing; return canvas.measureText(text).width; };

  // ---- text node -> one node per rendered line
  const emitText = (tn, styleEl, parent, clip, op, layer) => {
    const data = tn.data; if (!data || !data.trim()) return;
    const cs = getComputedStyle(styleEl);
    const st = textStyleOf(cs, op);
    if (st.color.a <= MIN_ALPHA) return;
    const range = document.createRange();
    let lines = [], cur = null;
    for (let i = 0; i < data.length; i++) {
      range.setStart(tn, i); range.setEnd(tn, i + 1);
      const rs = range.getClientRects(); if (!rs.length) continue;
      const r = rs[0]; if (r.width <= 0 || r.height <= 0) continue;
      if (cur && Math.abs(cur.top - r.top) < 1 && Math.abs(cur.bottom - r.bottom) < 1) { cur.text += data[i]; cur.x1 = Math.min(cur.x1, r.left); cur.x2 = Math.max(cur.x2, r.right); }
      else { cur = { text: data[i], top: r.top, bottom: r.bottom, x1: r.left, x2: r.right }; lines.push(cur); }
    }
    for (const ln of lines) {
      const text = ln.text.replace(/\s+$/, '').replace(/^\s+/, '');
      if (!text) continue;
      // trim the leading/trailing whitespace from the box as well
      const leadWS = ln.text.length - ln.text.replace(/^\s+/, '').length;
      // IR contract: a text box is its LINE box. Range rects are content areas (ascent+descent),
      // the line box adds half-leading (lh - content)/2 above and below (CSS inline layout), so a
      // renderer laying `line-height: lh` at the top of the box reproduces the glyph baseline.
      const ch = ln.bottom - ln.top;
      const lh = st.line_height && st.line_height > 0 ? st.line_height : ch;
      const top = ln.top - (lh - ch) / 2;
      // visibility and horizontal clipping come from the glyph content area; an overflow clip that
      // only cuts into the (empty) half-leading must not shorten the line box, or the renderer would
      // shift the glyphs by the clipped amount
      const content = boxOf({ left: ln.x1, top: ln.top, width: ln.x2 - ln.x1, height: ch }, clip);
      if (!big(content)) continue;
      const vClipped = content.y > ln.top + 0.01 || content.y + content.h < ln.bottom - 0.01;
      const lineBox = boxOf({ left: content.x, top, width: content.w, height: lh }, vClipped ? clip : null);
      const box = lineBox;
      if (!big(box)) continue;
      const n = { type: 'text', parent, box, layer, text, text_style: st, meta: { src: 'dom-text', tag: styleEl.tagName.toLowerCase() } };
      if (leadWS) n.meta.lead_ws = leadWS;
      emit(n);
    }
  };

  const emitInput = (el, cs, parent, clip, op, layer) => {
    const value = el.value || ''; const ph = el.getAttribute('placeholder') || '';
    const text = value || ph; if (!text) return;
    const r = el.getBoundingClientRect();
    const x = r.left + px(cs.borderLeftWidth) + px(cs.paddingLeft), y = r.top + px(cs.borderTopWidth) + px(cs.paddingTop);
    const cw = r.width - px(cs.borderLeftWidth) - px(cs.borderRightWidth) - px(cs.paddingLeft) - px(cs.paddingRight);
    const ch = r.height - px(cs.borderTopWidth) - px(cs.borderBottomWidth) - px(cs.paddingTop) - px(cs.paddingBottom);
    const w = Math.min(cw, measure(cs, text));
    const lh = cs.lineHeight === 'normal' ? px(cs.fontSize) * 1.2 : px(cs.lineHeight);
    const h = lh;  // the line box (centred in the content box, may overflow it like the real input)
    const st = textStyleOf(cs, op);
    if (!value) { const pc = parseColor(getComputedStyle(el, '::placeholder').color); if (pc) st.color = { ...pc, a: pc.a * op }; }
    let tx = x; if (st.align === 'center') tx = x + (cw - w) / 2; else if (st.align === 'right') tx = x + cw - w;
    const box = boxOf({ left: tx, top: y + (ch - h) / 2, width: w, height: h }, clip);
    if (!big(box) || st.color.a <= MIN_ALPHA) return;
    emit({ type: 'text', parent, box, layer, text, text_style: st, meta: { src: 'input', tag: el.tagName.toLowerCase(), placeholder: !value } });
  };

  const paddingBox = (e) => {
    const r = e.getBoundingClientRect(), c = getComputedStyle(e);
    return { x: r.left + px(c.borderLeftWidth), y: r.top + px(c.borderTopWidth), w: r.width - px(c.borderLeftWidth) - px(c.borderRightWidth), h: r.height - px(c.borderTopWidth) - px(c.borderBottomWidth) };
  };
  const containingBlock = (el, fixed) => {
    // nearest positioned ancestor (crossing shadow boundaries); viewport otherwise
    if (!fixed) {
      let p = el;
      while (p) {
        const c = getComputedStyle(p);
        if (c.position !== 'static' || (c.transform && c.transform !== 'none')) return paddingBox(p);
        p = p.parentElement || (p.getRootNode() && p.getRootNode().host) || null;
      }
    }
    return { x: 0, y: 0, w: VW, h: VH };
  };
  const pseudoBox = (el, cs, ps, r) => {
    // static/relative pseudo: approximate with the element's padding box; absolute/fixed: resolve
    // inset + size against the containing block (nearest positioned ancestor)
    const own = { x: r.left + px(cs.borderLeftWidth), y: r.top + px(cs.borderTopWidth), w: r.width - px(cs.borderLeftWidth) - px(cs.borderRightWidth), h: r.height - px(cs.borderTopWidth) - px(cs.borderBottomWidth) };
    // computed width/height are content-box sizes unless box-sizing: border-box; the painted box is
    // the border box (a 0px-tall pseudo with a 1px border-bottom is a 1px line, e.g. md-filled-field's
    // active indicator)
    const cb = ps.boxSizing === 'border-box';
    const ew = cb ? 0 : px(ps.paddingLeft) + px(ps.paddingRight) + px(ps.borderLeftWidth) + px(ps.borderRightWidth);
    const eh = cb ? 0 : px(ps.paddingTop) + px(ps.paddingBottom) + px(ps.borderTopWidth) + px(ps.borderBottomWidth);
    if (ps.position !== 'absolute' && ps.position !== 'fixed') {
      const w = ps.width === 'auto' ? own.w : px(ps.width) + ew, h = ps.height === 'auto' ? own.h : px(ps.height) + eh;
      return { left: own.x, top: own.y, width: w, height: h };
    }
    const base = cs.position !== 'static' ? own : containingBlock(el.parentElement || (el.getRootNode() && el.getRootNode().host) || null, ps.position === 'fixed');
    let w = ps.width === 'auto' ? null : px(ps.width) + ew, h = ps.height === 'auto' ? null : px(ps.height) + eh;
    let x, y;
    const l = ps.left === 'auto' ? null : px(ps.left), rt = ps.right === 'auto' ? null : px(ps.right);
    const t = ps.top === 'auto' ? null : px(ps.top), b = ps.bottom === 'auto' ? null : px(ps.bottom);
    if (w === null) w = base.w - (l || 0) - (rt || 0);
    if (h === null) h = base.h - (t || 0) - (b || 0);
    x = l !== null ? base.x + l : (rt !== null ? base.x + base.w - rt - w : base.x);
    y = t !== null ? base.y + t : (b !== null ? base.y + base.h - b - h : base.y);
    x += px(ps.marginLeft); y += px(ps.marginTop);
    const m = ps.transform && ps.transform !== 'none' && ps.transform.match(/matrix\(([^)]+)\)/);
    if (m) { const v = m[1].split(',').map(parseFloat); const sx = v[0], sy = v[3]; const cx = x + w / 2, cy = y + h / 2; w *= Math.abs(sx); h *= Math.abs(sy); x = cx - w / 2 + v[4]; y = cy - h / 2 + v[5]; }
    return { left: x, top: y, width: w, height: h };
  };
  // clip-path: inset(t r b l [round ...]) shrinks the painted box (e.g. md-slider's active track
  // is a full-width ::after clipped to the value); lengths may be calc() mixing % and px, so they are
  // resolved by a probe element sized like the reference box. Returns null when nothing is left.
  const probeHost = document.createElement('div');
  probeHost.style.cssText = 'position:absolute;left:-99999px;top:0;visibility:hidden;pointer-events:none';
  const resolveLen = (v, ref, vertical) => {
    if (/^-?[\d.]+px$/.test(v)) return parseFloat(v);
    if (/^-?[\d.]+%$/.test(v)) return parseFloat(v) / 100 * ref;
    if (!probeHost.isConnected) document.body.appendChild(probeHost);
    probeHost.style.width = ref + 'px'; probeHost.style.height = ref + 'px';
    const p = document.createElement('div'); p.style.position = 'absolute';
    if (vertical) p.style.height = v; else p.style.width = v;
    probeHost.appendChild(p); const rr = p.getBoundingClientRect(); probeHost.removeChild(p);
    return vertical ? rr.height : rr.width;
  };
  const clipInset = (b, cp) => {
    if (!cp || !cp.startsWith('inset(')) return b;
    const body = cp.slice(6, cp.lastIndexOf(')')).split(/\sround\s/)[0].trim();
    const toks = []; let depth = 0, cur = '';
    for (const ch of body) { if (ch === '(') depth++; if (ch === ')') depth--; if (/\s/.test(ch) && depth === 0) { if (cur) toks.push(cur); cur = ''; } else cur += ch; }
    if (cur) toks.push(cur);
    if (!toks.length) return b;
    const [t, rt, bt, l] = [toks[0], toks[1] ?? toks[0], toks[2] ?? toks[0], toks[3] ?? toks[1] ?? toks[0]];
    const it = resolveLen(t, b.height, true), ir = resolveLen(rt, b.width, false), ib = resolveLen(bt, b.height, true), il = resolveLen(l, b.width, false);
    const w = b.width - il - ir, h = b.height - it - ib;
    if (!(w > 0 && h > 0)) return null;
    return { left: b.left + il, top: b.top + it, width: w, height: h };
  };
  const emitPseudo = (el, cs, which, parent, clip, op, r, layer) => {
    const ps = getComputedStyle(el, '::' + which);
    if (!ps || ps.content === 'none' || ps.content === '' || ps.display === 'none' || ps.visibility === 'hidden') return;
    const pop = op * px(ps.opacity === '' ? '1' : ps.opacity);
    if (pop <= MIN_ALPHA) return;
    const pb = clipInset(pseudoBox(el, cs, ps, r), ps.clipPath);
    if (!pb) return;
    const box = boxOf(pb, clip);
    if (!big(box)) return;
    const p = paintOf(ps, pop, pb.width, pb.height);
    if (!painted(p)) return;
    emit({ type: 'box', parent, box, layer: layerOf(ps, '', layer), paint: p, meta: { src: 'pseudo', pseudo: which, tag: el.tagName.toLowerCase(), cls: typeof el.className === 'string' ? el.className.trim() : '' } });
  };

  // ---- md-elevation: attach its pseudo shadows to the nearest painted ancestor
  const attachElevation = (el, parent, op) => {
    const span = el.shadowRoot && el.shadowRoot.querySelector('.shadow'); if (!span) return;
    const r = el.getBoundingClientRect(); if (r.width < MIN || r.height < MIN) return;
    const sh = [];
    for (const which of ['before', 'after']) {
      const ps = getComputedStyle(span, '::' + which); if (!ps || ps.content === 'none') continue;
      sh.push(...parseShadows(ps.boxShadow, op * px(ps.opacity === '' ? '1' : ps.opacity)));
    }
    if (!sh.length) return;
    let i = parent;
    while (i >= 0 && !(out[i].paint && out[i].paint.fill) && !out[i].component) i = out[i].parent;
    if (i < 0) i = parent; if (i < 0) return;
    out[i].paint = out[i].paint || { fill: null, strokes: [], effects: [], radius: [0, 0, 0, 0], image: null };
    out[i].paint.effects = out[i].paint.effects.concat(sh.filter(s => s.blur > 0 || s.spread > 0 || s.dy || s.dx));
    (out[i].meta = out[i].meta || {}).elevation = true;
  };

  const slotName = n => n.nodeType === 1 ? (n.getAttribute('slot') || '') : '';
  const walkChildren = (host, parent, clip, op, layer) => {
    const root = host.shadowRoot || host;
    for (const c of root.childNodes) visit(c, parent, clip, op, host, layer);
  };
  // CSS paint order approximation: positioned boxes paint after in-flow content (layer 1 + z-index),
  // the <dialog> top layer paints last. Nodes carry `layer`; python sorts by (layer, pre-order).
  // Top layer (open <dialog>, :modal, :popover-open popovers such as a popover-positioned md-menu)
  // paints above everything else regardless of z-index.
  const TOP_LAYER = 1000;
  const inTopLayer = el => { if (!el || !el.matches) return false; try { return el.matches(':popover-open') || el.matches(':modal'); } catch (e) { return false; } };
  const layerOf = (cs, tag, layer, el) => {
    if (tag === 'dialog' || inTopLayer(el)) return Math.max(layer, TOP_LAYER);
    if (cs.position !== 'static') { const z = parseInt(cs.zIndex); return Math.max(layer, 1 + (isNaN(z) ? 0 : Math.max(0, z))); }
    return layer;
  };
  const visit = (node, parent, clip, op, styleParent, layer) => {
    if (node.nodeType === 3) { emitText(node, styleParent, parent, clip, op, layer); return; }
    if (node.nodeType !== 1) return;
    const el = node, tag = el.tagName.toLowerCase();
    if (SKIP.has(tag)) return;
    if (tag === 'slot') {
      const assigned = el.assignedNodes({ flatten: true });
      if (assigned.length) { for (const a of assigned) visit(a, parent, clip, op, el, layer); }
      else for (const c of el.childNodes) visit(c, parent, clip, op, el, layer);
      return;
    }
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    // a singular transform (scale(0): md-slider's hidden value label) paints nothing, pseudos included
    const tm = cs.transform && cs.transform.match(/^matrix\(([^)]+)\)/);
    if (tm) { const v = tm[1].split(',').map(parseFloat); if (Math.abs(v[0] * v[3] - v[1] * v[2]) < 1e-6) return; }
    layer = layerOf(cs, tag, layer, el);
    if (cs.clipPath && cs.clipPath.startsWith('inset(')) return;
    const eop = op * px(cs.opacity === '' ? '1' : cs.opacity);
    if (eop <= MIN_ALPHA) return;
    if (tag === 'md-elevation') { attachElevation(el, parent, eop); return; }
    const r = el.getBoundingClientRect();
    const box = boxOf(r, clip);
    const isHost = (tag.startsWith('md-') && el.getRootNode() === document && tag !== 'md-icon') || !!el.dataset.component;
    const data = {}; for (const k in el.dataset) data[k] = el.dataset[k];
    const attrs = {}; for (const a of el.attributes) attrs[a.name] = a.value;
    const base = { parent, box, layer, meta: { tag, cls: typeof el.className === 'string' ? el.className.trim() : '', shadow: el.getRootNode() !== document } };
    if (tag === 'md-icon') {
      const name = (el.textContent || '').trim();
      if (!big(box)) return;
      const c = parseColor(cs.color) || { r: 0, g: 0, b: 0, a: 1 };
      if (name) { const n = { ...base, type: 'icon', icon_name: name, color: { ...c, a: c.a * eop }, size: px(cs.fontSize) }; const ib = inkBox(el, cs); if (ib) n.ink_box = ib; emit(n); }
      else { const sv = el.querySelector('svg'); if (sv) { const fc = svgColor(sv, eop); emit({ ...base, type: 'vector', color: fc }); } }
      return;
    }
    if (tag === 'svg') {
      if (!big(box)) return;
      emit({ ...base, type: 'vector', color: svgColor(el, eop) });
      return;
    }
    if (tag === 'img') { if (big(box)) emit({ ...base, type: 'image', image_ref: el.getAttribute('src') || '' }); return; }
    if (tag === 'md-divider' || tag === 'hr') {
      if (!big(box)) return;
      let c = null;
      const ps = getComputedStyle(el, '::before');
      if (ps && ps.content !== 'none') c = parseColor(ps.backgroundColor);
      if (!c || c.a <= MIN_ALPHA) { const p = paintOf(cs, eop, r.width, r.height); c = p.fill || (p.strokes[0] && p.strokes[0].color); }
      const n = { ...base, type: 'line', color: c ? { ...c, a: c.a * eop } : null };
      if (isHost) n.component = { tag, attrs, data, info: { height: r.height } };
      emit(n);
      return;
    }
    const p = paintOf(cs, eop, r.width, r.height);
    const has = big(box);
    let myIdx = parent;
    const childClip = (cs.overflow !== 'visible' && cs.overflowX !== 'visible') && has ? inter(clip || VIEW, { x: r.left, y: r.top, w: r.width, h: r.height }) : clip;
    if (isHost) {
      const info = componentInfo(el, tag, r);
      const n = { ...base, type: 'box', paint: painted(p) ? p : null, component: { tag, attrs, data, info }, fit: !has };
      if (cs.opacity !== '1' && cs.opacity !== '') n.opacity = px(cs.opacity);
      myIdx = emit(n);
    } else if (has && painted(p)) {
      const n = { ...base, type: 'box', paint: p };
      myIdx = emit(n);
    }
    emitPseudo(el, cs, 'before', myIdx, childClip, eop, r, layer);  // a 0px element can still paint pseudos (md-filled-field's active indicator)
    if (tag === 'input' || tag === 'textarea') { if (has && (tag === 'textarea' || TEXT_INPUTS.has((el.getAttribute('type') || 'text').toLowerCase()))) emitInput(el, cs, myIdx, childClip, eop, layer); }
    else walkChildren(el, myIdx, childClip, eop, layer);
    emitPseudo(el, cs, 'after', myIdx, childClip, eop, r, layer);
  };
  // ---- colour science for the svg backdrop test (sRGB -> CIE Lab, CIE76 ΔE)
  const lab = c => {
    const lin = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    const r = lin(c.r), g = lin(c.g), b = lin(c.b);
    const f = t => t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116;
    const x = f((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047), y = f(0.2126 * r + 0.7152 * g + 0.0722 * b), z = f((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
    return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
  };
  const dE = (a, b) => { const p = lab(a), q = lab(b); return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); };
  const parentOf = e => e.parentElement || ((e.getRootNode() && e.getRootNode().host) || null);
  const backdropOf = el => {
    // the opaque background painted behind the element: elements under its centre first
    // (siblings such as a checkbox's container), then its ancestors, then the page
    const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    let list = [];
    try { const root = el.getRootNode(); list = (root.elementsFromPoint ? root.elementsFromPoint(cx, cy) : document.elementsFromPoint(cx, cy)); } catch (e) { list = []; }
    for (const e of list) {
      if (e === el || el.contains(e)) continue;
      const c = parseColor(getComputedStyle(e).backgroundColor);
      if (c && c.a > 0.5) return c;
    }
    for (let p = parentOf(el); p; p = parentOf(p)) { const c = parseColor(getComputedStyle(p).backgroundColor); if (c && c.a > 0.5) return c; }
    return pageBg();
  };
  const SVG_SKIP = 'mask, clipPath, defs, pattern, marker, symbol';
  const svgColor = (sv, op) => {
    // colour of the first visible shape whose fill (else stroke) differs from the backdrop;
    // shapes inside mask/clipPath/defs never paint. Never default to white: fall back to the
    // svg's own `color` only when it differs from the backdrop, else no colour.
    const back = backdropOf(sv);
    for (const e of sv.querySelectorAll('path, circle, rect, ellipse, polygon, polyline, line')) {
      if (e.closest(SVG_SKIP)) continue;
      const s = getComputedStyle(e);
      if (s.display === 'none' || s.visibility === 'hidden' || px(s.opacity === '' ? '1' : s.opacity) <= MIN_ALPHA) continue;
      const f = parseColor(s.fill), st = parseColor(s.stroke);
      const cands = [];
      if (s.fill !== 'none' && f && f.a * px(s.fillOpacity || '1') * op > MIN_ALPHA) cands.push(f);
      if (s.stroke !== 'none' && st && st.a * op > MIN_ALPHA && px(s.strokeWidth) > 0) cands.push(st);
      for (const c of cands) if (dE(c, back) > BACKDROP_DE) return { ...c, a: c.a * op };
    }
    const c = parseColor(getComputedStyle(sv).color);
    return c && dE(c, back) > BACKDROP_DE ? { ...c, a: c.a * op } : null;
  };
  // ---- md-icon glyph ink bounds: canvas metrics of the ligature around the text baseline
  const inkBox = (el, cs) => {
    const tn = Array.from(el.childNodes).find(n => n.nodeType === 3 && n.data.trim());
    if (!tn) return null;
    const lead = tn.data.length - tn.data.replace(/^\s+/, '').length, word = tn.data.trim();
    const range = document.createRange(); range.setStart(tn, lead); range.setEnd(tn, lead + word.length);
    const rr = range.getBoundingClientRect();
    if (!(rr.width > 0 && rr.height > 0)) return null;
    canvas.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    canvas.letterSpacing = '0px';
    const m = canvas.measureText(word);
    if (!(m.width > 0) || m.width > 1.5 * px(cs.fontSize)) return null;  // ligature not shaped
    const asc = m.fontBoundingBoxAscent, desc = m.fontBoundingBoxDescent;
    const baseline = rr.top + (rr.height - (asc + desc)) / 2 + asc;
    const x1 = rr.left - m.actualBoundingBoxLeft, x2 = rr.left + m.actualBoundingBoxRight;
    const y1 = baseline - m.actualBoundingBoxAscent, y2 = baseline + m.actualBoundingBoxDescent;
    if (!(x2 > x1 && y2 > y1)) return null;
    return { x: R(x1), y: R(y1), w: R(x2 - x1), h: R(y2 - y1) };
  };
  const componentInfo = (el, tag, r) => {
    const info = { height: R(r.height), width: R(r.width), child_tags: {} };
    for (const c of el.children) { const s = c.getAttribute('slot') || ''; (info.child_tags[s] = info.child_tags[s] || []).push(c.tagName.toLowerCase()); }
    const slotText = name => { const parts = []; for (const c of el.children) if ((c.getAttribute('slot') || '') === name && c.tagName.toLowerCase() !== 'md-icon') parts.push((c.textContent || '').trim()); return parts.filter(Boolean).join(' '); };
    const own = Array.from(el.childNodes).filter(n => n.nodeType === 3).map(n => n.data.trim()).filter(Boolean).join(' ');
    const ic = el.querySelector(':scope > md-icon'); if (ic) info.icon = (ic.textContent || '').trim();
    const label = el.getAttribute('label') || slotText('headline') || slotText('') || own;
    if (label) info.label = label;
    if (el.getAttribute('value') !== null && el.getAttribute('value') !== '') info.value = el.getAttribute('value');
    if (el.getAttribute('placeholder')) info.placeholder = el.getAttribute('placeholder');
    const sup = slotText('supporting-text'); if (sup) info.supporting_text = sup;
    const tr = slotText('trailing-supporting-text'); if (tr) info.trailing_text = tr;
    const hl = slotText('headline'); if (hl) info.headline = hl;
    if (tag.endsWith('-select')) { const sel = el.querySelector('md-select-option[selected]'); if (sel) info.value = (sel.textContent || '').trim(); }
    if (tag === 'md-dialog') { const h = slotText('headline'); if (h) info.label = h; }
    return info;
  };

  // disable remaining animations before measuring
  try { for (const a of document.getAnimations()) { try { a.finish(); } catch (e) { a.cancel(); } } } catch (e) {}
  function pageBg() {
    let c = parseColor(getComputedStyle(document.body).backgroundColor);
    if (!c || c.a <= MIN_ALPHA) c = parseColor(getComputedStyle(document.documentElement).backgroundColor);
    if (!c || c.a <= MIN_ALPHA) c = { r: 255, g: 255, b: 255, a: 1 };
    return c;
  }
  const bg = pageBg();
  walkChildren(document.body, -1, null, 1, 0);
  return { width: VW, height: VH, bg, nodes: out, ready: !!window.__mwcReady };
})()
"""


def extract_script() -> str:
    """The ground-truth extraction JS with the current ``dt.params`` thresholds substituted."""
    return (EXTRACT_JS.replace("__MIN_SIZE__", repr(float(P["selftest.mwc.min_size"])))
            .replace("__MIN_ALPHA__", repr(float(P["selftest.mwc.min_alpha"])))
            .replace("__BACKDROP_DE__", repr(float(P["selftest.mwc.backdrop_de"])))
            .replace("__SETTLE_ROUNDS__", str(int(P["selftest.mwc.settle_rounds"]))))


# --------------------------------------------------------------------------- JS result -> Document
def _color(d: Optional[dict]) -> Optional[Color]:
    if not d:
        return None
    return Color(int(round(d["r"])), int(round(d["g"])), int(round(d["b"])), float(d.get("a", 1.0)))


def _paint_to_node(n: Node, p: Optional[dict]) -> None:
    if not p:
        return
    if p.get("fill"):
        n.fills.append(Fill.solid(_color(p["fill"])))
    if p.get("image"):
        n.fills.append(Fill(kind="image", image_ref=p["image"]))
    for s in p.get("strokes", []):
        n.strokes.append(Stroke(color=_color(s["color"]), width=float(s["width"]), align="inside", sides=tuple(s["sides"]) if s.get("sides") else None))
    for e in p.get("effects", []):
        n.effects.append(Shadow(color=_color(e["color"]), dx=e["dx"], dy=e["dy"], blur=e["blur"], spread=e["spread"], inner=bool(e.get("inner"))))
    n.radius = tuple(float(v) for v in p.get("radius", [0, 0, 0, 0]))
    if p.get("gradient"):
        n.meta["gradient"] = p["gradient"]


def _text_style(d: dict) -> TextStyle:
    dec = d.get("decoration", "none")
    tr = d.get("transform", "none")
    return TextStyle(
        family=d.get("family") or "Roboto", size=float(d["size"]), weight=int(d["weight"]),
        line_height=float(d["line_height"]) if d.get("line_height") else None,
        letter_spacing=float(d.get("letter_spacing", 0)), color=_color(d["color"]) or Color(),
        align=d.get("align", "left"), italic=bool(d.get("italic")),
        decoration=dec if dec in ("none", "underline", "line-through") else "none",
        transform=tr if tr in ("none", "uppercase", "lowercase", "capitalize") else "none",
    )


def _raw_to_node(raw: dict, idx: int) -> Node:
    """Convert one flat JS record into a ``Node`` (children attached later)."""
    b = raw["box"]
    n = Node(id=f"n{idx}", box=Box(b["x"], b["y"], b["w"], b["h"]), meta=dict(raw.get("meta", {})))
    if raw.get("layer"):
        n.meta["z"] = int(raw["layer"])
    t = raw["type"]
    if t == "text":
        n.type = "text"
        n.text = raw["text"]
        n.text_style = _text_style(raw["text_style"])
        n.name = f"text:{raw['text'][:24]}"
    elif t == "icon":
        n.type = "icon"
        n.icon_name = raw["icon_name"]
        n.fills = [Fill.solid(_color(raw["color"]))]
        n.name = f"icon:{raw['icon_name']}"
        n.meta["icon_size"] = raw.get("size")
        ib = raw.get("ink_box")
        if ib:
            n.meta["ink_box"] = [float(ib["x"]), float(ib["y"]), float(ib["w"]), float(ib["h"])]
    elif t == "vector":
        n.type = "vector"
        if raw.get("color"):
            n.fills = [Fill.solid(_color(raw["color"]))]
        n.name = "vector"
    elif t == "image":
        n.type = "image"
        n.image_ref = raw.get("image_ref")
        n.name = "image"
    elif t == "line":
        n.type = "line"
        if raw.get("color"):
            n.fills = [Fill.solid(_color(raw["color"]))]
        n.name = "divider"
    else:
        n.type = "frame"
        _paint_to_node(n, raw.get("paint"))
        n.name = raw.get("meta", {}).get("tag", "box")
        if raw.get("opacity") is not None:
            n.opacity = float(raw["opacity"])
    comp = raw.get("component")
    if comp:
        info = dict(comp.get("info", {}))
        info["data"] = comp.get("data", {})
        ref = component_for(comp["tag"], comp.get("attrs", {}), info)
        if ref is not None:
            n.component = ref
            n.name = ref.name + "".join(f"/{v}" for v in ref.variant.values())
            if n.type == "frame":
                n.clip = False
        elif part_for(comp["tag"]):
            n.meta["part"] = part_for(comp["tag"])
            n.name = n.meta["part"]
            if n.type == "frame":
                n.clip = False
    return n


def _move_dialog_component(root: Node) -> None:
    """``md-dialog``'s host is the full-viewport layer (scrim + surface); the Dialog *component* is
    its rounded surface container. Move the ComponentRef there and nest the dialog content
    (icon, headline, body, actions) under it; the host becomes ``meta.part = 'DialogHost'``."""
    opaque = 0.99
    for host in list(root.walk()):
        if host.component is None or host.component.name != "Dialog":
            continue
        surfaces = [c for c in host.children if c.type in ("frame", "rect") and c.component is None
                    and c.fill_color is not None and c.fill_color.a * c.opacity >= opaque and c.box.area < 0.9 * host.box.area]
        if not surfaces:
            continue
        surf = max(surfaces, key=lambda c: c.box.area)
        tol = float(P["selftest.mwc.contain_tol"])
        i = host.children.index(surf)
        inside = [c for c in host.children[i + 1:] if surf.box.contains(c.box, tol)]
        host.children = [c for c in host.children if c not in inside]
        surf.children.extend(inside)
        surf.type = "frame"
        surf.component, host.component = host.component, None
        surf.name = "Dialog"
        surf.clip = False
        surf.meta["component_moved_from"] = host.id
        host.meta["part"] = "DialogHost"
        host.name = "DialogHost"


def _lift_paint(root: Node) -> None:
    """Lift the paint of a covering child onto an unpainted md-* host (see module docstring).

    The first child (paint order) that is a box without its own component, covers at least
    ``selftest.mwc.lift_cover`` of the host and paints an opaque fill or a stroke gives the host
    its fill/stroke/radius/shadows; a later covering stroke-only child (outline drawn separately
    from the background) contributes its stroke. Lifted children are removed, their own children
    are re-parented to the host in place, and ``meta.lifted_from`` lists what was lifted.
    """
    cover = float(P["selftest.mwc.lift_cover"])
    opaque = 0.99
    for host in list(root.walk()):
        if host is root or host.type != "frame" or not str(host.meta.get("tag", "")).startswith("md-"):
            continue
        if host.component is None and not host.meta.get("part"):
            continue
        if host.fills or host.strokes or host.box.area <= 0:
            continue
        cands = []
        for c in host.children:
            if c.type not in ("frame", "rect", "ellipse") or c.component is not None or c.meta.get("part"):
                continue
            fill_ok = bool(c.fills) and all(f.color is None or f.color.a * c.opacity >= opaque for f in c.fills)
            if not (fill_ok or (c.strokes and not c.fills)):
                continue
            if c.box.intersect(host.box).area >= cover * host.box.area:
                # a paint in a higher stacking layer than other content of the host (an outlined
                # field's outline over its leading icon) cannot become the host's own paint: the
                # host paints before all its children, so the layers could not be kept apart
                cz = int(c.meta.get("z", 0))
                own = {id(x) for x in c.walk()}
                if cz > int(host.meta.get("z", 0)) and any(int(o.meta.get("z", 0)) < cz for o in host.walk()
                                                           if o is not host and id(o) not in own):
                    continue
                cands.append(c)
        if not cands:
            continue
        first = cands[0]
        lifted = [first]
        host.fills = list(first.fills)
        host.strokes = list(first.strokes)
        host.radius = first.radius
        if not host.strokes:
            extra = next((c for c in cands[1:] if c.strokes and not c.fills), None)
            if extra is not None:
                host.strokes = list(extra.strokes)
                if not any(host.radius):
                    host.radius = extra.radius
                lifted.append(extra)
        for c in lifted:
            host.effects = list(host.effects) + [e for e in c.effects if e not in host.effects]
            # the paint keeps its stacking layer (a menu surface paints above the page content)
            if int(c.meta.get("z", 0)) > int(host.meta.get("z", 0)):
                host.meta["z"] = int(c.meta["z"])
        kids: list[Node] = []
        for c in host.children:
            if c in lifted:
                kids.extend(c.children)
            else:
                kids.append(c)
        host.children = kids
        host.meta["lifted_from"] = [{"id": c.id, "tag": c.meta.get("tag"), "cls": c.meta.get("cls", ""),
                                     **({"pseudo": c.meta["pseudo"]} if c.meta.get("pseudo") else {})} for c in lifted]


def _dom_index(n: Node) -> int:
    """Emission (DOM paint pre-order) index of an extracted node (ids are ``n<index>``)."""
    try:
        return int(n.id[1:])
    except ValueError:
        return 0


def _paint_key(n: Node) -> tuple[int, int]:
    """CSS paint order recorded by the extractor: (stacking layer ``meta.z``, DOM pre-order)."""
    return int(n.meta.get("z", 0)), _dom_index(n)


def _paints(n: Node) -> bool:
    return bool(n.fills or n.strokes or n.effects) or n.type in ("text", "icon", "vector", "image", "line")


def _restack(root: Node) -> None:
    """Make tree pre-order (the order every renderer paints in) follow the CSS paint order
    :func:`_paint_key` wherever two painted nodes overlap.

    1. Every node's children are stably sorted by the lowest painted layer in their subtree, so
       positioned boxes paint after in-flow siblings and an open md-menu / popover / dialog
       (layer 21 / 1000) after the content that follows it in the DOM.
    2. A tree cannot express every CSS order (a host whose outline paints in layer 2 above a
       later snackbar while its icon in layer 1 stays under it). For each remaining conflict --
       overlapping painted nodes, neither containing the other, drawn in the wrong order -- the
       node that must paint later is re-parented to the root (``meta.reparented_from``) and the
       root's children re-sorted by (lowest subtree layer, DOM order). Each node moves at most
       once, so this terminates; hierarchy only changes where paint order demands it.
    """
    minz: dict[int, int] = {}

    def subtree_min(n: Node) -> int:
        # lowest layer that actually paints in the subtree: an unpainted wrapper (md-dialog's
        # host around its top-layer scrim + surface) has no paint position of its own
        own = int(n.meta.get("z", 0)) if n is not root else 0
        kids = [subtree_min(c) for c in n.children]
        m = min(([own] if _paints(n) or not kids else []) + kids)
        minz[id(n)] = m
        return m

    def sort(n: Node) -> None:
        n.children.sort(key=lambda c: minz[id(c)])  # stable: DOM order within a layer
        for c in n.children:
            sort(c)

    subtree_min(root)
    sort(root)
    moved: set[int] = set()
    min_overlap = float(P["selftest.mwc.restack_overlap"])
    while True:
        order = list(root.walk())
        pos = {id(n): i for i, n in enumerate(order)}
        end = {id(n): pos[id(n)] + n.count() for n in order}
        painted = [n for n in order if n is not root and _paints(n) and n.box.area > 0]
        culprit = None
        for a in painted:  # a is drawn before b ...
            for b in painted:
                if pos[id(b)] <= pos[id(a)] or pos[id(b)] < end[id(a)]:
                    continue  # b not after a, or b inside a's subtree
                if _paint_key(a) > _paint_key(b) and a.box.intersect(b.box).area > min_overlap and id(a) not in moved:
                    culprit = a  # ... but must paint above b
                    break
            if culprit is not None:
                break
        if culprit is None:
            return
        parent = next(p for p in order if culprit in p.children)
        parent.children.remove(culprit)
        culprit.meta.setdefault("reparented_from", parent.id)
        moved.add(id(culprit))
        root.children.append(culprit)
        subtree_min(root)
        root.children.sort(key=lambda c: (minz[id(c)], _dom_index(c)))


def _flag_occluded(root: Node) -> None:
    """Flag nodes covered by boxes painted later (paint order == pre-order).

    * ``meta.occluded=True`` -- fully inside a later, non-descendant, opaque solid rect/frame
      (menus, snackbars, dialog containers hide what is underneath).
    * ``meta.overlay_alpha=a`` -- fully inside a later translucent box (e.g. the dialog scrim),
      so its rendered colours are blended; ``a`` is the strongest such alpha.

    "Later" means a larger ``(meta.z stacking layer, pre-order index)`` key: positioned boxes
    (menus, snackbars, FABs) and the dialog top layer paint above in-flow content.
    """
    order = list(root.walk())
    index = {id(n): i for i, n in enumerate(order)}

    def key(n: Node) -> tuple[int, int]:
        return _paint_key(n)

    covers: list[tuple[tuple[int, int], int, int, Box, float]] = []
    for i, n in enumerate(order):
        c = n.fill_color
        if n is not root and n.type in ("frame", "rect", "ellipse") and c is not None and min(n.radius) < min(n.box.w, n.box.h) / 2:
            a = c.a * n.opacity
            if a > 0:
                covers.append((key(n), i, i + n.count(), n.box, a))
    for n in order:
        if n is root:
            continue
        i, k = index[id(n)], key(n)
        subtree_end = i + n.count()
        alphas = [a for ck, j, jend, b, a in covers if ck > k and not (i <= j < subtree_end) and not (j <= i < jend) and b.contains(n.box)]
        if any(a >= 0.99 for a in alphas):
            n.meta["occluded"] = True
        elif alphas:
            n.meta["overlay_alpha"] = round(max(alphas), 3)


def build_document(result: dict, width: int, height: int) -> Document:
    """Nest the flat JS node list into a ``Document`` (DOM ancestry, containment-repaired)."""
    raws: list[dict] = result["nodes"]
    bg = _color(result.get("bg")) or Color(255, 255, 255)
    doc = Document.blank(width, height, bg)
    root = doc.root
    nodes: list[Node] = [_raw_to_node(r, i + 1) for i, r in enumerate(raws)]
    parents: list[int] = [int(r.get("parent", -1)) for r in raws]
    # resolve "fit" hosts (zero-size host boxes) from the union of their descendants
    desc_union: dict[int, Optional[Box]] = {}
    for i in range(len(nodes) - 1, -1, -1):
        p = parents[i]
        if p >= 0:
            u = desc_union.get(p)
            mine = nodes[i].box if not raws[i].get("fit") else desc_union.get(i)
            if mine is not None and mine.area > 0:
                desc_union[p] = mine if u is None else u.union(mine)
    alive = [True] * len(nodes)
    for i, r in enumerate(raws):
        if r.get("fit"):
            u = desc_union.get(i)
            if u is None:
                alive[i] = False
            else:
                nodes[i].box = u
                nodes[i].meta["fit_to_children"] = True
    tol = float(P["selftest.mwc.contain_tol"])

    def resolve_parent(i: int) -> Node:
        p = parents[i]
        while p >= 0 and not alive[p]:
            p = parents[p]
        cand = nodes[p] if p >= 0 else root
        orig = cand
        while cand is not root and not cand.box.contains(nodes[i].box, tol):
            pi = node_index[cand.id]
            pp = parents[pi]
            while pp >= 0 and not alive[pp]:
                pp = parents[pp]
            cand = nodes[pp] if pp >= 0 else root
        if cand is not orig:
            nodes[i].meta["reparented_from"] = orig.id
        return cand

    node_index = {n.id: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        if not alive[i]:
            continue
        n.box = n.box.intersect(root.box) if not root.box.contains(n.box, tol) else n.box
        if n.box.area <= 0:
            alive[i] = False
            continue
        resolve_parent(i).children.append(n)
    _move_dialog_component(root)
    _lift_paint(root)
    # leaf painted boxes become rects/ellipses; containers stay frames
    for n in root.walk():
        if n is root or n.type != "frame" or n.component is not None or n.meta.get("part"):
            continue
        if not n.children:
            r = min(n.radius)
            if n.box.w > 0 and abs(n.box.w - n.box.h) < 1 and r >= n.box.w / 2 - 0.5:
                n.type = "ellipse"
            else:
                n.type = "rect"
    _restack(root)
    _flag_occluded(root)
    doc.fonts = sorted({t.text_style.family for t in doc.texts() if t.text_style} | {"Material Symbols Outlined"})
    doc.design_system = "material3"
    return doc


def extract_ground_truth(html_path: str, width: int, height: int, png_path: Optional[str] = None) -> Document:
    """Render ``html_path`` (file path) at ``width``x``height`` and return the DOM-derived IR."""
    url = "file://" + os.path.abspath(html_path)
    _rgb, res = render_url(url, width, height, out_path=png_path, wait_ms=int(P["selftest.mwc.wait_ms"]), script=extract_script())
    assert isinstance(res, dict), "extraction script returned no result"
    if not res.get("ready"):
        raise RuntimeError("Material Web bundle did not load (window.__mwcReady unset)")
    doc = build_document(res, width, height)
    if png_path:
        doc.source_image = os.path.abspath(png_path)
    return doc


# --------------------------------------------------------------------------- page grammar
class _Page:
    """Accumulates HTML for one screen; all randomness goes through ``rng``."""

    def __init__(self, rng: random.Random, width: int, height: int):
        self.rng = rng
        self.w = width
        self.h = height
        self.parts: list[str] = []
        self.overlays: list[str] = []
        self.menu_id = 0

    # ---- helpers
    def icon(self, name: str, slot: Optional[str] = None) -> str:
        s = f' slot="{slot}"' if slot else ""
        return f"<md-icon{s}>{name}</md-icon>"

    def button(self, label: Optional[str] = None, kind: Optional[str] = None, icon: bool = False) -> str:
        rng = self.rng
        kind = kind or rng.choice(["filled", "outlined", "text", "elevated", "filled-tonal"])
        label = label or rng.choice(G.ACTIONS)
        ic = self.icon(rng.choice(G.ACTION_ICONS), "icon") if icon else ""
        return f"<md-{kind}-button>{ic}{label}</md-{kind}-button>"

    def icon_button(self, name: Optional[str] = None, kind: str = "") -> str:
        name = name or self.rng.choice(G.ACTION_ICONS)
        tag = {"": "md-icon-button", "filled": "md-filled-icon-button", "tonal": "md-filled-tonal-icon-button", "outlined": "md-outlined-icon-button"}[kind]
        return f"<{tag}>{self.icon(name)}</{tag}>"

    # ---- blocks
    def app_bar(self) -> str:
        rng = self.rng
        kind = rng.choice(["small", "center"])
        title = rng.choice(G.APP_TITLES + G.SECTION_TITLES)
        tone = " container" if rng.random() < 0.4 else ""
        actions = "".join(self.icon_button(i) for i in G.pick(rng, G.ACTION_ICONS, rng.randint(1, 3)))
        badge = ""
        if rng.random() < 0.5:
            cnt = rng.randint(1, 99)
            badge = f'<span class="badge-wrap">{self.icon_button("notifications")}<span class="badge" data-component="Badge" data-variant-size="large">{cnt}</span></span>'
        nav = self.icon_button(rng.choice(["menu", "arrow_back"]))
        return (f'<div class="app-bar {kind}{tone}" data-component="TopAppBar" data-variant-size="{kind}">{nav}'
                f'<div class="title">{title}</div>{actions}{badge}</div>')

    def tabs(self) -> str:
        rng = self.rng
        labels = rng.choice(G.TABS)
        active = rng.randrange(len(labels))
        kind = rng.choice(["primary", "primary", "secondary"])
        items = []
        with_icons = kind == "primary" and rng.random() < 0.4
        for lab in labels:
            ic = self.icon(rng.choice(G.LEADING_ICONS), "icon") if with_icons else ""
            items.append(f'<md-{kind}-tab>{ic}{lab}</md-{kind}-tab>')
        # the active tab is selected after upgrade (see page_html) so exactly one tab is active
        return f'<md-tabs data-active="{active}">' + "".join(items) + "</md-tabs>"

    def list_block(self, kind: Optional[str] = None) -> str:
        rng = self.rng
        kind = kind or rng.choice(["mail", "files", "settings", "events", "people"])
        n = rng.randint(3, 6)
        items = []
        for i in range(n):
            if kind == "mail":
                hl, sup, tr = rng.choice(G.PEOPLE), rng.choice(G.SUBJECTS), rng.choice(G.TIMES)
                lead = f'<div slot="start" class="avatar">{hl[0]}</div>' if rng.random() < 0.5 else self.icon(rng.choice(G.LEADING_ICONS), "start")
                two = rng.random() < 0.35
                sup2 = (" " + rng.choice(G.SNIPPETS)) if two else ""
                items.append(f'<md-list-item>{lead}<div slot="headline">{hl}</div><div slot="supporting-text">{sup}{sup2}</div><div slot="trailing-supporting-text">{tr}</div></md-list-item>')
            elif kind == "files":
                hl = rng.choice(G.FILES)
                items.append(f'<md-list-item>{self.icon(rng.choice(["folder", "description", "image", "table_chart", "slideshow"]), "start")}<div slot="headline">{hl}</div><div slot="supporting-text">Modified {rng.choice(G.TIMES)}</div>{self.icon("more_vert", "end")}</md-list-item>')
            elif kind == "settings":
                hl = rng.choice(G.SETTINGS)
                ctl = rng.choice(["switch", "checkbox", "radio"])
                on = rng.random() < 0.5
                trail = {"switch": f'<md-switch slot="end"{" selected" if on else ""}></md-switch>', "checkbox": f'<md-checkbox slot="end"{" checked" if on else ""}></md-checkbox>', "radio": f'<md-radio slot="end" name="g{i}"{" checked" if on else ""}></md-radio>'}[ctl]
                sup = f'<div slot="supporting-text">{rng.choice(G.SNIPPETS)}</div>' if rng.random() < 0.5 else ""
                items.append(f'<md-list-item>{self.icon(rng.choice(G.LEADING_ICONS), "start") if rng.random() < 0.5 else ""}<div slot="headline">{hl}</div>{sup}{trail}</md-list-item>')
            elif kind == "events":
                items.append(f'<md-list-item>{self.icon("event", "start")}<div slot="headline">{rng.choice(G.EVENTS)}</div><div slot="supporting-text">{rng.choice(G.TIMES)}</div></md-list-item>')
            else:
                hl = rng.choice(G.PEOPLE)
                items.append(f'<md-list-item><div slot="start" class="avatar">{hl[0]}</div><div slot="headline">{hl}</div><div slot="supporting-text">{hl.split()[0].lower()}@example.com</div></md-list-item>')
            if rng.random() < 0.4 and i < n - 1:
                items.append("<md-divider inset></md-divider>" if rng.random() < 0.5 else "<md-divider></md-divider>")
        title = f'<div class="section-title">{rng.choice(G.SECTION_TITLES)}</div>' if rng.random() < 0.6 else ""
        return f'{title}<md-list>{"".join(items)}</md-list>'

    def card_grid(self) -> str:
        rng = self.rng
        cols = 1 if self.w < 600 else (2 if self.w < 1000 else 3)
        n = rng.randint(cols, cols * 2)
        cards = []
        for _ in range(n):
            kind = rng.choice(["elevated", "filled", "outlined"])
            media = '<div class="media"></div>' if rng.random() < 0.3 else ""
            acts = ""
            if rng.random() < 0.6:
                acts = '<div class="actions">' + self.button(kind="text") + self.button(kind=rng.choice(["filled", "filled-tonal", "outlined"])) + "</div>"
            cards.append(f'<div class="card {kind}" data-component="Card" data-variant-style="{kind}">{media}<div class="headline">{rng.choice(G.HEADLINES + G.SUBJECTS)}</div><div class="supporting">{rng.choice(G.SNIPPETS)}</div>{acts}</div>')
        return f'<div class="grid" style="grid-template-columns: repeat({cols}, 1fr)">{"".join(cards)}</div>'

    def form(self) -> str:
        rng = self.rng
        n = rng.randint(2, 4)
        fields = []
        idx = G.pick(rng, list(range(len(G.FIELD_LABELS))), n)
        for i in idx:
            kind = rng.choice(["outlined", "filled"])
            val = G.FIELD_VALUES[i]
            v = f' value="{val}"' if val and rng.random() < 0.7 else ""
            ph = ' placeholder="Type here"' if not v and rng.random() < 0.3 else ""
            lead = self.icon(rng.choice(["search", "person", "mail", "lock", "place"]), "leading-icon") if rng.random() < 0.3 else ""
            sup = f' supporting-text="{rng.choice(["Required", "Optional", "We will never share this"])}"' if rng.random() < 0.3 else ""
            fields.append(f'<md-{kind}-text-field label="{G.FIELD_LABELS[i]}"{v}{ph}{sup}>{lead}</md-{kind}-text-field>')
        if rng.random() < 0.5:
            lab, opts = rng.choice(G.SELECT_LABELS)
            kind = rng.choice(["outlined", "filled"])
            sel = rng.randrange(len(opts))
            fields.append(f'<md-{kind}-select label="{lab}">' + "".join(f'<md-select-option{" selected" if j == sel else ""} value="{j}"><div slot="headline">{o}</div></md-select-option>' for j, o in enumerate(opts)) + f"</md-{kind}-select>")
        if rng.random() < 0.6:
            labeled = " labeled" if rng.random() < 0.5 else ""
            ticks = " ticks step=10" if rng.random() < 0.3 else ""
            rng_ = rng.random() < 0.3
            if rng_:
                fields.append(f'<md-slider range value-start="{rng.randint(10, 40)}" value-end="{rng.randint(50, 90)}"{labeled}{ticks}></md-slider>')
            else:
                fields.append(f'<md-slider value="{rng.randint(5, 95)}"{labeled}{ticks}></md-slider>')
        btns = '<div class="row end">' + self.button(kind="text", label="Cancel") + self.button(kind="filled", label=rng.choice(["Save", "Send", "Create", "Done"])) + "</div>"
        return f'<div class="form">{"".join(fields)}</div>{btns}'

    def chip_row(self) -> str:
        rng = self.rng
        kinds = ["assist", "filter", "input", "suggestion"]
        kind = rng.choice(kinds)
        labels = G.pick(rng, G.CHIP_LABELS, rng.randint(3, 5))
        chips = []
        for lab in labels:
            sel = " selected" if kind == "filter" and rng.random() < 0.4 else ""
            ic = self.icon(rng.choice(G.LEADING_ICONS), "icon") if rng.random() < 0.3 else ""
            el = " elevated" if kind == "assist" and rng.random() < 0.3 else ""
            chips.append(f'<md-{kind}-chip label="{lab}"{sel}{el}>{ic}</md-{kind}-chip>')
        return f'<div class="row"><md-chip-set>{"".join(chips)}</md-chip-set></div>'

    def button_row(self) -> str:
        rng = self.rng
        n = rng.randint(2, 4)
        kinds = G.pick(rng, ["filled", "outlined", "text", "elevated", "filled-tonal"], n)
        btns = "".join(self.button(kind=k, icon=rng.random() < 0.3) for k in kinds)
        if rng.random() < 0.4:
            btns += "".join(self.icon_button(kind=k) for k in G.pick(rng, ["", "filled", "tonal", "outlined"], rng.randint(1, 3)))
        return f'<div class="row">{btns}</div>'

    def headline_block(self) -> str:
        rng = self.rng
        return f'<div class="headline-text">{rng.choice(G.HEADLINES)}</div><div class="body-text">{rng.choice(G.SNIPPETS)}</div>'

    def progress_block(self) -> str:
        rng = self.rng
        parts = []
        if rng.random() < 0.7:
            v = rng.choice([0.2, 0.35, 0.5, 0.65, 0.8])
            buf = f' buffer="{min(1.0, v + 0.2)}"' if rng.random() < 0.4 else ""
            parts.append(f'<div class="progress-row"><md-linear-progress value="{v}"{buf} style="flex:1"></md-linear-progress></div>')
        if rng.random() < 0.6:
            parts.append(f'<div class="progress-row"><md-circular-progress value="{rng.choice([0.25, 0.5, 0.75])}"></md-circular-progress><div class="body-text" style="padding:0">{rng.choice(["Uploading 3 files", "Syncing", "Loading"])}</div></div>')
        return "".join(parts)

    def choice_block(self) -> str:
        rng = self.rng
        rows = []
        kind = rng.choice(["checkbox", "radio"])
        for i, lab in enumerate(G.pick(rng, G.SETTINGS, rng.randint(2, 4))):
            on = (i == 0) if kind == "radio" else rng.random() < 0.5
            ctl = f'<md-checkbox{" checked" if on else ""}></md-checkbox>' if kind == "checkbox" else f'<md-radio name="grp"{" checked" if on else ""}></md-radio>'
            rows.append(f'<label class="row" style="gap:16px">{ctl}<span class="body-text" style="padding:0;color:var(--md-sys-color-on-surface)">{lab}</span></label>')
        return "".join(rows)

    def menu_block(self) -> str:
        rng = self.rng
        self.menu_id += 1
        mid = f"menu{self.menu_id}"
        items = []
        for lab in G.pick(rng, G.MENU_ITEMS, rng.randint(3, 5)):
            ic = self.icon(rng.choice(G.ACTION_ICONS), "start") if rng.random() < 0.5 else ""
            items.append(f'<md-menu-item>{ic}<div slot="headline">{lab}</div></md-menu-item>')
        anchor = f'<span class="menu-anchor"><md-outlined-button id="{mid}-a">{self.icon("arrow_drop_down", "icon")}{rng.choice(["Sort by", "More", "Options"])}</md-outlined-button><md-menu id="{mid}" anchor="{mid}-a" open quick>{"".join(items)}</md-menu></span>'
        return f'<div class="row">{anchor}</div>'

    def nav_bar(self) -> str:
        rng = self.rng
        dests = G.pick(rng, G.NAV_DESTS, rng.randint(3, 5))
        active = rng.randrange(len(dests))
        items = []
        for i, (ic, lab) in enumerate(dests):
            badge = (f'<span class="badge" data-component="Badge" data-variant-size="large">{rng.randint(1, 20)}</span>' if rng.random() < 0.25
                     else ('<span class="badge small" data-component="Badge" data-variant-size="small"></span>' if rng.random() < 0.2 else ""))
            items.append(f'<div class="dest{" active" if i == active else ""}"><div class="pill"><span class="badge-wrap">{self.icon(ic)}{badge}</span></div><div class="lbl">{lab}</div></div>')
        return f'<div class="nav-bar" data-component="NavigationBar" data-destinations="{len(dests)}">{"".join(items)}</div>'

    def nav_rail(self) -> str:
        rng = self.rng
        dests = G.pick(rng, G.NAV_DESTS, rng.randint(3, 5))
        active = rng.randrange(len(dests))
        fab = f'<md-fab variant="primary" lowered>{self.icon(rng.choice(G.FAB_ICONS), "icon")}</md-fab>' if rng.random() < 0.5 else ""
        items = "".join(f'<div class="dest{" active" if i == active else ""}"><div class="pill">{self.icon(ic)}</div><div class="lbl">{lab}</div></div>' for i, (ic, lab) in enumerate(dests))
        return f'<div class="nav-rail" data-component="NavigationRail">{self.icon_button("menu")}{fab}{items}</div>'

    def fab(self, bottom: int) -> str:
        rng = self.rng
        size = rng.choice(["small", "medium", "medium", "large"])
        variant = rng.choice(["surface", "primary", "secondary", "tertiary"])
        extended = size == "medium" and rng.random() < 0.4
        lab = f' label="{rng.choice(["Compose", "New event", "Create", "Add"])}"' if extended else ""
        sz = "" if size == "medium" else f' size="{size}"'
        return f'<div class="fab-holder" style="bottom:{bottom}px"><md-fab variant="{variant}"{sz}{lab}>{self.icon(rng.choice(G.FAB_ICONS), "icon")}</md-fab></div>'

    def snackbar(self, bottom: int) -> str:
        rng = self.rng
        act = rng.random() < 0.6
        a = f'<span class="act">{rng.choice(["Undo", "Retry", "View"])}</span>' if act else ""
        return f'<div class="snackbar" data-component="Snackbar" data-variant-action="{"true" if act else "false"}" style="bottom:{bottom}px"><span class="msg">{rng.choice(G.SNACKBAR_TEXT)}</span>{a}{self.icon_button("close") if rng.random() < 0.3 else ""}</div>'

    def dialog(self) -> str:
        rng = self.rng
        ic = self.icon(rng.choice(["delete", "info", "warning", "location_on"]), "icon") if rng.random() < 0.4 else ""
        return (f'<md-dialog open>{ic}<div slot="headline">{rng.choice(G.DIALOG_TITLES)}</div>'
                f'<form slot="content" method="dialog">{rng.choice(G.DIALOG_BODY)}</form>'
                f'<div slot="actions">{self.button(kind="text", label="Cancel")}{self.button(kind=rng.choice(["text", "filled"]), label=rng.choice(["Delete", "Discard", "Allow", "Leave", "Save"]))}</div></md-dialog>')

    # ---- composition
    def compose(self) -> str:
        rng = self.rng
        phone = self.w < 600
        rail = self.w >= 840 and rng.random() < 0.6
        blocks = []
        if rng.random() < 0.9:
            blocks.append(self.app_bar())
        if rng.random() < 0.5:
            blocks.append(self.tabs())
        pool = [self.list_block, self.card_grid, self.form, self.chip_row, self.button_row, self.headline_block, self.progress_block, self.choice_block, self.menu_block]
        weights = [4, 3, 3, 3, 2, 2, 1, 1, 1]
        n_blocks = rng.randint(3, 4 + self.h // 300)
        for _ in range(n_blocks):
            fn = rng.choices(pool, weights)[0]
            blocks.append(fn())
        body = "".join(blocks)
        overlays = []
        bottom = 16
        nav = ""
        if phone and rng.random() < 0.7:
            nav = self.nav_bar()
            bottom = 96
        if rng.random() < 0.5:
            overlays.append(self.fab(bottom))
            bottom += 72
        if rng.random() < 0.3:
            overlays.append(self.snackbar(bottom))
        if rng.random() < 0.2:
            overlays.append(self.dialog())
        main = f'<div class="main">{body}</div>'
        if rail:
            main = f'<div class="shell">{self.nav_rail()}{main}</div>'
        nav_html = f'<div class="bottom">{nav}</div>' if nav else ""
        return main + nav_html + "".join(overlays)


def _asset_url(path: str, html_dir: Optional[str]) -> str:
    """URL of a repo asset as referenced from a page saved in ``html_dir``: relative when the
    page's directory is known (shipped fixtures must not embed the absolute path of the checkout
    that generated them), else an absolute ``file://`` URL."""
    if html_dir:
        try:
            return os.path.relpath(path, os.path.abspath(html_dir)).replace(os.sep, "/")
        except ValueError:  # different drive (Windows)
            pass
    return "file://" + path


def page_html(rng: random.Random, width: int, height: int, body_html: Optional[str] = None,
              html_dir: Optional[str] = None) -> str:
    """Compose one random Material 3 screen as a complete HTML document.

    ``body_html`` replaces the random composition with the given body markup (same fonts,
    bundle, tokens and CSS), which is handy for deterministic kitchen-sink pages in tests.
    ``html_dir`` is the directory the page will be saved in; fonts and the Material Web bundle
    are then referenced relative to it so the saved page keeps working after the checkout moves.
    """
    page = _Page(rng, width, height)
    body = body_html if body_html is not None else page.compose()
    css = f"""
{G.root_css_vars()}
* {{ transition: none !important; animation: none !important; }}
html, body {{ margin:0; padding:0; width:{width}px; height:{height}px; overflow:hidden; }}
body {{ font-family: Roboto, sans-serif; background: var(--md-sys-color-surface); color: var(--md-sys-color-on-surface); position:relative; }}
.shell {{ display:flex; height:100%; }}
.main {{ flex:1; min-width:0; display:flex; flex-direction:column; gap:4px; }}
.main > * {{ flex: none; }}
.bottom {{ position:absolute; left:0; right:0; bottom:0; }}
md-list {{ --md-list-container-color: transparent; }}
md-list-item {{ --md-list-item-label-text-color: var(--md-sys-color-on-surface); }}
{G.HAND_STYLED_CSS}
"""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{_asset_url(os.path.join(FONTS_DIR, "roboto.local.css"), html_dir)}">
<link rel="stylesheet" href="{_asset_url(os.path.join(FONTS_DIR, "material-symbols.local.css"), html_dir)}">
<script>
(() => {{ const orig = Element.prototype.animate; Element.prototype.animate = function(k, o) {{ const a = orig.call(this, k, o); try {{ a.finish(); }} catch (e) {{}} return a; }}; }})();
</script>
<script src="{_asset_url(BUNDLE, html_dir)}"></script>
<style>{css}</style>
</head><body>{body}
<script>
document.querySelectorAll('md-tabs[data-active]').forEach(t => t.updateComplete.then(() => {{ t.activeTabIndex = parseInt(t.dataset.active); }}));
// md-select resolves `selected` options asynchronously (racy); pin the value after upgrade
document.querySelectorAll('md-outlined-select, md-filled-select').forEach(s => s.updateComplete.then(() => {{ const o = s.querySelector('md-select-option[selected]'); if (o) {{ s.value = o.getAttribute('value'); }} }}));
</script>
</body></html>
"""


# --------------------------------------------------------------------------- corpus
def _histogram(doc: Document) -> dict[str, int]:
    h: dict[str, int] = {}
    for n in doc.walk():
        if n.component:
            h[n.component.key] = h.get(n.component.key, 0) + 1
    return dict(sorted(h.items()))


def _type_histogram(doc: Document) -> dict[str, int]:
    h: dict[str, int] = {}
    for n in doc.walk():
        h[n.type] = h.get(n.type, 0) + 1
    return dict(sorted(h.items()))


def generate(n: int, out_dir: str, seed: int = 1, widths: tuple[int, ...] = WIDTHS, height_range: tuple[int, int] = HEIGHT_RANGE) -> dict:
    """Generate ``n`` Material Web ground-truth pages under ``out_dir``; returns the manifest.

    Files per page: ``<id>.html``, ``<id>.png``, ``<id>.gt.json``. Also writes ``manifest.json``.
    Deterministic for a given ``(n, seed, widths, height_range)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    cases = []
    comp_hist: dict[str, int] = {}
    for i in range(n):
        width = rng.choice(widths)
        height = rng.randint(height_range[0], height_range[1])
        cid = f"mwc_{seed}_{i:03d}"
        html = page_html(random.Random(rng.random()), width, height, html_dir=out_dir)
        html_path = os.path.join(out_dir, cid + ".html")
        with open(html_path, "w") as f:
            f.write(html)
        png_path = os.path.join(out_dir, cid + ".png")
        doc = extract_ground_truth(html_path, width, height, png_path)
        # paths are stored relative to the corpus dir so the fixtures are portable (see load_case)
        doc.source_image = cid + ".png"
        doc.meta = {"corpus": "mwc", "id": cid, "seed": seed, "index": i, "html": cid + ".html", "html_sha1": hashlib.sha1(html.encode()).hexdigest()}
        doc.save(os.path.join(out_dir, cid + ".gt.json"))
        hist = _histogram(doc)
        for k, v in hist.items():
            comp_hist[k] = comp_hist.get(k, 0) + v
        cases.append({"id": cid, "width": width, "height": height, "nodes": doc.root.count(), "components": hist, "types": _type_histogram(doc), "html_sha1": doc.meta["html_sha1"]})
    manifest = {"kind": "mwc", "seed": seed, "n": n, "widths": list(widths), "height_range": list(height_range), "ids": [c["id"] for c in cases], "cases": cases, "component_histogram": dict(sorted(comp_hist.items()))}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_case(out_dir: str, cid: str) -> tuple[Document, str]:
    """Return ``(gt_document, png_path)`` for a corpus case id (``source_image`` made absolute)."""
    doc = Document.load(os.path.join(out_dir, cid + ".gt.json"))
    png = os.path.join(out_dir, cid + ".png")
    doc.source_image = png
    return doc, png
