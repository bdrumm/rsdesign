"""Tests for dt.mapping (module C): catalog, token snapping, component matching, ingestion.

Ground truth for the component-matching tests comes from the REAL renderer:
  * hand-built IR documents are rendered with `dt.render.render_doc` and the pixels under each
    component are checked against the IR fills before mapping is asserted;
  * `out/mwc_test.html` (real Material Web components) is rendered with `render_url`, the IR is
    extracted from the (shadow) DOM, and the matcher must recover every component + variant.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dt.ir import Box, Color, Document, Fill, Node, Shadow, Stroke, TextStyle, assign_ids
from dt.mapping import (
    ChildPattern, DesignSystem, Signature, Token, map_document, match_node, material3, score_signature, derive_signature,
    collapse_instances, expand_instances, instances,
)
from dt.mapping.material3 import LIGHT_SCHEME, TYPESCALE, _c, verify_against_material_web, MW_TOKENS_DIR, FIXTURES_DIR
from dt.mapping.matcher import features, snap_color, snap_radius, snap_text, snap_spacing, snap_elevation
from dt.mapping import ingest_library, ingest_figma, ingest_screenshots
from dt.params import P
from dt.render.screenshot import render_doc, render_url

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MWC_HTML = os.path.join(ROOT, "out", "mwc_test.html")

EXPECTED_COMPONENTS = {
    "Button", "FAB", "IconButton", "Chip", "TextField", "Switch", "Checkbox", "Radio", "Slider", "ListItem", "Card",
    "TopAppBar", "NavigationBar", "NavigationRail", "NavigationDrawer", "Tabs", "Dialog", "Snackbar", "Divider", "Badge",
    "Progress", "Menu", "SearchBar", "SegmentedButton", "Select",
}


@pytest.fixture(scope="module")
def ds() -> DesignSystem:
    return material3()


# --------------------------------------------------------------------------- IR builders
def text(t: str, x: float, y: float, w: float, h: float, size: float = 14, weight: int = 500, lh: float = 20, color: str = "#ffffff") -> Node:
    return Node(type="text", text=t, box=Box(x, y, w, h),
                text_style=TextStyle(size=size, weight=weight, line_height=lh, color=Color.from_hex(color)))


def icon(name: str, x: float, y: float, s: float, color: str) -> Node:
    return Node(type="icon", icon_name=name, box=Box(x, y, s, s), fills=[Fill.solid(color)])


def filled_button(x: float = 16, y: float = 24) -> Node:
    return Node(type="frame", name="btn", box=Box(x, y, 110, 40), radius=20, fills=[Fill.solid(_c("primary"))],
                children=[icon("add", x + 16, y + 11, 18, _c("on-primary")), text("Save", x + 42, y + 10, 50, 20)])


def outlined_text_field(x: float = 16, y: float = 84) -> Node:
    return Node(type="frame", name="tf", box=Box(x, y, 200, 56), radius=4,
                strokes=[Stroke(color=Color.from_hex(_c("outline")), width=1)],
                children=[text("Email", x + 12, y - 6, 40, 16, size=12, weight=400, lh=16, color=_c("on-surface-variant")),
                          text("a@b.com", x + 16, y + 16, 120, 24, size=16, weight=400, lh=24, color=_c("on-surface"))])


def switch(x: float, y: float) -> Node:
    return Node(type="frame", name="switch", box=Box(x, y, 52, 32), radius=16, fills=[Fill.solid(_c("primary"))],
                children=[Node(type="ellipse", box=Box(x + 24, y + 4, 24, 24), fills=[Fill.solid(_c("on-primary"))])])


def list_item(x: float = 16, y: float = 160) -> Node:
    return Node(type="frame", name="li", box=Box(x, y, 380, 72),
                children=[icon("wifi", x + 16, y + 24, 24, _c("on-surface-variant")),
                          text("Wi-Fi", x + 56, y + 16, 200, 24, size=16, weight=400, lh=24, color=_c("on-surface")),
                          text("Connected to Home", x + 56, y + 40, 200, 20, size=14, weight=400, lh=20, color=_c("on-surface-variant")),
                          switch(x + 314, y + 20)])


def fab(x: float = 500, y: float = 300) -> Node:
    return Node(type="frame", name="fab", box=Box(x, y, 56, 56), radius=16, fills=[Fill.solid(_c("primary-container"))],
                effects=[Shadow(dy=2, blur=10)], children=[icon("edit", x + 16, y + 16, 24, _c("on-primary-container"))])


def elevated_card(x: float = 16, y: float = 260) -> Node:
    return Node(type="frame", name="card", box=Box(x, y, 300, 120), radius=12, fills=[Fill.solid(_c("surface-container-low"))],
                effects=[Shadow(dy=1, blur=3)],
                children=[text("Title", x + 16, y + 16, 200, 24, size=16, weight=500, lh=24, color=_c("on-surface")),
                          text("Body text here", x + 16, y + 44, 260, 20, size=14, weight=400, lh=20, color=_c("on-surface-variant"))])


def sample_doc() -> Document:
    doc = Document.blank(600, 400, _c("surface"))
    doc.root.children = [filled_button(), outlined_text_field(), list_item(), fab(), elevated_card()]
    assign_ids(doc.root)
    return doc


def _pixel(rgb: np.ndarray, x: float, y: float) -> Color:
    r, g, b = rgb[int(y), int(x)]
    return Color(int(r), int(g), int(b))


# --------------------------------------------------------------------------- catalog
def test_catalog_completeness(ds: DesignSystem):
    assert set(ds.component_names()) == EXPECTED_COMPONENTS
    keys = [c.key for c in ds.components]
    assert len(keys) == len(set(keys))
    for spec in ds.components:
        assert spec.library == "material3"
        for variant, sig in spec.effective_signatures():
            assert not sig.is_empty(), spec.name
            assert sig.height is not None or sig.width is not None or sig.radius is not None, (spec.name, variant)
            for ref in (sig.fills or []) + (sig.text_colors or []) + (sig.stroke.get("tokens", []) if isinstance(sig.stroke, dict) else []):
                assert ref == "none" or ds.resolve_color(ref) is not None, (spec.name, ref)
            for role in sig.text_roles or []:
                assert ds.token(role) is not None and ds.token(role).category == "typescale", (spec.name, role)


def test_variant_vocabulary_is_the_one_catalog_vocabulary(ds: DesignSystem):
    """Every catalog variant emits exactly its component's vocabulary (docs/VARIANTS.md), the
    vocabulary ships in material3.json, and corpus helpers reject anything outside it."""
    from dt.mapping.material3 import VARIANTS, variant, component_key
    vocab = ds.variant_vocabulary()
    assert set(vocab) == set(VARIANTS) and set(ds.component_names()) <= set(vocab)
    for spec in ds.components:
        for v in spec.variants:
            assert set(v.props) == set(vocab[spec.name]), (spec.name, v.name, v.props)
            assert all(val in vocab[spec.name][k] for k, val in v.props.items()), (spec.name, v.name, v.props)
            assert ds.normalize_variant(spec.name, v.props) == v.props
        if not spec.variants:
            assert vocab[spec.name] == {}, spec.name
    loaded = DesignSystem.load(os.path.join(FIXTURES_DIR, "material3.json"))
    assert loaded.variant_vocabulary() == vocab
    assert variant("FAB", size="small") == {"size": "small", "extended": "false"}
    assert variant("Chip", type="filter", selected=True) == {"type": "filter", "selected": "true"}
    assert component_key("Button", {"style": "filled"}) == "m3.button:style=filled" and component_key("Divider", {}) == "m3.divider"
    with pytest.raises(KeyError):
        variant("Button", icon="true")  # props are not variants
    with pytest.raises(ValueError):
        variant("FAB", size="medium")  # Material Web's attribute value is not the vocabulary value
    assert ds.normalize_variant("FAB", {"size": "large", "color": "tertiary"}) == {"size": "large", "extended": "false"}
    assert ds.normalize_variant("Unknown", {"a": 1}) == {"a": "1"}


def test_catalog_tokens(ds: DesignSystem):
    roles = {t.short for t in ds.tokens_by_category("color") if t.name.startswith("md.sys.color.")}
    assert roles == set(LIGHT_SCHEME)
    assert ds.resolve_color("primary").hex() == "#6750a4"
    assert ds.resolve_color("on-primary").hex() == "#ffffff"
    assert ds.resolve_color("primary-container").hex() == "#eaddff"
    assert ds.resolve_color("surface").hex() == "#fef7ff"
    assert ds.resolve_color("outline").hex() == "#79747e"
    assert ds.resolve_color("outline-variant").hex() == "#cac4d0"
    assert ds.resolve_color("on-surface").hex() == "#1d1b20"
    assert ds.resolve_color("on-surface-variant").hex() == "#49454f"
    assert set(ds.typescale()) == {f"md.sys.typescale.{r}" for r in TYPESCALE} and len(TYPESCALE) == 15
    assert ds.typescale()["md.sys.typescale.label-large"] == {"size": 14.0, "line_height": 20.0, "weight": 500, "tracking": 0.1, "family": "Roboto"}
    assert ds.shape_scale()["md.sys.shape.corner-full"] == "full" and ds.shape_scale()["md.sys.shape.corner-extra-large"] == 28
    assert [t.value["dp"] for t in ds.tokens_by_category("elevation")] == [0, 1, 3, 6, 8, 12]
    assert ds.spacing_grid() == 4.0
    assert ds.fonts == ["Roboto"]


def test_catalog_metrics_match_material_web(ds: DesignSystem):
    """Signatures encode the exact DOM metrics of Material Web components."""
    def sig(name: str, variant: str) -> Signature:
        spec = ds.component(name)
        return next(s for v, s in spec.effective_signatures() if v and v.name == variant)
    assert sig("Button", "filled").height == (40, 40) and sig("Button", "filled").radius == "pill" and sig("Button", "filled").icon_size == (18, 18)
    assert sig("FAB", "regular").height == (56, 56) and sig("FAB", "regular").radius == 16 and sig("FAB", "regular").icon_size == (24, 24)
    assert sig("Switch", "selected").height == (32, 32) and sig("Switch", "selected").width == (52, 52)
    assert sig("Checkbox", "checked").height == (18, 18)
    assert sig("TextField", "outlined").height == (56, 56) and sig("TextField", "outlined").radius == 4
    assert sig("Chip", "assist").height == (32, 32) and sig("Chip", "assist").radius == 8


def test_catalog_in_sync_with_material_web_package():
    if not os.path.isdir(MW_TOKENS_DIR):
        pytest.skip("@material/web not installed")
    issues = verify_against_material_web()
    assert issues == [], issues


def test_fixture_json_roundtrip(ds: DesignSystem):
    path = os.path.join(FIXTURES_DIR, "material3.json")
    assert os.path.exists(path), "run python -m dt.mapping.material3"
    loaded = DesignSystem.load(path)
    assert loaded.to_dict() == ds.to_dict()
    assert DesignSystem.from_dict(json.loads(json.dumps(ds.to_dict()))).to_dict() == ds.to_dict()
    assert DesignSystem.load_named("material3").name == "material3"


# --------------------------------------------------------------------------- snapping
def test_snap_colors(ds: DesignSystem):
    assert snap_color(Color.from_hex("#6750a4"), ds)[0] == "md.sys.color.primary"
    tok, ev = snap_color(Color.from_hex("#6a52a6"), ds)  # slightly off primary
    assert tok == "md.sys.color.primary" and 0 < ev["de"] < P["map.color_de"]
    assert snap_color(Color.from_hex("#ffffff"), ds)[0] == "md.sys.color.on-primary"  # tie broken by priority
    assert snap_color(Color.from_hex("#1d1b20"), ds)[0] == "md.sys.color.on-surface"
    assert snap_color(Color.from_hex("#fef7ff"), ds)[0] == "md.sys.color.surface"
    tok, ev = snap_color(Color.from_hex("#00ff00"), ds)
    assert tok is None and ev["de"] > P["map.color_de"]


def test_snap_radius_text_spacing_elevation(ds: DesignSystem):
    assert snap_radius(20, 110, 40, ds)[0] == "md.sys.shape.corner-full"
    assert snap_radius(9999, 110, 40, ds)[0] == "md.sys.shape.corner-full"
    assert snap_radius(12, 300, 200, ds)[0] == "md.sys.shape.corner-medium"
    assert snap_radius(13, 300, 200, ds)[0] == "md.sys.shape.corner-medium"
    assert snap_radius(0, 300, 200, ds)[0] == "md.sys.shape.corner-none"
    assert snap_radius(22, 300, 200, ds)[0] is None  # 6px from both 16 and 28: no snap
    assert snap_text(TextStyle(size=14, weight=500, line_height=20), ds)[0] == "md.sys.typescale.label-large"
    assert snap_text(TextStyle(size=16, weight=400, line_height=24), ds)[0] == "md.sys.typescale.body-large"
    assert snap_text(TextStyle(size=22, weight=400, line_height=28), ds)[0] == "md.sys.typescale.title-large"
    assert snap_text(TextStyle(size=15, weight=400, line_height=22), ds)[0] is not None  # within tolerance
    assert snap_text(TextStyle(size=19, weight=900, line_height=40), ds)[0] is None
    assert snap_spacing(16, ds)[0] == "md.sys.spacing.16"
    assert snap_spacing(16.8, ds)[0] == "md.sys.spacing.16"
    assert snap_spacing(18, ds)[0] is None
    n = Node(effects=[Shadow(dy=1, blur=3)])
    assert snap_elevation(n, ds)[0] == "md.sys.elevation.level1"
    assert snap_elevation(Node(effects=[Shadow(dy=2, blur=10)]), ds)[0] == "md.sys.elevation.level3"


# --------------------------------------------------------------------------- matching: hand-built IR vs real renderer
def test_hand_built_ir_renders_as_specified():
    """The IR we match against is validated by the real renderer: pixels under each component equal its fill."""
    doc = sample_doc()
    rgb = render_doc(doc)
    assert rgb.shape == (400, 600, 3)
    btn, tf, li, fb, card = doc.root.children
    assert _pixel(rgb, btn.box.x + 8, btn.box.cy).delta_e(Color.from_hex(_c("primary"))) < 3
    assert _pixel(rgb, fb.box.x + 6, fb.box.y + 6).delta_e(Color.from_hex(_c("primary-container"))) < 3
    assert _pixel(rgb, card.box.x2 - 6, card.box.y2 - 6).delta_e(Color.from_hex(_c("surface-container-low"))) < 3
    assert _pixel(rgb, tf.box.x + 0.5, tf.box.cy).delta_e(Color.from_hex(_c("outline"))) < 12  # 1px antialiased stroke
    sw = li.children[3]
    assert _pixel(rgb, sw.box.x + 8, sw.box.cy).delta_e(Color.from_hex(_c("primary"))) < 3


@pytest.mark.parametrize("builder,name,variant", [
    (filled_button, "Button", {"style": "filled"}),
    (outlined_text_field, "TextField", {"style": "outlined"}),
    (list_item, "ListItem", {"lines": "2"}),
    (fab, "FAB", {"size": "regular", "extended": "false"}),
    (elevated_card, "Card", {"style": "elevated"}),
])
def test_match_hand_built(ds: DesignSystem, builder, name, variant):
    node = builder()
    doc = Document.blank(600, 400, _c("surface"))
    doc.root.children = [node]
    assign_ids(doc.root)
    map_document(doc, ds)
    assert node.component is not None, name
    assert node.component.name == name and node.component.variant == variant
    assert node.component.confidence >= 0.6
    assert node.type == "instance" and node.component.library == "material3"
    ev = node.component.evidence
    assert ev["candidates"][0]["name"] == name and "constraints" in ev
    runner_up = ev["candidates"][1]["score"] if len(ev["candidates"]) > 1 else 0
    assert node.component.confidence - runner_up > 0.2, ev["candidates"]


def test_map_document_full(ds: DesignSystem):
    doc = map_document(sample_doc(), ds)
    assert doc.design_system == "material3"
    by_name = {n.name: n for n in doc.walk()}
    btn = by_name["btn"].component
    assert btn.props["label"] == "Save" and btn.props["icon"] == "add"
    assert by_name["btn"].tokens["fill"] == "md.sys.color.primary" and by_name["btn"].tokens["shape"] == "md.sys.shape.corner-full"
    tf = by_name["tf"].component
    assert tf.props["label"] == "Email" and tf.props["value"] == "a@b.com"
    assert by_name["tf"].tokens["stroke"] == "md.sys.color.outline" and by_name["tf"].tokens["shape"] == "md.sys.shape.corner-extra-small"
    li = by_name["li"].component
    assert li.props["headline"] == "Wi-Fi" and li.props["supporting"] == ["Connected to Home"] and li.props["leading"] == "wifi"
    assert li.props["trailing"]["instance"] == "Switch" and li.props["trailing"]["variant"] == {"selected": "true"}
    sw = by_name["switch"]
    assert sw.component.name == "Switch" and sw.children[0].component is None  # handle stays a plain ellipse
    assert by_name["fab"].tokens["elevation"] == "md.sys.elevation.level3"
    assert by_name["card"].tokens["elevation"] == "md.sys.elevation.level1"
    label = next(n for n in doc.walk() if n.text == "Save")
    assert label.tokens["typescale"] == "md.sys.typescale.label-large" and label.tokens["text_color"] == "md.sys.color.on-primary"
    assert label.meta["tokens_evidence"]["typescale"]["from_component"] == "Button"
    assert doc.meta["mapping"]["components"] == 6
    for n in doc.walk():
        if n.tokens:
            assert "tokens_evidence" in n.meta
    # JSON round trip keeps everything
    doc2 = Document.from_json(doc.to_json())
    assert doc2.find(by_name["btn"].id).component.to_dict() == btn.to_dict()


def test_collapse_and_expand(ds: DesignSystem):
    doc = map_document(sample_doc(), ds, collapse=True)
    inst = instances(doc)
    assert inst and all(not n.children for n in inst)
    assert all("collapsed_children" in n.meta for n in inst)
    expand_instances(doc)
    assert all(n.children for n in inst if n.name != "switch" or True)
    assert doc.find("n1").children


def test_match_node_returns_ranked_candidates(ds: DesignSystem):
    cands = match_node(filled_button(), ds)
    assert [c.spec.name for c in cands][0] == "Button" and cands[0].variant.name == "filled"
    assert cands[0].score > cands[1].score
    # a plain unknown blob gets no confident match but may be recorded as a candidate
    blob = Node(type="frame", box=Box(0, 0, 300, 300), fills=[Fill.solid("#ff0000")], radius=50)
    doc = Document.blank(400, 400)
    doc.root.children = [blob]
    map_document(doc, ds)
    assert blob.component is None


def test_variant_resolution_buttons(ds: DesignSystem):
    def btn(fill=None, stroke=None, shadow=False, text_color="#ffffff"):
        n = Node(type="frame", box=Box(0, 0, 100, 40), radius=20, children=[text("Go", 24, 10, 40, 20, color=text_color)])
        if fill:
            n.fills = [Fill.solid(_c(fill))]
        if stroke:
            n.strokes = [Stroke(color=Color.from_hex(_c(stroke)), width=1)]
        if shadow:
            n.effects = [Shadow(dy=1, blur=3)]
        return n
    cases = [
        (btn(fill="primary"), "filled"), (btn(fill="secondary-container", text_color=_c("on-secondary-container")), "tonal"),
        (btn(stroke="outline", text_color=_c("primary")), "outlined"), (btn(text_color=_c("primary")), "text"),
        (btn(fill="surface-container-low", shadow=True, text_color=_c("primary")), "elevated"),
    ]
    for node, style in cases:
        c = match_node(node, ds)[0]
        assert c.spec.name == "Button" and c.variant.props["style"] == style and c.score >= 0.6, (style, c.summary())


def test_score_signature_soft_penalties(ds: DesignSystem):
    f = features(filled_button())
    sig = next(s for v, s in ds.component("Button").effective_signatures() if v.name == "filled")
    s_ok, _ = score_signature(f, sig, ds)
    assert s_ok > 0.95
    taller = filled_button()
    taller.box = Box(16, 24, 110, 44)
    s_tall, det = score_signature(features(taller), sig, ds)
    assert 0.6 < s_tall < s_ok and det["height"]["score"] < 1
    recolored = filled_button()
    recolored.fills = [Fill.solid("#ff0000")]
    s_red, det = score_signature(features(recolored), sig, ds)
    assert s_red < 0.5 and det["fill"]["score"] == 0


def test_params_registered():
    for key in ("map.color_de", "map.min_conf", "map.radius_tol", "map.grid", "map.grid_tol", "map.learn.cluster_thr"):
        assert key in P.all() and key in P.ranges() and key in P.docs()


# --------------------------------------------------------------------------- matching: real Material Web components
MWC_EXTRACT_JS = r"""
(() => {
  const alpha = (s) => { const m = s && s.match(/rgba?\(([^)]+)\)/); if (!m) return 0; const p = m[1].split(',').map(Number); return p.length > 3 ? p[3] : 1; };
  const px = (s) => parseFloat(s) || 0;
  const hosts = document.querySelectorAll('md-filled-button, md-outlined-button, md-text-button, md-filled-tonal-button, md-fab, md-switch, md-checkbox, md-outlined-text-field, md-assist-chip, md-filter-chip');
  const out = [];
  const walkShadow = (root, fn) => { for (const el of root.querySelectorAll('*')) { fn(el); if (el.shadowRoot) walkShadow(el.shadowRoot, fn); } };
  const visible = (cs) => cs.visibility !== 'hidden' && px(cs.opacity) > 0.1 && cs.display !== 'none';
  for (const host of hosts) {
    const hr = host.getBoundingClientRect();
    const rec = { tag: host.tagName.toLowerCase(), box: [hr.x, hr.y, hr.width, hr.height], painted: [], texts: [], icons: [], elevation: 0 };
    const seen = [];
    const addText = (txt, tr, cs) => {
      if (!txt || tr.width <= 0 || px(cs.lineHeight) === 0) return;
      for (const s of seen) if (s.text === txt && Math.abs(s.box[1] - tr.y) < 6) return;
      const t = { text: txt, box: [tr.x, tr.y, tr.width, tr.height], size: px(cs.fontSize), weight: +cs.fontWeight || 400, lh: px(cs.lineHeight), color: cs.color };
      seen.push(t); rec.texts.push(t);
    };
    if (host.shadowRoot) walkShadow(host.shadowRoot, (el) => {
      const r = el.getBoundingClientRect(); if (r.width < 1 || r.height < 1) return;
      const cs = getComputedStyle(el); if (!visible(cs)) return;
      const tag = el.tagName.toLowerCase();
      const cls = typeof el.className === 'string' ? el.className : '';
      if (tag === 'md-elevation') { rec.elevation = Math.max(rec.elevation, px(cs.getPropertyValue('--md-elevation-level'))); return; }
      if (tag === 'md-ripple' || tag === 'md-focus-ring') return;
      const entry = { tag, cls, box: [r.x, r.y, r.width, r.height], radius: px(cs.borderTopLeftRadius) };
      let keep = false;
      let bg = cs.backgroundColor;
      for (const ps of ['::before', '::after']) {
        const pcs = getComputedStyle(el, ps);
        if (pcs.content === 'none' || !visible(pcs)) continue;
        if (alpha(pcs.backgroundColor) > 0.05 && alpha(bg) <= 0.05) { bg = pcs.backgroundColor; entry.radius = px(pcs.borderTopLeftRadius) || entry.radius; }
        const pbw = Math.max(px(pcs.borderTopWidth), px(pcs.borderBottomWidth), px(pcs.borderLeftWidth), px(pcs.borderRightWidth));
        if (pbw > 0 && alpha(pcs.borderTopColor) > 0.05 && !entry.border) { entry.border = [pcs.borderTopColor, pbw]; keep = true; }
      }
      if (alpha(bg) > 0.05) { entry.bg = bg; keep = true; }
      const bw = Math.max(px(cs.borderTopWidth), px(cs.borderBottomWidth), px(cs.borderLeftWidth), px(cs.borderRightWidth));
      const bc = [cs.borderTopColor, cs.borderBottomColor, cs.borderLeftColor, cs.borderRightColor].find(c => alpha(c) > 0.05);
      if (bw > 0 && bc && [cs.borderTopStyle, cs.borderBottomStyle, cs.borderLeftStyle, cs.borderRightStyle].some(s => s !== 'none')) { entry.border = [bc, bw]; keep = true; }
      if (tag === 'md-icon' || tag === 'svg') { rec.icons.push({ name: el.textContent.trim() || 'check', box: [r.x, r.y, r.width, r.height], color: cs.color }); return; }
      if (tag === 'input' && !['checkbox', 'radio', 'range'].includes(el.type)) addText(el.value, r, cs);
      if (tag === 'slot' || /\blabel\b|headline|supporting/.test(cls)) {
        const range = document.createRange(); range.selectNodeContents(el);
        addText(el.textContent.trim(), range.getBoundingClientRect(), cs);
      }
      if (keep) rec.painted.push(entry);
    });
    for (const ic of host.querySelectorAll('md-icon')) { const r = ic.getBoundingClientRect(); rec.icons.push({ name: ic.textContent.trim(), box: [r.x, r.y, r.width, r.height], color: getComputedStyle(ic).color }); }
    for (const n of host.childNodes) {
      if (n.nodeType === 3 && n.textContent.trim()) {
        const range = document.createRange(); range.selectNodeContents(n);
        const ref = n.assignedSlot ? (n.assignedSlot.parentElement || n.assignedSlot) : host;
        addText(n.textContent.trim(), range.getBoundingClientRect(), getComputedStyle(ref));
      }
    }
    out.push(rec);
  }
  return out;
})()
"""


def _css_color(s: str) -> Color:
    nums = [float(x) for x in s[s.index("(") + 1:s.index(")")].split(",")]
    return Color(int(nums[0]), int(nums[1]), int(nums[2]), nums[3] if len(nums) > 3 else 1.0)


def _box(b: list[float]) -> Box:
    return Box(*b)


def mwc_record_to_ir(rec: dict) -> Node:
    """DOM record (one md-* host) -> IR frame with fills/strokes/radius/shadow + text/icon/shape children."""
    host = _box(rec["box"])
    node = Node(type="frame", name=rec["tag"], box=host)
    covering = [p for p in rec["painted"] if _box(p["box"]).iou(host) >= 0.9]
    bg_entries = [p for p in covering if p.get("bg")]
    if bg_entries:
        p = bg_entries[-1]
        node.fills = [Fill.solid(_css_color(p["bg"]))]
        node.radius = (p["radius"],) * 4
    borders = [p for p in rec["painted"] if p.get("border")]
    if borders:
        union = _box(borders[0]["box"])
        for p in borders[1:]:
            union = union.union(_box(p["box"]))
        covered = any(_box(q["box"]).iou(union) >= 0.9 for q in bg_entries)
        if union.iou(host) >= 0.9 and not covered:
            c, w = borders[0]["border"]
            node.strokes = [Stroke(color=_css_color(c), width=float(w))]
            node.radius = (max(p["radius"] for p in borders),) * 4
    if rec["elevation"] > 0:
        node.effects = [Shadow(dy=rec["elevation"] / 2, blur=rec["elevation"] * 2)]
    for p in rec["painted"]:
        if p.get("bg") and p not in covering:
            b = _box(p["box"])
            pill = p["radius"] >= min(b.w, b.h) / 2 - 1
            node.children.append(Node(type="ellipse" if pill and abs(b.w - b.h) < 1 else "rect", box=b, fills=[Fill.solid(_css_color(p["bg"]))], radius=(p["radius"],) * 4))
    for ic in rec["icons"]:
        node.children.append(Node(type="icon", icon_name=ic["name"], box=_box(ic["box"]), fills=[Fill.solid(_css_color(ic["color"]))]))
    for t in rec["texts"]:
        node.children.append(Node(type="text", text=t["text"], box=_box(t["box"]),
                                  text_style=TextStyle(size=t["size"], weight=t["weight"], line_height=t["lh"] or None, color=_css_color(t["color"]))))
    return node


@pytest.fixture(scope="module")
def mwc_doc() -> Document:
    if not os.path.exists(MWC_HTML):
        pytest.skip("out/mwc_test.html missing")
    rgb, recs = render_url("file://" + MWC_HTML, 600, 300, script=MWC_EXTRACT_JS)
    doc = Document.blank(600, 300, "#fef7ff")
    doc.root.children = [mwc_record_to_ir(r) for r in recs]
    assign_ids(doc.root)
    doc.meta["rgb"] = rgb
    return doc


def test_mwc_dom_ir_matches_pixels(mwc_doc: Document):
    """The DOM-derived IR agrees with the real screenshot of the Material Web page."""
    rgb = mwc_doc.meta["rgb"]
    by = {n.name: n for n in mwc_doc.root.children}
    b = by["md-filled-button"]
    assert (b.box.w, b.box.h) == (pytest.approx(81.8, abs=1), 40) and b.uniform_radius() >= 20
    assert _pixel(rgb, b.box.x + 10, b.box.cy).delta_e(b.fill_color) < 3
    f = by["md-fab"]
    assert (f.box.w, f.box.h) == (56, 56) and f.uniform_radius() == 16 and f.effects
    assert _pixel(rgb, f.box.x + 6, f.box.y + 6).delta_e(f.fill_color) < 3
    s = by["md-switch"]
    assert (s.box.w, s.box.h) == (52, 32) and s.children[0].type == "ellipse"
    assert _pixel(rgb, s.box.x + 8, s.box.cy).delta_e(s.fill_color) < 3
    assert (by["md-checkbox"].box.w, by["md-checkbox"].box.h) == (18, 18)
    tf = by["md-outlined-text-field"]
    assert tf.box.h == 56 and tf.strokes and tf.uniform_radius() == 4
    assert by["md-assist-chip"].box.h == 32 and by["md-assist-chip"].uniform_radius() == 8


def test_mwc_components_mapped(mwc_doc: Document, ds: DesignSystem):
    doc = Document.from_dict(mwc_doc.to_dict())
    map_document(doc, ds)
    expected = {
        "md-filled-button": ("Button", {"style": "filled"}),
        "md-outlined-button": ("Button", {"style": "outlined"}),
        "md-text-button": ("Button", {"style": "text"}),
        "md-filled-tonal-button": ("Button", {"style": "tonal"}),
        "md-fab": ("FAB", {"size": "regular", "extended": "false"}),
        "md-switch": ("Switch", {"selected": "true"}),
        "md-checkbox": ("Checkbox", {"checked": "true"}),
        "md-outlined-text-field": ("TextField", {"style": "outlined"}),
        "md-filter-chip": ("Chip", {"type": "filter", "selected": "true"}),
    }
    got = {n.name: n for n in doc.root.children}
    for tag, (name, variant) in expected.items():
        n = got[tag]
        assert n.component is not None, (tag, n.meta.get("component_candidates"))
        assert (n.component.name, n.component.variant) == (name, variant), (tag, n.component.evidence["candidates"])
        assert n.component.confidence >= 0.6, (tag, n.component.evidence)
    chip = got["md-assist-chip"].component
    assert chip is not None and chip.name == "Chip" and chip.variant["type"] in ("assist", "suggestion") and chip.props["label"] == "Assist"
    assert got["md-filled-button"].component.props["label"] == "Filled"
    assert got["md-filled-tonal-button"].component.props["icon"] == "add"
    assert got["md-outlined-text-field"].component.props["value"] == "a@b.com"
    assert got["md-fab"].tokens["fill"] == "md.sys.color.surface-container-high"  # md-fab default variant is 'surface'
    assert got["md-filled-button"].tokens["fill"] == "md.sys.color.primary"
    assert got["md-outlined-button"].tokens["stroke"] == "md.sys.color.outline"


# --------------------------------------------------------------------------- matching: ground truth of the Material Web corpus
AB_TABS_BODY = """<div class="main">
<div class="app-bar small" data-component="TopAppBar" data-variant-size="small"><md-icon-button><md-icon>menu</md-icon></md-icon-button><div class="title">Inbox</div><md-icon-button><md-icon>search</md-icon></md-icon-button><md-icon-button><md-icon>more_vert</md-icon></md-icon-button></div>
<md-tabs data-active="1"><md-primary-tab>Mail</md-primary-tab><md-primary-tab>Chat</md-primary-tab><md-primary-tab>Meet</md-primary-tab></md-tabs>
<div class="app-bar center container" data-component="TopAppBar" data-variant-size="center"><md-icon-button><md-icon>arrow_back</md-icon></md-icon-button><div class="title">Photos</div><md-icon-button><md-icon>share</md-icon></md-icon-button></div>
<md-tabs data-active="0"><md-secondary-tab>Upcoming</md-secondary-tab><md-secondary-tab>Past</md-secondary-tab></md-tabs>
<md-list><md-divider></md-divider></md-list>
</div>"""


@pytest.fixture(scope="module")
def mwc_gt(tmp_path_factory) -> Document:
    """Ground truth extracted from a real Material Web render (mwc corpus extractor)."""
    import random
    from dt.selftest.mwc_corpus import extract_ground_truth, page_html
    d = str(tmp_path_factory.mktemp("abtabs"))
    path = os.path.join(d, "page.html")
    with open(path, "w") as f:
        f.write(page_html(random.Random(0), 600, 400, body_html=AB_TABS_BODY))
    return extract_ground_truth(path, 600, 400, os.path.join(d, "page.png"))


def test_top_app_bar_vs_tabs_on_material_web(mwc_gt: Document, ds: DesignSystem):
    """Mapping the DOM ground truth (components stripped) recovers every gt component + variant:
    a title bar with actions is a TopAppBar (small vs center by title placement), a strip of
    equal cells with an indicator is Tabs (primary 3px vs secondary 2px indicator)."""
    gt = {n.id: n.component for n in mwc_gt.walk() if n.component}
    assert sorted(c.name for c in gt.values()).count("TopAppBar") == 2 and sorted(c.name for c in gt.values()).count("Tabs") == 2
    doc = Document.from_dict(mwc_gt.to_dict())
    for n in doc.walk():
        n.component, n.tokens = None, {}
    map_document(doc, ds)
    for nid, ref in gt.items():
        if ref.name not in ("TopAppBar", "Tabs", "Divider"):
            continue
        got = doc.find(nid).component
        assert got is not None and (got.name, got.variant) == (ref.name, ref.variant), (ref.name, ref.variant, got and got.evidence["candidates"])
    # a matched divider becomes an instance (documented contract) and remembers its geometric
    # type; with map.keep_geom_types="line" it keeps 'line' instead (pairs with gt lines in compare)
    div = next(n for n in doc.walk() if n.component is not None and n.component.name == "Divider")
    assert div.type == "instance" and div.meta["orig_type"] == "line"
    P.set("map.keep_geom_types", "line")
    try:
        doc2 = Document.from_dict(mwc_gt.to_dict())
        for n in doc2.walk():
            n.component, n.tokens = None, {}
        map_document(doc2, ds)
        assert doc2.find(div.id).type == "line" and doc2.find(div.id).component.name == "Divider"
    finally:
        P.reset("map.keep_geom_types")
    # every emitted variant is exactly the vocabulary of its component
    vocab = ds.variant_vocabulary()
    for n in doc.walk():
        if n.component is not None:
            assert set(n.component.variant) == set(vocab[n.component.name]), (n.component.name, n.component.variant)


def test_tabs_and_title_constraints_are_critical(ds: DesignSystem):
    """Without an indicator a 64px row of icons + text never scores as Tabs; without a title it is
    not a confident TopAppBar either."""
    bar = Node(type="frame", box=Box(0, 0, 600, 64), fills=[Fill.solid(_c("surface"))], children=[
        icon("menu", 12, 20, 24, _c("on-surface")),
        text("Inbox", 60, 18, 55, 28, size=22, weight=400, lh=28, color=_c("on-surface")),
        icon("search", 516, 20, 24, _c("on-surface-variant")), icon("more_vert", 564, 20, 24, _c("on-surface-variant")),
    ])
    cands = {c.spec.name: c for c in match_node(bar, ds, top=10)}
    assert max(cands, key=lambda k: cands[k].score) == "TopAppBar" and cands["TopAppBar"].variant.props == {"size": "small"}
    assert "Tabs" not in cands or cands["Tabs"].score < 0.4
    strip = Node(type="frame", box=Box(0, 0, 600, 48), children=[
        text(t, 100 * (2 * i + 1) - 20, 14, 40, 20, size=14, weight=500, lh=20, color=_c("on-surface")) for i, t in enumerate(("Mail", "Chat", "Meet"))
    ] + [Node(type="rect", box=Box(180, 45, 40, 3), fills=[Fill.solid(_c("primary"))])])
    best = match_node(strip, ds)[0]
    assert best.spec.name == "Tabs" and best.variant.props == {"type": "primary"} and best.score >= 0.6, best.summary()
    # unequal cells (title + actions) fail the critical cells constraint even with an indicator line
    uneven = Node(type="frame", box=Box(0, 0, 600, 48), children=[strip.children[0], strip.children[2], strip.children[3]])
    uneven.children[0] = text("Mail", 30, 14, 40, 20, size=14, weight=500, lh=20, color=_c("on-surface"))
    tabs = next((c for c in match_node(uneven, ds, top=10) if c.spec.name == "Tabs"), None)
    assert tabs is None or tabs.score < 0.5


# --------------------------------------------------------------------------- confidence calibration
# A wrong or undecidable match must not look certain: anything below
# pipeline.decisions.component_conf is queued for the driver in decisions.json.
def _queued(doc: Document) -> set[str]:
    from dt.pipeline import build_decisions
    return {d["node_id"] for d in build_decisions(doc) if d["kind"] == "component"}


def _outlined_chip(x: float, y: float, label: str) -> Node:
    t = text(label, x + 16, y + 6, 8 * len(label), 20, size=14, weight=500, lh=20, color=_c("on-surface-variant"))
    return Node(type="frame", name="chip", box=Box(x, y, 32 + 8 * len(label), 32), radius=8,
                strokes=[Stroke(color=Color.from_hex(_c("outline")), width=1)], children=[t])


def test_variant_tie_is_not_confident(ds: DesignSystem):
    """A text-only outlined chip is pixel-identical as a suggestion chip and an unselected filter
    chip. The mapper must still pick one (best effort) but report the tie: confidence below the
    decision threshold, so the chip lands in decisions.json. Checked on the IR and on what
    perceive recovers from the real render."""
    from dt.perceive import perceive
    doc = Document.blank(400, 120, _c("surface"))
    doc.root.children = [_outlined_chip(16, 16, "Travel"), _outlined_chip(140, 16, "Hotels")]
    assign_ids(doc.root)
    for d in (Document.from_dict(doc.to_dict()), perceive(render_doc(doc))):
        map_document(d, ds)
        chips = [n for n in d.walk() if n.component is not None and n.component.name == "Chip"]
        assert len(chips) == 2
        queued = _queued(d)
        for n in chips:
            ev = n.component.evidence
            assert ev["runner_up"]["name"] == "Chip" and ev["runner_up"]["variant"] != n.component.variant, ev["runner_up"]
            assert n.component.confidence < P["pipeline.decisions.component_conf"], (n.component.variant, n.component.confidence)
            assert n.id in queued
            assert ev["score"] >= P["map.min_conf"]  # the best guess itself is unchanged


def _toc_html(color: str, weight: int) -> str:
    fonts = os.path.join(ROOT, "fixtures", "fonts", "roboto.local.css")
    labels = ["Interactive Demo", "Usage", "Continuous", "Range", "Theming", "Example", "Properties"]
    items = "".join(f'<div style="height:30px;line-height:30px;margin-left:{16 + 24 * (i % 2)}px;color:{color};'
                    f'font:{weight} 14px Roboto">{t}</div>' for i, t in enumerate(labels))
    return (f'<html><head><link rel="stylesheet" href="file://{fonts}"></head><body style="margin:0;background:#f6ebe0">'
            f'<div style="margin:16px;padding:24px;background:#fffbff;border-radius:24px;width:300px">{items}</div></body></html>')


def test_inferred_frame_confidence_rests_on_observed_evidence(ds: DesignSystem):
    """material-web.dev's table of contents: short accent-coloured links (#7f5700, 14px/400) inside
    a panel. perceive's grouping pass wraps each in an unpainted 40px 'text_button' frame, so height,
    aspect and 'no fill/stroke/shadow' match a text Button by construction. Confidence must rest on
    what was observed (label colour ΔE ~89 from primary, body weight): below the decision threshold.
    Real M3 text buttons (primary, label-large) in the same layout stay confident."""
    from dt.perceive import perceive
    from dt.render.screenshot import html_to_png
    thr = P["pipeline.decisions.component_conf"]
    links = perceive(html_to_png(_toc_html("#7f5700", 400), 400, 320))
    map_document(links, ds)
    btns = [n for n in links.walk() if n.component is not None and n.component.name == "Button"]
    assert btns and all(n.meta.get("src") == "group" for n in btns)
    queued = _queued(links)
    for n in btns:
        assert n.component.confidence < thr, (n.component.props.get("label"), n.component.confidence)
        assert n.id in queued
        assert n.component.evidence["observed_score"] < n.component.evidence["score"]
    real = perceive(html_to_png(_toc_html(_c("primary"), 500), 400, 320))
    map_document(real, ds)
    btns = [n for n in real.walk() if n.component is not None and n.component.name == "Button"]
    assert len(btns) >= 4 and all(n.component.confidence >= thr for n in btns), [n.component.confidence for n in btns]


def test_glyph_strokes_are_not_dividers(ds: DesignSystem):
    """A hamburger glyph that perceive does not identify reaches mapping as three 18x2 bars. They
    are icon strokes, not M3 dividers or linear progress bars (both span containers; nothing in the
    bench GT is shorter than 380 px). Real 1px dividers and 4px progress tracks still map."""
    from dt.perceive import perceive
    from dt.render.screenshot import html_to_png
    bar = '<div style="position:absolute;left:{x}px;top:{y}px;width:18px;height:2px;background:%s"></div>' % _c("on-surface-variant")
    bars = "".join(bar.format(x=15 + dx, y=y + dy) for dx, dy in ((0, 0), (300, 40)) for y in (26, 31, 36))
    html = (f'<html><body style="margin:0;background:{_c("surface")}">{bars}'
            f'<div style="position:absolute;left:0;top:120px;width:412px;height:1px;background:{_c("outline-variant")}"></div>'
            f'<div style="position:absolute;left:16px;top:160px;width:380px;height:4px;background:{_c("primary")}"></div></body></html>')
    doc = perceive(html_to_png(html, 412, 200))
    short = [n for n in doc.walk() if n.type in ("rect", "line") and max(n.box.w, n.box.h) < 24]
    assert len(short) == 6  # precondition: the glyph reaches mapping as bare strokes
    map_document(doc, ds)
    assert all(n.component is None or n.component.name not in ("Divider", "Progress") for n in short), \
        [(n.component.name, n.box) for n in short if n.component is not None]
    comps = sorted((n.component.name, round(n.box.w)) for n in doc.walk() if n.component is not None and n.box.w >= 24)
    assert ("Divider", 412) in comps and ("Progress", 380) in comps


def test_calibration_params_registered():
    for key in ("map.conf.margin_full", "map.conf.tie_factor", "map.conf.imposed", "map.conf.inferred_critical"):
        assert key in P.all() and key in P.docs()


# --------------------------------------------------------------------------- ingestion: Figma
FIGMA_FIXTURE = os.path.join(FIXTURES_DIR, "figma_fixture")


def _load(name: str) -> dict:
    with open(os.path.join(FIGMA_FIXTURE, name)) as f:
        return json.load(f)


def test_figma_fixture_ingestion():
    ds = ingest_figma.from_figma_json(_load("file.json"), _load("components.json"), _load("styles.json"), _load("variables.json"))
    assert ds.name == "dt-fixture-kit" and ds.fonts == ["Roboto"]
    names = {t.name: t for t in ds.tokens}
    assert names["figma.style.primary"].value == "#6750a4" and names["figma.style.primary"].category == "color"
    assert names["figma.style.label-large"].value["size"] == 14 and names["figma.style.label-large"].value["weight"] == 500
    assert names["figma.style.elevation-level-1"].category == "elevation" and names["figma.style.elevation-level-1"].value["blur"] == 3
    assert names["figma.var.colors-primary"].value == "#6750a4"
    assert names["figma.var.colors-brand"].value == "#6750a4"  # alias resolved
    assert names["figma.var.spacing-md"].category == "spacing" and names["figma.var.spacing-md"].value == 16
    assert names["figma.var.radius-full"].category == "shape"
    assert ds.component_names() == ["Button", "Card/Elevated"]
    btn = ds.component("Button")
    assert btn.key == "b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8" and [v.name for v in btn.variants] == ["Style=Filled", "Style=Outlined"]
    assert btn.variants[0].props["style"] == "filled" and btn.variants[0].props["figma_key"].startswith("f1a2")
    assert btn.signature.radius == "pill" and btn.signature.height == (38, 42)
    assert btn.variants[0].signature.fills == ["figma.style.primary"] and btn.variants[1].signature.fills == ["none"]
    assert isinstance(btn.variants[1].signature.stroke, dict)
    ex = btn.exemplar_node()
    assert ex.box.x == 0 and ex.box.y == 0 and ex.children[0].type == "text" and ex.children[0].text == "Label"
    assert ex.layout.mode == "row" and ex.layout.gap == 8 and ex.layout.padding == (10, 24, 10, 24)
    card = ds.component("Card/Elevated")
    assert card.signature.shadow == "required" and card.signature.radius == 12 and card.signature.fills == ["#f7f2fa"]
    assert card.signature.icon_count == (1, 1) and card.signature.text_count == (2, 2)
    assert [s.name for s in card.slots] == ["headline", "supporting-text", "icon-more-vert"]
    assert DesignSystem.from_dict(ds.to_dict()).to_dict() == ds.to_dict()


def test_figma_ingested_system_matches_ir():
    ds = ingest_figma.from_figma_json(_load("file.json"), _load("components.json"), _load("styles.json"), _load("variables.json"))
    doc = Document.blank(400, 200, "#ffffff")
    btn = Node(type="frame", box=Box(20, 20, 96, 40), radius=20, fills=[Fill.solid("#6750a4")], children=[text("Label", 44, 30, 48, 20)])
    out = Node(type="frame", box=Box(140, 20, 96, 40), radius=20, strokes=[Stroke(color=Color.from_hex("#79747e"), width=1)],
               children=[text("Label", 164, 30, 48, 20, color="#6750a4")])
    doc.root.children = [btn, out]
    map_document(doc, ds)
    assert btn.component.name == "Button" and btn.component.variant["style"] == "filled" and btn.component.confidence >= 0.6
    assert out.component.name == "Button" and out.component.variant["style"] == "outlined"
    assert btn.tokens["fill"] in ("figma.style.primary", "figma.var.colors-primary")
    assert btn.component.props["label"] == "Label"


def test_figma_node_to_ir_shapes():
    n = {"id": "1", "name": "Dot", "type": "ELLIPSE", "absoluteBoundingBox": {"x": 10, "y": 10, "width": 8, "height": 8},
         "fills": [{"type": "SOLID", "color": {"r": 1, "g": 0, "b": 0, "a": 1}}], "effects": [{"type": "DROP_SHADOW", "offset": {"x": 0, "y": 2}, "radius": 4, "color": {"r": 0, "g": 0, "b": 0, "a": 0.3}}]}
    ir = ingest_figma.figma_node_to_ir(n, (10, 10))
    assert ir.type == "ellipse" and ir.fill_color.hex() == "#ff0000" and ir.effects[0].dy == 2 and ir.box.x == 0
    assert ingest_figma.parse_variant_name("Style=Filled, State=Enabled") == {"style": "filled", "state": "enabled"}


# --------------------------------------------------------------------------- ingestion: token libraries
DTCG = {
    "color": {"$type": "color", "primary": {"$value": "#6750A4"}, "brand": {"$value": "{color.primary}"},
              "surface": {"$value": {"colorSpace": "srgb", "components": [0.996, 0.969, 1.0], "alpha": 1, "hex": "#fef7ff"}}},
    "space": {"$type": "dimension", "md": {"$value": "16px"}, "lg": {"$value": {"value": 1.5, "unit": "rem"}}},
    "type": {"body": {"$type": "typography", "$value": {"fontFamily": "Roboto", "fontSize": "16px", "fontWeight": 400, "lineHeight": "24px", "letterSpacing": "0.5px"}}},
    "shadow": {"low": {"$type": "shadow", "$value": {"color": "#00000033", "offsetX": "0px", "offsetY": "1px", "blur": "3px", "spread": "0px"}}},
}


def test_dtcg_ingestion():
    toks = {t.name: t for t in ingest_library.from_dtcg(DTCG)}
    assert toks["color.primary"].value == "#6750a4" and toks["color.primary"].category == "color"
    assert toks["color.brand"].value == "#6750a4"  # alias
    assert toks["color.surface"].value == "#fef7ff"
    assert toks["space.md"].value == 16 and toks["space.lg"].value == 24 and toks["space.md"].category == "dimension"
    assert toks["type.body"].category == "typescale" and toks["type.body"].value == {"family": "Roboto", "size": 16, "weight": 400, "line_height": 24, "tracking": 0.5}
    assert toks["shadow.low"].category == "elevation" and toks["shadow.low"].value["blur"] == 3
    ds = ingest_library.design_system_from_tokens("lib", toks.values())
    assert ds.resolve_color("primary").hex() == "#6750a4" and ds.fonts == ["Roboto"]
    assert snap_color(Color.from_hex("#6750a4"), ds)[0] == "color.primary"


def test_material_theme_builder_ingestion_retheme():
    mtb = {"seed": "#0B57D0", "coreColors": {"primary": "#0B57D0"},
           "schemes": {"light": {"primary": "#0B57D0", "onPrimary": "#FFFFFF", "surfaceContainerHighest": "#E1E3E1", "surface": "#FCFCFF"},
                       "dark": {"primary": "#A8C7FA", "onPrimary": "#062E6F"}},
           "palettes": {"primary": {"0": "#000000", "40": "#0B57D0", "100": "#FFFFFF"}}}
    toks = ingest_library.from_material_theme_builder(mtb)
    names = {t.name: t.value for t in toks}
    assert names["md.sys.color.primary"] == "#0b57d0" and names["md.sys.color.on-primary"] == "#ffffff"
    assert names["md.sys.color.surface-container-highest"] == "#e1e3e1" and names["md.ref.palette.primary40"] == "#0b57d0"
    assert ingest_library.detect_format(mtb) == "material-theme-builder" and ingest_library.detect_format(DTCG) == "dtcg"
    # re-theme the M3 catalog: a blue filled button now matches with the new primary
    themed = ingest_library.design_system_from_tokens("blue", toks, base=material3())
    assert themed.resolve_color("primary").hex() == "#0b57d0" and themed.component("Button") is not None
    btn = Node(type="frame", box=Box(0, 0, 100, 40), radius=20, fills=[Fill.solid("#0b57d0")], children=[text("Go", 24, 10, 40, 20)])
    c = match_node(btn, themed)[0]
    assert c.spec.name == "Button" and c.variant.name == "filled" and c.score >= 0.9
    assert match_node(btn, material3())[0].score < c.score


def test_load_tokens_file(tmp_path):
    p = tmp_path / "tokens.json"
    p.write_text(json.dumps(DTCG))
    assert any(t.name == "color.primary" for t in ingest_library.load_tokens_file(str(p)))


# --------------------------------------------------------------------------- ingestion: screenshots (learned)
def _learn_docs() -> list[Document]:
    docs = []
    for k in range(2):
        d = Document.blank(600, 400, "#fef7ff")
        d.root.children = [filled_button(16, 24 + k * 10), filled_button(200, 100), fab(500, 300 - k * 20),
                           Node(type="frame", box=Box(16, 200, 300, 120), radius=12, fills=[Fill.solid("#f7f2fa")], effects=[Shadow(dy=1, blur=3)],
                                children=[text("Card", 32, 216, 100, 24, size=16, weight=500, lh=24, color="#1d1b20")]),
                           Node(type="frame", box=Box(330, 200, 60 + k * 100, 30), radius=4, fills=[Fill.solid("#%02x0000" % (50 + k * 90))])]
        assign_ids(d.root, prefix=f"d{k}n")
        docs.append(d)
    return docs


def test_learn_from_documents_clusters_recurring_subtrees():
    docs = _learn_docs()
    ds = ingest_screenshots.learn_from_documents(docs, name="learned")
    assert ds.name == "learned" and ds.fonts == ["Roboto"]
    specs = {c.key: c for c in ds.components}
    sizes = sorted((c.meta["members"] for c in ds.components), reverse=True)
    assert sizes[0] == 4  # the four filled buttons cluster together
    big = max(ds.components, key=lambda c: c.meta["members"])
    assert big.signature.radius == "pill" and 38 <= big.signature.height[0] <= 40 <= big.signature.height[1] <= 42
    assert big.signature.text_count == (1, 1) and big.signature.icon_count == (1, 1)
    assert big.exemplar["box"]["x"] == 0 and big.exemplar["box"]["y"] == 0 and len(big.exemplar["children"]) == 2
    assert [s.name for s in big.slots] == ["label", "icon"]
    # distinct things (FAB, card) land in their own clusters (2 members each); the odd red rects do not
    assert {c.meta["members"] for c in ds.components} <= {4, 2}
    assert all(c.meta["members"] >= 2 for c in ds.components)
    # learned tokens
    colors = {t.name: t for t in ds.tokens_by_category("color")}
    assert "learned.color.surface" in colors and colors["learned.color.surface"].value == "#fef7ff"
    assert "learned.color.primary" in colors and colors["learned.color.primary"].value == "#6750a4"
    types = ds.typescale()
    assert "learned.type.14-500" in types and types["learned.type.14-500"]["line_height"] == 20
    assert "learned.shape.full" in ds.shape_scale() and ds.shape_scale().get("learned.shape.r12") == 12
    # the learned system maps a fresh button to the learned component
    doc = Document.blank(600, 400, "#fef7ff")
    b = filled_button(100, 100)
    doc.root.children = [b]
    map_document(doc, ds)
    assert b.component is not None and b.component.key == big.key and b.component.confidence >= 0.6
    assert b.tokens["fill"] == "learned.color.primary"
    assert DesignSystem.from_dict(ds.to_dict()).to_dict() == ds.to_dict()


def test_greedy_cluster_basic():
    labels = ingest_screenshots.greedy_cluster([[0, 0], [0.2, 0], [5, 5], [5.1, 5], [20, 20]], thr=1.0)
    assert labels[0] == labels[1] and labels[2] == labels[3] and len(set(labels)) == 3


def test_from_reference_screenshots_with_perceive():
    """Runs against dt.perceive when module A exists; a stand-in perceive_fn is always exercised."""
    docs = _learn_docs()
    ds = ingest_screenshots.from_reference_screenshots(["a.png", "b.png"], perceive_fn=lambda p: docs[0] if p == "a.png" else docs[1])
    assert ds.meta["sources"] == ["a.png", "b.png"] and ds.components
    perceive_mod = pytest.importorskip("dt.perceive")
    if not hasattr(perceive_mod, "perceive"):
        pytest.skip("dt.perceive.perceive not implemented yet")
    png = os.path.join(ROOT, "out", "mwc_test.png")
    if not os.path.exists(png):
        pytest.skip("out/mwc_test.png missing")
    learned = ingest_screenshots.from_reference_screenshots([png, png])
    assert isinstance(learned, DesignSystem) and learned.tokens
