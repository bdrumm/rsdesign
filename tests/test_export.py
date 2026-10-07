"""Tests for dt.export: IR -> Figma build plan -> headless plugin run (fake Figma API) -> tree checks.

Ground truth for geometry/colors comes from the REAL renderer (dt.render) and the Material Web
reference page (out/mwc_test.html), never from mocked images.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess

import numpy as np
import pytest

from dt.export import RUN_FAKE_JS, font_style, gradient_handles, save_plan, to_build_plan, validate_plan, write_report
from dt.ir import Box, Color, ComponentRef, Document, Fill, GradientStop, Layout, Node, Shadow, Stroke, TextStyle
from dt.params import P


@pytest.fixture
def raw_layout():
    """Export auto-layout exactly as the IR states it (no Figma-layout verification). The vocabulary tests below
    use hand-made layouts that do not reproduce their own child boxes (e.g. a 100px text box around ~40px of
    glyphs first in a row); with verification on (the default) those frames are exported with absolute
    children -- see tests/test_roundtrip.py."""
    old = P["export.figma.verify_layout"]
    P.set("export.figma.verify_layout", False)
    yield
    P.set("export.figma.verify_layout", old)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MWC_HTML = os.path.join(ROOT, "out", "mwc_test.html")
MWC_PNG = os.path.join(ROOT, "out", "mwc_test.png")


# --------------------------------------------------------------------------- fixtures / helpers
def build_doc() -> Document:
    """Frame with an auto-layout row (text, icon, gradient rect), an instance (unknown key), line, ellipse."""
    doc = Document.blank(400, 300, "#fef7ff")
    doc.design_system = "material3"
    row = Node(
        id="row", type="frame", name="Row", box=Box(16, 16, 368, 56), fills=[Fill.solid("#ffffff")],
        radius=(12, 12, 12, 12), clip=True,
        layout=Layout(mode="row", gap=12, padding=(8, 16, 8, 16), align_items="center", justify="start",
                      sizing_h="fixed", sizing_v="hug"),
        effects=[Shadow(Color(0, 0, 0, 0.2), 0, 2, 4, 0)],
        strokes=[Stroke(Color(0, 0, 0), 1, "inside")],
    )
    txt = Node(id="t1", type="text", box=Box(32, 32, 100, 24), text="Hello",
               text_style=TextStyle(family="Google Sans", size=16, weight=500, color=Color(29, 27, 32), valign="middle",
                                    line_height=24, letter_spacing=0.1),
               tokens={"color": "md.sys.color.on-surface"})
    ico = Node(id="i1", type="icon", box=Box(144, 32, 24, 24), icon_name="search", fills=[Fill.solid("#6750a4")])
    grad = Node(id="g1", type="rect", name="Gradient", box=Box(180, 24, 100, 40),
                fills=[Fill(kind="linear", stops=[GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255))], angle=90)],
                radius=(20, 20, 0, 0))
    row.children = [txt, ico, grad]
    inst = Node(id="b1", type="instance", box=Box(16, 100, 120, 40), fills=[Fill.solid("#6750a4")], radius=(20, 20, 20, 20),
                tokens={"fill": "md.sys.color.primary"},
                component=ComponentRef(key="unknown-key", name="Button", library="material3",
                                       variant={"Style": "Filled"}, props={"label": "Save"}, confidence=0.9))
    line = Node(id="l1", type="line", box=Box(16, 160, 368, 1), fills=[Fill.solid("#cac4d0")])
    ell = Node(id="e1", type="ellipse", box=Box(16, 180, 40, 40), fills=[Fill.solid("#00ff00")],
               strokes=[Stroke(Color(0, 0, 0), 2, "center")])
    bordered = Node(id="r2", type="rect", name="Bordered", box=Box(80, 180, 60, 40), fills=[Fill.solid("#ffffff")],
                    strokes=[Stroke(Color(100, 100, 100), 1, "inside", sides=(2, 0, 1, 0))])
    doc.root.children = [row, inst, line, ell, bordered]
    return doc


def run_plugin(plan: dict, tmp_path, components: list[str] | None = None, variables: list[str] | None = None,
               extra: list[str] | None = None) -> dict:
    """Run code.js headlessly (fake Figma API) and return {summary, tree, ...}."""
    path = str(tmp_path / "plan.json")
    save_plan(plan, path)
    cmd = ["node", RUN_FAKE_JS, "--plan", path]
    if components:
        cmd += ["--components", ",".join(components)]
    if variables:
        cmd += ["--variables", ",".join(variables)]
    cmd += extra or []
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"run_fake failed: {r.stdout}\n{r.stderr}"
    out = json.loads(r.stdout)
    assert out["ok"], out
    return out


def walk(node: dict):
    yield node
    for c in node.get("children", []):
        yield from walk(c)


def by_name(tree_root: dict, name: str) -> dict:
    hits = [n for n in walk(tree_root) if n["name"] == name]
    assert hits, f"no node named {name!r}"
    return hits[0]


def rgb255(c: dict) -> tuple[int, int, int]:
    return tuple(int(round(c[k] * 255)) for k in ("r", "g", "b"))


# --------------------------------------------------------------------------- pure conversion
def test_font_style_mapping():
    assert font_style(400) == "Regular"
    assert font_style(500) == "Medium"
    assert font_style(700) == "Bold"
    assert font_style(100) == "Thin"
    assert font_style(400, italic=True) == "Italic"
    assert font_style(700, italic=True) == "Bold Italic"
    assert font_style(450) == "Regular"  # rounds to nearest hundred


def test_gradient_handles_match_css_angles():
    s, e, w = gradient_handles(180, 100, 40)  # CSS top -> bottom
    assert (s["x"], s["y"]) == (0.5, 0.0) and (e["x"], e["y"]) == (0.5, 1.0)
    assert (w["x"], w["y"]) == (1.0, 0.0)  # Figma default third handle
    s, e, _ = gradient_handles(90, 100, 40)  # left -> right
    assert (s["x"], s["y"]) == (0.0, 0.5) and (e["x"], e["y"]) == (1.0, 0.5)
    s, e, _ = gradient_handles(0, 100, 40)  # bottom -> top
    assert (s["x"], s["y"]) == (0.5, 1.0) and (e["x"], e["y"]) == (0.5, 0.0)
    s, e, _ = gradient_handles(45, 100, 100)  # diagonal on a square: corner to corner
    assert math.isclose(s["x"], 0.0, abs_tol=1e-6) and math.isclose(s["y"], 1.0, abs_tol=1e-6)
    assert math.isclose(e["x"], 1.0, abs_tol=1e-6) and math.isclose(e["y"], 0.0, abs_tol=1e-6)


def test_to_build_plan_structure(raw_layout):
    doc = build_doc()
    plan = to_build_plan(doc)
    assert validate_plan(plan) == []
    assert plan["version"] == "1.0" and plan["width"] == 400 and plan["height"] == 300
    assert {"family": "Google Sans", "style": "Medium"} in plan["fonts"]
    assert {"family": "Material Symbols Outlined", "style": "Regular"} in plan["fonts"]
    root = plan["root"]
    assert root["type"] == "FRAME" and (root["x"], root["y"]) == (0, 0)
    assert root["fills"][0]["type"] == "SOLID" and rgb255(root["fills"][0]["color"]) == (0xFE, 0xF7, 0xFF)
    kids = {c["id"]: c for c in root["children"]}
    row = kids["row"]
    # parent-relative coordinates
    assert (row["x"], row["y"], row["width"], row["height"]) == (16, 16, 368, 56)
    t1 = row["children"][0]
    assert (t1["x"], t1["y"]) == (16, 16)
    # auto-layout
    assert row["layoutMode"] == "HORIZONTAL" and row["itemSpacing"] == 12
    assert (row["paddingTop"], row["paddingRight"], row["paddingBottom"], row["paddingLeft"]) == (8, 16, 8, 16)
    assert row["primaryAxisAlignItems"] == "MIN" and row["counterAxisAlignItems"] == "CENTER"
    assert row["primaryAxisSizingMode"] == "FIXED" and row["counterAxisSizingMode"] == "AUTO"
    assert row["layoutWrap"] == "NO_WRAP" and row["clipsContent"] is True and row["cornerRadius"] == 12
    assert t1["layoutSizingHorizontal"] == "HUG" and t1["layoutSizingVertical"] == "HUG"
    # stroke + effect
    assert row["strokeWeight"] == 1 and row["strokeAlign"] == "INSIDE"
    eff = row["effects"][0]
    assert eff["type"] == "DROP_SHADOW" and eff["offset"] == {"x": 0, "y": 2} and eff["radius"] == 4
    assert math.isclose(eff["color"]["a"], 0.2)
    # text
    assert t1["type"] == "TEXT" and t1["characters"] == "Hello"
    assert t1["fontName"] == {"family": "Google Sans", "style": "Medium"} and t1["fontSize"] == 16
    assert t1["lineHeight"] == {"value": 24, "unit": "PIXELS"} and t1["letterSpacing"] == {"value": 0.1, "unit": "PIXELS"}
    # single-line left-aligned text is auto-width (never wraps in Figma, mirroring dt.render)
    assert t1["textAlignHorizontal"] == "LEFT" and t1["textAlignVertical"] == "CENTER" and t1["textAutoResize"] == "WIDTH_AND_HEIGHT"
    assert rgb255(t1["fills"][0]["color"]) == (29, 27, 32)
    assert t1["variableBindings"] == {"fills": "md.sys.color.on-surface"}
    # icon
    i1 = row["children"][1]
    assert i1["type"] == "TEXT" and i1["isIcon"] and i1["characters"] == "search" and i1["fontSize"] == 24
    assert i1["fontName"] == {"family": "Material Symbols Outlined", "style": "Regular"}
    assert rgb255(i1["fills"][0]["color"]) == (0x67, 0x50, 0xA4)
    # gradient rect with per-corner radius
    g1 = row["children"][2]
    assert g1["type"] == "RECTANGLE"
    paint = g1["fills"][0]
    assert paint["type"] == "GRADIENT_LINEAR" and len(paint["gradientStops"]) == 2
    assert paint["gradientHandlePositions"][0] == {"x": 0.0, "y": 0.5} and paint["gradientHandlePositions"][1] == {"x": 1.0, "y": 0.5}
    assert (g1["topLeftRadius"], g1["topRightRadius"], g1["bottomRightRadius"], g1["bottomLeftRadius"]) == (20, 20, 0, 0)
    # instance
    b1 = kids["b1"]
    assert b1["type"] == "INSTANCE" and b1["componentKey"] == "unknown-key"
    assert b1["variantProperties"] == {"Style": "Filled"} and b1["props"] == {"label": "Save"}
    assert b1["variableBindings"] == {"fills": "md.sys.color.primary"}
    fb = b1["fallback"]
    assert fb["type"] == "FRAME" and (fb["x"], fb["y"], fb["width"], fb["height"]) == (16, 100, 120, 40)
    assert fb["cornerRadius"] == 20 and fb["children"][0]["type"] == "TEXT" and fb["children"][0]["characters"] == "Save"
    # line -> rectangle, ellipse, per-side strokes
    assert kids["l1"]["type"] == "RECTANGLE" and kids["l1"]["height"] == 1 and rgb255(kids["l1"]["fills"][0]["color"]) == (0xCA, 0xC4, 0xD0)
    assert kids["e1"]["type"] == "ELLIPSE" and kids["e1"]["strokeAlign"] == "CENTER" and kids["e1"]["strokeWeight"] == 2
    r2 = kids["r2"]
    assert (r2["strokeTopWeight"], r2["strokeRightWeight"], r2["strokeBottomWeight"], r2["strokeLeftWeight"]) == (2, 0, 1, 0)
    assert r2["strokeWeight"] == 2
    # determinism + JSON round trip
    assert json.loads(json.dumps(plan)) == to_build_plan(build_doc())


def test_stretch_and_column_layout_sizing(raw_layout):
    doc = Document.blank(200, 200)
    col = Node(id="col", type="frame", box=Box(0, 0, 200, 200),
               layout=Layout(mode="column", gap=4, align_items="stretch", justify="space-between", sizing_h="hug", sizing_v="fixed"))
    inner = Node(id="inner", type="frame", box=Box(0, 0, 200, 50), layout=Layout(mode="row", sizing_h="hug", sizing_v="hug"))
    leaf = Node(id="leaf", type="rect", box=Box(0, 60, 200, 50))
    col.children = [inner, leaf]
    doc.root.children = [col]
    plan = to_build_plan(doc)
    assert validate_plan(plan) == []
    c = plan["root"]["children"][0]
    assert c["layoutMode"] == "VERTICAL" and c["primaryAxisAlignItems"] == "SPACE_BETWEEN" and c["counterAxisAlignItems"] == "MIN"
    assert c["primaryAxisSizingMode"] == "FIXED" and c["counterAxisSizingMode"] == "AUTO"  # main=v fixed, cross=h hug
    inner_p, leaf_p = c["children"]
    assert inner_p["layoutSizingHorizontal"] == "FILL" and inner_p["layoutSizingVertical"] == "HUG"
    assert leaf_p["layoutSizingHorizontal"] == "FILL" and leaf_p["layoutSizingVertical"] == "FIXED"


def test_validate_plan_catches_errors():
    plan = to_build_plan(build_doc())
    bad = copy.deepcopy(plan)
    bad["version"] = "2.0"
    bad["root"]["children"][0]["fills"][0]["color"]["r"] = 1.5
    bad["root"]["children"][0]["children"][0]["fontSize"] = 0
    bad["root"]["children"][0]["layoutMode"] = "DIAGONAL"
    bad["root"]["children"][1]["componentKey"] = ""
    bad["root"]["children"][2]["width"] = float("nan")
    bad["root"]["children"][3]["id"] = "row"  # duplicate id
    bad["root"]["children"][4]["children"] = []  # children on a RECTANGLE
    errs = validate_plan(bad)
    joined = "\n".join(errs)
    for needle in ("version", "color.r", "fontSize", "layoutMode", "componentKey", "width", "duplicate id", "only FRAME"):
        assert needle in joined, f"missing error for {needle}: {errs}"
    assert validate_plan({"version": "1.0"}) and validate_plan("nope") == ["plan must be an object"]
    assert validate_plan(plan) == []


# --------------------------------------------------------------------------- headless plugin
def test_plugin_builds_tree_with_fallback(tmp_path, raw_layout):
    doc = build_doc()
    plan = to_build_plan(doc)
    out = run_plugin(plan, tmp_path, variables=["md.sys.color.on-surface"])
    summary, (page_root,) = out["summary"], out["tree"]
    assert summary["created"] == 10 and summary["fallbacks"] == 1 and summary["instances"] == 0 and summary["texts"] == 3
    assert any("unknown-key" in w and "fallback" in w for w in summary["warnings"])
    assert "Google Sans / Medium" in summary["fontsMissing"]
    assert any(w.startswith("font Google Sans / Medium missing; used Roboto / Medium") for w in summary["warnings"])
    assert out["selection"] == [page_root["id"]]

    assert page_root["type"] == "FRAME" and page_root["name"] == "Screen"
    assert (page_root["width"], page_root["height"]) == (400, 300)
    assert rgb255(page_root["fills"][0]["color"]) == (0xFE, 0xF7, 0xFF)
    assert [c["name"] for c in page_root["children"]] == ["Row", "Button/Filled", "Line", "Ellipse", "Bordered"]

    row = by_name(page_root, "Row")
    assert (row["x"], row["y"], row["width"], row["height"]) == (16, 16, 368, 56)
    assert row["layoutMode"] == "HORIZONTAL" and row["itemSpacing"] == 12 and row["paddingLeft"] == 16
    assert row["counterAxisAlignItems"] == "CENTER" and row["primaryAxisSizingMode"] == "FIXED"
    assert row["cornerRadius"] == 12 and row["clipsContent"] is True
    assert row["effects"][0]["type"] == "DROP_SHADOW" and row["strokeWeight"] == 1
    assert len(row["children"]) == 3

    hello = by_name(page_root, "Hello")
    assert hello["type"] == "TEXT" and hello["characters"] == "Hello" and hello["fontSize"] == 16
    assert hello["fontName"] == {"family": "Roboto", "style": "Medium"}  # fallback family, same weight
    assert hello["textAutoResize"] == "WIDTH_AND_HEIGHT"  # auto-width: the fake measures nothing, Figma sizes it
    assert hello["lineHeight"] == {"value": 24, "unit": "PIXELS"} and hello["textAlignVertical"] == "CENTER"
    assert hello["fills"][0]["boundVariables"]["color"]["id"] == "VariableID:1"  # bound by name

    icon = by_name(page_root, "icon/search")
    assert icon["characters"] == "search" and icon["fontName"]["family"] == "Material Symbols Outlined"
    assert rgb255(icon["fills"][0]["color"]) == (0x67, 0x50, 0xA4)

    grad = by_name(page_root, "Gradient")
    assert grad["type"] == "RECTANGLE" and grad["fills"][0]["type"] == "GRADIENT_LINEAR"
    assert (grad["topLeftRadius"], grad["bottomRightRadius"]) == (20, 0)

    fb = by_name(page_root, "Button/Filled")  # unknown component key -> fallback FRAME named after the instance
    assert fb["type"] == "FRAME" and (fb["x"], fb["y"], fb["width"], fb["height"]) == (16, 100, 120, 40)
    assert fb["cornerRadius"] == 20 and rgb255(fb["fills"][0]["color"]) == (0x67, 0x50, 0xA4)
    assert fb["children"][0]["type"] == "TEXT" and fb["children"][0]["characters"] == "Save"
    assert any("md.sys.color.primary" in w and "not found" in w for w in summary["warnings"])

    line = by_name(page_root, "Line")
    assert line["type"] == "RECTANGLE" and (line["width"], line["height"]) == (368, 1)
    ell = by_name(page_root, "Ellipse")
    assert ell["type"] == "ELLIPSE" and ell["strokeAlign"] == "CENTER" and ell["strokeWeight"] == 2
    bordered = by_name(page_root, "Bordered")
    assert (bordered["strokeTopWeight"], bordered["strokeBottomWeight"], bordered["strokeLeftWeight"]) == (2, 1, 0)
    assert out["posted"][-1]["type"] == "done"


def test_plugin_uses_component_when_key_known(tmp_path):
    plan = to_build_plan(build_doc())
    out = run_plugin(plan, tmp_path, components=["unknown-key"], variables=["md.sys.color.primary"])
    summary, (page_root,) = out["summary"], out["tree"]
    assert summary["instances"] == 1 and summary["fallbacks"] == 0
    inst = by_name(page_root, "Button/Filled")
    assert inst["type"] == "INSTANCE" and inst["componentKey"] == "unknown-key"
    assert (inst["x"], inst["y"], inst["width"], inst["height"]) == (16, 100, 120, 40)
    assert inst["componentProperties"]["Style"]["value"] == "Filled"
    assert inst["componentProperties"]["label#1:0"]["value"] == "Save"
    assert inst["children"][0]["characters"] == "Save"
    # instance fills cannot be bound (component owns paints) -> warning, no crash
    assert any("md.sys.color.primary" in w for w in summary["warnings"])


def test_plugin_without_variables_api(tmp_path):
    out = run_plugin(to_build_plan(build_doc()), tmp_path, extra=["--no-variables"])
    assert out["summary"]["created"] == 10
    assert any("not found" in w for w in out["summary"]["warnings"])


# --------------------------------------------------------------------------- real renderer ground truth
def test_plan_paints_match_rendered_pixels(tmp_path):
    """Render the IR with the real renderer; plan paint colors must match the pixels at node centers."""
    from dt.render.screenshot import render_doc

    doc = build_doc()
    rgb = render_doc(doc, out_path=str(tmp_path / "render.png"))
    plan = to_build_plan(doc)
    assert rgb.shape[:2] == (plan["height"], plan["width"])

    def abs_center(node: Node) -> tuple[int, int]:
        return int(node.box.cx), int(node.box.cy)

    # solid shapes: ellipse, bordered rect, root background, instance fallback
    checks = {"e1": (0, 255, 0), "r2": (255, 255, 255), "b1": (0x67, 0x50, 0xA4)}
    plan_nodes = {n["id"]: n for n in walk(plan["root"])}
    for nid, expected in checks.items():
        ir = doc.find(nid)
        pn = plan_nodes[nid]
        fills = pn["fills"] if pn["type"] != "INSTANCE" else pn["fallback"]["fills"]
        assert rgb255(fills[0]["color"]) == expected
        cx, cy = abs_center(ir)
        px = rgb[cy, cx]
        if nid == "b1":  # label text sits at the exact center; sample off-center inside the pill
            px = rgb[int(ir.box.y + 6), int(ir.box.x + 30)]
        assert Color(*px).delta_e(Color(*expected)) < 3, (nid, px, expected)
    # root background
    bg = rgb[int(doc.height) - 5, int(doc.width) - 5]
    assert Color(*bg).delta_e(Color(0xFE, 0xF7, 0xFF)) < 2
    # gradient: left edge red-ish, right edge blue-ish (angle 90 = left -> right), matching the plan handles
    g = doc.find("g1")
    left = rgb[int(g.box.cy), int(g.box.x) + 3]
    right = rgb[int(g.box.cy), int(g.box.x2) - 4]
    assert left[0] > 180 and left[2] < 80 and right[2] > 180 and right[0] < 80
    paint = plan_nodes["g1"]["fills"][0]
    assert paint["gradientHandlePositions"][0]["x"] < paint["gradientHandlePositions"][1]["x"]


def test_mwc_reference_instances(tmp_path):
    """DOM boxes of the Material Web reference page -> instance nodes -> plan -> plugin tree sizes match."""
    if not os.path.exists(MWC_HTML):
        pytest.skip("reference page missing")
    from dt.common.image import load_rgb
    from dt.render.screenshot import render_url

    ref = load_rgb(MWC_PNG)
    H, W = ref.shape[:2]
    script = """(() => Array.from(document.querySelectorAll('body > *')).map(el => {
        const r = el.getBoundingClientRect();
        return {tag: el.tagName.toLowerCase(), x: r.x, y: r.y, w: r.width, h: r.height,
                text: (el.textContent || '').trim(), label: el.getAttribute('label')};
    }))()"""
    _, boxes = render_url("file://" + MWC_HTML, W, H, script=script)
    assert len(boxes) >= 8
    doc = Document.blank(W, H, "#fef7ff")
    keys = []
    for i, b in enumerate(boxes):
        key = f"m3/{b['tag'][3:]}"
        keys.append(key)
        doc.root.children.append(Node(
            id=f"c{i}", type="instance", name=b["tag"], box=Box(b["x"], b["y"], b["w"], b["h"]),
            fills=[Fill.solid("#6750a4")], radius=(20, 20, 20, 20),
            component=ComponentRef(key=key, name=b["tag"], library="material3",
                                   props={"label": b["text"] or b["label"] or ""}, confidence=1.0),
        ))
    plan = to_build_plan(doc)
    assert validate_plan(plan) == []
    assert (plan["width"], plan["height"]) == (W, H)
    out = run_plugin(plan, tmp_path, components=sorted(set(keys)))
    summary, (page_root,) = out["summary"], out["tree"]
    assert summary["instances"] == len(boxes) and summary["fallbacks"] == 0
    for b, node in zip(boxes, page_root["children"]):
        assert node["type"] == "INSTANCE" and node["name"] == b["tag"]
        assert abs(node["x"] - b["x"]) < 0.01 and abs(node["width"] - b["w"]) < 0.01 and abs(node["height"] - b["h"]) < 0.01
    filled = page_root["children"][0]
    assert filled["children"][0]["characters"] == "Filled"
    # the filled button really is primary-colored in the reference screenshot
    fb = boxes[0]
    px = ref[int(fb["y"] + 6), int(fb["x"] + 8)]
    assert Color(*px).delta_e(Color(0x67, 0x50, 0xA4)) < 6


# --------------------------------------------------------------------------- report
def test_write_report(tmp_path):
    from dt.render.screenshot import render_doc

    doc = build_doc()
    rendered = render_doc(doc, out_path=str(tmp_path / "rendered.png"))
    history = [
        {"iter": 0, "move": "shift", "node": "row", "loss_before": 0.5, "loss_after": 0.4, "accepted": True, "detail": "dx=2"},
        {"iter": 1, "move": "recolor", "node": "e1", "loss_before": 0.4, "loss_after": 0.45, "accepted": False},
        {"iter": 2, "move": "resize", "node": "g1", "loss_before": 0.4, "loss_after": 0.25, "accepted": True},
    ]
    metrics = {"pixel": {"ssim": 0.93, "mean_de": 1.2}, "structure": {"node_recall": 0.8}}
    out_dir = tmp_path / "report"
    path = write_report(str(out_dir), doc, history, metrics,
                        images={"target": str(tmp_path / "rendered.png"), "diff": np.zeros_like(rendered)})
    text = open(path).read()
    assert "# designtranslator run report" in text
    assert "| Nodes | 9 |" in text and "| Accepted moves | 2 |" in text and "50.0%" in text
    assert "pixel.ssim" in text and "0.9300" in text
    assert "| 0 | shift | row |" in text and "recolor" not in text.split("## Node tree")[0]
    assert "frame:row" in text and 'text:t1' in text
    assert "![target](target.png)" in text and "![diff](diff.png)" in text
    assert (out_dir / "target.png").exists() and (out_dir / "diff.png").exists()
    assert (out_dir / "ir.json").exists() and (out_dir / "build_plan.json").exists() and (out_dir / "metrics.json").exists()
    assert Document.load(str(out_dir / "ir.json")).root.count() == 9
    # tolerant inputs: floats and empty history
    path2 = write_report(str(tmp_path / "r2"), doc, [0.5, 0.3], {}, None)
    t2 = open(path2).read()
    assert "| Final loss | 0.3000 |" in t2 and "_No images._" in t2


def test_line_mode_and_token_refs(tmp_path):
    """P['export.figma.line_mode']='line' emits Figma LINE nodes for thin horizontal lines; unknown tokens -> tokenRefs."""
    from dt.params import P

    doc = Document.blank(100, 100)
    thin = Node(id="thin", type="line", box=Box(10, 20, 80, 1), fills=[Fill.solid("#cac4d0")], tokens={"typescale": "md.sys.typescale.body"})
    thick = Node(id="thick", type="line", box=Box(10, 40, 80, 4), fills=[Fill.solid("#000000")])
    vertical = Node(id="vert", type="line", box=Box(50, 50, 1, 40), strokes=[Stroke(Color(255, 0, 0), 1)])
    doc.root.children = [thin, thick, vertical]
    P.set("export.figma.line_mode", "line")
    try:
        plan = to_build_plan(doc)
    finally:
        P.reset("export.figma.line_mode")
    assert validate_plan(plan) == []
    t, k, v = plan["root"]["children"]
    assert t["type"] == "LINE" and (t["x"], t["y"], t["width"], t["height"]) == (10, 20.5, 80, 0)
    assert t["strokeWeight"] == 1 and t["strokeAlign"] == "CENTER" and rgb255(t["strokes"][0]["color"]) == (0xCA, 0xC4, 0xD0)
    assert t["tokenRefs"] == {"typescale": "md.sys.typescale.body"} and "variableBindings" not in t
    assert k["type"] == "RECTANGLE" and k["height"] == 4  # too thick for a LINE
    assert v["type"] == "RECTANGLE" and rgb255(v["fills"][0]["color"]) == (255, 0, 0)  # vertical: stroke color as fill
    out = run_plugin(plan, tmp_path)
    page_root = out["tree"][0]
    line = page_root["children"][0]
    assert line["type"] == "LINE" and line["height"] == 0 and line["width"] == 80 and line["strokeWeight"] == 1
    assert to_build_plan(doc)["root"]["children"][0]["type"] == "RECTANGLE"  # default mode restored


def test_embedded_raster_becomes_figma_image(tmp_path):
    """An as-is raster (image node with a data: URI crop) is exported with its bytes and the plugin
    registers it as a real Figma image instead of a placeholder."""
    import base64
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from dt.ir import Box, Document, Node

    buf = BytesIO()
    Image.fromarray(np.full((12, 20, 3), 200, np.uint8)).save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    doc = Document.blank(100, 60)
    doc.root.children.append(Node(id="art", type="image", name="illustration", box=Box(10, 10, 20, 12), image_ref=uri,
                                  meta={"rasterised": True}))
    plan = to_build_plan(doc)
    assert validate_plan(plan) == []
    out = run_plugin(plan, tmp_path)
    (page_root,) = out["tree"]
    art = by_name(page_root, "illustration")
    paint = art["fills"][-1]
    assert paint["type"] == "IMAGE" and str(paint.get("imageHash", "")).startswith("img_"), paint
