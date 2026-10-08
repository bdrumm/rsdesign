"""Figma channel: what a designer changed in the frame the plugin built becomes feedback.

The plugin's "Export corrections" action (dt/export/figma_plugin/code.js) serialises the selected rsdesign frame
(``rsdesign.figma-corrections/1``: node tree with ``pluginData['dt.id']``, geometry, fills, characters, fonts,
instance main components, plus the plan's node ids stamped at build time). :func:`diff_export` converts that tree
back to IR with :func:`dt.validate.roundtrip.plan_tree_to_ir` and diffs it against the run's ``ir.mapped.json``:

* a node with the same ``dt.id`` whose characters / icon glyph / main component / position or size / fill changed
  -> ``text`` / ``icon`` / ``component`` / ``geometry`` / ``color`` items;
* a plan node that is gone (or hidden) while its parent is still there -> ``extra``;
* a node without ``dt.id`` under an rsdesign node -> ``missing`` (with its box, type, text, fill, component).

``run_session`` drives the real plugin code on fake_figma.js with scripted edits (tests, agents without Figma).
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional

from dt.ir import Color, Document, Node
from dt.params import P, register

EXPORT_SCHEMA = "rsdesign.figma-corrections/1"
SESSION_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figma_session.js")

register("feedback.figma.geom_tol", 2.0, "px a node may move / resize in Figma before it counts as a geometry correction",
         (0.5, 10.0))
register("feedback.figma.color_de", 3.0, "CIE76 ΔE a fill may change in Figma before it counts as a colour correction",
         (1.0, 15.0))

_ADDED = "__added_"


def run_session(plan: dict, edits: list[dict], components: Optional[list[str]] = None,
                component_names: Optional[dict[str, str]] = None, fonts: Optional[list[dict]] = None) -> dict:
    """Build ``plan`` with the plugin on fake Figma, apply ``edits`` (see figma_session.js), export corrections.
    Returns ``{export, summary, build}``; raises RuntimeError on failure."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH (needed to run the Figma plugin headlessly)")
    if fonts is None:
        fonts = [{"family": f["family"], "style": f["style"]} for f in plan.get("fonts", [])] + \
                [{"family": "Roboto", "style": "Regular"}, {"family": "Inter", "style": "Regular"}]
    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, "session.json")
        with open(sp, "w") as f:
            json.dump({"plan": plan, "edits": edits, "components": components or [], "componentNames": component_names or {},
                       "fonts": fonts}, f)
        r = subprocess.run([node, SESSION_JS, sp], capture_output=True, text=True, timeout=120)
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # pragma: no cover - surfaced to the caller
        raise RuntimeError(f"figma_session.js produced no JSON (rc={r.returncode}): {r.stdout[-2000:]} {r.stderr[-2000:]}") from e
    if not out.get("ok"):
        raise RuntimeError(f"figma session failed: {out.get('error')}")
    return out


def _walk(n: dict, parent: Optional[dict] = None):
    yield n, parent
    for c in n.get("children") or []:
        yield from _walk(c, n)


def _did(n: dict) -> Optional[str]:
    pd = n.get("pluginData")
    return pd.get("dt.id") if isinstance(pd, dict) and pd.get("dt.id") else None


def _norm_variant(v: dict) -> dict[str, str]:
    return {str(k).split("#")[0].strip().lower(): str(x).strip().lower() for k, x in (v or {}).items()}


def _component_of(raw: dict) -> Optional[dict]:
    m = raw.get("mainComponent") if raw.get("type") == "INSTANCE" else None
    if not m:
        return None
    name = m.get("setName") or m.get("name") or ""
    if "=" in name and not m.get("setName"):  # a variant's own name ("Style=Filled") says nothing about the component
        name = ""
    return {"name": name.strip() or None, "key": m.get("key"), "variant": _norm_variant(raw.get("variantProperties") or {})}


def _same_name(a: Optional[str], b: Optional[str]) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _box(n: Node) -> list[float]:
    return [round(v, 2) for v in (n.box.x, n.box.y, n.box.w, n.box.h)]


def _color_of(n: Node) -> Optional[Color]:
    if n.type == "text":
        return n.text_style.color if n.text_style is not None else None
    return n.fill_color


def pick_frame(doc: Document, export: dict) -> dict:
    if not isinstance(export, dict) or export.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"not an rsdesign Figma corrections export (schema {EXPORT_SCHEMA!r})")
    frames = export.get("frames") or []
    if not frames:
        raise ValueError("the export has no frames")
    ids = {n.id for n in doc.walk()}
    return max(frames, key=lambda fr: len(ids & set(fr.get("planIds") or [])))


def plan_as_tree(plan: dict, instance_ids: Optional[set] = None) -> dict:
    """The build plan as the node tree the plugin creates from it (``dt.id`` plugin data): instances whose id is
    in ``instance_ids`` (resolved from the library in that file) stay atomic, the others become the plan's
    fallback frame, exactly as the plugin builds them. Converted with the same Figma layout model as the export,
    it is the "nobody edited it" baseline, so auto-layout re-positioning or a fallback frame's paint is never
    mistaken for a designer's edit."""
    inst = instance_ids or set()

    def conv(spec: dict) -> dict:
        if spec.get("type") == "INSTANCE" and str(spec.get("id")) not in inst and isinstance(spec.get("fallback"), dict):
            fb = dict(spec["fallback"])
            fb["id"] = spec.get("id")
            for k in ("x", "y", "width", "height", "visible", "opacity", "layoutSizingHorizontal", "layoutSizingVertical"):
                if k in spec:
                    fb[k] = spec[k]
            return conv(fb)
        n = {k: v for k, v in spec.items() if k not in ("children", "fallback", "id")}
        n["pluginData"] = {"dt.id": str(spec["id"])} if spec.get("id") is not None else {}
        if spec.get("type") == "INSTANCE":
            return n
        n["children"] = [conv(c) for c in spec.get("children") or []]
        return n
    return conv(plan["root"])


def diff_export(doc: Document, export: dict, plan: Optional[dict] = None) -> list[dict]:
    """Feedback items (unnormalised) for every difference between the exported Figma frame and ``doc``.
    ``plan`` (the run's figma_plan.json) gives the geometry / colour baseline; without it the IR is used."""
    from dt.validate.roundtrip import plan_tree_to_ir
    frame = pick_frame(doc, export)
    plan_ids = set(frame.get("planIds") or [])
    tree = copy.deepcopy(frame["tree"])
    raw_by_id: dict[str, dict] = {}
    added_top: set[str] = set()
    k = 0
    for n, parent in _walk(tree):
        did = _did(n)
        if did is None:  # a node the designer added: give it an id the round trip keeps
            k += 1
            did = f"{_ADDED}{k}"
            n.setdefault("pluginData", {})["dt.id"] = did
            if parent is None or not str(_did(parent) or "").startswith(_ADDED):
                added_top.add(did)
        raw_by_id[did] = n
    fig = plan_tree_to_ir(tree, doc.width, doc.height, images={}, measure=False)
    fig_by_id = {n.id: n for n in fig.walk()}
    base_by_id = {n.id: n for n in doc.walk()}
    if plan is not None:
        base = plan_tree_to_ir(plan_as_tree(plan, {i for i, r in raw_by_id.items() if r.get('type') == 'INSTANCE'}), doc.width, doc.height, images={}, measure=False)
        base_by_id.update({n.id: n for n in base.walk() if n.id in base_by_id})
    fig_parent = {c.id: p for p in fig.walk() for c in p.children}
    geom_tol, col_tol = float(P["feedback.figma.geom_tol"]), float(P["feedback.figma.color_de"])
    items: list[dict] = []
    for n in doc.walk():
        if n is doc.root:
            continue
        f = fig_by_id.get(n.id)
        if f is None:
            parent = doc.parent_of(n.id)
            praw = raw_by_id.get(parent.id) if parent is not None else None
            parent_kept = parent is doc.root or (parent is not None and parent.id in fig_by_id and (praw or {}).get("type") != "INSTANCE")
            if n.id in plan_ids and parent_kept:
                items.append({"kind": "extra", "node_id": n.id, "note": "deleted in Figma"})
            continue
        raw = raw_by_id.get(n.id, {})
        if raw.get("visible") is False and n.visible:
            items.append({"kind": "extra", "node_id": n.id, "note": "hidden in Figma"})
            continue
        if n.type == "text" and f.type == "text" and (f.text or "").strip() != (n.text or "").strip():
            items.append({"kind": "text", "node_id": n.id, "value": {"text": f.text or ""}, "note": "edited in Figma"})
        if n.type == "icon" and f.type == "icon" and (f.icon_name or "") != (n.icon_name or ""):
            items.append({"kind": "icon", "node_id": n.id, "value": {"icon_name": f.icon_name}, "note": "edited in Figma"})
        comp = _component_of(raw)
        if comp is not None and comp["name"]:
            cur = n.component
            if cur is None or not _same_name(cur.name, comp["name"]):
                items.append({"kind": "component", "node_id": n.id, "value": {"name": comp["name"], "variant": comp["variant"]},
                              "note": "swapped in Figma"})
            else:
                mine = _norm_variant(cur.variant)
                common = set(mine) & set(comp["variant"])
                if any(mine[k2] != comp["variant"][k2] for k2 in common):
                    items.append({"kind": "component", "node_id": n.id, "value": {"name": cur.name, "variant": comp["variant"]},
                                  "note": "variant changed in Figma"})
        bn = base_by_id.get(n.id, n)
        a, b = bn.box, f.box
        # displacement relative to the parent's: moving a button moves its label, which is not a label edit
        fp, dp = fig_parent.get(n.id), doc.parent_of(n.id)
        pdx = pdy = 0.0
        if fp is not None and dp is not None and dp.id == fp.id:
            pb = base_by_id.get(dp.id, dp).box
            pdx, pdy = fp.box.x - pb.x, fp.box.y - pb.y
        dx, dy = (b.x - a.x) - pdx, (b.y - a.y) - pdy
        resized = 0.0 if n.type in ("text", "icon") else max(abs(a.w - b.w), abs(a.h - b.h))
        # Figma sizes text from its glyphs: only a text's position is the designer's
        if max(abs(dx), abs(dy), resized) > geom_tol:
            new_box = [round(n.box.x + (b.x - a.x), 2), round(n.box.y + (b.y - a.y), 2),
                       round(n.box.w + (0.0 if n.type in ("text", "icon") else b.w - a.w), 2),
                       round(n.box.h + (0.0 if n.type in ("text", "icon") else b.h - a.h), 2)]
            items.append({"kind": "geometry", "node_id": n.id, "value": {"box": new_box}, "note": "moved / resized in Figma"})
        if raw.get("type") != "INSTANCE":  # an instance paints with its library component, not with our fills
            ca, cb = _color_of(bn), _color_of(f)
            if (ca is None) != (cb is None) or (ca is not None and cb is not None and ca.delta_e(cb) > col_tol):
                items.append({"kind": "color", "node_id": n.id, "value": {"color": cb.hex() if cb is not None else None},
                              "note": "recoloured in Figma"})
    for did in sorted(added_top, key=lambda s: int(s[len(_ADDED):])):
        f = fig_by_id.get(did)
        if f is None:
            continue
        raw = raw_by_id[did]
        value: dict[str, Any] = {"type": {"text": "text", "icon": "icon", "rect": "rect", "ellipse": "ellipse",
                                          "line": "line"}.get(f.type, "frame")}
        if f.type == "text" and f.text:
            value["text"] = f.text
        if f.type == "icon" and f.icon_name:
            value["icon_name"] = f.icon_name
        c = _color_of(f)
        if c is not None:
            value["fill"] = c.hex()
        comp = _component_of(raw)
        if comp is not None and comp["name"]:
            value["component"] = {"name": comp["name"], "variant": comp["variant"]}
        items.append({"kind": "missing", "node_id": None, "box": _box(f), "value": value,
                      "note": f"added in Figma ({raw.get('name') or raw.get('type')})"})
    return items


def from_figma(run_dir: str, export: Any, rating: Any = None, consent: Optional[dict] = None,
               home: Optional[str] = None) -> dict:
    """``dt feedback from-figma <run_dir> export.json`` -> a local bundle (channel ``figma``)."""
    from dt.feedback.capture import create_bundle
    if isinstance(export, str):
        with open(export) as f:
            export = json.load(f)
    doc = Document.load(os.path.join(run_dir, "ir.mapped.json"))
    pp = os.path.join(run_dir, "figma_plan.json")
    plan = None
    if os.path.exists(pp):
        with open(pp) as f:
            plan = json.load(f)
    items = diff_export(doc, export, plan)
    return create_bundle(run_dir, items, "figma", rating=rating, consent=consent, home=home)


__all__ = ["EXPORT_SCHEMA", "run_session", "diff_export", "from_figma", "pick_frame", "plan_as_tree"]
