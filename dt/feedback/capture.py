"""Capture channels: turn a consumer's corrections into a local feedback bundle.

* ``add_corrections(run_dir, corrections)``   -- ``dt feedback add`` / review.html downloads / MCP ``submit_feedback``
* ``log_decisions(run_dir, decisions, base)``  -- every answer given through ``dt apply-decisions``
* ``dt.feedback.figma.from_figma``             -- a Figma "Export corrections" file diffed against the run

Each item is enriched with the *context* the model saw (IR type, box, current component, and for containers the
matcher's learned signature), so learning never depends on the run directory still existing.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from dt.feedback.store import (CORRECTIONS_SCHEMA, LICENSE_NOTE, SCHEMA, bundle_dir, default_consent, load_bundle,
                               load_target, new_bundle_id, normalize_item, now_iso, run_info, save_bundle)
from dt.ir import Box, Color, ComponentRef, Document, Fill, Node, TextStyle

_DS_CACHE: dict[str, Any] = {}
CONTAINER_SKIP = ("text", "icon", "image", "line")


def design_system_for(name: Optional[str]):
    """The design system a run was mapped with (``material3`` when unknown or unloadable)."""
    key = name or "material3"
    if key not in _DS_CACHE:
        from dt.pipeline import resolve_design_system
        try:
            _DS_CACHE[key] = resolve_design_system(key)
        except Exception:  # noqa: BLE001 - e.g. a figma: spec without a token; signatures still work on M3 roles
            _DS_CACHE[key] = resolve_design_system("material3")
    return _DS_CACHE[key]


def node_context(node: Node, ds) -> dict:
    """What the model saw for ``node``: type, box, component, and (containers) the learned-rule signature."""
    ctx: dict[str, Any] = {"type": node.meta.get("orig_type", node.type) if node.type == "instance" else node.type,
                           "box": [round(v, 2) for v in (node.box.x, node.box.y, node.box.w, node.box.h)],
                           "component": ({"name": node.component.name, "variant": dict(node.component.variant)}
                                         if node.component is not None else None)}
    if ctx["type"] not in CONTAINER_SKIP and ds is not None:
        from dt.mapping.matcher import learned_signature
        h, feat = learned_signature(node, ds)
        ctx["signature_hash"], ctx["features"] = h, feat
    return ctx


def _rating(r: Any) -> Optional[int]:
    if r is None or r == "":
        return None
    if isinstance(r, str) and r.lower() in ("up", "thumbs_up", "+1", "good"):
        return 5
    if isinstance(r, str) and r.lower() in ("down", "thumbs_down", "-1", "bad"):
        return 1
    v = int(r)
    if not 1 <= v <= 5:
        raise ValueError("run_rating must be 1..5 (or 'up' / 'down')")
    return v


# --------------------------------------------------------------------------- corrections -> IR
def _deepest_container(root: Node, box: Box) -> Node:
    best = root
    for n in root.walk():
        if n.type in CONTAINER_SKIP or n.component is not None and n is not root:
            continue
        if n.box.contains(box, tol=1.0) and n.box.area <= best.box.area:
            best = n
    return best


def _translate(n: Node, dx: float, dy: float) -> None:
    for d in n.walk():
        d.box = d.box.translate(dx, dy)


def apply_corrections(doc: Document, items: list[dict], ds=None) -> tuple[Document, dict]:
    """A copy of ``doc`` with ``items`` applied (the user's ground truth for this screen).
    Returns ``(doc, {applied: [item ids], skipped: [{id, reason}]})``."""
    from dt.pipeline import _apply_choice
    out = Document.from_dict(doc.to_dict())
    rep: dict[str, list] = {"applied": [], "skipped": []}
    for it in items:
        kind, v = it["kind"], it.get("value")
        node = out.find(it["node_id"]) if it.get("node_id") else None
        if kind != "missing" and node is None:
            rep["skipped"].append({"id": it["id"], "reason": f"node {it.get('node_id')} not in IR"})
            continue
        if kind == "component":
            _apply_choice(node, "component", {"name": v["name"], "variant": v.get("variant") or {}}, ds)
        elif kind == "text":
            if node.type != "text":
                rep["skipped"].append({"id": it["id"], "reason": f"node {node.id} is {node.type}, not text"})
                continue
            node.text = v["text"]
        elif kind == "icon":
            node.type, node.icon_name, node.children = "icon", v["icon_name"], []
        elif kind == "geometry":
            x, y, w, h = v["box"]
            _translate(node, x - node.box.x, y - node.box.y)
            node.box = Box(x, y, w, h)
        elif kind == "color":
            c = Color.from_hex(v["color"]) if v.get("color") else None
            if node.type == "text" and node.text_style is not None and c is not None:
                node.text_style.color = c
            else:
                rest = [f for f in node.fills if not (f.kind == "solid" and f.color is not None)]
                node.fills = ([Fill.solid(c)] if c is not None else []) + rest
        elif kind == "missing":
            box = Box(*it["box"])
            parent = _deepest_container(out.root, box)
            t = v.get("type", "rect")
            new = Node(id=f"fb_{it['id']}", type=t if t != "instance" else "frame", box=box, meta={"feedback": "missing"})
            if v.get("fill"):
                new.fills = [Fill.solid(v["fill"])]
            if t == "text":
                new.text = v.get("text", "")
                new.text_style = TextStyle(size=max(8.0, round(box.h / 1.4, 1)), color=Color.from_hex(v.get("fill") or "#1d1b20"))
                new.fills = []
            if t == "icon":
                new.icon_name = v.get("icon_name")
            if v.get("component") and v["component"].get("name"):
                _apply_choice(new, "component", v["component"], ds)
            parent.children.append(new)
            rep["applied"].append(it["id"])
            continue
        elif kind == "extra":
            parent = out.root.parent_of(node.id)
            if parent is None:
                rep["skipped"].append({"id": it["id"], "reason": "cannot remove the root"})
                continue
            parent.children = [c for c in parent.children if c.id != node.id]
        elif kind == "should_be_image":
            node.type, node.children, node.component = "image", [], None
            node.meta["should_be_image"] = True
        elif kind == "should_be_editable":
            node.meta["should_be_editable"] = True
        elif kind == "ok":
            node.meta["confirmed"] = True
        node.meta["feedback"] = kind
        rep["applied"].append(it["id"])
    # the corpora's gt convention: geometric types, components carried by node.component
    for n in out.walk():
        if n.type == "instance" and "orig_type" in n.meta:
            n.type = n.meta.pop("orig_type")
    return out, rep


# --------------------------------------------------------------------------- bundles
def _enrich(items: list[dict], base: Document, ds) -> list[dict]:
    out = []
    for it in items:
        it = dict(it)
        if it.get("node_id"):
            node = base.find(it["node_id"])
            if node is None:
                raise ValueError(f"item {it['id']}: node {it['node_id']!r} is not in the run's ir.mapped.json")
            it["context"] = node_context(node, ds)
            if it.get("box") is None:
                it["box"] = list(it["context"]["box"])
        out.append(it)
    return out


def create_bundle(run_dir: str, items: list[dict], channel: str, rating: Any = None, consent: Optional[dict] = None,
                  home: Optional[str] = None, base_doc: Optional[Document] = None, bundle_id: Optional[str] = None,
                  merge: bool = False) -> dict:
    """Write (or, with ``merge``, extend) a local bundle for ``run_dir`` and return it.

    Files: ``bundle.json``; ``ir.base.json`` (the IR the items refer to, kept from the first capture);
    ``ir.json`` (base + every item: the user's ground truth); ``target.png`` only with
    ``consent.store_screenshot``."""
    info = run_info(run_dir)
    bid = bundle_id or new_bundle_id(f"{channel}:{info['run_dir']}:{info.get('screenshot_sha256')}:{time.time_ns()}")
    d = bundle_dir(bid, home)
    existing = load_bundle(d) if merge and os.path.exists(os.path.join(d, "bundle.json")) else None
    base_path = os.path.join(d, "ir.base.json")
    if existing is not None and os.path.exists(base_path):
        base = Document.load(base_path)
    else:
        base = base_doc if base_doc is not None else Document.load(os.path.join(run_dir, "ir.mapped.json"))
    ds = design_system_for(base.design_system or info.get("design_system"))
    norm = [normalize_item(it, channel, i) for i, it in enumerate(items)]
    norm = _enrich(norm, base, ds)
    if existing is not None:
        keyed = {(it["kind"], it.get("node_id"), tuple(it["box"] or []) if it["kind"] == "missing" else None): it
                 for it in existing.get("items", [])}
        for it in norm:
            keyed[(it["kind"], it.get("node_id"), tuple(it["box"] or []) if it["kind"] == "missing" else None)] = it
        norm = list(keyed.values())
        for i, it in enumerate(norm):
            it["id"] = f"i{i + 1}"
    cons = default_consent({**(existing or {}).get("consent", {}), **(consent or {})})
    bundle = {
        "schema": SCHEMA, "id": bid, "created": (existing or {}).get("created") or now_iso(), "updated": now_iso(),
        "channel": (existing or {}).get("channel") or channel, "run": info, "items": norm,
        "run_rating": _rating(rating) if rating is not None else (existing or {}).get("run_rating"),
        "consent": cons, "license": LICENSE_NOTE, "files": {"ir_base": "ir.base.json", "ir": "ir.json", "target": None},
    }
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(base_path):
        base.save(base_path)
    corrected, rep = apply_corrections(base, norm, ds)
    corrected.meta["feedback"] = {"bundle": bid, "items": len(norm), "applied": len(rep["applied"])}
    corrected.source_image = None
    corrected.save(os.path.join(d, "ir.json"))
    bundle["apply_report"] = rep
    tpath = os.path.join(d, "target.png")
    if cons["store_screenshot"]:
        if not os.path.exists(tpath):
            rgb = load_target(info)
            if rgb is not None and rgb.shape[:2] == (info["size"]["h"], info["size"]["w"]):
                from dt.common.image import save_rgb
                save_rgb(rgb, tpath)
        bundle["files"]["target"] = "target.png" if os.path.exists(tpath) else None
    elif os.path.exists(tpath):  # consent withdrawn: drop the copy
        os.remove(tpath)
    save_bundle(bundle, home)
    return bundle


def read_corrections(corrections: Any) -> dict:
    """A corrections file / dict / bare item list -> ``{run, items, run_rating, consent}``."""
    if isinstance(corrections, str):
        with open(corrections) as f:
            corrections = json.load(f)
    if isinstance(corrections, list):
        corrections = {"items": corrections}
    if not isinstance(corrections, dict):
        raise ValueError("corrections must be a JSON object {items, run_rating, consent} or a list of items")
    if corrections.get("schema") not in (None, CORRECTIONS_SCHEMA, SCHEMA):
        raise ValueError(f"unknown corrections schema {corrections.get('schema')!r}")
    return corrections


def add_corrections(run_dir: str, corrections: Any, rating: Any = None, consent: Optional[dict] = None,
                    channel: Optional[str] = None, home: Optional[str] = None) -> dict:
    """``dt feedback add``: a corrections file (review.html download, hand-written JSON, MCP items) -> bundle.
    Refuses corrections recorded against a different screenshot than the run's."""
    c = read_corrections(corrections)
    info = run_info(run_dir)
    sha = (c.get("run") or {}).get("screenshot_sha256")
    if sha and info.get("screenshot_sha256") and sha != info["screenshot_sha256"]:
        raise ValueError("these corrections were made on a different screenshot than this run's")
    cons = dict(c.get("consent") or {})
    cons.update({k: v for k, v in (consent or {}).items() if v is not None})
    return create_bundle(run_dir, list(c.get("items") or []), channel or c.get("channel") or "cli",
                         rating=rating if rating is not None else c.get("run_rating"), consent=cons, home=home)


def log_decisions(run_dir: str, answered: list[dict], base_doc: Document, home: Optional[str] = None) -> Optional[dict]:
    """Record answered decisions (``{id, kind, node_id, answer}``) as items of the run's decisions bundle
    (one bundle per run and screenshot; a later answer for the same node replaces the earlier one)."""
    items = []
    for d in answered:
        a = d.get("answer") or {}
        if d.get("kind") == "component":
            items.append({"kind": "component", "node_id": d["node_id"], "value": {"name": a.get("name"), "variant": a.get("variant") or {}},
                          "note": f"decision {d.get('id')}"})
        elif d.get("kind") == "text" and "text" in a:
            items.append({"kind": "text", "node_id": d["node_id"], "value": {"text": a["text"]}, "note": f"decision {d.get('id')}"})
        elif d.get("kind") == "icon" and a.get("icon_name"):
            items.append({"kind": "icon", "node_id": d["node_id"], "value": {"icon_name": a["icon_name"]}, "note": f"decision {d.get('id')}"})
    if not items:
        return None
    info = run_info(run_dir)
    bid = new_bundle_id(f"decisions:{info['run_dir']}:{info.get('screenshot_sha256')}")
    return create_bundle(run_dir, items, "decisions", home=home, base_doc=base_doc, bundle_id=bid, merge=True)


__all__ = ["add_corrections", "apply_corrections", "create_bundle", "log_decisions", "node_context", "read_corrections",
           "design_system_for"]
