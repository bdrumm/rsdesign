"""Markdown run report for a translation (perceive -> map -> refine -> export).

    write_report(out_dir, doc, loss_history, metrics, images) -> path of report.md

Inputs are deliberately loose so any stage can call this without importing phase-2 modules:
  * ``loss_history``: sequence of refine steps; each a dict / dataclass / object with any of
    ``iter, move|hypothesis|kind, node|node_id, loss_before, loss_after, accepted, detail``.
    Plain floats are accepted too (treated as the loss after each iteration).
  * ``metrics``: dict of name -> number (nested dicts are flattened with dots).
  * ``images``: dict of label -> path or RGB numpy array; copied/saved into ``out_dir`` and
    embedded as relative links.
Also writes ``ir.json`` (the IR) and ``build_plan.json`` next to the report.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
from typing import Any, Iterable, Mapping, Optional

from dt.export.figma_json import save_plan, to_build_plan, validate_plan
from dt.ir import Document, summarize
from dt.params import P, register

register("export.report.max_tree_nodes", 80, "Nodes shown in the report's IR tree dump.", (10, 1000))
register("export.report.max_moves", 50, "Accepted refine moves listed in the report.", (5, 1000))
register("export.report.float_precision", 4, "Decimal places for numbers in the report.", (1, 8))

_MOVE_KEYS = ("move", "hypothesis", "kind", "type", "name")
_NODE_KEYS = ("node", "node_id", "target")


def _as_dict(step: Any) -> dict:
    if isinstance(step, Mapping):
        return dict(step)
    if dataclasses.is_dataclass(step) and not isinstance(step, type):
        return dataclasses.asdict(step)
    if isinstance(step, (int, float)):
        return {"loss_after": float(step)}
    return {k: v for k, v in vars(step).items() if not k.startswith("_")} if hasattr(step, "__dict__") else {"detail": str(step)}


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{int(P['export.report.float_precision'])}f}"
    if v is None:
        return "-"
    return str(v)


def _first(d: dict, keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_metrics(v, key + "."))
        else:
            out[key] = v
    return out


def _normalize_history(loss_history: Optional[Iterable[Any]]) -> list[dict]:
    steps = [_as_dict(s) for s in (loss_history or [])]
    for i, s in enumerate(steps):
        s.setdefault("iter", i)
    return steps


def _save_image(label: str, img: Any, out_dir: str) -> Optional[str]:
    """Copy/save an image into out_dir; returns the relative path or None."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label) or "image"
    if isinstance(img, str):
        if not os.path.exists(img):
            return None
        src = os.path.abspath(img)
        if os.path.dirname(src) == os.path.abspath(out_dir):
            return os.path.basename(src)
        dst = os.path.join(out_dir, f"{safe}{os.path.splitext(src)[1] or '.png'}")
        if os.path.abspath(dst) != src:
            shutil.copyfile(src, dst)
        return os.path.basename(dst)
    try:  # numpy array
        from dt.common.image import save_rgb
        dst = os.path.join(out_dir, f"{safe}.png")
        save_rgb(img, dst)
        return os.path.basename(dst)
    except Exception:
        return None


def _summary_rows(doc: Document, steps: list[dict], plan: dict, plan_errors: list[str]) -> list[tuple[str, str]]:
    nodes = list(doc.walk())
    accepted = [s for s in steps if s.get("accepted", True)]
    # the loss after the run is the loss after the last ACCEPTED move (rejected moves are rolled back)
    losses = [s.get("loss_after") for s in accepted if isinstance(s.get("loss_after"), (int, float))]
    first = next((s.get("loss_before") for s in steps if isinstance(s.get("loss_before"), (int, float))), None)
    if first is None and losses:
        first = losses[0]
    rows = [
        ("Source image", doc.source_image or "-"),
        ("Size", f"{doc.width} x {doc.height} (dpr {doc.dpr})"),
        ("Design system", doc.design_system or "-"),
        ("Nodes", str(len(nodes))),
        ("Text nodes", str(sum(1 for n in nodes if n.type == "text"))),
        ("Component instances", str(sum(1 for n in nodes if n.component is not None))),
        ("Token bindings", str(sum(len(n.tokens) for n in nodes))),
        ("Refine iterations", str(len(steps))),
        ("Accepted moves", str(len(accepted)) if steps else "-"),
        ("Initial loss", _fmt(first)),
        ("Final loss", _fmt(losses[-1]) if losses else "-"),
        ("Loss reduction", f"{(1 - losses[-1] / first) * 100:.1f}%" if losses and first else "-"),
        ("Build plan", f"{plan['meta']['nodeCount']} nodes, {len(plan['fonts'])} fonts, "
                       f"{len(plan['warnings'])} warnings, {'valid' if not plan_errors else f'{len(plan_errors)} errors'}"),
    ]
    return rows


def _table(header: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _moves_section(steps: list[dict]) -> str:
    accepted = [s for s in steps if s.get("accepted", True)]
    if not steps:
        return "_No refine history._"
    rows = []
    for s in accepted[: int(P["export.report.max_moves"])]:
        rows.append((
            _fmt(s.get("iter")), _fmt(_first(s, _MOVE_KEYS)), _fmt(_first(s, _NODE_KEYS)),
            _fmt(s.get("loss_before")), _fmt(s.get("loss_after")), _fmt(s.get("detail")),
        ))
    extra = f"\n\n_... {len(accepted) - len(rows)} more accepted moves_" if len(accepted) > len(rows) else ""
    rejected = len(steps) - len(accepted)
    return _table(("iter", "move", "node", "loss before", "loss after", "detail"), rows) + extra + \
        f"\n\n{len(accepted)} accepted, {rejected} rejected."


def write_report(out_dir: str, doc: Document, loss_history: Optional[Iterable[Any]] = None,
                 metrics: Optional[Mapping[str, Any]] = None, images: Optional[Mapping[str, Any]] = None,
                 title: str = "designtranslator run report") -> str:
    """Write report.md (+ ir.json, build_plan.json, copied images) into out_dir. Returns report path."""
    os.makedirs(out_dir, exist_ok=True)
    steps = _normalize_history(loss_history)
    plan = to_build_plan(doc)
    plan_errors = validate_plan(plan)
    doc.save(os.path.join(out_dir, "ir.json"))
    save_plan(plan, os.path.join(out_dir, "build_plan.json"))

    parts = [f"# {title}", "", "## Summary", "", _table(("field", "value"), _summary_rows(doc, steps, plan, plan_errors)), ""]

    flat = flatten_metrics(metrics or {})
    parts += ["## Metrics", "", _table(("metric", "value"), [(k, _fmt(v)) for k, v in flat.items()]) if flat else "_No metrics._", ""]

    parts += ["## Accepted refine moves", "", _moves_section(steps), ""]

    parts += ["## Node tree", "", "```", summarize(doc, max_nodes=int(P["export.report.max_tree_nodes"])), "```", ""]

    parts += ["## Images", ""]
    any_img = False
    for label, img in (images or {}).items():
        rel = _save_image(label, img, out_dir)
        if rel:
            parts += [f"### {label}", "", f"![{label}]({rel})", ""]
            any_img = True
    if not any_img:
        parts += ["_No images._", ""]

    parts += ["## Export", "", f"- build plan: `build_plan.json` (schema v{plan['version']})", "- IR: `ir.json`"]
    if plan["warnings"]:
        parts += ["", "Plan warnings:", ""] + [f"- {w}" for w in plan["warnings"]]
    if plan_errors:
        parts += ["", "Plan validation errors:", ""] + [f"- {e}" for e in plan_errors]
    parts.append("")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": flat, "history": steps}, f, indent=2, default=str)
    return path
