"""Feedback bundles: schema, local storage under ``DT_HOME`` and run identity (docs/FEEDBACK.md).

A *bundle* is one consumer's corrections to one translate run::

    {"schema": "rsdesign.feedback/1", "id": "fb-<12 hex>", "created": iso8601, "channel": "cli|decisions|review|figma|mcp",
     "run": {"screenshot_sha256", "size": {"w", "h"}, "dpr", "design_system", "model": {"git", "params", "id"},
             "source_path", "run_dir"},
     "items": [{"id", "kind", "node_id", "box": [x, y, w, h], "value", "note", "source", "context"}],
     "run_rating": 1..5 | null,
     "consent": {"store_screenshot": bool, "share_screenshot": bool, "share_text": bool},
     "license": "..."}

Stored as ``$DT_HOME/feedback/<id>/bundle.json`` with ``ir.base.json`` (the run's IR as the model produced
it), ``ir.json`` (that IR with every correction applied: the user's ground truth) and, only when
``consent.store_screenshot``, ``target.png`` (the screenshot at the IR's resolution). Nothing here
touches the network; a bundle leaves the machine only through ``dt feedback export``.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
from typing import Any, Optional

from dt.params import PARAMS_PATH, P, register

SCHEMA = "rsdesign.feedback/1"
CORRECTIONS_SCHEMA = "rsdesign.corrections/1"
KINDS = ("component", "text", "icon", "geometry", "color", "missing", "extra", "should_be_image",
         "should_be_editable", "ok")
CHANNELS = ("cli", "decisions", "review", "figma", "mcp")
NODE_TYPES = ("frame", "rect", "ellipse", "line", "text", "icon", "image", "vector", "instance")
LICENSE_NOTE = ("Feedback stays on this machine. If you export it with `dt feedback export` and send it to the "
                "rsdesign maintainers, the corrections (and, only where consent.share_screenshot and "
                "consent.share_text are true, the screenshot and its text) are contributed under the project's "
                "Apache-2.0 license for its public test corpus; you confirm you have the right to share them.")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

register("feedback.log_decisions", True,
         "log every answer given through `dt apply-decisions` (CLI / MCP) as a feedback item of the run's local bundle")
register("feedback.consent.store_screenshot", True,
         "default consent: keep a copy of the screenshot in the local bundle (it never leaves the machine unless exported "
         "with share consent); without it learning uses the original file while it exists unchanged")
register("feedback.consent.share_screenshot", False, "default consent: an export may include the screenshot")
register("feedback.consent.share_text", False,
         "default consent: an export may include text content (otherwise characters are redacted, geometry kept)")


# --------------------------------------------------------------------------- paths
def dt_home() -> str:
    """User state directory: ``$DT_HOME`` or ``~/.rsdesign`` (never committed)."""
    return os.path.abspath(os.path.expanduser(os.environ.get("DT_HOME") or os.path.join("~", ".rsdesign")))


def feedback_root(home: Optional[str] = None) -> str:
    return os.path.join(home or dt_home(), "feedback")


def bundle_dir(bundle_id: str, home: Optional[str] = None) -> str:
    if not bundle_id.startswith("fb-") or os.sep in bundle_id or ".." in bundle_id:
        raise ValueError(f"not a bundle id: {bundle_id!r}")
    return os.path.join(feedback_root(home), bundle_id)


def corpus_root(home: Optional[str] = None) -> str:
    return os.path.join(feedback_root(home), "corpus")


def history_path(home: Optional[str] = None) -> str:
    return os.path.join(feedback_root(home), "history.jsonl")


def local_rules_path(home: Optional[str] = None) -> str:
    return os.path.join(home or dt_home(), "learned_rules.json")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- model identity
_GIT: dict[str, str] = {}


def _git_sha() -> str:
    if "sha" not in _GIT:
        try:
            r = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=10)
            _GIT["sha"] = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
        except Exception:  # noqa: BLE001 - not a git checkout (installed package)
            _GIT["sha"] = "unknown"
    return _GIT["sha"]


def params_hash(home: Optional[str] = None) -> str:
    """Hash of the learned parameter state: dt/params.json, the local params overlay and every learned-rule
    file the matcher would load (so a local rule change is a different model version)."""
    from dt.mapping.matcher import learned_rule_paths
    h = hashlib.sha1()
    for p in [PARAMS_PATH, os.path.join(home or dt_home(), "params.local.json")] + learned_rule_paths():
        h.update(p.encode())
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def model_version(home: Optional[str] = None) -> dict:
    """``{git, params, id}``: the code revision and the learned state the run was produced with."""
    git, ph = _git_sha(), params_hash(home)
    return {"git": git, "params": ph, "id": f"{git}+{ph}"}


# --------------------------------------------------------------------------- run identity
def run_info(run_dir: str) -> dict:
    """Identity of a translate run (``dt translate -o run_dir``): screenshot hash, size, dpr, design system,
    model version. Raises FileNotFoundError when ``run_dir`` has no ``ir.mapped.json``."""
    run_dir = os.path.abspath(run_dir)
    ir_path = os.path.join(run_dir, "ir.mapped.json")
    if not os.path.exists(ir_path):
        raise FileNotFoundError(f"{run_dir} is not a translate run (no ir.mapped.json)")
    metrics: dict = {}
    mp = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            metrics = json.load(f)
    with open(ir_path) as f:
        ir = json.load(f)
    src = metrics.get("source") or ir.get("source_image")
    dpr = metrics.get("dpr", ir.get("dpr", 1.0))
    try:
        dpr = float(dpr)
    except (TypeError, ValueError):
        dpr = 1.0
    return {
        "run_dir": run_dir,
        "source_path": os.path.abspath(src) if src else None,
        "screenshot_sha256": sha256_file(src) if src and os.path.exists(src) else None,
        "size": {"w": int(ir["width"]), "h": int(ir["height"])},
        "dpr": dpr,
        "design_system": ir.get("design_system") or metrics.get("design_system"),
        "model": model_version(),
    }


def load_target(run: dict) -> Optional["Any"]:
    """The screenshot at IR resolution (downscaled by dpr) when the source still exists unchanged."""
    src = run.get("source_path")
    if not src or not os.path.exists(src):
        return None
    if run.get("screenshot_sha256") and sha256_file(src) != run["screenshot_sha256"]:
        return None
    from dt.common.image import downscale_dpr, load_rgb
    rgb = load_rgb(src)
    if float(run.get("dpr") or 1.0) != 1.0:
        rgb = downscale_dpr(rgb, float(run["dpr"]))
    return rgb


# --------------------------------------------------------------------------- items
def _box(v: Any) -> Optional[list[float]]:
    if v is None:
        return None
    if isinstance(v, dict):
        v = [v.get("x"), v.get("y"), v.get("w"), v.get("h")]
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError(f"box must be [x, y, w, h] or {{x, y, w, h}}, got {v!r}")
    out = [round(float(x), 2) for x in v]
    if out[2] < 0 or out[3] < 0:
        raise ValueError(f"box has negative size: {v!r}")
    return out


def _hex(v: Any) -> Optional[str]:
    if v is None:
        return None
    from dt.ir import Color
    return Color.from_hex(str(v)).hex()


def normalize_value(kind: str, value: Any) -> Any:
    """Canonical value of an item (see docs/FEEDBACK.md); raises ValueError for malformed values."""
    if kind == "component":
        if value is None or isinstance(value, str):
            value = {"name": value}
        if not isinstance(value, dict) or "name" not in value:
            raise ValueError("component value must be {name, variant} (name null = not a component)")
        name = value.get("name")
        return {"name": str(name).strip() if name else None,
                "variant": {str(k): str(v) for k, v in dict(value.get("variant") or {}).items()}}
    if kind == "text":
        t = value.get("text") if isinstance(value, dict) else value
        if t is None:
            raise ValueError("text value must be a string or {text}")
        return {"text": str(t)}
    if kind == "icon":
        n = value.get("icon_name") if isinstance(value, dict) else value
        if not n:
            raise ValueError("icon value must be a Material Symbol name or {icon_name}")
        return {"icon_name": str(n).strip()}
    if kind == "geometry":
        b = value.get("box") if isinstance(value, dict) and "box" in value else value
        return {"box": _box(b)}
    if kind == "color":
        c = value.get("color") if isinstance(value, dict) else value
        return {"color": _hex(c)}
    if kind == "missing":
        v = dict(value or {}) if isinstance(value, dict) else {"type": value or "rect"}
        t = str(v.get("type") or "rect")
        if t not in NODE_TYPES:
            raise ValueError(f"missing.type must be one of {NODE_TYPES}")
        out: dict[str, Any] = {"type": t}
        if v.get("text") is not None:
            out["text"] = str(v["text"])
        if v.get("icon_name"):
            out["icon_name"] = str(v["icon_name"])
        if v.get("fill"):
            out["fill"] = _hex(v["fill"])
        if v.get("component"):
            out["component"] = normalize_value("component", v["component"])
        return out
    return None  # extra / should_be_image / should_be_editable / ok carry no value


def normalize_item(item: dict, source: str, idx: int) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"item {idx}: must be an object")
    kind = str(item.get("kind", "")).strip()
    if kind not in KINDS:
        raise ValueError(f"item {idx}: kind {kind!r} not in {KINDS}")
    node_id = item.get("node_id")
    box = _box(item.get("box"))
    if kind == "missing":
        if box is None:
            raise ValueError(f"item {idx}: 'missing' needs a box")
    elif not node_id:
        raise ValueError(f"item {idx}: {kind!r} needs a node_id")
    out = {"id": str(item.get("id") or f"i{idx + 1}"), "kind": kind, "node_id": str(node_id) if node_id else None,
           "box": box, "value": normalize_value(kind, item.get("value")), "note": str(item.get("note") or "")[:2000],
           "source": str(item.get("source") or source)}
    if item.get("context"):
        out["context"] = item["context"]
    return out


def default_consent(overrides: Optional[dict] = None) -> dict:
    c = {k: bool(P[f"feedback.consent.{k}"]) for k in ("store_screenshot", "share_screenshot", "share_text")}
    for k, v in (overrides or {}).items():
        if k in c and v is not None:
            c[k] = bool(v)
    return c


# --------------------------------------------------------------------------- bundles on disk
def new_bundle_id(seed: str) -> str:
    return "fb-" + hashlib.sha1(seed.encode()).hexdigest()[:12]


def save_bundle(bundle: dict, home: Optional[str] = None) -> str:
    d = bundle_dir(bundle["id"], home)
    os.makedirs(d, exist_ok=True)
    errs = validate_bundle(bundle)
    if errs:
        raise ValueError("invalid bundle: " + "; ".join(errs))
    tmp = os.path.join(d, "bundle.json.tmp")
    with open(tmp, "w") as f:
        json.dump(bundle, f, indent=2)
    os.replace(tmp, os.path.join(d, "bundle.json"))
    return d


def load_bundle(bundle_id_or_dir: str, home: Optional[str] = None) -> dict:
    d = bundle_id_or_dir if os.path.isdir(bundle_id_or_dir) else bundle_dir(bundle_id_or_dir, home)
    with open(os.path.join(d, "bundle.json")) as f:
        return json.load(f)


def list_bundles(home: Optional[str] = None) -> list[dict]:
    root = feedback_root(home)
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "bundle.json")
        if name.startswith("fb-") and os.path.exists(p):
            try:
                with open(p) as f:
                    out.append(json.load(f))
            except (OSError, ValueError):
                continue
    return out


def delete_bundle(bundle_id: str, home: Optional[str] = None) -> bool:
    """Remove a local bundle (and its user-corpus case). Returns whether it existed."""
    d = bundle_dir(bundle_id, home)
    existed = os.path.isdir(d)
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(os.path.join(corpus_root(home), bundle_id), ignore_errors=True)
    return existed


def validate_bundle(b: dict) -> list[str]:
    """Schema errors of a bundle (empty = valid)."""
    errs: list[str] = []
    if b.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}")
    if not str(b.get("id", "")).startswith("fb-"):
        errs.append("id must start with 'fb-'")
    if not b.get("created"):
        errs.append("created missing")
    run = b.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("size"), dict) or "model" not in run:
        errs.append("run must carry size and model")
    r = b.get("run_rating")
    if r is not None and (not isinstance(r, int) or not 1 <= r <= 5):
        errs.append("run_rating must be null or an integer 1..5")
    c = b.get("consent")
    if not isinstance(c, dict) or set(c) != {"store_screenshot", "share_screenshot", "share_text"} \
            or not all(isinstance(v, bool) for v in c.values()):
        errs.append("consent must be {store_screenshot, share_screenshot, share_text} booleans")
    for i, it in enumerate(b.get("items") or []):
        try:
            normalize_item(it, it.get("source", "cli"), i)
        except ValueError as e:
            errs.append(str(e))
    if not isinstance(b.get("items"), list):
        errs.append("items must be a list")
    return errs


def deep(o: Any) -> Any:
    return copy.deepcopy(o)


__all__ = ["SCHEMA", "CORRECTIONS_SCHEMA", "KINDS", "CHANNELS", "LICENSE_NOTE", "dt_home", "feedback_root",
           "bundle_dir", "corpus_root", "history_path", "local_rules_path", "model_version", "params_hash", "run_info",
           "load_target", "normalize_item", "normalize_value", "default_consent", "new_bundle_id", "save_bundle",
           "load_bundle", "list_bundles", "delete_bundle", "validate_bundle", "now_iso", "sha256_file"]
