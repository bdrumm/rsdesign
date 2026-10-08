"""Promotion path: from one consumer's local feedback to the shared model.

Consumer side -- ``dt feedback export -o bundle.zip [--redact-text] [--no-screenshots]``:
  * consent is per bundle; flags can only make an export *more* private;
  * text content ships only with ``consent.share_text`` and without ``--redact-text``; otherwise every
    character is replaced by a placeholder of the same class (x / X / 0, spaces and punctuation kept) so
    geometry, line lengths and text metrics survive but the words do not; free-text notes are dropped,
    raster crops (``image_ref``) are dropped, local paths are always stripped;
  * the screenshot ships only with ``consent.share_screenshot`` *and* shared text (a screenshot shows its text)
    and without ``--no-screenshots``.

Maintainer side -- ``dt feedback import bundle.zip [--promote]``:
  * bundles whose consent allows it become a ``user_reported`` scenario seed set
    (``fixtures/scenarios/user_reported/<id>.png`` + ``.gt.json`` + ``.feedback.json`` + ``manifest.json``);
  * every component answer becomes an observation in ``knowledge/feedback_candidates.json`` (signatures only,
    no text, no pixels), from which candidate *global* rules are derived (``feedback.global_min_support``
    distinct screenshots);
  * ``--promote`` gates candidate rules with the global bench (and the scenario seeds, when any) and, only when
    every gate passes, writes ``knowledge/learned_rules.json``; either way a ``feedback_promotion`` ledger entry
    (scope global) goes to ``knowledge/ledger.jsonl``.
"""
from __future__ import annotations

import io
import json
import os
import re
import string
import unicodedata
import zipfile
from typing import Any, Callable, Optional

from dt.feedback import ledger
from dt.feedback.learn import (_metrics, _user_gate, bench_gate, derive_rules, evaluate, observations, read_rules,
                               rules_diff, write_rules)
from dt.feedback.store import (LICENSE_NOTE, REPO_ROOT, SCHEMA, feedback_root, list_bundles, model_version, now_iso,
                               validate_bundle)
from dt.params import P

EXPORT_SCHEMA = "rsdesign.feedback-export/1"
Logger = Optional[Callable[[str], None]]
_PUNCT = set(string.punctuation)
_MEMBER = re.compile(r"^bundles/(fb-[0-9a-f]{12})/(bundle\.json|ir\.json|target\.png)$")
_META_KEEP = ("src", "conf", "orig_type", "feedback", "should_be_image", "should_be_editable", "confirmed", "logo")


def redact_text(s: Optional[str]) -> Optional[str]:
    """Same length, same character classes, no content: 'Inbox 12, Q3' -> 'Xxxxx 00, X0'."""
    if s is None:
        return None
    out = []
    for c in str(s):
        if c.isspace() or c in _PUNCT or unicodedata.category(c).startswith("P"):
            out.append(c)
        elif c.isdigit():
            out.append("0")
        elif c.isupper():
            out.append("X")
        else:
            out.append("x")
    return "".join(out)


def _redact_obj(v: Any) -> Any:
    if isinstance(v, str):
        return redact_text(v)
    if isinstance(v, list):
        return [_redact_obj(x) for x in v]
    if isinstance(v, dict):
        return {k: _redact_obj(x) for k, x in v.items()}
    return v


def scrub_ir(ir: dict, share_text: bool, keep_images: bool) -> dict:
    """Exportable copy of an IR dict: no local paths; without ``share_text`` no characters, names, props or
    free-form meta; without ``keep_images`` no raster crops."""
    ir = json.loads(json.dumps(ir))
    ir["source_image"] = None
    ir["meta"] = {k: v for k, v in (ir.get("meta") or {}).items() if k in ("feedback",)}

    def walk(n: dict) -> None:
        if not keep_images:
            n["image_ref"] = None
            n["fills"] = [f for f in n.get("fills") or [] if f.get("kind") != "image"]
        if not share_text:
            if n.get("text") is not None:
                n["text"] = redact_text(n["text"])
            n["name"] = n.get("type", "")
            n["meta"] = {k: v for k, v in (n.get("meta") or {}).items() if k in _META_KEEP and not isinstance(v, str) or k in ("src", "orig_type", "feedback")}
            if n.get("component"):
                n["component"]["props"] = _redact_obj(n["component"].get("props") or {})
                n["component"]["evidence"] = {}
        for c in n.get("children") or []:
            walk(c)
    walk(ir["root"])
    return ir


def scrub_bundle(b: dict, share_text: bool, screenshot: bool) -> dict:
    b = json.loads(json.dumps(b))
    run = b.get("run") or {}
    for k in ("source_path", "run_dir"):
        run.pop(k, None)
    b["run"] = run
    b.pop("apply_report", None)
    if not share_text:
        for it in b.get("items") or []:
            it["note"] = ""
            v = it.get("value")
            if it["kind"] == "text" and isinstance(v, dict):
                v["text"] = redact_text(v.get("text"))
            if it["kind"] == "missing" and isinstance(v, dict) and v.get("text") is not None:
                v["text"] = redact_text(v["text"])
    b["files"] = {"ir": "ir.json", "target": "target.png" if screenshot else None}
    b["export"] = {"text": "shared" if share_text else "redacted", "screenshot": screenshot, "exported": now_iso()}
    return b


def export_bundles(out_zip: str, home: Optional[str] = None, redact: bool = False, no_screenshots: bool = False,
                   ids: Optional[list[str]] = None) -> dict:
    """Write the local bundles (all, or ``ids``) to ``out_zip`` under each bundle's consent (see module doc)."""
    bundles = [b for b in list_bundles(home) if not ids or b["id"] in ids]
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(out_zip)) or ".", exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for b in bundles:
            cons = b.get("consent") or {}
            share_text = bool(cons.get("share_text")) and not redact
            d = os.path.join(feedback_root(home), b["id"])
            tpath = os.path.join(d, "target.png")
            shot = bool(cons.get("share_screenshot")) and share_text and not no_screenshots and os.path.exists(tpath)
            reason = None
            if not shot:
                reason = ("--no-screenshots" if no_screenshots else "no share_screenshot consent" if not cons.get("share_screenshot")
                          else "text is redacted (a screenshot shows its text)" if not share_text else "no stored screenshot")
            z.writestr(f"bundles/{b['id']}/bundle.json", json.dumps(scrub_bundle(b, share_text, shot), indent=2))
            irp = os.path.join(d, "ir.json")
            if os.path.exists(irp):
                with open(irp) as f:
                    z.writestr(f"bundles/{b['id']}/ir.json", json.dumps(scrub_ir(json.load(f), share_text, shot)))
            if shot:
                z.write(tpath, f"bundles/{b['id']}/target.png")
            rows.append({"id": b["id"], "items": len(b.get("items") or []), "text": "shared" if share_text else "redacted",
                         "screenshot": shot, "screenshot_excluded": reason})
        z.writestr("manifest.json", json.dumps({"schema": EXPORT_SCHEMA, "exported": now_iso(), "bundles": rows,
                                                "model": model_version(home), "license": LICENSE_NOTE}, indent=2))
    return {"out": os.path.abspath(out_zip), "bundles": rows}


# --------------------------------------------------------------------------- maintainer side
def default_scenarios_dir() -> str:
    return os.path.join(REPO_ROOT, "fixtures", "scenarios", "user_reported")


def default_knowledge_dir() -> str:
    return os.path.join(REPO_ROOT, "knowledge")


def _read_zip(zip_path: str) -> tuple[dict, dict[str, dict[str, bytes]]]:
    out: dict[str, dict[str, bytes]] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if "manifest.json" not in names:
            raise ValueError("not an rsdesign feedback export (no manifest.json)")
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("schema") != EXPORT_SCHEMA:
            raise ValueError(f"unknown export schema {manifest.get('schema')!r}")
        for name in names:
            m = _MEMBER.match(name)
            if m:  # anything else in the archive (paths, scripts) is ignored, never extracted
                out.setdefault(m.group(1), {})[m.group(2)] = z.read(name)
    return manifest, out


def _candidates_path(kdir: str) -> str:
    return os.path.join(kdir, "feedback_candidates.json")


def _load_obs(kdir: str) -> list[dict]:
    p = _candidates_path(kdir)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(json.load(f).get("observations", []))


def import_bundle(zip_path: str, scenarios_dir: Optional[str] = None, knowledge_dir: Optional[str] = None,
                  promote_rules: bool = False, gate: bool = True, workers: int = 3, bench_limit: Optional[int] = None,
                  bench_fn: Optional[Callable[..., dict]] = None, log: Logger = None) -> dict:
    """Maintainer side of the promotion path (see module doc). Returns a summary."""
    sdir = scenarios_dir or default_scenarios_dir()
    kdir = knowledge_dir or default_knowledge_dir()
    manifest, members = _read_zip(zip_path)
    res: dict[str, Any] = {"imported": [], "rejected": [], "seeds": [], "seed_skipped": []}
    imported: list[dict] = []
    for bid, files in sorted(members.items()):
        try:
            b = json.loads(files.get("bundle.json", b"{}"))
        except ValueError:
            res["rejected"].append({"id": bid, "reason": "bundle.json is not JSON"})
            continue
        errs = validate_bundle(b)
        if errs or b.get("id") != bid:
            res["rejected"].append({"id": bid, "reason": "; ".join(errs) or "id mismatch"})
            continue
        imported.append(b)
        res["imported"].append(bid)
        cons, exp = b["consent"], b.get("export") or {}
        allowed = cons.get("share_screenshot") and cons.get("share_text") and exp.get("text") == "shared"
        if allowed and "target.png" in files and "ir.json" in files:
            ir = json.loads(files["ir.json"])
            from PIL import Image
            im = Image.open(io.BytesIO(files["target.png"]))
            if im.size != (int(ir["width"]), int(ir["height"])):
                res["seed_skipped"].append({"id": bid, "reason": f"screenshot {im.size} != IR {ir['width']}x{ir['height']}"})
                continue
            os.makedirs(sdir, exist_ok=True)
            with open(os.path.join(sdir, bid + ".png"), "wb") as f:
                f.write(files["target.png"])
            ir["source_image"] = bid + ".png"
            ir.setdefault("meta", {})["provenance"] = {"family": "user_reported", "bundle": bid, "model": b["run"]["model"].get("id")}
            with open(os.path.join(sdir, bid + ".gt.json"), "w") as f:
                json.dump(ir, f)
            with open(os.path.join(sdir, bid + ".feedback.json"), "w") as f:
                json.dump(b, f, indent=2)
            res["seeds"].append(bid)
        else:
            res["seed_skipped"].append({"id": bid, "reason": "consent does not allow sharing the screenshot and its text"
                                        if not allowed else "screenshot not in the export"})
    if res["seeds"]:
        mp = os.path.join(sdir, "manifest.json")
        man = json.load(open(mp)) if os.path.exists(mp) else {"family": "user_reported", "kind": "user_reported", "cases": []}
        have = {c["id"] for c in man.get("cases", [])}
        for bid in res["seeds"]:
            if bid not in have:
                man["cases"].append({"id": bid, "source": "feedback", "imported": now_iso()})
        man["ids"] = [c["id"] for c in man["cases"]]
        man["n"] = len(man["cases"])
        with open(mp, "w") as f:
            json.dump(man, f, indent=2)
    obs = _load_obs(kdir)
    seen = {(o["bundle"], o["item"]) for o in obs}
    new_obs = [o for o in observations(imported) if (o["bundle"], o["item"]) not in seen]
    obs += new_obs
    os.makedirs(kdir, exist_ok=True)
    with open(_candidates_path(kdir), "w") as f:
        json.dump({"schema": "rsdesign.feedback-candidates/1", "updated": now_iso(), "observations": obs}, f, indent=2)
    cands = candidate_rules(kdir)
    res.update(observations_added=len(new_obs), candidates=len(cands), eligible=sum(1 for r in cands if r["active"]))
    if promote_rules:
        res["promotion"] = promote(sdir, kdir, gate=gate, workers=workers, bench_limit=bench_limit, bench_fn=bench_fn, log=log)
    return res


def candidate_rules(knowledge_dir: Optional[str] = None) -> list[dict]:
    return derive_rules(_load_obs(knowledge_dir or default_knowledge_dir()), scope="global",
                        min_support=int(P["feedback.global_min_support"]))


def promote(scenarios_dir: Optional[str] = None, knowledge_dir: Optional[str] = None, gate: bool = True, workers: int = 3,
            bench_limit: Optional[int] = None, bench_fn: Optional[Callable[..., dict]] = None, log: Logger = None) -> dict:
    """Try to promote eligible candidate rules into ``knowledge/learned_rules.json`` (global). Never ungated."""
    sdir = scenarios_dir or default_scenarios_dir()
    kdir = knowledge_dir or default_knowledge_dir()
    if not gate:
        return {"promoted": False, "reason": "promotion into the shared model always runs the global bench gate"}
    path = os.path.join(kdir, "learned_rules.json")
    current = read_rules(path)
    eligible = [r for r in candidate_rules(kdir) if r["active"]]
    by_id = {r["id"]: r for r in current}
    by_id.update({r["id"]: r for r in eligible})
    new = [by_id[k] for k in sorted(by_id)]
    diff = rules_diff(current, new)
    if not (diff["added"] or diff["removed"] or diff["changed"]):
        return {"promoted": False, "reason": "no eligible rule changes", "eligible": len(eligible)}
    cand = path + ".candidate.json"
    write_rules(cand, new, "global")
    try:
        ov_after, ov_before = {"map.learned_rules": cand}, {"map.learned_rules": path}
        gates: dict[str, bool] = {}
        bench_ev = (bench_fn or bench_gate)(ov_after, workers=workers, limit=bench_limit, log=log)
        gates["bench"] = bool(bench_ev["pass"])
        before = after = {}
        if os.path.exists(os.path.join(sdir, "manifest.json")):
            before, after = evaluate(overrides=ov_before, dirs=[sdir], log=log), evaluate(overrides=ov_after, dirs=[sdir], log=log)
            ug = _user_gate(before, after)
            if ug is not None:
                gates["scenarios"] = ug
        accepted = all(gates.values())
        sources = sorted({s for r in eligible for s in r.get("sources", [])})
        entry = ledger.make_entry("feedback_promotion", "global", "feedback", sources, {"file": "knowledge/learned_rules.json", "rules": diff},
                                  {"bench_composite": bench_ev.get("baseline_composite"), **_metrics(before)},
                                  {"bench_composite": bench_ev.get("composite"), **_metrics(after)}, gates, accepted,
                                  extra_evidence={"bench": bench_ev, "model": model_version()})
        if accepted:
            os.replace(cand, path)
        ledger.append(entry, os.path.join(kdir, "ledger.jsonl"))
        return {"promoted": accepted, "gates": gates, "ledger": entry["id"], "diff": diff}
    finally:
        if os.path.exists(cand):
            os.remove(cand)


__all__ = ["redact_text", "scrub_ir", "scrub_bundle", "export_bundles", "import_bundle", "candidate_rules", "promote",
           "EXPORT_SCHEMA", "SCHEMA"]
