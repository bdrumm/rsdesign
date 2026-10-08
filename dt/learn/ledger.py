"""The shared learning ledger: every learned change, its evidence, and how to undo it.

Both the scenario harness (``dt train``, ``dt scenario promote``) and the feedback loop write
the same schema, one JSON object per line::

    {"id": "<kind>-<short hash>", "ts": "<iso8601>",
     "kind": "params" | "rule" | "scenario_baseline" | "feedback_promotion" | "revert",
     "scope": "global" | "local",
     "source": {"type": "scenario" | "feedback" | "manual", "ids": [...]},
     "change": {...exact diff...},
     "evidence": {"before": {...metrics}, "after": {...metrics}, "gates": {"bench": bool, ...}},
     "accepted": bool, "reverts": "<id or null>"}

Global entries live in ``knowledge/ledger.jsonl`` (committed), local ones in
``$DT_HOME/ledger.jsonl`` (a consumer's own; never committed). :func:`read` merges both.

Reversible ``change`` shapes (anything else is recorded but cannot be auto-reverted):

* ``{"params": {key: [old, new]}, "layer": "global" | "local"}`` -- ``None`` = key absent from
  that layer file; :func:`revert` writes ``old`` back exactly (removing keys that were absent);
* ``{"suite": {family: [old_entry, new_entry]}}`` -- an active scenario suite entry
  (``knowledge/scenarios.json``); ``None`` = family not in the suite.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Iterable, Optional

from dt.learn import state

KINDS = ("params", "rule", "scenario_baseline", "feedback_promotion", "revert")
SOURCE_TYPES = ("scenario", "feedback", "manual")


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def make_entry(kind: str, scope: str, source_type: str, source_ids: Iterable[str], change: dict,
               evidence: Optional[dict] = None, accepted: bool = True, reverts: Optional[str] = None,
               ts: Optional[str] = None, **extra: Any) -> dict:
    """A schema-valid entry with a content-derived id ``<kind>-<10 hex>``."""
    e: dict[str, Any] = {"ts": ts or now_iso(), "kind": kind, "scope": scope,
                         "source": {"type": source_type, "ids": [str(i) for i in source_ids]},
                         "change": change, "evidence": evidence or {"before": {}, "after": {}, "gates": {}},
                         "accepted": bool(accepted), "reverts": reverts}
    e.update(extra)
    digest = hashlib.sha1(json.dumps(e, sort_keys=True, default=str).encode()).hexdigest()[:10]
    out = {"id": f"{kind}-{digest}"}
    out.update(e)
    errs = validate_entry(out)
    if errs:
        raise ValueError("invalid ledger entry: " + "; ".join(errs))
    return out


def validate_entry(e: dict) -> list[str]:
    """Schema errors of one entry (empty = valid)."""
    errs: list[str] = []
    for k in ("id", "ts", "kind", "scope", "source", "change", "evidence", "accepted", "reverts"):
        if k not in e:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if e["kind"] not in KINDS:
        errs.append(f"kind {e['kind']!r} not in {KINDS}")
    if not str(e["id"]).startswith(f"{e['kind']}-"):
        errs.append("id must be '<kind>-<hash>'")
    if e["scope"] not in state.SCOPES:
        errs.append(f"scope {e['scope']!r} not in {state.SCOPES}")
    src = e["source"]
    if not isinstance(src, dict) or src.get("type") not in SOURCE_TYPES or not isinstance(src.get("ids"), list):
        errs.append("source must be {type: scenario|feedback|manual, ids: [...]}")
    if not isinstance(e["change"], dict):
        errs.append("change must be an object")
    ev = e["evidence"]
    if not isinstance(ev, dict) or not all(k in ev for k in ("before", "after", "gates")):
        errs.append("evidence must have before, after, gates")
    if not isinstance(e["accepted"], bool):
        errs.append("accepted must be a bool")
    if e["reverts"] is not None and not isinstance(e["reverts"], str):
        errs.append("reverts must be an id or null")
    return errs


def append(entry: dict, path: Optional[str] = None) -> dict:
    """Validate and append ``entry`` to the ledger of its scope (or ``path``)."""
    errs = validate_entry(entry)
    if errs:
        raise ValueError("invalid ledger entry: " + "; ".join(errs))
    path = path or state.ledger_path(entry["scope"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return entry


def read(paths: Optional[Iterable[str]] = None) -> list[dict]:
    """All entries of the global and local ledgers (or ``paths``), oldest first. Corrupt lines
    are skipped."""
    if paths is None:
        paths = [state.ledger_path("global"), state.ledger_path("local")]
    out: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(e, dict) and e.get("id") and e["id"] not in seen:
                    seen.add(e["id"])
                    out.append(e)
    out.sort(key=lambda e: str(e.get("ts", "")))
    return out


def get(entry_id: str) -> Optional[dict]:
    """Entry by id (a unique id prefix also works)."""
    entries = read()
    exact = [e for e in entries if e["id"] == entry_id]
    if exact:
        return exact[0]
    pref = [e for e in entries if e["id"].startswith(entry_id)]
    return pref[0] if len(pref) == 1 else None


def reverted_by(entry_id: str) -> Optional[dict]:
    """The accepted revert entry that undid ``entry_id``, if any."""
    for e in read():
        if e.get("kind") == "revert" and e.get("reverts") == entry_id and e.get("accepted"):
            return e
    return None


def revert(entry_id: str, force: bool = False, note: str = "") -> dict:
    """Undo an accepted entry exactly and record a ``revert`` entry (returned).

    Refuses (``ValueError``) when the entry is unknown, was not accepted, is already reverted, has
    no reversible change, or -- unless ``force`` -- when the live layer no longer holds the
    entry's ``new`` values (something changed them since; reverting would clobber that change).
    """
    e = get(entry_id)
    if e is None:
        raise ValueError(f"no ledger entry {entry_id!r}")
    if not e.get("accepted"):
        raise ValueError(f"{e['id']} was not accepted; nothing to revert")
    if reverted_by(e["id"]):
        raise ValueError(f"{e['id']} is already reverted by {reverted_by(e['id'])['id']}")
    ch = e.get("change") or {}
    inverse: dict[str, Any] = {}
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    if "params" in ch:
        layer = ch.get("layer") or e["scope"]
        cur = state.read_layer(layer)
        drift = {k: {"expected": new, "current": cur.get(k)} for k, (old, new) in ch["params"].items() if cur.get(k) != new}
        if drift and not force:
            raise ValueError(f"{layer} params changed since {e['id']}: {drift} (use force to revert anyway)")
        restored = state.write_layer(layer, {k: old for k, (old, new) in ch["params"].items()})
        inverse = {"params": {k: [new, old] for k, (old, new) in ch["params"].items()}, "layer": layer}
        before = {"params": {k: cur.get(k) for k in ch["params"]}}
        after = {"params": {k: old for k, (old, _new) in ch["params"].items()}, "written": restored}
    elif "suite" in ch:
        from dt.scenarios import suite as _suite
        s = _suite.load()
        drift = {f: True for f, (old, new) in ch["suite"].items() if s["families"].get(f) != new}
        if drift and not force:
            raise ValueError(f"suite entries changed since {e['id']}: {sorted(drift)} (use force to revert anyway)")
        before = {"suite": {f: s["families"].get(f) for f in ch["suite"]}}
        for f, (old, _new) in ch["suite"].items():
            if old is None:
                s["families"].pop(f, None)
            else:
                s["families"][f] = old
        _suite.save(s)
        inverse = {"suite": {f: [new, old] for f, (old, new) in ch["suite"].items()}}
        after = {"suite": {f: old for f, (old, _new) in ch["suite"].items()}}
    else:
        raise ValueError(f"{e['id']} ({e['kind']}) has no reversible change (params or suite)")
    rev = make_entry("revert", e["scope"], "manual", [e["id"]], inverse,
                     evidence={"before": before, "after": after, "gates": {}, "note": note}, accepted=True,
                     reverts=e["id"])
    return append(rev)


def format_entry(e: dict, verbose: bool = False) -> str:
    """One-line (or detailed) human summary."""
    ch = e.get("change") or {}
    if "params" in ch:
        what = ", ".join(f"{k}: {o!r}->{n!r}" for k, (o, n) in list(ch["params"].items())[:4])
        what = f"[{ch.get('layer', e.get('scope'))}] {what}" + (" ..." if len(ch["params"]) > 4 else "")
    elif "suite" in ch:
        what = "suite " + ", ".join(ch["suite"])
    else:
        what = json.dumps(ch, default=str)[:100]
    gates = (e.get("evidence") or {}).get("gates") or {}
    g = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in gates.items() if isinstance(v, bool))
    line = (f"{e['id']:28s} {e.get('ts', '')[:19]:19s} {e['kind']:18s} {e['scope']:6s} "
            f"{'accepted' if e.get('accepted') else 'rejected'}  {what}  {g}")
    if e.get("reverts"):
        line += f"  (reverts {e['reverts']})"
    if not verbose:
        return line
    return line + "\n" + json.dumps(e, indent=2, sort_keys=True, default=str)
