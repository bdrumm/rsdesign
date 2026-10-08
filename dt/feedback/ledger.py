"""The shared learning ledger (one JSON object per line), written by the scenario harness and the feedback loop.

    {"id": "<kind>-<short hash>", "ts": iso8601, "kind": "params"|"rule"|"scenario_baseline"|"feedback_promotion"|"revert",
     "scope": "global"|"local", "source": {"type": "scenario"|"feedback"|"manual", "ids": [...]},
     "change": {...exact diff...}, "evidence": {"before": {...}, "after": {...}, "gates": {"bench": bool, ...}},
     "accepted": bool, "reverts": "<id>" | null}

Global entries go to ``knowledge/ledger.jsonl`` (committed with the learned state they describe); local entries
(a consumer's own learning) go to ``$DT_HOME/ledger.jsonl`` and never leave the machine.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

from dt.feedback.store import REPO_ROOT, dt_home, now_iso

KINDS = ("params", "rule", "scenario_baseline", "feedback_promotion", "revert")
SCOPES = ("global", "local")
SOURCE_TYPES = ("scenario", "feedback", "manual")
FIELDS = ("id", "ts", "kind", "scope", "source", "change", "evidence", "accepted", "reverts")


def global_path() -> str:
    return os.path.join(REPO_ROOT, "knowledge", "ledger.jsonl")


def local_path(home: Optional[str] = None) -> str:
    return os.path.join(home or dt_home(), "ledger.jsonl")


def make_entry(kind: str, scope: str, source_type: str, source_ids: list[str], change: dict, before: dict,
               after: dict, gates: dict[str, bool], accepted: bool, reverts: Optional[str] = None,
               extra_evidence: Optional[dict] = None, ts: Optional[str] = None) -> dict:
    ts = ts or now_iso()
    body = {"ts": ts, "kind": kind, "scope": scope, "source": {"type": source_type, "ids": list(source_ids)},
            "change": change, "evidence": {"before": before, "after": after, "gates": dict(gates), **(extra_evidence or {})},
            "accepted": bool(accepted), "reverts": reverts}
    digest = hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:10]
    entry = {"id": f"{kind}-{digest}", **body}
    errs = validate_entry(entry)
    if errs:
        raise ValueError("invalid ledger entry: " + "; ".join(errs))
    return entry


def validate_entry(e: Any) -> list[str]:
    """Errors against the shared schema (empty = conforms)."""
    if not isinstance(e, dict):
        return ["entry must be an object"]
    errs = [f"missing {k}" for k in FIELDS if k not in e]
    if errs:
        return errs
    if e["kind"] not in KINDS:
        errs.append(f"kind {e['kind']!r} not in {KINDS}")
    if not isinstance(e["id"], str) or not e["id"].startswith(f"{e['kind']}-") or len(e["id"]) <= len(e["kind"]) + 1:
        errs.append("id must be '<kind>-<short hash>'")
    if e["scope"] not in SCOPES:
        errs.append(f"scope {e['scope']!r} not in {SCOPES}")
    src = e["source"]
    if not isinstance(src, dict) or src.get("type") not in SOURCE_TYPES or not isinstance(src.get("ids"), list):
        errs.append("source must be {type: scenario|feedback|manual, ids: [...]}")
    if not isinstance(e["change"], dict):
        errs.append("change must be an object")
    ev = e["evidence"]
    if not isinstance(ev, dict) or not isinstance(ev.get("before"), dict) or not isinstance(ev.get("after"), dict) \
            or not isinstance(ev.get("gates"), dict) or not all(isinstance(v, bool) for v in ev["gates"].values()):
        errs.append("evidence must be {before: {}, after: {}, gates: {name: bool}}")
    if not isinstance(e["accepted"], bool):
        errs.append("accepted must be a bool")
    if e["reverts"] is not None and not isinstance(e["reverts"], str):
        errs.append("reverts must be an id or null")
    if not isinstance(e["ts"], str) or "T" not in e["ts"]:
        errs.append("ts must be ISO 8601")
    return errs


def append(entry: dict, path: str) -> str:
    errs = validate_entry(entry)
    if errs:
        raise ValueError("invalid ledger entry: " + "; ".join(errs))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return path


def read(path: str) -> list[dict]:
    out: list[dict] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


__all__ = ["KINDS", "SCOPES", "SOURCE_TYPES", "global_path", "local_path", "make_entry", "validate_entry", "append", "read"]
