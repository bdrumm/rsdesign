"""Where learned state lives, and the only writer of the persisted params layers.

Two scopes, same formats:

* ``global`` -- ships with the repo: ``dt/params.json`` (``dt.params.PARAMS_PATH``),
  ``knowledge/ledger.jsonl``, ``knowledge/scenarios.json``, ``knowledge/real_crops.json``.
* ``local``  -- one consumer's own learned state under ``$DT_HOME`` (default ``~/.rsdesign``),
  never committed: ``params.local.json`` (overlay loaded after the global file, see
  :mod:`dt.params`), ``ledger.jsonl``, ``scenarios/`` (real crops, local families).

Every learned change goes through :func:`write_layer`, which returns the exact per-key diff
``{key: [old, new]}`` (``None`` = absent from that layer file) that the ledger records and
:func:`dt.learn.ledger.revert` replays backwards.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import dt.params as _params
from dt.params import P, dt_home

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCOPES = ("global", "local")


def home(*parts: str) -> str:
    """A path under ``$DT_HOME`` (not created)."""
    return os.path.join(dt_home(), *parts)


def knowledge(*parts: str) -> str:
    return os.path.join(ROOT, "knowledge", *parts)


def layer_path(scope: str) -> str:
    """The params file backing ``scope`` (raises when the local layer is disabled)."""
    if scope == "global":
        return _params.PARAMS_PATH
    if scope == "local":
        p = _params.local_params_path()
        if not p:
            raise RuntimeError("the local params layer is disabled (DT_PARAMS_LOCAL is empty)")
        return p
    raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")


def read_layer(scope: str) -> dict[str, Any]:
    p = layer_path(scope)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_layer(scope: str, changes: dict[str, Any], reload: bool = True) -> dict[str, list]:
    """Apply ``changes`` (``{key: value}``; ``None`` removes the key) to the ``scope`` params file.

    Returns the exact diff ``{key: [old, new]}`` for the keys that changed (``None`` = absent),
    and reloads :data:`dt.params.P` so the live process sees the new layer (runtime overrides are
    dropped by the reload)."""
    path = layer_path(scope)
    data = read_layer(scope)
    diff: dict[str, list] = {}
    for k, v in changes.items():
        old = data.get(k)
        if v is None:
            if k in data:
                del data[k]
                diff[k] = [old, None]
        elif old != v or k not in data:
            data[k] = v
            diff[k] = [old, v]
    if diff:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    if reload:
        P.reload()
    return diff


def ledger_path(scope: str) -> str:
    """``knowledge/ledger.jsonl`` (global, ``$DT_LEDGER`` overrides) or ``$DT_HOME/ledger.jsonl`` (local)."""
    if scope == "global":
        return os.environ.get("DT_LEDGER") or knowledge("ledger.jsonl")
    if scope == "local":
        return home("ledger.jsonl")
    raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")


def suite_path() -> str:
    """The active scenario suite (``$DT_SCENARIO_SUITE`` overrides ``knowledge/scenarios.json``)."""
    return os.environ.get("DT_SCENARIO_SUITE") or knowledge("scenarios.json")


def real_manifest_path() -> str:
    """Manifest of mined real crops (provenance + criteria only; ``$DT_REAL_MANIFEST`` overrides)."""
    return os.environ.get("DT_REAL_MANIFEST") or knowledge("real_crops.json")


def real_crops_dir() -> str:
    """Third-party crops are stored here, never in git."""
    return home("scenarios", "real")


def local_families_dir() -> str:
    return home("scenarios", "families")


def snapshot(keys: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    """``{key: {value, layer}}`` of the live params (for ledger evidence)."""
    allv, layers = P.all(), P.layers()
    keys = sorted(allv) if keys is None else keys
    return {k: {"value": allv.get(k), "layer": layers.get(k, "default")} for k in keys}
