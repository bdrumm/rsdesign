"""Central registry of tunable parameters.

Every heuristic threshold in the pipeline reads from here, so the self-tuning loop
(`dt.selftest.tune`) can search over them and persist the best set to `dt/params.json`.

Usage:
    from dt.params import P
    thr = P["perceive.edge.canny_low"]

Rules:
  * Register defaults with `register()` at import time of the owning module.
  * Never hardcode a threshold in a module; register it with a short doc string.
  * `params.json` overrides defaults; `tune` writes it; humans may edit it.
  * Keys are dotted: "<stage>.<submodule>.<name>".
"""
from __future__ import annotations

import json
import os
from typing import Any

_DEFAULTS: dict[str, Any] = {}
_DOCS: dict[str, str] = {}
_RANGES: dict[str, tuple[float, float]] = {}
_OVERRIDES: dict[str, Any] = {}

PARAMS_PATH = os.environ.get("DT_PARAMS", os.path.join(os.path.dirname(__file__), "params.json"))


def register(key: str, default: Any, doc: str = "", rng: tuple[float, float] | None = None) -> None:
    """Register a tunable. `rng` = (lo, hi) search range for numeric params (None = not tuned)."""
    _DEFAULTS[key] = default
    if doc:
        _DOCS[key] = doc
    if rng is not None:
        _RANGES[key] = rng


def _load_overrides() -> None:
    global _OVERRIDES
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH) as f:
                _OVERRIDES = json.load(f)
        except Exception:
            _OVERRIDES = {}


_load_overrides()


class _Params:
    def __getitem__(self, key: str) -> Any:
        if key in _OVERRIDES:
            return _OVERRIDES[key]
        if key in _DEFAULTS:
            return _DEFAULTS[key]
        raise KeyError(f"unregistered param {key!r}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        """Runtime override (not persisted). Used by the tuner during search."""
        _OVERRIDES[key] = value

    def reset(self, key: str | None = None) -> None:
        if key is None:
            _OVERRIDES.clear()
            _load_overrides()
        else:
            _OVERRIDES.pop(key, None)

    def all(self) -> dict[str, Any]:
        d = dict(_DEFAULTS)
        d.update(_OVERRIDES)
        return d

    def defaults(self) -> dict[str, Any]:
        return dict(_DEFAULTS)

    def ranges(self) -> dict[str, tuple[float, float]]:
        return dict(_RANGES)

    def docs(self) -> dict[str, str]:
        return dict(_DOCS)

    def save(self, path: str | None = None, only: dict[str, Any] | None = None) -> str:
        """Persist overrides (or `only`) as params.json. Returns path."""
        path = path or PARAMS_PATH
        data = only if only is not None else {k: v for k, v in _OVERRIDES.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path


P = _Params()
