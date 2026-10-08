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

Layers (lowest to highest precedence; ``P.layers()`` reports which one set each key):
  1. ``default``  -- the value given to ``register()``;
  2. ``global``   -- ``dt/params.json`` (or ``$DT_PARAMS``): learned state that ships with the repo;
  3. ``local``    -- ``$DT_HOME/params.local.json`` (``DT_HOME`` defaults to ``~/.rsdesign``;
     ``$DT_PARAMS_LOCAL`` overrides the path, an empty value disables the layer): a consumer's
     own learned state, written by ``dt train --local`` / the feedback loop, never committed;
  4. ``runtime``  -- ``P.set()`` (the tuner's candidates, ``bench.param_overrides``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

_DEFAULTS: dict[str, Any] = {}
_DOCS: dict[str, str] = {}
_RANGES: dict[str, tuple[float, float]] = {}
_OVERRIDES: dict[str, Any] = {}
_LAYER: dict[str, str] = {}  # key -> "global" | "local" | "runtime" for every key in _OVERRIDES
_FILE_VALUES: dict[str, dict[str, Any]] = {"global": {}, "local": {}}  # persisted layers as loaded

PARAMS_PATH = os.environ.get("DT_PARAMS", os.path.join(os.path.dirname(__file__), "params.json"))


def dt_home() -> str:
    """The consumer's state directory (``$DT_HOME``, default ``~/.rsdesign``); never committed."""
    return os.path.abspath(os.path.expanduser(os.environ.get("DT_HOME") or os.path.join("~", ".rsdesign")))


def local_params_path() -> Optional[str]:
    """Path of the local params layer (``$DT_PARAMS_LOCAL`` or ``$DT_HOME/params.local.json``);
    ``None`` when the layer is disabled (``DT_PARAMS_LOCAL=""``)."""
    env = os.environ.get("DT_PARAMS_LOCAL")
    if env is not None:
        return os.path.abspath(os.path.expanduser(env)) if env.strip() else None
    return os.path.join(dt_home(), "params.local.json")


def _read_json(path: Optional[str]) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def register(key: str, default: Any, doc: str = "", rng: tuple[float, float] | None = None) -> None:
    """Register a tunable. `rng` = (lo, hi) search range for numeric params (None = not tuned)."""
    _DEFAULTS[key] = default
    if doc:
        _DOCS[key] = doc
    if rng is not None:
        _RANGES[key] = rng


def _load_overrides() -> None:
    """(Re)load the global then the local layer; the local layer wins for keys both set."""
    global _OVERRIDES
    glob, loc = _read_json(PARAMS_PATH), _read_json(local_params_path())
    _OVERRIDES = dict(glob)
    _OVERRIDES.update(loc)
    _LAYER.clear()
    _LAYER.update({k: "global" for k in glob})
    _LAYER.update({k: "local" for k in loc})
    _FILE_VALUES["global"], _FILE_VALUES["local"] = glob, loc


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
        # restoring a persisted value (e.g. bench.param_overrides on exit) keeps its layer label
        if key in _FILE_VALUES["local"] and _FILE_VALUES["local"][key] == value:
            _LAYER[key] = "local"
        elif key in _FILE_VALUES["global"] and key not in _FILE_VALUES["local"] and _FILE_VALUES["global"][key] == value:
            _LAYER[key] = "global"
        else:
            _LAYER[key] = "runtime"

    def reset(self, key: str | None = None) -> None:
        if key is None:
            _OVERRIDES.clear()
            _load_overrides()
        else:
            _OVERRIDES.pop(key, None)
            _LAYER.pop(key, None)

    def reload(self) -> None:
        """Re-read the global and local layer files (drops runtime overrides). Same as ``reset()``."""
        self.reset()

    def layers(self) -> dict[str, str]:
        """``{key: layer}`` for every registered or overridden key, where layer is the one that set
        the live value: ``default`` | ``global`` (dt/params.json) | ``local`` ($DT_HOME/params.local.json)
        | ``runtime`` (``P.set``)."""
        out = {k: "default" for k in _DEFAULTS}
        out.update({k: _LAYER.get(k, "global") for k in _OVERRIDES})
        return out

    def layer_values(self, layer: str) -> dict[str, Any]:
        """The values stored in one persisted layer file (``global`` or ``local``) as on disk."""
        if layer == "global":
            return _read_json(PARAMS_PATH)
        if layer == "local":
            return _read_json(local_params_path())
        raise ValueError(f"unknown params layer {layer!r} (global | local)")

    def layer_path(self, layer: str) -> Optional[str]:
        """File backing a persisted layer (``None`` for a disabled local layer)."""
        if layer == "global":
            return PARAMS_PATH
        if layer == "local":
            return local_params_path()
        raise ValueError(f"unknown params layer {layer!r} (global | local)")

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
        """Persist overrides (or `only`) as params.json. Returns path. (Prefer
        ``dt.learn.state.write_layer`` for learned changes: it writes one layer and the ledger.)"""
        path = path or PARAMS_PATH
        data = only if only is not None else {k: v for k, v in _OVERRIDES.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path


P = _Params()
