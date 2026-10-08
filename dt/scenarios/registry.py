"""Family registry: auto-discovers ``dt/scenarios/families/*.py`` (global, shipped) and
``$DT_HOME/scenarios/families/*.py`` (local, a consumer's own failure families).

A family module exposes ``FAMILY`` (one :class:`~dt.scenarios.spec.ScenarioFamily`) or
``FAMILIES`` (a list). Modules whose name starts with ``_`` are helpers and are skipped.
A local family with the same name as a global one is ignored (globals cannot be shadowed).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import sys
from typing import Optional

from dt.scenarios.spec import ScenarioFamily

_CACHE: Optional[dict[str, ScenarioFamily]] = None
_ERRORS: dict[str, str] = {}


def _from_module(mod) -> list[ScenarioFamily]:
    out: list[ScenarioFamily] = []
    fam = getattr(mod, "FAMILY", None)
    if isinstance(fam, ScenarioFamily):
        out.append(fam)
    for f in getattr(mod, "FAMILIES", []) or []:
        if isinstance(f, ScenarioFamily):
            out.append(f)
    return out


def _discover() -> dict[str, ScenarioFamily]:
    from dt.scenarios import families as pkg
    found: dict[str, ScenarioFamily] = {}
    _ERRORS.clear()
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        name = f"{pkg.__name__}.{info.name}"
        try:
            mod = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - one broken family must not hide the others
            _ERRORS[name] = f"{type(e).__name__}: {e}"
            continue
        for f in _from_module(mod):
            found.setdefault(f.name, f)
    from dt.learn.state import local_families_dir
    d = local_families_dir()
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            mod_name = f"dt_local_scenarios_{fn[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, os.path.join(d, fn))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            except Exception as e:  # noqa: BLE001
                _ERRORS[os.path.join(d, fn)] = f"{type(e).__name__}: {e}"
                continue
            for f in _from_module(mod):
                if f.name not in found:
                    f.failure_refs = list(f.failure_refs)
                    found[f.name] = f
    return found


def families(refresh: bool = False) -> dict[str, ScenarioFamily]:
    """``{name: family}`` of every discovered family."""
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = _discover()
    return dict(_CACHE)


def get(name: str) -> ScenarioFamily:
    fams = families()
    if name not in fams:
        raise KeyError(f"unknown scenario family {name!r}; known: {sorted(fams)}"
                       + (f"; import errors: {_ERRORS}" if _ERRORS else ""))
    return fams[name]


def register(family: ScenarioFamily) -> None:
    """Register a family programmatically (tests, notebooks)."""
    families()
    assert _CACHE is not None
    _CACHE[family.name] = family


def errors() -> dict[str, str]:
    families()
    return dict(_ERRORS)
