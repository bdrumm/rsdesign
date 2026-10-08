"""Shared pytest configuration.

1. Test-session isolation: user state (``$DT_HOME``: feedback bundles, local learned rules, local ledger,
   mined real crops) goes to a temporary directory, never to the developer's ``~/.rsdesign`` (e.g.
   ``apply_decisions`` logs feedback).
2. Marks the tests listed in tests/platform_sensitive.txt with ``platform_sensitive``: their outcome depends
   on the OCR engine version and the browser's font rasterisation. A plain ``pytest`` run still runs them
   (the reference machine must pass everything); CI deselects them in the blocking step
   (``-m "not platform_sensitive"``) and runs them separately as a non-blocking calibration signal.
"""
from __future__ import annotations

import os
import tempfile

import pytest

_LIST = os.path.join(os.path.dirname(__file__), "platform_sensitive.txt")


@pytest.fixture(autouse=True, scope="session")
def _isolated_dt_home():
    old = os.environ.get("DT_HOME")
    with tempfile.TemporaryDirectory(prefix="dt_home_") as d:
        os.environ["DT_HOME"] = d
        try:
            yield d
        finally:
            if old is None:
                os.environ.pop("DT_HOME", None)
            else:
                os.environ["DT_HOME"] = old


def _sensitive() -> set[str]:
    out: set[str] = set()
    if os.path.exists(_LIST):
        for line in open(_LIST):
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


def pytest_configure(config):
    config.addinivalue_line("markers", "platform_sensitive: depends on the OCR engine / font rasterisation of the "
                                       "reference machine (see tests/platform_sensitive.txt)")


def pytest_collection_modifyitems(config, items):
    sensitive = _sensitive()
    if not sensitive:
        return
    for item in items:
        if item.nodeid.split("[", 1)[0] in sensitive:
            item.add_marker(pytest.mark.platform_sensitive)
