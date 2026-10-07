"""Shared pytest configuration.

Marks the tests listed in tests/platform_sensitive.txt with ``platform_sensitive``: their outcome depends
on the OCR engine version and the browser's font rasterisation. A plain ``pytest`` run still runs them
(the reference machine must pass everything); CI deselects them in the blocking step
(``-m "not platform_sensitive"``) and runs them separately as a non-blocking calibration signal.
"""
from __future__ import annotations

import os

_LIST = os.path.join(os.path.dirname(__file__), "platform_sensitive.txt")


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
    import pytest
    sensitive = _sensitive()
    if not sensitive:
        return
    for item in items:
        base = item.nodeid.split("[", 1)[0]
        if base in sensitive:
            item.add_marker(pytest.mark.platform_sensitive)
