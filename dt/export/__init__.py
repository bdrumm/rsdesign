"""Export stage: IR -> Figma build plan JSON (+ plugin that builds the nodes) and run reports.

    from dt.export import to_build_plan, validate_plan, save_plan, write_report

The Figma plugin lives in ``dt/export/figma_plugin`` (manifest.json / code.js / ui.html) and can be
executed headlessly with ``node dt/export/figma_plugin/run_fake.js --plan plan.json``.
"""
from __future__ import annotations

import os

from dt.export.figma_json import PLAN_VERSION, font_style, gradient_handles, save_plan, to_build_plan, validate_plan
from dt.export.report import write_report

PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figma_plugin")
RUN_FAKE_JS = os.path.join(PLUGIN_DIR, "run_fake.js")

__all__ = [
    "PLAN_VERSION", "PLUGIN_DIR", "RUN_FAKE_JS",
    "to_build_plan", "validate_plan", "save_plan", "font_style", "gradient_handles", "write_report",
]
