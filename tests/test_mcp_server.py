"""The MCP server exposes the harness loop as tools and they work in-process."""
import asyncio
import json
import os

import pytest

pytest.importorskip("mcp")

from dt import mcp_server as S  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(coro):
    """Drive the server's coroutine from its own thread: the shared Playwright driver may already own an
    event loop on the main thread (earlier tests rendered there)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _call(name, args):
    res = _run(S.server.call_tool(name, args))
    # MCPServer.call_tool returns content blocks and/or structured output depending on SDK version
    if isinstance(res, tuple):
        res = res[-1] if isinstance(res[-1], dict) else res[0]
    if isinstance(res, dict):
        return res.get("result", res)
    if isinstance(res, list):
        return json.loads(res[0].text)
    return getattr(res, "structured_content", None) or json.loads(res.content[0].text)


def test_tools_listed():
    tools = _run(S.server.list_tools())
    names = {t.name for t in (tools.tools if hasattr(tools, "tools") else tools)}
    assert {"translate", "perceive", "render", "validate", "export_figma", "list_decisions", "apply_decisions",
            "bench", "improve_brief", "attribution", "move_stats"} <= names


def test_validate_identical_images_pass(tmp_path):
    png = os.path.join(ROOT, "fixtures", "corpus", "synth", "synth_1_000.png")
    out = _call("validate", {"target_path": png, "render_path": png})
    assert out["gates"]["structurally-faithful"] is True
    assert out["fidelity"]["mean_de"] == 0.0


def test_translate_and_export_roundtrip(tmp_path):
    png = os.path.join(ROOT, "fixtures", "corpus", "synth", "synth_1_000.png")
    out = _call("translate", {"image_path": png, "out_dir": str(tmp_path / "run"), "refine_iters": 0})
    assert out["errors"] == [] and os.path.exists(out["files"]["figma_plan"])
    assert set(out["gates"]) == {"pixel-exact", "visually-identical", "structurally-faithful"}
    plan = _call("export_figma", {"ir_path": out["files"]["ir_mapped"], "out_path": str(tmp_path / "plan.json")})
    assert plan["errors"] == []
    dec = _call("list_decisions", {"run_dir": out["out_dir"]})
    assert "count" in dec
