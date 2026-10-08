"""Learned-state layers (dt.params global/local overlay) and the shared learning ledger."""
from __future__ import annotations

import json
import os

import pytest

import dt.params as dtp
from dt.learn import ledger, state
from dt.params import P

KEY = "tune.sigma"          # registered with a range at import of dt.selftest.tune
KEY2 = "tune.coord_step"


@pytest.fixture()
def layers(tmp_path, monkeypatch):
    """Isolated global + local params files, DT_HOME and ledgers."""
    import dt.selftest.tune  # noqa: F401 - registers KEY / KEY2
    glob = tmp_path / "params.json"
    home = tmp_path / "home"
    monkeypatch.setattr(dtp, "PARAMS_PATH", str(glob))
    monkeypatch.setenv("DT_HOME", str(home))
    monkeypatch.delenv("DT_PARAMS_LOCAL", raising=False)
    monkeypatch.setenv("DT_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("DT_SCENARIO_SUITE", str(tmp_path / "scenarios.json"))
    P.reload()
    yield {"global": glob, "local": home / "params.local.json", "home": home, "tmp": tmp_path}
    monkeypatch.undo()
    P.reload()


def test_local_overlay_precedence_and_layers(layers):
    default = P.defaults()[KEY]
    assert P[KEY] == default and P.layers()[KEY] == "default"
    layers["global"].write_text(json.dumps({KEY: 0.3, KEY2: 0.1}))
    P.reload()
    assert P[KEY] == 0.3 and P.layers()[KEY] == "global"
    os.makedirs(layers["home"], exist_ok=True)
    layers["local"].write_text(json.dumps({KEY: 0.4}))
    P.reload()
    assert P[KEY] == 0.4 and P.layers()[KEY] == "local"        # local wins over global
    assert P[KEY2] == 0.1 and P.layers()[KEY2] == "global"     # untouched global key stays global
    P.set(KEY, 0.5)
    assert P[KEY] == 0.5 and P.layers()[KEY] == "runtime"
    P.set(KEY, 0.4)                                           # restoring the persisted value keeps its layer
    assert P.layers()[KEY] == "local"
    assert P.layer_values("local") == {KEY: 0.4}
    P.reload()
    assert P[KEY] == 0.4


def test_local_layer_can_be_disabled(layers, monkeypatch):
    os.makedirs(layers["home"], exist_ok=True)
    layers["local"].write_text(json.dumps({KEY: 0.4}))
    monkeypatch.setenv("DT_PARAMS_LOCAL", "")
    P.reload()
    assert P[KEY] == P.defaults()[KEY]
    with pytest.raises(RuntimeError):
        state.layer_path("local")


def test_write_layer_returns_exact_diff(layers):
    layers["global"].write_text(json.dumps({KEY2: 0.1}))
    diff = state.write_layer("global", {KEY: 0.33, KEY2: 0.1})
    assert diff == {KEY: [None, 0.33]}                         # unchanged keys are not part of the diff
    assert json.loads(layers["global"].read_text()) == {KEY: 0.33, KEY2: 0.1}
    assert P[KEY] == 0.33
    diff = state.write_layer("global", {KEY2: None})
    assert diff == {KEY2: [0.1, None]} and KEY2 not in json.loads(layers["global"].read_text())


def test_ledger_entry_schema():
    e = ledger.make_entry("params", "global", "scenario", ["fam"], {"params": {KEY: [None, 1.0]}, "layer": "global"})
    assert e["id"].startswith("params-") and not ledger.validate_entry(e)
    for bad in ({**e, "kind": "nope"}, {**e, "scope": "team"}, {**e, "source": {"type": "x", "ids": []}},
                {k: v for k, v in e.items() if k != "evidence"}):
        assert ledger.validate_entry(bad)
    with pytest.raises(ValueError):
        ledger.make_entry("bogus", "global", "scenario", [], {})


@pytest.mark.parametrize("scope", ["global", "local"])
def test_ledger_revert_restores_params_exactly(layers, scope):
    path = layers[scope]
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(json.dumps({KEY2: 0.11}))
    before = json.loads(path.read_text())
    diff = state.write_layer(scope, {KEY: 0.42, KEY2: 0.2})
    e = ledger.append(ledger.make_entry("params", scope, "scenario", ["fam"], {"params": diff, "layer": scope}))
    assert ledger.get(e["id"])["id"] == e["id"]
    assert P[KEY] == 0.42 and P[KEY2] == 0.2
    rev = ledger.revert(e["id"])
    assert rev["kind"] == "revert" and rev["reverts"] == e["id"]
    assert json.loads(path.read_text()) == before               # KEY removed again, KEY2 back to 0.11
    assert P[KEY2] == 0.11 and P[KEY] == P.defaults()[KEY]
    assert ledger.reverted_by(e["id"])["id"] == rev["id"]
    with pytest.raises(ValueError, match="already reverted"):
        ledger.revert(e["id"])
    ids = [x["id"] for x in ledger.read()]
    assert e["id"] in ids and rev["id"] in ids


def test_ledger_revert_refuses_drift_unless_forced(layers):
    diff = state.write_layer("global", {KEY: 0.42})
    e = ledger.append(ledger.make_entry("params", "global", "manual", [], {"params": diff, "layer": "global"}))
    state.write_layer("global", {KEY: 0.5})                    # someone changed it since
    with pytest.raises(ValueError, match="changed since"):
        ledger.revert(e["id"])
    ledger.revert(e["id"], force=True)
    assert KEY not in json.loads(layers["global"].read_text())


def test_rejected_entries_are_not_revertible(layers):
    e = ledger.append(ledger.make_entry("params", "global", "scenario", ["f"], {"params": {KEY: [None, 1.0]}, "layer": "global"},
                                        accepted=False))
    with pytest.raises(ValueError, match="not accepted"):
        ledger.revert(e["id"])


def test_suite_change_revert(layers):
    from dt.scenarios import suite
    s = suite.load()
    old = None
    new = {"family_version": 1, "train_seeds": [0], "holdout_seeds": [1], "baseline": {"holdout_pass_rate": 0.5}}
    s["families"]["fam"] = new
    suite.save(s)
    e = ledger.append(ledger.make_entry("scenario_baseline", "global", "scenario", ["fam"], {"suite": {"fam": [old, new]}}))
    ledger.revert(e["id"])
    assert "fam" not in suite.load()["families"]


def test_cli_learn_log_and_show(layers, capsys):
    from dt.cli import main
    diff = state.write_layer("global", {KEY: 0.42})
    e = ledger.append(ledger.make_entry("params", "global", "manual", [], {"params": diff, "layer": "global"}))
    assert main(["learn", "log", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [x["id"] for x in out] == [e["id"]]
    assert main(["learn", "show", e["id"][:12]]) == 0
    assert e["id"] in capsys.readouterr().out
    assert main(["learn", "revert", e["id"]]) == 0
    assert main(["learn", "revert", e["id"]]) == 1             # refused the second time
