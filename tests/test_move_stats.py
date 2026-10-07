"""Move statistics split batch gains and rank kinds by gain per try."""
import json

from dt.selftest import move_stats


def test_summary_splits_batch_gain(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    hist = [
        {"iter": 1, "kind": "shift", "accepted": True, "before": 1.0, "after": 0.8, "batch": 2},
        {"iter": 1, "kind": "recolor", "accepted": True, "before": 1.0, "after": 0.8, "batch": 2},
        {"iter": 2, "kind": "missing", "accepted": False, "before": 0.8, "after": 0.81, "batch": 1},
        {"iter": 2, "kind": "stop", "accepted": True, "before": 1.0, "after": 0.8},
    ]
    (run / "refine_history.json").write_text(json.dumps(hist))
    s = move_stats.summarise(move_stats.collect([str(tmp_path)]))
    assert s["moves"] == 3 and s["runs"] == 1
    assert abs(s["kinds"]["shift"]["gain"] - 0.1) < 1e-9 and abs(s["kinds"]["recolor"]["gain"] - 0.1) < 1e-9
    assert s["kinds"]["missing"]["accept_rate"] == 0.0
    assert "| missing |" in move_stats.markdown(s)
