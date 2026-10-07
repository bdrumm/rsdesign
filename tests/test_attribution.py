"""Stage attribution runs the bench ladder and splits the composite gap across stages."""
from dt.selftest import attribution


def test_attribution_ladder_is_consistent():
    res = attribution.run(limit=1, workers=1)
    c, k = res["composite"], res["stage_cost"]
    assert set(c) == {"oracle_render", "oracle_map", "full"}
    # the costs telescope: 1 - full == render + map + perceive
    assert abs((1 - c["full"]) - (k["render"] + k["map"] + k["perceive"])) < 1e-9
    # rendering the ground truth IR must be near-perfect on the corpus it was extracted from
    assert c["oracle_render"] > 0.9
    assert res["largest"] in k
    md = attribution.markdown(res)
    assert "Largest cost" in md
