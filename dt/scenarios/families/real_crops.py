"""real_crops: failures mined from real translate runs (``dt scenario mine``).

Each case is a context crop of a real screenshot around a region / edge tile that failed the
visually-identical gate, translated as a screenshot of its own (perceive -> map -> refine k).
No ground truth: criteria are the validator's measures on the crop, carried per case in the
manifest (``knowledge/real_crops.json``), always with an editability guard (rasterised area).
Seed ``s`` is the crop with permanent manifest ``index`` ``s % n``; pixels come from ``$DT_HOME``.
"""
from __future__ import annotations

from dt.scenarios.mine import load_crop, load_manifest
from dt.scenarios.sources import real_case
from dt.scenarios.spec import Criterion, ScenarioFamily

NAME = "real_crops"


def _entries() -> list[dict]:
    return sorted(load_manifest().get("crops", []), key=lambda c: int(c["index"]))


def _sample(seed: int) -> dict:
    ents = _entries()
    if not ents:
        raise RuntimeError("no mined real crops (run `dt scenario mine <run_dir...>`)")
    i = int(seed) % len(ents)
    e = next((c for c in ents if int(c["index"]) == i), None)
    if e is None:
        raise KeyError(f"no real crop with index {i} (crops must never be deleted from the manifest)")
    return {"index": i, "crop": e["id"]}


def generate(p: dict, seed: int):
    ents = {e["id"]: e for e in _entries()}
    e = ents[p["crop"]]
    rgb = load_crop(e)
    if rgb is None:
        raise FileNotFoundError(f"crop {e['id']} not in $DT_HOME/scenarios/real and its source "
                                f"{(e.get('source') or {}).get('file')} is not in fixtures/screens")
    meta = {"id": e["id"], "criteria": e.get("criteria"), "provenance": {**e.get("source", {}), "box": e["box"],
                                                                         "measure": e.get("measure")}}
    return real_case(NAME, seed, p, rgb, meta)


def _available() -> bool:
    return any(load_crop(e) is not None for e in _entries()[:3])


FAMILY = ScenarioFamily(
    name=NAME,
    description="Context crops of real screens around regions / edge tiles that failed the visually-identical gate.",
    stage="translate",
    refine_iters=3,
    failure_refs=["mined from translate runs' validation/validation.json (worst regions, worst chamfer tile)"],
    param_space={"index": (0, 0)},
    generate=generate,
    criteria=[Criterion("region_de_worst", "<", "param:validate.gate.ident.worst_region_de", scale=10.0),
              Criterion("raster_frac", "<=", "param:validate.gate.raster_frac", scale=0.35)],
    source="real",
    tune_prefixes=("perceive.seg.", "refine.critic.", "refine.raster."),
    sample=_sample,
    fidelity_text="ocr",
    available=_available,
    n_cases=lambda: len(_entries()),
)
