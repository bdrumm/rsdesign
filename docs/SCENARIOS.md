# Failure scenarios: every failure becomes a family that gates and trains the pipeline

The bench (`dt bench`) scores 24 fixed pages. A failure found on a real screen is one data point;
fixing that page alone is overfitting. The scenario harness turns each failure *mode* into a
**family**: a seeded generator of many cases spanning the mode, plus pass/fail criteria measured
with the same independent measures as the validator. Families join the regression gate and
drive parameter tuning, and every learned change lands in one ledger with its evidence and an
exact undo. This is how the engine refactors itself from its own failures (mined from runs and
`knowledge/failures.jsonl`) and from its consumers' failures (local families and local params).

## The protocol

```
failure  ->  family  ->  baseline  ->  fix / tune on TRAIN seeds  ->  HOLDOUT + global gates  ->  ledger
```

1. **Failure.** From `knowledge/failures.jsonl`, a translate run (`validation/validation.json`) or
   a consumer report. Mine real runs with `dt scenario mine out/run1 [...]`: crops around every
   region / edge tile that broke the visually-identical gate become `real_crops` cases (pixels
   in `$DT_HOME/scenarios/real/`, never in git; provenance and criteria in
   `knowledge/real_crops.json`; each crop keeps a permanent `index` = its scenario seed, so never
   delete one). New crops join the gate when `real_crops` is re-promoted.
2. **Family.** Write `dt/scenarios/families/<name>.py` (below). Model the *mode* with wide priors
   (theme, sizes, colours, counts, layout), not the page you saw.
3. **Baseline.** `dt scenario run <name> --n 12` to look at it (worst cases with diff images in
   `out/scenarios/<run>/<name>/`), then `dt scenario promote <name>`: runs the TRAIN seeds and the
   disjoint HOLDOUT seeds, records the baseline pass rates in `knowledge/scenarios.json` and a
   `scenario_baseline` ledger entry.
4. **Fix or tune on TRAIN seeds only.** Code fixes: iterate with `dt scenario run <name>` (train
   seeds `0..n-1`). Parameter fixes: `dt train --family <name> --iters N` searches the params
   relevant to the family (its `tune_prefixes`, or `--params a.,b.`) with the mean clipped criterion
   margin over the train seeds as the objective. Candidates perturb at most
   `scenarios.train.max_keys` keys, an Occam step drops every changed key the gain does not need,
   and a gain below `scenarios.train.min_gain` is not even gated.
5. **Gates.** A change is kept only if the family's HOLDOUT pass rate strictly improves, every
   other active family keeps `holdout pass rate >= baseline - scenarios.gate.eps`, and the global
   bench gate passes (`dt bench --gate --scenarios` runs both gates; `dt train` runs all three
   automatically, cheapest first).
6. **Ledger.** `dt train` writes an accepted change to `dt/params.json` (global) or
   `$DT_HOME/params.local.json` (`--local`) and appends a `params` entry to the ledger with the
   exact diff and the evidence. `dt learn log | show <id> | revert <id>`. After a code fix,
   re-promote the family so the new pass rate becomes the floor (`dt scenario promote <name>`; the
   ledger keeps the old baseline and `dt learn revert` restores it).

Rules: never tune or iterate on HOLDOUT seeds · never weaken a criterion to pass (criteria that
reference gate thresholds use `param:validate.gate.*`, which no tuner may touch) · every family
names the failures it generalises (`failure_refs`) · a fix that helps its family but costs another
family or the bench is rejected — that is the overfitting signal, see `dt train --report-generalisation`.

## Active suite and baselines (this commit)

| family | stage | source | train / holdout cases | train pass | holdout pass | what fails today (criterion pass rate, train / holdout) |
|---|---|---|---|---|---|---|
| `pale_surface` | perceive | IR | 8 / 10 | 0.375 | 0.400 | surface found 0.50 / 0.60 (≈2 ΔE surfaces never segmented; dark full-width bands repaint the page); ROI JND 0.75 / 0.50 |
| `page_tint` | translate, refine 4 | IR | 8 / 10 | 0.750 | 0.700 | whole-page JND 0.75 / 0.70 (tinted page or near-white bar lost when they are 1-3 ΔE apart) |
| `custom_glyph` | translate, refine 4 | HTML | 8 / 10 | 0.000 | 0.100 | glyph reproduced 0.12 / 0.10 (small marks render as nothing); not misnamed 0.62 / 0.80 (triangles/dots named as Material icons) |
| `desktop_list` | perceive | HTML | 8 / 10 | 0.000 | 0.000 | rows found 0.0 / 0.0 (tinted read rows ≈2 ΔE from white merge into one rect); text segments 1.0 / 0.9; no merged lines 1.0 / 1.0 |
| `real_crops` | translate, refine 3 | real (23 mined crops) | 12 / 11 | 0.167 | 0.000 | worst region ΔE < 5: 0.0 / 0.0; worst tile chamfer < 2: 0.4 / 0.0; editability 1.0 / 1.0 |

Measured on the pipeline at main @ 82e37e2 (Chrome + Vision). `dt scenario report` prints the
live table; these numbers are the floor fixers drive up. The scenario gate on the 51 holdout
cases takes about 2 minutes with 4 workers and is deterministic (two runs, zero drift).

## Families shipped

| family | failure refs | stage | criteria |
|---|---|---|---|
| `pale_surface` | R3 pale illustration card #f3edf7 on #fdf7fe dropped by segmentation | perceive | every surface found as a filled, non-raster node (IoU >= `scenarios.pale.match_iou`, fill within `scenarios.pale.fill_de` ΔE2000); surfaces' ROI non-text JND fraction <= visually-identical gate |
| `page_tint` | R3 whole page background ~2.5 ΔE off (root fill from a white app bar) | translate (refine 4) | whole-page non-text JND fraction <= visually-identical gate |
| `custom_glyph` | R3 sparkle bullets render as nothing; R4 hexagon/ring named as Material icons | translate (refine 4) | every glyph's box within `scenarios.glyph.max_de` mean ΔE2000; no glyph carries a Material `icon_name` |
| `desktop_list` | R4 dense 1280 px list: lost ' - ' / '&' / '/', grouping blow-up; R3 unpainted list items | perceive | >= 95% text segments recovered (line-fair CER), no predicted line spanning two segments, >= 90% rows found as containers |
| `real_crops` | mined from translate runs | translate (refine 3) | per crop: worst region ΔE (or worst edge tile chamfer) under the visually-identical gate, rasterised area <= `validate.gate.raster_frac` |

Sources: **IR-generated** cases render a `Document` with `render_doc` (the IR is the ground truth;
text boxes measured in the browser). **HTML-generated** cases lay out real HTML/CSS/SVG in the
browser (Roboto, Material Symbols, M3 tokens of the corpus pages) and take the ground truth from
the DOM with the Material Web corpus extractor (`dt.selftest.mwc_corpus.EXTRACT_JS`).
**Real crops** have pixels only; their criteria are fidelity measures on the crop (OCR'd text
excluded) plus the editability guard, so pasting the target back cannot pass.

## Measures

Every case runs a pipeline subset (`perceive`; `map` = perceive + map; `translate` = perceive ->
map -> refine k (no wall-clock budget, deterministic) -> re-map), renders the result and measures,
on the whole frame and on the case ROI:

* structural vs gt (`dt.compare.structural_metrics` on the nodes inside the ROI): `node_recall`,
  `node_precision`, `mean_iou`, `color_de`, `text_cer`, `text_recall`, `component_acc`, ...
* fidelity (`dt.validate.fidelity`, on the target frame): `jnd_frac_nontext`, `mean_de`,
  `chamfer`, `chamfer_tile_max`, `edge_within1`, `edge_f1`, `region_de_worst`, `region_de_mean`,
  `raster_frac`;
* family metrics (`surface_recall`, `glyph_reproduced`, `glyph_misnamed`, `text_line_recall`,
  `merged_lines`, `row_recall`, ...).

A criterion is `metric op threshold` (`roi="roi"` for the ROI scope); its margin is the signed
distance to the threshold divided by `scale`. A case passes when all its criteria pass. The
tuning objective is the mean margin clipped to `[-1, scenarios.margin_cap]` (failing cases
dominate; over-satisfying a criterion earns little).

## Learned state: global vs local, and the ledger

| | global (ships with the repo) | local (one consumer, never committed) |
|---|---|---|
| params | `dt/params.json` | `$DT_HOME/params.local.json` (loaded after the global file; `P.layers()` says which layer set each key) |
| ledger | `knowledge/ledger.jsonl` | `$DT_HOME/ledger.jsonl` |
| scenarios | `knowledge/scenarios.json`, `knowledge/real_crops.json` | `$DT_HOME/scenarios/families/*.py`, `$DT_HOME/scenarios/real/` |

`DT_HOME` defaults to `~/.rsdesign`; `DT_PARAMS_LOCAL=""` disables the local layer. Ledger
entries (written by both this harness and the feedback loop):

```json
{"id": "params-3f9a0c12d4", "ts": "2026-10-07T21:40:00+02:00", "kind": "params", "scope": "global",
 "source": {"type": "scenario", "ids": ["pale_surface"]},
 "change": {"params": {"perceive.seg.bg_dist": [null, 9.5]}, "layer": "global"},
 "evidence": {"before": {"train_objective": -0.31, "pass_rate": 0.6}, "after": {"train_objective": -0.12, "pass_rate": 0.8},
              "gates": {"holdout": true, "scenarios": true, "bench": true}, "generalisation": {...}},
 "accepted": true, "reverts": null}
```

`kind` is one of `params | rule | scenario_baseline | feedback_promotion | revert`. In a `params`
change `null` means "absent from that layer file"; `dt learn revert <id>` restores the old values
exactly (removing keys that were absent), refuses when the layer has drifted since (`--force`),
and appends a `revert` entry.

## Generalisation (overfitting) report

`dt train --family F --report-generalisation` runs every guard even after a failure and reports,
for the candidate change, each other family's holdout pass rate / objective before and after and
the bench gate result; it is stored in the ledger evidence. `dt train --report-generalisation`
(no family) builds the leave-one-family-out matrix: tune on each active family in turn (nothing
written) and measure the change on all the others and the bench.

### First runs of the loop (evidence, nothing written)

* `pale_surface`, `perceive.seg.*` (51 keys), 12 iterations: before the sparse search and Occam
  step, the best candidate moved 15 keys for a +0.0008 train gain and was rejected by two
  guards: `page_tint` holdout fell 0.7 -> 0.5 and the bench gate failed (component/token accuracy
  -0.02). With sparse candidates and pruning, the same gain reduces to one key
  (`perceive.seg.fill_tol` 3.0), below `scenarios.train.min_gain`. Conclusion: this family needs
  a segmentation code fix, not a threshold.
* `custom_glyph`, `refine.raster.{min_side,min_de,min_texture,min_parts}` +
  `perceive.icons.{accept_mis,accept_de,min_score}`, 10 iterations: pruned to
  `refine.raster.min_side` 20 -> 8 and `refine.raster.min_texture` 0.12 -> 0.27. Train objective
  -0.535 -> -0.410 and holdout objective -0.483 -> -0.392. It generalises: `real_crops` holdout
  0.000 -> 0.091, the other families are unchanged, and the bench composite is identical. It
  was still rejected because the holdout pass rate did not strictly improve (0.1 -> 0.1). This
  is a strong lead for a fixer: small marks need a raster or vector path below 20 px.

## How an AI driver adds a family

```python
# dt/scenarios/families/my_failure.py   (or $DT_HOME/scenarios/families/ for a local one)
from dt.ir import Box, Document
from dt.scenarios.families._common import rect, rng_for, text_node
from dt.scenarios.sources import ir_case          # or html_case(...) for real browser layout + DOM gt
from dt.scenarios.spec import Criterion, ScenarioFamily

def generate(p: dict, seed: int):
    rng = rng_for(seed, "my_failure")             # all randomness from the seed
    doc = Document.blank(p["width"], 400, "#ffffff")
    ...                                           # build the failure from the params
    return ir_case("my_failure", seed, p, doc, roi=Box(...), meta={"facts": ...})

def metrics(case, pred, rendered) -> dict:        # optional family measures
    return {"thing_found": ...}

FAMILY = ScenarioFamily(
    name="my_failure", description="...", stage="perceive",           # perceive | map | translate
    failure_refs=["<case + symptom from knowledge/failures.jsonl>"],
    param_space={"width": (320, 1280), "theme": ["light", "dark"], ...},   # wide priors
    generate=generate, metrics=metrics,
    criteria=[Criterion("thing_found", ">=", 1.0),
              Criterion("jnd_frac_nontext", "<=", "param:validate.gate.ident.jnd_frac", roi="roi", scale=0.05)],
    tune_prefixes=("perceive.seg.",), refine_iters=0)
```

Then: `dt scenario run my_failure --n 12` (inspect the worst cases), `python -m pytest
tests/test_scenarios.py`, `dt scenario promote my_failure`, commit the family +
`knowledge/scenarios.json` + `knowledge/ledger.jsonl`. Bump `version` whenever the generator's
output changes (the gate refuses to compare against a baseline of another version).

## Commands

```bash
dt scenario list | report                      # families, active suite, baselines
dt scenario run <family> --n 8 --seed 0        # [--holdout] [--stage translate --refine-iters 4] [--workers 3]
dt scenario mine <run_dir...>                  # real crops from translate runs
dt scenario promote <family>                   # [--n-train 8 --n-holdout 10]
dt scenario gate [--families a,b]              # holdout pass rates vs baselines
dt bench --gate --scenarios --workers 3        # global bench gate + scenario gate
dt train --family <f> [--params a.,b.] --iters 12 [--local] [--report-generalisation] [--record-rejected]
dt train --report-generalisation --iters 6     # leave-one-family-out matrix
dt learn log | show <id> | revert <id>
```
