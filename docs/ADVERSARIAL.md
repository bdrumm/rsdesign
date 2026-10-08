# Adversarial comparison: a recursive improvement model driven by typed findings

The thesis of this repository is **self-informed and user-informed tuning**: the engine refactors
itself from its own failures and from its consumers' use cases. This document describes the research
track that makes that loop *typed*. An adversary compares a target screenshot with our render and says
*what* is wrong, *where*, and *how much*. Those typed findings then drive four things:

* which refine operator to run,
* which scenario family to generate,
* which parameters to tune,
* which new failure types to add.

An anti-overfitting protocol sits between every proposed change and the shipped state.

Everything below is measured. Numbers come from `out/adversary/` (git-ignored), and how to regenerate
them is at the end. Code:

* `dt/adversary/ensemble.py`: fusion of the approaches.
* `dt/adversary/policy.py`: finding type → refine operator.
* `dt/adversary/novelty.py`: new types → scenario stubs.
* `dt/adversary/protocol.py`: the gate.
* Tests: `tests/test_adversary_synthesis.py`.

The shared benchmark, the taxonomy and the four approaches are documented in their own reports:
`out/adversary/benchmark_baseline.md`, `decompose.md`, `ir-space.md`, `a-contrario.md` and `metamorphic.md`.

---

## 1. The ingredients (recap)

| piece | what it is |
|---|---|
| **Taxonomy v1.0** (`dt/adversary/taxonomy.py`) | 18 error leaves in 7 parents (geometry, color, structure, text, icon, effect, layout), plus 4 `noise.*` leaves an adversary must stay silent on. Each leaf names the IR property at fault and the critic kinds that usually repair it. |
| **Benchmark** (`dt/adversary/benchmark.py`) | Clean ground-truth IR documents with 1–4 typed perturbations, each verified in the real renderer, plus pure-noise cases (sub-pixel offset, font smoothing, blur, JPEG, another browser build). The CALIBRATION split has 48 cases on 12 documents. EVALUATION has 64 cases on 16 *other* documents with other seeds, and includes the held-out generator family `synth_gen`. Headline = 0.4·detection F1 + 0.3·macro leaf F1 + 0.3·(1 − noise false-case rate)·recall. |
| **decompose** | Explains the difference image as a sum of causes: global tint, an explicit renderer-noise model, optical-flow geometry, colour transfer, and residual structure. Renderer-free, 0.3 s per case. |
| **ir-space** | Runs our own perceiver on both images, diffs the two readings, and backs every finding with pixel evidence. |
| **a-contrario** | Registers the nuisance in the real renderer, then tests multi-scale tiles against a calibrated H0 (NFA < 1). An interpretable forest assigns the type. |
| **metamorphic** | A differential translator: perceives both images with the same translator and diffs the two IRs. It also checks exact relations (translate, recolor, dpr2, rows, crop, mirror, jpeg) on any screenshot, with no ground truth. |

## 2. Novel methods in this synthesis

1. **Supervised Dawid–Skene fusion with hierarchical shrinkage** (`ensemble.fuse`).
   - **Clustering.** The four approaches' findings are clustered into candidate *causes* by greedy
     multi-way matching. Each approach casts one vote per cluster; duplicates from the same approach
     are kept as fragments.
   - **Annotator model.** Each approach is treated as a noisy annotator with three learned parts:
     - a per-class firing rate P(fire | y);
     - a confidence-bucket likelihood;
     - a confusion row P(type_a = t | fire, y).

     These tables are Beta/Dirichlet-smoothed towards approach-level and parent-level priors.
   - **Output.** The posterior over `types ∪ {none}` gives the fused type and a *calibrated*
     confidence, 1 − P(none | votes).
   - **Silence is evidence.** If an approach that reliably fires on a type stays silent, the
     posterior for that type drops.
2. **Fusion-calibration on fresh seeds.** Every approach was tuned on CALIBRATION seed 0, so its
   reliability measured there is optimistic. The fusion is therefore fitted on 144 *new* cases: the
   benchmark's own builder run with seeds 101–103 on the 12 calibration documents. No approach was
   tuned on these seeds, and no evaluation document is used.
3. **Replay-learned operator policy** (`policy.py`). For every ground-truth finding, each of the
   critic's own hypotheses of each operator kind that touches it is rendered in the real renderer,
   giving P(proposed | type, kind), P(fix | proposed, type, magnitude tercile, kind), the gain, and
   the renders per try. The policy's priors are the taxonomy's `refine_kinds`; the data overrules
   them where they are wrong. Typed findings are noisy, so the policy works in two steps:
   - It *marginalises over the runtime finder's confusion*: P(fix | finding typed t) = Σ_y
     P(y | t)·P(fix | y). The confusion is learned on the same fresh seeds.
   - The guided critic prunes *safely*. After a guided iteration that accepts nothing, it falls back
     to the full critic list for one iteration, and it never returns an empty list.
4. **Conformal novelty with a multi-page support rule** (`novelty.py`).
   - Every residual region and every fused finding gets 16 interpretable features.
   - Novelty is the k-NN distance to the *known* distribution (fusion-calibration items labelled by
     ground truth), turned into a conformal p-value against the known items' leave-one-out distances.
   - Candidates are clustered with DBSCAN on clipped z-scores.
   - A cluster is promoted to a candidate type only if it appears on at least 2 distinct pages. A
     cluster seen on one page is a one-off and is never promoted.
   - Each promoted cluster gets a draft `ScenarioFamily` stub.
5. **Metamorphic relations as a regulariser on the adversaries themselves** (`protocol.py`). These
   relations constrain any adversary, whatever its parameters:
   - identity: `find(T, T)` reports nothing;
   - nuisance invariance: JPEG q92 or a 0.25 px offset on the target changes nothing;
   - translation equivariance: padding both images moves every finding by the pad;
   - swap symmetry: `find(C, T)` reports the same places as `find(T, C)`.

   They need no labels. A change that wins the benchmark but breaks them is rejected.

## 3. Comparative results (EVALUATION split, 64 cases, 107 ground-truth findings)

All combiners are fitted on the fusion-calibration set unless the row says otherwise.
"Oracle" rows are fitted *on evaluation itself*. They are shown only to measure how much a
select-on-evaluation would inflate the number, and they are never shipped.

| adversary | headline | det P | det R | det F1 | AP | macro leaf F1 | type acc (leaf) | noise cases with a finding | spurious / noisy-target case | s/case* | recall mwc / synth / synth_gen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| null | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/12 | 0.00 | 0.00 | 0 / 0 / 0 |
| baseline_residual (floor) | 0.122 | 7.6 | 88.8 | 13.9 | 31.4 | 0.0 | 5.3 | 9/12 | 24.45 | 0.08 | 93 / 89 / 81 |
| metamorphic | 0.420 | 56.2 | 46.7 | 51.0 | 31.7 | 33.2 | 74.0 | 2/12 | 1.03 | 0.07 (32 uncached) | 46 / 49 / 43 |
| ir-space | 0.607 | 54.1 | 74.8 | 62.7 | 58.6 | 56.4 | 83.8 | 2/12 | 1.58 | 0.61 | 73 / 71 / 86 |
| decompose | 0.708 | 87.6 | 72.9 | 79.6 | 67.3 | 62.9 | 78.2 | 1/12 | 0.16 | 0.63 | 78 / 67 / 76 |
| a-contrario (best single) | 0.891 | **96.2** | 93.5 | **94.8** | 90.6 | 77.1 | 83.0 | 0/12 | **0.06** | 5.3 | 90 / **98** / **90** |
| **ensemble: route** ("each where strongest") | 0.825 | 94.8 | 85.0 | 89.7 | 80.3 | 70.3 | 81.3 | 0/12 | 0.10 | 6.6 | 85 / 82 / 90 |
| **ensemble: fuse** (shipped) | **0.905** | 92.7 | **94.4** | 93.5 | **93.4** | **82.6** | **89.1** | 0/12 | 0.13 | 6.6 | **98** / 96 / 86 |
| fuse, fitted on the tuned seed-0 calibration | 0.900 | 95.1 | 91.6 | 93.3 | 90.3 | 84.1 | 90.8 | 0/12 | 0.06 | 6.6 | 88 / 93 / 95 |
| fuse[decompose + ir-space + metamorphic] (no a-contrario) | 0.711 | 82.1 | 72.9 | 77.2 | 69.2 | 67.1 | 87.2 | 1/12 | 0.32 | 1.3 | 80 / 64 / 76 |
| fuse[a-contrario + decompose] | 0.875 | 91.0 | 94.4 | 92.7 | 93.7 | 81.5 | 88.1 | 1/12 | 0.13 | 6.0 | 93 / 98 / 90 |
| *route ORACLE (fit on evaluation)* | *0.919* | 97.1 | 94.4 | 95.7 | 93.2 | 84.4 | 88.1 | 0/12 | 0.10 | – | – |
| *fuse ORACLE (fit on evaluation)* | *0.967* | 97.2 | 99.1 | 98.1 | 98.4 | 92.6 | 94.3 | 0/12 | 0.03 | – | – |

\* Seconds per case are the sum of the approaches' cached run times on a shared machine (load 10–30).
ir-space and metamorphic reuse cached perceptions; uncached, the metamorphic adversary costs about 32 s per case.

### Per-type typed F1 (evaluation)

| type | a-contrario | decompose | ir-space | metamorphic | route | **fuse** | fuse oracle |
|---|---|---|---|---|---|---|---|
| geometry.shift | **0.83** | 0.77 | 0.67 | 0.27 | 0.83 | 0.67 | 1.00 |
| geometry.size | **0.67** | 0.50 | 0.31 | 0.20 | 0.57 | **0.67** | 0.77 |
| geometry.radius | **0.92** | 0.25 | 0.46 | 0.25 | 0.92 | 0.83 | 0.92 |
| color.fill | 0.86 | 0.12 | 0.40 | 0.00 | 0.86 | **0.92** | 0.92 |
| color.stroke | 0.67 | 0.60 | 0.44 | 0.00 | 0.67 | **0.86** | 0.86 |
| color.tint_global | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| structure.missing | 0.50 | **0.73** | 0.40 | 0.44 | 0.50 | 0.67 | 0.91 |
| structure.extra | 0.57 | 0.67 | 0.35 | 0.36 | 0.57 | **0.80** | 0.89 |
| structure.split | 0.73 | 0.50 | 0.73 | 0.00 | 0.60 | **0.91** | 1.00 |
| structure.merge | **0.75** | 0.25 | 0.73 | 0.57 | 0.25 | **0.75** | 0.89 |
| text.content | 0.86 | 0.67 | 0.71 | **0.93** | 0.63 | **0.93** | 0.93 |
| text.style | **0.89** | 0.57 | 0.11 | 0.26 | 0.57 | 0.75 | 0.89 |
| text.position | 0.75 | 0.71 | 0.62 | 0.31 | 0.77 | **0.88** | 0.93 |
| icon.glyph | 0.91 | **1.00** | 0.92 | 0.71 | 0.91 | **1.00** | 1.00 |
| icon.missing | 0.29 | **0.89** | 0.67 | 0.67 | 0.29 | 0.75 | 0.89 |
| effect.shadow | **0.94** | 0.33 | 0.22 | 0.00 | 0.94 | 0.89 | 1.00 |
| effect.opacity | 0.86 | 0.91 | 0.75 | 0.60 | **0.92** | **0.92** | 0.86 |
| layout.spacing | **0.89** | 0.86 | 0.67 | 0.40 | 0.86 | 0.67 | 1.00 |

What the numbers say:

* **Fusion beats every single approach on the headline** (0.905 vs 0.891), on AP (93.4 vs 90.6), on
  macro leaf F1 (82.6 vs 77.1, +5.5 pt) and on recall (94.4 vs 93.5). It stays at 0/12 noise false
  cases. The gain comes from *typing*, where the approaches disagree most: color.stroke, structure.extra/split,
  text.content/position and icon.missing. Leaf type accuracy rises from 83 % to 89 %.
* **It loses** 3.5 pt of detection precision (92.7 vs 96.2) and 1.3 pt of detection F1, plus some per-type F1, to a-contrario on shift,
  radius, text.style, shadow and spacing. Its recall on the held-out generator family `synth_gen`
  (86 vs 90) is lower: the fusion never saw `synth_gen`. The fusion without a-contrario (0.711) is
  barely better than decompose alone (0.708). The fused system is only as good as its best
  component plus the typing gains.
* **Literal routing ("use each approach where it is strongest") over-fits and loses** (0.825). With
  about 14 instances per type, the per-type argmax picked on fusion-calibration is often wrong on
  evaluation. Examples: icon.missing was routed to a-contrario (eval F1 0.29, where decompose has
  0.89), text.style to decompose, structure.merge to decompose. The Bayesian fusion shrinks those
  noisy per-type estimates and keeps every approach's vote.
* **The oracle gap is the overfitting a selection-on-evaluation would hide**: +0.062 for fusion and
  +0.094 for routing. Routing is the more over-fittable combiner, as expected.
* **Leave-one-family-out (fusion fitted without that family).**
  - mwc: fitting without mwc gives 0.873 on evaluation mwc, against 0.900 for the full fit and
    0.819 for the best single approach.
  - synth: fitting without synth gives 0.896, against 0.907 for the full fit and 0.912 for
    a-contrario.

  So the fusion transfers to an unseen family. On synth it does not beat a-contrario alone.
* **Seed stability.** Fits on one fresh seed (48 cases) score 0.845–0.898 on evaluation, against
  0.905 with all three seeds. More fusion-calibration data still helps.
* **Hyperparameter robustness.** A seed-fold cross-validation over 54 settings of the four
  smoothing hyperparameters, run on fusion-calibration only, spans 0.898–0.925 (default 0.916). The
  best 4-parameter setting gains +0.009 in CV. That is below the protocol's MDL price of 4 × 0.003, and no
  single-parameter move gains ≥ 0.003 consistently across folds. **The defaults were kept.** The
  evaluation split was looked at twice, both times with the same shipped model.
* **Magnitudes.** The fused magnitude comes from the voting member with the lowest
  fusion-calibration median relative error for that type. a-contrario supplies shift, size,
  spacing, split and text.position. ir-space supplies the colours and text CER. The metamorphic
  differ supplies radius and opacity.

Evidence panels: `out/adversary/synthesis/examples/*.png`. Each panel has one row per adversary (fused first), showing target |
candidate | ΔE heatmap. Ground truth is drawn in green and findings in red. The panels show the evaluation cases where
fusion gains most over a-contrario, and the case where it loses most.

## 4. Operator policy: finding type → refine operator (`policy.py`)

**Replay.** 117 perturbed fusion-calibration cases with 256 ground-truth findings. The real critic
proposed hypotheses, each was rendered, and 2,560 (finding, operator) records were kept. **123 of
256 findings (48 %) are fixed by one operator step.**

| finding type | operator that fixes it (P(fix ∣ proposed)) | fixable by one step | what the replay teaches |
|---|---|---|---|
| geometry.shift | `shift` 0.84 | 12/13 | operator works |
| geometry.size | `resize` 0.16 (proposed 11×, fixed 1×) | **1/13** | `resize` edge search is the weak operator: it proposes, but its edges are wrong |
| geometry.radius | `radius` 0.62 | 6/13 | works; `resize` competes for the same region and never helps |
| color.fill | `recolor` 0.83 | 11/14 | works |
| color.stroke | `recolor` 0.49 | 6/13 | recolor samples the fill, not the stroke band: a `restroke` operator is missing |
| color.tint_global | `recolor` 0.53 | 6/10 | per-node recolor fixes a page tint only partially: a page-level palette shift operator is missing |
| structure.missing | `missing` 0.28 | 2/16 | `perceive_region` inserts rarely reproduce the subtree |
| structure.extra | `extra` 0.86 | 15/15 | works |
| structure.split | **`shift` 0.71** (taxonomy said `extra`/`resize`) | 12/13 | the data overrules the prior: moving one part closes the seam |
| structure.merge | none (`resize` 0/6) | **0/13** | **no operator can re-introduce a gap**: a `split` hypothesis kind is missing from the critic |
| text.content | `text` 0.69 | 13/18 | works (re-OCR) |
| text.style | `text` 0.65 | 11/16 | works |
| text.position | `shift` 0.74, `text` 0.28 | 13/15 | works |
| icon.glyph | none (`shift` 2/10) | **2/16** | **no operator renames an icon**: an `icon` re-identification operator is missing |
| icon.missing | `missing` 0.28 | 2/14 | the inserted region is perceived without the icon atlas |
| effect.shadow | `shadow` 0.63 (proposed only 3×) | 3/14 | the shadow probe rarely fires: its under-strip test misses side and soft shadows |
| effect.opacity | none (`opacity` never proposed) | **0/16** | the opacity probe never fires on these cases: the blend test needs fixing |
| layout.spacing | `shift` 0.52 | 8/14 | per-node shifts fix one sibling at a time: a sibling-group (gap) operator is missing |

The shipped table is `dt/adversary/policy_model.json`. It holds the counts, the magnitude terciles and the
confusion of the runtime finder (decompose) on the same fresh seeds. `policy.suggest(findings,
doc)` returns ordered `{kind, node_id, box, finding_type, p_fix, value}` items.

**Held-out comparison** (52 EVALUATION perturbed cases; full `refine` with default parameters; the
guided critic's findings come from decompose on (target, current render, current IR) at every iteration):

| critic | total renders | final loss (mean) | loss recovered (mean) | better / worse final loss than default | faster / slower to the default's final loss | loss advantage at 5 / 10 / 20 / 40 renders (share of start loss) |
|---|---|---|---|---|---|---|
| current critic | 2,859 | 0.02009 | 80.2 % | – | – | – |
| guided (reorder only) | 2,813 (−1.6 %) | 0.01947 (−3.1 %) | 80.7 % | 2 / 3 | 12 / 5 | +0.03 % / −0.22 % / +0.61 % / +0.77 % |
| guided + safe pruning | 2,802 (−2.0 %) | 0.01945 (−3.2 %) | 81.0 % | 2 / 3 | 24 / 12 | **+1.59 %** / +0.98 % / +0.11 % / −0.81 % |

The honest reading: the policy gets to a given loss faster on twice as many cases as it slows down,
and its final loss is slightly better. Total renders barely move, for two reasons.

* The optimizer tries the whole critic list every iteration and stops on patience, so ordering mainly
  changes batch composition.
* The stall fallback (71 of 213 guided calls) restores the full list exactly when the run is ending.

**The large lever is in the table above, not in ordering:**

* five finding types have no working operator: merge, icon.glyph, opacity, size, and missing/icon.missing;
* two operators fire too rarely: shadow and opacity.

The replay found these gaps in 117 cases without a human reading a single render.

## 5. Novel types from real pages (`novelty.py`)

**Known distribution.** 2,981 items on fusion-calibration (2,278 noise, 703 labelled with 18 types).

**Sanity check on EVALUATION** (all known types): 1.3 % of items have p < α = 0.02, so the test is
calibrated. 34 candidates fall in one large "page-scale residual" cluster (JPEG / tint-wide residue)
plus one single-page cluster.

**Real pages:** 8 pages of `out/real_r3m`, 412 items, 75 candidates.

| cluster | items | pages | proposed type | nearest known type | signature (centroid) | exemplar | draft family |
|---|---|---|---|---|---|---|---|
| C0 | 29 | **6** | `icon.unrendered_glyph`: multi-colour brand glyphs (Google "G", product logos, coloured bullets) painted in the target, absent in the candidate | icon.missing | ~180 px², target-only ink (balance 1.0), on icon nodes (0.79), strongly chromatic (Δchroma z ≫ known) | `exemplars/C0_0_gws_calendar.png` | `adv_icon_unrendered_glyph` |
| C1 | 8 | 4 | `text.glyph_shape`: text whose glyphs differ in both shape *and* colour. Coloured link text is rendered in default ink in a substitute font, and no shift or affine colour transfer explains it | icon.missing (the known set has no such text item) | text 0.92, balanced ink, ΔE 26, shift gain 0.06 | `exemplars/C1_0_mw_button_types_mobile.png` | `adv_text_glyph_shape` |
| C2 | 4 | 4 | chromatic partial art: both sides textured, high ΔE, nothing explains it | structure.missing | ~270 px², Δchroma high, colour gain 0 | `exemplars/C2_0_gws_keep.png` | `adv_residual_hi_d_chroma_hi_de_mean` (finite real-crop family until a generator exists) |
| C3 | 4 | **1** | text glyph shape on mw_chip_types only | color.fill | | | **not promoted (one-off)** |

Each stub (`out/adversary/synthesis/novelty/<name>.family.json` and `.py.txt`) has the fields of
`dt.scenarios.spec.ScenarioFamily`:

* name, description, stage, source, refine_iters;
* `failure_refs` mined from `knowledge/failures.jsonl`. C0 links #8 (icons missed), #45 (sparkle
  bullets render as nothing) and #64 (foreign glyphs misnamed);
* a `param_space` widened beyond the observed ranges;
* criteria on the validator's measures (for example `jnd_frac_nontext <= param:validate.gate.ident.jnd_frac` in the ROI);
* tune prefixes, a generator sketch, and a Python skeleton that raises until the generator is written.

`novelty.to_family(stub)` instantiates a real `ScenarioFamily` when the scenario harness is installed.

What novelty teaches: the synthetic taxonomy has **no chromatic content**. Every real-page novelty
is first of all a chroma outlier: brand logos, coloured link text, colourful art. The next benchmark
round should add multi-colour glyphs and coloured text runs to the perturbation grammar, and the
taxonomy should get a minor version (1.1) with `icon.unrendered_glyph` and `text.glyph_shape` as
candidate leaves *after* their scenario families exist and pass the protocol.

## 6. The anti-overfitting protocol (`protocol.py`)

`protocol.check(change_fn)` runs these checks and returns `{accepted, reasons, checks, ledger}`.
`change_fn` can be a `{param: value}` dict, a `Change`, or `train_cases -> context manager`. The
protocol decides which cases the change may be fitted on.

| check | rule (registered `adversary.protocol.*` params) | what it prevents |
|---|---|---|
| bounded priors | every key registered; inside its registered range; step ≤ 25 % of the range; ≤ 5 keys; never `validate.*`, `bench.*`, `adversary.bench.*` (the yardstick) | moving into unexplored corners; tuning the judge |
| calibration gain | the headline gain on CALIBRATION must be ≥ 0.003 × number of changed params (MDL price), and the noise false-case rate must not rise | many-knob over-fits; "wins" bought with false alarms |
| leave-one-family-out | the change is re-fitted without each family, and no held-out family may lose > 0.01 | family-specific over-fits |
| fresh seeds | scored on the 144 fusion-calibration cases (unseen seeds), it may lose ≤ 0.005 | seed over-fits (the 48-case split is small) |
| metamorphic regulariser | violations of identity / nuisance / translation / swap on 12 calibration cases may grow by ≤ max(1, 5 %) | wins that break invariances the benchmark does not test |
| EVALUATION | report only, after the decision; each look is counted in `eval_looks.json` and a warning fires past 5 looks | silent selection on the test set |
| real pages | never passed to `change_fn`; finding counts reported, and a flood (≥ 3× and ≥ +20) is flagged | degenerate behaviour on the distribution that matters |
| ledger | the verdict carries a `knowledge/ledger.jsonl`-schema entry (via `dt.learn.ledger.make_entry` when installed); `protocol.record` appends it to `out/adversary/synthesis/ledger_draft.jsonl` | unprovenanced learned state |

**Demonstration.** A naive tuner (`protocol.naive_proposals`) does what an unprotected loop does. It
tries ±20 % of the range on 15 decompose parameters against the seed-0 calibration split and proposes
every winner. Two of the 30 steps won. The protocol judged them, plus their merge (the greedy "ship
every winner" step):

| proposed change | priors | calibration gain (required) | leave-one-family-out (mwc / synth) | fresh seeds (144) | metamorphic violations (identity / nuisance / translate / swap) | verdict | EVALUATION, report only |
|---|---|---|---|---|---|---|---|
| `tau_min` 3.0 → 1.2 | ok (step 0.20) | +0.0049 (0.003) | **−0.0116** / +0.024 | +0.023 | 30 → 26 (0/21/5/4 → 0/20/3/3) | **rejected**: mwc lost 0.0116 > 0.01 | +0.021 |
| `explain_min` 0.6 → 0.73 | ok | **+0.0001** (0.003) | −0.0001 / 0.000 | −0.001 | 30 → 30 | **rejected**: gain below the price | 0.000 |
| both (greedy merge) | ok | **+0.0010** (0.006) | **−0.0117** / +0.018 | +0.023 | 30 → 26 | **rejected**: both reasons | +0.021 |

Real-page finding counts (report only): `tau_min` 1.2 reduces findings on gws_calendar (17 → 11) and
gws_gmail (9 → 4), and no page floods.

What the demonstration shows, honestly:

* **The merge is the classic over-fit.** Two individually "winning" steps interact. The merged
  change gains a tenth of what one step alone gained, and it still carries that step's family
  regression.
* **The rejection of `tau_min` is conservative and the protocol paid for it.** Evaluation (looked at
  only after the decision) would have gained +0.021, and fresh seeds also gain. The change is
  *family-dependent*: it helps synthetic pages and hurts Material Web pages. The loop's correct
  response is not to override the protocol. Instead it should open a scenario family for
  low-contrast mwc surfaces, the regime where a lower noise floor hurts, or make the floor
  depend on a measurable page property. Then it re-proposes.
* **The adversary's own invariances are a finding.** Decompose already has 21 nuisance violations on
  12 cases: a JPEG q92 or 0.25 px nuisance on the target changes its findings. That is a concrete
  robustness target ("graded noise gate", decompose's own known gap). The relation counts it without labels.
* **Ledger entries.** The three verdicts' entries (`params-f3eafe9141`, `params-83c3b7cbe7`,
  `params-42bf5a9557`, all `accepted: false`) are in `out/adversary/synthesis/ledger_draft.jsonl` with
  their exact diffs and every check's before/after.

`tests/test_adversary_synthesis.py` also checks the protocol on a toy adversary. A threshold change
that generalises is accepted. The same change plus an always-on finding is rejected by the identity
relation. `change_fn` only ever receives calibration cases, one family held out at a time.

## 7. What each finding type teaches the loop

| type | best detector (eval F1) | operator (P(fix)) | real pages | loop action it triggers |
|---|---|---|---|---|
| geometry.shift | a-contrario 0.83 | shift 0.84 | rare | nothing: detected and fixed |
| geometry.size | a-contrario / fuse 0.67 | resize 0.16 | via text boxes | **fix the `resize` edge search** (1/13 fixable); scenario: thin strokes and 1–3 px edge moves under JPEG |
| geometry.radius | a-contrario 0.92 | radius 0.62 | 1 (gmail) | decompose's corner probe is gated off on noisy captures → graded noise gate |
| color.fill | fuse 0.92 | recolor 0.83 | chip fills | nothing |
| color.stroke | fuse 0.86 | recolor 0.49 | – | **add a stroke-band recolor operator** |
| color.tint_global | all 1.00 | recolor 0.53 | none detected (but see failures #44) | **add a page palette-shift operator**; the `page_tint` family exists in the scenario harness |
| structure.missing | decompose 0.73 | missing 0.28 | illustrations, button groups | inserted regions need the icon/font passes, not plain `perceive_region` |
| structure.extra | fuse 0.80 | extra 0.86 | flattened hero blocks | nothing |
| structure.split | fuse 0.91 | **shift 0.71** | – | update the taxonomy's `refine_kinds` (data beat the prior) |
| structure.merge | fuse / a-contrario 0.75 | **none** | 1 (calendar) | **add a `split` hypothesis to the critic** |
| text.content | fuse / metamorphic 0.93 | text 0.69 | 1–3 per page (tracks validator CER) | OCR voting at sub-image offsets (metamorphic finding) |
| text.style | a-contrario 0.89 | text 0.65 | 1 per page | the font pass. Novelty C1 shows real text also loses its *colour* |
| text.position | fuse 0.88 | shift 0.74 | – | nothing |
| icon.glyph | decompose / fuse 1.00 | **none** | – | **add an icon re-identification operator** (atlas match on the target crop) |
| icon.missing | decompose 0.89 | missing 0.28 | **dominant on real pages** (1–7 per page) | novelty C0: chromatic brand glyphs → `adv_icon_unrendered_glyph` family; logo raster path |
| effect.shadow | a-contrario 0.94 | shadow 0.63 (rarely proposed) | – | widen the shadow probe (sides, soft halos) |
| effect.opacity | fuse 0.92 | **never proposed** | – | **fix the opacity blend test** in the critic |
| layout.spacing | a-contrario 0.89 | shift 0.52 | – | **add a sibling-gap operator** (move the tail of a row/column together) |
| noise.* | a-contrario 0/12, fuse 0/12 | – | font substitution on every real page | stays an abstention on synthetic data; on real pages it is the novelty `text.glyph_shape` |

## 8. The recursive model

```
              ┌──────────────────────────────────────────────────────────────────────────┐
              │                                                                          │
   target + render + IR                                                                  │
              │                                                                          │
              ▼                                                                          │
  [findings]  ensemble.fuse: four adversaries → typed, localised, calibrated findings    │
              │        (noise abstentions kept; magnitudes from the best member)         │
              ▼                                                                          │
  [types]     taxonomy v1.x  ◀── novelty: unexplained clusters on ≥2 pages → candidate   │
              │                    leaves (minor version bump only after their family    │
              │                    passes the protocol)                                  │
              ▼                                                                          │
  [operators] policy: P(fix | type, magnitude, kind) learned by replay → guided critic;  │
              │        types with no working operator = critic work items                │
              ▼                                                                          │
  [families]  scenario families: novelty stubs + metamorphic relations (meta.*) +        │
              │        fuzz envelopes + failure-replay families (dt/scenarios)           │
              ▼                                                                          │
  [tuning]    proposals (dt train / tune / an AI driver's code edit) ──▶ protocol.check  │
              │        bounded priors · calibration gain · LOFO · fresh seeds ·          │
              │        metamorphic · eval report-only · ledger entry                     │
              ▼                                                                          │
  [state]     dt/params.json + knowledge/ (global) or $DT_HOME (local consumer state)    │
              │                                                                          │
              └──────────── re-run the adversaries on the next real pages ───────────────┘
                                       (= new findings)
```

User-informed tuning enters the loop at two points.

* **A consumer's own screenshots become real-page holdouts and novelty inputs.** The ensemble and
  novelty need no labels, and the metamorphic relations need no ground truth.
* **A consumer's decision answers become labelled findings.**

Local findings and families live under `$DT_HOME`. A local change is promoted to global only through
`protocol.check` on the *global* calibration set: a consumer's use case may propose a change, but it
must generalise to be shipped.

The loop has three fixed points that keep it from over-fitting itself:

1. **The yardstick is never tuned.** `validate.*`, `bench.*` and `adversary.bench.*` are rejected by the protocol.
2. **Evaluation is look-counted and report-only.**
3. **A new type needs multi-page support, and its family must pass the same gate** as any parameter change.

## 9. Integration steps

### dt/validate
1. Add `typed_findings` to `validation.json`. Use `ensemble.find` when the renderer is available,
   otherwise fuse whichever approaches can run (`fuse[decompose+ir-space+metamorphic]`, renderer-free
   apart from perception). The independent gates stay unchanged: findings are diagnostics, never a gate.
2. Render the fused findings in `validation.md` and `worst_regions.png`, coloured by parent.
   Novelty candidates (p < α) go to `decisions.json` as `{"kind": "novel_discrepancy", "box",
   "features", "nearest_type"}`, so a consumer's answer labels them.
3. Report a-contrario's registered nuisance ("smoothing off, +0.33 px") as a capture diagnostic. It
   is a pipeline-level fix on 5 of 8 real pages.

### dt/refine
1. Add a critic hook to `optimizer.refine` (`critique_fn: Optional[Callable] = None`, a 3-line change) instead
   of the monkey-patch in `policy.guided`. Ship `policy.guided` as the default only after its renders/loss
   comparison passes the protocol on a fresh evaluation seed.
2. Feed `refine` histories back into the replay tables online. Every accepted or rejected hypothesis
   next to a fused finding is a (type, kind, fixed) record, so the policy keeps learning from production runs.
3. Close the operator gaps in this order (each found by the replay):
   - `split` (structure.merge, 0/13);
   - icon re-identification (icon.glyph, 2/16);
   - the opacity blend test (0/16);
   - `resize` edges (1/13);
   - stroke-band recolor;
   - sibling-gap shift (layout.spacing);
   - page palette shift (tint).

   Each gap becomes a failing-first test from its replay records.

### dt/scenarios
1. `dt scenario add --from out/adversary/synthesis/novelty/<name>.family.json` registers a novelty stub as a
   draft family once its generator is written. The skeleton is `<name>.py.txt` and uses `dt.scenarios.spec`
   unchanged. Promote it to the active suite only through `protocol.check`.
2. Register the metamorphic relations (`meta.translate`, `meta.recolor_*`, `meta.dpr2`, `meta.jpeg`,
   ...; 158 specs in `out/adversary/metamorphic/scenarios.jsonl`) as label-free families. Their
   violation count is the regulariser in `dt train`.
3. Make `protocol.check` the acceptance gate of `dt train --family`:
   - its leave-one-family-out runs over the suite's families;
   - its fresh-seed check draws new family seeds;
   - its ledger entry is appended with `dt.learn.ledger.append` (scope `local` for consumer state,
     `global` after promotion).

## 10. Reproduce

```bash
python -m dt.adversary.ensemble --workers 3        # predictions (cached), fusion fit, evaluation table, examples
python -m dt.adversary.policy both --workers 4     # replay on fresh seeds -> policy_model.json; held-out refine comparison
python -m dt.adversary.novelty                     # known distribution, real-page clusters, scenario stubs
python -m dt.adversary.protocol --adversary decompose --naive 3 --keys ... --evaluation --real --record
python -m pytest -q tests/test_adversary_synthesis.py
```

Outputs: `out/adversary/synthesis/{ensemble.json, fusion_cv.json, examples/, policy/, novelty/, protocol/,
ledger_draft.jsonl}`. Shipped learned state: `dt/adversary/ensemble_model.json`, `dt/adversary/policy_model.json`.

## 11. Known limits

* The evaluation split has 4–8 instances per type, so per-type F1 differences under about 0.15 are within noise.
* Fusion inherits a-contrario's cost (renders, about 3–5 s per case).
* Without a-contrario the fusion adds little over decompose.
* The guided critic's measured render savings are small (−2 %). The ordering lever is weak against
  the optimizer's whole-list iterations.
* Novelty features are hand-designed (16). Naming is rule-based and descriptive only. A cluster's
  name is a hypothesis for a human or AI driver, and its family must still be written and pass the protocol.
* The real-page set is 8 pages, so the multi-page rule (≥ 2 pages) is a weak bar. The next real pages are the real test.
