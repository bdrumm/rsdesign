# Self-improvement: how the harness gets better without a human in the loop

The engine is deterministic; what "learns" is (1) its parameters, (2) its knowledge
(component signatures, priors, failure taxonomy), (3) its test corpus, and (4) its code — the last
via an AI driver that follows the protocol below. Every mechanism is measurable against ground
truth, and every change passes a regression gate before it is kept.

## Signals we can optimise against
| Signal | Source | Ground truth? |
|---|---|---|
| Structural metrics (node IoU/recall, colour ΔE, text CER, radius MAE, component+variant acc, token acc, layout consistency) | `fixtures/corpus/mwc` (real Material Web pages, DOM-derived IR), `fixtures/corpus/synth` (pure IR) | yes |
| Fidelity (JND fraction, tile chamfer, edges-within-1px, region ΔE, OCR text Δpos) | any screenshot, incl. `fixtures/screens` (public M3 docs + Google pages) | no (self-supervised: target vs our render) |
| Re-perception consistency | any screenshot | no (perceive(render(doc)) ≈ doc) |
| Decision-queue outcomes | the user's/AI's answers to `decisions.json` | human/AI labels |

## Mechanisms (in order of leverage)
1. **Parameter tuning** (`dt tune`) — random search + coordinate descent over every registered
   threshold (`dt.params.P.ranges()`), train/holdout split over corpora, writes `dt/params.json`
   only when holdout improves. Cheap, safe, repeatable; run nightly.
2. **Failure-driven repair loop** (`dt improve`) — bench → worst-10 cases by (metric, stage) →
   a structured markdown brief (symptom, suspected stage, diff crops, GT vs pred tree) → an AI
   driver edits the stage → bench → keep if composite ↑ and no metric regresses > ε → append to
   `knowledge/failures.jsonl` → **auto-generate a regression case** reproducing the failure
   (synth grammar rule or MWC page) so the fix can never silently regress. Loop until two rounds
   produce nothing new ("dry").
3. **Curriculum expansion** — when the corpus is "solved" (composite > target), the grammars
   add harder content: dark theme, dense desktop (1280–1920 px), overlapping/transparent layers,
   images, gradients, 2× DPR captures, tiny text, long lists, dialogs over scrims.
4. **Learned component signatures** — `ingest_screenshots` clusters recurring subtrees in
   reference screens into new `ComponentSpec`s; accepted ones ship in
   `fixtures/design_systems/learned.json`. Decision-queue answers that repeat (same signature →
   same component) are promoted to rules automatically.
5. **Font identification** — biggest residual on real screenshots is glyph mismatch. Render each
   OCR'd line in candidate fonts (Roboto, Google Sans, Inter, SF/‑apple‑system, Helvetica, Arial)
   at the estimated size/weight and pick the family minimising ΔE; store as `TextStyle.family`.
6. **DPR / scale auto-detection** — try dpr ∈ {1, 1.5, 2, 3}: downscale, perceive, render, pick
   the dpr minimising fidelity loss. Also detect letterboxing / device frames and crop.
7. **Unsupervised tuning on real screens** — objective = post-refine fidelity on `fixtures/screens`
   (no GT needed) with the GT corpora as a guard against degenerate solutions.
8. **Stage attribution** — on GT corpora, run perceive→map→export with GT substituted at each
   stage boundary; the drop in composite when a stage is swapped for the real thing is that
   stage's error budget. Invest where the budget is largest.
9. **Refine-move statistics** — log which hypothesis kinds get accepted and their gains; prune
   useless moves, reorder by empirical gain (a learned move policy, no ML needed).

## Protocol for an AI driver (summarised in AGENTS.md)
```
dt bench --json                      # baseline
dt improve --top 10 > brief.md       # worst cases with evidence
<edit one stage; keep the change local>
dt bench --json                      # must improve composite, no metric regresses > 0.5 pt
python -m pytest -q                  # must pass
dt regress-case add <case-id>        # freeze the failure as a test
git commit                           # learned state + code
```
Rules: one stage per iteration; never tune on the holdout; never edit `dt/ir.py` semantics; never
delete a regression case; record every accepted change in `knowledge/failures.jsonl`.
