# AGENTS.md — operating manual for an AI driver

You (an AI agent in the user's environment) drive this harness. The engine is deterministic and
does the pixel work; you make the few judgment calls it surfaces and you run the improvement loop.
Never try to "eyeball" pixels yourself — every question you might ask of an image has a command.

## 0. Setup (once)
```bash
bash scripts/setup.sh            # idempotent; macOS + Linux; ends with `dt doctor`
source .venv/bin/activate
dt doctor --json                 # {"ok": bool, "checks": [{name, status, detail, fix}]}; exit 1 if not ready
```
If a check is `fail`, run its `fix` line and re-run `dt doctor`. Do not continue with a failing
`browser`, `font render` or `ocr` check: every downstream number would be wrong.
* OCR: macOS → Vision (built in). Linux/Windows → the `linux` extra (RapidOCR); `DT_OCR` forces a backend.
* Browser: system Chrome, else Playwright's Chromium (`python -m playwright install chromium`);
  `DT_BROWSER` forces one. Bench numbers are only comparable on the same browser + OCR backend as the
  baseline (Chrome + Vision); `dt doctor` warns otherwise.
* `out/` is git-ignored: `python scripts/make_reference_page.py` rebuilds the reference page the
  tests use; `npm run build:mwc` rebuilds `fixtures/mwc/mwc.bundle.js` byte-for-byte.

## 1. Translate a screenshot
```bash
dt translate shot.png --ds material3 -o out/run1 --json
```
Outputs in `out/run1/`: `ir.json` (perceived), `ir.mapped.json` (components + tokens),
`render.png`, `diff.png`, `figma_plan.json`, `decisions.json`, `metrics.json`, `report.md`,
`validation/` (gates, heatmap, blink comparator, worst regions).

Read `metrics.json` first: `loss_before`, `loss_after`, `validation.gates`. Then `decisions.json`.

## 2. Resolve decisions (the only place your judgment is needed)
`decisions.json` lists items the engine could not settle, each with evidence and candidates:
```json
{"id":"d3","kind":"component","node_id":"n41","candidates":[{"name":"Card","variant":{"style":"elevated"},"conf":0.52}, ...],
 "evidence":{"box":[...],"fill":"#fff","radius":12,"shadow":true,"children":["text","text","frame"]}}
```
Answer with `dt apply-decisions out/run1 --answers answers.json` where answers map id → choice
(`{"d3": {"name":"Card","variant":{"style":"elevated"}}}` or `{"d7": {"text":"Inbox"}}`).
Then re-run `dt translate … --resume out/run1` (or `dt refine` + `dt export`) to regenerate.
Unanswered decisions never block output; the engine keeps its best guess.

## 3. Get it into Figma
Figma → Plugins → Development → Import plugin from manifest → `dt/export/figma_plugin/manifest.json`
→ run → paste `figma_plan.json`. Instances resolve to the Material 3 Design Kit when that library is
enabled in the file; otherwise the plugin builds the fallback frames (same geometry, same tokens).

## 4. Judge the result (never by looking)
```bash
dt validate shot.png out/run1/render.png --ir out/run1/ir.mapped.json -o out/run1/validation
```
Gates: `pixel-exact` > `visually-identical` > `structurally-faithful` (docs/VALIDATION.md).
If a gate fails, the report names the measure and the worst tile/region/text line. Typical fixes:
* tile chamfer high in one tile → a node is displaced/missized → `dt refine --focus <box>`
* region ΔE high → wrong fill/stroke → check `tokens` on that node; maybe a shadow is missing
* text CER > 0 → OCR error or font transform → fix `text` in ir.json, or add a decision answer
* text pixels bad but geometry fine → font family mismatch (known; see self-improvement #5)

## 5. Improve the harness (the RSI loop)
```bash
dt bench --json                 # baseline on fixtures/corpus (GT) + fixtures/screens (fidelity)
dt improve --top 10             # markdown brief: worst cases, suspected stage, evidence crops
# edit ONE stage (dt/perceive | dt/mapping | dt/refine | dt/export); keep it local
python -m pytest -q && dt bench --json     # composite must improve; no metric may regress > 0.5pt
dt tune --iters 50              # optional: parameter search; writes dt/params.json only if holdout improves
git commit -am "improve: <stage>: <what>"   # include knowledge/failures.jsonl entry
```
Rules: one stage per iteration · never edit `dt/ir.py` semantics · never tune on the holdout ·
never delete a regression case · every accepted change gets a `knowledge/failures.jsonl` entry.

## 6. Add a design system
* Figma: `dt ds from-figma <file_key>` (needs `FIGMA_TOKEN`) → `fixtures/design_systems/<name>.json`
* Reference screenshots: `dt ds from-screens dir/` → learned components with exemplars
* Tokens (W3C DTCG / Material Theme Builder): `dt ds from-tokens tokens.json`
Then `dt translate … --ds <name>`.
