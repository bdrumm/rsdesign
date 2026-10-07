# rsdesign

**Screenshot → structured IR → design-system mapping → Figma, with a render-verified refinement
loop — fully programmatic, measurable, and self-tuning.**

Give it a screenshot of a UI. It OCRs the text, segments the shapes, infers the hierarchy and
auto-layout, matches what it finds against a design system (Material 3 ships in the box), renders
its own understanding back to pixels, measures the difference against the original, and iterates
until the render matches. The result is a JSON build plan that a bundled Figma plugin turns into
real frames, text and component instances — plus a list of the few judgment calls the engine could
not settle itself.

No stage ever asks a language model to "look at" an image. Everything is OpenCV, OCR and
algorithms, so every number is reproducible, every threshold is tunable, and the whole system can be
benchmarked against ground truth and improved in a closed loop. An AI agent (or a human) *drives*
the harness; the harness does the pixel work. See [AGENTS.md](AGENTS.md) for the operating manual.

```
 screenshot.png
     │  perceive  (OCR via macOS Vision / RapidOCR / Tesseract, segmentation via OpenCV,
     │             hierarchy + auto-layout inference)
     ▼
 Document (IR)  ──── mapping ────▶ Document (IR + ComponentRef + design tokens)
     │                                   │
     │ render (IR→HTML→PNG, Playwright)  │ export (IR→Figma build plan JSON → plugin creates nodes)
     ▼                                   ▼
 rendered.png ◀──── compare ────▶ loss + per-node error map + residual regions
     ▲                                   │
     └──── refine (critic proposes shift/resize/recolor/text/radius/missing/extra moves;
                   optimizer keeps a move only if the render gets closer; loop) ◀──┘

 selftest: ground-truth corpora (Material Web pages with DOM-derived IR; pure-IR synth screens)
           → bench (node IoU/recall, colour ΔE, text CER, radius MAE, component accuracy, SSIM …)
           → tune (search over every registered threshold → dt/params.json)
           → improve (worst cases → a brief an AI can act on)
```

## Status (measured, not eyeballed)

Numbers from the shipped benchmark (`dt bench`, 24 ground-truth screens: 12 real Material Web pages
with IR extracted from the DOM, 12 synthetic Material 3 screens) and the regression baseline in
`knowledge/baseline.json`:

| measure | value |
|---|---|
| composite (metrics v2, 1 = perfect) | **0.893** |
| node recall / precision | 0.84 / 0.83 |
| component + variant accuracy | 0.67 (0.88 when mapping runs on ground-truth geometry) |
| mean pixel ΔE2000 (render vs screenshot) | 0.51 |
| non-text pixels past the visibility threshold (ΔE > 2.3) | 2.2 % |
| target edges within 1 px of a rendered edge | 94 % |
| text recall / character error rate | 0.98 / 0.003 |

On real pages with no ground truth (Google product pages, the Material 3 docs, material-web.dev),
mobile layouts land at 76–93 % of edges within 1 px; desktop pages are the current frontier and
no real page passes the strict `pixel-exact` / `visually-identical` gates yet
([docs/VALIDATION.md](docs/VALIDATION.md) defines them).

What is render-verified today (the renderer, not a model, picks the answer):
* **icons**: all 4,299 Material Symbols, outlined and filled, with pose search (59/60 common icons
  re-render identically; glyphs from other icon sets stay unnamed and become decisions);
* **fonts**: Roboto, Google Sans (and Google Sans Text when installed locally), size, weight,
  tracking and italic;
* **unexplained art**: guarded as-is crops (never over text, never inside a component, within a
  screen-area budget, reported as an editability measure);
* **device pixel ratio**: `--dpr auto` detects 1x/2x/3x captures.

## Use it from an AI agent (MCP)

```bash
uv pip install -e ".[mcp]"
python -m dt.mcp_server          # or: rsdesign-mcp   (stdio transport)
```

Tools: `translate`, `perceive`, `render`, `validate`, `export_figma`, `list_decisions`,
`apply_decisions`, `bench` (with the regression gate), `improve_brief`, `attribution`, `move_stats`.
Example client config (Claude Desktop / Claude Code / Cursor):

```json
{"mcpServers": {"rsdesign": {"command": "/path/to/rsdesign/.venv/bin/python", "args": ["-m", "dt.mcp_server"]}}}
```

## Install

Requirements: Python ≥ 3.12, Node ≥ 18, and Google Chrome (or Playwright's bundled Chromium).
macOS and Linux are supported.

```bash
git clone <this repo> designtranslator && cd designtranslator
bash scripts/setup.sh        # idempotent: .venv, Python deps (+ OCR extra off macOS), npm ci,
                             # Material Web bundle, browser, test reference page, then `dt doctor`
source .venv/bin/activate
dt doctor                    # re-check any time: prints [ ok ] / [warn] / [FAIL] + a fix per item
```

`dt doctor` (also `python -m dt.cli doctor`, `--json` for machines; exit 1 if something core is
broken) runs the real thing for each item: launches the browser and renders, checks that the
self-hosted Roboto actually loads, OCRs a rendered line with the selected backend, loads the icon
atlas, and checks the Material Web bundle and the reference page used by the tests.

The same steps by hand:

```bash
uv venv .venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -e .                             # macOS (Vision OCR via pyobjc)
uv pip install -e '.[linux]'                    # Linux/Windows: adds RapidOCR (alias: '.[ocr]')
uv pip install --reinstall opencv-python-headless   # Linux: RapidOCR pulls the GUI OpenCV, which needs libGL
npm ci                                          # playwright, @material/web, esbuild
npm run build:mwc                               # only if fixtures/mwc/mwc.bundle.js is missing (reproducible)
python scripts/make_reference_page.py           # out/mwc_test.{html,png}: out/ is git-ignored
dt doctor
```

Browser: system Chrome is used (`channel="chrome"`); when it is missing the renderer falls back to
Playwright's bundled Chromium (`python -m playwright install chromium`; on Linux also
`sudo python -m playwright install-deps chromium`). `DT_BROWSER=chrome|chromium|msedge|/path/to/exe`
forces one. Renders on Chromium can differ in text antialiasing from the Chrome-made baselines in
`knowledge/baseline.json`; `dt doctor` warns about it. Fonts (Roboto, Google Sans, Material
Symbols) are self-hosted under `fixtures/fonts`, so renders are offline.

OCR backend: macOS uses the built-in Vision framework (pyobjc, installed only on macOS). Elsewhere
the `linux`/`ocr` extra installs RapidOCR (ONNX, pure pip); Tesseract (`pytesseract`) also works.
The backend is picked automatically (`vision` → `rapidocr` → `tesseract`) or forced with
`DT_OCR=<name>`. RapidOCR's recogniser drops word spaces in Latin text ("GoogleWorkspace"); the
adapter restores them from the measured ink gaps (bench text CER 0.021 -> 0.005). On the
ground-truth bench RapidOCR now scores composite 0.874 against Vision's 0.889 (text CER 0.005 vs
0.003); the shipped baselines were made with Vision.

Optional, git-ignored inputs: `fixtures/screens/*.png` (public reference pages,
`dt corpus build --kind screens`, needs network). Tests that need them skip when they were never
captured.

## Quickstart

```bash
dt translate shot.png --ds material3 -o out/run1
```

```
out_dir               out/run1
design_system         material3
loss before -> after  0.0405 -> 0.0336
refine                8 accepted, 5 iters, stop=patience
components            9 {'Button': 3, 'Chip': 2, 'Card': 1, 'Checkbox': 1, 'Switch': 1, 'TextField': 1}
decisions             4 {'icon': 4}
gates                 {'pixel-exact': False, 'visually-identical': False, 'structurally-faithful': False}
report: out/run1/report.md
```

Add `--json` to get the same summary as one JSON object on stdout (progress goes to stderr).
Useful flags: `--refine-iters N` (0 disables refinement), `--time-budget S` (cap the refine loop),
`--dpr 2` for @2x captures, `--ds none` to skip mapping, `--resume out/run1` to reuse a previous
run's IR (after answering decisions).

### What `dt translate` writes

| file | what it is |
|---|---|
| `ir.json` | the perceived IR: absolute boxes, fills, strokes, radii, shadows, text with style, icons, inferred auto-layout |
| `ir.mapped.json` | the final IR after refinement and mapping: component refs (`name`, `variant`, `props`, `confidence`, evidence) and design tokens on every node |
| `render.png` / `render.before.png` | our render of the final / initial IR (what the output *looks like*) |
| `diff.png` / `diff.before.png` | target \| render \| ΔE heatmap with the residual error regions boxed |
| `figma_plan.json` | the Figma build plan (schema in `dt/export/BUILD_PLAN.md`) |
| `decisions.json` | open judgment calls: unmatched components with top-3 candidates + evidence, low-confidence OCR lines, unidentified icon glyphs |
| `metrics.json` | `loss_before`, `loss_after`, pixel metrics, refine stats (iterations, accepted moves by kind, renders, stop reason), components found, decisions count, validation gates, per-stage timings, `errors` |
| `report.md` | all of the above as a readable page with the IR tree |
| `refine_history.json` | every hypothesis the critic tried, with before/after loss and accepted flag |
| `validation/` | independent fidelity gates ([docs/VALIDATION.md](docs/VALIDATION.md)): `validation.json`/`.md`, `heatmap.png`, `side_by_side.png`, `worst_regions.png`, `blink.html` |

Every stage is wrapped: if one fails (no OCR backend, a design system that cannot be loaded, a
renderer crash) the failure is recorded in `metrics.json["errors"]`, the pipeline continues with the
best document it has, and still writes the whole output set. Exit code `3` signals such a partial
run; `1` means no document could be produced at all.

### Resolving decisions

`decisions.json` is the only place human/AI judgment enters:

```json
{"id": "d3", "kind": "component", "node_id": "r_41", "question": "Which component (if any) is this frame?",
 "candidates": [{"name": "Card", "variant": {"style": "elevated"}, "score": 0.52}, ...],
 "evidence": {"box": {...}, "fill": "#ffffff", "radius": 12, "shadow": true, "children": ["text", "text"]},
 "answer": null}
```

Answer with a JSON object mapping ids to choices and merge it back:

```bash
dt apply-decisions out/run1 --answers answers.json
# answers.json: {"d3": {"name": "Card", "variant": {"style": "elevated"}}, "d7": {"text": "Inbox"}, "d9": {"icon_name": "search"}}
dt translate shot.png --ds material3 -o out/run2 --resume out/run1     # optional: refine/export again
```

Unanswered decisions never block anything; the engine always ships its best guess.

## Getting the result into Figma

1. In Figma: **Plugins → Development → Import plugin from manifest…** and pick
   `dt/export/figma_plugin/manifest.json`.
2. Run the plugin, paste (or upload) `figma_plan.json`, click **Build**.
3. The plugin loads the fonts, builds frames / rectangles / ellipses / text / Material Symbols icon
   text nodes, applies auto-layout where the IR inferred it, and for every mapped component calls
   `importComponentByKeyAsync`. When the component key is not available in your file, it builds the
   fallback frame instead (same geometry, same tokens) and lists it in the summary. Design tokens are
   bound to local variables by name when they exist.

The plugin can be run headlessly for testing: `node dt/export/figma_plugin/run_fake.js --plan figma_plan.json`.

## Stage-by-stage CLI (the agent protocol)

Every stage is a subcommand with file in / file out, `--json` output and stable exit codes
(0 ok · 1 failure · 2 usage · 3 partial). This is the protocol an AI driver uses:

```bash
dt perceive shot.png -o ir.json                       # screenshot -> IR
dt map ir.json --ds material3 -o ir.mapped.json       # + components & tokens
dt render ir.mapped.json -o render.png [--mode flex]  # IR -> pixels (flex mode verifies the layout)
dt compare ir.mapped.json shot.png --diff diff.png [--gt gt.json]   # loss, per-node errors, metrics
dt refine ir.mapped.json shot.png -o ir.refined.json --iters 10     # adversarial loop
dt export ir.refined.json -o figma_plan.json          # Figma build plan (alias: export-figma)
dt validate shot.png render.png --ir ir.mapped.json -o out/validation   # independent gates
dt ds from-tokens tokens.json | from-figma <file_key> | from-screens dir/ | show material3
dt params [--prefix perceive.]                        # every tunable, with value / range / doc
```

Design-system specs for `--ds`: `material3` (or any `fixtures/design_systems/<name>.json`),
`figma:<file_key>` (needs `FIGMA_TOKEN`), `screens:<dir>` (learn recurring components from
reference screenshots), `tokens:<file.json>` (W3C DTCG or Material Theme Builder export re-theming
the Material 3 catalog), a path to a design-system JSON, or `none`.

The Python API mirrors the CLI: `dt.pipeline.translate(...)`, `dt.perceive.perceive`,
`dt.mapping.map_document`, `dt.render.screenshot.render_doc`, `dt.compare.evaluate`,
`dt.refine.optimizer.refine`, `dt.export.to_build_plan`, `dt.validate.validate`. The IR contract is
`dt/ir.py`; every threshold lives in `dt/params.py` (`P["stage.name"]`, overridable by `dt/params.json`).

## The self-improvement loop

The engine is deterministic; what learns is its parameters, its knowledge, its corpus and — through
an AI driver — its code. Everything is measured against ground truth and gated before it is kept
([docs/SELF_IMPROVEMENT.md](docs/SELF_IMPROVEMENT.md)).

```bash
dt corpus build --kind mwc --n 12 --seed 1      # real Material Web pages, IR extracted from the DOM
dt corpus build --kind synth --n 12 --seed 1    # pure-IR random M3 screens rendered by dt.render
dt corpus build --kind screens                  # (network) refresh public reference screenshots

dt bench --json                                 # perceive -> map -> render on the corpora; composite score
dt bench --gate                                 # regression gate vs knowledge/baseline.json (exit 1 on regression)
dt bench --set-baseline                         # accept a verified improvement as the new baseline
python -m dt.selftest.attribution --workers 4   # which stage (render / map / perceive) costs the most
python -m dt.selftest.move_stats out/           # which refine moves pay for their renders
python -m dt.validate.roundtrip ir.json         # IR -> Figma plan -> plugin (headless) -> IR -> render: lossless?

dt improve --top 10 -o brief.md                 # bench -> worst (case, metric) pairs -> brief with evidence,
                                                # suspected stage, prior knowledge, failures.jsonl templates
dt tune --iters 50                              # random + coordinate search over dt.params ranges;
                                                # writes dt/params.json only if the holdout split improves
```

The loop an AI driver runs: `dt bench` → `dt improve` → edit **one** stage → `python -m pytest -q`
→ `dt bench --gate` (composite must not drop, no metric may regress) → append the brief's filled
template to `knowledge/failures.jsonl` → commit. Rules: one stage per iteration, never tune on the
holdout, never change `dt/ir.py` semantics, never delete a regression case.

Learned state that ships with the repo: `dt/params.json` (tuned thresholds),
`fixtures/design_systems/*.json` (Material 3 catalog + learned systems), `knowledge/failures.jsonl`
(failure taxonomy: symptom, root cause, fix, metric before/after), `knowledge/rounds.jsonl` (the
improvement rounds that produced this version) and `knowledge/baseline.json` (the regression gate).

## Repository map

```
dt/ir.py              the IR contract (Document / Node / Box / Color / Fill / Stroke / Shadow / TextStyle / Layout / ComponentRef)
dt/params.py          registry of every tunable threshold (+ dt/params.json overrides)
dt/perceive/          OCR, segmentation, hierarchy, containers (group.py), layout, icons.py, fonts.py, dpr.py
dt/mapping/           design systems (Material 3, Figma, tokens, learned) and the matcher
dt/render/            IR -> HTML/CSS -> PNG (Playwright); also screenshots arbitrary URLs at any DPR
dt/compare/           ΔE maps, pixel + structural metrics, the loss, diff images
dt/refine/            critic (hypotheses incl. guarded as-is rasters) + optimizer (accept-if-loss-drops)
dt/export/            Figma build plan + plugin (+ headless fake Figma) + Figma auto-layout model + run report
dt/validate/          independent fidelity gates + export round-trip check
dt/selftest/          corpora, bench (+ gate), tune, stage attribution, move statistics
dt/pipeline.py        translate(): the end-to-end run, decisions, apply_decisions
dt/cli.py             the `dt` command (incl. `dt doctor`)
dt/mcp_server.py      the harness as MCP tools
fixtures/             fonts, icon atlas, Material Web bundle, design systems, ground-truth corpora
knowledge/            failures.jsonl, rounds.jsonl, baseline.json
scripts/              setup.sh (idempotent install), make_reference_page.py
tools/                progress report and real-screen evaluation helpers
tests/                real-renderer tests per stage (`python -m pytest -q`)
```

## Current limitations (honest list)

* **No real page passes the strict gates yet.** Mobile pages reproduce geometry closely; desktop
  pages (sidebars, dense tables, many small icons) do not. Every known failure is in
  `knowledge/failures.jsonl` with its suspected cause.
* **Low-contrast surfaces.** Surfaces within ~2 ΔE of their background (pale cards, tonal bands)
  are often not segmented, and a slightly wrong page background is not yet corrected by refine.
* **Non-Material glyphs** (custom bullets, brand marks) are left unnamed; small ones can render
  blank until the raster fallback covers them.
* **Translucency** beyond full-page dialog scrims (glass panels, overlapping translucent cards) has
  no blend model.
* **Mapping** claims some sub-parts of composite components as components of their own (tab cells,
  slider tracks) and expects the baseline Material 3 colour scheme; custom themes need a design
  system built from their tokens (`dt ds from-tokens`) or reference screens.
* **Figma plugin** is verified against a headless model of the Plugin API (including auto-layout
  re-layout and auto-width text) and by a round-trip render check, not yet inside a live Figma
  seat. Instances need the component keys of the library you have enabled.
* **Speed.** A mobile screen takes about 30–60 s with refinement, a desktop screen 1–2 min; most of
  it is browser renders in the refine loop.

## License

Apache License 2.0 (see [LICENSE](LICENSE)). Bundled third-party assets keep their own licenses:
Roboto, Material Symbols and Material Web (Apache 2.0), Google Sans and Google Sans Flex (SIL OFL
1.1). See [NOTICE.md](NOTICE.md).
