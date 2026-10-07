# rsdesign — screenshot → IR → design system → Figma, with adversarial refinement

Goal: pixel-faithful, **programmatic** translation of a UI screenshot into (a) a structured IR,
(b) a mapping onto a known design system (Material 3 first), and (c) a Figma build plan —
then an adversarial compare/refine loop that drives the output to match the reference screenshot.
No stage may rely on an LLM "looking at" an image; everything is CV + OCR + algorithms, so the
pipeline is deterministic, measurable, and self-tunable (RSI = closed loop with ground truth).

```
 screenshot.png
     │  perceive  (OCR via macOS Vision, segmentation via OpenCV, hierarchy + layout inference)
     ▼
 Document (IR)  ──── mapping ────▶ Document (IR + ComponentRef + tokens)
     │                                   │
     │ render (IR→HTML→PNG, Playwright)  │ export (IR→Figma build plan JSON → plugin creates nodes)
     ▼                                   ▼
 rendered.png ◀──── compare ────▶ loss + per-node error map + residual regions
     ▲                                   │
     └────────── refine (critic proposes moves; optimizer accepts if loss ↓; loop) ◀──┘

 selftest: ground-truth corpora (Material Web pages with DOM-derived IR; pure-IR synth) →
           bench (node IoU/recall, color ΔE, text CER, radius MAE, mapping accuracy, pixel SSIM) →
           tune (search over dt/params.py registry → dt/params.json)
```

## Environment
* `source .venv/bin/activate` (Python 3.14). Run tests with `python -m pytest -q`.
* Renderer: Playwright on system Chrome (`channel="chrome"`), DPR 1. Fonts self-hosted in
  `fixtures/fonts` (Roboto, Material Symbols Outlined). Pages must be loaded via `file://`
  (`set_content()` cannot load file:// fonts) — `dt.render.screenshot` already handles this.
* Material Web (`@material/web` 2.5) is bundled as a classic script at `fixtures/mwc/mwc.bundle.js`
  (ES modules do not load over file://). `window.__mwcReady` is set when loaded. Reference page:
  `out/mwc_test.html`.
* OCR: macOS Vision via pyobjc (`Vision.VNRecognizeTextRequest`, accurate level, language
  correction off). Word-level boxes via `candidate.boundingBoxForRange_error_`. Proven working.
* All thresholds go through `dt.params.P` (`register()` at import; `P["key"]` at use).

## Contract: `dt/ir.py` (DONE — do not change field semantics; additive changes only)
`Document(width, height, root, dpr, source_image, palette, fonts, design_system, meta)`;
`Node(id, type, name, box, fills, strokes, effects, radius, opacity, clip, visible, text,
text_style, icon_name, image_ref, layout, component, tokens, meta, children)`;
`Box(x,y,w,h)` absolute px; `Color(r,g,b,a)` with `delta_e()`; `Fill`, `Stroke`, `Shadow`,
`TextStyle`, `Layout`, `ComponentRef`. JSON roundtrip via `to_dict/from_dict`, `Document.load/save`.
`node.meta` is free-form evidence (e.g. `{"src": "ocr", "conf": 0.98}`); never *required* downstream.

## Contract: `dt/render/*` (DONE)
* `render_html(doc, mode="absolute"|"flex") -> str`
* `render_doc(doc, out_path=None, mode="absolute") -> np.ndarray(H,W,3)` ; `html_to_png(html,w,h)`
* `render_url(url, w, h, out_path, wait_ms, script) -> (rgb, script_result)` — screenshot any page
  and evaluate JS in it (used to extract DOM ground truth).

## Module ownership (parallel builders; files are DISJOINT — do not edit another module's files)

### A. `dt/perceive/` → `perceive(image) -> Document`
* `ocr.py` — `ocr_lines(rgb) -> list[TextLine]` (`text, box, words:[(str, Box)], conf`), plus
  `estimate_text_style(rgb, line) -> TextStyle` (size from glyph x-height/cap-height calibration,
  weight from stroke width via distance transform, color = dominant non-background glyph color,
  line_height from inter-line spacing within a paragraph). Must calibrate the Vision-box→font-size
  ratio against the renderer (render known sizes, measure, fit a linear map; store as params).
* `segment.py` — `find_regions(rgb, text_boxes) -> list[Region]`: background estimation; flat-color
  connected components (quantized); bordered boxes via edges/contours; rounded-rect radius
  estimation (fit corner arcs on the mask); circles/pills; dividers/lines; icon blobs (small,
  high-contrast, non-text, inside text-free areas); image regions (high color entropy); shadows
  (soft gradient halo just outside a region). Region: `box, kind, fill, stroke(color,width), radius,
  has_shadow, conf, mask`.
* `hierarchy.py` — `build_tree(regions, lines, W, H) -> Node` containment nesting (area-desc; child
  if ≥ P["perceive.hier.contain_ratio"] inside), text attached to deepest container, z-order.
* `layout.py` — `infer_layout(root)` sets `node.layout` for frames with ≥2 children: row/column
  from projection non-overlap, gap = median spacing, padding = parent box − children union, align &
  justify from cross/main-axis positions. Validation rule: rendering in `mode="flex"` must reproduce
  the absolute render (compare exposes this; bench tracks `layout_consistency`).
* `__init__.py` — `perceive(path_or_rgb, dpr=1.0) -> Document`; also `perceive_region(rgb, box)`
  used by the refine critic for "missing" regions.
* Tests: `tests/test_perceive.py` — render IR docs with `dt.render` (buttons, cards, text lines,
  icons, dividers at known geometry) and assert recovered boxes within ±2px, colors ΔE<3, text
  exact, radius within ±2, hierarchy correct. Use `out/mwc_test.png` too.

### B. `dt/compare/` → loss & metrics
* `pixel.py` — `diff_map(a,b)->HxW float32 (ΔE Lab)`, `pixel_metrics(a,b)->{mse,psnr,ssim,mean_de,
  frac_bad(ΔE>thr)}`, `residual_regions(diff, thr, min_area)->list[Box]` (connected components of
  error), `alignment_offset(a_crop, b_crop)->(dx,dy,conf)` via `cv2.phaseCorrelate`.
* `structural.py` — `match_nodes(pred, gt, iou_thr)->list[(pred_id, gt_id, iou)]` (Hungarian with
  type compatibility), `structural_metrics(pred, gt)->dict`: node_precision, node_recall, mean_iou,
  color_de (matched fills), text_cer, text_recall, radius_mae, type_acc, component_acc (name+variant
  when gt has component), token_acc.
* `loss.py` — `evaluate(doc, target_rgb, rendered_rgb=None)->LossReport{total, pixel, structure,
  per_node:{id: err}, residual_regions, diff_map}`; per-node error = mean ΔE inside node box
  excluding children; `total` is the scalar the optimizer minimizes (weights in params).
* `visualize.py` — side-by-side + heatmap PNG for reports (`save_diff_image`).
* Tests: `tests/test_compare.py` — identical images → 0; shifted node → residual region located at
  the node; phase-correlate recovers a known shift; structural metrics on perturbed IR.

### C. `dt/mapping/` → design-system ingestion & matching
* `design_system.py` — `Token(name, category, value)`, `ComponentSpec(key, name, library,
  variants, signature, exemplar, slots)`, `Signature` (height range, radius rule e.g. "pill"|px|
  "none", fill token candidates, stroke rule, text role, children pattern, icon?, aspect), and
  `DesignSystem(name, tokens, components, fonts)` with JSON load/save (`fixtures/design_systems/`).
* `material3.py` — `material3() -> DesignSystem`: full M3 baseline light scheme color roles
  (md.sys.color.*, from Material Web tokens: `node_modules/@material/web/tokens/`), the 15 typescale
  roles, shape scale (none/xs/sm/md/lg/xl/full), elevation levels, 4px spacing grid, and a component
  catalog with signatures covering: buttons (filled/outlined/text/elevated/tonal), FAB (small/
  regular/large/extended), icon buttons, chips (assist/filter/input/suggestion), text fields
  (filled/outlined), switch, checkbox, radio, slider, list item (1/2/3-line), cards (elevated/filled/
  outlined), top app bar (small/center/medium/large), navigation bar/rail/drawer, tabs, dialog,
  snackbar, divider, badge, progress, menu, search bar, segmented button. Derive exact metrics from
  Material Web's own token files/CSS rather than guessing.
* `ingest_figma.py` — `from_figma_file(file_key, token=env FIGMA_TOKEN)` using REST
  (`/v1/files/:key`, `/components`, `/styles`, `/variables/local`) and a pure
  `from_figma_json(file_json, components_json, styles_json, variables_json) -> DesignSystem` tested
  with a small fixture (no network in tests).
* `ingest_screenshots.py` — `from_reference_screenshots(paths) -> DesignSystem`: perceive each,
  cluster recurring subtrees by signature, emit learned ComponentSpecs with exemplars + palette/
  typescale tokens from frequency analysis. (Depends on `dt.perceive`; write against the interface
  above and guard the test with `pytest.importorskip`.)
* `ingest_library.py` — W3C DTCG design-tokens JSON and Material Theme Builder export → tokens.
* `matcher.py` — `map_document(doc, ds) -> Document`: (1) subtree pattern matching → `ComponentRef`
  with variant resolution and props (label/icon), confidence; (2) token snapping for everything:
  colors (nearest role with ΔE < P["map.color_de"]), radius → shape scale, text → typescale role,
  spacing/gaps/padding → 4px grid; (3) as-is translation preference: if a subtree matches a component
  ≥ P["map.min_conf"], children are collapsed into the instance's props. Record evidence.
* Tests: `tests/test_mapping.py` — M3 catalog sanity (every component has a signature), token
  snapping, matching on hand-built IR of a filled button / outlined text field / list item / FAB;
  Figma JSON fixture ingestion.

### D. `dt/export/` → Figma
* `figma_json.py` — `to_build_plan(doc) -> dict` (schema in `dt/export/BUILD_PLAN.md`): tree with
  parent-relative coords, Figma paints (0..1 RGB + opacity), strokes (weight, align), effects
  (DROP_SHADOW), cornerRadius / per-corner, auto-layout (layoutMode, itemSpacing, padding*, primary/
  counter axis align/sizing), text (characters, fontName{family,style}, fontSize, lineHeight,
  letterSpacing, textAlignHorizontal, fills), icon nodes (Material Symbols text node), instances
  `{componentKey, variantProperties, props, fallback: <frame>}`, token/variable bindings by name.
  `validate_plan(plan) -> list[str]` errors.
* `figma_plugin/manifest.json`, `code.js`, `ui.html` — plugin: paste/upload plan JSON → builds nodes
  (loadFontAsync per font/style, createFrame/createText/createRectangle/createEllipse, auto-layout,
  `importComponentByKeyAsync` for instances with fallback to the frame, bind variables by name via
  `figma.variables.getLocalVariablesAsync` when present). Keep `code.js` plain ES2019 (no bundler).
* `figma_plugin/fake_figma.js` + `tests/test_export.py` — a Node-side fake `figma` API that executes
  `code.js` headlessly and asserts the created node tree matches the plan (node count, names, sizes,
  fills, text). Run via `node` subprocess from pytest.
* Also `dt/export/report.py` — `write_report(out_dir, doc, loss_history, metrics)` → `report.md` +
  images for a translation run.

### E. `dt/selftest/` → ground-truth corpora (phase 1) ; bench + tune (phase 2)
* `mwc_corpus.py` — `generate(n, out_dir, seed)`: compose random Material Web pages (grammar:
  app bars, lists with switches/checkboxes, card grids, forms with text fields + buttons, chip rows,
  FABs, tabs, dialogs, nav bars/rails) with Roboto + Material Symbols via the bundle; render with
  `render_url`; extract **ground-truth IR from the DOM** (walk light + shadow DOM:
  `getBoundingClientRect`, computed background-color/border-radius/border/box-shadow/color/font-*,
  text content, slot icons) → `Document` with `component=ComponentRef(name, variant from tag+attrs)`
  for each md-* element. Save `<id>.png`, `<id>.gt.json`, `<id>.html` under
  `fixtures/corpus/mwc/`. Must produce a deterministic set from a seed.
* `synth.py` — `generate(n, out_dir, seed)`: pure-IR random docs built from M3 tokens (cards,
  buttons, text lines, dividers, icons, lists) → `render_doc` → png + gt json (gt == the IR).
* (phase 2) `bench.py` — `run(corpus_dirs, stages, limit) -> report` with per-metric mean/p50/worst
  and per-case rows; writes `out/bench/<ts>/report.json|md`.
* (phase 2) `tune.py` — search `P.ranges()` (random + coordinate descent, train/holdout split) →
  `dt/params.json`; appends `out/tune/history.jsonl`.
* Tests: `tests/test_selftest.py` — generate 3 mwc pages + 3 synth docs, assert gt docs load, every
  md-* element has a component ref, boxes inside the page, png size matches.

### F. (phase 2) `dt/refine/` → adversarial loop
* `critic.py` — `critique(doc, target, rendered, report) -> list[Hypothesis]`: residual region →
  classify {missing, extra, shifted (phase-correlation offset per node), recolor (sample target
  under node), resize (edge search), text (re-OCR region), radius}; each with an `apply(doc)`.
* `optimizer.py` — `refine(doc, target, max_iters, patience) -> (doc, history)`: greedy accept-if-
  loss-decreases over hypotheses ordered by expected gain; per-node local moves; converge on
  plateau. Must log every accepted move (`history`) for the report.
* Tests: perturb a synth doc (shift/recolor/resize/delete node) → refine recovers ≥90% of the loss.

### G. (phase 2) `dt/cli.py` — `dt translate shot.png --ds material3 -o out/run1`,
`dt corpus build --kind mwc --n 50`, `dt bench`, `dt tune --iters 50`, `dt export-figma ir.json`.

## Quality bar for every builder
* Register thresholds in `dt.params` with ranges; no magic numbers.
* Write tests that use the real renderer for ground truth (not mocks). Tests must pass:
  `python -m pytest tests/test_<module>.py -q`.
* Return a structured report: files written, public functions, test results (paste the pytest
  summary line), known gaps, and the 3 most likely failure modes on real Google-app screenshots.

## Product shape: a standalone harness any AI model (or human) can run
This repo is the deliverable: an engine + a protocol + a self-improvement loop, published as a
GitHub project. An AI model in the user's environment *drives* the harness; the harness does the
deterministic work and surfaces only genuine judgment calls.

* **Portability** — every platform-specific piece has a fallback behind an interface:
  OCR backends `vision` (macOS, default when available) → `rapidocr` (ONNX, pip) → `tesseract`
  (`dt/perceive/ocr.py: get_backend()`); browser `channel="chrome"` → Playwright's bundled
  chromium (`dt/render/screenshot.py`). Never import pyobjc at module top level.
* **Agent protocol** — every stage is a CLI subcommand with JSON in/out and stable exit codes
  (`dt perceive|map|render|compare|refine|export|bench|tune|improve`), and the same operations are
  exposed as an MCP server (`dt/mcp_server.py`, phase 3) so Claude/Cursor/any agent can call them as
  tools. `AGENTS.md` (phase 3) is the operating manual for AI drivers: the loop to run, how to read
  a `LossReport`, how to resolve decisions, how to add a component signature, how to run the
  regression gate before committing learned state.
* **Decision queue** — ambiguities the engine cannot settle (unmatched component with top-3
  candidates + evidence, conflicting hierarchy, low-confidence OCR, unknown icon) are written to
  `decisions.json`; `dt apply-decisions` merges answers back. The engine must always produce a
  complete best-effort output even with zero decisions answered.
* **Learned state ships with the repo** — `dt/params.json` (tuned thresholds),
  `fixtures/design_systems/*.json` (M3 + learned signatures), `knowledge/failures.jsonl` (failure
  taxonomy: case, stage, symptom, root cause, fix, metric delta), and `out/bench/baseline.json`
  (regression gate). `dt improve` = bench → worst cases → structured report an AI can act on.
