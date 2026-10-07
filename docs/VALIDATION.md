# Validation: how we decide the output is visually 1:1 with the input

The refine loop minimises `dt.compare.loss`. A system must not grade its own homework, so final
validation uses an **independent** module, `dt/validate/`, with different measurements and
explicit pass/fail gates. Every `dt translate` run ends with `validation.json` + `validation.md`.

## Measurements (all programmatic, all reported per run)

| Family | Measure | Why it is needed |
|---|---|---|
| Pixel | ΔE2000 map; fraction of pixels with ΔE > 2.3 (JND), > 5, > 10; max ΔE; SSIM, MS-SSIM | Overall closeness; the JND fraction is the headline "pixel-exact" number |
| Pixel (split) | Same numbers restricted to text regions vs non-text regions | Font substitution corrupts text pixels; must not hide geometry/colour errors (or be hidden by them) |
| Geometry | Edge maps (Canny) of target and render → edge IoU, symmetric chamfer distance (px), % of target edges within 1 px of a render edge | Detects misplaced / missized / missing shapes independently of colour and font |
| Colour | Flat-colour regions segmented on the **target**; per-region mean ΔE vs the render; worst regions listed | Catches wrong fills, wrong strokes, wrong shadows, region by region |
| Text | OCR both images with the same backend; match lines by position; CER, Δx/Δy (px), Δsize | Same-OCR-both-sides cancels OCR bias; checks characters, position, size |
| Consistency | perceive(render) vs IR (are perceiver and renderer inverses?) and perceive(render) vs perceive(target) | A stable fixed point means the IR explains the pixels; disagreement localises errors |
| Export round-trip | Figma build plan → fake-figma tree → (Figma auto-layout + text auto-resize model) → IR → render → mean ΔE2000 vs original render (`dt/validate/roundtrip.py`, lossy if > `validate.roundtrip.max_mean_de` = 0.5) | Export must be lossless |
| Ground truth (corpora only) | Node recall/precision @IoU 0.7, mean IoU, colour ΔE, text CER, radius MAE, component+variant accuracy, token accuracy, layout consistency | Only possible where GT exists; the self-improvement loop optimises these |

## Gates (reported as PASS/FAIL per tier)

| Tier | Criteria |
|---|---|
| **pixel-exact** | JND fraction (non-text) < 0.5 % · chamfer < 0.5 px · edges-within-1px > 99 % · text CER = 0 · text Δpos ≤ 1 px |
| **visually identical** | JND fraction (non-text) < 2 % · chamfer < 1 px · edges-within-1px > 97 % · text CER = 0 · text Δpos ≤ 2 px · worst region ΔE < 5 |
| **structurally faithful** | edge F1 @1 px > 0.85 · text CER < 0.05 · all flat regions ΔE < 10 · consistency node recall > 0.9 |
| **every tier** | render size == target size · rasterised area ≤ 35 % · no text node under a crop · no target text line (OCR box ≥ 9 px ≈ 12 px type) painted by a non-logo crop |

Everything is measured on the **target frame**: a render of another size is padded with a flat
colour opposite to the target (never cropped into agreement). Exact-pixel edge IoU is still
reported but not gated: anti-aliasing alone puts the gt IR of a Material Web page at 0.56, and
it ranked a perfect doc shifted by 1 px below a 6 px colour mosaic of the target. The bench
composite applies the same editability rule (`bench.raster_penalty` × rasterised share, and the
canvas IoU) as a multiplicative factor; see `dt/selftest/metrics.py`.

Text-region pixel error is reported but **not** gated (fonts are a known, separately tracked
limitation; font identification is a self-improvement item). Gates can be tightened via
`dt.params` (`validate.gate.*`).

## Human-verifiable artefacts
`validation/side_by_side.png`, `validation/heatmap.png`, `validation/blink.html` (flicker
comparator), `validation/worst_regions.png` (crops of the 8 worst regions, target vs render).

## Where this runs
* `dt translate … ` → always, at the end.
* `dt bench` → on every corpus case (GT metrics + fidelity); the composite score includes the
  JND fraction and chamfer so tuning cannot "win" by gaming one measure.
* `dt validate target.png render.png [--ir ir.json]` → standalone.
