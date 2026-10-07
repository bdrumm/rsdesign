# Component variants: one vocabulary

Every `ComponentRef.variant` in this repo uses one vocabulary per component, whether it comes
from the matcher (`dt.mapping.matcher.map_document`), the Material Web corpus
(`dt.selftest.mwc_corpus.component_for`) or the synth corpus (`dt.selftest.synth.ref`).
`compare.structural.component_acc` counts a match only when the name and every gt variant
value agree, so a corpus and a matcher that describe the same component with different keys
score 0 on correct predictions. Before this vocabulary existed that mismatch kept
`component_acc` at 0.06.

* Source of truth: `dt.mapping.material3.VARIANTS`. It ships as `meta.variants` in
  `fixtures/design_systems/material3.json` (regenerate with `python -m dt.mapping.material3`).
* Values are lower-case strings. Booleans are `"true"` / `"false"`. The first value listed for a
  key is its default.
* Variants describe appearance classes only. Content goes to `ComponentRef.props`: label, icon,
  text, value, headline, supporting text and destination count.
* `material3.variant(name, **values)` returns a complete, validated dict and raises on an unknown
  key or value. Both corpora build their refs through `material3.component_ref`, so they cannot
  drift from the catalog.
* `DesignSystem.normalize_variant(name, variant)` projects any dict onto the vocabulary (exact
  keys, unknown values become the default). The matcher applies it to every match. A design
  system without `meta.variants` (for example a Figma ingest) passes variants through unchanged.
* Ground-truth keys are `m3.<name>[:k=v,...]`, for example `m3.button:style=filled`,
  `m3.fab:extended=false,size=small` or `m3.divider` (`material3.component_key`).

## Vocabulary

| Component | Variant keys (default first) | Material Web ground truth | Matcher cue |
|---|---|---|---|
| Button | `style`: filled · outlined · text · elevated · tonal | tag (`md-filled-tonal-button` → tonal) | fill / stroke / shadow / label colour |
| FAB | `size`: regular · small · large; `extended`: false · true | `size` attr (medium → regular); `extended` = has `label` | height 56/40/96, radius 16/12/28; extended = aspect ≥ 1.4 (weighted ×3) |
| IconButton | `style`: standard · filled · tonal · outlined | tag | fill / outline on a 40px square |
| Chip | `type`: assist · filter · input · suggestion; `selected`: false · true | tag + `selected` attr (an elevated assist chip is still `assist`) | icon pattern, fill, stroke |
| TextField | `style`: filled · outlined | tag | 56px; outline vs container fill (+ optional active-indicator line) |
| Select | `style`: filled · outlined | tag | as TextField plus a required trailing arrow (icon or 10×5 vector) |
| Switch | `selected`: false · true | `selected` attr | 52×32 pill + handle; track fill primary vs surface-container-highest |
| Checkbox | `checked`: false · true | `checked` or `indeterminate` attr | filled primary vs 2px outline |
| Radio | `checked`: false · true | `checked` attr | ring colour + inner dot |
| Slider | – | – | – |
| ListItem | `lines`: 1 · 2 · 3 | host height ≤ 60 / ≤ 80 / more | height 48–60 / 64–80 / 84–96; body text + headline required (critical) |
| Card | `style`: elevated · filled · outlined | `data-variant-style` | fill / shadow / outline-variant stroke |
| TopAppBar | `size`: small · center · medium · large | `data-variant-size` | 64/112/152px; title-large/medium text required (critical); `title_align` start vs center |
| Tabs | `type`: primary · secondary | secondary when any child is `md-secondary-tab` | equal-width cells + bottom indicator (both critical); indicator 3px primary, 2px secondary |
| NavigationBar | – | – | – |
| NavigationRail | – | – | – |
| NavigationDrawer | `type`: standard · modal | – | shadow |
| Snackbar | `action`: false · true | `data-variant-action` | inverse surface; 1 text vs 2 texts; heights 48 / 68 only |
| Dialog | – | the rounded surface, not the full-viewport host | – |
| Divider | – | – | – |
| Badge | `size`: small · large | `data-variant-size` on the red pill itself | 6px dot vs 16px pill |
| Progress | `type`: linear · circular | `md-linear-progress` / `md-circular-progress` | 4px track vs 48px ring |
| Menu | – | – | – |
| SearchBar | – | – | – |
| SegmentedButton | – | – | – |

### Parts and containers are not components

`md-list` (List), `md-chip-set` (ChipSet), `md-primary-tab` / `md-secondary-tab` (Tab),
`md-menu-item` (MenuItem) and `md-select-option` (SelectOption) belong to their parent
component and have no catalog entry. Ground truth records them as `meta.part`, with no
`ComponentRef`, so they do not count toward `component_acc`. The synth corpus does the same for
its list frame, and `md-dialog`'s full-viewport host becomes `meta.part = "DialogHost"`.

## Ground-truth conventions that go with the vocabulary (mwc corpus)

* **Paint lifting.** Material Web paints most surfaces on an inner element, so the `md-*` host
  is an unpainted square while a child carries the pill. When a host has no fill and no stroke,
  and a child box covers at least `selftest.mwc.lift_cover` (0.9) of it with an opaque fill or a
  stroke, the host takes that child's fill, stroke, radius and shadows. The child is removed, its
  own children move up to the host, and `meta.lifted_from` records what was lifted. A later
  stroke-only covering child (an outline drawn separately from the background) contributes its
  stroke. The lifted paint keeps its stacking layer (`meta.z`), so a menu surface still occludes
  the page beneath it.
* **Dialog.** The `Dialog` ref sits on the rounded surface container. The dialog content is
  nested under it, and the translucent scrim stays on the `DialogHost` part.
* **Inline svg colour.** The colour of the first visible shape outside `mask` / `clipPath` /
  `defs` whose fill (or stroke) differs from the backdrop behind the svg by more than
  `selftest.mwc.backdrop_de` ΔE. Without such a shape the svg's own `color` is used if it differs
  from the backdrop; otherwise there is no colour. White is never a default: a radio ring is
  `#6750a4` or `#49454f`, and only a check mark on a primary container is white.
* **Icons.** `box` stays the 24px font box. `meta.ink_box` = `[x, y, w, h]` holds the tight glyph
  bounds from canvas text metrics around the text baseline. On the shipped corpus 86% of ink
  boxes are within 1px of the pixel ink; the rest are icons under badges or navigation pills,
  where the pixel check itself is ambiguous.
* **Badges** are the red pill itself (`span.badge`), not the icon button they decorate.

## Matcher changes that came with it

* `Signature` gained `critical` (extra constraint names whose zero score triggers
  `map.zero_penalty`), `cells` ("equal": ≥ 2 equal-width cells across the container),
  `indicator` (thickness range of a thin horizontal line in the bottom of the container,
  searched in the whole subtree) and `title_align` ("start" / "center": the primary text
  relative to the bar, or to the free span between its neighbours). The new thresholds are
  `map.cells.*`, `map.indicator.*`, `map.title.*` and `map.w.cells|indicator|title_align` in
  `dt.params`.
* `map.keep_geom_types` (default `""`) lists IR types that keep their geometric type when
  matched. The documented contract, which the CLI and Figma export rely on, is that matched
  nodes become `instance`. `compare.structural.TYPE_COMPAT` has no `line ↔ instance` pairing,
  so every matched divider (60 gt nodes) then counts as unmatched. Setting the parameter to
  `"line"`, or adding `instance` to `TYPE_COMPAT["line"]`, moves the bench from 0.7641 to 0.7903
  (component_acc 0.251 → 0.380, node_recall 0.596 → 0.640). That is a decision for the compare
  and CLI owners.

## Known ambiguities (not fixable by mapping)

* An assist chip without its icon is pixel-identical to a suggestion chip, and an unselected
  filter chip without an icon looks the same as both. 13 of the 18 remaining chip-type misses on the
  bench are assist chips read as suggestion chips.
* Unpainted components (text buttons, standard icon buttons, list items, the surface-coloured
  top app bar) have no perceived container yet, so nothing can be named. Fixing this needs a
  grouping pass in perceive (see `knowledge/failures.jsonl`). On ground-truth geometry the
  matcher reaches component_acc 0.88 (mwc 0.85, synth 0.91).

## Adding a component or a variant

1. Add the keys and values to `VARIANTS` (default first). Add the spec, or the variant with its
   signature override, in `material3.py`; every `Variant.props` must be exactly the vocabulary
   (`tests/test_mapping.py::test_variant_vocabulary_is_the_one_catalog_vocabulary`).
2. Emit it from `mwc_corpus.component_for` / `synth.ref` through `material3.component_ref`.
3. `python -m dt.mapping.material3`, regenerate the corpora (`seed=1`, `n=12`), then run
   `python -m pytest -q` and `dt bench`.
4. Add a row to the table above.
