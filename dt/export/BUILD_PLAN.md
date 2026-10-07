# Figma build plan — schema v1.0

A **build plan** is the JSON handed from `dt.export.figma_json.to_build_plan(doc)` to the Figma plugin
(`dt/export/figma_plugin/code.js`). It is a dumb, fully resolved description of the nodes to create:
no IR concepts, no design-system lookups, no pixels. Anything that can be decided in Python is decided
in Python so the plugin only has to call the Figma API in the right order.

Versioning: `version` is `MAJOR.MINOR`. The plugin accepts any plan with the same MAJOR. Additive
fields bump MINOR; renames/semantic changes bump MAJOR. `validate_plan(plan)` checks this document.

```jsonc
{
  "version": "1.0",
  "generator": "designtranslator",
  "width": 400,  "height": 300,                   // document size (px, DPR 1)
  "fonts": [ { "family": "Roboto", "style": "Medium" } ],   // every (family, style) referenced
  "fallbackFont": { "family": "Roboto", "style": "Regular" },
  "imagePlaceholder": { "r": 0.88, "g": 0.88, "b": 0.88 },  // solid used for unresolvable IMAGE fills
  "root": { /* PlanNode (FRAME) */ },
  "warnings": [ "..." ],                            // produced during conversion (lossy spots)
  "meta": { "sourceImage": null, "designSystem": "material3", "dpr": 1.0, "nodeCount": 9 }
}
```

## PlanNode

Common fields (every node):

| field | type | notes |
|---|---|---|
| `id` | string | unique in the plan (IR node id) |
| `name` | string | IR name, else text snippet / `icon/<name>` / `<Component>/<variant>` / type |
| `type` | `FRAME` `RECTANGLE` `ELLIPSE` `LINE` `TEXT` `INSTANCE` | |
| `x`, `y` | number | **relative to the parent's top-left**; root is `0,0` |
| `width`, `height` | number | px; `LINE` has `height: 0` |
| `visible` | bool | |
| `opacity` | 0..1 | |
| `variableBindings` | `{field: variableName}` | see *Variables* |
| `tokenRefs` | `{irKey: tokenName}` | IR tokens that have no Figma field (provenance only) |
| `layoutSizingHorizontal` / `layoutSizingVertical` | `FIXED` `HUG` `FILL` | only on children of auto-layout frames |
| `source` | `{irId, irType}` | provenance |

### Paint (`fills`, `strokes`)
Figma paints with 0..1 colors. `SOLID.opacity = IR color alpha × IR fill opacity`.

```jsonc
{ "type": "SOLID", "color": {"r":0,"g":0,"b":1}, "opacity": 1 }
{ "type": "GRADIENT_LINEAR", "opacity": 1,
  "gradientStops": [ { "position": 0, "color": {"r":1,"g":0,"b":0,"a":1} }, ... ],
  "gradientHandlePositions": [ start, end, width ] }     // normalized node space, see below
{ "type": "GRADIENT_RADIAL", ... "gradientHandlePositions": [ center, xRadius, yRadius ] }
{ "type": "IMAGE", "scaleMode": "FILL", "imageRef": "<path|url>", "opacity": 1 }   // plugin: placeholder unless imageHash present
```

Linear handles come from the CSS angle: the gradient line passes through the center with length
`|w·sin a| + |h·cos a|`; `start`/`end` are its ends normalized by `(w, h)`; the third handle is the
perpendicular at half length (Figma's own default convention). `180deg` (CSS top→bottom) gives
`[{0.5,0},{0.5,1},{1,0}]`.

### Shape fields (`FRAME`, `RECTANGLE`, `ELLIPSE`, `LINE`)
| field | notes |
|---|---|
| `fills` | list of Paint (may be empty; the plugin clears Figma's default fill) |
| `strokes`, `strokeWeight`, `strokeAlign` | `INSIDE` `CENTER` `OUTSIDE`; one weight/align per node |
| `strokeTopWeight` `strokeRightWeight` `strokeBottomWeight` `strokeLeftWeight` | per-side (FRAME/RECTANGLE only; implies INSIDE) |
| `effects` | `[{type: DROP_SHADOW|INNER_SHADOW, color{r,g,b,a}, offset{x,y}, radius (=CSS blur), spread, visible, blendMode}]` |
| `cornerRadius` | uniform; else `topLeftRadius` `topRightRadius` `bottomRightRadius` `bottomLeftRadius` |
| `clipsContent` | FRAME only (IR `clip`) |
| `children` | FRAME only |

IR `line` nodes become filled `RECTANGLE`s by default (pixel exact; radius and strokes kept, e.g. a
32×4 r=2 drag handle). With
`P["export.figma.line_mode"] == "line"`, thin horizontal lines become Figma `LINE` nodes whose `y`
is the line's vertical center and `strokeWeight` its thickness (`strokeAlign: CENTER`).

### Auto-layout (FRAME with IR `layout.mode != none`)
| IR | plan |
|---|---|
| `mode` row / column | `layoutMode` `HORIZONTAL` / `VERTICAL` |
| `gap` | `itemSpacing` |
| `padding` (t, r, b, l) | `paddingTop` `paddingRight` `paddingBottom` `paddingLeft` |
| `justify` start/center/end/space-between | `primaryAxisAlignItems` `MIN` `CENTER` `MAX` `SPACE_BETWEEN` |
| `align_items` start/center/end/stretch | `counterAxisAlignItems` `MIN` `CENTER` `MAX` (stretch → `MIN` + children `FILL` on the cross axis) |
| `sizing_h`/`sizing_v` fixed/hug/fill | `primaryAxisSizingMode` / `counterAxisSizingMode` `FIXED` / `AUTO` (mapped to main/cross axis by mode) |
| `wrap` | `layoutWrap` `WRAP` / `NO_WRAP` |

**Layout verification** (`P["export.figma.verify_layout"]`, default on): Figma ignores the x/y of
auto-layout children and re-lays them out, so `to_build_plan` simulates Figma's auto-layout
(`dt/export/figma_layout.py`) and keeps `layoutMode` only on frames where (a) every child lands within
`export.figma.layout_exact_tol` px of its IR box and (b) nothing moves when auto-width text is grown by
`export.figma.layout_text_probe` px (Figma sizes such text from its own glyph metrics). Other frames are
exported with absolute children (innermost first) and a warning. Inferred layouts that are only
approximately right (e.g. children appended by refine, or `center` alignment of left-aligned lines) would
otherwise move content on import.

Children of such frames carry `layoutSizingHorizontal/Vertical` (from their own `layout.sizing_*`,
`FIXED` when absent; `HUG` is demoted to `FIXED` on nodes that cannot hug). The plugin sets these
**after** `appendChild`, and re-applies the frame's fixed size after children are added.

### TEXT
| field | notes |
|---|---|
| `characters` | string |
| `fontName` | `{family, style}`; style from weight+italic: 100 Thin, 200 ExtraLight, 300 Light, 400 Regular, 500 Medium, 600 SemiBold, 700 Bold, 800 ExtraBold, 900 Black, italic → `"Italic"` / `"<Style> Italic"` |
| `fontSize` | px |
| `lineHeight` | `{value, unit: "PIXELS"}` or `{unit: "AUTO"}` |
| `letterSpacing` | `{value, unit: "PIXELS"}` |
| `textAlignHorizontal` / `textAlignVertical` | `LEFT CENTER RIGHT` / `TOP CENTER BOTTOM` |
| `textDecoration`, `textCase` | `NONE UNDERLINE STRIKETHROUGH` / `ORIGINAL UPPER LOWER TITLE` |
| `textAutoResize` | left-aligned text: `WIDTH_AND_HEIGHT` (never wraps, like dt.render); centred/right: `NONE` (fixed box). Auto-width text has no vertical slack, so `valign` middle/bottom is baked in: pixel line height → `y`/`height` moved to the line box; one line with normal line height and `middle` → `lineHeight` = box height; otherwise `NONE` |
| `fills` | from the text color |
| `isIcon` | `true` for icon nodes: `fontName {Material Symbols Outlined, Regular}`, `characters` = icon name, `fontSize = min(w, h)`, centered |
| `iconFill` | `1` for filled Material Symbols (IR `meta.icon_fill`); the plugin records it as plugin data `dt.iconFill` and warns: set the font's FILL axis to 1 (the plugin API cannot set variable-font axes) |

Plugin font cascade when `(family, style)` is not installed: fallback family @ same style →
requested family @ Regular → `fallbackFont` → Roboto/Inter; every substitution is a warning.

### INSTANCE
```jsonc
{ "type": "INSTANCE", "id": "...", "name": "Button/filled", "x": 16, "y": 100, "width": 120, "height": 40,
  "componentKey": "<figma component key>", "componentName": "Button", "library": "material3",
  "variantProperties": { "style": "filled" },     // applied via instance.setProperties (VARIANT props)
  "props": { "label": "Save", "icon": "add" },    // TEXT/BOOLEAN component props, else a text sub-node named like the prop
  "confidence": 0.9,
  "fallback": { /* FRAME PlanNode with the same geometry + the IR children */ } }
```
The fallback is built from the instance's children; an instance collapsed by `dt map --collapse` uses its
`meta.collapsed_children` (the real subtree), and only an instance with neither gets a synthesized label.
The plugin calls `importComponentByKeyAsync(componentKey)`; on any failure it builds `fallback`
instead (named after the instance) and records a warning. An instance without children but with a
string `label` prop gets a synthesized centered text in its fallback.

### Variables
`variableBindings` maps Figma fields to variable **names** (`md.sys.color.primary`), derived from IR
`node.tokens` (`fill`/`background`/`color`/`text_color` → `fills`, `stroke`/`border` → `strokes`,
`radius` → `cornerRadius`, `gap` → `itemSpacing`, `padding*` → `padding*`, `font_size`, `line_height`,
`letter_spacing`, `opacity`, `width`, `height`). The plugin resolves names with
`figma.variables.getLocalVariablesAsync()`; `fills`/`strokes` bind the first SOLID paint's color via
`setBoundVariableForPaint`, other fields via `node.setBoundVariable`. Missing variables → warnings,
never failures. Unknown IR token keys are kept in `tokenRefs`.

### Images
`IMAGE` paints whose `imageRef` is a local file (path or `file://`) are embedded as PNG data URIs
(`export.figma.embed_local_images`, up to `export.figma.embed_max_bytes`); the plugin turns data URIs into
Figma images (`figma.createImage`). Remote URLs keep the grey placeholder (plugins cannot fetch them).

### Provenance
The plugin stamps every node with `setPluginData("dt.id", <plan id>)`, so a Figma file can be mapped back
to the IR (used by the export round-trip check, `dt/validate/roundtrip.py`).

## Plugin protocol
`ui.html → code.js`: `{type: "build", plan, options: {closeWhenDone?, offsetX?, offsetY?}}`
`code.js → ui.html`: `{type: "done", summary: {created, texts, instances, fallbacks, warnings[], fontsLoaded[], fontsMissing[], rootId}}` or `{type: "error", message}`.

Headless: `node dt/export/figma_plugin/run_fake.js --plan plan.json [--components k1,k2] [--variables n1,n2] [--fonts "Roboto:Regular;..."] [--no-variables] [--code other_code.js]`
prints `{ok, summary, tree, posted, closed, images}` where `tree` is the created page subtree and `images`
maps each `imageHash` (content hash, like Figma's) to its base64 PNG bytes. The fake mirrors Figma's
setter semantics that matter for round-trips: `cornerRadius` writes the four corners and `strokeWeight`
the four side weights (differing ones read back as `"MIXED"`, standing in for `figma.mixed`).

Round-trip: `python -m dt.validate.roundtrip ir.json [-o out_dir]` → plan → plugin on the fake → Figma
auto-layout/text model → IR → render → mean ΔE2000 vs the direct render (> `validate.roundtrip.max_mean_de`
= lossy export).

## Validation (`validate_plan`)
Returns a list of error strings: version major, positive width/height, font list shape, node types,
unique ids, finite coordinates within `export.figma.max_dimension`, sizes ≥ 0, opacities in [0,1],
paint/effect shapes and 0..1 colors, gradient stops (≥ 2, positions in [0,1], three handles),
enum values for stroke/layout/text fields, `FILL` sizing only under auto-layout parents, TEXT
needs `characters`/`fontName`/`fontSize ≥ export.figma.min_font_size`, INSTANCE needs
`componentKey` + a FRAME `fallback`, children only on FRAMEs, node count ≤ `export.figma.max_nodes`.
