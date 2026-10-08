# Feedback: user-informed tuning

The engine improves itself from two kinds of evidence: its own failures on ground-truth corpora
(docs/SELF_IMPROVEMENT.md) and the corrections of the people who use it on *their* screens. This
document covers the second loop: how corrections are captured, how they become local learning, how
a consumer measures it, and how (with consent) it is promoted into the shared model.

Principles:

* **Private by default.** Everything lives under `$DT_HOME` (default `~/.rsdesign`, never committed).
  No feedback code opens a network connection (tested). Nothing leaves the machine unless the user
  runs `dt feedback export` and sends the zip.
* **Generalise, never patch one case.** Learning produces *rules over signatures* (a kind of node), not
  edits to one screen. Evidence is counted per distinct screenshot, conflicting answers lower
  confidence, and every rule is measured on the user's corpus and on the global bench before it is kept.
* **Engine evidence wins when it is strong.** A rule never overrides a match the engine has strong
  evidence for (`map.learned.veto_margin`, `map.learned.veto_score`).
* **Every change is ledgered.** Accepted or rejected, each learned change is one line in the shared
  learning ledger with its exact diff and evidence.

## 1. The feedback bundle (schema `rsdesign.feedback/1`)

```json
{"schema": "rsdesign.feedback/1", "id": "fb-3f2a9c01d4e5", "created": "2026-10-07T15:40:00+00:00",
 "channel": "cli | decisions | review | figma | mcp",
 "run": {"screenshot_sha256": "...", "size": {"w": 412, "h": 730}, "dpr": 1.0, "design_system": "material3",
         "model": {"git": "82e37e24d861", "params": "1ae27152e24b", "id": "82e37e24d861+1ae27152e24b"},
         "source_path": "/abs/shot.png", "run_dir": "/abs/out/run1"},
 "items": [{"id": "i1", "kind": "component", "node_id": "r_ae96fef0", "box": [16, 64, 89, 32],
            "value": {"name": "Chip", "variant": {"type": "filter"}}, "note": "", "source": "review",
            "context": {"type": "frame", "box": [16, 64, 89, 32], "component": {"name": "Chip", "variant": {...}},
                        "signature_hash": "452780f7272b1cd2", "features": {...}}}],
 "run_rating": 4,
 "consent": {"store_screenshot": true, "share_screenshot": false, "share_text": false},
 "license": "..."}
```

* `run.model` = code revision + hash of the learned state (`dt/params.json`, the local params overlay and
  every learned-rule file), so accuracy can be tracked per model version.
* `items[].kind` and `value`:

| kind | value | meaning |
|---|---|---|
| `component` | `{name, variant}` (`name: null` = not a component) | the node is this design-system component |
| `text` | `{text}` | correct characters |
| `icon` | `{icon_name}` | correct Material Symbol |
| `geometry` | `{box: [x, y, w, h]}` | correct position / size (children move with it) |
| `color` | `{color: "#rrggbb" \| null}` | correct fill (text colour for text nodes) |
| `missing` | `{type, text?, icon_name?, fill?, component?}` + `box` | something we did not produce |
| `extra` | – | we produced something that is not there |
| `should_be_image` / `should_be_editable` | – | raster vs editable judgement |
| `ok` | – | confirmed correct (a positive label; on a matched node it is a vote for that component) |

* `context` is what the model saw (IR type, box, component and, for containers, the matcher's learned
  signature), captured at submission time so learning never depends on the run directory surviving.
* Stored as `$DT_HOME/feedback/<id>/`: `bundle.json`, `ir.base.json` (the run's IR as produced),
  `ir.json` (base + all items: the user's ground truth for this screen) and `target.png` only with
  `consent.store_screenshot` (otherwise learning uses the original file while it exists unchanged; it is
  never copied).
* Default consent: `feedback.consent.*` params (store locally yes; share screenshot / text no).

## 2. Capture channels

| channel | how |
|---|---|
| CLI file | `dt feedback add <run_dir> corrections.json [--rating 1-5\|up\|down] [--share-screenshot] [--share-text] [--no-store-screenshot]`; a file made on another screenshot (sha256 mismatch) is refused |
| decisions | `dt apply-decisions` (CLI or MCP) logs every answer as an item of the run's decisions bundle (`feedback.log_decisions`, default on); re-answering a node replaces its item |
| review page | every `dt translate` writes `review.html` (`pipeline.review_html`): a self-contained offline page, target \| render with the IR boxes; click a box -> pick a correction and value; "Draw missing" -> drag a box; rate the run; set consent; "Download corrections.json" (Blob download, works from file://) -> `dt feedback add`. `dt feedback review <run_dir>` rewrites it |
| Figma | the plugin's **Export corrections** button serialises the selected frame it built (plugin data `dt.id`, type, name, geometry, auto-layout fields, fills/strokes, characters, fonts, instance main component + variant properties, the plan's node ids stamped at build time) to `figma_corrections.json` (`rsdesign.figma-corrections/1`); `dt feedback from-figma <run_dir> figma_corrections.json` diffs it against `ir.mapped.json` |
| MCP | `submit_feedback(run_dir, items, rating, consent)`, `feedback_status()` |

The Figma diff (`dt/feedback/figma.py`) converts the export with the round-trip's `plan_tree_to_ir` and
compares it with the run's own plan converted the same way (so auto-layout re-positioning, fallback frames
and text auto-sizing are never mistaken for edits): changed characters / glyph / main component or variant
/ position or size (relative to the parent, tolerance `feedback.figma.geom_tol`) / fill (`feedback.figma.color_de`)
-> `text` / `icon` / `component` / `geometry` / `color`; a plan node gone or hidden while its parent remains
-> `extra` (children of library instances are atomic and never reported); a node without `dt.id` -> `missing`.
`figma_session.js` drives the real plugin on `fake_figma.js` with scripted edits for tests and agents
without Figma.

## 3. Learning from feedback (`dt feedback learn`)

1. **User corpus.** Each bundle with a screenshot becomes `$DT_HOME/feedback/corpus/<id>/` (`<id>.png`,
   corrected `<id>.gt.json`, `manifest.json`), so `dt bench --corpus $DT_HOME/feedback/corpus/*` works.
2. **Rules.** Component answers (and `ok` on a matched node) on containers are grouped by
   `learned_signature` (`dt/mapping/matcher.py`): a hash of categorical features that survive
   re-perception -- IR type, corner class (pill / none / rounded), coarse fill, stroke and label-ink colour
   classes (`map.learned.neutral_chroma`, `light_l`, `dark_l`; classes, not roles: the same outline reads as
   `outline` on one screen and `on-surface-variant` on another), shadow, and the ordered kinds of the content
   atoms with repeated runs collapsed. Height and aspect stay out of the hash and are matched with a tolerance
   (`map.learned.h_tol`, `map.learned.aspect_tol`), so a rule learned on two chips covers chips with other
   labels and widths. Within a signature, observations are split by height; votes are counted **per
   distinct screenshot** (six chips corrected on one screen are one piece of evidence). The answer most
   screens agree on wins; `support` = its screens, `against` = screens that said otherwise (for a variant
   rule, another variant of the same component counts against; a screen that answered both ways counts on
   both sides: the signature is ambiguous there), `confidence = support / (support + against)`.
   A rule is active at `support >= feedback.rule_min_support` (2) and `confidence >= feedback.rule_min_conf` (0.67).
3. **Applying rules (`map_document`).** `P['map.learned_rules']` (default `global,local`:
   `knowledge/learned_rules.json` and `$DT_HOME/learned_rules.json`; `''` disables). For a matching node:
   * *override* (`confidence >= map.learned.override_conf`): the rule's component (and variant) is
     assigned, confidence raised to at most `map.learned.conf_cap` (so it is no longer asked);
   * *boost*: its candidate gains `map.learned.boost x confidence`; the normal threshold decides;
   * *reject* (`component: null`): no match, no decision question;
   * *vetoed*: the best *other* component outscores the rule's component by more than
     `map.learned.veto_margin`, the component can never be this IR type, or (for a reject rule) the
     engine's calibrated confidence in its match is at least `map.learned.veto_score`. Strong engine
     evidence always stands; the decision is recorded in `node.meta["learned_rule"]`.
4. **Evidence and gates.** Candidate rules are evaluated on the user corpus before / after: *item
   accuracy* (does the model now agree with each correction, `check_item`) and the structural composite
   with unverified component labels left out (`composite_unlabelled`: geometry, text and paint must not
   regress). Then the **global bench gate** (`knowledge/baseline.json`, the rules applied in every bench
   worker). Accepted only if every gate passes (`--no-gate` skips the bench for local experiments and is
   recorded as skipped). The previous rules are snapshotted; `dt feedback revert <ledger id>` restores them.
   A rule set that fails the gates is not written, and its newly active rules are remembered in
   `$DT_HOME/feedback/rejected_rules.json`: they stay off until their evidence (answer, support, against)
   changes, so one bad rule does not block later good ones. When only the evidence counts of inactive rules
   move (nothing the matcher does changes), the rules file is updated without gates or a ledger entry.
5. **Ledger.** One entry per change in `$DT_HOME/ledger.jsonl` (`kind: rule`, `scope: local`,
   `source: {type: feedback, ids: [bundle ids]}`, `change.rules = {added, removed, changed: {id: {field: [old, new]}}}`).

`dt feedback eval` re-runs every feedback case with the current model and appends to
`$DT_HOME/feedback/history.jsonl` (`ts`, `model`, item accuracy by kind, composite, per case), so a consumer
sees accuracy on *their* use cases across model versions.

## 4. Promotion to the shared model

Consumer: `dt feedback export -o bundle.zip [--redact-text] [--no-screenshots] [--ids ...]`
(`rsdesign.feedback-export/1`). Consent is per bundle and flags only make an export more private:

* text ships only with `consent.share_text` and without `--redact-text`; otherwise every character is
  replaced by one of the same class (`Inbox 12, Q3` -> `Xxxxx 00, X0`; spaces and punctuation kept), so
  boxes, line lengths and text metrics survive; notes, node names, component props and raster crops are
  dropped; local paths are always stripped;
* the screenshot ships only with `consent.share_screenshot` **and** shared text (a screenshot shows its
  text) and without `--no-screenshots`.

Maintainer: `dt feedback import bundle.zip [--promote]`:

* only expected archive members are read (no path traversal); bundles are schema-validated;
* bundles whose consent allows it become `fixtures/scenarios/user_reported/<id>.png` + `.gt.json` +
  `.feedback.json` + `manifest.json` (a bench-compatible scenario family);
* every component answer becomes an observation in `knowledge/feedback_candidates.json` (signatures and
  answers only); candidate *global* rules need `feedback.global_min_support` (3) distinct screenshots;
* `--promote` runs the global bench gate (and the user_reported seeds before / after when present) on the
  candidate rule set; only if every gate passes is `knowledge/learned_rules.json` written. Promotion is
  never ungated. Either way a `feedback_promotion` entry (scope `global`) is appended to `knowledge/ledger.jsonl`.

## 5. The shared learning ledger

Written by the scenario harness and the feedback loop alike (`dt/feedback/ledger.py` validates it):

```json
{"id": "<kind>-<short hash>", "ts": "<iso8601>", "kind": "params|rule|scenario_baseline|feedback_promotion|revert",
 "scope": "global|local", "source": {"type": "scenario|feedback|manual", "ids": [...]},
 "change": {...exact diff...}, "evidence": {"before": {...}, "after": {...}, "gates": {"bench": true, ...}},
 "accepted": true, "reverts": null}
```

Global entries live in `knowledge/ledger.jsonl`, local ones in `$DT_HOME/ledger.jsonl`.

## 6. Commands

```
dt feedback add <run_dir> corrections.json      dt feedback from-figma <run_dir> figma_corrections.json
dt feedback review <run_dir>                    dt feedback status | list
dt feedback learn [--no-gate] [--workers N]     dt feedback eval
dt feedback export -o bundle.zip [--redact-text] [--no-screenshots]
dt feedback import bundle.zip [--promote]       dt feedback revert <ledger id>      dt feedback forget <bundle id>
```

## Known limits

* Rules learn component / variant / not-a-component judgements. Text, icon, geometry, colour, missing,
  extra and raster judgements feed the user corpus and item accuracy (what the self-improvement loop and
  `dt improve` can act on) but do not yet become automatic rules.
* The user corpus gt is "the model's output plus the user's corrections": uncorrected nodes are the
  model's own (unverified) answer, which is why the user gate scores labels only through the items.
