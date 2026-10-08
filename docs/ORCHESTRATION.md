# Running improvement rounds with many agents

The harness improves fastest when several fixes run at once. Two things decide whether that helps or
hurts: whether parallel work collides, and whether it saturates the machine. This is the protocol that
came out of the first five rounds.

## The round shape: compete, judge, merge, attack

`tools/workflows/rsi_tournament.js` (also runnable by name as a saved workflow `rsi-tournament`):

1. **Compete.** Each failure family gets two or three contestants with different approaches, each in
   its own git worktree, each allowed to edit only that family's stage files.
2. **Judge.** An independent agent re-measures every contestant itself: the family's HOLDOUT seeds,
   the scenario gate, the bench gate and the generalisation report. The winner is the biggest holdout
   gain that passes every gate without lowering another family. Ideas from losers are recorded.
3. **Merge queue.** Winners merge into `main` one at a time as soon as they are judged; gates run after
   each merge and a failing merge is backed out. Families never wait for each other at a barrier.
4. **Attack.** A skeptic starts on each merged change immediately, in its own worktree.

```js
// args for the workflow
{ root: "/abs/repo", python: "/abs/repo/.venv/bin/python",
  families: [{ name: "page_tint", failure: "...", files: ["dt/refine/critic.py"],
               approaches: ["recolour by error mass", "background from uncovered pixels"] }] }
```

Families in one round must have disjoint `files`. Contestants of the same family may overlap: only the
winner merges.

## Not saturating the machine

* Start one shared evaluation service per machine (`dt serve`) and export `DT_EVAL_SERVICE` in every
  agent: browsers become a fixed pool instead of one or more per agent (we measured ~100 Chrome
  processes and full swap on a 24 GB machine without it).
* Keep per-agent parallelism small (`--workers 2`); the pool provides the parallelism.
* Per-pixel maths in search loops runs on the GPU through `dt.accel` (MLX on Apple Silicon); final
  verdicts stay on the CPU reference in `dt.validate`.
* `render_doc` keeps one long-lived page per thread with fonts loaded (`render.persistent_page`).

## Lessons that are now rules

* Workers never edit the main checkout; only the merge queue does.
* `knowledge/baseline.json` is regenerated after merges, never hand-merged.
* Subagents never publish pages or artifacts; the lead owns the report.
* A metric win that adds metamorphic violations, rasterises text, or lowers another family is rejected.
