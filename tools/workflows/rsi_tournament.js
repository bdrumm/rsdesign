export const meta = {
  name: 'rsi-tournament',
  description: 'Improvement round: per failure family, competing fixers in worktrees -> independent judge on holdout + gates -> serialized merge queue -> immediate skeptic',
  whenToUse: 'A round of fixes for several failure families with disjoint stage files, once the scenario harness (dt scenario / dt train) exists',
  phases: [
    { title: 'Compete', detail: '2-3 contestants per family, different approaches, isolated worktrees' },
    { title: 'Judge', detail: 're-measure every contestant on the family holdout, the scenario gate and the bench gate' },
    { title: 'Merge', detail: 'one merge at a time into main; gates after each; back out on failure' },
    { title: 'Attack', detail: 'a skeptic per merged change, started as soon as it lands' },
  ],
}

// args = {
//   root: '/abs/path/to/repo',                 // main checkout (must be clean; nothing else may work in it during the run)
//   python: '/abs/path/.venv/bin/python',
//   families: [{ name, failure, files: [paths the contestants may edit], approaches: ['approach A ...', 'approach B ...'] }],
// }
const A = args || {}
const ROOT = A.root
const PY = A.python || `${ROOT}/.venv/bin/python`
if (!ROOT || !Array.isArray(A.families) || !A.families.length) throw new Error('args.root and args.families are required')

const RESULT = {
  type: 'object',
  properties: {
    family: { type: 'string' }, approach: { type: 'string' }, worktree: { type: 'string' }, branch: { type: 'string' }, commit: { type: 'string' },
    holdout_before: { type: 'number' }, holdout_after: { type: 'number' }, bench_gate: { type: 'string' }, scenario_gate: { type: 'string' },
    tests_failed: { type: 'integer' }, summary: { type: 'string' },
  },
  required: ['family', 'approach', 'worktree', 'branch', 'commit', 'holdout_before', 'holdout_after', 'bench_gate', 'scenario_gate', 'tests_failed', 'summary'],
}
const JUDGE = {
  type: 'object',
  properties: {
    family: { type: 'string' }, winner_branch: { type: 'string' }, winner_worktree: { type: 'string' },
    table: { type: 'string' }, reason: { type: 'string' }, ideas_from_losers: { type: 'array', items: { type: 'string' } },
  },
  required: ['family', 'winner_branch', 'table', 'reason', 'ideas_from_losers'],
}
const MERGE = {
  type: 'object',
  properties: { family: { type: 'string' }, merged: { type: 'boolean' }, main_commit: { type: 'string' }, gates: { type: 'string' }, notes: { type: 'string' } },
  required: ['family', 'merged', 'gates'],
}
const RULES = `Repository ${ROOT}. Python: ${PY} (dt imports from your cwd). Read AGENTS.md, docs/SCENARIOS.md, docs/ADVERSARIAL.md, knowledge/failures.jsonl.
Rules: thresholds in dt.params; real-renderer tests; never weaken a test to pass; commit with explicit paths (never node_modules, out/); do NOT publish artifacts; if a shared evaluation service is running (dt service status), export DT_EVAL_SERVICE so you do not start extra browsers. Final output: the structured result only.`

// serialized merge queue: each merge waits for the previous one, but families do not wait for each other
let queue = Promise.resolve()
const enqueue = fn => { const p = queue.then(fn, fn); queue = p.catch(() => null); return p }

const results = await pipeline(
  A.families,
  // Compete: contestants per family, in parallel
  fam => parallel(fam.approaches.map((approach, i) => () => agent(
    `${RULES}\nYou work in an ISOLATED worktree (your cwd). If node_modules is missing: ln -s ${ROOT}/node_modules node_modules.\n` +
    `FAILURE FAMILY '${fam.name}': ${fam.failure}\nYOUR APPROACH (contestant ${i + 1}): ${approach}\nYOU MAY EDIT ONLY: ${fam.files.join(', ')} plus new tests and dt/scenarios/families/${fam.name}*.py.\n` +
    `Protocol: (1) \`${PY} -m dt.cli scenario run ${fam.name} --json\` on TRAIN seeds for the baseline; (2) fix the stage with your approach; ` +
    `optionally \`${PY} -m dt.cli train --family ${fam.name}\` for its parameters; (3) report HOLDOUT pass rate before/after, ` +
    `\`${PY} -m dt.cli bench --workers 2 --gate --scenarios --quiet\` and the full test suite. Commit; report worktree, branch, commit.`,
    { label: `compete:${fam.name}:${i + 1}`, phase: 'Compete', schema: RESULT, effort: 'high', isolation: 'worktree' }))),
  // Judge: independent re-measurement, pick a winner (or none)
  (contestants, fam) => {
    const live = (contestants || []).filter(Boolean)
    if (!live.length) return null
    return agent(`${RULES}\nYou are an INDEPENDENT JUDGE for failure family '${fam.name}'. Contestants: ${JSON.stringify(live)}\n` +
      `For EACH contestant: cd into its worktree and re-run yourself (do not trust reported numbers): the family on HOLDOUT seeds, ` +
      `\`${PY} -m dt.cli bench --workers 2 --gate --scenarios --quiet\`, the generalisation report (\`${PY} -m dt.cli train --report-generalisation\` if available), and the tests touching the edited files. ` +
      `Winner = biggest holdout gain that passes every gate and does not lower any other family; ties go to the simpler diff. If none passes, winner_branch = "". Collect ideas worth keeping from the losers.`,
      { label: `judge:${fam.name}`, phase: 'Judge', schema: JUDGE, effort: 'high' })
  },
  // Merge: serialized queue into main
  (verdict, fam) => (!verdict || !verdict.winner_branch) ? { family: fam.name, merged: false, gates: 'no winner' } : enqueue(() => agent(
    `${RULES}\nYou are the MERGE QUEUE. Work in the MAIN checkout ${ROOT} (cwd). Merge branch ${verdict.winner_branch} (family '${fam.name}') into main: ` +
    `\`git merge --no-edit ${verdict.winner_branch}\`; resolve conflicts conservatively (keep both sides' intent); run the full test suite, ` +
    `\`${PY} -m dt.cli bench --workers 3 --gate --scenarios --quiet\`. If anything fails and a small fix is not obvious, back the merge out ` +
    `(\`git merge --abort\` or \`git reset --hard ORIG_HEAD\` on the merge you just made — only that) and report merged=false. ` +
    `On success append a knowledge/rounds.jsonl-style note to knowledge/ledger.jsonl if the ledger exists, commit, and report the main commit.`,
    { label: `merge:${fam.name}`, phase: 'Merge', schema: MERGE, effort: 'high' })),
  // Attack: skeptic per merged change, as soon as it lands
  (merge, fam) => (!merge || !merge.merged) ? merge : agent(
    `${RULES}\nSKEPTIC for the change just merged into main (${merge.main_commit}) for family '${fam.name}'. Work in an isolated worktree branched from main. ` +
    `Try to break it: new adversarial cases from the family generator outside its prior ranges, the real pages in ${ROOT}/fixtures/screens, metamorphic relations (dt.adversary.metamorphic if present). ` +
    `Fix clear bugs with failing-first tests on your branch and report; the lead merges your branch.`,
    { label: `attack:${fam.name}`, phase: 'Attack', effort: 'high', isolation: 'worktree' }).then(attack => ({ ...merge, attack })),
)
return results