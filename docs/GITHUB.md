# GitHub as a resource

The public repository (https://github.com/bdrumm/rsdesign) is more than a mirror: hosted runners give
the harness extra, parallel compute for free on a public repository.

## CI (`.github/workflows/ci.yml`)

Every push to `main` and every pull request runs the full test suite on two images:

| runner | OCR backend | browser | role |
|---|---|---|---|
| `macos-latest` (Apple Silicon) | macOS Vision | Google Chrome | closest to the pixel reference |
| `ubuntu-latest` | RapidOCR | Playwright Chromium | proves the Linux install path |

The benchmark runs on both and is uploaded as an artifact. Pixel numbers depend on the platform (font
rasterisation, browser build), so `knowledge/baseline.json` stays the reference measured on the
maintainer's Apple Silicon machine; CI gates against `knowledge/baseline.ci-<os>.json` once such a
platform baseline has been recorded from a green run.

## Parallel sweeps (`.github/workflows/sweep.yml`)

Parameter search is embarrassingly parallel: each shard runs an independent, seeded search.

```bash
gh workflow run sweep.yml -R bdrumm/rsdesign -f shards=12 -f iters=20 -f include="perceive.seg refine.critic"
gh run list -R bdrumm/rsdesign --workflow sweep.yml --limit 1          # find the run id
gh run download <run-id> -R bdrumm/rsdesign -D out/sweep               # shards + ranked.json
```

Nothing found in the cloud is accepted automatically. The best candidates from `ranked.json` are
re-measured on the reference machine (`dt bench --gate`, plus the scenario gate once it lands) and only
then written to `dt/params.json` with a ledger entry (`knowledge/ledger.jsonl`).

## Publishing (`tools/publish.sh`)

```bash
bash tools/publish.sh main "what changed"
```

publishes the exact file tree of a local branch as one new commit on top of `origin/main`. Local
history never leaves the machine. The script refuses to publish when the tree contains absolute home
paths or the local account name, captured third-party screenshots, the locally installed Google Sans
Text font (license unconfirmed), or anything that looks like a secret.
