#!/usr/bin/env bash
# Publish the exact file tree of a local ref onto origin/main as ONE new commit. Local history (which may
# contain absolute paths, local-only fonts or intermediate states) never leaves the machine.
#   bash tools/publish.sh <ref> "<commit message>"
set -euo pipefail
REF="${1:?ref to publish (e.g. main)}"
MSG="${2:?commit message}"
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin main
fail=0
HOME_PREFIX="$(dirname "$HOME")/"   # e.g. /Users/ or /home/
ME="$(id -un)"
SCAN=(-- . ':!*.png' ':!*.woff2' ':!*.npz' ':!*.jpg' ':!tools/publish.sh')
if git grep -l -i -F -e "$HOME_PREFIX" -e "$ME" "$REF" "${SCAN[@]}" | grep -q .; then
  echo "refusing: absolute home paths or the local account name in $REF:"; git grep -l -i -F -e "$HOME_PREFIX" -e "$ME" "$REF" "${SCAN[@]}"; fail=1
fi
if git ls-tree -r --name-only "$REF" | grep -q "gsans_text"; then echo "refusing: Google Sans Text (license unconfirmed) is tracked in $REF"; fail=1; fi
if git ls-tree -r --name-only "$REF" | grep -qE "^fixtures/screens/.*\.png$"; then echo "refusing: captured third-party screenshots are tracked in $REF"; fail=1; fi
if git grep -l -E "ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}" "$REF" -- . | grep -q .; then echo "refusing: something that looks like a secret"; fail=1; fi
[ "$fail" = 0 ] || exit 1
TREE="$(git rev-parse "$REF^{tree}")"
if [ "$TREE" = "$(git rev-parse origin/main^{tree})" ]; then echo "origin/main already has this tree"; exit 0; fi
C="$(git commit-tree "$TREE" -p origin/main -m "$MSG")"
git push origin "$C:refs/heads/main"
git branch -f publish "$C" >/dev/null
echo "published $C ($(git rev-parse --short "$REF") tree) to origin/main"
