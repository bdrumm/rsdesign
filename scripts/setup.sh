#!/usr/bin/env bash
# One-shot, idempotent setup for a fresh clone (macOS or Linux). Safe to re-run: every step checks
# before it acts. Ends with `python -m dt.cli doctor`, which verifies the result and prints fixes.
#
#   bash scripts/setup.sh            # venv + Python deps + node deps + browser + reference page
#   DT_SETUP_NO_BROWSER=1 bash scripts/setup.sh   # never download Playwright's Chromium
#   PYTHON=python3.13 bash scripts/setup.sh       # interpreter for the venv (default: uv's choice / python3)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OS="$(uname -s)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
say() { printf '\n==> %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- 1. Python venv + package
if [ ! -x "$PY" ]; then
  say "creating .venv"
  if have uv; then
    uv venv ${PYTHON:+--python "$PYTHON"} "$VENV"
  else
    "${PYTHON:-python3}" -m venv "$VENV"
  fi
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), f"Python {sys.version.split()[0]} < 3.12: re-run with PYTHON=python3.12"' \
  || { echo "delete .venv and re-run with PYTHON=python3.12 (or newer)"; exit 1; }

if have uv; then PIP=(uv pip install --python "$PY"); else PIP=("$PY" -m pip install); fi
EXTRA=""
if [ "$OS" != "Darwin" ]; then EXTRA="[linux]"; fi   # RapidOCR: macOS uses the built-in Vision framework
say "installing the package (editable) ${EXTRA:+with extra $EXTRA}"
"${PIP[@]}" -e ".${EXTRA}"
if [ "$OS" != "Darwin" ] && "$PY" -c 'import importlib.metadata as m; m.version("opencv-python")' >/dev/null 2>&1; then
  # rapidocr pulls the GUI opencv build, which shares cv2/ with the headless one and needs libGL:
  # reinstalling the headless wheel last makes cv2 importable on servers without X/GL libraries
  say "making opencv-python-headless own cv2/ (no libGL needed)"
  "${PIP[@]}" --reinstall opencv-python-headless
fi

# ---------------------------------------------------------------- 2. Node deps + Material Web bundle
if have npm; then
  if [ ! -d node_modules/@material/web ] || [ ! -d node_modules/esbuild ]; then
    say "installing node modules"
    if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi
  fi
  if [ ! -s fixtures/mwc/mwc.bundle.js ]; then
    say "building fixtures/mwc/mwc.bundle.js"
    npm run --silent build:mwc
  fi
else
  echo "warning: npm not found; install Node >= 18 for corpus builds and the Figma plugin tests"
fi

# ---------------------------------------------------------------- 3. Browser for rendering
if "$PY" -c 'from dt.render.screenshot import html_to_png as h; h("<b>ok</b>", 40, 20)' >/dev/null 2>&1; then
  echo "renderer: ok"
elif [ -z "${DT_SETUP_NO_BROWSER:-}" ]; then
  say "no usable Chrome: installing Playwright's Chromium"
  "$PY" -m playwright install chromium
  if [ "$OS" = "Linux" ]; then
    echo "if the browser still fails to start, install its system libraries: sudo $PY -m playwright install-deps chromium"
  fi
fi

# ---------------------------------------------------------------- 4. Reference page used by the tests (out/ is git-ignored)
"$PY" scripts/make_reference_page.py || echo "warning: could not build out/mwc_test.* (see doctor below)"

# ---------------------------------------------------------------- 5. Verify
say "doctor"
"$PY" -m dt.cli doctor
