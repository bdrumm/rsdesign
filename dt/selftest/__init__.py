"""Self-test corpora (module E).

Ground-truth corpora used to benchmark and tune the perception / mapping stages:

* :mod:`dt.selftest.mwc_corpus` -- real Material Web pages rendered with the real renderer;
  the IR is extracted from the DOM (light + shadow DOM), so every node has exact geometry,
  paint, text style and a ``ComponentRef`` for each ``md-*`` element.
* :mod:`dt.selftest.synth` -- pure-IR random Material-3 screens rendered with ``dt.render``;
  the ground truth *is* the IR.
* :mod:`dt.selftest.grammar` -- shared vocabulary: M3 baseline tokens, word lists, CSS.

Both generators are CLI-free: ``generate(n, out_dir, seed)`` writes ``<id>.png``,
``<id>.gt.json`` (and ``<id>.html`` for mwc) plus a ``manifest.json`` and returns the manifest.
Shipped corpora live under ``fixtures/corpus/{mwc,synth}`` (n=12, seed=1).
"""
from __future__ import annotations

import os

CORPUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "corpus"))


def corpus_dir(kind: str) -> str:
    """Path of a shipped corpus (``"mwc"`` or ``"synth"``)."""
    return os.path.join(CORPUS_DIR, kind)


__all__ = ["CORPUS_DIR", "corpus_dir"]
