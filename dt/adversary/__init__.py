"""Adversarial comparison research track: shared taxonomy, perturbation operators and benchmark.

* ``taxonomy``  -- ``Finding`` + the versioned type taxonomy (what an adversary may report)
* ``perturb``   -- typed, magnitude-controlled IR perturbations with exact ground truth, and noise
* ``benchmark`` -- deterministic CALIBRATION/EVALUATION benchmark, scorer, floor adversaries

An adversary is ``fn(target_rgb, candidate_rgb, candidate_ir | None) -> list[Finding]``; score it with
``dt.adversary.benchmark.score(fn, build())``. Submodules import lazily (the renderer is heavy).
"""
from dt.adversary.taxonomy import ERROR_TYPES, NOISE_TYPES, PARENTS, TAXONOMY_VERSION, TYPES, Finding  # noqa: F401

__all__ = ["Finding", "TYPES", "PARENTS", "ERROR_TYPES", "NOISE_TYPES", "TAXONOMY_VERSION"]
