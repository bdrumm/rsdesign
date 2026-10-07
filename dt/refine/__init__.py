"""Module F: adversarial refinement (critic proposes, optimizer accepts if the loss drops).

    from dt.refine import refine
    doc2, history = refine(doc, target_rgb)           # render -> evaluate -> critique -> accept loop
    hyps = critique(doc, target_rgb, rendered, report)  # just the proposals

Everything is CV/OCR-driven and verified by the real renderer; thresholds live in
``dt.params`` under ``refine.*``.
"""
from dt.refine.critic import Hypothesis, critique
from dt.refine.optimizer import accepted_moves, loss_of, refine, target_text_lines

__all__ = ["Hypothesis", "critique", "refine", "loss_of", "accepted_moves", "target_text_lines"]
