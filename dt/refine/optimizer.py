"""Optimizer: greedy, render-verified acceptance of critic hypotheses.

    doc, history = refine(doc, target_rgb)

Loop:  render -> evaluate -> critique -> try hypotheses in order of expected gain.
A hypothesis (or a batch of mutually independent ones) is applied to a copy, rendered once and
kept only if the total loss drops by more than ``refine.opt.eps``. A rejected batch is bisected
so a single bad proposal cannot veto good ones. The loop stops when the loss is below
``refine.target_loss``, when ``patience`` consecutive iterations accept nothing, when
``max_iters`` is reached, or when the time budget runs out. The loss is monotonically
non-increasing by construction and the input document is never mutated.

``history`` is a list of ``{iter, kind, node_id, params, before, after, accepted}`` records,
one per tried hypothesis (accepted moves included), for reports and for tuning the critic.
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Optional

import numpy as np

from dt.compare.loss import LossReport, evaluate, text_term_from_lines
from dt.ir import Document, Node
from dt.params import P, register
from dt.refine.critic import Hypothesis, critique

register("refine.opt.max_iters", 8, "Maximum critique/accept iterations.", (1, 50))
register("refine.opt.patience", 2, "Stop after this many consecutive iterations without an accepted move.", (1, 10))
register("refine.opt.eps", 1e-6, "A move must lower the total loss by more than this to be accepted.", (0.0, 0.01))
register("refine.target_loss", 1e-5, "Stop once the pixel part of the loss is below this.", (0.0, 0.1))
register("refine.opt.max_hyps", 48, "Try at most this many hypotheses per iteration.", (4, 256))
register("refine.opt.batch_size", 6, "Apply up to this many independent hypotheses per render.", (1, 32))
register("refine.opt.batch_margin", 2, "Regions closer than this (px) are not independent.", (0, 16))
register("refine.opt.use_ocr", True, "OCR the target once and add the text term to the loss.")
register("refine.opt.incremental", True, "score trials by re-comparing only the pixels that differ from the current render (bit-identical loss; False = full evaluate per trial)")

Logger = Optional[Callable[[str], None]]


# --------------------------------------------------------------------------- loss
def target_text_lines(target_rgb: np.ndarray) -> Optional[list[str]]:
    """Normalised OCR lines of the target (cached by callers), or None when OCR is unavailable."""
    try:
        from dt.compare.structural import normalize_text
        from dt.perceive.ocr import ocr_lines
    except Exception:
        return None
    try:
        min_conf = float(P["compare.loss.ocr_min_conf"])
        return [normalize_text(l.text) for l in ocr_lines(target_rgb) if normalize_text(l.text) and l.conf >= min_conf]
    except Exception:
        return None


def _doc_texts(doc: Document) -> tuple[str, ...]:
    """Texts the render shows as text: text nodes plus the wordmarks kept as crops (``meta.alt_text``)."""
    return tuple([n.text for n in doc.texts()] +
                 [n.meta["alt_text"] for n in doc.walk() if n.type == "image" and isinstance(n.meta.get("alt_text"), str)])


_TEXT_TERM_CACHE: dict[tuple, Optional[float]] = {}


def _text_term(target_lines: list[str], doc: Document) -> Optional[float]:
    """``text_term_from_lines`` over :func:`_doc_texts`, memoised on (lines, texts): most trials only move
    geometry, and the Levenshtein sweep over every (line, text) pair dominated the evaluation."""
    texts = _doc_texts(doc)
    key = (tuple(target_lines), texts)
    if key not in _TEXT_TERM_CACHE:
        if len(_TEXT_TERM_CACHE) > 4096:
            _TEXT_TERM_CACHE.clear()
        proxy = Document.blank(1, 1)
        proxy.root.children = [Node(type="text", text=t) for t in texts]
        _TEXT_TERM_CACHE[key] = text_term_from_lines(target_lines, proxy)
    return _TEXT_TERM_CACHE[key]


def loss_of(doc: Document, target_rgb: np.ndarray, rendered_rgb: Optional[np.ndarray] = None,
            target_lines: Optional[list[str]] = None) -> tuple[float, LossReport]:
    """Total loss of ``doc`` against ``target_rgb`` (renders when no render is given).

    Identical to ``dt.compare.evaluate`` except that the OCR text term is computed from the
    pre-computed ``target_lines`` (so the target is OCR'd once per refine run, not per trial).
    Returns ``(total, report)``; ``report.total`` is updated to include the text term.
    """
    rep = evaluate(doc, target_rgb, rendered_rgb, use_ocr=False, keep_diff=True)
    total = rep.total
    if target_lines:
        term = _text_term(target_lines, doc)
        rep.structure["text_term"] = term
        total += float(P["compare.loss.w_text"]) * (term or 0.0)
    rep.total = float(total)
    return rep.total, rep


def _render(doc: Document) -> np.ndarray:
    from dt.render.screenshot import render_doc
    return render_doc(doc)


# --------------------------------------------------------------------------- batching
def _independent(a: Hypothesis, b: Hypothesis, ancestors: dict[str, list[str]], margin: float) -> bool:
    if a.region.expand(margin).intersect(b.region.expand(margin)).area > 0:
        return False
    if a.node_id and b.node_id:
        if a.node_id == b.node_id or a.node_id in ancestors.get(b.node_id, []) or b.node_id in ancestors.get(a.node_id, []):
            return False
    return True


def _batches(hyps: list[Hypothesis], doc: Document) -> list[list[Hypothesis]]:
    """Group hypotheses (kept in gain order) into batches of mutually independent edits."""
    size = int(P["refine.opt.batch_size"])
    margin = float(P["refine.opt.batch_margin"])
    anc: dict[str, list[str]] = {}

    def rec(n, chain):
        anc[n.id] = chain
        for c in n.children:
            rec(c, chain + [n.id])

    rec(doc.root, [])
    pending = list(hyps)
    out: list[list[Hypothesis]] = []
    while pending:
        batch: list[Hypothesis] = []
        rest: list[Hypothesis] = []
        for h in pending:
            if len(batch) < size and all(_independent(h, o, anc, margin) for o in batch):
                batch.append(h)
            else:
                rest.append(h)
        out.append(batch)
        pending = rest
    return out


# --------------------------------------------------------------------------- refine
class _State:
    def __init__(self, doc: Document, target: np.ndarray, lines: Optional[list[str]], log: Logger, deadline: Optional[float]):
        self.doc = doc
        self.target = target
        self.lines = lines
        self.log = log
        self.deadline = deadline
        self.rendered = _render(doc)
        self.loss, self.report = loss_of(doc, target, self.rendered, lines)
        self.history: list[dict] = []
        self.iter = 0
        self.n_renders = 1
        self._partial: set[int] = set()

    def score(self, cand: Document, rendered: np.ndarray) -> tuple[float, LossReport]:
        """``loss_of(cand, target, rendered, lines)``, bit-identical, without re-comparing the whole screen.

        Only pixels where ``rendered`` differs from the current render can change the ΔE map, so the map is
        patched in the row band that holds them (ΔE is per-pixel) and the total is re-reduced from it. Per-node
        errors, residual regions, SSIM and alignment are only read for an accepted state; ``accept``
        fills them in. Falls back to ``loss_of`` when sizes differ.
        """
        d0 = self.report.diff_map
        if not bool(P["refine.opt.incremental"]) or d0 is None or rendered.shape != self.rendered.shape \
                or rendered.shape != self.target.shape or d0.shape != rendered.shape[:2]:
            return loss_of(cand, self.target, rendered, self.lines)
        from dt.compare.pixel import diff_map
        rows = np.flatnonzero(np.any(rendered != self.rendered, axis=(1, 2)))
        d = d0.copy()
        if len(rows):
            # full-width row band: contiguous slices give bit-identical Lab values (narrower column crops
            # take another SIMD path in rgb2lab and drift by ~1e-5 ΔE)
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            d[y0:y1] = diff_map(self.target[y0:y1], rendered[y0:y1])
        mean_de = float(d.mean()) if d.size else 0.0
        frac_bad = float((d > float(P["compare.pixel.bad_de"])).mean()) if d.size else 0.0
        # the expression (and order) of dt.compare.loss.evaluate for equal sizes without OCR, then loss_of's text term
        total = (float(P["compare.loss.w_pixel"]) * min(1.0, mean_de / float(P["compare.loss.de_norm"]))
                 + float(P["compare.loss.w_bad"]) * frac_bad + float(P["compare.loss.w_text"]) * 0.0
                 + float(P["compare.loss.w_size"]) * 0.0)
        structure: dict = {"text_term": None}
        if self.lines:
            term = _text_term(self.lines, cand)
            structure["text_term"] = term
            total += float(P["compare.loss.w_text"]) * (term or 0.0)
        report = LossReport(total=float(total), pixel={"mean_de": mean_de, "frac_bad": frac_bad}, structure=structure,
                            diff_map=d, size=(d.shape[1], d.shape[0]))
        self._partial.add(id(report))
        return report.total, report

    def accept(self, cand: Document, rendered: np.ndarray, loss: float, report: LossReport) -> None:
        """Make ``cand`` the current state; complete a ``score`` report the way ``evaluate`` would."""
        if id(report) in self._partial:
            from dt.compare.loss import per_node_errors
            from dt.compare.pixel import alignment_offset, pixel_metrics, residual_regions
            d = report.diff_map
            pix = pixel_metrics(self.target, rendered, diff=d)
            pix["size_mismatch"], pix["size_term"] = False, 0.0
            report.pixel = pix
            report.residual_regions = residual_regions(d)
            report.per_node = per_node_errors(cand.root, d)
            report.alignment = alignment_offset(self.target, rendered)
        self._partial.clear()  # reports of rejected trials are never read again
        self.doc, self.rendered, self.loss, self.report = cand, rendered, loss, report

    def out_of_time(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def say(self, msg: str) -> None:
        if self.log:
            self.log(msg)

    def record(self, h: Hypothesis, before: float, after: float, accepted: bool, batch: int) -> None:
        self.history.append({
            "iter": self.iter, "kind": h.kind, "node_id": h.node_id,
            "params": {k: v for k, v in h.params.items() if not k.startswith("_")},
            "expected_gain": float(h.expected_gain), "before": float(before), "after": float(after),
            "accepted": bool(accepted), "batch": int(batch),
        })

    def _helpful(self, batch: list[Hypothesis], rep: LossReport) -> list[Hypothesis]:
        """Members whose own region lost error (regions are disjoint, so the change is theirs)."""
        if self.report.diff_map is None or rep.diff_map is None:
            return list(batch)
        out = []
        for h in batch:
            x0, y0, x1, y1 = h.region.expand(float(P["refine.opt.batch_margin"])).as_int()
            x0, y0 = max(0, x0), max(0, y0)
            before = float(self.report.diff_map[y0:y1, x0:x1].sum())
            after = float(rep.diff_map[y0:y1, x0:x1].sum())
            if after < before - 1e-6:
                out.append(h)
        return out

    def try_alternatives(self, h: Hypothesis) -> bool:
        """Score ``h`` and each of ``h.alternatives`` from the current state; accept the best if it lowers the loss."""
        if self.out_of_time():
            return False
        scored = []
        for opt in [h] + list(h.alternatives):
            cand = opt.apply(self.doc)
            rendered = _render(cand)
            self.n_renders += 1
            new, rep = self.score(cand, rendered)
            scored.append((new, opt, cand, rendered, rep))
        scored.sort(key=lambda t: t[0])
        best_loss, best, cand, rendered, rep = scored[0]
        for new, opt, *_ in scored[1:]:
            self.record(opt, self.loss, new, False, 1)
        if best_loss < self.loss - float(P["refine.opt.eps"]):
            self.record(best, self.loss, best_loss, True, 1)
            self.say(f"  accept {best.describe()}  loss {self.loss:.5f} -> {best_loss:.5f} (best of {len(scored)})")
            self.accept(cand, rendered, best_loss, rep)
            return True
        self.record(best, self.loss, best_loss, False, 1)
        return False

    def try_batch(self, batch: list[Hypothesis]) -> bool:
        """Apply ``batch`` to a copy, render, keep if the loss drops; else bisect. Returns True if anything was accepted."""
        if not batch or self.out_of_time():
            return False
        cand = self.doc
        for h in batch:
            cand = h.apply(cand)
        rendered = _render(cand)
        self.n_renders += 1
        new, rep = self.score(cand, rendered)
        if new < self.loss - float(P["refine.opt.eps"]):
            if len(batch) > 1:
                helpful = self._helpful(batch, rep)
                if len(helpful) < len(batch):  # drop members that changed nothing in their own region
                    cand2 = self.doc
                    for h in helpful:
                        cand2 = h.apply(cand2)
                    rendered2 = _render(cand2)
                    self.n_renders += 1
                    new2, rep2 = self.score(cand2, rendered2)
                    if new2 <= new + float(P["refine.opt.eps"]):
                        for h in batch:
                            if h not in helpful:
                                self.record(h, self.loss, new2, False, len(batch))
                        batch, cand, rendered, new, rep = helpful, cand2, rendered2, new2, rep2
            for h in batch:
                self.record(h, self.loss, new, True, len(batch))
                self.say(f"  accept {h.describe()}  loss {self.loss:.5f} -> {new:.5f}")
            self.accept(cand, rendered, new, rep)
            return True
        if len(batch) == 1:
            self.record(batch[0], self.loss, new, False, 1)
            return False
        mid = len(batch) // 2
        a = self.try_batch(batch[:mid])
        b = self.try_batch(batch[mid:])
        return a or b


def refine(doc: Document, target_rgb: np.ndarray, max_iters: Optional[int] = None, patience: Optional[int] = None,
           time_budget_s: Optional[float] = None, log: Logger = None) -> tuple[Document, list[dict]]:
    """Adversarial refinement of ``doc`` against the ``target_rgb`` screenshot.

    Returns ``(refined_doc, history)``. ``refined_doc`` is a new document (the input is not
    mutated) whose loss is <= the input's. ``history`` lists every tried hypothesis with
    ``accepted`` flags; the last record is a ``{"kind": "stop", ...}`` summary with the reason,
    the final loss and the number of renders. Safe on empty and single-node documents.
    """
    max_iters = int(P["refine.opt.max_iters"] if max_iters is None else max_iters)
    patience = int(P["refine.opt.patience"] if patience is None else patience)
    target_rgb = np.ascontiguousarray(np.asarray(target_rgb)[..., :3], dtype=np.uint8)
    deadline = time.monotonic() + float(time_budget_s) if time_budget_s else None
    lines = target_text_lines(target_rgb) if bool(P["refine.opt.use_ocr"]) else None
    st = _State(copy.deepcopy(doc), target_rgb, lines, log, deadline)
    start_loss = st.loss
    st.say(f"refine: start loss {st.loss:.5f} ({doc.root.count()} nodes)")
    stale = 0
    reason = "max_iters"
    for it in range(1, max_iters + 1):
        st.iter = it
        if st.report.total - float(P["compare.loss.w_text"]) * (st.report.structure.get("text_term") or 0.0) < float(P["refine.target_loss"]):
            reason = "target_loss"
            break
        if st.out_of_time():
            reason = "time_budget"
            break
        hyps = critique(st.doc, st.target, st.rendered, st.report)[: int(P["refine.opt.max_hyps"])]
        st.say(f"iter {it}: loss {st.loss:.5f}, {len(hyps)} hypotheses")
        if not hyps:
            reason = "no_hypotheses"
            break
        accepted_any = False
        multi = [h for h in hyps if h.alternatives]
        hyps = [h for h in hyps if not h.alternatives]
        for h in multi:  # exclusive alternatives first: they are the most specific edits
            if st.out_of_time():
                break
            if st.try_alternatives(h):
                accepted_any = True
        for batch in _batches(hyps, st.doc):
            if st.out_of_time():
                break
            if st.try_batch(batch):
                accepted_any = True
        if st.out_of_time() and not accepted_any:
            reason = "time_budget"
            break
        stale = 0 if accepted_any else stale + 1
        if stale >= patience:
            reason = "patience"
            break
    st.history.append({"iter": st.iter, "kind": "stop", "node_id": None, "params": {"reason": reason, "renders": st.n_renders},
                       "before": float(start_loss), "after": float(st.loss), "accepted": st.loss < start_loss, "batch": 0})
    st.say(f"refine: stop ({reason}) loss {start_loss:.5f} -> {st.loss:.5f} after {st.n_renders} renders")
    return st.doc, st.history


def accepted_moves(history: list[dict]) -> list[dict]:
    """Only the accepted edit records of a ``refine`` history (the ``stop`` summary excluded)."""
    return [h for h in history if h.get("accepted") and h.get("kind") != "stop"]
