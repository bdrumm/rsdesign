"""The persistent-page fast path of render_doc is pixel-identical to the fresh-page path and much faster."""
import glob
import os
import time

import numpy as np

from dt.ir import Document
from dt.params import P
from dt.render.html import render_html
from dt.render.screenshot import html_to_png, render_doc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fresh(d):
    """Reference: a fresh page per render (the persistent fast path disabled)."""
    P.set("render.persistent_page", False)
    try:
        return html_to_png(render_html(d), d.width, d.height)
    finally:
        P.reset("render.persistent_page")


def test_identical_to_fresh_page_on_every_corpus_doc():
    docs = [Document.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "fixtures", "corpus", "*", "*.gt.json")))]
    assert len(docs) >= 20
    for d in docs + docs[:3]:  # repeat a few to cover reuse after other documents and sizes
        fast = render_doc(d)
        assert np.array_equal(fast, _fresh(d)), d.meta.get("id")
        assert np.array_equal(html_to_png(render_html(d), d.width, d.height), fast)  # html_to_png fast path too


def test_unmarked_pages_never_take_the_fast_path():
    from dt.render.screenshot import _split_doc_html
    assert _split_doc_html("<html><head><style>x{}</style><script src=a.js></script></head><body>hi</body></html>") is None


def test_faster_than_fresh_page():
    docs = [Document.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "fixtures", "corpus", "synth", "*.gt.json")))[:6]]
    render_doc(docs[0]); _fresh(docs[0])
    t = time.time(); [render_doc(d) for d in docs]; fast = time.time() - t
    t = time.time(); [_fresh(d) for d in docs]; slow = time.time() - t
    assert fast < slow, (fast, slow)


def test_disabled_falls_back():
    d = Document.load(sorted(glob.glob(os.path.join(ROOT, "fixtures", "corpus", "synth", "*.gt.json")))[0])
    P.set("render.persistent_page", False)
    try:
        a = render_doc(d)
    finally:
        P.reset("render.persistent_page")
    assert np.array_equal(a, render_doc(d))


def test_back_to_back_same_size_variants_match_fresh_pages():
    """Regression: consecutive renders of SAME-SIZE documents that differ slightly (a shifted node, edited text)
    used to reuse Chrome's raster tiles and repaint only the invalidated region, whose anti-aliasing differed
    from a fresh page by a few pixels (found by the adversary benchmark's consistency test)."""
    import copy
    base = Document.load(sorted(glob.glob(os.path.join(ROOT, "fixtures", "corpus", "synth", "*.gt.json")))[2])
    variants = [base]
    texts = [n for n in base.walk() if n.type == "text" and n.text]
    for k in range(1, 5):
        v = copy.deepcopy(base)
        t = [n for n in v.walk() if n.type == "text" and n.text][k % max(1, len(texts))]
        t.box = t.box.translate(k, 0)
        t.text = t.text + "!" * k
        variants.append(v)
    fast = [render_doc(v) for v in variants]  # back to back, same viewport size
    for v, img in zip(variants, fast):
        assert np.array_equal(img, _fresh(v))
