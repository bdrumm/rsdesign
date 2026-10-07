"""Font identification: text rendered in Google Sans / Google Sans Text / Roboto at known sizes and
weights must be recovered from pixels, and the re-render must match the target."""
import pytest

from dt.ir import Box, Color, Document, Node, TextStyle
from dt.perceive import perceive
from dt.render.screenshot import render_doc
from dt.validate.fidelity import delta_e2000_map

ALL_SPECS = [("Google Sans", 28, 500, "#1f1f1f", "Gmail is email that's intuitive"),
         ("Roboto", 16, 400, "#444746", "Supporting text in Roboto regular"),
         ("Google Sans Text", 14, 400, "#444746", "Body copy set in Google Sans Text"),
         ("Google Sans", 22, 400, "#0b57d0", "Get started with Workspace"),
         ("Roboto", 14, 500, "#1d1b20", "Label large medium"),
         ("Google Sans", 16, 700, "#1f1f1f", "Bold Google Sans heading")]

# Google Sans Text is local-only (not in the public Google Fonts catalogue, not distributed): a clone without it
# can neither render nor identify it, so its lines are only part of the test where it is installed
from dt.perceive.fonts import _family_installed  # noqa: E402

SPECS = [s for s in ALL_SPECS if _family_installed(s[0])]


@pytest.fixture(scope="module")
def page():
    doc = Document.blank(520, 330, "#ffffff")
    y = 16
    for fam, sz, w, col, txt in SPECS:
        lh = round(sz * 1.35)
        doc.root.children.append(Node(type="text", text=txt, box=Box(20, y, 480, lh),
                                      text_style=TextStyle(family=fam, size=sz, weight=w, color=Color.from_hex(col), line_height=lh)))
        y += lh + 18
    return render_doc(doc)


def test_families_sizes_weights_recovered(page):
    d = perceive(page)
    got = {n.text.strip(): n.text_style for n in d.texts()}
    for fam, sz, w, _, txt in SPECS:
        ts = got[txt]
        assert ts.family == fam, (txt, ts.family)
        assert abs(ts.size - sz) <= 0.5, (txt, ts.size)
        assert ts.weight == w, (txt, ts.weight)
    assert d.meta["perceive"]["fonts"]["texts"] == len(SPECS)


def test_rerender_matches_target(page):
    d = perceive(page)
    assert float(delta_e2000_map(page, render_doc(d)).mean()) < 0.2


# ---- adversarial cases (skeptic icons-fonts): tracking, italic, fonts outside the candidate list, page limits
TRACKED = [("Roboto", 14, 400, 0.5, False, "Label with half pixel tracking applied"),
           ("Roboto", 12, 500, 0.5, False, "Label small medium tracking"),
           ("Roboto", 14, 400, 0.25, False, "Body medium with quarter pixel tracking"),
           ("Roboto", 16, 400, 0.0, True, "Italic emphasis in Roboto regular"),
           ("Google Sans", 16, 400, 0.0, True, "Synthetic oblique Google Sans text"),
           # untracked, but the scaled-ink placement lands ~1px off: was matched as Google Sans 11.5/500 at ΔE 4.7
           ("Google Sans", 12, 400, 0.0, False, "Line 4 of sixty test strings abc")]


@pytest.fixture(scope="module")
def tracked_page():
    doc = Document.blank(520, 340, "#ffffff")
    y = 16
    for fam, sz, w, ls, it, txt in TRACKED:
        lh = round(sz * 1.35)
        doc.root.children.append(Node(type="text", text=txt, box=Box(20, y, 480, lh),
                                      text_style=TextStyle(family=fam, size=sz, weight=w, color=Color.from_hex("#1f1f1f"), line_height=lh,
                                                           letter_spacing=ls, italic=it)))
        y += lh + 22
    return render_doc(doc)


def test_tracking_and_italic_recovered(tracked_page):
    """M3 label/body roles carry 0.1-0.5px tracking; a tracked line used to be matched by a larger and/or
    bolder untracked one (Roboto 14 ls 0.5 -> Roboto 15 w500), and italics were never tried."""
    d = perceive(tracked_page)
    got = {n.text.strip(): n for n in d.texts()}
    for fam, sz, w, ls, it, txt in TRACKED:
        ts = got[txt].text_style
        assert (ts.family, ts.weight, ts.italic) == (fam, w, it), (txt, ts.family, ts.weight, ts.italic)
        assert abs(ts.size - sz) <= 0.5, (txt, ts.size)
        assert abs(ts.letter_spacing - ls) <= 0.05, (txt, ts.letter_spacing)
        assert got[txt].meta["font_de"] < 0.5, (txt, got[txt].meta["font_de"])


def test_candidate_pages_split_below_screenshot_limit(page):
    """Very tall batch pages are fragile: Chrome intermittently fails 'Unable to capture screenshot' (seen on
    about_google.png, where the whole font pass then failed and every text kept the Roboto guess; a probe
    failed at 17000 px but not at 20000 px) and PIL refuses images > ~179M px. A narrow batch page (one
    ~500px cell per row) makes this 6-line page ~30k px tall: the cells must be split over several pages
    and give the same identification."""
    from dt.params import P
    P.set("perceive.fonts.page_width", 600)
    try:
        d = perceive(page)
    finally:
        P.reset("perceive.fonts.page_width")
    assert "error" not in d.meta["perceive"]["fonts"], d.meta["perceive"]["fonts"]
    got = {n.text.strip(): n.text_style for n in d.texts()}
    for fam, sz, w, _, txt in SPECS:
        assert (got[txt].family, got[txt].weight) == (fam, w) and abs(got[txt].size - sz) <= 0.5, txt


def test_foreign_fonts_never_made_worse():
    """Georgia / Courier are not candidates: identification may only lower the per-line ΔE, never raise it."""
    from dt.params import P
    from dt.render.html import _font_face_css
    from dt.render.screenshot import html_to_png
    lines = [("font-family:Georgia,serif;font-size:18px", "The quick brown fox jumps over"),
             ("font-family:'Courier New',monospace;font-size:14px", "monospace code_sample(x) = 42;")]
    divs = "".join(f'<div style="position:absolute;left:20px;top:{16 + 44 * i}px;white-space:pre;color:#1f1f1f;{css}">{t}</div>'
                   for i, (css, t) in enumerate(lines))
    img = html_to_png(f"<!doctype html><meta charset=utf-8><style>{_font_face_css()}</style><body style='margin:0;background:#fff'>{divs}</body>", 480, 110)
    P.set("perceive.fonts.enabled", False)
    try:
        base = perceive(img)
    finally:
        P.reset("perceive.fonts.enabled")
    d = perceive(img)
    for n in d.texts():
        assert n.meta["font_de"] <= n.meta["font_de_before"] + 1e-6, (n.text, n.meta)
    full_before = float(delta_e2000_map(img, render_doc(base)).mean())
    full_after = float(delta_e2000_map(img, render_doc(d)).mean())
    assert full_after <= full_before + 0.02, (full_before, full_after)
