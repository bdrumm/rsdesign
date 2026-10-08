"""desktop_list: dense 1150-1440 px inbox-style rows (sender, subject - snippet, time, 8-30 px icons).

Generalises knowledge/failures.jsonl R4 perceive entries on dense desktop lists: "every subject
line lost its ' - ' (24/24 lines), '&' and '/' deleted" (perceive.ocr icon-word split), the
60-120-row grouping blow-up (perceive.group) and the flex re-layout drift of long lists
(perceive.layout), plus R3 "unpainted containers were never produced" (list items).

Real browser HTML (flex rows, bold unread rows, grey snippets, tinted read rows and/or 1 px
dividers, svg checkbox/star/important/attachment icons, light or dark); ground truth from the
DOM. A *segment* is one visual text run: the sender, "subject - snippet", the time.

Success: >= 95% of segments recovered (line-fair, CER <= compare.struct.text_match_cer), no
predicted text node spanning two segments (merged columns or rows), and >= 90% of the rows
found as containers (IoU >= ``scenarios.list.row_iou``).
"""
from __future__ import annotations

from dt.ir import Box, Node
from dt.params import P, register
from dt.scenarios.families._common import NAMES, WORDS, pick, rng_for
from dt.scenarios.sources import html_case
from dt.scenarios.spec import Criterion, ScenarioFamily

register("scenarios.list.row_iou", 0.7, "desktop_list: IoU a predicted container needs with a row to count as found", (0.4, 0.95))
register("scenarios.list.merge_cover", 0.3,
         "desktop_list: a predicted text node 'covers' a segment when it overlaps this fraction of the segment's box",
         (0.1, 0.9))

NAME = "desktop_list"

ICONS = {
    "checkbox": '<rect x="4" y="4" width="16" height="16" rx="2" fill="none" stroke="{c}" stroke-width="2"/>',
    "star": '<polygon points="12,2.5 14.8,8.9 21.8,9.5 16.5,14.1 18.1,21 12,17.3 5.9,21 7.5,14.1 2.2,9.5 9.2,8.9" '
            'fill="none" stroke="{c}" stroke-width="1.8" stroke-linejoin="round"/>',
    "important": '<path d="M3 5h12l6 7-6 7H3l5-7z" fill="none" stroke="{c}" stroke-width="1.8" stroke-linejoin="round"/>',
    "attach": '<path d="M16.5 6.5v10a4.5 4.5 0 0 1-9 0V5a3 3 0 0 1 6 0v10.5a1.5 1.5 0 0 1-3 0V6.5" fill="none" '
              'stroke="{c}" stroke-width="1.8" stroke-linecap="round"/>',
}
THEMES = {
    "light": {"bg": "#ffffff", "read": "#f2f6fc", "div": "#e3e3e3", "ink": "#1f1f1f", "snip": "#5f6368", "icon": "#444746"},
    "dark": {"bg": "#1f1f1f", "read": "#2b2b2b", "div": "#3c4043", "ink": "#e3e3e3", "snip": "#9aa0a6", "icon": "#c4c7c5"},
}
TIMES = ("9:41 AM", "10:05 AM", "11:30 AM", "1:12 PM", "4:58 PM", "Oct 3", "Sep 28", "Aug 14", "Yesterday", "Jun 9")
SEPARATORS = (" - ", " - ", " & ", " / ")


def _icon(name: str, size: int, color: str) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="flex:none;display:block">'
            + ICONS[name].format(c=color) + "</svg>")


def _fit(words: list[str], max_px: float, fs: float) -> str:
    out = ""
    for w in words:
        cand = (out + " " + w) if out else w
        if len(cand) * fs * 0.6 > max_px:
            break
        out = cand
    return out


def generate(p: dict, seed: int):
    rng = rng_for(seed, NAME)
    th = THEMES[p["theme"]]
    W, fs, ip = int(p["width"]), int(p["font_px"]), int(p["icon_px"])
    row_h = max(int(p["row_h"]), ip + 8, round(fs * 1.6))
    n = int(p["n_rows"])
    top = 8
    sender_w = max(int(p["sender_w"]), int(max(len(s) for s in NAMES) * fs * 0.66))
    time_w = int(fs * 6)
    pad = 16
    n_lead = 2 + int(bool(p["important"]))
    mid_w = W - 2 * pad - n_lead * (ip + 12) - sender_w - 24 - time_w - (ip + 12)
    rows_html, rows = [], []
    for i in range(n):
        unread = rng.random() < float(p["unread_frac"])
        sender = pick(rng, NAMES)
        subject = " ".join(pick(rng, WORDS) for _ in range(rng.randint(2, 5)))
        subject = subject[:1].upper() + subject[1:]
        sep = pick(rng, SEPARATORS) if p["punct"] else " - "
        snippet = _fit([pick(rng, WORDS) for _ in range(30)], max(40.0, mid_w - 24 - len(subject + sep) * fs * 0.62), fs)
        mid = subject + sep + snippet
        tm = pick(rng, TIMES)
        attach = rng.random() < float(p["attach_frac"])
        style = p["row_style"]
        bg = th["bg"] if (unread or style == "divider") else th["read"]
        border = f"border-bottom:1px solid {th['div']};" if style in ("divider", "both") else ""
        wt = 700 if unread else 400
        lead = _icon("checkbox", ip, th["icon"]) + _icon("star", ip, th["icon"]) + (_icon("important", ip, th["icon"]) if p["important"] else "")
        rows_html.append(
            f'<div class="row" style="background:{bg};{border}">{lead}'
            f'<div class="snd" style="font-weight:{wt}">{sender}</div>'
            f'<div class="mid"><span style="font-weight:{wt}">{subject}</span><span class="snip">{sep}{snippet}</span></div>'
            f'<div class="att">{_icon("attach", ip, th["icon"]) if attach else ""}</div>'
            f'<div class="tm" style="font-weight:{wt}">{tm}</div></div>')
        rows.append({"box": Box(0, top + i * row_h, W, row_h), "texts": [sender, mid, tm]})
    H = top + n * row_h + 16
    css = (f"body{{background:{th['bg']};}} .list{{padding-top:{top}px;}} "
           f".row{{box-sizing:border-box;height:{row_h}px;display:flex;align-items:center;gap:12px;padding:0 {pad}px;"
           f"font:{fs}px Roboto;color:{th['ink']};white-space:nowrap;}} "
           f".snd{{flex:none;width:{sender_w}px;overflow:hidden;}} .mid{{flex:1;min-width:0;overflow:hidden;}} "
           f".snip{{color:{th['snip']};font-weight:400;}} .att{{flex:none;width:{ip}px;}} "
           f".tm{{flex:none;width:{time_w}px;text-align:right;}}")
    body = f'<div class="list">{"".join(rows_html)}</div>'
    case = html_case(NAME, seed, p, body, W, H, extra_css=css)
    case.meta["rows"] = [r["box"] for r in rows]
    case.meta["segments"] = _segments(case.gt, rows)
    case.meta["highlight_boxes"] = []
    return case


def _segments(gt, rows: list[dict]) -> list[dict]:
    """Group the gt text nodes into visual segments: (row, column) from the generator's strings."""
    segs: dict[tuple[int, int], dict] = {}
    for n in gt.walk():
        if n.type != "text" or not (n.text or "").strip():
            continue
        cy = n.box.cy
        ri = next((i for i, r in enumerate(rows) if r["box"].y <= cy < r["box"].y2), None)
        if ri is None:
            continue
        t = n.text.strip()
        texts = rows[ri]["texts"]
        col = 0 if t in texts[0] and n.box.x < rows[ri]["box"].x + 0.5 * rows[ri]["box"].w else (2 if t == texts[2] else 1)
        s = segs.setdefault((ri, col), {"row": ri, "col": col, "text": texts[col], "box": n.box})
        s["box"] = s["box"].union(n.box)
    return [segs[k] for k in sorted(segs)]


def metrics(case, pred, rendered) -> dict:
    from dt.compare.structural import normalize_text, text_recovered
    from dt.scenarios.families._common import doc_nodes
    nodes = doc_nodes(pred)
    ptexts = [n for n in nodes if n.type == "text" and normalize_text(n.text)]
    segs = case.meta["segments"]
    seg_nodes = [Node(type="text", box=s["box"], text=s["text"]) for s in segs]
    rec = [text_recovered(g, ptexts) for g in seg_nodes]
    cover = float(P["scenarios.list.merge_cover"])
    merged = 0
    for p_ in ptexts:
        hit = [s for s in segs if p_.box.intersect(s["box"]).area >= cover * max(1.0, s["box"].area)]
        if len(hit) >= 2:
            merged += 1
    thr = float(P["scenarios.list.row_iou"])
    conts = [n for n in nodes if n is not pred.root and n.type not in ("text", "icon", "vector", "line")]
    rows_found = sum(1 for b in case.meta["rows"] if any(c.box.iou(b) >= thr for c in conts))
    return {"text_line_recall": (sum(rec) / len(rec)) if rec else None, "merged_lines": merged,
            "row_recall": rows_found / max(1, len(case.meta["rows"])), "n_segments": len(segs)}


FAMILY = ScenarioFamily(
    name=NAME,
    description="Dense desktop inbox rows: every text segment recovered, no merged columns/rows, row containers found.",
    stage="perceive",
    failure_refs=["R4 adversarial dense 1280px Gmail-like list: subject lines lost ' - ', '&', '/' (perceive.ocr)",
                  "R4 long lists 60-120 rows: build_tree O(n^3) grouping (perceive.group)",
                  "R3 unpainted containers never produced: 2-line list items (perceive)"],
    param_space={"theme": ["light", "light", "dark"], "width": [1152, 1280, 1280, 1440], "n_rows": (6, 14),
                 "row_h": (32, 48), "font_px": (13, 15), "icon_px": (8, 30), "sender_w": (150, 220),
                 "unread_frac": (0.0, 0.7), "attach_frac": (0.0, 0.5), "row_style": ["divider", "tinted", "both"],
                 "important": [False, True], "punct": [False, True]},
    generate=generate,
    criteria=[Criterion("text_line_recall", ">=", 0.95, doc="segments recovered (line-fair)"),
              Criterion("merged_lines", "<=", 0, doc="no predicted text node spans two segments"),
              Criterion("row_recall", ">=", 0.9, doc="row containers found")],
    source="html",
    tune_prefixes=("perceive.ocr.", "perceive.group.", "perceive.hier."),
    metrics=metrics,
)
