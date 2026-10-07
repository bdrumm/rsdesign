"""Build the visual progress report (static HTML + images) from tracked knowledge + latest outputs.

    python tools/progress_report.py --bench out/bench/<run> --e2e out/e2e_mwc [--e2e ...] --out out/report

Inputs: knowledge/rounds.jsonl (one JSON object per improvement round), a bench report.json
(current) + out/bench/baseline.json (reference), e2e run dirs with validation/validation.json.
Output: <out>/index.html + <out>/img/*.jpg (downscaled). The HTML is complete at rest (no JS needed).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from typing import Optional

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E = html.escape

GOOD_DIR = {  # metric -> (label, higher_is_better, fmt)
    "node_recall": ("Node recall", True, "{:.3f}"),
    "node_precision": ("Node precision", True, "{:.3f}"),
    "mean_iou": ("Matched box IoU", True, "{:.3f}"),
    "color_de": ("Fill colour ΔE", False, "{:.2f}"),
    "text_cer": ("Text CER", False, "{:.3f}"),
    "text_recall": ("Text recall", True, "{:.3f}"),
    "radius_mae": ("Radius error (px)", False, "{:.2f}"),
    "type_acc": ("Node type accuracy", True, "{:.3f}"),
    "component_acc": ("Component + variant accuracy", True, "{:.3f}"),
    "token_acc": ("Token accuracy", True, "{:.3f}"),
    "mean_de": ("Mean pixel ΔE", False, "{:.2f}"),
    "frac_bad": ("Bad-pixel fraction", False, "{:.3f}"),
    "ssim": ("SSIM", True, "{:.3f}"),
    "layout_consistency": ("Auto-layout consistency", True, "{:.3f}"),
    "jnd_frac_nontext": ("Non-text pixels past JND", False, "{:.3f}"),
    "edge_within1": ("Target edges within 1 px", True, "{:.3f}"),
    "chamfer_tile_max": ("Worst-tile edge chamfer (px)", False, "{:.1f}"),
}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _img(src: str, out_dir: str, name: str, max_w: int = 1600) -> Optional[str]:
    if not src or not os.path.exists(src):
        return None
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    rel = f"img/{name}.jpg"
    im.save(os.path.join(out_dir, rel), quality=88, optimize=True)
    return rel


def _rounds() -> list[dict]:
    p = os.path.join(ROOT, "knowledge", "rounds.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def _sparkline(rounds: list[dict]) -> str:
    pts = [(r["id"], r["composite"]) for r in rounds if r.get("composite") is not None]
    vers = [int(r.get("metrics_version", 1)) for r in rounds if r.get("composite") is not None]
    if len(pts) < 1:
        return '<p class="muted">No benchmark yet.</p>'
    W, H, pl, pr, pt, pb = 560, 170, 44, 16, 14, 30
    lo = min(0.6, min(v for _, v in pts) - 0.02)
    hi = max(1.0, max(v for _, v in pts))
    n = len(pts)

    def X(i: int) -> float:
        return pl + (W - pl - pr) * (i / max(1, n - 1)) if n > 1 else pl + (W - pl - pr) / 2

    def Y(v: float) -> float:
        return pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo))

    grid = []
    for g in [lo, (lo + hi) / 2, hi]:
        grid.append(f'<line x1="{pl}" x2="{W - pr}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="grid"/>'
                    f'<text x="{pl - 6}" y="{Y(g) + 4:.1f}" class="axis" text-anchor="end">{g:.2f}</text>')
    # segments never join across a metric-definition change: the scores are not comparable
    segs, cur = [], []
    for i, (_, v) in enumerate(pts):
        if cur and vers[i] != vers[i - 1]:
            segs.append(cur)
            cur = []
        cur.append((i, v))
    segs.append(cur)
    path = " ".join(" ".join(f"{'M' if j == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for j, (i, v) in enumerate(sg)) for sg in segs)
    area = ""
    for i in range(1, n):
        if vers[i] != vers[i - 1]:
            xm = (X(i) + X(i - 1)) / 2
            grid.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{pt}" y2="{H - pb}" class="grid" stroke-dasharray="3 3"/>'
                        f'<text x="{xm + 4:.1f}" y="{pt + 10}" class="axis">metrics v{vers[i]}</text>')
    dots = []
    for i, (rid, v) in enumerate(pts):
        last = i == n - 1
        dots.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="{5 if last else 3.5}" class="{"dot last" if last else "dot"}"/>'
                    f'<text x="{X(i):.1f}" y="{H - 10}" class="axis" text-anchor="middle">{E(rid)}</text>'
                    f'<text x="{X(i):.1f}" y="{Y(v) - 10:.1f}" class="val" text-anchor="middle">{v:.3f}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" class="spark" role="img" aria-label="Composite score by round">'
            + "".join(grid) + f'<path d="{path}" class="line"/>' + "".join(dots) + "</svg>")


def _metric_rows(cur: dict, base: dict) -> str:
    rows = []
    for k, (label, hib, fmt) in GOOD_DIR.items():
        c = cur.get("metrics", {}).get(k) or {}
        b = base.get("metrics", {}).get(k) or {}
        cg, bg = c.get("goodness"), b.get("goodness")
        cm, bm = c.get("mean"), b.get("mean")
        if cg is None and bg is None:
            continue
        delta = (cg - bg) if (cg is not None and bg is not None) else None
        cls = "flat" if delta is None or abs(delta) < 0.005 else ("up" if delta > 0 else "down")
        dtxt = "–" if delta is None else (f"{delta:+.3f}")
        bar_b = f'<span class="bar base" style="width:{(bg or 0) * 100:.1f}%"></span>' if bg is not None else ""
        bar_c = f'<span class="bar cur" style="width:{(cg or 0) * 100:.1f}%"></span>' if cg is not None else ""
        rows.append(
            f'<tr><th scope="row">{E(label)}<span class="dir">{"higher is better" if hib else "lower is better"}</span></th>'
            f'<td class="num">{fmt.format(bm) if bm is not None else "–"}</td>'
            f'<td class="num strong">{fmt.format(cm) if cm is not None else "–"}</td>'
            f'<td class="bars"><span class="track">{bar_b}{bar_c}</span></td>'
            f'<td class="num delta {cls}">{dtxt}</td></tr>')
    return "\n".join(rows)


def _humanize(msg: str) -> str:
    """'tile chamfer 27.33px @ {'x': 416, 'y': 32, ...}' -> 'tile chamfer 27.33px at x 416, y 32'."""
    import re
    m = re.search(r"@ \{'x': (\d+), 'y': (\d+)[^}]*\}", msg)
    if m:
        msg = msg[: m.start()] + f"at x {m.group(1)}, y {m.group(2)}" + msg[m.end():]
    return msg.replace("JND frac", "pixels past JND").replace("frac", "fraction")


def _gate_chips(gates: dict, fails: dict) -> str:
    out = []
    for tier in ("pixel-exact", "visually-identical", "structurally-faithful"):
        ok = gates.get(tier)
        if ok is None:
            continue
        why = "; ".join(_humanize(f) for f in fails.get(tier, [])[:2])
        out.append(f'<li class="chip {"pass" if ok else "fail"}"><b>{E(tier.replace("-", " "))}</b>'
                   f'<span>{"pass" if ok else "fail"}</span>{f"<small>{E(why)}</small>" if why and not ok else ""}</li>')
    return '<ul class="chips">' + "".join(out) + "</ul>"


def _e2e_block(run_dir: str, out_dir: str, idx: int) -> str:
    vpath = os.path.join(run_dir, "validation", "validation.json")
    mpath = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(vpath):
        return ""
    v = json.load(open(vpath))
    m = json.load(open(mpath)) if os.path.exists(mpath) else {}
    img = _img(os.path.join(run_dir, "validation", "side_by_side.png"), out_dir, f"e2e_{idx}", 1800)
    worst = _img(os.path.join(run_dir, "validation", "worst_regions.png"), out_dir, f"e2e_{idx}_worst", 700)
    name = os.path.basename(os.path.normpath(run_dir))
    lb, la = m.get("loss_before"), m.get("loss_after")
    meas = [
        ("Non-text pixels past JND", f'{v.get("jnd_frac_nontext", 0) * 100:.2f}%'),
        ("Edge chamfer, mean / worst tile", f'{v.get("chamfer", 0):.2f} / {v.get("chamfer_tile_max", 0):.1f} px'),
        ("Target edges within 1 px", f'{v.get("edge_within1", 0) * 100:.1f}%'),
        ("Worst flat region ΔE", f'{v.get("region_de_worst", 0):.1f}'),
        ("Text lines matched", f'{v.get("text_lines_matched", 0)} / {v.get("text_lines_target", 0)}'),
        ("Text CER, max offset", f'{v.get("text_cer", 0):.3f}, {v.get("text_dpos_max", 0):.1f} px'),
    ]
    if lb is not None and la is not None:
        meas.insert(0, ("Refine loss", f"{lb:.4f} → {la:.4f}"))
    dl = "".join(f"<div><dt>{E(k)}</dt><dd>{E(val)}</dd></div>" for k, val in meas)
    fig = (f'<figure class="evidence"><img src="{img}" alt="Target, render and ΔE heat map for {E(name)}" loading="lazy">'
           f'<figcaption><span>Target screenshot</span><span>Our render</span><span>ΔE2000 heat map</span></figcaption></figure>') if img else ""
    wfig = (f'<figure class="worst"><img src="{worst}" alt="Worst regions, target left, render right" loading="lazy">'
            f'<figcaption>Worst regions, target left and render right</figcaption></figure>') if worst else ""
    return (f'<article class="run"><header><h3>{E(name)}</h3><p class="muted">{v.get("width")}×{v.get("height")} px</p></header>'
            f'{_gate_chips(v.get("gates", {}), v.get("gate_failures", {}))}{fig}<div class="run-detail"><dl class="meas">{dl}</dl>{wfig}</div></article>')


def _bench_gallery(bench_dir: str, out_dir: str, cases: list[str]) -> str:
    figs = []
    for c in cases:
        src = os.path.join(bench_dir, "diffs", f"{c}.png")
        rel = _img(src, out_dir, f"bench_{c}", 1800)
        if rel:
            figs.append(f'<figure class="evidence"><img src="{rel}" alt="Bench case {E(c)}: target, render, heat map" loading="lazy">'
                        f'<figcaption><span>{E(c)} target, red boxes = ground truth</span><span>Perceived and re-rendered</span><span>ΔE heat map</span></figcaption></figure>')
    return "".join(figs)


CSS = r"""
/* Layout: one reading column, max 1120px; summary strip, then evidence, then numbers, then history. */
:root{
  --bg:#f5f4f8; --surface:#ffffff; --ink:#1c1a22; --muted:#5f5b6b; --line:#dedae6; --accent:#4a3aa8;
  --pass:#1d7348; --pass-bg:#e3f3ea; --fail:#a8261d; --fail-bg:#fbe7e5; --warn:#8a5a00;
  --bar-base:#c9c3d8; --bar-cur:#4a3aa8;
  --display:'Bricolage Grotesque', 'Hanken Grotesk', system-ui, sans-serif;
  --body:'Hanken Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono:'Martian Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#121118; --surface:#1b1a23; --ink:#ece9f3; --muted:#a6a2b4; --line:#2f2c3a; --accent:#b4a6ff;
  --pass:#6fd6a1; --pass-bg:#173326; --fail:#ff9a8f; --fail-bg:#3a1c1a; --warn:#f2bb63;
  --bar-base:#4a4658; --bar-cur:#b4a6ff; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121118; --surface:#1b1a23; --ink:#ece9f3; --muted:#a6a2b4; --line:#2f2c3a; --accent:#b4a6ff;
  --pass:#6fd6a1; --pass-bg:#173326; --fail:#ff9a8f; --fail-bg:#3a1c1a; --warn:#f2bb63;
  --bar-base:#4a4658; --bar-cur:#b4a6ff; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);padding-inline:clamp(16px,4vw,40px);padding-block:28px 64px}
main{max-width:1120px;margin:0 auto;display:grid;gap:44px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(28px,4vw,40px);font-weight:700;letter-spacing:-.01em}
h2{font-size:22px;font-weight:650}
h3{font-size:16px;font-weight:650;font-family:var(--mono);letter-spacing:-.01em}
p{margin:0}
.muted{color:var(--muted)}
.eyebrow{font:500 11px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
section{display:grid;gap:16px}
.intro{display:grid;gap:12px;max-width:72ch}
.strip{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.strip div{background:var(--surface);padding:14px 16px;display:grid;gap:4px;min-width:0}
.strip b{font:600 22px/1.1 var(--display);font-variant-numeric:tabular-nums}
.strip span{font:500 11px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.chips{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:8px}
.chip{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 8px;padding:6px 10px;border-radius:6px;font-size:13px;min-width:0;max-width:100%}
.chip b{font-weight:600;text-transform:capitalize}
.chip span{font:600 11px/1 var(--mono);text-transform:uppercase;letter-spacing:.06em}
.chip small{flex-basis:100%;color:var(--muted);font:12px/1.4 var(--mono);overflow-wrap:anywhere}
.chip.pass{background:var(--pass-bg);color:var(--pass)} .chip.fail{background:var(--fail-bg);color:var(--fail)}
.chip.pass b,.chip.fail b{color:var(--ink)}
.run{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;display:grid;gap:14px;min-width:0}
.run header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
figure{margin:0;display:grid;gap:6px;min-width:0}
figure img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line);background:#fff}
.evidence figcaption{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font:12px/1.3 var(--mono);color:var(--muted)}
.worst figcaption{font:12px/1.3 var(--mono);color:var(--muted)}
.run-detail{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,320px);gap:20px;align-items:start}
@media (max-width:760px){.run-detail{grid-template-columns:1fr}}
.meas{margin:0;display:grid;gap:0}
.meas div{display:flex;justify-content:space-between;gap:16px;padding:7px 0;border-bottom:1px solid var(--line)}
.meas dt{color:var(--muted)} .meas dd{margin:0;font:500 13px/1.5 var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:640px;font-size:14px}
th,td{padding:9px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
thead th{font:500 11px/1.2 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
tbody th{font-weight:500}
.dir{display:block;font:11px/1.3 var(--mono);color:var(--muted)}
td.num{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
td.strong{font-weight:600}
.delta.up{color:var(--pass)} .delta.down{color:var(--fail)} .delta.flat{color:var(--muted)}
.bars{width:34%}
.track{display:grid;gap:3px}
.bar{display:block;height:6px;border-radius:3px}
.bar.base{background:var(--bar-base)} .bar.cur{background:var(--bar-cur)}
.legend{display:flex;gap:16px;font:12px/1 var(--mono);color:var(--muted);flex-wrap:wrap}
.legend i{display:inline-block;width:14px;height:6px;border-radius:3px;margin-right:6px;vertical-align:middle}
.two{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:24px;align-items:start}
@media (max-width:860px){.two{grid-template-columns:1fr}}
.spark{width:100%;height:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
.spark .grid{stroke:var(--line);stroke-width:1}
.spark .axis{fill:var(--muted);font:11px var(--mono)}
.spark .val{fill:var(--ink);font:600 11px var(--mono)}
.spark .line{fill:none;stroke:var(--accent);stroke-width:2}
.spark .area{fill:var(--accent);opacity:.12}
.spark .dot{fill:var(--surface);stroke:var(--accent);stroke-width:2}
.spark .dot.last{fill:var(--accent)}
.rounds{list-style:none;margin:0;padding:0;display:grid;gap:0;border-left:2px solid var(--line)}
.rounds li{padding:0 0 18px 18px;position:relative;display:grid;gap:6px}
.rounds li::before{content:"";position:absolute;left:-7px;top:4px;width:12px;height:12px;border-radius:50%;background:var(--surface);border:2px solid var(--accent)}
.rounds li:last-child::before{background:var(--accent)}
.rounds .rh{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.rounds .rid{font:600 12px/1 var(--mono);color:var(--accent)}
.rounds ul{margin:0;padding-left:18px;color:var(--muted);font-size:14px;display:grid;gap:2px}
.next{margin:0;padding-left:20px;display:grid;gap:8px;max-width:80ch}
.next b{font-weight:600}
footer{font:12px/1.5 var(--mono);color:var(--muted)}
@media (min-width:560px){.strip{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (min-width:1000px){.strip{grid-template-columns:repeat(6,minmax(0,1fr))}}
@media (prefers-reduced-motion:no-preference){figure img{transition:opacity .2s}}
"""


def build(bench_dir: Optional[str], e2e_dirs: list[str], out_dir: str, gallery: list[str], next_items: list[str],
          updated: str, extra: dict, baseline_path: Optional[str] = None, baseline_label: str = "Baseline (R1)") -> str:
    os.makedirs(out_dir, exist_ok=True)
    rounds = _rounds()
    bp = baseline_path or os.path.join(ROOT, "out", "bench", "baseline.json")
    base = json.load(open(bp)) if os.path.exists(bp) else {}
    cur = json.load(open(os.path.join(bench_dir, "report.json"))) if bench_dir and os.path.exists(os.path.join(bench_dir, "report.json")) else base
    commit = _git("rev-parse", "--short", "HEAD")
    latest = rounds[-1] if rounds else {}
    comp = cur.get("composite")
    strip = [
        ("Latest round", f'{latest.get("id", "–")}'),
        ("Benchmark composite", f"{comp:.3f}" if comp is not None else "–"),
        ("Reference composite", f'{base.get("composite", 0):.3f}' if base else "–"),
        ("Tests passing", str(extra.get("tests", latest.get("tests", "–")))),
        ("Ground-truth screens", str(cur.get("n_cases", "–"))),
        ("Commit", commit or "–"),
    ]
    strip_html = "".join(f"<div><span>{E(k)}</span><b>{E(v)}</b></div>" for k, v in strip)
    e2e_html = "".join(_e2e_block(d, out_dir, i) for i, d in enumerate(e2e_dirs))
    rounds_html = "".join(
        f'<li><div class="rh"><span class="rid">{E(r["id"])}</span><h3>{E(r["title"])}</h3>'
        f'<span class="muted">{E(r.get("when", ""))}{" · composite " + format(r["composite"], ".3f") if r.get("composite") is not None else ""}'
        f'{" · " + str(r["tests"]) + " tests" if r.get("tests") else ""}</span></div>'
        f'<ul>{"".join(f"<li>{E(s)}</li>" for s in r.get("summary", []))}</ul></li>' for r in rounds)
    gal = _bench_gallery(bench_dir, out_dir, gallery) if bench_dir else ""
    nxt = "".join(f"<li>{n}</li>" for n in next_items)
    extra_html = extra.get("html", "")
    page = f"""<title>rsdesign Progress</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Hanken+Grotesk:wght@400;500;600&family=Martian+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<main>
<section class="intro">
  <p class="eyebrow">Screenshot → IR → Material 3 → Figma · self-improving harness</p>
  <h1>rsdesign build progress</h1>
  <p class="muted">Every number on this page is measured by code against ground truth or against the input screenshot. Nothing here is judged by eye. Updated {E(updated)}.</p>
</section>
<div class="strip">{strip_html}</div>
<section>
  <h2>Translations of real Material 3 screens</h2>
  <p class="muted">Each run perceives the screenshot, maps it to Material 3, refines it against the original, and is then graded by the independent validator.</p>
  {e2e_html or '<p class="muted">No end-to-end runs yet.</p>'}
</section>
{extra_html}
<section>
  <h2>Benchmark against ground truth</h2>
  <p class="muted">Ground truth comes from the DOM of real Material Web pages and from synthetic Material 3 screens. Bars show goodness from 0 to 1.</p>
  <div class="legend"><span><i style="background:var(--bar-base)"></i>{E(baseline_label)}</span><span><i style="background:var(--bar-cur)"></i>Current</span></div>
  <div class="tablewrap"><table><thead><tr><th>Metric</th><th class="num">Baseline</th><th class="num">Current</th><th>Goodness</th><th class="num">Δ</th></tr></thead>
  <tbody>{_metric_rows(cur, base)}</tbody></table></div>
  {gal}
</section>
<section class="two">
  <div style="display:grid;gap:12px;min-width:0"><h2>Composite by round</h2>{_sparkline(rounds)}</div>
  <div style="display:grid;gap:12px;min-width:0"><h2>Rounds</h2><ol class="rounds">{rounds_html}</ol></div>
</section>
<section>
  <h2>Next round</h2>
  <ol class="next">{nxt}</ol>
</section>
<footer>Generated by tools/progress_report.py from knowledge/rounds.jsonl, bench reports and validation outputs.</footer>
</main>"""
    path = os.path.join(out_dir, "index.html")
    open(path, "w").write(page)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--e2e", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "report"))
    ap.add_argument("--gallery", nargs="*", default=[])
    ap.add_argument("--next", action="append", default=[])
    ap.add_argument("--updated", default="")
    ap.add_argument("--extra-json", default="")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--baseline-label", default="Baseline (R1)")
    a = ap.parse_args()
    extra = json.load(open(a.extra_json)) if a.extra_json else {}
    print(build(a.bench, a.e2e, a.out, a.gallery, a.next, a.updated, extra, a.baseline or None, a.baseline_label))
