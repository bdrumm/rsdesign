"""review.html: a self-contained local page for correcting one translate run.

Written by ``dt translate`` into the run directory (``pipeline.review_html``) and by ``dt feedback review``.
Open it from disk (file://, no server, no network): target screenshot and our render side by side with the
IR boxes overlaid; click a box to correct it (component, text, icon, geometry, colour, extra, should be an
image / editable, ok), toggle "draw missing" and drag a box for something we missed, rate the run, then
"Download corrections.json" and feed it back with ``dt feedback add <run_dir> corrections.json``.
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Optional

from dt.feedback.store import CORRECTIONS_SCHEMA, KINDS, default_consent, model_version, sha256_file
from dt.ir import Document, Node

_TYPES_FOR_MISSING = ("rect", "frame", "text", "icon", "image", "ellipse", "line")


def _png_data_uri(rgb: Any) -> Optional[str]:
    if rgb is None:
        return None
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb[..., :3]).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _node_rows(doc: Document) -> list[dict]:
    rows = []
    for n, depth in doc.root.walk_with_depth():
        if n is doc.root or not n.visible:
            continue
        fill = n.fill_color if n.type != "text" else (n.text_style.color if n.text_style else None)
        rows.append({
            "id": n.id, "type": n.meta.get("orig_type", n.type) if n.type == "instance" else n.type, "name": n.name,
            "box": [round(v, 1) for v in (n.box.x, n.box.y, n.box.w, n.box.h)], "depth": depth,
            "text": n.text if n.type == "text" else None, "icon": n.icon_name if n.type == "icon" else None,
            "component": {"name": n.component.name, "variant": dict(n.component.variant),
                          "confidence": round(float(n.component.confidence), 2)} if n.component is not None else None,
            "fill": fill.hex() if fill is not None else None,
        })
    return rows


def _components(ds) -> list[dict]:
    if ds is None:
        return []
    vocab = ds.variant_vocabulary() if hasattr(ds, "variant_vocabulary") else {}
    return [{"name": c.name, "variants": {k: list(v) for k, v in (vocab.get(c.name) or {}).items()}} for c in ds.components]


def review_payload(run_dir: str, doc: Document, target: Any = None, render: Any = None, ds=None,
                   source: Optional[str] = None, dpr: Any = None) -> dict:
    src = source or doc.source_image
    return {
        "schema": CORRECTIONS_SCHEMA,
        "run": {"run_dir": os.path.abspath(run_dir), "source_name": os.path.basename(src) if src else None,
                "screenshot_sha256": sha256_file(src) if src and os.path.exists(src) else None,
                "size": {"w": int(doc.width), "h": int(doc.height)}, "dpr": dpr if dpr is not None else doc.dpr,
                "design_system": doc.design_system, "model": model_version()},
        "target": _png_data_uri(target), "render": _png_data_uri(render),
        "nodes": _node_rows(doc), "components": _components(ds), "kinds": list(KINDS),
        "missing_types": list(_TYPES_FOR_MISSING), "consent": default_consent(),
    }


def write_review(run_dir: str, doc: Optional[Document] = None, target: Any = None, ds=None, source: Optional[str] = None,
                 dpr: Any = None, out_path: Optional[str] = None) -> str:
    """Write ``<run_dir>/review.html`` (or ``out_path``); returns the path. Missing pieces (no render yet, source
    moved) degrade gracefully: the page still works on the boxes alone."""
    from dt.common.image import load_rgb
    doc = doc if doc is not None else Document.load(os.path.join(run_dir, "ir.mapped.json"))
    if target is None:
        from dt.feedback.store import load_target
        src = source or doc.source_image
        metrics = os.path.join(run_dir, "metrics.json")
        if dpr is None and os.path.exists(metrics):
            with open(metrics) as f:
                dpr = json.load(f).get("dpr")
        target = load_target({"source_path": src, "dpr": dpr or 1.0}) if src else None
    if ds is None and doc.design_system:
        from dt.feedback.capture import design_system_for
        ds = design_system_for(doc.design_system)
    rp = os.path.join(run_dir, "render.png")
    render = load_rgb(rp) if os.path.exists(rp) else None
    payload = review_payload(run_dir, doc, target, render, ds, source=source, dpr=dpr)
    data = json.dumps(payload).replace("</", "<\\/").replace("<!--", "<\\!--")
    html = TEMPLATE.replace("__PAYLOAD__", data).replace("__TITLE__", _esc(payload["run"]["source_name"] or "run"))
    out = out_path or os.path.join(run_dir, "review.html")
    with open(out, "w") as f:
        f.write(html)
    return out


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review __TITLE__</title>
<style>
  :root { --bg:#f7f7f8; --panel:#fff; --ink:#1d1b20; --muted:#625b71; --line:#d9d6de; --accent:#6750a4;
          --comp:#3b6fd8; --text:#2e8b57; --other:#8a8494; --sel:#d93025; --miss:#e8710a; }
  @media (prefers-color-scheme: dark) { :root { --bg:#141218; --panel:#1d1b20; --ink:#e6e0e9; --muted:#cac4d0;
          --line:#49454f; --accent:#d0bcff; --other:#938f99; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center;
           padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line); }
  header h1 { font-size:14px; margin:0 8px 0 0; font-weight:600; }
  header .grow { flex:1; }
  button, select, input { font:inherit; color:inherit; }
  button { border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:5px 10px; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button.on { outline:2px solid var(--accent); }
  .layout { display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr) 320px; gap:12px; padding:12px 16px; align-items:start; }
  @media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }
  .pane { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px; min-width:0; }
  .pane h2 { font-size:12px; color:var(--muted); font-weight:600; margin:0 0 6px; }
  .wrap { position:relative; width:100%; line-height:0; user-select:none; }
  .wrap img { width:100%; display:block; }
  .wrap .ph { line-height:1.4; padding:40px 0; text-align:center; color:var(--muted); border:1px dashed var(--line); }
  .ov { position:absolute; inset:0; }
  .nb { position:absolute; border:1px solid var(--other); background:transparent; cursor:pointer; }
  .nb.comp { border-color:var(--comp); }
  .nb.text { border-color:var(--text); }
  .nb:hover { background:rgba(103,80,164,.12); }
  .nb.sel { border:2px solid var(--sel); background:rgba(217,48,37,.10); }
  .nb.corrected { box-shadow: inset 0 0 0 2px var(--miss); }
  .hideboxes .nb:not(.sel) { border-color:transparent; }
  .draw { position:absolute; border:2px dashed var(--miss); background:rgba(232,113,10,.12); pointer-events:none; }
  .drawing .ov { cursor:crosshair; }
  .side { position:sticky; top:64px; display:flex; flex-direction:column; gap:12px; }
  .side section { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; }
  .side h3 { font-size:12px; margin:0 0 8px; color:var(--muted); font-weight:600; }
  .row { display:flex; gap:6px; align-items:center; margin:6px 0; flex-wrap:wrap; }
  .row label { min-width:64px; color:var(--muted); }
  .row input[type=text], .row input[type=number], .row select { flex:1; min-width:0; padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:var(--panel); }
  .row input[type=number] { width:64px; flex:0 0 64px; }
  #info { font-family: ui-monospace, Menlo, monospace; font-size:11px; white-space:pre-wrap; color:var(--muted); }
  ul#items { list-style:none; margin:0; padding:0; max-height:300px; overflow:auto; }
  ul#items li { display:flex; gap:6px; align-items:center; padding:4px 0; border-bottom:1px solid var(--line); }
  ul#items li span { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .muted { color:var(--muted); }
  .rating button { min-width:36px; }
</style>
</head>
<body>
<header>
  <h1>Review: __TITLE__</h1>
  <span class="rating" role="group" aria-label="Overall rating">
    <button id="thumbs-up" title="Looks right">Looks right</button>
    <button id="thumbs-down" title="Needs work">Needs work</button>
    <select id="rating" aria-label="Rating 1-5"><option value="">rate 1-5</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
  </span>
  <button id="mode-missing" title="Drag a box on the screenshot around something we missed">Draw missing</button>
  <label><input type="checkbox" id="show-boxes" checked> boxes</label>
  <span class="grow"></span>
  <label title="Keep a copy of the screenshot in your local feedback store"><input type="checkbox" id="c-store"> keep screenshot locally</label>
  <label title="Allow an export to include the screenshot"><input type="checkbox" id="c-share-shot"> share screenshot</label>
  <label title="Allow an export to include text content"><input type="checkbox" id="c-share-text"> share text</label>
  <span id="count" class="muted">0 corrections</span>
  <button id="download" class="primary">Download corrections.json</button>
</header>
<div class="layout">
  <div class="pane"><h2>Screenshot: click a box to correct it</h2><div class="wrap" id="target-wrap"></div></div>
  <div class="pane"><h2>Our render</h2><div class="wrap" id="render-wrap"></div></div>
  <div class="side">
    <section>
      <h3>Selected</h3>
      <div id="info">Click a box (or draw a missing one).</div>
      <div class="row"><label for="kind">correction</label><select id="kind"></select></div>
      <div id="fields"></div>
      <div class="row"><label for="v-note">note</label><input type="text" id="v-note" placeholder="optional"></div>
      <div class="row"><button id="add" class="primary" disabled>Add correction</button><button id="clear">Clear</button></div>
    </section>
    <section>
      <h3>Corrections</h3>
      <ul id="items"></ul>
      <p class="muted">Then run <code>dt feedback add &lt;run_dir&gt; corrections.json</code>. Nothing leaves this machine unless you export it.</p>
    </section>
  </div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  'use strict';
  var P = JSON.parse(document.getElementById('payload').textContent);
  var W = P.run.size.w, H = P.run.size.h;
  var state = { items: [], rating: null, selected: null, pendingBox: null, drawMode: false };
  var byId = {};
  P.nodes.forEach(function (n) { byId[n.id] = n; });
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function pct(v, of) { return (100 * v / of) + '%'; }

  // ---------------------------------------------------------------- panes
  function buildPane(wrapId, src, interactive) {
    var wrap = $(wrapId);
    if (src) wrap.appendChild(el('img', { src: src, alt: '' }));
    else { var ph = el('div', { 'class': 'ph' }, 'image unavailable'); ph.style.aspectRatio = W + ' / ' + H; wrap.appendChild(ph); }
    var ov = el('div', { 'class': 'ov' });
    var sorted = P.nodes.slice().sort(function (a, b) { return b.box[2] * b.box[3] - a.box[2] * a.box[3]; });
    sorted.forEach(function (n) {
      var d = el('div', { 'class': 'nb' + (n.component ? ' comp' : n.type === 'text' ? ' text' : ''), 'data-node-id': n.id,
                          title: label(n) });
      d.style.left = pct(n.box[0], W); d.style.top = pct(n.box[1], H);
      d.style.width = pct(Math.max(n.box[2], 1), W); d.style.height = pct(Math.max(n.box[3], 1), H);
      if (interactive) d.addEventListener('click', function (ev) { if (state.drawMode) return; ev.stopPropagation(); select(n.id); });
      else d.style.pointerEvents = 'none';
      ov.appendChild(d);
    });
    wrap.appendChild(ov);
    return ov;
  }
  function label(n) {
    var s = n.type + ':' + n.id;
    if (n.component) s += ' -> ' + n.component.name + fmtVariant(n.component.variant);
    if (n.text) s += ' "' + n.text.slice(0, 40) + '"';
    if (n.icon) s += ' icon ' + n.icon;
    return s;
  }
  function fmtVariant(v) {
    var k = Object.keys(v || {});
    return k.length ? '{' + k.map(function (x) { return x + '=' + v[x]; }).join(', ') + '}' : '';
  }

  // ---------------------------------------------------------------- selection + form
  function select(id) {
    state.selected = id; state.pendingBox = null;
    document.querySelectorAll('.nb').forEach(function (d) { d.classList.toggle('sel', d.getAttribute('data-node-id') === id); });
    var n = byId[id];
    $('info').textContent = label(n) + '\nbox ' + n.box.join(', ') + (n.fill ? '\nfill ' + n.fill : '') +
      (n.component ? '\nconfidence ' + n.component.confidence : '');
    var kinds = P.kinds.filter(function (k) { return k !== 'missing'; });
    if (n.type !== 'text') kinds = kinds.filter(function (k) { return k !== 'text'; });
    fillKinds(kinds, n.component ? 'component' : (n.type === 'text' ? 'text' : 'ok'));
    $('add').disabled = false;
  }
  function selectBox(box) {
    state.selected = null; state.pendingBox = box;
    document.querySelectorAll('.nb.sel').forEach(function (d) { d.classList.remove('sel'); });
    $('info').textContent = 'missing element at ' + box.map(function (v) { return Math.round(v); }).join(', ');
    fillKinds(['missing'], 'missing');
    $('add').disabled = false;
  }
  function fillKinds(kinds, preferred) {
    var s = $('kind'); s.innerHTML = '';
    kinds.forEach(function (k) { s.appendChild(el('option', { value: k }, k.replace(/_/g, ' '))); });
    s.value = kinds.indexOf(preferred) >= 0 ? preferred : kinds[0];
    renderFields();
  }
  function field(labelText, input) {
    var r = el('div', { 'class': 'row' });
    r.appendChild(el('label', { 'for': input.id }, labelText));
    r.appendChild(input);
    return r;
  }
  function input(id, type, value) { var i = el('input', { id: id, type: type || 'text' }); if (value !== undefined && value !== null) i.value = value; return i; }
  function componentSelect(id, current) {
    var s = el('select', { id: id });
    s.appendChild(el('option', { value: '' }, '(not a component)'));
    P.components.forEach(function (c) { s.appendChild(el('option', { value: c.name }, c.name)); });
    if (current && !P.components.some(function (c) { return c.name === current; })) s.appendChild(el('option', { value: current }, current));
    s.value = current || '';
    return s;
  }
  function renderFields() {
    var f = $('fields'); f.innerHTML = '';
    var k = $('kind').value, n = state.selected ? byId[state.selected] : null;
    if (k === 'component') {
      f.appendChild(field('component', componentSelect('v-component', n && n.component ? n.component.name : '')));
      f.appendChild(field('variant', input('v-variant', 'text', n && n.component ? fmtVariant(n.component.variant).replace(/[{}]/g, '') : '')));
    } else if (k === 'text') {
      f.appendChild(field('text', input('v-text', 'text', n ? n.text : '')));
    } else if (k === 'icon') {
      f.appendChild(field('icon', input('v-icon', 'text', n ? n.icon : '')));
    } else if (k === 'geometry') {
      var r = el('div', { 'class': 'row' }); r.appendChild(el('label', {}, 'x y w h'));
      ['x', 'y', 'w', 'h'].forEach(function (a, i) { r.appendChild(input('v-' + a, 'number', n ? n.box[i] : 0)); });
      f.appendChild(r);
    } else if (k === 'color') {
      f.appendChild(field('colour', input('v-color', 'color', n && n.fill ? n.fill : '#000000')));
    } else if (k === 'missing') {
      var t = el('select', { id: 'v-type' });
      P.missing_types.forEach(function (x) { t.appendChild(el('option', { value: x }, x)); });
      f.appendChild(field('type', t));
      f.appendChild(field('text', input('v-text', 'text', '')));
      f.appendChild(field('icon', input('v-icon', 'text', '')));
      f.appendChild(field('component', componentSelect('v-component', '')));
    }
  }
  function parseVariant(s) {
    var out = {};
    String(s || '').split(',').forEach(function (p) {
      var kv = p.split('='); if (kv.length === 2 && kv[0].trim()) out[kv[0].trim()] = kv[1].trim();
    });
    return out;
  }
  function currentValue(k) {
    var v = function (id) { var e = $(id); return e ? e.value : ''; };
    if (k === 'component') return { name: v('v-component') || null, variant: parseVariant(v('v-variant')) };
    if (k === 'text') return { text: v('v-text') };
    if (k === 'icon') return { icon_name: v('v-icon') };
    if (k === 'geometry') return { box: ['x', 'y', 'w', 'h'].map(function (a) { return Number(v('v-' + a)); }) };
    if (k === 'color') return { color: v('v-color') };
    if (k === 'missing') {
      var m = { type: v('v-type') };
      if (v('v-text')) m.text = v('v-text');
      if (v('v-icon')) m.icon_name = v('v-icon');
      if (v('v-component')) m.component = { name: v('v-component'), variant: {} };
      return m;
    }
    return null;
  }
  function addCorrection() {
    var k = $('kind').value;
    if (!k || (!state.selected && !state.pendingBox)) return;
    var item = { kind: k, value: currentValue(k), note: $('v-note').value || '' };
    if (state.selected) { item.node_id = state.selected; item.box = byId[state.selected].box.slice(); }
    else { item.node_id = null; item.box = state.pendingBox.map(function (x) { return Math.round(x * 10) / 10; }); }
    // one correction per (kind, node): a later one replaces the earlier
    state.items = state.items.filter(function (it) { return !(it.node_id && it.node_id === item.node_id && it.kind === item.kind); });
    state.items.push(item);
    $('v-note').value = '';
    refreshItems();
  }
  function refreshItems() {
    var ul = $('items'); ul.innerHTML = '';
    state.items.forEach(function (it, i) {
      var li = el('li', { 'data-index': i });
      var v = it.value ? JSON.stringify(it.value) : '';
      li.appendChild(el('span', { title: v }, it.kind + ' ' + (it.node_id || 'new box') + ' ' + v));
      var b = el('button', { 'aria-label': 'remove' }, 'remove');
      b.addEventListener('click', function () { state.items.splice(i, 1); refreshItems(); });
      li.appendChild(b);
      ul.appendChild(li);
    });
    document.querySelectorAll('.nb').forEach(function (d) {
      var id = d.getAttribute('data-node-id');
      d.classList.toggle('corrected', state.items.some(function (it) { return it.node_id === id; }));
    });
    $('count').textContent = state.items.length + ' correction' + (state.items.length === 1 ? '' : 's');
  }

  // ---------------------------------------------------------------- draw a missing box
  var drawStart = null, drawEl = null;
  function toImage(ev, wrap) {
    var r = wrap.getBoundingClientRect();
    return [Math.min(W, Math.max(0, (ev.clientX - r.left) * W / r.width)), Math.min(H, Math.max(0, (ev.clientY - r.top) * H / r.height))];
  }
  function setupDraw(wrap) {
    wrap.addEventListener('mousedown', function (ev) {
      if (!state.drawMode) return;
      ev.preventDefault();
      drawStart = toImage(ev, wrap);
      drawEl = el('div', { 'class': 'draw' }); wrap.appendChild(drawEl);
    });
    window.addEventListener('mousemove', function (ev) {
      if (!drawStart || !drawEl) return;
      var p = toImage(ev, wrap), x = Math.min(p[0], drawStart[0]), y = Math.min(p[1], drawStart[1]);
      drawEl.style.left = pct(x, W); drawEl.style.top = pct(y, H);
      drawEl.style.width = pct(Math.abs(p[0] - drawStart[0]), W); drawEl.style.height = pct(Math.abs(p[1] - drawStart[1]), H);
    });
    window.addEventListener('mouseup', function (ev) {
      if (!drawStart) return;
      var p = toImage(ev, wrap);
      var box = [Math.min(p[0], drawStart[0]), Math.min(p[1], drawStart[1]), Math.abs(p[0] - drawStart[0]), Math.abs(p[1] - drawStart[1])];
      drawStart = null;
      if (drawEl) { drawEl.remove(); drawEl = null; }
      if (box[2] >= 2 && box[3] >= 2) selectBox(box);
    });
  }

  // ---------------------------------------------------------------- output
  function buildCorrections() {
    return {
      schema: P.schema, created: new Date().toISOString(), channel: 'review', run: P.run,
      items: state.items.slice(), run_rating: state.rating,
      consent: { store_screenshot: $('c-store').checked, share_screenshot: $('c-share-shot').checked, share_text: $('c-share-text').checked }
    };
  }
  function download() {
    var blob = new Blob([JSON.stringify(buildCorrections(), null, 2)], { type: 'application/json' });
    var a = el('a', { href: URL.createObjectURL(blob), download: 'corrections.json' });
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  function setRating(r) {
    state.rating = r;
    $('rating').value = r ? String(r) : '';
    $('thumbs-up').classList.toggle('on', r === 5); $('thumbs-down').classList.toggle('on', r === 1);
  }

  // ---------------------------------------------------------------- wire up
  buildPane('target-wrap', P.target, true);
  buildPane('render-wrap', P.render, false);
  setupDraw($('target-wrap'));
  $('c-store').checked = !!P.consent.store_screenshot;
  $('c-share-shot').checked = !!P.consent.share_screenshot;
  $('c-share-text').checked = !!P.consent.share_text;
  $('kind').addEventListener('change', renderFields);
  $('add').addEventListener('click', addCorrection);
  $('clear').addEventListener('click', function () { state.selected = null; state.pendingBox = null; $('info').textContent = 'Click a box (or draw a missing one).'; $('kind').innerHTML = ''; $('fields').innerHTML = ''; $('add').disabled = true; document.querySelectorAll('.nb.sel').forEach(function (d) { d.classList.remove('sel'); }); });
  $('download').addEventListener('click', download);
  $('thumbs-up').addEventListener('click', function () { setRating(state.rating === 5 ? null : 5); });
  $('thumbs-down').addEventListener('click', function () { setRating(state.rating === 1 ? null : 1); });
  $('rating').addEventListener('change', function () { setRating(this.value ? Number(this.value) : null); });
  $('mode-missing').addEventListener('click', function () {
    state.drawMode = !state.drawMode; this.classList.toggle('on', state.drawMode);
    document.body.classList.toggle('drawing', state.drawMode);
  });
  $('show-boxes').addEventListener('change', function () { document.body.classList.toggle('hideboxes', !this.checked); });
  document.addEventListener('keydown', function (ev) { if (ev.key === 'Escape') $('clear').click(); });
  window.rsReview = { state: state, buildCorrections: buildCorrections, select: select };
})();
</script>
</body>
</html>
"""

__all__ = ["write_review", "review_payload"]
