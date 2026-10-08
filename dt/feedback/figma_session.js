#!/usr/bin/env node
/*
 * figma_session.js -- a headless Figma editing session for the feedback loop (tests, agents without Figma).
 *
 *   node figma_session.js session.json
 *
 * session.json: {
 *   "plan": <build plan>,                      built with the real plugin (dt/export/figma_plugin/code.js)
 *   "components": [key], "componentNames": {key: name}, "fonts": [{family, style}],
 *   "edits": [                                 what a designer does to the built frame afterwards
 *     {"op": "set", "dtId": "...", "key": "characters", "value": "Inbox"},   (fonts are loaded first, like a real edit)
 *     {"op": "remove", "dtId": "..."},
 *     {"op": "add", "parentDtId": "...", "type": "RECTANGLE|ELLIPSE|FRAME|TEXT", "x":0, "y":0, "width":10, "height":10,
 *      "fills": [...], "characters": "...", "name": "..."},
 *     {"op": "swap", "dtId": "...", "componentKey": "..."}
 *   ]
 * }
 * stdout: {"ok": true, "export": <rsdesign.figma-corrections/1>, "summary": {...}} or {"ok": false, "error": "..."}.
 * The export is produced by the plugin's own 'export-corrections' message handler, exactly as in Figma.
 */
'use strict';

var fs = require('fs');
var path = require('path');
var vm = require('vm');
var PLUGIN_DIR = path.join(__dirname, '..', 'export', 'figma_plugin');
var fake = require(path.join(PLUGIN_DIR, 'fake_figma.js'));

function findByDtId(root, id) {
  var hits = root.findAll(function (n) { return typeof n.getPluginData === 'function' && n.getPluginData('dt.id') === String(id); });
  if (!hits.length) throw new Error('no node with dt.id ' + id);
  return hits[0];
}

async function applyEdit(figma, root, e) {
  if (e.op === 'set') {
    var n = findByDtId(root, e.dtId);
    if (n.type === 'TEXT') await figma.loadFontAsync(n.fontName);
    n[e.key] = e.value;
  } else if (e.op === 'remove') {
    findByDtId(root, e.dtId).remove();
  } else if (e.op === 'add') {
    var parent = e.parentDtId ? findByDtId(root, e.parentDtId) : root;
    var make = { RECTANGLE: 'createRectangle', ELLIPSE: 'createEllipse', FRAME: 'createFrame', TEXT: 'createText' }[e.type];
    if (!make) throw new Error('cannot add ' + e.type);
    var c = figma[make]();
    parent.appendChild(c);
    c.x = e.x || 0; c.y = e.y || 0;
    if (e.name) c.name = e.name;
    if (e.type === 'TEXT') {
      var font = e.fontName || { family: 'Roboto', style: 'Regular' };
      await figma.loadFontAsync(font);
      c.fontName = font;
      c.characters = String(e.characters || '');
      if (e.fontSize) c.fontSize = e.fontSize;
      c.textAutoResize = 'NONE';
    }
    c.resize(e.width || 10, e.height || 10);
    if (e.fills) c.fills = e.fills;
  } else if (e.op === 'swap') {
    var inst = findByDtId(root, e.dtId);
    var comp = await figma.importComponentByKeyAsync(e.componentKey);
    inst.swapComponent(comp);
  } else {
    throw new Error('unknown edit op ' + e.op);
  }
}

async function main() {
  var session = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  var figma = fake.createFakeFigma({ fonts: session.fonts, components: session.components || [],
                                     componentNames: session.componentNames || {} });
  global.figma = figma;
  global.__html__ = '<html></html>';
  var code = path.join(PLUGIN_DIR, 'code.js');
  vm.runInThisContext(fs.readFileSync(code, 'utf8'), { filename: code });
  var built = await figma.ui.onmessage({ type: 'build', plan: session.plan, options: {} });
  var root = figma.currentPage.children[figma.currentPage.children.length - 1];
  var edits = session.edits || [];
  for (var i = 0; i < edits.length; i++) await applyEdit(figma, root, edits[i]);
  figma.currentPage.selection = [root.children && root.children.length ? root.children[0] : root];  // any node inside works
  var res = await figma.ui.onmessage({ type: 'export-corrections' });
  process.stdout.write(JSON.stringify({ ok: true, export: res.data, summary: res.summary, build: built }) + '\n');
}

main().catch(function (err) {
  process.stdout.write(JSON.stringify({ ok: false, error: String(err && err.stack ? err.stack : err) }) + '\n');
  process.exit(1);
});
