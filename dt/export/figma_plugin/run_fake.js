#!/usr/bin/env node
/*
 * run_fake.js — execute code.js headlessly against fake_figma.js and print the resulting tree.
 *
 *   node run_fake.js --plan plan.json [--components k1,k2] [--variables n1,n2]
 *                    [--fonts "Roboto:Regular;Inter:Regular"] [--no-variables] [--code code.js]
 *   (plan is read from stdin when --plan is omitted)
 *
 * stdout: {"ok": true, "summary": {...}, "tree": [<page children>], "posted": [...], "closed": ...}
 *         {"ok": false, "error": "..."} with exit code 1 on failure.
 */
'use strict';

var fs = require('fs');
var path = require('path');
var vm = require('vm');
var fake = require('./fake_figma.js');

function parseArgs(argv) {
  var args = { plan: null, components: [], variables: [], fonts: null, noVariables: false, code: path.join(__dirname, 'code.js') };
  for (var i = 0; i < argv.length; i++) {
    var a = argv[i];
    if (a === '--plan') args.plan = argv[++i];
    else if (a === '--components') args.components = splitList(argv[++i]);
    else if (a === '--variables') args.variables = splitList(argv[++i]);
    else if (a === '--fonts') args.fonts = splitList(argv[++i], ';').map(function (s) {
      var parts = s.split(':');
      return { family: parts[0], style: parts[1] || 'Regular' };
    });
    else if (a === '--no-variables') args.noVariables = true;
    else if (a === '--code') args.code = argv[++i];
    else throw new Error('unknown argument ' + a);
  }
  return args;
}

function splitList(s, sep) {
  return String(s || '').split(sep || ',').map(function (x) { return x.trim(); }).filter(Boolean);
}

function readPlan(file) {
  var text = file ? fs.readFileSync(file, 'utf8') : fs.readFileSync(0, 'utf8');
  return JSON.parse(text);
}

async function main() {
  var args = parseArgs(process.argv.slice(2));
  var plan = readPlan(args.plan);
  var figma = fake.createFakeFigma({ fonts: args.fonts, components: args.components, variables: args.variables, noVariables: args.noVariables });
  global.figma = figma;
  global.__html__ = '<html></html>';
  vm.runInThisContext(fs.readFileSync(args.code, 'utf8'), { filename: args.code });
  if (typeof figma.ui.onmessage !== 'function') throw new Error('code.js did not register figma.ui.onmessage');

  var summary = await figma.ui.onmessage({ type: 'build', plan: plan, options: {} });
  var done = figma.__state.posted.filter(function (m) { return m.type === 'done'; })[0];
  var result = {
    ok: true,
    summary: summary || (done && done.summary) || null,
    tree: figma.currentPage.children.map(fake.serialize),
    selection: figma.currentPage.selection.map(function (n) { return n.id; }),
    posted: figma.__state.posted,
    closed: figma.__closed || null,
    fontsLoaded: Object.keys(figma.__state.loaded),
    images: figma.__state.images  // imageHash -> base64 PNG (what figma.getImageByHash(h).getBytesAsync() returns)
  };
  process.stdout.write(JSON.stringify(result) + '\n');
}

main().catch(function (err) {
  process.stdout.write(JSON.stringify({ ok: false, error: String(err && err.stack ? err.stack : err) }) + '\n');
  process.exit(1);
});
