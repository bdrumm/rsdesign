/*
 * designtranslator Figma plugin — turns a build plan (dt/export/BUILD_PLAN.md, v1.x) into nodes.
 *
 * Plain ES2019, no imports, no bundler. Runs inside Figma's plugin sandbox, or headlessly against
 * fake_figma.js (see run_fake.js). Protocol with ui.html:
 *   ui -> main : { type: 'build', plan: <plan>, options: { closeWhenDone: bool } }
 *   main -> ui : { type: 'done', summary: {...} } | { type: 'error', message: string }
 */
'use strict';

var PLAN_MAJOR = 1;
var HARD_FALLBACK_FONTS = [
  { family: 'Roboto', style: 'Regular' },
  { family: 'Inter', style: 'Regular' }
];

figma.showUI(__html__, { width: 440, height: 560 });

figma.ui.onmessage = function (msg) {
  if (!msg || msg.type !== 'build') return undefined;
  return buildPlan(msg.plan, msg.options || {}).then(function (summary) {
    figma.ui.postMessage({ type: 'done', summary: summary });
    if (msg.options && msg.options.closeWhenDone) figma.closePlugin('Built ' + summary.created + ' nodes');
    return summary;
  }).catch(function (err) {
    var message = String(err && err.stack ? err.stack : err);
    figma.ui.postMessage({ type: 'error', message: message });
    throw err;
  });
};

// ------------------------------------------------------------------ entry point
async function buildPlan(plan, options) {
  if (!plan || typeof plan !== 'object' || !plan.root) throw new Error('plan has no root');
  var major = parseInt(String(plan.version || '0').split('.')[0], 10);
  if (major !== PLAN_MAJOR) throw new Error('unsupported plan version ' + plan.version);
  var state = {
    created: 0, texts: 0, instances: 0, fallbacks: 0,
    warnings: [], fontsLoaded: [], fontsMissing: [],
    fontCache: {}, vars: null, plan: plan
  };
  await preloadFonts(plan, state);
  var root = await createNode(plan.root, figma.currentPage, state);
  if (root && options.offsetX !== undefined) root.x = Number(options.offsetX) || 0;
  if (root && options.offsetY !== undefined) root.y = Number(options.offsetY) || 0;
  if (root) {
    figma.currentPage.selection = [root];
    if (figma.viewport && figma.viewport.scrollAndZoomIntoView) figma.viewport.scrollAndZoomIntoView([root]);
  }
  return {
    created: state.created, texts: state.texts, instances: state.instances, fallbacks: state.fallbacks,
    warnings: state.warnings, fontsLoaded: state.fontsLoaded, fontsMissing: state.fontsMissing,
    rootId: root ? root.id : null
  };
}

// ------------------------------------------------------------------ fonts
function fontKey(f) { return f.family + ' / ' + f.style; }

function collectFonts(spec, into) {
  if (!spec) return into;
  if (spec.fontName) into[fontKey(spec.fontName)] = spec.fontName;
  if (spec.fallback) collectFonts(spec.fallback, into);
  (spec.children || []).forEach(function (c) { collectFonts(c, into); });
  return into;
}

/** Try to load a font once; remembers success/failure. Returns true when usable. */
async function ensureFont(font, state) {
  if (!font || !font.family || !font.style) return false;
  var key = fontKey(font);
  if (key in state.fontCache) return state.fontCache[key];
  try {
    await figma.loadFontAsync({ family: font.family, style: font.style });
    state.fontCache[key] = true;
    state.fontsLoaded.push(key);
  } catch (e) {
    state.fontCache[key] = false;
    state.fontsMissing.push(key);
  }
  return state.fontCache[key];
}

async function preloadFonts(plan, state) {
  var wanted = collectFonts(plan.root, {});
  (plan.fonts || []).forEach(function (f) { if (f && f.family) wanted[fontKey(f)] = f; });
  var keys = Object.keys(wanted);
  for (var i = 0; i < keys.length; i++) await ensureFont(wanted[keys[i]], state);
}

/**
 * Resolve (family, style) -> a loaded font. Cascade keeps the weight first, then the family:
 *   requested -> fallback family @ same style -> requested family @ Regular -> plan fallback -> Roboto/Inter.
 */
async function resolveFont(font, state) {
  var fb = state.plan.fallbackFont;
  var candidates = [font];
  if (font && fb && fb.family && font.style) candidates.push({ family: fb.family, style: font.style });
  if (font && font.style !== 'Regular') candidates.push({ family: font.family, style: 'Regular' });
  if (fb) candidates.push(fb);
  HARD_FALLBACK_FONTS.forEach(function (f) {
    if (font && font.style) candidates.push({ family: f.family, style: font.style });
    candidates.push(f);
  });
  for (var i = 0; i < candidates.length; i++) {
    var c = candidates[i];
    if (!c || !c.family) continue;
    if (await ensureFont(c, state)) {
      if (i > 0) state.warnings.push('font ' + fontKey(font) + ' missing; used ' + fontKey(c));
      return { family: c.family, style: c.style };
    }
  }
  throw new Error('no usable font for ' + fontKey(font));
}

// ------------------------------------------------------------------ node creation
async function createNode(spec, parent, state) {
  if (!spec || typeof spec !== 'object') return null;
  if (spec.type === 'INSTANCE') return createInstance(spec, parent, state);
  var node = createShape(spec.type, state);
  if (!node) return null;
  node.name = spec.name || spec.type;
  stamp(node, spec);
  if (spec.isIcon && spec.iconFill) {
    state.warnings.push('icon "' + node.name + '" is filled: set the Material Symbols Fill axis to ' + spec.iconFill +
      ' (stored as plugin data dt.iconFill)');
  }
  parent.appendChild(node);
  applyGeometry(node, spec);
  applyCommon(node, spec, state);
  if (spec.type === 'TEXT') await applyText(node, spec, state);
  if (spec.type === 'FRAME') {
    applyLayout(node, spec);
    var children = spec.children || [];
    for (var i = 0; i < children.length; i++) await createNode(children[i], node, state);
    if (spec.layoutMode && spec.layoutMode !== 'NONE') {
      node.resize(Math.max(spec.width, 0.01), Math.max(spec.height, 0.01));
      // real Figma turns an AUTO (hug) axis into FIXED on resize; restore the plan's declared modes
      if (spec.primaryAxisSizingMode) node.primaryAxisSizingMode = spec.primaryAxisSizingMode;
      if (spec.counterAxisSizingMode) node.counterAxisSizingMode = spec.counterAxisSizingMode;
    }
  }
  applyChildSizing(node, spec, parent, state);
  await bindVariables(node, spec, state);
  state.created += 1;
  return node;
}

/** Provenance on the Figma node (setPluginData): the plan/IR id, and the icon FILL axis Figma cannot set. */
function stamp(node, spec) {
  if (typeof node.setPluginData !== 'function') return;
  if (spec.id !== undefined) node.setPluginData('dt.id', String(spec.id));
  if (spec.isIcon && spec.iconFill) node.setPluginData('dt.iconFill', String(spec.iconFill));
}

function createShape(type, state) {
  switch (type) {
    case 'FRAME': return figma.createFrame();
    case 'RECTANGLE': return figma.createRectangle();
    case 'ELLIPSE': return figma.createEllipse();
    case 'LINE': return figma.createLine();
    case 'TEXT': return figma.createText();
    default:
      state.warnings.push('unknown node type ' + type + ' skipped');
      return null;
  }
}

function applyGeometry(node, spec) {
  node.x = Number(spec.x) || 0;
  node.y = Number(spec.y) || 0;
  var w = Math.max(Number(spec.width) || 0, 0.01);
  var h = spec.type === 'LINE' ? 0 : Math.max(Number(spec.height) || 0, 0.01);
  node.resize(w, h);
}

function solidPaint(rgb, opacity) {
  return { type: 'SOLID', color: { r: rgb.r, g: rgb.g, b: rgb.b }, opacity: opacity === undefined ? 1 : opacity };
}

/** Normalize plan paints into what the API accepts; IMAGE paints without an imageHash become placeholders. */
function toPaints(paints, state, owner) {
  var out = [];
  (paints || []).forEach(function (p) {
    if (!p || !p.type) return;
    if (p.type === 'IMAGE' && !p.imageHash && typeof p.imageRef === 'string' && p.imageRef.indexOf('data:image/') === 0
        && typeof figma.createImage === 'function' && typeof figma.base64Decode === 'function') {
      try {  // embedded crop (as-is raster): decode and register the bytes with Figma
        var b64 = p.imageRef.slice(p.imageRef.indexOf(',') + 1);
        var img = figma.createImage(figma.base64Decode(b64));
        out.push({ type: 'IMAGE', scaleMode: p.scaleMode || 'FILL', imageHash: img.hash, opacity: p.opacity === undefined ? 1 : p.opacity });
        state.images = (state.images || 0) + 1;
        return;
      } catch (e) {
        state.warnings.push(owner + ': embedded image could not be decoded (' + e + ')');
      }
    }
    if (p.type === 'IMAGE' && !p.imageHash) {
      var ph = state.plan.imagePlaceholder || { r: 0.88, g: 0.88, b: 0.88 };
      state.warnings.push(owner + ': IMAGE fill ' + (p.imageRef || '') + ' replaced by placeholder');
      out.push(solidPaint(ph, p.opacity));
      return;
    }
    var copy = {};
    Object.keys(p).forEach(function (k) { if (k !== 'imageRef') copy[k] = p[k]; });
    if (copy.type === 'SOLID') copy.color = { r: p.color.r, g: p.color.g, b: p.color.b };
    if (copy.opacity === undefined) copy.opacity = 1;
    out.push(copy);
  });
  return out;
}

function applyCommon(node, spec, state) {
  var owner = spec.type + ' "' + (spec.name || spec.id) + '"';
  if (spec.visible === false) node.visible = false;
  if (typeof spec.opacity === 'number') node.opacity = spec.opacity;
  if (spec.type !== 'LINE' && spec.type !== 'TEXT') node.fills = toPaints(spec.fills, state, owner);
  if (spec.strokes && spec.strokes.length) {
    node.strokes = toPaints(spec.strokes, state, owner);
    if (typeof spec.strokeWeight === 'number') node.strokeWeight = spec.strokeWeight;
    if (spec.strokeAlign) node.strokeAlign = spec.strokeAlign;
    if (typeof spec.strokeTopWeight === 'number' && typeof node.strokeTopWeight !== 'undefined') {
      node.strokeTopWeight = spec.strokeTopWeight;
      node.strokeRightWeight = spec.strokeRightWeight;
      node.strokeBottomWeight = spec.strokeBottomWeight;
      node.strokeLeftWeight = spec.strokeLeftWeight;
    }
  }
  if (spec.effects && spec.effects.length) {
    node.effects = spec.effects.map(function (e) {
      return {
        type: e.type, color: e.color, offset: e.offset, radius: e.radius, spread: e.spread || 0,
        visible: e.visible !== false, blendMode: e.blendMode || 'NORMAL'
      };
    });
  }
  if (spec.type === 'FRAME' || spec.type === 'RECTANGLE') {
    if (typeof spec.cornerRadius === 'number') node.cornerRadius = spec.cornerRadius;
    if (typeof spec.topLeftRadius === 'number') {
      node.topLeftRadius = spec.topLeftRadius;
      node.topRightRadius = spec.topRightRadius;
      node.bottomRightRadius = spec.bottomRightRadius;
      node.bottomLeftRadius = spec.bottomLeftRadius;
    }
  }
  if (spec.type === 'FRAME') node.clipsContent = spec.clipsContent === true;
}

var LAYOUT_FIELDS = ['itemSpacing', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
  'primaryAxisAlignItems', 'counterAxisAlignItems', 'primaryAxisSizingMode', 'counterAxisSizingMode', 'layoutWrap'];

function applyLayout(node, spec) {
  if (!spec.layoutMode || spec.layoutMode === 'NONE') return;
  node.layoutMode = spec.layoutMode;
  LAYOUT_FIELDS.forEach(function (k) { if (spec[k] !== undefined) node[k] = spec[k]; });
}

function applyChildSizing(node, spec, parent, state) {
  var parentAuto = parent && parent.layoutMode && parent.layoutMode !== 'NONE';
  if (!parentAuto) return;
  ['layoutSizingHorizontal', 'layoutSizingVertical'].forEach(function (k) {
    if (!spec[k]) return;
    try { node[k] = spec[k]; } catch (e) { state.warnings.push(spec.name + ': ' + k + '=' + spec[k] + ' rejected (' + e.message + ')'); }
  });
}

async function applyText(node, spec, state) {
  var font = await resolveFont(spec.fontName, state);
  node.fontName = font;
  node.characters = String(spec.characters || '');
  if (typeof spec.fontSize === 'number') node.fontSize = spec.fontSize;
  if (spec.lineHeight) node.lineHeight = spec.lineHeight;
  if (spec.letterSpacing) node.letterSpacing = spec.letterSpacing;
  if (spec.textAlignHorizontal) node.textAlignHorizontal = spec.textAlignHorizontal;
  if (spec.textAlignVertical) node.textAlignVertical = spec.textAlignVertical;
  if (spec.textDecoration) node.textDecoration = spec.textDecoration;
  if (spec.textCase) node.textCase = spec.textCase;
  node.textAutoResize = spec.textAutoResize || 'NONE';
  node.fills = toPaints(spec.fills, state, 'TEXT "' + spec.name + '"');
  if (node.textAutoResize === 'NONE') node.resize(Math.max(spec.width, 0.01), Math.max(spec.height, 0.01));
  state.texts += 1;
}

// ------------------------------------------------------------------ instances
async function createInstance(spec, parent, state) {
  var component = null;
  try {
    component = await figma.importComponentByKeyAsync(spec.componentKey);
  } catch (e) {
    state.warnings.push('component ' + spec.componentKey + ' (' + (spec.componentName || '') + ') unavailable: ' +
      (e && e.message ? e.message : e) + '; built fallback frame');
  }
  if (!component) {
    state.fallbacks += 1;
    var fb = {};
    Object.keys(spec.fallback || {}).forEach(function (k) { fb[k] = spec.fallback[k]; });
    fb.name = spec.name || fb.name;
    return createNode(fb, parent, state);
  }
  var inst = component.createInstance();
  inst.name = spec.name || inst.name;
  stamp(inst, spec);
  parent.appendChild(inst);
  applyGeometry(inst, spec);
  if (spec.visible === false) inst.visible = false;
  if (typeof spec.opacity === 'number') inst.opacity = spec.opacity;
  await applyInstanceProperties(inst, spec, state);
  applyChildSizing(inst, spec, parent, state);
  await bindVariables(inst, spec, state);
  state.created += 1;
  state.instances += 1;
  return inst;
}

/** Find the component property key matching a plan name: exact, or "name#id" with the same prefix. */
function findPropKey(defs, name) {
  if (!defs) return null;
  if (name in defs) return name;
  var lower = String(name).toLowerCase();
  var keys = Object.keys(defs);
  for (var i = 0; i < keys.length; i++) {
    if (keys[i].split('#')[0].toLowerCase() === lower) return keys[i];
  }
  return null;
}

function findTextChild(inst, name) {
  var lower = String(name).toLowerCase();
  var texts = inst.findAll ? inst.findAll(function (n) { return n.type === 'TEXT'; }) : [];
  for (var i = 0; i < texts.length; i++) if (String(texts[i].name).toLowerCase() === lower) return texts[i];
  if ((lower === 'label' || lower === 'text') && texts.length) return texts[0];
  return null;
}

async function applyInstanceProperties(inst, spec, state) {
  var defs = inst.componentProperties || {};
  var updates = {};
  var variants = spec.variantProperties || {};
  Object.keys(variants).forEach(function (k) {
    var key = findPropKey(defs, k);
    if (key && defs[key].type === 'VARIANT') updates[key] = String(variants[k]);
    else state.warnings.push(spec.name + ': variant property ' + k + ' not found on component');
  });
  var props = spec.props || {};
  var keys = Object.keys(props);
  for (var i = 0; i < keys.length; i++) {
    var k = keys[i], v = props[k];
    var key = findPropKey(defs, k);
    if (key) {
      var t = defs[key].type;
      updates[key] = t === 'BOOLEAN' ? Boolean(v) : (t === 'TEXT' ? String(v) : v);
      continue;
    }
    if (typeof v === 'string') {
      var textNode = findTextChild(inst, k);
      if (textNode) {
        var font = textNode.fontName;
        if (font && font.family && (await ensureFont(font, state))) textNode.characters = v;
        else state.warnings.push(spec.name + ': cannot load font of text "' + k + '"');
        continue;
      }
    }
    state.warnings.push(spec.name + ': prop ' + k + ' not applied');
  }
  if (Object.keys(updates).length) {
    // text properties rewrite TEXT sub-nodes, which requires their fonts to be loaded
    var texts = inst.findAll ? inst.findAll(function (n) { return n.type === 'TEXT'; }) : [];
    for (var j = 0; j < texts.length; j++) if (texts[j].fontName && texts[j].fontName.family) await ensureFont(texts[j].fontName, state);
    try { inst.setProperties(updates); } catch (e) { state.warnings.push(spec.name + ': setProperties failed: ' + e.message); }
  }
}

// ------------------------------------------------------------------ variables
async function loadVariables(state) {
  if (state.vars !== null) return state.vars;
  state.vars = {};
  if (!figma.variables || typeof figma.variables.getLocalVariablesAsync !== 'function') return state.vars;
  try {
    var list = await figma.variables.getLocalVariablesAsync();
    list.forEach(function (v) { state.vars[v.name] = v; });
  } catch (e) {
    state.warnings.push('variables unavailable: ' + e.message);
  }
  return state.vars;
}

async function bindVariables(node, spec, state) {
  var bindings = spec.variableBindings;
  if (!bindings || !Object.keys(bindings).length) return;
  var vars = await loadVariables(state);
  var fields = Object.keys(bindings);
  for (var i = 0; i < fields.length; i++) {
    var field = fields[i], name = bindings[field], v = vars[name];
    if (!v) { state.warnings.push(spec.name + ': variable "' + name + '" not found for ' + field); continue; }
    try {
      if (field === 'fills' || field === 'strokes') {
        var paints = (node[field] || []).slice();
        if (!paints.length || paints[0].type !== 'SOLID') throw new Error('no solid paint to bind');
        paints[0] = figma.variables.setBoundVariableForPaint(paints[0], 'color', v);
        node[field] = paints;
      } else {
        node.setBoundVariable(field, v);
      }
    } catch (e) {
      state.warnings.push(spec.name + ': binding ' + field + ' -> ' + name + ' failed: ' + e.message);
    }
  }
}
