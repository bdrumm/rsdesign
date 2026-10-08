/*
 * fake_figma.js — a minimal in-memory implementation of the Figma Plugin API surface used by
 * code.js, so the plugin can be executed headlessly (tests, CI, agents without Figma).
 *
 * Mirrors the real API's sharp edges that matter for correctness:
 *   - TEXT properties (characters, fontSize, fontName, ...) throw unless the font is loaded
 *   - loadFontAsync throws for fonts not in the available list
 *   - importComponentByKeyAsync throws for unknown keys
 *   - layoutSizing FILL requires an auto-layout parent; HUG only on auto-layout frames / text
 *   - per-side stroke weights exist only on FRAME / RECTANGLE / INSTANCE / COMPONENT
 *
 *   - with documentAccess "dynamic-page", instance.mainComponent is read via getMainComponentAsync()
 *
 * createFakeFigma({fonts, components, componentNames, variables, noVariables}) -> figma
 *   fonts:      [{family, style}]  available fonts (default: Roboto family, Inter Regular, Material Symbols)
 *   components: [key]              keys importComponentByKeyAsync resolves (a button-like component)
 *   componentNames: {key: name}    display names of those components (default "Component <key>")
 *   variables:  [name]             local COLOR variables
 *   noVariables: true              remove figma.variables entirely (older editors)
 * serialize(node) -> plain JSON tree.
 */
'use strict';

var DEFAULT_FONTS = [
  { family: 'Roboto', style: 'Thin' }, { family: 'Roboto', style: 'Light' }, { family: 'Roboto', style: 'Regular' },
  { family: 'Roboto', style: 'Medium' }, { family: 'Roboto', style: 'Bold' }, { family: 'Roboto', style: 'Black' },
  { family: 'Roboto', style: 'Italic' }, { family: 'Roboto', style: 'Bold Italic' }, { family: 'Roboto', style: 'Medium Italic' },
  { family: 'Inter', style: 'Regular' }, { family: 'Inter', style: 'Medium' }, { family: 'Inter', style: 'Bold' },
  { family: 'Material Symbols Outlined', style: 'Regular' }
];

var TEXT_FONT_FIELDS = ['characters', 'fontSize', 'lineHeight', 'letterSpacing', 'textAlignHorizontal',
  'textAlignVertical', 'textDecoration', 'textCase', 'textAutoResize'];
var CONTAINER_TYPES = { PAGE: 1, FRAME: 1, COMPONENT: 1, INSTANCE: 1 };
var SKIP_KEYS = { parent: 1, children: 1, mainComponent: 1 };
var CORNER_KEYS = ['topLeftRadius', 'topRightRadius', 'bottomRightRadius', 'bottomLeftRadius'];
var SIDE_KEYS = ['strokeTopWeight', 'strokeRightWeight', 'strokeBottomWeight', 'strokeLeftWeight'];
var MIXED = 'MIXED';  // stands in for figma.mixed (a Symbol in the real API) so it survives JSON serialization

function fontKey(f) { return f.family + ' / ' + f.style; }

function createFakeFigma(opts) {
  opts = opts || {};
  var available = {};
  (opts.fonts || DEFAULT_FONTS).forEach(function (f) { available[fontKey(f)] = f; });
  var loaded = {};
  var components = {};
  var variables = (opts.variables || []).map(function (name, i) {
    return { id: 'VariableID:' + (i + 1), name: name, resolvedType: 'COLOR' };
  });
  var nextId = 1;
  var posted = [];
  var notices = [];
  var images = {};  // imageHash -> base64 PNG bytes

  function isLoaded(font) { return !!(font && loaded[fontKey(font)]); }

  function defaults(type) {
    var d = {
      visible: true, opacity: 1, x: 0, y: 0, width: 100, height: 100, boundVariables: {}, pluginData: {}, removed: false,
      fills: [], strokes: [], strokeWeight: 1, strokeAlign: 'INSIDE', effects: []
    };
    if (type === 'FRAME' || type === 'COMPONENT' || type === 'INSTANCE') {
      Object.assign(d, {
        fills: type === 'FRAME' ? [{ type: 'SOLID', color: { r: 1, g: 1, b: 1 }, opacity: 1 }] : [],
        clipsContent: true, cornerRadius: 0, topLeftRadius: 0, topRightRadius: 0, bottomRightRadius: 0, bottomLeftRadius: 0,
        strokeTopWeight: 1, strokeRightWeight: 1, strokeBottomWeight: 1, strokeLeftWeight: 1,
        layoutMode: 'NONE', itemSpacing: 0, paddingTop: 0, paddingRight: 0, paddingBottom: 0, paddingLeft: 0,
        primaryAxisAlignItems: 'MIN', counterAxisAlignItems: 'MIN', primaryAxisSizingMode: 'AUTO',
        counterAxisSizingMode: 'AUTO', layoutWrap: 'NO_WRAP', layoutSizingHorizontal: 'FIXED', layoutSizingVertical: 'FIXED'
      });
    } else if (type === 'RECTANGLE') {
      Object.assign(d, {
        fills: [{ type: 'SOLID', color: { r: 0.85, g: 0.85, b: 0.85 }, opacity: 1 }],
        cornerRadius: 0, topLeftRadius: 0, topRightRadius: 0, bottomRightRadius: 0, bottomLeftRadius: 0,
        strokeTopWeight: 1, strokeRightWeight: 1, strokeBottomWeight: 1, strokeLeftWeight: 1,
        layoutSizingHorizontal: 'FIXED', layoutSizingVertical: 'FIXED'
      });
    } else if (type === 'ELLIPSE') {
      d.fills = [{ type: 'SOLID', color: { r: 0.85, g: 0.85, b: 0.85 }, opacity: 1 }];
      d.layoutSizingHorizontal = 'FIXED'; d.layoutSizingVertical = 'FIXED';
    } else if (type === 'LINE') {
      Object.assign(d, { height: 0, strokes: [{ type: 'SOLID', color: { r: 0, g: 0, b: 0 }, opacity: 1 }], strokeAlign: 'CENTER' });
    } else if (type === 'TEXT') {
      Object.assign(d, {
        fills: [{ type: 'SOLID', color: { r: 0, g: 0, b: 0 }, opacity: 1 }],
        characters: '', fontName: { family: 'Inter', style: 'Regular' }, fontSize: 12,
        lineHeight: { unit: 'AUTO' }, letterSpacing: { value: 0, unit: 'PERCENT' },
        textAlignHorizontal: 'LEFT', textAlignVertical: 'TOP', textAutoResize: 'WIDTH_AND_HEIGHT',
        textDecoration: 'NONE', textCase: 'ORIGINAL', layoutSizingHorizontal: 'FIXED', layoutSizingVertical: 'FIXED'
      });
    }
    return d;
  }

  /** Validation rules applied on property assignment (the parts of the real API that throw). */
  function checkSet(node, key, value) {
    if (node.type === 'TEXT') {
      if (key === 'fontName') {
        if (!isLoaded(value)) throw new Error('Cannot write to node with unloaded font "' + fontKey(value) + '". Please call figma.loadFontAsync');
      } else if (TEXT_FONT_FIELDS.indexOf(key) >= 0 && !isLoaded(node.fontName)) {
        throw new Error('Cannot write to node with unloaded font "' + fontKey(node.fontName) + '". Please call figma.loadFontAsync');
      }
    }
    if (key === 'layoutSizingHorizontal' || key === 'layoutSizingVertical') {
      var parentAuto = node.parent && node.parent.layoutMode && node.parent.layoutMode !== 'NONE';
      if (value === 'FILL' && !parentAuto) throw new Error('FILL sizing is only valid on auto-layout children');
      var selfAuto = node.type === 'TEXT' || (node.layoutMode && node.layoutMode !== 'NONE');
      if (value === 'HUG' && !selfAuto) throw new Error('HUG sizing is only valid on auto-layout frames and text nodes');
    }
    if ((key === 'fills' || key === 'strokes' || key === 'effects') && !Array.isArray(value)) {
      throw new Error(key + ' must be an array');
    }
    if (key === 'opacity' && !(typeof value === 'number' && value >= 0 && value <= 1)) throw new Error('opacity must be in [0,1]');
  }

  function makeNode(type) {
    var node = Object.assign({ id: '1:' + (nextId++), type: type, name: type, parent: null }, defaults(type));
    if (CONTAINER_TYPES[type]) node.children = [];
    node.resize = function (w, h) {
      if (!(isFinite(w) && isFinite(h)) || w < 0 || h < 0) throw new Error('resize expects non-negative finite numbers');
      this.width = w; this.height = h;
    };
    node.appendChild = function (child) {
      if (!this.children) throw new Error(this.type + ' cannot have children');
      if (child.parent && child.parent.children) {
        var idx = child.parent.children.indexOf(child);
        if (idx >= 0) child.parent.children.splice(idx, 1);
      }
      child.parent = this;
      this.children.push(child);
    };
    node.remove = function () {
      if (this.parent && this.parent.children) {
        var i = this.parent.children.indexOf(this);
        if (i >= 0) this.parent.children.splice(i, 1);
      }
      this.parent = null; this.removed = true;
    };
    node.findAll = function (fn) {
      var out = [];
      (function walk(n) { (n.children || []).forEach(function (c) { if (!fn || fn(c)) out.push(c); walk(c); }); })(this);
      return out;
    };
    node.findOne = function (fn) { return this.findAll(fn)[0] || null; };
    node.setPluginData = function (key, value) { this.pluginData[String(key)] = String(value); };
    node.getPluginData = function (key) { return this.pluginData[String(key)] || ''; };
    node.getPluginDataKeys = function () { return Object.keys(this.pluginData); };
    node.setBoundVariable = function (field, variable) {
      var v = typeof variable === 'string' ? variables.filter(function (x) { return x.id === variable; })[0] : variable;
      if (!v || !v.id) throw new Error('setBoundVariable: unknown variable');
      this.boundVariables[field] = { type: 'VARIABLE_ALIAS', id: v.id };
    };
    return new Proxy(node, {
      set: function (target, key, value, receiver) {
        checkSet(receiver, key, value);
        target[key] = value;
        // real Figma: cornerRadius writes all four corners; differing corners read back as figma.mixed
        if (key === 'cornerRadius' && 'topLeftRadius' in target) {
          CORNER_KEYS.forEach(function (k) { target[k] = value; });
        } else if (CORNER_KEYS.indexOf(key) >= 0) {
          var same = CORNER_KEYS.every(function (k) { return target[k] === value; });
          target.cornerRadius = same ? value : MIXED;
        } else if (key === 'strokeWeight' && 'strokeTopWeight' in target) {  // same rule for per-side stroke weights
          SIDE_KEYS.forEach(function (k) { target[k] = value; });
        } else if (SIDE_KEYS.indexOf(key) >= 0) {
          var sameSide = SIDE_KEYS.every(function (k) { return target[k] === value; });
          target.strokeWeight = sameSide ? value : MIXED;
        }
        return true;
      }
    });
  }

  function makeComponent(key) {
    var comp = makeNode('COMPONENT');
    comp.key = key;
    comp.name = (opts.componentNames && opts.componentNames[key]) || ('Component ' + key);
    comp.createInstance = function () {
      var inst = makeNode('INSTANCE');
      inst.name = comp.name;
      inst.componentKey = key;
      inst.mainComponent = comp;
      inst.getMainComponentAsync = function () { return Promise.resolve(this.mainComponent); };
      inst.swapComponent = function (other) {  // real API: keeps the instance (and its plugin data), swaps the main
        if (!other || other.type !== 'COMPONENT') throw new Error('swapComponent expects a COMPONENT');
        this.mainComponent = other; this.componentKey = other.key;
      };
      inst.componentProperties = {
        'Style': { type: 'VARIANT', value: 'Filled' },
        'Size': { type: 'VARIANT', value: 'md' },
        'label#1:0': { type: 'TEXT', value: 'Label' },
        'showIcon#2:0': { type: 'BOOLEAN', value: false }
      };
      // the component's own text is set with its font "pre-loaded"; restore the plugin-visible state afterwards
      var label = makeNode('TEXT');
      label.name = 'label';
      var labelFont = { family: 'Roboto', style: 'Medium' };
      var wasLoaded = !!loaded[fontKey(labelFont)];
      loaded[fontKey(labelFont)] = true;
      label.fontName = labelFont;
      label.characters = 'Label';
      if (!wasLoaded) delete loaded[fontKey(labelFont)];
      inst.appendChild(label);
      inst.setProperties = function (obj) {
        var self = this;
        Object.keys(obj).forEach(function (k) {
          if (!(k in self.componentProperties)) throw new Error('Unknown component property "' + k + '"');
          self.componentProperties[k].value = obj[k];
          if (k.indexOf('label') === 0) label.characters = String(obj[k]);
        });
      };
      return inst;
    };
    return comp;
  }
  (opts.components || []).forEach(function (key) { components[key] = makeComponent(key); });

  var page = makeNode('PAGE');
  page.name = 'Page 1';
  page.selection = [];

  var figma = {
    apiVersion: '1.0.0',
    editorType: 'figma',
    currentPage: page,
    root: { type: 'DOCUMENT', children: [page] },
    ui: {
      onmessage: null,
      postMessage: function (m) { posted.push(m); },
      show: function () {}, hide: function () {}, close: function () {}, resize: function () {}
    },
    showUI: function (html, o) { figma.__ui = { html: html, options: o }; },
    closePlugin: function (msg) { figma.__closed = { message: msg === undefined ? null : msg }; },
    notify: function (msg) { notices.push(msg); return { cancel: function () {} }; },
    createFrame: function () { return makeNode('FRAME'); },
    createRectangle: function () { return makeNode('RECTANGLE'); },
    base64Decode: function (b64) { return Uint8Array.from(Buffer.from(b64, 'base64')); },
    createImage: function (bytes) {
      if (!bytes || !bytes.length || bytes[0] !== 0x89 || bytes[1] !== 0x50) throw new Error('createImage: not a PNG');
      // content hash like real Figma (equal-length but different images must not collide)
      var hash = 'img_' + require('crypto').createHash('sha1').update(Buffer.from(bytes)).digest('hex');
      images[hash] = Buffer.from(bytes).toString('base64');
      return { hash: hash, getBytesAsync: function () { return Promise.resolve(Uint8Array.from(Buffer.from(images[hash], 'base64'))); } };
    },
    getImageByHash: function (hash) {
      if (!(hash in images)) return null;
      return { hash: hash, getBytesAsync: function () { return Promise.resolve(Uint8Array.from(Buffer.from(images[hash], 'base64'))); } };
    },
    createEllipse: function () { return makeNode('ELLIPSE'); },
    createLine: function () { return makeNode('LINE'); },
    createText: function () { return makeNode('TEXT'); },
    loadFontAsync: function (f) {
      return new Promise(function (resolve, reject) {
        var key = f && fontKey(f);
        if (!available[key]) return reject(new Error('Font not found: ' + key));
        loaded[key] = true;
        resolve();
      });
    },
    listAvailableFontsAsync: function () {
      return Promise.resolve(Object.keys(available).map(function (k) { return { fontName: available[k] }; }));
    },
    importComponentByKeyAsync: function (key) {
      return new Promise(function (resolve, reject) {
        if (!components[key]) return reject(new Error('Could not find component with key "' + key + '"'));
        resolve(components[key]);
      });
    },
    mixed: MIXED,
    viewport: { center: { x: 0, y: 0 }, zoom: 1, scrollAndZoomIntoView: function () {} },
    __state: { posted: posted, notices: notices, loaded: loaded, available: available, components: components, variables: variables, images: images }
  };
  if (!opts.noVariables) {
    figma.variables = {
      getLocalVariablesAsync: function (type) {
        return Promise.resolve(variables.filter(function (v) { return !type || v.resolvedType === type; }));
      },
      setBoundVariableForPaint: function (paint, field, variable) {
        if (!variable || !variable.id) throw new Error('setBoundVariableForPaint: unknown variable');
        var copy = JSON.parse(JSON.stringify(paint));
        copy.boundVariables = Object.assign({}, copy.boundVariables || {});
        copy.boundVariables[field] = { type: 'VARIABLE_ALIAS', id: variable.id };
        return copy;
      }
    };
  }
  return figma;
}

/** Plain-JSON snapshot of a node subtree (functions, parent links and component back-refs dropped). */
function serialize(node) {
  var out = {};
  Object.keys(node).forEach(function (k) {
    if (SKIP_KEYS[k] || typeof node[k] === 'function') return;
    out[k] = node[k];
  });
  if (node.children) out.children = node.children.map(serialize);
  return out;
}

module.exports = { createFakeFigma: createFakeFigma, serialize: serialize, DEFAULT_FONTS: DEFAULT_FONTS };
