// core/trace.js — loading trace data (site/data/*.json, written by tools/trace/) and referring to it from prose.
//
// Every trace file is an object whose top-level key "scenes" is an array of scenes {id, nodes[], ...}. Other top-level keys
// (schema, library, kind, generator, tables for other lessons...) are ignored by the loader but stay reachable by T().
// D (what lesson text functions receive) = { files: {<file stem>: json}, scenes: {<scene id>: scene}, anchors: {<name>: number} | null }.
// D.anchors comes from site/data/anchors.json (written by tools/anchors.mjs from the trace paths in site/anchors.json); lesson-boot.js
// always loads it, so T.anchor(name) works in every lesson without loading the big trace file the anchor comes from.
import { fmt } from '../ops/texutil.js';
import { escapeHtml } from './dom.js';
import { ROLES, HUMAN_KINDS, proseNumber, sigFor, humanText, formatBinding } from './numbers.js';

export const ANCHORS_SCHEMA = 'torc-anchors/1';
// site/data/anchors.json {schema, generator, anchors: {<name>: {path, value}}} -> {<name>: value}
export function anchorValues(json) {
  if (!json || json.schema !== ANCHORS_SCHEMA || typeof json.anchors !== 'object') throw new Error(`data/anchors.json: expected {schema: "${ANCHORS_SCHEMA}", anchors: {…}}`);
  return Object.fromEntries(Object.entries(json.anchors).map(([k, a]) => [k, a.value]));
}

export async function loadAnchors(baseUrl, fetchFn = fetch) {
  const res = await fetchFn(new URL('data/anchors.json', baseUrl));
  if (!res.ok) throw new Error(`data/anchors.json: HTTP ${res.status}`);
  return anchorValues(await res.json());
}

export function buildD(files, anchors = null) {
  const D = { files: {}, scenes: {}, anchors };
  for (const [stem, json] of Object.entries(files)) {
    if (json === null || typeof json !== 'object' || Array.isArray(json)) throw new Error(`trace file ${stem}.json: expected a JSON object`);
    D.files[stem] = json;
    if (json.scenes === undefined) continue;
    if (!Array.isArray(json.scenes)) throw new Error(`trace file ${stem}.json: "scenes" must be an array`);
    for (const s of json.scenes) {
      if (!s || typeof s.id !== 'string') throw new Error(`trace file ${stem}.json: a scene has no string "id"`);
      if (D.scenes[s.id]) throw new Error(`scene id "${s.id}" appears in more than one trace file`);
      D.scenes[s.id] = s;
    }
  }
  return D;
}

export const stemOf = (file) => file.replace(/\.json$/, '');

export async function loadTrace(files, baseUrl, fetchFn = fetch) {
  const loaded = {};
  await Promise.all(files.map(async (f) => {
    const res = await fetchFn(new URL('data/' + f, baseUrl));
    if (!res.ok) throw new Error(`data/${f}: HTTP ${res.status}`);
    loaded[stemOf(f)] = await res.json();
  }));
  // keep the order the lesson listed them in
  return buildD(Object.fromEntries(files.map((f) => [stemOf(f), loaded[stemOf(f)]])));
}

// Path syntax:  "<sceneId>.<nodeName>.<value|grad>[.<index>]"   e.g. reuse.x.grad   mlp.W1.value.2
//               otherwise a path descends into the data: in arrays a segment is an index or the "name" of an item.
//               "files.<stem>.<key>..." reaches the non-scene tables of a trace file.
// A tensor {shape, values} with one element resolves to that number; with several it needs an index.
export function resolvePath(D, path) {
  const segs = String(path).split('.');
  let cur = D, i = 0;
  if (D.scenes[segs[0]]) { cur = D.scenes[segs[0]]; i = 1; }
  for (; i < segs.length; i++) {
    const s = segs[i];
    if (cur === null || typeof cur !== 'object') throw new Error(`T("${path}"): cannot descend into "${s}"`);
    if (Array.isArray(cur)) {
      const item = /^\d+$/.test(s) ? cur[Number(s)] : cur.find((x) => x && x.name === s);
      if (item === undefined) throw new Error(`T("${path}"): no item "${s}"`);
      cur = item;
    } else if (Array.isArray(cur.nodes) && cur.nodes.some((n) => n.name === s)) cur = cur.nodes.find((n) => n.name === s);   // scene -> node by name
    else if (Object.hasOwn(cur, s)) cur = cur[s];
    else if (Array.isArray(cur.values) && /^\d+$/.test(s) && Number(s) < cur.values.length) cur = cur.values[Number(s)];   // tensor -> element
    else throw new Error(`T("${path}"): no "${s}" (have: ${Object.keys(cur).join(', ')})`);
  }
  return cur;
}

function toNumber(path, v) {
  if (typeof v === 'number') return v;
  if (v && Array.isArray(v.values)) {
    if (v.values.length === 1) return v.values[0];
    throw new Error(`T("${path}"): tensor has ${v.values.length} elements; add an index, e.g. ${path}.values.0`);
  }
  if (v === null) throw new Error(`T("${path}"): value is null (for example the root node has no stored gradient)`);
  throw new Error(`T("${path}"): does not resolve to a number (got ${typeof v})`);
}

// T = makeT(D). Every form throws if a path is missing, so prose can never drift from the data silently.
//   T(path)                       legacy, untagged: '<span class="num" data-trace="reuse.x.grad">7</span>' (fmt, no comma). Lessons
//                                 with template: 3 must not use it (tested).
//   T(path, role, {sig}?)         role setup | worked | reveal | contrast (also T(path, {role, sig})):
//                                 '<span class="num" data-trace="…" data-role="setup" data-sig="exact">26,506</span>'
//   T.anchor(name, {sig}?)        a course-wide anchor: data-trace="anchors.<name>" data-role="anchor" data-anchor="<name>"
//   T.human(path, kind, {of, sig}?)  a loss or a count in a human unit (core/numbers.js): data-role="human" data-human data-of
//   T.num(path) / T.text(path)    the bare number / fmt text (for formulas, code and attributes)
// The tests look for any <span class="num"> without data-trace (a number someone typed by hand) and re-check each bound one with
// formatBinding (core/numbers.js), reading the data-* attributes.
const attr = (k, v) => (v === undefined || v === null ? '' : ` data-${k}="${escapeHtml(String(v))}"`);
const span = (a, text) => `<span class="num"${attr('trace', a.path)}${attr('role', a.role)}${attr('sig', a.sig)}${attr('human', a.human)}${attr('of', a.of)}${attr('anchor', a.anchor)}>${text}</span>`;

export function makeT(D) {
  const T = (path, role, opts = {}) => {
    if (role === undefined) return `<span class="num" data-trace="${escapeHtml(path)}">${fmt(T.num(path))}</span>`;
    if (role && typeof role === 'object') { opts = role; role = role.role; }
    if (role === 'anchor') throw new Error(`T("${path}", "anchor"): use T.anchor(name) with a name from site/anchors.json`);
    if (role === 'human') throw new Error(`T("${path}", "human"): use T.human(path, kind, {of})`);
    if (!ROLES.includes(role)) throw new Error(`T("${path}"): unknown role "${role}" (use ${ROLES.join(', ')})`);
    const v = T.num(path), sig = sigFor(role, v, opts.sig);
    return span({ path, role, sig }, proseNumber(v, sig));
  };
  T.num = (path) => toNumber(path, resolvePath(D, path));
  T.text = (path) => fmt(T.num(path));
  T.anchor = (name, opts = {}) => {
    if (!D.anchors) throw new Error(`T.anchor("${name}"): the anchors are not loaded (data/anchors.json)`);
    if (!Object.hasOwn(D.anchors, name)) throw new Error(`T.anchor("${name}"): no such anchor (have: ${Object.keys(D.anchors).join(', ')}); add it to site/anchors.json and run node tools/anchors.mjs`);
    const path = `anchors.${name}`, v = T.num(path), sig = sigFor('anchor', v, opts.sig);
    return span({ path, role: 'anchor', sig, anchor: name }, proseNumber(v, sig));
  };
  T.human = (path, kind, opts = {}) => {
    if (!HUMAN_KINDS.includes(kind)) throw new Error(`T.human("${path}"): unknown kind "${kind}" (use ${HUMAN_KINDS.join(', ')})`);
    if (opts.of !== undefined && !['pct', 'per100'].includes(kind)) throw new Error(`T.human("${path}", "${kind}"): "of" is only for pct and per100`);
    if (opts.sig !== undefined && kind !== 'rmse') throw new Error(`T.human("${path}", "${kind}"): percentages are whole numbers, sig is only for rmse`);
    const sig = kind === 'rmse' ? sigFor('human', 0.5, opts.sig) : undefined;
    const text = humanText(kind, T.num(path), opts.of === undefined ? undefined : T.num(opts.of), sig);
    return span({ path, role: 'human', sig, human: kind, of: opts.of }, text);
  };
  // the text a parsed binding {path, role, sig, human, of} must show (tests)
  T.reformat = (b) => (b.role ? formatBinding(b, T.num) : T.text(b.path));
  return T;
}
