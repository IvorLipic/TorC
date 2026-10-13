// core/trace.js — loading trace data (site/data/*.json, written by tools/trace/) and referring to it from prose.
//
// Every trace file is an object whose top-level key "scenes" is an array of scenes {id, nodes[], ...}. Other top-level keys
// (schema, library, kind, generator, tables for other lessons...) are ignored by the loader but stay reachable by T().
// D (what lesson text functions receive) = { files: {<file stem>: json}, scenes: {<scene id>: scene} }.
import { fmt } from '../ops/texutil.js';
import { escapeHtml } from './dom.js';

export function buildD(files) {
  const D = { files: {}, scenes: {} };
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

// T = makeT(D). T('reuse.x.grad') -> '<span class="num" data-trace="reuse.x.grad">7</span>'. Throws if the path is missing.
// The tests look for any <span class="num"> without data-trace (a number someone typed by hand) and re-check each bound one.
export function makeT(D) {
  const T = (path) => `<span class="num" data-trace="${escapeHtml(path)}">${fmt(T.num(path))}</span>`;
  T.num = (path) => toNumber(path, resolvePath(D, path));
  T.text = (path) => fmt(T.num(path));
  return T;
}
