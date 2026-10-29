#!/usr/bin/env node
// tools/anchors.mjs — writes site/data/anchors.json: the value of every course-wide anchor listed in site/anchors.json (R2).
//
//   node tools/anchors.mjs           (re)write site/data/anchors.json from the committed trace files in site/data/
//   node tools/anchors.mjs --check   exit 1 if site/data/anchors.json differs from what the traces give (CI, tools/trace/run.sh --check)
//
// site/anchors.json is hand-written: { "<name>": "<trace path>" }, at most about 15 names, each one used in at least two lessons
// (for example the parameter counts of the two MNIST networks, or the 60,000 training images). A path has the T() syntax of
// site/js/core/trace.js, usually "files.<trace file stem>.<key>…", and must resolve to a number. Values are never typed by hand.
// The output {schema: "torc-anchors/1", generator, anchors: {<name>: {path, value}}} is small, so lesson-boot.js loads it on every
// lesson page and T.anchor(name) works without the 400 KB MNIST trace. site/test/anchors.test.mjs checks that it equals the traces.
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const ANCHOR_NAME = /^[a-z][a-z0-9_]*$/;
export const MAX_ANCHORS = 15;

export async function computeAnchors(siteDir = path.join(ROOT, 'site')) {
  const { buildD, resolvePath, ANCHORS_SCHEMA } = await import(pathToFileURL(path.join(siteDir, 'js/core/trace.js')).href);
  const list = JSON.parse(readFileSync(path.join(siteDir, 'anchors.json'), 'utf8'));
  const files = Object.fromEntries(readdirSync(path.join(siteDir, 'data')).filter((f) => f.endsWith('.json') && f !== 'anchors.json').sort()
    .map((f) => [f.replace(/\.json$/, ''), JSON.parse(readFileSync(path.join(siteDir, 'data', f), 'utf8'))]));
  const D = buildD(files);
  const anchors = {};
  for (const [name, p] of Object.entries(list)) {
    if (!ANCHOR_NAME.test(name)) throw new Error(`site/anchors.json: anchor name "${name}" must be lowercase_with_underscores`);
    if (typeof p !== 'string') throw new Error(`site/anchors.json: "${name}" must map to a trace path string`);
    const v = resolvePath(D, p);
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`site/anchors.json: "${name}" -> ${p} is not a finite number`);
    anchors[name] = { path: p, value: v };
  }
  if (Object.keys(anchors).length > MAX_ANCHORS) throw new Error(`site/anchors.json: ${Object.keys(anchors).length} anchors, at most ${MAX_ANCHORS}`);
  return { schema: ANCHORS_SCHEMA, generator: 'tools/anchors.mjs', source: 'site/anchors.json', anchors };
}

export const serialize = (json) => JSON.stringify(json, null, 2) + '\n';

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const out = path.join(ROOT, 'site/data/anchors.json');
  const want = serialize(await computeAnchors());
  if (process.argv.includes('--check')) {
    let have = '';
    try { have = readFileSync(out, 'utf8'); } catch { /* missing */ }
    if (have !== want) { console.error('site/data/anchors.json is stale or missing; run node tools/anchors.mjs and commit'); process.exit(1); }
    console.log('site/data/anchors.json matches the traces');
  } else {
    writeFileSync(out, want);
    console.log(`wrote site/data/anchors.json (${Object.keys(JSON.parse(want).anchors).length} anchors)`);
  }
}
