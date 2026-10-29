// Course-wide anchors (R2): site/anchors.json names trace paths, tools/anchors.mjs copies their values into site/data/anchors.json,
// and lesson-boot.js always loads that file. The committed file must equal what the traces give now.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { computeAnchors, serialize, ANCHOR_NAME, MAX_ANCHORS } from '../../tools/anchors.mjs';
import { anchorValues, buildD, makeT } from '../js/core/trace.js';
import { readJson } from './helpers.mjs';

test('site/data/anchors.json equals the values in the traces (run node tools/anchors.mjs after the traces change)', async () => {
  const committed = readFileSync(new URL('../data/anchors.json', import.meta.url), 'utf8');
  assert.equal(committed, serialize(await computeAnchors()));
});

test('site/anchors.json: at most 15 named trace paths, each resolving to a number', () => {
  const list = readJson('anchors.json');
  const names = Object.keys(list);
  assert.ok(names.length >= 1 && names.length <= MAX_ANCHORS, `${names.length} anchors`);
  for (const n of names) { assert.match(n, ANCHOR_NAME); assert.equal(typeof list[n], 'string'); }
  const values = anchorValues(readJson('data/anchors.json'));
  assert.deepEqual(Object.keys(values), names);
  for (const n of names) assert.ok(Number.isFinite(values[n]), n);
  // spot checks against the trace files, read directly
  assert.equal(values.mlp_parameters, readJson('data/mnist-mlp.json').setup.parameters);
  assert.equal(values.cnn_parameters, readJson('data/mnist-cnn.json').setup.parameters);
  assert.equal(values.train_images, readJson('data/mnist-mlp.json').setup.train_samples);
  assert.equal(values.bowl_w_min, readJson('data/why-gradients.json').setup.w_min);
});

test('T.anchor works in a lesson that loads none of the anchor trace files', () => {
  const T = makeT(buildD({}, anchorValues(readJson('data/anchors.json'))));
  assert.match(T.anchor('mlp_parameters'), />26,506<\/span>$/);
  assert.match(T.anchor('train_images'), />60,000<\/span>$/);
});

test('tools/trace/run.sh runs tools/anchors.mjs after the traces and checks it in --check', () => {
  const sh = readFileSync(new URL('../../tools/trace/run.sh', import.meta.url), 'utf8');
  assert.match(sh, /node "\$ROOT\/tools\/anchors\.mjs" --check/);
  assert.match(sh, /run_all "\$DATA"[\s\S]*node "\$ROOT\/tools\/anchors\.mjs"\n/);
  assert.match(sh, /\[ "\$b" = "anchors\.json" \] && known=1/, 'the drift check knows who writes anchors.json');
});
