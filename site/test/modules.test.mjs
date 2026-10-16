// Lesson modules: the claims of the page against what TorC recorded (data/modules.json), every number bound to the trace,
// no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/modules.json');
const D = buildD({ modules: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/modules.js')).default;
const byId = (id) => data.models.find((m) => m.id === id);

test('parameter counts: in*out + out per Linear, none for ReLU, Sequential::parameters() collects them all', () => {
  for (const m of data.models) {
    let total = 0;
    for (const l of m.layers) {
      const lin = /^Linear\((\d+), (\d+)\)$/.exec(l.label);
      if (lin) {
        const [i, o] = [Number(lin[1]), Number(lin[2])];
        assert.deepEqual(l.params.map((p) => [p.name, p.shape]), [['bias', [o]], ['weight', [o, i]]], l.label);
        total += i * o + o;
      } else assert.deepEqual(l.params, [], 'ReLU has no parameters');
    }
    assert.equal(m.total, total, m.id);
    assert.equal(m.sequential_parameters_total, total, `${m.id}: Sequential collects every parameter`);
    assert.equal(m.sequential_parameters_count, m.tensors);
  }
  assert.equal(byId('mlp').total, 25450);
  assert.equal(byId('mnist').total, 26506);
});

test('a size mismatch is found in the forward pass, at the layer that does not fit', () => {
  const m = byId('mismatch');
  const bad = m.layers.findIndex((l) => l.what);
  assert.equal(bad, 2);
  assert.match(m.layers[bad].what, /^matmul inner dimensions must match, got \(8, 32\) and \(64, 10\)$/);
  assert.equal(m.sequential_what, m.layers[bad].what);
  assert.deepEqual(byId('mlp').sequential_out_shape, [8, 10]);
});

test('call order: a second forward before the first backward makes that backward throw; the other orders work', () => {
  const o = Object.fromEntries(data.orders.map((x) => [x.id, x]));
  assert.equal(o['in-order'].exception, 'none');
  assert.equal(o['latest-backward'].exception, 'none');
  assert.equal(o['early-second-forward'].exception, 'TorcError');
  assert.match(o['early-second-forward'].what, /graph input expired before backward/);
  assert.match(o['early-second-forward'].code, /model\(x2\)[\s\S]*L1\.backward\(\)/);
});

test('every number in the prose is bound to the trace, with no hand-typed digits', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const all = parts.map((p) => p.html).join('\n');
  assert.match(all, /TorcError: autograd graph input expired before backward\(\); keep all graph Variables alive/);
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/module-widgets.js');
  assert.equal(typeof m.mountModelViewer, 'function');
  assert.equal(typeof m.mountCallOrder, 'function');
  assert.equal(typeof document, 'undefined');
});
