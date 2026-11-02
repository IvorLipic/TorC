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

test('the order of parameters(): inside one module it is the map\'s, across modules it is the order of add()', () => {
  const o = data.parameter_order;
  assert.deepEqual([...o.linear_names].sort(), ['bias', 'weight']);
  assert.equal(o.linear_named_count, 2);
  assert.equal(o.sequential_named_count, 0, 'a Sequential has no named parameters of its own');
  // the recorded Sequential lists the two Linear layers in add() order: the first has 32 neurons and 784 inputs, the second 10 and 32
  const sorted = (a, b) => JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
  assert.ok(sorted(o.sequential_shapes.slice(0, 2), [[32], [32, 784]]));
  assert.ok(sorted(o.sequential_shapes.slice(2, 4), [[10], [10, 32]]));
  // the prose names the first entry and the weight that follows it: it must say what was recorded
  const all = renderLesson(lesson, D).map((p) => p.html).join('\n');
  const reveal = all.match(/In one run of TorC, the first entry has the shape <code>\(([^)]*)\)<\/code>\. That is a bias, and the weight has the shape <code>\(([^)]*)\)<\/code>/);
  assert.ok(reveal, 'the reveal of p-first names the recorded shapes');
  assert.equal(JSON.stringify(o.sequential_shapes[0]), JSON.stringify(reveal[1].split(', ').map(Number)));
  assert.equal(o.sequential_shapes[0].length, 1, 'a bias has one dimension');
  assert.equal(o.sequential_shapes[1].length, 2, 'a weight has two');
});

test('a module of our own: a local in-between Variable breaks backward(), forward_cache_ keeps it alive', () => {
  const [local, cached] = data.custom_module;
  assert.equal(local.id, 'local');
  assert.equal(local.exception, 'TorcError');
  assert.match(local.what, /^autograd graph input expired before backward\(\)/);
  assert.equal(cached.exception, 'none');
  assert.deepEqual([cached.scale_has_grad, cached.shift_has_grad], [true, true]);
  const all = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.match(all, /forward_cache_\.emplace_back\(mul\(x, named_parameters\(\)\.at\("scale"\)\)\)/);
});

test('every claim and message of the prose against the data, and every number bound to the trace', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const b of boundNums(html)) assert.equal(T.reformat(b), b.shown, b.path);
  }
  const all = parts.map((p) => p.html).join('\n');
  assert.match(all, /TorcError: autograd graph input expired before backward\(\); keep all graph Variables alive/);
  assert.match(all, /ShapeError: matmul inner dimensions must match, got \(8, 32\) and \(64, 10\)/);
  // the quiz answers
  assert.equal(byId('mlp').total, 25450);
  assert.deepEqual([byId('mlp').layers[0].params.map((p) => p.numel), byId('mlp').layers[2].params.map((p) => p.numel)], [[32, 25088], [10, 320]]);
  const shown = Object.fromEntries(boundNums(all).map(({ path, shown: v }) => [path, v]));
  assert.equal(shown['files.modules.models.0.total'], '25,450');
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/module-widgets.js');
  assert.equal(typeof m.mountModelViewer, 'function');
  assert.equal(typeof m.mountCallOrder, 'function');
  assert.equal(typeof document, 'undefined');
});
