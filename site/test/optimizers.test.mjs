// Lesson optimizers: the claims of the page against what TorC recorded (data/optimizers.json), the update rules recomputed from
// the recorded gradients, every number bound to the trace, no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/optimizers.json');
const D = buildD({ optimizers: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/optimizers.js')).default;
const V = data.valley;
const run = (id) => V.runs.find((r) => r.id === id);
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol * (1 + Math.abs(b)), `${msg}: ${a} vs ${b}`);
const lossAt = (w1, w2) => 0.5 * V.curvature[0] * w1 * w1 + 0.5 * V.curvature[1] * w2 * w2;
const grad = (q) => [V.curvature[0] * q.w1, V.curvature[1] * q.w2];

test('the valley: every recorded loss is 0.5 w1² + 50 w2², on the paths and on the grid', () => {
  assert.deepEqual(V.curvature, [1, 100]);
  for (const r of V.runs) {
    assert.equal(r.path.length, V.steps + 1);
    assert.deepEqual([r.path[0].w1, r.path[0].w2].map((v) => Math.round(v * 10) / 10), V.start.map((v) => Math.round(v * 10) / 10));
    r.path.forEach((q, k) => close(q.loss, lossAt(q.w1, q.w2), 1e-5, `${r.id} step ${k}`));
  }
  V.surface.loss.forEach((row, j) => row.forEach((L, i) => close(L, lossAt(V.surface.w1[i], V.surface.w2[j]), 1e-5, `grid ${i},${j}`)));
});

test('SGD, momentum and Adam follow their update rules, recomputed from the recorded positions', () => {
  const sgd = run('sgd'), mom = run('momentum'), adam = run('adam');
  for (let k = 0; k < V.steps; k++) {
    const g = grad(sgd.path[k]);
    close(sgd.path[k + 1].w1, sgd.path[k].w1 - sgd.lr * g[0], 1e-5, `sgd w1 ${k}`);
    close(sgd.path[k + 1].w2, sgd.path[k].w2 - sgd.lr * g[1], 1e-5, `sgd w2 ${k}`);
  }
  let v = [0, 0];
  for (let k = 0; k < V.steps; k++) {
    const g = grad(mom.path[k]);
    v = v.map((x, i) => mom.momentum * x + g[i]);
    close(mom.path[k + 1].w1, mom.path[k].w1 - mom.lr * v[0], 1e-4, `momentum w1 ${k}`);
    close(mom.path[k + 1].w2, mom.path[k].w2 - mom.lr * v[1], 1e-4, `momentum w2 ${k}`);
  }
  let m = [0, 0], s = [0, 0];
  for (let k = 0; k < V.steps; k++) {
    const g = grad(adam.path[k]), t = k + 1;
    m = m.map((x, i) => 0.9 * x + 0.1 * g[i]);
    s = s.map((x, i) => 0.999 * x + 0.001 * g[i] * g[i]);
    const up = m.map((x, i) => (x / (1 - 0.9 ** t)) / (Math.sqrt(s[i] / (1 - 0.999 ** t)) + 1e-8));
    close(adam.path[k + 1].w1, adam.path[k].w1 - adam.lr * up[0], 1e-4, `adam w1 ${k}`);
    close(adam.path[k + 1].w2, adam.path[k].w2 - adam.lr * up[1], 1e-4, `adam w2 ${k}`);
  }
  // the first Adam step moves both parameters by lr
  close(adam.path[1].w1 - adam.path[0].w1, adam.lr, 1e-5, 'adam first step w1');
  close(adam.path[0].w2 - adam.path[1].w2, adam.lr, 1e-5, 'adam first step w2');
});

test('the race: Adam < momentum < plain SGD after the last step, and SGD is still far along the valley', () => {
  const end = (id) => run(id).path.at(-1).loss;
  assert.ok(end('adam') < end('momentum') && end('momentum') < end('sgd'), `${end('adam')} ${end('momentum')} ${end('sgd')}`);
  assert.ok(run('sgd').path.at(-1).w1 < -1);
  assert.ok(run('sgd').lr < data.derived.sgd_lr_limit);
  assert.equal(data.derived.sgd_lr_limit, 2 / V.curvature[1]);
});

test('first step: Adam moves each parameter by about lr, SGD by lr times the gradient', () => {
  const [sgd, adam] = data.first_step;
  assert.deepEqual(adam.grad.map((g) => Math.fround(g)), [Math.fround(0.001), 1000]);
  adam.change.forEach((c) => close(c, -adam.lr, 1e-4, 'adam change'));
  sgd.change.forEach((c, i) => close(c, -sgd.lr * sgd.grad[i], 1e-3, 'sgd change'));
});

test('weight decay: AdamW multiplies a zero-gradient weight by 1 - lr·λ each step, Adam keeps it, and step() skips a parameter with no gradient', () => {
  const [adam, adamw] = data.decay;
  const f = 1 - data.decay_setup.lr * data.decay_setup.weight_decay;
  assert.equal(data.derived.decay_factor, f);
  for (const run of [adam, adamw]) run.steps.slice(0, -1).forEach((s, k) => {
    // read between backward() and step(): w has a gradient (its w2 part is exactly zero), u has none
    assert.equal(s.w_has_grad, true, `${run.optimizer} step ${k}`);
    assert.equal(s.w2_grad, 0);
    assert.equal(s.u_has_grad, false);
  });
  adam.steps.forEach((s) => { assert.equal(s.w2, 3); assert.equal(s.u, 3); });
  adamw.steps.forEach((s, k) => { close(s.w2, 3 * f ** k, 1e-5, `adamw w2 step ${k}`); assert.equal(s.u, 3, 'no gradient, no decay'); });
  close(adamw.steps.at(-1).w2, data.derived.w2_after_50_adamw, 1e-5, 'derived matches');
});

test('every number in the prose is bound to the trace, with no hand-typed digits', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<div class="op-eq"[\s\S]*?<\/div>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>')
    .replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/optimizer-widgets.js');
  assert.equal(typeof m.mountOptimizerRace, 'function');
  assert.deepEqual(m.RUNS.map((r) => r[0]), V.runs.map((r) => r.id));
  assert.equal(typeof document, 'undefined');
});
