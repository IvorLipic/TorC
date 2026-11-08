// Lesson one-step ("One training step, followed end to end", template 3): every claim of the page checked from what TorC recorded
// (data/one-step.json, tools/trace/trace_one_step.cpp; the weights come from data/mnist-mlp.json), a JS port of the forward pass
// against TorC's recorded values, the widget stages, and no hand-typed digits. The v3 rules run on this lesson in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { humanValue } from '../js/core/numbers.js';
import { readJson, course } from './helpers.mjs';

const O = readJson('data/one-step.json');
const M = readJson('data/mnist-mlp.json');
const D = buildD({ 'one-step': O }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const mod = await import('../js/lessons/one-step.js');
const lesson = mod.default;
const { linear, softmax, argmax } = await import('../js/widgets/digit-widgets.js');
const { stages, s3 } = await import('../js/widgets/one-step-widgets.js');
const close = (a, b, tol = 1e-5, msg = '') => assert.ok(Math.abs(a - b) <= tol * (1 + Math.abs(b)), `${msg} ${a} vs ${b}`);
const html = (() => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
})();
const relu = (v) => v.map((x) => (x > 0 ? x : 0));
const [first, outTrue, outRival, bias] = O.weights;

test('the setup: a training image of a one, the trained network, plain SGD with a batch of one image', () => {
  assert.equal(O.image.set, 'train', 'a training image, not a test image (C2)');
  assert.equal(O.image.label, 1);
  assert.deepEqual([O.setup.optimizer, O.setup.momentum, O.setup.batch_size], ['optim::SGD', 0, 1]);
  assert.equal(O.setup.parameters, D.anchors.mlp_parameters);
  assert.equal(O.full_run.batch_size, M.setup.batch_size);
  assert.equal(O.full_run.optimizer, 'optim::AdamW');
  assert.equal(O.full_run.steps, M.settings.steps, 'the real run took this many steps');
  assert.equal(O.full_run.steps, O.full_run.steps_per_epoch * O.full_run.epochs, 'a step for every batch of every epoch');
  assert.ok(Object.values(O.checks).every(Boolean), JSON.stringify(O.checks));
  assert.deepEqual(O.weights.map((w) => w.id), ['first_layer', 'output_true', 'output_rival', 'output_bias_true']);
  assert.deepEqual([mod.FIRST, mod.TRUE, mod.RIVAL], [0, 1, 2]);
});

test('the forward pass: a JS port with the trained weights gives the values TorC recorded', () => {
  const x = O.image.pixels.map((p) => p / 255);
  const h1 = linear(M.layers[0], x), a1 = relu(h1), h2 = linear(M.layers[1], a1), a2 = relu(h2), z = linear(M.layers[2], a2);
  h1.forEach((v, i) => close(v, O.forward.hidden1.pre[i], 1e-4, `h1 ${i}`));
  h2.forEach((v, i) => close(v, O.forward.hidden2.pre[i], 1e-4, `h2 ${i}`));
  z.forEach((v, i) => close(v, O.forward.scores[i], 1e-4, `score ${i}`));
  const p = softmax(z);
  p.forEach((v, i) => close(v, O.forward.probabilities[i], 1e-4, `p ${i}`));
  // section 1: the one is most probable, a three comes second; the surprise is -ln of the probability of the one
  assert.equal(argmax(z), 1);
  assert.equal(O.forward.rival, 3);
  const sorted = [...O.forward.probabilities].sort((a, b) => b - a);
  assert.equal(sorted[1], O.forward.probabilities[3]);
  close(O.forward.loss, -Math.log(O.forward.p_true), 1e-6, 'surprise');
  close(O.after.loss, -Math.log(O.after.p_true), 1e-6, 'surprise after');
});

test('the backward pass: probability minus one for the right score, the probability for the others; input times the gradient', () => {
  O.backward.scores_grad.forEach((g, j) => close(g, O.forward.probabilities[j] - (j === 1 ? 1 : 0), 1e-5, `score ${j}`));
  assert.ok(O.backward.scores_grad.every((g, j) => (j === 1 ? g < 0 : g > 0)), 'task 2: negative for the one, positive for every other score');
  assert.equal(Math.max(...O.backward.scores_grad), O.backward.scores_grad[3], 'the largest for the three');
  // the worked example and the "your turn" quiz: the neuron is the most active of the second layer, and the gradient is the product
  assert.equal(outTrue.path.unit, argmax(O.forward.hidden2.post));
  assert.equal(outRival.path.unit, outTrue.path.unit, 'the same neuron');
  assert.equal(outRival.path.class, 3);
  close(outTrue.grad, outTrue.path.unit_post * outTrue.path.score_grad, 1e-6);
  close(outRival.grad, outRival.path.unit_post * outRival.path.score_grad, 1e-6);
    assert.ok(outRival.grad > 0 && outRival.after < outRival.before, 'positive gradient, smaller weight');
  close(bias.grad, O.forward.p_true - 1, 1e-5, 'the bias gets the gradient of its score');
  // section 3: the first-layer weight from the brightest pixel; position counts from zero
  const px = first.path;
  assert.equal(px.pixel, px.pixel_row * 28 + px.pixel_col);
  assert.equal(O.image.pixels[px.pixel], Math.max(...O.image.pixels));
  assert.equal(first.row, px.unit);
  assert.ok(px.grad_equals_product && Math.abs(first.grad - px.pixel_value * px.unit_grad) < 1e-6);
  close(px.unit_grad, O.backward.hidden1_pre_grad[px.unit], 1e-7);
});

test('"A black pixel gets the gradient zero, a neuron that is off passes none, a few changes are too small": about 15 % change', () => {
  const x = O.image.pixels, f = O.forward, g1 = O.backward.hidden1_pre_grad, g2 = O.backward.hidden2_pre_grad;
  // a neuron that ReLU sets to zero passes no gradient back
  f.hidden1.post.forEach((v, i) => { if (v === 0) assert.equal(g1[i], 0); });
  f.hidden2.post.forEach((v, i) => { if (v === 0) assert.equal(g2[i], 0); });
  const nz = (a) => a.filter((v) => v !== 0).length;
  const upper = nz(x) * nz(g1) + nz(g1) + nz(f.hidden1.post) * nz(g2) + nz(g2) + nz(f.hidden2.post) * 10 + 10;
  const C = O.changed_detail;
  assert.ok(C.nonzero_grad <= upper, `${C.nonzero_grad} of at most ${upper}`);
  assert.equal(C.nonzero_grad - C.unchanged_with_nonzero_grad, O.changed_parameters, 'every other nonzero gradient changes its parameter');
  assert.ok(C.unchanged_with_nonzero_grad > 0 && C.unchanged_with_nonzero_grad < 0.02 * C.nonzero_grad, 'a few changes are too small to store');
  assert.ok(nz(x) < 0.25 * x.length, 'most pixels are black');
  assert.equal(Math.round(humanValue('pct', O.changed_parameters, O.setup.parameters)), 15);
  // "one of the brightest pixels": the brightest grey value is shared
  assert.ok(x.filter((v) => v === Math.max(...x)).length > 1);
});

test('the chain rule for two hidden neurons (section 3 and the widget): weights times later gradients, summed, times the ReLU slope', () => {
  const C = O.chain, k2 = outTrue.path.unit;
  assert.equal(C.second.unit, k2, 'the most active neuron of the second layer');
  assert.equal(C.first.unit, first.path.unit, 'the first-layer neuron of the named weight');
  for (const [L, n, Wl, gNext, gHere] of [[C.second, 10, M.layers[2], O.backward.scores_grad, O.backward.hidden2_pre_grad], [C.first, 32, M.layers[1], O.backward.hidden2_pre_grad, O.backward.hidden1_pre_grad]]) {
    assert.equal(L.terms.length, n);
    L.terms.forEach((t, o) => {
      assert.equal(t.to, o);
      assert.equal(Math.fround(t.weight), Math.fround(Wl.weight[o * 32 + L.unit]), 'the trained weight, before the step');
      assert.equal(Math.fround(t.grad), Math.fround(gNext[o]));
    });
    assert.equal(L.relu_slope, 1, 'the neuron is on');
    close(L.sum, gHere[L.unit], 1e-5, 'the sum is the recorded gradient');
    const ranked = L.ranked.map((o) => Math.abs(L.terms[o].product));
    for (let i = 1; i < ranked.length; i++) assert.ok(ranked[i] <= ranked[i - 1]);
    close(L.rest_after_3 + L.ranked.slice(0, 3).reduce((s, o) => s + L.terms[o].product, 0), L.grad, 1e-5, 'three largest parts plus the rest');
  }
  close(C.first.grad, first.path.unit_grad, 1e-7);
  assert.equal(C.first.ranked[0], stages(D)[6].highlight[0], 'the widget names the largest part');
});

test('the step: new = old - lr × gradient exactly, and the size of the step is lr × gradient', () => {
  for (const w of O.weights) {
    assert.ok(w.update_exact, w.id);
    assert.equal(Math.fround(Math.fround(w.before) - Math.fround(w.lr_times_grad)), Math.fround(w.after), w.id);
    assert.equal(Math.fround(Math.fround(w.grad) * Math.fround(O.setup.lr)), Math.fround(w.lr_times_grad), w.id);
  }
  assert.ok(outTrue.lr_times_grad < 0 && outTrue.after > outTrue.before, 'subtracting a negative number adds, so the weight grows');
  assert.equal(T.text('files.one-step.weights.1.lr_times_grad'), '-0.002642');
  // the "your turn" quiz: the tolerance tells the new value from the old one, and accepts a gradient rounded to 2.117 or 2.118
  const q = lesson.sections.find((s) => s.id === 'rival');
  assert.ok(Math.abs(outRival.before - outRival.after) > 3 * q.tolerance);
  for (const g of [2.117, 2.118]) assert.ok(Math.abs(outRival.before - O.setup.lr * g - outRival.after) < q.tolerance, String(g));
});

test('after the step: the one rises clearly but stays below one; its score rises most; every wrong probability falls', () => {
  const A = O.after, F0 = O.forward;
  assert.ok(A.p_true > F0.p_true + 0.1 && A.p_true < 0.9);
  assert.equal(A.p_rival, A.probabilities[3]);
  assert.equal(A.predicted, 1);
  assert.ok(O.changed_parameters > 1000, 'thousands of changes');
  assert.equal(A.scores.length, 10);
  const d = A.scores.map((s, j) => s - F0.scores[j]);
  assert.equal(Math.max(...d), d[1], 'the score of the one rose much more than any other');
  assert.ok(d[1] > 3 * Math.max(...d.filter((_, j) => j !== 1)));
  assert.ok(d[3] < 0, 'the step lowered the score of the three');
  assert.deepEqual(d.map((v, j) => (j !== 1 && v > 0 ? j : -1)).filter((j) => j >= 0), [4, 6, 7], 'a few other wrong scores rose a little');
  A.probabilities.forEach((p, j) => { if (j !== 1) assert.ok(p < F0.probabilities[j], `digit ${j}`); });
  assert.ok(outRival.grad > 0 && outRival.after < outRival.before, 'positive gradient, smaller weight');
});

test('the widget stages: every readout comes from the trace', () => {
  const S = stages(D);
  assert.equal(S.length, 11);
  assert.match(S[0].text, new RegExp(`grey value is ${first.path.pixel_255}, so its input is ${s3(first.path.pixel_value)}\\.`));
  assert.match(S[1].text, new RegExp(`so ${O.forward.hidden1.post.filter((v) => v === 0).length} neurons are off\\. Neuron ${first.path.unit} has the value ${s3(first.path.unit_post)}\\.`));
  assert.match(S[4].text, /The one gets 0\.606, and the three gets 0\.315\./);
  assert.match(S[5].text, /The one gets −0\.394, and the three gets 0\.315\./);
  assert.match(S[6].text, new RegExp(`Neuron ${O.chain.first.ranked[0]} gets ${s3(O.backward.hidden2_pre_grad[O.chain.first.ranked[0]])}\\.`));
  assert.match(S[7].text, new RegExp(`Neuron ${first.path.unit} gets ${s3(first.path.unit_grad)}\\.`));
  assert.match(S[9].text, /The weight to the one goes from 0\.5099 to 0\.5125\. The weight to the three goes from 0\.4185 to 0\.4164\./);
  assert.match(S[10].text, /the one gets 0\.799 in place of 0\.606\. The three gets 0\.156 in place of 0\.315\./);
  assert.deepEqual(S[10].values, O.after.probabilities);
  assert.deepEqual(S[10].compare, O.forward.probabilities);
  close(bias.grad, O.backward.scores_grad[1], 1e-7, 'the bias gets the gradient of its score');
  assert.equal(s3(-0.39352), '−0.394');
});

test('the capstone: every main-path lesson before this one is linked, in course order', () => {
  const before = ['what-is-learning', 'why-gradients', 'sgd-step', 'linear-layer', 'nonlinearity', 'losses', 'backprop', 'deep-nets', 'training-loop', 'optimizers', 'generalization', 'mnist-mlp'];
  for (const id of before) assert.match(html, new RegExp(`href="\\.\\./${id}/"`), id);
});

test('no hand-typed digits in the prose: every number is bound', () => {
  const prose = html.replace(/<style[\s\S]*?<\/style>/g, ' ').replace(/<pre[\s\S]*?<\/pre>/g, ' ').replace(/<svg[\s\S]*?<\/svg>/g, ' ')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, ' ').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, ' ').replace(/<h2[^>]*>\s*\d+ ·/g, '<h2>').replace(/<[^>]+>/g, ' ');
  assert.deepEqual(prose.match(/\S*\d\S*/g), null);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/one-step-widgets.js');
  assert.equal(typeof m.mountOneStep, 'function');
  assert.equal(typeof document, 'undefined');
});
