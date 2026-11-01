// Lesson backprop (template 3): every claim that the prose, the quizzes and the widgets make, checked from the real TorC runs
// (data/autograd_trace.json scenes `reuse` and `net`, data/why-gradients.json scene `why` and `nudge`), the pure helpers of the
// nudge counter and the small-network widget, the graph-lab layout of scene `reuse` (both drawings, D16), the phone hint of the
// math window, and that the widgets import without a DOM. The v3 rules (roles, budgets, glossary, notation order) run on this
// lesson in lessons.test.mjs.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { buildGraph, runBackward } from '../js/engine.js';
import { renderLesson } from '../js/core/sections.js';
import { costModels, gridFor, thousands } from '../js/widgets/backprop-cost.js';
import { netValues, STAGES, fmt3 } from '../js/widgets/blame-net.js';
import { layoutFor } from '../js/widgets/graph-lab.js';
import { startHint } from '../js/widgets/math-window.js';
import { NODE_W, NODE_H } from '../js/widgets/graph.js';
import { reuseLayout } from '../js/lessons/backprop.scenes.js';
import lesson from '../js/lessons/backprop.js';
import { readJson, course, proseProblems, uiStringProblems } from './helpers.mjs';

const auto = readJson('data/autograd_trace.json'), why = readJson('data/why-gradients.json'), timing = readJson('data/backprop-timing.json');
const D = buildD({ autograd_trace: auto, 'why-gradients': why, 'backprop-timing': timing }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const tol = (a, b, t = 1e-5) => Math.abs(a - b) <= t * (1 + Math.abs(b));
const node = (scene, name) => D.scenes[scene].nodes.find((n) => n.name === name);
const val = (scene, name) => node(scene, name).value.values;
const grad = (scene, name) => node(scene, name).grad.values;

test('the lesson reads the three data files it declares, and course.json agrees', () => {
  const entry = course().parts.flatMap((p) => p.lessons).find((l) => l.id === 'backprop');
  assert.deepEqual(lesson.trace, entry.trace);
  assert.equal(lesson.title, entry.title);
  assert.equal(entry.status, 'published');
  assert.deepEqual(auto.scenes.map((s) => s.id), ['reuse', 'chain', 'mlp', 'net']);
  assert.ok(D.scenes.net.finite_difference.leaves.every((l) => l.ok), 'the backward pass of scene net agrees with finite differences');
});

// ---- 1 · the cost of nudging ------------------------------------------------------------------------------------------------
test('nudge counter: three timed networks, the first is the digit network, nudging needs one forward pass more than its parameters', () => {
  const m = costModels(D);
  assert.equal(m.length, 3);
  assert.equal(m[0].parameters, D.anchors.mlp_parameters, 'the first timed network has the digit network\'s parameter count');
  for (const x of m) { assert.equal(x.nudgePasses, x.parameters + 1); assert.ok(x.backpropCost > 1 && x.backpropCost < 20); }
  assert.equal(thousands(26507), '26,507');
});

test('timing file: structure, ratios that agree with its own times, and the derived values the prose binds', () => {
  assert.equal(timing.kind, 'timing');
  for (const r of timing.runs) {
    assert.ok(Math.abs(r.backprop_over_nudge - r.forward_backward_us / r.forward_nograd_us) < 1e-9);
    assert.ok(Math.abs(r.nudging_over_backprop - (r.parameters + 1) * r.forward_nograd_us / r.forward_backward_us) < 1e-6 * r.nudging_over_backprop);
    assert.ok(r.forward_backward_us > r.forward_us, 'a backward pass takes time');
    assert.ok(r.nudging_over_backprop > 100, 'backpropagation is far cheaper than nudging for every timed network');
  }
  const d = timing.derived, ratios = timing.runs.map((r) => r.backprop_over_nudge);
  assert.equal(d.backprop_over_nudge_min, Math.min(...ratios));
  assert.equal(d.backprop_over_nudge_max, Math.max(...ratios));
  assert.ok(Math.abs(d.largest_over_digit - timing.runs[2].parameters / timing.runs[0].parameters) < 1e-9);
  // "a forward and a backward pass together cost as much as a few nudges, whatever the size": the ratio does not grow with size
  assert.ok(d.backprop_over_nudge_max / d.backprop_over_nudge_min < 3, 'the cost in nudges stays within a factor 3 over a 25-fold size range');
});

test('nudge counter: the grid holds every pass, fits its box at phone and laptop width, and scales pixels when needed', () => {
  for (const w of [300, 356, 640]) for (const n of [2, 10, ...timing.runs.map((r) => r.parameters + 1)]) {
    const g = gridFor(n, w);
    assert.ok(g.cols * g.rows >= g.cells && g.cells * g.per >= n, `${n} passes fit`);
    assert.ok(g.cols * (g.cell + g.gap) - g.gap <= w, `the grid is not wider than ${w}px`);
    assert.ok(g.height <= 230, `the grid is not much taller than its box (${g.height}px)`);
  }
  assert.equal(gridFor(D.anchors.mlp_parameters + 1, 356).per, 1, 'the digit network is drawn one pass per square');
});

// ---- 2 · the chain rule -------------------------------------------------------------------------------------------------------
test('chain rule: each link changes by its ratio times the change before it, and the product of the ratios is the gradient', () => {
  const s = why.nudge.stages;
  assert.deepEqual(s.map((x) => x.name), ['p', 'e', 'L'], 'prediction, miss, loss');
  assert.ok(tol(s[0].change / why.nudge.delta, s[0].local_slope, 1e-5), 'the prediction rises by 2 times the nudge');
  assert.ok(tol(s[1].change / s[0].change, s[1].local_slope, 1e-5), 'the miss rises by 1 times the change of the prediction');
  assert.ok(Math.abs(s[2].change / s[1].change - s[2].local_slope) < 0.03, 'the loss changes by about -12 times the change of the miss');
  assert.equal(s[0].local_slope * s[1].local_slope * s[2].local_slope, why.nudge.local_slope_product);
  assert.equal(why.nudge.local_slope_product, grad('why', 'w')[0], 'the backward pass of TorC gives the same product');
  assert.equal(why.nudge.local_slope_product, why.probes[0].grad, 'the same gradient as by nudging w directly (the lesson why-gradients)');
});

test('C7: the gradient of the miss is twice the miss, and TorC gets it as two uses that each send the miss back', () => {
  const e = val('why', 'e')[0], eg = grad('why', 'e')[0];
  assert.equal(eg, 2 * e);
  const bw = runBackward(buildGraph(D.scenes.why));
  const sends = bw.steps.flatMap((st) => st.sends).filter((x) => x.to === 'e' && x.tracked);
  assert.equal(sends.length, 2, 'L = e·e uses e twice');
  for (const x of sends) assert.ok(tol(x.tensor.v[0], e * 1), 'in a product each factor gets the other factor times the arriving gradient (one)');
  assert.equal(val('why', 'p')[0] - why.setup.t, e, 'the miss is the prediction minus the price');
});

// ---- 3 · one value, several uses -----------------------------------------------------------------------------------------------
test('reuse: x gets three gradients (x, x and the gradient of m) and their sum is the gradient of x', () => {
  const bw = runBackward(buildGraph(D.scenes.reuse));
  const sends = bw.steps.flatMap((st) => st.sends).filter((x) => x.to === 'x' && x.tracked).map((x) => x.tensor.v[0]).sort((a, b) => b - a);
  const x = val('reuse', 'x')[0], mg = grad('reuse', 'm')[0];
  assert.deepEqual(sends, [x, x, mg]);
  const u = auto.reuse_uses;
  assert.deepEqual([u.mul_first, u.mul_second, u.add], [x, x, 1], 'TorC, with one leaf per use, gives each use its own gradient');
  assert.equal(u.mul_first + u.mul_second + u.add, grad('reuse', 'x')[0]);
  assert.ok(tol(sends.reduce((a, b) => a + b, 0), grad('reuse', 'x')[0]));
  assert.equal(grad('reuse', 'x')[0], 2 * x + 1, 'the sum rule of calculus, d(x² + x)/dx = 2x + 1');
});

test('reuse: the layout binds its numbers, and both drawings place every node inside the view (D16)', () => {
  const L = reuseLayout(T);
  assert.match(L.takeaway, /data-trace="reuse\.x\.grad"/);
  for (const k of ['goal', 'intro', 'forwardHint', 'backwardHint', 'takeaway']) assert.deepEqual(proseProblems(`<p>${L[k]}</p>`, { tics: true }), [], k);
  for (const v of [L, { ...L, ...L.narrow }]) {
    for (const n of D.scenes.reuse.nodes) {
      const p = v.pos[n.name];
      assert.ok(p, `${n.name} has a position`);
      assert.ok(p[0] - NODE_W / 2 >= 0 && p[0] + NODE_W / 2 <= v.view[0], `${n.name} fits the width`);
      assert.ok(p[1] - NODE_H / 2 - 30 >= 0 && p[1] + NODE_H / 2 + 40 <= v.view[1], `${n.name} and its labels fit the height`);
    }
  }
  assert.ok(L.narrow.view[0] <= 356 && L.narrow.minWidth <= 356, 'the phone drawing fits a 390 px screen at full size');
});

test('graph lab (D16): the phone drawing is used only below the wide layout\'s width, and the start hint fits the device', () => {
  const L = reuseLayout(T);
  assert.equal(layoutFor(L, 900), L);
  assert.equal(layoutFor(L, 0), L, 'an unmeasured box keeps the wide layout');
  assert.deepEqual(layoutFor(L, 356).pos, L.narrow.pos);
  const plain = { view: [10, 10], minWidth: 600, pos: {} };
  assert.equal(layoutFor(plain, 356), plain, 'a layout without a narrow drawing scrolls');
  assert.doesNotMatch(startHint(true), /→/);
  assert.match(startHint(false), /→/);
});

// ---- 4 · the small network ------------------------------------------------------------------------------------------------------
test('net: one sample, both neurons on, and the loss is the miss squared', () => {
  const N = netValues(D.scenes.net);
  assert.deepEqual(node('net', 'x').value.shape, [1, 2], 'one sample of two inputs');
  assert.ok(N.h.every((h) => h > 0.1), 'both neurons are on, far from the ReLU kink');
  assert.ok(tol(N.L, N.miss * N.miss));
  assert.ok(tol(N.miss, N.o - N.t));
});

test('net: the rules the text and the widget state hold for every recorded gradient', () => {
  const N = netValues(D.scenes.net);
  assert.ok(tol(N.gO, 2 * N.miss), 'the gradient of the prediction is twice the miss');
  for (let j = 0; j < 2; j++) assert.ok(tol(N.gH[j], N.gO * N.W2[j]), `neuron ${j + 1}: the prediction's gradient times its output weight`);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) assert.ok(tol(N.gW1[i * 2 + j], N.x[i] * N.gH[j]), `weight ${i}→${j}: its neuron's gradient times its input`);
  for (let j = 0; j < 2; j++) assert.ok(tol(N.gW2[j], N.a[j] * N.gO), 'an output weight: the prediction\'s gradient times its input');
  assert.ok(tol(N.gB2, N.gO) && N.gB1.every((g, j) => tol(g, N.gH[j])), 'a bias carries the input 1');
  // worked example and your turn
  assert.ok(tol(grad('net', 'h')[1], T.num('net.o.grad') * T.num('net.W2.value.1')));
  assert.ok(tol(T.num('net.W1.grad.3'), T.num('net.x.value.1') * T.num('net.h.grad.1')));
  // task 1: the second neuron's gradient is positive, because its output weight is negative
  assert.ok(N.W2[1] < 0 && N.gH[1] > 0 && N.gH[0] < 0);
  // task 2: the weights from the second input get twice the gradient of the weights from the first input
  assert.equal(N.x[1], 2 * N.x[0]);
  for (let j = 0; j < 2; j++) assert.ok(tol(N.gW1[2 + j], 2 * N.gW1[j]));
  // section 5: the two weights the text compares get different gradients, of opposite sign
  assert.ok(T.num('net.W1.grad.2') < 0 && T.num('net.W1.grad.1') > 0);
  assert.equal(new Set(N.gW1.map((g) => g.toFixed(4))).size, 4, 'the four first-layer weights get four different gradients');
});

test('net: transfer quiz, a neuron whose sum is negative passes no gradient to its weights (engine, checked against TorC)', () => {
  const g = buildGraph(D.scenes.net, { x: [-2, 0] });
  const h = g.byName.h.value.v;
  assert.ok(h[0] < -0.1 && h[1] > 0.1, 'the first neuron is off, the second is on, both far from the kink');
  const bw = runBackward(g);
  assert.deepEqual([bw.grad.W1.v[0], bw.grad.W1.v[2]].map(Math.abs), [0, 0], 'the two weights into the first neuron get zero');
  assert.equal(Math.abs(bw.grad.b1.v[0]), 0);
});

test('net: no gradient that the lesson quotes equals the miss or minus the miss (review H4)', () => {
  const miss = T.num('net.d.value');
  for (const p of ['net.o.grad', 'net.h.grad.1', 'net.W1.grad.3', 'net.W1.grad.2', 'net.W1.grad.1'])
    assert.ok(Math.abs(Math.abs(T.num(p)) - Math.abs(miss)) > 0.1, `${p} is not ± the miss`);
});

test('accumulation (TorC, recorded): a second backward() on the same loss changes nothing, a new forward pass and backward() add', () => {
  const a = auto.accumulation;
  assert.equal(a.after_first_backward, 2 * a.w, 'dL/dw = 2w for L = w·w');
  assert.equal(a.after_second_backward_same_loss, a.after_first_backward, 'backward_with_grad clears the root\'s tape');
  assert.equal(a.after_new_forward_and_backward, 2 * a.after_first_backward, 'the new gradient adds to the stored one');
  assert.equal(a.has_grad_after_zero_grad, false);
});

test('net: the cumulative quiz counts the parameters as four weights and two biases, then two weights and one bias', () => {
  const c = auto.net_counts;
  assert.equal(c.parameters, c.inputs * c.hidden + c.hidden + c.hidden * c.outputs + c.outputs);
  assert.equal(T.num('files.autograd_trace.net_counts.parameters'), 9);
});

test('blame-net widget: stage captions and labels follow the writing rules, and readouts have 2 to 3 significant figures', () => {
  for (const s of STAGES) { assert.deepEqual(uiStringProblems(s.caption), [], s.caption); assert.deepEqual(proseProblems(`<p>${s.caption}</p>`, { tics: true }), []); }
  assert.equal(fmt3(-2.25), '−2.25');
  assert.equal(fmt3(0.5625), '0.563');
  assert.equal(fmt3(1.25), '1.25');
});

// ---- the page --------------------------------------------------------------------------------------------------------------------
test('the page introduces ∂ once, keeps the overline out of the prose, and says why TorC adds gradients', () => {
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.equal((html.match(/means the same as d/g) ?? []).length, 1);
  assert.doesNotMatch(html, /\\bar|\\overline|̄/, 'the overline notation stays in the graph-lab windows');
  assert.match(html, /After a new forward pass, <code>L\.backward\(\)<\/code> adds the new gradient/);
  assert.doesNotMatch(html, /second <code>L\.backward\(\)<\/code> adds/, 'a second backward() on the same loss does nothing in TorC');
  assert.match(html, /L\.backward\(\)/);
  assert.doesNotMatch(html, /w\.grad\(\)|requires_grad/, 'no C++ in the concept text');
});

test('the widgets and the lesson import without a DOM, and the lesson source has no LaTeX escape bug', () => {
  const src = readFileSync(new URL('../js/lessons/backprop.js', import.meta.url), 'utf8');
  assert.doesNotMatch(src, /(?<!\\)\\[;,!: ]/);
  assert.equal(typeof lesson.footer, 'function');
  assert.match(lesson.footer(D), /TorC timing/);
});
