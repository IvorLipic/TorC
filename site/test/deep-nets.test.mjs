// Lesson deep-nets (template 3): every claim that the prose, the quizzes and the widgets make, checked from the real TorC run
// (data/deep.json), and the pure helpers of the three widgets. The v3 rules (roles, budgets, glossary, notation order) run on this
// lesson in lessons.test.mjs.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { twinView, TWIN_STARTS, vanishingRows, signalSeries, SIGNAL_STARTS, powLabel, fmt2 } from '../js/widgets/deep-widgets.js';
import lesson from '../js/lessons/deep-nets.js';
import { readJson, course } from './helpers.mjs';

const deep = readJson('data/deep.json');
const D = buildD({ deep }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const tol = (a, b, t = 1e-5) => Math.abs(a - b) <= t * (1 + Math.abs(b));
const twin = (id) => deep.twins.find((t) => t.id === id);

test('the lesson reads deep.json, and course.json agrees', () => {
  const entry = course().parts.flatMap((p) => p.lessons).find((l) => l.id === 'deep-nets');
  assert.deepEqual(lesson.trace, entry.trace);
  assert.equal(lesson.title, entry.title);
  assert.equal(entry.status, 'published');
  assert.deepEqual(deep.twins.map((t) => t.id), ['twins', 'different', 'zeros'], 'the indices TWINS and ZEROS of the lesson');
  assert.deepEqual(deep.init.map((r) => r.name), ['kaiming', 'too_small', 'too_large', 'no_two'], 'the indices of init[]');
  assert.deepEqual(deep.vanishing.map((r) => r.activation), ['sigmoid', 'relu']);
  assert.equal(deep.twins[0].long_run[4].step, 2000);
});

// ---- twins -----------------------------------------------------------------------------------------------------------------------
test('twins: the equal start stays equal at every step, and the different start does not', () => {
  assert.equal(twin('twins').equal_every_step, true);
  assert.equal(twin('different').equal_every_step, false);
  for (const r of twin('twins').long_run) assert.deepEqual(r.neuron0, r.neuron1, `step ${r.step}: bit for bit equal`);
  for (const s of twin('twins').steps) assert.deepEqual(s.grad0, s.grad1, `step ${s.step}: equal gradients`);
  const start = twin('twins').start;
  assert.equal(start.W1[0], start.W1[2]); assert.equal(start.W1[1], start.W1[3]); assert.equal(start.b1[0], start.b1[1]); assert.equal(start.W2[0], start.W2[1]);
});

test('twins: the equal start stops at a typical miss of about 0.41, the different start learns all four points', () => {
  const t = twin('twins').long_run, d = twin('different').long_run, target = deep.twins_setup.t.values;
  assert.deepEqual(target, [0, 1, 1, 0], 'the four points: one when exactly one input is one');
  assert.ok(Math.abs(Math.sqrt(t[4].loss) - 0.41) < 0.005);
  assert.ok(tol(t[4].loss, t[3].loss, 1e-6), 'the typical miss stops falling');
  d[4].prediction.forEach((p, i) => assert.ok(Math.abs(p - target[i]) < 0.01, `point ${i} is learned`));
});

test('zeros: no weight ever changes, only the output bias learns, and the network predicts 0.5 for every point', () => {
  const z = twin('zeros');
  assert.equal(z.equal_every_step, true);
  for (const r of z.long_run) for (const n of [r.neuron0, r.neuron1]) assert.deepEqual([...n.w, n.b, n.v], [0, 0, 0, 0], `step ${r.step}`);
  for (const s of z.steps) for (const g of [s.grad0, s.grad1]) assert.deepEqual([...g.w, g.b, g.v].map(Math.abs), [0, 0, 0, 0], 'each neuron gets a gradient of zero');
  const last = z.long_run[4];
  assert.ok(last.prediction.every((p) => Math.abs(p - 0.5) < 1e-6));
  assert.ok(Math.abs(last.output_bias - 0.5) < 1e-6);
  const t = deep.twins_setup.t.values;
  assert.ok(Math.abs(last.output_bias - t.reduce((a, b) => a + b, 0) / t.length) < 1e-6, 'the output bias moves to the mean of the four targets');
  const s0 = z.steps[0];
  assert.deepEqual([...s0.grad0.w, ...s0.grad1.w].map(Math.abs), [0, 0, 0, 0], 'first-layer weights: zero gradient (the output weights are zero)');
  assert.deepEqual([s0.grad0.v, s0.grad1.v].map(Math.abs), [0, 0], 'output weights: zero gradient (ReLU of zero is zero, so their inputs are zero)');
  assert.equal(T(`files.deep.twins.2.long_run.4.prediction.0`, 'contrast').replace(/<[^>]+>/g, ''), '0.5');
});

test('twins widget: the view of each start and step reads the recorded numbers', () => {
  for (const s of TWIN_STARTS) for (let k = 0; k < 5; k++) {
    const v = twinView(deep, s.id, k);
    assert.equal(v.a.length, 4); assert.equal(v.b.length, 4);
    assert.equal(v.equal, v.a.every((x, i) => x === v.b[i]));
    // the drawing: zero line at 118 px, 54 px per unit, value labels 11 px beyond the bar, axis labels from 200 px (H - 22)
    for (const x of [...v.a, ...v.b]) {
      const end = x >= 0 ? 118 - 54 * x - 4 - 10 : 118 - 54 * x + 11;
      assert.ok(end > 18 && end < 200 - 10, `start ${s.id}, step ${v.step}: a bar of ${x} and its label stay clear of the key and the axis labels`);
    }
  }
  assert.equal(fmt2(-0.5171386), '−0.52');
  assert.equal(fmt2(0), '0');
});

// ---- vanishing ------------------------------------------------------------------------------------------------------------------
test('vanishing: with sigmoid the gradient grows at every layer toward the output, with ReLU it stays about level', () => {
  const rows = vanishingRows(deep);
  assert.equal(rows.length, deep.setup.depth);
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i][1] > rows[i - 1][1], `sigmoid layer ${i + 1} > layer ${i}`);
  const relu = rows.map((r) => r[2]);
  assert.ok(Math.max(...relu) / Math.min(...relu) < 3, 'ReLU bars stay within a factor 3');
  assert.ok(tol(deep.vanishing[0].derived.last_over_first_hidden, rows[7][1] / rows[0][1]));
  assert.equal(T('files.deep.vanishing.0.derived.last_over_first_hidden', 'contrast').replace(/<[^>]+>/g, ''), '16,000');
  assert.equal(T('files.deep.vanishing.1.derived.last_over_first_hidden', 'contrast').replace(/<[^>]+>/g, ''), '2');
  assert.equal(deep.setup.sigmoid_max_slope, 0.25);
  // the plot's log axis covers every bar
  for (const [, s, r] of rows) for (const v of [s, r]) assert.ok(Math.log10(v) > -6.3 && Math.log10(v) < 0.8);
  assert.equal(powLabel(-6), '10⁻⁶');
});

// ---- the size of the start ------------------------------------------------------------------------------------------------------
test('signal: TorC\'s start stays level, too small falls, too large climbs, and without the 2 it falls at every layer', () => {
  const s = Object.fromEntries(signalSeries(deep).map((x) => [x.id, x.points.map((p) => p[1])]));
  assert.deepEqual(SIGNAL_STARTS.map((x) => x.id).sort(), Object.keys(s).sort());
  const k = s.kaiming;
  assert.ok(k.every((v) => v > 0.5 && v < 2), 'Kaiming stays between 0.5 and 2');
  for (let l = 1; l < 9; l++) {
    assert.ok(s.too_small[l] < s.too_small[l - 1], 'too small falls at every layer');
    assert.ok(s.too_large[l] > s.too_large[l - 1], 'too large climbs at every layer');
    assert.ok(s.no_two[l] < s.no_two[l - 1], 'without the 2 it falls at every layer');
  }
  for (const v of Object.values(s).flat()) assert.ok(Math.log10(v) > -6 && Math.log10(v) < 3, 'every point is inside the plot');
});

test('the 2 in √(2/in): ReLU zeroes about half the values, the no-two std is √(1/32), and the worked example and quiz hold', () => {
  const fz = deep.init[0].layers.slice(1).map((l) => l.frac_zero);
  assert.ok(fz.every((f) => f > 0.35 && f < 0.6), 'about half of the values are zero after each ReLU');
  assert.ok(tol(deep.setup.no_two_std, Math.sqrt(1 / deep.setup.width)));
  assert.ok(tol(deep.init[3].init_std, deep.setup.no_two_std, 1e-6));
  assert.ok(tol(deep.setup.kaiming_std, Math.sqrt(2 / deep.setup.width)));
  assert.ok(tol(deep.setup.kaiming_digit.std, Math.sqrt(2 / deep.setup.kaiming_digit.inputs)));
  assert.equal(deep.setup.kaiming_digit.inputs, D.anchors.pixels, 'the first layer of the digit network has one input per pixel');
  assert.ok(deep.init[1].init_std < deep.setup.kaiming_std, 'the transfer quiz start is smaller than TorC\'s start');
});

test('transfer quiz: with the too small start, every layer gets almost no gradient, the first one included', () => {
  const ts = deep.init[1], k = deep.init[0];
  assert.equal(ts.init_std, 0.05000000074505806);
  assert.ok(ts.init_std < deep.setup.kaiming_std);
  ts.layers.slice(1).forEach((l, i) => assert.ok(k.layers[i + 1].mean_abs_grad / l.mean_abs_grad >= ts.derived.kaiming_over_this_min * (1 - 1e-9)));
  assert.ok(ts.derived.kaiming_over_this_min > 1e6, 'every layer at least a million times smaller');
  for (let l = 1; l < 9; l++) assert.ok(ts.layers[l].std < ts.layers[l - 1].std, 'the signal shrinks on the way forward');
});

test('the page states the claims it should and no hand-typed result', () => {
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.match(html, /same output weight, so the backward pass gives them the same/);
  assert.match(html, /makes up for that lost half/);
  assert.match(html, /one random start/, 'single-seed claim labelled (C5)');
  assert.match(html, /href="\.\.\/backprop\/"/, 'the part review links back');
});
