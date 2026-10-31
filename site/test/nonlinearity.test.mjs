// Lesson nonlinearity, "Activation functions" (template 3): the claims of the page against what TorC recorded
// (data/nonlinearity.json), the JS of the hinge builder against the trace, every number bound, no hand-typed digits, all LaTeX,
// and the widget modules importing without a DOM. The softmax rows stay tested here because `losses` reads them.
// The v3 rules (roles, budgets, glossary, notation order) run on this lesson in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson, course, renderStrictAll } from './helpers.mjs';
import { hingeSum, typicalMiss, dataOnGrid } from '../js/widgets/hinge-builder.js';

const data = readJson('data/nonlinearity.json');
const D = buildD({ nonlinearity: data }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const lesson = (await import('../js/lessons/nonlinearity.js')).default;
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
const N = data, H = data.hinges;
const ALL = new Set(H.units.map((u) => u.unit));
const [lin, net] = data.models;

// ---- the recorded data ----------------------------------------------------------------------------------------------------------
test('relu: max(0, z); gradient 1 for z > 0 and 0 for z <= 0 (TorC uses 0 at exactly zero)', () => {
  const [relu] = data.activations;
  data.z.forEach((z, i) => {
    assert.equal(relu.value[i], Math.max(0, z));
    assert.equal(relu.grad[i], z > 0 ? 1 : 0, `z = ${z}`);
  });
  assert.equal(data.z[40], 0);
});

test('sigmoid: 1/(1 + e^-z), gradient s(1 - s), largest at zero (0.25), about 0.018 at z = 4 where sigmoid is almost one', () => {
  const sig = data.activations[1];
  data.z.forEach((z, i) => {
    assert.ok(near(sig.value[i], 1 / (1 + Math.exp(-z))), `value at ${z}`);
    assert.ok(near(sig.grad[i], sig.value[i] * (1 - sig.value[i]), 1e-5), `grad at ${z}`);
  });
  assert.equal(Math.max(...sig.grad), sig.grad[40]);
  assert.equal(sig.grad[40], 0.25);
  assert.equal(data.z[80], 4);
  assert.ok(sig.grad[80] < 0.02 && sig.grad[80] > 0.017, 'shown as 0.018');
  assert.ok(sig.value[80] > 0.98, 'sigmoid is almost one at z = 4');
});

test('softmax rows (read by losses): add to one, a shift and large inputs change nothing, equal inputs share equally', () => {
  const [small, shifted, large, equal] = data.softmax;
  for (const r of data.softmax) assert.ok(near(r.output.reduce((a, b) => a + b, 0), 1));
  assert.deepEqual(shifted.output, small.output);
  assert.deepEqual(large.output, small.output);
  assert.ok(equal.output.every((v) => near(v, 1 / 3)));
});

test('the input grids are exact: built from integers, so 0 is exactly 0 on every machine', () => {
  data.z.forEach((z, k) => assert.equal(Math.fround(z), Math.fround((k - 40) / 10), `z[${k}]`));
  data.data.x.forEach((x, k) => assert.equal(Math.fround(x), Math.fround((k - 10) / 5), `x[${k}]`));
  data.data.grid.forEach((x, k) => assert.equal(Math.fround(x), Math.fround((k - 22) / 10), `grid[${k}]`));
});

// ---- section 1 and the Your turn quiz: f, g and ReLU between them --------------------------------------------------------------
test('collapse: real Linear(1, 1) modules f = 2x + 1 and g = 3x + 4 give g(f(x)) = 6x + 7; with ReLU between, x = -2 gives 4, not -5', () => {
  const c = data.collapse;
  assert.deepEqual([c.f.w, c.f.b, c.g.w, c.g.b], [2, 1, 3, 4]);
  assert.deepEqual([c.derived.w, c.derived.b], [c.g.w * c.f.w, c.g.w * c.f.b + c.g.b]);
  assert.deepEqual([c.derived.w, c.derived.b], [6, 7]);
  assert.equal(c.equal, true);
  c.x.forEach((x, i) => {
    assert.equal(c.f_x[i], c.f.w * x + c.f.b);
    assert.equal(c.g_f_x[i], c.derived.w * x + c.derived.b, `the one line at x = ${x}`);
    assert.equal(c.combined_x[i], c.g_f_x[i]);
    assert.equal(c.relu_f_x[i], Math.max(0, c.f_x[i]));
    assert.equal(c.g_relu_f_x[i], c.g.w * c.relu_f_x[i] + c.g.b);
  });
  assert.deepEqual([c.x[4], c.f_x[4], c.g_f_x[4]], [2, 5, 19], 'the worked example at x = 2');
  assert.deepEqual([c.x[0], c.f_x[0], c.relu_f_x[0], c.g_relu_f_x[0], c.g_f_x[0]], [-2, -3, 0, 4, -5], 'the Your turn quiz');
  assert.equal(T.num(lesson.sections.find((s) => s.id === 'relu-between').answer), 4);
});

// ---- the predict quiz and the fit widget -----------------------------------------------------------------------------------------
test('fit: with no activation the model ends as a flat line at the mean of the targets; with ReLU it bends and fits far better', () => {
  assert.deepEqual(lin.checkpoints.map((c) => c.step), [0, 25, 100, data.setup.steps]);
  assert.equal(data.setup.points, data.data.x.length);
  assert.deepEqual([data.data.x[0], data.data.x[20]], [-2, 2]);
  data.data.x.forEach((x, i) => assert.ok(near(data.data.t[i], x * x, 1e-5)));
  assert.ok(lin.derived.max_abs_diff_line_vs_model < 1e-4, 'two linear layers are one line');
  assert.ok(Math.abs(lin.derived.w_eff) < 1e-4, 'the line is flat');
  const mean = data.data.t.reduce((a, b) => a + b, 0) / data.data.t.length;
  assert.ok(near(lin.derived.b_eff, mean, 1e-5), 'at the average height of the points');
  assert.equal(data.data.grid[22], 0);
  assert.ok(near(lin.checkpoints.at(-1).prediction[22], mean, 1e-5), 'the reveal quotes the flat line at x = 0');
  const start = lin.checkpoints[0].prediction, end = lin.checkpoints.at(-1).prediction;
  assert.ok(start.at(-1) - start[0] > 1, 'the start line is tilted');
  assert.ok(Math.max(...end) - Math.min(...end) < 1e-4, 'the end line is flat');
  for (const c of lin.checkpoints) {                                    // never bends: every prediction lies on a straight line
    const s = (c.prediction.at(-1) - c.prediction[0]) / (data.data.grid.at(-1) - data.data.grid[0]);
    c.prediction.forEach((p, i) => assert.ok(Math.abs(p - (c.prediction[0] + s * (data.data.grid[i] - data.data.grid[0]))) < 1e-4));
  }
  const linEnd = lin.checkpoints.at(-1).loss, netEnd = net.checkpoints.at(-1).loss;
  assert.ok(netEnd * 100 < linEnd);
  assert.equal(T.human(`files.nonlinearity.models.0.checkpoints.3.loss`, 'rmse').replace(/<[^>]+>/g, ''), '1.3');
  assert.equal(T.human(`files.nonlinearity.models.1.checkpoints.3.loss`, 'rmse').replace(/<[^>]+>/g, ''), '0.064');
});

// ---- the hinge builder -------------------------------------------------------------------------------------------------------------
test('hinge builder: all neurons on equals sum_plus_b2 and the model, its typical miss squared is the recorded loss', () => {
  assert.equal(H.units.length, data.setup.hidden);
  assert.equal(H.max_abs_diff_vs_prediction, 0);
  const all = hingeSum(N, ALL);
  all.forEach((v, i) => assert.ok(near(v, H.sum_plus_b2[i], 1e-5), `grid ${i}`));
  all.forEach((v, i) => assert.ok(near(v, net.checkpoints[3].prediction[i], 1e-5)));
  assert.ok(near(typicalMiss(N, all) ** 2, net.checkpoints[3].loss, 1e-3), 'the JS readout stands in for the TorC loss');
  assert.deepEqual(dataOnGrid(N), data.data.x.map((_, k) => 2 + 2 * k));
  assert.ok(hingeSum(N, new Set()).every((v) => v === H.b2), 'all off: a flat line at the output bias');
  for (const u of H.units) {                                     // each contribution is W2·relu(W1 x + b1), kinked at -b1/W1
    data.data.grid.forEach((x, i) => assert.ok(near(u.contribution[i], u.w2 * Math.max(0, u.w1 * x + u.b1), 1e-5)));
    assert.ok(near(u.derived.kink_x, -u.b1 / u.w1, 1e-6));
  }
});

test('hinge tasks: neuron 4 alone is zero left of its kink and a straight line right of it; neuron 1 off drops the left arm only', () => {
  const g = data.data.grid, u4 = H.units[4], u1 = H.units[1];
  assert.equal(u4.unit, 4);
  assert.ok(u4.w1 > 0 && u4.w2 > 0 && u4.derived.active_for === 'x > kink_x');
  const one = hingeSum(N, new Set([4]));
  g.forEach((x, i) => {
    if (x < u4.derived.kink_x) assert.ok(near(one[i], H.b2, 1e-6), 'flat at the bias on the left');
    else assert.ok(near(one[i] - H.b2, u4.w2 * (u4.w1 * x + u4.b1), 1e-5), 'a straight line on the right');
  });
  assert.ok(H.units[0].w1 > 0 && H.units[0].w2 > 0, 'task 2: neuron 0 lifts the right side');
  assert.ok(u1.w1 < 0 && u1.w2 > 0, 'task 2: neuron 1 lifts the left side');
  assert.ok(H.units.filter((u) => u.w2 < 0.2).length >= 2, 'some neurons have a small weight in the second layer');
  assert.ok(H.b2 < 0 && H.b2 > -0.3, 'task 1: the sum sits a little lower than the single hinge');
  assert.equal(u1.unit, 1);
  assert.ok(u1.derived.active_for === 'x < kink_x' && u1.derived.kink_x < 0);
  const off = new Set(ALL); off.delete(1);
  const all = hingeSum(N, ALL), less = hingeSum(N, off);
  assert.ok(all[0] - less[0] > 1, 'the left end drops by more than one');
  g.forEach((x, i) => { if (x > u1.derived.kink_x) assert.ok(near(less[i], all[i], 1e-6), 'the right side stays the same'); });
  const leftRise = H.units.filter((u) => u.w1 < 0).map((u) => u.contribution[0]);
  assert.ok(u1.contribution[0] >= Math.max(...leftRise) - 0.05, 'neuron 1 gives much of the rise on the left');
});

// ---- the lesson text --------------------------------------------------------------------------------------------------------------
const parts = renderLesson(lesson, D, { course: course() });
const html = parts.map((p) => p.html).join('\n');
const rawHtml = () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
};

test('the lesson is a v3 lesson that lights the model, softmax and overflow are gone, the predict has no hint', () => {
  assert.equal(lesson.template, 3);
  assert.equal(lesson.loop, 'model');
  assert.equal(lesson.title, 'Activation functions');
  assert.equal(course().parts.flatMap((p) => p.lessons).find((l) => l.id === 'nonlinearity').title, lesson.title);
  assert.doesNotMatch(rawHtml().replace(/<style[\s\S]*?<\/style>/g, ''), /softmax|overflow|symmetric/i);
  const flat = lesson.sections.find((s) => s.id === 'flat');
  assert.equal(flat.purpose, 'predict');
  assert.ok(lesson.sections.indexOf(flat) < lesson.sections.findIndex((s) => s.kind === 'text' && !s.setup && s !== lesson.sections[0]));
  assert.ok(flat.options.find((o) => o.correct).html.includes('flat straight line'));
  assert.match(html, /data-sym="sigma"/);
  assert.match(html, /data-sym="exp"/);
  assert.match(html, /aria-label="Analogy"/);
  const t = rawHtml().replace(/<style[\s\S]*?<\/style>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
  assert.match(t, /Nothing sits between the layers, so each layer only multiplies and adds, as in the last lesson\./, 'the predict is fair');
  assert.match(t, /typical miss, the square root of the mean squared miss/);
  assert.doesNotMatch(t, /on average|kink|\bunit\b/, 'one name: typical miss, bend, neuron');
  assert.match(t, /This small sigma is not the Σ of sums\./);
  assert.match(t, /letter W\. It goes down, up, down and up again, so it has three bends\./, 'the transfer asks the reverse question');
  assert.match(t, /zero for every negative one/);
  // the cumulative quiz comes before any sentence that says the weight barely learns
  const quiz = rawHtml().indexOf('Weight A sits before a flat part');
  assert.ok(quiz > 0 && rawHtml().slice(0, quiz).indexOf('barely learn') < 0);
  assert.match(html, /Where it breaks/);
});

test('every number is bound, and no hand-typed digit survives in the prose, quizzes or tasks', () => {
  for (const { section, html: h } of parts) assert.deepEqual(findUnboundNums(h), [], section.kind);
  for (const b of boundNums(html)) assert.equal(T.reformat(b), b.shown, b.path);
  const digits = rawHtml()
    .replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code>[\s\S]*?<\/code>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '')
    .replace(/<h[23][^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(digits, []);
  assert.equal(lesson.footer, undefined);
});

test('every formula of the lesson parses in KaTeX (throwOnError)', () => {
  const formulas = [...rawHtml().matchAll(/<code>([^<]*)<\/code>/g)].map((m) => m[1].replace(/&amp;/g, '&')).filter((s) => /[\\^_{}]/.test(s));
  assert.ok(formulas.length >= 2);
  renderStrictAll(formulas);
  assert.doesNotMatch(html, /katex-error/);
});

test('the gradient of relu is drawn in two pieces, with no line across the jump at zero', async () => {
  const { pieces } = await import('../js/widgets/nonlinear-widgets.js');
  const [relu, sig] = data.activations;
  const p = pieces(data.z, relu.grad);
  assert.equal(p.length, 2);
  assert.equal(pieces(data.z, sig.grad).length, 1);
});

test('the widget modules import in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/nonlinear-widgets.js');
  assert.equal(typeof m.mountActivationPlot, 'function');
  assert.equal(typeof m.mountCurveFit, 'function');
  assert.equal(typeof (await import('../js/widgets/hinge-builder.js')).mountHingeBuilder, 'function');
  assert.equal(typeof document, 'undefined');
});
