// Lesson linear-layer, "The neuron and the layer" (template 3): the taxi layer of data/linear.json against the JS arithmetic of the
// widgets, every claim the prose, the quizzes and the tasks make (checked from the data, not only that the numbers are bound), the
// parameter counts of the digit network, the network picture, all LaTeX, and that the widgets import without a DOM. The recorded
// scenes `linear` and `linear-dx` and the signal table stay tested here: `backprop` and `deep-nets` use them.
// The v3 rules (roles, budgets, glossary, notation order) run on this lesson in lessons.test.mjs.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, runBackward } from '../js/engine.js';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson, close, katex, course, renderStrictAll } from './helpers.mjs';
import lesson from '../js/lessons/linear-layer.js';
import { linearShapes, paramCount } from '../js/widgets/linear-shapes.js';
import { exampleFromScene, exampleFromTensors, linearForward, gradW, gradB, gradX, colSums, rowsOf, neuronTex } from '../js/widgets/linear-math.js';
import { signalSeries, signalTable, growthRatios } from '../js/widgets/linear-signal-data.js';
import { digitLayers, digitNetwork, layerPicture } from '../js/widgets/layer-count.js';
import { neuronFigure } from '../js/widgets/neuron-figure.js';

const data = readJson('data/linear.json');
const anchors = anchorValues(readJson('data/anchors.json'));
const D = buildD({ linear: data }, anchors);
const T = makeT(D);
const scene = (id) => D.scenes[id];
const node = (s, name) => s.nodes.find((n) => n.name === name);
const sum = (a) => a.reduce((x, y) => x + y, 0);
const taxi = data.taxi;
const ex = exampleFromTensors(taxi);
const A = (name) => anchors.anchors?.[name]?.value ?? anchors[name];

// ---- the recorded scenes (kept for backprop) ---------------------------------------------------------------------------------
test('engine reproduces scenes `linear` and `linear-dx` (values and gradients), parity with nn::Linear, finite differences', () => {
  for (const id of ['linear', 'linear-dx']) {
    const s = scene(id);
    const g = buildGraph(s), bw = runBackward(g);
    for (const n of s.nodes) {
      assert.deepEqual(g.byName[n.name].value.shape, n.value.shape, `${id}.${n.name} shape`);
      assert.ok(close(g.byName[n.name].value.v, n.value.values), `${id}.${n.name} value`);
      if (n.grad) assert.ok(close(bw.grad[n.name].v, n.grad.values), `${id}.${n.name} grad`);
    }
    const p = s.parity;
    assert.ok(p.output_equal && p.grads_equal && p.loss_equal && p.max_abs_diff === 0, id);
    for (const l of s.finite_difference.leaves) assert.equal(l.ok, true, `${id}.${l.name} finite difference`);
  }
  assert.equal(data.checks.all_internal_checks_passed, true);
});

test('hand formulas equal the library: gW = dY^T x, gb = column sums of dY, dX = dY W (for the backprop In TorC panel)', () => {
  const s = scene('linear'), e = exampleFromScene(s);
  const dY = rowsOf(node(s, 'y').grad.values, 2, 2);
  assert.ok(close(gradW(dY, e.x).flat(), node(s, 'W').grad.values));
  assert.ok(close(gradB(dY), node(s, 'b').grad.values));
  const dx = scene('linear-dx'), edx = exampleFromScene(dx);
  assert.ok(close(gradX([[1, 1], [1, 1]], edx.W).flat(), node(dx, 'x').grad.values));
  assert.ok(close(node(dx, 'x').grad.values.slice(0, 3), colSums(edx.W)));
  assert.ok(close(linearForward(e.x, e.W, e.b).flat(), node(s, 'y').value.values));
});

test('signal table (kept for deep-nets): six layers per init; Kaiming stays near 1, too small fades, too big explodes', () => {
  assert.deepEqual(data.signal.map((s) => s.name), ['kaiming', 'std_0.01', 'std_1.0']);
  const [k, small, big] = signalSeries(data.signal);
  assert.ok(k.points.slice(1).every((q) => q.std > 0.6 && q.std < 1.2));
  assert.ok(small.points[6].std < 1e-5 && big.points[6].std > 1e5);
  assert.ok(growthRatios(big).every((r) => r > 10 && r < 13));
  assert.equal(signalTable(signalSeries(data.signal)).rows.length, 7);
});

// ---- the taxi layer ------------------------------------------------------------------------------------------------------------
test('taxi: a real nn::Linear(3, 2) with W of shape (out, in) = (2, 3); its y equals the JS arithmetic of the grid, exactly', () => {
  assert.deepEqual(taxi.inputs, ['distance in km', 'minutes of waiting', 'bags']);
  assert.deepEqual(taxi.outputs, ['price in euros', 'time in minutes']);
  assert.deepEqual(taxi.weight_shape, [2, 3], 'one row per neuron, one column per input');
  assert.deepEqual(taxi.bias_shape, [2]);
  assert.equal(taxi.exact, true);
  assert.deepEqual(linearForward(ex.x, ex.W, ex.b).flat(), taxi.y.values, 'the grid shows what TorC computed');
  assert.deepEqual(ex.y.flat(), taxi.y.values);
  assert.deepEqual(linearShapes(3, 2, 2), { x: [2, 3], W: [2, 3], Wt: [3, 2], b: [2], y: [2, 2] });
});

test('the story and the quizzes: price = 2·4 + 0.5·6 + 1·2 + 1 = 14, time = 3·4 + 1·6 + 0·2 + 2 = 20, y10 = 22', () => {
  const [x0, x1] = ex.x, [wp, wt] = ex.W, [bp, bt] = ex.b;
  assert.deepEqual(wp, [2, 0.5, 1], 'kilometre 2 euros, minute 0.5 euro, bag 1 euro');
  assert.equal(bp, 1, 'the fee is 1 euro');
  assert.deepEqual(wt, [3, 1, 0], 'a bag adds no time, so its weight is 0');
  assert.equal(bt, 2);
  assert.deepEqual(x0, [4, 6, 2]);
  assert.deepEqual(x1, [10, 0, 1]);
  assert.equal(sum(x0.map((v, j) => v * wp[j])) + bp, taxi.y.values[0]);
  assert.equal(taxi.y.values[0], 14);
  assert.equal(sum(x0.map((v, j) => v * wt[j])) + bt, taxi.y.values[1]);
  assert.equal(sum(x1.map((v, j) => v * wp[j])) + bp, taxi.y.values[2], 'your turn: row 1 of x, row 0 of W, plus b0');
  const q = lesson.sections.find((s) => s.id === 'price'), yt = lesson.sections.find((s) => s.id === 'your-entry');
  assert.equal(T.num(q.answer), 14);
  assert.equal(T.num(yt.answer), taxi.y.values[2]);
});

test('grid tasks: b0 + 1 raises both prices by one and leaves the times; bag weight 0 lowers each price by its bags', () => {
  const y = rowsOf(taxi.y.values, 2, 2), y1 = rowsOf(taxi.y_bias0_plus1.values, 2, 2);
  y.forEach((r, i) => { assert.equal(y1[i][0], r[0] + 1, 'TorC: the bias reaches every row'); assert.equal(y1[i][1], r[1]); });
  assert.deepEqual(linearForward(ex.x, ex.W, [ex.b[0] + 1, ex.b[1]]), y1, 'the JS grid agrees');
  const W = ex.W.map((r) => r.slice()); W[0][2] = 0;
  const yz = linearForward(ex.x, W, ex.b);
  yz.forEach((r, i) => { assert.equal(y[i][0] - r[0], ex.x[i][2], 'one euro per bag'); assert.equal(r[1], y[i][1]); });
  assert.deepEqual(ex.x.map((r) => r[2]), [2, 1], 'the task names 2 and 1 euros');
});

// ---- the digit network ---------------------------------------------------------------------------------------------------------
test('digit network: the three Linear layers, (out, in) shapes, count = out·in + out, total = the anchor, first layer holds most', () => {
  const L = digitLayers(D);
  assert.deepEqual(L.map((l) => [l.neurons, l.inputs]), [[32, 784], [32, 32], [10, 32]]);
  for (const l of L) {
    assert.deepEqual(l.weightShape, [l.neurons, l.inputs]);
    assert.equal(l.weights, l.neurons * l.inputs);
    assert.equal(l.biases, l.neurons);
    assert.equal(l.count, paramCount(l.inputs, l.neurons).total);
  }
  assert.equal(sum(L.map((l) => l.count)), digitNetwork(D).total);
  assert.equal(digitNetwork(D).total, A('mlp_parameters'));
  assert.ok(L[0].count > A('mlp_parameters') / 2, 'most parameters are in the first layer');
  assert.equal(L[0].inputs, A('pixels'));
  assert.equal(A('image_side') ** 2, A('pixels'), '784 = 28 × 28');
  assert.equal(L[2].neurons, A('classes'), 'one neuron for each digit');
  // the quizzes
  const last = lesson.sections.find((s) => s.id === 'last-layer');
  assert.equal(T.num(last.answer), L[2].count);
  assert.equal(L[2].count, 330);
  const shape = lesson.sections.find((s) => s.id === 'w-shape');
  const opts = shape.options.map((o) => o.html(D, T).replace(/<[^>]+>/g, ''));
  assert.deepEqual(opts, ['(32, 784)', '(784, 32)', '(784)', '(32)']);
  assert.ok(shape.options[0].correct && shape.options.filter((o) => o.correct).length === 1);
  for (const w of [260, 560]) assert.match(layerPicture(L[0], 784, w), /784 columns, one per input.*32 rows, one per neuron/s);
});

// ---- the lesson text --------------------------------------------------------------------------------------------------------
const parts = renderLesson(lesson, D, { course: course() });
const html = parts.map((p) => p.html).join('\n');
const rawHtml = () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
};
const plain = (h) => h.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

test('the lesson is a v3 lesson that lights the model, with the course title, and initialization and backward rules are gone', () => {
  assert.equal(lesson.template, 3);
  assert.equal(lesson.loop, 'model');
  const entry = course().parts.flatMap((p) => p.lessons).find((l) => l.id === 'linear-layer');
  assert.equal(entry.title, lesson.title);
  assert.equal(lesson.title, 'The neuron and the layer');
  assert.ok(lesson.sections.filter((s) => s.kind === 'widget').every((s) => ['matmul-grid', 'layer-count'].includes(s.id)), 'no init-signal widget');
  assert.doesNotMatch(rawHtml(), /overline|Kaiming|init_std|softmax|ShapeError|batch/i);
  assert.doesNotMatch(plain(html), /copy the first neuron/i);
});

test('the network pictures: inputs and weights from the trace, the bias in the circle, no output value (the reader computes it)', () => {
  const one = neuronFigure(taxi, { neurons: [0] }), two = neuronFigure(taxi, { neurons: [0, 1] });
  for (const v of ['4 km', '6 min waiting', '2 bags', '× 2', '× 0.5', '× 1', '+ 1', 'price']) assert.ok(one.includes(v), v);
  for (const v of ['× 3', '× 0', '+ 2', 'time']) assert.ok(two.includes(v), v);
  assert.ok(!/>\s*(14|20|22)\s*</.test(one + two), 'no output values in the pictures');
  assert.match(one, /role="img" aria-label="One neuron\./);
  assert.match(two, /aria-label="A layer of two neurons\./);
  assert.equal((two.match(/<line /g) ?? []).length, 2 * 3 + 2, 'every input reaches every neuron');
});

test('section claims: positions count from 0, Σ and subscripts and ᵀ are linked, W is (neurons, inputs), b is broadcast', () => {
  const t = plain(rawHtml());
  assert.match(t, /positions count from 0/);
  assert.match(html, /data-sym="sum"/);
  assert.match(html, /data-sym="subscript"/);
  assert.match(html, /data-sym="transpose"/);
  assert.match(t, /TorC stores one row per neuron, so the shape of W is \(neurons, inputs\)\./);
  assert.match(t, /TorC copies b into each row of the result and calls this copying broadcasting\s*\./);
  assert.match(html, /data-refresher="matrix-product"/);
  assert.match(t, /not a small brain/);
});

test('every number is bound, and no hand-typed digit survives in the prose, quizzes or tasks', () => {
  for (const { section, html: h } of parts) assert.deepEqual(findUnboundNums(h), [], section.kind);
  for (const b of boundNums(html)) assert.equal(T.reformat(b), b.shown, b.path);
  const digits = rawHtml()
    .replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<details class="refresher"[\s\S]*?<\/details>/g, '')     // the refresher cites its textbook section; it has no trace numbers
    .replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code>[\s\S]*?<\/code>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '')
    .replace(/<h[23][^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(digits, []);
  assert.equal(lesson.footer, undefined);
});

test('every LaTeX string of the lesson and of the grid parses in KaTeX (throwOnError), indices from 0', () => {
  const formulas = [...rawHtml().matchAll(/<code>([^<]*)<\/code>/g)].map((m) => m[1].replace(/&amp;/g, '&')).filter((s) => /[\\^_{}]/.test(s));
  assert.ok(formulas.length >= 4);
  renderStrictAll(formulas);
  for (let i = 0; i < 2; i++) for (let k = 0; k < 2; k++) katex.renderToString(neuronTex(ex.x, ex.W, ex.b, i, k), { throwOnError: true, displayMode: true });
  assert.match(neuronTex(ex.x, ex.W, ex.b, 0, 1), /^\\begin\{aligned\}y_\{01\} &= \\sum_\{j=0\}\^\{2\} x_\{0j\}\\,W_\{1j\} \+ b_\{1\}/);
  assert.match(neuronTex(ex.x, ex.W, ex.b, 0, 1), /= 20\\end\{aligned\}$/);
  assert.doesNotMatch(html, /katex-error/);
});

test('the widgets import in Node without touching the DOM', async () => {
  assert.equal(typeof document, 'undefined');
  assert.equal(typeof (await import('../js/widgets/matmul-grid.js')).mountMatmulGrid, 'function');
  assert.equal(typeof (await import('../js/widgets/layer-count.js')).mountLayerCount, 'function');
  assert.equal(typeof document, 'undefined');
});
