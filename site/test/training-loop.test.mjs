// Run: node --test site/test/
// The training-loop lesson (template 3): every claim of the prose, the guided tasks and the quizzes, each one asserted against what
// TorC recorded (data/training.json, tools/trace/trace_training.cpp); the JS gradient port against the batch gradients of the trace;
// the widgets' pure helpers; the five-line loop against course.json. The v3 rules themselves (roles, budgets, glossary) run in
// lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson, course } from './helpers.mjs';
import lesson, { LOOP, STEPS_32 } from '../js/lessons/training-loop.js';
import { loaderState, compareView, valueAt, typicalMiss, sig } from '../js/widgets/training-widgets.js';
import { spreadLabels } from '../js/widgets/curve-labels.js';

const data = readJson('data/training.json');
const D = buildD({ training: data }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const N = data.setup.samples;
const run = (bs) => data.runs.find((r) => r.batch_size === bs);
const BC = data.batch_compare, GE = BC.gradient_estimate, S = data.sorted_order, AC = data.accumulate;
const near = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
const r2 = (v) => sig(v, 2);                                          // the 2-figure display of a typical miss
// the gradient of the mean squared error of y = w·x + b on the samples `idx` (the JS stand-in for L.backward() on a batch)
const gradOn = (idx, w = 0, b = 0) => {
  let gw = 0, gb = 0;
  for (const i of idx) { const r = w * data.data.x[i] + b - data.data.y[i]; gw += 2 * r * data.data.x[i]; gb += 2 * r; }
  return { gw: gw / idx.length, gb: gb / idx.length };
};
const ALL = Array.from({ length: N }, (_, i) => i);

// ---- the data and the loader ------------------------------------------------------------------------------------------
test('the samples follow t = 3x + 1 plus small noise', () => {
  assert.equal(data.data.x.length, N);
  const res = data.data.x.map((x, i) => data.data.y[i] - (data.setup.true_weight * x + data.setup.true_bias));
  const mse = res.reduce((a, r) => a + r * r, 0) / N;
  assert.ok(near(mse, data.true_line_loss, 1e-5), `${mse} vs ${data.true_line_loss}`);
  assert.ok(Math.abs(Math.sqrt(mse) - data.setup.noise_std) < 0.02, 'the noise is small, about the configured size');
});

test('every epoch of every loader gives each sample exactly once; batch 10 takes 10 steps, batch 32 takes 4 with a smaller last batch', () => {
  for (const l of data.loaders) {
    for (const [e, batches] of l.epochs.entries()) {
      assert.deepEqual(batches.flat().slice().sort((a, b) => a - b), ALL, `${l.batch_size} ${l.shuffle} epoch ${e}`);
      assert.equal(batches.length, Math.ceil(N / l.batch_size));
      batches.slice(0, -1).forEach((b) => assert.equal(b.length, l.batch_size));
    }
    if (!l.shuffle) l.epochs.forEach((ep) => assert.deepEqual(ep.flat(), ALL, 'no shuffle keeps the order'));
    else assert.notDeepEqual(l.epochs[0].flat(), l.epochs[1].flat(), 'a new epoch draws a new order');
  }
  // the worked example and the "Your turn" quiz
  assert.equal(run(10).epochs[1].steps, N / 10);
  assert.equal(T.num(STEPS_32), Math.ceil(N / 32));
  assert.equal(data.runs[1].batch_size, 32, 'STEPS_32 points at the batch-32 run');
  const l32 = data.loaders.find((l) => l.batch_size === 32 && !l.shuffle).epochs[0];
  assert.deepEqual(l32.map((b) => b.length).slice(0, 3), [32, 32, 32], 'three full batches');
  assert.ok(l32.at(-1).length < 32, 'and a smaller last batch');
  for (const r of data.runs) r.epochs.forEach((e, i) => assert.equal(e.steps, i * Math.ceil(N / r.batch_size)), 'an epoch is many steps');
});

test('loaderState follows the buttons of the batch viewer through two epochs', () => {
  const l = data.loaders.find((x) => x.batch_size === 10 && x.shuffle);
  const s0 = loaderState(l, 0, 0);
  assert.equal(s0.batches, 10);
  assert.ok(s0.hasNext);
  assert.deepEqual(s0.current, []);
  const s2 = loaderState(l, 0, 2);
  assert.deepEqual(s2.current, l.epochs[0][1]);
  assert.deepEqual([...s2.used].sort((a, b) => a - b), l.epochs[0][0].slice().sort((a, b) => a - b));
  assert.equal(loaderState(l, 0, 10).hasNext, false, 'the epoch is over after ten presses');
  assert.equal(loaderState(l, 0, 10).steps, 10);
  assert.equal(loaderState(l, 1, 3).steps, 13, 'steps count on into the next epoch');
});

// ---- the batch gradient as an estimate ------------------------------------------------------------------------------------
test('gradient port: the MSE gradient of each batch and of all samples equals what L.backward() gave in TorC', () => {
  const full = gradOn(ALL);
  assert.ok(near(full.gw, GE.full.gw) && near(full.gb, GE.full.gb), JSON.stringify(full));
  assert.equal(GE.batches.length, N / GE.batch_size);
  GE.batches.forEach((b, i) => { const g = gradOn(b.indices); assert.ok(near(g.gw, b.gw, 1e-4) && near(g.gb, b.gb, 1e-4), `batch ${i}`); });
  // the batches are the first epoch of training: the loader order after one reset(), as in the loop
  assert.deepEqual(GE.batches.map((b) => b.indices), data.loaders.find((l) => l.batch_size === 10 && l.shuffle).epochs[1]);
  assert.equal(GE.at.w, 0);
  assert.equal(GE.at.b, 0);
});

test('the estimate claims: close to the full gradient but never equal; the average of one epoch is the full gradient', () => {
  const gws = GE.batches.map((b) => b.gw), gbs = GE.batches.map((b) => b.gb);
  assert.equal(GE.range.gw_min, Math.min(...gws));
  assert.equal(GE.range.gw_max, Math.max(...gws));
  assert.equal(GE.range.gb_min, Math.min(...gbs));
  assert.equal(GE.range.gb_max, Math.max(...gbs));
  // "it points roughly the same way" / "roughly downhill": every batch gradient has the signs of the full gradient
  for (const b of GE.batches) assert.ok(Math.sign(b.gw) === Math.sign(GE.full.gw) && Math.sign(b.gb) === Math.sign(GE.full.gb));
  // "none sits on the star" (in the widget's units, a dot of radius 5 on a range of about 3)
  for (const b of GE.batches) assert.ok(Math.hypot(b.gw - GE.full.gw, b.gb - GE.full.gb) > 0.05);
  // "the ring of the average sits on the star", and the wrong option "ten times smaller" is refuted
  assert.ok(near(GE.mean.gw, GE.full.gw, 1e-6) && near(GE.mean.gb, GE.full.gb, 1e-6));
  assert.ok(near(GE.mean.gw, gws.reduce((a, v) => a + v, 0) / gws.length, 1e-9));
  assert.ok(Math.abs(GE.mean.gw) > 5 * Math.abs(GE.full.gw / 10), 'a batch gradient is not ten times smaller');
});

test('cumulative quiz: one SGD step from w = 0 on the first batch lands at 0 − lr × gradient, as TorC stepped', () => {
  const b = GE.batches[0];
  assert.ok(near(b.step.w, GE.at.w - GE.lr * b.gw, 1e-6) && near(b.step.b, GE.at.b - GE.lr * b.gb, 1e-6));
  assert.ok(b.step.w > 0, 'w goes up');
  const q = lesson.sections.find((s) => s.id === 'one-step');
  assert.equal(T.num(q.answer), b.step.w);
});

// ---- why shuffle ----------------------------------------------------------------------------------------------------------------
test('why shuffle: in sorted order without shuffle the miss grows during the first steps; shuffled, it falls at once', () => {
  const xs = S.sorted_indices.map((i) => data.data.x[i]);
  xs.slice(1).forEach((x, k) => assert.ok(x >= xs[k], 'sorted by x, smallest first'));
  assert.deepEqual(S.sorted_indices.slice().sort((a, b) => a - b), ALL);
  const L = S.sorted_loss;
  assert.ok(L[1] > L[0] && L[2] > L[1] && L[3] > L[0], `${L.slice(0, 4)}`);
  assert.ok(S.shuffled_loss[1] < S.shuffled_loss[0]);
  assert.deepEqual(S.shuffled_loss, BC.runs.find((r) => r.batch_size === 10 && r.seed === S.shuffled_seed).loss);
  // "the first batches hold only the leftmost points": the first batch is the ten smallest x
  const first = S.sorted_indices.slice(0, 10).map((i) => data.data.x[i]);
  assert.ok(Math.max(...first) < Math.min(...S.sorted_indices.slice(10).map((i) => data.data.x[i])) + 1e-12);
  assert.ok(Math.max(...first) < -0.5, 'the left end');
});

// ---- the fair batch-size comparison (C1, C5) -------------------------------------------------------------------------------
test('equal steps: after 30 steps the two batch sizes miss by about the same; three orders; batch 100 does not depend on the order', () => {
  assert.deepEqual(BC.batch_sizes, [10, 100]);
  assert.deepEqual(BC.seeds.length, 3);
  const [s10, s100] = BC.equal_steps[0].summary;
  assert.equal(BC.equal_steps[0].steps, 30);
  assert.equal(s10.batch_size, 10);
  assert.equal(s100.batch_size, 100);
  assert.equal(s100.min, s100.max, 'one batch is all the data, so the order does not matter');
  // the reveal shows "about 0.31 to 0.32" and "about 0.32": all within a few hundredths
  for (const v of [s10.min, s10.max]) assert.ok(Math.abs(typicalMiss(v) - typicalMiss(s100.min)) < 0.01);
  assert.equal(r2(typicalMiss(s10.max)), r2(typicalMiss(s100.min)));
  // the widget task: "the two curves almost cover each other" over the first 100 steps
  const v = compareView(data, 'steps');
  const c100 = v.curves.find((c) => c.key === 'b100').pts;
  for (const c of v.curves.filter((x) => x.key === 'b10')) for (let k = 3; k <= 100; k++) assert.ok(Math.abs(c.pts[k][1] - c100[k][1]) < 0.12 * c100[k][1] + 0.01, `seed ${c.seed} step ${k}`);
  // "the large batches read ten times more samples"
  const res = BC.equal_steps[0].results;
  assert.equal(res.find((r) => r.batch_size === 100).samples_seen, 10 * res.find((r) => r.batch_size === 10).samples_seen);
});

test('equal samples: 3000 samples are 300 steps of batch 10 and 30 of batch 100, and the small batches miss far less', () => {
  const E = BC.equal_samples[0];
  assert.equal(E.samples, 3000);
  assert.equal(E.results[0].batch_size, 10);
  assert.equal(E.results[3].batch_size, 100);
  assert.equal(E.results[0].steps, E.samples / 10);
  assert.equal(E.results[3].steps, E.samples / 100);
  const [s10, s100] = E.summary;
  assert.ok(typicalMiss(s10.max) * 3 < typicalMiss(s100.min), 'far less');
  // "after equal epochs, the small batches win only because they take ten times more steps": 30 epochs are 3000 samples each
  assert.equal(run(10).epochs[30].steps, 10 * run(100).epochs[30].steps);
  assert.ok(run(10).epochs[30].loss < run(100).epochs[30].loss);
  assert.equal(run(10).epochs[30].loss, E.results[0].loss, 'seed 7 after 3000 samples is the 30-epoch run');
  // the widget's samples view at the end of the axis shows these values
  const v = compareView(data, 'samples');
  assert.equal(v.xMax, E.samples);
  assert.equal(valueAt(v.curves.find((c) => c.key === 'b100').pts, E.samples), typicalMiss(s100.min));
});

test('each size has a cost: the batch gradient is noisy (batch 10 runs differ by order), the full batch is not', () => {
  const band10 = BC.band.find((b) => b.batch_size === 10), band100 = BC.band.find((b) => b.batch_size === 100);
  band100.min.forEach((v, k) => assert.ok(near(v, band100.max[k], 1e-6), `step ${k}`));
  assert.ok(band10.max[1] - band10.min[1] > 0.1, 'the orders spread apart in the first steps');
});

// ---- the noise that no line removes ---------------------------------------------------------------------------------------------
test('the noise: the learned line is close to 3 and 1, its typical miss is about that of the true line, and the loss stops falling', () => {
  const last = run(10).epochs[30];
  assert.ok(Math.abs(last.weight - data.setup.true_weight) < 0.01 && Math.abs(last.bias - data.setup.true_bias) < 0.02);
  assert.equal(r2(typicalMiss(last.loss)), '0.09');
  assert.ok(Math.abs(typicalMiss(last.loss) - typicalMiss(data.true_line_loss)) < 0.002);
  const L = BC.runs.find((r) => r.batch_size === 10 && r.seed === 7).loss;
  assert.ok(Math.abs(L[300] - L[200]) < 0.01 * L[300], 'the loss has stopped falling');
  // the least-squares line of these samples: no line has a lower loss, and the run reaches it
  const x = data.data.x, y = data.data.y;
  const mx = x.reduce((a, v) => a + v, 0) / N, my = y.reduce((a, v) => a + v, 0) / N;
  const w = x.reduce((a, v, i) => a + (v - mx) * (y[i] - my), 0) / x.reduce((a, v) => a + (v - mx) ** 2, 0), b = my - w * mx;
  const lsq = x.reduce((a, v, i) => a + (w * v + b - y[i]) ** 2, 0) / N;
  assert.ok(Math.abs(last.loss - lsq) < 1e-5);
});

// ---- zero_grad (the In TorC panel) -------------------------------------------------------------------------------------------
test('zero_grad: without it TorC stores the sum of the two gradients, about twice the new one; a second backward on one L adds nothing', () => {
  assert.ok(near(AC.second_without_zero_grad.gw, AC.first.gw + AC.second_with_zero_grad.gw, 1e-5));
  assert.ok(near(AC.second_without_zero_grad.gb, AC.first.gb + AC.second_with_zero_grad.gb, 1e-5));
  const ratio = AC.second_without_zero_grad.gw / AC.second_with_zero_grad.gw;
  assert.ok(ratio > 1.8 && ratio < 2.2, `about twice: ${ratio}`);
  assert.deepEqual(AC.first, GE.full, 'the first gradient is the full gradient at the start');
  assert.deepEqual(AC.same_loss_twice, AC.first);
  // the second gradient is what one SGD step on the full gradient leaves (the JS port at the new w and b)
  const g2 = gradOn(ALL, -AC.lr * AC.first.gw, -AC.lr * AC.first.gb);
  assert.ok(near(g2.gw, AC.second_with_zero_grad.gw, 1e-4) && near(g2.gb, AC.second_with_zero_grad.gb, 1e-4));
});

// ---- the page --------------------------------------------------------------------------------------------------------------------
test('the five-line loop on the page is the code of this lesson in course.json, with the five numbered lines', () => {
  const c = course().parts.flatMap((p) => p.lessons).find((l) => l.id === 'training-loop');
  assert.equal(c.code, LOOP);
  assert.deepEqual([...LOOP.matchAll(/\/\/ (\d)\. /g)].map((m) => Number(m[1])), [1, 2, 3, 4, 5]);
  assert.match(LOOP, /L\.backward\(\);[\s\S]*opt\.step\(\);[\s\S]*opt\.zero_grad\(\);/);
  assert.match(LOOP, /epoch < 30/);
  assert.equal(data.setup.epochs, 30);
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.ok(html.includes('class="real-line"'));
  assert.doesNotMatch(html, /reset\(\) check|no_reset/);
});

test('every number in the prose is bound to the trace, with no hand-typed digits outside code', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const b of boundNums(html)) assert.equal(T.reformat(b), b.shown, b.path);
  }
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

// ---- widget helpers --------------------------------------------------------------------------------------------------------------
test('compareView gives the curves of each view, as typical misses, and the sorted view starts higher than shuffled after a step', () => {
  const steps = compareView(data, 'steps');
  assert.equal(steps.curves.filter((c) => c.key === 'b10').length, 3);
  assert.equal(steps.curves.find((c) => c.key === 'b100').pts.length, steps.xMax + 1);
  assert.equal(steps.curves[0].pts[0][1], typicalMiss(BC.runs[0].loss[0]));
  const samples = compareView(data, 'samples');
  assert.deepEqual(samples.curves.find((c) => c.key === 'b100').pts.map((p) => p[0]).slice(0, 3), [0, 100, 200]);
  assert.equal(samples.curves.find((c) => c.key === 'b10').pts.at(-1)[0], 3000);
  const sorted = compareView(data, 'sorted');
  assert.ok(valueAt(sorted.curves.find((c) => c.key === 'sort').pts, 2) > valueAt(sorted.curves.find((c) => c.key === 'b10').pts, 2));
  assert.equal(valueAt([[0, 1], [10, 2]], 9), 1);
  assert.equal(sig(-1.50492883), '−1.5');
  assert.equal(sig(0.0900449, 2), '0.09');
});

test('spreadLabels keeps labels apart and inside the plot', () => {
  const ys = spreadLabels([100, 102, 300], { gap: 16, min: 20, max: 280 });
  assert.ok(ys[1] - ys[0] >= 16 && ys[2] <= 280 && ys[0] >= 20);
  assert.deepEqual(spreadLabels([50, 10], { gap: 10 }), [50, 10]);
  const tight = spreadLabels([270, 275, 279], { gap: 14, min: 0, max: 280 });
  assert.ok(tight[2] <= 280 && tight[1] <= tight[2] - 14 && tight[0] <= tight[1] - 14);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/training-widgets.js');
  for (const f of ['mountBatchViewer', 'mountGradientEstimate', 'mountBatchCompare']) assert.equal(typeof m[f], 'function');
  assert.equal(typeof document, 'undefined');
});

test('float32 values from the trace print as float32 literals (brief D5, cpp-literal.js)', async () => {
  const { cf } = await import('../js/widgets/cpp-literal.js');
  assert.equal(data.setup.noise_std, 0.10000000149011612, 'the trace stores float32 0.1 as a double');
  assert.equal(cf(data.setup.noise_std), '0.1f');
  assert.equal(cf(3), '3.0f');
});
