// Run: node --test site/test/
// The optimizers lesson (template 3): every claim of the prose, the guided tasks and the quizzes, each one asserted against what TorC
// recorded (data/optimizers.json, tools/trace/trace_optimizers.cpp); the update rules of SGD, momentum, Adam and AdamW recomputed from
// the recorded positions; the widget's pure helpers. The v3 rules themselves (roles, budgets, glossary) run in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';
import lesson from '../js/lessons/optimizers.js';
import { raceOf, logLoss, FLOOR, LEVELS, RUNS, surfaceOf } from '../js/widgets/optimizer-widgets.js';

const data = readJson('data/optimizers.json');
const D = buildD({ optimizers: data }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const V = data.valley, SW = data.sweep, B = data.bowl;
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol * (1 + Math.abs(b)), `${msg}: ${a} vs ${b}`);
const half = (curv) => curv.map((c) => c / 2);
const lossAt = (curv, w1, w2) => half(curv)[0] * w1 * w1 + half(curv)[1] * w2 * w2;
const sweepOf = (sw, id) => sw.optimizers.find((o) => o.id === id);
const raceRun = (race, id) => race.find((r) => r.id === id);
const final = (race, id) => raceRun(race, id).final_loss;

// recompute a path with the update rules of src/optim.cpp (SGD with and without momentum, Adam, AdamW) on L = sum(c/2 · w²)
function replay(r, curv) {
  const g = (q) => [curv[0] * q.w1, curv[1] * q.w2];
  let v = [0, 0], m = [0, 0], s = [0, 0];
  for (let k = 0; k < r.path.length - 1; k++) {
    const q = r.path[k], gr = g(q), w = [q.w1, q.w2], t = k + 1;
    let next;
    if (r.optimizer === 'optim::SGD') {
      if (r.momentum) { v = v.map((x, i) => r.momentum * x + gr[i]); next = w.map((x, i) => x - r.lr * v[i]); }
      else next = w.map((x, i) => x - r.lr * gr[i]);
    } else {
      m = m.map((x, i) => 0.9 * x + 0.1 * gr[i]);
      s = s.map((x, i) => 0.999 * x + 0.001 * gr[i] * gr[i]);
      const up = m.map((x, i) => (x / (1 - 0.9 ** t)) / (Math.sqrt(s[i] / (1 - 0.999 ** t)) + 1e-8));
      next = w.map((x, i) => x - r.lr * up[i] - (r.optimizer === 'optim::AdamW' ? r.lr * r.weight_decay * x : 0));
    }
    close(r.path[k + 1].w1, next[0], 2e-3, `${r.id} lr ${r.lr} w1 step ${k}`);
    close(r.path[k + 1].w2, next[1], 2e-3, `${r.id} lr ${r.lr} w2 step ${k}`);
  }
}

test('the two problems: every recorded loss is c/2 · w² summed, on the paths and on the grids', () => {
  assert.deepEqual(V.curvature, [1, 100]);
  assert.deepEqual(B.curvature, [4, 4], 'the round bowl is equally steep in both directions');
  for (const [curv, race, surface] of [[V.curvature, data.race, V.surface], [B.curvature, B.race, B.surface]]) {
    for (const r of race) r.path.forEach((q, k) => close(q.loss, lossAt(curv, q.w1, q.w2), 1e-5, `${r.id} step ${k}`));
    surface.loss.forEach((row, j) => row.forEach((L, i) => close(L, lossAt(curv, surface.w1[i], surface.w2[j]), 1e-5, `grid ${i},${j}`)));
  }
  data.race_surface.loss.forEach((row, j) => row.forEach((L, i) => close(L, lossAt(V.curvature, data.race_surface.w1[i], data.race_surface.w2[j]), 1e-5, `race grid ${i},${j}`)));
  // "the lowest point at w1 = w2 = 0" and the start of every run
  assert.equal(V.surface.w1[45], 0);
  assert.deepEqual(SW.start.map((v) => Math.fround(v)), [Math.fround(-4), Math.fround(0.4)]);
  assert.deepEqual(B.start, SW.start);
  for (const r of [...data.race, ...B.race]) { close(r.path[0].w1, -4, 1e-6, 'start'); close(r.path[0].w2, 0.4, 1e-6, 'start'); }
});

test('every race path follows the update rule of its optimizer in TorC (src/optim.cpp), recomputed step by step', () => {
  for (const r of data.race) replay(r, V.curvature);
  for (const r of B.race) if (r.path.length > 1) replay(r, B.curvature);
  for (const r of V.runs) replay({ ...r, weight_decay: 0 }, V.curvature);
});

test('the sweep: ten rates; each race runs at the best rate of its optimizer and ends at the best final loss', () => {
  assert.equal(SW.lrs.length, 10);
  assert.equal(SW.lrs[0], 0.001);
  assert.equal(SW.lrs[9], 3);
  for (const [sw, race] of [[SW, data.race], [B.sweep, B.race]]) {
    assert.deepEqual(race.map((r) => r.id), ['sgd', 'momentum', 'adam', 'adamw']);
    for (const o of sw.optimizers) {
      const r = raceRun(race, o.id);
      assert.equal(r.lr, o.best.lr, o.id);
      close(r.final_loss, o.best.final_loss, 1e-6, `${o.id} best`);
      const ok = o.runs.filter((x) => !x.diverged);
      assert.equal(Math.min(...ok.map((x) => x.final_loss)), o.best.final_loss, `${o.id}: best is the lowest final loss that did not diverge`);
    }
  }
});

test('setup: plain SGD is safe at 0.019, explodes at 0.03, and after 50 steps w1 is still about −1.5', () => {
  assert.equal(SW.lrs[3], 0.019);
  assert.equal(SW.lrs[4], 0.03);
  const sgd = sweepOf(SW, 'sgd');
  assert.equal(sgd.best.lr, SW.lrs[3]);
  assert.equal(sgd.runs[3].diverged, false);
  assert.ok(sgd.runs[4].diverged && sgd.runs[4].final_loss > 1e6, 'the loss explodes');
  assert.equal(V.steps, 50);
  const w1 = raceRun(data.race, 'sgd').path[50].w1;
  assert.ok(w1 < -1.4 && w1 > -1.6);
  // the steps across the valley grow at 0.03: plain SGD multiplies w2 by 1 - lr·100 = -2 at each step
  assert.ok(Math.abs(1 - SW.lrs[4] * V.curvature[1]) > 1 && Math.abs(1 - SW.lrs[3] * V.curvature[1]) < 1);
});

test('momentum: mu 0.9; its loss rises at some steps, ends far below plain SGD, and survives 0.03 better than plain SGD at its best', () => {
  const mom = sweepOf(SW, 'momentum');
  close(mom.momentum, 0.9, 1e-6, 'mu');
  const path = raceRun(data.race, 'momentum').path;
  const rises = path.slice(1).filter((q, k) => q.loss > path[k].loss).length;
  assert.ok(rises >= 5, `rises at ${rises} steps`);
  assert.ok(final(data.race, 'momentum') * 10 < final(data.race, 'sgd'), 'far lower');
  // the predict reveal: across the valley it swings (w2 changes sign), along the floor it moves faster than plain SGD
  assert.ok(path.some((q) => q.w2 < 0));
  assert.ok(Math.abs(path[20].w1) < Math.abs(raceRun(data.race, 'sgd').path[20].w1));
  // section 2: at 0.03 plain SGD blew up, momentum ended lower than plain SGD at its own best rate
  assert.ok(sweepOf(SW, 'sgd').runs[4].diverged);
  assert.ok(!mom.runs[4].diverged && mom.runs[4].final_loss < sweepOf(SW, 'sgd').best.final_loss);
  // the wrong option "it blows up" is wrong at a safe rate
  for (const r of mom.runs.slice(0, 5)) assert.equal(r.diverged, false);
});

test('Adam: the first step moves each parameter by about lr, plain SGD by lr times the gradient', () => {
  const [sgd, adam] = data.first_step;
  assert.equal(sgd.optimizer, 'optim::SGD');
  assert.equal(adam.optimizer, 'optim::Adam');
  assert.deepEqual(adam.grad.map((g) => Math.fround(g)), [Math.fround(0.001), 1000]);
  adam.change.forEach((c) => close(c, -adam.lr, 1e-4, 'adam change'));
  sgd.change.forEach((c, i) => close(c, -sgd.lr * sgd.grad[i], 1e-3, 'sgd change'));
  assert.equal(adam.lr, sgd.lr);
  const q = lesson.sections.find((s) => s.id === 'first-step');
  assert.equal(q.options.findIndex((o) => o.correct), 0);
});

test('AdamW: a zero-gradient weight is multiplied by 1 − lr·λ = 0.99 per step; step() skips a parameter with no gradient', () => {
  const [adam, adamw] = data.decay;
  const f = 1 - data.decay_setup.lr * data.decay_setup.weight_decay;
  assert.equal(data.derived.decay_factor, f);
  assert.equal(data.decay_setup.lr, data.decay_setup.weight_decay, 'the page says lr = λ');
  for (const run of [adam, adamw]) run.steps.slice(0, -1).forEach((s) => { assert.equal(s.w_has_grad, true); assert.equal(s.w2_grad, 0); assert.equal(s.u_has_grad, false); });
  adam.steps.forEach((s) => { assert.equal(s.w2, 3); assert.equal(s.u, 3); });
  adamw.steps.forEach((s, k) => { close(s.w2, 3 * f ** k, 1e-5, `adamw w2 step ${k}`); assert.equal(s.u, 3, 'no gradient, no leak'); });
  const q = lesson.sections.find((s) => s.id === 'decay-2');
  close(T.num(q.answer), adamw.steps[1].w2 * f, 1e-6, 'your turn');
  assert.ok(q.tolerance < Math.abs(adamw.steps[1].w2 - adamw.steps[2].w2), 'the tolerance does not accept the old value');
});

test('which optimizer is best: Adam lowest on the valley, plain SGD lowest on the round bowl, Adam worse than SGD at SGD\'s rate', () => {
  const ids = ['sgd', 'momentum', 'adam', 'adamw'];
  const order = (race) => ids.slice().sort((a, b) => final(race, a) - final(race, b));
  assert.deepEqual(order(data.race).slice(0, 2), ['adam', 'adamw'], 'Adam and AdamW end lowest on the valley');
  assert.equal(order(data.race)[0], 'adam');
  assert.equal(order(data.race).at(-1), 'sgd', 'plain SGD ends highest on the valley');
  assert.equal(order(B.race)[0], 'sgd', 'plain SGD ends lowest on the bowl');
  for (const id of ['momentum', 'adam', 'adamw']) assert.ok(final(B.race, id) > 1e-3, `${id} ends higher on the bowl`);
  // "plain SGD reaches the lowest point within a few steps"
  const k = raceRun(B.race, 'sgd').path.findIndex((q) => q.loss < FLOOR);
  assert.ok(k > 0 && k <= 5, `below the floor at step ${k}`);
  // Adam at the best rate of plain SGD ends higher than plain SGD (the valley)
  assert.ok(sweepOf(SW, 'adam').runs[3].final_loss > sweepOf(SW, 'sgd').runs[3].final_loss);
  // the transfer feedback: on the round bowl, plain SGD beat Adam
  assert.ok(final(B.race, 'sgd') < final(B.race, 'adam'));
});

test('every number in the prose is bound to the trace, with no hand-typed digits outside formulas and code', () => {
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
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>')
    .replace(/<[^>]+>/g, ' ').replace(/[₀-₉]/g, '').replace(/(^|\s)1 −/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
  // the valley formula shows the recorded curvature, and the page never claims that one optimizer is always best
  assert.match(raw, /w_2\^2/);
  assert.match(raw, new RegExp(`${V.curvature[1]}\\\\,w_2`));
  assert.doesNotMatch(raw, /always the best optimizer<\/label>[\s\S]*correct/);
});

test('widget helpers: raceOf follows the race of each problem; the loss axis floors at 0.0001', () => {
  assert.deepEqual(RUNS.map((r) => r[0]), ['sgd', 'momentum', 'adam', 'adamw']);
  assert.deepEqual(raceOf(data, 'valley', ['sgd', 'momentum']).map((r) => r.run.lr), [data.race[0].lr, data.race[1].lr]);
  assert.equal(raceOf(data, 'bowl')[0].run, B.race[0]);
  assert.equal(logLoss(0), Math.log10(FLOOR));
  assert.equal(logLoss(100), 2);
  // the highest contour level of each problem is about its start loss, so the start sits on the outer line
  close(LEVELS.valley.at(-1), SW.start_loss, 0.05, 'valley');
  close(LEVELS.bowl.at(-1), B.sweep.start_loss, 0.05, 'bowl');
  // every path stays on its map
  for (const [race, S] of [[data.race, surfaceOf(data, 'valley')], [B.race, surfaceOf(data, 'bowl')]]) for (const r of race) for (const q of r.path) {
    assert.ok(q.w1 >= S.w1[0] - 0.01 && q.w1 <= S.w1.at(-1) + 0.01 && q.w2 >= S.w2[0] - 0.01 && q.w2 <= S.w2.at(-1) + 0.01, `${r.id} off the map`);
  }
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/optimizer-widgets.js');
  assert.equal(typeof m.mountOptimizerRace, 'function');
  assert.equal(typeof document, 'undefined');
});
