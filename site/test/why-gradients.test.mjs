// Lesson why-gradients (template 3): the JS model behind the widgets against the numbers TorC recorded (data/why-gradients.json),
// the pure helpers of the nudge lens and the two bowls, every claim that the prose makes (checked from the data, not only that the
// numbers are bound), the notation and naming rules of this lesson, all LaTeX, and that the widgets import without a DOM.
// The v3 rules (roles, budgets, glossary, notation order) run on this lesson in lessons.test.mjs.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { buildGraph, runBackward } from '../js/engine.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { makeModel, recordedTrajectory } from '../js/widgets/landscape-model.js';
import { fmtX, texX, fmt3, updateTex } from '../js/widgets/landscape-tex.js';
import { PRESET_LRS, MAX_STEPS } from '../js/widgets/bowl.js';
import { nudgeRows, lensView } from '../js/widgets/nudge-lens.js';
import { bowlRows, lossOn } from '../js/widgets/two-bowls.js';
import { whyLayout } from '../js/lessons/why-gradients.scenes.js';
import { readJson, renderStrictAll, course } from './helpers.mjs';

const data = readJson('data/why-gradients.json');
const D = buildD({ 'why-gradients': data }, anchorValues(readJson('data/anchors.json')));
const scene = D.scenes.why;
const m = makeModel(scene);
const T = makeT(D);
const tol = (a, b, t = 1e-5) => Math.abs(a - b) <= t * (1 + Math.abs(b));      // float32 recording vs double port (docs/trace-schema.md)
const tr = data.trajectories, steep = data.steep;
const SLOW = 0, FAST = 1, BOUNCE = 3, DIVERGE = 4;

test('the trace file is the one the lesson expects (scene why, five learning rates, 33 grid points, the steep bowl)', () => {
  assert.equal(data.kind, 'why');
  assert.equal(scene.nodes.map((n) => n.name).join(','), 'w,x,t,p,e,L');
  assert.equal(data.grid.length, 33);
  assert.deepEqual(tr.map((t) => t.lr), [0.05, 0.1, 0.125, 0.25, 0.3]);   // the lesson's SLOW/FAST/BOUNCE/DIVERGE indices
  for (const lr of PRESET_LRS) assert.ok(recordedTrajectory(data, lr), `the widget's preset lr ${lr} is a recorded run`);
  assert.deepEqual(PRESET_LRS, [SLOW, FAST, BOUNCE, DIVERGE].map((i) => tr[i].lr), 'the fog bowl offers the four runs the lesson talks about');
  assert.equal(m.x, data.setup.x);
  assert.equal(m.t, data.setup.t);
  assert.equal(m.wMin, data.setup.w_min);
  assert.equal(m.criticalLr, data.setup.critical_lr);
  assert.deepEqual(data.secants.map((s) => s.h), [1, 0.5, 0.1, 0.01, 0.001]);
  assert.equal(steep.trajectories[0].lr, 0.05, 'the transfer quiz uses the steep run with lr 0.05');
  assert.deepEqual(steep.compare.map((c) => c.w), [0, 1, data.probes[3].w, 6]);
});

// ---- the JS model (bowl widget) == TorC --------------------------------------------------------------------------------------
test('(a) buildGraph/runBackward at every grid point and probe equals what TorC recorded', () => {
  for (const r of [...data.grid, ...data.probes]) {
    const g = buildGraph(scene, { w: [r.w] }), bw = runBackward(g);
    const v = (n) => g.byName[n].value.v[0];
    assert.ok(tol(v('p'), r.p) && tol(v('e'), r.e) && tol(v('L'), r.L), `forward at w=${r.w}`);
    assert.ok(tol(bw.grad.w.v[0], r.grad), `grad at w=${r.w}: ${bw.grad.w.v[0]} vs ${r.grad}`);
    const q = m.at(r.w);
    assert.ok(tol(q.L, r.L) && tol(q.grad, r.grad), `model.at(${r.w})`);
  }
  assert.equal(m.at(0).grad, scene.nodes.find((n) => n.name === 'w').grad.values[0]);
});

test('(c) grad == 8(w-3) and L == 4(w-3)^2 on the grid and at 500 other weights', () => {
  assert.equal(m.curvature, 8);
  for (const r of data.grid) assert.ok(tol(r.grad, 8 * (r.w - 3), 1e-12) && tol(r.L, 4 * (r.w - 3) ** 2, 1e-12), `recorded w=${r.w}`);
  for (let i = 0; i <= 500; i++) {
    const w = -3 + (i * 14) / 500, q = m.at(w);
    assert.ok(tol(q.grad, m.curvature * (w - m.wMin), 1e-12), `grad at ${w}`);
    assert.ok(tol(q.L, 4 * (w - m.wMin) ** 2, 1e-12), `L at ${w}`);
    assert.ok(tol(m.localSlopes(w).reduce((a, s) => a * s.slope, 1), q.grad, 1e-12), `chain product at ${w}`);
  }
});

test('(b) w -= lr*g reproduces every recorded trajectory, and the closed form (1 - 8 lr)^k holds', () => {
  for (const rec of tr) {
    const st = m.iterate(m.w0, rec.lr, 8), f = m.factor(rec.lr);
    assert.equal(st.length, rec.states.length);
    rec.states.forEach((r, k) => {
      for (const [mine, theirs, what] of [[st[k].w, r.w, 'w'], [st[k].L, r.L, 'L'], [st[k].grad, r.grad, 'grad'], [st[k].wNext, r.w_next, 'w_next'], [st[k].promise, r.tangent_L_next, 'promise']]) {
        assert.ok(tol(mine, theirs, 1e-5), `lr=${rec.lr} k=${k} ${what}: ${mine} vs ${theirs}`);
      }
      assert.ok(tol(r.w - m.wMin, f ** k * (m.w0 - m.wMin), 1e-4), `closed form, recorded lr=${rec.lr} k=${k}`);
    });
    assert.equal(rec.matches_manual, true);
    assert.ok(tol(f, rec.factor, 1e-12), `factor for ${rec.lr}`);
  }
  assert.equal(data.checks.sgd_matches_manual, true);
  assert.equal(steep.checks.sgd_matches_manual, true);
});

test('(e) |1 - 8 lr| < 1 predicts which runs settle: matches stable_predicted and the recorded loss of all five runs', () => {
  for (const rec of tr) {
    const L0 = rec.states[0].L, L8 = rec.states[8].L;
    assert.equal(m.stable(rec.lr), rec.stable_predicted, `lr=${rec.lr}`);
    assert.equal(m.stable(rec.lr), L8 < L0 - 1e-9, `observed: L8=${L8} vs L0=${L0} at lr=${rec.lr}`);
    assert.equal(Math.abs(m.factor(rec.lr)) > 1, L8 > L0 + 1e-9, `diverges iff |factor| > 1 at lr=${rec.lr}`);
  }
});

test('the nudge of the model reproduces the recorded stages and the second-order term 4 delta^2', () => {
  const rec = data.nudge, n = m.nudge(rec.w0, rec.delta);
  n.stages.forEach((s, i) => assert.ok(tol(s.after, rec.stages[i].after) && s.slope === rec.stages[i].local_slope, s.name));
  assert.equal(n.product, rec.backward_slope);
  assert.ok(Math.abs(n.actualDL - rec.actual_dL) < 1e-5 && Math.abs(n.secondOrder - rec.second_order_term) < 1e-5);
  for (const w of [-1, 0, 2, 5]) for (const d of [-0.5, -0.1, 0.01, 0.5]) assert.ok(Math.abs(m.nudge(w, d).secondOrder - 4 * d * d) < 1e-9, `${w} ${d}`);
});

// ---- nudge lens ---------------------------------------------------------------------------------------------------------------
test('nudge lens: the rows are the recorded secants, and the JS curve it draws agrees with them', () => {
  const rows = nudgeRows(data);
  assert.equal(rows.length, data.secants.length);
  rows.forEach((r, i) => {
    const s = data.secants[i];
    assert.deepEqual([r.h, r.before, r.after, r.actual, r.ratio, r.grad], [s.h, s.L0, s.L_h, s.dL, s.slope, s.backward_slope]);
    assert.equal(r.grad, data.probes[0].grad);
    assert.ok(tol(r.predicted, -24 * s.h, 1e-12), 'gradient x nudge');
    // the model draws the curve between the recorded points: double vs float32 (the 0.001 secant loses digits in float32)
    assert.ok(Math.abs(m.lossAt(s.h) - s.L_h) < 1e-5, `L(${s.h})`);
    assert.ok(Math.abs((m.lossAt(s.h) - m.lossAt(0)) / s.h - s.slope) < 1e-2, `secant at ${s.h}`);
    assert.ok(tol(s.expected_slope, 4 * s.h - 24, 1e-12), 'expected mathematics: (L(h) - L(0))/h = 4h - 24');
    assert.ok(r.gap > 0, 'the bowl bends upward, so the curve is above the straight line');
    assert.ok(Math.abs(r.gap - 4 * s.h * s.h) < 1e-2 * Math.max(s.h, 0.01), `the promise misses by about 4h² at ${s.h}`);
    // the view holds every point it draws
    const v = lensView(r, m.lossAt);
    for (const [x, y] of [[r.w0, r.before], [r.w0 + r.h, r.after], [r.w0 + r.h, r.before + r.predicted]]) {
      assert.ok(x > v.x[0] && x < v.x[1] && y > v.y[0] && y < v.y[1], `point (${x}, ${y}) inside the view for h=${r.h}`);
    }
  });
  // the task's observation: the gap shrinks faster than the nudge, so the curve looks straighter at every zoom
  const share = rows.map((r) => r.gap / Math.abs(r.predicted));
  for (let i = 1; i < share.length; i++) assert.ok(share[i] < share[i - 1], `the gap is a smaller share of the change at h=${rows[i].h}`);
  for (let i = 1; i < rows.length; i++) assert.ok(Math.abs(rows[i].ratio - rows[i].grad) < Math.abs(rows[i - 1].ratio - rows[i - 1].grad), 'the ratio comes closer to the gradient');
});

// ---- two bowls ----------------------------------------------------------------------------------------------------------------
test('two bowls: the rows are steep.compare, with the same distance and four times the gradient at every weight', () => {
  const rows = bowlRows(data);
  assert.equal(rows.length, steep.compare.length);
  for (const r of rows) {
    assert.ok(tol(r.distance, Math.abs(r.w - data.setup.w_min), 1e-9), `distance at ${r.w}`);
    assert.equal(r.ratio, 4);
    assert.ok(tol(r.steepGrad, 4 * r.grad, 1e-6), `steep gradient at ${r.w}`);
    const p = data.probes.find((q) => q.w === r.w), sp = steep.probes.find((q) => q.w === r.w);
    assert.ok(p && sp, `both bowls have a probe at ${r.w}`);
    assert.deepEqual([r.grad, r.L, r.steepGrad, r.steepL], [p.grad, p.L, sp.grad, sp.L]);
  }
  assert.ok(Math.sign(rows[3].grad) > 0 && Math.sign(rows[3].steepGrad) > 0, 'past the bottom both gradients are positive');
  // the recorded steep bowl is 16(w-3)^2 with gradient 32(w-3)
  for (const q of steep.grid) assert.ok(tol(q.L, 16 * (q.w - 3) ** 2, 1e-9) && tol(q.grad, 32 * (q.w - 3), 1e-9), `steep grid at ${q.w}`);
  const pts = data.grid.map((q) => [q.w, q.L]);
  assert.ok(Math.abs(lossOn(pts, 5.1) - 4 * 2.1 ** 2) < 0.1, 'label placement interpolates the recorded curve');
});

// ---- the lesson -------------------------------------------------------------------------------------------------------------
const lesson = (await import('../js/lessons/why-gradients.js')).default;
const parts = renderLesson(lesson, D, { course: course() });
const html = parts.map((p) => p.html).join('\n');
const shown = Object.fromEntries(boundNums(html).map(({ path, shown: v }) => [path, v]));
const F = 'files.why-gradients';
const sectionOf = (id) => parts.find((p) => p.section.id === id || p.html.includes(`id="wg-${id}"`)).html;
const plainText = (h) => h.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');

test('the lesson is a v3 lesson that lights the adjust step, and the chain rule and the graph lab have left it', () => {
  assert.equal(lesson.template, 3);
  assert.equal(lesson.loop, 'adjust');
  assert.equal(lesson.title, 'The gradient');
  assert.ok(!lesson.sections.some((s) => s.kind === 'widget' && s.id === 'chain-lab'), 'no graph lab (it moves to backprop)');
  assert.doesNotMatch(plainText(html), /chain rule|local derivative|w_\{k\+1\}/i);
  assert.deepEqual(lesson.sections.filter((s) => s.kind === 'widget').map((s) => s.id), ['nudge-lens', 'two-bowls', 'bowl']);
  const quizzes = lesson.sections.filter((s) => s.kind === 'quiz');
  assert.deepEqual(quizzes.map((q) => `${q.id}:${q.purpose}`), ['nudge:predict', 'your-step:check', 'best-lr:check', 'steep-step:transfer', 'loss-of-lines:cumulative']);
  assert.equal(quizzes.find((q) => q.purpose === 'cumulative').from, 'what-is-learning');
});

test('the story: two notebooks cost 6 euros, the loss is the squared miss, the start w = 0 has loss 36, and 6 / 2 is the price', () => {
  const p0 = data.probes[0];
  assert.deepEqual([data.setup.x, data.setup.t, p0.w, p0.p, p0.L], [2, 6, 0, 0, 36]);
  assert.equal(data.nudge.miss_before, Math.abs(p0.e));
  assert.equal(p0.L, p0.e ** 2, 'the loss is the square of the miss');
  assert.equal(data.setup.t / data.setup.x, data.setup.w_min, 'dividing gives the price');
  const s = sectionOf('price');
  assert.match(s, /<a class="term"[^>]*>loss<\/a> <code>L<\/code> measures/, 'the loss is named L where it is defined (review M1)');
  const raw = rawHtml();
  assert.ok(raw.includes(`<code>L = (${data.setup.x}w - ${data.setup.t})^2</code>`), 'the loss formula, with the constants of the trace');
  assert.equal(shown[`${F}.probes.0.L`], '36');
  assert.equal(shown[`${F}.setup.w_min`], '3', 'the bottom is named before any step (review H1)');
  assert.ok(data.probes.find((q) => q.w === data.setup.w_min).L === 0, 'at the true price the loss is zero');
  // the static picture of the whole curve: the recorded grid, the start and the bottom
  assert.match(s, /<figure class="wg-fig"><svg class="plot"[^>]*role="img"/);
  const path = s.match(/<path class="plot-curve" d="([^"]+)"/)[1];
  assert.equal(path.split(/[ML]/).filter(Boolean).length, data.grid.length, 'one point per recorded grid point');
  assert.match(s, />start<\/text>/);
  assert.match(s, />bottom<\/text>/);
  assert.equal(shown['anchors.mlp_parameters'], '26,506');
});

test('the predict quiz: nudging w to 0.01 gives the miss 5.98 and the loss 35.76, a change of about −0.24, about −24 per unit', () => {
  const n = data.nudge;
  assert.ok(tol(n.stages[0].after, data.setup.x * n.delta, 1e-6), 'the prediction is 2 × 0.01');
  assert.ok(tol(n.miss_after, n.miss_before - data.setup.x * n.delta, 1e-6));
  assert.ok(tol(n.stages[2].after, n.miss_after ** 2, 1e-6), 'the new loss is the new miss squared');
  assert.equal(lesson.sections.find((q) => q.id === 'nudge').answer, `${F}.nudge.stages.2.after`);
  assert.deepEqual([shown[`${F}.nudge.delta`], shown[`${F}.nudge.stages.0.after`], shown[`${F}.nudge.miss_after`], shown[`${F}.nudge.stages.2.after`]], ['0.01', '0.02', '5.98', '35.76']);
  assert.equal(shown[`${F}.nudge.actual_dL`], '−0.24');
});

test('section 1: a nudge ten times smaller changes the loss about ten times less, the ratio stays near −24, and the signs', () => {
  const s = data.secants;
  assert.ok(Math.abs(s[3].dL / s[4].dL - 10) < 0.1, 'ten times smaller nudge, about ten times smaller change');
  assert.ok(Math.abs(s[2].dL / s[3].dL - 10) < 0.2);
  for (const q of s.slice(2)) assert.ok(Math.abs(q.slope - data.probes[0].grad) < 0.5, `ratio near −24 at h=${q.h}`);
  // a negative gradient: a larger w gives a smaller loss; a positive one: a smaller w gives a smaller loss
  const [p0, p1] = data.probes, p6 = data.probes[5], p72 = data.probes[6];
  assert.ok(p0.grad < 0 && p1.w > p0.w && p1.L < p0.L);
  assert.ok(p6.grad > 0 && p72.w > p6.w && p72.L > p6.L);
  // "slope" appears once, to define the gradient (decision 14); dL/dw is read aloud
  const prose = plainText(html.replace(/<pre[\s\S]*?<\/pre>/g, ' '));
  assert.equal((prose.match(/\bslope/gi) ?? []).length, 1, 'the one allowed use of "slope"');
  assert.match(prose, /This number is the gradient \. It is the slope of the loss curve at the current point/);
  // the gradient is the number the ratio approaches (review M2): every recorded ratio is closer to it than the one before
  for (let i = 1; i < s.length; i++) assert.ok(Math.abs(s[i].slope - s[i].backward_slope) < Math.abs(s[i - 1].slope - s[i - 1].backward_slope));
  assert.match(prose, /comes closer and closer to one number/);
  assert.doesNotMatch(prose, /dL\/dw/, 'no dL/dw in this lesson');
});

test('section 2 and the learner step: 0 − 0.05 × (−24) = 1.2, then 1.2 − 0.05 × (−14.4) = 1.92, steps shrink, 60% kept', () => {
  const t0 = tr[SLOW];
  const [s0, s1] = t0.states;
  assert.ok(tol(s0.w - t0.lr * s0.grad, s0.w_next, 1e-6) && tol(s0.w_next, s1.w, 0));
  assert.ok(tol(s1.w - t0.lr * s1.grad, s1.w_next, 1e-6));
  assert.ok(s0.w < s0.w_next && s0.w_next < data.setup.w_min, 'w goes up, toward the true price');
  assert.deepEqual([shown[traj(SLOW, 'states.0.w_next')], shown[traj(SLOW, 'states.1.w')], shown[traj(SLOW, 'states.1.grad')], shown[traj(SLOW, 'states.1.w_next')]], ['1.2', '1.2', '−14.4', '1.92']);
  assert.ok(tol(s0.step, s0.w_next - s0.w, 1e-6) && tol(s1.step, s1.w_next - s1.w, 1e-6) && s1.step < s0.step, 'the step shrank');
  for (const st of t0.states) assert.ok(Math.abs(st.kept - 0.6) < 1e-5, `each step keeps 60 % (k=${st.k})`);
  assert.equal(shown[traj(SLOW, 'kept_percent')], '60');
  // the worked distance line that defines "keeps" (review H1): 3, then 1.8, then 1.08, each 60 % of the one before
  assert.deepEqual([s0.distance, s1.distance, s1.distance_next].map((v) => Number(v.toPrecision(4))), [3, 1.8, 1.08]);
  assert.ok(tol(s1.distance / s0.distance, 0.6, 1e-6) && tol(s1.distance_next / s1.distance, 0.6, 1e-6));
  assert.deepEqual([shown[traj(SLOW, 'states.0.distance')], shown[traj(SLOW, 'states.1.distance')], shown[traj(SLOW, 'states.1.distance_next')]], ['3', '1.8', '1.08']);
  assert.ok(s1.grad > s0.grad && s1.grad < 0, 'the curve is flatter at 1.2 than at 0');
  // the arrow is read "replace w by", lr is upright, the gradient has no letter g, and the real line is L.backward()
  const raw = rawHtml();
  assert.match(plainText(html), /Programs write “new w =” as w ← \. Read the arrow as “replace w by”/);
  assert.equal((html.match(/data-sym="arrow"/g) ?? []).length, 3, 'the arrow is introduced once and used in two worked lines (review L2)');
  assert.match(html, /This line finds the gradient without nudges, and a later lesson shows how/);
  assert.doesNotMatch(plainText(html), /w\.grad\(\)/, 'no C++ call in the prose (review L1)');
  assert.doesNotMatch(raw, /\\mathit\{lr\}|\\cdot g\b/);
  assert.match(html, /<figure class="real-line"><pre[^>]*><code>L\.backward\(\);<\/code><\/pre>/);
});

test('section 3: four notebooks cost 12 euros, same bottom and distance, gradient −96, the miss and the effect of w both double', () => {
  const s0 = steep.probes[0], p0 = data.probes[0];
  assert.deepEqual([steep.setup.x, steep.setup.t, steep.setup.w_min, s0.w], [4, 12, data.setup.w_min, p0.w]);
  assert.equal(s0.grad, -96);
  assert.equal(s0.grad, 4 * p0.grad, 'four times larger');
  assert.equal(s0.e, 2 * p0.e, 'the miss is twice as large');
  assert.equal(steep.setup.x, 2 * data.setup.x, 'each nudge changes the prediction twice as much');
  assert.equal(steep.compare[0].distance, Math.abs(p0.w - data.setup.w_min), 'the same distance');
  assert.deepEqual([shown[`${F}.steep.probes.0.grad`], shown[`${F}.probes.0.grad`]], ['−96', '−24']);
  assert.match(sectionOf('steep'), /four times larger/);
});

function traj(i, rest) { return `${F}.trajectories.${i}.${rest}`; }
function rawHtml() {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
}

test('the fog bowl tasks and the learning-rate section: 0.05 settles, 0.25 bounces 0, 6, 0, 0.3 grows, shares 60/20/100/140 %', () => {
  const slow = tr[SLOW], fast = tr[FAST], bounce = tr[BOUNCE], div = tr[DIVERGE];
  for (let k = 1; k < 9; k++) assert.ok(slow.states[k].L < slow.states[k - 1].L && Math.abs(slow.states[k].step) < Math.abs(slow.states[k - 1].step), `0.05 settles, k=${k}`);
  assert.ok(Math.abs(slow.states[8].grad) < 0.5, 'after eight steps the gradient is near zero');
  assert.deepEqual(bounce.states.slice(0, 5).map((s) => s.w), [0, 6, 0, 6, 0]);
  assert.ok(bounce.states.every((s) => s.crossed && s.kept === 1 && s.L === 36), 'as far past the bottom as it started, forever');
  for (let k = 1; k < 9; k++) assert.ok(div.states[k].distance > div.states[k - 1].distance && div.states[k].L > div.states[k - 1].L, `0.3 jumps farther, k=${k}`);
  // best-lr: after eight steps, lr 0.1 is closest to the bottom
  const after8 = [SLOW, FAST, BOUNCE, DIVERGE].map((i) => Math.abs(tr[i].states[8].w - data.setup.w_min));
  assert.equal(Math.min(...after8), after8[1]);
  assert.equal(lesson.sections.find((q) => q.id === 'best-lr').options.findIndex((o) => o.correct), 1);
  // a larger rate keeps a smaller share, up to a point; the shares the text and the reveal state
  assert.ok(fast.kept < slow.kept && bounce.kept > fast.kept);
  const human = Object.fromEntries(boundNums(html).filter((b) => b.human).map((b) => [b.path, b.shown]));
  assert.deepEqual([traj(FAST, 'kept'), traj(SLOW, 'kept'), traj(BOUNCE, 'kept'), traj(DIVERGE, 'kept')].map((p) => human[p]), ['20%', '60%', '100%', '140%']);
  assert.ok(tol(fast.factor, 1 - 8 * fast.lr, 1e-12) && tol(div.factor, -1.4, 1e-12), 'expected mathematics: the share is |1 − 8 lr|');
});

test('the transfer quiz: on the steep bowl, 0 − 0.05 × (−96) = 4.8 lands past the bottom', () => {
  const s = steep.trajectories[0], s0 = s.states[0];
  assert.ok(tol(s0.w - s.lr * s0.grad, s0.w_next, 1e-6));
  assert.ok(Math.abs(s0.w_next - 4.8) < 1e-5);
  assert.ok(s0.w_next > steep.setup.w_min && s0.crossed, 'past the bottom');
  assert.ok(Math.abs(s0.step) > Math.abs(tr[SLOW].states[0].step), 'the same learning rate makes a longer step on steeper ground');
  assert.equal(lesson.sections.find((q) => q.id === 'steep-step').answer, `${F}.steep.trajectories.0.states.0.w_next`);
});

test('the cumulative quiz (what-is-learning): the mean of the squared misses, line A one, line B two, the same total miss', () => {
  const A = Array(10).fill(1), B = [...Array(5).fill(0), ...Array(5).fill(2)];
  const mse = (m) => m.reduce((a, v) => a + v * v, 0) / m.length, total = (m) => m.reduce((a, v) => a + v, 0);
  assert.deepEqual([mse(A), mse(B), total(A), total(B)], [1, 2, 10, 10]);
  assert.equal((2 * 2) / (1 * 1), 4, 'one miss of two points counts as much as four misses of one point');
  const q = lesson.sections.find((x) => x.id === 'loss-of-lines');
  assert.equal(q.options.findIndex((o) => o.correct), 0);
  assert.equal(q.from, 'what-is-learning');
});

test('the SGD::step() line is not on this page (it is the real line of sgd-step), and there is no In TorC panel', () => {
  assert.doesNotMatch(html, /param-&gt;data\(\)|class="in-torc"/);
});

test('the best-lr prompt names the start and the number of steps from the trace (review M7, L8)', () => {
  assert.equal(shown[traj(FAST, 'states.8.k')], '8');
  assert.equal(tr[FAST].states.length - 1, 8);
  assert.equal(shown[traj(FAST, 'states.0.w')], '0');
});

test('every number is bound, and no hand-typed digit survives in the prose, quizzes or tasks', () => {
  for (const { section, html: h } of parts) assert.deepEqual(findUnboundNums(h), [], section.kind);
  const digits = rawHtml()
    .replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code>[\s\S]*?<\/code>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '')
    .replace(/<h[23][^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(digits, []);
  assert.equal(lesson.footer, undefined, 'the standard one-line footer');
});

// ---- KaTeX and number formats ---------------------------------------------------------------------------------------------
test('every LaTeX string of the bowl widget and of the lesson parses in KaTeX (throwOnError)', () => {
  let count = 0;
  for (const lr of PRESET_LRS) {
    for (let w = -1; w <= 7.0001; w += 0.05) { renderStrictAll([updateTex(m, w, lr)]); count++; }
    const st = m.iterate(m.w0, lr, MAX_STEPS);
    renderStrictAll(st.map((q) => updateTex(m, q.w, lr)));
    count += st.length;
  }
  assert.ok(count > 600);
  assert.doesNotMatch(html, /katex-error/);
  const formulas = [...rawHtml().matchAll(/<code>([^<]*\^2|\dw)<\/code>/g)].map((x) => x[1].replace(/&amp;/g, '&'));
  assert.ok(formulas.length >= 2, formulas.join(' | '));
  renderStrictAll(formulas);
  assert.equal(updateTex(m, 0, 0.05), '\\text{new } w = 0 - 0.05 \\times (-24) = 1.2');
});

test('readouts use 3 significant figures and the minus sign; huge values use exponent form', () => {
  assert.equal(fmt3(-23.9601135), '−24');
  assert.equal(fmt3(-2.3599968), '−2.36');
  assert.equal(fmt3(-0.0239944458), '−0.024');
  assert.equal(fmt3(35.7603989), '35.8');
  assert.equal(fmt3(1.20000005), '1.2');
  assert.equal(fmt3(1.19e-10), '0');
  assert.equal(fmt3(8.219e18), '8.22e18');
  assert.equal(fmt3(-2.93e21), '−2.93e21');
  assert.equal(fmtX(8.219e18), '8.219e18');
  assert.equal(texX(-1.68e7), '-1.68\\times 10^{7}');
  const st = m.iterate(m.w0, 0.35, 40);
  for (const s of st) for (const v of [s.w, s.L, s.grad]) assert.doesNotMatch(fmt3(v), /\d{7,}/, `${v}`);
  renderStrictAll(st.map((q) => updateTex(m, q.w, 0.35)));
  assert.doesNotMatch(updateTex(m, -2.1e6, 0.3), /\de\d/);
});

// ---- widgets import without a DOM; the graph-lab layout stays for the backprop lesson ---------------------------------------
test('the widgets import in Node without touching the DOM', async () => {
  assert.equal(typeof document, 'undefined');
  for (const f of ['bowl.js', 'nudge-lens.js', 'two-bowls.js']) {
    const mod = await import(`../js/widgets/${f}`);
    assert.ok(Object.values(mod).some((v) => typeof v === 'function' && /^mount/.test(v.name)), f);
  }
  assert.equal(typeof document, 'undefined');
});

test('the graph-lab layout of scene why (kept for the lesson backprop) binds its numbers and places every node', () => {
  const lay = whyLayout(T);
  for (const field of ['goal', 'intro', 'forwardHint', 'backwardHint', 'takeaway']) {
    assert.deepEqual(findUnboundNums(lay[field]), [], field);
    for (const { path, shown: v } of boundNums(lay[field])) assert.equal(T.text(path), v, `${field}: ${path}`);
  }
  assert.deepEqual(new Set(Object.keys(lay.pos)), new Set(scene.nodes.map((n) => n.name)));
  for (const [name, [x, y]] of Object.entries(lay.pos)) assert.ok(x > 0 && x < lay.view[0] && y > 0 && y < lay.view[1], `${name} inside the view`);
});
