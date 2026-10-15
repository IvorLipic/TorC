// Lesson why-gradients: the JS model behind the bowl widget against the numbers TorC recorded (data/why-gradients.json),
// the stability predicate, every number in the prose, the style budget, all LaTeX, and that the widget imports without a DOM.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, runBackward } from '../js/engine.js';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { makeModel, recordedTrajectory } from '../js/widgets/landscape-model.js';
import { fmtX, texX, updateTex } from '../js/widgets/landscape-tex.js';
import { PRESET_LRS, MAX_STEPS } from '../js/widgets/bowl.js';
import { whyLayout } from '../js/lessons/why-gradients.scenes.js';
import { readJson, renderStrictAll, lessonStats } from './helpers.mjs';

const data = readJson('data/why-gradients.json');
const D = buildD({ 'why-gradients': data });
const scene = D.scenes.why;
const m = makeModel(scene);
const T = makeT(D);
const tol = (a, b, t = 1e-5) => Math.abs(a - b) <= t * (1 + Math.abs(b));      // float32 recording vs double port (docs/trace-schema.md)

test('the trace file is the one the lesson expects (scene why, five learning rates, 33 grid points)', () => {
  assert.equal(data.kind, 'why');
  assert.equal(scene.nodes.map((n) => n.name).join(','), 'w,x,t,p,e,L');
  assert.equal(data.grid.length, 33);
  assert.deepEqual(data.trajectories.map((t) => t.lr), [0.05, 0.1, 0.125, 0.25, 0.3]);   // the lesson's SLOW/EXACT/BOUNCE/DIVERGE indices
  for (const lr of PRESET_LRS) assert.ok(recordedTrajectory(data, lr), `the widget's preset lr ${lr} is a recorded run`);
  assert.equal(m.x, data.setup.x);
  assert.equal(m.t, data.setup.t);
  assert.equal(m.wMin, data.setup.w_min);
  assert.equal(m.criticalLr, data.setup.critical_lr);
});

// ---- (a) engine port == TorC's grid and probes ----------------------------------------------------------------------------
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

// ---- (c) the identity ------------------------------------------------------------------------------------------------------
test('(c) grad == 8(w-3) and L == 4(w-3)^2 on the grid and at 500 other weights', () => {
  assert.equal(m.curvature, 8);
  for (const r of data.grid) assert.ok(tol(r.grad, 8 * (r.w - 3), 1e-12) && tol(r.L, 4 * (r.w - 3) ** 2, 1e-12), `recorded w=${r.w}`);
  for (let i = 0; i <= 500; i++) {
    const w = -3 + (i * 14) / 500, q = m.at(w);
    assert.ok(tol(q.grad, m.curvature * (w - m.wMin), 1e-12), `grad at ${w}`);
    assert.ok(tol(q.L, 4 * (w - m.wMin) ** 2, 1e-12), `L at ${w}`);
    // the chain rule: the product of the local slopes IS the gradient
    const prod = m.localSlopes(w).reduce((a, s) => a * s.slope, 1);
    assert.ok(tol(prod, q.grad, 1e-12), `chain product at ${w}`);
  }
});

// ---- (b) iteration reproduces every trajectory --------------------------------------------------------------------------------
test('(b) w -= lr*g reproduces every recorded trajectory (w, p, e, L, grad, next w, tangent promise)', () => {
  for (const rec of data.trajectories) {
    const st = m.iterate(m.w0, rec.lr, 8);
    assert.equal(st.length, rec.states.length);
    rec.states.forEach((r, k) => {
      for (const [mine, theirs, what] of [[st[k].w, r.w, 'w'], [st[k].p, r.p, 'p'], [st[k].e, r.e, 'e'], [st[k].L, r.L, 'L'], [st[k].grad, r.grad, 'grad'], [st[k].wNext, r.w_next, 'w_next'], [st[k].promise, r.tangent_L_next, 'promise']]) {
        assert.ok(tol(mine, theirs, 1e-5), `lr=${rec.lr} k=${k} ${what}: ${mine} vs ${theirs}`);
      }
    });
    assert.equal(rec.matches_manual, true);
    assert.equal(recordedTrajectory(data, rec.lr), rec);
    assert.ok(tol(m.factor(rec.lr), rec.factor, 1e-12), `factor for ${rec.lr}`);
  }
  assert.equal(data.checks.sgd_matches_manual, true);
});

test('the closed form w_k - 3 = (1 - 8 lr)^k (w_0 - 3) holds for the model and the recording', () => {
  for (const rec of data.trajectories) {
    const f = m.factor(rec.lr), st = m.iterate(m.w0, rec.lr, 8);
    st.forEach((s, k) => assert.ok(tol(s.w - m.wMin, f ** k * (m.w0 - m.wMin), 1e-9), `lr=${rec.lr} k=${k}`));
    rec.states.forEach((s, k) => assert.ok(tol(s.w - m.wMin, f ** k * (m.w0 - m.wMin), 1e-4), `recorded lr=${rec.lr} k=${k}`));
  }
});

// ---- (e) stability predicate -----------------------------------------------------------------------------------------------------
test('(e) |1 - 8 lr| < 1 predicts divergence: matches stable_predicted and the recorded loss of all five runs', () => {
  for (const rec of data.trajectories) {
    const L0 = rec.states[0].L, L8 = rec.states[8].L;
    assert.equal(m.stable(rec.lr), rec.stable_predicted, `lr=${rec.lr}`);
    assert.equal(m.stable(rec.lr), L8 < L0 - 1e-9, `observed: L8=${L8} vs L0=${L0} at lr=${rec.lr}`);   // lr 0.25 stays at 36: not stable
    assert.equal(Math.abs(m.factor(rec.lr)) > 1, L8 > L0 + 1e-9, `diverges iff |factor| > 1 at lr=${rec.lr}`);
  }
  assert.equal(m.newtonLr, 0.125);
  assert.equal(m.factor(m.newtonLr), 0);
  assert.equal(m.factor(m.criticalLr), -1);
  // beyond the recordings: the predicate agrees with simulation on a sweep
  for (let lr = 0.01; lr <= 0.35; lr += 0.01) {
    const st = m.iterate(m.w0, lr, 60), grows = st[60].L > st[0].L * (1 + 1e-9);
    if (Math.abs(lr - 0.25) > 1e-6) assert.equal(!m.stable(lr), grows, `lr=${lr}`);
  }
});

// ---- nudge (chain rule) ------------------------------------------------------------------------------------------------------------
test('the nudge widget reproduces the recorded stages, products and second-order term', () => {
  const rec = data.nudge, n = m.nudge(rec.w0, rec.delta);
  n.stages.forEach((s, i) => {
    const r = rec.stages[i];
    assert.equal(s.name, r.name);
    assert.equal(s.from, r.from);
    assert.equal(s.slope, r.local_slope);
    assert.ok(tol(s.before, r.before) && tol(s.after, r.after) && Math.abs(s.change - r.change) < 1e-5, `${s.name}`);
  });
  assert.equal(n.product, rec.local_slope_product);
  assert.equal(n.product, rec.backward_slope);
  assert.ok(Math.abs(n.predictedDL - rec.predicted_dL) < 1e-7 && Math.abs(n.actualDL - rec.actual_dL) < 1e-5 && Math.abs(n.secondOrder - rec.second_order_term) < 1e-5);
  assert.ok(Math.abs(n.secondOrder - (m.curvature / 2) * rec.delta ** 2) < 1e-12, 'second-order term is 4 delta^2');
  for (const w of [-1, 0, 2, 5]) for (const d of [-0.5, -0.1, 0.01, 0.5]) {
    const k = m.nudge(w, d);
    assert.ok(Math.abs(k.secondOrder - 4 * d * d) < 1e-9, `${w} ${d}`);
    assert.ok(Math.abs(k.product - m.at(w).grad) < 1e-12);
  }
});

// ---- numbers in prose ---------------------------------------------------------------------------------------------------------------------
const lesson = (await import('../js/lessons/why-gradients.js')).default;
const html = renderLesson(lesson, D).map((p) => p.html).join('\n');

test('every number in the lesson prose is bound to the trace, and the bound text is what T() gives', () => {
  for (const { section, html: h } of renderLesson(lesson, D)) {
    assert.deepEqual(findUnboundNums(h), [], section.kind);
    for (const { path, shown } of boundNums(h)) assert.equal(T.text(path), shown, path);
  }
  // the numbers the lesson's argument rests on: L = 36 and slope -24 at w = 0, one step at lr 0.05 to 1.2 / 12.96, the bounce 6, 0, 36
  const shown = Object.fromEntries(boundNums(html).map(({ path, shown: v }) => [path, v]));
  const t = 'files.why-gradients.trajectories';
  assert.deepEqual([shown['why.L.value'], shown['why.w.grad'], shown[`${t}.0.states.1.w`], shown[`${t}.0.states.1.L`], shown[`${t}.3.states.1.w`], shown[`${t}.3.states.2.w`], shown[`${t}.3.states.8.L`], shown['why.e.grad']],
    ['36', '-24', '1.2', '12.96', '6', '0', '36', '-12']);
  assert.deepEqual([shown[`${t}.2.lr`], shown[`${t}.3.lr`], shown[`${t}.4.lr`]], ['0.125', '0.25', '0.3'], 'exact / bounce / diverge');
});

test('the lesson keeps to the style budget (docs/authoring.md)', () => {
  const st = lessonStats(lesson, D, renderLesson, boundNums);
  assert.ok(st.words <= 800, `${st.words} words`);
  assert.ok(st.numbers <= 25, `${st.numbers} numbers`);
  assert.ok(st.widgets <= 2, `${st.widgets} widgets`);
  assert.ok(st.questions <= 3, `${st.questions} questions`);
  assert.equal(lesson.sections[1].kind, 'tldr', 'the TL;DR comes right after the hero');
  assert.ok(lesson.sections[1].items(D, T).length <= 4);
  assert.equal(lesson.footer, undefined, 'the standard one-line footer');
  assert.doesNotMatch(html, /recorded|recomputed|printed by|in your browser|show as table/i, 'provenance is stated once, on the home page');
});

// Strip everything that is not running prose (tags, bound numbers, code, formulas, headings' numerals); no digit may survive.
function proseDigits(h) {
  return h
    .replace(/<style[\s\S]*?<\/style>/g, '')
    .replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code>[\s\S]*?<\/code>/g, '')
    .replace(/<h[23][^>]*>\s*\d\s·/g, '<h2>')                                    // “1 · The gradient is a slope”
    .replace(/<[^>]+>/g, ' ')
    .match(/[^\s]*\d[^\s]*/g) ?? [];
}

test('no hand-typed digits survive in the prose, the layout text or the check/predict sections', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;                                     // formulas then render as <code>latex</code>, which the lint skips
  let leftover;
  try { leftover = renderLesson(lesson, D).flatMap(({ html: h }) => proseDigits(h)); } finally { globalThis.katex = saved; }
  assert.deepEqual(leftover, []);
  const layout = whyLayout(T);
  for (const field of ['goal', 'intro', 'forwardHint', 'backwardHint', 'takeaway']) {
    assert.deepEqual(findUnboundNums(layout[field]), [], field);
    for (const { path, shown } of boundNums(layout[field])) assert.equal(T.text(path), shown, `${field}: ${path}`);
    assert.deepEqual(proseDigits(layout[field]), [], field);
  }
  assert.deepEqual(new Set(layout.above), new Set(['x', 't']));
  assert.deepEqual(new Set(Object.keys(layout.pos)), new Set(scene.nodes.map((n) => n.name)), 'every node of the scene has a position');
});

test('the formulas in the prose (the bowl, the factor, the chain product) come from the scene and are true', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  assert.ok(raw.includes(`L(w) = (${m.x}w-${m.t})^2 = ${m.x * m.x}(w-${m.wMin})^2`), 'L(w) written out');
  assert.equal(m.x * m.x, m.curvature / 2);
  assert.ok(raw.includes(`f = 1-${m.curvature}\\,\\mathit{lr}`), 'the factor');
  for (const t of data.trajectories) assert.ok(tol(t.factor, m.factor(t.lr), 1e-12), `factor at ${t.lr}`);
  assert.deepEqual([2, 3, 4].map((i) => m.factor(data.trajectories[i].lr)), [0, -1, 1 - m.curvature * 0.3], 'exact, bounce, diverge');
  const [sp, se, sL] = m.localSlopes(m.w0).map((q) => q.slope);
  assert.deepEqual([sp, se, sL], [m.x, 1, 2 * m.at(m.w0).e]);
  assert.ok(raw.includes(`= ${sp}\\cdot ${se}\\cdot (${sL}) = ${m.at(m.w0).grad}`), 'the chain product with the recorded local slopes');
  assert.equal(sp * se * sL, data.nudge.local_slope_product);
  // the code snippet is faithful to the trace program: only w requires grad, and the API is w.grad()
  assert.match(html, /Variable w\(Tensor\(\{<span[^>]*>0<\/span>\.0f\}, \{1\}\), true\)/);
  assert.match(html, /Variable x\(Tensor\(\{<span[^>]*>2<\/span>\.0f\}, \{1\}\), false\)/);
  assert.ok([m.w0, m.x, m.t].every(Number.isInteger), 'integer leaves, so "N.0f" is right');
  assert.match(html, /w\.grad\(\) now holds the slope/);
});

test('predict and check sections exist, are answered from the trace, and have unique ids', () => {
  const kinds = lesson.sections.map((s) => s.kind);
  assert.equal(kinds.filter((k) => k === 'predict').length, 1);
  assert.equal(kinds.filter((k) => k === 'check').length, 2);
  const ids = lesson.sections.filter((s) => s.id).map((s) => s.id);
  assert.equal(new Set(ids).size, ids.length);
  // the bounce answer: lr 0.25 sits at 6, 0, 6 ... with the loss pinned at 36
  const bounce = data.trajectories[3];
  assert.deepEqual(bounce.states.slice(0, 5).map((s) => s.w), [0, 6, 0, 6, 0]);
  assert.ok(bounce.states.every((s) => s.L === 36));
  // the sign check quotes w = 6, slope 24
  assert.deepEqual([data.probes[5].w, data.probes[5].grad], [6, 24]);
});

// ---- (g) KaTeX parses all paper work --------------------------------------------------------------------------------------------------
test('(g) every LaTeX string the bowl widget produces parses in KaTeX (throwOnError)', () => {
  let count = 0;
  for (const lr of PRESET_LRS) {
    for (let w = -1; w <= 7.0001; w += 0.05) { renderStrictAll([updateTex(m, w, lr)]); count++; }
    const st = m.iterate(m.w0, lr, MAX_STEPS);
    renderStrictAll(st.map((q) => updateTex(m, q.w, lr)));
    count += st.length;
  }
  assert.ok(count > 600);
  assert.doesNotMatch(html, /katex-error/);
});

test('the layout of the graph lab and the lab itself agree with the scene', () => {
  const lay = whyLayout(T);
  assert.ok(lay.view[0] > 0 && lay.view[1] > 0 && lay.minWidth > 0);
  for (const [name, [x, y]] of Object.entries(lay.pos)) assert.ok(x > 0 && x < lay.view[0] && y > 0 && y < lay.view[1], `${name} inside the view`);
  assert.equal(scene.root, 'L');
  assert.equal(scene.nodes.find((n) => n.name === 'L').inputs.join(), 'e,e');
});

// ---- widgets import without a DOM ------------------------------------------------------------------------------------------------------
test('the bowl widget imports in Node without touching the DOM', async () => {
  assert.equal(typeof document, 'undefined');
  const a = await import('../js/widgets/bowl.js');
  assert.equal(typeof a.mountBowl, 'function');
  assert.equal(typeof document, 'undefined');
});

test('huge values from a diverging run are shown in exponent form, never as 8219000000000000000', () => {
  assert.equal(fmtX(8.219e18), '8.219e18');
  assert.equal(fmtX(-2.93e21), '-2.93e21');
  assert.equal(fmtX(1.5e7), '1.5e7');
  assert.equal(fmtX(-2.1e6), '-2.1e6', 'the readout switches to exponent form at 1e6, like the slope next to it');
  assert.equal(texX(-1.68e7), '-1.68\\times 10^{7}', 'inside KaTeX a bare e would read as a variable');
  assert.doesNotMatch(updateTex(m, -2.1e6, 0.3), /\de\d/);
  assert.equal(fmtX(7841.2), '7841');
  assert.equal(fmtX(-24), '-24');
  assert.equal(fmtX(0.01016), '0.01016');
  const st = m.iterate(m.w0, 0.35, 40);
  for (const s of st) for (const v of [s.w, s.L, s.grad]) assert.doesNotMatch(fmtX(v), /^-?\d{9,}$/, `${v}`);
  renderStrictAll([fmtX(1e19), `${fmtX(-1e19)}`, ...st.map((q) => updateTex(m, q.w, 0.35))]);
});
