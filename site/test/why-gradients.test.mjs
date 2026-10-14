// Lesson why-gradients: the JS model behind the widgets against the numbers TorC recorded (data/why-gradients.json),
// the stability predicate, every number in the prose, all LaTeX, and that the widget modules import without a DOM.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, runBackward } from '../js/engine.js';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { makeModel, sameNumber, recordedPoint, recordedTrajectory, recordedSecant } from '../js/widgets/landscape-model.js';
import { fmtX } from '../js/widgets/landscape-tex.js';
import { knobTex, secantTex, updateTex, lastStepTex, factorTex, gearsTex } from '../js/widgets/landscape-tex.js';
import { H_STOPS, PRESET_LRS, JUMPS, MAX_STEPS } from '../js/widgets/landscape1d.js';
import { whyLayout } from '../js/lessons/why-gradients.scenes.js';
import { DERIVED } from '../js/lessons/why-gradients.js';
import { readJson, renderStrictAll, katex } from './helpers.mjs';

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
  assert.deepEqual(data.trajectories.map((t) => t.lr), PRESET_LRS);
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

test('the knob jump buttons, the probes the lesson quotes and the recorded points exist', () => {
  for (const w of JUMPS) assert.ok(recordedPoint(data, w), `probe w=${w}`);
  assert.equal(recordedPoint(data, 0.1).w, data.probes[1].w);        // 0.1 from the slider matches the float32 0.100000001
  assert.equal(recordedPoint(data, 0.3), null);
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

// ---- (d) secants ----------------------------------------------------------------------------------------------------------------
test('(d) secant slopes are exactly -24 + 4h, shrink monotonically toward the gradient, and match the float32 recording', () => {
  assert.equal(m.at(0).grad, -24);
  let prevGap = Infinity;
  for (const h of [...H_STOPS].sort((a, b) => b - a)) {
    const s = m.secant(0, h);
    assert.ok(tol(s.slope, 4 * h - 24, 1e-9), `h=${h}`);
    assert.ok(s.gap < prevGap && s.gap > 0, `gap shrinks at h=${h}`);
    prevGap = s.gap;
    const rec = recordedSecant(data, h);
    if (rec) {
      assert.ok(Math.abs(rec.slope - s.slope) <= 1e-2, `h=${h}: browser ${s.slope} vs TorC float32 ${rec.slope}`);   // float32 cancellation at h = 0.001
      assert.equal(rec.backward_slope, m.at(0).grad);
      assert.ok(tol(rec.expected_slope, 4 * h - 24, 1e-12));
      assert.ok(tol(s.L0, rec.L0) && tol(s.Lh, rec.L_h));
    }
  }
  assert.equal(data.secants.filter((r) => recordedSecant(data, r.h)).length, 5, 'five recorded secants');
  const rec = data.secants.map((r) => Math.abs(r.slope - r.backward_slope));
  assert.ok(rec.every((v, i) => i === 0 || v < rec[i - 1]), `recorded |secant - slope| decreases: ${rec}`);
  assert.deepEqual(data.secants.map((r) => r.h), [...data.secants.map((r) => r.h)].sort((a, b) => b - a));
});

// ---- (e) stability predicate -----------------------------------------------------------------------------------------------------
test('(e) |1 - 8 lr| < 1 predicts divergence: matches stable_predicted and the recorded loss of all five runs', () => {
  for (const rec of data.trajectories) {
    const L0 = rec.states[0].L, L8 = rec.states[8].L;
    assert.equal(m.stable(rec.lr), rec.stable_predicted, `lr=${rec.lr}`);
    assert.equal(m.stable(rec.lr), L8 < L0 - 1e-9, `observed: L8=${L8} vs L0=${L0} at lr=${rec.lr}`);   // lr 0.25 stays at 36: not stable
    const v = m.verdict(rec.lr);
    assert.equal(v.id === 'diverge', L8 > L0 + 1e-9, `verdict ${v.id}`);
  }
  assert.deepEqual(PRESET_LRS.map((lr) => m.verdict(lr).id), ['smooth', 'smooth', 'exact', 'bounce', 'diverge']);
  assert.equal(m.verdict(0.2).id, 'zigzag');
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

test('every number in the lesson prose is bound to the trace, and the bound text is what T() gives', () => {
  const parts = renderLesson(lesson, D);
  assert.ok(parts.length > 15);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const all = parts.map((p) => p.html).join('\n');
  assert.ok(boundNums(all).length > 60, 'the lesson quotes many trace numbers');
  for (const quoted of ['why.L.value', 'why.w.grad', 'why.e.grad', `files.why-gradients.trajectories.0.states.1.L`, `files.why-gradients.trajectories.0.states.0.tangent_L_next`, `files.why-gradients.nudge.stages.2.change`]) {
    assert.ok(all.includes(`data-trace="${quoted}"`) || quoted.endsWith('change'), `lesson quotes ${quoted}`);
  }
  // the six numbers the content plan names: 36, -24, 12.96, 7.2, -12, -0.2396
  assert.deepEqual([T.text('why.L.value'), T.text('why.w.grad'), T.text(`files.why-gradients.trajectories.0.states.1.L`), T.text(`files.why-gradients.trajectories.0.states.0.tangent_L_next`), T.text('why.e.grad'), T.text('files.why-gradients.nudge.actual_dL')],
    ['36', '-24', '12.96', '7.2', '-12', '-0.2396']);
});

// Strip everything that is not running prose (tags, bound numbers, code, formulas, headings' numerals); no digit may survive.
function proseDigits(html) {
  return html
    .replace(/<style[\s\S]*?<\/style>/g, '')
    .replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<span class="dc"[^>]*>[^<]*<\/span>/g, '')                          // derived constants, checked in the next test
    .replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code>[\s\S]*?<\/code>/g, '')
    .replace(/<h[23][^>]*>\s*\d\s·/g, '<h2>')                                    // “1 · A knob and a score”
    .replace(/<[^>]+>/g, ' ')
    .match(/[^\s]*\d[^\s]*/g) ?? [];
}

test('no hand-typed digits survive in the prose, the layout text or the check/predict sections', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;                                     // formulas then render as <code>latex</code>, which the lint skips
  let leftover;
  try { leftover = renderLesson(lesson, D).flatMap(({ html }) => proseDigits(html)); } finally { globalThis.katex = saved; }
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

test('the derived constants the prose prints as formulas (8 and 3) are the scene’s own, and the table cells match the trace', () => {
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  const dcs = [...html.matchAll(/<span class="dc" data-derived="(\w+)">([^<]*)<\/span>/g)];
  assert.ok(dcs.length >= 2, 'derived constants appear in the step-size explanation');
  for (const [, what, shown] of dcs) assert.equal(what === 'curvature' ? shown : null, String(m.curvature), what);
  assert.equal(m.curvature, 2 * m.x * m.x);
  assert.equal(m.wMin, m.t / m.x);
  assert.equal(m.curvature, data.trajectories.length ? Math.round((1 - data.trajectories[0].factor) / data.trajectories[0].lr * 1e9) / 1e9 : NaN, 'curvature agrees with the recorded factor 1 - c*lr');
  // the learning-rate table lists every recorded run, in order
  const rows = [...html.matchAll(/<tr><td data-label="learning rate"><span class="num" data-trace="files\.why-gradients\.trajectories\.(\d)\.lr">/g)].map((x) => Number(x[1]));
  assert.deepEqual(rows, [0, 1, 2, 3, 4]);
});

test('numbers that are arithmetic on printed ones are marked as derived (dotted underline), printed ones are not', () => {
  const all = renderLesson(lesson, D).map((p) => p.html).join('\n');
  let derived = 0, printed = 0;
  for (const mt of all.matchAll(/(<span class="dv"[^>]*>)?<span class="num" data-trace="([^"]*)">/g)) {
    const isDerived = DERIVED.some((k) => mt[2].includes(k));
    assert.equal(Boolean(mt[1]), isDerived, `${mt[2]} ${isDerived ? 'must be marked derived' : 'is printed by TorC and must not be marked'}`);
    if (isDerived) derived++; else printed++;
  }
  assert.ok(derived >= 10 && printed >= 60, `${derived} derived, ${printed} printed`);
  // the schema says which fields are derived: factor, critical_lr, tangent_L_next, predicted_dL, second_order_term, local_slope_product, w_min
  for (const t of data.trajectories) assert.ok(tol(t.factor, 1 - 8 * t.lr, 1e-12));
  assert.equal(data.setup.critical_lr, 2 / m.curvature);
  assert.ok(tol(data.nudge.predicted_dL, data.nudge.local_slope_product * data.nudge.delta, 1e-6));
  // footer and hero state the provenance honestly
  assert.match(lesson.footer(D, T), /derived from those numbers/);
  assert.match(all, /dotted underline/);
  assert.doesNotMatch(lesson.footer(D, T), /Every number/);
});

test('the formulas the prose derives (L(w) = 4(w-3)^2, slope 8(w-3), local slopes) come from the scene and are true', () => {
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.ok(html.includes(`L(w) = (${m.x}w-${m.t})^2 = ${m.x * m.x}(w-${m.wMin})^2`), 'L(w) written out');
  assert.ok(html.includes(`2\\cdot ${m.curvature / 2}(w-${m.wMin})`), 'power rule line');
  assert.equal(m.x * m.x, m.curvature / 2);
  const [sp, se, sL] = m.localSlopes(m.w0).map((q) => q.slope);
  assert.deepEqual([sp, se, sL], [m.x, 1, 2 * m.at(m.w0).e]);
  for (const [i, v] of [sp, se, sL].entries()) assert.ok(html.includes(`data-trace="files.why-gradients.nudge.stages.${i}.local_slope">${v}<`), `local slope ${i} quoted`);
  // the code snippet is faithful to the trace program: only w requires grad, and the API is w.grad()
  assert.match(html, /Variable w\(Tensor\(\{<span[^>]*>0<\/span>\.0f\}, \{1\}\), true\)/);
  assert.match(html, /Variable x\(Tensor\(\{<span[^>]*>2<\/span>\.0f\}, \{1\}\), false\)/);
  assert.ok([m.w0, m.x, m.t].every(Number.isInteger), 'integer leaves, so "N.0f" is right');
  assert.match(html, /w\.grad\(\) holds the slope/);
});

test('predict and check sections exist, are answered from the trace, and have unique ids', () => {
  const kinds = lesson.sections.map((s) => s.kind);
  assert.ok(kinds.includes('predict'));
  assert.ok(kinds.filter((k) => k === 'check').length >= 3);
  const ids = lesson.sections.filter((s) => s.id).map((s) => s.id);
  assert.equal(new Set(ids).size, ids.length);
  const direction = lesson.sections.find((s) => s.id === 'direction');
  assert.match(direction.prompt(D, T), /left/);
  assert.match(direction.reveal(D, T), /Right/);
  // the bounce answer: lr 0.25 sits at 6, 0, 6 ... with the loss pinned at 36
  const bounce = data.trajectories[3];
  assert.deepEqual(bounce.states.slice(0, 5).map((s) => s.w), [0, 6, 0, 6, 0]);
  assert.ok(bounce.states.every((s) => s.L === 36));
});

// ---- (g) KaTeX parses all paper work --------------------------------------------------------------------------------------------------
test('(g) every paper-work string the widgets produce parses in KaTeX (throwOnError)', () => {
  let count = 0;
  const check = (list) => { renderStrictAll(list); count += list.length; };
  for (let w = -1; w <= 7.0001; w += 0.05) {
    check(knobTex(m, w));
    for (const h of H_STOPS) check(secantTex(m, w, h));
    check([updateTex(m, w, 0.05), updateTex(m, w, 0.3)]);
  }
  for (let lr = 0.01; lr <= 0.3501; lr += 0.005) {
    check(factorTex(m, lr));
    const st = m.iterate(m.w0, lr, MAX_STEPS);
    for (const s of st.slice(0, 5)) check(lastStepTex({ ...s, wNext: s.wNext, actual: m.lossAt(s.wNext) }));
  }
  for (let d = -0.5; d <= 0.5001; d += 0.01) check(gearsTex(m, m.nudge(m.w0, d)));
  assert.ok(count > 1000);
  // and the display-mode formulas the lesson text builds
  for (const { html } of renderLesson(lesson, D)) assert.doesNotMatch(html, /katex-error/);
  assert.ok(katex.renderToString(factorTex(m, 0.05)[0], { throwOnError: true }).length > 20);
});

test('the layout of the graph lab and the lab itself agree with the scene', () => {
  const lay = whyLayout(T);
  assert.ok(lay.view[0] > 0 && lay.view[1] > 0 && lay.minWidth > 0);
  for (const [name, [x, y]] of Object.entries(lay.pos)) assert.ok(x > 0 && x < lay.view[0] && y > 0 && y < lay.view[1], `${name} inside the view`);
  assert.equal(scene.root, 'L');
  assert.equal(scene.nodes.find((n) => n.name === 'L').inputs.join(), 'e,e');
});

// ---- widgets import without a DOM ------------------------------------------------------------------------------------------------------
test('the widget modules import in Node without touching the DOM and export their mount functions', async () => {
  assert.equal(typeof document, 'undefined');
  const a = await import('../js/widgets/landscape1d.js');
  const b = await import('../js/widgets/why-gears.js');
  for (const f of ['mountKnob', 'mountSlope', 'mountStepper', 'mountStepSize']) assert.equal(typeof a[f], 'function', f);
  assert.equal(typeof b.mountGears, 'function');
  a.injectStyles();                                                    // a no-op without a document
  assert.equal(typeof document, 'undefined');
});

test('sameNumber is the mixed tolerance of docs/trace-schema.md', () => {
  assert.ok(sameNumber(33.6400032, 33.64) && !sameNumber(33.7, 33.64));
});

test('huge values from a diverging run are shown in exponent form, never as 8219000000000000000', () => {
  assert.equal(fmtX(8.219e18), '8.219e18');
  assert.equal(fmtX(-2.93e21), '-2.93e21');
  assert.equal(fmtX(1.5e7), '1.5e7');
  assert.equal(fmtX(7841.2), '7841');
  assert.equal(fmtX(-24), '-24');
  assert.equal(fmtX(0.01016), '0.01016');
  const st = m.iterate(m.w0, 0.35, 40);
  for (const s of st) for (const v of [s.w, s.L, s.grad]) assert.doesNotMatch(fmtX(v), /^-?\d{9,}$/, `${v}`);
  renderStrictAll([fmtX(1e19), `${fmtX(-1e19)}`, ...st.slice(0, 41).map((s) => lastStepTex({ ...s, actual: m.lossAt(s.wNext) })).flat()]);
});
