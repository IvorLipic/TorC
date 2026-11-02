// Lesson generalization: every claim of the page against what TorC recorded (data/generalization.json), the widget helpers
// against the trace, the code on the page against the trace program (D11), and no hand-typed digits in the prose.
// The v3 rules (roles, budgets, glossary, notation) run on this lesson in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const G = readJson('data/generalization.json');
const anchors = anchorValues(readJson('data/anchors.json'));
const D = buildD({ generalization: G }, anchors);
const T = makeT(D);
const mod = await import('../js/lessons/generalization.js');
const lesson = mod.default;
const W = await import('../js/widgets/generalization-widgets.js');
const V = await import('../js/widgets/val-stop.js');
const course = readJson('course.json');
const plain = () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
};
const html = plain();
const shown = (path) => boundNums(html).filter((b) => b.path === path).map((b) => b.shown);
const runOf = (wd) => G.runs.find((r) => r.weight_decay === wd);
const last = (r) => r.checkpoints.at(-1);
const rm = Math.sqrt;
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

test('the data: ten evenly spaced training points, thirty validation and a hundred test points near sin(pi x), noise within ±0.3', () => {
  const d = G.data, st = G.settings;
  assert.equal(d.train_x.length, st.train_samples);
  assert.equal(d.val_x.length, st.val_samples);
  assert.equal(d.test_x.length, st.test_samples);
  assert.deepEqual([st.train_samples, st.val_samples, st.test_samples], [10, 30, 100], 'the words "ten", "thirty" and "a hundred" of the page');
  d.train_x.forEach((x, i) => assert.ok(Math.abs(x - (-1 + (2 * i) / (d.train_x.length - 1))) < 1e-6));
  const within = (xs, ts) => xs.forEach((x, i) => assert.ok(Math.abs(ts[i] - W.wave(x)) <= st.noise + 1e-5, `x = ${x}`));
  within(d.train_x, d.train_t);
  within(d.val_x, d.val_t);
  within(d.test_x, d.test_t);
  within(G.more_data.train_x, G.more_data.train_t);
  assert.equal(G.more_data.train_samples, 100, '"a hundred" training points');
  assert.deepEqual(G.more_data.train_x.slice(0, 10), d.train_x, 'the hundred points start with the ten');
});

test('the model: a polynomial of degree 15 with 15 weights and a bias, 16 parameters, more than the 10 training points', () => {
  assert.equal(G.settings.degree, 15);
  assert.equal(G.settings.parameters, G.settings.degree + 1);
  assert.ok(G.settings.parameters > G.settings.train_samples);
  assert.equal(G.settings.model, `nn::Linear(${G.settings.degree}, 1)`);
  for (const r of [...G.runs, G.more_data]) {
    const c = r.checkpoints[0];
    assert.equal(c.weight_sum_sq, 0, 'every parameter starts at zero');
    c.curve.forEach((y) => assert.equal(y, 0));
  }
});

test('the typical misses of derived.* are the square roots of the losses at the validation stop and at the last step', () => {
  const runs = [...G.runs.map((r, i) => [r, G.derived.misses[i]]), [G.more_data, G.derived.more_data_misses]];
  for (const [r, m] of runs) {
    const s = r.checkpoints.find((c) => c.step === r.val_stop.step), l = last(r);
    assert.equal(m.stop.step, r.val_stop.step);
    for (const [k, key] of [['train', 'train_loss'], ['val', 'val_loss'], ['test', 'test_loss']]) {
      assert.ok(near(m.stop[k], rm(s[key])), `stop ${k}`);
      assert.ok(near(m.last[k], rm(l[key])), `last ${k}`);
    }
    assert.ok(near(m.stop.ratio, m.stop.val / m.stop.train));
    assert.ok(near(m.last.ratio, m.last.val / m.last.train));
  }
  // the wave's own miss is the mean squared miss of sin(pi x) on each set
  const wl = (xs, ts) => xs.reduce((a, x, i) => a + (W.wave(x) - ts[i]) ** 2, 0) / xs.length;
  assert.ok(near(G.derived.true_wave_loss.val, wl(G.data.val_x, G.data.val_t), 1e-6));
  assert.ok(near(G.derived.true_wave_loss.test, wl(G.data.test_x, G.data.test_t), 1e-6));
});

test('the predict and the game: without weight decay the training miss falls almost to zero, the validation miss falls with a bump, then rises', () => {
  const r = runOf(0), cs = r.checkpoints, b = W.lowestVal(cs);
  assert.equal(cs[b].step, r.val_stop.step, 'lowestVal() finds the validation stop of the trace');
  assert.ok(b > 0 && b < cs.length - 1, 'the lowest validation miss lies between the first and the last step');
  assert.ok(rm(last(r).train_loss) < 0.01, '"almost zero"');
  assert.ok(last(r).val_loss > 2 * cs[b].val_loss, 'the validation miss rises after the turn');
  // "with one bump": a local rise before the lowest point
  const v = cs.slice(0, b + 1).map((c) => c.val_loss);
  const rises = v.slice(1).filter((x, i) => x > v[i]).length;
  assert.ok(rises >= 1, 'a bump before the lowest point');
  // the test curve turns at the same step (shown only after the commit, to compare)
  assert.equal(r.best_test.step, r.val_stop.step);
  // "It is close to the miss that the validation set promised" at the validation stop
  assert.ok(Math.abs(rm(r.val_stop.test_loss) - rm(r.val_stop.val_loss)) < 0.05);
  // "near the right end, it swings far above the wave"
  const g = G.data.grid, cv = last(r).curve;
  const right = g.map((x, i) => [x, cv[i] - W.wave(x)]).filter(([x]) => x > 0.75);
  assert.ok(Math.max(...right.map(([, d]) => d)) > 0.8, 'the curve rises far above the wave near x = 0.9');
});

test('overfitting has degrees: the ratio of the two misses is about 3 at the stop and about 50 at the last step', () => {
  const m = G.derived.misses[0];
  assert.deepEqual(shown(`files.generalization.derived.misses.0.last.ratio`), ['50']);
  assert.deepEqual(shown(`files.generalization.derived.misses.0.stop.ratio`), ['3.1']);
  assert.ok(m.last.ratio > 10 * m.stop.ratio);
  // the learner divides the two rounded misses of the prompt: the answer must fall within the tolerance of the trace value
  const q = lesson.sections.find((s) => s.id === 'ratio');
  const rounded = Number(m.stop.val.toPrecision(2)) / Number(m.stop.train.toPrecision(2));
  assert.ok(Math.abs(rounded - m.stop.ratio) < q.tolerance, `${rounded} vs ${m.stop.ratio}`);
  // the worked block: the rounded inputs give the rounded result
  assert.equal(Number((Number(m.last.val.toPrecision(2)) / Number(m.last.train.toPrecision(2))).toPrecision(1)), Number(m.last.ratio.toPrecision(1)));
});

test('weight decay 0.1 keeps the weights small and the validation miss from climbing; weight decay 1 underfits with a gap', () => {
  const none = runOf(0), mid = runOf(0.1), high = runOf(1);
  assert.ok(last(mid).weight_sum_sq < last(none).weight_sum_sq / 10, 'small weights');
  const vm = mid.checkpoints.map((c) => c.val_loss), lo = Math.min(...vm);
  assert.ok(last(mid).val_loss < 1.05 * lo, '"the validation miss does not climb"');
  assert.ok(rm(last(high).train_loss) > 2 * rm(last(mid).train_loss), '"misses even its training points by a lot"');
  assert.ok(last(high).val_loss > last(high).train_loss, 'the underfit model also has a gap');
  assert.ok(rm(last(high).val_loss) > rm(last(mid).val_loss), '"both stay high"');
  assert.deepEqual(G.derived.decay_factor.map((f, i) => near(f, 1 - G.settings.lr * G.settings.weight_decays[i])), [true, true, true]);
  // the reveal of the cumulative quiz: 0.95 and 0.995
  assert.deepEqual(shown('files.generalization.derived.decay_factor.2'), ['0.95']);
  // section 4: "the size of the weights ends near 441, against 13 with weight decay"
  assert.deepEqual(shown('files.generalization.runs.0.checkpoints.26.weight_sum_sq'), ['440']);
  assert.deepEqual(shown('files.generalization.runs.1.checkpoints.26.weight_sum_sq'), ['13']);
  // "It still looks like the wave but sags": with weight decay 1 the curve has the sign of the wave at its peaks, but is smaller
  const g = G.data.grid, hc = last(high).curve, at = (x) => hc[g.findIndex((v) => Math.abs(v - x) < 1e-6)];
  assert.ok(at(0.5) > 0 && at(0.5) < 0.8 * W.wave(0.5) && at(-0.5) < 0 && at(-0.5) > 0.8 * W.wave(-0.5), 'it sags at the peaks of the wave');
  // the leak quiz reveal: 0.05 × 1 and 0.95
  assert.deepEqual(shown('files.generalization.settings.lr'), ['0.05', '0.05']);
});

test('more data: the same 16 parameters on a hundred points end with a small gap, near the wave itself and not below it', () => {
  const m = G.derived.more_data_misses, n = G.derived.misses[0];
  assert.ok(m.last.val < 0.5 * n.last.val, '"much smaller with more points"');
  assert.ok(m.last.ratio < 1.5, 'a small gap');
  const waveVal = rm(G.derived.true_wave_loss.val);
  assert.ok(m.last.val > waveVal && m.last.val < 1.2 * waveVal, 'close to the wave, not below it');
  for (const r of [...G.runs, G.more_data]) for (const c of r.checkpoints) assert.ok(c.val_loss > 0.9 * G.derived.true_wave_loss.val, 'no run beats the wave by much');
});

test('C2: the test set is never used to choose, and the page says so; no test number in the prose before the game result', () => {
  assert.match(html, /You look at them once, at the end, to measure the final model/);
  assert.deepEqual(shown('files.generalization.runs.0.val_stop.test_loss'), [], 'the test miss at the stop appears only in the game, after the commit');
  assert.match(html, /They chose with the test set, so its miss is too hopeful/);
  // the game shows no test value before the commit; the remedies widget never shows the test set
  const vs = readFileSync(new URL('../js/widgets/val-stop.js', import.meta.url), 'utf8');
  assert.match(vs, /if \(stopped\) points\.unshift/);
  assert.match(vs, /if \(stopped\) lines\.push\(\{ cs, key: 'test_loss'/);
  const rw = readFileSync(new URL('../js/widgets/generalization-widgets.js', import.meta.url), 'utf8');
  assert.doesNotMatch(rw.slice(rw.indexOf('export function mountRemedies')), /test_loss|test_x/);
});

test('the result text of the game: the chosen step, its test miss, and where the lowest validation miss is', () => {
  const r = runOf(0), b = W.lowestVal(r.checkpoints);
  const atBest = V.resultText(G, b);
  assert.match(atBest[0], new RegExp(`after step ${r.val_stop.step}\\. .* about ${W.fmt2(rm(r.val_stop.test_loss))}\\.`));
  assert.match(atBest[2], /lowest validation miss on the slider/);
  const atEnd = V.resultText(G, r.checkpoints.length - 1);
  assert.match(atEnd[0], /after step 20,000/);
  assert.match(atEnd[2], new RegExp(`after step ${r.val_stop.step},`));
  // review H2: at the stop the test miss is smaller than the validation miss, because this validation set is harder by chance
  // (the wave itself misses it by more); the text says so with the wave's own misses
  assert.ok(r.val_stop.test_loss < r.val_stop.val_loss);
  assert.ok(G.derived.true_wave_loss.val > G.derived.true_wave_loss.test);
  assert.match(atBest[3], /^Here the test miss is even a little smaller\. Small sets differ by chance\./);
  assert.ok(atBest[3].includes(`about ${W.fmt2(rm(G.derived.true_wave_loss.val))} and a test point by about ${W.fmt2(rm(G.derived.true_wave_loss.test))}`));
  assert.match(atBest[4], /Each further choice with the same points makes it more hopeful/);
  // from step 20 on, the test curve lies below the validation curve (what the reader sees after the commit)
  for (const c of r.checkpoints.filter((q) => q.step >= 20)) assert.ok(c.test_loss < c.val_loss, `step ${c.step}`);
});

test('the widget helpers: log step axis, typical miss, 2 significant figures, labels kept apart, four runs; no DOM in Node', () => {
  assert.equal(W.stepX(0), 0);
  assert.equal(W.stepX(9), 1);
  assert.equal(W.miss(0.04), 0.2);
  assert.equal(W.fmt2(0.0869), '0.087');
  assert.equal(W.fmt2(0.46127), '0.46');
  assert.equal(W.fmt2(0.2038), '0.2');
  assert.equal(W.fmt2(0.2038), T.human('files.generalization.more_data.checkpoints.26.val_loss', 'rmse').match(/>([^<]*)</)[1], 'the readout and the prose agree');
  assert.equal(W.fmt2(50.25), '50');
  assert.equal(W.fmt2(441.2), '440', 'the weight readout agrees with the prose');
  assert.equal(W.fmtInt(20000), '20,000');
  const ys = W.spreadLabels([100, 104, 300], 10, 200, 14);
  assert.ok(ys[1] - ys[0] >= 14 && ys[2] <= 200 && ys[0] >= 10);
  assert.deepEqual(W.cases(G).map((c) => c.label), ['no weight decay', 'weight decay 0.1', 'weight decay 1', '100 training points']);
  assert.equal(typeof W.mountRemedies, 'function');
  assert.equal(typeof V.mountValStop, 'function');
  assert.equal(typeof document, 'undefined');
});

test('the code on the page is the code of the experiment (brief C10, D11) and the home page snippet', () => {
  const block = html.match(/<pre class="gn-code"[^>]*><code>([\s\S]*?)<\/code><\/pre>/)[1].replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const src = readFileSync(new URL('../../tools/trace/trace_generalization.cpp', import.meta.url), 'utf8');
  const st = G.settings;
  assert.ok(src.includes('data::DataLoader train(train_set, batch, /*shuffle=*/false);') && src.includes(`BATCH = ${st.batch_size};`));
  assert.ok(src.includes(`val(val_set, ${st.batch_size})`) && src.includes(`test(test_set, ${mod.TEST_BATCH})`), 'the evaluation loaders of the trace program');
  assert.ok(src.includes('Run out = train_run(train_set, BATCH, wd);'));
  assert.ok(block.includes(`data::DataLoader train(train_set, ${st.batch_size}, /*shuffle=*/false);`));
  assert.ok(block.includes(`data::DataLoader val(val_set, ${st.batch_size}), test(test_set, ${mod.TEST_BATCH});`));
  assert.ok(block.includes('optim::AdamW opt(params, 0.05f, /*weight_decay=*/0.1f);'));
  assert.ok(block.includes('// 10 training samples') && block.includes('// 30 validation and 100 test samples'));
  assert.ok(block.includes('weight decay of 0.0f and 1.0f'));
  assert.deepEqual(st.weight_decays, G.runs.map((r) => r.weight_decay));
  assert.equal(st.shuffle, false);
  assert.ok(st.train_samples <= st.batch_size && st.val_samples <= st.batch_size && st.test_samples <= mod.TEST_BATCH, 'each loader gives its set in one batch');
  const home = course.parts.flatMap((p) => p.lessons).find((l) => l.id === 'generalization').code;
  assert.equal(home, block, 'the home page snippet is the code of the lesson');
});

test('every number in the prose is bound, with no hand-typed digits outside code and formulas', () => {
  const prose = html.replace(/<style[\s\S]*?<\/style>/g, ' ').replace(/<svg[\s\S]*?<\/svg>/g, ' ').replace(/<pre[\s\S]*?<\/pre>/g, ' ')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, ' ').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, ' ').replace(/<h2[^>]*>\s*\d+ ·/g, '<h2>')
    .replace(/<[^>]+>/g, ' ');
  assert.deepEqual(prose.match(/\S*\d\S*/g), null);
  assert.deepEqual(shown('files.generalization.settings.parameters'), ['16', '16']);
  assert.deepEqual(shown('anchors.mlp_parameters'), ['26,506', '26,506']);
});
