// Lesson what-is-learning: the claims of the page against what TorC computed (data/whole-game.json, data/mnist-mlp.json), the JS
// that scores the learner's line (widgets/fit-game.js) against TorC's MSELoss, and no hand-typed digits in the prose.
// The v3 rules (roles, budgets, glossary, notation) run on this lesson in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, boundNums } from '../js/core/sections.js';
import { readJson, rng } from './helpers.mjs';

const G = readJson('data/whole-game.json');
const M = readJson('data/mnist-mlp.json');
const anchors = anchorValues(readJson('data/anchors.json'));
const D = buildD({ 'whole-game': G, 'mnist-mlp': M }, anchors);
const T = makeT(D);
const mod = await import('../js/lessons/what-is-learning.js');
const lesson = mod.default;
const { mse, typicalMiss, lineThrough, fmt2, equation, HANDLE_X, START } = await import('../js/widgets/fit-game.js');
const { argmax } = await import('../js/widgets/digit-widgets.js');

const tx = G.data.train.x, ty = G.data.train.y, hx = G.data.hidden.x, hy = G.data.hidden.y;
const run = G.runs[0], fin = run.final;
const near = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
const plain = () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: readJson('course.json') }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
};
const html = plain();
const shown = (path) => boundNums(html).filter((b) => b.path === path).map((b) => b.shown);

test('the data: fifteen made-up students, ten training points and five hidden points, scores out of ten', () => {
  assert.equal(G.data.n_train, 10);
  assert.equal(G.data.n_hidden, 5);
  assert.equal(G.data.points.length, 15);
  assert.equal(tx.length, 10);
  assert.equal(hx.length, 5);
  for (const y of [...ty, ...hy]) assert.ok(y >= 0 && y <= 10, `score ${y} is out of ten`);
  for (const x of [...tx, ...hx]) assert.ok(x >= 0.5 && x <= 4, `hours ${x} fit the plot`);
  assert.ok(G.data.story.includes('hours of practice'), 'the story of the page is the story of the data');
});

test('mse() scores a line exactly as TorC MSELoss does, on the training and the hidden points (every probe)', () => {
  assert.ok(G.probes.length >= 5);
  for (const p of G.probes) {
    assert.ok(near(mse(p.w, p.b, tx, ty), p.train_loss), `${p.id} training: ${mse(p.w, p.b, tx, ty)} vs ${p.train_loss}`);
    assert.ok(near(mse(p.w, p.b, hx, hy), p.hidden_loss), `${p.id} hidden: ${mse(p.w, p.b, hx, hy)} vs ${p.hidden_loss}`);
    p.train_residual.forEach((r, i) => {
      assert.ok(near(p.w * tx[i] + p.b - ty[i], r), `${p.id} residual ${i}`);
      assert.ok(near(r * r, p.train_square[i]), `${p.id} square ${i}`);
    });
    const meanSq = p.train_square.reduce((a, b) => a + b, 0) / p.train_square.length;
    assert.ok(near(meanSq, p.train_loss), `${p.id}: the loss is the mean of the squares`);
  }
});

test('mse() agrees with TorC at every step of the run, and one JS gradient step gives TorC\'s next line (SGD on MSELoss)', () => {
  for (const s of run.steps) {
    assert.ok(near(mse(s.w, s.b, tx, ty), s.train_loss, 2e-5), `step ${s.step} training`);
    assert.ok(near(mse(s.w, s.b, hx, hy), s.hidden_loss, 2e-5), `step ${s.step} hidden`);
  }
  for (const k of [0, 1, 2, 10, 100]) {
    const { w, b } = run.steps[k], n = tx.length;
    let gw = 0, gb = 0;
    for (let i = 0; i < n; i++) { const r = w * tx[i] + b - ty[i]; gw += (2 / n) * r * tx[i]; gb += (2 / n) * r; }
    assert.ok(near(w - run.lr * gw, run.steps[k + 1].w, 1e-4) && near(b - run.lr * gb, run.steps[k + 1].b, 1e-4), `step ${k + 1}`);
  }
});

test('the game starts with the flat line of the probe flat-5, and the handles define the line through them', () => {
  const flat = G.probes.find((p) => p.id === 'flat-5');
  assert.deepEqual([START.w, START.b], [flat.w, flat.b]);
  const l = lineThrough(HANDLE_X[0], 2, HANDLE_X[1], 8);
  assert.ok(near(l.w * HANDLE_X[0] + l.b, 2) && near(l.w * HANDLE_X[1] + l.b, 8));
  assert.equal(fmt2(typicalMiss(flat.train_loss)), '1.9');
  assert.equal(fmt2(0.39968), '0.4');
  assert.equal(fmt2(-0.4999), '−0.5');
  assert.equal(equation(fin.w, fin.b), 'score = 1.6 × hours + 1.9');
  assert.equal(equation(2, -0.19), 'score = 2 × hours − 0.19');
  assert.equal(equation(0, 0), 'score = 0 × hours + 0');
});

test('predict: no line passes through all ten points, and no line beats the computer\'s line on them', () => {
  assert.equal(run.id, 'from-zero');
  assert.equal(run.settled, true, 'the run stopped because the line stopped moving');
  assert.ok(fin.train_loss > 0.1, 'the best line still misses');
  assert.ok(near(fin.train_loss, G.derived.least_squares.train_loss, 1e-4), 'the computer reached the smallest possible loss');
  const r = rng(7);
  for (let i = 0; i < 2000; i++) {
    const w = -1 + 4 * r(), b = -2 + 10 * r();
    assert.ok(mse(w, b, tx, ty) >= fin.train_loss - 1e-6, `w ${w}, b ${b} beat the computer`);
  }
  for (const p of G.probes) assert.ok(p.train_loss >= fin.train_loss - 1e-6, `${p.id}`);
  // close to the rule that made the data, but not equal to it
  assert.ok(Math.abs(fin.w - G.data.true_line.w) < 0.2 && Math.abs(fin.b - G.data.true_line.b) < 0.2);
  assert.ok(Math.abs(fin.w - G.data.true_line.w) > 0.01 || Math.abs(fin.b - G.data.true_line.b) > 0.01);
  assert.deepEqual(shown('files.whole-game.runs.0.final.train_loss'), ['0.4', '0.4'], 'typical miss on the training points');
});

test('the worked example and "your turn" use the line w = 1, b = 3, and the answer cannot be the miss itself', () => {
  assert.equal(mod.LINE, 'files.whole-game.probes.3');
  const p = G.probes[3];
  assert.equal(p.id, 'w1-b3');
  for (const i of [mod.ONE, mod.TWO]) assert.ok(near(p.train_pred[i], p.w * tx[i] + p.b));
  assert.deepEqual(shown(`files.whole-game.probes.3.train_square.${mod.ONE}`), ['1.21']);
  assert.deepEqual(shown(`files.whole-game.probes.3.train_residual.${mod.ONE}`), ['−1.1']);
  assert.deepEqual(shown(`files.whole-game.probes.3.train_square.${mod.TWO}`), ['0.25']);
  assert.deepEqual(shown('files.whole-game.probes.3.train_loss'), ['0.624', '0.79']);
  const q = lesson.sections.find((s) => s.id === 'your-square');
  const ans = T.num(q.answer), miss = Math.abs(p.train_residual[mod.TWO]);
  assert.ok(Math.abs(ans - 0.25) < 1e-5);
  assert.ok(Math.abs(miss - ans) > q.tolerance, 'typing the miss instead of its square is marked wrong');
});

test('the game tasks: the computer starts flat at zero, moves a lot at first and very little later', () => {
  assert.deepEqual([run.start.w, run.start.b, run.steps[0].w, run.steps[0].b], [0, 0, 0, 0]);
  const d = (a, b) => Math.hypot(a.w - b.w, a.b - b.b);
  assert.ok(d(run.steps[0], run.steps[1]) > 2, 'the first step moves the line a lot');
  const n = run.steps.length - 1;
  assert.ok(d(run.steps[n - 50], run.steps[n]) < 0.01, 'the last fifty steps move it very little');
  assert.ok(run.steps[1].train_loss < run.steps[0].train_loss / 10, 'the loss falls fast at first');
});

test('new data: the computer\'s line misses the hidden points by more than the training points', () => {
  assert.ok(fin.hidden_loss > fin.train_loss);
  assert.ok(typicalMiss(fin.hidden_loss) > typicalMiss(fin.train_loss) * 1.5);
  assert.deepEqual(shown('files.whole-game.runs.0.final.hidden_loss'), ['0.83']);
  // the sentence names the hidden miss first, then the training miss
  assert.match(html, /hidden points by more than the training points, by about <span[^>]*runs\.0\.final\.hidden_loss[^>]*>0\.83<\/span> against <span[^>]*runs\.0\.final\.train_loss[^>]*>0\.4<\/span>/);
});

test('the digit pad: the first test images are read correctly, the last four buttons are misread', () => {
  const s = M.shown, last4 = s.slice(-4), rest = s.slice(0, -4);
  assert.ok(last4.every((x) => argmax(x.logits) !== x.label), 'the last four are mistakes');
  assert.ok(rest.every((x) => argmax(x.logits) === x.label), 'the others are read correctly');
  assert.equal(anchors.train_images, 60000);
  assert.deepEqual(shown('anchors.train_images'), ['60,000']);
  assert.deepEqual(shown('anchors.mlp_parameters'), ['26,506']);
});

test('the reveal states the computer\'s line and the rule that made the data', () => {
  assert.deepEqual(shown('files.whole-game.runs.0.final.w'), ['1.595']);
  assert.deepEqual(shown('files.whole-game.runs.0.final.b'), ['1.897']);
  assert.deepEqual(shown('files.whole-game.data.true_line.w'), ['1.5']);
  assert.deepEqual(shown('files.whole-game.data.true_line.b'), ['2']);
});

test('no hand-typed digits in the prose: every number is bound, and the ChatGPT aside has none', () => {
  const prose = html.replace(/<style[\s\S]*?<\/style>/g, ' ').replace(/<svg[\s\S]*?<\/svg>/g, ' ').replace(/<code[^>]*>[\s\S]*?<\/code>/g, ' ')
    .replace(/<span class="num"[^>]*>[^<]*<\/span>/g, ' ').replace(/<h2[^>]*>\s*\d+ ·/g, '<h2>').replace(/<[^>]+>/g, ' ');
  assert.deepEqual(prose.match(/\S*\d\S*/g), null);
  const aside = html.match(/<aside class="chatgpt"[\s\S]*?<\/aside>/)[0];
  assert.doesNotMatch(aside, /class="num"/, 'the ChatGPT aside has no numbers (no source on this page)');
});
