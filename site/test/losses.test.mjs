// Lesson losses ("Softmax and cross-entropy", template 3): every claim of the page checked from what TorC recorded
// (data/losses.json), the widget helpers, the data the appendix lesson torc-pitfalls needs (mse_mismatch, errors), and no
// hand-typed digits. The v3 rules (roles, budgets, glossary, notation order) run on this lesson in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { humanValue } from '../js/core/numbers.js';
import { explorerRows, sig3, pct, FRUITS } from '../js/widgets/loss-widgets.js';
import { readJson, course } from './helpers.mjs';

const data = readJson('data/losses.json');
const D = buildD({ losses: data }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const lesson = (await import('../js/lessons/losses.js')).default;
const close = (a, b, tol = 1e-5, msg = '') => assert.ok(Math.abs(a - b) <= tol * (1 + Math.abs(b)), `${msg} ${a} vs ${b}`);
const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
const render = () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
};

test('the rows the page uses: uniform, mild [2, 1, 0.1], confident [4, 1, -2], in this order', () => {
  assert.deepEqual(data.ce.map((r) => r.id), ['uniform', 'mild', 'confident']);
  assert.deepEqual(data.exp_rows.map((r) => r.id), ['uniform', 'mild', 'confident']);
  data.exp_rows.forEach((r, i) => assert.deepEqual(r.logits, data.ce[i].logits));
  assert.deepEqual(data.ce[2].logits, [4, 1, -2]);
});

test('softmax: e to the power of each score, divided by the sum; every share is above zero and the shares add to one', () => {
  for (const r of data.exp_rows) {
    r.logits.forEach((z, j) => close(r.exp[j], Math.exp(z), 1e-6, `${r.id} exp ${j}`));
    close(r.exp_sum, r.exp.reduce((a, b) => a + b, 0), 1e-6, `${r.id} sum`);
    r.shares.forEach((p, j) => close(p, r.exp[j] / r.exp_sum, 1e-6, `${r.id} share ${j}`));
    r.shares.forEach((p, j) => assert.equal(p, data.ce.find((c) => c.id === r.id).cases[0].softmax[j], 'the same shares as CrossEntropyLoss'));
    close(r.shares.reduce((a, b) => a + b, 0), 1, 1e-6, r.id);
    assert.ok(r.shares.every((p) => p > 0), 'every class keeps a share above zero');
    // a larger score gets a larger share
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) if (r.logits[a] > r.logits[b]) assert.ok(r.shares[a] > r.shares[b]);
  }
  const conf = data.exp_rows[2];
  assert.ok(conf.exp[2] > 0 && conf.exp[2] < 1, 'a negative score gives a positive result below one');
  assert.ok(conf.shares[2] > 0 && conf.shares[2] < 0.01, 'the predict: the cherry gets a small share above zero');
  assert.ok(conf.shares[0] > 0.9, 'the model is very sure of the apple');
});

test('cross-entropy is -ln of the share of the right class (the worked example, the uniform check, ten digits)', () => {
  for (const r of data.ce) {
    const p = softmax(r.logits);
    for (const c of r.cases) {
      c.softmax.forEach((v, j) => close(v, p[j], 1e-6, `${r.id} softmax ${j}`));
      close(c.loss, -Math.log(p[c.target]), 1e-5, `${r.id} t${c.target} loss`);
      c.grad.forEach((g, j) => close(g, p[j] - (j === c.target ? 1 : 0), 1e-5, `${r.id} t${c.target} grad ${j}`));
    }
  }
  const mild = data.ce[1];
  close(mild.cases[0].loss, -Math.log(data.exp_rows[1].shares[0]), 1e-6, 'worked: -ln(0.659)');
  assert.equal(Math.round(humanValue('prob', mild.cases[0].loss)), 66, 'the model gave the apple 66 %');
  // the surprise grows as the right class gets a smaller share (task 1)
  const L = mild.cases.map((c) => c.loss);
  assert.ok(L[0] < L[1] && L[1] < L[2]);
  for (const c of data.ce[0].cases) close(c.loss, Math.log(3), 1e-6, 'three equal scores: ln 3');
  // the transfer quiz: ten equal scores give each digit 0.1 and the surprise ln 10, while school log of 10 is 1
  assert.equal(data.ten.classes, 10);
  assert.equal(data.ten.logits.length, 10);
  data.ten.softmax.forEach((p) => close(p, 0.1, 1e-6));
  close(data.ten.loss, Math.log(10), 1e-6, 'ln 10');
  assert.equal(Math.log10(10), 1);
  assert.equal(D.anchors.classes, data.ten.classes, 'the anchor "classes" is the ten digits');
  // task 2: very sure of the apple, but the cherry is right: the largest surprise of the page
  assert.ok(data.ce[2].cases[2].loss > 5 && data.ce[2].cases[2].loss === Math.max(...data.ce.flatMap((r) => r.cases.map((c) => c.loss))));
});

test('the misconception: the same share for the right class gives the same surprise, however the rest is spread', () => {
  const [even, uneven] = data.spread;
  assert.deepEqual([even.id, uneven.id, even.target, uneven.target], ['even', 'uneven', 0, 0]);
  close(even.softmax[0], uneven.softmax[0], 1e-6, 'same apple share');
  close(even.loss, uneven.loss, 1e-6, 'same surprise');
  assert.equal(even.softmax[1], even.softmax[2], 'the even row splits the rest evenly');
  assert.ok(uneven.softmax[1] > 10 * uneven.softmax[2], 'the uneven row does not');
});

test('section 4 and the second predict: the surprise keeps a large gradient where the squared miss has almost none', () => {
  const C = data.curves;
  assert.equal(C.z.length, 49);
  assert.deepEqual([C.z[0], C.z[48]], [-6, 6]);
  C.z.forEach((z, i) => {
    const p = 1 / (1 + Math.exp(-z));
    close(C.p[i], p, 1e-5, `share at ${z}`);
    close(C.ce_loss[i], -Math.log(p), 1e-5, `ce loss at ${z}`);
    close(C.ce_grad[i], C.p[i] - 1, 1e-5, `the gradient is the share minus one at ${z}`);
    close(C.sq_loss[i], (1 - p) ** 2, 1e-5, `sq loss at ${z}`);
    close(C.sq_grad[i], -2 * (1 - p) ** 2 * p, 1e-5, `sq grad at ${z}`);
  });
  assert.ok(C.ce_grad[0] < -0.99, 'near minus one');
  assert.ok(Math.abs(C.sq_grad[0]) < 0.01, 'near zero');
  assert.ok(Math.abs(C.ce_grad[0]) > 100 * Math.abs(C.sq_grad[0]));
});

test('the data that the appendix lesson torc-pitfalls needs is kept, and none of it is on this page', () => {
  const m = data.mse_mismatch;
  assert.deepEqual([m.prediction_shape, m.target_shape, m.exception], [[3, 1], [3], 'ShapeError']);
  assert.equal(m.what, 'MSELoss input shape (3, 1) must match target shape (3)');
  const e = Object.fromEntries(data.errors.map((x) => [x.id, x]));
  assert.deepEqual(Object.keys(e), ['not-integer', 'out-of-range', 'rank-1-logits', 'length', 'column-targets']);
  assert.equal(e['out-of-range'].what, 'CrossEntropyLoss target 3 at index 0 is out of range [0, 3)');
  const html = render();
  assert.doesNotMatch(html, /MSELoss|ShapeError|mse_mismatch/, 'the MSE shape pitfall and the label errors live in torc-pitfalls');
});

test('the explorer offers the page rows, and its readouts round like the page says', () => {
  const rows = explorerRows(data);
  assert.deepEqual(rows.map((r) => r.key), ['uniform', 'mild', 'confident', 'even', 'uneven']);
  assert.equal(rows[1].label, 'first photo (2, 1, 0.1)');
  assert.equal(rows[2].label, 'second photo (4, 1, −2)');
  assert.match(rows[3].label, /same apple probability/);
  assert.match(rows[4].label, /same apple probability/);
  assert.deepEqual(rows[3].cases.map((c) => c.target), [0], 'the spread rows have the apple as the right class');
  assert.deepEqual(FRUITS, ['apple', 'banana', 'cherry']);
  assert.equal(pct(data.exp_rows[1].shares[0]), '66%');
  assert.equal(pct(data.exp_rows[2].shares[2]), '0.24%');
  assert.equal(pct(0.9975), '99.8%');
  assert.equal(sig3(data.ce[1].cases[0].loss), '0.417');
  assert.equal(sig3(-0.00123456), '−0.00123');
});

test('C2-type honesty: the ChatGPT aside names no vocabulary size, says it is not a TorC run, and the footer cites its source', () => {
  const html = render();
  const aside = html.match(/<aside class="chatgpt"[\s\S]*?<\/aside>/)[0];
  assert.doesNotMatch(aside.replace(/<[^>]+>/g, ''), /\d/);
  assert.match(aside, /not a TorC run/);
  assert.match(lesson.footer(D, T), /language_understanding_paper\.pdf">Radford et al\. \(2018\)[^<]*section 3\.1/);
  assert.match(lesson.footer(D, T), /arxiv\.org\/abs\/2203\.02155/);
  assert.match(aside, /pretraining/);
  assert.doesNotMatch(aside, /thousand/);
  assert.match(lesson.footer(D, T), /Numbers: TorC/);
  assert.match(html, /log always means ln/);
});

test('no hand-typed digits in the prose (code, formulas and bound numbers aside)', () => {
  const left = render().replace(/<style[\s\S]*?<\/style>/g, '').replace(/<details class="refresher"[\s\S]*?<\/details>/g, '')
    .replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code[^>]*>[\s\S]*?<\/code>/g, '')
    .replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, ['100%.'], 'only the analogy "shares of 100%" is typed');
  assert.doesNotMatch(render().replace(/<details class="refresher"[\s\S]*?<\/details>/g, '').replace(/<[^>]+>/g, ' '), /\bshares?\b(?! of 100%)/, 'the name is probability; "shares of 100%" is only the analogy');
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/loss-widgets.js');
  assert.equal(typeof m.mountLossExplorer, 'function');
  assert.equal(typeof m.mountGradientCompare, 'function');
  assert.equal(typeof document, 'undefined');
});
