// Lesson classification (template 3): every claim of the page checked from what TorC recorded (data/classification.json), the
// pure helpers of the playground (widgets/class-plane.js) against the data, the C2 rule (new points only once, for one model),
// and no hand-typed digits. The v3 rules (roles, budgets, glossary, notation order) run on this lesson in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, boundNums } from '../js/core/sections.js';
import { humanValue } from '../js/core/numbers.js';
import { modelFor, checkpointFor, pointsOf, gridOf, levelSegments, clipLine, unitLines, readout, stepLabel, pointsFigure, shade } from '../js/widgets/class-plane.js';
import { readJson, course } from './helpers.mjs';

const data = readJson('data/classification.json');
const D = buildD({ classification: data }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const lesson = (await import('../js/lessons/classification.js')).default;
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const END = 1000;
const last = (h) => checkpointFor(data, h, END);
// the scores of the trained network at a point, from the recorded layers (Linear -> ReLU -> Linear, weight (out, in))
function scores(cp, x, y) {
  const lin = (l, v) => { const [o, i] = l.weight.shape; return Array.from({ length: o }, (_, r) => l.bias.values[r] + v.reduce((s, vv, c) => s + l.weight.values[r * i + c] * vv, 0)); };
  let v = lin(cp.layers[0], [x, y]);
  if (cp.layers.length > 1) v = lin(cp.layers[1], v.map((a) => Math.max(0, a)));
  return v;
}

test('the trace is the one the lesson expects: models 0, 2, 4, 8 and checkpoints 0, 25, 100, 1000 in this order', () => {
  assert.equal(data.kind, 'classification');
  assert.deepEqual(data.models.map((m) => m.hidden), [0, 2, 4, 8]);
  assert.deepEqual(data.setup.hidden, [0, 2, 4, 8]);
  assert.deepEqual(data.setup.checkpoints, [0, 25, 100, 1000]);
  for (const m of data.models) assert.deepEqual(m.checkpoints.map((c) => c.step), data.setup.checkpoints);
  assert.equal(data.setup.steps, END);
  assert.equal(data.setup.train_samples, 200);
  assert.equal(data.grid.n, 41);
});

test('the points: orange (class 1) fill the disc, blue (class 0) the ring around it, so no straight line separates them', () => {
  for (const set of ['train', 'test']) {
    const pts = pointsOf(data, set);
    assert.equal(pts.length, data.setup[`${set}_samples`]);
    pts.forEach((p, i) => assert.equal(p.label, i % 2));
    for (const p of pts) {
      const r = Math.hypot(p.x, p.y);
      if (p.label === 1) assert.ok(r <= data.setup.r_inner + 1e-6, `orange point at radius ${r}`);
      else assert.ok(r >= data.setup.r_ring[0] - 1e-6 && r <= data.setup.r_ring[1] + 1e-6, `blue point at radius ${r}`);
    }
  }
  // blue points lie on every side of the disc: their angles leave no gap of 180 degrees, so any half-plane that holds the
  // whole disc (which contains the centre) also holds blue points
  const ang = pointsOf(data).filter((p) => p.label === 0).map((p) => Math.atan2(p.y, p.x)).sort((a, b) => a - b);
  const gaps = ang.map((a, i) => (i ? a - ang[i - 1] : a + 2 * Math.PI - ang.at(-1)));
  assert.ok(Math.max(...gaps) < Math.PI / 4);
});

test('predict: the best straight line gets 70 of 100 training points right (checked by a search here), this run ends lower', () => {
  const pts = pointsOf(data);
  let best = 0;
  for (let a = 0; a < 720; a++) {
    const t = (a * Math.PI) / 360, pr = pts.map((p) => [Math.cos(t) * p.x + Math.sin(t) * p.y, p.label]).sort((u, v) => u[0] - v[0]);
    const blue = pr.filter((q) => q[1] === 0).length;
    let ob = 0, bb = 0;
    for (let k = 0; k <= pr.length; k++) { best = Math.max(best, ob + blue - bb); if (k < pr.length) { if (pr[k][1] === 1) ob++; else bb++; } }
  }
  assert.equal(data.best_line.correct, best);
  assert.equal(Math.round(humanValue('pct', data.best_line.correct, data.setup.train_samples)), 70);
  assert.ok(last(0).train_correct < data.best_line.correct);
});

test('predict and task 1: one layer ends near a coin flip, straight and unsure everywhere', () => {
  const cp = last(0);
  assert.ok(cp.train_accuracy > 0.45 && cp.train_accuracy < 0.6, `train accuracy ${cp.train_accuracy}`);
  assert.equal(Math.round(humanValue('pct', cp.train_correct, data.setup.train_samples)), 53);
  for (const c of modelFor(data, 0).checkpoints) {
    // the boundary (the 50 % level of the recorded grid) lies on the recorded decision line: it is straight
    const { w1, w2, b } = c.derived.decision_line, n = Math.hypot(w1, w2);
    const segs = levelSegments(gridOf(data, c));
    assert.ok(segs.length > 10, `step ${c.step}`);
    for (const [x1, y1, x2, y2] of segs) for (const [x, y] of [[x1, y1], [x2, y2]]) assert.ok(Math.abs(w1 * x + w2 * y + b) / n < 0.02, `step ${c.step}: (${x}, ${y}) is off the line`);
    // and the probability of orange is the sigmoid of the line's value (two scores, one difference)
    data.grid.axis.forEach((x, i) => { const y = data.grid.axis[(i * 7) % 41]; assert.ok(Math.abs(c.prob1[((i * 7) % 41) * 41 + i] - sigmoid(w1 * x + w2 * y + b)) < 1e-5); });
  }
  // "the shading fades": at the end every probability is close to 50 %
  assert.ok(cp.prob1.every((p) => p > 0.35 && p < 0.65));
  assert.ok(Math.max(...checkpointFor(data, 0, 0).prob1) > 0.95, 'before training the shading is strong');
});

test('section 1: a unit is off on one side of its line, and the recorded counts agree with the lines', () => {
  const pts = pointsOf(data);
  for (const m of data.models.filter((x) => x.hidden > 0)) for (const c of m.checkpoints) {
    assert.equal(c.units.length, m.hidden);
    c.units.forEach((u, k) => {
      assert.equal(u.w1, c.layers[0].weight.values[2 * k]);
      assert.equal(u.b, c.layers[0].bias.values[k]);
      const on = pts.filter((p) => u.w1 * p.x + u.w2 * p.y + u.b > 0).length;
      assert.ok(Math.abs(on - u.active_train_points) <= 1, `${m.hidden} units, step ${c.step}, unit ${k}`);
    });
  }
  // the scores rebuilt from the recorded layers give the recorded probability of orange (the sigmoid of the difference)
  for (const h of [2, 4, 8]) {
    const c = last(h);
    for (const [i, j] of [[3, 5], [20, 20], [30, 11], [40, 0]]) {
      const [s0, s1] = scores(c, data.grid.axis[i], data.grid.axis[j]);
      assert.ok(Math.abs(c.prob1[j * 41 + i] - sigmoid(s1 - s0)) < 1e-4, `${h} units at (${i}, ${j})`);
    }
  }
});

test('task 2 and section 2: the four neuron lines cross inside the disc, every neuron votes for blue, the bias for orange', () => {
  const c = last(4), [l1, l2] = c.layers;
  assert.ok(c.units.every((u) => u.b < 0), 'at the centre every neuron is off');
  for (let k = 0; k < 4; k++) assert.ok(l2.weight.values[4 + k] - l2.weight.values[k] < 0, `neuron ${k + 1} votes for blue`);
  assert.ok(l2.bias.values[1] - l2.bias.values[0] > 0, 'the bias favours orange');
  // "the four lines cross inside the disc": each line is closer to the centre than the disc's radius
  for (const u of c.units) assert.ok(Math.abs(u.b) / Math.hypot(u.w1, u.w2) < data.setup.r_inner, `line at ${Math.abs(u.b) / Math.hypot(u.w1, u.w2)}`);
  // "the boundary lies further out": along the x axis the 50 % level is outside the disc's radius on both sides
  const row = 20, z = gridOf(data, c).z[row], ax = data.grid.axis;
  const right = ax.findIndex((x, i) => x > 0 && z[i] < 0.5), left = ax.findLastIndex((x, i) => x < 0 && z[i] < 0.5);
  assert.ok(ax[right] > data.setup.r_inner && ax[left] < -data.setup.r_inner, `${ax[left]} .. ${ax[right]}`);
  assert.equal(l1.weight.shape.join(), '4,2');
  assert.equal(c.train_accuracy, 1);
  // the worked vote: the recorded votes are output x (orange weight - blue weight), and they add up to the score difference
  const V = data.vote, [centre, ring] = V.points;
  assert.deepEqual([V.hidden, V.step], [4, 1000]);
  V.vote_weights.forEach((w, k) => assert.ok(Math.abs(w - (l2.weight.values[4 + k] - l2.weight.values[k])) < 1e-6));
  assert.ok(Math.abs(V.bias_orange - (l2.bias.values[1] - l2.bias.values[0])) < 1e-6);
  for (const q of V.points) {
    q.votes.forEach((v, k) => assert.ok(Math.abs(v - q.outputs[k] * V.vote_weights[k]) < 1e-6 * (1 + Math.abs(v))));
    assert.ok(Math.abs(q.total - (V.bias_orange + q.votes.reduce((a, b) => a + b, 0))) < 1e-9);
    assert.ok(Math.abs(q.total - q.score_difference) < 1e-4);
    const [s0, s1] = scores(c, q.x, q.y);
    assert.ok(Math.abs(s1 - s0 - q.score_difference) < 1e-3);
  }
  assert.ok(centre.outputs.every((o) => o === 0) && centre.total > 0, 'in the middle all neurons are off and the bias wins');
  assert.deepEqual(ring.outputs.map((o) => o > 0), [true, false, false, true], 'only neurons 1 and 4 are on at the ring point');
  assert.ok(ring.total < 0, 'the ring point is blue');
  assert.ok(Math.hypot(ring.x, ring.y) > data.setup.r_ring[0], 'the point lies in the ring');
});

test('task 3: two units leave the orange region open, eight units close it', () => {
  const edge = (c) => { const n = 41, out = []; for (let k = 0; k < n; k++) out.push(c.prob1[k], c.prob1[(n - 1) * n + k], c.prob1[k * n], c.prob1[k * n + n - 1]); return out; };
  for (const s of [100, 1000]) assert.ok(edge(checkpointFor(data, 2, s)).filter((p) => p > 0.5).length > 4, `with two units orange reaches the edge of the plane at step ${s}`);
  assert.ok(edge(last(8)).every((p) => p < 0.5), 'with eight units the orange region is closed');
  assert.ok(edge(last(4)).every((p) => p < 0.5));
});

test('the sigmoid section and the your-turn quiz: softmax of two scores is the sigmoid of their difference', () => {
  for (const e of data.sigmoid_identity) {
    assert.equal(e.abs_diff, 0);
    assert.ok(Math.abs(e.difference - (e.scores[1] - e.scores[0])) < 1e-6);
    assert.ok(Math.abs(e.sigmoid - sigmoid(e.difference)) < 1e-6);
    assert.ok(Math.abs(e.softmax[1] - e.sigmoid) < 1e-6 && Math.abs(e.softmax[0] + e.softmax[1] - 1) < 1e-6);
  }
  const [eq, one, , q] = data.sigmoid_identity;
  assert.deepEqual([eq.scores, eq.difference, eq.sigmoid], [[0, 0], 0, 0.5]);
  assert.deepEqual([one.scores, one.difference], [[0, 1], 1]);
  assert.deepEqual([q.scores, q.difference], [[-1, 2], 3]);
});

test('section 4: the typical probability rises with training, and the 2-neuron model does not improve from step 100 to 1000', () => {
  for (const m of data.models) for (const c of m.checkpoints) assert.ok(Math.abs(c.typical_right_prob - Math.exp(-c.train_loss)) < 1e-9);
  const before = checkpointFor(data, 8, 0).typical_right_prob, after = last(8).typical_right_prob;
  assert.ok(before < 0.55 && after > 0.999 && after < 1, `${before} -> ${after}`);
  assert.equal(T(`files.classification.models.3.checkpoints.3.typical_right_prob`, { role: 'contrast', sig: 4 }).match(/>([^<]*)</)[1], '0.9998', 'never shown as 1 or 100 %');
  const per = (s) => humanValue('per100', checkpointFor(data, 2, s).train_correct, data.setup.train_samples);
  assert.equal(per(100), per(1000));
  assert.ok(per(100) > 15, 'two units still make many mistakes');
});

test('cumulative quiz: the parameter count is the weights and biases of both layers', () => {
  const m = modelFor(data, 4), [l1, l2] = m.checkpoints[0].layers;
  const n = (l) => l.weight.shape[0] * l.weight.shape[1] + l.bias.shape[0];
  assert.equal(m.parameters, n(l1) + n(l2));
  assert.equal(m.parameters, 22);
  assert.equal(modelFor(data, 8).parameters, 42, 'the Sequential of the real line (and the course.json code) has 42 parameters');
});

test('C2: the new points appear once, for one model at the end, and every comparison uses training numbers', () => {
  const html = renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n');
  const testPaths = boundNums(html).map((b) => b.path).filter((p) => /test_(correct|loss|accuracy)/.test(p));
  assert.deepEqual(testPaths, ['files.classification.models.3.checkpoints.3.test_correct']);
  assert.equal(Math.round(humanValue('pct', last(8).test_correct, data.setup.test_samples)), 100);
  assert.match(html, /training points only/);
  assert.match(html, /ran once/, 'the single run is labelled (C5)');
});

test('the playground helpers: readouts round like T.human, lines are clipped to the plane, labels', () => {
  for (const m of data.models) for (const c of m.checkpoints) {
    const r = readout(data, c);
    assert.equal(r.mistakes, humanValue('per100', c.train_correct, data.setup.train_samples));
    const lines = unitLines(data, c);
    assert.equal(lines.length, Math.max(1, m.hidden));
    for (const u of lines) if (u.seg) for (const [x, y] of u.seg) {
      assert.ok(Math.abs(u.w1 * x + u.w2 * y + u.b) < 1e-6 && x >= -2 - 1e-9 && x <= 2 + 1e-9 && y >= -2 - 1e-9 && y <= 2 + 1e-9);
    }
  }
  assert.deepEqual(clipLine(1, 0, 0, -2, 2).map((p) => p.map((v) => v + 0)), [[0, -2], [0, 2]]);
  assert.equal(clipLine(1, 0, -5, -2, 2), null);
  assert.deepEqual(clipLine(1, -1, 0, -2, 2).map((p) => p.map(Math.abs)), [[2, 2], [2, 2]]);
  assert.deepEqual(data.setup.checkpoints.map(stepLabel), ['before training', '25 steps', '100 steps', '1,000 steps']);
  assert.equal(shade(0.5), 'rgb(14,17,22)');
  assert.match(pointsFigure(data), /<svg[^>]*role="img"/);
  assert.equal((pointsFigure(data).match(/<circle/g) ?? []).length, 100);
  // a level line of a plane z = x crosses at x = 0.5 exactly
  const g = { xs: [0, 1], ys: [0, 1], z: [[0, 1], [0, 1]] };
  assert.deepEqual(levelSegments(g), [[0.5, 0, 0.5, 1]]);
});

test('no hand-typed digits in the prose (code, formulas and bound numbers aside)', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>')
    .replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('one name per thing: hidden neurons, never "units"; no 100 % shown for a probability below one', () => {
  const html = renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n');
  const text = html.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<[^>]+>/g, ' ');
  assert.doesNotMatch(text, /\bunits?\b/);
  assert.doesNotMatch(text, /\btypical training point\b/);
  for (const b of boundNums(html)) if (/typical_right_prob/.test(b.path)) assert.notEqual(b.shown, '1');
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/class-plane.js');
  assert.equal(typeof m.mountClassPlane, 'function');
  assert.equal(typeof document, 'undefined');
});
