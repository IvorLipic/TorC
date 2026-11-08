// Lesson torc-pitfalls (appendix): every claim of the page against what TorC recorded (autograd_trace, optim, training, losses,
// linear and tensors), the zero_grad loop widget against the recorded loop, every message quoted on the page, and every number
// of the prose bound to the trace.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { makeModel, trainingLoop } from '../js/optim.js';
import { readJson } from './helpers.mjs';

const lesson = (await import('../js/lessons/torc-pitfalls.js')).default;
const stems = lesson.trace.map((f) => f.replace(/\.json$/, ''));
const files = Object.fromEntries(lesson.trace.map((f, i) => [stems[i], readJson('data/' + f)]));
const D = buildD(files, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
const A = files.autograd_trace;
const EXPIRED = 'autograd graph input expired before backward(); keep all graph Variables alive';

test('gradients add up across new forward passes, and a second backward() on the same loss does nothing', () => {
  assert.deepEqual([A.accumulation.w, A.accumulation.after_first_backward, A.accumulation.after_second_backward_same_loss, A.accumulation.after_new_forward_and_backward],
    [3, 6, 6, 12]);
  assert.equal(A.accumulation.has_grad_after_zero_grad, false, 'zero_grad removes the stored gradient');
  assert.equal(A.accumulation.after_first_backward, 2 * A.accumulation.w, 'L = w * w has the gradient 2w');
});

test('the zero_grad loop: with zero_grad the loss falls, without it the stored gradient changes sign and the loss jumps', () => {
  const optim = files.optim, lr = optim.no_zero_grad.training_loop.lr, m = makeModel(D.scenes.sgd);
  const withZ = trainingLoop(m, { lr, iterations: 10, zeroGrad: true }), noZ = trainingLoop(m, { lr, iterations: 10, zeroGrad: false });
  assert.ok(withZ.every((r, i) => i === 0 || r.loss < withZ[i - 1].loss), 'with zero_grad the loss falls at every iteration');
  assert.ok(withZ.at(-1).loss < withZ[0].loss / 1000);
  assert.ok(noZ.some((r, i) => i > 0 && r.loss > noZ[i - 1].loss), 'without it the loss rises at some iteration');
  assert.ok(noZ.some((r) => r.gw > 0) && noZ.some((r) => r.gw < 0), 'the stored gradient changes sign');
  assert.equal(noZ[0].loss, withZ[0].loss, 'the first iteration is the same');
  assert.ok(Math.abs(noZ[1].gw) > Math.abs(withZ[1].gw), 'later bars are larger');
});

test('step() skips a parameter without a stored gradient, and gives no error', () => {
  const s = files.optim.skipped_step;
  for (const name of ['step_without_backward', 'step_after_zero_grad', 'momentum_step_without_backward']) {
    const c = s.find((x) => x.name === name);
    assert.deepEqual([c.changed, c.threw], [false, false], name);
  }
  assert.equal(s.find((x) => x.name === 'step_after_backward').changed, true);
  const u = A.unused_parameter;
  assert.deepEqual([u.used_has_grad, u.unused_has_grad], [true, false]);
  assert.equal(u.unused_after, u.start);
  assert.equal(u.used_grad, 2 * u.start, 'the gradient of w * w is 2w');
  assert.ok(Math.abs(u.used_after - (u.start - u.lr * u.used_grad)) < 1e-5, 'the step of the worked block');
});

test('graph lifetime: temporaries and locals break backward(), named Variables work', () => {
  const by = Object.fromEntries(A.lifetime.map((c) => [c.id, c]));
  assert.deepEqual([by.named.exception, by['model-named'].exception], ['none', 'none']);
  assert.equal(by.named.grad, 6);
  for (const id of ['temporary', 'model-temporary', 'local']) { assert.equal(by[id].exception, 'TorcError', id); assert.equal(by[id].what, EXPIRED, id); }
  assert.match(by.temporary.code, /sum\(mul\(w, w\)\)/);
  assert.match(by['model-temporary'].code, /mse\(layer\(x\), t\)/);
});

test('without reset() the second and third epoch take no steps, and the loss does not change', () => {
  const r = files.training.no_reset;
  assert.deepEqual(r.map((e) => e.steps), [10, 0, 0]);
  assert.ok(r.every((e) => e.loss === r[0].loss));
  assert.equal(files.training.setup.samples / files.training.loaders[0].batch_size, r[0].steps, 'ten batches of ten samples');
  assert.equal(files.training.exhausted.what, 'DataLoader::next_batch: no more batches in current epoch');
});

test('the messages of the shape and setting errors', () => {
  assert.equal(files.losses.mse_mismatch.what, 'MSELoss input shape (3, 1) must match target shape (3)');
  const ce = Object.fromEntries(files.losses.errors.map((e) => [e.id, e]));
  assert.equal(ce['not-integer'].what, 'CrossEntropyLoss target at index 0 must be an integer');
  assert.equal(ce['out-of-range'].what, 'CrossEntropyLoss target 3 at index 0 is out of range [0, 3)');
  assert.equal(ce['rank-1-logits'].what, 'CrossEntropyLoss expects rank-2 logits, got shape (3)');
  assert.equal(ce.length.what, 'CrossEntropyLoss target length 3 does not match batch size 2');
  assert.equal(ce['column-targets'].exception, 'none', 'a column of targets (batch, 1) is accepted');
  const lin = Object.fromEntries(files.linear.shape_errors.map((e) => [e.id, e]));
  assert.equal(lin['linear-784x32-(8, 10)'].what, 'matmul inner dimensions must match, got (8, 10) and (784, 32)');
  assert.equal(lin['linear-3x2-(3)'].what, 'matmul requires rank >= 2 operands, got (3) and (3, 2)');
  assert.equal(lin['linear-zero-features'].what, 'Linear features must be positive');
  assert.equal(files.optim.ctor_errors[0].what, 'SGD: lr must be positive');
  assert.equal(files.tensors.index_errors[0].what, 'Index out of bounds at dim 0: 2 not in [0, 2)');
  assert.equal(files.tensors.reduce_error.what, 'Invalid axis 2 for tensor with shape (2, 3)');
  for (const what of [files.losses.mse_mismatch.what, ce['not-integer'].what, ce['out-of-range'].what, ce['rank-1-logits'].what, ce.length.what,
    lin['linear-784x32-(8, 10)'].what, lin['linear-3x2-(3)'].what, lin['linear-zero-features'].what, files.optim.ctor_errors[0].what,
    files.tensors.index_errors[0].what, files.tensors.reduce_error.what, EXPIRED, files.training.exhausted.what]) {
    assert.ok(html.includes(what.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')), `the page quotes: ${what}`);
  }
});

test('every number in the prose is bound to the trace, and the quiz answers are the recorded values', () => {
  for (const { section, html: h } of renderLesson(lesson, D)) {
    assert.deepEqual(findUnboundNums(h), [], section.kind);
    for (const b of boundNums(h)) assert.equal(T.reformat(b), b.shown, b.path);
  }
  const quiz = lesson.sections.find((s) => s.id === 'p-reset');
  assert.equal(T.num(quiz.answer), 0, 'the second epoch takes no steps');
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/loop-timeline.js');
  assert.equal(typeof m.mountLoopTimeline, 'function');
  assert.equal(typeof document, 'undefined');
});
