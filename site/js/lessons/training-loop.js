// Lesson "training-loop" (template 3, curriculum v3 Part 4 `training`, brief §6 and §7): 100 noisy samples of the line t = 3x + 1
// and the model y = w·x + b. Batches and epochs (the batch viewer), the batch gradient as an estimate of the gradient of all the
// samples (the S in SGD, `gradient_estimate`), why the loader shuffles (a sorted order without shuffle, `sorted_order`), a fair
// batch-size comparison at equal steps and at equal samples seen with three loader seeds (brief C1, C5: `batch_compare`), the
// five-line loop as the "real line" moment, zero_grad in an In TorC panel with the reason (`accumulate`), and the noise that no
// line removes. The old reset() check moved to the appendix (torc-pitfalls). Every number comes from data/training.json
// (tools/trace/trace_training.cpp) through a role-tagged T(); site/test/training-loop.test.mjs checks each claim of the prose.
import { worked, realLine, inTorc, term } from '../core/blocks.js';
import { mountBatchViewer, mountGradientEstimate, mountBatchCompare } from '../widgets/training-widgets.js';

const F = 'files.training';
const BC = `${F}.batch_compare`;
const GE = `${BC}.gradient_estimate`;
const AC = `${F}.accumulate`;
const ES = `${BC}.equal_steps.0`;       // after 30 steps each
const EQ = `${BC}.equal_samples.0`;     // after 3000 samples each
// summaries of the comparison by batch size, and the results of seed 7 (the test pins this order: batch 10 first, then 100)
const SUM10 = 0, SUM100 = 1, RES10 = 0, RES100 = 3;
// the runs and loaders of the trace, found by batch size
const run = (D, bs) => `${F}.runs.${D.files.training.runs.findIndex((r) => r.batch_size === bs)}`;
const loader = (D, bs) => `${F}.loaders.${D.files.training.loaders.findIndex((l) => l.batch_size === bs)}`;
export const STEPS_32 = `${F}.runs.1.epochs.1.steps`;   // the numeric answer: steps of one epoch with batch size 32
const B10 = (T, role = 'setup') => T(`${BC}.batch_sizes.0`, role);
const B100 = (T, role = 'setup') => T(`${BC}.batch_sizes.1`, role);
const N = (T, role = 'setup') => T(`${F}.setup.samples`, role);
const rmse = (T, path) => T.human(path, 'rmse');

// The training loop as it runs in tools/trace/trace_training.cpp (batch 10, shuffled, 30 epochs); also the `code` of this lesson
// in course.json (tools/snippets/check.mjs compiles and runs it, the test checks that the two are equal).
export const LOOP = `for (int epoch = 0; epoch < 30; ++epoch) {
  // a new random order
  loader.reset();
  while (loader.has_next()) {
    auto [xb, tb] = loader.next_batch();
    Variable x(xb, false), t(tb, false);
    // 1. predict
    Variable y = model(x);
    // 2. the loss of this batch
    Variable L = loss_fn(y, t);
    // 3. the gradients
    L.backward();
    // 4. change the parameters
    opt.step();
    // 5. clear the gradients
    opt.zero_grad();
  }
}`;

const sec = (id, title, body) => `<section class="tr" aria-labelledby="tr-${id}"><h2 id="tr-${id}">${title}</h2>${body}</section>`;
const STYLE = `<style>
.tr p,.tr ul{max-width:44rem}
.tr h2{margin-top:2rem}
</style>`;

export default {
  id: 'training-loop',
  title: 'The training loop',
  trace: ['training.json'],
  template: 3,
  loop: 'all',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>The training loop</h1>
    <p class="lead">A computer that learns from millions of examples does not use all of them in each step.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        `A ${term('data-set', 'data set')} holds a million ${term('sample', 'samples')}. Must every ${term('training')} step use all of them?`,
        'Are small groups of samples better than large ones?',
        'What does a training program repeat?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('points', 'One hundred points near a line', `
    <p>A short program made a data set, a collection of ${N(T)} samples. Each sample is an input x with its ${term('target')} t = ${T(`${F}.setup.true_weight`, 'setup')}x + ${T(`${F}.setup.true_bias`, 'setup')}, plus a small random number, the noise.</p>
    <p>The ${term('model')} is a line, y = w·x + b. Its ${term('weight')} w and its ${term('bias')} b are the ${term('parameter', 'parameters')}, and both start at ${T(`${GE}.at.w`, 'setup')}. The ${term('loss')} L is the ${term('mean-squared-error', 'mean squared error')} of the ${term('prediction', 'predictions')} y.</p>
    <p>At the start, the typical miss, the square root of the loss, is about ${rmse(T, `${BC}.runs.0.loss.0`)}.</p>`),
    },
    {
      kind: 'quiz', id: 'estimate', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>Take ${B10(T)} of the ${N(T)} samples at random, and compute the ${term('gradient')} for w from them. How does it compare with the gradient from all ${N(T)}?</p>`,
      options: [
        { html: 'It is the same number.', feedback: 'Another random choice gives other samples, so the number changes.' },
        { html: 'It points roughly the same way, but it can be far off.', correct: true, feedback: 'Ten samples give only a rough picture.' },
        { html: 'It is about ten times smaller, because it uses ten times fewer samples.', feedback: 'The loss is a mean, so fewer samples do not shrink it.' },
      ],
      reveal: (D, T) => `<p>For all ${N(T, 'reveal')} samples, the gradient for w is ${T(`${GE}.full.gw`, { role: 'reveal', sig: 2 })}. Ten random groups gave values from ${T(`${GE}.range.gw_min`, { role: 'reveal', sig: 2 })} to ${T(`${GE}.range.gw_max`, { role: 'reveal', sig: 2 })}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('batches', '1 · Batches and epochs', `
    <p>A ${term('batch')} is a small group of samples that one step uses. Its size is the batch size B. TorC's data loader hands out the batches.</p>
    <p>One pass through all the samples is an ${term('epoch')}, so an epoch has many steps:</p>
    ${worked(`<p>${T(`${F}.setup.samples`, 'worked')} samples ÷ ${T(`${loader(D, 10)}.batch_size`, 'worked')} samples per batch = ${T(`${run(D, 10)}.epochs.1.steps`, 'worked')} steps in one epoch</p>`)}
    <p>Before each epoch, the loader can shuffle the samples into a new random order.</p>`),
    },
    {
      kind: 'widget', id: 'batches', className: 'widget', label: 'Interactive: a data loader hands out batches of the samples',
      mount: (host, D) => mountBatchViewer(host, D),
      tasks: () => [
        { do: 'Press Next batch until the epoch is over.', see: 'Each press is one step, and one epoch takes many.' },
      ],
    },
    {
      kind: 'quiz', id: 'steps-32', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: STEPS_32, tolerance: 0.5,
      prompt: (D, T) => `<p>Another loader has ${N(T)} samples, a batch size of ${T(`${loader(D, 32)}.batch_size`, 'setup')}, and a smaller last batch for the rest. How many steps does one epoch take?</p>`,
      reveal: (D, T) => `<p>Three batches of ${T(`${loader(D, 32)}.batch_size`, 'reveal')} leave a few samples for the smaller last batch. So one epoch takes ${T(`${run(D, 32)}.epochs.1.steps`, 'reveal')} steps.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('estimate', '2 · A batch gives an estimate of the gradient', `
    <p>Each step of ${term('gradient-descent', 'gradient descent')} uses the gradient of one batch. It is an estimate of the gradient of all the samples. It points roughly downhill, but it can be far off.</p>
    <p>At the same w and b, the ten batch gradients average to the gradient of all the samples, because the batches are equal in size. So over many steps, estimates that are too high and too low partly cancel. And a batch costs much less to compute.</p>
    <p>This method is ${term('stochastic-gradient-descent', 'stochastic gradient descent')}, or SGD, where stochastic means random.</p>`),
    },
    {
      kind: 'widget', id: 'scatter', className: 'widget', label: 'Interactive: the gradient of each batch next to the gradient of all samples',
      mount: (host, D) => mountGradientEstimate(host, D),
      tasks: () => [
        { do: 'Press Next batch until all ten dots show. Then press “Show the average of the dots”.', see: 'The dots scatter around the star, and most sit far from it. Their average sits on the star.' },
      ],
    },
    {
      kind: 'text',
      html: () => sec('shuffle', '3 · Why the loader shuffles', `
    <p>Why shuffle? Suppose that the loader keeps the samples sorted by x. The first batches then hold only the leftmost points, and each step pulls the line toward them. So the miss on all the samples grows at first.</p>
    <p>A random order makes each batch a fair mix. A new order in each epoch gives new batches, so the estimates are not off the same way twice.</p>`),
    },
    {
      kind: 'quiz', id: 'batch-size', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>One run uses batches of ${B10(T)}. Another uses batches of ${B100(T)}, all the samples. Both start from the same line. After ${T(`${ES}.steps`, 'setup')} steps each, which line misses less?</p>`,
      options: [
        { html: (D, T) => `The run with batches of ${B100(T)}, by far, because its steps used more samples.`, feedback: 'Its gradient is exact. But over many steps, the too high and too low estimates of the small batches partly cancel.' },
        { html: (D, T) => `The run with batches of ${B10(T)}, by far, because small batches learn better.`, feedback: 'Small batches are not better per step.' },
        { html: 'Both miss by about the same amount.', correct: true, feedback: 'Per step, both make about the same progress here.' },
      ],
      reveal: (D, T) => `<p>Both sizes miss by about ${rmse(T, `${ES}.summary.${SUM100}.min`)}, in all three random orders. But the large batches read ten times more samples.</p>`,
    },
    {
      kind: 'widget', id: 'compare', className: 'widget', label: 'Interactive: the typical miss of two batch sizes, by steps and by samples seen',
      mount: (host, D) => mountBatchCompare(host, D),
      tasks: (D, T) => [
        { do: `Choose “Same number of steps” and move the marker to ${T(`${ES}.steps`, 'setup')}.`, see: 'The two curves almost cover each other.' },
        { do: `Choose “Same number of samples” and move it to ${T(`${EQ}.samples`, 'setup')}.`, see: `Batch ${B10(T)} took ten times more steps and misses far less.` },
        { do: 'Choose “Sorted or shuffled” and look at the first steps.', see: 'Sorted, the miss grows at first. Shuffled, it falls at once.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('fair', '4 · Which batch size is better?', `
    <p>Both sizes used the same ${term('learning-rate', 'learning rate')}, ${T(`${BC}.lr`, 'setup')}, chosen by us and not tuned. After equal steps, they missed by about the same amount. After equal samples, the small batches took more steps:</p>
    ${worked(`<p>${T(`${EQ}.samples`, 'worked')} ÷ ${T(`${BC}.batch_sizes.0`, 'worked')} = ${T(`${EQ}.results.${RES10}.steps`, 'worked')} steps, and ${T(`${EQ}.samples`, 'worked')} ÷ ${T(`${BC}.batch_sizes.1`, 'worked')} = ${T(`${EQ}.results.${RES100}.steps`, 'worked')} steps</p>`)}
    <p>Then the small batches miss by about ${rmse(T, `${EQ}.summary.${SUM10}.max`)}, and the large ones by about ${rmse(T, `${EQ}.summary.${SUM100}.max`)}. So the better size depends on what you count. A small batch costs less per step, but its gradient is noisier. A large batch can often use a larger learning rate, untried here. Each random order gives slightly different numbers, so we ran three.</p>`),
    },
    {
      kind: 'quiz', id: 'friend', purpose: 'transfer', type: 'choice',
      prompt: '<p>A friend trains a model twice for the same number of epochs. The run with ten times smaller batches ends lower. Your friend calls small batches better. What did this comparison hold equal?</p>',
      options: [
        { html: 'The number of samples seen.', correct: true, feedback: 'Each epoch reads every sample once, so equal epochs mean equal samples.' },
        { html: 'The number of steps.', feedback: 'With ten times smaller batches, each epoch takes ten times more steps.' },
        { html: 'Nothing, so the result means nothing.', feedback: 'It held the samples equal.' },
      ],
      reveal: '<p>Per sample, the small batches won, because they took ten times more steps.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('loop', '5 · The loop in TorC', `
    <p>Every model here trains with this loop. Its last line clears the gradients, because TorC would otherwise add the next gradients to the old ones.</p>
    ${realLine(LOOP, 'The outer loop counts the epochs. The inner loop runs once per batch, and each pass is one step. Its first two lines take the batch and hand it to TorC, with no gradients for the data (false).')}
    ${inTorc('Why the loop clears the gradients', `<p>TorC stores a gradient with each parameter and adds each new gradient to it. So the loop clears them with <code>opt.zero_grad()</code>.</p>
    <p>Without it, the gradient stored after the second pass is the sum of both passes. The next step would then be about twice too large.</p>`)}`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('noise', '6 · The noise that no line removes', `
    <p>Take the run with batches of ${B10(T)} after ${T(`${EQ}.results.${RES10}.steps`, 'setup')} steps. Its line has w = ${T(`${run(D, 10)}.epochs.30.weight`, 'contrast', { sig: 4 })} and b = ${T(`${run(D, 10)}.epochs.30.bias`, 'contrast', { sig: 4 })}. The true values are ${T(`${F}.setup.true_weight`, 'setup')} and ${T(`${F}.setup.true_bias`, 'setup')}.</p>
    <p>The line misses a sample by about ${T.human(`${run(D, 10)}.epochs.30.loss`, 'rmse', { sig: 1 })}, and so does the true line, because of the noise. So the loss stops falling there. The learned line even fits a little of it.</p>`),
    },
    {
      kind: 'quiz', id: 'one-step', purpose: 'cumulative', from: 'sgd-step', type: 'numeric',
      answer: `${GE}.batches.0.step.w`, tolerance: 0.005,
      prompt: (D, T) => `<p>The line starts at w = ${T(`${GE}.at.w`, 'setup')}. The first batch gives the gradient ${T(`${GE}.batches.0.gw`, 'setup')} for w, and the learning rate is ${T(`${GE}.lr`, 'setup')}. What is w after one step?</p>`,
      reveal: (D, T) => `<p>The step subtracts ${T(`${GE}.lr`, 'reveal')} × (${T(`${GE}.batches.0.gw`, 'reveal')}). So w goes up to ${T(`${GE}.batches.0.step.w`, 'reveal')}.</p>`,
    },
    {
      kind: 'explain', id: 'why-batches',
      prompt: '<p>Explain why a step can use a small batch and not all the samples.</p>',
      model: '<p>A random batch has a gradient close to that of all the samples, so each step still goes roughly downhill, at a much smaller cost.</p>',
    },
    {
      kind: 'answer',
      html: (D, T) => `<p>No. Each step can use a small random batch, whose gradient is a cheap estimate of the gradient of all the samples. At our learning rate, batches of ${B10(T)} and ${B100(T)} learned about equally per step. Per sample, the small batches learned faster, with noisier steps. The loop repeats five lines, from the prediction to clearing the gradients.</p>`,
      next: () => '<p>Can a better rule for the step move faster through a narrow valley?</p>',
    },
  ],
};
