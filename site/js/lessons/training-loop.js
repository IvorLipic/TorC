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
  loader.reset();                // a new order
  while (loader.has_next()) {
    auto [xb, tb] = loader.next_batch();
    Variable x(xb, false), t(tb, false);
    Variable y = model(x);       // 1. predict
    Variable L = loss_fn(y, t);  // 2. the loss
    L.backward();                // 3. gradients
    opt.step();                  // 4. the step
    opt.zero_grad();             // 5. clear
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
    <p>We made ${N(T)} samples with a short program. Each one has an input x and a ${term('target')} t = ${T(`${F}.setup.true_weight`, 'setup')}x + ${T(`${F}.setup.true_bias`, 'setup')}, plus a small random error, the noise.</p>
    <p>The ${term('model')} is a line, y = w·x + b, as for the taxi rides. Its ${term('weight')} w and its ${term('bias')} b are the ${term('parameter', 'parameters')}, and both start at ${T(`${GE}.at.w`, 'setup')}. The ${term('loss')} L is the ${term('mean-squared-error', 'mean squared error')} of the ${term('prediction', 'predictions')} y.</p>
    <p>At the start, the typical miss, the square root of the loss, is about ${rmse(T, `${BC}.runs.0.loss.0`)}.</p>`),
    },
    {
      kind: 'quiz', id: 'estimate', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>Take ${B10(T)} of the ${N(T)} samples at random, and compute the ${term('gradient')} for w from them. How does it compare with the gradient from all ${N(T)}?</p>`,
      options: [
        { html: 'It is the same number.', feedback: 'Another random choice gives other samples, so the number changes.' },
        { html: 'It is close, but a little different for each random choice.', correct: true, feedback: 'A few random samples give a rough picture of all of them.' },
        { html: 'It is about ten times smaller, because it uses ten times fewer samples.', feedback: 'The loss is a mean, so fewer samples do not shrink it.' },
      ],
      reveal: (D, T) => `<p>For all ${N(T, 'reveal')} samples, the gradient for w is ${T(`${GE}.full.gw`, 'reveal')}. Ten random groups gave values from ${T(`${GE}.range.gw_min`, 'reveal')} to ${T(`${GE}.range.gw_max`, 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('batches', '1 · Batches and epochs', `
    <p>A ${term('batch')} is a small group of samples that one step uses. Its size is the batch size B. TorC's data loader hands out the batches.</p>
    <p>One pass through all the samples is an ${term('epoch')}, so an epoch has many steps:</p>
    ${worked(`<p>${T(`${F}.setup.samples`, 'worked')} samples ÷ ${T(`${loader(D, 10)}.batch_size`, 'worked')} samples per batch = ${T(`${run(D, 10)}.epochs.1.steps`, 'worked')} steps in one epoch</p>`)}
    <p>Before each epoch, the loader can shuffle, which puts the samples in a new random order.</p>`),
    },
    {
      kind: 'widget', id: 'batches', className: 'widget', label: 'Interactive: a data loader hands out batches of the samples',
      mount: (host, D) => mountBatchViewer(host, D),
      tasks: (D, T) => [
        { do: `With batch size ${B10(T)}, press Next batch until the epoch is over.`, see: 'Each press is one step. One epoch takes many steps.' },
      ],
    },
    {
      kind: 'quiz', id: 'steps-32', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: STEPS_32, tolerance: 0.5,
      prompt: (D, T) => `<p>The loader has ${N(T)} samples and a batch size of ${T(`${loader(D, 32)}.batch_size`, 'setup')}. How many steps does one epoch take?</p>`,
      reveal: (D, T) => `<p>Three batches of ${T(`${loader(D, 32)}.batch_size`, 'reveal')} leave a few samples for a smaller last batch. So one epoch takes ${T(`${run(D, 32)}.epochs.1.steps`, 'reveal')} steps.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('estimate', '2 · A batch gives an estimate of the gradient', `
    <p>Each step of ${term('gradient-descent', 'gradient descent')} uses the gradient of one batch. It is an estimate of the gradient of all the samples, so it points roughly downhill, but not exactly.</p>
    <p>Over one epoch, the batch gradients average out to the gradient of all the samples, because the batches are equally large. And a batch costs much less to compute.</p>
    <p>This method is ${term('stochastic-gradient-descent', 'stochastic gradient descent')}, or SGD, where stochastic means random.</p>`),
    },
    {
      kind: 'widget', id: 'scatter', className: 'widget', label: 'Interactive: the gradient of each batch next to the gradient of all samples',
      mount: (host, D) => mountGradientEstimate(host, D),
      tasks: () => [
        { do: 'Press Next batch until all ten dots show.', see: 'The dots scatter around the star, and none sits on it.' },
        { do: 'Press “Show the average of the dots”.', see: 'The ring of the average sits on the star.' },
      ],
    },
    {
      kind: 'text',
      html: () => sec('shuffle', '3 · Why the loader shuffles', `
    <p>Why shuffle? Suppose that the samples are sorted by x, with no shuffle. The first batches then hold only the leftmost points, and each step pulls the line toward them. So the miss on all the samples grows at first.</p>
    <p>A random order makes each batch a fair mix of the samples. A new order in each epoch gives new batches, so their errors do not repeat.</p>`),
    },
    {
      kind: 'quiz', id: 'batch-size', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>One run uses batches of ${B10(T)}. Another uses batches of ${B100(T)}, all the samples. Both start from the same line. After ${T(`${ES}.steps`, 'setup')} steps each, which line misses less?</p>`,
      options: [
        { html: (D, T) => `The run with batches of ${B100(T)}, by far, because its steps used more samples.`, feedback: 'Its gradient is more exact, but a batch of ten points almost the same way.' },
        { html: (D, T) => `The run with batches of ${B10(T)}, by far, because small batches learn better.`, feedback: 'Small batches are not better per step.' },
        { html: 'Both lines miss by about the same amount.', correct: true, feedback: 'Per step, both runs make about the same progress here.' },
      ],
      reveal: (D, T) => `<p>In three random orders, the small batches miss by about ${rmse(T, `${ES}.summary.${SUM10}.min`)} to ${rmse(T, `${ES}.summary.${SUM10}.max`)}. The large batches miss by about ${rmse(T, `${ES}.summary.${SUM100}.min`)}, but they read ten times more samples.</p>`,
    },
    {
      kind: 'widget', id: 'compare', className: 'widget', label: 'Interactive: the typical miss of two batch sizes, by steps and by samples seen',
      mount: (host, D) => mountBatchCompare(host, D),
      tasks: (D, T) => [
        { do: `Choose “Same number of steps” and move the marker to ${T(`${ES}.steps`, 'setup')} steps.`, see: 'The two curves almost cover each other.' },
        { do: `Choose “Same number of samples” and move the marker to ${T(`${EQ}.samples`, 'setup')} samples.`, see: `Batch ${B10(T)} took ten times more steps and misses far less.` },
        { do: 'Choose “Sorted or shuffled” and look at the first steps.', see: 'In sorted order, the miss grows at first. In shuffled order, it falls at once.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('fair', '4 · A fair comparison of batch sizes', `
    <p>After equal epochs, the small batches win only because they take ten times more steps. After equal steps, both missed by about the same amount. After equal samples, the small batches took more steps:</p>
    ${worked(`<p>${T(`${EQ}.samples`, 'worked')} ÷ ${T(`${BC}.batch_sizes.0`, 'worked')} = ${T(`${EQ}.results.${RES10}.steps`, 'worked')} steps, and ${T(`${EQ}.samples`, 'worked')} ÷ ${T(`${BC}.batch_sizes.1`, 'worked')} = ${T(`${EQ}.results.${RES100}.steps`, 'worked')} steps</p>`)}
    <p>Then the small batches miss by about ${rmse(T, `${EQ}.summary.${SUM10}.max`)}, and the large ones by about ${rmse(T, `${EQ}.summary.${SUM100}.max`)}.</p>
    <p>So each size has a cost. A small batch needs more steps for the same samples, and its gradient is noisier. A large batch reads more samples per step. Each random order gives slightly different numbers, so we ran three.</p>`),
    },
    {
      kind: 'quiz', id: 'friend', purpose: 'transfer', type: 'choice',
      prompt: '<p>A friend trains a model twice for the same number of epochs. Ten times smaller batches end lower, so your friend says that small batches are better. What do you answer?</p>',
      options: [
        { html: 'The small batches took ten times more steps. Compare at equal steps, too.', correct: true, feedback: 'Equal epochs give the small batches more steps.' },
        { html: 'Your friend is right, because a small batch gives a more exact gradient.', feedback: 'A small batch gives the less exact gradient.' },
        { html: 'Your friend is right, because the loss is lower.', feedback: 'The extra steps alone can explain the lower loss.' },
      ],
      reveal: '<p>Compare at equal steps and at equal samples.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('loop', '5 · The loop in TorC', `
    <p>Every model in this course trains with a loop like this:</p>
    ${realLine(LOOP, 'Each pass of the inner loop is one step with one batch. The numbered lines are the model, the loss and the adjust step.')}
    ${inTorc('Why the loop clears the gradients', `<p>TorC adds every new gradient to the stored one. Inside one backward pass, that sums the gradients of a value with several uses. So between steps, the loop must clear them with <code>opt.zero_grad()</code>.</p>
    ${worked(`<p>At the start, <code>L.backward()</code> stores ${T(`${AC}.first.gw`, 'worked')} for w. After one step, a new forward pass and <code>L.backward()</code> give ${T(`${AC}.second_with_zero_grad.gw`, 'worked')}. Without <code>zero_grad()</code>, TorC stores ${T(`${AC}.first.gw`, 'worked')} + (${T(`${AC}.second_with_zero_grad.gw`, 'worked')}) = ${T(`${AC}.second_without_zero_grad.gw`, 'worked')}.</p>`)}
    <p>The next step would use this sum, which is about twice too long.</p>`)}`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('noise', '6 · The noise that no line removes', `
    <p>Take the run with batches of ${B10(T)} after ${T(`${EQ}.results.${RES10}.steps`, 'setup')} steps. Its line has w = ${T(`${run(D, 10)}.epochs.30.weight`, 'contrast', { sig: 4 })} and b = ${T(`${run(D, 10)}.epochs.30.bias`, 'contrast', { sig: 4 })}. The true values are ${T(`${F}.setup.true_weight`, 'setup')} and ${T(`${F}.setup.true_bias`, 'setup')}.</p>
    <p>Its typical miss is about ${rmse(T, `${run(D, 10)}.epochs.30.loss`)}. The true line itself misses by about ${rmse(T, `${F}.true_line_loss`)}, because of the noise. No line removes that noise, so the loss stops falling there.</p>`),
    },
    {
      kind: 'quiz', id: 'one-step', purpose: 'cumulative', from: 'sgd-step', type: 'numeric',
      answer: `${GE}.batches.0.step.w`, tolerance: 0.005,
      prompt: (D, T) => `<p>The line starts at w = ${T(`${GE}.at.w`, 'setup')}. The first batch gives the gradient ${T(`${GE}.batches.0.gw`, 'setup')} for w, and the ${term('learning-rate', 'learning rate')} is ${T(`${GE}.lr`, 'setup')}. What is w after one step?</p>`,
      reveal: (D, T) => `<p>The step subtracts ${T(`${GE}.lr`, 'reveal')} × (${T(`${GE}.batches.0.gw`, 'reveal')}). So w goes up to ${T(`${GE}.batches.0.step.w`, 'reveal')}.</p>`,
    },
    {
      kind: 'explain', id: 'why-batches',
      prompt: '<p>Explain why a step can use a small batch and not all the samples.</p>',
      model: '<p>The gradient of a random batch is close to that of all the samples. So each step still goes roughly downhill, for much less work.</p>',
    },
    {
      kind: 'answer',
      html: (D, T) => `<p>No. Each step can use a small random batch, whose gradient is a cheap estimate of the gradient of all the samples. Per step, batches of ${B10(T)} and ${B100(T)} learned about equally here. Per sample, the small batches learned faster, with noisier steps. The program repeats five lines that predict, measure the loss, find the gradients, step and clear.</p>`,
      next: () => '<p>Can a better rule for the step move faster through a narrow valley?</p>',
    },
  ],
};
