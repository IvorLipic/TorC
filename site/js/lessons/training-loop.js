// Lesson "training-loop": datasets and data loaders, the five calls of one training step, epochs and reset(), the batch size, and
// the loss that the noise in the data leaves. Every number and message comes from data/training.json
// (tools/trace/trace_training.cpp) through T() or the recorded text. Writing: docs/authoring.md "Writing rules" (ASD-STE100,
// STE-flavored mode).
import { escapeHtml } from '../core/dom.js';
import { mountBatchViewer, mountTrainingCurves } from '../widgets/training-widgets.js';

const F = 'files.training';
// runs in the trace: batch size 100, 32, 10
const RUN = { b100: `${F}.runs.0`, b32: `${F}.runs.1`, b10: `${F}.runs.2` };
const sec = (id, title, body) => `<section class="tr" aria-labelledby="tr-${id}"><h2 id="tr-${id}">${title}</h2>${body}</section>`;
const sizes = (D, bs) => D.files.training.loaders.find((l) => l.batch_size === bs && !l.shuffle).epochs[0].map((b) => b.length);
const lastEpoch = (D) => D.files.training.setup.epochs;

const STYLE = `<style>
.tr p{max-width:44rem}
.tr h2{margin-top:2rem}
.tr-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.tr-code code{background:none;padding:0}
</style>`;

export default {
  id: 'training-loop',
  title: 'The training loop',
  trace: ['training.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>The training loop</h1>
    <p class="lead">Training repeats one short loop many times. This lesson shows the loop, and the data loader that gives the loop its data.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'A data loader splits a dataset into batches. A batch is a small group of samples.',
        'One step trains on one batch. One epoch is one pass over all the batches.',
        'Each step runs the model, computes the loss, calls <code>backward()</code>, updates the parameters and clears the gradients.',
        'Call <code>loader.reset()</code> at the start of each epoch.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => { const s = D.files.training.setup; return sec('data', '1 · Datasets and batches', `
    <p>A dataset holds the samples. Each sample is an input <code>x</code> and a target <code>t</code>. Here, <code>data::SyntheticRegression</code> makes ${T(`${F}.setup.samples`)} samples of the line <code>t = ${s.true_weight}x + ${s.true_bias}</code>, and it adds noise to each target.</p>
    <pre class="tr-code" tabindex="0" role="group" aria-label="A dataset and a data loader in TorC"><code>data::SyntheticRegression dataset(${s.samples}, 1, ${s.true_weight}.0f, ${s.true_bias}.0f, ${s.noise_std}f);
data::DataLoader loader(dataset, 32, /*shuffle=*/true);
while (loader.has_next()) {
    auto [x, t] = loader.next_batch();   // one batch
}
loader.reset();                          // the next epoch</code></pre>
    <p>A data loader gives out the samples in batches. A batch is a small group of samples. <code>next_batch()</code> returns the next batch, and <code>has_next()</code> tells you if the loader has another batch. With a batch size of <code>32</code>, the batches hold <code>${sizes(D, 32).join(', ')}</code> samples. After the last batch, <code>next_batch()</code> throws this error.</p>
    <p><code>${escapeHtml(D.files.training.exhausted.exception)}: ${escapeHtml(D.files.training.exhausted.what)}</code></p>`); },
    },
    {
      kind: 'widget', id: 'batches', className: 'widget', label: 'Interactive: call next_batch() and reset() on a data loader over the samples',
      mount: (host, D) => mountBatchViewer(host, D),
    },
    {
      kind: 'text',
      html: () => sec('step', '2 · One training step', `
    <p>Training runs this loop. Each pass through the inner loop is one step, and it uses one batch. One pass over all the batches is one epoch.</p>
    <pre class="tr-code" tabindex="0" role="group" aria-label="The training loop in TorC"><code>for (int epoch = 0; epoch &lt; epochs; ++epoch) {
    loader.reset();
    while (loader.has_next()) {
        auto [xb, tb] = loader.next_batch();
        Variable x(xb, false), t(tb, false);
        Variable y = model(x);            // 1. forward pass
        Variable loss = loss_fn(y, t);    // 2. loss
        loss.backward();                  // 3. gradients
        opt.step();                       // 4. update
        opt.zero_grad();                  // 5. clear the gradients
    }
}</code></pre>
    <p>The forward pass computes the outputs, and the loss compares them with the targets. <code>backward()</code> computes the gradients. <code>step()</code> changes the parameters, and <code>zero_grad()</code> sets the gradients back to zero. With <code>shuffle</code> on, each <code>reset()</code> also puts the samples in a new random order.</p>`),
    },
    {
      kind: 'check', id: 'c-reset',
      prompt: () => '<p>You remove the line <code>loader.reset()</code>. How many steps does the model take in the second epoch?</p>',
      reveal: (D, T) => `<p>It takes ${T(`${F}.no_reset.1.steps`)} steps. After the first epoch, <code>has_next()</code> returns <code>false</code>, so the inner loop does not run. TorC gives no error, and the loss stays at ${T(`${F}.no_reset.1.loss`)}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('batch', '3 · Batch size', `
    <p>Three runs train the same model on the same data, from the same start. Each run uses the learning rate ${T(`${F}.setup.lr`)} for ${T(`${F}.setup.epochs`)} epochs. Only the batch size is different. A small batch gives more steps in each epoch. But each step uses fewer samples, so its gradient is less exact.</p>`),
    },
    {
      kind: 'predict', id: 'p-batch', title: 'Before you try it',
      prompt: () => '<p>After the same number of epochs, which run has the lower loss, the run with batch size <code>100</code> or the run with batch size <code>10</code>?</p>',
      reveal: (D, T) => { const E = lastEpoch(D); return `<p>The run with batch size <code>10</code>. It took ${T(`${RUN.b10}.epochs.${E}.steps`)} steps, and its loss is ${T(`${RUN.b10}.epochs.${E}.loss`)}. The run with batch size <code>100</code> took only ${T(`${RUN.b100}.epochs.${E}.steps`)} steps, and its loss is ${T(`${RUN.b100}.epochs.${E}.loss`)}.</p>`; },
    },
    {
      kind: 'widget', id: 'curves', className: 'widget', label: 'Interactive: the loss after each epoch for three batch sizes',
      mount: (host, D) => mountTrainingCurves(host, D),
    },
    {
      kind: 'check', id: 'c-steps',
      prompt: () => '<p>A dataset has <code>100</code> samples, and the batch size is <code>32</code>. How many steps does one epoch take?</p>',
      reveal: (D, T) => `<p>It takes ${T(`${RUN.b32}.epochs.1.steps`)} steps. The last batch holds only <code>${sizes(D, 32).at(-1)}</code> samples.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => { const E = lastEpoch(D); return sec('noise', '4 · The noise in the data', `
    <p>The loss does not go to zero. The noise in the targets has a standard deviation of ${T(`${F}.setup.noise_std`)}, and no model can predict it. Even the true line has a loss of ${T(`${F}.true_line_loss`)} on these samples. After ${T(`${F}.setup.epochs`)} epochs with batch size <code>10</code>, the model has the weight ${T(`${RUN.b10}.epochs.${E}.weight`)} and the bias ${T(`${RUN.b10}.epochs.${E}.bias`)}. These values are close to the true line, so the model learned what it can learn.</p>`); },
    },
    {
      kind: 'text',
      html: () => '<section class="tr"><p>The next lesson, <a href="../optimizers/">Better optimizers</a>, compares plain gradient descent with momentum, Adam and AdamW.</p></section>',
    },
  ],
};
