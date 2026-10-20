// Lesson "generalization": a test set, overfitting (the training loss falls while the test loss rises), weight decay against it,
// underfitting, and the validation set. Every number comes from data/generalization.json (tools/trace/trace_generalization.cpp)
// through T(). Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { mountGeneralization } from '../widgets/generalization-widgets.js';

const F = 'files.generalization';
// runs in the trace: weight decay 0, 0.1, 1
const RUN = { none: `${F}.runs.0`, mid: `${F}.runs.1`, high: `${F}.runs.2` };
const sec = (id, title, body) => `<section class="gn" aria-labelledby="gn-${id}"><h2 id="gn-${id}">${title}</h2>${body}</section>`;
const lastIdx = (D) => D.files.generalization.runs[0].checkpoints.length - 1;
const bestIdx = (D) => { const cs = D.files.generalization.runs[0].checkpoints; return cs.reduce((b, c, i) => (c.test_loss < cs[b].test_loss ? i : b), 0); };

const STYLE = `<style>
.gn p{max-width:44rem}
.gn h2{margin-top:2rem}
.gn-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.gn-code code{background:none;padding:0}
</style>`;

export default {
  id: 'generalization',
  title: 'Generalization',
  trace: ['generalization.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Generalization</h1>
    <p class="lead">A model learns from its training data, but it must work on new data. This lesson shows a model that fits its training data too closely, and how weight decay helps.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'Keep some samples apart as a test set, and never train on them.',
        'If the training loss falls and the test loss rises, the model learns the noise. This is overfitting.',
        'Weight decay keeps the weights small, and this can reduce overfitting.',
        'Choose settings with a validation set, not with the test set.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('test', '1 · Training data and test data', `
    <p>The goal is a model that works on new data. So keep some samples apart as a test set, and never train on them. The loss on the test set shows how well the model works on new data.</p>
    <pre class="gn-code" tabindex="0" role="group" aria-label="A training set and a test set in TorC"><code>data::DataLoader train(train_set, 64, /*shuffle=*/true);
data::DataLoader test(test_set, 256);    // never trained on
optim::AdamW opt(params, 1e-3f, /*weight_decay=*/0.01f);</code></pre>
    <p>Here, the training set has ${T(`${F}.setup.train_samples`)} samples of a sine curve with noise. The test set has ${T(`${F}.setup.test_samples`)} other samples of the same curve. The model is <code>nn::Linear</code> on the powers of <code>x</code> up to the power ${T(`${F}.setup.degree`)}. So it can make a very flexible curve.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('overfit', '2 · Overfitting', `
    <p>A flexible model can follow the noise in the training samples. Its curve then passes close to every training sample, but it swings between them. The training loss becomes very small, and the test loss becomes larger. The name of this problem is overfitting.</p>`),
    },
    {
      kind: 'predict', id: 'p-test', title: 'Before you try it',
      prompt: () => '<p>Without weight decay, the training loss falls at every step. Does the test loss also fall at every step?</p>',
      reveal: (D, T) => `<p>No. The test loss is lowest after step ${T(`${RUN.none}.checkpoints.${bestIdx(D)}.step`)}, at ${T(`${RUN.none}.checkpoints.${bestIdx(D)}.test_loss`)}. Then it rises to ${T(`${RUN.none}.checkpoints.${lastIdx(D)}.test_loss`)}, while the training loss falls to ${T(`${RUN.none}.checkpoints.${lastIdx(D)}.train_loss`)}.</p>`,
    },
    {
      kind: 'widget', id: 'fit', className: 'widget', label: 'Interactive: the model and its training and test loss for three values of weight decay',
      mount: (host, D) => mountGeneralization(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => { const L = lastIdx(D); return sec('decay', '3 · Weight decay', `
    <p>Weight decay makes every weight a little smaller at each step. A curve that swings between the samples needs large weights, so weight decay works against it. With a weight decay of ${T(`${RUN.mid}.weight_decay`)}, the sum of the squared weights ends at ${T(`${RUN.mid}.checkpoints.${L}.weight_sum_sq`)}, against ${T(`${RUN.none}.checkpoints.${L}.weight_sum_sq`)} without it. The test loss ends at ${T(`${RUN.mid}.checkpoints.${L}.test_loss`)} and does not rise.</p>
    <p>Too much weight decay makes the model too simple. With a weight decay of ${T(`${RUN.high}.weight_decay`)}, both losses stay high, at ${T(`${RUN.high}.checkpoints.${L}.train_loss`)} and ${T(`${RUN.high}.checkpoints.${L}.test_loss`)}. The name of this problem is underfitting.</p>`); },
    },
    {
      kind: 'check', id: 'c-floor',
      prompt: (D, T) => `<p>The noise in the targets has a variance of ${T(`${F}.derived.noise_variance`)}. Can a model get a test loss far below this value?</p>`,
      reveal: (D, T) => `<p>No. The noise in each test target is new, so no model can predict it. The best test loss in these runs is ${T(`${RUN.none}.checkpoints.${bestIdx(D)}.test_loss`)}, which is above the noise.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('validation', '4 · The validation set', `
    <p>If you choose the weight decay or the number of steps by the test loss, the test set takes part in training. Then the test loss is too low, and it no longer shows how the model works on new data. So take a third set, the validation set, from the training data. Use it to choose such settings. Use the test set only once, at the end.</p>`),
    },
    {
      kind: 'check', id: 'c-diagnose',
      prompt: (D, T) => { const L = lastIdx(D); return `<p>A model has a training loss of ${T(`${RUN.none}.checkpoints.${L}.train_loss`)} and a test loss of ${T(`${RUN.none}.checkpoints.${L}.test_loss`)}. What is the problem, and what can help?</p>`; },
      reveal: () => '<p>The model overfits. Weight decay can help. More training data and fewer steps can also help. Choose the weight decay and the number of steps with the validation loss.</p>',
    },
    {
      kind: 'text',
      html: () => '<section class="gn"><p>The next lesson, <a href="../mnist-mlp/">Digits with an MLP</a>, trains a network on handwritten digits from start to end.</p></section>',
    },
  ],
};
