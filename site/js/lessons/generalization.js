// Lesson "generalization" (part 4, training; brief §6 Part 3, §7, C2, C5, C10, D11). A polynomial of degree 15 learns from ten noisy
// points of a wave (data/generalization.json, tools/trace/trace_generalization.cpp). The training, validation and test sets come
// first, then the validation-stop game (widgets/val-stop.js): the learner picks the stopping step from the validation curve,
// commits, and only then sees the test miss. Then overfitting as a matter of degree (the ratio of the two typical misses), weight
// decay and underfitting, and more data (widgets/generalization-widgets.js), which refutes "more parameters always overfit".
// Every number comes from the trace through T(); the claims are tested in site/test/generalization.test.mjs. Notes:
// docs/lessons/generalization.md. Rules: docs/authoring.md, docs/v3-implementation.md.
import { math } from '../core/katex.js';
import { worked, analogy, inTorc, term } from '../core/blocks.js';
import { mountValStop } from '../widgets/val-stop.js';
import { mountRemedies } from '../widgets/generalization-widgets.js';
import { cf } from '../widgets/cpp-literal.js';

const F = 'files.generalization';
export const RUN = { none: `${F}.runs.0`, mid: `${F}.runs.1`, high: `${F}.runs.2` };
export const MISS = `${F}.derived.misses.0`;            // typical misses of the run without weight decay (stop and last step)
const WAVE = `${F}.derived.true_wave_loss.val`;          // the mean squared miss of the wave itself on the validation set
const S = `${F}.settings`;

// The code of the experiment, as tools/trace/trace_generalization.cpp runs it (D11). The sample counts, the batch size, the
// learning rate and the weight decays come from the trace. The test batch size is not in the JSON, so a test pins it to the source.
export const TEST_BATCH = 256;
export const code = (D) => {
  const G = D.files.generalization, st = G.settings, wd = st.weight_decays.map(cf);
  return `// ${st.train_samples} training samples, all in one batch of up to ${st.batch_size}
data::DataLoader train(train_set, ${st.batch_size}, /*shuffle=*/false);
// ${st.val_samples} validation and ${st.test_samples} test samples, never trained on
data::DataLoader val(val_set, ${st.batch_size}), test(test_set, ${TEST_BATCH});
// the other two runs use a weight decay of ${wd[0]} and ${wd[2]}
optim::AdamW opt(params, ${cf(st.lr)}, /*weight_decay=*/${wd[1]});`;
};
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const sec = (id, title, body) => `<section class="gn" aria-labelledby="gn-${id}"><h2 id="gn-${id}">${title}</h2>${body}</section>`;
const eq = (latex, label) => `<div class="gn-eq" tabindex="0" role="group" aria-label="${label}">${math(latex)}</div>`;
export const lastIdx = (D) => D.files.generalization.runs[0].checkpoints.length - 1;
const poly = (n) => `\\text{prediction} = w_1 x + w_2 x^2 + \\dots + w_{${n}} x^{${n}} + b`;

const STYLE = `<style>
.gn p,.gn li{max-width:44rem}
.gn h2{margin-top:2rem}
.gn-eq{overflow-x:auto;margin:.2rem 0}
.gn-eq .katex-display{margin:.4em 0}
.gn-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.gn-code code{background:none;padding:0}
</style>`;

export default {
  id: 'generalization',
  title: 'Does it work on new data?',
  trace: ['generalization.json'],
  template: 3,
  loop: 'data',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Does it work on new data?</h1>
    <p class="lead">Here you find out how to check a curve on new points, and how to measure it fairly.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'A curve passes almost exactly through ten points. How well does it predict a new point?',
        'How can you tell whether a curve is good for new points?',
        'You chose with some points. Can they still tell you how good the result is?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('setup', 'Ten points near a wave', `
    <p>The points lie near a wave, which the game below draws as a dashed curve. Each ${term('target')} has noise, a small random number added to it. The ${term('model')} is a polynomial of degree ${T(`${S}.degree`, 'setup')}:</p>
    ${eq(poly(D.files.generalization.settings.degree), 'The model: a polynomial of degree 15 in x, with one weight per power and a bias b')}
    <p>The small number under each <code>w</code> names its power of <code>x</code>. With these ${term('weight', 'weights')} and the ${term('bias')} <code>b</code>, the model has ${T(`${S}.parameters`, 'setup')} ${term('parameter', 'parameters')}. That is more than its ${T(`${S}.train_samples`, 'setup')} points to learn from.</p>
    <p>${term('training', 'Training')} starts with every parameter at zero. It uses AdamW, the ${term('optimizer')} of the last lesson, with no ${term('weight-decay', 'weight decay')}.</p>`),
    },
    {
      kind: 'quiz', id: 'turn', purpose: 'predict', type: 'choice',
      prompt: () => '<p>Training runs for many steps, and the miss on the training points falls almost to zero. What happens to the miss on new points near the same wave?</p>',
      options: [
        { html: 'It also gets smaller at every step.', feedback: 'You expect that more training always helps on new points. That holds only at first.' },
        { html: 'Overall it falls at first and then rises.', correct: true, feedback: 'At first the curve learns the wave. Later it also learns the noise of its training points.' },
        { html: 'It stays the same, because training never sees the new points.', feedback: 'The curve changes, and it passes near the new points too.' },
        { html: 'It falls almost to zero too.', feedback: 'Each new point has its own noise, which no curve can predict.' },
      ],
      reveal: () => '<p>Early on, it also rises for a while. In the game below, look for the step where it is lowest.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('sets', '1 · Three sets of points', `
    <ul>
      <li>The ${term('training-set', 'training set')} has the ${T(`${S}.train_samples`, 'setup')} points that training uses.</li>
      <li>The ${term('validation-set', 'validation set')} has ${T(`${S}.val_samples`, 'setup')} other points. You use them to make choices, such as when to stop.</li>
      <li>The ${term('test-set', 'test set')} has ${T(`${S}.test_samples`, 'setup')} more points. You look at them once, at the end, to measure the final model.</li>
    </ul>
    <p>The model does not train on the last two sets. Why keep two? Each choice that you make with some points makes their miss a little too hopeful. The test set stays untouched, so its miss fairly measures ${term('generalization')}, how well the model works on new data.</p>
    <p>Each set gets a typical miss, the square root of its ${term('loss')}.</p>`),
    },
    {
      kind: 'widget', id: 'stop', className: 'widget', label: 'Interactive: choose the step where training stops, then see the test miss once',
      mount: (host, D) => mountValStop(host, D),
      tasks: () => [
        {
          do: 'Move the slider slowly from the first step to the last.',
          see: 'The training miss falls almost to zero. The validation miss falls, rises for a while, falls to its lowest value, and rises again.',
        },
        {
          do: 'Find the step with the smallest validation miss, and press “Stop training here”.',
          see: 'The test miss appears once. Is it larger or smaller than the miss that the validation set promised?',
        },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('overfit', '2 · The curve learns the noise', `
    <p>At its lowest point, the model misses a validation point by about ${T.human(`${RUN.none}.val_stop.val_loss`, 'rmse')}. After that step, the curve bends toward each training point. Near the right end, it swings far above the wave.</p>
    <p>The curve then follows the noise of the ten points instead of the wave, so it gets worse on new points. This is ${term('overfitting')}.</p>
    ${analogy('A student who memorizes last year’s exam answers does well on that exam and badly on a new one.',
    'The model does not choose to memorize. It fits the noise because nothing stops it, and it can do this a little or a lot.')}`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('degree', '3 · How much a model overfits', `
    <p>A model overfits more when its training miss is much smaller than its validation miss. So divide the two.</p>
    ${worked(`<p>At the last step, the training miss is about ${T(`${MISS}.last.train`, { role: 'worked', sig: 2 })}, and the validation miss is about ${T(`${MISS}.last.val`, { role: 'worked', sig: 2 })}.</p>
    <p>${T(`${MISS}.last.val`, { role: 'worked', sig: 2 })} ÷ ${T(`${MISS}.last.train`, { role: 'worked', sig: 2 })} ≈ ${T(`${MISS}.last.ratio`, { role: 'worked', sig: 2 })}</p>`)}`),
    },
    {
      kind: 'quiz', id: 'ratio', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${MISS}.stop.ratio`, tolerance: 0.15,
      prompt: (D, T) => `<p>At the lowest validation miss, after step ${T(`${RUN.none}.val_stop.step`, 'setup')}, the training miss is about ${T(`${MISS}.stop.train`, { role: 'setup', sig: 2 })}. The validation miss is about ${T(`${MISS}.stop.val`, { role: 'setup', sig: 2 })}. How many times as large is the validation miss? A calculator is fine.</p>`,
      reveal: (D, T) => `<p>It is about ${T(`${MISS}.stop.ratio`, { role: 'reveal', sig: 2 })} times as large. So this step overfits much less than the last step. Overfitting is not a yes or no matter.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('decay', '4 · Weight decay keeps the curve calm', `
    <p>Without weight decay, the weights grow, and the curve swings. In the widget, the size of the weights ends near ${T(`${RUN.none}.checkpoints.${lastIdx(D)}.weight_sum_sq`, 'contrast')}, against ${T(`${RUN.mid}.checkpoints.${lastIdx(D)}.weight_sum_sq`, 'contrast')} with weight decay. Weight decay, the leak of the last lesson, keeps the weights small.</p>
    <p>Too much weight decay holds the curve too close to zero. It still looks like the wave but sags, and it misses even its training points by a lot. This is ${term('underfitting')}. Both misses stay high.</p>
    <p>The factor of the two misses compares the steps of one run. To compare runs, look at the validation miss itself.</p>`),
    },
    {
      kind: 'widget', id: 'remedies', className: 'widget', label: 'Interactive: the same model with weight decay or with more training points',
      mount: (host, D) => mountRemedies(host, D),
      tasks: (D, T) => [
        {
          do: `Choose weight decay ${T(`${RUN.mid}.weight_decay`, 'setup')} and go to the last step.`,
          see: 'The curve stays smooth, the weights stay small, and the validation miss does not climb.',
        },
        {
          do: `Choose weight decay ${T(`${RUN.high}.weight_decay`, 'setup')}.`,
          see: 'The curve sags below the wave and misses many training points. Both misses stay high.',
        },
        {
          do: `Choose ${T(`${F}.more_data.train_samples`, 'setup')} training points.`,
          see: 'The same polynomial now follows the wave closely.',
        },
      ],
    },
    {
      kind: 'text',
      html: (D) => `<section class="gn">${inTorc('the polynomial and the three sets', `<p>The program of this page gives the powers of <code>x</code> to <code>${esc(D.files.generalization.settings.model)}</code>, a ${term('layer')} with one weight for each power and a bias. This code makes the sets and the optimizer:</p>
    <pre class="gn-code" tabindex="0" role="group" aria-label="The data loaders and the optimizer of this lesson in TorC"><code>${esc(code(D))}</code></pre>`)}</section>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('count', '5 · More parameters do not always overfit', `
    <p>Many people believe that many parameters always mean overfitting. The widget shows that the count alone does not decide it. The same ${T(`${S}.parameters`, 'setup')} parameters overfit strongly on ten points, but not on a hundred.</p>
    <p>At the last step, the validation miss is about ${T.human(`${F}.more_data.checkpoints.${lastIdx(D)}.val_loss`, 'rmse')} with a hundred points, against ${T.human(`${RUN.none}.checkpoints.${lastIdx(D)}.val_loss`, 'rmse')} with ten.</p>
    <p>The digit network of the next lesson has ${T.anchor('mlp_parameters')} parameters, but it also learns from ${T.anchor('train_images')} images. The next lesson checks it on new digits.</p>
    <p>Even the wave itself misses a validation point by about ${T.human(WAVE, 'rmse')}, because nobody can predict the noise of a new point.</p>
    <p>These runs use one draw of random points, so other points give other numbers.</p>`),
    },
    {
      kind: 'quiz', id: 'houses', purpose: 'transfer', type: 'choice',
      prompt: () => '<p>A team tries five values of weight decay for a model of house prices. They keep the best value on the test set and report its test miss. What is wrong?</p>',
      options: [
        { html: 'They chose with the test set, so its miss is too hopeful.', correct: true, feedback: 'Each value was another look at the test set. The best of five looks is too hopeful.' },
        { html: 'Nothing, because the test set is there to choose settings.', feedback: 'You choose settings with the validation set. The test set measures only the final model.' },
        { html: 'They should choose with the training miss.', feedback: 'The model that overfits most has the smallest training miss.' },
        { html: 'Five values are too few.', feedback: 'More values would make it worse, not better.' },
      ],
      reveal: () => '<p>Choose with the validation set. Then measure the chosen model once on the test set.</p>',
    },
    {
      kind: 'quiz', id: 'leak', purpose: 'cumulative', from: 'optimizers', type: 'numeric',
      answer: `${F}.derived.decay_factor.2`, tolerance: 0.002,
      prompt: (D, T) => `<p>These runs use AdamW with the ${term('learning-rate', 'learning rate')} ${T(`${S}.lr`, 'setup')}. With weight decay ${T(`${RUN.high}.weight_decay`, 'setup')}, what part of each parameter does the leak keep at each step? Give a decimal number.</p>`,
      reveal: (D, T) => `<p>The leak takes away the learning rate times the weight decay, here ${T(`${S}.lr`, 'reveal')} × ${T(`${RUN.high}.weight_decay`, 'reveal')}. So it keeps ${T(`${F}.derived.decay_factor.2`, 'reveal')} of each parameter.</p>`,
    },
    {
      kind: 'explain', id: 'why-validation',
      prompt: () => '<p>Explain why you choose the stopping step with the validation set, not the test set.</p>',
      model: () => '<p>Each choice made with a set makes its miss a little too hopeful, and the choices add up. The test set stays untouched, so its one miss stays fair.</p>',
    },
    {
      kind: 'answer',
      html: () => '<p>A curve that fits its training points almost exactly can miss new points by a lot, because it follows their noise. To tell if a curve is good, measure its miss on points that it did not train on. Points that helped you to choose are no longer a fair check. So choose with the validation set, and measure the final model once on the test set.</p>',
      next: (D, T) => `<p>Does this also work for a network with ${T.anchor('mlp_parameters')} parameters that reads handwritten digits?</p>`,
    },
  ],
};
