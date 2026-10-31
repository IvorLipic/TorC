// Lesson "losses" (Part 3 `networks`, curriculum v3, template 3), title "Softmax and cross-entropy" (brief §6 and §7): softmax
// turns scores into shares that add to one, with exponentials so that every share is positive; cross-entropy is the surprise at
// the right class, −ln of its share (the exp-ln refresher; "log" means ln on this site); why it beats the squared miss for classes
// (the gradient comparison); ten digits as ten classes, with the uniform-scores check ln 10; a ChatGPT aside (no vocabulary size,
// source in the footer). MSE is taught in what-is-learning, and the MSE shape pitfall lives in the appendix lesson torc-pitfalls
// (decision 22), so neither is on this page. Every number comes from data/losses.json through a role-tagged T()
// (docs/authoring.md "Number roles"). Notes: docs/lessons/losses.md.
import { math } from '../core/katex.js';
import { worked, refresher, chatgpt, term, sym } from '../core/blocks.js';
import { defaultFooter } from '../core/sections.js';
import { mountLossExplorer, mountGradientCompare } from '../widgets/loss-widgets.js';

const F = 'files.losses';
// rows by index (the test pins this order): exp_rows and ce are uniform, mild [2, 1, 0.1], confident [4, 1, −2]
const MILD = 1, CONF = 2;
const ex = (r, rest) => `${F}.exp_rows.${r}.${rest}`;
const sec = (id, title, body) => `<section class="ls" aria-labelledby="ls-${id}"><h2 id="ls-${id}">${title}</h2>${body}</section>`;
const eq = (latex, label) => `<div class="ls-eq" tabindex="0" role="group" aria-label="${label}">${math(latex, true)}</div>`;
const ln = () => sym('ln', 'ln');
const SOURCE = { title: 'Jurafsky and Martin, Speech and Language Processing (3rd edition draft), the chapters on language models', url: 'https://web.stanford.edu/~jurafsky/slp3/' };

const STYLE = `<style>
.ls p{max-width:44rem}
.ls h2{margin-top:2rem}
.ls-eq{overflow-x:auto;margin:.3rem 0}
.ls-eq .katex-display{margin:.4em 0}
</style>`;

const lesson = {
  id: 'losses',
  title: 'Softmax and cross-entropy',
  trace: ['losses.json'],
  template: 3,
  loop: 'loss',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Softmax and cross-entropy</h1>
    <p class="lead">A ${term('model')} gives one score to each ${term('class')}. You turn the scores into ${term('probability', 'probabilities')} and measure how wrong they are.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'How do the scores of three fruits become probabilities?',
        'How wrong is a small probability for the right fruit?',
        'Why not use the squared miss, as for prices?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('fruit', 'Three fruits, three scores', `
    <p>A model says which fruit a photo shows, an apple, a banana or a cherry. Each fruit is a class with its own score.</p>
    <p>For one photo the apple gets the score ${T(ex(MILD, 'logits.0'), 'setup')} and the banana ${T(ex(MILD, 'logits.1'), 'setup')}. The cherry gets ${T(ex(MILD, 'logits.2'), 'setup')}.</p>
    <p>Books call such scores ${term('logit', 'logits')}. A score can be any number, also a negative one.</p>
    <p>In the lesson on the ${term('decision-boundary')} of two classes, the ${term('sigmoid')} of a difference of scores gave a probability. Three classes need a rule that turns scores into shares of 100%, with a larger share for a larger score.</p>`),
    },
    {
      kind: 'quiz', id: 'negative', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>A second photo gets the scores apple ${T(ex(CONF, 'logits.0'), 'setup')} and banana ${T(ex(CONF, 'logits.1'), 'setup')}. The cherry gets ${T(ex(CONF, 'logits.2'), 'setup')}. What share does the rule give the cherry?</p>`,
      options: [
        { html: 'A negative share, because its score is negative.', feedback: 'A share cannot be negative. The rule first makes every score positive.' },
        { html: 'Exactly zero, because a negative score means no chance.', feedback: 'A negative score means a small chance, not none.' },
        { html: 'A small share above zero.', correct: true, feedback: 'Yes. Every class keeps a share above zero.' },
      ],
      reveal: (D, T) => `<p>The cherry gets ${T(ex(CONF, 'shares.2'), 'reveal')}, which is small but above zero. The apple gets ${T(ex(CONF, 'shares.0'), 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('softmax', '1 · Softmax turns scores into shares', `
    <p>First the rule takes e to the power of each score. The result is always positive, also for a negative score, and a larger score gives a larger result.</p>
    <p>Then the rule divides each result by the sum of all of them, so the shares add to one. This rule is ${term('softmax')}.</p>
    ${eq('\\text{share of the apple} = \\frac{e^{s_{\\text{apple}}}}{e^{s_{\\text{apple}}} + e^{s_{\\text{banana}}} + e^{s_{\\text{cherry}}}}', 'Softmax for three classes')}
    ${worked(`<p>e to the power of the three scores gives ${T(ex(MILD, 'exp.0'), 'worked')}, ${T(ex(MILD, 'exp.1'), 'worked')} and ${T(ex(MILD, 'exp.2'), 'worked')}. Their sum is ${T(ex(MILD, 'exp_sum'), 'worked')}.</p>
    <p>The share of the apple is ${T(ex(MILD, 'exp.0'), 'worked')} / ${T(ex(MILD, 'exp_sum'), 'worked')} = <strong>${T(ex(MILD, 'shares.0'), 'worked')}</strong>.</p>`)}
    <p>A share says how sure the model is. It does not say that the model is right.</p>`),
    },
    {
      kind: 'quiz', id: 'cherry-share', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: ex(MILD, 'shares.2'), tolerance: 0.005,
      prompt: '<p>Use the numbers above. What is the share of the cherry?</p>',
      reveal: (D, T) => `${worked(`<p>${T(ex(MILD, 'exp.2'), 'worked')} / ${T(ex(MILD, 'exp_sum'), 'worked')} = <strong>${T(ex(MILD, 'shares.2'), 'worked')}</strong></p>`)}
    <p>The banana gets ${T(ex(MILD, 'shares.1'), 'reveal')}, and the three shares add to one.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('surprise', '2 · Cross-entropy measures the surprise', `
    <p>The photo shows an apple, so the apple is the right class. The model gave it ${T.human(`${F}.ce.${MILD}.cases.0.loss`, 'prob')}. How wrong is that?</p>
    <p>For classes, the ${term('loss')} <code>L</code> is the surprise at the right answer:</p>
    ${eq('L = -\\ln(\\text{share of the right class})', 'Cross-entropy for one photo')}
    <p>${ln()} is the natural logarithm, with base e. On this site, log always means ln. School log has base ten, so use the ln key of a calculator.</p>
    <p>A share of one gives no surprise. As the share comes near zero, the surprise grows without limit. This loss is ${term('cross-entropy', 'cross-entropy')}, and for many photos it is the mean surprise.</p>
    ${worked(`<p>−ln(${T(ex(MILD, 'shares.0'), 'worked')}) = <strong>${T(`${F}.ce.${MILD}.cases.0.loss`, 'worked')}</strong></p>`)}
    ${refresher('exp-ln')}`),
    },
    {
      kind: 'widget', id: 'explorer', className: 'widget', label: 'Interactive: scores, shares and the surprise for one photo',
      mount: (host, D) => mountLossExplorer(host, D),
      tasks: () => [
        { do: 'Keep the scores that are chosen. Make each fruit the right class in turn.',
          see: 'The shares stay the same. The surprise grows as the right class gets a smaller share.' },
        { do: 'Choose the scores that start with four, and make the cherry the right class.',
          see: 'The model is very sure of the apple, so the surprise is large.' },
        { do: 'Choose the two “same apple share” rows.',
          see: 'The other two fruits split the rest in different ways, but the surprise is the same.' },
      ],
    },
    {
      kind: 'text',
      html: () => sec('ignores', '3 · What the surprise looks at', `
    <p>Many people think that the loss cares how the wrong classes share the rest. The two “same apple share” rows show that it does not. Only the share of the right class counts.</p>
    <p>So ${term('training')} raises the share of the right class.</p>`),
    },
    {
      kind: 'quiz', id: 'very-wrong', purpose: 'predict', type: 'choice',
      prompt: () => `<p>A model with two classes gives the right class almost no share. Which loss gives the larger ${term('gradient')} for the score of the right class? The second loss compares the shares with the ${term('target')} shares, one for the right class and zero for the other.</p>`,
      options: [
        { html: 'The squared miss, because the miss is as large as it can be.', feedback: 'The miss is large, but the share hardly moves there. So the gradient of the squared miss is close to zero.' },
        { html: 'The surprise, because it grows without limit as the share comes near zero.', correct: true, feedback: 'Yes. So its gradient stays large.' },
        { html: 'Both give the same gradient, because they measure the same mistake.', feedback: 'They react in different ways to a very wrong model.' },
      ],
      reveal: (D, T) => `<p>At the score ${T(`${F}.curves.z.0`, 'reveal')}, the surprise has the gradient ${T(`${F}.curves.ce_grad.0`, 'reveal')}. The squared miss has only ${T(`${F}.curves.sq_grad.0`, 'reveal')}.</p>`,
    },
    {
      kind: 'widget', id: 'compare', className: 'widget', label: 'Interactive: the gradient of two losses for the score of the right class',
      mount: (host, D) => mountGradientCompare(host, D),
      tasks: (D, T) => [{
        do: `Move the score of the right class from ${T(`${F}.curves.z.0`, 'setup')} to ${T(`${F}.curves.z.48`, 'setup')}.`,
        see: 'Where the model is very wrong, the cross-entropy gradient is large. The squared miss gradient is near zero there.',
      }],
    },
    {
      kind: 'text',
      html: () => sec('why', '4 · Why the surprise beats the squared miss', `
    <p>With the squared miss, a very wrong model gets almost no gradient, so it learns slowest where it is most wrong.</p>
    <p>The surprise has no flat part there. For the score of the right class, its gradient is the share minus one. When the share is near zero, the gradient is near minus one, so each step pushes hard.</p>
    <p>For numbers such as prices, the squared miss stays the right loss. For classes, you use cross-entropy.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('digits', '5 · Ten digits are ten classes', `
    <p>A network that reads handwritten digits gives ${T.anchor('classes')} scores, one for each digit. Softmax turns them into shares, and the loss is the surprise at the right digit.</p>
    ${chatgpt('ChatGPT picks its next word piece in the same way. It scores many thousands of word pieces, and softmax turns the scores into shares. Its training loss is the surprise at the piece that really came next. This is a concept, not a TorC run.')}`),
    },
    {
      kind: 'quiz', id: 'ten-equal', purpose: 'transfer', type: 'numeric',
      answer: `${F}.ten.loss`, tolerance: 0.01,
      prompt: '<p>A digit network has not learned anything yet. It gives all ten digits the same score. What is its surprise, whatever the right digit is? Use ln.</p>',
      reveal: (D, T) => `<p>Each digit gets a share of ${T(`${F}.ten.softmax.0`, 'reveal')}. The surprise is −ln(${T(`${F}.ten.softmax.0`, 'reveal')}) = ${T(`${F}.ten.loss`, 'reveal')}.</p>
    <p>Did you get a number near one? That is school log. Use ln.</p>`,
    },
    {
      kind: 'quiz', id: 'boundary', purpose: 'cumulative', from: 'classification', type: 'choice',
      prompt: () => `<p>In the lesson on two classes, a point lies on the ${term('decision-boundary')}. What probability of orange does the model give it?</p>`,
      options: [
        { html: 'One half, because the two scores are equal there.', correct: true, feedback: 'Yes. The sigmoid of zero is one half.' },
        { html: 'Zero, because the point belongs to no colour.', feedback: 'Neither colour wins there.' },
        { html: 'It depends on how far the point is from the disc.', feedback: 'The two scores are equal along the whole boundary.' },
      ],
      reveal: '<p>Softmax of two equal scores also gives each class one half.</p>',
    },
    {
      kind: 'explain', id: 'what-ce',
      prompt: '<p>Explain what cross-entropy measures.</p>',
      model: '<p>Cross-entropy is the surprise at the right answer, minus the ln of the share that the model gives the right class.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>Softmax takes e to the power of each score and divides by the sum. So every class gets a positive share, and the shares add to one.</p>
    <p>Cross-entropy is the surprise at the right class, minus the ln of its share. Its gradient stays large when the model is very wrong, so it beats the squared miss.</p>`,
      next: () => `<p>A network has thousands of ${term('parameter', 'parameters')}. How does TorC find the gradient for each of them at once?</p>`,
    },
  ],
};
// the default footer, plus the source of the ChatGPT aside (decision 18)
lesson.footer = (D) => `${defaultFooter(lesson, D)}<p>The note on ChatGPT uses this source. <a href="${SOURCE.url}">${SOURCE.title}</a></p>`;
export default lesson;
