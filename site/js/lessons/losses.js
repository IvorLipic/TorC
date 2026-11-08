// Lesson "losses" (Part 3 `networks`, curriculum v3, template 3), title "Softmax and cross-entropy" (brief §6 and §7): softmax
// turns scores into probabilities that add to one, with exponentials so that every probability is above zero ("shares of 100 %"
// is the analogy, stated once with its limit); cross-entropy is the surprise at the right class, −ln of its probability (the
// exp-ln refresher; "log" means ln on this site); why it beats the mean squared error for classes (the gradient comparison); ten digits
// as ten classes, with the equal-scores check ln 10; a ChatGPT aside (no vocabulary size, sources in the footer). MSE is taught in
// what-is-learning, and the MSE shape pitfall lives in the appendix lesson torc-pitfalls (decision 22), so neither is on this page.
// Every number comes from data/losses.json through a role-tagged T() (docs/authoring.md "Number roles").
// Notes: docs/lessons/losses.md.
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
// the ChatGPT aside (decision 18): next-token softmax and its log-likelihood loss in pretraining, and the later training stages
export const SOURCES = [
  { title: 'Radford et al. (2018), Improving Language Understanding by Generative Pre-Training, section 3.1 (softmax over the vocabulary, the pretraining loss)', url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf' },
  { title: 'Ouyang et al. (2022), Training language models to follow instructions with human feedback (the later training stages)', url: 'https://arxiv.org/abs/2203.02155' },
];

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
        `Why not use the ${term('mean-squared-error', 'mean squared error')}, as for prices?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('fruit', 'Three fruits, three scores', `
    <p>A model says which fruit a photo shows, an apple, a banana or a cherry. Each fruit is a class with a score.</p>
    <p>For the first photo the apple gets the score ${T(ex(MILD, 'logits.0'), 'setup')} and the banana ${T(ex(MILD, 'logits.1'), 'setup')}. The cherry gets ${T(ex(MILD, 'logits.2'), 'setup')}.</p>
    <p>A score can be any number, even a negative one. Books call such scores ${term('logit', 'logits')}.</p>
    <p>In the lesson on the ${term('decision-boundary')}, the ${term('sigmoid')} of a difference gave a probability. Three classes need a rule that gives larger scores larger probabilities, which add to one.</p>`),
    },
    {
      kind: 'quiz', id: 'negative', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>A second photo gets the scores apple ${T(ex(CONF, 'logits.0'), 'setup')} and banana ${T(ex(CONF, 'logits.1'), 'setup')}. The cherry gets ${T(ex(CONF, 'logits.2'), 'setup')}. What probability does the rule give the cherry?</p>`,
      options: [
        { html: 'A negative probability, because its score is negative.', feedback: 'A probability cannot be negative.' },
        { html: 'Exactly zero, because a negative score means no chance.', feedback: 'A negative score still gets a chance.' },
        { html: 'A small probability above zero.', correct: true, feedback: 'Yes. Every class gets a probability above zero.' },
      ],
      reveal: '<p>You compute it below.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('softmax', '1 · Softmax turns scores into probabilities', `
    <p>First the rule takes e to the power of each score. The result is always positive, and a larger score gives a larger result.</p>
    <p>Then the rule divides each result by the sum of all of them, so the probabilities add to one. This rule is ${term('softmax')}.</p>
    ${eq('\\text{probability of the apple} = \\frac{e^{s_{\\text{apple}}}}{e^{s_{\\text{apple}}} + e^{s_{\\text{banana}}} + e^{s_{\\text{cherry}}}}', 'Softmax for three classes')}
    ${worked(`<p>For the first photo, e to the power of each score gives ${T(ex(MILD, 'exp.0'), 'worked')}, ${T(ex(MILD, 'exp.1'), 'worked')} and ${T(ex(MILD, 'exp.2'), 'worked')}. Their sum is ${T(ex(MILD, 'exp_sum'), 'worked')}.</p>
    <p>The probability of the apple is ${T(ex(MILD, 'exp.0'), 'worked')} / ${T(ex(MILD, 'exp_sum'), 'worked')} = <strong>${T(ex(MILD, 'shares.0'), 'worked')}</strong>.</p>`)}
    <p>Picture the probabilities as shares of 100%. They say how sure the model is. They do not say that it is right.</p>`),
    },
    {
      kind: 'quiz', id: 'cherry-prob', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: ex(CONF, 'shares.2'), tolerance: 0.0002,
      prompt: '<p>Take the second photo. What probability does softmax give the cherry? A calculator helps.</p>',
      reveal: (D, T) => `${worked(`<p>e to the power of each score gives ${T(ex(CONF, 'exp.0'), 'worked')}, ${T(ex(CONF, 'exp.1'), 'worked')} and ${T(ex(CONF, 'exp.2'), 'worked')}. Their sum is ${T(ex(CONF, 'exp_sum'), 'worked')}.</p>
    <p>${T(ex(CONF, 'exp.2'), 'worked')} / ${T(ex(CONF, 'exp_sum'), 'worked')} = <strong>${T(ex(CONF, 'shares.2'), 'worked')}</strong></p>`)}
`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('surprise', '2 · Cross-entropy measures the surprise', `
    <p>The first photo shows an apple, the right class. The model gave it ${T.human(`${F}.ce.${MILD}.cases.0.loss`, 'prob')}. How wrong is that?</p>
    ${refresher('exp-ln')}
    <p>For classes, the ${term('loss')} <code>L</code> of one photo is the surprise at the right answer:</p>
    ${eq('L = -\\ln(\\text{probability of the right class})', 'Cross-entropy for one photo')}
    <p>${ln()} is the natural logarithm. On this site, log always means ln, so use the ln key of a calculator. The ln of a number below one is negative, so the minus sign makes the surprise positive.</p>
    <p>A probability of one gives no surprise. Near zero, the surprise has no limit. This loss is ${term('cross-entropy', 'cross-entropy')}, and for many photos TorC takes the mean.</p>
    ${worked(`<p>−ln(${T(ex(MILD, 'shares.0'), 'worked')}) = <strong>${T(`${F}.ce.${MILD}.cases.0.loss`, 'worked')}</strong></p>`)}`),
    },
    {
      kind: 'widget', id: 'explorer', className: 'widget', label: 'Interactive: scores, probabilities and the surprise for one photo',
      mount: (host, D) => mountLossExplorer(host, D),
      tasks: () => [
        { do: 'Choose the first photo. Make each fruit the right class in turn.',
          see: 'The surprise grows as the right class gets a smaller probability.' },
        { do: 'Choose the second photo, and make the cherry the right class.',
          see: 'The model is very sure of the apple, so the surprise is large.' },
        { do: 'Choose the two “same apple probability” rows.',
          see: 'The other fruits split the rest differently, but the surprise is the same.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('ignores', '3 · What the surprise looks at', `
    <p>Many people think that the loss cares how the wrong classes split the rest. In both “same apple probability” rows the apple gets ${T.human(`${F}.spread.0.loss`, 'prob')}. The surprise is the same in both rows, as the widget shows.</p>
    <p>Only the probability of the right class counts. The wrong scores still change it, because they are part of the sum that divides.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('squared', '4 · The mean squared error for classes', `
    <p>The first lesson used the mean squared error, or MSE. For classes, it compares the probabilities with ${term('target')} probabilities, one for the right class and zero for the others.</p>
    <p>${term('training', 'Training')} needs the ${term('gradient')} of the loss for every number that feeds it, also for a score. It says how much the loss changes per unit change of the score.</p>`),
    },
    {
      kind: 'quiz', id: 'very-wrong', purpose: 'predict', type: 'choice',
      prompt: '<p>A model with two classes gives the right class a probability near zero. Which loss has the larger gradient for the score of the right class, in size?</p>',
      options: [
        { html: 'MSE, because the miss is as large as it can be.', feedback: 'The miss is large, but it hardly changes when the score changes.' },
        { html: 'The surprise, because it falls by about one per unit that the score rises.', correct: true, feedback: 'Yes.' },
        { html: 'Both the same, because they measure the same mistake.', feedback: 'They react in different ways to a very wrong model.' },
      ],
      reveal: (D, T) => `<p>Near zero, the probability hardly moves when the score changes. So the gradient of MSE almost vanishes, while the gradient of the surprise stays near minus one.</p>
    <p>At the score ${T(`${F}.curves.z.0`, 'reveal')}, MSE has the gradient ${T(`${F}.curves.sq_grad.0`, 'reveal')}. The surprise has ${T(`${F}.curves.ce_grad.0`, 'reveal')}, which is further from zero. Both are negative, because a higher score lowers both losses.</p>`,
    },
    {
      kind: 'widget', id: 'compare', className: 'widget', label: 'Interactive: the gradient of two losses for the score of the right class',
      mount: (host, D) => mountGradientCompare(host, D),
      tasks: (D, T) => [{
        do: `Move the score of the right class from ${T(`${F}.curves.z.0`, 'setup')} to ${T(`${F}.curves.z.48`, 'setup')}. The other score stays zero.`,
        see: 'Where the model is very wrong, the cross-entropy gradient stays near minus one. The MSE gradient is near zero there.',
      }],
    },
    {
      kind: 'text',
      html: () => sec('why', '5 · Why the surprise beats MSE', `
    <p>With MSE, a very wrong model learns slowest.</p>
    <p>For one photo, TorC’s <code>backward()</code> gives the surprise the gradient “probability minus one” for the score of the right class. You can check this in the widget. So a very wrong model gets a strong push.</p>
    <p>For prices, MSE is a good choice. For classes, you use cross-entropy.</p>
    ${chatgpt(`A model like ChatGPT scores every word piece that it knows. Softmax gives the probabilities, and then the model chooses one piece. In the first stage of its training, the loss is the surprise at the piece that really came next. Later training uses other methods. This is a concept, not a TorC run.`)}`),
    },
    {
      kind: 'quiz', id: 'ten-equal', purpose: 'transfer', type: 'numeric',
      answer: `${F}.ten.loss`, tolerance: 0.01,
      prompt: (D, T) => `<p>A network for handwritten digits gives ${T.anchor('classes')} scores, one for each digit. Before it learns, it gives all digits the same score. What is its surprise, whatever the right digit is? Use ln.</p>`,
      reveal: (D, T) => `<p>Each digit gets ${T(`${F}.ten.softmax.0`, 'reveal')}. The surprise is −ln(${T(`${F}.ten.softmax.0`, 'reveal')}) = ${T(`${F}.ten.loss`, 'reveal')}.</p>
    <p>Did you get a number near one? That is school log. Use ln.</p>`,
    },
    {
      kind: 'quiz', id: 'boundary', purpose: 'cumulative', from: 'classification', type: 'choice',
      prompt: '<p>Two classes again. A point lies on the decision boundary. What probability of orange does it get?</p>',
      options: [
        { html: 'One half', correct: true, feedback: 'Yes. The two scores are equal there, and the sigmoid of zero is one half.' },
        { html: 'Zero', feedback: 'On the boundary neither colour wins.' },
        { html: 'One', feedback: 'On the boundary the model cannot decide.' },
      ],
      reveal: '<p>Softmax of two equal scores gives the same.</p>',
    },
    {
      kind: 'explain', id: 'what-ce',
      prompt: '<p>Explain what cross-entropy measures.</p>',
      model: '<p>Cross-entropy measures the surprise at the right answer. It equals −ln of the probability that the model gives the right class.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>Softmax takes e to the power of each score and divides by the sum. Every class gets a probability above zero, and they add to one.</p>
    <p>Cross-entropy is the surprise at the right class, −ln of its probability. Its gradient stays near minus one when the model is very wrong.</p>`,
      next: () => `<p>A network has thousands of ${term('parameter', 'parameters')}. How does TorC find the gradient for each of them at once?</p>`,
    },
  ],
};
// the default footer, plus the sources of the ChatGPT aside (decision 18)
lesson.footer = (D) => `${defaultFooter(lesson, D)}<p>The note on ChatGPT uses these sources.</p><ul class="sources">${SOURCES.map((s) => `<li><a href="${s.url}">${s.title}</a></li>`).join('')}</ul>`;
export default lesson;
