// Lesson "from-digits-to-chatgpt" (Part 6 `next`, curriculum v3, template 3, the last page of the main path), title "From digits to
// ChatGPT" (brief §6 Part 5, decision 18). Content beyond TorC, so `conceptual: true` with `sources` in the footer: text becomes
// tokens and tokens become embeddings; next-token prediction with cross-entropy, where the text gives its own targets (the
// widget makes the training examples of one sentence); attention as a weighted average with weights computed from the tokens,
// looking back only; pretraining runs the loop of this course at a much larger size; fine-tuning and learning from human
// feedback come after; where TorC stops, and a list of resources. No TorC code and no numbers except two TorC anchors (the
// pixels and the parameters of the digit network). The attention figure uses weights chosen by hand and says so.
// Sources were not fetched in the session that wrote the page: docs/lessons/from-digits-to-chatgpt.md ("verify before quoting").
import { term, refresher } from '../core/blocks.js';
import { mountNextToken } from '../widgets/next-token.js';

const sec = (id, title, body) => `<section class="ch" aria-labelledby="ch-${id}"><h2 id="ch-${id}">${title}</h2>${body}</section>`;

export const SOURCES = [
  { title: 'Sennrich, Haddow and Birch (2016), Neural Machine Translation of Rare Words with Subword Units (tokens that are parts of words)', url: 'https://arxiv.org/abs/1508.07909' },
  { title: 'Vaswani et al. (2017), Attention Is All You Need, sections 3.2 and 3.4 (attention as a weighted sum with softmax weights, the mask, learned embeddings, the feed-forward layers with ReLU)', url: 'https://arxiv.org/abs/1706.03762' },
  { title: 'Radford et al. (2018), Improving Language Understanding by Generative Pre-Training, sections 3.1 and 4.1 (pretraining on unlabeled text by next-token prediction, Adam with decoupled weight decay)', url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf' },
  { title: 'Loshchilov and Hutter (2019), Decoupled Weight Decay Regularization (AdamW)', url: 'https://arxiv.org/abs/1711.05101' },
  { title: 'Brown et al. (2020), Language Models are Few-Shot Learners (much larger models trained on much more text)', url: 'https://arxiv.org/abs/2005.14165' },
  { title: 'Ouyang et al. (2022), Training language models to follow instructions with human feedback (fine-tuning on written answers, a reward model from rankings, reinforcement learning)', url: 'https://arxiv.org/abs/2203.02155' },
  { title: 'OpenAI (2022), Introducing ChatGPT (trained with the methods of the paper by Ouyang et al.)', url: 'https://openai.com/index/chatgpt/' },
  { title: '3Blue1Brown, Neural networks (video series)', url: 'https://www.3blue1brown.com/topics/neural-networks' },
  { title: 'Andrej Karpathy, Neural Networks: Zero to Hero (video series)', url: 'https://karpathy.ai/zero-to-hero.html' },
  { title: 'Michael Nielsen, Neural Networks and Deep Learning (online book)', url: 'https://neuralnetworksanddeeplearning.com/' },
  { title: 'Zhang, Lipton, Li and Smola, Dive into Deep Learning (online book)', url: 'https://d2l.ai/' },
];
const src = (start) => SOURCES.find((s) => s.title.startsWith(start)).url;

// The made-up attention weights of the figure: what "it" takes from itself and the tokens before it. Chosen by hand, not from a
// model; they add to one (tested). The tokens after "it" are hidden from it.
export const FIGURE = {
  tokens: ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.'],
  reader: 7,
  weights: [0.04, 0.55, 0.06, 0.03, 0.04, 0.12, 0.08, 0.08],
};
const BAR = 54;   // px of a weight of one
function figure() {
  const cells = FIGURE.tokens.map((t, i) => {
    const w = FIGURE.weights[i];
    const bar = w === undefined ? '<span class="ch-bar none"></span>' : `<span class="ch-bar" style="height:${Math.max(2, Math.round(w * BAR * 1.6))}px"></span>`;
    const cls = i === FIGURE.reader ? ' reader' : w === undefined ? ' later' : '';
    return `<span class="ch-cell${cls}">${bar}<span class="ch-word">${t}</span></span>`;
  }).join('');
  return `<figure class="ch-fig"><div class="ch-row" role="img" aria-label="A made-up example. The token it gives the largest weight to cat, small weights to the other earlier tokens, and no weight to the tokens after it.">${cells}</div>`
    + '<figcaption>A made-up illustration with weights chosen by hand, not by a trained model or by TorC. A taller bar is a larger weight for “it”. The dashed tokens are hidden from “it”.</figcaption></figure>';
}

const STYLE = `<style>
.ch p,.ch li{max-width:44rem}
.ch h2{margin-top:2rem}
.ch ul{padding-left:1.2rem}
.ch li{margin:.3rem 0}
.ch-fig{margin:.8rem 0;max-width:44rem}
.ch-row{display:flex;flex-wrap:wrap;gap:8px 6px;align-items:flex-end;background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.7rem .8rem}
.ch-cell{display:flex;flex-direction:column;align-items:center;gap:4px}
.ch-bar{display:block;width:26px;background:var(--fwd);border-radius:3px 3px 0 0}
.ch-bar.none{height:2px;background:transparent}
.ch-word{padding:.15rem .45rem;border:1px solid var(--border);border-radius:7px}
.ch-cell.reader .ch-word{border-color:var(--ok);font-weight:600}
.ch-cell.later .ch-word{color:var(--muted);border-style:dashed;opacity:.65}
.ch-fig figcaption{color:var(--muted);font-size:.9rem;margin-top:.4rem}
</style>`;

const lesson = {
  id: 'from-digits-to-chatgpt',
  title: 'From digits to ChatGPT',
  trace: [],
  template: 3,
  loop: 'all',
  conceptual: true,
  sources: SOURCES,
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>From digits to ChatGPT</h1>
    <p class="lead">This page follows a ${term('model')} like ChatGPT from text to ${term('training')}. It explains ideas beyond TorC and shows no TorC run.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'ChatGPT writes its answer piece by piece. How does it choose the next piece?',
        'It learns from books and web pages. Who gives it the right answers?',
        'How much of this course does ChatGPT use?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: () => sec('setup', 'A model that writes text', `
    <p>You type a question into ChatGPT, and it writes the answer one small piece at a time.</p>
    <p>Every ${term('neural-network')} in this course learned from ${term('sample', 'samples')} with ${term('target', 'targets')}. Each image of a digit came with its target, the digit that it shows.</p>
    <p>ChatGPT first learns from a very large amount of ordinary text, such as books and web pages. A model that learns to continue text is a <strong>language model</strong>.</p>`),
    },
    {
      kind: 'quiz', id: 'where-target', purpose: 'predict', type: 'choice',
      prompt: () => `<p>Each training example of the language model needs a target, so that the ${term('loss')} can measure the miss. Where do the targets come from?</p>`,
      options: [
        { html: 'People write the right continuation next to each sentence.',
          feedback: 'That is how the digits got their targets. In this stage, nobody writes targets.' },
        { html: 'The text itself gives them. The target is the piece that really comes next.', correct: true,
          feedback: 'Yes. Nobody has to write a target.' },
        { html: 'There is no target. The model reads the text and stores it.',
          feedback: `Without a target there is no loss and no ${term('gradient')}, so the model cannot learn.` },
      ],
      reveal: '<p>The next sections show how one sentence gives many examples.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('tokens', '1 · Text becomes tokens, and tokens become vectors', `
    <p>First, a program called a tokenizer cuts the text into ${term('token', 'tokens')}. A token is often a word, a part of a word or a punctuation mark. All the tokens that the model knows form its <strong>vocabulary</strong>.</p>
    <p>Each token of the vocabulary has its own ${term('vector')}, a list of numbers, called its ${term('embedding')}.</p>
    <p>The numbers of the embeddings are ${term('parameter', 'parameters')}, like the ${term('weight', 'weights')} of a ${term('layer')}. Training changes them together with the other parameters.</p>
    <p>The digit network read ${T.anchor('pixels')} ${term('pixel')} values. A language model reads one embedding for each token.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('next', '2 · The next token is the target', `
    <p>Take the sentence “The cat sat on the mat.” The model reads “The” and must predict “cat”. Then it reads “The cat” and must predict “sat”.</p>
    <p>So one sentence gives many training examples. In each one, the target is the token that really comes next. The text holds the targets, so nobody has to write them. Books call this <strong>self-supervised learning</strong>.</p>
    <p>The loss is ${term('cross-entropy', 'cross-entropy')}, as for the digits. The model gives a score to every token of the vocabulary, and ${term('softmax')} turns the scores into ${term('probability', 'probabilities')}. The loss is the surprise at the token that really came next.</p>
    <p>For digits, the ${term('class', 'classes')} were the ten digits. Here, the classes are the tokens of the vocabulary.</p>`),
    },
    {
      kind: 'widget', id: 'pairs', className: 'widget', label: 'Interactive: the training examples of one sentence',
      mount: (host) => mountNextToken(host),
      tasks: () => [
        { do: 'Press “Next example” until the end of the sentence.',
          see: 'At each step, the model reads one more token. Every token after the first is a target once.' },
        { do: 'Type a sentence of your own, and step through it again.',
          see: 'Any text gives training examples, and nobody wrote their targets.' },
      ],
    },
    {
      kind: 'quiz', id: 'equal-scores', purpose: 'cumulative', from: 'losses', type: 'choice',
      prompt: '<p>Before training, a language model gives every token the same score. It has far more tokens than there are digits. How does its surprise compare with that of a digit network with equal scores?</p>',
      options: [
        { html: 'Its surprise is larger.', correct: true,
          feedback: 'Yes. Each token gets a smaller probability than each digit did.' },
        { html: 'It is the same, because all the scores are equal in both cases.',
          feedback: 'With equal scores, each probability depends on how many classes there are.' },
        { html: 'It is smaller, because more tokens share the loss.',
          feedback: 'The surprise looks only at the probability of the right class, which is smaller here.' },
      ],
      reveal: '<p>Minus the log of a smaller probability is larger.</p>',
    },
    {
      kind: 'text',
      html: () => sec('attention', '3 · Attention looks back at the earlier tokens', `
    <p>Read “The cat sat on the mat because it was tired.” To continue after “it”, the model must know that “it” means the cat.</p>
    <p>${term('attention', 'Attention')} gives the model this information. For each token, it gives a weight to the token itself and to every earlier token. Then it takes the weighted sum of vectors that stand for those tokens.</p>
    <p>In a ${term('neuron')}, the weights of the weighted sum were parameters. Here the model computes new weights for every text, from dot products of vectors that it makes from the embeddings.</p>
    <p>Softmax turns the dot products into weights that are positive and add to one, so the sum is a weighted average. Training learns how to make the vectors.</p>
    <p>A token may look only at itself and at the tokens before it.</p>`),
    },
    { kind: 'text', html: () => figure() },
    {
      kind: 'quiz', id: 'look-back', purpose: 'check', type: 'choice',
      prompt: '<p>While the model learns, why may a token look only at itself and at the tokens before it?</p>',
      options: [
        { html: 'So that it cannot read the token that it must predict.', correct: true,
          feedback: 'Yes. Otherwise it could copy its target and would not learn to predict.' },
        { html: 'Because later words matter less for the meaning.',
          feedback: 'Later words can matter a lot. The reason is the target.' },
        { html: 'To save computer time.',
          feedback: 'The reason is the target that the model must predict.' },
      ],
      reveal: '<p>In the widget, the model reads only the tokens before the target. It cannot see the target or the dashed tokens after it.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('loop', '4 · Pretraining runs the loop of this course', `
    <p>This first stage of training is ${term('pretraining')}.</p>
    <ul>
      <li>The data is a very large amount of text, cut into training examples.</li>
      <li>The model has embeddings, attention, and layers of neurons with an ${term('activation-function')} such as ${term('relu', 'ReLU')}. This design is a transformer.</li>
      <li>The loss is the cross-entropy at the next token.</li>
      <li>In the adjust step, ${term('backpropagation')} finds every gradient, and an ${term('optimizer')} changes the parameters.</li>
    </ul>
    <p>GPT is the family of models behind ChatGPT. The first GPT model trained with ${term('adam', 'Adam')} and the ${term('weight-decay')} of AdamW. The digit network of this course used AdamW too.</p>
    <p>The digit network has ${T.anchor('mlp_parameters')} parameters. Large language models have far more parameters and read far more text.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('after', '5 · After pretraining', `
    <p>Now the model can continue a text. To make an assistant, ChatGPT got more stages of training.</p>
    <p>In <strong>fine-tuning</strong>, people write good answers, and the model trains on them with the same loss.</p>
    <p>In <strong>learning from human feedback</strong>, people rank answers of the model. A second model learns to score answers as people do. The chat model then trains to get high scores.</p>
    <p>That goal is not a next token, so it needs another method, called reinforcement learning.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('more', '6 · Where TorC stops, and where to learn more', `
    <p>TorC has the parts of this course, from linear layers and ${term('convolution', 'convolutions')} to AdamW. It has no layer for embeddings or attention, so this course stops at networks for images.</p>
    <p>These free resources go further.</p>
    <ul>
      <li><a href="${src('3Blue1Brown')}">Neural networks</a>, by 3Blue1Brown, is a series of videos with moving pictures of a network, ${term('gradient-descent')} and backpropagation.</li>
      <li><a href="${src('Andrej Karpathy')}">Zero to Hero</a>, by Andrej Karpathy, is a series of videos in which you build backpropagation and later a small GPT in Python.</li>
      <li><a href="${src('Michael Nielsen')}">Neural Networks and Deep Learning</a>, by Michael Nielsen, is a free online book about a network for handwritten digits.</li>
      <li>The free online book at <a href="${src('Zhang')}">d2l.ai</a> has Python code, from linear layers to attention and transformers.</li>
    </ul>`),
    },
    {
      kind: 'explain', id: 'self-target',
      prompt: '<p>Explain where a language model gets the target of each training example in pretraining.</p>',
      model: '<p>The target is the token that really comes next in the text, so the text gives its own targets.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>ChatGPT predicts the next token again and again. In pretraining, the token that really comes next is the target, so the text gives its own targets.</p>
    <p>The loss is cross-entropy over the vocabulary, and the loop of this course trains the model. Attention lets each token take a weighted average of the earlier tokens. Later stages learn from people.</p>`,
      next: () => '<p>You reached the end of the main path. Which part of the loop do you want to build yourself first?</p>',
    },
  ],
};
export default lesson;
