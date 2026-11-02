// Lesson "from-digits-to-chatgpt" (Part 6 `next`, curriculum v3, template 3, the last page of the main path), title "From digits to
// ChatGPT" (brief §6 Part 5, decision 18). Content beyond TorC, so `conceptual: true` with `sources` in the footer: text becomes
// tokens and tokens become embeddings; next-token prediction with cross-entropy, where the text gives its own targets (the
// widget makes the samples of one sentence, the reader counts them for one more); writing is the same prediction, a pick and
// a repeat; attention as a weighted average with attention weights computed from the tokens, looking back only; pretraining
// runs the loop of this course at a much larger size; fine-tuning and learning from human feedback come after; where TorC
// stops, and a list of resources. No TorC code and no numbers except the year of the first ChatGPT, which its source gives.
// The attention figure uses attention weights chosen by hand and says so.
// Sources were not fetched in the session that wrote the page: docs/lessons/from-digits-to-chatgpt.md ("verify before quoting").
import { term } from '../core/blocks.js';
import { mountNextToken } from '../widgets/next-token.js';

const sec = (id, title, body) => `<section class="ch" aria-labelledby="ch-${id}"><h2 id="ch-${id}">${title}</h2>${body}</section>`;

export const SOURCES = [
  { title: 'Sennrich, Haddow and Birch (2016), Neural Machine Translation of Rare Words with Subword Units (tokens that are parts of words)', url: 'https://arxiv.org/abs/1508.07909' },
  { title: 'Vaswani et al. (2017), Attention Is All You Need, sections 3.2 (attention as a weighted sum with softmax weights, the mask), 3.3 (the feed-forward layers with ReLU), 3.4 (learned embeddings) and 3.5 (the position of each token)', url: 'https://arxiv.org/abs/1706.03762' },
  { title: 'Radford et al. (2018), Improving Language Understanding by Generative Pre-Training, sections 3.1 and 4.1 (pretraining on unlabeled text by next-token prediction, Adam with decoupled weight decay)', url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf' },
  { title: 'Loshchilov and Hutter (2019), Decoupled Weight Decay Regularization (AdamW)', url: 'https://arxiv.org/abs/1711.05101' },
  { title: 'Brown et al. (2020), Language Models are Few-Shot Learners (much larger models trained on much more text)', url: 'https://arxiv.org/abs/2005.14165' },
  { title: 'Holtzman et al. (2020), The Curious Case of Neural Text Degeneration (writing text by picking each next token at random with the predicted probabilities)', url: 'https://arxiv.org/abs/1904.09751' },
  { title: 'Ouyang et al. (2022), Training language models to follow instructions with human feedback (fine-tuning on written answers, a reward model from rankings, reinforcement learning)', url: 'https://arxiv.org/abs/2203.02155' },
  { title: 'OpenAI (2022), Introducing ChatGPT (the first ChatGPT, trained with the methods of the paper by Ouyang et al.)', url: 'https://openai.com/index/chatgpt/' },
  { title: '3Blue1Brown, Neural networks (video series)', url: 'https://www.3blue1brown.com/topics/neural-networks' },
  { title: 'Andrej Karpathy, Neural Networks: Zero to Hero (video series)', url: 'https://karpathy.ai/zero-to-hero.html' },
  { title: 'Michael Nielsen, Neural Networks and Deep Learning (online book)', url: 'https://neuralnetworksanddeeplearning.com/' },
  { title: 'Zhang, Lipton, Li and Smola, Dive into Deep Learning (online book; the name self-supervised learning)', url: 'https://d2l.ai/' },
];
const src = (start) => SOURCES.find((s) => s.title.startsWith(start)).url;

// The made-up attention weights of the figure: what "it" takes from itself and the tokens before it. Chosen by hand, not from a
// model; they add to one (tested). The tokens after "it" are hidden from it.
export const FIGURE = {
  tokens: ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.'],
  reader: 7,
  weights: [0.04, 0.55, 0.06, 0.03, 0.04, 0.12, 0.08, 0.08],
};
const BAR = 54;   // px of an attention weight of one
function figure() {
  const cells = FIGURE.tokens.map((t, i) => {
    const w = FIGURE.weights[i];
    const bar = w === undefined ? '<span class="ch-bar none"></span>' : `<span class="ch-bar" style="height:${Math.max(2, Math.round(w * BAR * 1.6))}px"></span>`;
    const cls = i === FIGURE.reader ? ' reader' : w === undefined ? ' later' : '';
    return `<span class="ch-cell${cls}">${bar}<span class="ch-word">${t}</span></span>`;
  }).join('');
  return `<figure class="ch-fig"><div class="ch-row" role="img" aria-label="A made-up example. The token “it” gives its largest attention weight to “cat”, small attention weights to the other earlier tokens, and none to the tokens after it.">${cells}</div>`
    + '<figcaption>A made-up illustration with attention weights chosen by hand, not by a trained model or by TorC. A taller bar is a larger attention weight for “it”. The dashed tokens are hidden from “it”.</figcaption></figure>';
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
    <p>ChatGPT first learns from a very large amount of ordinary text.</p>`),
    },
    {
      kind: 'quiz', id: 'where-target', purpose: 'predict', type: 'choice',
      prompt: () => `<p>Each of ChatGPT’s samples needs a target, so that the ${term('loss')} can measure the miss. Where do the targets come from?</p>`,
      options: [
        { html: 'From people, who write each continuation.',
          feedback: 'That is how the digits got their targets. For this much text, nobody writes targets.' },
        { html: 'From the text, because the piece that really comes next is the target.', correct: true,
          feedback: 'Yes. Nobody has to write a target.' },
        { html: 'From nowhere, because the model only stores the text.',
          feedback: `This is the belief that the model memorizes the text. It learns ${term('parameter', 'parameters')} that score each possible next piece, and only a loss and its ${term('gradient')} change them.` },
      ],
      reveal: '<p>The next sections show how one sentence gives many samples.</p>',
    },
    {
      kind: 'text',
      html: () => sec('tokens', '1 · Text becomes tokens, and tokens become vectors', `
    <p>First, a program called a tokenizer cuts the text into ${term('token', 'tokens')}. A token is often a word, a part of a word or a punctuation mark. All the tokens that the model knows form its <strong>vocabulary</strong>.</p>
    <p>Each token of the vocabulary has its own ${term('vector')}, a list of numbers, called its ${term('embedding')}. These numbers are parameters, like the ${term('weight', 'weights')} of a ${term('layer')}, and training changes them.</p>
    <p>A model that predicts the next token of a text is a <strong>language model</strong>.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('next', '2 · The next token is the target', `
    <p>Take the sentence “The cat sat on the mat.” The model reads “The” and must predict “cat”. Then it reads “The cat” and must predict “sat”.</p>
    <p>Each of these is one sample, and its target is the token that really comes next. The text holds the targets, so nobody has to write them. This is also called <strong>self-supervised learning</strong>.</p>
    <p>The loss is ${term('cross-entropy', 'cross-entropy')}, as for the digits. The model gives a score to every token of the vocabulary, and ${term('softmax')} turns the scores into ${term('probability', 'probabilities')}. The loss is the surprise at the token that really came next.</p>
    <p>For digits, the ${term('class', 'classes')} were the ten digits. Here, the classes are the tokens of the vocabulary.</p>`),
    },
    {
      kind: 'quiz', id: 'count', purpose: 'check', type: 'choice', title: 'Your turn',
      prompt: '<p>The text “I like green tea” has no full stop, so it has four tokens. How many samples does it give?</p>',
      options: [
        { html: 'Three samples', correct: true, feedback: 'Yes.' },
        { html: 'Four samples', feedback: 'This counts one sample per token, but the first token is never a target.' },
        { html: 'One sample', feedback: 'One image gave one sample, but a sentence gives one sample for each token after the first.' },
      ],
      reveal: '<p>The targets are “like”, “green” and “tea”.</p>',
    },
    {
      kind: 'widget', id: 'pairs', className: 'widget', label: 'Interactive: the samples of one sentence',
      mount: (host) => mountNextToken(host),
      tasks: () => [
        { do: 'Press “Next sample” until the end of the sentence.',
          see: 'The model reads one more token at each step. Every token after the first is a target once.' },
        { do: 'Go back one sample, then press “Next sample” again.',
          see: 'The target joins the tokens that the model reads. When the model writes, its pick joins the text in the same way.' },
        { do: 'Type a sentence of your own, and step through it.',
          see: 'Nobody wrote the targets of your sentence.' },
      ],
    },
    {
      kind: 'text',
      html: () => `<section class="ch" aria-label="How the model writes">
    <p>When ChatGPT writes, it makes the same ${term('prediction')}. It reads the text so far, and softmax gives every token a probability.</p>
    <p>It picks a likely token, often at random with these probabilities. It adds the token to the text and predicts again.</p></section>`,
    },
    {
      kind: 'quiz', id: 'equal-scores', purpose: 'cumulative', from: 'losses', type: 'choice',
      prompt: '<p>Suppose that a language model gives every token the same score. It has far more tokens than there are digits. How does its surprise compare with that of a digit network with equal scores?</p>',
      options: [
        { html: 'Its surprise is larger.', correct: true,
          feedback: 'Yes. Each token gets a smaller probability than each digit did.' },
        { html: 'It is the same, because all scores are equal.',
          feedback: 'Each probability depends on how many classes there are.' },
        { html: 'It is smaller, because more tokens share the loss.',
          feedback: 'The surprise looks only at the probability of the right class, which is smaller here.' },
      ],
      reveal: '<p>Minus the log of a smaller probability is larger.</p>',
    },
    {
      kind: 'text',
      html: () => sec('attention', '3 · Attention looks back at the earlier tokens', `
    <p>In “The cat sat on the mat because it was tired”, the word “it” means the cat. The model must know this to continue.</p>
    <p>In a ${term('neuron')}, the weights were parameters. ${term('attention', 'Attention')} computes new <strong>attention weights</strong> for every text.</p>
    <p>For each token, the model makes vectors from the embeddings. It takes dot products with the vectors of the token itself and of every earlier token. A dot product is large when two vectors have large numbers in the same places.</p>
    <p>Softmax turns the dot products into attention weights that add to one. The token then takes the weighted average of vectors for those tokens.</p>
    <p>Training makes the vectors of “it” and “cat” match, so “it” takes much from “cat”. A token may look only at itself and at earlier tokens.</p>`),
    },
    { kind: 'text', html: () => figure() },
    {
      kind: 'text',
      html: () => sec('loop', '4 · Pretraining runs the loop of this course', `
    <p>This first stage of training is ${term('pretraining')}.</p>
    <ul>
      <li>The data is text, cut into samples.</li>
      <li>The model is a transformer. After the embeddings, attention and layers of neurons with an ${term('activation-function')} such as ${term('relu', 'ReLU')} take turns many times. It also uses token positions.</li>
      <li>The loss is cross-entropy at the next token.</li>
      <li>In the adjust step, ${term('backpropagation')} finds every gradient, and an ${term('optimizer')} changes the parameters.</li>
    </ul>
    <p>GPT is the family of models behind ChatGPT. The first GPT model trained with ${term('adam', 'Adam')} and the ${term('weight-decay')} of AdamW. The digit network of this course used AdamW too.</p>
    <p>Large language models are far larger than any network of this course.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('after', '5 · After pretraining', `
    <p>The first ChatGPT, from 2022, got more stages of training to make it an assistant.</p>
    <p>In <strong>fine-tuning</strong>, people write good answers, and the model trains on them with the same loss.</p>
    <p>In <strong>learning from human feedback</strong>, people rank answers of the model. A second model learns to score answers as people do. The chat model then trains to get high scores.</p>
    <p>That goal is not a next token, so it needs another method, called reinforcement learning.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('more', '6 · Where TorC stops, and where to learn more', `
    <p>TorC has the parts of this course, from linear layers and ${term('convolution', 'convolutions')} to AdamW. It has no layer for embeddings or attention, so this course stops at networks for images.</p>
    <ul>
      <li><a href="${src('3Blue1Brown')}">Neural networks</a>, by 3Blue1Brown, is a series of videos with moving pictures of a network, ${term('gradient-descent')} and backpropagation.</li>
      <li><a href="${src('Andrej Karpathy')}">Zero to Hero</a>, by Andrej Karpathy, is a series of videos that build backpropagation and later a small GPT in Python.</li>
      <li><a href="${src('Michael Nielsen')}">Neural Networks and Deep Learning</a>, by Michael Nielsen, is a free online book about a network for handwritten digits.</li>
      <li>The free book at <a href="${src('Zhang')}">d2l.ai</a> goes from linear layers to transformers, with Python code.</li>
    </ul>`),
    },
    {
      kind: 'explain', id: 'self-target',
      prompt: '<p>Explain where a language model gets its targets in pretraining.</p>',
      model: '<p>The target is the token that really comes next in the text, so the text gives its own targets.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>ChatGPT writes by predicting the next token, picking one and adding it to the text. In pretraining, the token that really comes next is the target, so the text gives its own targets.</p>
    <p>The loss is cross-entropy, and the loop of this course trains the model. Attention lets each token take a weighted average of itself and the earlier tokens. Later stages learn from people.</p>`,
      next: () => `<p>How can you build one yourself? Start with the videos of <a href="${src('Andrej Karpathy')}">Zero to Hero</a>, or use TorC with the appendix <a href="../tensors/">Using TorC</a>.</p>`,
    },
  ],
};
export default lesson;
