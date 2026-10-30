// Lesson "what-is-learning" (part 1, the whole game; brief §6 Part 0, §3 "Whole game first" and "Productive failure", R4).
// The digit pad first (the trained MLP of data/mnist-mlp.json), then the predict quiz and right after it the hand-fitting game on
// data/whole-game.json: the learner fits a line to ten made-up students, sees each miss as a square, lets TorC's run take over,
// and then scores both lines on thirty hidden students. After the game come the formula of the loss, the loop in words, the
// parameters (the one "knob" analogy of the course), new data, and what the course delivers. Every number comes from the
// traces through T(); the claims are tested in site/test/what-is-learning.test.mjs. Notes: docs/lessons/what-is-learning.md.
// Rules: docs/authoring.md, docs/v3-implementation.md.
import { math } from '../core/katex.js';
import { worked, analogy, chatgpt, term, loopDiagram } from '../core/blocks.js';
import { mountDigitPad } from '../widgets/digit-widgets.js';
import { mountFitGame } from '../widgets/fit-game.js';

const F = 'files.whole-game';
export const RUN = `${F}.runs.0`;          // the run from w = 0, b = 0 (the test pins its id)
export const LINE = `${F}.probes.3`;       // the line w = 1, b = 3 of the worked example (the test pins its id)
export const ONE = 1, TWO = 6;             // the training students of the worked example and of the "your turn" quiz
const sec = (id, title, body) => `<section class="wl" aria-labelledby="wl-${id}"><h2 id="wl-${id}">${title}</h2>${body}</section>`;
const eq = (latex, label) => `<div class="wl-eq" tabindex="0" role="group" aria-label="${label}">${math(latex)}</div>`;

const STYLE = `<style>
.wl p,.wl li{max-width:44rem}
.wl h2{margin-top:2rem}
.wl-eq{overflow-x:auto;margin:.2rem 0}
.wl-eq .katex-display{margin:.4em 0}
.wl .loop{margin:.6rem 0}
.wl-pre{color:var(--muted);max-width:46rem}
</style>`;

// one student of the worked line: hours, score, prediction, miss and its square, all from TorC's probe of that line
const student = (T, i, role) => ({
  x: T(`${F}.data.train.x.${i}`, role), y: T(`${F}.data.train.y.${i}`, role),
  pred: T(`${LINE}.train_pred.${i}`, role), miss: T(`${LINE}.train_residual.${i}`, role), sq: T(`${LINE}.train_square.${i}`, role),
});
const lineSum = (T, s) => `${T(`${LINE}.w`, 'worked')} × ${s.x} + ${T(`${LINE}.b`, 'worked')} = ${s.pred}`;

export default {
  id: 'what-is-learning',
  title: 'What does it mean for a machine to learn?',
  trace: ['whole-game.json', 'mnist-mlp.json'],
  template: 3,
  loop: 'all',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>What does it mean for a machine to learn?</h1>
    <p class="lead">A computer can learn to read handwriting, and nobody writes rules for it. Here you find out how, and you try it yourself.</p>
    <p class="wl-pre">You need algebra and graphs of functions, such as straight lines. The course explains everything else when you need it. The picture below names four words that this page explains.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'A computer reads a digit that you draw, but nobody wrote rules for it. How did it learn?',
        'How does a computer find a line through the test scores of ten students?',
        'How can you tell if that line works for new students?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('digits', 'A computer that reads digits', `
    <p>Press Clear and draw a digit in the black square. A ${term('model')} reads it. A model is a rule with adjustable numbers that turns an input into an answer.</p>
    <p>Nobody gave this model rules such as “a seven has a line at the top”. It learned from ${T.anchor('train_images')} handwritten digits with their right answers. How can a machine learn from examples?</p>`),
    },
    {
      kind: 'widget', id: 'pad', className: 'widget', label: 'Interactive: draw a digit and see how sure the model is of each digit',
      mount: (host, D) => mountDigitPad(host, D, { simple: true }),
      tasks: () => [{
        do: 'Press Clear and draw a few digits. Then press the last four buttons above the square.',
        see: 'The model reads most clear digits correctly, but it reads the last four wrongly.',
      }],
    },
    {
      kind: 'text', setup: true,
      html: () => sec('story', 'Hours of practice and test scores', `
    <p>The digit model is too large to watch, so we start with a small problem.</p>
    <p>Forty made-up students give their hours of practice in a week and their test score out of ten. The computer learns from ten of them. This learning is ${term('training')}, and the ten students are the training points. The other thirty are ${term('hidden-points', 'hidden points')}, which nobody uses until the end.</p>
    <p>The model is a straight line. Its input is the hours, and its ${term('prediction')} is a score. The real score is the ${term('target')}. The line has two numbers, <code>w</code> and <code>b</code>:</p>
    ${eq('\\text{predicted score} = w \\times \\text{hours} + b', 'The model: predicted score equals w times hours plus b')}`),
    },
    {
      kind: 'quiz', id: 'perfect-line', purpose: 'predict', type: 'choice',
      prompt: () => '<p>In the game below, you place a line by hand. Then the computer fits a line to the same ten points. What do you expect from the computer’s line?</p>',
      options: [
        { html: 'It passes exactly through all ten points.', feedback: 'The points do not lie on one straight line, so every line misses some.' },
        { html: 'It misses some points, but no line misses them less.', correct: true, feedback: 'Even the best line misses some points.' },
        { html: 'A careful hand line can miss the points less.', feedback: 'A hand line can come close or tie, but it cannot miss less.' },
        { html: 'It finds the exact rule behind the made-up scores.', feedback: 'The computer sees only the ten points, not the rule behind them.' },
      ],
      reveal: () => '<p>Every straight line misses some of the points. In the game, watch the squares under the computer’s line shrink without disappearing.</p>',
    },
    {
      kind: 'widget', id: 'fit', className: 'widget', label: 'Interactive: place a line by hand, then let the computer fit one',
      mount: (host, D) => mountFitGame(host, D),
      tasks: () => [
        { do: 'Drag the two white handles, or use Tab and the arrow keys. Make the yellow squares as small as you can.', see: `Each square shows one miss. The ${term('loss')} is the mean area of the squares. No line removes every square.` },
        { do: 'Press “Let the computer fit the line”, then Step a few times, then To the end.', see: 'Its first steps change the line a lot, and later steps very little. At the end, its loss is smaller than yours, or equal.' },
        { do: 'Then press “Show the hidden points”. Your line is then locked.', see: 'The computer’s line misses the hidden points by more than the training points.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => {
        const s = student(T, ONE, 'worked');
        return sec('misses', '1 · How to measure the misses of a line', `
    <p>The miss at one student is the prediction minus the target. The computer squares each miss, so misses above and below the line both count, and big misses count more. The loss is the mean of the ten squares. Its name is the ${term('mean-squared-error', 'mean squared error')}.</p>
    ${worked(`<p>Take the line with <code>w</code> = ${T(`${LINE}.w`, 'worked')} and <code>b</code> = ${T(`${LINE}.b`, 'worked')}. One student practised ${s.x} hours and scored ${s.y}. The prediction is ${lineSum(T, s)}. The miss is ${s.pred} − ${s.y} = ${s.miss}, and its square is ${s.sq}.</p>`)}
    <p>A loss is in squared points. Its square root brings it back to the unit of the score, and the game calls it the typical miss.</p>`);
      },
    },
    {
      kind: 'quiz', id: 'your-square', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${LINE}.train_square.${TWO}`, tolerance: 0.01,
      prompt: (D, T) => `<p>Use the same line, with <code>w</code> = ${T(`${LINE}.w`, 'setup')} and <code>b</code> = ${T(`${LINE}.b`, 'setup')}. Another student practised ${T(`${F}.data.train.x.${TWO}`, 'setup')} hours and scored ${T(`${F}.data.train.y.${TWO}`, 'setup')}. What is the square of the miss?</p>`,
      reveal: (D, T) => {
        const s = student(T, TWO, 'worked');
        return worked(`<p>The prediction is ${lineSum(T, s)}. The miss is ${s.pred} − ${s.y} = ${s.miss}, and its square is ${s.sq}.</p>`);
      },
    },
    {
      kind: 'text',
      html: (D, T) => sec('loop', '2 · What the computer did', `
    <p>The computer’s line comes from TorC, a set of ready-made program parts that trains every model of this course. It started at <code>w</code> = ${T(`${RUN}.start.w`, 'setup')} and <code>b</code> = ${T(`${RUN}.start.b`, 'setup')} and repeated one round, a step, many times:</p>
    <ol class="wl-loop">
      <li><b>Data.</b> It takes the ten training points.</li>
      <li><b>Model.</b> The line predicts a score for each of them.</li>
      <li><b>Loss.</b> The loss measures the misses.</li>
      <li><b>Adjust.</b> It finds in which direction <code>w</code> and <code>b</code> must change to make the loss smaller, and it moves them that way.</li>
    </ol>
    <p>The next lessons show this training loop with their own part lit.</p>
    ${loopDiagram('all')}
    <p>The scores came from the rule score = ${T(`${F}.data.true_line.w`, 'setup')} × hours + ${T(`${F}.data.true_line.b`, 'setup')}, plus random noise. After training, the line has <code>w</code> ≈ ${T(`${RUN}.final.w`, 'contrast')} and <code>b</code> ≈ ${T(`${RUN}.final.b`, 'contrast')}. It is close to the rule but not equal to it, because it saw only ten noisy points.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('parameters', '3 · The numbers that training changes', `
    <p>The number <code>w</code> multiplies the input, so it is a ${term('weight')}. The number <code>b</code> is added, so it is a ${term('bias')}. Training changes both, and together they are the ${term('parameter', 'parameters')} of the model.</p>
    ${analogy('A model is a machine with knobs, and each setting gives other predictions. Training turns the knobs until the predictions miss less. The knobs are the parameters.',
    'A person turns one knob at a time. The computer changes all the parameters in one step.')}
    <p>The digit model has ${T.anchor('mlp_parameters')} parameters instead of two.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('new-data', '4 · Learning means working on new data', `
    <p>On the training points, the computer’s line misses a student by about ${T.human(`${RUN}.final.train_loss`, 'rmse')}. On the thirty hidden students it misses by about ${T.human(`${RUN}.final.hidden_loss`, 'rmse')}, and that is what to expect for a new student.</p>
    <p>New students are harder, because the line fits the noise of its ten points a little. Still, even the best possible line for the hidden points misses by about ${T.human(`${F}.derived.best_hidden_line.hidden_loss`, 'rmse')}. On a few points, a hand line can tie or win by luck.</p>
    <p>So a machine learns only if it also works on examples that it never saw. The students are made up, so these numbers are only an example.</p>`),
    },
    {
      kind: 'quiz', id: 'curve', purpose: 'transfer', type: 'choice',
      prompt: () => '<p>A friend draws a wiggly curve through all ten training points, so its loss on them is zero. What do you check before you call it better?</p>',
      options: [
        { html: 'How much it misses the thirty hidden students.', correct: true, feedback: 'The hidden students show how it works on new data.' },
        { html: 'Nothing, because a loss of zero is the best result.', feedback: 'A curve through every training point also follows their noise.' },
        { html: 'That it also passes through the rule behind the scores.', feedback: 'With real data, nobody knows the rule.' },
      ],
      reveal: () => '<p>Measure its miss on the hidden students. A curve that follows the noise of ten points can miss new students by more than a straight line.</p>',
    },
    {
      kind: 'text',
      html: () => sec('course', '5 · What the course explains', `
    <p>The rest of the course explains each part of the loop.</p>
    <ul>
      <li>The next part shows how the computer finds the direction.</li>
      <li>Then you build networks, models with many parameters that can fit curves.</li>
      <li>Then you see how to train networks well, and how the digit model learns from images.</li>
    </ul>
    ${chatgpt('The last lesson of the course shows how ChatGPT learns from text with this same loop.')}`),
    },
    {
      kind: 'explain', id: 'learn',
      prompt: () => '<p>Say in your own words what it means for a machine to learn.</p>',
      model: () => '<p>A machine learns when it changes its parameters to make its misses smaller, and its predictions then work on new examples too.</p>',
    },
    {
      kind: 'answer',
      html: (D, T) => `<p>Nobody wrote rules for the digit model. It ran the same loop as the line, on ${T.anchor('train_images')} images. It predicted, measured the loss, changed its parameters a little, and repeated. To find out if a model learned, keep some examples hidden and measure its miss on them.</p>`,
      next: () => '<p>How does the computer know in which direction to change each parameter?</p>',
    },
  ],
};
