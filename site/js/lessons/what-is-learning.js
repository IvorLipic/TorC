// Lesson "what-is-learning" (part 1, the whole game; brief §6 Part 0, §3 "Whole game first" and "Productive failure", R4).
// The digit pad first (the trained MLP of data/mnist-mlp.json), then the hand-fitting game on data/whole-game.json: the learner
// fits a line to ten made-up students, sees each miss as a square, lets TorC's run take over, and scores both lines on five
// hidden students. Then the parts of the training loop, the weights (the one "knob" analogy of the course), new data, and what
// the course delivers. Every number comes from the traces through T(); the claims are tested in
// site/test/what-is-learning.test.mjs. Notes: docs/lessons/what-is-learning.md. Rules: docs/authoring.md, docs/v3-implementation.md.
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
    <p class="wl-pre">You need algebra and graphs of functions, such as straight lines. The course explains everything else when you need it.</p></section>`,
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
    <p>Draw a digit in the black square. A ${term('model')} reads it. A model is a rule with numbers in it that turns an input into an answer. The bars show how sure it is of each digit.</p>
    <p>Nobody gave this model rules such as “a seven has a line at the top”. It learned from ${T.anchor('train_images')} handwritten digits with their right answers. How can a machine learn from examples?</p>`),
    },
    {
      kind: 'widget', id: 'pad', className: 'widget', label: 'Interactive: draw a digit and see how sure the model is of each digit',
      mount: (host, D) => mountDigitPad(host, D),
      tasks: () => [{
        do: 'Draw a few digits. Then press the last four buttons with test images.',
        see: 'The model reads most clear digits correctly. The last four images show that it still makes mistakes.',
      }],
    },
    {
      kind: 'text', setup: true,
      html: () => sec('story', 'Hours of practice and test scores', `
    <p>The digit model is too large to watch, so we start with a small problem.</p>
    <p>Fifteen made-up students give their hours of practice in a week and their test score out of ten. Each score comes from a fixed rule plus a random part, the noise.</p>
    <p>The computer learns from ten students. This learning is ${term('training')}, and the ten students are the training points. The other five are ${term('hidden-points', 'hidden points')}, which nobody uses until the end.</p>
    <p>The model is a straight line. Its input is the hours, and its ${term('prediction')} is a score. The real score is the ${term('target')}. The line has two numbers, <code>w</code> and <code>b</code>:</p>
    ${eq('\\text{predicted score} = w \\times \\text{hours} + b', 'The model: predicted score equals w times hours plus b')}`),
    },
    {
      kind: 'quiz', id: 'perfect-line', purpose: 'predict', type: 'choice',
      prompt: () => '<p>In the game below, you place a line by hand. Then the computer places its own line on the same ten points. What do you expect?</p>',
      options: [
        { html: 'The computer’s line passes through all ten points.', feedback: 'The noise moves the points off any straight line, so every line misses some of them.' },
        { html: 'The computer’s line fits the ten points at least as well as yours, but it still misses some.', correct: true, feedback: 'No line hits every point, but the misses of the computer are the smallest.' },
        { html: 'The computer finds the exact rule that made the scores.', feedback: 'The computer sees only the ten points, not the rule. So its line is close to the rule, but not equal to it.' },
      ],
      reveal: (D, T) => `<p>The computer’s line ends at <code>w</code> = ${T(`${RUN}.final.w`, 'reveal')} and <code>b</code> = ${T(`${RUN}.final.b`, 'reveal')}. The rule that made the scores has <code>w</code> = ${T(`${F}.data.true_line.w`, 'reveal')} and <code>b</code> = ${T(`${F}.data.true_line.b`, 'reveal')}. The computer’s line still misses a typical student by about ${T.human(`${RUN}.final.train_loss`, 'rmse')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => {
        const s = student(T, ONE, 'worked');
        return sec('score', '1 · How to score a line', `
    <p>The miss at one student is the prediction minus the target. The computer squares each miss, so misses above and below the line both count. Big misses count much more.</p>
    <p>The ${term('loss')} of the line is the mean of the ten squares. The name of this loss is the ${term('mean-squared-error', 'mean squared error')}.</p>
    ${worked(`<p>Take the line with <code>w</code> = ${T(`${LINE}.w`, 'worked')} and <code>b</code> = ${T(`${LINE}.b`, 'worked')}. One student practised ${s.x} hours and scored ${s.y}. The prediction is ${lineSum(T, s)}. The miss is ${s.pred} − ${s.y} = ${s.miss}, and its square is ${s.sq}.</p>
    <p>The mean of all ten squares is ${T(`${LINE}.train_loss`, 'worked')}. Its square root is about ${T.human(`${LINE}.train_loss`, 'rmse')}.</p>`)}
    <p>This square root, the typical miss, is in the unit of the score.</p>`);
      },
    },
    {
      kind: 'quiz', id: 'your-square', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${LINE}.train_square.${TWO}`, tolerance: 0.01,
      prompt: (D, T) => `<p>Use the same line, with <code>w</code> = ${T(`${LINE}.w`, 'setup')} and <code>b</code> = ${T(`${LINE}.b`, 'setup')}. Another student practised ${T(`${F}.data.train.x.${TWO}`, 'setup')} hours and scored ${T(`${F}.data.train.y.${TWO}`, 'setup')}. What is the square of the miss?</p>`,
      reveal: (D, T) => {
        const s = student(T, TWO, 'worked');
        return worked(`<p>The prediction is ${lineSum(T, s)}. The miss is ${s.pred} − ${s.y} = ${s.miss}, and its square is ${s.sq}.</p>`)
          + '<p>A miss smaller than one gives an even smaller square.</p>';
      },
    },
    {
      kind: 'widget', id: 'fit', className: 'widget', label: 'Interactive: place a line by hand, then let the computer fit one',
      mount: (host, D) => mountFitGame(host, D),
      tasks: () => [
        { do: 'Drag the two white handles, or use the Tab and arrow keys. Make the orange squares as small as you can.', see: 'No line removes every square, so the loss stays above zero.' },
        { do: 'Press “Let the computer fit the line”. Press Step a few times, then Play.', see: 'The computer starts with a flat line at zero. Its first steps change the line a lot, and later steps very little.' },
        { do: 'Press “Show the hidden points”.', see: 'The computer’s line misses the hidden points by more than the training points.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('loop', '2 · What the computer did', `
    <p>The computer’s line comes from a real run of TorC, the library of this course. It started at <code>w</code> = ${T(`${RUN}.start.w`, 'setup')} and <code>b</code> = ${T(`${RUN}.start.b`, 'setup')}. Then it repeated one round, a step, many times:</p>
    <ol class="wl-loop">
      <li><b>Data.</b> It takes the ten training points.</li>
      <li><b>Model.</b> The line predicts a score for each of them.</li>
      <li><b>Loss.</b> The loss measures the misses.</li>
      <li><b>Adjust.</b> It finds the direction in which <code>w</code> and <code>b</code> must change to make the loss smaller. It changes them a small amount that way. Then it forgets this direction, because the next step finds a new one.</li>
    </ol>
    <p>Every lesson shows this training loop with its own part lit.</p>
    ${loopDiagram('all')}
    <p>The name of this method is ${term('gradient-descent')}. The next two lessons show how the computer finds the direction.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('weights', '3 · The numbers that training changes', `
    <p>Training changes only <code>w</code> and <code>b</code>. The numbers that training changes are the ${term('weight', 'weights')} of the model. Later lessons call <code>b</code> the ${term('bias')}.</p>
    ${analogy('A model is a machine with knobs, and each setting gives other predictions. Training turns the knobs until the predictions miss less. The knobs are the weights.',
    'A person turns one knob at a time. The computer finds a direction for every weight and changes all of them in one step.')}
    <p>The digit model has ${T.anchor('mlp_parameters')} weights instead of two.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('new-data', '4 · Learning means working on new data', `
    <p>A small loss on the training points does not prove that the model learned. A model can fit them well and still fail on new students.</p>
    <p>The computer’s line misses the hidden points by more than the training points, by about ${T.human(`${RUN}.final.hidden_loss`, 'rmse')} against ${T.human(`${RUN}.final.train_loss`, 'rmse')}. The training points made the line look better than it is.</p>
    <p>So a machine learns only if it also works on examples that it never saw. Five made-up students are few, and other students would give other numbers.</p>`),
    },
    {
      kind: 'quiz', id: 'weather', purpose: 'transfer', type: 'choice',
      prompt: () => '<p>A model predicts the temperature of tomorrow. On the days that it trained on, it misses very little. What do you check before you trust it?</p>',
      options: [
        { html: 'How much it misses on days that it did not train on.', correct: true, feedback: 'Only new days show if it learned.' },
        { html: 'Nothing, because the small miss proves that it learned.', feedback: 'A model can fit its training days and still fail on new days.' },
        { html: 'That its loss on the training days is exactly zero.', feedback: 'Weather has noise, so even a good model misses a little.' },
      ],
      reveal: () => '<p>Test the model on new days, as the game did with the hidden points.</p>',
    },
    {
      kind: 'text',
      html: () => sec('course', '5 · What the course explains', `
    <p>You saw the whole loop once, on a line with two weights. The rest of the course explains each part.</p>
    <ul>
      <li>The next part shows how the computer finds the direction.</li>
      <li>Then you build networks, models with many weights that can fit curves.</li>
      <li>Then you train networks well, also the digit model.</li>
    </ul>
    ${chatgpt('ChatGPT learns with the same loop. Its data is a huge amount of text, and its model predicts the next word. It has far more weights than the digit model. The last lesson shows what else is different.')}`),
    },
    {
      kind: 'explain', id: 'learn',
      prompt: () => '<p>Say in your own words what it means for a machine to learn.</p>',
      model: () => '<p>A machine learns when it changes its weights so that it misses less, also on new data.</p>',
    },
    {
      kind: 'answer',
      html: () => '<p>A machine learns from examples. It starts with a model whose weights are poor guesses. A loss measures how far its predictions miss. The computer changes the weights a little to make the loss smaller, and it repeats this many times. The machine learns only if its predictions also work on new data.</p>',
      next: () => '<p>How does the computer know in which direction to change each weight?</p>',
    },
  ],
};
