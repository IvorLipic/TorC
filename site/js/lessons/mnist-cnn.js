// Lesson "mnist-cnn" (Part 5 `images`, curriculum v3, template 3; brief §6 and §7 "mnist-cnn", C5, C9). TorC trains the CNN of
// examples/mnist_cnn on MNIST. The page compares it with the MLP and says plainly that the comparison is not perfectly fair (C9),
// with the spread over three random starts (C5, mnist-seeds.json); shows what the first kernels learned; asks why the CNN mixes up
// fours and nines less often (a reasoning question that the confusion matrices support); runs the CNN on a drawing; and ends the
// course with a recap of every part and the bridge to the epilogue. It is the last lesson of Part 5, so the recap doubles as the
// part's mixed review (it recommends going back and never blocks).
// Every number comes from data/mnist-cnn.json (trace_mnist_cnn.cpp), data/mnist-mlp.json (trace_mnist_mlp.cpp),
// data/mnist-seeds.json (trace_mnist_seeds.cpp) and the anchors through a role-tagged T() (docs/authoring.md "Number roles").
// Notes: docs/lessons/mnist-cnn.md.
import { escapeHtml } from '../core/dom.js';
import { worked, inTorc, term } from '../core/blocks.js';
import { mountDigitPad, mountCnnMaps, forwardCnn } from '../widgets/digit-widgets.js';

const F = 'files.mnist-cnn', MLP = 'files.mnist-mlp', SEEDS = 'files.mnist-seeds';
const FOUR = 4, NINE = 9, SEVEN = 7, TWO = 2;          // the confusion pairs the page discusses (tested against the matrices)
const K_TOP = [2, 5], K_RIGHT = 7;                     // first-layer kernels that follow top and right stroke edges (tested)
// the same lines as the `code` of this lesson in course.json, which tools/snippets/check.mjs compiles and runs (tested)
const CODE = `using std::make_unique;
// (B, 1, 28, 28) becomes (B, 8, 14, 14)
model.add(make_unique<nn::Conv2d>(1, 8, 3, 2, 1));
model.add(make_unique<nn::ReLU>());
// (B, 8, 14, 14) becomes (B, 16, 7, 7)
model.add(make_unique<nn::Conv2d>(8, 16, 3, 2, 1));
model.add(make_unique<nn::ReLU>());
// (B, 16, 7, 7) becomes (B, 784)
model.add(make_unique<nn::Flatten>());
// 784 cells give 10 scores
model.add(make_unique<nn::Linear>(16 * 7 * 7, 10));`;
const S1 = `${'files.mnist-seeds'}.derived.seed_set_1`;
const sec = (id, title, body) => `<section class="mc" aria-labelledby="mc-${id}"><h2 id="mc-${id}">${title}</h2>${body}</section>`;

// one line of the course review: the part, a question, a collapsed answer and a link to the lesson that teaches it
const review = (part, q, a, id, title) => `<strong>${part}.</strong> ${q} <details class="mc-ans"><summary>Answer</summary><p>${a} (<a href="../${id}/">${title}</a>)</p></details>`;

const STYLE = `<style>
.mc p{max-width:44rem}
.mc h2{margin-top:2rem}
.mc-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.mc-code code{background:none;padding:0}
.mc-review{margin:.3rem 0 .3rem 1.2rem;padding:0;max-width:42rem}
.mc-review li{margin:.35rem 0}
.mc-ans{margin:.15rem 0 .3rem}
.mc-ans summary{cursor:pointer;color:var(--muted)}
.mc-ans p{margin:.2rem 0}
</style>`;

export default {
  id: 'mnist-cnn',
  title: 'Reading digits with a convolutional network',
  trace: ['mnist-cnn.json', 'mnist-mlp.json', 'mnist-seeds.json'],
  template: 3,
  loop: 'all',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Reading digits with a convolutional network</h1>
    <p class="lead">The ${term('convolution')} ${term('layer', 'layers')} of the last lesson now learn the handwritten digits that the MLP learned.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        `The ${term('cnn', 'CNN')} has about a third of the ${term('parameter', 'parameters')} of the MLP. Does it read new digits better or worse?`,
        `What do its ${term('kernel', 'kernels')} learn to find?`,
        'Which digits does it still mix up?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('model', 'The digit CNN', `
    <p>The CNN reads the same images of handwritten digits as the MLP. Its first convolution layer has ${T(`${F}.layers.0.weight_shape.0`, 'setup')} kernels. Each kernel gives a ${term('feature-map')} of ${T(`${F}.first_maps.shape.2`, 'setup')} × ${T(`${F}.first_maps.shape.3`, 'setup')} cells.</p>
    <p>The second convolution layer has ${T(`${F}.layers.1.weight_shape.0`, 'setup')} kernels, and each reads all the maps of the first. A ${term('relu', 'ReLU')} follows each of the two layers. Then a linear layer reads all ${T(`${F}.layers.2.weight_shape.1`, 'setup')} cells of the ${T(`${F}.layers.1.weight_shape.0`, 'setup')} maps and gives one score per digit.</p>
    <p>The CNN has ${T.anchor('cnn_parameters')} parameters, and the MLP has ${T.anchor('mlp_parameters')}. TorC trains both on the same ${T.anchor('train_images')} ${term('training')} images.</p>`),
    },
    {
      kind: 'quiz', id: 'compare', purpose: 'predict', type: 'choice',
      prompt: '<p>Which network makes fewer mistakes on the test images, which it never saw in training?</p>',
      options: [
        { html: 'The CNN', correct: true, feedback: 'Yes, with fewer parameters.' },
        { html: 'The MLP, because more parameters can learn more', feedback: `More parameters do not mean better reading. The CNN uses its few ${term('weight', 'weights')} everywhere.` },
        { html: 'About the same, because both learn from the same images', feedback: 'The images are the same, but the CNN reads them with kernels.' },
      ],
      reveal: (D, T) => `<p>The CNN makes about ${T.human(`${F}.final.test_correct`, 'per100', { of: `${F}.final.test_samples` })} mistakes per hundred test digits. The MLP makes about ${T.human(`${MLP}.final.test_correct`, 'per100', { of: `${MLP}.final.test_samples` })}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('count', '1 · Counting the mistakes', `
    <p>These counts come from the ${term('test-set')}, so they measure how well each network reads new digits. That is its ${term('generalization')}.</p>
    ${worked(`<p>The MLP reads ${T(`${MLP}.final.test_correct`, 'worked')} of the ${T(`${MLP}.final.test_samples`, 'worked')} test images correctly. So it reads ${T(`${MLP}.final.test_samples`, 'worked')} − ${T(`${MLP}.final.test_correct`, 'worked')} = <strong>${T(`${MLP}.final.test_mistakes`, 'worked')}</strong> wrongly, about ${T.human(`${MLP}.final.test_correct`, 'per100', { of: `${MLP}.final.test_samples` })} per hundred.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'cnn-mistakes', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${S1}.percent_of_mlp_mistakes_avoided`, tolerance: 1, unit: '%',
      prompt: (D, T) => `<p>The MLP reads ${T(`${S1}.mlp_mistakes`, 'setup')} test images wrongly, and the CNN ${T(`${S1}.cnn_mistakes`, 'setup')}. What percentage of the MLP’s mistakes does the CNN avoid? A calculator is fine.</p>`,
      reveal: (D, T) => `<p>The CNN avoids ${T(`${S1}.fewer_mistakes`, 'reveal')} of the ${T(`${S1}.mlp_mistakes`, 'reveal')} mistakes. That is about ${T(`${S1}.percent_of_mlp_mistakes_avoided`, { role: 'reveal', sig: 2 })} percent.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('fair', '2 · Is the comparison fair?', `
    <p>This comparison is not perfectly fair. The two runs use different numbers of ${term('epoch', 'epochs')} and different ${term('learning-rate', 'learning rates')}. The MLP trains for ${T(`${MLP}.settings.epochs`, 'setup')} epochs, four times as many as the CNN’s ${T(`${F}.settings.epochs`, 'setup')}. Its learning rate of ${T(`${MLP}.settings.lr`, 'setup')} is half the CNN’s ${T(`${F}.settings.lr`, 'setup')}. Both use the ${term('optimizer')} AdamW. The CNN still wins, with about a third of the parameters.</p>
    <p>A run also depends on its random start, so TorC trained each network from three starts. The best MLP run made about ${T.human(`${SEEDS}.mlp.max_accuracy`, 'per100')} mistakes per hundred, and the worst CNN run about ${T.human(`${SEEDS}.cnn.min_accuracy`, 'per100')}. So the CNN was ahead with every start.</p>
    ${inTorc('The network in code', `
    <pre class="mc-code" tabindex="0" role="group" aria-label="The digit CNN in TorC"><code>${escapeHtml(CODE)}</code></pre>
    <p><em>B</em> is the number of images in a ${term('batch')}. The TorC example <code>mnist_cnn</code> trains this network for ${T(`${F}.settings.example.epochs`, 'setup')} epochs, and the run on this page for ${T(`${F}.settings.epochs`, 'setup')}.</p>`)}`),
    },
    {
      kind: 'text',
      html: () => sec('kernels', '3 · What the first kernels learned', `
    <p>Training started each kernel from random numbers. In the widget, each small square on top is a kernel, amber for positive weights and blue for negative ones. Under it is its feature map. A map lights up, with bright cells, where its kernel’s pattern appears.</p>`),
    },
    {
      kind: 'widget', id: 'maps', className: 'widget', label: 'The kernels of the first convolution and their feature maps on a test image',
      mount: (host, D) => mountCnnMaps(host, D, { kernelLabels: true }),
      tasks: (D, T) => [
        { do: `Pick image #${T(`${F}.shown.1.index`, 'setup')}. Look at kernel ${K_TOP[0]} and kernel ${K_TOP[1]}, which are blue in the top row.`,
          see: 'Both maps light up along the top edges of the strokes, where ink lies below dark.' },
        { do: `Look at kernel ${K_RIGHT}, which is amber on the left and blue on the right. Then pick other images.`,
          see: 'Its map lights up along the right edges of the strokes, wherever they are in the image.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('mixups', '4 · Which digits the CNN mixes up', `
    <p>A ${term('confusion-matrix')} counts how often a network read each true digit as each other digit. Both networks read some fours as nines. In this one run of each, the MLP did so with ${T(`${MLP}.confusion.matrix.${FOUR}.${NINE}`, 'contrast')} test images, and the CNN with ${T(`${F}.confusion.matrix.${FOUR}.${NINE}`, 'contrast')}.</p>
`),
    },
    {
      kind: 'quiz', id: 'four-nine', purpose: 'check', type: 'choice',
      prompt: '<p>Does this one difference show that kernels help to tell fours from nines?</p>',
      options: [
        { html: 'No. A difference of a few images can come from the random start.', correct: true,
          feedback: 'Yes. One run shows how often, not why.' },
        { html: 'Yes, because the CNN made fewer mistakes on this pair.',
          feedback: 'A few images can differ by chance. Only a difference over several starts is solid.' },
        { html: 'Yes, because kernels find a short stroke at any position.',
          feedback: 'That is a possible reason, but one count cannot prove it.' },
      ],
      reveal: (D, T) => `<p>The CNN read sevens as twos ${T(`${F}.confusion.matrix.${SEVEN}.${TWO}`, 'reveal')} times, and the MLP ${T(`${MLP}.confusion.matrix.${SEVEN}.${TWO}`, 'reveal')} times. Only the total held with all three starts.</p>`,
    },
    {
      kind: 'widget', id: 'pad', className: 'widget', label: 'Interactive: draw a digit and see the probability of each digit, from the CNN',
      mount: (host, D) => mountDigitPad(host, D, { file: 'mnist-cnn', forward: forwardCnn, name: 'The CNN', torcNote: false }),
      tasks: () => [
        { do: 'Press each of the last four image buttons above the pad.',
          see: `The bars show the ${term('probability')} of each digit. The longest bar is never the label, so the CNN reads all four wrongly.` },
        { do: 'Draw a digit with a mouse, a pen or a finger, near one side of the pad.',
          see: `The small image shows your drawing moved to the centre, as in ${term('mnist', 'MNIST')}. So the moves of <a href="../convolution/">Convolution</a> do not happen here.` },
      ],
    },
    {
      kind: 'quiz', id: 'bigger', purpose: 'transfer', type: 'choice',
      prompt: '<p>You train the same CNN on images twice as wide and twice as tall. What happens to its parameters?</p>',
      options: [
        { html: 'The convolution layers keep theirs, but the last linear layer gets more.', correct: true,
          feedback: 'Yes. The maps get larger, so the linear layer reads more cells.' },
        { html: 'Nothing changes anywhere, because kernels do not depend on the image size.',
          feedback: 'The last layer has a weight for every cell of the larger maps.' },
        { html: 'Every layer gets about four times as many.',
          feedback: `Not even in the MLP. There only the first layer would grow, because it has a weight per ${term('pixel')}.` },
      ],
      reveal: '<p>The maps get twice as wide and tall, so the linear layer reads four times as many cells.</p>',
    },
    {
      kind: 'quiz', id: 'choose-lr', purpose: 'cumulative', from: 'generalization', type: 'choice',
      prompt: '<p>A friend tries ten learning rates for the CNN and keeps the best one on the test images. What is wrong?</p>',
      options: [
        { html: `The test images then help to choose the ${term('model')}, so they no longer measure new digits.`, correct: true,
          feedback: `Yes. Choose with a ${term('validation-set')}, and keep the test set for the end.` },
        { html: 'Nothing, because the test images are the most honest measure.',
          feedback: 'They are honest only while nothing is chosen with them.' },
        { html: 'Ten learning rates are too few.',
          feedback: 'The number is not the problem.' },
      ],
      reveal: '<p>Measure on the test set once, at the end.</p>',
    },
    {
      kind: 'explain', id: 'why-cnn',
      prompt: '<p>Explain why the CNN needs fewer parameters than the MLP and still reads digits better.</p>',
      model: '<p>Its kernels use the same few weights at every position, so each kernel finds its stroke anywhere.</p>',
    },
    {
      kind: 'text',
      html: () => sec('review', 'The whole course in review', `
    <p>This is the last lesson about images, and the epilogue follows. Answer each question, then open the answer. If one is hard, go back to its lesson.</p>
    <ul class="mc-review">
      <li>${review('What it means to learn', `What does the ${term('loss')} measure?`, 'How far the outputs are from the right answers.', 'what-is-learning', 'What does it mean for a machine to learn?')}</li>
      <li>${review(`The ${term('gradient')} and ${term('gradient-descent', 'gradient descent')}`, 'Why does a step go against the gradient?', 'The loss grows in the direction of the gradient.', 'why-gradients', 'The gradient')}</li>
      <li>${review(term('neural-network', 'Neural networks'), `What do ${term('softmax')} and ${term('cross-entropy', 'cross-entropy')} do?`, 'Softmax turns scores into probabilities, and cross-entropy measures the surprise at the right answer.', 'losses', 'Softmax and cross-entropy')}</li>
    </ul>`),
    },
    {
      kind: 'text',
      html: () => `<section class="mc" aria-label="The whole course in review, continued">
    <ul class="mc-review">
      <li>${review('Neural networks', `How does ${term('backpropagation')} find every gradient?`, 'It goes backward once and reuses each result.', 'backprop', 'How the computer finds every gradient at once')}</li>
      <li>${review('Training in practice', `What does ${term('adam', 'Adam')} change in a step?`, 'Each parameter gets its own step size.', 'optimizers', 'Better optimizers')}</li>
      <li>${review('Reading handwritten digits', 'Why does a feature map move with the digit?', 'The kernel has the same weights everywhere.', 'convolution', 'Convolution')}</li>
    </ul>
    <p>The epilogue asks how far this loop reaches, from digits to ChatGPT.</p></section>`,
    },
    {
      kind: 'answer',
      html: () => `<p>It reads them better. The CNN makes fewer mistakes than the MLP with about a third of the parameters, with every random start. The runs differ in epochs and learning rate, so the comparison is not perfectly fair.</p>
    <p>Three of its eight first kernels clearly follow the edges of strokes. It still mixes up some digits, and one run cannot tell why.</p>`,
      next: () => '<p>How far does the same loop reach, from digits to ChatGPT?</p>',
    },
  ],
};
