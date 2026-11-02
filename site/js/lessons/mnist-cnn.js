// Lesson "mnist-cnn" (Part 5 `images`, curriculum v3, template 3; brief §6 and §7 "mnist-cnn", C5, C9). TorC trains the CNN of
// examples/mnist_cnn on MNIST. The page compares it with the MLP and says plainly that the comparison is not perfectly fair (C9),
// with the spread over three random starts (C5, mnist-seeds.json); shows what the first kernels learned; asks why the CNN mixes up
// fours and nines less often (a reasoning question that the confusion matrices support); runs the CNN on a drawing; and ends the
// course with a recap of every part and the bridge to the epilogue. It is the last lesson of Part 5, so the recap doubles as the
// part's mixed review (it recommends going back and never blocks).
// Every number comes from data/mnist-cnn.json (trace_mnist_cnn.cpp), data/mnist-mlp.json (trace_mnist_mlp.cpp),
// data/mnist-seeds.json (trace_mnist_seeds.cpp) and the anchors through a role-tagged T() (docs/authoring.md "Number roles").
// Notes: docs/lessons/mnist-cnn.md.
import { worked, inTorc, term } from '../core/blocks.js';
import { mountDigitPad, mountCnnMaps, forwardCnn } from '../widgets/digit-widgets.js';

const F = 'files.mnist-cnn', MLP = 'files.mnist-mlp', SEEDS = 'files.mnist-seeds';
const FOUR = 4, NINE = 9, SEVEN = 7, TWO = 2;          // the confusion pairs the page discusses (tested against the matrices)
const sec = (id, title, body) => `<section class="mc" aria-labelledby="mc-${id}"><h2 id="mc-${id}">${title}</h2>${body}</section>`;

const STYLE = `<style>
.mc p{max-width:44rem}
.mc h2{margin-top:2rem}
.mc-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.mc-code code{background:none;padding:0}
.mc-review{margin:.3rem 0 .3rem 1.2rem;padding:0;max-width:42rem}
.mc-review li{margin:.35rem 0}
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
    <p>The second convolution layer has ${T(`${F}.layers.1.weight_shape.0`, 'setup')} kernels, and each reads all the maps of the first. A ${term('relu', 'ReLU')} follows each of the two layers. Then a linear layer reads all ${T(`${F}.layers.2.weight_shape.1`, 'setup')} cells and gives one score per digit.</p>
    <p>The CNN has ${T.anchor('cnn_parameters')} parameters, and the MLP has ${T.anchor('mlp_parameters')}. TorC trains both on the same ${T.anchor('train_images')} ${term('training')} images.</p>`),
    },
    {
      kind: 'quiz', id: 'compare', purpose: 'predict', type: 'choice',
      prompt: '<p>Which network makes fewer mistakes on the test images, which it never saw in training?</p>',
      options: [
        { html: 'The CNN', correct: true, feedback: 'Yes. It reads more test digits correctly, with fewer parameters.' },
        { html: 'The MLP, because more parameters can learn more', feedback: `More parameters do not mean better reading. The CNN uses its few ${term('weight', 'weights')} at every position.` },
        { html: 'About the same, because both learn from the same images', feedback: 'The images are the same, but the CNN reads them in another way, with kernels.' },
      ],
      reveal: (D, T) => `<p>The CNN makes about ${T.human(`${F}.final.test_correct`, 'per100', { of: `${F}.final.test_samples` })} mistakes per hundred test digits. The MLP makes about ${T.human(`${MLP}.final.test_correct`, 'per100', { of: `${MLP}.final.test_samples` })}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('count', '1 · Counting the mistakes', `
    <p>These counts come from the ${term('test-set')}, so they measure how well each network reads new digits. That is its ${term('generalization')}.</p>
    ${worked(`<p>The MLP reads ${T(`${MLP}.final.test_correct`, 'worked')} of the ${T(`${MLP}.final.test_samples`, 'worked')} test images correctly. So it reads ${T(`${MLP}.final.test_samples`, 'worked')} − ${T(`${MLP}.final.test_correct`, 'worked')} = <strong>${T(`${MLP}.final.test_mistakes`, 'worked')}</strong> wrongly, about three in every hundred.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'cnn-mistakes', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${F}.final.test_mistakes`, tolerance: 0.5,
      prompt: (D, T) => `<p>The CNN reads ${T(`${F}.final.test_correct`, 'setup')} of the ${T(`${F}.final.test_samples`, 'setup')} test images correctly. How many does it read wrongly?</p>`,
      reveal: (D, T) => `<p>It reads ${T(`${F}.final.test_mistakes`, 'reveal')} wrongly, about two in every hundred.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('fair', '2 · Is the comparison fair?', `
    <p>This comparison is not perfectly fair. The two runs use different numbers of ${term('epoch', 'epochs')} and different ${term('learning-rate', 'learning rates')}. The MLP trains for ${T(`${MLP}.settings.epochs`, 'setup')} epochs with a learning rate of ${T(`${MLP}.settings.lr`, 'setup')}. The CNN trains for ${T(`${F}.settings.epochs`, 'setup')} epochs with ${T(`${F}.settings.lr`, 'setup')}. The CNN still wins, with about a third of the parameters.</p>
    <p>One run also depends on its random start. So TorC trained each network from three random starts. The best MLP run made about ${T.human(`${SEEDS}.mlp.max_accuracy`, 'per100')} mistakes per hundred, and the worst CNN run about ${T.human(`${SEEDS}.cnn.min_accuracy`, 'per100')}. So the CNN was ahead with every start.</p>
    ${inTorc('The network in code', `
    <pre class="mc-code" tabindex="0" role="group" aria-label="The digit CNN in TorC"><code>using std::make_unique;
// (B, 1, 28, 28) becomes (B, 8, 14, 14)
model.add(make_unique&lt;nn::Conv2d&gt;(1, 8, 3, 2, 1));
model.add(make_unique&lt;nn::ReLU&gt;());
// (B, 8, 14, 14) becomes (B, 16, 7, 7)
model.add(make_unique&lt;nn::Conv2d&gt;(8, 16, 3, 2, 1));
model.add(make_unique&lt;nn::ReLU&gt;());
// (B, 16, 7, 7) becomes (B, 784)
model.add(make_unique&lt;nn::Flatten&gt;());
// 784 cells give 10 scores
model.add(make_unique&lt;nn::Linear&gt;(16 * 7 * 7, 10));</code></pre>
    <p>This is the network of the TorC example <code>mnist_cnn</code>. The example trains for ${T(`${F}.settings.example.epochs`, 'setup')} epochs, and the run on this page trains for ${T(`${F}.settings.epochs`, 'setup')}.</p>`)}`),
    },
    {
      kind: 'text',
      html: () => sec('kernels', '3 · What the first kernels learned', `
    <p>Training started each kernel from random numbers. Now each kernel lights up on its own pattern. In the widget, each small square on top is a kernel, amber for positive weights and blue for negative ones. Under it is its feature map.</p>`),
    },
    {
      kind: 'widget', id: 'maps', className: 'widget', label: 'The kernels of the first convolution and their feature maps on a test image',
      mount: (host, D) => mountCnnMaps(host, D),
      tasks: (D, T) => [
        { do: `Pick image #${T(`${F}.shown.1.index`, 'setup')}. Find a kernel that is blue in its top row and amber below it.`,
          see: 'Its map lights up along the top edges of the strokes, where ink lies below dark.' },
        { do: 'Find the kernel that is amber on the left and blue on the right. Then pick other images.',
          see: 'Its map lights up along the right edges of the strokes, wherever they are in the image.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('mixups', '4 · Which digits the CNN mixes up', `
    <p>A ${term('confusion-matrix')} counts how often a network read each true digit as each other digit. Both networks read some fours as nines. The MLP did so with ${T(`${MLP}.confusion.matrix.${FOUR}.${NINE}`, 'contrast')} test images, the CNN with ${T(`${F}.confusion.matrix.${FOUR}.${NINE}`, 'contrast')}.</p>
`),
    },
    {
      kind: 'quiz', id: 'four-nine', purpose: 'check', type: 'choice',
      prompt: '<p>Many fours look like nines with an open top. A short stroke at the top makes the difference. Why might the CNN mix them up less often than the MLP?</p>',
      options: [
        { html: 'A kernel finds a short stroke at any position, and every training image teaches it at every position.', correct: true,
          feedback: 'Yes, this is a likely reason. The MLP must learn the stroke separately for each place.' },
        { html: 'The CNN has more parameters for the hard digits.', feedback: 'The CNN has fewer parameters than the MLP, about a third.' },
        { html: 'The CNN trained for more epochs.', feedback: 'The CNN trained for fewer epochs than the MLP.' },
        { html: 'The CNN saw these test images during training.', feedback: 'Neither network trains on the test images.' },
      ],
      reveal: (D, T) => `<p>These counts show how often, not why, so the first answer is a likely reason and not a proof. The CNN does not win on every pair. It read sevens as twos ${T(`${F}.confusion.matrix.${SEVEN}.${TWO}`, 'reveal')} times, the MLP ${T(`${MLP}.confusion.matrix.${SEVEN}.${TWO}`, 'reveal')} times.</p>`,
    },
    {
      kind: 'widget', id: 'pad', className: 'widget', label: 'Interactive: draw a digit and see the probability of each digit, from the CNN',
      mount: (host, D) => mountDigitPad(host, D, { file: 'mnist-cnn', forward: forwardCnn }),
      tasks: () => [
        { do: 'Pick each of the last four test images.', see: 'The CNN reads all four wrongly. Its highest bar is not the label.' },
        { do: 'Draw a digit near one side of the pad.', see: `The pad moves your drawing to the centre first, as the ${term('mnist', 'MNIST')} images are. So the CNN reads it as a centred digit.` },
      ],
    },
    {
      kind: 'quiz', id: 'bigger', purpose: 'transfer', type: 'choice',
      prompt: '<p>You train the same CNN on images twice as wide and twice as tall. What happens to its parameters?</p>',
      options: [
        { html: 'The convolution layers keep theirs, but the last linear layer gets more.', correct: true,
          feedback: 'Yes. The kernels do not depend on the image size. The maps get larger, so the linear layer reads more cells.' },
        { html: 'Nothing changes anywhere, because kernels do not depend on the image size.',
          feedback: 'The last layer has a weight for every cell of the larger maps.' },
        { html: 'Every layer gets about four times as many.',
          feedback: 'That holds for an MLP. A kernel has the same weights for any image size.' },
      ],
      reveal: '<p>The maps get twice as wide and twice as tall. So the linear layer reads four times as many cells.</p>',
    },
    {
      kind: 'quiz', id: 'choose-lr', purpose: 'cumulative', from: 'generalization', type: 'choice',
      prompt: '<p>A friend tries ten learning rates for the CNN and keeps the one with the best result on the test images. What is wrong with that?</p>',
      options: [
        { html: `The test images then help to choose the ${term('model')}, so they no longer measure new digits.`, correct: true,
          feedback: `Yes. Choose with a ${term('validation-set')}, and keep the test set for the end.` },
        { html: 'Nothing, because the test images are the most honest measure.',
          feedback: 'They are honest only while nothing is chosen with them.' },
        { html: 'Ten learning rates are too few.',
          feedback: 'The number is not the problem. The choice with the test images is.' },
      ],
      reveal: '<p>Choose with a validation set, and measure on the test set once, at the end.</p>',
    },
    {
      kind: 'explain', id: 'why-cnn',
      prompt: '<p>Explain why the CNN needs fewer parameters than the MLP and still reads digits better.</p>',
      model: '<p>Its kernels use the same few weights at every position, so each kernel finds its stroke anywhere, and every image teaches it everywhere.</p>',
    },
    {
      kind: 'text',
      html: () => sec('review', 'The whole course in review', `
    <p>This lesson ends the course on TorC. Each line names one part and asks one question. If a question is hard, go back to its lesson.</p>
    <ul class="mc-review">
      <li>What it means to learn. What does the ${term('loss')} measure? (<a href="../what-is-learning/">the first lesson</a>)</li>
      <li>The ${term('gradient')}. Why does a step go against it? (<a href="../why-gradients/">the gradient</a>)</li>
      <li>${term('neural-network', 'Neural networks')}. Why does a network need an ${term('activation-function')} between its layers? (<a href="../nonlinearity/">activation functions</a>)</li>
      <li>Training in practice. Why does a model need data that it never trains on? (<a href="../generalization/">generalization</a>)</li>
      <li>Reading digits. Why does a feature map move with the digit? (<a href="../convolution/">convolution</a>)</li>
    </ul>
    <p>The epilogue asks how far this loop of data, model, loss and adjust step reaches, from digits to ChatGPT.</p>`),
    },
    {
      kind: 'answer',
      html: () => `<p>It reads them better. The CNN makes fewer mistakes than the MLP with about a third of the parameters, and with every random start. The comparison is not perfectly fair, because the runs differ in epochs and learning rate.</p>
    <p>Its first kernels learned to find the edges of strokes. It still mixes up some fours and nines, and some sevens and twos.</p>`,
      next: () => '<p>How far does the same loop reach, from digits to ChatGPT?</p>',
    },
  ],
};
