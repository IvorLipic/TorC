// Lesson "mnist-mlp" (Part 5 `images`, curriculum v3, template 3), title "Reading handwritten digits with a network" (brief §6 and
// §7, C2, C5, D9). What MNIST is (anchors as setup values), pixels and channels (red, green, blue, ahead of convolution), the digit
// network with its real line (brief §2), the digit pad, the first-layer weights with guidance on what to look for, overfitting
// diagnosed from training vs validation mistakes (the validation run, C2), the test set reported once, the spread over three
// random starts (C5, data/mnist-seeds.json), the confusion matrix and the grid of the first mistakes ("why might a four and a nine
// get confused?"), and a worked row of the matrix. Every number comes from data/mnist-mlp.json, data/mnist-mistakes.json and
// data/mnist-seeds.json through a role-tagged T() (docs/authoring.md "Number roles"); accuracies are mistakes per 100 (T.human).
// Tests: site/test/mnist-mlp.test.mjs. Notes: docs/lessons/mnist-mlp.md.
import { worked, realLine, term } from '../core/blocks.js';
import { mountDigitPad, mountWeightsGallery } from '../widgets/digit-widgets.js';
import { mountTrainValidation, mountConfusion } from '../widgets/mnist-widgets.js';

const F = 'files.mnist-mlp';
const V = `${F}.validation_run`;
const S = 'files.mnist-seeds';
export const LAST = 20;                      // the last pass of both runs (the test pins setup.epochs and validation_run.epochs)
export const ROW = 4, MINE = 9;              // the worked row of the matrix and the row the learner reads
const sec = (id, title, body) => `<section class="mm" aria-labelledby="mm-${id}"><h2 id="mm-${id}">${title}</h2>${body}</section>`;
const plain = (body) => `<section class="mm">${body}</section>`;
const per100 = (T, path) => T.human(path, 'per100');
// the digit network as in examples/mnist_mlp (the test checks the shapes against the trace)
export const MODEL_CODE = 'nn::Sequential model;\n// 784 pixel values in, 32 neurons out\nmodel.add(std::make_unique<nn::Linear>(784, 32));\nmodel.add(std::make_unique<nn::ReLU>());\nmodel.add(std::make_unique<nn::Linear>(32, 32));\nmodel.add(std::make_unique<nn::ReLU>());\n// 32 neurons in, ten scores out, one for each digit\nmodel.add(std::make_unique<nn::Linear>(32, 10));';

const STYLE = `<style>
.mm p{max-width:44rem}
.mm h2{margin-top:2rem}
</style>`;

export default {
  id: 'mnist-mlp',
  title: 'Reading handwritten digits with a network',
  trace: ['mnist-mlp.json', 'mnist-mistakes.json', 'mnist-seeds.json'],
  template: 3,
  loop: 'all',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Reading handwritten digits with a network</h1>
    <p class="lead">A network learns to read handwritten digits from many examples. Then you count its mistakes on digits that it never saw.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'A network learns from thousands of handwritten digits. How often does it read a new digit wrongly?',
        'Which digits does it mix up, and why those?',
        `The network has many ${term('parameter', 'parameters')}. Does it only memorize the images that it learned from?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('digits', 'The digits', `
    <p>${term('mnist', 'MNIST')} is a well-known ${term('data-set')} of handwritten digits. It has ${T(`${F}.setup.train_samples`, 'setup')} images for ${term('training')} and ${T(`${F}.setup.test_samples`, 'setup')} test images, each with its digit.</p>
    <p>An image is a grid of ${T(`${F}.setup.image_side`, 'setup')} by ${T(`${F}.setup.image_side`, 'setup')} ${term('pixel', 'pixels')}. Each pixel has a grey value from zero for black to ${T(`${F}.setup.pixel_max`, 'setup')} for white. TorC divides it by ${T(`${F}.setup.pixel_max`, 'setup')}, so every input is between zero and one.</p>
    <p>One grid of values is a ${term('channel')}. A colour photo has three channels, for red, green and blue. An MNIST image has one grey channel.</p>
    <p>The network gets the ${T(`${F}.setup.pixels`, 'setup')} grey values as one long row of numbers.</p>`),
    },
    {
      kind: 'quiz', id: 'how-many', purpose: 'predict', type: 'choice',
      prompt: '<p>The network trains until it reads almost all its training images correctly. How many mistakes per hundred new images do you expect?</p>',
      options: [
        { html: 'None, because it learned each digit.', feedback: 'New handwriting still fools it.' },
        { html: 'A few, a little more than on its training images.', correct: true, feedback: 'Yes.' },
        { html: 'About half, because it only memorized its images.', feedback: 'This is the belief that a large network only memorizes. It does much better.' },
      ],
      reveal: (D, T) => `<p>On new images it makes about ${per100(T, `${V}.epochs.${LAST}.validation.accuracy`)} mistakes per hundred.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('model', '1 · The network', `
    <p>The network has three ${term('layer', 'layers')} of ${term('neuron', 'neurons')}. The first two have ${T(`${F}.layers.0.weight_shape.0`, 'setup')} neurons each, with ${term('relu', 'ReLU')}. The last layer gives ten scores, one for each digit. ${term('softmax', 'Softmax')} turns them into ${term('probability', 'probabilities')}, and the most probable digit is the answer.</p>
    <p>Such a network is a multilayer perceptron, or MLP. It has ${T.anchor('mlp_parameters')} parameters, as you counted in the lesson on <a href="../linear-layer/">the layer</a>.</p>
    ${realLine(MODEL_CODE, 'This is the digit network in TorC. Each Linear line is one layer of neurons, and a ReLU line sits between two layers.')}
    <p>The ${term('loss')} is the ${term('cross-entropy', 'cross-entropy')}, and the ${term('optimizer')} is AdamW. Each step uses a ${term('batch')} of ${T(`${F}.setup.batch_size`, 'setup')} images. Training goes through all training images ${T(`${F}.setup.epochs`, 'setup')} times, once per ${term('epoch')}.</p>`),
    },
    {
      kind: 'widget', id: 'pad', className: 'widget', label: 'Interactive: draw a digit and see the probability of each digit',
      mount: (host, D) => mountDigitPad(host, D, { markWrong: true, torcNote: false }),
      tasks: () => [
        { do: 'Press Clear and draw a digit small, in a corner.',
          see: 'The small image shows what the network reads. The page scales your digit and moves it to the middle, as MNIST does.' },
        { do: 'Press the buttons with a red edge.',
          see: 'The network reads these test images wrongly. Each looks a little like the digit that it chose.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('weights', '2 · What the first layer looks for', `
    <p>Each neuron of the first layer has one ${term('weight')} for each pixel. So its weights make an image of ${T(`${F}.setup.image_side`, 'setup')} by ${T(`${F}.setup.image_side`, 'setup')} squares.</p>
    <p>Amber is a positive weight, so ink there raises the neuron. Blue is a negative weight, so ink there lowers it.</p>
    <p>Do not expect digits. A neuron reacts to pieces of strokes, and the next layers combine the pieces.</p>`),
    },
    {
      kind: 'widget', id: 'weights', className: 'widget', label: 'The weights of the first layer, one image for each neuron',
      mount: (host, D) => mountWeightsGallery(host, D),
      tasks: () => [
        { do: 'Look at the middle of a few images.',
          see: 'Neighbouring pixels there often share a colour. They form patches and parts of strokes.' },
        { do: 'Compare the middle of an image with its edge.',
          see: `The edges look more speckled. Few digits have ink there, so these weights rarely get a ${term('gradient')}. The next lesson shows why.` },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('memorize', '3 · Learned, or only memorized?', `
    <p>In the lesson on <a href="../generalization/">new data</a>, a curve fitted the noise of its ten points. This network has far more parameters. Does it only memorize its training images?</p>
    <p>To find out, TorC trains the same network again on ${T(`${V}.train_samples`, 'setup')} training images. It keeps the last ${T(`${V}.validation_samples`, 'setup')} apart as the ${term('validation-set', 'validation set')}.</p>
    <p>After each epoch, it counts the mistakes on the validation images. It also counts them on the first ${T(`${V}.train_subset_samples`, 'setup')} training images, as many as in the validation set.</p>`),
    },
    {
      kind: 'widget', id: 'train-validation', className: 'widget', label: 'Interactive: mistakes per hundred images after each epoch, on training images and on validation images',
      mount: (host, D) => mountTrainValidation(host, D),
      tasks: () => [
        { do: 'Set the slider to the first epoch, then the tenth, then the last.',
          see: 'At first the two curves are close. Later the training curve is lower, and the gap between them grows slowly.' },
        { do: 'Look at the validation curve over the last few epochs.',
          see: 'It flattens and even rises a little, while the training curve keeps falling.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => plain(`
    <p>After the last epoch, the network makes about ${per100(T, `${V}.epochs.${LAST}.train_subset.accuracy`)} mistakes per hundred training images and about ${per100(T, `${V}.epochs.${LAST}.validation.accuracy`)} per hundred validation images. The gap is small, so the network ${term('overfitting', 'overfits')} only a little. It has many parameters, but also many images.</p>
    <p>The two sets hold different images, so their counts differ a little by chance. The gap that opens later is what counts.</p>
    <p>After training, this run reads the ${term('test-set', 'test set')} once. It makes about ${per100(T, `${V}.final_test.accuracy`)} mistakes per hundred test images.</p>
    <p>The network of the digit pad trains on all training images. From three random starts, it read between ${T(`${S}.mlp.min_accuracy`, 'contrast', { sig: 4 })} and ${T(`${S}.mlp.max_accuracy`, 'contrast', { sig: 4 })} of the test images correctly. These runs also measured the test images after each epoch, but this course chooses nothing with those counts.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('confusion', '4 · Which digits it mixes up', `
    <p>A ${term('confusion-matrix', 'confusion matrix')} counts the answers of the network for each true digit. Below it is a heat map. Each row is a true digit, and each column is an answer. Green squares are right answers. Amber squares are mistakes, brighter for more images.</p>
    <p>We only look at these test counts, and we choose nothing with them.</p>`),
    },
    {
      kind: 'widget', id: 'confusion', className: 'widget', label: 'Interactive: the confusion matrix of the test images and the first mistakes',
      mount: (host, D) => mountConfusion(host, D),
      tasks: () => [
        { do: 'Find the brightest amber square, and select it.',
          see: 'It is a four read as a nine. A nine read as a four is less common.' },
        { do: 'Press one of the images below the squares.',
          see: 'The ring moves to the square of its pair.' },
      ],
    },
    {
      kind: 'text',
      html: () => plain(`
    <p>Why might a four and a nine get confused? Both have a stroke down the right side and a closed part at the top. When the top lines of a four almost meet, its pixels are close to those of a nine. Similar pixels give similar scores. The first mistake in the grid is such a four.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('row', '5 · Reading one row', `
    <p>One row of the ${term('matrix')} holds every test image of one digit.</p>
    ${worked(`<p>Row four holds all ${T(`${F}.test_per_class.total.${ROW}`, 'worked')} test images of a four. Its diagonal square counts ${T(`${F}.confusion.matrix.${ROW}.${ROW}`, 'worked')} right answers. So the network reads ${T(`${F}.confusion.matrix.${ROW}.${ROW}`, 'worked')} / ${T(`${F}.test_per_class.total.${ROW}`, 'worked')} = <strong>${T(`${F}.test_per_class.accuracy.${ROW}`, 'worked')}</strong> of the fours correctly.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'nines', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${F}.test_per_class.accuracy.${MINE}`, tolerance: 0.002,
      prompt: '<p>Now find row nine in the matrix above. Add up its squares to get all the test images of a nine. What fraction of them does the network read correctly? A calculator is fine.</p>',
      reveal: (D, T) => `${worked(`<p>${T(`${F}.confusion.matrix.${MINE}.${MINE}`, 'worked')} / ${T(`${F}.test_per_class.total.${MINE}`, 'worked')} = ${T(`${F}.test_per_class.accuracy.${MINE}`, 'worked')}</p>`)}<p>Nines are the hardest row.</p>`,
    },
    {
      kind: 'quiz', id: 'letters', purpose: 'transfer', type: 'choice',
      prompt: '<p>A network like this one learns handwritten capital letters. Which pair will it mix up most often?</p>',
      options: [
        { html: 'O and Q', correct: true, feedback: 'Yes. A Q is an O with one short stroke more.' },
        { html: 'O and L, because both letters are common', feedback: 'Common letters do not look alike.' },
        { html: `Every pair equally often, because each letter is its own ${term('class')}`, feedback: 'Letters with similar pixels still get similar scores.' },
      ],
      reveal: '<p>Letters that share most of their pixels get similar scores.</p>',
    },
    {
      kind: 'quiz', id: 'steps', purpose: 'cumulative', from: 'training-loop', type: 'numeric',
      answer: `${F}.settings.steps_per_epoch`, tolerance: 0.3,
      prompt: (D, T) => `<p>Training takes the ${T(`${F}.settings.train_samples`, 'setup')} training images in batches of ${T(`${F}.settings.batch_size`, 'setup')}. How many training steps does one epoch take? A calculator is fine.</p>`,
      reveal: (D, T) => `<p>If your answer has a fraction, round it up. A last, smaller batch takes the images that are left, so one epoch takes ${T(`${F}.settings.steps_per_epoch`, 'reveal')} steps. All ${T(`${F}.settings.epochs`, 'reveal')} epochs take ${T(`${F}.settings.steps`, 'reveal')} steps.</p>`,
    },
    {
      kind: 'explain', id: 'memorize',
      prompt: '<p>Explain how you can tell that a network does not only memorize its training images.</p>',
      model: '<p>Its mistakes on images that it never trained on stay close to its mistakes on its training images.</p>',
    },
    {
      kind: 'answer',
      html: (D, T) => `<p>On test images that it never saw, the network makes about ${per100(T, `${F}.final.test_accuracy`)} mistakes per hundred. Its mistakes gather in digits with similar strokes, such as a four and a nine.</p>
    <p>Its mistakes on the validation images stay close to those on its training images. So it learned rules that work on new digits, and it overfits only a little.</p>`,
      next: () => '<p>What happens to the numbers of the network during one training step?</p>',
    },
  ],
};
