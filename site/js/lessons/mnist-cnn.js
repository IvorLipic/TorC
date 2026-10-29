// Lesson "mnist-cnn": the second project. TorC trains the CNN of examples/mnist_cnn on MNIST; the page compares it with the MLP
// of the lesson mnist-mlp, shows what the first convolution learned, and runs the same forward pass on a drawing. Every number
// comes from data/mnist-cnn.json (tools/trace/trace_mnist_cnn.cpp) and data/mnist-mlp.json through T().
// Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { mountDigitPad, mountCnnMaps, forwardCnn } from '../widgets/digit-widgets.js';

const F = 'files.mnist-cnn', MLP = 'files.mnist-mlp';
const sec = (id, title, body) => `<section class="mc" aria-labelledby="mc-${id}"><h2 id="mc-${id}">${title}</h2>${body}</section>`;
const last = (D, f) => D.files[f].setup.epochs;

const STYLE = `<style>
.mc p{max-width:44rem}
.mc h2{margin-top:2rem}
.mc-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.mc-code code{background:none;padding:0}
</style>`;

export default {
  id: 'mnist-cnn',
  title: 'Digits with a CNN',
  trace: ['mnist-cnn.json', 'mnist-mlp.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Digits with a CNN</h1>
    <p class="lead">This project trains the CNN of the lesson Convolution on the same handwritten digits. You compare it with the MLP, and you see what its first layer learned.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'The CNN reads the image as a grid, so each kernel sees pixels that are near each other.',
        'It has fewer parameters than the MLP, but it reads more test images correctly.',
        'Each kernel of its first layer responds to a simple pattern of light and dark.',
        'Each training step takes longer, because each kernel works at every position.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('train', '1 · The model and the training', `
    <pre class="mc-code" tabindex="0" role="group" aria-label="The CNN training setup in TorC"><code>// each sample has the shape (1, 28, 28)
data::MNISTDataset train_set("mnist_train.csv", 0, {1, 28, 28});
model.add(std::make_unique&lt;nn::Conv2d&gt;(1, 8, 3, 2, 1));
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Conv2d&gt;(8, 16, 3, 2, 1));
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Flatten&gt;());
model.add(std::make_unique&lt;nn::Linear&gt;(16 * 7 * 7, 10));
optim::AdamW opt(model.parameters(), 1e-3f);</code></pre>
    <p>The last argument of <code>MNISTDataset</code> gives each sample the shape of an image with one channel. The model has ${T(`${F}.setup.parameters`)} parameters. TorC trains it for ${T(`${F}.setup.epochs`)} epochs with batches of ${T(`${F}.setup.batch_size`)} images. The TorC example trains for more epochs. In TorC, each epoch of the CNN takes much longer than an epoch of the MLP, because <code>conv2d</code> is a plain loop over every position.</p>`),
    },
    {
      kind: 'predict', id: 'p-compare', title: 'Before you try it',
      prompt: (D, T) => `<p>The MLP has ${T(`${MLP}.setup.parameters`)} parameters, and the CNN has ${T(`${F}.setup.parameters`)}. Which one reads more test images correctly?</p>`,
      reveal: (D, T) => `<p>The CNN. After ${T(`${F}.setup.epochs`)} epochs, it reads ${T(`${F}.epochs.${last(D, 'mnist-cnn')}.test_correct`)} of the test images correctly. The MLP reads ${T(`${MLP}.epochs.${last(D, 'mnist-mlp')}.test_correct`)} after ${T(`${MLP}.setup.epochs`)} epochs. This comparison is not perfectly fair, because the two runs use different numbers of epochs and different learning rates. The CNN still wins, with about a third of the parameters. Each kernel of the CNN uses the same weights at every position. So the model needs few parameters to find a stroke anywhere in the image.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('kernels', '2 · What the first layer learned', `
    <p>The first convolution has one kernel for each of its output channels. TorC started the kernels from random values. After training, each kernel responds to its own pattern of light and dark. Pick a test image to see the feature map of each kernel. A bright area shows where the kernel finds its pattern.</p>`),
    },
    {
      kind: 'widget', id: 'maps', className: 'widget', label: 'The kernels of the first convolution and their feature maps on a test image',
      mount: (host, D) => mountCnnMaps(host, D),
    },
    {
      kind: 'check', id: 'c-size',
      prompt: (D) => { const s = D.files['mnist-cnn'].setup.image_side; return `<p>The input image has <code>${s} × ${s}</code> pixels. Why does each feature map of the first layer have fewer values?</p>`; },
      reveal: (D, T) => `<p>The first convolution has stride two, so it moves the kernel two pixels at a time. Each feature map has ${T(`${F}.first_maps.shape.2`)} rows and ${T(`${F}.first_maps.shape.3`)} columns.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('try', '3 · Draw a digit', `
    <p>The page runs the forward pass of the CNN with the weights that TorC learned, on your drawing or on a test image. The last four test images are images that the CNN reads wrongly. The MLP reads most of them wrongly too.</p>`),
    },
    {
      kind: 'widget', id: 'pad', className: 'widget', label: 'Interactive: draw a digit and see the probability of each digit, from the CNN',
      mount: (host, D) => mountDigitPad(host, D, { file: 'mnist-cnn', forward: forwardCnn }),
    },
    {
      kind: 'check', id: 'c-worst',
      prompt: () => '<p>Which digit does the CNN read worst on the test images?</p>',
      reveal: (D, T) => { const c = D.files['mnist-cnn'].test_per_class, k = c.correct.reduce((b, v, i) => (v / c.total[i] < c.correct[b] / c.total[b] ? i : b), 0); return `<p>The digit <code>${k}</code>, with ${T(`${F}.test_per_class.correct.${k}`)} of ${T(`${F}.test_per_class.total.${k}`)} images correct.</p>`; },
    },
    {
      kind: 'text',
      html: () => '<section class="mc"><p>This is the last lesson of the course. Go back to the <a href="../../">course map</a>, or try the <a href="../backprop-lab/">Backprop lab</a>.</p></section>',
    },
  ],
};
