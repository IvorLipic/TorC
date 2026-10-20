// Lesson "mnist-mlp": the project that puts the course together. TorC trains the model of examples/mnist_mlp on MNIST; the page
// shows the data, the model, the training result, the first layer's weights, and a digit pad that runs the same forward pass with
// TorC's weights. Every number comes from data/mnist-mlp.json (tools/trace/trace_mnist_mlp.cpp) through T().
// Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { mountDigitPad, mountWeightsGallery } from '../widgets/digit-widgets.js';

const F = 'files.mnist-mlp';
const sec = (id, title, body) => `<section class="mm" aria-labelledby="mm-${id}"><h2 id="mm-${id}">${title}</h2>${body}</section>`;
const lastEpoch = (D) => D.files['mnist-mlp'].setup.epochs;
const worst = (D) => { const c = D.files['mnist-mlp'].test_per_class; return c.correct.reduce((b, v, k) => (v / c.total[k] < c.correct[b] / c.total[b] ? k : b), 0); };

const STYLE = `<style>
.mm p{max-width:44rem}
.mm h2{margin-top:2rem}
.mm-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.mm-code code{background:none;padding:0}
</style>`;

export default {
  id: 'mnist-mlp',
  title: 'Digits with an MLP',
  trace: ['mnist-mlp.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Digits with an MLP</h1>
    <p class="lead">This project puts the course together. TorC trains a network on images of handwritten digits. Then you can draw a digit and see what the network reads.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'Each image is a grid of pixels. The model reads it as one long row of numbers.',
        'Two hidden layers with ReLU and an output layer give one logit for each digit.',
        'TorC trains the model with cross-entropy and AdamW, one batch at a time.',
        'The accuracy on the test images shows how well the model works on new digits.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => { const s = D.files['mnist-mlp'].setup; return sec('data', '1 · The data', `
    <p>MNIST has ${T(`${F}.setup.train_samples`)} training images and ${T(`${F}.setup.test_samples`)} test images of handwritten digits. Each image has <code>${s.image_side} × ${s.image_side}</code> pixels with values from zero to <code>${s.pixel_max}</code>. Each line of the CSV file holds the label and the pixels of one image. <code>data::MNISTDataset</code> divides each pixel by <code>${s.pixel_max}</code>, so the inputs are between zero and one.</p>
    <pre class="mm-code" tabindex="0" role="group" aria-label="The MNIST data in TorC"><code>data::MNISTDataset train_set("mnist_train.csv");   // (60000, 784) and labels
data::MNISTDataset test_set("mnist_test.csv");
data::DataLoader loader(train_set, 128, /*shuffle=*/true);</code></pre>`); },
    },
    {
      kind: 'text',
      html: (D, T) => sec('model', '2 · The model', `
    <pre class="mm-code" tabindex="0" role="group" aria-label="The model in TorC"><code>nn::Sequential model;
model.add(std::make_unique&lt;nn::Linear&gt;(784, 32));
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Linear&gt;(32, 32));
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Linear&gt;(32, 10));   // one logit for each digit
optim::AdamW opt(model.parameters(), 5e-4f);
nn::CrossEntropyLoss loss_fn;</code></pre>
    <p>This is the model of the TorC example <code>mnist_mlp</code>. It has ${T(`${F}.setup.parameters`)} parameters. The first layer has one weight row for each of its neurons, with one weight for each pixel. So you can show each row as an image.</p>`),
    },
    {
      kind: 'widget', id: 'weights', className: 'widget', label: 'The rows of the first weight matrix, drawn as images',
      mount: (host, D) => mountWeightsGallery(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => { const E = lastEpoch(D); return sec('train', '3 · Training', `
    <p>TorC trains the model for ${T(`${F}.setup.epochs`)} epochs with batches of ${T(`${F}.setup.batch_size`)} images. That is ${T(`${F}.epochs.${E}.steps`)} steps. After the first epoch, the model reads ${T(`${F}.epochs.1.test_correct`)} of the test images correctly. After the last epoch, it reads ${T(`${F}.epochs.${E}.test_correct`)} correctly. The share of correct answers is the accuracy.</p>`); },
    },
    {
      kind: 'predict', id: 'p-overfit', title: 'Before you try it',
      prompt: (D, T) => `<p>The training loss falls to ${T(`${F}.epochs.${lastEpoch(D)}.train_loss`)}. Does the model overfit, as in the lesson Generalization?</p>`,
      reveal: (D, T) => `<p>Not in these epochs. The test accuracy rises in almost every epoch, from ${T(`${F}.epochs.1.test_correct`)} to ${T(`${F}.epochs.${lastEpoch(D)}.test_correct`)} correct images. The model has many parameters, but it also has many training images.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('try', '4 · Draw a digit', `
    <p>Draw a digit in the black square. The page runs the same forward pass as TorC, with the weights that TorC learned. First it scales and moves your drawing the way MNIST scales and moves its digits. The buttons load test images. The last four are images that the model reads wrongly.</p>`),
    },
    {
      kind: 'widget', id: 'pad', className: 'widget', label: 'Interactive: draw a digit and see the probability of each digit',
      mount: (host, D) => mountDigitPad(host, D),
    },
    {
      kind: 'check', id: 'c-worst',
      prompt: () => '<p>For each digit, the model reads a different share of the test images correctly. How can you find the digit that it reads worst?</p>',
      reveal: (D, T) => { const k = worst(D); return `<p>Divide the correct images of each digit by all test images of that digit, and take the smallest share. Here it is the digit <code>${k}</code>, with ${T(`${F}.test_per_class.correct.${k}`)} of ${T(`${F}.test_per_class.total.${k}`)} images correct.</p>`; },
    },
    {
      kind: 'text',
      html: () => '<section class="mm"><p>The next lesson, <a href="../convolution/">Convolution</a>, shows a layer that looks at small areas of an image. The lesson after it trains a smaller network on the same digits.</p></section>',
    },
  ],
};
