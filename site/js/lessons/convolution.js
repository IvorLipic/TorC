// Lesson "convolution": a kernel that slides over an image, stride and padding with the output-size formula, channels and the
// shared weights of a kernel, and the layers of the CNN example. Every number and message comes from data/conv.json
// (tools/trace/trace_conv.cpp) through T() or the recorded text. Writing: docs/authoring.md "Writing rules" (ASD-STE100,
// STE-flavored mode).
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { mountSlidingWindow, mountFeatureMaps } from '../widgets/conv-widgets.js';

const F = 'files.conv';
const eq = (latex) => `<div class="cv-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="cv" aria-labelledby="cv-${id}"><h2 id="cv-${id}">${title}</h2>${body}</section>`;
const err = (D, id) => { const e = D.files.conv.errors.find((x) => x.id === id); return `<code>${escapeHtml(e.exception)}: ${escapeHtml(e.what)}</code>`; };
const shp = (a) => `<code>(${a.join(', ')})</code>`;
const conv1 = (D) => D.files.conv.cnn[0];
const wshape = (D) => conv1(D).params.find((p) => p.name === 'weight').shape;   // (C_out, C_in, K, K)

const STYLE = `<style>
.cv p{max-width:44rem}
.cv h2{margin-top:2rem}
.cv-eq{overflow-x:auto;margin:.2rem 0}
.cv-eq .katex-display{margin:.4em 0}
.cv-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.cv-code code{background:none;padding:0}
</style>`;

export default {
  id: 'convolution',
  title: 'Convolution',
  trace: ['conv.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Convolution</h1>
    <p class="lead">A linear layer connects every pixel to every output. A convolution layer looks at one small area of the image at a time, with the same weights everywhere. This lesson shows how it works in TorC.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'A kernel is a small grid of weights. It slides over the image, and at each position it gives one output value.',
        'Stride and padding set the size of the output.',
        'The layer uses each kernel at every position, so it has few parameters.',
        'Each output channel has one kernel for each input channel.',
      ],
    },
    {
      kind: 'text',
      html: () => sec('slide', '1 · A kernel slides over the image', `
    ${eq('y_{ij} = b + \\sum_{a}\\sum_{b\'} x_{\\,iS + a - P,\\; jS + b\' - P}\\; w_{a b\'}')}
    <p>The kernel ${math('w', false)} is a small grid of weights. At each position, TorC multiplies the pixels under the kernel by the kernel weights and adds the products and the bias. ${math('S', false)} is the stride and ${math('P', false)} is the padding. TorC does not flip the kernel. This kernel gives large values where the image changes from left to right, so it finds vertical edges.</p>`),
    },
    {
      kind: 'widget', id: 'window', className: 'widget', label: 'Interactive: the window of each output on a small input, for three settings of stride and padding',
      mount: (host, D) => mountSlidingWindow(host, D),
    },
    {
      kind: 'text',
      html: () => sec('size', '2 · Stride and padding', `
    ${eq('H_{\\text{out}} = \\left\\lfloor \\frac{H + 2P - K}{S} \\right\\rfloor + 1')}
    <p>The stride is the step of the kernel from one position to the next. The padding adds rows and columns of zeros around the image. ${math('H', false)} is the size of the input and ${math('K', false)} is the size of the kernel. Without padding, the output is smaller than the input. With stride two, the output has about half the size.</p>`),
    },
    {
      kind: 'check', id: 'c-size',
      prompt: (D) => { const c = conv1(D); return `<p>An image of size <code>${c.in_shape[2]}</code> goes through a convolution with kernel size <code>${c.kernel}</code>, stride <code>${c.stride}</code> and padding <code>${c.padding}</code>. Then the result goes through the same convolution again. What are the two output sizes?</p>`; },
      reveal: (D, T) => `<p>The first output has size ${T(`${F}.cnn.0.out_shape.2`)}, because the floor of the fraction is thirteen. The second has size ${T(`${F}.cnn.2.out_shape.2`)}.</p>`,
    },
    {
      kind: 'text',
      html: (D) => sec('channels', '3 · Channels', `
    <p>An image can have several channels, and a convolution layer can give several output channels. Each output channel has one kernel for each input channel. It adds the results of these kernels and one bias. So the weight has the shape <code>(C_out, C_in, K, K)</code>. If the input has another number of channels, TorC throws this error.</p>
    <p>${err(D, 'channels')}</p>
    <p>Each output channel is a feature map. Below, two kernels find the vertical and the horizontal edges of a digit.</p>`),
    },
    {
      kind: 'widget', id: 'maps', className: 'widget', label: 'Interactive: the feature maps of two edge kernels on a handwritten digit',
      mount: (host, D) => mountFeatureMaps(host, D),
    },
    {
      kind: 'predict', id: 'p-params', title: 'Before you try it',
      prompt: (D, T) => { const [co, ci, k] = wshape(D), s = conv1(D).in_shape[2]; return `<p>A linear layer from a <code>${s} × ${s}</code> image to <code>${co}</code> outputs has ${T(`${F}.dense_compare.parameters`)} parameters. How many parameters does <code>Conv2d(${ci}, ${co}, ${k})</code> have?</p>`; },
      reveal: (D, T) => { const [co, , k] = wshape(D); return `<p>It has ${T(`${F}.cnn.0.parameters`)}. Each of the <code>${co}</code> kernels has <code>${k} × ${k}</code> weights and one bias. The layer uses the same kernels at every position of the image.</p>`; },
    },
    {
      kind: 'text',
      html: (D, T) => sec('cnn', '4 · A small CNN', `
    <pre class="cv-code" tabindex="0" role="group" aria-label="The CNN of the TorC example"><code>model.add(std::make_unique&lt;nn::Conv2d&gt;(1, 8, 3, /*stride=*/2, /*padding=*/1));   // (N, 8, 14, 14)
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Conv2d&gt;(8, 16, 3, 2, 1));                         // (N, 16, 7, 7)
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Flatten&gt;());                                      // (N, 784)
model.add(std::make_unique&lt;nn::Linear&gt;(16 * 7 * 7, 10));</code></pre>
    <p>This is the model of the TorC example <code>mnist_cnn</code>. A network with convolution layers is a convolutional neural network, or CNN. <code>Flatten</code> turns each sample of shape ${shp(D.files.conv.cnn[3].out_shape.slice(1))} into one row for the linear layer. The model has ${T(`${F}.cnn_parameters`)} parameters, and most of them are in the last layer.</p>`),
    },
    {
      kind: 'text',
      html: () => '<section class="cv"><p>Go back to the <a href="../../">course map</a>. The next lesson trains this CNN on the handwritten digits.</p></section>',
    },
  ],
};
