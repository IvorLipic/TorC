// Lesson "convolution" (Part 5 `images`, curriculum v3, template 3; brief §6 and §7 "convolution", C8, D14). An MLP has its own
// weight for every pixel. A kernel uses the same few weights at every position, so it needs few parameters, and its feature map
// moves with the digit, exactly for a move by a whole number of strides (C8). The CNN is still not shift-invariant: the
// measured accuracy of both trained models on moved test images says so. The sliding window comes as a picture before the ⌊ ⌋
// rule. Channels, the parameter count, and "TorC does not flip the kernel" in the In TorC panel.
// Every number comes from data/conv.json (tools/trace/trace_conv.cpp), data/shift.json (trace_shift.cpp) and the anchors through
// a role-tagged T() (docs/authoring.md "Number roles"). The shift widgets use the weights of data/mnist-cnn.json.
// Notes: docs/lessons/convolution.md.
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { worked, inTorc, analogy, term, sym } from '../core/blocks.js';
import { mountSlidingWindow } from '../widgets/conv-widgets.js';
import { mountShiftMaps, mountShiftAccuracy } from '../widgets/shift-widgets.js';

const F = 'files.conv', SH = 'files.shift';
const L0 = `${F}.derived.layers.0`, L1 = `${F}.derived.layers.1`;   // the two convolutions of the digit CNN
const FM = `${SH}.feature_maps`;
const MOVE = 3;                                                       // the move of the misconception section, in pixels
const sec = (id, title, body) => `<section class="cv" aria-labelledby="cv-${id}"><h2 id="cv-${id}">${title}</h2>${body}</section>`;
const eq = (latex) => `<div class="cv-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;
const err = (D, id) => { const e = D.files.conv.errors.find((x) => x.id === id); return `<code>${escapeHtml(e.exception)}: ${escapeHtml(e.what)}</code>`; };

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
  trace: ['conv.json', 'shift.json', 'mnist-cnn.json'],
  template: 3,
  loop: 'model',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Convolution</h1>
    <p class="lead">The same few ${term('weight', 'weights')} can look for a pattern in every part of an image.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'How can one small detector find a stroke anywhere in a digit?',
        `How many ${term('parameter', 'parameters')} does such a detector need?`,
        `A digit moves a few ${term('pixel', 'pixels')} to the right. Does the network still read it?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('setup', 'A weight for every pixel', `
    <p>In the first ${term('layer')} of the MLP for digits, each ${term('neuron')} has its own weight for each of the ${T.anchor('pixels')} pixels. So a stroke at the top meets other weights than the same stroke lower down.</p>
    <p>A ${term('convolution')} layer works in another way. Its ${term('kernel')} is a small grid of weights. At each position, the layer multiplies each pixel under the kernel by the weight on top of it and adds the products. It uses the same weights at every position. ${term('training', 'Training')} changes them with their ${term('gradient', 'gradients')}, as in every layer.</p>
    <p>The grid of all these results is a ${term('feature-map')}. A network that starts with convolution layers is a ${term('cnn', 'CNN')}.</p>`),
    },
    {
      kind: 'quiz', id: 'params', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>A convolution layer has ${T(`${L0}.out_channels`, 'setup')} kernels. Each has ${T(`${L0}.kernel`, 'setup')} × ${T(`${L0}.kernel`, 'setup')} weights and one ${term('bias')}. The layer slides them over an image of ${T(`${L0}.side_in`, 'setup')} × ${T(`${L0}.side_in`, 'setup')} pixels. How many parameters does it have?</p>`,
      options: [
        { html: (D, T) => `${T(`${F}.cnn.0.parameters`, 'setup')}`, correct: true,
          feedback: 'Yes. The same kernels work at every position, so the size of the image does not matter.' },
        { html: (D, T) => `${T(`${F}.dense_compare.parameters`, 'setup')}, as many as a linear layer with ${T(`${L0}.out_channels`, 'setup')} neurons`,
          feedback: 'That layer has its own weight for every pixel and neuron.' },
        { html: 'More for a larger image, because the kernels visit more positions',
          feedback: 'The kernels visit more positions, but with the same weights.' },
      ],
      reveal: (D, T) => `<p>Each kernel has ${T(`${L0}.kernel_area`, 'reveal')} weights and one bias. That makes ${T(`${L0}.parameters_per_kernel`, 'reveal')} parameters per kernel and ${T(`${F}.cnn.0.parameters`, 'reveal')} for the layer.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('slide', '1 · A kernel slides over the image', `
    <p>The small image below is dark on the left and bright on the right. Its kernel adds the pixels under its left column and subtracts those under its right column. So the result is far from zero only at the edge. A real layer also adds a bias, but this example has none.</p>
    <p>The step from one position to the next is the ${term('stride')}. ${term('padding', 'Padding')} adds a border of zeros, so the kernel can also sit at the edges.</p>
    ${analogy('The kernel is a stencil that you slide over the image. At each position, it asks whether its pattern is there.', 'The feature map moves with the image, but the final answer of a network can still change.')}`),
    },
    {
      kind: 'widget', id: 'window', className: 'widget', label: 'Interactive: the window of each output on a small image, for three settings of stride and padding',
      mount: (host, D) => mountSlidingWindow(host, D),
      tasks: (D, T) => [
        { do: 'Keep the first setting. Select the top row of the output, from left to right.',
          see: 'The window moves one pixel at a time. The first two values are far from zero, because their windows cover the edge.' },
        { do: `Choose the setting with stride ${T(`${F}.example.runs.2.stride`, 'setup')} and padding ${T(`${F}.example.runs.2.padding`, 'setup')}.`,
          see: `A border of zeros appears. The window jumps two pixels, so the output has only ${T(`${F}.example.runs.2.output.shape.2`, 'setup')} × ${T(`${F}.example.runs.2.output.shape.3`, 'setup')} values.` },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('size', '2 · The size of the feature map', `
    <p>Take an image of <em>H</em> × <em>H</em> pixels, a kernel of <em>K</em> × <em>K</em>, stride <em>S</em> and padding <em>P</em>. The feature map then has this many rows and columns:</p>
    ${eq('\\left\\lfloor \\frac{H + 2P - K}{S} \\right\\rfloor + 1')}
    <p>The brackets ${sym('floor')} mean “round down to a whole number”. The fraction counts the steps that fit, and adding one counts the first position. Each value of a feature map is a cell.</p>
    ${worked(`<p>The first layer of the digit CNN reads ${T(`${L0}.side_in`, 'worked')} × ${T(`${L0}.side_in`, 'worked')} pixels with <em>K</em> = ${T(`${L0}.kernel`, 'worked')}, <em>S</em> = ${T(`${L0}.stride`, 'worked')} and <em>P</em> = ${T(`${L0}.padding`, 'worked')}.</p>
    <p>(${T(`${L0}.side_in`, 'worked')} + ${T(`${L0}.twice_padding`, 'worked')} − ${T(`${L0}.kernel`, 'worked')}) ÷ ${T(`${L0}.stride`, 'worked')} = ${T(`${L0}.numerator`, 'worked')} ÷ ${T(`${L0}.stride`, 'worked')} = ${T(`${L0}.quotient`, 'worked')}. Rounded down, that is ${T(`${L0}.rounded_down`, 'worked')}. Adding one gives <strong>${T(`${L0}.side_out`, 'worked')}</strong>.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'size-two', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${F}.cnn.2.out_shape.2`, tolerance: 0.01,
      prompt: (D, T) => `<p>The second layer reads these maps of ${T(`${L1}.side_in`, 'setup')} × ${T(`${L1}.side_in`, 'setup')} cells. It also has <em>K</em> = ${T(`${L1}.kernel`, 'setup')}. It uses <em>S</em> = ${T(`${L1}.stride`, 'setup')} and <em>P</em> = ${T(`${L1}.padding`, 'setup')}. How many rows do its feature maps have?</p>`,
      reveal: (D, T) => `<p>The fraction is ${T(`${L1}.quotient`, 'reveal')}. Rounded down, it is ${T(`${L1}.rounded_down`, 'reveal')}, and adding one gives ${T(`${L1}.side_out`, 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('channels', '3 · Channels', `
    <p>A colour image has three ${term('channel', 'channels')}, for red, green and blue. A kernel for it has one grid of weights per channel and adds all the products, so it gives one feature map.</p>
    <p>A layer with ${T(`${L0}.out_channels`, 'setup')} kernels gives ${T(`${L0}.out_channels`, 'setup')} feature maps. The next layer reads them like an image with that many channels.</p>
    ${worked(`<p>The second layer has ${T(`${L1}.out_channels`, 'worked')} kernels and reads ${T(`${L1}.in_channels`, 'worked')} channels. Each kernel has ${T(`${L1}.in_channels`, 'worked')} × ${T(`${L1}.kernel`, 'worked')} × ${T(`${L1}.kernel`, 'worked')} = ${T(`${L1}.weights_per_kernel`, 'worked')} weights plus a bias, so ${T(`${L1}.parameters_per_kernel`, 'worked')} parameters. The layer has ${T(`${L1}.out_channels`, 'worked')} × ${T(`${L1}.parameters_per_kernel`, 'worked')} = <strong>${T(`${L1}.parameters`, 'worked')}</strong> parameters.</p>`)}
    ${inTorc('The convolution layer in code', `
    <pre class="cv-code" tabindex="0" role="group" aria-label="A convolution layer in TorC"><code>// 1 input channel, 8 output channels, kernel 3, stride 2, padding 1
nn::Conv2d conv(1, 8, 3, 2, 1);
// x holds B images with 1 channel of 28 x 28 pixels: (B, 1, 28, 28)
// y holds their 8 feature maps of 14 x 14 cells: (B, 8, 14, 14)
Variable y = conv(x);</code></pre>
    <p>Some books turn the kernel half a turn before they use it. TorC does not flip the kernel. With the wrong number of channels, it stops with ${err(D, 'channels')}.</p>`)}`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('moves', '4 · The feature map moves with the digit', `
    <p>The kernel has the same weights at every position. So a stroke that moves by one stride gives the same results one position further, and the map moves by one cell.</p>
    <p>This holds exactly only for whole strides. With the stride ${T(`${FM}.stride`, 'setup')}, a move of one pixel puts the stroke between two positions, and the map changes.</p>`),
    },
    {
      kind: 'widget', id: 'shift-maps', className: 'widget', label: 'Interactive: a test image moved to the right, and one feature map of the trained CNN before and after the move',
      mount: (host, D) => mountShiftMaps(host, D),
      tasks: (D, T) => [
        { do: `Before you choose a move of ${T(`${SH}.setup.shifts.2`, 'setup')} pixels, guess where the bright cells will go. Then choose it.`,
          see: 'The bright cells sit one cell further right. The text under the maps says that every map moved by one cell.' },
        { do: `Choose a move of ${T(`${SH}.setup.shifts.1`, 'setup')} pixel, then ${T(`${SH}.setup.shifts.3`, 'setup')} pixels.`,
          see: 'The maps change instead of moving by whole cells.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('wrong', '5 · A tempting idea that is wrong', `
    <p>Many people conclude that a CNN reads a digit equally well anywhere. But its last layer is a linear layer with its own weight for every cell. After a move, the cells meet other weights.</p>
    <p>TorC measured both networks on all ${T.anchor('test_images')} test images. Without a move, the CNN reads ${T.human(`${SH}.accuracy.cnn.right.0.accuracy`, 'pct')} of them correctly. Moved ${T(`${SH}.accuracy.cnn.right.${MOVE}.shift`, 'setup')} pixels to the right, it reads ${T.human(`${SH}.accuracy.cnn.right.${MOVE}.accuracy`, 'pct')}. The MLP then reads ${T.human(`${SH}.accuracy.mlp.right.${MOVE}.accuracy`, 'pct')}. So the CNN copes better, but it is not immune to the move.</p>`),
    },
    {
      kind: 'widget', id: 'shift-accuracy', className: 'widget', label: 'Interactive: the test images that the MLP and the CNN read correctly after a move',
      mount: (host, D) => mountShiftAccuracy(host, D),
      tasks: (D, T) => [
        { do: `Read both lines at a move of ${T(`${SH}.setup.shifts.${MOVE}`, 'setup')} pixels to the right.`,
          see: 'The CNN line is higher, but both lines are far below their start.' },
        { do: `Choose the direction down, then a move of ${T(`${SH}.setup.shifts.4`, 'setup')} pixels.`,
          see: 'Both networks now read most test images wrongly.' },
      ],
    },
    {
      kind: 'quiz', id: 'stride-one', purpose: 'transfer', type: 'choice',
      prompt: '<p>Another CNN has a first layer with the stride one. A digit moves one pixel to the right. What happens to the feature maps of that layer?</p>',
      options: [
        { html: 'They move one cell to the right.', correct: true,
          feedback: 'Yes. Here one pixel is one whole stride.' },
        { html: 'They change, because a move of one pixel always breaks the pattern.',
          feedback: 'That happens only when the move is not a whole number of strides.' },
        { html: 'They move half a cell.',
          feedback: 'With the stride one, the positions of the kernel are one pixel apart.' },
      ],
      reveal: '<p>With the stride one, every move by whole pixels moves the maps by whole cells.</p>',
    },
    {
      kind: 'quiz', id: 'shared-gradient', purpose: 'cumulative', from: 'backprop', type: 'choice',
      prompt: '<p>During training, the layer uses one kernel weight at every position. How does the backward pass find the gradient of this weight?</p>',
      options: [
        { html: 'It adds the gradients from all positions.', correct: true,
          feedback: 'Yes. A value with several uses gets a gradient from each use, and TorC adds them.' },
        { html: 'It takes only the position with the largest result.',
          feedback: `Every position changes the ${term('loss')}, so every position counts.` },
        { html: 'It needs a copy of the weight for each position.',
          feedback: 'The weight is one number with many uses, and the backward pass needs no copy.' },
      ],
      reveal: '<p>So every position of every training image teaches the same kernel.</p>',
    },
    {
      kind: 'explain', id: 'why-moves',
      prompt: '<p>Explain why the feature map moves with the digit, but the answer of the CNN can still change.</p>',
      model: '<p>The kernel has the same weights everywhere, so a moved stroke gives the same results further on. The last layer has a weight per cell.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>A kernel is a small grid of weights that the layer uses at every position. So it finds a stroke anywhere, with few parameters.</p>
    <p>When the digit moves by whole strides, the feature maps move with it. The answer can still change, because the last layer has a weight for every cell.</p>`,
      next: () => '<p>Does a CNN read the test digits better than the MLP?</p>',
    },
  ],
};
