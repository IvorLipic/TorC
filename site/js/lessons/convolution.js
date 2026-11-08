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
const SO = `${F}.derived.stride_one`;                                 // the size rule for stride 1, padding 1 (the learner's step)
const FM = `${SH}.feature_maps`;
const ONE = 2, TWO = 4;                                               // moves of one and two strides (shift.json setup.shifts)
const sec = (id, title, body) => `<section class="cv" aria-labelledby="cv-${id}"><h2 id="cv-${id}">${title}</h2>${body}</section>`;
const eq = (latex) => `<div class="cv-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;

// the same lines as the `code` of this lesson in course.json, which tools/snippets/check.mjs compiles and runs (tested)
const CODE = `// in 1, out 8, kernel 3, stride 2, padding 1
nn::Conv2d conv(1, 8, 3, 2, 1);
// (B, 1, 28, 28) becomes (B, 8, 14, 14)
Variable y = conv(x);
// (B, C, H, W) becomes (B, C·H·W)
nn::Flatten flat;`;

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
        'How can one small detector find a stroke anywhere?',
        `How many ${term('parameter', 'parameters')} does such a detector need?`,
        `Does the network still read a digit that moves a few ${term('pixel', 'pixels')}?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('setup', 'A weight for every pixel', `
    <p>In the MLP for digits, each ${term('neuron')} of the first ${term('layer')} has its own weight for each of the ${T.anchor('pixels')} pixels. So a stroke at the top meets other weights than a stroke lower down.</p>
    <p>A ${term('convolution')} layer works in another way. Its ${term('kernel')} is a small grid of weights. At one position, the layer multiplies each pixel under the kernel by the weight on top of it and adds the products. Then the kernel moves to the next position. ${term('training', 'Training')} adjusts the weights with their ${term('gradient', 'gradients')}.</p>
    <p>The grid of all these results is a ${term('feature-map')}. A network that starts with convolution layers is a ${term('cnn', 'CNN')}.</p>`),
    },
    {
      kind: 'quiz', id: 'params', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>A convolution layer has ${T(`${L0}.out_channels`, 'setup')} kernels. Each has ${T(`${L0}.kernel`, 'setup')} × ${T(`${L0}.kernel`, 'setup')} weights and one ${term('bias')}. The layer slides them over an image of ${T(`${L0}.side_in`, 'setup')} × ${T(`${L0}.side_in`, 'setup')} pixels. How many parameters does it have?</p>`,
      options: [
        { html: (D, T) => `${T(`${F}.cnn.0.parameters`, 'setup')}`, correct: true,
          feedback: 'Yes. The same kernels work at every position, whatever the image size.' },
        { html: (D, T) => `${T(`${L0}.weights`, 'setup')}`,
          feedback: 'That forgets the bias of each kernel.' },
        { html: (D, T) => `${T(`${F}.dense_compare.parameters`, 'setup')}, as many as a linear layer with ${T(`${L0}.out_channels`, 'setup')} neurons`,
          feedback: 'That layer has its own weight for every pixel and neuron.' },
        { html: 'More for a larger image, because the kernels visit more positions',
          feedback: 'The kernels visit more positions with the same weights.' },
      ],
      reveal: (D, T) => `<p>Each kernel has ${T(`${L0}.kernel_area`, 'reveal')} weights and one bias. That makes ${T(`${L0}.parameters_per_kernel`, 'reveal')} parameters per kernel and ${T(`${F}.cnn.0.parameters`, 'reveal')} for the layer.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('slide', '1 · A kernel slides over the image', `
    <p>The small image below is dark on the left and bright on the right. One pixel in the middle is twice as bright. The kernel has positive weights on the left and negative weights on the right, and its middle row counts twice. So a cell is far from zero where the brightness changes.</p>
    <p>The ${term('stride')} is the number of pixels that the kernel moves each time. ${term('padding', 'Padding')} adds a border of zeros, so the kernel can also sit at the edges. Beside the bright side, that border makes a second edge. Rows and columns count from zero, and this example has no bias.</p>
    ${analogy('The kernel is a stencil that you slide over the image.', 'A stencil only shows yes or no. A kernel gives a sum with a sign.')}`),
    },
    {
      kind: 'widget', id: 'window', className: 'widget', label: 'Interactive: the window of each cell of the feature map on a small image, for three settings of stride and padding',
      mount: (host, D) => mountSlidingWindow(host, D),
      tasks: (D, T) => [
        { do: 'Keep the first setting. Select the top row of the feature map, from left to right.',
          see: 'The first two cells cover the edge, so they are far from zero.' },
        { do: `Choose the setting with stride ${T(`${F}.example.runs.2.stride`, 'setup')} and padding ${T(`${F}.example.runs.2.padding`, 'setup')}.`,
          see: `Zeros appear around the image, and the map’s right column is far from zero too. The second setting gives ${T(`${F}.example.runs.1.output.shape.2`, 'setup')} × ${T(`${F}.example.runs.1.output.shape.3`, 'setup')} cells. Here the window jumps two pixels, so the map has ${T(`${F}.example.runs.2.output.shape.2`, 'setup')} × ${T(`${F}.example.runs.2.output.shape.3`, 'setup')}.` },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('size', '2 · The size of the feature map', `
    <p>Take an image of <em>H</em> × <em>H</em> pixels and a kernel of <em>K</em> × <em>K</em>, with stride <em>S</em> and padding <em>P</em>. The feature map has this many rows:</p>
    ${eq('\\left\\lfloor \\frac{H + 2P - K}{S} \\right\\rfloor + 1')}
    <p>The brackets ${sym('floor')} mean “round down”. The fraction counts the steps that fit, and adding one counts the first position. TorC trained a digit CNN on the ${term('mnist', 'MNIST')} images, with two convolution layers and a linear layer.</p>
    ${worked(`<p>The first layer of the digit CNN reads ${T(`${L0}.side_in`, 'worked')} × ${T(`${L0}.side_in`, 'worked')} pixels with <em>K</em> = ${T(`${L0}.kernel`, 'worked')}, <em>S</em> = ${T(`${L0}.stride`, 'worked')} and <em>P</em> = ${T(`${L0}.padding`, 'worked')}.</p>
    <p>(${T(`${L0}.side_in`, 'worked')} + ${T(`${L0}.twice_padding`, 'worked')} − ${T(`${L0}.kernel`, 'worked')}) ÷ ${T(`${L0}.stride`, 'worked')} = ${T(`${L0}.numerator`, 'worked')} ÷ ${T(`${L0}.stride`, 'worked')} = ${T(`${L0}.quotient`, 'worked')}. Rounded down, that is ${T(`${L0}.rounded_down`, 'worked')}. Adding one gives <strong>${T(`${L0}.side_out`, 'worked')}</strong>.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'size-two', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${F}.derived.stride_one.side_out`, tolerance: 0.01,
      prompt: (D, T) => `<p>Another layer reads maps of ${T(`${SO}.side_in`, 'setup')} × ${T(`${SO}.side_in`, 'setup')} cells. It has <em>K</em> = ${T(`${SO}.kernel`, 'setup')}. It uses <em>S</em> = ${T(`${SO}.stride`, 'setup')} and <em>P</em> = ${T(`${SO}.padding`, 'setup')}. How many rows does its feature map have?</p>`,
      reveal: (D, T) => `<p>The fraction is ${T(`${SO}.quotient`, 'reveal')}, and adding one gives ${T(`${SO}.side_out`, 'reveal')}. The padding keeps the size.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('channels', '3 · Channels', `
    <p>A colour image has three ${term('channel', 'channels')}, for red, green and blue. A kernel for it has one grid of weights per channel and adds all the products into one feature map.</p>
    <p>A layer with ${T(`${L0}.out_channels`, 'setup')} kernels gives ${T(`${L0}.out_channels`, 'setup')} feature maps. The next layer reads them like an image with that many channels.</p>
    ${worked(`<p>The second layer has ${T(`${L1}.out_channels`, 'worked')} kernels and reads ${T(`${L1}.in_channels`, 'worked')} channels. Each kernel has ${T(`${L1}.in_channels`, 'worked')} × ${T(`${L1}.kernel`, 'worked')} × ${T(`${L1}.kernel`, 'worked')} = ${T(`${L1}.weights_per_kernel`, 'worked')} weights plus a bias, so ${T(`${L1}.parameters_per_kernel`, 'worked')} parameters. The layer has ${T(`${L1}.out_channels`, 'worked')} × ${T(`${L1}.parameters_per_kernel`, 'worked')} = <strong>${T(`${L1}.parameters`, 'worked')}</strong> parameters.</p>`)}
    ${inTorc('The convolution layer in code', `
    <pre class="cv-code" tabindex="0" role="group" aria-label="A convolution layer in TorC"><code>${escapeHtml(CODE)}</code></pre>
    <p><em>B</em> is the number of images in a ${term('batch')}, and <em>C</em> is the number of channels. Some books flip the kernel, a half turn. TorC does not flip it, so each weight sits on the pixel that it multiplies.</p>`)}`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('moves', '4 · The feature map moves with the digit', `
    <p>The kernel has the same weights at every position. So a stroke that moves by one stride gives the same results one position further, and the map moves one cell.</p>
    <p>This holds exactly only for whole strides. With the stride ${T(`${FM}.stride`, 'setup')}, a move of one pixel puts the stroke between two positions, and the map changes. The maps below come from the trained digit CNN, after its bias and ${term('relu', 'ReLU')}.</p>`),
    },
    {
      kind: 'widget', id: 'shift-maps', className: 'widget', label: 'Interactive: a test image moved to the right, and one feature map of the trained CNN before and after the move',
      mount: (host, D) => mountShiftMaps(host, D),
      tasks: (D, T) => [
        { do: `Choose a move of ${T(`${SH}.setup.shifts.${ONE}`, 'setup')} pixels. Count the dark columns left of the bright cells in both maps.`,
          see: 'The second map has one more, because it moved one cell.' },
        { do: `Choose another kernel, then a move of ${T(`${SH}.setup.shifts.1`, 'setup')} pixel.`,
          see: 'The map changes instead of moving a whole cell.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('wrong', '5 · A tempting idea that is wrong', `
    <p>Many people conclude that a CNN reads a digit equally well anywhere. TorC tested the trained CNN on all ${T.anchor('test_images')} test images, moved right.</p>
    <p>After a move of ${T(`${SH}.accuracy.cnn.right.${ONE}.shift`, 'setup')} pixels, the first maps move exactly one cell, and the CNN reads ${T.human(`${SH}.accuracy.cnn.right.${ONE}.accuracy`, 'pct')} correctly. After ${T(`${SH}.accuracy.cnn.right.${TWO}.shift`, 'setup')} pixels, they move exactly two cells, but it reads only ${T.human(`${SH}.accuracy.cnn.right.${TWO}.accuracy`, 'pct')}.</p>
    <p>So the maps moved exactly, but the answer still fell. The last layer has its own weight for every cell, so moved cells meet other weights. The MLP loses even more.</p>`),
    },
    {
      kind: 'widget', id: 'shift-accuracy', className: 'widget', label: 'Interactive: the percentage points that the MLP and the CNN lose after a move',
      mount: (host, D) => mountShiftAccuracy(host, D),
      tasks: (D, T) => [
        { do: `Read both lines at moves of ${T(`${SH}.setup.shifts.${ONE}`, 'setup')} and ${T(`${SH}.setup.shifts.${TWO}`, 'setup')} pixels to the right.`,
          see: 'Both networks lose points at both moves, and the CNN loses fewer.' },
        { do: 'Choose the direction down.',
          see: 'Both lines climb higher than for the right.' },
      ],
    },
    {
      kind: 'quiz', id: 'stride-one', purpose: 'transfer', type: 'choice',
      prompt: '<p>Another CNN has a first layer with the stride one. A digit moves one pixel to the right. What happens to the feature maps?</p>',
      options: [
        { html: 'They move one cell to the right.', correct: true,
          feedback: 'Yes. Here one pixel is one whole stride.' },
        { html: 'They change, because a move of one pixel always breaks the pattern.',
          feedback: 'That happens only for a move that is not whole strides.' },
        { html: 'They move half a cell.',
          feedback: 'The positions are one pixel apart here.' },
      ],
      reveal: '<p>Every whole-pixel move is now a whole number of strides.</p>',
    },
    {
      kind: 'quiz', id: 'shared-gradient', purpose: 'cumulative', from: 'backprop', type: 'choice',
      prompt: '<p>The layer uses one kernel weight at every position. How does the backward pass find its gradient?</p>',
      options: [
        { html: 'It adds the gradients from all positions.', correct: true,
          feedback: 'Yes. A value with several uses gets a gradient from each use, and TorC adds them.' },
        { html: 'It takes only the position with the largest result.',
          feedback: `Every position changes the ${term('loss')}, so every position counts.` },
        { html: 'It needs a copy of the weight for each position.',
          feedback: 'The weight is one number with many uses, and needs no copy.' },
      ],
      reveal: '<p>So every position teaches the same kernel.</p>',
    },
    {
      kind: 'explain', id: 'why-moves',
      prompt: '<p>Explain why the feature map moves with the digit, but the CNN’s answer can still change.</p>',
      model: '<p>The kernel has the same weights everywhere, so a moved stroke gives the same results. The last layer has a weight per cell.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>A kernel is a small grid of weights used at every position. It finds a stroke anywhere, with ten parameters for three by three weights and a bias.</p>
    <p>When the digit moves by whole strides, the maps move with it. The answer can still change, because the last layer has a weight per cell.</p>`,
      next: () => '<p>Does a CNN read the test digits better than the MLP?</p>',
    },
  ],
};
