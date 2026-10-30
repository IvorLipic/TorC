// Lesson "linear-layer" (Part 3 `networks`, curriculum v3, template 3): "The neuron and the layer". One neuron is a weighted sum
// plus a bias (a taxi price from three inputs, drawn as circles and arrows; the dot product is named there); a layer is many
// neurons that read the same inputs, and one matrix product computes all of them (the matmul grid, positions from 0, with the
// matrix-product refresher); the bias reaches every row (broadcasting, in words); W is (out, in) and is transposed; the parameter
// count of the digit network (784 = 28 × 28). Brief §6 Part 2 and §7. The backward rules left for `backprop`, initialization for
// `deep-nets`. The dot-product refresher is not opened here because of the word budget (docs/lessons/linear-layer.md).
// Every number comes from data/linear.json (taxi, param_counts) or site/data/anchors.json through a role-tagged T()
// (docs/authoring.md "Number roles"); the formulas take their numbers from the trace with T.num.
import { math } from '../core/katex.js';
import { worked, refresher, term, sym } from '../core/blocks.js';
import { mountMatmulGrid } from '../widgets/matmul-grid.js';
import { mountLayerCount } from '../widgets/layer-count.js';
import { neuronFigure, NEURON_FIGURE_CSS } from '../widgets/neuron-figure.js';

const P = 'files.linear';
const TX = `${P}.taxi`;
const x = (i, j) => `${TX}.x.values.${i * 3 + j}`;          // ride i, input j (km, minutes of waiting, bags)
const w = (k, j) => `${TX}.W.values.${k * 3 + j}`;          // neuron k (0 price, 1 time), weight of input j
const b = (k) => `${TX}.b.values.${k}`;
const y = (i, k) => `${TX}.y.values.${i * 2 + k}`;
// the digit network (examples/mnist_mlp): param_counts[1]; its Linear layers are layers 0, 2 and 4; params sorted by name
const NET = `${P}.param_counts.1`;
const LAYER = (n) => `${NET}.layers.${n}`;
const WEIGHT = (n) => `${LAYER(n)}.params.1`;
const BIAS = (n) => `${LAYER(n)}.params.0`;

const sec = (id, title, body) => `<section class="ll" aria-labelledby="ll-${id}"><h2 id="ll-${id}">${title}</h2>${body}</section>`;
const eq = (latex, label = 'Formula') => `<div class="ll-eq" tabindex="0" role="group" aria-label="${label}">${math(latex, true)}</div>`;
const mat = (T, rows) => `\\begin{bmatrix} ${rows.map((r) => r.map((p) => T.num(p)).join(' & ')).join(' \\\\ ')} \\end{bmatrix}`;
// a sum of products "a × b + c × d + … + bias = result" with every number bound for a worked block
const sumOf = (T, pairs, bias, result, unit) => `${pairs.map(([a, c]) => `${T(a, 'worked')} × ${T(c, 'worked')}`).join(' + ')} + ${T(bias, 'worked')} = <strong>${T(result, 'worked')}</strong> ${unit}`;

const STYLE = `<style>
.ll p{max-width:44rem}
.ll h2{margin-top:2rem}
.ll-eq{overflow-x:auto;margin:.2rem 0}
.ll-eq .katex-display{margin:.4em 0}
${NEURON_FIGURE_CSS}
</style>`;

export default {
  id: 'linear-layer',
  title: 'The neuron and the layer',
  trace: ['linear.json'],
  template: 3,
  loop: 'model',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>The neuron and the layer</h1>
    <p class="lead">Every ${term('neural-network')} uses one small formula many times. A taxi price shows the formula.</p></section>`,
    },
    {
      kind: 'questions',
      items: (D, T) => [
        'A taxi price depends on the distance, the waiting time and the bags. How does one formula combine three inputs?',
        `The digit network of this course has ${T.anchor('mlp_parameters')} ${term('parameter', 'parameters')}. Where does that number come from?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('taxi', 'Taxi prices', `
    <p>In the last lesson, a taxi ride cost a fixed fee plus a price per kilometre. The ${term('gradient')} showed the ${term('model')} how to improve both. This method is ${term('gradient-descent')}.</p>
    <p>That model had one input, the distance. Our taxi company also charges for waiting and for bags.</p>
    <p>Each kilometre costs ${T(w(0, 0), 'setup')} euros. Each minute of waiting costs ${T(w(0, 1), 'setup')} euro. Each bag costs ${T(w(0, 2), 'setup')} euro. Every ride also pays a fee of ${T(b(0), 'setup')} euro.</p>
    <p>We choose these prices by hand. ${term('training', 'Training')} could find them from data.</p>`),
    },
    {
      kind: 'quiz', id: 'price', purpose: 'predict', type: 'numeric',
      answer: y(0, 0), tolerance: 0.01, unit: 'euros',
      prompt: (D, T) => `<p>A ride is ${T(x(0, 0), 'setup')} km long. It has ${T(x(0, 1), 'setup')} minutes of waiting and ${T(x(0, 2), 'setup')} bags. What does it cost?</p>`,
      reveal: (D, T) => `${worked(`<p>price = ${sumOf(T, [[w(0, 0), x(0, 0)], [w(0, 1), x(0, 1)], [w(0, 2), x(0, 2)]], b(0), y(0, 0), 'euros')}</p>`)}
    <p>The fee comes once, at the end.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('neuron', '1 · One neuron', `
    <p>You multiplied each input by its own number, added the products, and then added the fee. A ${term('neuron')} computes exactly this.</p>
    <p>The numbers that multiply the inputs are its ${term('weight', 'weights')}. The number added at the end is its ${term('bias')}, here the fee. Multiplying matching numbers and adding the products is a dot product.</p>
    <p>With letters, the inputs are ${sym('subscript', '<i>x</i>₀')}, <i>x</i>₁ and <i>x</i>₂, and the weights are <i>w</i>₀, <i>w</i>₁ and <i>w</i>₂. The small number is a position, and positions count from <code>0</code>.</p>
    <p>The sign ${sym('sum', 'Σ')} means “add up all the terms”. So the neuron computes:</p>
    ${eq('y = w_0 x_0 + w_1 x_1 + w_2 x_2 + b = \\sum_{j=0}^{2} w_j\\, x_j + b', 'The formula of one neuron')}
    ${neuronFigure(D.files.linear.taxi, { neurons: [0], caption: 'One neuron. Each arrow multiplies an input by its weight. The circle adds the products and the bias.' })}
    <p>The name comes from nerve cells, but this neuron is not a small brain. It only multiplies and adds.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('layer', '2 · A layer is many neurons', `
    <p>The taxi app also guesses how long a ride takes. A second neuron reads the same three inputs with its own weights and bias.</p>
    <p>Each kilometre adds ${T(w(1, 0), 'setup')} minutes, and each minute of waiting adds ${T(w(1, 1), 'setup')} minute. A bag adds no time, so its weight is ${T(w(1, 2), 'setup')}. The bias is ${T(b(1), 'setup')} minutes for getting in and out.</p>
    <p>Neurons that read the same inputs side by side form a ${term('layer')}.</p>
    ${neuronFigure(D.files.linear.taxi, { neurons: [0, 1], caption: 'A layer of two neurons. Each has its own weights and bias.' })}
    ${worked(`<p>time = ${sumOf(T, [[w(1, 0), x(0, 0)], [w(1, 1), x(0, 1)], [w(1, 2), x(0, 2)]], b(1), y(0, 1), 'minutes')}</p>`)}`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('matrix', '3 · One matrix product for the whole layer', `
    <p>A ${term('matrix')} is a grid of numbers in rows and columns. The matrix <i>x</i> has one row per ride, and <i>W</i> has one row per neuron:</p>
    ${eq(`x = ${mat(T, [[x(0, 0), x(0, 1), x(0, 2)], [x(1, 0), x(1, 1), x(1, 2)]])} \\qquad W = ${mat(T, [[w(0, 0), w(0, 1), w(0, 2)], [w(1, 0), w(1, 1), w(1, 2)]])} \\qquad b = (${T.num(b(0))},\\ ${T.num(b(1))})`, 'The two rides, the weights and the biases')}
    <p>Rows count from <code>0</code> too, so row <code>1</code> of <i>W</i> is the time neuron. The output <i>y</i>ᵢₖ of neuron <i>k</i> for ride <i>i</i> combines row <i>i</i> of <i>x</i> with row <i>k</i> of <i>W</i>:</p>
    ${eq('y_{ik} = \\sum_{j} x_{ij}\\, W_{kj} + b_k', 'The formula of a layer')}
    <p>This is a matrix product, and it computes every neuron for every ride at once. TorC copies <i>b</i> into each row of the result and calls this copying ${term('broadcasting')}.</p>
    ${refresher('matrix-product')}`),
    },
    {
      kind: 'quiz', id: 'your-entry', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: y(1, 0), tolerance: 0.01, unit: 'euros',
      prompt: '<p>Compute <i>y</i>₁₀, the price of the second ride, from the matrices above.</p>',
      reveal: (D, T) => `${worked(`<p><i>y</i>₁₀ = ${sumOf(T, [[x(1, 0), w(0, 0)], [x(1, 1), w(0, 1)], [x(1, 2), w(0, 2)]], b(0), y(1, 0), 'euros')}</p>`)}
    <p>It uses row <code>1</code> of <i>x</i>, row <code>0</code> of <i>W</i> and <i>b</i>₀.</p>`,
    },
    {
      kind: 'widget', id: 'matmul-grid', className: 'lin-w lin-mg', label: 'Interactive: the taxi layer as one matrix product',
      mount: (host, D, ctx) => mountMatmulGrid(host, D, { ...ctx, rowName: 'ride', neuronNames: ['price neuron', 'time neuron'], inputNames: ['kilometres', 'minutes of waiting', 'bags'] }),
      tasks: (D, T) => [
        { do: 'Choose the output <i>y</i>₀₁ in the grid.',
          see: 'Row <code>0</code> of <i>x</i> and row <code>1</code> of <i>W</i> light up. Under the grid is the time of the first ride.' },
        { do: 'Raise <i>b</i>₀ by one.',
          see: 'Both prices go up by one euro, and the times stay the same.' },
        { do: 'In row <code>0</code> of <i>W</i>, set the weight of the bags to zero.',
          see: `The first price falls by ${T(x(0, 2), 'setup')} euros and the second by ${T(x(1, 2), 'setup')} euro, one euro for each bag.` },
      ],
    },
    {
      kind: 'quiz', id: 'w-shape', purpose: 'transfer', type: 'choice',
      prompt: (D, T) => `<p>A digit image is a square of ${T.anchor('image_side')} by ${T.anchor('image_side')} ${term('pixel', 'pixels')}. Its ${T.anchor('pixels')} pixels are the inputs of the first layer, which has ${T(`${WEIGHT(0)}.shape.0`, 'setup')} neurons. What ${term('shape')} does TorC give its <i>W</i>, as (rows, columns)?</p>`,
      options: [
        { html: (D, T) => `(${T(`${WEIGHT(0)}.shape.0`, 'setup')}, ${T(`${WEIGHT(0)}.shape.1`, 'setup')})`, correct: true,
          feedback: 'Yes. <i>W</i> has one row per neuron, and each row holds one weight per input.' },
        { html: (D, T) => `(${T(`${WEIGHT(0)}.shape.1`, 'setup')}, ${T(`${WEIGHT(0)}.shape.0`, 'setup')})`,
          feedback: 'This gives one row per input. But each row of <i>W</i> belongs to one neuron, as in the taxi grid.' },
        { html: (D, T) => `(${T(`${WEIGHT(0)}.shape.1`, 'setup')})`,
          feedback: 'This is enough for one neuron. But each neuron has its own weights.' },
        { html: (D, T) => `(${T(`${BIAS(0)}.shape.0`, 'setup')})`,
          feedback: 'This is the shape of <i>b</i>, one bias per neuron.' },
      ],
      reveal: (D, T) => `<p>TorC stores this <i>W</i> with ${T(`${WEIGHT(0)}.shape.0`, 'reveal')} rows and ${T(`${WEIGHT(0)}.shape.1`, 'reveal')} columns. So it holds ${T(`${WEIGHT(0)}.numel`, 'reveal')} weights.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('shape', '4 · The shape of W and the size of a layer', `
    <p>Many people expect <i>W</i> to have one row per input. TorC stores one row per neuron, so the shape of <i>W</i> is (neurons, inputs).</p>
    <p>A matrix product pairs each row of the left matrix with each column of the right matrix. So TorC first turns the rows of <i>W</i> into columns. This is the transpose ${sym('transpose', '<i>W</i>ᵀ')}, and the layer computes:</p>
    ${eq('y = x\\, W^{\\top} + b', 'The layer as one formula')}
    <p>Each neuron has one weight per input and one bias. So a layer has (neurons × inputs) weights and one bias per neuron.</p>`),
    },
    {
      kind: 'quiz', id: 'last-layer', purpose: 'check', type: 'numeric',
      answer: `${LAYER(4)}.count`, tolerance: 0.5,
      prompt: (D, T) => `<p>The third layer of the digit network reads the ${T(`${WEIGHT(4)}.shape.1`, 'setup')} outputs of the second layer. It has ${T(`${WEIGHT(4)}.shape.0`, 'setup')} neurons, one for each digit. How many parameters does it have?</p>`,
      reveal: (D, T) => `<p><i>W</i> holds ${T(`${WEIGHT(4)}.numel`, 'reveal')} weights. With the ${T(`${BIAS(4)}.numel`, 'reveal')} biases, the layer has ${T(`${LAYER(4)}.count`, 'reveal')} parameters.</p>`,
    },
    {
      kind: 'widget', id: 'layer-count', className: 'widget', label: 'Interactive: the parameters of the digit network, layer by layer',
      mount: (host, D) => mountLayerCount(host, D),
      tasks: (D, T) => [
        { do: 'Choose the first layer.',
          see: `<i>W</i> is a long strip with ${T(`${WEIGHT(0)}.shape.0`, 'setup')} rows and ${T(`${WEIGHT(0)}.shape.1`, 'setup')} columns. It holds most parameters of the network.` },
        { do: 'Choose the whole network.',
          see: `The three layers add up to ${T.anchor('mlp_parameters')} parameters.` },
      ],
    },
    {
      kind: 'quiz', id: 'gradient-size', purpose: 'cumulative', from: 'sgd-step', type: 'choice',
      prompt: () => `<p>In the last lesson, the taxi model had two parameters, and its gradient was a list of two numbers. How many numbers does the gradient of the first digit layer hold?</p>`,
      options: [
        { html: 'One number for each parameter', correct: true,
          feedback: 'Yes. Each weight and each bias gets its own number.' },
        { html: 'One number for the whole layer',
          feedback: 'One number cannot tell each weight which way to move.' },
        { html: 'One number for each neuron',
          feedback: 'Each neuron has many weights, and each weight needs its own number.' },
      ],
      reveal: (D, T) => `<p>The first layer has ${T(`${LAYER(0)}.count`, 'reveal')} parameters. So its gradient is a list of ${T(`${LAYER(0)}.count`, 'reveal')} numbers.</p>`,
    },
    {
      kind: 'explain', id: 'what-a-layer-does',
      prompt: '<p>Explain what a layer of neurons computes.</p>',
      model: '<p>In a layer, every neuron computes its own weighted sum of the same inputs plus its bias, and one matrix product does all of them.</p>',
    },
    {
      kind: 'answer',
      html: (D, T) => `<p>A neuron multiplies each input by its own weight, adds the products and adds its bias. A layer is many neurons that read the same inputs, and one matrix product computes them all.</p>
    <p>Each neuron has one weight per input and one bias. In the digit network this gives ${T.anchor('mlp_parameters')} parameters.</p>`,
      next: () => '<p>Each output of a layer is a weighted sum, so it changes along a straight line. Can a second layer on top fit a curve?</p>',
    },
  ],
};
