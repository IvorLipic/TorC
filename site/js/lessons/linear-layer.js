// Lesson "linear-layer": nn::Linear as matmul plus bias. Shapes and parameter counts, one neuron = one row of W, the three operations
// inside forward(), the backward formulas, and why the weights start as random numbers of a particular size.
// Interactive parts live in widgets/ (shape-explorer, matmul-grid, linear-backward, init-signal and the shared graph lab);
// layouts and derived numbers in linear-layer.scenes.js. Every number from the real run is put into prose with T('path').
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { mountGraphLab } from '../widgets/graph-lab.js';
import { injectStyle, useBaseStyle } from '../widgets/linear-ui.js';
import { mountShapeExplorer } from '../widgets/shape-explorer.js';
import { mountMatmulGrid } from '../widgets/matmul-grid.js';
import { mountLinearBackward } from '../widgets/linear-backward.js';
import { mountWeightHistogram, mountInitSignal } from '../widgets/init-signal.js';
import { linearShapes, fmtShape } from '../widgets/linear-shapes.js';
import { SCENES, makeCalc, makeExact, makeSci } from './linear-layer.scenes.js';

const prose = (html) => `<section class="lin-prose" style="max-width:50rem">${html}</section>`;
const tm = (latex) => math(latex, false);
const shp = (T, base) => `(${T(`${base}.0`)}, ${T(`${base}.1`)})`;
const P = 'files.linear';                                  // prefix of the extra tables of linear.json
const WEIGHT = `${P}.param_counts.0.layers.0.params.1`;    // Linear(784, 32): weight record (params are sorted by name: bias, weight)
const BIAS = `${P}.param_counts.0.layers.0.params.0`;
const MLP = `${P}.param_counts.1`;
const SIG = (i, layer, key = 'post_std') => `${P}.signal.${i}.layers.${layer}.${key}`;   // i: 0 kaiming, 1 std 0.01, 2 std 1.0
const recorded = (D, id) => D.files.linear.shape_errors.find((r) => r.id === id);

const labCss = `
.lin-parity{background:var(--panel);border:1px solid var(--border);border-left:4px solid var(--ok);border-radius:var(--r);padding:.55rem 1rem;margin:12px 0 0;font-size:.93rem}
.lin-parity ul{display:inline-flex;flex-wrap:wrap;gap:.1rem 1rem;margin:0;padding:0;list-style:none;font-size:.82rem;color:var(--muted);vertical-align:baseline}
.lin-parity .flag::before{content:"\\2713  ";color:var(--ok)}
.lin-parity .flag.bad::before{content:"\\2717  ";color:var(--bwd)}
.lin-labwrap{margin:0}
/* the shared lab lets a wide formula stretch the one-column math window past a phone screen; let the grid track shrink instead */
@media(min-width:901px){.lin-labwrap .lab{height:calc(100vh - 170px);min-height:560px}}
@media(max-width:900px){.lin-labwrap .win-body{grid-template-columns:minmax(0,1fr)}}
`;

export default {
  id: 'linear-layer',
  title: 'nn::Linear: a layer as matmul plus bias',
  trace: ['linear.json'],
  sections: [
    // ---------------------------------------------------------------------------------------------- hero
    {
      kind: 'text',
      html: () => `<section class="hero">
    <h1>nn::Linear: a layer is a matmul plus a bias</h1>
    <p class="lead">A neural-network layer sounds mysterious until you see that it is one matrix product and one addition. Here you predict every shape, compute an output by hand, open up the three operations inside <code>forward()</code>, derive the gradients, and see why the weights start as random numbers of one particular size. The numbers come from a real run of <a href="https://github.com/IvorLipic/TorC">TorC</a>.</p>
    <p class="hint" style="margin-top:.6rem">Reading the numbers: <span style="color:var(--fwd)">cyan</span> numbers in the text were printed by the real library; <span style="color:var(--warn)">amber</span> numbers are plain arithmetic on them. Each interactive part says whether it is recomputed in your browser or precomputed from the TorC run.</p>
  </section>`,
    },

    // ---------------------------------------------------------------------------------------------- 1 shapes
    {
      kind: 'text',
      html: () => prose(`<h2 id="shapes">1 · Shapes first</h2>
    <p>A <code>Linear(in, out)</code> layer takes a <strong>batch</strong> of <em>B</em> samples, each described by <code>in</code> numbers, and returns <em>B</em> rows of <code>out</code> numbers. The input <code>x</code> has shape (B, in), the output <code>y</code> has shape (B, out). To get from one to the other the layer owns two tensors: a weight <code>W</code> of shape (<strong>out, in</strong>) and a bias <code>b</code> of shape (<code>out</code>). TorC computes ${tm('y = x\\,W^{\\top} + b')}.</p>
    <p>Change the three numbers and watch which boxes grow. Then press <em>Break it</em> and read the real error message.</p>`),
    },
    {
      kind: 'widget', id: 'shape-explorer', className: 'lin-w lin-se', label: 'Shape explorer',
      mount: (host, D, ctx) => mountShapeExplorer(host, D, ctx),
    },
    {
      kind: 'text',
      html: (D, T) => { const X = makeExact(D); return prose(`<ul>
      <li><strong>The width of <code>x</code> must equal <code>in</code>.</strong> It is the inner dimension of the matrix product, so it is where a mismatch shows up.</li>
      <li><strong>The batch size only changes the height of <code>x</code> and <code>y</code>.</strong> The parameters do not depend on it.</li>
      <li><strong>Counting parameters is <code>in·out + out</code>.</strong> <code>Linear(784, 32)</code> has a weight of shape ${shp(T, `${WEIGHT}.shape`)}, which is ${X(`${WEIGHT}.numel`)} numbers, and a bias of ${T(`${BIAS}.numel`)} numbers: ${X(`${P}.param_counts.0.total`)} parameters in ${T(`${P}.param_counts.0.tensors`)} tensors. That is the count the real library reported for this layer.</li>
      <li>The MNIST MLP of TorC’s examples is <code>Linear(784, 32)</code>, ReLU, <code>Linear(32, 32)</code>, ReLU, <code>Linear(32, 10)</code>: ${X(`${MLP}.layers.0.count`)} + ${X(`${MLP}.layers.2.count`)} + ${X(`${MLP}.layers.4.count`)} = ${X(`${MLP}.total`)} parameters in ${T(`${MLP}.tensors`)} tensors. Each ReLU adds ${T(`${MLP}.layers.1.count`)}: it has nothing to learn.</li>
    </ul>`); },
    },
    {
      kind: 'gotcha', title: 'Why did my Linear fail?',
      html: (D) => `<p>These are the real messages TorC threw (recorded by <code>tools/trace</code>):</p>
    <ul>${D.files.linear.shape_errors.filter((r) => r.exception !== 'none').map((r) => `<li>${escapeHtml(r.call)}<br><code>${escapeHtml(r.exception)}: ${escapeHtml(r.what)}</code></li>`).join('')}</ul>
    <p><strong>How to read a matmul message.</strong> The first shape is your input <code>x</code>. The second is <strong>Wᵀ</strong>, shape (in, out), not the (out, in) weight you might picture. If the last number of the first shape and the first number of the second differ, your data has a different number of features from the layer’s <code>in</code>. An input with a single axis has no batch axis; give it one (a batch of one sample has shape (1, in)).</p>`,
    },

    // ---------------------------------------------------------------------------------------------- 2 one neuron
    {
      kind: 'text',
      html: () => prose(`<h2 id="neuron">2 · One neuron is one row of W</h2>
    <p>Each of the <code>out</code> rows of <code>W</code> belongs to one <strong>neuron</strong>. A neuron looks at one sample (one row of <code>x</code>), multiplies it entry by entry with its own weights, adds the products, and adds its own bias. That single number is one entry of <code>y</code>:</p>
    <div style="overflow-x:auto">${math('y_{ik} = \\sum_{j} x_{ij}\\,W_{kj} + b_k')}</div>
    <p>Pick an output below (hover it, tab to it, or click it) and the two rows and the bias it was built from light up, with the arithmetic written out.</p>`),
    },
    {
      kind: 'widget', id: 'matmul-grid', className: 'lin-w lin-mg', label: 'One neuron, one row',
      mount: (host, D, ctx) => mountMatmulGrid(host, D, ctx),
    },
    {
      kind: 'text',
      html: (D, T) => prose(`<p>With the preset numbers, TorC printed <code>y</code> = [[${T('linear.y.value.0')}, ${T('linear.y.value.1')}], [${T('linear.y.value.2')}, ${T('linear.y.value.3')}]]; the grid recomputes exactly that. Things to try:</p>
    <ul>
      <li>Change <code>b[2]</code>. The <strong>whole second column</strong> of <code>y</code> moves by the same amount, for every sample: the bias is added to each sample’s answer of neuron 2.</li>
      <li>Change one entry in the first row of <code>x</code>. Only the <strong>first row</strong> of <code>y</code> changes: samples do not influence each other.</li>
      <li>Press <em>Set b = 0</em> (TorC’s starting bias) and <em>Copy neuron 1 into neuron 2</em>. Both columns of <code>y</code> become identical for every sample: the two neurons are twins and the forward pass cannot tell them apart. That is why weights must not start all equal (more in section 5).</li>
    </ul>`),
    },

    // ---------------------------------------------------------------------------------------------- 3 three ops
    {
      kind: 'text',
      html: () => prose(`<h2 id="ops">3 · Under the hood: three operations</h2>
    <p>The forward pass of <code>nn::Linear</code> is three operations (<code>src/nn/linear.cpp</code>): <code>add(matmul(x, transpose(weight)), bias)</code>. The layer keeps its three intermediate results private, so the graph below is built by hand from the same three TorC operations, with the loss added on top. Press <kbd>Next</kbd> (or <kbd>→</kbd>) to step forward, then backward; click a node to open its math window; <em>✎ Inputs</em> lets you change any number.</p>`),
    },
    {
      kind: 'widget', id: 'linear-lab', className: 'lin-labwrap', label: 'Interactive graph of the three operations',
      mount: (host, D, ctx) => {
        useBaseStyle();
        injectStyle('lin-lab-style', labCss);
        const T = ctx.makeT();
        const sc = D.scenes;
        const flags = ['output_equal', 'grads_equal', 'loss_equal'].map((k) => `<li class="flag${['linear', 'linear-dx'].every((id) => sc[id].parity[k]) ? '' : ' bad'}"><code>${k}</code></li>`).join('');
        host.innerHTML = `<div class="lin-parity"><strong>TorC’s <code>nn::Linear</code> output equals this hand-built graph</strong> for the preset numbers (largest difference found: ${T('linear.parity.max_abs_diff')}). Flags recorded for both scenes: <ul>${flags}</ul></div>
          <div class="lab" id="linear-lab-inner"></div>`;
        return mountGraphLab(host.querySelector('#linear-lab-inner'), { scenes: { linear: sc.linear, 'linear-dx': sc['linear-dx'] }, layouts: SCENES, defaultScene: 'linear' });
      },
    },

    // ---------------------------------------------------------------------------------------------- 4 backward
    {
      kind: 'text',
      html: (D, T) => prose(`<h2 id="backward">4 · Backward: what do W and b receive?</h2>
    <p>Put a loss on top: the mean squared error against targets <code>t</code>. In scene <code>linear</code> it is ${tm('L')} = ${T('linear.L.value')}. The loss hands <code>y</code> the gradient ${tm('\\overline{y} = 2\\,(y - t)/N')} (the widgets below call this matrix <code>dY</code>), where <em>N</em> is the number of outputs, here ${tm('\\overline{y}')} = [[${T('linear.y.grad.0')}, ${T('linear.y.grad.1')}], [${T('linear.y.grad.2')}, ${T('linear.y.grad.3')}]]. Three facts, each visible in the backward steps of the graph above:</p>
    <ol>
      <li><strong><code>add</code></strong> passes ${tm('\\overline{y}')} unchanged to both of its inputs. But <code>b</code> was a single row that got copied to every sample, so the copies’ gradients are <strong>added up</strong>: ${tm('\\overline{b}_k = \\sum_i \\overline{y}_{ik}')}, the column sums. For neuron 1: ${T('linear.y.grad.0')} + ${T('linear.y.grad.2')} = ${T('linear.b.grad.0')}; for neuron 2: ${T('linear.y.grad.1')} + ${T('linear.y.grad.3')} = ${T('linear.b.grad.1')}. (It is the same thing you saw for <code>b1</code> in <a href="../backprop-lab/#mlp">scene 3 of the Backprop lab</a>.) So the bias gradient is <em>not</em> a copy of ${tm('\\overline{y}')}.</li>
      <li><strong><code>matmul</code></strong> sends ${tm('x^{\\top}\\,\\overline{y}')} back to <code>Wt</code>.</li>
      <li><strong><code>transpose</code></strong> carries that back to <code>W</code> turned round: ${tm('\\overline{W} = \\overline{y}^{\\top}\\,x')}, shape (out, in) like <code>W</code> itself: [[${T('linear.W.grad.0')}, ${T('linear.W.grad.1')}, ${T('linear.W.grad.2')}], [${T('linear.W.grad.3')}, ${T('linear.W.grad.4')}, ${T('linear.W.grad.5')}]].</li>
    </ol>
    <p>Pick any gradient entry below to see which numbers it is the sum of.</p>`),
    },
    {
      kind: 'widget', id: 'linear-backward', className: 'lin-w lin-bw', label: 'Where the gradients come from',
      mount: (host, D, ctx) => mountLinearBackward(host, D, ctx),
    },
    {
      kind: 'text',
      html: (D, T) => prose(`<p><strong>What about <code>x</code>?</strong> In scene <code>linear</code> the input is data (dashed outline), so TorC computes no gradient for it. But if <code>x</code> were the output of an earlier layer, that layer would need ${tm('\\overline{x} = \\overline{y}\\,W')}: the gradient a layer hands backwards. Switch the widget above to the <code>sum</code> loss to see it.</p>`),
    },
    {
      kind: 'predict', id: 'predict-dx',
      prompt: (D, T) => `<p>With loss ${tm('S = \\mathrm{sum}(y)')}, every output has gradient 1. What is each row of <code>x</code>’s gradient, given <code>W</code> = [[${T('linear.W.value.0')}, ${T('linear.W.value.1')}, ${T('linear.W.value.2')}], [${T('linear.W.value.3')}, ${T('linear.W.value.4')}, ${T('linear.W.value.5')}]]?</p>`,
      reveal: (D, T) => `<p>Each row is ${tm('1\\cdot W_{1j} + 1\\cdot W_{2j}')}, the <strong>column sums of <code>W</code></strong>: ${T('linear.W.value.0')} + ${T('linear.W.value.3')} = ${T('linear-dx.x.grad.0')}, ${T('linear.W.value.1')} + ${T('linear.W.value.4')} = ${T('linear-dx.x.grad.1')}, ${T('linear.W.value.2')} + ${T('linear.W.value.5')} = ${T('linear-dx.x.grad.2')}. TorC printed ${T('linear-dx.x.grad.0')}, ${T('linear-dx.x.grad.1')}, ${T('linear-dx.x.grad.2')} for the first sample and the same row for the second (scene <code>linear-dx</code>).</p>`,
    },

    // ---------------------------------------------------------------------------------------------- 5 init
    {
      kind: 'text',
      html: (D, T) => prose(`<h2 id="init">5 · Why initialise like that?</h2>
    <p>The constructor is <code>Linear(in, out, init_std = 0, seed)</code>. The <strong>bias starts at zero</strong>. The <strong>weights start random</strong>: with the default <code>init_std = 0</code> each is drawn from a normal distribution with mean 0 and standard deviation ${tm('\\sqrt{2/\\text{in}}')} (<code>kaiming_normal_std</code>, “Kaiming” or “He” initialisation); a positive <code>init_std</code> replaces that number. Choose a recorded case to see the weights TorC really drew:</p>`),
    },
    {
      kind: 'widget', id: 'init-hist', className: 'lin-w lin-is', label: 'The weights TorC drew',
      mount: (host, D, ctx) => mountWeightHistogram(host, D, ctx),
    },
    {
      kind: 'text',
      html: (D, T) => prose(`<p>For <code>Linear(784, 32)</code> the formula predicts a standard deviation of ${T(`${P}.init_stats.2.expected_std`)}; the ${makeExact(D)(`${P}.init_stats.2.numel`)} weights TorC drew have ${T(`${P}.init_stats.2.std`)}, and their mean is ${T(`${P}.init_stats.2.mean`)}, close to zero. Now pick the three <code>Linear(784, 32)</code> cases one after the other: the bars keep their shape and only the numbers on the horizontal axis change. <strong>The initialisation chooses the scale of the bell, nothing else.</strong> So which scale is right?</p>
    <h3 style="margin-top:1.4rem">Why ${tm('\\sqrt{2/\\text{in}}')}?</h3>
    <p><em>Expected mathematics (not something TorC prints).</em> Every output is a sum of <code>in</code> products <code>w·a</code>. If the weights are independent with variance ${tm('s^2')}, that sum has variance about ${tm('\\text{in}\\cdot s^2\\cdot(\\text{typical }a^2)')}. ReLU then zeroes roughly half of the values, which halves the typical ${tm('a^2')}. Choosing ${tm('s^2 = 2/\\text{in}')} makes the two effects cancel: the typical size of the activations neither grows nor fades from layer to layer. With the default, about half of the ReLU outputs are exactly zero (${T(SIG(0, 1, 'frac_zero'))} of them after layer 1).</p>
    <p>Test it: send a batch through <strong>six</strong> layers of <code>Linear(256, 256)</code> + ReLU and look at the spread (standard deviation) of the numbers that come out of each layer.</p>`),
    },
    {
      kind: 'widget', id: 'init-signal', className: 'lin-w lin-is', label: 'Signal through six layers',
      mount: (host, D, ctx) => mountInitSignal(host, D, ctx),
    },
    {
      kind: 'text',
      html: (D, T) => { const C = makeCalc(D), S = makeSci(D); return prose(`<ul>
      <li><strong>Kaiming (default).</strong> The spread stays close to the same value: ${T(SIG(0, 1))} after layer 1 and ${T(SIG(0, 6))} after layer 6 in this run; over ${T(`${P}.signal.0.n_extra_seeds`)} other random draws of the weights, the layer-6 value lies between ${T(`${P}.signal.0.across_seeds.5.post_std_min`)} and ${T(`${P}.signal.0.across_seeds.5.post_std_max`)}. It wobbles because every neuron has only a few hundred random weights, but it neither explodes nor fades.</li>
      <li><strong><code>init_std</code> 0.01 (too small).</strong> The signal fades: ${S(SIG(1, 1))} after layer 1, ${S(SIG(1, 6))} after layer 6. Each layer multiplies it by roughly the same factor, between ${C('shrink-lo')} and ${C('shrink-hi')}; the formula ${tm('\\sqrt{\\text{in}/2}\\cdot s')} (here in = 256) predicts ${C('shrink-pred')}.</li>
      <li><strong><code>init_std</code> 1.0 (too big).</strong> The signal explodes: ${S(SIG(2, 1))} after layer 1, ${S(SIG(2, 6))} after layer 6, a factor between ${C('grow-lo')} and ${C('grow-hi')} per layer (the formula predicts ${C('grow-pred')}).</li>
    </ul>
    <p>So a bigger initialisation does not “learn faster”: by layer 6 the numbers are far outside anything a loss function can use sensibly. <em>Expected, not measured here:</em> the gradients travel backwards through the same weights, so they blow up or fade in the same way. Likewise, <strong>all-zero weights are not a safe start</strong>: every neuron of a layer would compute the same thing (you saw the twins in section 2), and the forward pass would give them no reason to differ.</p>`); },
    },

    // ---------------------------------------------------------------------------------------------- misconceptions
    {
      kind: 'text',
      html: (D, T) => { const S = makeSci(D); return `<section class="lin-prose" aria-labelledby="myths">
    <h2 id="myths" style="max-width:50rem">Six things that look right but are not</h2>
    <div class="cards">
      <div class="card"><h3>“W has shape (in, out).”</h3><p>It is (out, in): for <code>Linear(3, 2)</code> the weight is ${shp(T, 'linear.W.value.shape')}. The (in, out) shape in the error message is <code>Wᵀ</code>.</p></div>
      <div class="card"><h3>“Linear includes the activation.”</h3><p>No. The graph in section 3 has three operations and none of them is ReLU. In the MLP, ReLU is its own module with ${T(`${MLP}.layers.1.count`)} parameters.</p></div>
      <div class="card"><h3>“Zero weights are a fine start.”</h3><p>No. Only the <em>bias</em> starts at zero. Equal weights make copies: every neuron computes the same output and the forward pass gives them no reason to differ. Random weights break the tie.</p></div>
      <div class="card"><h3>“Bigger initial weights learn faster.”</h3><p>No. With <code>init_std</code> 1.0 the activations grow to ${S(SIG(2, 6))} in six layers; with the default they stay near ${T(SIG(0, 6))}.</p></div>
      <div class="card"><h3>“There is one bias per input feature.”</h3><p>One per <em>output neuron</em>: <code>Linear(784, 32)</code> has ${T(`${BIAS}.numel`)} biases, not 784.</p></div>
      <div class="card"><h3>“The input needs a gradient.”</h3><p>Usually not: <code>x</code> is data, and TorC stores no gradient for it in scene <code>linear</code>. It only matters when <code>x</code> comes from an earlier layer.</p></div>
    </div>
  </section>`; },
    },

    // ---------------------------------------------------------------------------------------------- recap + checks
    {
      kind: 'text',
      html: () => prose(`<h2 id="recap">Recap</h2>
    <ul>
      <li><code>Linear(in, out)</code>: x (B, in), W (out, in), b (out), y (B, out); parameters ${tm('\\text{in}\\cdot\\text{out} + \\text{out}')}.</li>
      <li>Row <em>k</em> of <code>W</code> is neuron <em>k</em>; ${tm('y_{ik}')} is the dot product of sample <em>i</em> with that row, plus ${tm('b_k')}.</li>
      <li>Inside: <code>transpose</code>, <code>matmul</code>, <code>add</code>. Backward: ${tm('\\overline{W} = \\overline{y}^{\\top}x')}, ${tm('\\overline{b}')} = column sums of ${tm('\\overline{y}')}, ${tm('\\overline{x} = \\overline{y}W')}.</li>
      <li>Weights start normal with standard deviation ${tm('\\sqrt{2/\\text{in}}')}, the bias at zero.</li>
    </ul>
    <h2 id="check" style="margin-top:2rem">Check yourself</h2>`),
    },
    {
      kind: 'check', id: 'check-shapes',
      prompt: () => `<p>You create <code>Linear(784, 32)</code> and feed it a batch of 8 samples. Predict the shapes of <code>W</code>, <code>b</code>, <code>x</code> and <code>y</code>, and the number of parameters. Then predict what TorC says if <code>x</code> has shape (8, 10) instead.</p>`,
      reveal: (D, T) => { const s = linearShapes(784, 32, 8); return `<p><code>W</code> ${fmtShape(s.W)}, <code>b</code> ${fmtShape(s.b)}, <code>x</code> ${fmtShape(s.x)}, <code>y</code> ${fmtShape(s.y)}. Parameters: ${makeExact(D)(`${WEIGHT}.numel`)} + ${T(`${BIAS}.numel`)} = ${makeExact(D)(`${P}.param_counts.0.total`)}, whatever the batch size.</p>
      <p>With (8, 10), the recorded message is <code>${escapeHtml(recorded(D, 'linear-784x32-(8, 10)').what)}</code>: your input and then <code>Wᵀ</code> (784, 32); 10 ≠ 784.</p>`; },
    },
    {
      kind: 'check', id: 'check-bias',
      prompt: (D, T) => `<p>In the grid, <code>b</code> = [${T('linear.b.value.0')}, ${T('linear.b.value.1')}]. Before you try it: if you raise <code>b[2]</code> by 1 (to 0.8), which entries of <code>y</code> change, and to what? (Remember <code>y</code> = [[${T('linear.y.value.0')}, ${T('linear.y.value.1')}], [${T('linear.y.value.2')}, ${T('linear.y.value.3')}]].)</p>`,
      reveal: (D, T) => { const C = makeCalc(D); return `<p>Only the second column changes, in both rows, by exactly +1: <code>y[1,2]</code> becomes ${T('linear.y.value.1')} + 1 = ${C('edit-y12')} and <code>y[2,2]</code> becomes ${T('linear.y.value.3')} + 1 = ${C('edit-y22')}. The first column does not move, because neuron 1 has its own bias. Check it in the grid above.</p>`; },
    },
    {
      kind: 'check', id: 'check-out-in',
      prompt: () => `<p>Why is the weight stored as (out, in) when the multiplication needs (in, out)? Use the error message of the gotcha above as a hint.</p>`,
      reveal: (D, T) => `<p>Row <em>k</em> of an (out, in) weight holds <strong>all the weights of neuron <em>k</em></strong> in one place (row-major storage keeps a row together), and the same row is what the dot product with a sample uses. <code>forward()</code> transposes the weight to (in, out) right before the <code>matmul</code>, which is why the error message shows ${shp(T, 'linear.Wt.value.shape')} for a layer whose weight is ${shp(T, 'linear.W.value.shape')}. The price is one <code>transpose</code> node in the graph.</p>`,
    },
  ],
  footer: (D) => `<p>Every number marked as measured was printed by the real TorC library (<code>tools/trace/trace_linear.cpp</code>, TorC ${escapeHtml(D.files.linear.library.ref)}) and the gradients agree with finite differences. The shape explorer, the grid and the sums in the gradient widget are recomputed in your browser; the weight histogram and the six-layer table are precomputed by the library, because a browser cannot reproduce C++’s random numbers. Source: <a href="https://github.com/IvorLipic/TorC">github.com/IvorLipic/TorC</a>.</p>`,
};
