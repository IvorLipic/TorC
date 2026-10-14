// Lesson "sgd-step": one real optimizer.step(), the loss landscape, the learning rate, zero_grad and a first look at momentum.
// Numbers in prose come from data/optim.json (a real TorC run) through T(); the widgets are in widgets/ (contour2d, lr-lab,
// loop-timeline, sgd-lifecycle, sgd-update-window) and the optimizer port in ../optim.js. Library lines are quoted from SOURCES.
import { mountGraphLab } from '../widgets/graph-lab.js';
import { mountUpdateWindow } from '../widgets/sgd-update-window.js';
import { mountLifecycle } from '../widgets/sgd-lifecycle.js';
import { mountDescent } from '../widgets/lr-lab.js';
import { mountLoopTimeline, DEFAULT_ITERATION } from '../widgets/loop-timeline.js';
import { fmt } from '../widgets/sgd-ui.js';
import { SOURCES } from '../widgets/sgd-code.js';
import { makeLayouts, CONCEPTS } from './sgd-step.scenes.js';
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { REPO_URL } from '../core/chrome.js';

// one real line of the library, copied from SOURCES (so it can never drift from the quoted source)
const quote = (key, n) => { const s = SOURCES[key]; return `<code class="quote" data-src="${key}:${n}">${escapeHtml(s.lines[n - s.from].trim())}</code>`; };
const M = (tex, display = false) => math(tex, display);
const step = (run, i, key) => `files.optim.runs.${run}.steps.${i}.${key}`;
const fin = (run, key) => `files.optim.runs.${run}.final.${key}`;
const loop = (i, key) => `files.optim.no_zero_grad.training_loop.iterations.${i}.${key}`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const ten = (D) => WORDS[D.files.optim.runs[0].steps.length] ?? String(D.files.optim.runs[0].steps.length);   // how many updates the recorded runs have
const derived = (x, why) => `<strong class="derived" title="${why}">${fmt(x)}</strong>`;   // a number worked out from recorded ones (not itself recorded)
const acc = (i, key) => `files.optim.no_zero_grad.accumulate_only.${i}.${key}`;

export default {
  id: 'sgd-step',
  title: 'The SGD update step',
  trace: ['optim.json'],
  sections: [
    // ---- hero + the problem ------------------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: () => `<section class="hero">
    <h1>The SGD update step</h1>
    <p class="lead">Backprop gave us a gradient. This lesson spends it: one real <code>optimizer.step()</code>, how a learning rate can be too big, the one line everybody forgets (<code>zero_grad()</code>) and a first look at momentum. Every number comes from <a href="${REPO_URL}">TorC</a> or is recomputed with the same rules.</p>
    <p class="hint">About 20 minutes. It helps to have done <a href="../why-gradients/">Why following gradients works</a> and the <a href="../backprop-lab/">Backprop lab</a> first.</p>
  </section>`,
    },
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="setup">
    <h2 id="setup">The problem: fit a line to three points</h2>
    <p>Three points lie exactly on a line: (x, t) = (${T('files.optim.data.x.0')}, ${T('files.optim.data.t.0')}), (${T('files.optim.data.x.1')}, ${T('files.optim.data.t.1')}), (${T('files.optim.data.x.2')}, ${T('files.optim.data.t.2')}). TorC does not know that. It starts from the model <code>y = w·x + b</code> with <code>w = b = 0</code> (a flat line at zero) and has to find better numbers on its own. In the library that model is <code>nn::Linear(1, 1)</code>: a weight <code>W</code> and a bias <code>b</code> are the only things that can change.</p>
    <p>“Better” needs a number, so we use the mean squared error ${M('L = \\tfrac{1}{N}\\sum_i (w\\,x_i + b - t_i)^2', false)} (<code>nn::MSELoss</code>), where N = ${T('files.optim.data.n')} is the number of points. With <code>w = b = 0</code> every prediction is 0, so the loss is just the average of the squared targets: <code>L</code> = ${T('sgd.L.value')}. The goal of training is to make it small. (The best line is the one that made the data: <code>w</code> = ${T('files.optim.data.optimum.w')}, <code>b</code> = ${T('files.optim.data.optimum.b')}, loss ${T('files.optim.data.optimum.loss')}.)</p>
  </section>`,
    },

    // ---- 1 the graph you already know ---------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: () => `<section aria-labelledby="sec-graph">
    <h2 id="sec-graph">1 · The graph you already know</h2>
    <p>This is the same kind of graph as in the Backprop lab, now with real names. <code>x</code> and <code>t</code> are data. <code>W</code> and <code>b</code> are the <strong>parameters</strong>. <code>Wt</code> is <code>W</code> transposed, <code>xW</code> the matrix product, <code>y</code> the prediction and <code>L</code> the loss. Press <kbd>Next ▶</kbd> (or <kbd>→</kbd>) until the end: forward computes <code>L</code>, then backward leaves a gradient on <code>W</code> and <code>b</code>. When you get there, ask yourself: <em>has anything been learned yet?</em></p>
  </section>`,
    },
    {
      kind: 'widget', id: 'lab', className: 'lab', label: 'Interactive graph: forward and backward for the line fit',
      mount: (host, D, ctx) => mountGraphLab(host, { scenes: D.scenes, layouts: makeLayouts(ctx.makeT()), defaultScene: 'sgd' }),
    },

    // ---- 2 who touches what ---------------------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: () => `<section aria-labelledby="who">
    <h2 id="who">2 · Who touches what</h2>
    <p>After backward, <code>W.grad</code> and <code>b.grad</code> hold numbers, but <code>W</code> and <code>b</code> are untouched. Changing them is a different call, made by a different object. One training iteration is three calls, and each has exactly one job:</p>
    <div class="sgd-scroll on-bg"><table class="sgd-tab text">
      <caption>The three calls of a training iteration</caption>
      <thead><tr><th scope="col">call</th><th scope="col">what it does</th><th scope="col">reads</th><th scope="col">writes</th></tr></thead>
      <tbody>
        <tr><th scope="row"><code>loss.backward()</code></th><td>computes gradients</td><td>the graph of this forward pass</td><td><code>.grad</code> of every tensor that needs one (adds to it)</td></tr>
        <tr><th scope="row"><code>optimizer.step()</code></th><td>moves the parameters</td><td><code>.grad</code></td><td><code>.data</code></td></tr>
        <tr><th scope="row"><code>optimizer.zero_grad()</code></th><td>forgets old gradients</td><td>nothing</td><td><code>.grad</code> (empties it)</td></tr>
      </tbody></table></div>
    <p><strong>The optimizer never computes a gradient.</strong> <code>optim::SGD</code> is built from a list of parameters, a learning rate and a momentum. It has never heard of the loss or of the graph: it only reads whatever <code>backward()</code> left in each parameter’s <code>.grad</code>. And <code>step()</code> does <em>not</em> clear <code>.grad</code>; that belongs to <code>zero_grad()</code>, which section 6 is about.</p>
    <p>Poke at it below. Press the calls in any order and watch <code>.data</code> and <code>.grad</code> of each tensor; the code on the right is the real function that runs. Worth trying: <code>step()</code> first; (forward +) <code>backward()</code> twice; <code>backward()</code>, <code>step()</code>, and then look at <code>W.grad</code> again. One detail: <code>backward()</code> uses up the graph it walks, so a second <code>backward()</code> on the <em>same</em> forward pass does nothing. Accumulation needs a fresh forward pass first, which is why the sandbox button runs one every time.</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'lifecycle', className: 'widget', label: 'Sandbox: backward, step and zero_grad on W and b',
      mount: (host, D) => mountLifecycle(host, D),
    },

    // ---- 3 the update on paper ------------------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: () => `<section aria-labelledby="paper">
    <h2 id="paper">3 · The update, on paper</h2>
    <p>Plain SGD (momentum 0) does the same thing to every parameter that has a gradient:</p>
    <div class="math">${M('p \\leftarrow p - \\eta\\,\\overline{p}', true)}</div>
    <p>Here η (eta) is the learning rate, <code>lr</code> in the code, and ${M('\\overline{p}', false)} is the gradient that <code>backward()</code> left in <code>p.grad()</code>. In the library the whole rule is one line: ${quote('step', 27)} (plus a momentum branch we meet in section 7). The minus sign is what makes it <em>descent</em>: the gradient points uphill, so we step the other way.</p>
    <p>The name SGD, stochastic gradient descent, comes from computing the gradient on random mini-batches of the data. Here every update uses all the points, so it is the plain version of the same rule.</p>
  </section>`,
    },
    {
      kind: 'predict', id: 'p-update',
      prompt: (D, T) => `<p>Right now <code>W</code> = ${T('sgd.W.value')}, <code>W.grad</code> = ${T('sgd.W.grad')} and <code>b.grad</code> = ${T('sgd.b.grad')}. The optimizer was created with <code>lr</code> = ${T('files.optim.runs.0.lr')}. After one <code>step()</code>, does <code>W</code> go up or down, and to what number? What about <code>b</code>?</p>`,
      reveal: (D, T) => `<p>Up, because the gradient is negative: raising <code>W</code> lowers the loss. <code>W</code> ← 0 − ${T('files.optim.runs.0.lr')} · (${T('sgd.W.grad')}) = <strong>${T(step(0, 1, 'w'))}</strong> and <code>b</code> ← 0 − ${T('files.optim.runs.0.lr')} · (${T('sgd.b.grad')}) = <strong>${T(step(0, 1, 'b'))}</strong>. Check it against the table below.</p>`,
    },
    {
      kind: 'widget', id: 'update-window', className: 'widget', label: 'The update on paper: rule, numbers and a before and after table',
      mount: (host, D) => mountUpdateWindow(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="paper-notes" class="notes">
    <h3 id="paper-notes">Two things to notice</h3>
    <ul>
      <li>Only parameters appear in the table. <code>x</code> and <code>t</code> have no <code>.grad</code>, so <code>step()</code> never sees them: <strong>parameters are updated, inputs are not.</strong></li>
      <li>One update already changed the loss from ${T(step(0, 0, 'loss'))} to ${T(step(0, 1, 'loss'))}. Repeating the recipe (forward, backward, step) is all that training is.</li>
    </ul>
  </section>`,
    },

    // ---- 4 the loss landscape ------------------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="land">
    <h2 id="land">4 · The loss as a landscape</h2>
    <p>Every pair (w, b) is a different line, and every line has a loss. Draw the loss over the (w, b) plane and you get a bowl: dark blue is low, orange is high, and the thin white lines are contours where the loss has the same value (their numbers are written on them). The star is the best line, <code>w</code> = ${T('files.optim.data.optimum.w')}, <code>b</code> = ${T('files.optim.data.optimum.b')}. The picture below is computed live from the same rules as the graph above.</p>
    <p>The gradient (${T('sgd.W.grad')}, ${T('sgd.b.grad')}) points <em>uphill</em>, where the loss grows fastest, and <code>step()</code> moves the other way. The dashed arrow shows where the next <code>step()</code> would land. Press <strong>Step</strong> and watch the trail and the table grow.</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'landscape', className: 'widget', label: 'Loss landscape over w and b with a Step button',
      mount: (host, D) => mountDescent(host, D, { mode: 'step' }),
    },
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="land-notes" class="notes">
    <h3 id="land-notes">What the trail tells you</h3>
    <ul>
      <li>The first moves are long and the loss falls fast: ${T(step(0, 0, 'loss'))} → ${T(step(0, 1, 'loss'))} → ${T(step(0, 2, 'loss'))} → ${T(step(0, 3, 'loss'))} → ${T(step(0, 4, 'loss'))}. Each gradient is smaller than the one before, so each move is shorter.</li>
      <li>The path does not aim at the star. It slides down the steep wall first and then creeps along the valley floor, so after ${ten(D)} updates the loss is still ${T(fin(0, 'loss'))}, not 0, and (w, b) = (${T(fin(0, 'w'))}, ${T(fin(0, 'b'))}), close to but not at (${T('files.optim.data.optimum.w')}, ${T('files.optim.data.optimum.b')}). More on why in the next section.</li>
      <li>The same <code>lr</code> gave a long first move (<code>w</code> from 0 to ${T(step(0, 1, 'w'))}) and tiny later ones: <strong>the step is <code>lr</code> times the gradient</strong>, not a fixed distance.</li>
    </ul>
  </section>`,
    },

    // ---- 5 the learning rate -------------------------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: () => `<section aria-labelledby="lr">
    <h2 id="lr">5 · The learning rate</h2>
    <p>Everything above used one learning rate. It multiplies the gradient, so it sets the size of every move, and that makes it the most important number you pick by hand.</p>
  </section>`,
    },
    {
      kind: 'predict', id: 'p-lr',
      prompt: (D, T) => `<p>We used <code>lr</code> = ${T('files.optim.runs.0.lr')}. If we make it four times bigger, ${T('files.optim.runs.3.lr')}, do we reach the bottom four times faster? Decide, then try it in the lab.</p>`,
      reveal: (D, T) => `<p>No. At ${T('files.optim.runs.3.lr')} the loss grows at every update: ${T(step(3, 0, 'loss'))}, ${T(step(3, 1, 'loss'))}, ${T(step(3, 2, 'loss'))}, ${T(step(3, 3, 'loss'))}, … and is ${T(fin(3, 'loss'))} after ${ten(D)}. A bigger step is only better up to a limit; past it, the steps overshoot the valley by more than they gained.</p>`,
    },
    {
      kind: 'widget', id: 'lr-lab', className: 'widget', label: 'Learning-rate lab: slider, presets and the critical rate',
      mount: (host, D) => mountDescent(host, D, { mode: 'lr' }),
    },
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="lr-notes" class="notes">
    <h3 id="lr-notes">What the four presets show</h3>
    <ul>
      <li><strong>${T('files.optim.runs.0.lr')}</strong>: steady, the loss falls at every update.</li>
      <li><strong>${T('files.optim.runs.1.lr')}</strong>: the first update jumps <em>past</em> the best <code>w</code> (to ${T(step(1, 1, 'w'))}, the best is ${T('files.optim.data.optimum.w')}) and yet the loss is already ${T(step(1, 1, 'loss'))}, much lower than the ${T(step(0, 1, 'loss'))} the small step reached. Overshooting a little is not a disaster.</li>
      <li><strong>${T('files.optim.runs.2.lr')}</strong>: <code>w</code> zig-zags across the minimum (${T(step(2, 1, 'w'))}, ${T(step(2, 2, 'w'))}, ${T(step(2, 3, 'w'))}, ${T(step(2, 4, 'w'))}, …) but each swing is smaller, and the loss still falls at every update.</li>
      <li><strong>${T('files.optim.runs.3.lr')}</strong>: each swing is bigger than the last (<code>w</code> = ${T(step(3, 1, 'w'))}, ${T(step(3, 2, 'w'))}, ${T(step(3, 3, 'w'))}, …) and the loss explodes.</li>
    </ul>
    <p><strong>The learning rate is not “how fast it gets correct”.</strong> It is the size of a step, and a step can be too long to be useful.</p>
    <details>
      <summary>Where does the limit ${T('files.optim.hessian.critical_lr')} come from? (derived mathematics, not a TorC output)</summary>
      <p>This loss is exactly a bowl (a quadratic in w and b), described by its matrix of second derivatives, the Hessian, ${M('H = \\tfrac{2}{N}\\begin{bmatrix}\\sum x^2 & \\sum x\\\\ \\sum x & N\\end{bmatrix}', false)}. For our three points that is ${'<code>[[' + [T('files.optim.hessian.matrix.0.0'), T('files.optim.hessian.matrix.0.1')].join(', ') + '], [' + [T('files.optim.hessian.matrix.1.0'), T('files.optim.hessian.matrix.1.1')].join(', ') + ']]</code>'}, with eigenvalues λ = ${T('files.optim.hessian.eigenvalues.0')} (steepest direction) and λ = ${T('files.optim.hessian.eigenvalues.1')} (flattest).</p>
      <p>One SGD update multiplies the distance from the minimum by ${M('I - \\eta H', false)}, which along an eigen-direction means “times ${M('1 - \\eta\\lambda', false)}”. The error shrinks only if ${M('|1-\\eta\\lambda| < 1', false)} in <em>both</em> directions, i.e. ${M('\\eta < 2/\\lambda_{\\max}', false)}: here 2 / ${T('files.optim.hessian.eigenvalues.0')} = <strong>${T('files.optim.hessian.critical_lr')}</strong>.</p>
      <p>The runs agree. ${T('files.optim.runs.0.lr')}, ${T('files.optim.runs.1.lr')} and ${T('files.optim.runs.2.lr')} are below the limit and converge. ${T('files.optim.runs.3.lr')} is above it: the error is multiplied by a number of size ${T('files.optim.runs.3.analysis.spectral_radius')} at every update (and its sign flips, which is the zig-zag), so it grows.</p>
      <p>The flat direction explains the slow creep along the valley: there the factor is only about ${T('files.optim.runs.0.analysis.spectral_radius')} per update at <code>lr</code> ${T('files.optim.runs.0.lr')}, so the last bit of error disappears slowly. That is why ${ten(D)} updates are not enough to reach loss 0.</p>
    </details>
  </section>`,
    },

    // ---- 6 the zero_grad trap --------------------------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="zg">
    <h2 id="zg">6 · The zero_grad trap</h2>
    <p>Back in section 2, <code>backward()</code> <em>added</em> to <code>.grad</code>. Here is what that means in a real TorC run: three forward-and-backward passes at the same <code>W</code> and <code>b</code>, with no <code>zero_grad()</code> in between. <code>W.grad</code> after each pass: <strong>${T(acc(0, 'gw'))}</strong>, <strong>${T(acc(1, 'gw'))}</strong>, <strong>${T(acc(2, 'gw'))}</strong>. The same gradient is counted again and again: nothing is replaced.</p>
    <p>A training loop calls <code>step()</code> every iteration. If it never clears <code>.grad</code>, then <code>step()</code> reads the <em>sum</em> of every gradient since the start.</p>
  </section>`,
    },
    {
      kind: 'predict', id: 'p-zero',
      prompt: (D, T) => `<p>Take the training loop of this lesson (<code>lr</code> = ${T('files.optim.runs.0.lr')}) and delete the <code>optimizer.zero_grad()</code> line. After ${ten(D)} iterations, will it (a) learn just as well, (b) learn more slowly, or (c) do something else? Decide, then untick the box below.</p>`,
      reveal: (D, T) => `<p>(c). It does not converge at all. With <code>zero_grad()</code> the loss falls to ${T(step(0, 9, 'loss'))}. Without it the loss goes ${T(loop(0, 'loss'))}, ${T(loop(1, 'loss'))}, ${T(loop(2, 'loss'))}, then <em>up</em> to ${T(loop(3, 'loss'))} and ${T(loop(4, 'loss'))}, and it is still ${T(loop(9, 'loss'))} at iteration 10.</p>`,
    },
    {
      kind: 'widget', id: 'loop', className: 'widget', label: 'A training loop with and without zero_grad',
      mount: (host, D) => mountLoopTimeline(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="zg-notes" class="notes">
    <h3 id="zg-notes">Why it bounces</h3>
    <p>Look at iteration ${DEFAULT_ITERATION} (where the widget starts). By then <code>W</code> = ${T(loop(DEFAULT_ITERATION - 1, 'w'))} has already passed the best value ${T('files.optim.data.optimum.w')}, so this pass’s gradient is ${T(loop(DEFAULT_ITERATION - 1, 'gw_this_pass'))}: “go back down”. But <code>.grad</code> still holds the ${T(loop(DEFAULT_ITERATION - 2, 'gw'))} left over from the ${WORDS[DEFAULT_ITERATION - 1]} iterations before it, so after <code>backward()</code> it is ${T(loop(DEFAULT_ITERATION - 1, 'gw'))} and <code>step()</code> pushes <code>W</code> <em>further up</em>, to ${T(loop(DEFAULT_ITERATION, 'w'))}. The accumulated gradient remembers where you were, not where you are.</p>
    <p>That running sum of every past gradient, with no forgetting at all, is exactly the momentum rule of section 7 with μ = 1, except that nobody asked for it.</p>
    <p><strong><code>zero_grad()</code> is not optional.</strong> In TorC’s own example the three lines run in this order every iteration: ${quote('loop', 66)} ${quote('loop', 67)} ${quote('loop', 68)}</p>
  </section>`,
    },
    {
      kind: 'gotcha',
      title: 'Why did my parameters never move? (TorC says nothing)',
      html: (D, T) => `<p>The opposite mistake is just as quiet. Every one of these left <code>(w, b)</code> exactly at (${T('files.optim.skipped_step.step_without_backward.before.w')}, ${T('files.optim.skipped_step.step_without_backward.before.b')}) in a real TorC run, and <strong>none of them threw an exception</strong>:</p>
      <ul>
        <li><code>step()</code> on parameters that never had <code>backward()</code> called: still (${T('files.optim.skipped_step.step_without_backward.after.w')}, ${T('files.optim.skipped_step.step_without_backward.after.b')}).</li>
        <li><code>backward()</code>, then <code>zero_grad()</code>, then <code>step()</code>: still (${T('files.optim.skipped_step.step_after_zero_grad.after.w')}, ${T('files.optim.skipped_step.step_after_zero_grad.after.b')}).</li>
        <li>For comparison, <code>backward()</code> then <code>step()</code> gives (${T('files.optim.skipped_step.step_after_backward.after.w')}, ${T('files.optim.skipped_step.step_after_backward.after.b')}).</li>
      </ul>
      <p>The cause is one line in <code>SGD::step()</code>: ${quote('step', 20)} A parameter without a gradient is skipped without a word. If your loss never moves, check the order of your three calls first. (The constructor, by contrast, is strict: <code>lr</code> ≤ 0 throws <code>${escapeHtml(D.files.optim.ctor_errors[0].exception)}: ${escapeHtml(D.files.optim.ctor_errors[0].what)}</code>, and a negative momentum throws <code>${escapeHtml(D.files.optim.ctor_errors[2].what)}</code>.)</p>`,
    },
    {
      kind: 'check', id: 'c-zero',
      prompt: (D, T) => `<p>You run a fresh forward pass and <code>loss.backward()</code> three times at the same <code>W</code> and <code>b</code>, with no <code>zero_grad()</code> in between, and then call <code>step()</code> once with <code>lr</code> ${T('files.optim.runs.0.lr')}. Both start at ${T('sgd.W.value')}. What are <code>W</code> and <code>b</code> afterwards?</p>`,
      reveal: (D, T) => { const lr = T.num('files.optim.runs.0.lr'); return `<p><code>W.grad</code> is ${T(acc(2, 'gw'))} and <code>b.grad</code> is ${T(acc(2, 'gb'))} (three times one pass each), so <code>W</code> ← 0 − ${T('files.optim.runs.0.lr')} · (${T(acc(2, 'gw'))}) = ${derived(-lr * T.num(acc(2, 'gw')), 'worked out from the recorded gradient')} and <code>b</code> ← 0 − ${T('files.optim.runs.0.lr')} · (${T(acc(2, 'gb'))}) = ${derived(-lr * T.num(acc(2, 'gb')), 'worked out from the recorded gradient')}. That is three times the step of a single pass (which gave ${T(step(0, 1, 'w'))} and ${T(step(0, 1, 'b'))}), just because the gradient was never cleared.</p>`; },
    },

    // ---- 7 momentum preview -----------------------------------------------------------------------------------------------------------
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="mom">
    <h2 id="mom">7 · A first look at momentum</h2>
    <p>Plain SGD uses only the current gradient. With momentum, TorC keeps a <em>velocity</em> <code>v</code> for every parameter, and moves along that. The velocity is a <em>decaying sum</em> of past gradients: at each update the old velocity is multiplied by μ (a value below 1 makes old gradients fade) and the new gradient is added. When successive gradients agree, the steps therefore grow to roughly 1/(1 − μ) times a plain step:</p>
    <div class="math">${M('v \\leftarrow \\mu\\,v + \\overline{p}\\qquad\\qquad p \\leftarrow p - \\eta\\,v', true)}</div>
    <p>These are two lines of the library: ${quote('step', 24)} and ${quote('step', 25)} The velocity starts at 0, so the very first update is the plain one (<code>w</code> = ${T(step(4, 1, 'w'))} after update 1 for μ = ${T('files.optim.runs.4.momentum')}, the same as for μ = 0). From then on the velocity remembers where you have been heading. With μ = 0 the library skips this branch.</p>
    <p>Slide μ and watch the loss curve. Compare it with the grey dashed line, which is μ = 0.</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'momentum', className: 'widget', label: 'Momentum lab: slider, presets and the recorded TorC runs',
      mount: (host, D) => mountDescent(host, D, { mode: 'momentum' }),
    },
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="mom-notes" class="notes">
    <h3 id="mom-notes">What to notice</h3>
    <ul>
      <li>With μ = ${T('files.optim.runs.4.momentum')} the loss reaches ${T(step(4, 2, 'loss'))} after two updates, far below the ${T(step(0, 2, 'loss'))} of plain SGD, and then <em>rises</em> to ${T(step(4, 3, 'loss'))}: the velocity carried it past the valley floor. It does not settle steadily: the loss is ${T(step(4, 9, 'loss'))} before the last update and ${T(fin(4, 'loss'))} after it, up again.</li>
      <li>With μ = ${T('files.optim.runs.5.momentum')} the overshoots are large (loss ${T(step(5, 3, 'loss'))} at update 3) and after ${ten(D)} updates it has not settled (${T(fin(5, 'loss'))}).</li>
      <li><strong>The loss does not have to fall at every update.</strong> That is normal for momentum and does not mean something is broken.</li>
    </ul>
    <p>Momentum is a tool with a price, not a free speed-up. A later lesson takes it apart; here you only need the formula and the fact that the optimizer now has memory.</p>
  </section>`,
    },
    {
      kind: 'check', id: 'c-mom',
      prompt: (D, T) => `<p>With μ = ${T('files.optim.runs.5.momentum')} the first update equals the plain one: <code>W</code> goes from 0 to ${T(step(5, 1, 'w'))}. Will the second update move <code>W</code> by more than plain SGD’s second update (${T(step(0, 1, 'w'))} → ${T(step(0, 2, 'w'))}), by the same amount, or by less? Why?</p>`,
      reveal: (D, T) => { const mu = T.num('files.optim.runs.5.momentum'), lr = T.num('files.optim.runs.5.lr'); const v = mu * T.num(step(5, 0, 'gw')) + T.num(step(5, 1, 'gw')); return `<p>More: <code>W</code> goes ${T(step(5, 1, 'w'))} → <strong>${T(step(5, 2, 'w'))}</strong>. The old velocity still counts: <code>v</code> = ${T('files.optim.runs.5.momentum')} · (${T(step(5, 0, 'gw'))}) + (${T(step(5, 1, 'gw'))}) = ${derived(v, 'worked out from the recorded gradients')}, so the step is ${T('files.optim.runs.5.lr')} × ${fmt(-v)} = ${derived(-lr * v, 'worked out from the recorded gradients')}, longer than the gradient alone gives. That extra push is also why the loss can go <em>up</em> afterwards (${T(step(4, 2, 'loss'))} → ${T(step(4, 3, 'loss'))} for μ = ${T('files.optim.runs.4.momentum')}): it is expected, not a bug.</p>`; },
    },

    // ---- wrap up ---------------------------------------------------------------------------------------------------------------------------
    {
      kind: 'check', id: 'c-params',
      prompt: () => `<p>After a full iteration (<code>backward()</code>, <code>step()</code>) which of <code>x</code>, <code>t</code>, <code>W</code>, <code>b</code>, <code>y</code> and <code>L</code> have a different <code>.data</code>? Which have a <code>.grad</code>?</p>`,
      reveal: () => `<p>Only <code>W</code> and <code>b</code> change: they are the parameters handed to the optimizer, and <code>step()</code> touches nothing else. <code>x</code> and <code>t</code> are data (<code>requires_grad</code> is false: no <code>.grad</code>, never touched). <code>y</code> and <code>L</code> are recomputed from scratch by the next forward pass. For <code>.grad</code>: <code>W</code> and <code>b</code> have one, and so do the intermediate results <code>Wt</code>, <code>xW</code> and <code>y</code> (they are on the path from the loss to the parameters; the graph above shows them), while <code>L</code>, the root, stores none. The gradients of <code>W</code> and <code>b</code> stay put after <code>step()</code> until <code>zero_grad()</code> clears them; the intermediate ones belong to that forward pass, and the next forward pass builds a new graph.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => `<section aria-labelledby="ideas" class="ideas">
    <h2 id="ideas">Five things to remember</h2>
    <div class="cards" id="concepts">${CONCEPTS(T).map((c) => `<div class="card"><h3>${c.h}</h3><p>${c.p}</p></div>`).join('')}</div>
  </section>`,
    },
  ],
  footer: () => `<p>Every number in the text, the graph and the “Precomputed from TorC” tables was printed by the real TorC library (<code>tools/trace/trace_optim.cpp</code>, release v0.1.0). The sandbox, landscape, learning-rate, loop and momentum widgets are recomputed in your browser with a JavaScript port of the same formulas, which is tested against TorC’s recorded runs. The critical learning rate and the stability factors are derived mathematics (the Hessian of this loss), not output of the library. The code excerpts are copied from the library source. Source: <a href="${REPO_URL}">github.com/IvorLipic/TorC</a>.</p>`,
};
