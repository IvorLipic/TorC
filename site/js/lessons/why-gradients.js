// Lesson "why-gradients": one knob, one score. Slope as loss-per-unit-of-w (secants -> tangent), a step against the slope,
// why a step can be too big, and the chain rule as multiplied local slopes (the graph lab on scene `why`).
// Every number in the prose comes from data/why-gradients.json through T(); the widgets recompute live in the browser with the
// same engine code and say so. Storyboard: docs/content-plan.md 3.1, notes: docs/lessons/why-gradients.md.
import { mountGraphLab } from '../widgets/graph-lab.js';
import { mountKnob, mountSlope, mountStepper, mountStepSize } from '../widgets/landscape1d.js';
import { mountGears } from '../widgets/why-gears.js';
import { whyLayout } from './why-gradients.scenes.js';
import { math } from '../core/katex.js';
import { fmt } from '../ops/texutil.js';
import { makeModel } from '../widgets/landscape-model.js';

const F = 'files.why-gradients';
const eq = (latex) => `<div class="wg-eq">${math(latex)}</div>`;
// the bowl's constants (curvature 2x^2 and minimum t/x) come from the scene, not from the keyboard
// In prose they are wrapped in <span class="dc" data-derived=…>; the lesson test checks each against the scene.
const consts = (D) => {
  const m = makeModel(D.scenes.why);
  return { k: fmt(m.curvature), w: fmt(m.wMin), kS: `<span class="dc" data-derived="curvature">${fmt(m.curvature)}</span>` };
};
const traj = (i, rest) => `${F}.trajectories.${i}.${rest}`;
// numbers that are plain arithmetic on printed ones (not printed by TorC) get a dotted underline: see DERIVED in the test
export const DERIVED = ['.factor', 'critical_lr', 'w_min', 'tangent_L_next', 'predicted_dL', 'local_slope_product', 'second_order_term'];
const dv = (T, path) => `<span class="dv" title="Derived from numbers TorC printed, by plain arithmetic">${T(path)}</span>`;

const STYLE = `<style>
.wg-eq{overflow-x:auto;margin:.3rem 0}
.wg-eq .katex-display{margin:.5em 0}
.wg-lesson h2{margin-top:2.4rem}
.wg-lesson p{max-width:46rem}
.wg-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.7rem 1rem;overflow-x:auto;max-width:46rem;line-height:1.5}
.wg-code code{background:none;padding:0}
.wg-list{max-width:46rem;padding-left:1.2rem}
.wg-list li{margin:.3rem 0}
.wg-rates{border-collapse:collapse;font-size:.92rem;margin:.6rem 0;width:100%;max-width:46rem}
.wg-rates caption{text-align:left;color:var(--muted);padding-bottom:.3rem}
.wg-rates th,.wg-rates td{border-top:1px solid var(--border);padding:.3rem .6rem;text-align:right}
.wg-rates th:first-child,.wg-rates td:last-child,.wg-rates th:last-child{text-align:left}
.wg-rates thead th{color:var(--muted);font-weight:600}
.wg-rates-wrap{overflow-x:auto}
#chain-lab #scene-tabs{display:none}
#chain-lab .lab-top{justify-content:flex-end}
.wg-next{max-width:46rem;margin-top:1.4rem}
.dv,.dc{border-bottom:1px dotted var(--muted)}
.wg-qt{font-weight:600;margin:.1rem 0 .3rem}
.wg-lesson .card h4{font-size:1.05rem;color:var(--fg);margin:0 0 .3rem;font-weight:600}
/* phone: the learning-rate table becomes one small card per learning rate (no sideways scrolling) */
@media(max-width:640px){
  .wg-rates,.wg-rates tbody,.wg-rates tr,.wg-rates td{display:block;width:100%}
  .wg-rates caption{display:block;width:100%}
  .wg-rates thead{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
  .wg-rates tr{border:1px solid var(--border);border-radius:10px;margin:.5rem 0;padding:.2rem .6rem;background:var(--panel)}
  .wg-rates td{border:0;display:flex;justify-content:space-between;gap:1rem;text-align:right!important;padding:.15rem 0}
  .wg-rates td::before{content:attr(data-label);color:var(--muted);text-align:left;flex:none}
}
</style>`;

// one row of the learning-rate summary table; every number is bound to the trace
const rateRow = (i, what, k) => (D, T) => `<tr><td data-label="learning rate">${T(traj(i, 'lr'))}</td><td data-label="factor 1 − ${k}·lr">${dv(T, traj(i, 'factor'))}</td><td data-label="w after one step">${T(traj(i, 'states.1.w'))}</td><td data-label="L after four steps">${T(traj(i, 'states.4.L'))}</td><td data-label="what happens">${what}</td></tr>`;

export default {
  id: 'why-gradients',
  title: 'Why following gradients works',
  trace: ['why-gradients.json'],
  sections: [
    // ------------------------------------------------------------------------------------------------ hero + 1
    {
      kind: 'text',
      html: (D) => `${STYLE}<section class="hero wg-lesson">
    <h1>Why following gradients works</h1>
    <p class="lead">Training a model means turning knobs to make a score smaller. Here you turn one knob, measure a slope, step against it, take a step that is too big, and then see where <a href="https://github.com/IvorLipic/TorC">TorC</a> gets that slope from. Everything is small enough to check by hand.</p>
    <p class="hint">Numbers on this page were printed by a real run of TorC <code>${D.files['why-gradients'].library.ref}</code>, or are recomputed live in your browser by a port of its rules (each interactive part says which). A number with a <span class="dv">dotted underline</span> is not printed by TorC but derived from printed numbers by plain arithmetic.</p>
  </section>`,
    },
    {
      kind: 'text',
      html: (D, T, fm = makeModel(D.scenes.why)) => `<section class="wg-lesson" aria-labelledby="wg-knob">
    <h2 id="wg-knob">1 · A knob and a score</h2>
    <p>Start with the smallest model there is: <strong>one knob</strong>, <code>w</code>, and <strong>one data point</strong>. The input is <code>x</code> = ${T('why.x.value')} and the target is <code>t</code> = ${T('why.t.value')}. The model multiplies the input by the knob to get a prediction <code>p</code>; the <em>loss</em> <code>L</code> is the squared error, a score we want small:</p>
    ${eq('p = w\\cdot x \\qquad e = p - t \\qquad L = e\\cdot e')}
    <p>In TorC that is three operations, each remembering how it was made so that <code>backward()</code> can run them in reverse later. Only <code>w</code> is created with <code>requires_grad = true</code>; <code>x</code> and <code>t</code> are plain data:</p>
    <pre class="wg-code"><code>Variable w(Tensor({${T('why.w.value')}.0f}, {1}), true);    // the knob (tracked)
Variable x(Tensor({${T('why.x.value')}.0f}, {1}), false);   // data: constant
Variable t(Tensor({${T('why.t.value')}.0f}, {1}), false);   // data: constant
Variable p = mul(w, x);     // prediction
Variable e = sub(p, t);     // error
Variable L = mul(e, e);     // the score: a single number
L.backward();               // afterwards w.grad() holds the slope</code></pre>
    <p>Put the three lines together and the score is a function of the knob alone: ${math(`L(w) = (${fm.x}w-${fm.t})^2 = ${fm.x * fm.x}(w-${fm.wMin})^2`, false)}. It is a bowl with its bottom at <code>w</code> = ${dv(T, `${F}.setup.w_min`)}.</p>
    <p>Turning the knob changes the score. In this toy world you could simply draw the whole curve. In a real network there are thousands of knobs and you never see the curve; all you can get is <em>local</em> information about where you stand. The rest of this page is about the most useful piece of local information there is.</p>
  </section>`,
    },
    {
      kind: 'predict', id: 'direction',
      prompt: (D, T) => `<p class="wg-qt">Which way is downhill?</p><p>The knob starts at <code>w</code> = ${T('why.w.value')}, so the prediction is <code>p</code> = ${T('why.p.value')} while the target is <code>t</code> = ${T('why.t.value')}, and the error is <code>e</code> = ${T('why.e.value')}. You cannot see the curve. Do you turn the knob <strong>left</strong> (smaller w) or <strong>right</strong> (bigger w) to shrink the score? Give a reason in terms of <code>p</code> and <code>t</code>, and say at which <code>w</code> you would expect the error to vanish.</p>`,
      reveal: (D, T) => `<p><strong>Right.</strong> The prediction is too small (<code>e</code> is negative) and <code>p = w·x</code> with a positive <code>x</code>, so a bigger <code>w</code> makes <code>p</code> bigger and closer to <code>t</code>. The error vanishes when <code>w·x = t</code>, that is at <code>w</code> = ${dv(T, `${F}.setup.w_min`)}. Check the first step: at <code>w</code> = ${T(`${F}.probes.2.w`)} the score is already ${T(`${F}.probes.2.L`)}, down from ${T('why.L.value')}.</p><p>Here we could reason it out because we know the whole model. With thousands of knobs we cannot, so the next section shows a measurement that tells us the direction at the spot where we stand.</p>`,
    },
    {
      kind: 'text',
      html: () => `<section class="wg-lesson" aria-label="Try it"><p><strong>Test your guess.</strong> Drag the knob (or focus it and use the arrow keys) and look at the score. Then see how low you can get it.</p></section>`,
    },
    {
      kind: 'widget', id: 'knob', className: 'widget wg-widget', label: 'Interactive: turn the knob and watch the score',
      mount: (host, D) => mountKnob(host, D),
    },
    // ------------------------------------------------------------------------------------------------ 2 slope
    {
      kind: 'text',
      html: (D, T) => `<section class="wg-lesson" aria-labelledby="wg-slope">
    <h2 id="wg-slope">2 · Slope: how fast the score changes</h2>
    <p>Trying both directions works for one knob. A better question to ask at the spot where you stand: <strong>if I nudge the knob a little, how much does the score change per unit of nudge?</strong></p>
    <p>Nudge <code>w</code> by <code>h</code> and divide the change in score by <code>h</code>. That ratio is the slope of the <em>secant</em>, the straight line through two points of the curve. Below, you stand at <code>w</code> = ${T('why.w.value')}, where the score is ${T(`${F}.secants.0.L0`)}. With <code>h</code> = ${T(`${F}.secants.0.h`)} the score drops to ${T(`${F}.secants.0.L_h`)}, so the secant slope is ${T(`${F}.secants.0.slope`)}. Now slide <code>h</code> toward zero. (For one knob this slope is the <em>derivative</em>; TorC and machine learning call it the <strong>gradient</strong>, which is why the result lands in <code>w.grad()</code>.)</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'slope', className: 'widget wg-widget', label: 'Interactive: secants turning into a tangent',
      mount: (host, D) => mountSlope(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => `<section class="wg-lesson" aria-labelledby="wg-slope2">
    <h3 id="wg-slope2">What you should have seen</h3>
    <p>As <code>h</code> shrinks the secant slopes go ${T(`${F}.secants.0.slope`)}, ${T(`${F}.secants.1.slope`)}, ${T(`${F}.secants.2.slope`)}, ${T(`${F}.secants.3.slope`)} and settle on <strong>${T('why.w.grad')}</strong>. The secant has become the <em>tangent</em>, the straight line that touches the curve and shares its steepness at that point. That limit is the <strong>derivative</strong>, and it is exactly what TorC's <code>backward()</code> leaves in <code>w.grad()</code>: ${T('why.w.grad')}, without trying a single <code>h</code>.</p>
    <ul class="wg-list">
      <li><strong>Units.</strong> ${T('why.w.grad')} means “score units per unit of <code>w</code>”. Raise <code>w</code> by a tiny amount ε and the score changes by about ${T('why.w.grad')}·ε. (A big nudge such as <code>h</code> = ${T(`${F}.secants.0.h`)} gives a different ratio, ${T(`${F}.secants.0.slope`)}: the curve bends.)</li>
      <li><strong>Sign.</strong> Negative: raising <code>w</code> lowers the score. Positive: raising it makes things worse. Zero: flat.</li>
      <li><strong>Size.</strong> The magnitude says how <em>steep</em> it is here. It does <em>not</em> say how far away the bottom is: the bottom is ${dv(T, `${F}.setup.w_min`)} units away, the slope is ${T('why.w.grad')}.</li>
    </ul>
    <p>Move the knob in the picture above to other places and watch the tangent tilt. The slope depends on where you stand, and the tangent is only a good picture of the curve <em>near</em> the dot.</p>
  </section>`,
    },
    // ------------------------------------------------------------------------------------------------ 3 step
    {
      kind: 'text',
      html: (D, T) => `<section class="wg-lesson" aria-labelledby="wg-step">
    <h2 id="wg-step">3 · Step against the slope</h2>
    <p>The slope says which way is uphill, so go the other way. The rule is one line:</p>
    ${eq('w \\leftarrow w - \\mathit{lr}\\cdot g')}
    <p>Here <code>g</code> is the slope (<code>w.grad()</code>) and <code>lr</code>, the <em>learning rate</em>, turns “score per unit of <code>w</code>” into “how far to move <code>w</code>”. With <code>lr</code> = ${T(traj(0, 'lr'))} and <code>g</code> = ${T('why.w.grad')}, the new weight is ${T(traj(0, 'states.1.w'))}. Press <strong>Step</strong> and follow the dot.</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'step', className: 'widget wg-widget', label: 'Interactive: take steps against the slope',
      mount: (host, D) => mountStepper(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => `<section class="wg-lesson" aria-labelledby="wg-step2">
    <h3 id="wg-step2">Three things worth noticing</h3>
    <p><strong>The steps shrink</strong> (at <code>lr</code> = ${T(traj(0, 'lr'))}). The score falls every time, and each step is shorter than the one before, because the slope itself gets smaller as you near the bottom: ${T(traj(0, 'states.0.grad'))} at the start, ${T(traj(0, 'states.1.grad'))} after one step.</p>
    <p><strong>The tangent over-promises.</strong> At the start the tangent said that moving to <code>w</code> = ${T(traj(0, 'states.1.w'))} would bring the score to ${dv(T, traj(0, 'states.0.tangent_L_next'))}. The curve, which bends upward, gave ${T(traj(0, 'states.1.L'))}: better than ${T('why.L.value')}, worse than promised. The slope is a <em>local</em> promise: reliable for small moves, optimistic for big ones.</p>
    <p><strong>The slope is a direction, not an address.</strong> From <code>w</code> = ${T(`${F}.probes.5.w`)} the slope is ${T(`${F}.probes.5.grad`)}, positive, so the same rule moves <code>w</code> <em>down</em>. The slope never says where the bottom is, only which side is uphill.</p>
  </section>`,
    },
    // ------------------------------------------------------------------------------------------------ 4 step size
    {
      kind: 'text',
      html: () => `<section class="wg-lesson" aria-labelledby="wg-size">
    <h2 id="wg-size">4 · How big a step?</h2>
    <p>The learning rate is yours to choose, and it matters more than it looks. Think about it before touching anything.</p>
  </section>`,
    },
    {
      kind: 'predict', id: 'big-step',
      prompt: (D, T) => `<p class="wg-qt">What does a bigger step do?</p><p>Same start, <code>w</code> = ${T('why.w.value')}, slope ${T('why.w.grad')}, but <code>lr</code> = ${T(traj(3, 'lr'))} instead of ${T(traj(0, 'lr'))}. Write down where <code>w</code> is after one step and after two, and what happens to the score.</p>`,
      reveal: (D, T) => `<p>It never gets there. The first step lands on <code>w</code> = ${T(traj(3, 'states.1.w'))}, the far side of the bowl, at exactly the same score (${T(traj(3, 'states.1.L'))}). The slope there is ${T(traj(3, 'states.1.grad'))}, so the next step jumps back to <code>w</code> = ${T(traj(3, 'states.2.w'))}. It bounces for ever, and the score stays at ${T(traj(3, 'states.8.L'))}. Check it with the <code>lr</code> = ${T(traj(3, 'lr'))} button below.</p>`,
    },
    {
      kind: 'widget', id: 'step-size', className: 'widget wg-widget', label: 'Interactive: learning-rate presets and eight steps',
      mount: (host, D) => mountStepSize(host, D),
    },
    {
      kind: 'text',
      html: (D, T, { k, w: w3, kS } = consts(D), fmx2 = fmt(makeModel(D.scenes.why).curvature / 2)) => `<section class="wg-lesson" aria-labelledby="wg-size2">
    <h3 id="wg-size2">Why that happens</h3>
    <p>In this bowl the slope is ${math(`${k}(w-${w3})`, false)} (the derivative of ${math(`L(w)=${fmx2}(w-${w3})^2`, false)} by the power rule: ${math(`2\\cdot ${fmx2}(w-${w3})`, false)}): it grows by ${kS} for every unit you move away from the bottom. A step of <code>lr</code> times the slope therefore covers a fixed fraction, ${kS}·<code>lr</code>, of the distance to <code>w</code> = ${dv(T, `${F}.setup.w_min`)}. After one step the distance is multiplied by ${math(`1-${k}\\,\\mathit{lr}`, false)}:</p>
    ${eq(`w_{k+1}-${w3} = (1-${k}\\,\\mathit{lr})\\,(w_k-${w3})`)}
    <p>You can see that factor in the table's last column. It decides everything:</p>
    <div class="wg-rates-wrap"><table class="wg-rates">
      <caption>The five learning rates TorC was run with, all starting at <code>w</code> = ${T('why.w.value')} (what each does is in the last column)</caption>
      <thead><tr><th scope="col">lr</th><th scope="col">factor ${math(`1-${k}\\,\\mathit{lr}`, false)}</th><th scope="col">w after one step</th><th scope="col">L after four steps</th><th scope="col">What happens</th></tr></thead>
      <tbody>
        ${rateRow(0, 'creeps in from one side', k)(D, T)}
        ${rateRow(1, 'faster, still from one side', k)(D, T)}
        ${rateRow(2, 'lands on the minimum in one step', k)(D, T)}
        ${rateRow(3, 'bounces between two points for ever', k)(D, T)}
        ${rateRow(4, 'overshoots more every step: diverges', k)(D, T)}
      </tbody></table></div>
    <ul class="wg-list">
      <li><strong>Below ${dv(T, `${F}.setup.critical_lr`)}</strong>, |factor| is less than one and every step lands closer to the bottom. A negative factor means the dot jumps over the bottom each time but still ends up closer.</li>
      <li><strong>Exactly ${dv(T, `${F}.setup.critical_lr`)}</strong> the factor is ${dv(T, traj(3, 'factor'))}: the distance never changes.</li>
      <li><strong>Above it</strong> the distance grows. With <code>lr</code> = ${T(traj(4, 'lr'))} the score goes ${T(traj(4, 'states.0.L'))}, ${T(traj(4, 'states.1.L'))}, ${T(traj(4, 'states.2.L'))}, ${T(traj(4, 'states.3.L'))}, …</li>
      <li><strong>Neither “small is safe” nor “big is fast” is true.</strong> The best <code>lr</code> here, ${T(traj(2, 'lr'))}, is one over ${kS}, the rate at which the slope grows (the bowl's <em>curvature</em>), and it only works because this bowl is a perfect parabola. Real losses are not, so in practice <code>lr</code> is a setting you try and watch. The <a href="../sgd-step/">SGD step</a> lesson does this with TorC's real <code>optim::SGD</code>.</li>
    </ul>
  </section>`,
    },
    // ------------------------------------------------------------------------------------------------ 5 chain rule
    {
      kind: 'text',
      html: (D, T) => `<section class="wg-lesson" aria-labelledby="wg-chain">
    <h2 id="wg-chain">5 · The chain rule as gears</h2>
    <p>Where does the slope ${T('why.w.grad')} come from? We got it for free from <code>backward()</code>; here is what it does. The knob does not touch the loss directly. It turns <code>p</code>, which turns <code>e</code>, which turns <code>L</code>, like a train of gears. Each hand-over has a <strong>local slope</strong>: how much the next stage moves per unit that the previous one moves. They are easy to find because each is a single operation:</p>
    <ul class="wg-list">
      <li><code>p = w·x</code> grows by <code>x</code> = ${T('why.x.value')} for each unit of <code>w</code>: slope ${T(`${F}.nudge.stages.0.local_slope`)}.</li>
      <li><code>e = p − t</code> moves one for one with <code>p</code>: slope ${T(`${F}.nudge.stages.1.local_slope`)}.</li>
      <li><code>L = e·e</code> has slope <code>2e</code> (the power rule again); with <code>e</code> = ${T('why.e.value')} that is ${T(`${F}.nudge.stages.2.local_slope`)}.</li>
    </ul>
    <p>Nudge the knob and watch the change travel through all three.</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'gears', className: 'widget wg-widget', label: 'Interactive: the chain rule as multiplied local slopes',
      mount: (host, D) => mountGears(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => `<section class="wg-lesson" aria-labelledby="wg-chain2">
    <p>The three local slopes are ${T(`${F}.nudge.stages.0.local_slope`)}, ${T(`${F}.nudge.stages.1.local_slope`)} and ${T(`${F}.nudge.stages.2.local_slope`)}, and their product is ${dv(T, `${F}.nudge.local_slope_product`)}, which is the slope ${T('why.w.grad')} that TorC reported. That is the <strong>chain rule</strong>: the slope of the whole chain is the product of the local slopes. For a nudge of ${T(`${F}.nudge.delta`)} the loss changes by ${T(`${F}.nudge.actual_dL`)}, and the product predicted ${dv(T, `${F}.nudge.predicted_dL`)}. The tiny difference is the curve bending: the tangent is only a local promise, as before.</p>
    <p>Now let the machine do it. Step through the graph below: <strong>Forward</strong> computes <code>p</code>, <code>e</code>, <code>L</code>; <strong>backward</strong> starts at <code>L</code> and multiplies one local slope per step until it reaches <code>w</code>. The math window under the graph opens by itself at each step; compare its numbers with the table above. In it, a bar such as <code>w̄</code> means the gradient of <code>L</code> with respect to that node, and <code>⊙</code> is element-wise multiplication, which for single numbers is plain multiplication. At <code>L</code> watch the two arrows from <code>e</code>: <code>e</code> is used twice in <code>e·e</code>, and TorC <em>adds</em> what comes back along both.</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'chain-lab', className: 'lab', label: 'Interactive lab: forward and backward through the chain',
      mount: (host, D, ctx) => mountGraphLab(host, { scenes: { why: D.scenes.why }, layouts: { why: whyLayout(ctx.makeT()) }, defaultScene: 'why' }),
    },
    // ------------------------------------------------------------------------------------------------ 6 recap
    {
      kind: 'text',
      html: (D, T) => `<section class="wg-lesson" aria-labelledby="wg-recap">
    <h2 id="wg-recap">6 · Recap</h2>
    <ul class="wg-list">
      <li>The <strong>gradient</strong> is a slope: score change per unit change of a weight (here ${T('why.w.grad')} at <code>w</code> = ${T('why.w.value')}).</li>
      <li>You move <strong>against</strong> it, <code>w ← w − lr · g</code>, and for a small enough step the score goes down.</li>
      <li>The step size must fit the curve: too small is slow, too big overshoots, and past ${dv(T, `${F}.setup.critical_lr`)} here it diverges.</li>
      <li><code>backward()</code> gets the slope by <strong>multiplying local slopes</strong> along the chain (and adding when a value is used more than once).</li>
    </ul>
    <h3>Four mix-ups to avoid</h3>
    <div class="cards">
      <div class="card"><h4>“The gradient says how far to go.”</h4><p>It is a rate. At <code>w</code> = ${T('why.w.value')} it is ${T('why.w.grad')} but the bottom is ${dv(T, `${F}.setup.w_min`)} away. The learning rate turns a rate into a distance.</p></div>
      <div class="card"><h4>“Smaller learning rates are always safer; bigger ones faster.”</h4><p>${T(traj(2, 'lr'))} reaches the bottom in one step, ${T(traj(3, 'lr'))} never settles, and ${T(traj(0, 'lr'))} is still at ${T(traj(0, 'states.8.L'))} after eight steps.</p></div>
      <div class="card"><h4>“The tangent predicts the new score.”</h4><p>It is a local promise. It said ${dv(T, traj(0, 'states.0.tangent_L_next'))}; the curve gave ${T(traj(0, 'states.1.L'))}.</p></div>
      <div class="card"><h4>“The gradient points at the minimum.”</h4><p>In one dimension it is only a sign: from <code>w</code> = ${T(`${F}.probes.5.w`)} it is ${T(`${F}.probes.5.grad`)}, from <code>w</code> = ${T('why.w.value')} it is ${T('why.w.grad')}. The step goes against it.</p></div>
    </div>
  </section>`,
    },
    {
      kind: 'check', id: 'sign',
      prompt: (D, T) => `<p class="wg-qt">Sign and direction</p><p>The knob is at <code>w</code> = ${T(`${F}.probes.5.w`)}. Without looking: is the slope positive or negative, and which way does the rule move <code>w</code>?</p>`,
      reveal: (D, T) => `<p>The score there is ${T(`${F}.probes.5.L`)} and the slope is ${T(`${F}.probes.5.grad`)}: <strong>positive</strong>, so raising <code>w</code> would raise the score. <code>w ← w − lr · g</code> subtracts a positive number, so <code>w</code> moves <strong>down</strong>, toward ${dv(T, `${F}.setup.w_min`)}.</p>`,
    },
    {
      kind: 'check', id: 'one-step',
      prompt: (D, T, { k } = consts(D)) => `<p class="wg-qt">The best step size</p><p>In the bowl the best learning rate is ${T(traj(2, 'lr'))}. Starting from <code>w</code> = ${T('why.w.value')}, where is <code>w</code> after one step, and what is the factor ${math(`1-${k}\\,\\mathit{lr}`, false)}? Why would this exact trick not carry over to a real network?</p>`,
      reveal: (D, T) => `<p>After one step <code>w</code> = ${T(traj(2, 'states.1.w'))}, the minimum, because the factor is ${dv(T, traj(2, 'factor'))} and the whole distance is covered. It works only because the loss is a perfect parabola whose slope-growth rate we know. A real loss bends differently in every direction and at every place, so you cannot read the right <code>lr</code> off a formula.</p>`,
    },
    {
      kind: 'check', id: 'chain',
      prompt: (D, T) => `<p class="wg-qt">Gears and contributions</p><p>Which three numbers multiply to give the slope ${T('why.w.grad')}, and why does <code>e</code> receive <em>two</em> contributions in the backward pass?</p>`,
      reveal: (D, T) => `<p>The local slopes ${T(`${F}.nudge.stages.0.local_slope`)} (<code>p = w·x</code>, so the slope is <code>x</code>), ${T(`${F}.nudge.stages.1.local_slope`)} (<code>e = p − t</code>) and ${T(`${F}.nudge.stages.2.local_slope`)} (<code>L = e·e</code>, so the slope is <code>2e</code>). Their product is ${dv(T, `${F}.nudge.local_slope_product`)}. In <code>e·e</code> the variable <code>e</code> is both inputs of the multiplication, so each input sends back the other one, <code>e</code> = ${T('why.e.value')}, and the two contributions add up to ${T('why.e.grad')}.</p>`,
    },
    {
      kind: 'text',
      html: () => `<section class="wg-lesson wg-next" aria-labelledby="wg-next">
    <h2 id="wg-next">What next</h2>
    <p>You have seen the whole idea with one weight. The <a href="../backprop-lab/">Backprop lab</a> does the same chain of local slopes for graphs where values are reused and for a small network with several weights, and <a href="../sgd-step/">the SGD step</a> turns <code>w ← w − lr · g</code> into a real training step with TorC's optimizer.</p>
  </section>`,
    },
  ],
  footer: (D) => `<p>Numbers outside the interactive parts were printed by the real TorC library, <code>${D.files['why-gradients'].library.ref}</code> (<code>tools/trace/trace_why.cpp</code>), or, with a dotted underline, derived from those numbers by plain arithmetic (the factor, the critical learning rate, the minimum, the tangent's promise). Inside the interactive parts, values are recomputed in your browser by a small JavaScript port of the same rules, tested against that output; each part says when it is showing a point TorC recorded. Source: <a href="https://github.com/IvorLipic/TorC">github.com/IvorLipic/TorC</a>.</p>`,
};
