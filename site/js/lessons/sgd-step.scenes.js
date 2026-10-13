// sgd-step.scenes.js — layout (where each node of the scene "sgd" is drawn) and the teaching text of its graph lab.
// Numbers are never typed here: makeLayouts(T) puts values from the trace into the text through T() (data/optim.json, a real TorC run);
// site/test/sgd-step.test.mjs renders this text and checks every bound number.

export function makeLayouts(T) {
  return {
    sgd: {
      tab: 'Fit a line: y = wx + b',
      view: [940, 330], minWidth: 760,
      above: ['y', 'L'],
      pos: { W: [90, 70], Wt: [260, 70], x: [90, 235], xW: [440, 150], b: [630, 265], y: [630, 150], t: [810, 265], L: [810, 150] },
      goal: 'What does one training step start with?',
      intro: 'A line <code>y = w·x + b</code> is fitted to three points. <code>x</code> and <code>t</code> are <strong>data</strong> (dashed boxes: no gradient is wanted for them). <code>W</code> and <code>b</code> are the <strong>parameters</strong>, the numbers we want to improve. The graph computes the prediction <code>y</code> and the mean squared error <code>L</code>.',
      forwardHint: 'Forward: compute the prediction and the loss.',
      backwardHint: 'Backward: walk back from the loss and leave a gradient on <code>W</code> and <code>b</code>.',
      takeaway: `<span>Backward left <code>W.grad</code> = ${T('sgd.W.grad')} and <code>b.grad</code> = ${T('sgd.b.grad')}. Now look at <code>W</code> and <code>b</code> themselves: still ${T('sgd.W.value')} and ${T('sgd.b.value')}. Computing a gradient changes nothing by itself; moving the parameters is the optimizer’s job.</span>`,
    },
  };
}

export const CONCEPTS = (T) => [
  { h: 'The optimizer never computes gradients', p: 'It reads <code>.grad</code>, which <code>backward()</code> filled in. Without a <code>backward()</code> first there is nothing to read, and <code>step()</code> quietly does nothing.' },
  { h: 'step() reads .grad and rewrites .data', p: `That is the whole job: <code>p = p − lr·grad</code>. Only parameters change (<code>W</code> and <code>b</code>); the data <code>x</code> and <code>t</code> never do. <code>step()</code> does not clear <code>.grad</code>.` },
  { h: 'zero_grad() is not optional', p: `<code>.grad</code> is added to, never overwritten. Forget to clear it and <code>step()</code> reads the sum of every past gradient: the loss bounces (${T('files.optim.no_zero_grad.training_loop.iterations.2.loss')}, ${T('files.optim.no_zero_grad.training_loop.iterations.3.loss')}, ${T('files.optim.no_zero_grad.training_loop.iterations.4.loss')}…) instead of falling.` },
  { h: 'The learning rate is a size, not a speed', p: `A bigger step is not a faster arrival. Past a limit that depends on the data (here about ${T('files.optim.hessian.critical_lr')}) every update makes things worse: lr ${T('files.optim.runs.3.lr')} sends the loss to ${T('files.optim.runs.3.final.loss')}.` },
  { h: 'Momentum can overshoot', p: `With momentum the loss need not fall at every update: with μ = ${T('files.optim.runs.4.momentum')} it goes ${T('files.optim.runs.4.steps.2.loss')} then ${T('files.optim.runs.4.steps.3.loss')}. That is the price of the extra speed.` },
];
