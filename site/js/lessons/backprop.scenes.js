// backprop.scenes.js — the graph-lab layout and teaching text of the scene `reuse` (y = x·x + x, data/autograd_trace.json) as the
// lesson backprop embeds it. The Backprop lab keeps its own layout (backprop-lab.scenes.js). Numbers are never typed here:
// reuseLayout(T) binds each one with T(), and site/test/backprop.test.mjs renders the strings against the real file.
// `narrow` is the phone drawing of the same graph (brief D16): x at the bottom left, the multiplication above, y at the bottom right.
// The intro also explains the notation that the lab's panel uses (x̄, c₁ to c₃, seed), which the lesson text does not.
const U = 'files.autograd_trace.reuse_uses';

export function reuseLayout(T) {
  const n = (path) => T(path, 'setup');
  return {
    tab: 'Reuse',
    view: [900, 210], minWidth: 600,
    pos: { x: [150, 130], m: [450, 55], y: [750, 130] },
    narrow: { view: [340, 300], minWidth: 300, pos: { x: [62, 214], m: [170, 70], y: [278, 214] } },
    goal: 'What happens when one value is used more than once?',
    intro: `The value <code>x</code> is ${n('reuse.x.value')}. The multiplication uses <code>x</code> twice, and the addition uses it once more. In this panel, x̄ is the gradient of <code>x</code>. The labels c₁, c₂ and c₃ are the gradients that come back from its three uses. The seed is the gradient one at the output.`,
    forwardHint: 'The forward pass computes <code>m</code> and then <code>y</code>.',
    backwardHint: 'The backward pass starts at <code>y</code> with a gradient of one and goes from right to left.',
    takeaway: `Each use of <code>x</code> sends back its own gradient. The two uses in the multiplication send ${n(`${U}.mul_first`)} each, and the addition sends ${n(`${U}.add`)}. TorC adds them, so the gradient of <code>x</code> is ${n('reuse.x.grad')}.`,
  };
}
