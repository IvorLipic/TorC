// backprop-lab.scenes.js — layout (where each named node is drawn) and the teaching text of each scene of the Backprop lab.
// Numbers never live here: values, edges and gradients come from data/autograd_trace.json (real TorC run)
// and from engine.js (live recomputation after the learner edits inputs).
export const SCENES = {
  reuse: {
    tab: '1 · Reuse',
    view: [900, 210], minWidth: 600,
    pos: { x: [150, 130], m: [450, 55], y: [750, 130] },
    goal: 'What if the same variable is used more than once?',
    intro: 'Here <code>x</code> is used <strong>twice</strong> in the multiplication and <strong>once more</strong> in the addition. Watch how its gradient is the <em>sum</em> of what each use sends back.',
    forwardHint: 'Forward: compute the value of every node, left to right.',
    backwardHint: 'Backward: start at the output with gradient 1 and walk right to left, applying each node’s rule.',
    takeaway: 'Each use of <code>x</code> sends back its own contribution (3, 3 and 1 with the preset numbers). Backprop <strong>adds</strong> them — that is the sum rule of calculus, <code>d(x²+x)/dx = 2x+1</code>.',
  },
  chain: {
    tab: '2 · Chain rule',
    view: [1000, 290], minWidth: 760,
    pos: { a: [100, 70], b: [100, 215], ab: [330, 150], c: [330, 35], z: [560, 95], r: [760, 95], f: [930, 95] },
    goal: 'How does a gradient travel through several operations?',
    intro: 'A short chain: multiply, add, ReLU, sum. At every node the gradient arriving from the right is <strong>multiplied by that node’s local slope</strong> and passed to the left. ReLU blocks the gradient wherever its input was negative.',
    forwardHint: 'Forward: each node computes its output from its inputs.',
    backwardHint: 'Backward: at each node, multiply the incoming gradient by the local slope.',
    takeaway: 'This is the chain rule: the gradient of the output with respect to an input is the product of the local slopes along the way. Try editing <code>c</code> so that every element of <code>z</code> is positive and watch ReLU stop blocking.',
  },
  mlp: {
    tab: '3 · A tiny network',
    view: [1280, 300], minWidth: 1000,
    above: ['x', 'W1', 'b1', 'W2', 'b2', 't'],
    pos: { x: [70, 85], W1: [230, 85], b1: [420, 85], W2: [590, 85], b2: [740, 85], t: [900, 85],
           xW1: [110, 225], h: [300, 225], a: [490, 225], aW2: [655, 225], o: [800, 225], d: [960, 225], q: [1100, 225], L: [1210, 225] },
    goal: 'What does one training step of a small network compute?',
    intro: 'A one-hidden-layer network on a <strong>batch of two examples</strong> (the two rows of <code>x</code>), with a mean-squared-error loss. The gradients of the <strong>weights and biases</strong> are what an optimizer uses to improve them.',
    forwardHint: 'Forward: the prediction and the loss.',
    backwardHint: 'Backward: from the loss back to every weight and bias.',
    takeaway: 'Matrix products and broadcasting follow the same two rules as everything else: multiply by the local slope, and add up what arrives. The bias <code>b1</code> was copied to both rows of the batch, so its gradient is the <strong>sum over the rows</strong>. ReLU blocked the gradient wherever <code>h</code> was negative — try changing <code>b1</code> or <code>x</code> and watch which entries switch on and off.',
  },
};
