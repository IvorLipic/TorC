// scenes.js — layout (where each named node is drawn) and the teaching text of each scene.
// Numbers never live here: values, edges and gradients come from data/autograd_trace.json
// (real TorC run) and from engine.js (live recomputation after the learner edits inputs).
export const SCENES = {
  reuse: {
    tab: '1 · Reuse',
    view: [900, 300], minWidth: 640,
    pos: { x: [150, 160], m: [450, 80], y: [750, 160] },
    goal: 'What if the same variable is used more than once?',
    intro: 'Here <code>x</code> is used <strong>twice</strong> in the multiplication and <strong>once more</strong> in the addition. Watch how its gradient is the <em>sum</em> of what each use sends back.',
    forwardHint: 'Forward: compute the value of every node, left to right.',
    backwardHint: 'Backward: start at the output with gradient 1 and walk right to left, applying each node’s rule.',
    takeaway: 'Each use of <code>x</code> sends back its own contribution (3, 3 and 1 with the preset numbers). Backprop <strong>adds</strong> them — that is the sum rule of calculus, <code>d(x²+x)/dx = 2x+1</code>.',
  },
  chain: {
    tab: '2 · Chain rule',
    view: [1000, 380], minWidth: 760,
    pos: { a: [100, 90], b: [100, 270], ab: [330, 180], c: [330, 40], z: [560, 110], r: [760, 110], f: [930, 110] },
    goal: 'How does a gradient travel through several operations?',
    intro: 'A short chain: multiply, add, ReLU, sum. At every node the gradient arriving from the right is <strong>multiplied by that node’s local slope</strong> and passed to the left. ReLU blocks the gradient wherever its input was negative.',
    forwardHint: 'Forward: each node computes its output from its inputs.',
    backwardHint: 'Backward: at each node, multiply the incoming gradient by the local slope.',
    takeaway: 'This is the chain rule: the gradient of the output with respect to an input is the product of the local slopes along the way. Try editing <code>c</code> so that every element of <code>z</code> is positive and watch ReLU stop blocking.',
  },
  mlp: {
    tab: '3 · A tiny network',
    view: [1300, 360], minWidth: 1040,
    above: ['x', 'W1', 'b1', 'W2', 'b2', 't'],
    pos: { x: [70, 100], W1: [230, 100], b1: [420, 100], W2: [600, 100], b2: [770, 100], t: [940, 100],
           xW1: [130, 270], h: [320, 270], a: [500, 270], aW2: [670, 270], o: [830, 270], d: [980, 270], q: [1120, 270], L: [1240, 270] },
    goal: 'What does one training step of a small network compute?',
    intro: 'A one-hidden-layer network with a mean-squared-error loss: <code>L = (relu(x·W1 + b1)·W2 + b2 − t)²</code>. The gradients of the <strong>weights and biases</strong> are what an optimizer uses to improve them.',
    forwardHint: 'Forward: the prediction and the loss.',
    backwardHint: 'Backward: from the loss back to every weight and bias.',
    takeaway: 'Matrix products and broadcasting follow the same two rules as everything else: multiply by the local slope, and add up what arrives. The second hidden unit is negative before ReLU, so it receives no gradient — change <code>b1</code> to revive it.',
  },
};

export const CONCEPTS = [
  { h: '1 · Forward', p: 'Every operation computes its output from its inputs, and TorC remembers which operation produced what. That memory is the <em>computation graph</em>.' },
  { h: '2 · Backward', p: 'Starting from the output (gradient 1), each node multiplies the gradient it received by its <em>local slope</em> and hands the result to its inputs. This is the chain rule.' },
  { h: '3 · Accumulate', p: 'If a value was used in several places, each place sends a gradient back and TorC <em>adds</em> them. Nothing is overwritten.' },
];
