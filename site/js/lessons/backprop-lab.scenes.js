// backprop-lab.scenes.js — layout (where each named node is drawn) and the teaching text of each scene of the Backprop lab.
// Numbers never live here: values, edges and gradients come from data/autograd_trace.json (real TorC run)
// and from engine.js (live recomputation after the learner edits inputs).
export const SCENES = {
  reuse: {
    tab: '1 · Reuse',
    view: [900, 210], minWidth: 600,
    pos: { x: [150, 130], m: [450, 55], y: [750, 130] },
    goal: 'What happens when one value is used more than once?',
    intro: 'The multiplication uses <code>x</code> twice, and the addition uses it once more. The gradient of <code>x</code> is the sum of the three gradients that come back.',
    forwardHint: 'The forward pass computes every node from left to right.',
    backwardHint: 'The backward pass starts at the output with a gradient of one and goes from right to left.',
    takeaway: 'Each use of <code>x</code> sends back its own gradient. With the preset numbers these are 3, 3 and 1. Backpropagation adds them, which is the sum rule of calculus, <code>d(x²+x)/dx = 2x+1</code>.',
  },
  chain: {
    tab: '2 · Chain rule',
    view: [1000, 290], minWidth: 760,
    pos: { a: [100, 70], b: [100, 215], ab: [330, 150], c: [330, 35], z: [560, 95], r: [760, 95], f: [930, 95] },
    goal: 'How does a gradient pass through several operations?',
    intro: 'This chain has four operations, which are multiply, add, ReLU and sum. Each node multiplies the gradient that comes from the right by its local derivative and passes the result to the left. ReLU stops the gradient wherever its input was negative.',
    forwardHint: 'The forward pass computes each output from its inputs.',
    backwardHint: 'The backward pass multiplies the incoming gradient by the local derivative at each node.',
    takeaway: 'This is the chain rule. The gradient of the output with respect to an input is the product of the local derivatives on the path between them. Change <code>c</code> so that every element of <code>z</code> is positive, and ReLU no longer stops the gradient.',
  },
  mlp: {
    tab: '3 · A tiny network',
    view: [1280, 300], minWidth: 1000,
    above: ['x', 'W1', 'b1', 'W2', 'b2', 't'],
    pos: { x: [70, 85], W1: [230, 85], b1: [420, 85], W2: [590, 85], b2: [740, 85], t: [900, 85],
           xW1: [110, 225], h: [300, 225], a: [490, 225], aW2: [655, 225], o: [800, 225], d: [960, 225], q: [1100, 225], L: [1210, 225] },
    goal: 'What does one training step of a small network compute?',
    intro: 'This network has one hidden layer. Its input is a batch of two examples, which are the two rows of <code>x</code>. The loss is the mean squared error. An optimizer uses the gradients of the weights and biases to change them.',
    forwardHint: 'The forward pass computes the prediction and the loss.',
    backwardHint: 'The backward pass goes from the loss back to every weight and bias.',
    takeaway: 'Matrix products and broadcasting follow the same two rules as every other operation. Each node multiplies by its local derivative, and a value adds up all the gradients it receives. The layer added the bias <code>b1</code> to both rows of the batch, so its gradient is the sum over the rows. ReLU stopped the gradient wherever <code>h</code> was negative. Change <code>b1</code> or <code>x</code> to see which entries change.',
  },
};
