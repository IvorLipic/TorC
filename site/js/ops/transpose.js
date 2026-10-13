// ops/transpose.js — swap rows and columns (autograd transpose(a): reverses the axes and copies into a new tensor).
// Backward: the gradient is moved back with the same swap (Variable transpose: input_grads[0] = grad_output.transpose(inverse axes)).
import { transpose } from './tensor.js';
import { tex, bar, mat } from './texutil.js';

export default {
  name: 'transpose', title: 'transpose', arity: 1,
  forward: transpose,
  backward: (g) => [transpose(g)],
  texForward: (n, [a]) => `${tex(n)}_{ij} = ${tex(a)}_{ji}`,
  whyForward: 'Swap rows and columns: entry (i, j) of the result is entry (j, i) of the input. TorC copies the data into a new tensor.',
  forwardNumbers: ({ name, inputs: [a], out }) => ({ note: '', latex: [`${tex(name)} = ${mat(a)}^{\\top} = ${mat(out)}`] }),
  texBackward: (n, [a]) => [`${bar(a)} = ${bar(n)}^{\\top}`],
  whyBackward: 'Transposing only moves entries to new places, so each input entry has slope 1 with respect to the one output entry it was moved to. The gradient is moved back the same way, which is another transpose.',
  backwardNumbers: ({ name, s, G }) => [`${bar(s.to)} = ${bar(name)}^{\\top} = ${mat(G)}^{\\top} = ${mat(s.raw)}`],
};
