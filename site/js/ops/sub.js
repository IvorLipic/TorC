// ops/sub.js — elementwise a - b with broadcasting.
import { zip, map } from './tensor.js';
import { tex, bar, mat, tn, par, negate } from './texutil.js';
import { elementwiseForwardNumbers } from './elementwise.js';

export default {
  name: 'sub', title: 'subtraction', arity: 2,
  forward: (a, b) => zip(a, b, (x, y) => x - y),
  backward: (g) => [g, map(g, (x) => -x)],
  texForward: (n, [a, b]) => `${tex(n)} = ${tex(a)} - ${tex(b)}`,
  whyForward: 'Subtract matching elements; a smaller input is copied (broadcast) to the other’s shape first.',
  forwardNumbers: elementwiseForwardNumbers('-', (a, b, o) => `${tn(a)} - ${par(b)} = ${tn(o)}`, true),
  texBackward: (n, [a, b]) => [`${bar(a)} = ${bar(n)}`, `${bar(b)} = -${bar(n)}`],
  whyBackward: 'Nudging the first input by δ moves the result by +δ (slope 1); nudging the second moves it by −δ (slope −1).',
  backwardNumbers: ({ name, s, G }) => [s.arg === 0
    ? `${bar(s.to)} = ${bar(name)} = ${mat(G)}`
    : `${bar(s.to)} = -${bar(name)} = ${mat(negate(G))}`],
};
