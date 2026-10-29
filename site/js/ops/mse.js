// ops/mse.js — nn::MSELoss as ONE node: inputs [prediction, target], output shape [1] = mean((prediction - target)^2).
// Library rule (src/nn/losses.cpp): grad = (prediction - target) * (2 / N) with N = numel(prediction);
//   prediction gets  grad * grad_output,   target gets  -grad * grad_output.
// The JS port requires prediction and target to have the same shape, as the library does (it throws ShapeError otherwise).
import { T, zip, same, numel } from './tensor.js';
import { tex, bar, mat, tn, par, plus, negate } from './texutil.js';

const check = (p, t) => { if (!same(p.shape, t.shape)) throw new Error(`mse expects prediction and target of the same shape, got ${p.shape} and ${t.shape}`); };

export default {
  name: 'mse', title: 'mean squared error', arity: 2,
  forward: (p, t) => {
    check(p, t);
    const d = zip(p, t, (x, y) => x - y);
    return T([1], [d.v.reduce((s, x) => s + x * x, 0) / d.v.length]);
  },
  backward: (g, [p, t]) => {
    check(p, t);
    const k = 2 / numel(p.shape);
    const grad = zip(zip(p, t, (x, y) => x - y), g, (d, go) => k * d * go);
    return [grad, negate(grad)];
  },
  texForward: (n, [p, t]) => `${tex(n)} = \\tfrac{1}{N}\\sum_i \\left(${tex(p)}_i - ${tex(t)}_i\\right)^2`,
  whyForward: 'Subtract the target from the prediction, square each difference, and average the squares (N is the number of elements).',
  forwardNumbers: ({ name, inputs: [p, t], out }) => {
    const sq = p.v.map((x, i) => `(${tn(x)} - ${par(t.v[i])})^2`);
    const vals = p.v.map((x, i) => tn((x - t.v[i]) ** 2));
    const lines = [`${tex(name)} = \\tfrac{1}{${p.v.length}}\\left(${plus(sq)}\\right)`];
    if (p.v.length > 1) lines.push(`${tex(name)} = \\tfrac{1}{${p.v.length}}\\left(${plus(vals)}\\right) = ${tn(out.v[0])}`);
    else lines[0] += ` = ${tn(out.v[0])}`;
    return { note: '', latex: lines };
  },
  texBackward: (n, [p, t]) => [
    `${bar(p)} = \\dfrac{2}{N}\\left(${tex(p)} - ${tex(t)}\\right)\\,${bar(n)}`,
    `${bar(t)} = -\\dfrac{2}{N}\\left(${tex(p)} - ${tex(t)}\\right)\\,${bar(n)}`,
  ],
  whyBackward: 'Each squared difference (p − t)² has slope 2(p − t) with respect to p, and the mean divides by N. The target moves the difference the other way, so its slope is the negative of the prediction’s.',
  backwardNumbers: ({ name, node, graph, s, G }) => {
    const [p, t] = node.inputs.map((i) => graph.byName[i].value);
    const sign = s.arg === 0 ? '' : '-';
    return [`${bar(s.to)} = ${sign}\\dfrac{2}{${p.v.length}}\\left(${mat(p)} - ${mat(t)}\\right)\\cdot ${tn(G.v[0])} = ${mat(s.raw)}`];
  },
};
