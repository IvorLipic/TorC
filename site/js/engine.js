// engine.js — builds a computation graph from a trace scene and runs the backward pass, using the op rules in ops/.
// It is a JavaScript port of TorC's autograd (src/autograd.cpp) so that learners can edit inputs and watch values and
// gradients recompute. It is NOT the source of truth: site/test/*.test.mjs check that, for the preset inputs, it
// reproduces every value and gradient of the real C++ run stored in data/*.json (written by tools/trace/).
//
// Tensors are {shape: number[], v: number[]} (row-major). The op rules themselves live one-per-file in ./ops/.
import { getOp, OP_NAMES } from './ops/index.js';
import { numel, T, fill, same, broadcastShape, expandTo, reduceSumToShape, matmul, transpose } from './ops/tensor.js';

export { numel, T, fill, broadcastShape, expandTo, reduceSumToShape, matmul, transpose };
export const OPS = OP_NAMES;

// ---- graph ------------------------------------------------------------------
// `scene.nodes` is the list from the real library trace: {name, op, role, requires_grad, inputs, value:{shape,values}}
// in construction order (which is a valid topological order). `overrides` maps leaf name -> number[].
export function buildGraph(scene, overrides = {}) {
  const nodes = [], byName = {};
  for (const n of scene.nodes) {
    const node = { name: n.name, op: n.role === 'leaf' ? 'leaf' : n.op, role: n.role, inputs: n.inputs.slice(), index: nodes.length };
    if (n.role === 'leaf') {
      node.requires_grad = n.requires_grad;
      node.value = T(n.value.shape, overrides[n.name] ?? n.value.values);
    } else {
      node.requires_grad = n.inputs.some((i) => byName[i].requires_grad);
      node.value = getOp(n.op).forward(...n.inputs.map((i) => byName[i].value));
    }
    nodes.push(node);
    byName[n.name] = node;
  }
  return { nodes, byName, root: scene.root };
}

// Follows Variable::backward_with_grad: the library visits nodes in reverse DFS post-order, this visits them in reverse
// construction order. For the scenes here the order of the operations is identical (only where leaves are visited can
// differ, and leaves do no work). Visit nodes in reverse construction order; each node's
// gradient is the SUM of everything its consumers sent; each op's rule then sends raw gradients to
// its inputs, which are reduced to the input shape and accumulated.
export function runBackward(graph) {
  const { nodes, byName, root } = graph;
  const received = {};  // name -> [{from, arg, tensor, running}]
  const steps = [];
  const grad = {};
  for (const n of nodes) received[n.name] = [];
  const sumTensors = (list, shape) => list.reduce((acc, c) => T(shape, acc.v.map((x, i) => x + c.tensor.v[i])), fill(shape, 0));

  for (let k = nodes.length - 1; k >= 0; k--) {
    const n = nodes[k];
    if (!n.requires_grad) continue;
    const isRoot = n.name === root;
    const incoming = received[n.name];
    if (!isRoot && incoming.length === 0) continue;          // not reachable from the root
    const G = isRoot ? fill(n.value.shape, 1) : sumTensors(incoming, n.value.shape);
    grad[n.name] = isRoot ? null : G;                        // the library never stores the root's own grad
    const step = { node: n.name, isRoot, incoming: incoming.slice(), G, sends: [] };
    if (n.role === 'op') {
      const ins = n.inputs.map((i) => byName[i].value);
      const raws = getOp(n.op).backward(G, ins, n.value);
      n.inputs.forEach((iname, arg) => {
        const inp = byName[iname];
        const raw = raws[arg];
        const tracked = inp.requires_grad;
        const send = { to: iname, arg, tracked, raw, tensor: null, reduced: false, running: null };
        if (tracked) {
          send.reduced = !same(raw.shape, inp.value.shape);
          send.tensor = reduceSumToShape(raw, inp.value.shape);
          received[iname].push({ from: n.name, arg, tensor: send.tensor });
          send.running = sumTensors(received[iname], inp.value.shape);
        }
        step.sends.push(send);
      });
    }
    steps.push(step);
  }
  return { steps, grad };
}

// The ordered list of everything the learner steps through.
export function timeline(graph, bw) {
  const out = [{ kind: 'inputs' }];
  for (const n of graph.nodes) if (n.role === 'op') out.push({ kind: 'forward', node: n.name });
  for (const s of bw.steps) out.push({ kind: 'backward', node: s.node });
  return out;
}
