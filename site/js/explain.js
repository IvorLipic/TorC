// explain.js — turns engine results into "paper work": for every operation, the rule used and the
// same rule with the actual numbers substituted, as LaTeX (rendered by the vendored KaTeX).
// The per-op text lives in ops/<op>.js; this file only assembles windows from it. Pure functions, no DOM,
// so the tests can check that every string parses.
import { getOp, opTitle } from './ops/index.js';
import { numel } from './ops/tensor.js';
import { fmt, tex, bar, mat, tn, txt, plus, shapeStr } from './ops/texutil.js';
import { math, katexAvailable } from './core/katex.js';

export { fmt, mat, tex, opTitle };
const IM = (l) => (katexAvailable() ? math(l, false) : l);
const ordinal = (k) => (k === 0 ? 'first' : 'second');

// ---- forward window -----------------------------------------------------------------
export function forwardWindow(graph, name) {
  const n = graph.byName[name];
  const op = getOp(n.op);
  const iv = n.inputs.map((i) => graph.byName[i].value);
  const blocks = [];
  blocks.push({ heading: 'The rule', latex: [op.texForward(name, n.inputs)], html: op.whyForward });
  const { note = '', latex } = op.forwardNumbers({ name, node: n, graph, inputs: iv, out: n.value });
  blocks.push({ heading: 'With our numbers', html: note, latex });
  blocks.push({ heading: 'Result', latex: [`${tex(name)} = ${mat(n.value)}`], html: `Stored as ${shapeStr(n.value)}.` });
  return { title: `${name} = ${op.title}`, kind: 'forward', blocks };
}

// ---- backward window for an operation node -------------------------------------------
function contributionLines(step) {
  const lines = [];
  const cs = step.incoming;
  if (step.isRoot) return lines;
  if (cs.length === 1) lines.push(`${bar(step.node)} = ${mat(cs[0].tensor)}`);
  else {
    lines.push(`${bar(step.node)} = ${cs.map((c, i) => `c_{${i + 1}}`).join(' + ')}`);
    lines.push(`${bar(step.node)} = ${plus(cs.map((c) => mat(c.tensor)))} = ${mat(step.G)}`);
  }
  return lines;
}
const sourceNote = (step, graph) => step.incoming.map((c, i) => {
  const from = graph.byName[c.from];
  return `<code>c<sub>${i + 1}</sub></code> came from <code>${c.from}</code> (${opTitle(from.op)}, as its ${ordinal(c.arg)} input)`;
}).join('; ');

export function backwardWindow(graph, bw, name) {
  const step = bw.steps.find((s) => s.node === name);
  const n = graph.byName[name];
  const op = getOp(n.op);
  const blocks = [];
  // 1. what has arrived
  if (step.isRoot) {
    blocks.push({ heading: 'Where backward starts', latex: [`${bar(name)} = \\dfrac{\\partial ${tex(name)}}{\\partial ${tex(name)}} = ${mat(step.G)}`],
      html: `<code>${name}</code> is the number we are differentiating, so its gradient with respect to itself is 1. TorC seeds the backward pass with this 1 (it is not stored on the variable afterwards). <strong>Notation:</strong> ${IM(bar(name))} means ${IM('\\partial\\,\\text{out}/\\partial\\,' + tex(name))}, how much the final result changes per unit change in <code>${name}</code>; every node gets such a gradient, written with a bar.` });
  } else {
    const many = step.incoming.length > 1;
    blocks.push({ heading: 'Gradient that has arrived here', latex: contributionLines(step),
      html: `Notation: ${IM(bar(name))} means ${IM('\\partial\\,\\text{out}/\\partial\\,' + tex(name))} — how much the final result changes per unit change in <code>${name}</code>. ` +
            (many ? `<code>${name}</code> is used by ${step.incoming.length} operations, so every consumer sends a contribution and they are <strong>added</strong>: ${sourceNote(step, graph)}.`
                  : `It was sent by <code>${step.incoming[0].from}</code>.`) });
  }
  // 2. the rule
  blocks.push({ heading: 'Local rule (chain rule)', latex: op.texBackward(name, n.inputs), html: op.whyBackward +
    (n.inputs.length === 2 && n.inputs[0] === n.inputs[1] ? ` <br><strong>Both inputs are the same variable <code>${n.inputs[0]}</code></strong>, so both lines apply to it and both results are added to its gradient.` : '') +
    ' <br><em>Chain rule: (gradient arriving at the output) × (local slope).</em>' });
  // 3. numbers per input
  const G = step.G;
  step.sends.forEach((s) => {
    const label = `Gradient for ${s.to} (${ordinal(s.arg)} input)`;
    if (!s.tracked) {
      blocks.push({ heading: label, html: `<code>${s.to}</code> is a constant (<code>requires_grad = false</code>), so TorC discards what would be sent here: nothing is accumulated for it.`, latex: [] });
      return;
    }
    const other = graph.byName[n.inputs[1 - s.arg]];
    const rows = op.backwardNumbers({ name, node: n, graph, s, G, other });
    let html = '';
    if (s.reduced) {
      const rowsN = numel(s.raw.shape) / numel(s.tensor.shape);
      if (rowsN > 1) rows.push(`${txt('shrink back to the shape of ')}${tex(s.to)}:\\quad ${bar(s.to)} = ${plus(Array.from({ length: rowsN }, (_, r) => mat({ shape: s.tensor.shape, v: s.raw.v.slice(r * s.tensor.v.length, (r + 1) * s.tensor.v.length) })))} = ${mat(s.tensor)}`);
      html = `<code>${s.to}</code> was copied (broadcast) to every row in the forward pass, so in the backward pass the gradients of all those copies are <strong>summed</strong> to give ${shapeStr(s.tensor)} again (${rowsN === 1 ? 'here there is only one row, so the sum has a single term' : rowsN + ' rows here'}).`;
    }
    blocks.push({ heading: label, latex: rows, html });
  });
  // 4. delivery / accumulation
  const del = [];
  const delHtml = [];
  step.sends.filter((s) => s.tracked).forEach((s) => {
    const k = bw.steps.flatMap((st) => st.sends).filter((x) => x.to === s.to && x.tracked);
    const total = k.length;
    const pos = k.indexOf(s) + 1;
    del.push(`${bar(s.to)} \\mathrel{+}= ${mat(s.tensor)}\\quad\\Rightarrow\\quad ${bar(s.to)}\\text{ so far} = ${mat(s.running)}`);
    delHtml.push(total > 1 ? `<code>${s.to}</code>: contribution ${pos} of ${total}` : `<code>${s.to}</code> receives only this one contribution`);
  });
  if (del.length) blocks.push({ heading: 'Deliver (accumulate into the inputs)', latex: del, html: delHtml.join(' · ') + '. Gradients are added, never overwritten.' });
  return { title: `${name} · ${op.title}`, kind: 'backward', blocks };
}

// ---- backward window for a leaf ------------------------------------------------------
export function leafWindow(graph, bw, name) {
  const n = graph.byName[name];
  const step = bw.steps.find((s) => s.node === name);
  const blocks = [];
  if (!n.requires_grad) {
    blocks.push({ heading: 'Constant', html: `<code>${name}</code> was created with <code>requires_grad = false</code>, so it takes no part in the backward pass.`, latex: [] });
    return { title: `${name} · constant input`, kind: 'leaf', blocks };
  }
  const cs = step ? step.incoming : [];
  const lines = [];
  if (cs.length === 1) lines.push(`${bar(name)} = ${mat(cs[0].tensor)}`);
  else if (cs.length > 1) {
    lines.push(`${bar(name)} = ${cs.map((_, i) => `c_{${i + 1}}`).join(' + ')}`);
    lines.push(`${bar(name)} = ${plus(cs.map((c) => mat(c.tensor)))} = ${mat(step.G)}`);
  }
  blocks.push({ heading: 'Everything that arrived', latex: lines,
    html: cs.length > 1
      ? `<code>${name}</code> is used in ${cs.length} places, so it received ${cs.length} contributions and <strong>adds them up</strong>: ${cs.map((c, i) => `<code>c<sub>${i + 1}</sub></code> from <code>${c.from}</code> (${opTitle(graph.byName[c.from].op)}, ${ordinal(c.arg)} input)`).join('; ')}.`
      : `<code>${name}</code> received one contribution, from <code>${cs[0]?.from}</code>.` });
  blocks.push({ heading: 'Final gradient', latex: [`${txt(name + '.grad')} = ${mat(step.G)}`],
    html: 'This is what TorC stores in <code>' + name + '.grad()</code>. An optimizer would use it to nudge <code>' + name + '</code> and make the output smaller (gradient descent).' });
  return { title: `${name} · final gradient`, kind: 'leaf', blocks };
}
