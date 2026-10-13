// explain.js — turns engine results into "paper work": for every operation, the rule used and the
// same rule with the actual numbers substituted, as LaTeX (rendered by the vendored KaTeX).
// Pure functions, no DOM, so site/test/engine.test.mjs can check that every string parses.
import { expandTo, matmul, transpose, numel } from './engine.js';

// ---- number / tensor formatting ------------------------------------------------
export function fmt(x) {
  if (Object.is(x, -0) || Math.abs(x) < 1e-9) return '0';
  const s = Number(x.toPrecision(4)).toString();
  return s.includes('e') ? Number(x.toPrecision(3)).toExponential().replace('e+', 'e') : s;
}
const tn = (x) => fmt(x).replace('-', '-');                     // number as TeX
const par = (x) => (x < 0 ? `(${tn(x)})` : tn(x));              // factor: parenthesise negatives
const isScalar = (t) => numel(t.shape) === 1;

export function tex(name) {                                     // W1 -> W_1, xW1 -> xW_1
  const m = /^([A-Za-z]+)(\d+)$/.exec(name);
  const base = name.length === 1 ? `\\mathit{${name}}` : `\\texttt{${name}}`;   // multi-letter node names are code names, not a product of variables
  return m ? (m[1].length === 1 ? `{\\mathit{${m[1]}}_{${m[2]}}}` : `{\\texttt{${m[1]}}_{${m[2]}}}`) : base;
}
const bar = (name) => `\\overline{${tex(name)}}`;
const txt = (s) => `\\text{${s}}`;
const IM = (l) => (globalThis.katex ? globalThis.katex.renderToString(l, { throwOnError: false, strict: 'ignore' }) : l);

export function mat(t) {                                        // tensor -> TeX
  if (isScalar(t)) return tn(t.v[0]);
  if (t.shape.length === 2 && t.shape[0] > 1) {
    const [r, c] = t.shape;
    const rows = [];
    for (let i = 0; i < r; i++) rows.push(t.v.slice(i * c, (i + 1) * c).map(tn).join(' & '));
    return `\\begin{bmatrix}${rows.join(' \\\\ ')}\\end{bmatrix}`;
  }
  return `\\begin{bmatrix}${t.v.map(tn).join(' & ')}\\end{bmatrix}`;
}
const shapeStr = (t) => (isScalar(t) ? 'a single number' : `shape (${t.shape.join(', ')})`);
const plus = (list) => list.map((s, i) => (i === 0 ? s : s.startsWith('-') ? ` - ${s.slice(1)}` : ` + ${s}`)).join('');

// ---- operation documentation ------------------------------------------------------
const OPNAME = { add: 'addition', sub: 'subtraction', mul: 'multiplication', relu: 'ReLU', sum: 'sum', mean: 'mean', matmul: 'matrix multiplication' };
export const opTitle = (op) => OPNAME[op] ?? op;
const ordinal = (k) => (k === 0 ? 'first' : 'second');

const SYM = { add: '+', sub: '-', mul: '\\odot' };
const FORWARD_RULE = {
  add: (n, [a, b]) => `${tex(n)} = ${tex(a)} + ${tex(b)}`,
  sub: (n, [a, b]) => `${tex(n)} = ${tex(a)} - ${tex(b)}`,
  mul: (n, [a, b]) => `${tex(n)} = ${tex(a)} \\odot ${tex(b)}\\quad(\\text{element by element})`,
  relu: (n, [a]) => `${tex(n)}_i = \\max(0,\\, ${tex(a)}_i)`,
  sum: (n, [a]) => `${tex(n)} = \\sum_i ${tex(a)}_i`,
  mean: (n, [a]) => `${tex(n)} = \\tfrac{1}{N}\\sum_i ${tex(a)}_i`,
  matmul: (n, [a, b]) => `${tex(n)}_{ij} = \\sum_k ${tex(a)}_{ik}\\,${tex(b)}_{kj}`,
};
const FORWARD_WHY = {
  add: 'Add the matching elements. If one input is smaller it is first copied (broadcast) to the other’s shape.',
  sub: 'Subtract matching elements; a smaller input is copied (broadcast) to the other’s shape first.',
  mul: 'Multiply the matching elements (not a matrix product).',
  relu: 'Keep positive numbers, replace negative ones by 0.',
  sum: 'Add all the elements into one number.',
  mean: 'Add all the elements and divide by how many there are (N).',
  matmul: 'Each output entry is the dot product of a row of the first matrix with a column of the second.',
};

const BACKWARD_RULE = {
  add: (n, [a, b]) => [`${bar(a)} = ${bar(n)}`, `${bar(b)} = ${bar(n)}`],
  sub: (n, [a, b]) => [`${bar(a)} = ${bar(n)}`, `${bar(b)} = -${bar(n)}`],
  mul: (n, [a, b]) => [`${bar(a)} = ${bar(n)} \\odot ${tex(b)}`, `${bar(b)} = ${bar(n)} \\odot ${tex(a)}`],
  relu: (n, [a]) => [`${bar(a)}_i = \\begin{cases}${bar(n)}_i & ${tex(a)}_i > 0\\\\ 0 & ${tex(a)}_i \\le 0\\end{cases}`],
  sum: (n, [a]) => [`${bar(a)}_i = ${bar(n)}\\quad\\text{for every } i`],
  mean: (n, [a]) => [`${bar(a)}_i = \\dfrac{${bar(n)}}{N}\\quad\\text{for every } i`],
  matmul: (n, [a, b]) => [`${bar(a)} = ${bar(n)}\\, ${tex(b)}^{\\top}`, `${bar(b)} = ${tex(a)}^{\\top}\\, ${bar(n)}`],
};
const BACKWARD_WHY = {
  add: 'Nudge either input by a tiny amount δ and the sum moves by exactly δ. The slope is 1, so the chain rule hands the incoming gradient on unchanged.',
  sub: 'Nudging the first input by δ moves the result by +δ (slope 1); nudging the second moves it by −δ (slope −1).',
  mul: 'If one factor grows by δ, the product grows by (the other factor)·δ. So the slope with respect to one input is the <em>other</em> input, and the chain rule multiplies it by the gradient that arrived.',
  relu: 'ReLU is a straight line of slope 1 for positive inputs and flat (slope 0) for negative ones, so the gradient passes only where the input was positive. At exactly 0 TorC uses slope 0.',
  sum: 'Every element is added exactly once, so each has slope 1: all elements receive the same gradient.',
  mean: 'Every element is added once and then divided by N, so each has slope 1/N.',
  matmul: 'Each output entry is a dot product, so entry (i,k) of the first input feeds every output in row i, weighted by the second matrix. Collecting those slopes gives the two matrix products shown.',
};

// ---- helpers for numeric substitution ----------------------------------------------
function elementLines(op, A, B, out) {                          // A,B,out already broadcast to the same shape
  const n = out.v.length;
  const rows = [];
  for (let i = 0; i < n; i++) {
    const a = A.v[i], b = B ? B.v[i] : null;
    const tag = n > 1 ? `_{${i + 1}}` : '';
    if (op === 'add') rows.push(`${tag ? 'n' + tag : 'n'} &= ${tn(a)} + ${par(b)} = ${tn(out.v[i])}`);
    else if (op === 'sub') rows.push(`${tag ? 'n' + tag : 'n'} &= ${tn(a)} - ${par(b)} = ${tn(out.v[i])}`);
    else if (op === 'mul') rows.push(`${tag ? 'n' + tag : 'n'} &= ${par(a)} \\cdot ${par(b)} = ${tn(out.v[i])}`);
    else if (op === 'relu') rows.push(`${tag ? 'n' + tag : 'n'} &= \\max(0,\\, ${tn(a)}) = ${tn(out.v[i])}`);
  }
  return rows;
}
const aligned = (rows, name) => `\\begin{aligned}${rows.map((r) => r.replace(/^n/, tex(name))).join(' \\\\ ')}\\end{aligned}`;

// ---- forward window -----------------------------------------------------------------
export function forwardWindow(graph, name) {
  const n = graph.byName[name];
  const ins = n.inputs.map((i) => graph.byName[i]);
  const iv = ins.map((i) => i.value);
  const blocks = [];
  blocks.push({ heading: 'The rule', latex: [FORWARD_RULE[n.op](name, n.inputs)], html: FORWARD_WHY[n.op] });
  const calc = [];
  let note = '';
  if (['add', 'sub', 'mul'].includes(n.op)) {
    const [a, b] = iv;
    const shape = n.value.shape;
    const A = expandTo(a, shape), B = expandTo(b, shape);
    const expanded = (a.shape.join() !== shape.join() ? [`${tex(n.inputs[0])} \\to ${mat(A)}`] : [])
      .concat(b.shape.join() !== shape.join() ? [`${tex(n.inputs[1])} \\to ${mat(B)}`] : []);
    if (expanded.length) { calc.push(expanded.join(',\\quad ')); note = 'The smaller input is copied (broadcast) so both have the same shape:'; }
    calc.push(`${mat(A)} ${SYM[n.op]} ${mat(B)} = ${mat(n.value)}`);
    if (n.value.v.length > 1 || ['add', 'sub'].includes(n.op)) calc.push(aligned(elementLines(n.op, A, B, n.value), name));
  } else if (n.op === 'relu') {
    calc.push(aligned(elementLines('relu', iv[0], null, n.value), name));
  } else if (n.op === 'sum') {
    calc.push(`${tex(name)} = ${plus(iv[0].v.map(tn))} = ${tn(n.value.v[0])}`);
  } else if (n.op === 'mean') {
    calc.push(`${tex(name)} = \\tfrac{1}{${iv[0].v.length}}\\left(${plus(iv[0].v.map(tn))}\\right) = ${tn(n.value.v[0])}`);
  } else if (n.op === 'matmul') {
    const [a, b] = iv;
    calc.push(`${mat(a)}\\,${mat(b)} = ${mat(n.value)}`);
    const [r, k] = a.shape, c = b.shape[1];
    const rows = [];
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
      const terms = [];
      for (let p = 0; p < k; p++) terms.push(`${par(a.v[i * k + p])}\\cdot ${par(b.v[p * c + j])}`);
      rows.push(`n_{${i + 1}${j + 1}} &= ${terms.join(' + ')} = ${tn(n.value.v[i * c + j])}`);
    }
    if (rows.length <= 8) calc.push(aligned(rows, name));
  }
  blocks.push({ heading: 'With our numbers', html: note, latex: calc });
  blocks.push({ heading: 'Result', latex: [`${tex(name)} = ${mat(n.value)}`], html: `Stored as ${shapeStr(n.value)}.` });
  return { title: `${name} = ${opTitle(n.op)}`, kind: 'forward', blocks };
}

// ---- backward window for an operation node -------------------------------------------
function contributionLines(step, graph) {
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
  const blocks = [];
  // 1. what has arrived
  if (step.isRoot) {
    blocks.push({ heading: 'Where backward starts', latex: [`${bar(name)} = \\dfrac{\\partial ${tex(name)}}{\\partial ${tex(name)}} = ${mat(step.G)}`],
      html: `<code>${name}</code> is the number we are differentiating, so its gradient with respect to itself is 1. TorC seeds the backward pass with this 1 (it is not stored on the variable afterwards). <strong>Notation:</strong> ${IM(bar(name))} means ${IM('\\partial\\,\\text{out}/\\partial\\,' + tex(name))}, how much the final result changes per unit change in <code>${name}</code>; every node gets such a gradient, written with a bar.` });
  } else {
    const many = step.incoming.length > 1;
    blocks.push({ heading: 'Gradient that has arrived here', latex: contributionLines(step, graph),
      html: `Notation: ${IM(bar(name))} means ${IM('\\partial\\,\\text{out}/\\partial\\,' + tex(name))} — how much the final result changes per unit change in <code>${name}</code>. ` +
            (many ? `<code>${name}</code> is used by ${step.incoming.length} operations, so every consumer sends a contribution and they are <strong>added</strong>: ${sourceNote(step, graph)}.`
                  : `It was sent by <code>${step.incoming[0].from}</code>.`) });
  }
  // 2. the rule
  blocks.push({ heading: 'Local rule (chain rule)', latex: BACKWARD_RULE[n.op](name, n.inputs), html: BACKWARD_WHY[n.op] +
    (n.inputs.length === 2 && n.inputs[0] === n.inputs[1] ? ` <br><strong>Both inputs are the same variable <code>${n.inputs[0]}</code></strong>, so both lines apply to it and both results are added to its gradient.` : '') +
    ' <br><em>Chain rule: (gradient arriving at the output) × (local slope).</em>' });
  // 3. numbers per input
  const G = step.G;
  step.sends.forEach((s) => {
    const inp = graph.byName[s.to];
    const rows = [];
    const label = `Gradient for ${s.to} (${ordinal(s.arg)} input)`;
    if (!s.tracked) {
      blocks.push({ heading: label, html: `<code>${s.to}</code> is a constant (<code>requires_grad = false</code>), so TorC discards what would be sent here: nothing is accumulated for it.`, latex: [] });
      return;
    }
    const other = graph.byName[n.inputs[1 - s.arg]];
    if (n.op === 'add') rows.push(`${bar(s.to)} = ${bar(name)} = ${mat(G)}`);
    else if (n.op === 'sub') rows.push(s.arg === 0 ? `${bar(s.to)} = ${bar(name)} = ${mat(G)}` : `${bar(s.to)} = -${bar(name)} = ${mat({ shape: G.shape, v: G.v.map((x) => -x) })}`);
    else if (n.op === 'mul') {
      const O = expandTo(other.value, G.shape);
      rows.push(`${bar(s.to)} = ${bar(name)} \\odot ${tex(other.name)} = ${mat(G)} \\odot ${mat(O)} = ${mat(s.raw)}`);
    } else if (n.op === 'relu') {
      const x = graph.byName[n.inputs[0]].value;
      rows.push(`\\text{slope: } ${mat({ shape: x.shape, v: x.v.map((z) => (z > 0 ? 1 : 0)) })}\\quad(1 \\text{ where } ${tex(n.inputs[0])} > 0)`);
      rows.push(`${bar(s.to)} = ${bar(name)} \\odot \\text{slope} = ${mat(G)} \\odot ${mat({ shape: x.shape, v: x.v.map((z) => (z > 0 ? 1 : 0)) })} = ${mat(s.raw)}`);
    } else if (n.op === 'sum') rows.push(`${bar(s.to)}_i = ${tn(G.v[0])}\\ \\Rightarrow\\ ${bar(s.to)} = ${mat(s.raw)}`);
    else if (n.op === 'mean') rows.push(`${bar(s.to)}_i = \\dfrac{${tn(G.v[0])}}{${inp.value.v.length}} = ${tn(s.raw.v[0])}\\ \\Rightarrow\\ ${bar(s.to)} = ${mat(s.raw)}`);
    else if (n.op === 'matmul') {
      if (s.arg === 0) { const Bt = transpose(other.value); rows.push(`${bar(s.to)} = ${bar(name)}\\,${tex(other.name)}^{\\top} = ${mat(G)}\\,${mat(Bt)} = ${mat(s.raw)}`); }
      else { const At = transpose(other.value); rows.push(`${bar(s.to)} = ${tex(other.name)}^{\\top}\\,${bar(name)} = ${mat(At)}\\,${mat(G)} = ${mat(s.raw)}`); }
    }
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
    const total = totalContributions(bw, s.to);
    const k = bw.steps.flatMap((st) => st.sends).filter((x) => x.to === s.to && x.tracked);
    const pos = k.indexOf(s) + 1;
    del.push(`${bar(s.to)} \\mathrel{+}= ${mat(s.tensor)}\\quad\\Rightarrow\\quad ${bar(s.to)}\\text{ so far} = ${mat(s.running)}`);
    delHtml.push(total > 1 ? `<code>${s.to}</code>: contribution ${pos} of ${total}` : `<code>${s.to}</code> receives only this one contribution`);
  });
  if (del.length) blocks.push({ heading: 'Deliver (accumulate into the inputs)', latex: del, html: delHtml.join(' · ') + '. Gradients are added, never overwritten.' });
  return { title: `${name} · ${opTitle(n.op)}`, kind: 'backward', blocks };
}
const bar_html = (name) => `<code>∂out/∂${name}</code>`;
const totalContributions = (bw, name) => bw.steps.flatMap((s) => s.sends).filter((x) => x.to === name && x.tracked).length;

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
