// widgets/math-window.js — the "math window": for the selected node, the rule and the same rule with the real numbers
// (built by explain.js, rendered with KaTeX). Before anything is selected it shows the scene's intro.
import { el, keepFocus } from '../core/dom.js';
import { math } from '../core/katex.js';
import { forwardWindow, backwardWindow, leafWindow, mat, opTitle } from '../explain.js';

function leafInputWindow(n) {
  return {
    title: `${n.name} · input`, kind: 'input',
    blocks: [
      { heading: n.requires_grad ? 'A learnable input' : 'A constant input', html: n.requires_grad
        ? `<code>${n.name}</code> was created with <code>requires_grad = true</code>. TorC computes the gradient of the output with respect to it. In training, these are the weights and biases.`
        : `<code>${n.name}</code> was created with <code>requires_grad = false</code>. It is data, so TorC computes no gradient for it.`, latex: [] },
      { heading: 'Value', latex: [`${'\\mathit{' + n.name + '}'} = ${mat(n.value)}`], html: `Shape (${n.value.shape.join(', ')}). Edit it in the <a href="#in-${n.name}" data-focus="${n.name}">Inputs</a> panel.` }],
  };
}

export function windowData(graph, bw, name, tab) {
  const n = graph.byName[name];
  if (n.role === 'leaf') return tab === 'grad' ? leafWindow(graph, bw, name) : leafInputWindow(n);
  if (tab === 'backward') {
    return bw.steps.some((s) => s.node === name) ? backwardWindow(graph, bw, name)
      : { title: `${name} · ${opTitle(n.op)}`, kind: 'backward', blocks: [{ heading: 'Not part of the backward pass', html: 'No gradient reaches this node from the output.', latex: [] }] };
  }
  return forwardWindow(graph, name);
}

// view = {graph, bw, tl, step, layout, sel, tab, pop}; actions = {onTab(id), onBackToStep(), onFocusInputs(name)}
export function renderWindow(box, { graph, bw, tl, step, layout: L, sel, tab, pop }, { onTab, onBackToStep, onFocusInputs }) {
  if (!sel) {
    box.className = 'win intro' + (pop ? ' pop' : '');
    box.replaceChildren(
      el('h3', { html: L.goal }),   // layout text is an HTML fragment everywhere (the lab's goal line renders it the same way)
      el('p', { html: L.intro }),
      el('p', { class: 'hint', html: `Press <kbd>Next</kbd> or <kbd>→</kbd> to start. ${L.forwardHint} ${L.backwardHint} At each step this panel shows the rule and the same rule with the numbers. You can also click any node.` }));
    return;
  }
  const n = graph.byName[sel];
  const w = windowData(graph, bw, sel, tab);
  const tabs = n.role === 'leaf' ? [['input', 'Value'], ['grad', 'Gradient']] : [['forward', 'Forward'], ['backward', 'Backward']];
  box.className = 'win ' + w.kind + (pop ? ' pop' : '');
  box.replaceChildren(...[
    el('div', { class: 'win-head' },
      el('h3', {}, w.title),
      el('div', { class: 'win-tabs', role: 'tablist' }, tabs.map(([id, label]) => el('button', { class: 'wtab ' + id, 'data-key': 'wtab:' + id, role: 'tab', 'aria-selected': tab === id, onclick: () => keepFocus(() => onTab(id)) }, label)))),
    el('div', { class: 'win-body' }, w.blocks.map((b) => el('section', { class: 'blk' },
      el('h4', {}, b.heading),
      b.html ? el('p', { html: b.html }) : null,
      ...(b.latex ?? []).map((l) => el('div', { class: 'math', html: math(l) }))))),
    sel && tl[step].node !== sel ? el('p', { class: 'hint' }, 'You are exploring ', el('code', {}, sel), '. ', el('button', { class: 'link', onclick: onBackToStep }, 'Back to the current step')) : null].filter(Boolean));
  box.querySelectorAll('a[data-focus]').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); onFocusInputs(a.dataset.focus); }));
}
