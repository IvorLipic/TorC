// widgets/inputs-popover.js — the "✎ Inputs" popover: one number box per element of every leaf tensor of the current scene.
// Typing a valid number calls onEdit(name, index, value); an invalid entry is flagged and the previous value stays in use.
// Markup expected in `host`: #inputs-btn, #inputs (the popover), #inputs-inner. Closes on Esc and on a click outside.
import { $, el } from '../core/dom.js';

const MATCH_TITLE = 'With the preset inputs, every value and gradient on this page equals what the real TorC C++ library printed.';
const EDITED_TITLE = 'You changed the inputs: values are recomputed in your browser by a JavaScript port of TorC’s rules.';

// 2-D tensors are shown as rows, matching how the matrices are written in the math windows
function rowsOf(n) {
  const idx = n.value.v.map((v, i) => [v, i]);
  if (n.value.shape.length !== 2) return [idx];
  const c = n.value.shape[1], rows = [];
  for (let r = 0; r < n.value.shape[0]; r++) rows.push(idx.slice(r * c, (r + 1) * c));
  return rows;
}

export function createInputsPopover(host, { onEdit, onReset, signal }) {
  const btn = $('#inputs-btn', host), pop = $('#inputs', host), inner = $('#inputs-inner', host);

  function open(isOpen) { pop.hidden = !isOpen; btn.setAttribute('aria-expanded', String(isOpen)); }

  // leaves: the graph's leaf nodes; edited: true when any input differs from the preset
  function render({ leaves, edited }) {
    btn.classList.toggle('edited', edited);
    inner.replaceChildren(
      el('div', { class: 'inputs-head' },
        el('strong', {}, 'Inputs'),
        el('span', { class: 'hint' }, ' — type new numbers; everything recomputes.'),
        el('span', { class: 'badge ' + (edited ? 'warn' : 'ok'), id: 'match-badge', title: edited ? EDITED_TITLE : MATCH_TITLE },
          edited ? 'your values (recomputed in browser)' : '✓ matches the real TorC run'),
        edited ? el('button', { class: 'link', onclick: onReset }, 'reset') : null),
      el('div', { class: 'inputs-grid' }, leaves.map((n) => el('label', { class: 'input-row', id: 'in-' + n.name },
        el('span', { class: 'in-name' }, n.name, el('small', {}, n.requires_grad ? 'learnable' : 'constant')),
        el('span', { class: 'in-vals' }, rowsOf(n).map((row) => el('span', { class: 'in-row' }, row.map(([v, i]) => el('input', {
          type: 'number', step: 'any', value: String(Number(v.toPrecision(7))), 'aria-label': `${n.name}[${i}]`,
          oninput: (ev) => {
            const x = parseFloat(ev.target.value);
            const ok = Number.isFinite(x) && Math.abs(x) <= 1e6;
            ev.target.classList.toggle('bad', !ok);
            ev.target.title = ok ? '' : 'Enter a number between -1,000,000 and 1,000,000 (the previous valid value is still in use).';
            if (ok) onEdit(n.name, i, x);
          },
        }))))),
        el('span', { class: 'in-shape' }, 'shape (' + n.value.shape.join(', ') + ')')))));
  }

  // after an edit the focused <input> must survive, so only the badge / button / reset link are updated in place
  function updateEdited(edited) {
    btn.classList.toggle('edited', edited);
    const b = $('#match-badge', host);
    if (b) { b.className = 'badge ' + (edited ? 'warn' : 'ok'); b.title = edited ? EDITED_TITLE : MATCH_TITLE; b.textContent = edited ? 'your values (recomputed in browser)' : '✓ matches the real TorC run'; }
    const head = $('.inputs-head', host), link = $('.inputs-head .link', host);
    if (edited && head && !link) head.append(el('button', { class: 'link', onclick: onReset }, 'reset'));
    else if (!edited && link) link.remove();
  }

  function focusInput(name) { open(true); $(`#in-${CSS.escape(name)} input`, host)?.focus(); }

  btn.addEventListener('click', () => open(pop.hidden), { signal });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !pop.hidden) { open(false); btn.focus(); } }, { signal });
  document.addEventListener('click', (e) => {
    if (!pop.hidden && !pop.contains(e.target) && !e.target.closest('#inputs-btn') && !e.target.closest('a[data-focus]')) open(false);
  }, { signal });

  return { render, updateEdited, open, focusInput };
}
