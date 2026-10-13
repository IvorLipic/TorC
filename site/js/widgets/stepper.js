// widgets/stepper.js — the step timeline: restart / back / next buttons, a counter and one chip per checkpoint.
// `tl` is the list from engine.timeline(); `step` the current index; onGo(i) moves. Pure render into `box`.
import { el } from '../core/dom.js';

export function renderStepper(box, { tl, step }, { onGo }) {
  const chips = [];
  let phase = null;
  tl.forEach((t, i) => {
    const p = t.kind === 'inputs' ? 'start' : t.kind;
    if (p !== phase) { phase = p; chips.push(el('span', { class: 'phase ' + p }, p === 'start' ? '' : p === 'forward' ? 'Forward →' : '← Backward')); }
    const label = t.kind === 'inputs' ? 'Start' : t.node;
    chips.push(el('button', { class: `chip ${t.kind} ${i === step ? 'on' : ''} ${i < step ? 'done' : ''}`, 'aria-current': i === step ? 'step' : false, 'data-key': 'chip:' + i, onclick: () => onGo(i), title: t.kind === 'inputs' ? 'Inputs' : `${t.kind}: ${t.node}` }, label));
  });
  box.replaceChildren(
    el('button', { class: 'btn', 'data-key': 'restart', onclick: () => onGo(0), 'aria-label': 'Restart', title: 'Restart' }, '⏮'),
    el('button', { class: 'btn', 'data-key': 'back', onclick: () => onGo(step - 1), disabled: step === 0, 'aria-label': 'Previous step' }, '◀ Back'),
    el('button', { class: 'btn primary', 'data-key': 'next', onclick: () => onGo(step + 1), disabled: step === tl.length - 1, 'aria-label': 'Next step' }, 'Next ▶'),
    el('span', { class: 'counter' }, `${step}/${tl.length - 1}`),
    el('div', { class: 'chips', role: 'group', 'aria-label': 'Steps (checkpoints)' }, chips));
}

// Keep the current chip centred when the chip row scrolls.
export function revealChip(box) {
  const chips = box.querySelector('.chips'), on = box.querySelector('.chip.on');
  if (chips && on && chips.scrollWidth > chips.clientWidth) chips.scrollLeft = on.offsetLeft - chips.clientWidth / 2 + on.offsetWidth / 2;
}
