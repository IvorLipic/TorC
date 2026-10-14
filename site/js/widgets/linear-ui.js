// widgets/linear-ui.js — small DOM building blocks shared by the `linear-layer` widgets (radio groups, matrix grids) and
// the one-time stylesheet they share. Nothing runs at import time except defining
// functions, so Node can import the module. Styles are injected once, as <style id="…"> elements, by the widget that needs them.
import { el } from '../core/dom.js';

export function injectStyle(id, css) {
  if (typeof document === 'undefined' || document.getElementById(id)) return;
  const s = document.createElement('style');
  s.id = id;
  s.textContent = css;
  document.head.append(s);
}

let counter = 0;
export const uid = (prefix) => `${prefix}-${++counter}`;

export const BASE_STYLE_ID = 'lin-base-style';
export const BASE_CSS = `
.lin-w{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.9rem 1.1rem 1.1rem;margin:14px 0}
.lin-w h3{margin:0;font-size:1.05rem}
.lin-head{display:flex;flex-wrap:wrap;gap:.3rem .8rem;align-items:center;justify-content:space-between;margin:0 0 .6rem}
.lin-note{color:#c4ccd5;font-size:.93rem;margin:.45rem 0}
.lin-note.dim{color:var(--muted);font-size:.86rem}
.lin-row{display:flex;flex-wrap:wrap;gap:.5rem .7rem;align-items:center}
.lin-fields{display:flex;flex-wrap:wrap;gap:.6rem 1.3rem;align-items:flex-end;margin:.2rem 0 .6rem}
.lin-field label{display:block;font-size:.82rem;color:var(--muted);margin-bottom:2px}
.lin-field label b{color:var(--fg);font-family:ui-monospace,monospace}
.lin-err{color:#ff9d93;font-size:.8rem;line-height:1.3;max-width:16rem}
.lin-err:empty{display:none}
.lin-err::before{content:"\\2717  "}
.lin-btns{display:flex;flex-wrap:wrap;gap:6px}
.lin-seg{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.lin-seg .lin-lbl{font-size:.85rem;color:var(--muted);margin-right:.2rem}
.lin-seg label{position:relative;display:inline-block}
.lin-seg input{position:absolute;opacity:0;inset:0;width:100%;height:100%;margin:0;cursor:pointer}
.lin-seg label span{display:inline-block;border:1px solid var(--border);border-radius:99px;padding:.22rem .85rem;font-size:.9rem;color:var(--muted)}
.lin-seg input:checked + span{color:var(--fg);border-color:var(--fwd);background:rgba(88,196,221,.12);font-weight:600}
.lin-seg input:checked + span::before{content:"\\25CF  "}
.lin-seg input:focus-visible + span{outline:2px solid var(--fwd);outline-offset:2px}
.lin-w button:focus-visible,.lin-w input:focus-visible,.lin-w select:focus-visible{outline:2px solid var(--fwd);outline-offset:2px}
.lin-mat{margin:0;display:flex;flex-direction:column;gap:.3rem;min-width:0}
.lin-mat figcaption{font-size:.84rem;color:var(--muted);line-height:1.3}
.lin-mat figcaption code{color:var(--fg)}
.lin-grid{display:grid;gap:3px 4px;align-items:center;justify-items:stretch;padding:5px 9px;border-left:2px solid var(--muted);border-right:2px solid var(--muted);border-radius:7px;width:max-content;max-width:100%}
.lin-tag{font:.7rem ui-monospace,monospace;color:var(--muted);text-align:center;padding:0 .15rem;white-space:nowrap}
.lin-tag.hl{color:var(--fg);font-weight:700;text-decoration:underline}
.lin-cell{display:block;min-width:3.3rem;height:2.15rem;line-height:2.15rem;text-align:center;font:.92rem ui-monospace,monospace;color:var(--fg);background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:0 .25rem;margin:0}
input.lin-cell{width:4rem;line-height:normal}
input.lin-cell[aria-invalid=true]{background:#2b1513;border-color:var(--bwd)}
.lin-cell.hl{background:rgba(88,196,221,.17);border-color:var(--fwd);box-shadow:0 0 0 1px var(--fwd);font-weight:700}
.lin-cell.hl2{background:rgba(252,98,85,.15);border-color:var(--bwd);box-shadow:0 0 0 1px var(--bwd);font-weight:700}
.lin-cell.off{opacity:.55;border-style:dashed}
button.lin-cell{cursor:pointer}
button.lin-cell:hover{border-color:var(--fg)}
.lin-op{font-size:1.5rem;color:var(--muted);align-self:center;padding:0 .1rem}
.lin-paper{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:.5rem .9rem;margin:.6rem 0;overflow-x:auto}
.lin-paper .katex-display{margin:.3em 0}
@media(max-width:640px){.lin-paper{padding:.4rem .6rem}.lin-paper .katex{font-size:1em}}
.lin-sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}
@media (prefers-reduced-motion:reduce){.lin-w *{transition:none!important;animation:none!important}}
`;

export function useBaseStyle() { injectStyle(BASE_STYLE_ID, BASE_CSS); }

// A radio group drawn as pills (native radios: arrow keys, Tab, screen readers all work). Returns {root, set(value)}.
export function segmented({ label, name, options, value, onChange }) {
  const gid = uid('lin-grp');
  const inputs = options.map((o) => el('input', { type: 'radio', name, value: o.value, checked: o.value === value, onchange: () => onChange(o.value) }));
  const root = el('div', { class: 'lin-seg', role: 'radiogroup', 'aria-labelledby': gid },
    el('span', { class: 'lin-lbl', id: gid }, label),
    ...options.map((o, i) => el('label', {}, inputs[i], el('span', {}, o.label))));
  return { root, set(v) { inputs.forEach((inp) => { inp.checked = inp.value === v; }); } };
}

// A matrix as a grid with row/column tags. cell(i, j) returns the element of that entry.
export function matrixEl({ caption, rows, cols, rowTag, colTag, cell }) {
  const grid = el('div', { class: 'lin-grid', style: `grid-template-columns:${rowTag ? 'auto ' : ''}repeat(${cols}, auto)`, role: 'group', 'aria-label': String(caption).replace(/<[^>]+>/g, '') });
  const rowTags = [], colTags = [], cells = [];
  if (colTag) {
    if (rowTag) grid.append(el('span'));
    for (let j = 0; j < cols; j++) { const t = el('span', { class: 'lin-tag', 'aria-hidden': 'true' }, colTag(j)); colTags.push(t); grid.append(t); }
  }
  for (let i = 0; i < rows; i++) {
    if (rowTag) { const t = el('span', { class: 'lin-tag', 'aria-hidden': 'true' }, rowTag(i)); rowTags.push(t); grid.append(t); }
    cells.push([]);
    for (let j = 0; j < cols; j++) { const c = cell(i, j); cells[i].push(c); grid.append(c); }
  }
  const root = el('figure', { class: 'lin-mat' }, el('figcaption', { html: caption }), grid);
  return { root, grid, cells, rowTags, colTags };
}

// Highlight helpers: mark whole rows / columns of a matrix_el result.
export function clearMarks(m) {
  for (const t of [...m.rowTags, ...m.colTags]) t.classList.remove('hl');
  for (const r of m.cells) for (const c of r) c.classList.remove('hl', 'hl2');
}
export function markRow(m, i, cls = 'hl') { m.rowTags[i]?.classList.add('hl'); for (const c of m.cells[i] ?? []) c.classList.add(cls); }
export function markCell(m, i, j, cls = 'hl') { m.cells[i]?.[j]?.classList.add(cls); }
