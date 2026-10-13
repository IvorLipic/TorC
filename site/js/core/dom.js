// core/dom.js — tiny DOM helpers shared by every page and widget.
const SVGNS = 'http://www.w3.org/2000/svg';

export const $ = (sel, root = document) => root.querySelector(sel);

// el('div', {class:'x', html:'<b>..</b>', onclick: fn, 'aria-label': 'y'}, child, 'text', [more])
export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}

export function svg(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null && v !== false) e.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}

export const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// Re-render without losing keyboard focus: elements opt in with a data-key attribute.
export function keepFocus(fn, root = document) {
  const a = document.activeElement;
  const key = a && a.dataset ? a.dataset.key : null;
  fn();
  if (key) {
    const t = root.querySelector(`[data-key="${CSS.escape(key)}"]`);
    if (t && !t.disabled) t.focus({ preventScroll: true });
  }
}
