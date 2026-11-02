// widgets/next-token.js — "One sentence, many training examples" (lesson from-digits-to-chatgpt, brief §6 Part 5, decision 18).
// A made-up illustration, not a TorC run and not a real tokenizer: the widget cuts a sentence at spaces and punctuation, then
// steps through the training examples that next-token prediction makes from it. Example k reads the first k tokens, and its
// target is token k + 1 (counting from 1). The tokens after the target stay grey, because the model may not read them. The
// reader can type a sentence of their own, so any text gives examples without a person writing a label.
//
//   splitTokens(text) -> [string]                                  (pure, tested)
//   trainingExamples(tokens) -> [{ context: [string], target }]   (pure, tested)
//   tokenRoles(n, k) -> ['read' | 'target' | 'later']              (pure, tested)
//   mountNextToken(host) -> { destroy(), go(k), setText(s) }
import { el, escapeHtml } from '../core/dom.js';

export const START_SENTENCE = 'The cat sat on the mat.';
export const MAX_CHARS = 120;

// Words (letters, digits, apostrophes and inner hyphens) and single punctuation marks. A real tokenizer also cuts rare words
// into smaller pieces; the widget says so in its note.
export function splitTokens(text) {
  return String(text).match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu) ?? [];
}

export function trainingExamples(tokens) {
  return tokens.slice(1).map((target, i) => ({ context: tokens.slice(0, i + 1), target }));
}

// The role of each of n tokens in the example whose context has k tokens: the model reads the first k, token k is the target,
// and every later token is hidden as well.
export function tokenRoles(n, k) {
  return Array.from({ length: n }, (_, i) => (i < k ? 'read' : i === k ? 'target' : 'later'));
}

const CSS = `
.nt{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.nt-note{margin:0 0 .6rem;color:var(--muted);font-size:.9rem}
.nt label{display:block;font-size:.92rem;margin:0 0 .25rem}
.nt input{box-sizing:border-box;width:100%;max-width:32rem;font:inherit;color:var(--fg);background:var(--panel2);border:1px solid var(--border);border-radius:9px;padding:.4rem .6rem}
.nt-row{display:flex;flex-wrap:wrap;gap:6px;margin:.7rem 0 .5rem}
.nt-tok{display:inline-block;padding:.2rem .5rem;border-radius:7px;border:1px solid var(--border);font-size:.98rem;line-height:1.4}
.nt-tok.read{border-color:var(--fwd);background:rgba(88,196,221,.14)}
.nt-tok.target{border-color:var(--ok);background:rgba(131,193,103,.18);font-weight:600}
.nt-tok.later{color:var(--muted);border-style:dashed;opacity:.65}
.nt-keys{display:flex;flex-wrap:wrap;gap:4px 14px;margin:0 0 .5rem;font-size:.85rem;color:var(--muted)}
.nt-keys span::before{content:"";display:inline-block;width:.8em;height:.8em;border-radius:3px;margin-right:.35em;vertical-align:-.05em;border:1px solid}
.nt-keys .k-read::before{border-color:var(--fwd);background:rgba(88,196,221,.14)}
.nt-keys .k-target::before{border-color:var(--ok);background:rgba(131,193,103,.18)}
.nt-keys .k-later::before{border-color:var(--border);border-style:dashed}
.nt-btns{display:flex;flex-wrap:wrap;gap:6px}
.nt-read{margin:.5rem 0 0;font-size:.95rem;line-height:1.5}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('nt-styles')) return;
  document.head.append(el('style', { id: 'nt-styles' }, CSS));
}

const quote = (ts) => `“${escapeHtml(ts.join(' '))}”`;

export function mountNextToken(host) {
  injectStyles();
  let tokens = [], examples = [], at = 0;
  const input = el('input', { id: 'nt-text', type: 'text', value: START_SENTENCE, maxlength: String(MAX_CHARS), autocomplete: 'off', spellcheck: 'false' });
  const row = el('div', { class: 'nt-row', 'aria-hidden': 'true' });
  const read = el('div', { class: 'nt-read', 'aria-live': 'polite' });
  const back = el('button', { type: 'button', class: 'btn small', onclick: () => go(at - 1) }, 'Previous example');
  const next = el('button', { type: 'button', class: 'btn small primary', onclick: () => go(at + 1) }, 'Next example');
  const keys = el('div', { class: 'nt-keys', 'aria-hidden': 'true' },
    el('span', { class: 'k-read' }, 'the model reads'), el('span', { class: 'k-target' }, 'the target, hidden from the model'), el('span', { class: 'k-later' }, 'later tokens, hidden too'));
  host.replaceChildren(el('div', { class: 'nt' },
    el('p', { class: 'nt-note' }, 'A made-up illustration, not a TorC run. This widget cuts the text at spaces and punctuation. A real tokenizer also cuts rare words into smaller pieces.'),
    el('label', { for: 'nt-text' }, 'Your sentence'), input, row, keys,
    el('div', { class: 'nt-btns' }, back, next), read));

  function draw() {
    row.replaceChildren();
    back.disabled = at <= 0;
    next.disabled = at >= examples.length - 1;
    if (!examples.length) {
      read.innerHTML = '<p>Type a sentence with at least two words.</p>';
      return;
    }
    const ex = examples[at], k = ex.context.length;
    const roles = tokenRoles(tokens.length, k);
    tokens.forEach((t, i) => row.append(el('span', { class: `nt-tok ${roles[i]}` }, t)));
    read.innerHTML = `<p>Example ${at + 1} of ${examples.length}. The model reads ${quote(ex.context)}. Its target is ${quote([ex.target])}.</p>`;
  }
  function go(k) {
    at = Math.max(0, Math.min(examples.length - 1, k));
    draw();
  }
  function setText(s) {
    tokens = splitTokens(s);
    examples = trainingExamples(tokens);
    at = 0;
    draw();
  }
  input.addEventListener('input', () => setText(input.value));
  setText(input.value);
  return { destroy() {}, go, setText };
}
