// widgets/next-token.js — "One sentence, many samples" (lesson from-digits-to-chatgpt, brief §6 Part 5, decision 18).
// A made-up illustration, not a TorC run and not a real tokenizer: the widget cuts a sentence at spaces and punctuation, then
// steps through the samples that next-token prediction makes from it. Sample k reads the first k tokens, and its target is
// token k + 1 (counting from 1). The target and the tokens after it are hidden from the model. The reader can type a sentence
// of their own, so any text gives samples without a person writing a target. Going one sample on shows the target joining the
// tokens that the model reads, as a picked token joins the text when the model writes.
//
//   splitTokens(text) -> [string]                                  (pure, tested)
//   samplesOf(tokens) -> [{ context: [string], target }]          (pure, tested)
//   tokenRoles(n, k) -> ['read' | 'target' | 'later']              (pure, tested)
//   mountNextToken(host, {announceDelay}) -> { destroy(), go(k), step(d), setText(s) }
import { el, escapeHtml } from '../core/dom.js';

export const START_SENTENCE = 'The cat sat on the mat.';
export const MAX_CHARS = 120;

// Words (letters, digits, apostrophes and inner hyphens) and single punctuation marks. A real tokenizer also cuts rare words
// into smaller pieces; the widget says so in its note.
export function splitTokens(text) {
  return String(text).match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]/gu) ?? [];
}

export function samplesOf(tokens) {
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
.nt .btn[aria-disabled=true]{opacity:.4;cursor:default}
.nt-sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('nt-styles')) return;
  document.head.append(el('style', { id: 'nt-styles' }, CSS));
}

const quote = (ts) => `“${escapeHtml(ts.join(' '))}”`;

export function mountNextToken(host, { announceDelay = 700 } = {}) {
  injectStyles();
  let tokens = [], samples = [], at = 0, timer = null;
  const input = el('input', { id: 'nt-text', type: 'text', value: START_SENTENCE, maxlength: String(MAX_CHARS), autocomplete: 'off', spellcheck: 'false' });
  const row = el('div', { class: 'nt-row', 'aria-hidden': 'true' });
  const read = el('div', { class: 'nt-read' });
  // screen readers hear a change after a step, or after the typing stops, not at every keystroke
  const live = el('p', { class: 'nt-sr', 'aria-live': 'polite' });
  // aria-disabled, not disabled, so that a button keeps the focus at the first and the last sample
  const back = el('button', { type: 'button', class: 'btn small', onclick: () => step(-1) }, 'Previous sample');
  const next = el('button', { type: 'button', class: 'btn small primary', onclick: () => step(1) }, 'Next sample');
  const keys = el('div', { class: 'nt-keys', 'aria-hidden': 'true' },
    el('span', { class: 'k-read' }, 'the model reads'), el('span', { class: 'k-target' }, 'the target, hidden from the model'), el('span', { class: 'k-later' }, 'later tokens, hidden too'));
  host.replaceChildren(el('div', { class: 'nt' },
    el('p', { class: 'nt-note' }, 'A made-up illustration, not a TorC run. This widget cuts the text at spaces and punctuation. A real tokenizer also cuts rare words into smaller pieces.'),
    el('label', { for: 'nt-text' }, 'Your sentence'), input, row, keys,
    el('div', { class: 'nt-btns' }, back, next), read, live));

  const message = () => {
    if (!samples.length) return 'Type a sentence with at least two words.';
    const x = samples[at];
    return `Sample ${at + 1} of ${samples.length}. The model reads ${quote(x.context)}. Its target is ${quote([x.target])}.`;
  };
  function draw() {
    row.replaceChildren();
    back.setAttribute('aria-disabled', String(at <= 0));
    next.setAttribute('aria-disabled', String(at >= samples.length - 1));
    read.innerHTML = `<p>${message()}</p>`;
    if (!samples.length) return;
    const roles = tokenRoles(tokens.length, samples[at].context.length);
    tokens.forEach((t, i) => row.append(el('span', { class: `nt-tok ${roles[i]}` }, t)));
  }
  function announce() { clearTimeout(timer); timer = null; live.innerHTML = message(); }
  function go(k) {
    at = Math.max(0, Math.min(samples.length - 1, k));
    draw();
  }
  function step(d) {
    const k = Math.max(0, Math.min(samples.length - 1, at + d));
    if (k === at) return;
    go(k);
    announce();
  }
  function setText(s) {
    tokens = splitTokens(s);
    samples = samplesOf(tokens);
    at = 0;
    draw();
  }
  input.addEventListener('input', () => {
    setText(input.value);
    clearTimeout(timer);
    timer = setTimeout(announce, announceDelay);
  });
  setText(input.value);
  return { destroy() { clearTimeout(timer); }, go, step, setText };
}
