// core/katex.js — the one place that talks to the vendored KaTeX (loaded as a classic <script>, so it is a global).
// In Node tests set globalThis.katex = require('.../vendor/katex/katex.min.js') first.
import { escapeHtml } from './dom.js';

export const katexAvailable = () => typeof globalThis.katex?.renderToString === 'function';

// LaTeX -> HTML. Never throws (a bad string shows as red source text); tests use renderStrict to catch bad strings.
export function math(latex, display = true) {
  if (!katexAvailable()) return `<code>${escapeHtml(latex)}</code>`;
  return globalThis.katex.renderToString(latex, { displayMode: display, throwOnError: false, strict: 'ignore' });
}

// Same, but throws on any LaTeX error. For tests and build-time checks.
export function renderStrict(latex, display = true) {
  return globalThis.katex.renderToString(latex, { displayMode: display, throwOnError: true });
}
