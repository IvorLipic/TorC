// core/sections.js — turns a lesson module's `sections` into HTML. Pure (no DOM), so Node renders every lesson
// against the real trace data in the tests, and the browser uses the very same strings.
//
// section kinds
//   tldr    { kind:'tldr',   items: (D, T) => ['…', …] }                         the 3-4 sentence summary under the hero
//   text    { kind:'text',   html:  (D, T) => '<section>…</section>' }            markup is used as given
//   gotcha  { kind:'gotcha', title, html: (D, T) => … }                           a "Why did my X fail?" box
//   predict { kind:'predict', id, title?, prompt: (D, T) => …, reveal: (D, T) => … }       think first, then open the answer
//   check   { kind:'check',  id, title?, prompt: (D, T) => …, reveal: (D, T) => … }        like predict, plus an "I got it" mark (stored)
//   widget  { kind:'widget', id, className?, label?, mount(host, D, ctx) }         interactive part; mount() runs in the browser
// D = loaded trace data (see core/trace.js); T('scene.path') = a number from the trace, shown as <span class="num" data-trace>.
import { makeT, stemOf } from './trace.js';
import { escapeHtml } from './dom.js';

export const SECTION_KINDS = ['tldr', 'text', 'gotcha', 'predict', 'check', 'widget'];
const call = (f, D, T) => (typeof f === 'function' ? f(D, T) : String(f ?? ''));

export function sectionHtml(section, D, T = makeT(D)) {
  switch (section.kind) {
    case 'text':
      return call(section.html, D, T);
    case 'tldr': {
      const items = typeof section.items === 'function' ? section.items(D, T) : section.items;
      return `<aside class="tldr" aria-labelledby="tldr-h"><h2 id="tldr-h">Summary</h2><ul>${items.map((i) => `<li>${i}</li>`).join('')}</ul></aside>`;
    }
    case 'gotcha':
      return `<aside class="gotcha"><h3>${escapeHtml(section.title ?? 'Why did it fail?')}</h3>${call(section.html, D, T)}</aside>`;
    case 'predict':
    case 'check': {
      const check = section.kind === 'check';
      return `<section class="${section.kind}" data-section="${escapeHtml(section.id ?? '')}">`
        + `<h3>${escapeHtml(section.title ?? (check ? 'Check yourself' : 'Predict, then reveal'))}</h3><div class="prompt">${call(section.prompt, D, T)}</div>`
        + `<details><summary>Show the answer</summary><div class="answer">${call(section.reveal, D, T)}</div></details>`
        + (check ? `<button type="button" class="btn small" data-check="${escapeHtml(section.id)}" aria-pressed="false">I got it</button>` : '')
        + '</section>';
    }
    case 'widget':
      return `<section id="${escapeHtml(section.id)}" class="${escapeHtml(section.className ?? 'widget')}" aria-label="${escapeHtml(section.label ?? section.id)}" data-widget="${escapeHtml(section.id)}"></section>`;
    default:
      throw new Error(`unknown section kind "${section.kind}"`);
  }
}

// Structural checks every lesson module must satisfy.
export function validateLesson(lesson) {
  const errs = [];
  if (!lesson || typeof lesson !== 'object') return ['lesson module has no default export'];
  if (typeof lesson.id !== 'string') errs.push('lesson.id missing');
  if (typeof lesson.title !== 'string') errs.push('lesson.title missing');
  if (!Array.isArray(lesson.trace) || !lesson.trace.every((f) => typeof f === 'string')) errs.push('lesson.trace must be an array of file names');
  if (!Array.isArray(lesson.sections) || !lesson.sections.length) errs.push('lesson.sections must be a non-empty array');
  const ids = new Set();
  for (const [i, s] of (lesson.sections ?? []).entries()) {
    if (!SECTION_KINDS.includes(s?.kind)) { errs.push(`section ${i}: unknown kind "${s?.kind}"`); continue; }
    if (s.kind === 'text' && typeof s.html !== 'function') errs.push(`section ${i}: text sections need html: (D, T) => string`);
    if (s.kind === 'tldr' && !(typeof s.items === 'function' || Array.isArray(s.items))) errs.push(`section ${i}: tldr sections need items: (D, T) => [html, …]`);
    if (s.kind === 'widget' && (typeof s.mount !== 'function' || typeof s.id !== 'string')) errs.push(`section ${i}: widget sections need id and mount(host, D, ctx)`);
    if (['predict', 'check'].includes(s.kind) && typeof s.id !== 'string') errs.push(`section ${i}: ${s.kind} sections need an id`);
    if (s.id !== undefined) { if (ids.has(s.id)) errs.push(`duplicate section id "${s.id}"`); ids.add(s.id); }
  }
  return errs;
}

// Every <span class="num"> must carry data-trace (a number that came from T()). Returns the offending spans.
export function findUnboundNums(html) {
  const bad = [];
  for (const m of html.matchAll(/<[a-z][^>]*\bclass=["'][^"']*\bnum\b[^"']*["'][^>]*>/gi)) if (!/\sdata-trace=["'][^"']+["']/.test(m[0])) bad.push(m[0]);
  return bad;
}
// Bound numerals as [{path, shown}] so a test can re-resolve each one against the data.
export function boundNums(html) {
  return [...html.matchAll(/<span class="num" data-trace="([^"]*)">([^<]*)<\/span>/g)].map((m) => ({ path: m[1].replace(/&quot;/g, '"').replace(/&amp;/g, '&'), shown: m[2] }));
}

// Render a whole lesson to [{section, html}] (widgets come out as empty hosts).
export function renderLesson(lesson, D) {
  const T = makeT(D);
  return lesson.sections.map((section) => ({ section, html: sectionHtml(section, D, T) }));
}

// The one-line footer every lesson gets unless it brings its own: which TorC run the numbers come from, and the source.
// Where the numbers come from is explained once, on the home page; lessons do not repeat it.
export function defaultFooter(lesson, D) {
  const ref = lesson.trace.map((f) => D.files[stemOf(f)]?.library?.ref).find(Boolean);
  const repo = 'https://github.com/IvorLipic/TorC';
  const shown = ref && /^[0-9a-f]{40}$/.test(ref) ? ref.slice(0, 7) : ref;   // a pinned commit SHA is shown short, linked in full
  return `<p>Numbers: TorC${ref ? ` <code>${escapeHtml(shown)}</code>` : ''} · <a href="${ref ? `${repo}/tree/${encodeURIComponent(ref)}` : repo}">source</a></p>`;
}
