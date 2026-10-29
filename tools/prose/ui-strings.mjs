// tools/prose/ui-strings.mjs — finds the visible UI strings in site/js/widgets/*.js and site/js/lesson-boot.js (brief D15), so the
// writing rules can be applied to them like to lesson prose. Pure: no DOM, no network. Used by tools/prose/extract.mjs (writes
// build/prose/ui.md for the ASD-STE100 linter) and by site/test/prose.test.mjs (applies the site rules with `uiStringProblems`).
//
// How a string is found. A small JavaScript tokenizer (comments, strings, template literals with nested ${…}, regex literals)
// yields every string and template literal with a "sink": the place it goes. A literal counts as visible prose when
//   - after the markup is removed it has at least three words (letters only, hyphens and apostrophes allowed), and
//   - it starts with a capital letter, OR it goes to a text sink: `.textContent/.innerHTML/.innerText/.title =`, an object key
//     such as `title:`, `ariaLabel:`, `caption:`, `label:`, `xLabel:`, `html:`, `text:`, `setAttribute('aria-label', …)`, or the
//     text argument of `el(tag, attrs, …)` / `svg(…)`, and
//   - it is not CSS, a selector, an SVG path, LaTeX, code, a class list, an import path or a format string.
// `${…}` becomes the digit 9 (like a bound number in lesson prose), `<code>…</code>` becomes the word code, other tags are removed
// (a closing block tag ends a line), and `aria-label`, `title`, `alt` and `placeholder` attributes inside markup are strings of
// their own. Whole files in SKIP_FILES are not read.
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

// Files that hold no UI prose. sgd-code.js is a listing of C++ source lines that the widget shows as code.
export const SKIP_FILES = new Set(['sgd-code.js']);

// Permanent allow-list: strings that break a site rule on purpose. Each entry is {file, includes, why}; a string is allowed when
// it is in `file` and contains `includes`. Keep it small.
export const UI_ALLOW = [
  { file: 'lesson-boot.js', includes: 'for example: python3 -m http.server', why: 'the colon introduces a command to type, which is the one case the rules allow' },
];

// TEMPORARY. Strings that still break the site rules on this branch (brief D15 plus the ones this linter found in the same
// widgets). They are excluded from the checks only so that the branch is green; the lead empties this list after merging the
// agent that fixes the strings. `from: 'brief'` is named in docs/pedagogy-review.md D15, `from: 'linter'` was found by this
// extraction. Entry = {file, includes, rule, from}; it matches a string of `file` that contains `includes`.
export const KNOWN_D15 = [
  { file: 'matmul-grid.js', includes: 'the last valid value is still in use', rule: 'semicolon', from: 'brief' },
  { file: 'matmul-grid.js', includes: 'The two columns of y are identical: the neurons are twins', rule: 'colon', from: 'brief' },
  { file: 'contour2d.js', includes: 'Positions on the trail (before each update; ', rule: 'semicolon', from: 'brief' },
  { file: 'contour2d.js', includes: 'The view zoomed out to keep the whole trail in sight; ', rule: 'semicolon', from: 'brief' },
  { file: 'lesson-boot.js', includes: '✓ Done — click to undo', rule: 'dash', from: 'brief' },
  { file: 'inputs-popover.js', includes: 'recomputed in your browser', rule: 'provenance', from: 'brief' },
  { file: 'inputs-popover.js', includes: '— type new numbers; everything recomputes.', rule: 'semicolon, dash, provenance', from: 'linter' },
  { file: 'inputs-popover.js', includes: 'your values (recomputed in browser)', rule: 'provenance (lines 31 and 52)', from: 'linter' },
  { file: 'digit-widgets.js', includes: '(brighter is larger; each map has its own scale)', rule: 'semicolon', from: 'linter' },
  { file: 'generalization-widgets.js', includes: 'at each recorded step', rule: 'provenance', from: 'linter' },
  { file: 'math-window.js', includes: 'was created with code: TorC will compute', rule: 'colon', from: 'linter' },
  { file: 'math-window.js', includes: 'was created with code: it is data', rule: 'colon', from: 'linter' },
];

const SINK_KEYS = new Set(['textContent', 'innerHTML', 'innerText', 'title', 'html', 'text', 'label', 'ariaLabel', 'aria-label',
  'aria-description', 'caption', 'alt', 'placeholder', 'xLabel', 'yLabel', 'tooltip']);
const ATTR_SINKS = new Set(['aria-label', 'aria-description', 'title', 'alt', 'placeholder']);
const EL_CALLS = new Set(['el', 'svg', 'h', 'text']);
const KEYWORDS_BEFORE_REGEX = new Set(['return', 'typeof', 'case', 'in', 'of', 'delete', 'void', 'throw', 'new', 'else', 'do', 'yield', 'await']);
const NOT_CALLS = new Set(['if', 'for', 'while', 'switch', 'catch', 'return', 'typeof', 'function']);
const PLACEHOLDER = '';

const decodeEscapes = (s) => s.replace(/\\(u\{([0-9a-fA-F]+)\}|u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2})|[\s\S])/g, (m, a, b, c, d) => {
  if (b) return String.fromCodePoint(parseInt(b, 16));
  if (c) return String.fromCharCode(parseInt(c, 16));
  if (d) return String.fromCharCode(parseInt(d, 16));
  return ({ n: '\n', t: '\t', r: '\r', b: '', f: '', v: '', 0: '\0', '\n': '' })[a] ?? a;
});

// Every string and template literal of a JavaScript source: [{line, text, sink}]. `text` has ${…} replaced by PLACEHOLDER.
export function literalsOf(src) {
  const out = [], n = src.length;
  let i = 0, line = 1, prev = '', recent = [];
  const frames = [];                                  // {t:'(', name, arg, first} | {t:'{', key} | {t:'['} | {t:'T', tpl}; all have `tern`
  const push = (tok) => { prev = typeof tok === 'string' ? tok : '"'; recent.push(tok); if (recent.length > 40) recent.shift(); };
  const top = () => frames[frames.length - 1];

  function sinkNow() {
    const f = top();
    if (f?.t === '{' && f.key && SINK_KEYS.has(f.key)) return f.key;
    if (f?.t === '(' && f.name === 'setAttribute' && f.arg === 1 && ATTR_SINKS.has(f.first)) return f.first;
    if (f?.t === '(' && EL_CALLS.has(f.name) && f.arg >= 1) return 'el-text';
    for (let k = recent.length - 1; k >= 1; k--) {      // `x.textContent = cond ? 'a' : 'b'` — stop at the statement start
      const tok = recent[k];
      if (tok === ';' || tok === '{' || tok === '}') break;
      if (tok === '=' && ['textContent', 'innerHTML', 'innerText', 'title', 'ariaLabel', 'placeholder', 'alt'].includes(recent[k - 1])) return recent[k - 1];
    }
    return null;
  }

  function readQuasi(t) {                              // reads the next piece of template `t`; true when the template ended
    let raw = '';
    while (i < n) {
      const c = src[i];
      if (c === '\\') { raw += c + (src[i + 1] ?? ''); i += 2; continue; }
      if (c === '`') { i++; t.parts.push(decodeEscapes(raw)); return true; }
      if (c === '$' && src[i + 1] === '{') { i += 2; t.parts.push(decodeEscapes(raw)); frames.push({ t: 'T', tpl: t, tern: 0 }); push('${'); return false; }
      if (c === '\n') line++;
      raw += c; i++;
    }
    t.parts.push(decodeEscapes(raw));
    return true;
  }
  const finish = (t) => { out.push({ line: t.line, text: t.parts.join(PLACEHOLDER), sink: t.sink }); push({ lit: true }); };

  while (i < n) {
    const c = src[i];
    if (c === '\n') { line++; i++; continue; }
    if (/\s/.test(c)) { i++; continue; }
    if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') { const e = src.indexOf('*/', i + 2); const stop = e < 0 ? n : e + 2; line += (src.slice(i, stop).match(/\n/g) || []).length; i = stop; continue; }
    if (c === '"' || c === "'") {
      let j = i + 1, raw = '';
      while (j < n && src[j] !== c && src[j] !== '\n') { if (src[j] === '\\') { raw += src[j] + (src[j + 1] ?? ''); j += 2; } else raw += src[j++]; }
      const text = decodeEscapes(raw);
      out.push({ line, text, sink: sinkNow() });
      const f = top();
      if (f?.t === '(' && f.arg === 0 && f.first === undefined) f.first = text;
      push({ lit: true, value: text }); i = j + 1; continue;
    }
    if (c === '`') {
      i++;
      const t = { parts: [], line, sink: sinkNow() };
      if (!readQuasi(t)) { /* inside ${…}: the tokens go on below, and the closing brace resumes the template */ } else finish(t);
      continue;
    }
    if (c === '{') { frames.push({ t: '{', key: null, tern: 0 }); push('{'); i++; continue; }
    if (c === '[') { frames.push({ t: '[', tern: 0 }); push('['); i++; continue; }
    if (c === '(') {
      const name = typeof prev === 'string' && /^[A-Za-z_$][\w$]*$/.test(prev) && !NOT_CALLS.has(prev) ? prev : null;
      frames.push({ t: '(', name, arg: 0, tern: 0 }); push('('); i++; continue;
    }
    if (c === '}' || c === ']' || c === ')') {
      const f = top();
      if (f?.t === 'T' && c === '}') {
        frames.pop(); i++;
        const t = f.tpl;
        if (readQuasi(t)) finish(t);
        continue;
      }
      if (f) frames.pop();
      push(c); i++; continue;
    }
    if (c === ',') { const f = top(); if (f?.t === '(') f.arg++; if (f?.t === '{') f.key = null; push(','); i++; continue; }
    if (c === '?') {
      if (src[i + 1] === '?') { push('??'); i += 2; continue; }                          // nullish coalescing
      if (src[i + 1] === '.' && !/\d/.test(src[i + 2] ?? '')) { push('?.'); i += 2; continue; }   // optional chaining
      const f = top(); if (f) f.tern++;
      push('?'); i++; continue;
    }
    if (c === ':') {
      const f = top();
      if (f && f.tern > 0) f.tern--;
      else if (f?.t === '{') { const k = recent[recent.length - 1]; f.key = typeof k === 'string' ? k : k?.value ?? null; }
      push(':'); i++; continue;
    }
    if (c === '/') {
      const last = recent[recent.length - 1];
      const regexOk = last === undefined || (typeof last === 'string' && (/^[(,=:[!&|?{};+\-*%<>~^]$/.test(last) || KEYWORDS_BEFORE_REGEX.has(last)));
      if (regexOk) {
        let j = i + 1, cls = false;
        while (j < n && src[j] !== '\n') { if (src[j] === '\\') j += 2; else if (src[j] === '[') { cls = true; j++; } else if (src[j] === ']') { cls = false; j++; } else if (src[j] === '/' && !cls) break; else j++; }
        j++; while (/[a-z]/i.test(src[j] ?? '')) j++;
        i = j; push({ lit: true, regex: true }); continue;
      }
      push('/'); i++; continue;
    }
    if (/[A-Za-z_$]/.test(c)) { let j = i; while (/[\w$]/.test(src[j] ?? '')) j++; push(src.slice(i, j)); i = j; continue; }
    if (/[0-9]/.test(c)) { let j = i; while (/[\w.]/.test(src[j] ?? '')) j++; push('0'); i = j; continue; }
    push(c); i++;
  }
  return out;
}

const ENTITIES = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'", '&nbsp;': ' ' };
const LOOKS_LIKE_HTML = /<\/?[a-z][^>]*>/i;

// Visible text of a literal: [{text, sink}] — the main text, plus one entry per aria-label/title/alt/placeholder attribute.
export function visibleTexts(raw, sink) {
  let text = raw;
  const found = [];
  if (LOOKS_LIKE_HTML.test(raw)) {
    for (const m of raw.matchAll(/\b(?:aria-label|aria-description|title|alt|placeholder)="([^"]*)"/g)) found.push({ text: m[1], sink: 'attribute' });
    text = raw.replace(/<style[\s\S]*?<\/style>/gi, ' ').replace(/<script[\s\S]*?<\/script>/gi, ' ')
      .replace(/<code[^>]*>[\s\S]*?<\/code>/gi, 'code').replace(/<\/(p|li|h[1-6]|div|section|aside|summary|figcaption|caption|label|tr)>/gi, '\n')
      .replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, ' ');
  }
  text = text.replace(/&(amp|lt|gt|quot|#39|nbsp);/g, (m) => ENTITIES[m]);
  found.unshift({ text, sink });
  return found.map((f) => ({
    sink: f.sink,
    text: f.text.split('\n').map((l) => l.replace(/\s+/g, ' ').replace(/ ([.,?!])/g, '$1').trim()).filter(Boolean).join('\n').replaceAll(PLACEHOLDER, '9'),
  })).filter((f) => f.text);
}

const WORD = /^[(["'“‘]*\p{L}[\p{L}'’-]*[)\]"'”’.,:;!?]*$/u;
export const wordCount = (text) => text.split(/\s+/).filter((t) => WORD.test(t)).length;

// Not prose: CSS, selectors, SVG paths, LaTeX, code, class lists, format strings.
function isNotProse(text) {
  const t = text.replaceAll('9', ' ').trim();
  if (/\\[A-Za-z]{2,}/.test(text)) return true;                                       // LaTeX
  if (/^[MmLlHhVvCcSsQqTtAaZz\d\s,.eE+-]+$/.test(text)) return true;                  // SVG path data
  if (/[a-z-]+:\s?[^\s;]+;/.test(text) || /^[a-z]+(-[a-z]+)+:\s?\S/.test(text)) return true;   // CSS declarations
  if (/\bvar\(--|\brgba?\(|\brepeat\(|\bcalc\(|[{}]\s*[.#\w-]+\s*[{:]/.test(text)) return true;
  if (/\{[^}]*:[^}]*\}/.test(text)) return true;                                       // CSS rule or object literal
  if (/^[.#[]/.test(text) || /\[data-[\w-]+/.test(text)) return true;                   // selector
  if (/^\s*(const|let|var|function|class|import|export|return|if|for|while|void|int|float|double|auto|torc::|std::|#include)\b/.test(text)) return true;
  if (/[;{}]\s*$/.test(text) || /=>|===|!==|&&|\|\||->|::/.test(text)) return true;   // code
  if (/%[ds]|\{\d\}/.test(text)) return true;                                          // format string
  return t.length === 0;
}

// Does this literal read as visible prose? `sink` is where it goes (see the header comment), or null.
export function isProseString(text, sink) {
  if (wordCount(text) < 3 || isNotProse(text)) return false;
  return /^[("'“‘]*\p{Lu}/u.test(text) || sink !== null;
}

// All visible prose strings of one source file: [{file, line, text}].
export function extractStringsFromSource(src, file = 'source.js') {
  const out = [];
  for (const lit of literalsOf(src)) {
    for (const v of visibleTexts(lit.text, lit.sink)) {
      if (isProseString(v.text, v.sink)) out.push({ file, line: lit.line, text: v.text });
    }
  }
  return out;
}

export const matches = (entry, rec) => entry.file === rec.file && rec.text.includes(entry.includes);
export const isKnownD15 = (rec) => KNOWN_D15.some((e) => matches(e, rec));
export const isAllowed = (rec) => UI_ALLOW.some((e) => matches(e, rec));

// The files whose strings are linted (relative to the site directory).
export function uiFiles(siteDir) {
  const widgets = readdirSync(path.join(siteDir, 'js/widgets')).filter((f) => f.endsWith('.js')).sort().map((f) => `js/widgets/${f}`);
  return [...widgets, 'js/lesson-boot.js'].filter((f) => !SKIP_FILES.has(path.basename(f)));
}

// Every visible prose string of the site's widgets and lesson-boot.js: [{file, line, text, known, allowed}].
export function extractUiStrings(siteDir) {
  return uiFiles(siteDir).flatMap((rel) => extractStringsFromSource(readFileSync(path.join(siteDir, rel), 'utf8'), path.basename(rel)))
    .map((r) => ({ ...r, known: isKnownD15(r), allowed: isAllowed(r) }));
}
