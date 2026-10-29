// Shared test helpers. Not a test file itself.
import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';

export const SITE = new URL('../', import.meta.url);
export const katex = createRequire(import.meta.url)('../vendor/katex/katex.min.js');
globalThis.katex = katex;                       // core/katex.js reads the global, exactly as in the browser

export const readJson = (rel) => JSON.parse(readFileSync(new URL(rel, SITE), 'utf8'));
export const dataFiles = () => readdirSync(new URL('data/', SITE)).filter((f) => f.endsWith('.json')).sort();
export const traceFiles = () => Object.fromEntries(dataFiles().map((f) => [f, readJson('data/' + f)]));
// [{file, scene}] for every scene of every trace file (the loader reads scenes[] at top level and ignores other keys)
export const allScenes = () => Object.entries(traceFiles()).flatMap(([file, j]) => (Array.isArray(j.scenes) ? j.scenes.map((scene) => ({ file, scene })) : []));
export const course = () => readJson('course.json');
export const close = (a, b, tol = 1e-5) => a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) <= tol * (1 + Math.abs(b[i])));
export const renderStrictAll = (latexList) => latexList.forEach((s) => katex.renderToString(s, { throwOnError: true }));

// small deterministic generator for "random tensors"
export function rng(seed = 12345) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

// Size of a rendered lesson against the style budget of docs/authoring.md: words of prose (code blocks, styles and formulas
// and inline code excluded; without KaTeX every formula renders as <code>, so formulas are excluded too), numbers bound to the
// trace outside code blocks, widgets, and predict/check questions.
export function lessonStats(lesson, D, renderLesson, boundNums) {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let parts;
  try { parts = renderLesson(lesson, D); } finally { globalThis.katex = saved; }
  const html = parts.map((p) => p.html).join('\n');
  const prose = html.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code>[\s\S]*?<\/code>/g, ' ').replace(/<[^>]+>/g, ' ');
  return {
    words: prose.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length,
    numbers: boundNums(html.replace(/<pre[\s\S]*?<\/pre>/g, '')).length,
    widgets: lesson.sections.filter((s) => s.kind === 'widget').length,
    questions: lesson.sections.filter((s) => s.kind === 'predict' || s.kind === 'check').length,
  };
}

// Words of one rendered section, the way the style budget counts them: no headings, styles, code blocks, inline code or formulas
// (render without KaTeX so formulas become <code>).
export function sectionWords(html) {
  return html.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code>[\s\S]*?<\/code>/g, ' ')
    .replace(/<h[1-3][^>]*>[\s\S]*?<\/h[1-3]>/g, ' ').replace(/<[^>]+>/g, ' ').split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
}

// The writing rules of docs/authoring.md that a program can check, on visible prose (code, formulas and numbers removed):
// no semicolons, no dashes used as punctuation, a colon only where it introduces what follows the block (a formula, code or a
// list), and at most 25 words in a sentence. Returns a list of problems with the sentence they were found in.
//   - Allowed (brief R8): the minus sign U+2212, a spaced en dash between maths tokens ("5 – 3", "x – y"), and one of the label
//     colons in LABEL_COLONS at the start of a block ("Task:", "Reveal:").
//   - `{ tics: true }` also fails the banned teacher-voice tics of brief section 8 (BANNED_TICS). The caller turns it on for
//     lessons with `template: 3` (see site/test/prose.test.mjs) and for widget and UI strings.
export function proseProblems(html, { tics = false } = {}) {
  const text = html.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, ' ')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, 'X').replace(/<span class="(num|rec)"[^>]*>[^<]*<\/span>/g, '9')
    .replace(/<\/(p|li|h1|h2|h3|summary|div|section|aside)>/g, '\n').replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/g, ' ');
  const problems = [];
  for (const block of text.split('\n').map((b) => b.replace(/\s+/g, ' ').trim()).filter(Boolean)) {
    if (block.includes(';')) problems.push(`semicolon: ${block}`);
    if (hasPunctuationDash(block)) problems.push(`dash: ${block}`);
    if (/:\s*\S/.test(withoutLabelColon(block))) problems.push(`colon inside a sentence: ${block}`);
    for (const sentence of block.split(/(?<=[.?!])\s+/)) {
      const n = sentence.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
      if (n > 25) problems.push(`${n} words: ${sentence}`);
    }
    if (tics) for (const t of ticsIn(block)) problems.push(`banned tic "${t}": ${block}`);
  }
  return problems;
}

// ---- helpers of proseProblems (brief section 8; docs/authoring.md "Writing rules") ----------------------------------------------

// Label colons that may start a block: the guided-task and reveal labels of the v3 template.
export const LABEL_COLONS = ['Task:', 'Task 1:', 'Task 2:', 'Task 3:', 'What you will see:', 'Reveal:', 'Where it breaks:', 'Next question:',
  'In TorC:', 'Analogy:', 'Your turn:'];

// The banned tics, as [name, pattern]. A match fails a lesson with `template: 3` and every widget or UI string. Flag-only items
// of the brief (a hedging "just", three-item lists for rhythm, a punchline that restates the paragraph) are NOT here: a reviewer
// reads for those (brief Appendix A).
export const BANNED_TICS = [
  ["let's", /\blet[’']s\b/i],
  ["here's the thing", /\bhere[’']s the thing\b/i],
  ['it turns out', /\bit turns out\b/i],
  ['simply', /\bsimply\b/i],
  ['magic', /\bmagic(?:al(?:ly)?)?\b/i],
  ['crucially', /\bcrucially\b/i],
  ['notice that', /\bnotice that\b/i],
  ['the key insight', /\bthe key insight\b/i],
  ['under the hood', /\bunder the hood\b/i],
  ['in other words', /\bin other words\b/i],
  ['think of it as', /\bthink of it as\b/i],
  ['imagine', /\bimagine\b/i],
  ['essentially', /\bessentially\b/i],
  ['at its core', /\bat its core\b/i],
  ['the heart of', /\bthe heart of\b/i],
  ['powerful', /\bpowerful(?:ly)?\b/i],
  ['elegant', /\belegan(?:t|tly)\b/i],
  ['beautiful', /\bbeautiful(?:ly)?\b/i],
  ['not just X but Y', /\bnot just\b[^.?!]*?\bbut\b/i],
  ["that's it", /\bthat[’']s it\b/i],
  ['this is where X comes in', /\bthis is where\b[^.?!]*?\bcomes? in\b/i],
  ['this lesson shows (opener)', /(?:^|[.?!]\s+)this lesson shows\b/i],
];

// Names of the banned tics found in a block of prose (one entry per match, repeated if it occurs twice).
export function ticsIn(block) {
  const found = [];
  for (const [name, re] of BANNED_TICS) for (const _ of block.matchAll(new RegExp(re.source, re.flags + 'g'))) found.push(name);
  return found;
}

// `block` without a leading label colon of LABEL_COLONS ("Task: press Step" -> " press Step"); other blocks come back unchanged.
export function withoutLabelColon(block) {
  const label = LABEL_COLONS.find((l) => block.startsWith(l) && (block.length === l.length || /\s/.test(block[l.length])));
  return label ? block.slice(label.length) : block;
}

// A word on either side of a dash that makes it a minus or a range in maths: a number, the code/number placeholders X and 9, or a
// single letter, optionally in brackets.
const isMathToken = (t) => /^[(\[]*(?:\d[\d.,]*|X|[A-Za-z])[)\]]*[.,;:!?]*$/.test(t);

// Dashes as punctuation: any em dash, and a spaced en dash unless it sits between two maths tokens. The minus sign U+2212 and
// the hyphen are never dashes in this sense.
export function hasPunctuationDash(block) {
  if (block.includes('—')) return true;
  for (const m of block.matchAll(/(\S*)\s–\s(\S*)/g)) if (!(isMathToken(m[1]) && isMathToken(m[2]))) return true;
  return false;
}

// ---- visible UI strings of widgets and lesson-boot.js (brief D15) ----------------------------------------------------------------

// The provenance wording that lesson pages may not use (the same pattern as site/test/lessons.test.mjs).
export const PROVENANCE_WORDS = /recorded|recomputed|printed by|in your browser|show as table/i;

// The site rules for one extracted UI string (`text` as tools/prose/ui-strings.mjs gives it: bound values are the digit 9, code is
// the word code, one block per line). Same rules as proseProblems, with three differences for short interface text:
//   - a colon is fine at the end of the text, after a short label ("dashed line: training loss"; at most 3 words, or 4 when the
//     label starts in lower case; no punctuation before the colon; so legend entries and "Label: value" readouts pass), or when
//     a bound value follows it in the same sentence ("Step 9: w = 9, L = 9.");
//   - provenance words fail ("recorded", "recomputed", "printed by", "in your browser", "show as table");
//   - the banned tics always fail.
export function uiStringProblems(text) {
  const problems = [];
  for (const block of text.split('\n').map((b) => b.replace(/\s+/g, ' ').trim()).filter(Boolean)) {
    if (block.includes(';')) problems.push(`semicolon: ${block}`);
    if (hasPunctuationDash(block)) problems.push(`dash: ${block}`);
    for (const sentence of block.split(/(?<=[.?!])\s+/)) {
      if (!colonsOk(withoutLabelColon(sentence))) problems.push(`colon inside a sentence: ${block}`);
      const n = sentence.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
      if (n > 25) problems.push(`${n} words: ${sentence}`);
    }
    if (PROVENANCE_WORDS.test(block)) problems.push(`provenance wording: ${block}`);
    for (const t of ticsIn(block)) problems.push(`banned tic "${t}": ${block}`);
  }
  return problems;
}

// May every colon of this sentence stay? (see uiStringProblems)
function colonsOk(sentence) {
  for (let k = sentence.indexOf(':'); k !== -1; k = sentence.indexOf(':', k + 1)) {
    const before = sentence.slice(0, k), after = sentence.slice(k + 1).trim();
    if (after === '' || after.startsWith('//') || /\d$/.test(before) && /^\d/.test(after) && sentence[k + 1] !== ' ') continue;
    const words = before.split(/\s+/).length;
    const label = words <= 4 && !/[.,;?!]/.test(before) && (words <= 3 || /^[^\p{Lu}]/u.test(before));
    const readout = /9/.test(after) && !/[.?!]\s+\S/.test(after);
    if (!label && !readout) return false;
  }
  return true;
}
