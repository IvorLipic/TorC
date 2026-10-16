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
export function proseProblems(html) {
  const text = html.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, ' ')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, 'X').replace(/<span class="(num|rec)"[^>]*>[^<]*<\/span>/g, '9')
    .replace(/<\/(p|li|h1|h2|h3|summary|div|section|aside)>/g, '\n').replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/g, ' ');
  const problems = [];
  for (const block of text.split('\n').map((b) => b.replace(/\s+/g, ' ').trim()).filter(Boolean)) {
    if (block.includes(';')) problems.push(`semicolon: ${block}`);
    if (/\s[—–]\s|—/.test(block)) problems.push(`dash: ${block}`);
    if (/:\s*\S/.test(block)) problems.push(`colon inside a sentence: ${block}`);
    for (const sentence of block.split(/(?<=[.?!])\s+/)) {
      const n = sentence.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
      if (n > 25) problems.push(`${n} words: ${sentence}`);
    }
  }
  return problems;
}
