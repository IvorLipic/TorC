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
