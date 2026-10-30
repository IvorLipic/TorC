// core/sections.js — turns a lesson module's `sections` into HTML. Pure (no DOM), so Node renders every lesson
// against the real trace data in the tests, and the browser uses the very same strings. Behaviour (quiz feedback, the explain
// good answer, the glossary popover, the widgets) is wired afterwards by lesson-boot.js (core/quiz.js, core/popover.js).
//
// section kinds of a v3 lesson (`template: 3`, docs/v3-implementation.md "Machinery API", R7, R10)
//   text      { kind:'text', html: (D, T) => '<section>…</section>', setup? }   markup is used as given; setup: true marks the
//                                                                             scenario text that may come before the first predict quiz
//   questions { kind:'questions', items: (D, T) => [html] }                   "Questions this lesson answers" (1-3), with the loop
//                                                                             diagram when the lesson has `loop`. Section 1
//   quiz      { kind:'quiz', id, purpose, type, prompt, reveal, title?, from?,
//               options: [{ html, correct, feedback }]   (type 'choice')
//               answer: '<trace path>', tolerance, unit?  (type 'numeric') }   answer first, then feedback and the reveal
//   explain   { kind:'explain', id, prompt, model }                           a text box, then "Compare with a good answer"
//   widget    { kind:'widget', id, className?, label?, mount(host, D, ctx), tasks?: (D, T) => [{ do, see }] }
//   answer    { kind:'answer', html: (D, T) => html, next: (D, T) => html }   "The answer", the next question, the next lesson. Last
//   gotcha    { kind:'gotcha', title, html: (D, T) => … }                     a "Why did my X fail?" box (appendix pages only in v3)
// legacy kinds, kept until every lesson is migrated
//   tldr    { kind:'tldr',   items: (D, T) => ['…', …] }                         the 3-4 sentence summary under the hero
//   predict { kind:'predict', id, title?, prompt: (D, T) => …, reveal: (D, T) => … }       think first, then open the answer
//   check   { kind:'check',  id, title?, prompt: (D, T) => …, reveal: (D, T) => … }        like predict, plus an "I got it" mark (stored)
// D = loaded trace data (see core/trace.js); T('scene.path', role) = a number from the trace, shown as <span class="num" data-trace>.
// Every prompt, option html, feedback, reveal, model, items and tasks field is (D, T) => html (a plain string also works).
// Option html sits inside a <label>, so it is inline markup.
import { makeT, stemOf } from './trace.js';
import { escapeHtml } from './dom.js';
import { loopDiagram } from './blocks.js';

export const SECTION_KINDS = ['tldr', 'text', 'gotcha', 'predict', 'check', 'widget', 'questions', 'quiz', 'explain', 'answer'];
export const QUIZ_PURPOSES = ['predict', 'check', 'transfer', 'cumulative'];
export const QUIZ_TYPES = ['choice', 'numeric'];
export const LOOP_VALUES = ['data', 'model', 'loss', 'adjust', 'all'];
const QUIZ_TITLE = { predict: 'Before you read on', check: 'Check yourself', transfer: 'A new situation', cumulative: 'From an earlier lesson' };
const call = (f, D, T) => (typeof f === 'function' ? f(D, T) : String(f ?? ''));
const list = (f, D, T) => (typeof f === 'function' ? f(D, T) : f) ?? [];
const lessonLink = (l) => `<a href="../${escapeHtml(l.id)}/">${escapeHtml(l.title)}</a>`;

function quizHtml(q, D, T, ctx) {
  const id = escapeHtml(q.id), base = `quiz-${id}`;
  const attrs = ['class="quiz"', `id="${base}"`, `data-quiz="${id}"`, `data-purpose="${escapeHtml(q.purpose)}"`, `data-type="${escapeHtml(q.type)}"`];
  let field;
  if (q.type === 'numeric') {
    attrs.push(`data-answer="${T.num(q.answer)}"`, `data-tolerance="${Number(q.tolerance)}"`);
    field = `<legend>Type a number.</legend><div class="numeric"><label for="${base}-n">Your answer</label>`
      + `<input type="number" step="any" inputmode="decimal" autocomplete="off" id="${base}-n" name="${base}">`
      + (q.unit ? `<span class="unit">${escapeHtml(q.unit)}</span>` : '') + '</div>';
  } else {
    attrs.push(`data-correct="${q.options.findIndex((o) => o.correct === true)}"`);
    field = '<legend>Choose one answer.</legend><div class="options">' + q.options.map((o, i) =>
      `<div class="option"><input type="radio" id="${base}-o${i}" name="${base}" value="${i}"><label for="${base}-o${i}">${call(o.html, D, T)}</label></div>`).join('') + '</div>';
  }
  if (q.from) attrs.push(`data-from="${escapeHtml(q.from)}"`);
  attrs.push(`aria-labelledby="${base}-h"`);
  const fromLesson = q.from && ctx.course ? (ctx.course.parts ?? []).flatMap((p) => p.lessons ?? []).find((l) => l.id === q.from) : null;
  const templates = q.type === 'choice' ? q.options.map((o, i) => `<template data-fb="${i}">${o.feedback ? call(o.feedback, D, T) : ''}</template>`).join('') : '';
  return `<section ${attrs.join(' ')}><h3 id="${base}-h">${escapeHtml(q.title ?? QUIZ_TITLE[q.purpose] ?? 'Check yourself')}</h3>`
    + (fromLesson ? `<p class="from">This question comes from the lesson ${lessonLink(fromLesson)}.</p>` : '')
    + `<div class="prompt" id="${base}-prompt">${call(q.prompt, D, T)}</div>`
    + `<form class="quiz-form" novalidate><fieldset aria-describedby="${base}-prompt">${field}</fieldset>`
    + '<div class="quiz-actions"><button type="submit" class="btn primary">Check</button></div></form>'
    + `<div class="feedback" id="${base}-fb" role="status" aria-live="polite"></div>${templates}`
    + `<div class="reveal" id="${base}-reveal" hidden><p class="reveal-h"><strong>Explanation</strong></p>${call(q.reveal, D, T)}</div></section>`;
}

function explainHtml(x, D, T) {
  const id = escapeHtml(x.id), base = `explain-${id}`;
  return `<section class="explain" id="${base}" data-explain="${id}" aria-labelledby="${base}-h"><h3 id="${base}-h">Explain it in one sentence</h3>`
    + `<div class="prompt" id="${base}-prompt">${call(x.prompt, D, T)}</div>`
    + `<div class="explain-field"><label for="${base}-text">Your sentence</label><textarea id="${base}-text" rows="3" aria-describedby="${base}-prompt"></textarea></div>`
    + '<div class="quiz-actions"><button type="button" class="btn" data-explain-compare>Compare with a good answer</button></div>'
    + '<div class="feedback" role="status" aria-live="polite"></div>'
    + `<div class="model" id="${base}-model" hidden><p class="reveal-h"><strong>One good answer</strong></p>${call(x.model, D, T)}</div></section>`;
}

function answerHtml(a, D, T, ctx) {
  let nextLine = '';
  if (ctx.next) nextLine = `<p class="next-lesson">The next lesson is ${lessonLink(ctx.next)}.</p>`;
  else if (ctx.next === null) nextLine = '<p class="next-lesson">This is the last lesson here. The <a href="../../">course map</a> lists all lessons.</p>';
  return `<section class="answer-box" aria-labelledby="answer-h"><h2 id="answer-h">The answer</h2><div class="answer-body">${call(a.html, D, T)}</div>`
    + `<div class="next-question"><h3>Next question</h3>${call(a.next, D, T)}</div>${nextLine}</section>`;
}

function tasksHtml(tasks) {
  if (!tasks.length) return '';
  return '<dl class="tasks" aria-label="Guided tasks">' + tasks.map((t) =>
    `<div class="task-do"><dt>Task</dt><dd>${t.do}</dd></div><div class="task-see"><dt>What you will see</dt><dd>${t.see}</dd></div>`).join('') + '</dl>';
}

// ctx (optional): { lesson, course, next } — next is the next lesson {id, title} from neighbors() (null: there is none, undefined:
// not known, so no next-lesson line); course gives the title of a cumulative quiz's `from` lesson.
export function sectionHtml(section, D, T = makeT(D), ctx = {}) {
  switch (section.kind) {
    case 'text':
      return call(section.html, D, T);
    case 'questions': {
      const loop = ctx.lesson?.loop ? loopDiagram(ctx.lesson.loop) : '';
      return '<aside class="questions" aria-labelledby="questions-h"><div class="questions-body"><h2 id="questions-h">Questions this lesson answers</h2>'
        + `<ul>${list(section.items, D, T).map((i) => `<li>${i}</li>`).join('')}</ul></div>${loop}</aside>`;
    }
    case 'quiz':
      return quizHtml(section, D, T, ctx);
    case 'explain':
      return explainHtml(section, D, T);
    case 'answer':
      return answerHtml(section, D, T, ctx);
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
      return `<section id="${escapeHtml(section.id)}" class="${escapeHtml(section.className ?? 'widget')}" aria-label="${escapeHtml(section.label ?? section.id)}" data-widget="${escapeHtml(section.id)}"></section>`
        + (section.tasks ? tasksHtml(list(section.tasks, D, T)) : '');
    default:
      throw new Error(`unknown section kind "${section.kind}"`);
  }
}

// The structure of a quiz (brief §9): 3-4 options, exactly one correct, feedback on every wrong option; or a numeric answer path
// with a tolerance above zero. Whether the path resolves is checked by the tests, which have the data.
export function quizProblems(q) {
  const errs = [];
  if (!QUIZ_PURPOSES.includes(q.purpose)) errs.push(`purpose must be one of ${QUIZ_PURPOSES.join(', ')}`);
  if (q.purpose === 'cumulative' && !(typeof q.from === 'string' && q.from)) errs.push('a cumulative quiz needs from: "<earlier lesson id>"');
  if (q.from !== undefined && q.purpose !== 'cumulative') errs.push('only a cumulative quiz has "from"');
  if (!q.prompt) errs.push('prompt missing');
  if (!q.reveal) errs.push('reveal missing');
  if (q.type === 'choice') {
    if (!Array.isArray(q.options) || q.options.length < 3 || q.options.length > 4) errs.push('a choice quiz has 3-4 options');
    else {
      if (q.options.filter((o) => o?.correct === true).length !== 1) errs.push('exactly one option is correct');
      q.options.forEach((o, j) => {
        if (!o?.html) errs.push(`option ${j} has no html`);
        if (o?.correct !== true && !o?.feedback) errs.push(`wrong option ${j} needs feedback that names the misconception`);
      });
    }
    if (q.answer !== undefined) errs.push('a choice quiz has no "answer"');
  } else if (q.type === 'numeric') {
    if (typeof q.answer !== 'string' || !q.answer) errs.push('a numeric quiz needs answer: "<trace path>"');
    if (!(typeof q.tolerance === 'number' && Number.isFinite(q.tolerance) && q.tolerance > 0)) errs.push('a numeric quiz needs a tolerance above 0');
    if (q.options !== undefined) errs.push('a numeric quiz has no options');
  } else errs.push(`type must be one of ${QUIZ_TYPES.join(', ')}`);
  return errs;
}

// Structural checks every lesson module must satisfy.
export function validateLesson(lesson) {
  const errs = [];
  if (!lesson || typeof lesson !== 'object') return ['lesson module has no default export'];
  if (typeof lesson.id !== 'string') errs.push('lesson.id missing');
  if (typeof lesson.title !== 'string') errs.push('lesson.title missing');
  if (!Array.isArray(lesson.trace) || !lesson.trace.every((f) => typeof f === 'string')) errs.push('lesson.trace must be an array of file names');
  if (!Array.isArray(lesson.sections) || !lesson.sections.length) errs.push('lesson.sections must be a non-empty array');
  if (lesson.template !== undefined && lesson.template !== 3) errs.push('lesson.template must be 3 or absent');
  if (lesson.loop !== undefined && !LOOP_VALUES.includes(lesson.loop)) errs.push(`lesson.loop must be one of ${LOOP_VALUES.join(', ')}`);
  if (lesson.conceptual !== undefined && typeof lesson.conceptual !== 'boolean') errs.push('lesson.conceptual must be true or false');
  if (lesson.conceptual && !(Array.isArray(lesson.sources) && lesson.sources.length && lesson.sources.every((x) => typeof x?.title === 'string' && /^https:\/\//.test(x?.url ?? ''))))
    errs.push('a conceptual lesson needs sources: [{title, url: "https://…"}, …]');
  const ids = new Set();
  for (const [i, s] of (lesson.sections ?? []).entries()) {
    if (!SECTION_KINDS.includes(s?.kind)) { errs.push(`section ${i}: unknown kind "${s?.kind}"`); continue; }
    if (s.kind === 'text' && typeof s.html !== 'function') errs.push(`section ${i}: text sections need html: (D, T) => string`);
    if (s.kind === 'tldr' && !(typeof s.items === 'function' || Array.isArray(s.items))) errs.push(`section ${i}: tldr sections need items: (D, T) => [html, …]`);
    if (s.kind === 'questions' && !(typeof s.items === 'function' || Array.isArray(s.items))) errs.push(`section ${i}: questions sections need items: (D, T) => [html, …]`);
    if (s.kind === 'widget' && (typeof s.mount !== 'function' || typeof s.id !== 'string')) errs.push(`section ${i}: widget sections need id and mount(host, D, ctx)`);
    if (s.kind === 'widget' && s.tasks !== undefined && !(typeof s.tasks === 'function' || Array.isArray(s.tasks))) errs.push(`section ${i}: widget tasks must be (D, T) => [{do, see}, …]`);
    if (['predict', 'check', 'quiz', 'explain'].includes(s.kind) && typeof s.id !== 'string') errs.push(`section ${i}: ${s.kind} sections need an id`);
    if (s.kind === 'explain' && (!s.prompt || !s.model)) errs.push(`section ${i}: explain sections need prompt and model`);
    if (s.kind === 'answer' && (!s.html || !s.next)) errs.push(`section ${i}: answer sections need html and next`);
    if (s.kind === 'quiz') errs.push(...quizProblems(s).map((e) => `section ${i} (quiz "${s.id}"): ${e}`));
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
// Bound numerals as [{path, shown, role?, sig?, human?, of?, anchor?}] so a test can re-resolve and re-format each one
// (T.reformat in core/trace.js). Legacy spans carry only data-trace.
const unq = (s) => s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
export function boundNums(html) {
  return [...html.matchAll(/<span class="num" data-trace="([^"]*)"((?: data-[a-z]+="[^"]*")*)>([^<]*)<\/span>/g)].map((m) => {
    const b = { path: unq(m[1]), shown: m[3] };
    for (const a of m[2].matchAll(/ data-([a-z]+)="([^"]*)"/g)) b[a[1]] = unq(a[2]);
    return b;
  });
}

// Render a whole lesson to [{section, html}] (widgets come out as empty hosts, followed by their tasks).
// ctx: { course, next } as for sectionHtml; lesson-boot.js passes the next lesson from neighbors().
export function renderLesson(lesson, D, ctx = {}) {
  const T = makeT(D);
  return lesson.sections.map((section) => ({ section, html: sectionHtml(section, D, T, { ...ctx, lesson }) }));
}

// The one-line footer every lesson gets unless it brings its own: which TorC run the numbers come from, and the source.
// Where the numbers come from is explained once, on the home page; lessons do not repeat it. A conceptual lesson (content beyond
// TorC, decision 18) also lists its sources; without trace files it has only that list.
export function defaultFooter(lesson, D) {
  const ref = lesson.trace.map((f) => D.files[stemOf(f)]?.library?.ref).find(Boolean);
  const repo = 'https://github.com/IvorLipic/TorC';
  const shown = ref && /^[0-9a-f]{40}$/.test(ref) ? ref.slice(0, 7) : ref;   // a pinned commit SHA is shown short, linked in full
  const numbers = `<p>Numbers: TorC${ref ? ` <code>${escapeHtml(shown)}</code>` : ''} · <a href="${ref ? `${repo}/tree/${encodeURIComponent(ref)}` : repo}">source</a></p>`;
  if (!lesson.conceptual) return numbers;
  const sources = `<p>This page explains ideas beyond TorC. It uses these sources.</p><ul class="sources">${(lesson.sources ?? []).map((x) => `<li><a href="${escapeHtml(x.url)}">${escapeHtml(x.title)}</a></li>`).join('')}</ul>`;
  return (lesson.trace.length ? numbers : '') + sources;
}
