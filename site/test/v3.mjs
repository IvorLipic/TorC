// The v3 lesson rules (docs/v3-implementation.md "Budgets", "Numbers", "Blocks"; brief §4, §8, §9, §10; R2, R7, R9, R10) as one
// function, checkV3(), that returns a list of problems. Not a test file itself: lessons.test.mjs runs it on every lesson whose
// module has `template: 3`, and v3.test.mjs runs it on the fixture lesson (site/test/fixtures/) and on broken copies of it, so
// each rule is proven to fire. Every problem starts with a rule code ("role:", "glossary:", …) so a test can ask for one.
//
// Exceptions (brief §9): an appendix lesson (its part has kind "appendix" in course.json) needs no Questions/Answer frame, no
// predict, transfer, explain or cumulative quiz, and may use any symbol (the notation page lists them all). A conceptual lesson
// (`conceptual: true`, content beyond TorC) needs no predict-before-text and no transfer quiz.
import { flatLessons } from '../js/core/course.js';
import { makeT } from '../js/core/trace.js';
import { renderLesson, sectionHtml, boundNums, findUnboundNums, quizProblems } from '../js/core/sections.js';
import { ROLES } from '../js/core/numbers.js';
import { sectionWords } from './helpers.mjs';

export const LIMITS = { questions: 3, answerWords: 80, textWords: 150, numbered: 6, quizzes: 6, words: 1400, worked: 3, perSentence: 2, outside: 10, palette: 2, tasksMin: 1, tasksMax: 3 };
const PALETTE = ['analogy', 'refresher', 'in-torc', 'chatgpt'];
const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);

// ---- a small HTML walker: every token with the open elements around it ------------------------------------------------
const attrsOf = (s) => Object.fromEntries([...s.matchAll(/([a-zA-Z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)].map((m) => [m[1].toLowerCase(), m[2] ?? m[3] ?? m[4] ?? '']));
export function tokens(html) {
  const out = [], stack = [];
  for (const m of html.matchAll(/<!--[\s\S]*?-->|<(\/?)([a-zA-Z][\w-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>|[^<]+|</g)) {
    if (m[0].startsWith('<!--')) continue;
    if (m[2]) {
      const tag = m[2].toLowerCase();
      if (m[1]) {                                   // close: pop to the matching element
        const i = stack.map((e) => e.tag).lastIndexOf(tag);
        if (i >= 0) stack.length = i;
        continue;
      }
      const el = { tag, attrs: attrsOf(m[3]), cls: (attrsOf(m[3]).class ?? '').split(/\s+/).filter(Boolean) };
      out.push({ type: 'open', el, within: [...stack] });
      if (!VOID.has(tag) && !/\/\s*$/.test(m[3])) stack.push(el);
    } else out.push({ type: 'text', text: m[0], within: [...stack] });
  }
  return out;
}
const inside = (t, pred) => t.within.some(pred);
const hasClass = (c) => (e) => e.cls.includes(c);

// render without KaTeX, so a formula becomes <code>latex</code> (word counts skip it; the notation check reads it)
function plainRender(fn) {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return fn(); } finally { globalThis.katex = saved; }
}
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const words = (s) => s.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
const text = (h) => h.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, ' ').replace(/<code[^>]*>[\s\S]*?<\/code>/g, ' ').replace(/<[^>]+>/g, ' ');

// ---- course helpers -----------------------------------------------------------------------------------------------------
const partOf = (course, id) => (course.parts ?? []).find((p) => (p.lessons ?? []).some((l) => l.id === id)) ?? null;
export const isAppendixLesson = (course, id) => partOf(course, id)?.kind === 'appendix';
// main-path lessons in course order (not labs, not the appendix), every status
export const mainPath = (course) => (course.parts ?? []).filter((p) => p.kind !== 'lab' && p.kind !== 'appendix').flatMap((p) => p.lessons.map((l) => l.id));

// ---- the rules ------------------------------------------------------------------------------------------------------------
// opts: { D (with anchors), course, glossary (site/glossary.json), notation (NOTATION), source (the module's source text) }
export function checkV3(lesson, { D, course, glossary, notation, source }) {
  const P = [];
  const add = (code, msg) => P.push(`${code}: ${msg}`);
  const T = makeT(D);
  const order = flatLessons(course).map((l) => l.id);
  const at = order.indexOf(lesson.id);
  if (at < 0) add('course', `lesson "${lesson.id}" is not in course.json`);
  const appendix = isAppendixLesson(course, lesson.id), conceptual = lesson.conceptual === true;
  const S = lesson.sections;

  // a numeric quiz's answer path is resolved while rendering, so check it first and say which quiz it is
  const badPaths = S.filter((s) => s.kind === 'quiz' && s.type === 'numeric').flatMap((q) => { try { T.num(q.answer); return []; } catch (e) { return [`quiz: quiz "${q.id}": ${e.message}`]; } });
  if (badPaths.length) return badPaths;
  let parts;
  try { parts = plainRender(() => renderLesson(lesson, D, { course })); } catch (e) { return [`render: ${e.message}`]; }
  const html = parts.map((p) => p.html).join('\n');
  const toks = tokens(html);

  // kinds
  for (const s of S) {
    if (['tldr', 'predict', 'check'].includes(s.kind)) add('kinds', `a v3 lesson uses quiz and questions, not the legacy kind "${s.kind}"`);
    if (s.kind === 'gotcha' && !appendix) add('kinds', 'a gotcha box belongs on an appendix page');
  }

  // numbers: bound, tagged, re-formatted, in their place
  for (const u of findUnboundNums(html)) add('role', `a number without data-trace: ${u}`);
  const spans = toks.filter((t) => t.type === 'open' && t.el.tag === 'span' && t.el.cls.includes('num'));
  for (const s of spans) {
    const role = s.el.attrs['data-role'], path = s.el.attrs['data-trace'];
    if (!ROLES.includes(role)) { add('role', `${path} has no role (use T(path, role), T.anchor or T.human)`); continue; }
    if (role === 'worked' && !inside(s, hasClass('worked'))) add('worked', `${path} has the role worked outside a worked() block`);
    if (role === 'reveal' && !inside(s, (e) => e.cls.includes('reveal') || e.cls.includes('model') || (e.tag === 'template' && 'data-fb' in e.attrs))) add('reveal', `${path} has the role reveal outside a quiz reveal or feedback`);
  }
  for (const b of boundNums(html)) {
    let want;
    try { want = T.reformat(b); } catch (e) { add('binding', e.message); continue; }
    if (want !== b.shown) add('binding', `${b.path} shows "${b.shown}", the data gives "${want}"`);
  }
  const workedBlocks = toks.filter((t) => t.type === 'open' && t.el.cls.includes('worked')).length;
  if (workedBlocks > LIMITS.worked) add('worked', `${workedBlocks} worked blocks, at most ${LIMITS.worked}`);
  const outside = spans.filter((s) => ['anchor', 'contrast', 'human'].includes(s.el.attrs['data-role']) && !inside(s, (e) => e.tag === 'pre' || 'data-widget' in e.attrs)).length;
  if (outside > LIMITS.outside) add('numbers', `${outside} numbers outside worked, setup and reveal roles, at most ${LIMITS.outside}`);
  // at most 2 numbers per sentence outside worked blocks (code blocks do not count)
  const marked = html.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '\n')
    .replace(/<span class="num"[^>]*data-role="([a-z]+)"[^>]*>[^<]*<\/span>/g, (m, r) => (r === 'worked' ? ' W ' : ' \u0001 '))
    .replace(/<\/(p|li|h[1-6]|div|section|aside|dd|dt|summary|figcaption|legend|label|template|td|th)>/g, '\n').replace(/<[^>]+>/g, ' ');
  for (const block of marked.split('\n')) for (const sentence of block.split(/(?<=[.?!])\s+/)) {
    const n = (sentence.match(/\u0001/g) ?? []).length;
    if (n > LIMITS.perSentence) add('numbers', `${n} numbers in one sentence: ${sentence.replace(/\u0001/g, '#').replace(/\s+/g, ' ').trim().slice(0, 100)}`);
  }

  // template and budgets
  if (!appendix) {
    const q = S[1];
    if (q?.kind !== 'questions') add('questions', 'section 1 (right after the hero) is the Questions box');
    else {
      const n = (typeof q.items === 'function' ? q.items(D, T) : q.items).length;
      if (n < 1 || n > LIMITS.questions) add('questions', `${n} questions, 1 to ${LIMITS.questions}`);
    }
    const last = S[S.length - 1];
    if (last?.kind !== 'answer') add('answer', 'the last section is The answer');
    else {
      const n = plainRender(() => sectionWords(typeof last.html === 'function' ? last.html(D, T) : last.html));
      if (n > LIMITS.answerWords) add('answer', `The answer has ${n} words, at most ${LIMITS.answerWords}`);
    }
  }
  // a refresher is collapsed, optional reading of 100-200 words (brief §8), so it does not count toward the section budget
  const noRefresher = (h) => h.replace(/<details class="refresher"[\s\S]*?<\/details>/g, ' ');
  S.forEach((s, i) => { if (s.kind === 'text') { const n = sectionWords(noRefresher(parts[i].html)); if (n > LIMITS.textWords) add('words', `text section ${i} has ${n} words, at most ${LIMITS.textWords}`); } });
  const numbered = (html.match(/<h2[^>]*>\s*\d+\s·/g) ?? []).length;
  if (numbered > LIMITS.numbered) add('budget', `${numbered} numbered sections, at most ${LIMITS.numbered}`);
  const quizzes = S.filter((s) => s.kind === 'quiz');
  if (quizzes.length > LIMITS.quizzes) add('budget', `${quizzes.length} quizzes, at most ${LIMITS.quizzes}`);
  const total = words(text(html));
  // the two reference pages of the appendix (notation, glossary) are generated lists whose length follows the course, not a
  // reading budget; their text sections keep the 150-word limit
  const reference = appendix && ['notation', 'glossary'].includes(lesson.id);
  if (total > LIMITS.words && !reference) add('words', `${total} words of prose, at most ${LIMITS.words}`);
  const palette = toks.filter((t) => t.type === 'open' && PALETTE.some((c) => t.el.cls.includes(c)) && ['aside', 'details'].includes(t.el.tag)).length;
  if (palette > LIMITS.palette) add('palette', `${palette} of analogy, refresher, In TorC and ChatGPT, at most ${LIMITS.palette}`);

  // the core template: predict first, transfer, explain, cumulative
  if (!appendix && !conceptual) {
    const firstText = S.findIndex((s, i) => i > 0 && s.kind === 'text' && !s.setup);
    const firstPredict = S.findIndex((s) => s.kind === 'quiz' && s.purpose === 'predict');
    if (firstPredict < 0) add('predict', 'no predict quiz');
    else if (firstText >= 0 && firstText < firstPredict) add('predict', `the first predict quiz (section ${firstPredict}) comes after text section ${firstText}; mark pure scenario text with setup: true`);
    if (!quizzes.some((s) => s.purpose === 'transfer')) add('transfer', 'no transfer quiz');
  }
  if (!appendix) {
    if (!S.some((s) => s.kind === 'explain')) add('explain', 'no explain prompt');
    const pos = mainPath(course).indexOf(lesson.id);
    if (pos >= 2 && !quizzes.some((s) => s.purpose === 'cumulative')) add('cumulative', `lesson ${pos + 1} of the main path needs a cumulative quiz`);
  }
  for (const s of quizzes.filter((x) => x.purpose === 'cumulative')) {
    const j = order.indexOf(s.from);
    if (j < 0) add('cumulative', `quiz "${s.id}": from "${s.from}" is not a lesson in course.json`);
    else if (j >= at) add('cumulative', `quiz "${s.id}": from "${s.from}" is not earlier in course order`);
  }

  // widgets have guided tasks
  for (const s of S.filter((x) => x.kind === 'widget')) {
    const n = s.tasks ? (typeof s.tasks === 'function' ? s.tasks(D, T) : s.tasks).length : 0;
    if (n < LIMITS.tasksMin || n > LIMITS.tasksMax) add('tasks', `widget "${s.id}" has ${n} tasks, 1 to 3`);
  }

  // quizzes: structure, the numeric answer resolves, no hand-typed digit in prompt, options, feedback or reveal
  for (const q of quizzes) {
    for (const e of quizProblems(q)) add('quiz', `quiz "${q.id}": ${e}`);
    if (q.type === 'numeric') { try { T.num(q.answer); } catch (e) { add('quiz', `quiz "${q.id}": ${e.message}`); } }
    const bits = [q.prompt, q.reveal, ...(q.options ?? []).flatMap((o) => [o.html, o.feedback])].filter(Boolean);
    for (const f of bits) {
      const h = plainRender(() => (typeof f === 'function' ? f(D, T) : String(f)));
      const digits = text(h.replace(/<span class="num"[^>]*>[^<]*<\/span>/g, ' ')).match(/\S*\d\S*/g);
      if (digits) add('quiz', `quiz "${q.id}": hand-typed digits ${JSON.stringify(digits)} (bind them with T)`);
    }
  }

  // glossary: every term the lesson uses is linked with term() at its first use (headings, code, formulas, SVG and <dfn> skipped)
  const terms = glossary?.terms ?? {};
  for (const t of toks.filter((x) => x.type === 'open' && x.el.cls.includes('term'))) if (!Object.hasOwn(terms, t.el.attrs['data-term'])) add('glossary', `term("${t.el.attrs['data-term']}") is not in site/glossary.json`);
  // <dfn> marks the defining instance of a term (the glossary page), which needs no link to its own definition
  const skip = (e) => ['pre', 'code', 'svg', 'h1', 'h2', 'h3', 'style', 'script', 'dfn'].includes(e.tag);
  const prose = toks.filter((t) => t.type === 'text' && !inside(t, skip));
  for (const [key, g] of Object.entries(terms)) {
    const re = new RegExp(`(?<![\\p{L}\\p{N}])(?:${g.forms.map(escRe).join('|')})(?![\\p{L}\\p{N}])`, 'iu');
    // a word inside the link of a longer term that contains it ("gradient" in "gradient descent") is a use of that longer term
    const inLonger = (t) => inside(t, (e) => e.tag === 'a' && e.cls.includes('term') && e.attrs['data-term'] !== key && re.test(terms[e.attrs['data-term']]?.term ?? ''));
    const first = prose.find((t) => re.test(t.text.replace(/&[a-z]+;/g, ' ')) && !inLonger(t));
    if (first && !inside(first, (e) => e.tag === 'a' && e.cls.includes('term') && e.attrs['data-term'] === key))
      add('glossary', `the first use of "${g.term}" is not linked with term('${key}'): …${first.text.trim().slice(0, 60)}…`);
  }

  // notation: no symbol before the lesson that introduces it, in course order (the appendix lists them all)
  if (!appendix) {
    const codeText = toks.filter((t) => t.type === 'text' && inside(t, (e) => e.tag === 'code')).map((t) => t.text).join('\n')
      .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
    const proseText = toks.filter((t) => t.type === 'text' && !inside(t, (e) => e.tag === 'code' || e.tag === 'pre')).map((t) => t.text).join('\n');
    const symLinks = toks.filter((t) => t.type === 'open' && t.el.cls.includes('sym')).map((t) => t.el.attrs['data-sym']);
    for (const n of notation) {
      const j = order.indexOf(n.lesson);
      if (j >= 0 && j <= at) continue;
      const used = (n.match.latex ?? []).some((m) => codeText.includes(m)) || (n.match.text ?? []).some((m) => proseText.includes(m)) || symLinks.includes(n.key);
      if (used) add('notation', `${n.symbol} (${n.name}) is introduced in "${n.lesson}", which comes ${j < 0 ? 'nowhere in the course yet' : 'later in course order'}`);
    }
  }

  // LaTeX escape bugs in the source: '\;' in a JS string is a bare ';' (D1). Write '\\;' (and the same for , ! : and a space).
  if (typeof source === 'string') for (const m of source.matchAll(/(?<!\\)\\[;,!: ]/g)) {
    const line = source.slice(0, m.index).split('\n').length;
    add('escape', `line ${line}: ${JSON.stringify(m[0])} is not a LaTeX space in a JS string; write ${JSON.stringify('\\' + m[0])}`);
  }
  return P;
}

// The next-lesson line of the answer section, as lesson-boot renders it with neighbors() (D2).
export function answerWithNext(lesson, D, course, next) {
  const i = lesson.sections.findIndex((s) => s.kind === 'answer');
  return i < 0 ? '' : sectionHtml(lesson.sections[i], D, makeT(D), { lesson, course, next });
}
