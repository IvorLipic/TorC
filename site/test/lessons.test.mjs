// Every published lesson renders in Node against the real trace data: text sections, no hand-typed numerals.
// A lesson whose module has `template: 3` is checked by the v3 rules (site/test/v3.mjs, docs/v3-implementation.md); every other
// lesson keeps the rules of the old template (TL;DR, the 800-word budget) until it is migrated.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { flatLessons, isPublished, neighbors } from '../js/core/course.js';
import { buildD, makeT, stemOf, anchorValues } from '../js/core/trace.js';
import { renderLesson, validateLesson, findUnboundNums, boundNums, defaultFooter } from '../js/core/sections.js';
import { NOTATION } from '../js/content/notation.js';
import { course, readJson, SITE, lessonStats, sectionWords, proseProblems } from './helpers.mjs';
import { checkV3, answerWithNext } from './v3.mjs';

// the pinned TorC ref (first non-comment line of tools/trace/TORC_REF); every trace envelope records it as library.ref
const TORC_REF = readFileSync(new URL('../tools/trace/TORC_REF', SITE), 'utf8').split('\n').map((l) => l.trim()).find((l) => l && !l.startsWith('#'));
const lessons = flatLessons(course());
const anchors = anchorValues(readJson('data/anchors.json'));
const glossary = readJson('glossary.json');

test('course.json lists at least one published lesson', () => {
  assert.ok(lessons.some(isPublished));
});

for (const entry of lessons.filter((l) => isPublished(l) || existsSync(new URL(`js/lessons/${l.id}.js`, SITE)))) {
  test(`lesson ${entry.id}: module agrees with course.json and its text sections render against the real data`, async () => {
    const lesson = (await import(new URL(`../js/lessons/${entry.id}.js`, import.meta.url))).default;
    const v3 = lesson.template === 3;
    assert.deepEqual(validateLesson(lesson), []);
    assert.equal(lesson.id, entry.id);
    assert.equal(lesson.title, entry.title);
    assert.deepEqual(lesson.trace, entry.trace);

    const D = buildD(Object.fromEntries(lesson.trace.map((f) => [stemOf(f), readJson('data/' + f)])), anchors);
    const T = makeT(D);
    const parts = renderLesson(lesson, D, { course: course() });   // throws if any T('path') is missing
    assert.equal(parts.length, lesson.sections.length);
    for (const { section, html } of parts) {
      assert.equal(typeof html, 'string');
      assert.ok(html.length > 0, `${section.kind} section rendered nothing`);
      assert.deepEqual(findUnboundNums(html), [], `unbound numerals in a ${section.kind} section (use T('scene.path'))`);
      for (const b of boundNums(html)) assert.equal(T.reformat(b), b.shown, `${b.path} shows ${b.shown}`);
      assert.doesNotMatch(html, /\bundefined\b|\[object Object\]|NaN/, `${section.kind} section contains a rendering accident`);
    }
    if (lesson.footer) assert.equal(typeof lesson.footer(D, T), 'string');
    else {
      const footer = defaultFooter(lesson, D);
      if (lesson.trace.length) {
        assert.ok(footer.includes(`href="https://github.com/IvorLipic/TorC/tree/${encodeURIComponent(TORC_REF)}"`), `the source link points at the pinned ref ${TORC_REF}`);
        assert.ok(footer.includes(`<code>${/^[0-9a-f]{40}$/.test(TORC_REF) ? TORC_REF.slice(0, 7) : TORC_REF}</code>`), 'the footer names the pinned ref');
      }
      if (lesson.conceptual) for (const s of lesson.sources) assert.ok(footer.includes(`<a href="${s.url}">`), `the footer cites ${s.url}`);
    }
    assert.match(parts[0].html, /<h1>/, 'the first section carries the lesson heading');
    assert.doesNotMatch(parts.map((p) => p.html).join(''), /recorded|recomputed|printed by|in your browser|show as table/i, 'provenance is said once, on the home page');
    const saved = globalThis.katex;
    globalThis.katex = undefined;
    let plain;
    try { plain = renderLesson(lesson, D, { course: course() }); } finally { globalThis.katex = saved; }
    // the writing rules (docs/authoring.md): no semicolons, dashes or colon punchlines, sentences of at most 25 words
    assert.deepEqual(plain.flatMap(({ html }) => proseProblems(html)), []);

    if (v3) {
      // the v3 rules: number roles and limits, the template and budgets, glossary, notation, LaTeX escapes, quiz structure
      const source = readFileSync(new URL(`js/lessons/${entry.id}.js`, SITE), 'utf8');
      assert.deepEqual(checkV3(lesson, { D, course: course(), glossary, notation: NOTATION, source }), []);
      const next = neighbors(course(), entry.id).next;
      if (lesson.sections.some((s) => s.kind === 'answer')) assert.match(answerWithNext(lesson, D, course(), next), next ? new RegExp(`The next lesson is <a href="\\.\\./${next.id}/">`) : /course map/, 'the next-lesson line follows neighbors() (D2)');
      return;
    }
    // style budget of the old template (docs/authoring.md): TL;DR right after the hero, short prose, few numbers, at most two widgets and three questions
    assert.equal(lesson.sections[1]?.kind, 'tldr', 'the TL;DR comes right after the hero');
    const st = lessonStats(lesson, D, renderLesson, boundNums);
    assert.ok(st.words <= 800 && st.numbers <= 25 && st.widgets <= 2 && st.questions <= 3, JSON.stringify(st));
    // the finer limits: summary <= 60 words, every text section <= 100, at most four numbered sections, one predict, two checks
    for (const { section, html } of plain) {
      if (section.kind === 'tldr') assert.ok(sectionWords(html) <= 60, `the summary has ${sectionWords(html)} words`);
      if (section.kind === 'text') assert.ok(sectionWords(html) <= 100, `a text section has ${sectionWords(html)} words: ${html.replace(/<[^>]+>/g, ' ').slice(0, 80)}`);
    }
    const numbered = plain.map((p) => p.html).join('').match(/<h2[^>]*>\s*\d+\s·/g) ?? [];
    assert.ok(numbered.length <= 4, `${numbered.length} numbered sections`);
    assert.ok(lesson.sections.filter((x) => x.kind === 'predict').length <= 1, 'at most one predict');
    assert.ok(lesson.sections.filter((x) => x.kind === 'check').length <= 2, 'at most two checks');
  });
}

test('writing rules hold for the course descriptions and the graph-lab texts (docs/authoring.md)', async () => {
  const outcomes = flatLessons(course()).map((l) => `<p>${l.outcome}</p>`);
  const { SCENES } = await import('../js/lessons/backprop-lab.scenes.js');
  const { whyLayout } = await import('../js/lessons/why-gradients.scenes.js');
  const D = buildD({ 'why-gradients': readJson('data/why-gradients.json') });
  const layouts = [...Object.values(SCENES), whyLayout(makeT(D))];
  const texts = layouts.flatMap((l) => ['goal', 'intro', 'forwardHint', 'backwardHint', 'takeaway'].map((k) => `<p>${l[k]}</p>`));
  assert.deepEqual([...outcomes, ...texts].flatMap(proseProblems), []);
});
