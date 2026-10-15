// Every published lesson renders in Node against the real trace data: text sections, no hand-typed numerals.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { flatLessons, isPublished } from '../js/core/course.js';
import { buildD, makeT, stemOf } from '../js/core/trace.js';
import { renderLesson, validateLesson, findUnboundNums, boundNums, defaultFooter } from '../js/core/sections.js';
import { course, readJson, SITE, lessonStats, sectionWords } from './helpers.mjs';

const lessons = flatLessons(course());

test('course.json lists at least one published lesson', () => {
  assert.ok(lessons.some(isPublished));
});

for (const entry of lessons.filter((l) => isPublished(l) || existsSync(new URL(`js/lessons/${l.id}.js`, SITE)))) {
  test(`lesson ${entry.id}: module agrees with course.json and its text sections render against the real data`, async () => {
    const lesson = (await import(new URL(`../js/lessons/${entry.id}.js`, import.meta.url))).default;
    assert.deepEqual(validateLesson(lesson), []);
    assert.equal(lesson.id, entry.id);
    assert.equal(lesson.title, entry.title);
    assert.deepEqual(lesson.trace, entry.trace);

    const D = buildD(Object.fromEntries(lesson.trace.map((f) => [stemOf(f), readJson('data/' + f)])));
    const T = makeT(D);
    const parts = renderLesson(lesson, D);                       // throws if any T('path') is missing
    assert.equal(parts.length, lesson.sections.length);
    for (const { section, html } of parts) {
      assert.equal(typeof html, 'string');
      assert.ok(html.length > 0, `${section.kind} section rendered nothing`);
      assert.deepEqual(findUnboundNums(html), [], `unbound numerals in a ${section.kind} section (use T('scene.path'))`);
      for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, `${path} shows ${shown}`);
      assert.doesNotMatch(html, /\bundefined\b|\[object Object\]|NaN/, `${section.kind} section contains a rendering accident`);
    }
    if (lesson.footer) assert.equal(typeof lesson.footer(D, T), 'string');
    else assert.match(defaultFooter(lesson, D), /href="https:\/\/github\.com\/IvorLipic\/TorC\/tree\/v0\.1\.0"/, 'the source link points at the pinned release');
    assert.match(parts[0].html, /<h1>/, 'the first section carries the lesson heading');
    // style budget (docs/authoring.md): TL;DR right after the hero, short prose, few numbers, at most two widgets and three questions
    assert.equal(lesson.sections[1]?.kind, 'tldr', 'the TL;DR comes right after the hero');
    const st = lessonStats(lesson, D, renderLesson, boundNums);
    assert.ok(st.words <= 800 && st.numbers <= 25 && st.widgets <= 2 && st.questions <= 3, JSON.stringify(st));
    assert.doesNotMatch(parts.map((p) => p.html).join(''), /recorded|recomputed|printed by|in your browser|show as table/i, 'provenance is said once, on the home page');
    // the finer limits: TL;DR <= 50 words, every text section <= 60, at most four numbered sections, one predict, two checks
    const saved = globalThis.katex;
    globalThis.katex = undefined;
    let plain;
    try { plain = renderLesson(lesson, D); } finally { globalThis.katex = saved; }
    for (const { section, html } of plain) {
      if (section.kind === 'tldr') assert.ok(sectionWords(html) <= 50, `TL;DR has ${sectionWords(html)} words`);
      if (section.kind === 'text') assert.ok(sectionWords(html) <= 60, `a text section has ${sectionWords(html)} words: ${html.replace(/<[^>]+>/g, ' ').slice(0, 80)}`);
    }
    const numbered = plain.map((p) => p.html).join('').match(/<h2[^>]*>\s*\d+\s·/g) ?? [];
    assert.ok(numbered.length <= 4, `${numbered.length} numbered sections`);
    assert.ok(lesson.sections.filter((x) => x.kind === 'predict').length <= 1, 'at most one predict');
    assert.ok(lesson.sections.filter((x) => x.kind === 'check').length <= 2, 'at most two checks');
  });
}
