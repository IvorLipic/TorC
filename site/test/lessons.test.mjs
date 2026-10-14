// Every published lesson renders in Node against the real trace data: text sections, no hand-typed numerals.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { flatLessons, isPublished } from '../js/core/course.js';
import { buildD, makeT, stemOf } from '../js/core/trace.js';
import { renderLesson, validateLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { course, readJson, SITE, lessonStats } from './helpers.mjs';

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
    assert.match(parts[0].html, /<h1>/, 'the first section carries the lesson heading');
    // style budget (docs/authoring.md): TL;DR right after the hero, short prose, few numbers, at most two widgets and three questions
    assert.equal(lesson.sections[1]?.kind, 'tldr', 'the TL;DR comes right after the hero');
    const st = lessonStats(lesson, D, renderLesson, boundNums);
    assert.ok(st.words <= 800 && st.numbers <= 25 && st.widgets <= 2 && st.questions <= 3, JSON.stringify(st));
    assert.doesNotMatch(parts.map((p) => p.html).join(''), /recorded|recomputed|printed by|in your browser|show as table/i, 'provenance is said once, on the home page');
  });
}
