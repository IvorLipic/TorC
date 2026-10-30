// The closing "next lesson" line of a lesson must name the lesson that the navigation buttons (neighbors() in core/course.js) go to.
// Brief D2: sgd-step said "Layers" while its Next button said "Tensors". Brief D3: linear-layer ended without a next lesson.
// The rule: in the last text section that says "The next lesson, <a href="../id/">Title</a>", id is neighbors().next and Title is
// its course.json title. The last lesson of a sequence says nothing about a next lesson.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { flatLessons, isPublished, neighbors } from '../js/core/course.js';
import { buildD, stemOf, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { course, readJson, SITE } from './helpers.mjs';

const C = course();
const entries = flatLessons(C).filter((l) => isPublished(l) || existsSync(new URL(`js/lessons/${l.id}.js`, SITE)));

for (const entry of entries) {
  test(`lesson ${entry.id}: the closing next-lesson link matches neighbors()`, async () => {
    const lesson = (await import(new URL(`../js/lessons/${entry.id}.js`, import.meta.url))).default;
    if (lesson.template === 3) return;   // a v3 lesson names its next lesson in The answer; lessons.test.mjs checks it (answerWithNext)
    const D = buildD(Object.fromEntries(lesson.trace.map((f) => [stemOf(f), readJson('data/' + f)])), anchorValues(readJson('data/anchors.json')));
    const naming = renderLesson(lesson, D).filter((p) => p.section.kind === 'text' && /next lesson/i.test(p.html));
    const next = neighbors(C, entry.id).next;
    if (!next) {
      assert.equal(naming.length, 0, 'the last lesson of its sequence must not name a next lesson');
      return;
    }
    assert.ok(naming.length > 0, `${entry.id} has no text section that names the next lesson (${next.id})`);
    const closing = naming.at(-1).html;
    const m = closing.match(/next lesson,?\s*<a href="\.\.\/([a-z0-9-]+)\/">([^<]*)<\/a>/i);
    assert.ok(m, 'the next-lesson line must start with: The next lesson, <a href="../id/">Title</a>');
    assert.equal(m[1], next.id, `links ../${m[1]}/ but the Next button goes to ../${next.id}/`);
    assert.equal(m[2], next.title, 'the link text is the title of the next lesson');
    assert.ok(existsSync(new URL(`lessons/${next.id}/index.html`, SITE)), 'the next lesson has a page');
  });
}

test('every lesson with a next neighbour was checked, so a new lesson cannot skip the rule', () => {
  assert.ok(entries.filter((l) => neighbors(C, l.id).next).length >= 10);
});
