// Every SVG that a lesson renders has a description (aria-label) or is hidden from screen readers (aria-hidden="true").
// tools/prose/extract.mjs writes a picture as "Picture. <aria-label>" and drops the text inside the SVG (labels are not sentences),
// so this test makes sure no picture is dropped silently from the prose check and from screen readers.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { flatLessons } from '../js/core/course.js';
import { readJson, course } from './helpers.mjs';

const anchors = anchorValues(readJson('data/anchors.json'));

for (const entry of flatLessons(course()).filter((l) => l.status !== 'planned')) {
  test(`lesson ${entry.id}: every SVG has an aria-label or aria-hidden="true"`, async () => {
    const lesson = (await import(`../js/lessons/${entry.id}.js`)).default;
    const D = buildD(Object.fromEntries(lesson.trace.map((f) => [f.replace(/\.json$/, ''), readJson(`data/${f}`)])), anchors);
    // without KaTeX: its own SVGs (the root sign) sit inside an aria-hidden span and are not pictures
    const saved = globalThis.katex;
    globalThis.katex = undefined;
    let html;
    try { html = renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
    for (const m of html.matchAll(/<svg\b[^>]*>/g)) {
      const tag = m[0];
      assert.ok(/\saria-label="[^"]+"/.test(tag) || /\saria-hidden="true"/.test(tag), `${entry.id}: ${tag.slice(0, 120)}`);
    }
  });
}
