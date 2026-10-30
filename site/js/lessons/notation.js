// lessons/notation.js — the notation guide of the appendix "Using TorC" (brief §8 "Notation", R9; decision 18). Built from
// content/notation.js: one entry per symbol with an anchor #<key> that sym(key) links to, its name, how to read it aloud and what it
// means, grouped by the lesson that introduces it, in course order. Static: rendered once at import time, with the glossary
// terms of the meanings linked at their first use on the page (lessons/glossary.js, linkTerms).
import { NOTATION } from '../content/notation.js';
import { escapeHtml } from '../core/dom.js';
import { referenceGroups, linkTerms, words } from './glossary.js';

const GROUPS = referenceGroups(NOTATION.map((n) => {
  const read = `Read it as “${n.read}”.`;
  return {
    key: n.key,
    lesson: n.lesson,
    head: `<span class="symbol">${escapeHtml(n.symbol)}</span> ${escapeHtml(n.name)}`,
    words: words(`${read} ${n.meaning}`),
    body: (linked) => `<p>${linkTerms(`${read} ${n.meaning}`, linked)}</p>`,
  };
}), 'notation-group');

export default {
  id: 'notation',
  title: 'The symbols of this course',
  trace: [],
  template: 3,
  conceptual: true,
  sources: [
    { title: 'OpenStax, Calculus Volume 1', url: 'https://openstax.org/details/books/calculus-volume-1' },
    { title: 'A. Zhang, Z. Lipton, M. Li, A. Smola, Dive into Deep Learning', url: 'https://d2l.ai/' },
  ],
  sections: [
    {
      kind: 'text',
      html: () => `<section class="hero"><h1>The symbols of this course</h1>
    <p class="lead">This page lists each symbol of the course with its name, how to read it aloud and what it means. Each symbol stands under the lesson that introduces it.</p></section>`,
    },
    ...GROUPS.map((html) => ({ kind: 'text', html: () => html })),
    {
      kind: 'text',
      html: () => '<section class="notation-group"><p>The next lesson, <a href="../glossary/">Glossary</a>, explains each term of the course in one plain sentence.</p></section>',
    },
  ],
};
