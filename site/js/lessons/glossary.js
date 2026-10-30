// lessons/glossary.js — the glossary page of the appendix "Using TorC" (brief §8 "Glossary", R9; decision 18). Built from
// site/glossary.json, the one source of the definitions that the term() popovers of every lesson show. One entry per term, with an
// anchor #<key> that term(key) links to, grouped by the lesson that introduces the term, in course order.
// The page is static: it is rendered once at import time. Every term that a definition mentions is linked with term() at its
// first use on the page (the v3 glossary rule, site/test/v3.mjs), so a reader can follow a definition to the next one.
// The JSON is imported as a JSON module; lesson-boot.js still loads glossary.json itself for the popovers.
import GLOSSARY from '../../glossary.json' with { type: 'json' };
import COURSE from '../../course.json' with { type: 'json' };
import { term } from '../core/blocks.js';
import { escapeHtml } from '../core/dom.js';
import { flatLessons, isPublished } from '../core/course.js';

const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const formRe = (forms, flags = 'giu') => new RegExp(`(?<![\\p{L}\\p{N}])(?:${[...forms].sort((a, b) => b.length - a.length).map(escRe).join('|')})(?![\\p{L}\\p{N}])`, flags);

// Links the first use on a page of every glossary term in `text` (plain text, escaped here) with term(). `linked` is the set of
// keys already linked on the page; it grows. A longer term that contains a shorter one ("weight decay" holds "weight") is linked
// only after the shorter one, because the v3 rule wants the first use of "weight" inside a link to "weight".
export function linkTerms(text, linked, glossary = GLOSSARY) {
  const terms = Object.entries(glossary.terms);
  const all = formRe(terms.flatMap(([, g]) => g.forms));
  const keysOf = (s) => terms.filter(([, g]) => formRe(g.forms, 'iu').test(s) && g.forms.some((f) => f.toLowerCase() === s.toLowerCase())).map(([k]) => k);
  let out = '', at = 0;
  for (const m of text.matchAll(all)) {
    const s = m[0], [key] = keysOf(s);
    out += escapeHtml(text.slice(at, m.index));
    at = m.index + s.length;
    // a shorter term inside this match that is not linked yet goes first
    const inner = terms.filter(([k, g]) => k !== key && !linked.has(k)).map(([k, g]) => [k, formRe(g.forms, 'iu').exec(s)]).find(([, x]) => x);
    if (inner) {
      const [k, x] = inner;
      linked.add(k);
      out += escapeHtml(s.slice(0, x.index)) + term(k, escapeHtml(x[0])) + escapeHtml(s.slice(x.index + x[0].length));
    } else if (key && !linked.has(key)) {
      linked.add(key);
      out += term(key, escapeHtml(s));
    } else out += escapeHtml(s);
  }
  return out + escapeHtml(text.slice(at));
}

const lessons = flatLessons(COURSE);
const titleOf = (id) => lessons.find((l) => l.id === id)?.title ?? id;
const lessonHead = (id) => {
  const l = lessons.find((x) => x.id === id);
  return l && isPublished(l) ? `<a href="../${escapeHtml(id)}/">${escapeHtml(l.title)}</a>` : escapeHtml(titleOf(id));
};
export const words = (s) => s.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
const BUDGET = 140;   // a text section holds at most 150 words (v3 budget); headings do not count

// A reference page body: one group per introducing lesson in course order, split into text sections of at most BUDGET words.
// entries: [{ key, lesson, head (html), body (linked, text) => html, words }], in the order to show them inside a group.
// `linked` is shared by the whole page, so each glossary term is linked at its first use on the page only.
export function referenceGroups(entries, cls) {
  const linked = new Set();
  const order = lessons.map((l) => l.id);
  const byLesson = new Map();
  for (const e of entries) {
    if (!byLesson.has(e.lesson)) byLesson.set(e.lesson, []);
    byLesson.get(e.lesson).push(e);
  }
  const ids = [...byLesson.keys()].sort((a, b) => (order.indexOf(a) + 1 || 1e9) - (order.indexOf(b) + 1 || 1e9));
  const sections = [];
  for (const id of ids) {
    let body = '', n = 0, first = true;
    const flush = () => {
      if (!body) return;
      const hid = `${cls}-${escapeHtml(id)}`;
      sections.push(first ? `<section class="${cls}" aria-labelledby="${hid}"><h2 id="${hid}">${lessonHead(id)}</h2>${body}</section>` : `<section class="${cls}">${body}</section>`);
      body = ''; n = 0; first = false;
    };
    for (const e of byLesson.get(id)) {
      if (n && n + e.words > BUDGET) flush();
      body += `<div class="entry"><h3 id="${escapeHtml(e.key)}">${e.head}</h3>${e.body(linked)}</div>`;
      n += e.words;
    }
    flush();
  }
  return sections;
}

const GROUPS = referenceGroups(Object.entries(GLOSSARY.terms).sort(([, a], [, b]) => a.term.localeCompare(b.term, 'en')).map(([key, g]) => ({
  key, lesson: g.lesson, head: escapeHtml(g.term), words: words(g.short), body: (linked) => `<p>${linkTerms(g.short, linked)}</p>`,
})), 'glossary-group');

export default {
  id: 'glossary',
  title: 'Glossary',
  trace: [],
  template: 3,
  conceptual: true,
  sources: [
    { title: 'Google for Developers, Machine Learning Glossary', url: 'https://developers.google.com/machine-learning/glossary' },
    { title: 'A. Zhang, Z. Lipton, M. Li, A. Smola, Dive into Deep Learning', url: 'https://d2l.ai/' },
  ],
  sections: [
    {
      kind: 'text',
      html: () => `<section class="hero"><h1>Glossary</h1>
    <p class="lead">This page explains each term of the course in one plain sentence. Each term stands under the lesson that introduces it.</p></section>`,
    },
    ...GROUPS.map((html) => ({ kind: 'text', html: () => html })),
  ],
};
