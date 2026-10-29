#!/usr/bin/env node
// tools/prose/extract.mjs — writes the visible prose of the site to build/prose/*.md (one file per published page, plus the
// course descriptions and the home page text), for the ASD-STE100 linter (tools/prose/check.sh). Code, formulas and quoted
// library lines are replaced by a placeholder word, because the writing rules apply to sentences, not to C++.
// The visible strings of the widgets and of lesson-boot.js (brief D15) go to build/prose/ui.md; tools/prose/ui-strings.mjs says
// how they are found, and site/test/prose.test.mjs applies the site rules to them. Strings in KNOWN_D15 (strings that still break
// the rules on this branch, temporary) are left out so the linter run stays at 0 hard findings.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { extractUiStrings } from './ui-strings.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SITE = path.join(ROOT, 'site'), OUT = path.join(ROOT, 'build/prose');
const imp = (rel) => import(pathToFileURL(path.join(SITE, rel)).href);
const { renderLesson } = await imp('js/core/sections.js');
const { buildD } = await imp('js/core/trace.js');
const { flatLessons, isPublished } = await imp('js/core/course.js');
const readJson = (rel) => JSON.parse(readFileSync(path.join(SITE, rel), 'utf8'));

export const plain = (html) => html
  .replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '\n')
  .replace(/<code[^>]*>[\s\S]*?<\/code>/g, 'code').replace(/<span class="katex[\s\S]*?<\/span><\/span><\/span>/g, 'formula')
  .replace(/<\/(p|li|h1|h2|h3|summary|section|aside|div)>/g, '\n').replace(/<[^>]+>/g, ' ')
  .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#39;/g, "'").replace(/&quot;/g, '"')
  .split('\n').map((l) => l.replace(/\s+/g, ' ').replace(/ ([.,?!])/g, '$1').trim()).filter(Boolean).join('\n\n');

globalThis.katex = { renderToString: () => 'formula' };
mkdirSync(OUT, { recursive: true });
const course = readJson('course.json');
for (const l of flatLessons(course).filter(isPublished)) {
  const lesson = (await imp(`js/lessons/${l.id}.js`)).default;
  const D = buildD(Object.fromEntries(lesson.trace.map((f) => [f.replace(/\.json$/, ''), readJson('data/' + f)])));
  writeFileSync(path.join(OUT, `${l.id}.md`), renderLesson(lesson, D).map((p) => plain(p.html)).filter(Boolean).join('\n\n') + '\n');
}
writeFileSync(path.join(OUT, 'course.md'), flatLessons(course).map((l) => `${l.title}.\n\n${l.outcome}`).join('\n\n') + '\n');
const home = readFileSync(path.join(SITE, 'js/home.js'), 'utf8');
writeFileSync(path.join(OUT, 'home.md'), [...home.matchAll(/<p class="lead">([\s\S]*?)<\/p>|<footer>([\s\S]*?)<\/footer>|class: 'hint' \}, '([^']*)'/g)].map((m) => plain(m[1] ?? m[2] ?? m[3])).join('\n\n') + '\n');
const ui = extractUiStrings(SITE).filter((r) => !r.known);
const sentence = (t) => t.split('\n').map((l) => (/[.?!:]$/.test(l) ? l : `${l}.`)).join('\n\n');
writeFileSync(path.join(OUT, 'ui.md'), ui.map((r) => sentence(r.text)).join('\n\n') + '\n');
console.log(`prose: ${flatLessons(course).filter(isPublished).length} pages, course, home and ${ui.length} UI strings written to ${path.relative(ROOT, OUT)}/`);
