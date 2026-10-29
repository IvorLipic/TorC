// lesson-boot.js — loaded by every lessons/<id>/index.html shell. Loads course.json, the lesson module, its trace data, the
// course-wide anchors (data/anchors.json, so T.anchor works in every lesson) and the glossary (glossary.json, for the term
// popovers), renders the lesson's sections in order (with the next lesson from neighbors() for the "answer" section), then wires
// the widgets, quizzes, explain prompts, glossary popovers, the "mark done" toggle and prev/next links.
// The anchors and the glossary are optional for a page: if one fails to load, a lesson that needs it fails with a clear message
// (T.anchor throws), and the term links stay plain links.
import { $, el, escapeHtml } from './core/dom.js';
import { SITE, siteUrl, headerHtml } from './core/chrome.js';
import { loadCourse, lessonById, neighbors, lessonHref } from './core/course.js';
import { loadTrace, loadAnchors, makeT } from './core/trace.js';
import { renderLesson, validateLesson, defaultFooter } from './core/sections.js';
import { store } from './core/store.js';
import { wireQuizzes, wireExplains } from './core/quiz.js';
import { wireTerms } from './core/popover.js';

function fail(app, message) {
  const main = $('main', app) ?? app.appendChild(el('main'));
  main.replaceChildren(el('div', { class: 'load-error', role: 'alert' }, message));
}

function navHtml(course, id) {
  const { prev, next } = neighbors(course, id);
  const link = (l, cls, label) => (l
    ? `<a class="btn ${cls}" href="${escapeHtml(siteUrl(lessonHref(l.id)))}" rel="${cls}">${label}<span class="t">${escapeHtml(l.title)}</span></a>`
    : `<span class="${cls} none"></span>`);
  return `<nav class="lesson-nav" aria-label="Lesson navigation">
    ${link(prev, 'prev', '← ')}
    <button type="button" class="btn done-toggle" id="done-toggle" aria-pressed="false"></button>
    ${link(next, 'next', '')}
  </nav>`;
}

function wireDone(id) {
  const b = $('#done-toggle');
  const paint = () => {
    const done = store.isDone(id);
    b.setAttribute('aria-pressed', String(done));
    b.textContent = done ? '✓ Done — click to undo' : 'Mark as done';
    b.classList.toggle('on', done);
  };
  b.addEventListener('click', () => { store.setDone(id, !store.isDone(id)); paint(); });
  paint();
}

// Formula and code boxes in the prose carry tabindex="0" so a keyboard user can scroll them when they are wider than the screen
// (axe: scrollable-region-focusable). A box that does not scroll should not be a tab stop, so the tabindex follows the width.
function wireScrollStops(root) {
  const boxes = [...root.querySelectorAll('[role="group"][tabindex]')].filter((e) => !e.closest('[data-widget]'));
  const fit = () => boxes.forEach((e) => (e.scrollWidth > e.clientWidth + 1 ? e.setAttribute('tabindex', '0') : e.removeAttribute('tabindex')));
  fit();
  document.fonts?.ready.then(fit);   // KaTeX fonts can arrive after boot and make a box wider than the screen
  if (typeof ResizeObserver !== 'undefined') { const ro = new ResizeObserver(fit); boxes.forEach((e) => ro.observe(e)); }
}

// Legacy check sections: the "I got it" toggle, stored as `seen` in the v2 store.
function wireChecks(root, id) {
  for (const b of root.querySelectorAll('button[data-check]')) {
    const key = b.dataset.check;
    const paint = () => { const on = store.get(id).seen[key] === true; b.setAttribute('aria-pressed', String(on)); b.textContent = on ? '✓ Got it' : 'I got it'; };
    b.addEventListener('click', () => { store.setSeen(id, key, store.get(id).seen[key] !== true); paint(); });
    paint();
  }
}

async function optionalJson(promise, what) {
  try { return await promise; } catch (err) { console.warn(`${what} could not be loaded: ${err.message}`); return null; }
}

(async function boot() {
  const app = $('#app');
  const id = document.body.dataset.lesson;
  app.innerHTML = headerHtml('lesson') + '<main id="top"></main>';
  try {
    const course = await loadCourse(new URL('course.json', SITE));
    const entry = lessonById(course, id);
    if (!entry) throw new Error(`"${id}" is not a lesson in course.json`);
    const lesson = (await import(`./lessons/${id}.js`)).default;
    const errs = validateLesson(lesson);
    if (errs.length) throw new Error(`lessons/${id}.js: ${errs.join('; ')}`);
    const anchorsP = optionalJson(loadAnchors(SITE), 'data/anchors.json');
    const glossaryP = optionalJson(fetch(new URL('glossary.json', SITE)).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }), 'glossary.json');
    let D;
    try { D = await loadTrace(lesson.trace, SITE); }
    catch (err) { throw new Error(`Could not load the trace data (${lesson.trace.join(', ')}). Serve the site over HTTP, for example: python3 -m http.server -d site. ${err.message}`); }
    D.anchors = await anchorsP;

    const main = $('main', app);
    const parts = renderLesson(lesson, D, { course, next: neighbors(course, id).next });
    main.innerHTML = parts.map((p) => p.html).join('\n') + navHtml(course, id);
    for (const { section } of parts) if (section.kind === 'widget') section.mount($(`[data-widget="${CSS.escape(section.id)}"]`, main), D, { lessonId: id, store, makeT: () => makeT(D) });
    wireScrollStops(main);
    wireChecks(main, id);
    wireQuizzes(main, { lessonId: id, store });
    wireExplains(main, { lessonId: id, store });
    wireDone(id);
    app.append(el('footer', { html: lesson.footer ? lesson.footer(D, makeT(D)) : defaultFooter(lesson, D) }));
    store.markVisited(id);
    const glossary = await glossaryP;
    if (glossary) wireTerms(main, glossary);
  } catch (err) {
    console.error(err);
    fail(app, `This lesson could not be loaded. ${err.message}`);
  }
})();
