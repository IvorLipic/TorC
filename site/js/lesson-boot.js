// lesson-boot.js — loaded by every lessons/<id>/index.html shell. Loads course.json, the lesson module and its trace data,
// renders the lesson's sections in order, then the "mark done" toggle and prev/next links.
import { $, el, escapeHtml } from './core/dom.js';
import { SITE, siteUrl, headerHtml } from './core/chrome.js';
import { loadCourse, lessonById, neighbors, lessonHref } from './core/course.js';
import { loadTrace, makeT } from './core/trace.js';
import { renderLesson, validateLesson } from './core/sections.js';
import { store } from './core/store.js';

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

function wireChecks(root, id) {
  for (const b of root.querySelectorAll('button[data-check]')) {
    const key = b.dataset.check;
    const paint = () => { const on = store.get(id).checks[key] === true; b.setAttribute('aria-pressed', String(on)); b.textContent = on ? '✓ Got it' : 'I got it'; };
    b.addEventListener('click', () => { store.setCheck(id, key, store.get(id).checks[key] !== true); paint(); });
    paint();
  }
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
    let D;
    try { D = await loadTrace(lesson.trace, SITE); }
    catch (err) { throw new Error(`Could not load the trace data (${lesson.trace.join(', ')}). Serve the site over HTTP, for example: python3 -m http.server -d site. ${err.message}`); }

    const main = $('main', app);
    const parts = renderLesson(lesson, D);
    main.innerHTML = parts.map((p) => p.html).join('\n') + navHtml(course, id);
    for (const { section } of parts) if (section.kind === 'widget') section.mount($(`[data-widget="${CSS.escape(section.id)}"]`, main), D, { lessonId: id, store, makeT: () => makeT(D) });
    wireChecks(main, id);
    wireDone(id);
    if (lesson.footer) app.append(el('footer', { html: lesson.footer(D, makeT(D)) }));
    store.markVisited(id);
  } catch (err) {
    console.error(err);
    fail(app, `This lesson could not be loaded. ${err.message}`);
  }
})();
