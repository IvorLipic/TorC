// home.js — the home page, in this order: the promise and a live demo (the digit pad of widgets/digit-widgets.js, which runs
// the forward pass of the MLP that TorC trained; its data loads only after the text is on screen), the prerequisites and the
// total time of the main path, the curriculum (numbered lessons by part, progress ticks, a filter box, each lesson's TorC code
// behind a closed "Show the TorC code" toggle; planned lessons shown but not linked), the appendix "Using TorC" and the labs.
// Old single-page deep links such as "#mlp/9" are forwarded to the page that claims them (legacy_hashes in course.json).
import { $, el, escapeHtml } from './core/dom.js';
import { SITE, siteUrl, headerHtml } from './core/chrome.js';
import { loadCourse, curriculum, appendix, labs, isPublished, isPlanned, legacyTarget, lessonHref, filterLessons, studyTime, aboutHours } from './core/course.js';
import { store } from './core/store.js';

// C++ with its // comments dimmed (escaped first, so the markup cannot leak)
const codeHtml = (code) => escapeHtml(code).split('\n').map((line) => line.replace(/(\/\/.*)$/, '<span class="c">$1</span>')).join('\n');

function row(l, h = 'h3') {
  const live = isPublished(l), done = live && store.isDone(l.id);
  const title = live
    ? el('a', { class: 'row-link', href: siteUrl(lessonHref(l.id)) }, l.title)
    : el('span', {}, l.title);
  return el('li', { class: `row${live ? '' : ' planned'}${done ? ' done' : ''}` },
    l.number ? el('span', { class: 'row-num', 'aria-hidden': 'true' }, String(l.number).padStart(2, '0')) : null,
    el('div', { class: 'row-body' },
      el(h, { class: 'row-title' }, title,
        done ? el('span', { class: 'tick', title: 'Marked as done' }, el('span', { 'aria-hidden': 'true' }, ' ✓'), el('span', { class: 'sr' }, ' (done)')) : null),
      el('p', { class: 'row-outcome' }, l.outcome),
      el('p', { class: 'row-meta' },
        live ? null : el('span', { class: 'tag' }, 'planned'),
        `${l.est_minutes} min`,
        ...(l.widgets ?? []).map((w) => el('span', { class: 'widget-tag' }, w))),
      l.code ? el('details', { class: 'row-code' },
        el('summary', {}, 'Show the TorC code'),
        el('pre', { role: 'group', tabindex: '0', 'aria-label': `TorC code: ${l.title}` }, el('code', { html: codeHtml(l.code) }))) : null));
}

const partNumber = new Map();   // part id -> its number on the full curriculum (kept while the filter hides parts)

function renderProgress(course) {
  const published = curriculum(course).flatMap((p) => p.lessons).filter(isPublished);
  const done = published.filter((l) => store.isDone(l.id)).length;
  $('#progress').textContent = `${done} of ${published.length} lessons done` + (store.persistent ? '' : ' · progress is not saved (browser storage is unavailable)');
  const first = published[0];
  $('#start').replaceChildren(first ? el('a', { class: 'btn primary big', href: siteUrl(lessonHref(first.id)) }, `Start: ${first.title} →`) : '');
}

function renderLists(course, query) {
  const parts = curriculum(course).map((p) => ({ ...p, lessons: filterLessons(p.lessons, query) })).filter((p) => p.lessons.length);
  const app = filterLessons(appendix(course), query), labList = filterLessons(labs(course), query);
  const count = parts.reduce((n, p) => n + p.lessons.length, 0) + app.length + labList.length;
  $('#map').replaceChildren(...parts.map((p) => el('section', { class: 'part', 'aria-labelledby': `part-${p.id}` },
    el('h3', { class: 'part-h', id: `part-${p.id}` }, el('span', { class: 'part-num' }, String(partNumber.get(p.id))), p.title),
    el('ol', { class: 'rows', start: String(p.lessons[0].number) }, p.lessons.map((l) => row(l, 'h4'))))));
  if (!count) $('#map').append(el('p', { class: 'hint empty' }, `No page matches “${query.trim()}”.`));
  $('#filter-status').textContent = query.trim() ? `${count} ${count === 1 ? 'page matches' : 'pages match'}.` : '';
  $('#appendix').replaceChildren(...(app.length ? [
    el('h2', { id: 'appendix-h' }, 'Using TorC'),
    el('p', { class: 'hint' }, 'These pages are about the TorC library itself. They are not part of the lesson order. Read them when you want to write your own TorC programs.'),
    el('ul', { class: 'rows' }, app.map((l) => row(l)))] : []));
  $('#labs').replaceChildren(...(labList.length ? [
    el('h2', { id: 'labs-h' }, 'Labs'),
    el('p', { class: 'hint' }, 'The labs are not part of the lesson order. You can use them at any time.'),
    el('ul', { class: 'rows' }, labList.map((l) => row(l)))] : []));
}

// The digit pad: the module and the 400 kB of weights load when the demo comes near the screen, after the text is drawn.
function startDemo() {
  const host = $('#demo-pad');
  let started = false;
  const start = async () => {
    if (started) return;
    started = true;
    try {
      const [{ mountDigitPad }, { loadTrace }] = await Promise.all([import('./widgets/digit-widgets.js'), import('./core/trace.js')]);
      const D = await loadTrace(['mnist-mlp.json'], SITE);
      mountDigitPad(host, D);
    } catch (err) {
      host.replaceChildren(el('p', { class: 'hint', role: 'alert' }, `The page could not load the network. The lessons work without it. (${err.message})`));
    }
  };
  if (typeof IntersectionObserver === 'function') {
    const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { io.disconnect(); start(); } }, { rootMargin: '400px 0px' });
    io.observe(host);
  } else setTimeout(start, 0);
}

(async function main() {
  const app = $('#app');
  let course;
  try { course = await loadCourse(new URL('course.json', SITE)); }
  catch (err) { app.innerHTML = headerHtml('home') + '<main id="top"></main>'; $('main', app).append(el('p', { class: 'load-error', role: 'alert' }, 'Could not load the course map. If you opened this file directly, serve the site over HTTP (python3 -m http.server -d site). ' + err.message)); return; }

  const target = legacyTarget(course, location.hash);
  if (target) { location.replace(new URL(target, SITE).href); return; }

  curriculum(course).forEach((p, i) => partNumber.set(p.id, i + 1));
  const time = studyTime(course);
  const planned = curriculum(course).flatMap((p) => p.lessons).filter(isPlanned).length;
  app.innerHTML = headerHtml('home') + `
<main id="top">
  <div class="home-top">
    <section class="hero home-hero" aria-labelledby="home-h">
      <h1 id="home-h">How a machine learns</h1>
      <p class="lead">This course explains how a machine learns from examples. You start with one gradient, a number that tells a model how to improve. You finish with a network that reads handwritten digits. At the end, you see how ChatGPT learns with the same method, on a much larger scale.</p>
      <div class="hero-actions"><span id="start"></span><span class="hint" id="progress"></span></div>
    </section>
    <section class="demo" aria-labelledby="demo-h">
      <h2 id="demo-h">Draw a digit</h2>
      <p class="demo-text">Draw a digit from 0 to 9 in the black square. A network that TorC trained reads your drawing and shows the probability of each digit. The buttons show handwritten digits that the network did not see in training. Later in the course, you build and train this network yourself.</p>
      <div id="demo-pad" class="demo-pad"><p class="hint">The page is loading the network.</p></div>
    </section>
    <section class="before" aria-labelledby="before-h">
      <h2 id="before-h">Before you start</h2>
      <p>You need algebra and graphs of functions. We explain everything else when you need it, including derivatives and matrices.</p>
      <p>The course uses TorC, a small library for machine learning in the C++ programming language. You do not need to know C++, because each lesson explains its ideas in words, pictures and parts that you can change.</p>
      <p id="time">The ${time.lessons} lessons take ${aboutHours(time.minutes)} in total. ${planned ? `Of these lessons, ${planned} have no page yet. The course lists them so that you can see the whole plan.` : ''}</p>
    </section>
  </div>
  <section class="course" aria-labelledby="course-h">
    <h2 id="course-h">The lessons</h2>
    <div class="filter"><label for="filter">Find a page</label>
      <input id="filter" type="search" placeholder="a word, for example gradient" autocomplete="off">
      <span id="filter-status" class="hint" role="status" aria-live="polite"></span></div>
    <div id="map"></div>
  </section>
  <section class="appendix" id="appendix" aria-labelledby="appendix-h"></section>
  <section class="labs" id="labs" aria-labelledby="labs-h"></section>
</main>
<footer><p>Tests check every number on these pages against real runs of TorC. The TorC code of each lesson compiles and runs with the TorC release that the site uses. Only this browser stores your progress. <a href="https://github.com/IvorLipic/TorC">Source</a>.</p></footer>`;
  renderProgress(course);
  renderLists(course, '');
  $('#filter').addEventListener('input', (e) => renderLists(course, e.target.value));
  startDemo();
  // a lesson marked done in another tab, or a back/forward-cache restore, should be reflected here
  window.addEventListener('pageshow', (e) => { if (e.persisted) { renderProgress(course); renderLists(course, $('#filter').value); } });
})();
