// home.js — the course map: lessons grouped by part, a filter box, and a tick on lessons marked done.
// Only published lessons are listed (drafts are reachable by URL). Old single-page deep links such as "#mlp/9" are
// forwarded to the lesson that claims them (legacy_hashes in course.json).
import { $, el } from './core/dom.js';
import { SITE, siteUrl, headerHtml } from './core/chrome.js';
import { loadCourse, publishedParts, filterLessons, legacyTarget, lessonHref } from './core/course.js';
import { store } from './core/store.js';

function card(l) {
  const done = store.isDone(l.id);
  return el('li', { class: 'lesson-card' + (done ? ' done' : '') },
    el('a', { class: 'lesson-link', href: siteUrl(lessonHref(l.id)) },
      el('span', { class: 'lesson-title' }, l.title),
      done ? el('span', { class: 'tick', title: 'Marked as done' }, el('span', { 'aria-hidden': 'true' }, '✓'), el('span', { class: 'sr' }, 'Done')) : null),
    el('p', { class: 'lesson-outcome' }, l.outcome),
    el('p', { class: 'lesson-meta' }, `${l.est_minutes} min`));
}

function renderMap(course, query) {
  const map = $('#map');
  const parts = publishedParts(course).map((p) => ({ ...p, lessons: filterLessons(p.lessons, query) })).filter((p) => p.lessons.length);
  const count = parts.reduce((n, p) => n + p.lessons.length, 0);
  map.replaceChildren(...parts.map((p) => el('section', { class: 'part', 'aria-labelledby': 'part-' + p.id },
    el('h2', { id: 'part-' + p.id }, p.title),
    el('ul', { class: 'lesson-grid' }, p.lessons.map(card)))));
  if (!count) map.append(el('p', { class: 'hint empty' }, `No lesson matches “${query}”.`));
  $('#filter-status').textContent = query ? `${count} ${count === 1 ? 'lesson matches' : 'lessons match'}` : '';
}

function renderProgress(course) {
  const all = publishedParts(course).flatMap((p) => p.lessons);
  const done = all.filter((l) => store.isDone(l.id)).length;
  $('#progress').textContent = `${done} of ${all.length} ${all.length === 1 ? 'lesson' : 'lessons'} done`
    + (store.persistent ? '' : ' · progress is not being saved (browser storage is unavailable)');
}

(async function main() {
  const app = $('#app');
  let course;
  try { course = await loadCourse(new URL('course.json', SITE)); }
  catch (err) { app.innerHTML = headerHtml('home') + '<main id="top"></main>'; $('main', app).append(el('p', { class: 'load-error', role: 'alert' }, 'Could not load the course map. If you opened this file directly, serve the site over HTTP (python3 -m http.server -d site). ' + err.message)); return; }

  const target = legacyTarget(course, location.hash);
  if (target) { location.replace(new URL(target, SITE).href); return; }

  app.innerHTML = headerHtml('home') + `
<main id="top">
  <section class="hero">
    <h1>Learn how a tensor library learns</h1>
    <p class="lead">Interactive lessons on <a href="https://github.com/IvorLipic/TorC">TorC</a>, a from-scratch C++ tensor and autograd library. Every number comes from a real run of the library; where you can edit inputs, a tested JavaScript port recomputes them in your browser.</p>
  </section>
  <section class="course" aria-labelledby="course-h">
    <div class="course-head">
      <h2 id="course-h">Course map</h2>
      <p class="hint" id="progress"></p>
    </div>
    <div class="filter"><label for="filter">Filter lessons</label>
      <input id="filter" type="search" placeholder="title, outcome or symbol, e.g. backward" autocomplete="off">
      <span id="filter-status" class="hint" role="status" aria-live="polite"></span></div>
    <div id="map"></div>
  </section>
</main>
<footer><p>Source and the library itself: <a href="https://github.com/IvorLipic/TorC">github.com/IvorLipic/TorC</a>. Progress is kept only in this browser.</p></footer>`;
  renderProgress(course);
  renderMap(course, '');
  $('#filter').addEventListener('input', (e) => renderMap(course, e.target.value));
  // a lesson marked done in another tab, or a back/forward-cache restore, should be reflected here
  window.addEventListener('pageshow', (e) => { if (e.persisted) { renderProgress(course); renderMap(course, $('#filter').value); } });
})();
