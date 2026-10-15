// home.js — the home page: the curriculum (numbered lessons by part, each with its TorC code beside it; planned lessons shown
// but not linked) and the labs. Old single-page deep links such as "#mlp/9" are forwarded to the page that claims them
// (legacy_hashes in course.json).
import { $, el, escapeHtml } from './core/dom.js';
import { SITE, siteUrl, headerHtml } from './core/chrome.js';
import { loadCourse, curriculum, labs, isPublished, legacyTarget, lessonHref } from './core/course.js';
import { store } from './core/store.js';

// C++ with its // comments dimmed (escaped first, so the markup cannot leak)
const codeHtml = (code) => escapeHtml(code).split('\n').map((line) => line.replace(/(\/\/.*)$/, '<span class="c">$1</span>')).join('\n');

function row(l) {
  const live = isPublished(l), done = live && store.isDone(l.id);
  const title = live
    ? el('a', { class: 'row-link', href: siteUrl(lessonHref(l.id)) }, l.title)
    : el('span', {}, l.title);
  return el('li', { class: `row${live ? '' : ' planned'}${done ? ' done' : ''}` },
    el('div', { class: 'row-text' },
      l.number ? el('span', { class: 'row-num', 'aria-hidden': 'true' }, String(l.number).padStart(2, '0')) : null,
      el('div', {},
        el('h3', { class: 'row-title' }, title,
          done ? el('span', { class: 'tick', title: 'Marked as done' }, el('span', { 'aria-hidden': 'true' }, ' ✓'), el('span', { class: 'sr' }, ' (done)')) : null),
        el('p', { class: 'row-outcome' }, l.outcome),
        el('p', { class: 'row-meta' },
          live ? `${l.est_minutes} min` : el('span', { class: 'tag' }, 'planned'),
          ...(l.widgets ?? []).map((w) => el('span', { class: 'widget-tag' }, w))))),
    l.code ? el('pre', { class: 'row-code', role: 'region', tabindex: '0', 'aria-label': `TorC code: ${l.title}` }, el('code', { html: codeHtml(l.code) })) : null);
}

function render(course) {
  const parts = curriculum(course), labList = labs(course);
  const published = parts.flatMap((p) => p.lessons).filter(isPublished);
  const done = published.filter((l) => store.isDone(l.id)).length;
  $('#progress').textContent = `${done} of ${published.length} lessons done` + (store.persistent ? '' : ' · progress is not saved (browser storage is unavailable)');
  const first = published[0];
  $('#start').replaceChildren(first ? el('a', { class: 'btn primary big', href: siteUrl(lessonHref(first.id)) }, `Start: ${first.title} →`) : '');
  $('#map').replaceChildren(...parts.map((p, i) => el('section', { class: 'part', 'aria-labelledby': `part-${p.id}` },
    el('h2', { id: `part-${p.id}` }, el('span', { class: 'part-num' }, `${i + 1}`), p.title),
    el('ol', { class: 'rows' }, p.lessons.map(row)))));
  $('#labs').replaceChildren(...(labList.length ? [
    el('h2', { id: 'labs-h' }, 'Labs'),
    el('p', { class: 'hint' }, 'Playgrounds outside the lesson order.'),
    el('ul', { class: 'rows' }, labList.map(row))] : []));
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
  <section class="hero home-hero">
    <h1>Deep learning, from the gradient up</h1>
    <p class="lead">A course in the usual order, from one gradient to a convolutional network, built on <a href="https://github.com/IvorLipic/TorC">TorC</a>, a small C++ tensor and autograd library. Each lesson pairs the real code with something to play with.</p>
    <div class="hero-actions"><span id="start"></span><span class="hint" id="progress"></span></div>
  </section>
  <section class="course" aria-label="Curriculum"><div id="map"></div></section>
  <section class="labs" id="labs" aria-labelledby="labs-h"></section>
</main>
<footer><p>Numbers on every page come from real runs of TorC; the code beside each lesson compiles against the pinned release. <a href="https://github.com/IvorLipic/TorC">Source</a>. Progress is kept only in this browser.</p></footer>`;
  render(course);
  // a lesson marked done in another tab, or a back/forward-cache restore, should be reflected here
  window.addEventListener('pageshow', (e) => { if (e.persisted) render(course); });
})();
