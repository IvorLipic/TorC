// core/course.js — the course map (site/course.json): loading, validation, ordering, filtering. Pure functions
// (no DOM) so Node tests and tools/check_site.mjs use exactly the code the browser uses.
//
// course.json:  { schema: "torc-course/1", title, parts: [ { id, title, kind?: "lab", lessons: [ {
//   id, title, outcome, prereqs: [id], symbols: [string], trace: [file in site/data/], est_minutes,
//   status: "published" | "draft" | "planned", widgets?: [string], code?: string, legacy_hashes?: [string] } ] } ] }
// A part with kind "lab" holds labs (playgrounds outside the lesson sequence). "planned" lessons have no page yet; the home page
// shows them as the rest of the curriculum. Drafts have a page reachable by URL but are listed nowhere.

export const STATUSES = ['published', 'draft', 'planned'];
export const isPublished = (l) => l.status === 'published';
export const isPlanned = (l) => l.status === 'planned';
export const isLabPart = (p) => p.kind === 'lab';
export const lessonHref = (id) => `lessons/${id}/`;          // relative to the site root

export function flatLessons(course) {
  return (course.parts ?? []).flatMap((p) => (p.lessons ?? []).map((l) => ({ ...l, partId: p.id, partTitle: p.title })));
}
export const lessonById = (course, id) => flatLessons(course).find((l) => l.id === id) ?? null;

// Home page, the curriculum: lesson parts (not labs) with their published and planned lessons, numbered in course order
// (drafts are left out and take no number); parts with nothing to show are omitted.
export function curriculum(course) {
  let n = 0;
  return (course.parts ?? []).filter((p) => !isLabPart(p))
    .map((p) => ({ id: p.id, title: p.title, lessons: (p.lessons ?? []).filter((l) => isPublished(l) || isPlanned(l)).map((l) => ({ ...l, number: ++n })) }))
    .filter((p) => p.lessons.length > 0);
}
// Home page, the labs: published lessons of the lab parts.
export const labs = (course) => (course.parts ?? []).filter(isLabPart).flatMap((p) => (p.lessons ?? []).filter(isPublished));

// Previous/next among published pages of the same kind (lessons with lessons, labs with labs), in course order; planned lessons
// have no page and are skipped. A draft (reachable by URL only) is placed by its own position.
export function neighbors(course, id) {
  const kindOf = new Map((course.parts ?? []).flatMap((p) => (p.lessons ?? []).map((l) => [l.id, isLabPart(p)])));
  const list = flatLessons(course).filter((l) => (isPublished(l) || l.id === id) && kindOf.get(l.id) === kindOf.get(id));
  const i = list.findIndex((l) => l.id === id);
  if (i < 0) return { prev: null, next: null };
  return { prev: list[i - 1] ?? null, next: list[i + 1] ?? null };
}

// Old deep links such as "#mlp/9" were served by the home page when the site was a single lab.
// Returns the relative URL to redirect to (from the site root), or null.
export function legacyTarget(course, hash) {
  const head = String(hash ?? '').replace(/^#/, '').split('/')[0];
  if (!head) return null;
  const l = flatLessons(course).find((x) => (x.legacy_hashes ?? []).includes(head));
  return l ? lessonHref(l.id) + hash : null;
}

const isStr = (x) => typeof x === 'string' && x.trim() !== '';
const ID_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

// Structural validation. Returns a list of human-readable problems (empty = valid).
export function validateCourse(course) {
  const errs = [];
  if (!course || typeof course !== 'object') return ['course.json: not an object'];
  if (!isStr(course.title)) errs.push('course.json: "title" missing');
  if (!Array.isArray(course.parts) || !course.parts.length) return [...errs, 'course.json: "parts" must be a non-empty array'];
  const ids = new Set(), partIds = new Set();
  for (const p of course.parts) {
    if (!isStr(p.id) || !isStr(p.title) || !Array.isArray(p.lessons)) { errs.push(`part ${JSON.stringify(p?.id)}: needs id, title, lessons[]`); continue; }
    if (p.kind !== undefined && p.kind !== 'lab') errs.push(`part "${p.id}": kind must be "lab" or absent`);
    if (partIds.has(p.id)) errs.push(`duplicate part id "${p.id}"`);
    partIds.add(p.id);
    for (const l of p.lessons) {
      const where = `lesson "${l?.id}"`;
      if (!isStr(l.id) || !ID_RE.test(l.id)) { errs.push(`${where}: id must be lowercase-hyphenated`); continue; }
      if (ids.has(l.id)) errs.push(`duplicate lesson id "${l.id}"`);
      ids.add(l.id);
      if (!isStr(l.title)) errs.push(`${where}: "title" missing`);
      if (!isStr(l.outcome)) errs.push(`${where}: "outcome" missing`);
      if (!STATUSES.includes(l.status)) errs.push(`${where}: status must be one of ${STATUSES.join(', ')}`);
      if (!Number.isFinite(l.est_minutes) || l.est_minutes <= 0) errs.push(`${where}: est_minutes must be a positive number`);
      for (const key of ['prereqs', 'symbols', 'trace']) if (!Array.isArray(l[key]) || !l[key].every(isStr)) errs.push(`${where}: "${key}" must be an array of strings`);
      if (l.code !== undefined && !isStr(l.code)) errs.push(`${where}: "code" must be a non-empty string`);
      if (l.widgets !== undefined && !(Array.isArray(l.widgets) && l.widgets.every(isStr))) errs.push(`${where}: "widgets" must be an array of strings`);
      if (l.legacy_hashes !== undefined && !(Array.isArray(l.legacy_hashes) && l.legacy_hashes.every(isStr))) errs.push(`${where}: "legacy_hashes" must be an array of strings`);
      for (const t of l.trace ?? []) if (typeof t === 'string' && !/^[A-Za-z0-9_.-]+\.json$/.test(t)) errs.push(`${where}: trace file "${t}" must be a plain *.json file name`);
    }
  }
  const all = flatLessons(course);
  for (const l of all) for (const q of l.prereqs ?? []) {
    if (!ids.has(q)) errs.push(`lesson "${l.id}": prereq "${q}" is not a lesson in course.json`);
    if (q === l.id) errs.push(`lesson "${l.id}": lists itself as a prereq`);
  }
  // prerequisite cycles
  const byId = Object.fromEntries(all.map((l) => [l.id, l]));
  const state = {};
  const visit = (id, path) => {
    if (state[id] === 2 || !byId[id]) return;
    if (state[id] === 1) { errs.push(`prereq cycle: ${[...path, id].join(' -> ')}`); return; }
    state[id] = 1;
    for (const q of byId[id].prereqs ?? []) visit(q, [...path, id]);
    state[id] = 2;
  };
  for (const l of all) visit(l.id, []);
  // legacy hashes must be unique across lessons
  const seen = new Map();
  for (const l of all) for (const h of l.legacy_hashes ?? []) {
    if (seen.has(h) && seen.get(h) !== l.id) errs.push(`legacy hash "${h}" claimed by both "${seen.get(h)}" and "${l.id}"`);
    seen.set(h, l.id);
  }
  return errs;
}

export async function loadCourse(url, fetchFn = fetch) {
  const res = await fetchFn(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const course = await res.json();
  const errs = validateCourse(course);
  if (errs.length) throw new Error('course.json is invalid: ' + errs.join('; '));
  return course;
}
