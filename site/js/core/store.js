// core/store.js — learner progress, schema v2 (R10). localStorage key "torc:v2" holds
//   { "<lessonId>": { visited: bool, done: bool, quiz: { "<quizId>": { tries: int >= 0, correct: bool } },
//                     explained: { "<explainId>": true }, seen: { "<checkId>": true } } }
// quiz.correct is the result of the latest attempt; tries counts every attempt. The store records what the learner did (an attempt
// and whether it was right), never a self-report. `seen` holds the legacy "I got it" marks of predict/check sections that are not
// migrated yet. On first load, v1 data ("torc:v1", { visited, checks, done }) migrates: v1 checks become `seen`, never `correct`.
// The v1 key is left as it is.
// Every access is wrapped in try/catch: storage may be absent, blocked, full, or throw on first touch (private
// windows, "block site data", sandboxed iframes). When it fails the store keeps working in memory for the
// life of the page, so nothing on the site depends on persistence. No accounts, no network, no analytics.

export const STORAGE_KEY = 'torc:v2';
export const LEGACY_KEY = 'torc:v1';
const blank = () => ({ visited: false, done: false, quiz: {}, explained: {}, seen: {} });
const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const okKey = (k) => k !== '__proto__';
const trueMap = (m) => {
  const out = {};
  if (isObj(m)) for (const [k, b] of Object.entries(m)) if (b === true && okKey(k)) out[k] = true;
  return out;
};

// Drops anything that is not the documented v2 shape (hand-edited or older data must never break a page).
export function sanitize(raw) {
  const out = {};
  if (!isObj(raw)) return out;
  for (const [id, v] of Object.entries(raw)) {
    if (!isObj(v) || !okKey(id)) continue;
    const quiz = {};
    if (isObj(v.quiz)) for (const [k, q] of Object.entries(v.quiz)) {
      if (!okKey(k) || !isObj(q) || !Number.isInteger(q.tries) || q.tries < 0) continue;
      quiz[k] = { tries: q.tries, correct: q.correct === true && q.tries > 0 };
    }
    out[id] = { visited: v.visited === true, done: v.done === true, quiz, explained: trueMap(v.explained), seen: trueMap(v.seen) };
  }
  return out;
}

// v1 { visited, checks: {id: bool}, done } -> v2. A v1 "I got it" was a self-report, so it becomes `seen`, never `correct`.
export function migrateV1(raw) {
  const out = {};
  if (!isObj(raw)) return out;
  for (const [id, v] of Object.entries(raw)) {
    if (!isObj(v) || !okKey(id)) continue;
    out[id] = { visited: v.visited === true, done: v.done === true, quiz: {}, explained: {}, seen: trueMap(v.checks) };
  }
  return out;
}

// getStorage is a function so that even *reading* window.localStorage (which can throw) happens inside try/catch.
export function createStore(getStorage = () => globalThis.localStorage) {
  let memory = null;      // set once a read or write has failed; the in-memory copy is then authoritative
  let persistent = true;

  function save(all) {
    try { getStorage().setItem(STORAGE_KEY, JSON.stringify(all)); }
    catch { persistent = false; memory = all; }
  }
  function load() {
    if (memory) return memory;
    try {
      const s = getStorage();
      const raw = s.getItem(STORAGE_KEY);
      if (raw !== null && raw !== undefined) return sanitize(JSON.parse(raw));
      const old = s.getItem(LEGACY_KEY);
      if (old === null || old === undefined) return {};
      let v1;
      try { v1 = JSON.parse(old); } catch { return {}; }   // unreadable v1 data is ignored, not fatal
      const all = migrateV1(v1);
      save(all);
      return memory ?? all;
    } catch {
      persistent = false;
      memory = {};
      return memory;
    }
  }
  const entry = (all, id) => (Object.hasOwn(all, id) ? all[id] : blank());   // ids like "constructor" are not inherited
  function update(id, fn) {
    if (!okKey(id)) return blank();
    const all = load();
    const cur = entry(all, id);
    fn(cur);
    all[id] = cur;
    save(all);
    return cur;
  }
  const safe = (k) => okKey(String(k));

  return {
    get persistent() { load(); return persistent; },
    all: () => load(),
    get: (id) => entry(load(), id),
    isDone: (id) => entry(load(), id).done,
    markVisited: (id) => update(id, (p) => { p.visited = true; }),
    setDone: (id, done) => update(id, (p) => { p.visited = true; p.done = !!done; }),
    // one quiz attempt; returns the quiz record {tries, correct}
    recordQuiz: (id, quizId, correct) => update(id, (p) => {
      if (!safe(quizId)) return;
      const q = Object.hasOwn(p.quiz, quizId) ? p.quiz[quizId] : { tries: 0, correct: false };
      p.visited = true;
      p.quiz[quizId] = { tries: q.tries + 1, correct: !!correct };
    }).quiz[quizId] ?? { tries: 0, correct: false },
    quiz: (id, quizId) => { const q = entry(load(), id).quiz; return Object.hasOwn(q, quizId) ? q[quizId] : { tries: 0, correct: false }; },
    setExplained: (id, explainId) => update(id, (p) => { if (safe(explainId)) { p.visited = true; p.explained[explainId] = true; } }),
    // the legacy "I got it" toggle of a check section (stored as seen)
    setSeen: (id, key, on) => update(id, (p) => { if (!safe(key)) return; p.visited = true; if (on) p.seen[key] = true; else delete p.seen[key]; }),
  };
}

// The shared instance used by pages.
export const store = createStore();
