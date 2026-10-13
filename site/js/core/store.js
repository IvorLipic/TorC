// core/store.js — learner progress. localStorage key "torc:v1" holds
//   { "<lessonId>": { visited: bool, checks: { "<checkId>": bool }, done: bool } }
// Every access is wrapped in try/catch: storage may be absent, blocked, full, or throw on first touch (private
// windows, "block site data", sandboxed iframes). When it fails the store keeps working in memory for the
// life of the page, so nothing on the site depends on persistence. No accounts, no network, no analytics.

export const STORAGE_KEY = 'torc:v1';
const blank = () => ({ visited: false, checks: {}, done: false });
const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);

// Drops anything that is not the documented shape (hand-edited or older data must never break a page).
export function sanitize(raw) {
  const out = {};
  if (!isObj(raw)) return out;
  for (const [id, v] of Object.entries(raw)) {
    if (!isObj(v) || id === '__proto__') continue;
    const checks = {};
    if (isObj(v.checks)) for (const [k, b] of Object.entries(v.checks)) if (typeof b === 'boolean' && k !== '__proto__') checks[k] = b;
    out[id] = { visited: v.visited === true, checks, done: v.done === true };
  }
  return out;
}

// getStorage is a function so that even *reading* window.localStorage (which can throw) happens inside try/catch.
export function createStore(getStorage = () => globalThis.localStorage) {
  let memory = null;      // set once a read or write has failed; the in-memory copy is then authoritative
  let persistent = true;

  function load() {
    if (memory) return memory;
    try {
      const raw = getStorage().getItem(STORAGE_KEY);
      return sanitize(raw ? JSON.parse(raw) : {});
    } catch {
      persistent = false;
      memory = {};
      return memory;
    }
  }
  function save(all) {
    try { getStorage().setItem(STORAGE_KEY, JSON.stringify(all)); }
    catch { persistent = false; memory = all; }
  }
  const entry = (all, id) => (Object.hasOwn(all, id) ? all[id] : blank());   // ids like "constructor" are not inherited
  function update(id, fn) {
    if (id === '__proto__') return blank();
    const all = load();
    const cur = entry(all, id);
    fn(cur);
    all[id] = cur;
    save(all);
    return cur;
  }

  return {
    get persistent() { load(); return persistent; },
    all: () => load(),
    get: (id) => entry(load(), id),
    isDone: (id) => entry(load(), id).done,
    markVisited: (id) => update(id, (p) => { p.visited = true; }),
    setDone: (id, done) => update(id, (p) => { p.visited = true; p.done = !!done; }),
    setCheck: (id, checkId, ok) => update(id, (p) => { p.visited = true; p.checks[checkId] = !!ok; }),
  };
}

// The shared instance used by pages.
export const store = createStore();
