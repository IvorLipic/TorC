// core/state.js — view state in the URL hash, so every view is a link.
// Two shapes are understood:   #reuse/3   (path style: segments split on "/", used by the Backprop lab)
//                              #step=3&sel=h   (key=value style, for lessons that need several fields)
// parseHash / formatHash are pure (testable in Node); readHash / writeHash / onHashChange touch the browser.

export function parseHash(hash) {
  const raw = String(hash ?? '').replace(/^#/, '');
  const isKV = raw.includes('=');
  const params = {};
  if (isKV) for (const [k, v] of new URLSearchParams(raw)) params[k] = v;
  return { raw, parts: raw === '' ? [] : raw.split('/'), params };
}

export function formatHash({ parts, params } = {}) {
  if (params && Object.keys(params).length) return '#' + new URLSearchParams(params).toString();
  return parts && parts.length ? '#' + parts.join('/') : '';
}

export const readHash = () => parseHash(typeof location === 'undefined' ? '' : location.hash);

// Replace (not push) the hash so Back leaves the lesson instead of walking through every step.
export function writeHash(parts) {
  try {
    const h = formatHash(Array.isArray(parts) ? { parts } : { params: parts });
    history.replaceState(null, '', h || location.pathname + location.search);
  } catch { /* sandboxed iframe or file:// */ }
}

export function onHashChange(fn, signal) {
  window.addEventListener('hashchange', () => fn(readHash()), signal ? { signal } : undefined);
}
