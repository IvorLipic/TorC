// core/numbers.js — how a number is written in lesson prose (the number rule v2 of docs/pedagogy-review.md §4, R2).
// Pure functions, no DOM. core/trace.js (makeT) uses them for the role-tagged T(path, role), T.anchor and T.human; the tests
// use formatBinding() to re-format every bound number from its data-* attributes and compare it with what the page shows.
//
// Roles (docs/v3-implementation.md "Numbers"):
//   setup | worked | reveal   exact: an integer in full, anything else as fmt() prints it (4 significant figures)
//   anchor | contrast | human  an integer in full, anything else to 2 significant figures
//   opts.sig (1-4) overrides both, and then also rounds integers ("about 27,000").
// Every integer part of 1,000 or more gets a comma every three digits (owner decision 10). A negative number gets the minus sign
// U+2212, which screen readers say as "minus". fmt() in ops/texutil.js stays as it is: KaTeX and the graph lab use it, and the
// untagged legacy T(path) keeps it, so lessons that are not migrated render exactly as before.
//
// human kinds (T.human):   rmse    √v at 2 significant figures ("off by about 0.24")
//                          prob    100·e^(−v) as a whole percent: the probability given to the right class
//                          pct     100·v/of (or 100·v without of) as a whole percent
//                          per100  mistakes per 100: 100 − (the whole-percent pct), so "97%" and "3 per 100" always add to 100
import { fmt } from '../ops/texutil.js';

export const ROLES = ['setup', 'worked', 'reveal', 'anchor', 'contrast', 'human'];
export const EXACT_ROLES = ['setup', 'worked', 'reveal'];
export const HUMAN_KINDS = ['rmse', 'prob', 'pct', 'per100'];
const MINUS = '−';

// "26506" -> "26,506", "-1234.5" -> "-1,234.5"; exponent forms and short numbers are left alone.
export function groupThousands(s) {
  const m = /^(-?)(\d+)(\.\d+)?$/.exec(s);
  if (!m || m[2].length < 4) return s;
  return m[1] + m[2].replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (m[3] ?? '');
}

const sign = (s) => (s.startsWith('-') ? MINUS + s.slice(1) : s);
const plain = (x) => (Math.abs(x) >= 1e21 ? String(x) : (Number.isInteger(x) ? BigInt(x).toString() : String(x)));

// sig: 'exact' (integers in full, others as fmt) or 1..4 significant figures.
export function proseNumber(x, sig = 'exact') {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error(`proseNumber: ${x} is not a finite number`);
  if (Object.is(x, -0) || Math.abs(x) < 1e-9) return '0';
  let s;
  if (sig === 'exact') s = Number.isInteger(x) ? plain(x) : fmt(x);
  else {
    if (!Number.isInteger(sig) || sig < 1 || sig > 4) throw new Error(`proseNumber: sig must be 1-4 or "exact", got ${sig}`);
    const r = Number(x.toPrecision(sig));
    s = Math.abs(r) < 1e-6 ? r.toPrecision(sig).replace('e+', 'e') : plain(r);
  }
  return sign(groupThousands(s));
}

// The significant figures a role uses for value x (what goes into data-sig).
export function sigFor(role, x, sig) {
  if (sig !== undefined) {
    if (!Number.isInteger(sig) || sig < 1 || sig > 4) throw new Error(`sig must be an integer from 1 to 4, got ${sig}`);
    return sig;
  }
  if (EXACT_ROLES.includes(role) || Number.isInteger(x)) return 'exact';
  return 2;
}

const wholePercent = (x) => Math.round(x);
// The number a human kind shows, before formatting.
export function humanValue(kind, v, of) {
  switch (kind) {
    case 'rmse':
      if (v < 0) throw new Error(`T.human rmse: a mean squared error cannot be negative (${v})`);
      return Math.sqrt(v);
    case 'prob': return 100 * Math.exp(-v);
    case 'pct': return of === undefined ? 100 * v : 100 * v / of;
    case 'per100': return 100 - wholePercent(of === undefined ? 100 * v : 100 * v / of);
    default: throw new Error(`T.human: unknown kind "${kind}" (use ${HUMAN_KINDS.join(', ')})`);
  }
}

// The text a human kind shows: rmse as a number (2 significant figures, or sig), the percentages with a "%" sign, per100 bare.
export function humanText(kind, v, of, sig) {
  const h = humanValue(kind, v, of);
  if (kind === 'rmse') return proseNumber(h, sig ?? 2);
  if (kind === 'per100') return proseNumber(h, 'exact');
  return proseNumber(wholePercent(h), 'exact') + '%';
}

// Re-formats a bound number from its attributes: {role, sig, human, of} plus num(path) (the resolver). Used by the tests.
export function formatBinding({ path, role, sig, human, of }, num) {
  const v = num(path);
  if (role === 'human') return humanText(human, v, of ? num(of) : undefined, sig === undefined || sig === '' ? undefined : Number(sig));
  return proseNumber(v, sig === 'exact' ? 'exact' : Number(sig));
}
