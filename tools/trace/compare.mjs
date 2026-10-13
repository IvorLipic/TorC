// Compares two trace JSON files (committed vs. freshly generated): identical structure (keys, array lengths, strings,
// booleans, node names/ops/edges/shapes) and numerically equal values.
//
//   default numeric tolerance: relative 1e-6, absolute 1e-7 (float32 results may differ in the last bits across
//   compilers/CPUs; structure must not).
//   library.compiler / library.flags are informational (they differ between machines) and are not compared;
//   library.ref IS compared, so a changed pin makes --check fail until the JSON is regenerated.
//
// Values that come out of std::normal_distribution (tools/trace/trace_linear.cpp: init_stats, signal) are
// implementation-defined: a different C++ standard library draws different weights from the same mt19937 seed. The
// POLICIES table therefore compares them loosely (relative 10% for the std of 1000+ weights, 50% for the activation
// stds in `signal`, which compound over six layers) or by structure only (individual weights, means, extremes, histogram
// bins). On one standard library the output is bit-for-bit repeatable, so the loose rules only matter when the
// generating toolchain changes; they exist so --check does not fail for a stdlib difference that is not a TorC change.
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';

const [fa, fb] = process.argv.slice(2);
const [a, b] = [fa, fb].map((f) => JSON.parse(readFileSync(f, 'utf8')));
const file = basename(fa);
const errors = [];

const IGNORE = [/^\$\.library\.(compiler|flags)$/];
const POLICIES = {
  'linear.json': [
    { re: /^\$\.init_stats\[\d+\]\.(weights|histogram)(\.|\[|$)/, mode: 'struct' },
    { re: /^\$\.init_stats\[\d+\]\.(mean|min|max)$/, mode: 'struct' },
    { re: /^\$\.init_stats\[\d+\]\.(std|std_over_expected)$/, mode: 'loose', rel: 0.1 },
    { re: /^\$\.signal\[\d+\]\.(layers|across_seeds)\[\d+\]\.layer$/, mode: 'exact' },
    { re: /^\$\.signal\[\d+\]\.(layers|across_seeds)\[\d+\]\./, mode: 'loose', rel: 0.5 },
  ],
};
const policies = POLICIES[file] ?? [];
const policyFor = (path) => policies.find((p) => p.re.test(path));

const near = (x, y, rel = 1e-6, abs = 1e-7) => Math.abs(x - y) <= abs + rel * Math.max(Math.abs(x), Math.abs(y));
function walk(x, y, path) {
  if (IGNORE.some((r) => r.test(path))) return;
  if (typeof x === 'number' && typeof y === 'number') {
    const p = policyFor(path);
    if (p?.mode === 'struct') return;
    if (p?.mode === 'loose' ? !near(x, y, p.rel, 1e-12) : !near(x, y)) errors.push(`${path}: ${x} vs ${y}`);
    return;
  }
  if (Array.isArray(x) && Array.isArray(y)) {
    if (x.length !== y.length) { errors.push(`${path}: length ${x.length} vs ${y.length}`); return; }
    x.forEach((v, i) => walk(v, y[i], `${path}[${i}]`)); return;
  }
  if (x && y && typeof x === 'object' && typeof y === 'object' && !Array.isArray(x) && !Array.isArray(y)) {
    const keys = new Set([...Object.keys(x), ...Object.keys(y)]);
    for (const k of keys) { if (!(k in x) || !(k in y)) errors.push(`${path}.${k}: missing on one side`); else walk(x[k], y[k], `${path}.${k}`); }
    return;
  }
  if (x !== y) errors.push(`${path}: ${JSON.stringify(x)} vs ${JSON.stringify(y)}`);
}
walk(a, b, '$');
if (errors.length) { console.error(`${file}:\n` + errors.slice(0, 15).join('\n')); process.exit(1); }
console.log(`${file}: matches (structure identical, numbers within tolerance${policies.length ? '; stdlib-RNG values compared loosely' : ''})`);
