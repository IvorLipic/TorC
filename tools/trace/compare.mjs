// Compares two trace JSON files: identical structure (scenes, node names/ops/edges/shapes) and numerically equal values
// (relative 1e-6, absolute 1e-7). Float32 results may differ in the last bits across compilers/CPUs; structure must not.
import { readFileSync } from 'node:fs';
const [a, b] = process.argv.slice(2).map((f) => JSON.parse(readFileSync(f, 'utf8')));
const errors = [];
const near = (x, y) => Math.abs(x - y) <= 1e-7 + 1e-6 * Math.max(Math.abs(x), Math.abs(y));
function walk(x, y, path) {
  if (typeof x === 'number' && typeof y === 'number') { if (!near(x, y)) errors.push(`${path}: ${x} vs ${y}`); return; }
  if (Array.isArray(x) && Array.isArray(y)) {
    if (x.length !== y.length) { errors.push(`${path}: length ${x.length} vs ${y.length}`); return; }
    x.forEach((v, i) => walk(v, y[i], `${path}[${i}]`)); return;
  }
  if (x && y && typeof x === 'object' && typeof y === 'object') {
    const keys = new Set([...Object.keys(x), ...Object.keys(y)]);
    for (const k of keys) { if (!(k in x) || !(k in y)) errors.push(`${path}.${k}: missing on one side`); else walk(x[k], y[k], `${path}.${k}`); }
    return;
  }
  if (x !== y) errors.push(`${path}: ${JSON.stringify(x)} vs ${JSON.stringify(y)}`);
}
walk(a, b, '$');
if (errors.length) { console.error(errors.slice(0, 15).join('\n')); process.exit(1); }
console.log('trace matches (structure identical, numbers within tolerance)');
