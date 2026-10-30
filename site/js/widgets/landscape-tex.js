// widgets/landscape-tex.js — number formats and the LaTeX of the why-gradients widgets (bowl, nudge lens, two bowls), as pure
// functions of the model, so the tests can render every string with KaTeX throwOnError over many inputs. No DOM.
import { fmt as fmtBase } from '../ops/texutil.js';

// Number display for the widgets: the shared 4-significant-digit format, but anything from 1e6 up (a diverging run) is always
// shown in exponent form, never as 8219000000000000000 next to 2.93e21. texX writes the same number for KaTeX (1.68 × 10^7), where
// a bare "e" would read as a variable.
export const fmtX = (x) => (Math.abs(x) >= 1e6 ? x.toExponential(3).replace(/\.?0+e/, 'e').replace('e+', 'e') : fmtBase(x));
export const texX = (x) => { const m = /^(-?[\d.]+)e(-?\d+)$/.exec(fmtX(x)); return m ? `${m[1]}\\times 10^{${m[2]}}` : fmtX(x); };

// Readouts (docs/authoring.md "Widgets"): 3 significant figures, the minus sign U+2212, exponent form from 1e6 up (8.22e18).
// A value below 1e-9 in size is shown as 0, so a run that lands on the bottom does not read "−1.19e-7".
export function fmt3(x) {
  if (!Number.isFinite(x)) return String(x);
  if (Math.abs(x) < 1e-9) return '0';
  const s = Math.abs(x) >= 1e6 ? x.toExponential(2).replace(/\.?0+e/, 'e').replace('e+', 'e') : String(Number(x.toPrecision(3)));
  return s.replace(/^-/, '−');
}
const tex3 = (x) => { const s = fmt3(x).replace('−', '-'); const m = /^(-?[\d.]+)e(-?\d+)$/.exec(s); return m ? `${m[1]}\\times 10^{${m[2]}}` : s; };
export const par = (x) => (x < 0 ? `(${tex3(x)})` : tex3(x));

// The pending step, with the numbers of the current point: new w = w − lr × gradient (lr upright, decision 14).
export function updateTex(m, w, lr) {
  const q = m.at(w);
  return `\\text{new } w = ${par(w)} - ${tex3(lr)} \\times ${par(q.grad)} = ${tex3(w - lr * q.grad)}`;
}
