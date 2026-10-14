// widgets/landscape-tex.js — the LaTeX of the why-gradients bowl widget, as pure functions of the model, so the tests can render
// every string with KaTeX throwOnError over many inputs. No DOM.
import { fmt as fmtBase } from '../ops/texutil.js';

// Number display for the widgets: the shared 4-significant-digit format, but anything from 1e7 up (a diverging run) is always
// shown in exponent form, never as 8219000000000000000 next to 2.93e21.
export const fmtX = (x) => (Math.abs(x) >= 1e7 ? x.toExponential(3).replace(/\.?0+e/, 'e').replace('e+', 'e') : fmtBase(x));
const fmt = fmtX;
export const par = (x) => (x < 0 ? `(${fmt(x)})` : fmt(x));

// the pending update w <- w - lr*g
export function updateTex(m, w, lr) {
  const q = m.at(w);
  return `w \\leftarrow w - \\mathit{lr}\\cdot g = ${par(w)} - ${fmt(lr)}\\cdot ${par(q.grad)} = ${fmt(w - lr * q.grad)}`;
}
