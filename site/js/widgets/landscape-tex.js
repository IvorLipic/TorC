// widgets/landscape-tex.js — the "paper work" (LaTeX) of the why-gradients widgets, as pure functions of the model, so the tests
// can render every string with KaTeX throwOnError over many inputs. No DOM. The widgets only place these strings on the page.
import { fmt as fmtBase, tex } from '../ops/texutil.js';

// Number display for the widgets: the shared 4-significant-digit format, but anything from 1e7 up (a diverging run) is always
// shown in exponent form, never as 8219000000000000000 next to 2.93e21.
export const fmtX = (x) => (Math.abs(x) >= 1e7 ? x.toExponential(3).replace(/\.?0+e/, 'e').replace('e+', 'e') : fmtBase(x));
const fmt = fmtX;
export const par = (x) => (x < 0 ? `(${fmt(x)})` : fmt(x));

// More digits where the arithmetic on the page must visibly add up (a secant subtracts two nearly equal numbers).
export const fmtN = (x, digits = 8) => (Object.is(x, -0) || Math.abs(x) < 1e-12 ? '0' : String(Number(x.toPrecision(digits))));
const parN = (x) => (x < 0 ? `(${fmtN(x)})` : fmtN(x));

// p = w·x, e = p − t, L = e·e with the live numbers
export function knobTex(m, w) {
  const q = m.at(w);
  return [
    `${tex('p')} = ${tex('w')}\\cdot ${tex('x')} = ${par(w)}\\cdot ${par(m.x)} = ${fmt(q.p)}`,
    `${tex('e')} = ${tex('p')} - ${tex('t')} = ${par(q.p)} - ${par(m.t)} = ${fmt(q.e)}`,
    `${tex('L')} = ${tex('e')}\\cdot ${tex('e')} = ${par(q.e)}\\cdot ${par(q.e)} = ${fmt(q.L)}`,
  ];
}

// the secant at w with second point w + h, next to what backward() reports
export function secantTex(m, w, h) {
  const s = m.secant(w, h);
  return [
    `\\dfrac{L(w+h)-L(w)}{h} = \\dfrac{L(${fmt(w)}+${fmt(h)})-L(${fmt(w)})}{${fmt(h)}} = \\dfrac{${fmtN(s.Lh)} - ${parN(s.L0)}}{${fmt(h)}} = ${fmtN(s.slope)}`,
    `\\text{TorC's }\\texttt{backward()}:\\quad \\left.\\dfrac{dL}{dw}\\right|_{w=${fmt(w)}} = ${fmt(s.grad)}`,
  ];
}

// the pending update w <- w - lr*g
export function updateTex(m, w, lr) {
  const q = m.at(w);
  return `w \\leftarrow w - \\mathit{lr}\\cdot g = ${par(w)} - ${fmt(lr)}\\cdot ${par(q.grad)} = ${fmt(w - lr * q.grad)}`;
}

// a finished step l = {L, grad, w, wNext, promise, actual}: the tangent's promise next to what happened
export function lastStepTex(l) {
  return [
    `\\text{tangent promised: } L + g\\,\\Delta w = ${fmt(l.L)} + ${par(l.grad)}\\cdot ${par(l.wNext - l.w)} = ${fmt(l.promise)}`,
    `\\text{what happened: } L(${fmt(l.wNext)}) = ${fmt(l.actual)}\\qquad \\text{gap} = ${fmt(l.actual - l.promise)}`,
  ];
}

// distance to the minimum is multiplied by 1 - curvature*lr
export function factorTex(m, lr) {
  return [
    `w_{k+1}-${fmt(m.wMin)} = (1-${fmt(m.curvature)}\\,\\mathit{lr})\\,(w_k-${fmt(m.wMin)})`,
    `1-${fmt(m.curvature)}\\cdot ${fmt(lr)} = ${fmt(m.factor(lr))}`,
  ];
}

// the chain rule: product of local slopes, the prediction and its error
export function gearsTex(m, n) {
  const slopes = n.stages.map((s) => s.slope);
  return [
    `${slopes.map(par).join('\\cdot ')} = ${fmt(n.product)}\\qquad \\text{TorC's }\\texttt{w.grad} = ${fmt(n.backwardSlope)}`,
    `\\Delta L \\approx ${fmt(n.product)}\\cdot ${par(n.delta)} = ${fmt(n.predictedDL)}\\qquad \\text{actual } \\Delta L = ${fmt(n.actualDL)}\\qquad \\text{difference} = ${fmt(n.secondOrder)}`,
    `\\text{difference} = ${fmt(m.curvature / 2)}\\,\\delta^{2} = ${fmt((m.curvature / 2) * n.delta * n.delta)}`,
  ];
}
