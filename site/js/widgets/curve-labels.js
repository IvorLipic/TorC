// widgets/curve-labels.js — direct labels for curves on a plot (brief §9 "label curves directly on the plot", D18). Pure.
//
//   spreadLabels(ys, { gap, min, max })   pixel y positions for labels that want to sit at `ys`, moved apart so that no two are
//                                         closer than `gap` and all stay within [min, max]. The labels keep the top-to-bottom order
//                                         of their wanted positions. The result holds one position per label, in the order of `ys`.
export function spreadLabels(ys, { gap = 14, min = -Infinity, max = Infinity } = {}) {
  const order = ys.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = new Array(ys.length);
  // push down from the top, then pull up from the bottom, so the group fits between min and max
  let prev = -Infinity;
  const pos = order.map(([y]) => { const p = Math.max(y, min, prev + gap); prev = p; return p; });
  let next = Infinity;
  for (let k = pos.length - 1; k >= 0; k--) { pos[k] = Math.min(pos[k], max, next - gap); next = pos[k]; }
  order.forEach(([, i], k) => { out[i] = pos[k]; });
  return out;
}
