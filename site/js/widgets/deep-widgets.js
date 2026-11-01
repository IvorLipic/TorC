// widgets/deep-widgets.js — the three widgets of the lesson deep-nets (brief §6 "deep-nets"). Every number is recorded by the real
// TorC run (data/deep.json, tools/trace/trace_deep.cpp); this file only reshapes and draws it. A browser cannot reproduce C++'s
// random numbers, so the widgets choose among recorded runs.
//   mountTwins(host, D)      two hidden neurons that start equal, different or at zero, at the recorded steps 0, 10, 100, 500, 2000
//   mountVanishing(host, D)  the mean size of each hidden layer's weight gradient after one backward(), sigmoid and ReLU, log axis
//   mountSignal(host, D)     the spread (std) of the signal after each ReLU layer for four starts, log axis
// The pure helpers (twinView, vanishingRows, signalSeries, powLabel, fmt2) are tested in site/test/deep-nets.test.mjs.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable } from './plot.js';

// 2 significant figures, a true minus sign, and plain decimals down to 0.001
export function fmt2(x) {
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a < 0.0005) return '0.000';   // a readout this small means "zero" to the reader (a twin's miss of 2e-7)
  const s = a < 1e5 ? String(Number(x.toPrecision(2))) : x.toExponential(1).replace('e+', 'e');
  return s.startsWith('-') ? `−${s.slice(1)}` : s;
}
// 10 to the power p, written with superscript digits ("10⁻⁶")
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export const powLabel = (p) => (p === 0 ? '1' : p === 1 ? '10' : `10${String(p).split('').map((c) => SUP[c]).join('')}`);

let seq = 0;
const btn = (label, onclick) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick }, label);
const thousands = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

// ---- twins ----------------------------------------------------------------------------------------------------------------------
export const TWIN_STARTS = [
  { id: 'twins', label: 'equal start' },
  { id: 'different', label: 'different start' },
  { id: 'zeros', label: 'all zero' },
];
// one recorded state: the four numbers of each neuron, whether they are equal, the predictions, the typical miss √(loss)
export function twinView(files, startId, k) {
  const run = files.twins.find((t) => t.id === startId), r = run.long_run[k];
  const nums = (n) => [n.w[0], n.w[1], n.b, n.v];
  return {
    step: r.step, a: nums(r.neuron0), b: nums(r.neuron1), equal: r.neurons_equal,
    prediction: r.prediction, target: files.twins_setup.t.values, miss: Math.sqrt(r.loss), equalEveryStep: run.equal_every_step,
  };
}

export function mountTwins(host, D) {
  injectStyles();
  const files = D.files.deep;
  const steps = files.twins[0].long_run.map((r) => r.step);
  const id = `dtw${++seq}`;
  let start = 0, k = 0;
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const startBtns = TWIN_STARTS.map((s, i) => btn(s.label, () => { start = i; draw(); announce(); }));
  const stepBtns = steps.map((s, i) => btn(thousands(s), () => { k = i; draw(); announce(); }));
  const pic = el('div', { class: 'dn-pic' });
  const read = el('div', { class: 'dn-read' });
  host.replaceChildren(
    el('div', { class: 'dn-ctl', role: 'group', 'aria-labelledby': `${id}-s` }, el('span', { class: 'dn-lbl', id: `${id}-s` }, 'Start'), ...startBtns),
    el('div', { class: 'dn-ctl', role: 'group', 'aria-labelledby': `${id}-k` }, el('span', { class: 'dn-lbl', id: `${id}-k` }, 'After step'), ...stepBtns),
    pic, read, status);

  function draw() {
    const v = twinView(files, TWIN_STARTS[start].id, k);
    startBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === start)));
    stepBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === k)));
    const W = 360, H = 222, base = 118, scale = 54;   // y of the zero line; 54 px per unit (values stay within ±1.2, so a bar and its label never reach the axis labels at H - 22)
    const names = ['weight from input 1', 'weight from input 2', 'bias', 'output weight'];
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'dn-svg', role: 'img', 'aria-label': describe(v) });
    s.append(svg('line', { x1: 8, x2: W - 8, y1: base, y2: base, class: 'dn-zero' }));
    names.forEach((name, g) => {
      const cx = 52 + g * 86;
      [v.a[g], v.b[g]].forEach((x, j) => {
        const h = Math.max(Math.abs(x) * scale, 0.5), y = x >= 0 ? base - h : base;
        const bx = cx - 37 + j * 40;   // two bars of 34 px with a 6 px gap, so each value label has its own width
        s.append(svg('rect', { x: bx, y, width: 34, height: h, class: j ? 'dn-bar two' : 'dn-bar one', rx: 2 }));
        s.append(svg('text', { x: bx + 17, y: x >= 0 ? y - 4 : y + h + 11, class: `dn-num ${j ? 'two' : 'one'}`, 'text-anchor': 'middle' }, fmt2(x)));
      });
      const words = name.split(' ');
      const line1 = words.length > 2 ? words.slice(0, 2).join(' ') : name, line2 = words.length > 2 ? words.slice(2).join(' ') : '';
      s.append(svg('text', { x: cx, y: H - 22, class: 'dn-axis', 'text-anchor': 'middle' }, line1));
      if (line2) s.append(svg('text', { x: cx, y: H - 9, class: 'dn-axis', 'text-anchor': 'middle' }, line2));
    });
    s.append(svg('text', { x: 8, y: 14, class: 'dn-key one' }, 'neuron 1'), svg('text', { x: 78, y: 14, class: 'dn-key two' }, 'neuron 2'));
    pic.replaceChildren(s);
    read.innerHTML = `<p><b>${v.equal ? 'The two neurons are equal.' : 'The two neurons differ.'}</b></p>`
      + `<p>Predictions for the four points: ${v.prediction.map(fmt2).join(', ')}. The targets are ${v.target.map(fmt2).join(', ')}.</p>`
      + `<p>Typical miss ${fmt2(v.miss)}, lower is better.</p>`;
  }
  function describe(v) {
    return `Neuron 1: ${v.a.map(fmt2).join(', ')}. Neuron 2: ${v.b.map(fmt2).join(', ')}. In order: the weight from input 1, the weight from input 2, the bias and the output weight.`;
  }
  function announce() {
    const v = twinView(files, TWIN_STARTS[start].id, k);
    status.textContent = `${TWIN_STARTS[start].label}, after step ${thousands(v.step)}. ${v.equal ? 'The two neurons are equal.' : 'The two neurons differ.'} Typical miss ${fmt2(v.miss)}.`;
  }
  draw();
  return { start: (i) => startBtns[i].click(), step: (i) => stepBtns[i].click() };
}

// ---- log-axis plots ------------------------------------------------------------------------------------------------------------
// a direct label in an unclipped group of the plot (plot.js clips its layers to the plot area, and a label may stand outside it)
function label(p, group, x, y, text, cls, dx, dy, anchor) {
  let g = p.root.querySelector(`g.${group}`);
  if (!g) { g = svg('g', { class: group }); p.root.append(g); }
  g.append(svg('text', { x: p.sx(x) + dx, y: p.sy(y) + dy, class: `plot-text ${cls}`, 'text-anchor': anchor }, text));
}
const clearLabels = (p, group) => p.root.querySelector(`g.${group}`)?.replaceChildren();
function logAxes(p, lo, hi, xs, xFmt) {
  const ticks = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) ticks.push(e);
  const every = ticks.length > 6 ? 2 : 1;
  p.axes({ xTicks: xs, yTicks: ticks.filter((e) => e % every === 0), xFmt, yFmt: powLabel });
}

// ---- vanishing gradients ----------------------------------------------------------------------------------------------------
// one row per hidden layer: [layer, sigmoid mean |gradient|, relu mean |gradient|]
export function vanishingRows(files) {
  const [sig, relu] = ['sigmoid', 'relu'].map((a) => files.vanishing.find((v) => v.activation === a));
  return sig.layers.filter((l) => l.name.startsWith('hidden')).map((l, i) => [l.layer, l.mean_abs_grad, relu.layers[i].mean_abs_grad]);
}

export function mountVanishing(host, D) {
  injectStyles();
  const rows = vanishingRows(D.files.deep);
  const box = el('div', { class: 'dn-plot' });
  host.replaceChildren(el('p', { class: 'dn-keyrow', 'aria-hidden': 'true' }, el('span', { class: 'dn-key-item' }, el('span', { class: 'dn-sw sig' }), 'sigmoid, the left bar of each layer'), el('span', { class: 'dn-key-item' }, el('span', { class: 'dn-sw relu' }), 'ReLU, the right bar')), box, el('p', { class: 'hint' }, 'Each step up on this axis is ten times larger. Short bars mean small gradients, so those layers barely change in a training step.'));
  srTable(host, { columns: ['layer', 'sigmoid', 'ReLU'], rows, caption: 'Mean size of each hidden layer\'s weight gradient after one backward pass' });
  const lo = -6.3, hi = 0.8;
  const p = createPlot(box, { width: 620, height: 300, x: [0.4, 8.6], y: [lo, hi], xLabel: 'hidden layer (1 is next to the input)', yLabel: 'mean size of the gradient', ariaLabel: 'Bars on a log axis. They show the mean size of the weight gradient in each hidden layer, for sigmoid and for ReLU.', margin: { l: 56 }, onResize: () => draw() });
  function draw() {
    p.clear('bars');
    clearLabels(p, 'bar-labels');
    p.root.querySelectorAll('.plot-axes').forEach((g) => g.remove());
    logAxes(p, lo, hi, rows.map((r) => r[0]), (x) => String(x));
    const bw = Math.min(16, (p.sx(2) - p.sx(1)) / 2.6);
    const g = p.layer('bars');
    for (const [layer, s, r] of rows) {
      [[s, 'sig', -1], [r, 'relu', 1]].forEach(([v, cls, side]) => {
        const x = p.sx(layer) + (side < 0 ? -bw - 1 : 1), y = p.sy(Math.log10(v));
        g.append(svg('rect', { x, y, width: bw, height: Math.max(0, p.sy(lo) - y), class: `dn-vbar ${cls}` }));
      });
    }
  }
  draw();
  return { destroy: () => p.destroy() };
}

// ---- the signal through eight ReLU layers ------------------------------------------------------------------------------------
export const SIGNAL_STARTS = [
  { id: 'kaiming', label: 'TorC’s start, √(2/in)', short: '√(2/in)' },
  { id: 'no_two', label: 'without the 2, √(1/in)', short: 'without the 2' },
  { id: 'too_small', label: 'too small', short: 'too small' },
  { id: 'too_large', label: 'too large', short: 'too large' },
];
export function signalSeries(files) {
  return SIGNAL_STARTS.map((s) => {
    const run = files.init.find((r) => r.name === s.id);
    return { ...s, weightStd: run.weight_std, points: run.layers.map((l) => [l.layer, l.std]) };
  });
}

export function mountSignal(host, D) {
  injectStyles();
  const series = signalSeries(D.files.deep);
  const shown = new Set(series.map((s) => s.id));
  const id = `dsg${++seq}`;
  const btns = series.map((s) => el('button', { type: 'button', class: `btn small dn-toggle dn-s-${s.id}`, 'aria-pressed': 'true', onclick: () => {
    if (shown.has(s.id) && shown.size > 1) shown.delete(s.id); else shown.add(s.id);
    draw();
    status.textContent = `Shown: ${series.filter((x) => shown.has(x.id)).map((x) => x.label).join(', ')}.`;
  } }, s.label));
  const box = el('div', { class: 'dn-plot' });
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'dn-ctl', role: 'group', 'aria-labelledby': `${id}-l` }, el('span', { class: 'dn-lbl', id: `${id}-l` }, 'Show'), ...btns), box,
    el('p', { class: 'hint' }, 'Each line follows the spread of the signal through eight ReLU layers. Layer 0 is the input. Each step up on this axis is ten times larger.'), status);
  srTable(host, { columns: ['layer', ...series.map((s) => s.label)], rows: series[0].points.map(([l], i) => [l, ...series.map((s) => s.points[i][1])]), caption: 'Spread (std) of the signal after each layer' });
  const lo = -6, hi = 3;
  const p = createPlot(box, { width: 620, height: 320, x: [0, 10.4], y: [lo, hi], xLabel: 'layer', yLabel: 'spread of the signal', ariaLabel: 'Spread of the signal after each of eight ReLU layers, on a log axis, for four starts of the weights.', margin: { l: 56, r: 8 }, onResize: () => draw() });
  function draw() {
    btns.forEach((b, i) => b.setAttribute('aria-pressed', String(shown.has(series[i].id))));
    p.clear('lines');
    clearLabels(p, 'line-labels');
    p.root.querySelectorAll('.plot-axes').forEach((g) => g.remove());
    logAxes(p, lo, hi, [0, 1, 2, 3, 4, 5, 6, 7, 8], (x) => String(x));
    for (const s of series.filter((x) => shown.has(x.id))) {
      const pts = s.points.map(([l, v]) => [l, Math.log10(v)]);
      p.curve(pts, { class: `plot-curve dn-line dn-s-${s.id}`, layer: 'lines' });
      const [lx, ly] = pts[pts.length - 1];
      const fits = p.sx(lx) + 8 + 7 * s.short.length <= p.width;   // else write the label under the end of its line
      label(p, 'line-labels', lx, ly, s.short, `dn-lab dn-s-${s.id}`, fits ? 6 : 0, fits ? 4 : 18, fits ? 'start' : 'end');
    }
  }
  draw();
  return { destroy: () => p.destroy(), toggle: (i) => btns[i].click() };
}

const CSS = `
.dn-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:.2rem 0 .5rem}
.dn-lbl{color:var(--muted);font-size:.9rem;margin-right:.3rem}
.dn-pic{max-width:560px}
.dn-svg{display:block;width:100%;height:auto}
.dn-svg text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.dn-zero{stroke:var(--border);stroke-width:1.5}
.dn-bar.one,.dn-vbar.relu{fill:var(--fwd)}
.dn-bar.two,.dn-vbar.sig{fill:var(--warn)}
.dn-num{font-size:10px}
.dn-num.one,.dn-key.one,.dn-lab.relu{fill:var(--fwd)}
.dn-num.two,.dn-key.two,.dn-lab.sig{fill:var(--warn)}
.dn-key{font-size:12px;font-family:system-ui,sans-serif!important}
.dn-axis{font-size:10px;fill:var(--muted);font-family:system-ui,sans-serif!important}
.dn-read p{margin:.2rem 0;font-size:.95rem}
.dn-plot{max-width:680px}
.dn-keyrow{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;font-size:.88rem;color:var(--muted);margin:.2rem 0}
.dn-sw{display:inline-block;width:12px;height:12px;border-radius:2px}
.dn-sw.sig{background:var(--warn)}
.dn-sw.relu{background:var(--fwd)}
.dn-key-item{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;margin-right:12px}
.dn-toggle[aria-pressed=false]{opacity:.55;border-style:dashed;text-decoration:line-through}
.dn-toggle[aria-pressed=false]:focus-visible{opacity:.8}
.dn-lab{font-size:12px}
.dn-line{fill:none;stroke-width:2.4}
.dn-line.dn-s-kaiming{stroke:var(--fwd)}
.dn-line.dn-s-no_two{stroke:var(--ok);stroke-dasharray:6 4}
.dn-line.dn-s-too_small{stroke:var(--warn);stroke-dasharray:2 4;stroke-linecap:round}
.dn-line.dn-s-too_large{stroke:var(--bwd)}
.dn-lab.dn-s-kaiming{fill:var(--fwd)}
.dn-lab.dn-s-no_two{fill:var(--ok)}
.dn-lab.dn-s-too_small{fill:var(--warn)}
.dn-lab.dn-s-too_large{fill:var(--bwd)}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('dn-styles')) return;
  document.head.append(el('style', { id: 'dn-styles' }, CSS));
}
