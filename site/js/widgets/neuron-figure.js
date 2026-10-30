// widgets/neuron-figure.js — the network picture of the lesson linear-layer (brief §6 Part 2: "one neuron, drawn as circles and
// arrows"). Pure: it returns an SVG string built from the taxi example of data/linear.json (`taxi`, a real nn::Linear(3, 2) with
// weights set by hand), so Node renders it in the tests exactly as the browser does.
//
//   neuronFigure(taxi, { neurons: [0] | [0, 1], ride: 0, caption })
//
// The inputs of one ride sit on the left (their values from taxi.x), one arrow per weight goes to each neuron (the weight from
// taxi.W, written on the arrow), each neuron circle holds Σ and its bias (taxi.b), and an arrow leaves it with the name of the
// output. Output values are not drawn: the lesson asks the reader to compute them. Neuron 0 is drawn in the forward colour and
// neuron 1 in the warning colour, the same colours as the rows of W in the matmul grid. Labels have a halo (paint-order stroke)
// so that they stay readable where they cross a line.
import { escapeHtml } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';

const W = 360;
const INPUT_UNITS = ['km', 'min waiting', 'bags'];
const OUTPUT_NAMES = ['price', 'time'];
const COLORS = ['var(--fwd)', 'var(--warn)'];
const BOX = { x: 6, w: 112, h: 32 };
const NX = 252;                                  // centre x of the neuron circles

export function layout(n) {
  return n === 1
    ? { h: 190, inputs: [35, 95, 155], neurons: [95], r: 30 }
    : { h: 224, inputs: [36, 112, 188], neurons: [62, 162], r: 27 };
}

// point at fraction t of the way from (x1, y1) to (x2, y2)
const along = (x1, y1, x2, y2, t) => [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t];
const f1 = (v) => v.toFixed(1);
const txt = (x, y, s, { anchor = 'middle', cls = 'nf-text', fill = 'var(--fg)', size = 13, weight = 400 } = {}) =>
  `<text class="${cls}" x="${f1(x)}" y="${f1(y)}" text-anchor="${anchor}" dominant-baseline="central" style="fill:${fill};font-size:${size}px;font-weight:${weight}">${escapeHtml(s)}</text>`;

export function neuronFigure(taxi, { neurons = [0], ride = 0, caption = '' } = {}) {
  const L = layout(neurons.length);
  const nIn = taxi.x.shape[1];
  const xv = (j) => taxi.x.values[ride * nIn + j];
  const wv = (k, j) => taxi.W.values[k * nIn + j];
  const parts = [];
  // arrows from every input to every neuron, with the weight near the neuron end
  neurons.forEach((k, a) => {
    const ny = L.neurons[a];
    for (let j = 0; j < nIn; j++) {
      const x1 = BOX.x + BOX.w, y1 = L.inputs[j];
      const ang = Math.atan2(ny - y1, NX - x1);
      const x2 = NX - L.r * Math.cos(ang), y2 = ny - L.r * Math.sin(ang);
      parts.push(`<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" style="stroke:${COLORS[k]};stroke-width:2;opacity:.85"/>`);
      const [lx, ly] = along(x1, y1, x2, y2, neurons.length === 1 ? 0.5 : 0.66);
      parts.push(txt(lx, ly, `× ${fmt(wv(k, j))}`, { cls: 'nf-w', fill: COLORS[k], weight: 700 }));
    }
  });
  // input boxes
  for (let j = 0; j < nIn; j++) {
    const y = L.inputs[j];
    parts.push(`<rect x="${BOX.x}" y="${f1(y - BOX.h / 2)}" width="${BOX.w}" height="${BOX.h}" rx="7" style="fill:var(--panel2);stroke:var(--border);stroke-width:1.5"/>`);
    parts.push(txt(BOX.x + BOX.w / 2, y, `${fmt(xv(j))} ${INPUT_UNITS[j] ?? ''}`.trim()));
  }
  // neurons, bias inside, output arrow and name
  neurons.forEach((k, a) => {
    const ny = L.neurons[a];
    parts.push(`<circle cx="${NX}" cy="${ny}" r="${L.r}" style="fill:var(--panel2);stroke:${COLORS[k]};stroke-width:2.5"/>`);
    parts.push(txt(NX, ny - 8, 'Σ', { size: 16, weight: 700 }));
    parts.push(txt(NX, ny + 11, `+ ${fmt(taxi.b.values[k])}`, { size: 12, fill: COLORS[k], weight: 700 }));
    const x1 = NX + L.r, x2 = W - 46;
    parts.push(`<line x1="${x1}" y1="${ny}" x2="${x2}" y2="${ny}" style="stroke:${COLORS[k]};stroke-width:2"/>`);
    parts.push(`<polygon points="${x2 + 6},${ny} ${x2 - 2},${ny - 5} ${x2 - 2},${ny + 5}" style="fill:${COLORS[k]}"/>`);
    parts.push(txt(x2 + 10, ny, OUTPUT_NAMES[k] ?? `output ${k}`, { anchor: 'start', weight: 700 }));
  });
  const ins = Array.from({ length: nIn }, (_, j) => `${fmt(xv(j))} ${INPUT_UNITS[j]}`).join(', ');
  const per = neurons.map((k) => `The ${OUTPUT_NAMES[k]} neuron multiplies the inputs by ${Array.from({ length: nIn }, (_, j) => fmt(wv(k, j))).join(', ')}, adds the products and adds ${fmt(taxi.b.values[k])}.`).join(' ');
  const aria = `${neurons.length === 1 ? 'One neuron' : 'A layer of two neurons'}. The inputs of the ride are ${ins}. Each input has an arrow to ${neurons.length === 1 ? 'the neuron' : 'each neuron'}, with its weight on the arrow. ${per}`;
  return `<figure class="nf-fig"><svg class="nf" viewBox="0 0 ${W} ${L.h}" width="${W}" height="${L.h}" role="img" aria-label="${escapeHtml(aria)}" focusable="false">${parts.join('')}</svg>`
    + (caption ? `<figcaption>${caption}</figcaption>` : '') + '</figure>';
}

export const NEURON_FIGURE_CSS = `
.nf-fig{margin:.7rem 0;max-width:30rem}
.nf-fig svg{display:block;width:100%;height:auto;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
.nf-fig .nf-w,.nf-fig .nf-text{paint-order:stroke;stroke:var(--bg);stroke-width:4px;stroke-linejoin:round}
.nf-fig figcaption{color:var(--muted);font-size:.9rem}
`;
