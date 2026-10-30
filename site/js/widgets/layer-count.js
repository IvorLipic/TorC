// widgets/layer-count.js — "Where do 26,506 parameters come from?" (lesson linear-layer, brief §6 Part 2 "shapes and parameter
// counts"). The three Linear layers of the digit network of this course (data/linear.json `param_counts`, entry "MNIST MLP": the
// shapes and sizes that the real modules hold, and the total of the real Sequential::parameters()). A preset picks one layer or
// the whole network. For a layer, the picture draws W as a rectangle with one row per neuron and one column per input, at a
// common scale, so the first layer is long and flat and the last one is small. The readout gives the count of weights, biases and
// parameters. It replaces the parameter-count viewer of modules (module-widgets.js), which shows module names, the functions
// between the layers and error cases that this lesson does not teach yet.
//
//   digitLayers(D) -> [{ name, inputs, neurons, weights, biases, count, weightShape }]   (pure, tested against the trace)
//   mountLayerCount(host, D) -> { destroy(), pick(i) }
import { el } from '../core/dom.js';

const count = (n) => n.toLocaleString('en-US');
const NAMES = ['first layer', 'second layer', 'third layer'];
const INPUT_WORDS = ['pixels', 'outputs of the first layer', 'outputs of the second layer'];
const NEURON_WORDS = ['neurons', 'neurons', 'neurons, one for each digit'];

export function digitNetwork(D) {
  return D.files.linear.param_counts.find((e) => e.name === 'MNIST MLP');
}

export function digitLayers(D) {
  return digitNetwork(D).layers.filter((l) => l.params.length).map((l, i) => {
    const w = l.params.find((p) => p.name === 'weight'), b = l.params.find((p) => p.name === 'bias');
    return { name: NAMES[i], inputs: w.shape[1], neurons: w.shape[0], weights: w.numel, biases: b.numel, count: l.count, weightShape: w.shape };
  });
}

const CSS = `
.lc{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.lc-presets{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 .6rem}
.lc .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.lc-fig{min-height:1px}
.lc-fig svg{display:block;max-width:100%;height:auto}
.lc-read{margin:.5rem 0 0;font-size:.95rem;line-height:1.5}
.lc-read p{margin:.15rem 0}
.lc-read b{font-variant-numeric:tabular-nums}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('lc-styles')) return;
  document.head.append(el('style', { id: 'lc-styles' }, CSS));
}

// The picture of one layer's W: a rectangle of neurons × inputs cells, scaled so that the widest W (the first layer) fills the
// width. Returns an SVG string.
export function layerPicture(layer, maxInputs, width = 560) {
  const left = 8, top = 34, scale = (width - left - 16) / maxInputs;
  const w = Math.max(6, layer.inputs * scale), h = Math.max(6, layer.neurons * scale);
  const H = top + h + 30;
  const label = `W of the ${layer.name}: ${layer.neurons} rows, one per neuron, and ${layer.inputs} columns, one per input.`;
  return `<svg viewBox="0 0 ${width} ${H.toFixed(0)}" width="${width}" height="${H.toFixed(0)}" role="img" aria-label="${label}" focusable="false">`
    + `<rect x="${left}" y="${top}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" style="fill:rgba(88,196,221,.25);stroke:var(--fwd);stroke-width:1.5"/>`
    + `<text x="${left}" y="${top - 10}" style="fill:var(--fg);font-size:13px">${count(layer.inputs)} columns, one per input</text>`
    + `<text x="${left}" y="${(top + h + 20).toFixed(1)}" style="fill:var(--fg);font-size:13px">${count(layer.neurons)} rows, one per neuron</text>`
    + '</svg>';
}

export function mountLayerCount(host, D) {
  injectStyles();
  const layers = digitLayers(D), net = digitNetwork(D);
  const maxIn = Math.max(...layers.map((l) => l.inputs));
  const items = [...layers.map((l, i) => [i, l.name]), [layers.length, 'whole network']];
  let at = 0;
  const btns = items.map(([i, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(i) }, text));
  const fig = el('div', { class: 'lc-fig' });
  const read = el('div', { class: 'lc-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'lc' }, el('div', { class: 'lc-presets', role: 'group', 'aria-label': 'Layer of the digit network' }, ...btns), fig, read));

  function drawFig() {
    if (at >= layers.length) { fig.innerHTML = ''; return; }
    const width = Math.max(260, Math.min(560, Math.round(fig.getBoundingClientRect().width) || 560));
    fig.innerHTML = layerPicture(layers[at], maxIn, width);
  }
  function pick(i) {
    at = i;
    btns.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
    if (i < layers.length) {
      const l = layers[i];
      drawFig();
      read.innerHTML = `<p>The ${l.name} reads <b>${count(l.inputs)}</b> ${INPUT_WORDS[i]} and has <b>${count(l.neurons)}</b> ${NEURON_WORDS[i]}.</p>`
        + `<p>W has the shape (${l.weightShape.map(count).join(', ')}), so it holds <b>${count(l.weights)}</b> weights.</p>`
        + `<p>b holds <b>${count(l.biases)}</b> biases. The layer has <b>${count(l.count)}</b> parameters.</p>`;
    } else {
      drawFig();
      read.innerHTML = `<p>${layers.map((l) => `<b>${count(l.count)}</b>`).join(' + ')} = <b>${count(net.total)}</b> parameters.</p>`
        + '<p>Between the layers sits a function without parameters. The next lesson explains it.</p>';
    }
  }
  pick(at);
  let lastW = 0;
  const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => { const w = Math.round(fig.getBoundingClientRect().width); if (w && w !== lastW) { lastW = w; drawFig(); } }) : null;
  ro?.observe(fig);
  return { destroy() { ro?.disconnect(); }, pick };
}
