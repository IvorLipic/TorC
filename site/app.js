// app.js — the interactive lab. Renders a persistent computation graph (SVG), a step timeline,
// editable inputs, and a "math window" that shows the rule and the actual numbers for any node.
import { buildGraph, runBackward, timeline } from './engine.js';
import { forwardWindow, backwardWindow, leafWindow, fmt, mat, opTitle } from './explain.js';
import { SCENES, CONCEPTS } from './scenes.js';

const SVGNS = 'http://www.w3.org/2000/svg';
const $ = (sel, root = document) => root.querySelector(sel);
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
function svg(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null && v !== false) e.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
const math = (latex, display = true) => window.katex.renderToString(latex, { displayMode: display, throwOnError: false, strict: 'ignore' });
const vecTxt = (t) => (t.v.length === 1 ? fmt(t.v[0]) : '[' + t.v.map(fmt).join(', ') + ']');

const state = { data: null, sceneId: 'reuse', overrides: {}, step: 0, sel: null, tab: null, playing: false, timer: null, pop: false };
const NODE_W = 96, NODE_H = 44;

// ---- derived data -------------------------------------------------------------------
function derive() {
  const scene = state.data.scenes.find((s) => s.id === state.sceneId);
  const graph = buildGraph(scene, state.overrides[state.sceneId] ?? {});
  const bw = runBackward(graph);
  const tl = timeline(graph, bw);
  state.step = Math.max(0, Math.min(state.step, tl.length - 1));
  return { scene, graph, bw, tl, layout: SCENES[state.sceneId] };
}
let D = null;

// ---- scene tabs, inputs ---------------------------------------------------------------
function renderTabs() {
  const box = $('#scene-tabs');
  box.replaceChildren(...Object.entries(SCENES).map(([id, s]) => el('button', {
    class: 'tab', role: 'tab', 'aria-selected': id === state.sceneId, onclick: () => selectScene(id),
  }, s.tab)));
}
function selectScene(id) {
  stopPlay();
  state.sceneId = id; state.step = 0; state.sel = null; state.tab = null;
  refresh(true);
}

function renderInputs() {
  const box = $('#inputs-inner');
  const leaves = D.graph.nodes.filter((n) => n.role === 'leaf');
  const edited = Object.keys(state.overrides[state.sceneId] ?? {}).length > 0;
  box.replaceChildren(
    el('div', { class: 'inputs-head' },
      el('strong', {}, 'Inputs'),
      el('span', { class: 'hint' }, ' — type new numbers and everything recomputes instantly.'),
      el('span', { class: 'badge ' + (edited ? 'warn' : 'ok'), id: 'match-badge', title: edited ? 'You changed the inputs: values are recomputed in your browser by a JavaScript port of TorC’s rules.' : 'With the preset inputs, every value and gradient on this page equals what the real TorC C++ library printed.' },
        edited ? 'your values (recomputed in browser)' : '✓ identical to the real TorC run'),
      edited ? el('button', { class: 'link', onclick: () => { delete state.overrides[state.sceneId]; refresh(); } }, 'reset') : null),
    el('div', { class: 'inputs-grid' }, leaves.map((n) => {
      const orig = D.scene.nodes.find((x) => x.name === n.name).value;
      return el('label', { class: 'input-row', id: 'in-' + n.name },
        el('span', { class: 'in-name' }, n.name, el('small', {}, n.requires_grad ? 'learnable' : 'constant')),
        el('span', { class: 'in-vals' }, n.value.v.map((v, i) => el('input', {
          type: 'number', step: 'any', value: String(Number(v.toPrecision(7))), 'aria-label': `${n.name}[${i}]`,
          oninput: (ev) => onEdit(n.name, i, ev.target.value, orig.values),
        }))),
        el('span', { class: 'in-shape' }, 'shape (' + n.value.shape.join(', ') + ')'));
    })));
}
function onEdit(name, i, text, defaults) {
  const x = parseFloat(text);
  if (!Number.isFinite(x)) return;
  const cur = (state.overrides[state.sceneId] ??= {});
  const arr = (cur[name] ?? defaults).slice();
  arr[i] = x;
  if (arr.every((v, k) => Math.abs(v - defaults[k]) < 1e-12)) delete cur[name]; else cur[name] = arr;
  if (!Object.keys(cur).length) delete state.overrides[state.sceneId];
  const focus = document.activeElement;
  const key = focus?.getAttribute('aria-label');
  refresh(false, true);
  if (key) $(`[aria-label="${key}"]`)?.focus();
}

// ---- graph ------------------------------------------------------------------------------
function clip(from, to, gap = 3) {
  const dx = to[0] - from[0], dy = to[1] - from[1];
  const t = Math.min((NODE_W / 2 + gap) / Math.abs(dx || 1e-9), (NODE_H / 2 + gap) / Math.abs(dy || 1e-9));
  return [from[0] + dx * t, from[1] + dy * t];
}
function renderGraph() {
  const L = D.layout, [W, H] = L.view;
  const g = D.graph, tl = D.tl;
  const cur = tl[state.step];
  const fwdIdx = {}, bwdIdx = {};
  tl.forEach((t, i) => { if (t.kind === 'forward') fwdIdx[t.node] = i; if (t.kind === 'backward') bwdIdx[t.node] = i; });
  const root = $('#graph');
  root.style.minWidth = L.minWidth + 'px';
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': 'Computation graph' });
  s.append(svg('defs', {}, ...['gray', 'cyan', 'red'].map((c) => svg('marker', { id: 'arr-' + c, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, svg('path', { d: 'M0,0 L10,5 L0,10 z', class: 'arrow-' + c })))));
  const edgeLayer = svg('g'), nodeLayer = svg('g'), pillLayer = svg('g');
  s.append(edgeLayer, nodeLayer, pillLayer);

  // edges (one per input slot of each op; duplicates are drawn as a second, curved arc)
  const seen = {};
  for (const n of g.nodes) {
    n.inputs.forEach((src, arg) => {
      const p = L.pos[src], q = L.pos[n.name];
      const key = src + '>' + n.name;
      const k = (seen[key] = (seen[key] ?? -1) + 1);
      const inp = g.byName[src];
      const tracked = inp.requires_grad;
      const bwStep = D.bw.steps.find((st) => st.node === n.name);
      const send = bwStep?.sends.find((x) => x.arg === arg);
      const fwdActive = cur.kind === 'forward' && cur.node === n.name;
      const bwdActive = cur.kind === 'backward' && cur.node === n.name && tracked;
      let a, b, d, mid;
      if (k === 0) { a = clip(p, q); b = clip(q, p); d = `M${a[0]},${a[1]} L${b[0]},${b[1]}`; mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }
      else {
        const cx = (p[0] + q[0]) / 2, cy = (p[1] + q[1]) / 2, dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1;
        const ctrl = [cx - (dy / len) * 70 * k, cy + (dx / len) * 70 * k];
        a = clip(p, ctrl); b = clip(q, ctrl); d = `M${a[0]},${a[1]} Q${ctrl[0]},${ctrl[1]} ${b[0]},${b[1]}`;
        mid = [(a[0] + 2 * ctrl[0] + b[0]) / 4, (a[1] + 2 * ctrl[1] + b[1]) / 4];
      }
      const color = bwdActive ? 'red' : fwdActive ? 'cyan' : 'gray';
      edgeLayer.append(svg('path', { d, class: `edge ${tracked ? '' : 'const'} ${bwdActive ? 'bwd-active' : ''} ${fwdActive ? 'fwd-active' : ''}`, 'marker-end': `url(#arr-${color})` }));
      // contribution label after the consumer's backward step has happened
      if (send?.tracked && bwdIdx[n.name] !== undefined && state.step >= bwdIdx[n.name] && D.bw.steps.flatMap((st) => st.sends).filter((x) => x.to === src && x.tracked).length > 1) {
        const txt = '+' + vecTxt(send.tensor);
        const w = Math.max(30, txt.length * 7 + 12);
        const lab = svg('g', { class: 'contrib', transform: `translate(${mid[0]},${mid[1]})` },
          svg('rect', { x: -w / 2, y: -10, width: w, height: 20, rx: 10 }), svg('text', { 'text-anchor': 'middle', y: 4 }, txt));
        pillLayer.append(lab);
      }
    });
  }

  // nodes
  for (const n of g.nodes) {
    const [x, y] = L.pos[n.name];
    const isCur = cur.node === n.name;
    const up = (L.above ?? []).includes(n.name);
    const showVal = n.role === 'leaf' || (fwdIdx[n.name] !== undefined && state.step >= fwdIdx[n.name]);
    const showGrad = bwdIdx[n.name] !== undefined && state.step >= bwdIdx[n.name];
    const stored = D.bw.grad[n.name];
    const cls = ['node', n.role, n.requires_grad ? 'learn' : 'const', isCur ? 'current' : '', state.sel === n.name ? 'selected' : '', cur.kind === 'forward' && isCur ? 'f' : '', cur.kind === 'backward' && isCur ? 'b' : ''].join(' ');
    const gnode = svg('g', { class: cls, transform: `translate(${x},${y})`, tabindex: 0, role: 'button', 'aria-label': `${n.name}, ${n.role === 'leaf' ? 'input' : opTitle(n.op)}. Open its math window.` },
      svg('rect', { x: -NODE_W / 2, y: -NODE_H / 2, width: NODE_W, height: NODE_H, rx: 10 }),
      svg('text', { class: 'nname', 'text-anchor': 'middle', y: n.role === 'op' ? -2 : 5 }, n.name),
      n.role === 'op' ? svg('text', { class: 'nop', 'text-anchor': 'middle', y: 13 }, opTitle(n.op) === 'matrix multiplication' ? 'matmul' : n.op) : null,
      showVal ? svg('text', { class: 'nval', 'text-anchor': 'middle', y: up ? -NODE_H / 2 - 22 : NODE_H / 2 + 17 }, vecTxt(n.value)) : null,
      showGrad ? svg('text', { class: 'ngrad', 'text-anchor': 'middle', y: up ? -NODE_H / 2 - 7 : NODE_H / 2 + 34 }, stored ? 'grad ' + vecTxt(stored) : 'seed 1') : null);
    gnode.addEventListener('click', () => selectNode(n.name));
    gnode.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectNode(n.name); } });
    nodeLayer.append(gnode);
  }
  root.replaceChildren(s);
}
function selectNode(name) {
  const cur = D.tl[state.step];
  state.sel = name;
  const n = D.graph.byName[name];
  const back = cur.kind === 'backward';
  state.tab = n.role === 'leaf' ? (back ? 'grad' : 'input') : (back ? 'backward' : 'forward');
  state.pop = true;
  renderGraph(); renderWindow();
  $('#window').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// ---- controls / timeline --------------------------------------------------------------------
function go(i) { state.step = Math.max(0, Math.min(i, D.tl.length - 1)); syncSelection(); state.pop = true; render(); }
function syncSelection() {
  const t = D.tl[state.step];
  if (t.kind === 'inputs') { state.sel = null; state.tab = null; return; }
  state.sel = t.node;
  const n = D.graph.byName[t.node];
  state.tab = n.role === 'leaf' ? 'grad' : t.kind;
}
function stopPlay() { state.playing = false; clearInterval(state.timer); }
function togglePlay() {
  if (state.playing) { stopPlay(); renderControls(); return; }
  if (state.step >= D.tl.length - 1) go(0);
  state.playing = true;
  state.timer = setInterval(() => { if (state.step >= D.tl.length - 1) { stopPlay(); renderControls(); } else go(state.step + 1); }, 4500);
  renderControls();
}
function renderControls() {
  const tl = D.tl, box = $('#controls');
  const chips = [];
  let phase = null;
  tl.forEach((t, i) => {
    const p = t.kind === 'inputs' ? 'start' : t.kind;
    if (p !== phase) { phase = p; chips.push(el('span', { class: 'phase ' + p }, p === 'start' ? '' : p === 'forward' ? 'Forward →' : '← Backward')); }
    const label = t.kind === 'inputs' ? 'Start' : t.node;
    chips.push(el('button', { class: `chip ${t.kind} ${i === state.step ? 'on' : ''} ${i < state.step ? 'done' : ''}`, 'aria-current': i === state.step ? 'step' : false, onclick: () => { stopPlay(); go(i); }, title: t.kind === 'inputs' ? 'Inputs' : `${t.kind}: ${t.node}` }, label));
  });
  const L = D.layout;
  const cur = tl[state.step];
  const hint = cur.kind === 'inputs' ? 'Press <kbd>Next</kbd> (or <kbd>→</kbd>) to start the forward pass.' : cur.kind === 'forward' ? L.forwardHint : L.backwardHint;
  box.replaceChildren(
    el('div', { class: 'ctl-row' },
      el('button', { class: 'btn', onclick: () => { stopPlay(); go(0); }, 'aria-label': 'Restart' }, '⏮'),
      el('button', { class: 'btn', onclick: () => { stopPlay(); go(state.step - 1); }, disabled: state.step === 0, 'aria-label': 'Previous step' }, '◀ Back'),
      el('button', { class: 'btn primary', onclick: () => { stopPlay(); go(state.step + 1); }, disabled: state.step === tl.length - 1, 'aria-label': 'Next step' }, 'Next ▶'),
      el('button', { class: 'btn', onclick: togglePlay, 'aria-pressed': state.playing }, state.playing ? '⏸ Pause' : '▶ Auto-play'),
      el('span', { class: 'counter' }, `step ${state.step} / ${tl.length - 1}`)),
    el('div', { class: 'chips', role: 'group', 'aria-label': 'Steps (checkpoints)' }, chips),
    el('p', { class: 'hint', html: hint }));
}

// ---- math window --------------------------------------------------------------------------------
function leafInputWindow(n) {
  const orig = D.scene.nodes.find((x) => x.name === n.name);
  return {
    title: `${n.name} · input`, kind: 'input',
    blocks: [
      { heading: n.requires_grad ? 'A learnable input' : 'A constant input', html: n.requires_grad
        ? `<code>${n.name}</code> was created with <code>requires_grad = true</code>: TorC will compute the gradient of the output with respect to it. (In training these are the weights and biases.)`
        : `<code>${n.name}</code> was created with <code>requires_grad = false</code>: it is data, not something we adjust, so no gradient is computed for it.`, latex: [] },
      { heading: 'Value', latex: [`${'\\mathit{' + n.name + '}'} = ${mat(n.value)}`], html: `Shape (${n.value.shape.join(', ')}). Edit it in the <a href="#in-${n.name}" data-focus="${n.name}">Inputs</a> panel.` }],
  };
}
function windowData(name, tab) {
  const n = D.graph.byName[name];
  if (n.role === 'leaf') return tab === 'grad' ? leafWindow(D.graph, D.bw, name) : leafInputWindow(n);
  if (tab === 'backward') {
    return D.bw.steps.some((s) => s.node === name) ? backwardWindow(D.graph, D.bw, name)
      : { title: `${name} · ${opTitle(n.op)}`, kind: 'backward', blocks: [{ heading: 'Not part of the backward pass', html: 'No gradient reaches this node from the output.', latex: [] }] };
  }
  return forwardWindow(D.graph, name);
}
function renderWindow() {
  const box = $('#window');
  const cur = D.tl[state.step];
  if (!state.sel) {
    const L = D.layout;
    box.className = 'win intro' + (state.pop ? ' pop' : '');
    box.replaceChildren(
      el('h3', {}, 'Math window'),
      el('p', {}, 'This is where the arithmetic appears. Each step shows the rule, then the same rule with your numbers filled in. ',
        'You can also click any node in the graph to open its window.'),
      el('p', { html: `<strong>This scene:</strong> ${L.goal}` }),
      el('div', { class: 'expr', html: math(exprTex(), true) }));
    state.pop = false; return;
  }
  const n = D.graph.byName[state.sel];
  const w = windowData(state.sel, state.tab);
  const tabs = n.role === 'leaf' ? [['input', 'Value'], ['grad', 'Gradient']] : [['forward', 'Forward'], ['backward', 'Backward']];
  box.className = 'win ' + w.kind + (state.pop ? ' pop' : '');
  box.replaceChildren(...[
    el('div', { class: 'win-head' },
      el('h3', { 'aria-live': 'polite' }, w.title),
      el('div', { class: 'win-tabs', role: 'tablist' }, tabs.map(([id, label]) => el('button', { class: 'wtab ' + id, role: 'tab', 'aria-selected': state.tab === id, onclick: () => { state.tab = id; state.pop = false; renderWindow(); } }, label)))),
    ...w.blocks.map((b, i) => el('section', { class: 'blk' },
      el('h4', {}, b.heading),
      b.html ? el('p', { html: b.html }) : null,
      ...(b.latex ?? []).map((l) => el('div', { class: 'math', html: math(l) })))),
    state.sel && D.tl[state.step].node !== state.sel ? el('p', { class: 'hint' }, 'You are exploring ', el('code', {}, state.sel), '. ', el('button', { class: 'link', onclick: () => { syncSelection(); state.pop = true; renderGraph(); renderWindow(); } }, 'Back to the current step')) : null].filter(Boolean));
  box.querySelectorAll('a[data-focus]').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); const d = $('#inputs-details'); if (d) d.open = true; $(`#in-${a.dataset.focus} input`)?.focus(); }));
  state.pop = false;
}
function exprTex() { return D.scene.expression.replace(/\*/g, '\\cdot ').replace(/relu/g, '\\mathrm{relu}').replace(/sum/g, '\\mathrm{sum}').replace(/mse/g, '\\mathrm{mse}').replace(/@/g, '\\,@\\,'); }

// ---- page ---------------------------------------------------------------------------------------------
function render() { renderGraph(); renderControls(); renderWindow(); renderIntro(); writeHash(); }
function renderIntro() {
  const L = D.layout;
  $('#scene-intro').innerHTML = `<h3>${L.goal}</h3><p>${L.intro}</p><div class="expr">${math(exprTex(), true)}</div>`;
  $('#takeaway').innerHTML = D.tl[state.step] === D.tl[D.tl.length - 1] ? `<strong>Takeaway.</strong> ${L.takeaway}` : '';
  $('#takeaway').hidden = state.step !== D.tl.length - 1;
}
function refresh(full = false, keepInputs = false) {
  D = derive();
  renderTabs();
  if (!keepInputs) renderInputs();
  else { // update the badge only, keep the focused input element
    const edited = Object.keys(state.overrides[state.sceneId] ?? {}).length > 0;
    const b = $('#match-badge'); b.className = 'badge ' + (edited ? 'warn' : 'ok'); b.textContent = edited ? 'your values (recomputed in browser)' : '✓ identical to the real TorC run';
    if (!$('.inputs-head .link') && edited) $('.inputs-head').append(el('button', { class: 'link', onclick: () => { delete state.overrides[state.sceneId]; refresh(); } }, 'reset'));
  }
  if (full) syncSelection();
  else if (state.sel && !D.graph.byName[state.sel]) syncSelection();
  render();
}
function writeHash() { try { history.replaceState(null, '', `#${state.sceneId}${state.step ? '/' + state.step : ''}`); } catch { /* sandboxed */ } }
function readHash() {
  const [id, st] = location.hash.replace('#', '').split('/');
  if (SCENES[id]) { state.sceneId = id; state.step = Math.max(0, parseInt(st ?? '0', 10) || 0); }
}

window.addEventListener('hashchange', () => { if (!state.data) return; stopPlay(); readHash(); state.sel = null; refresh(true); });

document.addEventListener('keydown', (e) => {
  if (!D || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'ArrowRight') { stopPlay(); go(state.step + 1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { stopPlay(); go(state.step - 1); e.preventDefault(); }
});

(async function main() {
  const concepts = $('#concepts');
  concepts.replaceChildren(...CONCEPTS.map((c) => el('div', { class: 'card' }, el('h3', {}, c.h), el('p', { html: c.p }))));
  try {
    state.data = await (await fetch('data/autograd_trace.json')).json();
  } catch (err) {
    $('#lab-body').textContent = 'Could not load data/autograd_trace.json — serve the site over HTTP (for example: python3 -m http.server -d site). ' + err;
    return;
  }
  readHash();
  refresh(false);
  syncSelection();
  render();
})();
