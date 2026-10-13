// widgets/graph.js — the persistent SVG computation graph. A pure render function: give it the derived graph state and it
// replaces the contents of `root`; clicking or pressing Enter/Space on a node calls onSelect(name).
// Reusable by any lesson that has a scene {nodes[], root} and a layout {view, minWidth, pos, above?}.
import { svg } from '../core/dom.js';
import { fmt, opTitle } from '../explain.js';

export const NODE_W = 96, NODE_H = 44;
export const vecTxt = (t) => (t.v.length === 1 ? fmt(t.v[0]) : '[' + t.v.map(fmt).join(', ') + ']');

function clip(from, to, gap = 3) {
  const dx = to[0] - from[0], dy = to[1] - from[1];
  const t = Math.min((NODE_W / 2 + gap) / Math.abs(dx || 1e-9), (NODE_H / 2 + gap) / Math.abs(dy || 1e-9));
  return [from[0] + dx * t, from[1] + dy * t];
}

// view = {graph, bw, tl, layout, step, sel}
export function renderGraph(root, { graph: g, bw, tl, layout: L, step, sel }, { onSelect }) {
  const [W, H] = L.view;
  const cur = tl[step];
  const fwdIdx = {}, bwdIdx = {};
  tl.forEach((t, i) => { if (t.kind === 'forward') fwdIdx[t.node] = i; if (t.kind === 'backward') bwdIdx[t.node] = i; });
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
      const bwStep = bw.steps.find((st) => st.node === n.name);
      const send = bwStep?.sends.find((x) => x.arg === arg);
      const fwdActive = cur.kind === 'forward' && cur.node === n.name;
      const bwdActive = cur.kind === 'backward' && cur.node === n.name && tracked;
      let a, b, d, mid;
      if (k === 0) { a = clip(p, q); b = clip(q, p); d = `M${a[0]},${a[1]} L${b[0]},${b[1]}`; mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }
      else {
        const cx = (p[0] + q[0]) / 2, cy = (p[1] + q[1]) / 2, dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1;
        const ctrl = [cx + (dy / len) * 70 * k, cy - (dx / len) * 70 * k];   // second arc bows upward, away from the value/gradient labels
        a = clip(p, ctrl); b = clip(q, ctrl); d = `M${a[0]},${a[1]} Q${ctrl[0]},${ctrl[1]} ${b[0]},${b[1]}`;
        mid = [(a[0] + 2 * ctrl[0] + b[0]) / 4, (a[1] + 2 * ctrl[1] + b[1]) / 4];
      }
      const color = bwdActive ? 'red' : fwdActive ? 'cyan' : 'gray';
      edgeLayer.append(svg('path', { d, class: `edge ${tracked ? '' : 'const'} ${bwdActive ? 'bwd-active' : ''} ${fwdActive ? 'fwd-active' : ''}`, ...(bwdActive ? { 'marker-start': 'url(#arr-red)' } : { 'marker-end': `url(#arr-${color})` }) }));
      // contribution label after the consumer's backward step has happened
      if (send?.tracked && bwdIdx[n.name] !== undefined && step >= bwdIdx[n.name] && bw.steps.flatMap((st) => st.sends).filter((x) => x.to === src && x.tracked).length > 1) {
        const sendsTo = bw.steps.flatMap((st) => st.sends).filter((x) => x.to === src && x.tracked);
        const txt = send.tensor.v.length === 1 ? '+' + vecTxt(send.tensor) : 'c' + (sendsTo.indexOf(send) + 1);
        const w = Math.max(30, txt.length * 7 + 12);
        pillLayer.append(svg('g', { class: 'contrib', transform: `translate(${mid[0]},${mid[1]})` },
          svg('rect', { x: -w / 2, y: -10, width: w, height: 20, rx: 10 }), svg('text', { 'text-anchor': 'middle', y: 4 }, txt)));
      }
    });
  }

  // nodes
  for (const n of g.nodes) {
    const [x, y] = L.pos[n.name];
    const isCur = cur.node === n.name;
    const up = (L.above ?? []).includes(n.name);
    const showVal = n.role === 'leaf' || (fwdIdx[n.name] !== undefined && step >= fwdIdx[n.name]);
    const showGrad = bwdIdx[n.name] !== undefined && step >= bwdIdx[n.name];
    const stored = bw.grad[n.name];
    const cls = ['node', n.role, n.requires_grad ? 'learn' : 'const', isCur ? 'current' : '', sel === n.name ? 'selected' : '', cur.kind === 'forward' && isCur ? 'f' : '', cur.kind === 'backward' && isCur ? 'b' : ''].join(' ');
    const gnode = svg('g', { class: cls, 'data-node': n.name, 'data-key': 'node:' + n.name, transform: `translate(${x},${y})`, tabindex: 0, role: 'button', 'aria-label': `${n.name}, ${n.role === 'leaf' ? 'input' : opTitle(n.op)}. Open its math window.` },
      svg('rect', { x: -NODE_W / 2, y: -NODE_H / 2, width: NODE_W, height: NODE_H, rx: 10 }),
      svg('text', { class: 'nname', 'text-anchor': 'middle', y: n.role === 'op' ? -2 : 5 }, n.name),
      n.role === 'op' ? svg('text', { class: 'nop', 'text-anchor': 'middle', y: 13 }, n.op) : null,
      showVal ? svg('text', { class: 'nval', 'text-anchor': 'middle', y: up ? -NODE_H / 2 - 22 : NODE_H / 2 + 17 }, vecTxt(n.value)) : null,
      showGrad ? svg('text', { class: 'ngrad', 'text-anchor': 'middle', y: up ? -NODE_H / 2 - 7 : NODE_H / 2 + 34 }, stored ? 'grad ' + vecTxt(stored) : 'seed 1') : null);
    gnode.addEventListener('click', () => onSelect(n.name));
    gnode.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(n.name); } });
    nodeLayer.append(gnode);
  }
  root.replaceChildren(s);
}

// Scroll the graph horizontally so the current node is inside the visible part (phone width).
export function revealNode(wrap, nodeName) {
  const node = wrap && nodeName ? wrap.querySelector(`.node[data-node="${CSS.escape(nodeName)}"]`) : null;
  if (!node) return;
  const r = node.getBoundingClientRect(), w = wrap.getBoundingClientRect();
  if (r.left < w.left + 8 || r.right > w.right - 8) wrap.scrollLeft += (r.left + r.width / 2) - (w.left + w.width / 2);
}
