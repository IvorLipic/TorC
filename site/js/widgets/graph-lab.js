// widgets/graph-lab.js — the interactive lab used by graph lessons: scene tabs, goal line, step timeline, persistent
// graph, math window and the inputs popover, wired together. Lesson code supplies the scenes (from the trace data)
// and a layout per scene {tab, view, minWidth, pos, above?, goal, intro, forwardHint, backwardHint, takeaway}.
//
//   mountGraphLab(host, { scenes: {id: scene}, layouts: {id: layout}, defaultScene })  ->  { destroy() }
//
// Phones (brief D16). A layout may carry `narrow: {view, minWidth, pos, above?}`, a second drawing of the same graph that fits a
// phone: the lab uses it whenever the graph box is narrower than the layout's `minWidth`, so the whole graph (and the output) is on
// screen from the start. A graph without `narrow` scrolls sideways, and a line under it says so. On a narrow screen the step chips
// get their own full-width row that scrolls, with a fade at the edge that has more chips. The math window's start hint names the
// keyboard only on a device with a fine pointer.
//
// Arrow keys step the lab only when focus is inside it, or nowhere (see labOwnsArrows). `goal`, `intro`, `takeaway` and the hints are
// HTML fragments, in the lab and in the math window alike (docs/authoring.md).
//
// State lives here (one lab per page). View state is kept in the URL hash as "#<sceneId>/<step>".
import { $, el, keepFocus } from '../core/dom.js';
import { math } from '../core/katex.js';
import { parseHash, writeHash } from '../core/state.js';
import { buildGraph, runBackward, timeline } from '../engine.js';
import { renderGraph, revealNode } from './graph.js';
import { renderStepper, revealChip } from './stepper.js';
import { renderWindow } from './math-window.js';
import { createInputsPopover } from './inputs-popover.js';

// D16: the phone styles live with the widget, so every page that mounts a lab gets them (widgets.css keeps the desktop lab).
const D16_CSS = `
[data-graph-lab] .graph-cue{display:none;margin:0;font-size:.85rem;color:var(--muted)}
[data-graph-lab].lab-scrolls .graph-cue{display:block}
[data-graph-lab] .graph-cue .cue-touch{display:none}
@media(pointer:coarse){[data-graph-lab] .graph-cue .cue-touch{display:inline}[data-graph-lab] .graph-cue .cue-mouse{display:none}}
@media(max-width:640px){
  [data-graph-lab] .controls{flex-wrap:wrap;row-gap:6px}
  [data-graph-lab] .controls .chips{flex:1 1 100%;order:9;margin:0;padding:2px 0 4px}
  [data-graph-lab] .controls .chips.more-right{-webkit-mask-image:linear-gradient(to right,#000 85%,transparent);mask-image:linear-gradient(to right,#000 85%,transparent)}
  [data-graph-lab] .controls .chips.more-left{-webkit-mask-image:linear-gradient(to left,#000 85%,transparent);mask-image:linear-gradient(to left,#000 85%,transparent)}
  [data-graph-lab] .controls .chips.few{flex-wrap:wrap;overflow-x:visible}
  [data-graph-lab] .controls .chips.few.more-left,[data-graph-lab] .controls .chips.few.more-right{-webkit-mask-image:none;mask-image:none}
  [data-graph-lab] .controls .chips.more-left.more-right{-webkit-mask-image:linear-gradient(to right,transparent,#000 12%,#000 88%,transparent);mask-image:linear-gradient(to right,transparent,#000 12%,#000 88%,transparent)}
}`;
function injectD16Style() {
  if (typeof document === 'undefined' || document.getElementById('graph-lab-d16')) return;
  const st = document.createElement('style');
  st.id = 'graph-lab-d16';
  st.textContent = D16_CSS;
  document.head.append(st);
}

// The layout the graph is drawn with: the `narrow` variant when the graph box is narrower than the wide layout needs. Pure.
export function layoutFor(layout, boxWidth) {
  if (!layout?.narrow || !(boxWidth > 0) || boxWidth >= layout.minWidth) return layout;
  return { ...layout, ...layout.narrow };
}

const MARKUP = `
    <div class="lab-top">
      <div id="scene-tabs" class="tabs" role="tablist" aria-label="Examples"></div>
      <button id="inputs-btn" class="btn small" aria-expanded="false" aria-controls="inputs" title="Edit the input numbers">✎ Inputs</button>
    </div>
    <div id="inputs" class="popover" role="dialog" aria-label="Edit the input numbers" hidden><div id="inputs-inner"></div></div>
    <div id="goal" class="goal"></div>
    <div id="controls" class="controls"></div>
    <div class="graph-wrap"><div id="graph" class="graph"></div></div>
    <p class="graph-cue"><span class="cue-touch">The graph is wider than the screen. Swipe it sideways to see every node.</span><span class="cue-mouse">The graph is wider than the window. Scroll it sideways to see every node.</span></p>
    <div id="window" class="win" aria-live="polite"></div>`;

// Which keyboard events may step the lab? Pure (no DOM) so it can be tested: `d` describes the event target.
//   tag / editable: the target is a form field (arrow keys move the caret or the value there, never the lab)
//   inLab:          the target is inside this lab's host element
//   atPage:         nothing is focused (the target is <body>/<html>) - the lab then owns the arrows, but only if it is the one lab
//   labCount:       how many labs are on the page
// A control of ANOTHER widget (a slider, a button, a table region ...) is neither inLab nor atPage, so its arrows are its own.
export function labOwnsArrows({ tag = '', editable = false, inLab = false, atPage = false, labCount = 1 }) {
  if (editable || /^(INPUT|TEXTAREA|SELECT)$/i.test(tag)) return false;
  return inLab || (atPage && labCount === 1);
}
function describeTarget(t, host) {
  const atPage = !t || t === document || t === document.body || t === document.documentElement;
  return { tag: t?.tagName ?? '', editable: !!t?.isContentEditable, inLab: !!t && host.contains(t), atPage, labCount: document.querySelectorAll('[data-graph-lab]').length };
}

export function mountGraphLab(host, { scenes, layouts, defaultScene }) {
  const ac = new AbortController();
  const { signal } = ac;
  injectD16Style();
  host.innerHTML = MARKUP;
  host.setAttribute('data-graph-lab', '');
  const state = { sceneId: defaultScene, overrides: {}, step: 0, sel: null, tab: null, pop: false };
  let D = null;
  const q = (sel) => $(sel, host);

  const popover = createInputsPopover(host, {
    signal,
    onEdit: (name, i, x) => {
      const defaults = D.scene.nodes.find((n) => n.name === name).value.values;
      const cur = (state.overrides[state.sceneId] ??= {});
      const arr = (cur[name] ?? defaults).slice();
      arr[i] = x;
      if (arr.every((v, k) => Math.abs(v - defaults[k]) < 1e-12)) delete cur[name]; else cur[name] = arr;
      if (!Object.keys(cur).length) delete state.overrides[state.sceneId];
      const key = document.activeElement?.getAttribute('aria-label');
      refresh(false, true);
      if (key) $(`[aria-label="${CSS.escape(key)}"]`, host)?.focus();
    },
    onReset: () => { delete state.overrides[state.sceneId]; refresh(); },
  });

  const isEdited = () => Object.keys(state.overrides[state.sceneId] ?? {}).length > 0;

  // ---- derived data ------------------------------------------------------------------
  function derive() {
    const scene = scenes[state.sceneId];
    const graph = buildGraph(scene, state.overrides[state.sceneId] ?? {});
    const bw = runBackward(graph);
    const tl = timeline(graph, bw);
    state.step = Math.max(0, Math.min(state.step, tl.length - 1));
    return { scene, graph, bw, tl, layout: layoutFor(layouts[state.sceneId], q('.graph-wrap')?.clientWidth) };
  }

  // ---- scene tabs ----------------------------------------------------------------------
  function renderTabs() {
    q('#scene-tabs').replaceChildren(...Object.entries(layouts).map(([id, s]) => el('button', {
      class: 'tab', role: 'tab', 'aria-selected': id === state.sceneId, onclick: () => selectScene(id),
    }, s.tab)));
  }
  function selectScene(id) {
    state.sceneId = id; state.step = 0; state.sel = null; state.tab = null;
    refresh(true);
  }

  // ---- selection / navigation -------------------------------------------------------------
  function selectNode(name) {
    const cur = D.tl[state.step];
    state.sel = name;
    const n = D.graph.byName[name];
    const back = cur.kind === 'backward';
    state.tab = n.role === 'leaf' ? (back ? 'grad' : 'input') : (back ? 'backward' : 'forward');
    state.pop = true;
    keepFocus(() => { drawGraph(); drawWindow(); });
    q('#window').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  function go(i) { state.step = Math.max(0, Math.min(i, D.tl.length - 1)); syncSelection(); state.pop = true; render(); }
  function syncSelection() {
    const t = D.tl[state.step];
    if (t.kind === 'inputs') { state.sel = null; state.tab = null; return; }
    state.sel = t.node;
    const n = D.graph.byName[t.node];
    state.tab = n.role === 'leaf' ? 'grad' : t.kind;
  }

  // ---- drawing ------------------------------------------------------------------------------
  const view = () => ({ ...D, step: state.step, sel: state.sel, tab: state.tab, pop: state.pop });
  function drawGraph() { renderGraph(q('#graph'), view(), { onSelect: selectNode }); }
  function drawWindow() {
    renderWindow(q('#window'), view(), {
      onTab: (id) => { state.tab = id; state.pop = false; drawWindow(); },
      onBackToStep: () => { syncSelection(); state.pop = true; drawGraph(); drawWindow(); },
      onFocusInputs: (name) => popover.focusInput(name),
    });
    state.pop = false;
  }
  function exprTex() { return D.scene.expression.replace(/\*/g, '\\cdot ').replace(/relu/g, '\\mathrm{relu}').replace(/sum/g, '\\mathrm{sum}').replace(/mse/g, '\\mathrm{mse}').replace(/@/g, '\\,@\\,'); }
  function drawGoal() {
    const L = D.layout, last = state.step === D.tl.length - 1;
    const g = q('#goal');
    g.className = 'goal' + (last ? ' done' : '');
    // goal / takeaway are HTML fragments (code, bold ... allowed). Each sits in ONE .goal-text span: .goal is a flex row, so loose
    // inline pieces would otherwise become separate flex items with gaps between them.
    g.innerHTML = last ? `<span class="goal-text"><strong>Takeaway.</strong> ${L.takeaway}</span>` : `<span class="goal-text"><strong>${L.goal}</strong></span> <span class="expr">${math(exprTex(), false)}</span>`;
  }
  function render() {
    keepFocus(() => {
      drawGraph();
      renderStepper(q('#controls'), view(), { onGo: go });
      drawWindow();
      drawGoal();
      writeHash([state.sceneId, ...(state.step ? [String(state.step)] : [])]);
    });
    revealNode(q('.graph-wrap'), D.tl[state.step].node);
    revealChip(q('#controls'));
    markScroll();
  }
  // D16: say so when the graph scrolls sideways, and fade the chip row at the side that has more chips.
  function markScroll() {
    const wrap = q('.graph-wrap');
    host.classList.toggle('lab-scrolls', !!wrap && wrap.scrollWidth > wrap.clientWidth + 2);
    const chips = q('#controls .chips');
    if (!chips) return;
    const scrolls = !chips.classList.contains('few');   // a short timeline wraps instead of scrolling
    chips.classList.toggle('more-left', scrolls && chips.scrollLeft > 2);
    chips.classList.toggle('more-right', scrolls && chips.scrollLeft + chips.clientWidth < chips.scrollWidth - 2);
    if (!chips.dataset.watch) { chips.dataset.watch = '1'; chips.addEventListener('scroll', markScroll, { passive: true }); }
  }
  function refresh(full = false, keepInputs = false) {
    D = derive();
    renderTabs();
    if (!keepInputs) popover.render({ leaves: D.graph.nodes.filter((n) => n.role === 'leaf'), edited: isEdited() });
    else popover.updateEdited(isEdited());
    if (full) syncSelection();
    else if (state.sel && !D.graph.byName[state.sel]) syncSelection();
    render();
  }

  // ---- URL hash and keyboard -----------------------------------------------------------------------
  function readHash() {
    const [id, st] = parseHash(location.hash).parts;
    if (layouts[id] && scenes[id]) { state.sceneId = id; state.step = Math.max(0, parseInt(st ?? '0', 10) || 0); return true; }
    return false;
  }
  window.addEventListener('hashchange', () => { if (!D || !layouts[parseHash(location.hash).parts[0]]) return; readHash(); state.sel = null; refresh(true); }, { signal });
  document.addEventListener('keydown', (e) => {
    if (!D || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || !labOwnsArrows(describeTarget(e.target, host))) return;
    if (e.key === 'ArrowRight') { go(state.step + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { go(state.step - 1); e.preventDefault(); }
  }, { signal });

  readHash();
  refresh(false);
  syncSelection();
  render();
  // redraw when the graph box crosses the width at which the layout changes (rotating a phone, resizing a window)
  if (typeof ResizeObserver !== 'undefined') {
    let lastNarrow = D.layout !== layouts[state.sceneId];
    const ro = new ResizeObserver(() => {
      if (!D) return;
      const narrow = layoutFor(layouts[state.sceneId], q('.graph-wrap')?.clientWidth) !== layouts[state.sceneId];
      if (narrow !== lastNarrow) { lastNarrow = narrow; D = derive(); render(); } else markScroll();
    });
    ro.observe(host);
    signal.addEventListener('abort', () => ro.disconnect());
  }

  return { destroy: () => { ac.abort(); host.removeAttribute('data-graph-lab'); }, state };
}
