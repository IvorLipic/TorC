// widgets/graph-lab.js — the interactive lab used by graph lessons: scene tabs, goal line, step timeline, persistent
// graph, math window and the inputs popover, wired together. Lesson code supplies the scenes (from the trace data)
// and a layout per scene {tab, view, minWidth, pos, above?, goal, intro, forwardHint, backwardHint, takeaway}.
//
//   mountGraphLab(host, { scenes: {id: scene}, layouts: {id: layout}, defaultScene })  ->  { destroy() }
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

const MARKUP = `
    <div class="lab-top">
      <div id="scene-tabs" class="tabs" role="tablist" aria-label="Examples"></div>
      <button id="inputs-btn" class="btn small" aria-expanded="false" aria-controls="inputs" title="Edit the input numbers">✎ Inputs</button>
    </div>
    <div id="inputs" class="popover" role="dialog" aria-label="Edit the input numbers" hidden><div id="inputs-inner"></div></div>
    <div id="goal" class="goal"></div>
    <div id="controls" class="controls"></div>
    <div class="graph-wrap"><div id="graph" class="graph"></div></div>
    <div id="window" class="win" aria-live="polite"></div>`;

export function mountGraphLab(host, { scenes, layouts, defaultScene }) {
  const ac = new AbortController();
  const { signal } = ac;
  host.innerHTML = MARKUP;
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
    return { scene, graph, bw, tl, layout: layouts[state.sceneId] };
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
    g.innerHTML = last ? `<strong>Takeaway.</strong> ${L.takeaway}` : `<strong>${L.goal}</strong> <span class="expr">${math(exprTex(), false)}</span>`;
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
    if (!D || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'ArrowRight') { go(state.step + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { go(state.step - 1); e.preventDefault(); }
  }, { signal });

  readHash();
  refresh(false);
  syncSelection();
  render();

  return { destroy: () => ac.abort(), state };
}
