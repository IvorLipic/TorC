// widgets/val-stop.js — the validation-stop game of the lesson generalization (brief §6 Part 3 and §9, C2). Data: the run without
// weight decay in data/generalization.json (tools/trace/trace_generalization.cpp), a polynomial of degree 15 trained by TorC on ten
// noisy points of a wave.
//
//   mountValStop(host, D)   1. A slider moves through the recorded steps. The top plot shows the model's curve with the training
//                              points and the validation points. The bottom plot shows the typical miss on the training set and on
//                              the validation set. The test set is not on the page yet.
//                           2. "Stop training here" commits the chosen step. The slider and the button are then locked, so the
//                              choice cannot change after the test set was seen (C2).
//                           3. Only then the test miss at the chosen step appears, and the whole test curve and the test points
//                              are drawn, labelled as shown only to compare. Nothing starts by itself; there is no animation.
// The module touches the DOM only in mountValStop. Its helpers are in widgets/generalization-widgets.js.
import { el } from '../core/dom.js';
import { srTable } from './plot.js';
import { injectStyles, fitPlot, drawFit, missPlot, drawMiss, miss, fmt2, fmtInt, lowestVal, stepX } from './generalization-widgets.js';

// The text of the result after the commit (pure, for the tests): the chosen step, the test miss there, the validation miss
// there, and where the lowest validation miss is.
export function resultText(run, k) {
  const c = run.checkpoints[k], b = lowestVal(run.checkpoints), best = run.checkpoints[b];
  const lines = [
    `You stopped after step ${fmtInt(c.step)}. There the model misses a test point by about ${fmt2(miss(c.test_loss))}.`,
    `The validation set promised a miss of about ${fmt2(miss(c.val_loss))}.`,
    b === k
      ? 'You chose the step with the lowest validation miss.'
      : `The lowest validation miss comes after step ${fmtInt(best.step)}, where it is about ${fmt2(miss(best.val_loss))}.`,
  ];
  return lines;
}

export function mountValStop(host, D) {
  injectStyles();
  const G = D.files.generalization, run = G.runs[0], cs = run.checkpoints;
  let k = 0, stopped = false;
  const input = el('input', { type: 'range', id: 'vs-step', min: 0, max: cs.length - 1, step: 1, value: k });
  const stopBtn = el('button', { type: 'button', class: 'btn primary' }, 'Stop training here');
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  const fitBox = el('div'), missBox = el('div');
  const key = el('p', { class: 'gw-key' },
    el('span', {}, el('i', { class: 'k-train', 'aria-hidden': 'true' }), 'training points'),
    el('span', {}, el('i', { class: 'k-val', 'aria-hidden': 'true' }), 'validation points'));
  const testKey = el('span', { hidden: true }, el('i', { class: 'k-test', 'aria-hidden': 'true' }), 'test points');
  key.append(testKey);
  const note = el('p', { class: 'gw-key' }, 'On the step axis, each mark is ten times the mark before it.');
  const rStep = el('p'), rTrain = el('p', { class: 'r-train' }), rVal = el('p', { class: 'r-val' });
  const result = el('div', { class: 'gw-result', hidden: true, tabindex: '-1' });
  host.replaceChildren(el('div', { class: 'gw' },
    el('div', { class: 'gw-ctl' }, el('div', { class: 'gw-field' }, el('label', { for: 'vs-step', class: 'gw-lbl' }, 'after step'), input), stopBtn),
    fitBox, key, missBox, note,
    el('div', { class: 'gw-read' }, rStep, rTrain, rVal, el('p', { class: 'r-note' }, 'A smaller miss is better.')), result, say));
  const table = srTable(host, { columns: ['step', 'typical miss on the training set', 'typical miss on the validation set'], rows: cs.map((c) => [c.step, miss(c.train_loss), miss(c.val_loss)]), caption: 'Typical misses of the run without weight decay, step by step' });

  const paintFit = (pl) => {
    pl.clear('wave'); pl.clear('data'); pl.clear('model'); pl.clear('labels');
    const points = [{ xs: G.data.val_x, ts: G.data.val_t, cls: 'gw-val', r: 3.4 }, { xs: G.data.train_x, ts: G.data.train_t, cls: 'gw-train', r: 4.5 }];
    if (stopped) points.unshift({ xs: G.data.test_x, ts: G.data.test_t, cls: 'gw-test', r: 2.2 });
    drawFit(pl, G, { points, curve: cs[k].curve });
  };
  const paintMiss = (pl) => {
    pl.clear('curves'); pl.clear('marks');
    pl.root.querySelectorAll(':scope > text.gw-t').forEach((n) => n.remove());
    const lines = [
      { cs, key: 'val_loss', cls: 'gw-lval', label: 'validation', lcls: 'c-val', dot: 'gw-dval' },
      { cs, key: 'train_loss', cls: 'gw-ltrain', label: 'training', lcls: 'c-train', dot: 'gw-dtrain' },
    ];
    if (stopped) lines.push({ cs, key: 'test_loss', cls: 'gw-ltest', label: 'test', lcls: 'c-test', dot: 'gw-dtest' });
    drawMiss(pl, { lines, at: k, stop: stopped ? cs[k].step : undefined, xMax: stepX(G.setup.steps) });
  };
  const fit = fitPlot(fitBox, '')((pl) => paintFit(pl));
  const loss = missPlot(missBox, '')((pl) => paintMiss(pl));
  const redraw = (pl, paint) => { pl.root.querySelectorAll(':scope > g.plot-axes').forEach((g) => g.remove()); paint(pl); };

  function draw() {
    const c = cs[k];
    input.setAttribute('aria-valuetext', `after step ${fmtInt(c.step)}`);
    redraw(fit, paintFit);
    redraw(loss, paintMiss);
    fit.root.setAttribute('aria-label', `The model without weight decay after step ${fmtInt(c.step)}, with ${G.data.train_x.length} training points, ${G.data.val_x.length} validation points${stopped ? `, ${G.data.test_x.length} test points` : ''} and the wave.`);
    loss.root.setAttribute('aria-label', `Typical miss on the training set and on the validation set${stopped ? ' and on the test set' : ''}, against the training step on a log axis. After step ${fmtInt(c.step)}: training ${fmt2(miss(c.train_loss))}, validation ${fmt2(miss(c.val_loss))}.`);
    rStep.textContent = `After step ${fmtInt(c.step)}`;
    rTrain.innerHTML = `The model misses a <b>training point</b> by about ${fmt2(miss(c.train_loss))}.`;
    rVal.innerHTML = `It misses a <b>validation point</b> by about ${fmt2(miss(c.val_loss))}.`;
  }
  input.addEventListener('input', () => { k = Number(input.value); draw(); });
  input.addEventListener('change', () => { const c = cs[k]; say.textContent = `After step ${fmtInt(c.step)}: training miss about ${fmt2(miss(c.train_loss))}, validation miss about ${fmt2(miss(c.val_loss))}.`; });
  stopBtn.addEventListener('click', () => {
    if (stopped) return;
    stopped = true;
    input.disabled = true;
    stopBtn.disabled = true;
    stopBtn.textContent = 'Training stopped';
    testKey.hidden = false;
    result.replaceChildren(...resultText(run, k).map((t, i) => el('p', {}, i === 0 ? el('b', {}, t) : t)),
      el('p', { class: 'r-note' }, 'We show the green test curve only so that you can compare it with the validation curve. In real work, you see only the one test miss.'));
    result.hidden = false;
    table.update(cs.map((c) => [c.step, miss(c.train_loss), miss(c.val_loss)]));
    draw();
    result.focus();   // the button is now disabled, so keyboard focus moves to the result, and a screen reader reads it
  });
  draw();
  return { draw, destroy() { fit.destroy(); loss.destroy(); } };
}
