// why-gradients.scenes.js — layout and teaching text of the graph-lab scene `why` (w -> p -> e -> L) used by the lesson why-gradients.
// Numbers are never typed here: whyLayout(T) builds the strings with T(), which binds each number to the trace data
// (site/test/why-gradients.test.mjs renders them against the real file and rejects any unbound number).
export function whyLayout(T) {
  const slope = (i) => T(`files.why-gradients.nudge.stages.${i}.local_slope`);
  return {
    tab: 'Chain',
    view: [880, 250], minWidth: 640,
    above: ['x', 't'],
    pos: { w: [110, 175], x: [300, 82], t: [510, 82], p: [300, 175], e: [510, 175], L: [740, 175] },
    goal: 'Where does w.grad() come from?',
    intro: 'Backward starts at <code>L</code> with a seed of one and multiplies by one local slope per step.',
    forwardHint: 'Forward: p, e, L from w.',
    backwardHint: 'Backward: from L to w, one local slope per step.',
    takeaway: `<code>w.grad()</code> = ${T('why.w.grad')} = ${slope(0)} · ${slope(1)} · ${slope(2)}. <code>e</code> is used twice, so its two contributions add up to ${T('why.e.grad')}.`,
  };
}
