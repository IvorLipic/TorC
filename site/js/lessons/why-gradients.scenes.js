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
    intro: 'The loss <code>L</code> depends on the knob <code>w</code> through three small operations: <code>p = w·x</code>, <code>e = p − t</code>, <code>L = e·e</code>. Each has its own <strong>local slope</strong>. The backward pass starts at <code>L</code> with a seed of one and multiplies by one local slope per step.',
    forwardHint: 'Forward: compute p, e and L from the knob.',
    backwardHint: 'Backward: from L back to w, one local slope at a time.',
    takeaway: `With the preset inputs <code>w.grad()</code> is ${T('why.w.grad')}: the local slopes ${slope(0)} (p from w), ${slope(1)} (e from p) and ${slope(2)} (L from e) multiplied together. `
      + `<code>e</code> is used twice in <code>e·e</code>, so it received two contributions, each equal to <code>e</code> = ${T('why.e.value')}, and they add up to ${T('why.e.grad')}. Change <code>w</code> in ✎ Inputs and the numbers follow.`,
  };
}
