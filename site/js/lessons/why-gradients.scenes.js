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
    intro: 'The backward pass starts at <code>L</code> with a gradient of one. At each step it multiplies by one local derivative.',
    forwardHint: 'The forward pass computes p, e and L from w.',
    backwardHint: 'The backward pass goes from L back to w, one local derivative at each step.',
    takeaway: `<code>w.grad()</code> = ${T('why.w.grad')} = ${slope(0)} · ${slope(1)} · ${slope(2)}. The error <code>e</code> appears twice in <code>e·e</code>. TorC adds the contributions of the two uses and gets ${T('why.e.grad')}, which is twice e. This is the gradient of e squared.`,
  };
}
