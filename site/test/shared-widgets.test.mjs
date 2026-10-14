// Pure (DOM-free) parts of the shared widgets: contribution pills, keyboard scoping of the graph lab, plot number formatting and
// heat-map cell edges. The browser behaviour of the same code is checked by loading the lessons (see CLAUDE.md).
import test from 'node:test';
import assert from 'node:assert/strict';
import './helpers.mjs';
import { signedTxt, pillTxt } from '../js/widgets/graph.js';
import { labOwnsArrows } from '../js/widgets/graph-lab.js';
import { fmtTick, cellEdges, ticks } from '../js/widgets/plot.js';

test('contribution pills carry a proper sign: "+2", "-6", "+0", never "+-6"', () => {
  assert.equal(signedTxt(2), '+2');
  assert.equal(signedTxt(-6), '-6');
  assert.equal(signedTxt(0), '+0');
  assert.equal(signedTxt(-0), '+0');
  assert.equal(signedTxt(0.5), '+0.5');
  assert.equal(signedTxt(-0.25), '-0.25');
  assert.equal(pillTxt({ v: [-6] }, 0), '-6');
  assert.equal(pillTxt({ v: [3] }, 1), '+3');
  assert.equal(pillTxt({ v: [1, -2] }, 1), 'c2');           // tensors are labelled c1, c2, … (the math window lists the values)
  for (const x of [-6, -0.001, 3, 12345.6, 25088, -26506]) assert.doesNotMatch(signedTxt(x), /\+-/);
});

test('graph lab arrows: only inside the lab, or with nothing focused when it is the only lab', () => {
  const inLab = { inLab: true };
  assert.equal(labOwnsArrows({ ...inLab, tag: 'BUTTON' }), true);                     // stepper chips and buttons in the lab
  assert.equal(labOwnsArrows({ ...inLab, tag: 'DIV' }), true);
  assert.equal(labOwnsArrows({ ...inLab, tag: 'INPUT' }), false);                     // the lab's own number boxes keep their arrows
  assert.equal(labOwnsArrows({ ...inLab, tag: 'select' }), false);
  assert.equal(labOwnsArrows({ ...inLab, tag: 'DIV', editable: true }), false);
  assert.equal(labOwnsArrows({ atPage: true, tag: 'BODY', labCount: 1 }), true);       // nothing focused, one lab
  assert.equal(labOwnsArrows({ atPage: true, tag: 'BODY', labCount: 2 }), false);      // ambiguous: no lab takes it
  assert.equal(labOwnsArrows({ atPage: true, tag: 'BODY', labCount: 0 }), false);
  for (const tag of ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A', 'SVG', 'DIV', 'SUMMARY']) {
    assert.equal(labOwnsArrows({ tag, labCount: 1 }), false, `${tag} in another widget never steps the lab`);
  }
});

test('plot tick numbers: 4 significant digits, exponent form beyond 1e6 and below 1e-3', () => {
  assert.equal(fmtTick(0), '0');
  assert.equal(fmtTick(-0), '0');
  assert.equal(fmtTick(0.5), '0.5');
  assert.equal(fmtTick(12345.678), '12350');
  assert.equal(fmtTick(0.001), '0.001');
  assert.equal(fmtTick(9999), '9999');
  assert.equal(fmtTick(999999), '1e6');                   // rounds to 1e6 at 4 digits, so it is written as 1e6
  assert.equal(fmtTick(1234567), '1.23e6');
  assert.equal(fmtTick(-2.5e7), '-2.5e7');
  assert.equal(fmtTick(0.00012), '1.2e-4');
  assert.equal(fmtTick(1e-9), '1e-9');
  assert.equal(fmtTick(NaN), 'NaN');
  assert.deepEqual(ticks([0, 1], 5).map(fmtTick), ['0', '0.2', '0.4', '0.6', '0.8', '1']);
});

test('heat-map cells tile the plot: edges are shared, the first and last cell are full width', () => {
  const p = [10, 20, 30, 40];
  const e = cellEdges(p);
  assert.deepEqual(e, [5, 15, 25, 35, 45]);
  for (let i = 1; i < e.length - 1; i++) assert.equal(e[i + 1] - e[i], e[i] - e[i - 1]);   // equal widths for equally spaced points
  assert.deepEqual(cellEdges([100, 80, 60]), [110, 90, 70, 50]);                            // descending (y pixels) works too
  assert.deepEqual(cellEdges([7]), [6.5, 7.5]);
  const uneven = cellEdges([0, 1, 4]);
  assert.deepEqual(uneven, [-0.5, 0.5, 2.5, 5.5]);                                          // no gap, no overlap: cell i = [e[i], e[i+1]]
});
