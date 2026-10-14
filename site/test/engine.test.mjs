// Run: node --test site/test/
// Checks the JS engine against the REAL TorC library runs stored in data/*.json (tools/trace/ writes them).
// Every trace file's top-level scenes[] is used; other top-level keys are ignored.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, runBackward, timeline } from '../js/engine.js';
import { forwardWindow, backwardWindow, leafWindow, fmt } from '../js/explain.js';
import { allScenes, close, renderStrictAll, katex } from './helpers.mjs';

const scenes = allScenes();
const autograd = scenes.filter((s) => s.file === 'autograd_trace.json').map((s) => s.scene);

test('trace files were found and contain scenes', () => {
  assert.ok(scenes.length >= 3, `expected at least the 3 backprop-lab scenes, found ${scenes.length}`);
  assert.deepEqual(autograd.map((s) => s.id), ['reuse', 'chain', 'mlp']);
});

for (const { file, scene } of scenes) {
  const tag = `${file} / ${scene.id}`;
  test(`${tag}: engine matches the real library (values and gradients)`, () => {
    const g = buildGraph(scene);
    const bw = runBackward(g);
    for (const n of scene.nodes) {
      const mine = g.byName[n.name];
      assert.deepEqual(mine.value.shape, n.value.shape, `${n.name} shape`);
      assert.ok(close(mine.value.v, n.value.values), `${n.name} value ${mine.value.v} vs ${n.value.values}`);
      if (n.grad) assert.ok(close(bw.grad[n.name].v, n.grad.values), `${n.name} grad ${bw.grad[n.name]?.v} vs ${n.grad.values}`);
      else assert.ok(!bw.grad[n.name], `${n.name} should have no stored grad`);
    }
  });

  test(`${tag}: every generated LaTeX string renders in KaTeX without errors`, () => {
    const g = buildGraph(scene);
    const bw = runBackward(g);
    const walk = (w) => { for (const b of w.blocks) renderStrictAll(b.latex ?? []); };
    for (const n of g.nodes) if (n.role === 'op') walk(forwardWindow(g, n.name));
    for (const s of bw.steps) walk(g.byName[s.node].role === 'op' ? backwardWindow(g, bw, s.node) : leafWindow(g, bw, s.node));
    for (const n of g.nodes) if (n.role === 'leaf') walk(leafWindow(g, bw, n.name));
    assert.ok(timeline(g, bw).length > 3);
  });

  test(`${tag}: finite differences of the JS engine agree with its own gradients`, () => {
    const g0 = buildGraph(scene), bw = runBackward(g0);
    const h = 1e-4;
    for (const leaf of g0.nodes.filter((n) => n.role === 'leaf' && n.requires_grad)) {
      leaf.value.v.forEach((_, k) => {
        const f = (d) => { const v = leaf.value.v.slice(); v[k] += d; return buildGraph(scene, { [leaf.name]: v }).byName[g0.root].value.v[0]; };
        const num = (f(h) - f(-h)) / (2 * h);
        assert.ok(Math.abs(num - bw.grad[leaf.name].v[k]) < 1e-3, `${leaf.name}[${k}] ${num} vs ${bw.grad[leaf.name].v[k]}`);
      });
    }
  });
}

test('editing an input recomputes (relu unit revived by raising b1)', () => {
  const scene = autograd.find((s) => s.id === 'mlp');
  const g = buildGraph(scene, { b1: [0.1, 0.2] });
  const bw = runBackward(g);
  assert.notEqual(bw.grad.b1.v[1], 0, 'second hidden unit should now receive gradient');
});

test('explanations contain the numbers a learner would write on paper', () => {
  const reuse = buildGraph(autograd.find((s) => s.id === 'reuse')), bwr = runBackward(reuse);
  const lw = leafWindow(reuse, bwr, 'x');
  assert.ok(lw.blocks[0].latex.some((l) => l.includes('1 + 3 + 3 = 7')), 'x.grad = 1 + 3 + 3 = 7');
  const mlp = buildGraph(autograd.find((s) => s.id === 'mlp')), bwm = runBackward(mlp);
  const bb = backwardWindow(mlp, bwm, 'a').blocks.map((b) => (b.latex ?? []).join(' ')).join(' ');
  assert.match(bb, /slope/, 'relu window shows the slope mask');
  const bo = backwardWindow(mlp, bwm, 'o').blocks.map((b) => (b.latex ?? []).join(' ')).join(' ');
  assert.match(bo, /shrink back to the shape/, 'broadcast reduction shown with numbers for b2');
});

test('the numbers quoted in the Backprop lab takeaways are the ones the engine computes', async () => {
  // scenes.js says "(3, 3 and 1 with the preset numbers)" for the reuse scene: the contributions x receives.
  const { SCENES } = await import('../js/lessons/backprop-lab.scenes.js');
  assert.match(SCENES.reuse.takeaway, /3, 3 and 1/);
  const g = buildGraph(autograd.find((s) => s.id === 'reuse')), bw = runBackward(g);
  const sent = bw.steps.flatMap((s) => s.sends).filter((s) => s.to === 'x' && s.tracked).map((s) => s.tensor.v[0]).sort();
  assert.deepEqual(sent, [1, 3, 3]);
});

test('fmt', () => { assert.equal(fmt(-0), '0'); assert.equal(fmt(0.22000006), '0.22'); assert.equal(fmt(1.1000000238), '1.1'); });
test('fmt keeps integers below 1e6 exact and rounds the rest to 4 significant digits', () => {
  assert.equal(fmt(25088), '25088'); assert.equal(fmt(26506), '26506'); assert.equal(fmt(-65536), '-65536'); assert.equal(fmt(999999), '999999');
  assert.equal(fmt(12345.6), '12350'); assert.equal(fmt(1234.5678), '1235'); assert.equal(fmt(2173000), '2173000');
});
test('katex is the vendored build', () => { assert.equal(typeof katex.renderToString, 'function'); });
