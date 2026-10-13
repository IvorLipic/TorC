// Run: node --test site/test/
// Checks the JS engine against the REAL TorC library run stored in data/autograd_trace.json.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { buildGraph, runBackward, timeline } from '../engine.js';
import { forwardWindow, backwardWindow, leafWindow, fmt } from '../explain.js';

const data = JSON.parse(readFileSync(new URL('../data/autograd_trace.json', import.meta.url)));
const katex = createRequire(import.meta.url)('../vendor/katex/katex.min.js');
const close = (a, b, tol = 1e-5) => a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) <= tol * (1 + Math.abs(b[i])));

for (const scene of data.scenes) {
  test(`${scene.id}: engine matches the real library (values and gradients)`, () => {
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

  test(`${scene.id}: every generated LaTeX string renders in KaTeX without errors`, () => {
    const g = buildGraph(scene);
    const bw = runBackward(g);
    const render = (s) => katex.renderToString(s, { throwOnError: true });
    const walk = (w) => { for (const b of w.blocks) for (const l of b.latex ?? []) render(l); };
    for (const n of g.nodes) if (n.role === 'op') walk(forwardWindow(g, n.name));
    for (const s of bw.steps) walk(s.node && g.byName[s.node].role === 'op' ? backwardWindow(g, bw, s.node) : leafWindow(g, bw, s.node));
    assert.ok(timeline(g, bw).length > 3);
  });

  test(`${scene.id}: finite differences of the JS engine agree with its own gradients`, () => {
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
  const scene = data.scenes.find((s) => s.id === 'mlp');
  const g = buildGraph(scene, { b1: [0.1, 0.2] });
  const bw = runBackward(g);
  assert.notEqual(bw.grad.b1.v[1], 0, 'second hidden unit should now receive gradient');
});

test('explanations contain the numbers a learner would write on paper', () => {
  const reuse = buildGraph(data.scenes.find((s) => s.id === 'reuse')), bwr = runBackward(reuse);
  const lw = leafWindow(reuse, bwr, 'x');
  assert.ok(lw.blocks[0].latex.some((l) => l.includes('1 + 3 + 3 = 7')), 'x.grad = 1 + 3 + 3 = 7');
  const mlp = buildGraph(data.scenes.find((s) => s.id === 'mlp')), bwm = runBackward(mlp);
  const bb = backwardWindow(mlp, bwm, 'a').blocks.map((b) => (b.latex ?? []).join(' ')).join(' ');
  assert.match(bb, /slope/, 'relu window shows the slope mask');
  const bo = backwardWindow(mlp, bwm, 'o').blocks.map((b) => (b.latex ?? []).join(' ')).join(' ');
  assert.match(bo, /shrink back to the shape/, 'broadcast reduction shown with numbers for b2');
});

test('fmt', () => { assert.equal(fmt(-0), '0'); assert.equal(fmt(0.22000006), '0.22'); assert.equal(fmt(1.1000000238), '1.1'); });
