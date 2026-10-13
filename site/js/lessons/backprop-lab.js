// Lesson "backprop-lab": step through forward and backward passes of three small graphs with the real numbers.
// The interactive part is the shared graph lab (widgets/graph-lab.js); this file supplies the scenes' teaching text and layout.
import { mountGraphLab } from '../widgets/graph-lab.js';
import { SCENES, CONCEPTS } from './backprop-lab.scenes.js';

export default {
  id: 'backprop-lab',
  title: 'Backprop lab',
  trace: ['autograd_trace.json'],
  sections: [
    {
      kind: 'text',
      html: () => `<section class="hero">
    <h1>Backpropagation, one step at a time</h1>
    <p class="lead">Step through a small computation forward, then backward, and see exactly what is multiplied by what, with the real numbers computed by <a href="https://github.com/IvorLipic/TorC">TorC</a>, a from-scratch C++ tensor library.</p>
  </section>`,
    },
    {
      kind: 'widget', id: 'lab', className: 'lab', label: 'Interactive lab',
      mount: (host, D) => mountGraphLab(host, { scenes: D.scenes, layouts: SCENES, defaultScene: 'reuse' }),
    },
    {
      kind: 'text',
      html: () => `<section aria-labelledby="ideas" class="ideas">
    <h2 id="ideas">Three ideas are enough</h2>
    <div class="cards" id="concepts">${CONCEPTS.map((c) => `<div class="card"><h3>${c.h}</h3><p>${c.p}</p></div>`).join('')}</div>
  </section>`,
    },
  ],
  footer: () => '<p>Every value and gradient shown for the preset inputs was printed by the real TorC library (<code>tools/trace/autograd_trace.cpp</code>) and cross-checked against finite differences; edited inputs are recomputed in your browser by a small JavaScript port of the same rules, which is tested against the C++ output. Source: <a href="https://github.com/IvorLipic/TorC">github.com/IvorLipic/TorC</a>.</p>',
};
