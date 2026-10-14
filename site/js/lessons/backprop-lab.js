// Lesson "backprop-lab": step through forward and backward passes of three small graphs with the real numbers.
// The interactive part is the shared graph lab (widgets/graph-lab.js); this file supplies the scenes' teaching text and layout.
import { mountGraphLab } from '../widgets/graph-lab.js';
import { SCENES } from './backprop-lab.scenes.js';

export default {
  id: 'backprop-lab',
  title: 'Backprop lab',
  trace: ['autograd_trace.json'],
  sections: [
    {
      kind: 'text',
      html: () => `<section class="hero">
    <h1>Backpropagation, one step at a time</h1>
    <p class="lead">Step a small computation forward, then backward, and see what is multiplied by what.</p>
  </section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        '<strong>Forward</strong>: each operation computes its output, and TorC remembers which operation made what: the computation graph.',
        '<strong>Backward</strong>: from the output (gradient one), each node multiplies what it receives by its <em>local slope</em> and passes it on: the chain rule.',
        '<strong>Accumulate</strong>: a value used in several places gets a gradient from each, and TorC <em>adds</em> them.',
      ],
    },
    {
      kind: 'widget', id: 'lab', className: 'lab', label: 'Interactive lab',
      mount: (host, D) => mountGraphLab(host, { scenes: D.scenes, layouts: SCENES, defaultScene: 'reuse' }),
    },
  ],
};
