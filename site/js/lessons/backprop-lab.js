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
    <p class="lead">This lab runs small computations forward and then backward, one step at a time. At each step you see which numbers TorC multiplies or adds.</p>
  </section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'In the forward pass, each operation computes its output. TorC records which operation made each value.',
        'In the backward pass, each operation multiplies the gradient it receives by its local derivative. This is the chain rule.',
        'If the computation uses a value in several places, the value gets a gradient from each place. TorC adds them.',
      ],
    },
    {
      kind: 'widget', id: 'lab', className: 'lab', label: 'Interactive lab',
      mount: (host, D) => mountGraphLab(host, { scenes: D.scenes, layouts: SCENES, defaultScene: 'reuse' }),
    },
  ],
};
