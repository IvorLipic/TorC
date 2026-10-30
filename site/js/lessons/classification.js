// Lesson "classification" (Part 3 `networks`, curriculum v3, template 3, brief §6 "classification"): orange points in a disc
// inside a ring of blue points. A model without a hidden layer draws a straight boundary and stays near a coin flip; with 2, 4
// or 8 hidden ReLU units the boundary bends where it crosses the units' lines (the playground, widgets/class-plane.js). The
// probability of orange is the sigmoid of the difference of the two scores (`sigmoid_identity`, the real torc::softmax and
// torc::sigmoid), and the loss is described in words as the surprise at the right colour (cross-entropy is named and written in
// `losses`). Models are compared on the training points only; the 200 new points are looked at once, for one model, at the end
// (brief C2). One run per model, said in the text (brief C5). Every number comes from data/classification.json through a
// role-tagged T() (docs/authoring.md "Number roles"). Notes: docs/lessons/classification.md.
import { math } from '../core/katex.js';
import { worked, realLine, analogy, term, sym } from '../core/blocks.js';
import { mountClassPlane, pointsFigure } from '../widgets/class-plane.js';

const F = 'files.classification';
// models by index (the test pins this order): 0, 2, 4 and 8 hidden units; checkpoints by index: steps 0, 25, 100, 1000
const H0 = 0, H2 = 1, H4 = 2, H8 = 3;
const START = 0, MID = 2, END = 3;
const cp = (m, c, rest) => `${F}.models.${m}.checkpoints.${c}.${rest}`;
const id = (k, rest) => `${F}.sigmoid_identity.${k}.${rest}`;
const sec = (sid, title, body) => `<section class="cl" aria-labelledby="cl-${sid}"><h2 id="cl-${sid}">${title}</h2>${body}</section>`;
const eq = (latex, label) => `<div class="cl-eq" tabindex="0" role="group" aria-label="${label}">${math(latex, true)}</div>`;
const pct = (T, path, of) => T.human(path, 'pct', of ? { of } : undefined);
const per100 = (T, m, c) => T.human(cp(m, c, 'train_correct'), 'per100', { of: `${F}.setup.train_samples` });
const steps = (T) => T(`${F}.setup.steps`, 'setup');
const hid = (T, k) => T(`${F}.setup.hidden.${k}`, 'setup');

const STYLE = `<style>
.cl p{max-width:44rem}
.cl h2{margin-top:2rem}
.cl-eq{overflow-x:auto;margin:.3rem 0}
.cl-eq .katex-display{margin:.4em 0}
.cp-fig{margin:.6rem 0;max-width:15rem}
.cp-fig svg{width:100%;height:auto;background:var(--panel);border-radius:8px}
.cp-fig figcaption{color:var(--muted);font-size:.9rem}
</style>`;

export default {
  id: 'classification',
  title: 'Two classes and the decision boundary',
  trace: ['classification.json'],
  template: 3,
  loop: 'model',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Two classes and the decision boundary</h1>
    <p class="lead">A ${term('model')} learns to tell orange points from blue points. You watch the line between the colours bend.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'How can a model learn which colour belongs where in a plane?',
        'Why does a straight line fail here, and what bends it?',
        'How does the model say how sure it is?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('points', 'Orange points and blue points', `
    <p>Each point has a position, given by two numbers <code>x</code> and <code>y</code>, and a colour. There are ${T(`${F}.setup.train_samples`, 'setup')} ${term('training')} points.</p>
    ${pointsFigure(D.files.classification, { caption: 'The orange points fill a disc. The blue points form a ring around it.' })}
    <p>Each colour is a ${term('class')}. ${term('classification', 'Classification')} means choosing a class for each input. The model gives a colour to every spot of the plane. The line where this colour changes is the ${term('decision-boundary')}.</p>
    <p>A model with one ${term('layer')} computes only weighted sums of <code>x</code> and <code>y</code>, plus a ${term('bias')}. The lesson on nonlinearity showed that such a model draws only straight lines. So its boundary is straight.</p>`),
    },
    {
      kind: 'quiz', id: 'straight', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>We train a model with one layer on these points for ${steps(T)} steps. About how many of every hundred training points does it colour correctly at the end?</p>`,
      options: [
        { html: 'Almost all, because training removes the mistakes one by one.', feedback: 'Training can only move and turn the line. No straight line has the disc on one side and the whole ring on the other.' },
        { html: 'About three of every four, because the line cuts off a part of the ring.', feedback: 'A line can do that, but training does not count mistakes. This model ends close to a coin flip.' },
        { html: 'About half, which is no better than a coin flip.', correct: true, feedback: 'Yes. A straight boundary cannot separate a disc from a ring around it.' },
      ],
      reveal: (D, T) => `<p>After ${T(`${F}.setup.steps`, 'reveal')} steps it colours ${pct(T, cp(H0, END, 'train_correct'), `${F}.setup.train_samples`)} of the training points correctly.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('bend', '1 · Hidden units bend the boundary', `
    <p>Every straight line leaves a part of the ring with the disc. A ${term('hidden-layer')} of ${term('relu', 'ReLU')} units changes this.</p>
    <p>In the lesson on nonlinearity, each ReLU unit made one bend in a curve. In the plane, each hidden unit has its own straight line, where its weighted sum is zero:</p>
    ${eq('w_1 x + w_2 y + b = 0', 'The line of one hidden unit')}
    <p>On one side of its line the unit is off, because ReLU gives zero. On the other side it is on, and its output grows with the distance from the line.</p>
    <p><strong>The boundary can bend only where it crosses the line of a hidden unit.</strong> More hidden units give it more places to bend.</p>`),
    },
    {
      kind: 'widget', id: 'plane', className: 'widget', label: 'Interactive: the decision boundary of four models',
      mount: (host, D) => mountClassPlane(host, D, { hidden: 0, step: 0 }),
      tasks: (D, T) => [
        { do: `Choose ${hid(T, H0)} hidden units. Press each training button in turn.`,
          see: 'The boundary stays straight. The shading fades, because the model becomes unsure everywhere.' },
        { do: `Choose ${hid(T, H4)} hidden units and ${T(`${F}.setup.checkpoints.3`, 'setup')} steps. Find the dashed line of each unit.`,
          see: 'The four lines enclose the disc. Each arrow points to the side where its unit is on.' },
        { do: `Choose ${hid(T, H2)} hidden units, then ${hid(T, H8)}.`,
          see: 'With two units the orange region is a band, open at both ends. Eight units close it around the disc.' },
      ],
    },
    {
      kind: 'text',
      html: () => sec('vote', '2 · How the units vote', `
    <p>The model gives every point two scores, one for blue and one for orange. Each score is a weighted sum of the unit outputs, plus a bias.</p>
    <p>In the model with four hidden units, every unit votes for blue. In the middle all four units are off, so only the biases count, and they favour orange. Outside the line of a unit, that unit is on and pushes the point toward blue.</p>
    ${analogy(`A ${term('neuron')} is a weighted vote. Each input votes with its ${term('weight')}, and the bias shifts how many votes it needs to say yes.`, 'A neuron is not a brain cell. It only multiplies, adds and applies one function.')}`),
    },
    {
      kind: 'text',
      html: (D, T) => {
        const d = T.num(id(1, 'difference'));
        return sec('prob', '3 · The probability of orange', `
    <p>To say how sure it is, the model turns its two scores into two ${term('probability', 'probabilities')} that add to one. For two classes, TorC gives this probability of orange:</p>
    ${eq('\\text{probability of orange} = \\sigma(s_{\\text{orange}} - s_{\\text{blue}})', 'The probability of orange')}
    <p>Here ${math('s', false)} is a score, and ${sym('sigma', 'σ')} is the ${term('sigmoid')} from the lesson on nonlinearity, ${math('\\sigma(z) = 1/(1 + e^{-z})', false)}.</p>
    <p>On the boundary the two scores are equal. Their difference is ${T(id(0, 'difference'), 'setup')}, so the probability of orange is ${pct(T, id(0, 'sigmoid'))}.</p>
    ${worked(`<p>A point gets the blue score ${T(id(1, 'scores.0'), 'worked')} and the orange score ${T(id(1, 'scores.1'), 'worked')}. The difference is ${T(id(1, 'difference'), 'worked')}.</p>
    <p>${math(`\\sigma(${d}) = 1/(1 + e^{-${d}})`, false)} = <strong>${T(id(1, 'sigmoid'), 'worked')}</strong></p>`)}`);
      },
    },
    {
      kind: 'quiz', id: 'your-sigmoid', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: id(3, 'sigmoid'), tolerance: 0.01,
      prompt: (D, T) => `<p>Another point gets the blue score ${T(id(3, 'scores.0'), 'setup')} and the orange score ${T(id(3, 'scores.1'), 'setup')}. What is its probability of orange, as a number between zero and one? A calculator helps.</p>`,
      reveal: (D, T) => {
        const d = T.num(id(3, 'difference'));
        return `${worked(`<p>The difference is ${T(id(3, 'scores.1'), 'worked')} − (${T(id(3, 'scores.0'), 'worked')}) = ${T(id(3, 'difference'), 'worked')}.</p>
    <p>${math(`\\sigma(${d}) = 1/(1 + e^{-${d}})`, false)} = <strong>${T(id(3, 'sigmoid'), 'worked')}</strong></p>`)}
    <p>The more the orange score is above the blue score, the closer the probability comes to one.</p>`;
      },
    },
    {
      kind: 'text',
      html: (D, T) => sec('surprise', '4 · Surprise at the right colour', `
    <p>Training needs a ${term('loss')} <code>L</code> that says how wrong the model is. Here it looks at the probability that the model gives the right colour.</p>
    <p>A probability close to one gives almost no surprise. A small probability gives a large surprise. <strong>The loss is the mean surprise over the training points.</strong> The next lesson writes it as a formula.</p>
    <p>With eight hidden units, a typical training point gets ${T.human(cp(H8, START, 'train_loss'), 'prob')} on its right colour before training and ${T.human(cp(H8, END, 'train_loss'), 'prob')} after.</p>
    <p>TorC finds the ${term('gradient', 'gradients')} for us. The lesson “How the computer finds every gradient at once” shows how.</p>
    ${realLine('nn::Sequential model;\nmodel.add(std::make_unique<nn::Linear>(2, 8));   // x and y in, 8 hidden units\nmodel.add(std::make_unique<nn::ReLU>());         // each unit is off on one side of its line\nmodel.add(std::make_unique<nn::Linear>(8, 2));   // 8 unit outputs in, 2 scores out',
    'This is the model with eight hidden units in TorC.')}`),
    },
    {
      kind: 'quiz', id: 'longer', purpose: 'check', type: 'choice',
      prompt: (D, T) => `<p>After ${T(`${F}.setup.checkpoints.${MID}`, 'setup')} steps, the model with two hidden units makes ${per100(T, H2, MID)} mistakes per hundred training points. What happens to its mistakes by step ${steps(T)}?</p>`,
      options: [
        { html: 'They fall close to zero, because more training always helps.', feedback: 'Here more steps do not change what two units draw. The orange region stays an open band.' },
        { html: 'They stay about the same.', correct: true, feedback: 'Yes. The orange region stays an open band, so blue points at its ends stay wrong.' },
        { html: 'They grow, because the model trains for too long.', feedback: 'On the training points the mistakes do not grow here.' },
      ],
      reveal: (D, T) => `<p>After ${T(`${F}.setup.steps`, 'reveal')} steps it still makes ${per100(T, H2, END)} mistakes per hundred. Training cannot give a model a boundary that its units cannot draw.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('new', '5 · Points the model never saw', `
    <p>We compared the models on training points only. We kept ${T(`${F}.setup.test_samples`, 'setup')} new points apart, from the same disc and ring. We look at them once, here.</p>
    <p>The model with eight hidden units colours ${pct(T, cp(H8, END, 'test_correct'), `${F}.setup.test_samples`)} of the new points correctly. So it learned the disc and the ring, not only the training points.</p>
    <p>Each model started from random weights and ran once. Another run would give somewhat different numbers.</p>`),
    },
    {
      kind: 'quiz', id: 'road', purpose: 'transfer', type: 'choice',
      prompt: '<p>On a map, red houses stand on one side of a straight road and green houses on the other side. Which model can learn the colour of a house from its position?</p>',
      options: [
        { html: 'A model with one layer, because a straight boundary separates the colours.', correct: true, feedback: 'Yes. The road is straight, and so is this boundary.' },
        { html: 'A model with many hidden units, because more units are always better.', feedback: 'A model needs hidden units only when the boundary must bend.' },
        { html: 'No model, because the houses do not form a disc and a ring.', feedback: 'A model can learn any boundary that its units can draw, and a straight line is the simplest.' },
      ],
      reveal: '<p>A model needs hidden units only when the boundary must bend.</p>',
    },
    {
      kind: 'quiz', id: 'count', purpose: 'cumulative', from: 'linear-layer', type: 'numeric',
      answer: `${F}.models.${H4}.parameters`, tolerance: 0.5,
      prompt: () => `<p>A network has two inputs, a hidden layer with four units and two outputs. How many ${term('parameter', 'parameters')}, weights and biases together, does it have?</p>`,
      reveal: (D, T) => {
        const L = (k, rest) => T(cp(H4, START, `layers.${k}.${rest}`), 'worked');
        return worked(`<p>The hidden layer has ${L(0, 'weight.shape.0')} × ${L(0, 'weight.shape.1')} weights and ${L(0, 'bias.shape.0')} biases.</p>
    <p>The output layer has ${L(1, 'weight.shape.0')} × ${L(1, 'weight.shape.1')} weights and ${L(1, 'bias.shape.0')} biases.</p>
    <p>Together that makes <strong>${T(`${F}.models.${H4}.parameters`, 'worked')}</strong> parameters.</p>`);
      },
    },
    {
      kind: 'explain', id: 'why-bend',
      prompt: '<p>Explain why hidden units let a model separate the disc from the ring.</p>',
      model: '<p>Each hidden unit adds a line where the boundary can bend, so enough units can close the boundary around the disc.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>The model gives each point two scores, and the sigmoid of their difference is the probability of orange. Training raises the probability of the right colour.</p>
    <p>A model with one layer has a straight boundary. Each hidden ReLU unit adds a line where the boundary can bend, so enough units close it around the disc.</p>`,
      next: () => '<p>How does a model turn ten scores into probabilities, and how do you measure its surprise?</p>',
    },
  ],
};
