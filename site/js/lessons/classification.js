// Lesson "classification" (Part 3 `networks`, curriculum v3, template 3, brief §6 "classification"): orange points in a disc
// inside a ring of blue points. A model without a hidden layer draws a straight boundary (the best line gets 70 %, this run ends unsure); with 2, 4
// or 8 hidden ReLU neurons the boundary bends (the playground, widgets/class-plane.js); a worked vote of the 4-neuron model. The
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
        'How can a model learn which colour belongs where?',
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
    <p>A model with one ${term('layer')} computes only weighted sums of <code>x</code> and <code>y</code>, plus a ${term('bias')}. The lesson on nonlinearity showed that its boundary is then a straight line.</p>`),
    },
    {
      kind: 'quiz', id: 'straight', purpose: 'predict', type: 'choice',
      prompt: '<p>We train a model with one layer on these points. Can its straight boundary get almost every training point right?</p>',
      options: [
        { html: 'Yes, because training moves the line until the mistakes are gone.', feedback: 'Training can move and turn a straight line, but it cannot bend it.' },
        { html: 'Yes, if the line goes through the middle of the disc.', feedback: 'That line cuts the disc in two, and half of the orange points land on the blue side.' },
        { html: 'No, because every straight line leaves blue points on the side of the disc.', correct: true, feedback: 'Yes. The ring goes all the way around the disc.' },
      ],
      reveal: (D, T) => `<p>The best straight line gets ${pct(T, `${F}.best_line.correct`, `${F}.setup.train_samples`)} of the training points right. After ${T(`${F}.setup.steps`, 'reveal')} steps this run gets only ${pct(T, cp(H0, END, 'train_correct'), `${F}.setup.train_samples`)}, because the model ends unsure everywhere. The first task below shows this.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('bend', '1 · Hidden neurons bend the boundary', `
    <p>A ${term('hidden-layer')} of ${term('relu', 'ReLU')} ${term('neuron', 'neurons')} can bend the boundary. In the lesson on nonlinearity, each of them made one bend in a curve. In the plane, each hidden neuron has its own straight line, where its weighted sum is zero:</p>
    ${eq('w_1 x + w_2 y + b = 0', 'The line of one hidden neuron')}
    <p>On one side of its line the neuron is off, because ReLU gives zero. On the other side it is on, and its output grows with the distance from the line. More neurons give the model more lines, so its boundary can bend in more places.</p>
    ${realLine('nn::Sequential model;\n// x and y in, 8 hidden neurons out\nmodel.add(std::make_unique<nn::Linear>(2, 8));\n// each neuron is off on one side of its line\nmodel.add(std::make_unique<nn::ReLU>());\n// 8 neuron outputs in, 2 scores out\nmodel.add(std::make_unique<nn::Linear>(8, 2));',
    'This is a model with eight hidden neurons in TorC. Its last layer gives two scores, one for each colour.')}`),
    },
    {
      kind: 'quiz', id: 'longer', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>A model with two hidden neurons makes ${per100(T, H2, MID)} mistakes per hundred training points after ${T(`${F}.setup.checkpoints.${MID}`, 'setup')} steps. What happens to its mistakes by step ${steps(T)}?</p>`,
      options: [
        { html: 'They fall close to zero, because more training always helps.', feedback: 'This is the belief that long training fixes every mistake. Here more steps hardly change the boundary.' },
        { html: 'They stay about the same.', correct: true, feedback: 'Yes. More steps hardly change the boundary, as the plane below shows.' },
        { html: 'They grow, because the model trains for too long.', feedback: 'This is the belief that long training makes a model worse on its own training points. Here that does not happen.' },
      ],
      reveal: (D, T) => `<p>After ${T(`${F}.setup.steps`, 'reveal')} steps it still makes ${per100(T, H2, END)} mistakes per hundred. The third task below shows why.</p>`,
    },
    {
      kind: 'widget', id: 'plane', className: 'widget', label: 'Interactive: the decision boundary of four models',
      mount: (host, D) => mountClassPlane(host, D, { hidden: 0, step: 0 }),
      tasks: (D, T) => [
        { do: `Choose ${hid(T, H0)} hidden neurons. Press each training button in turn.`,
          see: 'The boundary stays straight. The shading fades, because the model becomes unsure everywhere.' },
        { do: `Choose ${hid(T, H4)} hidden neurons and ${T(`${F}.setup.checkpoints.3`, 'setup')} steps. Find the dashed line of each neuron.`,
          see: 'The four lines cross inside the disc. Each arrow points to the side where its neuron is on. The boundary lies further out.' },
        { do: `Choose ${hid(T, H2)} hidden neurons, then ${hid(T, H8)}.`,
          see: 'With two neurons the orange region is an open band. Eight neurons close it around the disc.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => {
        const V = (rest) => `${F}.vote.${rest}`, p = (rest) => V(`points.1.${rest}`);
        return sec('vote', '2 · How the neurons vote', `
    <p>The model gives every point a blue score and an orange score, and the larger one wins.</p>
    <p>Each neuron that is on adds its output times a ${term('weight')} to the orange score minus the blue score. This product is its vote, and a bias sets the start value.</p>
    <p>With four neurons, every neuron votes for blue, and the bias favours orange. In the middle all neurons are off, so the bias wins.</p>
    ${worked(`<p>At the ring point (${T(p('x'), 'worked')}, ${T(p('y'), 'worked')}), only the first and the fourth neuron are on. Their votes are ${T(p('votes.0'), 'worked')} and ${T(p('votes.3'), 'worked')}.</p>
    <p>${T(V('bias_orange'), 'worked')} + (${T(p('votes.0'), 'worked')}) + (${T(p('votes.3'), 'worked')}) = <strong>${T(p('total'), 'worked')}</strong>, so the point is blue.</p>`)}
    <p>The boundary lies where the votes just outweigh the bias.</p>
    ${analogy('A neuron is a weighted vote, and the bias decides when no neuron votes.', 'A real vote is yes or no. The vote of a neuron grows with the distance from its line.')}`);
      },
    },
    {
      kind: 'text',
      html: (D, T) => {
        const d = T.num(id(1, 'difference'));
        return sec('prob', '3 · The probability of orange', `
    <p>To say how sure it is, the model turns its two scores into two ${term('probability', 'probabilities')} that add to one. TorC uses a rule that the next lesson names. For two classes, that rule gives the same as this formula:</p>
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
`;
      },
    },
    {
      kind: 'text',
      html: (D, T) => sec('surprise', '4 · Surprise at the right colour', `
    <p>Training needs a ${term('loss')} <code>L</code> that says how wrong the model is. Here it uses the probability of the right colour.</p>
    <p>A probability close to one gives almost no surprise. A small probability gives a large surprise. <strong>The loss is the mean surprise over the training points.</strong> The next lesson writes it as a formula.</p>
    <p>The probability whose surprise equals this mean is the typical probability of the right colour. With eight hidden neurons it is ${T(cp(H8, START, 'typical_right_prob'), 'contrast')} before training and ${T(cp(H8, END, 'typical_right_prob'), { role: 'contrast', sig: 4 })} after.</p>
    <p>TorC finds the ${term('gradient', 'gradients')} for us. The lesson “How the computer finds every gradient at once” shows how.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('new', '5 · Points the model never saw', `
    <p>We compared the models on training points only. We kept ${T(`${F}.setup.test_samples`, 'setup')} new points apart, from the same disc and ring. We look at them once, here.</p>
    <p>The model with eight hidden neurons gets ${pct(T, cp(H8, END, 'test_correct'), `${F}.setup.test_samples`)} of the new points right. So it learned the disc and the ring.</p>
    <p>Each model started from random weights and ran once. Another run would give somewhat different numbers.</p>`),
    },
    {
      kind: 'quiz', id: 'road', purpose: 'transfer', type: 'choice',
      prompt: '<p>On a map, red houses stand on one side of a straight road and green houses on the other side. What is the smallest model that can learn the colour of a house from its position?</p>',
      options: [
        { html: 'A model with one layer, because a straight boundary separates the colours.', correct: true, feedback: 'Yes. The road is straight, and so is this boundary.' },
        { html: 'A model with two hidden neurons, because every boundary needs a bend.', feedback: 'A straight road needs no bend.' },
        { html: 'A model with eight hidden neurons, because more neurons are always better.', feedback: 'Eight neurons can learn it too, but fewer are enough.' },
      ],
      reveal: '<p>A model needs hidden neurons only when the boundary must bend.</p>',
    },
    {
      kind: 'quiz', id: 'count', purpose: 'cumulative', from: 'linear-layer', type: 'numeric',
      answer: `${F}.models.${H4}.parameters`, tolerance: 0.5,
      prompt: () => `<p>A network has two inputs, a hidden layer with four neurons and two outputs. How many ${term('parameter', 'parameters')}, weights and biases together, does it have?</p>`,
      reveal: (D, T) => {
        const L = (k, rest) => T(cp(H4, START, `layers.${k}.${rest}`), 'reveal');
        return `<p>The hidden layer has ${L(0, 'weight.shape.0')} × ${L(0, 'weight.shape.1')} weights. It also has ${L(0, 'bias.shape.0')} biases.</p>
    <p>The output layer has ${L(1, 'weight.shape.0')} × ${L(1, 'weight.shape.1')} weights. It also has ${L(1, 'bias.shape.0')} biases.</p>
    <p>Together that makes ${T(`${F}.models.${H4}.parameters`, 'reveal')} parameters.</p>`;
      },
    },
    {
      kind: 'explain', id: 'why-bend',
      prompt: '<p>Explain why hidden neurons let a model separate the disc from the ring.</p>',
      model: '<p>Each hidden neuron adds a line where the boundary can bend, so enough neurons can close the boundary around the disc.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>The model gives each point two scores, and the sigmoid of their difference is the probability of orange.</p>
    <p>A model with one layer has a straight boundary. Each hidden ReLU neuron adds a line where the boundary can bend, so enough neurons close it around the disc.</p>`,
      next: () => '<p>How does a model turn ten scores into probabilities, and how do you measure its surprise?</p>',
    },
  ],
};
