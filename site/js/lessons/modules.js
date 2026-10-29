// Lesson "modules": a module holds named parameters and a forward(); Sequential; the order of parameters(); and why a module
// must not run again before backward(). Numbers and messages come from data/modules.json (tools/trace/trace_modules.cpp).
// Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { escapeHtml } from '../core/dom.js';
import { mountModelViewer, mountCallOrder } from '../widgets/module-widgets.js';

const F = 'files.modules';
const sec = (id, title, body) => `<section class="md" aria-labelledby="md-${id}"><h2 id="md-${id}">${title}</h2>${body}</section>`;
const model = (D, id) => D.files.modules.models.find((m) => m.id === id);
const idx = (D, id) => D.files.modules.models.findIndex((m) => m.id === id);
const err = (exception, what) => `<code>${escapeHtml(exception)}: ${escapeHtml(what)}</code>`;

const STYLE = `<style>
.md p{max-width:44rem}
.md h2{margin-top:2rem}
.md-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.md-code code{background:none;padding:0}
</style>`;

export default {
  id: 'modules',
  title: 'Modules',
  trace: ['modules.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Modules</h1>
    <p class="lead">In TorC, a model is a module. A module holds parameters and has a forward function. This lesson shows how modules fit together, and one rule that you must follow when you use them.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'A module holds named parameters and has a <code>forward()</code> function.',
        '<code>Sequential</code> runs its modules in order and collects all their parameters.',
        '<code>parameters()</code> returns the parameters in no fixed order. Find a parameter by its name.',
        'A module keeps the values of its last forward pass. Call <code>backward()</code> before you call the module again.',
      ],
    },
    {
      kind: 'text',
      html: () => sec('module', '1 · What a module is', `
    <p><code>nn::Linear</code> is a module. Its constructor registers two parameters, which are called <code>weight</code> and <code>bias</code>. Its <code>forward()</code> function computes the output from the input. <code>nn::ReLU</code> is also a module, but it has no parameters. You call a module like a function, and the call runs <code>forward()</code>.</p>
    <pre class="md-code" tabindex="0" role="group" aria-label="A module in TorC"><code>nn::Linear layer(784, 32);
for (const auto&amp; [name, p] : layer.named_parameters())
    std::cout &lt;&lt; name &lt;&lt; "\\n";   // weight and bias
Variable y = layer(x);          // runs forward(x)</code></pre>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('sequential', '2 · Sequential', `
    <p><code>nn::Sequential</code> is a module that holds other modules. Its <code>forward()</code> passes the output of each module to the next one, in the order you added them. Its <code>parameters()</code> collects the parameters of all its modules, and you give that list to the optimizer.</p>
    <pre class="md-code" tabindex="0" role="group" aria-label="A Sequential model in TorC"><code>nn::Sequential model;
model.add(std::make_unique&lt;nn::Linear&gt;(784, 32));
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Linear&gt;(32, 10));
optim::SGD opt(model.parameters(), 0.01f);</code></pre>
    <p>This model has ${T(`${F}.models.${idx(D, 'mlp')}.total`)} parameters in ${T(`${F}.models.${idx(D, 'mlp')}.tensors`)} tensors. Select a model below to see the parameters and the output shape of each module.</p>`),
    },
    {
      kind: 'widget', id: 'models', className: 'widget', label: 'Interactive: the modules of four models, their parameters and output shapes',
      mount: (host, D) => mountModelViewer(host, D),
    },
    {
      kind: 'text',
      html: () => sec('order', '3 · Find parameters by name', `
    <p>A module keeps its parameters in an unordered map. So <code>parameters()</code> returns them in an order that the standard library chooses. This order is not the order in which the module registered them, and it can change between compilers. To get one parameter, use its name, as in <code>layer.named_parameters().at("weight")</code>. To check a model, compare names and shapes, not positions in the list.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('calls', '4 · Call backward() before the next forward pass', `
    <p>Each module keeps the values of its last forward pass, because <code>backward()</code> needs them. When you call the module again, it clears those values and stores new ones. So the graph of the first call is gone.</p>`),
    },
    {
      kind: 'predict', id: 'p-order', title: 'Before you try it',
      prompt: () => '<p>You call <code>model(x1)</code> and compute the loss <code>L1</code>. Then you call <code>model(x2)</code>. What happens when you now call <code>L1.backward()</code>?</p>',
      reveal: (D) => { const o = D.files.modules.orders.find((x) => x.id === 'early-second-forward'); return `<p>TorC throws an error, because the values that <code>L1</code> needs are gone. The message is ${err(o.exception, o.what)}. Call <code>L1.backward()</code> before the second forward pass.</p>`; },
    },
    {
      kind: 'widget', id: 'orders', className: 'widget', label: 'Interactive: three orders of forward and backward calls',
      mount: (host, D) => mountCallOrder(host, D),
    },
    {
      kind: 'check', id: 'c-count',
      prompt: () => '<p>How many parameters does the model <code>Linear(784, 32)</code>, <code>ReLU</code>, <code>Linear(32, 10)</code> have? How many of them come from <code>ReLU</code>?</p>',
      reveal: (D, T) => `<p>It has ${T(`${F}.models.${idx(D, 'mlp')}.total`)} parameters. <code>ReLU</code> adds none, because it has no parameters.</p>`,
    },
    {
      kind: 'check', id: 'c-mismatch',
      prompt: () => '<p>A model has <code>Linear(784, 32)</code> and then <code>Linear(64, 10)</code>. When does TorC report the mistake?</p>',
      reveal: (D) => { const l = model(D, 'mismatch').layers.find((x) => x.what); return `<p>TorC reports it when the forward pass reaches the second <code>Linear</code>. The constructors do not check that the sizes fit. The message is ${err(l.exception, l.what)}.</p>`; },
    },
    {
      kind: 'text',
      html: () => '<section class="md"><p>Go back to the <a href="../../">course map</a> to continue the lessons.</p></section>',
    },
  ],
};
