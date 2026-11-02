// Lesson "modules" (appendix "Using TorC", curriculum v3, template 3): nn::Module (named parameters and forward()), Sequential,
// named_parameters() and why parameters() has no fixed order inside one module (an unordered map; D7), writing a module of our
// own, and the forward-cache rule (a module call clears the in-between results of its last call). Brief §6 appendix and §7.
// Numbers and messages come from data/modules.json (tools/trace/trace_modules.cpp) through a role-tagged T() or the recorded text.
// Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { escapeHtml } from '../core/dom.js';
import { term } from '../core/blocks.js';
import { mountModelViewer, mountCallOrder } from '../widgets/module-widgets.js';

const F = 'files.modules';
const sec = (id, title, body) => `<section class="md" aria-labelledby="md-${id}"><h2 id="md-${id}">${title}</h2>${body}</section>`;
const model = (D, id) => D.files.modules.models.find((m) => m.id === id);
const idx = (D, id) => D.files.modules.models.findIndex((m) => m.id === id);
const err = (exception, what) => `<code>${escapeHtml(exception)}: ${escapeHtml(what)}</code>`;
const shp = (a) => `<code>(${a.join(', ')})</code>`;
const code = (label, body) => `<pre class="md-code" tabindex="0" role="group" aria-label="${label}"><code>${body}</code></pre>`;

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
  template: 3,
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Modules</h1>
    <p class="lead">In TorC, a ${term('model')} is a module. This page shows how a module holds its ${term('parameter', 'parameters')} and how <code>Sequential</code> joins modules. It also gives the rule for calling a module again.</p></section>`,
    },
    {
      kind: 'text',
      html: () => sec('module', '1 · What a module is', `
    <p>A module is an object that holds named parameters and has a <code>forward()</code> function. <code>nn::Linear</code> is a module. Its constructor registers two parameters, named <code>weight</code> and <code>bias</code>. <code>nn::ReLU</code> is also a module, and it has no parameters.</p>
    <p>You call a module like a function, and the call runs <code>forward()</code>.</p>
    ${code('A module in TorC', `nn::Linear layer(784, 32);
for (const auto&amp; [name, p] : layer.named_parameters())
    std::cout &lt;&lt; name &lt;&lt; "\\n";   // weight and bias, in no fixed order
Variable y = layer(x);          // runs forward(x)`)}`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('sequential', '2 · Sequential', `
    <p><code>nn::Sequential</code> is a module that holds other modules. Its <code>forward()</code> passes the output of each module to the next one, in the order you added them. Its <code>parameters()</code> collects the parameters of all its modules. You give that list to the ${term('optimizer')}.</p>
    ${code('A Sequential model in TorC', `nn::Sequential model;
model.add(std::make_unique&lt;nn::Linear&gt;(784, 32));
model.add(std::make_unique&lt;nn::ReLU&gt;());
model.add(std::make_unique&lt;nn::Linear&gt;(32, 10));
optim::SGD opt(model.parameters(), 0.01f);`)}
    <p>This model has ${T(`${F}.models.${idx(D, 'mlp')}.total`, 'setup')} parameters in ${T(`${F}.models.${idx(D, 'mlp')}.tensors`, 'setup')} ${term('tensor', 'tensors')}.</p>`),
    },
    {
      kind: 'widget', id: 'models', className: 'widget', label: 'Interactive: the modules of four models, their parameters and output shapes',
      mount: (host, D) => mountModelViewer(host, D),
      tasks: () => [
        { do: 'Select the model <code>784 → 32 → 10</code>.', see: `Each <code>Linear</code> lists a ${term('bias')} and a ${term('weight')}. <code>ReLU</code> lists no parameters. The total counts every one.` },
        { do: 'Select <code>no ReLU</code>.', see: `The total stays the same, because <code>ReLU</code> has no parameters. The output ${term('shape', 'shapes')} stay the same too.` },
        { do: 'Select <code>wrong sizes</code>.', see: 'The forward pass stops at the second <code>Linear</code> with a <code>ShapeError</code>. The constructor did not check the sizes.' },
      ],
    },
    {
      kind: 'quiz', id: 'p-first', purpose: 'predict', type: 'choice', title: 'Before you read on',
      prompt: () => '<p>You add <code>Linear(784, 32)</code>, <code>ReLU</code> and <code>Linear(32, 10)</code> to a <code>Sequential</code>. Then you call <code>parameters()</code>. Which parameter is the first entry of the list?</p>',
      options: [
        { html: 'The weight of the first <code>Linear</code>, because the constructor registers it first.', feedback: () => 'The module keeps its parameters in a map, not in a list. The map does not remember the order of registration.' },
        { html: 'One of the two parameters of the first <code>Linear</code>, but TorC does not fix which one.', correct: true, feedback: () => 'Right. Only the order of the modules is fixed.' },
        { html: 'The bias of the first <code>Linear</code>, always.', feedback: () => 'One build may list the bias first. Another standard library may list the weight first, so you cannot rely on it.' },
      ],
      reveal: (D) => `<p>In one run of TorC, the first entry has the shape ${shp(D.files.modules.parameter_order.sequential_shapes[0])}. That is a bias, and the weight has the shape ${shp(D.files.modules.parameter_order.sequential_shapes[1])}. Another compiler may give the other order.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('order', '3 · Find a parameter by its name', `
    <p><code>named_parameters()</code> returns a map from names to parameters, and <code>parameters()</code> returns pointers in a list. The map is an <code>std::unordered_map</code>, and the C++ standard does not fix its order. So the two parameters of one module come in an order that you cannot rely on.</p>
    <p>A <code>Sequential</code> lists its modules in the order you added them. Its own <code>named_parameters()</code> is empty, because each module keeps its own parameters. To reach a parameter by name, keep a pointer to the ${term('layer')}.</p>
    ${code('Find a parameter by name', `auto first = std::make_unique&lt;nn::Linear&gt;(784, 32);
nn::Linear* first_layer = first.get();       // keep a pointer
model.add(std::move(first));
Variable&amp; w = first_layer->named_parameters().at("weight");`)}
    <p>To check a model, compare names and shapes. Never compare positions in the list.</p>`),
    },
    {
      kind: 'quiz', id: 'p-order', purpose: 'predict', type: 'choice', title: 'Before you read the rule',
      prompt: () => `<p>You call <code>model(x1)</code> and compute the ${term('loss')} <code>L1</code>. Then you call <code>model(x2)</code>. What happens when you call <code>L1.backward()</code> now?</p>`,
      options: [
        { html: 'It works. The graph of <code>L1</code> is a separate object that the second call does not touch.', feedback: () => 'The graph needs in-between results that the module keeps. The second call replaces them.' },
        { html: 'TorC throws an error, because the in-between results of the first call are gone.', correct: true, feedback: () => 'Right. The second call cleared them.' },
        { html: 'It works, but it uses the data of <code>x2</code>.', feedback: () => 'There is no mix of the two graphs. TorC reports an error before it computes anything.' },
      ],
      reveal: (D) => { const o = D.files.modules.orders.find((x) => x.id === 'early-second-forward'); return `<p>The message is ${err(o.exception, o.what)}.</p>`; },
    },
    {
      kind: 'text',
      html: (D) => sec('cache', '4 · Call backward() before the next forward pass', `
    <p><code>backward()</code> needs the in-between results of the forward pass, and TorC does not keep them alive for you. A module that makes more than one in-between result keeps them in <code>forward_cache_</code>. <code>nn::Linear</code> keeps the transposed weight, the product and the sum there.</p>
    <p>Every call of a module clears this cache first. So call <code>backward()</code> on a loss before you call the same module again. After two forward passes, only the loss of the latest one still works.</p>`),
    },
    {
      kind: 'widget', id: 'orders', className: 'widget', label: 'Interactive: three orders of forward and backward calls',
      mount: (host, D) => mountCallOrder(host, D),
      tasks: () => [
        { do: 'Select <code>backward, then forward</code>.', see: `Both <code>backward()</code> calls run, and every parameter gets its ${term('gradient')}.` },
        { do: 'Select <code>second forward first</code>.', see: 'The first loss fails with a <code>TorcError</code>, because the second forward pass cleared its in-between results.' },
        { do: 'Select <code>backward of the second loss</code>.', see: 'It runs. Only the latest forward pass is still valid.' },
      ],
    },
    {
      kind: 'text',
      html: (D) => sec('own', 'Your own module', `
    <p>A module of your own registers its parameters in the constructor. If <code>forward()</code> makes an in-between result, keep it in <code>forward_cache_</code>, as below.</p>
    ${code('A module with two operations', `class Affine : public nn::Module {
public:
    Affine() {
        register_parameter("scale", Variable(Tensor({2.f}, {1}), true));
        register_parameter("shift", Variable(Tensor({1.f}, {1}), true));
    }
    Variable forward(const Variable&amp; x) const override {
        forward_cache_.emplace_back(mul(x, named_parameters().at("scale")));
        Variable&amp; scaled = forward_cache_.back();    // kept alive
        return add(scaled, named_parameters().at("shift"));
    }
};`)}
    <p>A local <code>Variable scaled = mul(…)</code> would die when <code>forward()</code> returns. Then <code>backward()</code> reports ${err(D.files.modules.custom_module[0].exception, D.files.modules.custom_module[0].what)}.</p>`),
    },
    {
      kind: 'gotcha', title: 'Why did my backward() fail?',
      html: (D) => `<p>Look for a second call of the same module between the forward pass and <code>backward()</code>, or for an in-between result that is a local variable. The message is ${err(D.files.modules.orders[1].exception, D.files.modules.orders[1].what)}. The next page lists more ways to meet it.</p>`,
    },
    {
      kind: 'quiz', id: 'c-count', purpose: 'check', type: 'numeric', title: 'Check yourself', unit: 'parameters',
      answer: `${F}.models.0.total`, tolerance: 0.5,
      prompt: () => '<p>A <code>Linear(in, out)</code> has <code>in × out</code> weights and <code>out</code> biases. How many parameters do <code>Linear(784, 32)</code>, <code>ReLU</code> and <code>Linear(32, 10)</code> have in total? A calculator is fine.</p>',
      reveal: (D, T) => `<p>The first layer has ${T(`${F}.models.0.layers.0.params.1.numel`, 'reveal')} weights and ${T(`${F}.models.0.layers.0.params.0.numel`, 'reveal')} biases. The second layer has ${T(`${F}.models.0.layers.2.params.1.numel`, 'reveal')} weights and ${T(`${F}.models.0.layers.2.params.0.numel`, 'reveal')} biases. <code>ReLU</code> adds none. The total is ${T(`${F}.models.0.total`, 'reveal')}.</p>`,
    },
    {
      kind: 'quiz', id: 'c-mismatch', purpose: 'check', type: 'choice', title: 'Check yourself',
      prompt: () => '<p>A model has <code>Linear(784, 32)</code> and then <code>Linear(64, 10)</code>. When does TorC report the mistake?</p>',
      options: [
        { html: 'When you build the model, because the constructors compare the sizes.', feedback: () => 'The constructors do not see each other. Each one checks only its own sizes.' },
        { html: 'In the forward pass, when it reaches the second <code>Linear</code>.', correct: true, feedback: () => 'Right. The two sizes first meet when the second layer multiplies its input.' },
        { html: 'In <code>backward()</code>, when the gradients do not fit.', feedback: () => 'The forward pass fails first, so there is no graph to run backward.' },
      ],
      reveal: (D) => { const l = model(D, 'mismatch').layers.find((x) => x.what); return `<p>The message is ${err(l.exception, l.what)}.</p>`; },
    },
    {
      kind: 'text',
      html: () => '<section class="md"><p>The next lesson, <a href="../torc-pitfalls/">Common errors in TorC programs</a>, collects the errors that TorC programs meet most often.</p></section>',
    },
  ],
};
