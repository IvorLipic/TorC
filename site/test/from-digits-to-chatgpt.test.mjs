// Lesson from-digits-to-chatgpt (template 3, conceptual): content beyond TorC (decision 18). The page binds no trace file. These
// tests check that it says it is conceptual, cites a source list in its footer, shows no number except one year that a source
// gives, makes only TorC claims that the course data supports, and that the widget and the figure show what the prose says.
// The widget also runs once on a small fake DOM. The v3 rules (budgets, glossary, notation order, quiz structure) run on this
// lesson in lessons.test.mjs.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, anchorValues } from '../js/core/trace.js';
import { renderLesson, defaultFooter, boundNums } from '../js/core/sections.js';
import { flatLessons, neighbors } from '../js/core/course.js';
import { splitTokens, samplesOf, tokenRoles, mountNextToken, START_SENTENCE, MAX_CHARS } from '../js/widgets/next-token.js';
import lesson, { SOURCES, FIGURE } from '../js/lessons/from-digits-to-chatgpt.js';
import { readJson, course } from './helpers.mjs';

const D = buildD({}, anchorValues(readJson('data/anchors.json')));
const C = course();
const entry = flatLessons(C).find((l) => l.id === 'from-digits-to-chatgpt');
const plain = () => { const k = globalThis.katex; globalThis.katex = undefined; try { return renderLesson(lesson, D, { course: C }); } finally { globalThis.katex = k; } };
const parts = plain();
const html = parts.map((p) => p.html).join('\n');
const textOf = (h) => h.replace(/<style[\s\S]*?<\/style>/g, ' ').replace(/<template[^>]*>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+([,.])/g, '$1').replace(/\s+/g, ' ');
const sectionById = (id) => parts.find((p) => p.html.includes(`id="ch-${id}"`)).html;
const quiz = (id) => lesson.sections.find((s) => s.id === id);
const str = (f) => (typeof f === 'function' ? f(D, () => '') : String(f));

test('the module agrees with course.json: published, conceptual, no trace file, the last page of the main path', () => {
  assert.equal(entry.status, 'published');
  assert.equal(lesson.title, entry.title);
  assert.deepEqual(lesson.trace, []);
  assert.deepEqual(entry.trace, []);
  assert.equal(lesson.template, 3);
  assert.equal(lesson.loop, 'all');
  assert.equal(lesson.conceptual, true);
  assert.ok(entry.est_minutes >= 8 && entry.est_minutes <= 10, 'a short epilogue (brief §6: 8 to 10 minutes)');
  assert.equal(neighbors(C, entry.id).next, null, 'no lesson after it on the main path');
  assert.equal(neighbors(C, entry.id).prev?.id, 'mnist-cnn');
});

test('decision 18: the page says it is conceptual and not a TorC run, and its footer lists every source', () => {
  assert.match(textOf(parts[0].html), /ideas beyond TorC and shows no TorC run/);
  for (const s of SOURCES) {
    assert.match(s.url, /^https:\/\//, s.title);
    assert.ok(s.title.length > 10);
  }
  assert.equal(new Set(SOURCES.map((s) => s.url)).size, SOURCES.length, 'no source twice');
  assert.equal(lesson.sources, SOURCES);
  assert.equal(lesson.footer, undefined, 'the default footer of a conceptual lesson');
  const footer = defaultFooter(lesson, D);
  assert.doesNotMatch(footer, /Numbers: TorC/, 'no trace file, so no "Numbers: TorC" line');
  assert.match(footer, /This page explains ideas beyond TorC/);
  for (const s of SOURCES) assert.ok(footer.includes(`<a href="${s.url}">`), s.url);
  // the four resources of the curated list are linked from the page and cited in the footer
  for (const name of ['3Blue1Brown', 'Andrej Karpathy', 'Michael Nielsen', 'Dive into Deep Learning']) {
    const s = SOURCES.find((x) => x.title.includes(name));
    assert.ok(s, name);
    assert.ok(sectionById('more').includes(`href="${s.url}"`), `${name} is linked in the list`);
  }
  // the papers that the claims rest on
  for (const id of ['1706.03762', '2203.02155', '1711.05101', '2005.14165', '1508.07909', '1904.09751']) assert.ok(SOURCES.some((s) => s.url.endsWith(id)), id);
  assert.match(SOURCES.find((s) => s.title.startsWith('Vaswani')).title, /3\.3 \(the feed-forward layers with ReLU\)/, 'the section of the feed-forward layers');
});

test('no number on the page except the year of the first ChatGPT, which the OpenAI source gives', () => {
  assert.deepEqual(boundNums(html), [], 'no trace number is bound');
  const rest = textOf(html).replace(/\b[1-6] ·/g, ' ').replace(/3Blue1Brown|d2l\.ai/g, ' ');   // section numbers and names
  assert.deepEqual(rest.match(/\S*\d\S*/g), ['2022,'], 'no hand-typed number (decision 18: keep numbers out or cite them)');
  assert.match(textOf(sectionById('after')), /The first ChatGPT, from 2022, got more stages of training/);
  assert.ok(SOURCES.some((s) => s.title.startsWith('OpenAI (2022), Introducing ChatGPT (the first ChatGPT')));
});

test('TorC claims: the course has linear layers, convolutions and AdamW, the digit network uses AdamW, and nothing for embeddings or attention', () => {
  const symbols = flatLessons(C).flatMap((l) => l.symbols ?? []);
  for (const s of ['nn::Linear', 'nn::Conv2d', 'optim::AdamW', 'nn::CrossEntropyLoss']) assert.ok(symbols.includes(s), s);
  assert.ok(!symbols.some((s) => /embed|attention|token/i.test(s)), 'no course symbol for embeddings, attention or tokens');
  const mlp = flatLessons(C).find((l) => l.id === 'mnist-mlp');
  assert.match(mlp.code, /optim::AdamW/, '"The digit network of this course used AdamW too."');
  assert.match(textOf(sectionById('more')), /It has no layer for embeddings or attention, so this course stops at networks for images\./);
  assert.match(textOf(sectionById('loop')), /The digit network of this course used AdamW too\./);
});

test('the widget: one sentence gives one sample per token after the first, and the target is the next token', () => {
  const tokens = splitTokens(START_SENTENCE);
  assert.deepEqual(tokens, ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']);
  const ex = samplesOf(tokens);
  assert.equal(ex.length, tokens.length - 1);
  assert.deepEqual(ex.map((e) => e.target), tokens.slice(1), '"Every token after the first is a target once."');
  ex.forEach((e, i) => assert.deepEqual(e.context, tokens.slice(0, i + 1), 'the model reads one more token at each step'));
  // "the target joins the tokens that the model reads": the context of the next sample is this context plus this target
  ex.slice(0, -1).forEach((e, i) => assert.deepEqual(ex[i + 1].context, [...e.context, e.target]));
  // the prose of section 2 walks through the first two samples of the widget's sentence
  const next = textOf(sectionById('next'));
  assert.ok(next.includes(`“${START_SENTENCE}”`));
  assert.deepEqual([ex[0].context, ex[0].target], [['The'], 'cat']);
  assert.deepEqual([ex[1].context, ex[1].target], [['The', 'cat'], 'sat']);
  assert.match(next, /The model reads “The” and must predict “cat”\. Then it reads “The cat” and must predict “sat”\./);
});

test('the learner step: "I like green tea" has four tokens and gives three samples, with the targets of the reveal', () => {
  const tokens = splitTokens('I like green tea');
  assert.equal(tokens.length, 4, 'the prompt says four tokens');
  const s = samplesOf(tokens);
  assert.equal(s.length, 3);
  const q = quiz('count');
  assert.match(str(q.prompt), /“I like green tea” has no full stop, so it has four tokens/);
  assert.equal(q.options.find((o) => o.correct).html, 'Three samples');
  assert.equal(str(q.reveal), `<p>The targets are ${s.map((x) => `“${x.target}”`).join(', ').replace(/, (?=[^,]*$)/, ' and ')}.</p>`);
  assert.equal(splitTokens('I like green tea.').length, 5, 'with a full stop the widget would count one more token');
});

test('the widget: the target and every later token are hidden from the model', () => {
  for (let n = 2; n <= 9; n++) for (let k = 1; k < n; k++) {
    const r = tokenRoles(n, k);
    assert.equal(r.filter((x) => x === 'read').length, k);
    assert.equal(r.indexOf('target'), k, 'the target comes right after what the model reads');
    assert.ok(r.slice(k + 1).every((x) => x === 'later'));
  }
});

test('the widget: the tokenizer stand-in keeps words with apostrophes and hyphens, splits punctuation, and handles empty text', () => {
  assert.deepEqual(splitTokens('Don’t stop-now, ok?'), ['Don’t', 'stop-now', ',', 'ok', '?']);
  assert.deepEqual(splitTokens('   '), []);
  assert.deepEqual(samplesOf(splitTokens('Hello')), [], 'one token gives no sample');
  assert.deepEqual(samplesOf([]), []);
  assert.ok(MAX_CHARS >= 60);
});

// ---- a small fake DOM, enough for core/dom.js el() and the widget -------------------------------------------------------------
class FakeNode {
  constructor(tag) { this.tag = tag; this.nodeType = 1; this.children = []; this.attrs = {}; this.listeners = {}; this._html = ''; this.className = ''; this.value = ''; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'value') this.value = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(t, f) { (this.listeners[t] ??= []).push(f); }
  dispatch(t) { for (const f of this.listeners[t] ?? []) f({ type: t }); }
  append(...c) { this.children.push(...c); }
  replaceChildren(...c) { this.children = c; }
  set innerHTML(h) { this._html = h; this.children = []; }
  get innerHTML() { return this._html; }
  get textContent() { return this.text ?? (this._html.replace(/<[^>]+>/g, '') + this.children.map((c) => c.textContent).join('')); }
  find(pred) { if (pred(this)) return this; for (const c of this.children) { const r = c.find?.(pred); if (r) return r; } return null; }
  findAll(pred, out = []) { if (pred(this)) out.push(this); for (const c of this.children) c.findAll?.(pred, out); return out; }
}
function withFakeDom(fn) {
  const saved = globalThis.document;
  globalThis.document = {
    createElement: (t) => new FakeNode(t),
    createTextNode: (s) => Object.assign(new FakeNode('#text'), { text: String(s) }),
    getElementById: () => null,
    head: new FakeNode('head'),
  };
  try { return fn(); } finally { globalThis.document = saved; }
}

test('the widget on a fake DOM: steps, keeps both buttons focusable at the ends, announces a step at once and typing later', async () => {
  await withFakeDom(async () => {
    const host = new FakeNode('section');
    const w = mountNextToken(host, { announceDelay: 5 });
    const btn = (label) => host.find((n) => n.tag === 'button' && n.textContent === label);
    const read = host.find((n) => n.className === 'nt-read');
    const live = host.find((n) => n.className === 'nt-sr');
    const input = host.find((n) => n.tag === 'input');
    const roles = () => host.findAll((n) => n.className.startsWith('nt-tok ')).map((n) => n.className.slice(7));
    assert.equal(live.getAttribute('aria-live'), 'polite');
    assert.equal(read.getAttribute('aria-live'), null, 'the visible readout is not a live region');
    assert.match(read.innerHTML, /Sample 1 of 6\. The model reads “The”\. Its target is “cat”\./);
    assert.deepEqual(roles(), ['read', 'target', 'later', 'later', 'later', 'later', 'later']);
    assert.equal(btn('Previous sample').getAttribute('aria-disabled'), 'true');
    assert.equal(live.innerHTML, '', 'nothing announced on load');
    btn('Next sample').dispatch('click');
    assert.match(live.innerHTML, /^Sample 2 of 6\./, 'a step is announced at once');
    for (let i = 0; i < 10; i++) btn('Next sample').dispatch('click');
    assert.match(read.innerHTML, /Sample 6 of 6\. The model reads “The cat sat on the mat”\. Its target is “\.”\./);
    assert.equal(btn('Next sample').getAttribute('aria-disabled'), 'true');
    assert.equal(btn('Next sample').getAttribute('disabled'), null, 'aria-disabled keeps the focus on the button');
    input.value = 'I like green tea';
    input.dispatch('input');
    assert.match(read.innerHTML, /Sample 1 of 3\./, 'typing redraws at once');
    assert.match(live.innerHTML, /^Sample 6 of 6\./, 'typing is not announced at every keystroke');
    await new Promise((r) => setTimeout(r, 20));
    assert.match(live.innerHTML, /^Sample 1 of 3\./, 'typing is announced after a pause');
    input.value = 'Hi';
    input.dispatch('input');
    assert.match(read.innerHTML, /Type a sentence with at least two words\./);
    w.destroy();
  });
});

test('writing: the trained model predicts, picks a likely token at random with the probabilities, adds it, and repeats', () => {
  const writes = textOf(parts.find((p) => p.html.includes('aria-label="How the model writes"')).html);
  assert.match(writes, /It reads the text so far, and softmax gives every token a probability\./);
  assert.match(writes, /It picks a likely token, often at random with these probabilities\. It adds the token to the text and predicts again\./);
  const answer = textOf(parts.at(-1).html);
  assert.match(answer, /ChatGPT writes by predicting the next token, picking one and adding it to the text\./, 'The answer restates the loop');
  const tasks = str(lesson.sections.find((s) => s.id === 'pairs').tasks);
  assert.equal(lesson.sections.find((s) => s.id === 'pairs').tasks().length, 3);
  assert.ok(lesson.sections.find((s) => s.id === 'pairs').tasks().some((t) => /its pick joins the text/.test(t.see)), tasks);
  // the next step points at real pages
  assert.match(parts.at(-1).html, /href="\.\.\/tensors\/"/);
  assert.ok(flatLessons(C).some((l) => l.id === 'tensors' && l.status === 'published'));
});

test('the attention figure: attention weights chosen by hand, for "it" and the tokens before it only, adding to one, largest on "cat"', () => {
  const att = textOf(sectionById('attention'));
  assert.ok(att.includes(`“${FIGURE.tokens.slice(0, -1).join(' ')}”`), 'the figure shows the sentence of the text');
  assert.match(att, /A dot product is large when two vectors have large numbers in the same places\./);
  assert.match(att, /Training makes the vectors of “it” and “cat” match, so “it” takes much from “cat”\./);
  assert.equal(FIGURE.tokens[FIGURE.reader], 'it');
  assert.equal(FIGURE.weights.length, FIGURE.reader + 1, 'an attention weight for itself and for every earlier token, none for later ones');
  assert.ok(FIGURE.weights.every((w) => w > 0));
  assert.ok(Math.abs(FIGURE.weights.reduce((a, b) => a + b, 0) - 1) < 1e-9, 'softmax attention weights add to one');
  const top = FIGURE.weights.indexOf(Math.max(...FIGURE.weights));
  assert.equal(FIGURE.tokens[top], 'cat', '"the word “it” means the cat"');
  const fig = parts.find((p) => p.html.includes('class="ch-fig"')).html;
  assert.match(textOf(fig), /A made-up illustration with attention weights chosen by hand, not by a trained model or by TorC\./);
  assert.match(fig, /aria-label="A made-up example\. The token “it” gives its largest attention weight to “cat”/);
  assert.equal((fig.match(/class="ch-cell later"/g) ?? []).length, FIGURE.tokens.length - FIGURE.reader - 1, 'the later tokens are dashed');
  // after the sentence that contrasts them with parameters, the page says "attention weights", never a bare "weights" for them
  const afterContrast = att.slice(att.indexOf('the weights were parameters') + 'the weights were parameters'.length);
  assert.doesNotMatch(afterContrast.replace(/attention weights?/g, ''), /\bweights?\b/);
});

test('the cumulative quiz (from losses): equal scores give each class one over the number of classes, and more classes a larger surprise', () => {
  const ten = readJson('data/losses.json').ten;
  assert.ok(ten.softmax.every((p) => Math.abs(p - 1 / ten.softmax.length) < 1e-6), 'TorC: ten equal scores give each class one tenth');
  assert.ok(Math.abs(ten.loss - Math.log(ten.softmax.length)) < 1e-5, 'TorC: the surprise is ln of the number of classes');
  const q = quiz('equal-scores');
  assert.equal(q.from, 'losses');
  assert.match(str(q.prompt), /^<p>Suppose that a language model gives every token the same score\./, 'a supposition, not a claim about untrained models');
  assert.ok(flatLessons(C).findIndex((l) => l.id === 'losses') < flatLessons(C).findIndex((l) => l.id === entry.id));
  assert.match(q.options.find((o) => o.correct).html, /larger/);
  for (const n of [11, 100, 1000]) assert.ok(-Math.log(1 / n) > ten.loss, `${n} classes give a larger surprise than ten`);
});

test('the predict quiz: options of similar length, each wrong one named in its feedback, and the setup does not define the answer', () => {
  const q = quiz('where-target');
  const lens = q.options.map((o) => o.html.split(/\s+/).length);
  assert.ok(Math.max(...lens) - Math.min(...lens) <= 8, `option lengths ${lens}`);
  assert.match(str(q.options[2].feedback), /This is the belief that the model memorizes the text\./);
  assert.doesNotMatch(str(q.options[0].feedback), /stage/, 'stages come in section 4');
  const setup = parts[lesson.sections.findIndex((s) => s.setup)].html;
  assert.doesNotMatch(setup, /language model|continue/, 'the setup does not hint at the answer');
  assert.match(textOf(sectionById('tokens')), /A model that predicts the next token of a text is a language model\./, 'defined after the quiz');
});

test('the glossary entries that this lesson introduces name it, the page links each of them, and attention looks back only', () => {
  const g = readJson('glossary.json').terms;
  for (const k of ['token', 'embedding', 'attention', 'pretraining']) {
    assert.equal(g[k].lesson, 'from-digits-to-chatgpt', k);
    assert.ok(html.includes(`data-term="${k}"`), `term('${k}') is linked`);
  }
  assert.match(g.attention.short, /itself and the tokens before it/);
  assert.match(g.sample.short, /^A sample is one item of a data set/);
  assert.doesNotMatch(textOf(html), /training examples?/i, 'one name: sample (glossary), never "training example"');
});

test('the later training stages are described as other objectives, and the explain model names the core idea', () => {
  const after = textOf(sectionById('after'));
  assert.match(after, /fine-tuning, people write good answers/);
  assert.match(after, /That goal is not a next token, so it needs another method, called reinforcement learning\./);
  assert.match(str(quiz('self-target')?.model ?? lesson.sections.find((s) => s.kind === 'explain').model), /the token that really comes next in the text, so the text gives its own targets/);
});
