// Lesson from-digits-to-chatgpt (template 3, conceptual): content beyond TorC (decision 18). The page binds no trace file. These
// tests check that it says it is conceptual, cites a source list in its footer, shows no number except two TorC anchors, makes
// only TorC claims that the course data supports, and that the widget and the figure show what the prose says about them.
// The v3 rules (budgets, glossary, notation order, quiz structure) run on this lesson in lessons.test.mjs.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson, defaultFooter, boundNums } from '../js/core/sections.js';
import { flatLessons, neighbors } from '../js/core/course.js';
import { splitTokens, trainingExamples, tokenRoles, START_SENTENCE, MAX_CHARS } from '../js/widgets/next-token.js';
import lesson, { SOURCES, FIGURE } from '../js/lessons/from-digits-to-chatgpt.js';
import { readJson, course } from './helpers.mjs';

const anchors = readJson('data/anchors.json');
const D = buildD({}, anchorValues(anchors));
const T = makeT(D);
const C = course();
const entry = flatLessons(C).find((l) => l.id === 'from-digits-to-chatgpt');
const plain = () => { const k = globalThis.katex; globalThis.katex = undefined; try { return renderLesson(lesson, D, { course: C }); } finally { globalThis.katex = k; } };
const parts = plain();
const html = parts.map((p) => p.html).join('\n');
const textOf = (h) => h.replace(/<style[\s\S]*?<\/style>/g, ' ').replace(/<template[^>]*>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
const sectionById = (id) => parts.find((p) => p.html.includes(`id="ch-${id}"`)).html;

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
  assert.ok(SOURCES.length >= 5);
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
  for (const id of ['1706.03762', '2203.02155', '1711.05101', '2005.14165', '1508.07909']) assert.ok(SOURCES.some((s) => s.url.endsWith(id)), id);
});

test('no number on the page except two TorC anchors, the pixels and the parameters of the digit network', () => {
  assert.deepEqual(boundNums(html).map((b) => b.path).sort(), ['anchors.mlp_parameters', 'anchors.pixels']);
  assert.equal(T.anchor('pixels').replace(/<[^>]+>/g, ''), '784');
  assert.equal(T.anchor('mlp_parameters').replace(/<[^>]+>/g, ''), '26,506');
  assert.equal(anchors.anchors.mlp_parameters.path, 'files.mnist-mlp.setup.parameters', 'the digit network is the MLP of mnist-mlp');
  assert.equal(anchors.anchors.pixels.path, 'files.mnist-mlp.setup.pixels');
  // every other digit on the page: only the section numbers of the headings
  const rest = textOf(html.replace(/<span class="num"[^>]*>[^<]*<\/span>/g, ' ')).replace(/\b[1-6] ·/g, ' ')
    .replace(/3Blue1Brown|d2l\.ai/g, ' ');   // names of resources, not numbers
  assert.deepEqual(rest.match(/\S*\d\S*/g), null, 'no hand-typed number (decision 18: keep numbers out or cite them)');
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

test('the widget: one sentence gives one training example per token after the first, and the target is the next token', () => {
  const tokens = splitTokens(START_SENTENCE);
  assert.deepEqual(tokens, ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']);
  const ex = trainingExamples(tokens);
  assert.equal(ex.length, tokens.length - 1);
  assert.deepEqual(ex.map((e) => e.target), tokens.slice(1), '"Every token after the first is a target once."');
  ex.forEach((e, i) => {
    assert.deepEqual(e.context, tokens.slice(0, i + 1), 'the model reads one more token at each step');
    assert.equal(e.context.length + 1 <= tokens.length, true);
  });
  // the prose of section 2 walks through the first two examples of the widget's sentence
  const next = textOf(sectionById('next'));
  assert.ok(next.includes(`“${START_SENTENCE}”`));
  assert.deepEqual([ex[0].context, ex[0].target], [['The'], 'cat']);
  assert.deepEqual([ex[1].context, ex[1].target], [['The', 'cat'], 'sat']);
  assert.match(next, /The model reads “The” and must predict “cat”\. Then it reads “The cat” and must predict “sat”\./);
});

test('the widget: the target and every later token are hidden from the model (the look-back quiz and its reveal)', () => {
  for (let n = 2; n <= 9; n++) for (let k = 1; k < n; k++) {
    const r = tokenRoles(n, k);
    assert.equal(r.filter((x) => x === 'read').length, k);
    assert.equal(r.indexOf('target'), k, 'the target comes right after what the model reads');
    assert.ok(r.slice(k + 1).every((x) => x === 'later'));
  }
  const reveal = parts.find((p) => p.section.id === 'look-back').html;
  assert.match(textOf(reveal), /reads only the tokens before the target/);
});

test('the widget: the tokenizer stand-in keeps words with apostrophes and hyphens, splits punctuation, and handles empty text', () => {
  assert.deepEqual(splitTokens('Don’t stop-now, ok?'), ['Don’t', 'stop-now', ',', 'ok', '?']);
  assert.deepEqual(splitTokens('   '), []);
  assert.deepEqual(trainingExamples(splitTokens('Hello')), [], 'one token gives no example');
  assert.deepEqual(trainingExamples([]), []);
  assert.ok(MAX_CHARS >= 60);
});

test('the attention figure: weights chosen by hand, for "it" and the tokens before it only, adding to one, largest on "cat"', () => {
  const att = textOf(sectionById('attention'));
  assert.ok(att.includes(`“${FIGURE.tokens.slice(0, -1).join(' ')}${FIGURE.tokens.at(-1)}”`), 'the figure shows the sentence of the text');
  assert.equal(FIGURE.tokens[FIGURE.reader], 'it');
  assert.equal(FIGURE.weights.length, FIGURE.reader + 1, 'a weight for itself and for every earlier token, none for later ones');
  assert.ok(FIGURE.weights.every((w) => w > 0));
  assert.ok(Math.abs(FIGURE.weights.reduce((a, b) => a + b, 0) - 1) < 1e-9, 'softmax weights add to one');
  const top = FIGURE.weights.indexOf(Math.max(...FIGURE.weights));
  assert.equal(FIGURE.tokens[top], 'cat', '"the model must know that “it” means the cat"');
  const fig = parts.find((p) => p.html.includes('class="ch-fig"')).html;
  assert.match(textOf(fig), /A made-up illustration with weights chosen by hand, not by a trained model or by TorC\./);
  assert.equal((fig.match(/class="ch-cell later"/g) ?? []).length, FIGURE.tokens.length - FIGURE.reader - 1, 'the later tokens are dashed');
});

test('the cumulative quiz (from losses): equal scores give each class one over the number of classes, and more classes a larger surprise', () => {
  const ten = readJson('data/losses.json').ten;
  assert.ok(ten.softmax.every((p) => Math.abs(p - 1 / ten.softmax.length) < 1e-6), 'TorC: ten equal scores give each class one tenth');
  assert.ok(Math.abs(ten.loss - Math.log(ten.softmax.length)) < 1e-5, 'TorC: the surprise is ln of the number of classes');
  const q = lesson.sections.find((s) => s.id === 'equal-scores');
  assert.equal(q.from, 'losses');
  assert.ok(flatLessons(C).findIndex((l) => l.id === 'losses') < flatLessons(C).findIndex((l) => l.id === entry.id));
  assert.match(q.options.find((o) => o.correct).html, /larger/);
  for (const n of [11, 100, 1000]) assert.ok(-Math.log(1 / n) > ten.loss, `${n} classes give a larger surprise than ten`);
});

test('the glossary entries that this lesson introduces name it, and the page links each of them', () => {
  const g = readJson('glossary.json').terms;
  for (const k of ['token', 'embedding', 'attention', 'pretraining']) {
    assert.equal(g[k].lesson, 'from-digits-to-chatgpt', k);
    assert.ok(html.includes(`data-term="${k}"`), `term('${k}') is linked`);
  }
});

test('the later training stages are described as other objectives, and the explain model names the core idea', () => {
  const after = textOf(sectionById('after'));
  assert.match(after, /fine-tuning\s*, people write good answers/);
  assert.match(after, /That goal is not a next token, so it needs another method, called reinforcement learning\./);
  const ex = lesson.sections.find((s) => s.kind === 'explain');
  assert.match(String(ex.model), /the token that really comes next in the text, so the text gives its own targets/);
});
