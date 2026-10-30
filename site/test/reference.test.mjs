// The reference content of v3 (brief §8 "Notation", "Maths refreshers", "Glossary"; decision 18; R9): the ten maths refreshers,
// the notation guide and the glossary, and the two appendix pages built from them (lessons/notation.js, lessons/glossary.js).
// The well-formedness rules that every entry shares are in v3.test.mjs; this file checks the content itself.
import test from 'node:test';
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { REFRESHERS } from '../js/content/refreshers.js';
import { NOTATION } from '../js/content/notation.js';
import { refresher, term, sym } from '../js/core/blocks.js';
import { flatLessons } from '../js/core/course.js';
import { buildD, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { linkTerms } from '../js/lessons/glossary.js';
import { proseProblems, sectionWords, readJson, course } from './helpers.mjs';
import { tokens } from './v3.mjs';

const glossary = readJson('glossary.json');

const KEYS = ['slope', 'sum', 'dot-product', 'matrix-product', 'exp-ln', 'probability', 'mean-variance', 'normal-distribution', 'contour-map', 'log-axis'];
const words = (h) => h.replace(/<svg[\s\S]*?<\/svg>/g, ' ').replace(/<[^>]+>/g, ' ').split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
const textOf = (h) => h.replace(/<svg[\s\S]*?<\/svg>/g, ' ').replace(/<code[^>]*>[\s\S]*?<\/code>/g, ' ').replace(/<[^>]+>/g, ' ');

// ---- refreshers -----------------------------------------------------------------------------------------------------------
test('refreshers: exactly the ten of brief §8, in its order', () => {
  assert.deepEqual(Object.keys(REFRESHERS), KEYS);
});

for (const [key, r] of Object.entries(REFRESHERS)) {
  test(`refresher ${key}: 100-200 words, one accessible picture, the writing rules, a cited source`, () => {
    const n = words(r.html);
    assert.ok(n >= 100 && n <= 200, `${n} words (with code), 100 to 200`);
    // a refresher may stand alone in a text section, whose budget is 150 words (v3.mjs counts the way sectionWords does)
    assert.ok(sectionWords(r.html) <= 150, `${sectionWords(r.html)} words count toward the 150-word budget of a text section`);
    const svgs = r.html.match(/<svg[^>]*>/g) ?? [];
    assert.equal(svgs.length, 1, 'one picture');
    assert.match(svgs[0], /role="img"/);
    assert.match(svgs[0], /aria-label="[A-Z][^"]{20,}\."/, 'the label describes the picture in a sentence');
    assert.deepEqual(proseProblems(`<p>${/aria-label="([^"]*)"/.exec(svgs[0])[1]}</p>`, { tics: true }), [], 'the label follows the writing rules');
    assert.match(svgs[0], /viewBox="0 0 (\d+) \d+"/);
    assert.ok(Number(/viewBox="0 0 (\d+)/.exec(svgs[0])[1]) <= 320, 'fits a 390 px phone without scaling the text below 12 px');
    assert.match(svgs[0], /max-width:\d+px;height:auto/, 'scales down with the column');
    // colours come from the theme tokens, so the picture follows the theme
    const colours = [...r.html.matchAll(/(?:fill|stroke):([^;"]+)/g)].map((m) => m[1].trim()).filter((c) => c !== 'none');
    assert.ok(colours.length > 0);
    for (const c of colours) assert.match(c, /^var\(--[a-z0-9]+\)$/, `colour ${c} is a theme token`);
    assert.deepEqual(proseProblems(r.html, { tics: true }), []);
    // no KaTeX, no T(): static markup, and every digit of the prose sits in <code> (school arithmetic, not a trace value)
    assert.doesNotMatch(r.html, /class="num"|katex/);
    assert.doesNotMatch(textOf(r.html.replace(/<a [^>]*>[^<]*<\/a>/g, ' ')), /\d/, 'digits in the prose are written inside <code> (the source title aside)');
    // decision 18: conceptual content says so and cites a source
    assert.match(r.source.url, /^https:\/\//);
    assert.ok(r.html.includes(`<a href="${r.source.url}">${r.source.title}</a>`), 'the body cites its source');
    assert.match(r.html, /not a TorC run/);
    // the html is well formed: every element that opens also closes
    const open = tokens(r.html).filter((t) => t.type === 'open').length;
    const closes = (r.html.match(/<\/[a-z]+>|\/>/g) ?? []).length;
    assert.equal(open, closes, 'balanced markup');
  });
}

test('refresher exp-ln says that log means ln on this site, and that school log has base 10', () => {
  const t = textOf(REFRESHERS['exp-ln'].html);
  assert.match(t, /On this site, log always means ln/);
  assert.match(REFRESHERS['exp-ln'].html, /At school, log often means the logarithm with base <code>10<\/code>/);
});

// The worked numbers of the refreshers are plain arithmetic: recompute each one.
test('refreshers: the worked arithmetic is right', () => {
  const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);
  near(3.1 ** 2, 9.61); near((3.1 ** 2 - 9) / 0.1, 6.1, 1e-9); near((3.01 ** 2 - 9) / 0.01, 6.01, 1e-9);   // slope
  assert.equal(3 + 1 + 4, 8);                                                                               // sum
  assert.equal(2 * 4 + 3 * 1, 11);                                                                          // dot product
  const A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]];
  const AB = A.map((r) => B[0].map((_, j) => r[0] * B[0][j] + r[1] * B[1][j]));
  assert.deepEqual(AB, [[19, 22], [43, 50]]);                                                               // matrix product
  near(Math.E, 2.718, 5e-4); near(Math.log10(100), 2);                                                      // exp and ln
  near(6 * (1 / 6), 1); near(1 - 0.5 - 0.3, 0.2, 1e-12);                                                   // probability
  const xs = [2, 4, 6, 8], mean = xs.reduce((a, b) => a + b) / xs.length;
  const v = xs.map((x) => (x - mean) ** 2).reduce((a, b) => a + b) / xs.length;
  assert.equal(mean, 5); assert.equal(v, 5); near(Math.sqrt(v), 2.24, 5e-3);                                // mean and variance
  near(Math.sqrt(1 * 100), 10);                                                                             // log axis: the geometric middle
  const html = Object.values(REFRESHERS).map((r) => r.html).join('');
  for (const s of ['3.1² = 9.61', '0.61 / 0.1 = 6.1', '0.0601 / 0.01 = 6.01', '3 + 1 + 4 = 8', '2 × 4 + 3 × 1 = 8 + 3 = 11', '1 × 5 + 2 × 7 = 19',
    '<code>22</code>, <code>43</code> and <code>50</code>', '20 / 4 = 5', '9, 1, 1, 9', '6 × 1/6 = 1', '1 × 10 = 10']) assert.ok(html.includes(s), s);
});

// The glossary terms that each refresher mentions in its prose. A lesson that opens a refresher links these terms with term()
// before it (the v3 glossary rule wants the first use of a term linked). Kept short on purpose; change it with the refresher.
const REFRESHER_TERMS = {
  slope: [], sum: [], 'dot-product': ['vector', 'weight'], 'matrix-product': ['matrix'], 'exp-ln': [], probability: ['probability'],
  'mean-variance': [], 'normal-distribution': [], 'contour-map': [], 'log-axis': [],
};
const skipTags = (e) => ['pre', 'code', 'svg', 'h1', 'h2', 'h3', 'style', 'script'].includes(e.tag);
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const formsRe = (g) => new RegExp(`(?<![\\p{L}\\p{N}])(?:${g.forms.map(escRe).join('|')})(?![\\p{L}\\p{N}])`, 'iu');

test('refreshers: the glossary terms in their prose are the listed few', () => {
  for (const key of KEYS) {
    const prose = tokens(refresher(key)).filter((t) => t.type === 'text' && !t.within.some(skipTags)).map((t) => t.text).join(' ');
    const hits = Object.entries(glossary.terms).filter(([, g]) => formsRe(g).test(prose)).map(([k]) => k);
    assert.deepEqual(hits, REFRESHER_TERMS[key], key);
  }
});

test('refreshers: a symbol of the notation guide appears only where a lesson after its introduction can open the refresher', () => {
  // Σ and subscripts (linear-layer): sum. exp-ln writes exp and ln only inside <code>, so the check does not see them there; a lesson
  // opens it from nonlinearity on (see the head of content/refreshers.js). No other refresher uses a symbol.
  const uses = (key) => {
    const toks = tokens(refresher(key));
    const prose = toks.filter((t) => t.type === 'text' && !t.within.some((e) => e.tag === 'code' || e.tag === 'pre')).map((t) => t.text).join('\n');
    const code = toks.filter((t) => t.type === 'text' && t.within.some((e) => e.tag === 'code')).map((t) => t.text).join('\n');
    return NOTATION.filter((n) => (n.match.text ?? []).some((m) => prose.includes(m)) || (n.match.latex ?? []).some((m) => code.includes(m))).map((n) => n.key);
  };
  const want = { sum: ['sum', 'subscript'] };
  for (const key of KEYS) assert.deepEqual(uses(key), want[key] ?? [], key);
});

// ---- notation -------------------------------------------------------------------------------------------------------------
const order = flatLessons(course()).map((l) => l.id);
const BRIEF_SYMBOLS = { arrow: 'why-gradients', 'dl-dw': 'why-gradients', sum: 'linear-layer', subscript: 'linear-layer', transpose: 'linear-layer',
  sigma: 'nonlinearity', exp: 'nonlinearity', ln: 'losses', partial: 'backprop', mu: 'optimizers', beta: 'optimizers', lambda: 'optimizers',
  epsilon: 'optimizers', hat: 'optimizers', floor: 'convolution', overline: 'backprop-lab' };

test('notation: every symbol of brief §8, with its introducing lesson, in course order', () => {
  const byKey = Object.fromEntries(NOTATION.map((n) => [n.key, n]));
  for (const [key, lesson] of Object.entries(BRIEF_SYMBOLS)) assert.equal(byKey[key]?.lesson, lesson, key);
  const at = NOTATION.map((n) => order.indexOf(n.lesson));
  assert.ok(at.every((i) => i >= 0), 'every introducing lesson is in course.json');
  assert.deepEqual(at, [...at].sort((a, b) => a - b), 'the guide lists the symbols in course order');
  assert.match(byKey.ln.meaning, /log always means ln/);
});

test('notation: the match patterns find the symbol and not ordinary words or TorC code', () => {
  const english = 'The export of the model explains each step. We use a lens, a kiln and a blog, and the result is fine.';
  const code = ['zero_grad()', 'requires_grad', 'L.backward();', 'optim::SGD', 'std::exp(x)', 'x.log()', 'named_parameters()'];
  for (const n of NOTATION) {
    for (const m of n.match.text ?? []) assert.ok(!english.includes(m), `${n.key}: text pattern "${m}" hits ordinary prose`);
    for (const m of n.match.latex ?? []) assert.ok(!code.some((c) => c.includes(m)), `${n.key}: "${m}" hits TorC code`);
  }
  for (const n of NOTATION.filter((x) => [...x.symbol].length === 1)) assert.ok(n.match.text.includes(n.symbol), `${n.key}: the symbol itself is a text pattern`);
});

// ---- glossary -------------------------------------------------------------------------------------------------------------
const BRIEF_TERMS = ['model', 'prediction', 'target', 'loss', 'weight', 'bias', 'parameter', 'training', 'data-set', 'sample', 'test-set',
  'validation-set', 'gradient', 'learning-rate', 'gradient-descent', 'mean-squared-error', 'neuron', 'layer', 'activation-function', 'relu',
  'sigmoid', 'softmax', 'probability', 'cross-entropy', 'logit', 'class', 'classification', 'decision-boundary', 'backpropagation',
  'chain-rule', 'batch', 'epoch', 'momentum', 'adam', 'weight-decay', 'overfitting', 'underfitting', 'generalization', 'convolution',
  'kernel', 'feature-map', 'channel', 'stride', 'padding', 'mnist', 'pixel', 'tensor', 'shape', 'broadcasting', 'token', 'embedding',
  'attention', 'pretraining'];

test('glossary: the terms of brief §6 and §8, keys in alphabetical order, one plain sentence each', () => {
  const keys = Object.keys(glossary.terms);
  for (const k of BRIEF_TERMS) assert.ok(keys.includes(k), k);
  assert.ok(keys.length >= 40, `${keys.length} terms`);
  assert.deepEqual(keys, [...keys].sort(), 'keys in alphabetical order (eases merges)');
  const seen = new Map();
  for (const [k, g] of Object.entries(glossary.terms)) {
    assert.equal(g.short.split(/(?<=[.?!])\s+/).length, 1, `${k}: one sentence`);
    assert.ok(words(g.short) <= 25, `${k}: at most 25 words`);
    assert.ok(order.includes(g.lesson), `${k}: lesson "${g.lesson}" is in course.json`);
    for (const f of g.forms) {
      assert.ok(!seen.has(f.toLowerCase()), `${k}: the form "${f}" also belongs to ${seen.get(f.toLowerCase())}`);
      seen.set(f.toLowerCase(), k);
    }
  }
  assert.equal(glossary.terms.loss.lesson, 'what-is-learning');
  assert.equal(glossary.terms.gradient.lesson, 'why-gradients');
});

// ---- the two appendix pages -----------------------------------------------------------------------------------------------
const pageOf = async (id) => {
  const lesson = (await import(`../js/lessons/${id}.js`)).default;
  const D = buildD({}, anchorValues(readJson('data/anchors.json')));
  return { lesson, html: renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n') };
};

test('the glossary page has one anchor per term, under the lesson that introduces it, and cites its sources', async () => {
  const { lesson, html } = await pageOf('glossary');
  assert.equal(lesson.template, 3);
  assert.deepEqual(lesson.trace, []);
  assert.ok(lesson.conceptual && lesson.sources.length > 0, 'content beyond TorC cites its sources (decision 18)');
  for (const [k, g] of Object.entries(glossary.terms)) {
    assert.equal((html.match(new RegExp(`<h3 id="${k}">`, 'g')) ?? []).length, 1, k);
    assert.ok(html.includes(`<h3 id="${k}">${g.term}</h3><p>`), `${k}: the heading is the term`);
  }
  for (const m of html.matchAll(/data-term="([^"]+)"/g)) assert.ok(Object.hasOwn(glossary.terms, m[1]), m[1]);
  let current = null;
  for (const m of html.matchAll(/aria-labelledby="glossary-group-([^"]+)"|<h3 id="([^"]+)">/g)) {
    if (m[1]) current = m[1];
    else assert.equal(glossary.terms[m[2]].lesson, current, `${m[2]} stands under its lesson`);
  }
});

test('the notation page has one anchor per symbol, and sym() links land on it', async () => {
  const { lesson, html } = await pageOf('notation');
  assert.equal(lesson.template, 3);
  assert.deepEqual(lesson.trace, []);
  assert.ok(lesson.conceptual && lesson.sources.length > 0);
  for (const n of NOTATION) {
    assert.equal((html.match(new RegExp(`<h3 id="${n.key}">`, 'g')) ?? []).length, 1, n.key);
    assert.match(sym(n.key), new RegExp(`href="\\.\\./notation/#${n.key}"`));
  }
  for (const m of html.matchAll(/data-term="([^"]+)"/g)) assert.ok(Object.hasOwn(glossary.terms, m[1]), m[1]);
});

test('linkTerms: links a term at its first use only, and a shorter term inside a longer one first', () => {
  const g = { terms: { weight: { term: 'weight', forms: ['weight', 'weights'] }, 'weight-decay': { term: 'weight decay', forms: ['weight decay'] } } };
  const linked = new Set();
  assert.equal(linkTerms('Weight decay pulls each weight down.', linked, g), `${term('weight', 'Weight')} decay pulls each weight down.`);
  assert.equal(linkTerms('Then weight decay acts.', linked, g), `Then ${term('weight-decay', 'weight decay')} acts.`);
  assert.equal(linkTerms('A weight & a <b>.', new Set(['weight', 'weight-decay']), g), 'A weight &amp; a &lt;b&gt;.', 'escapes the text');
  // in its own definition a term is the defining instance (<dfn>), not a link to itself, and it stays unlinked for later mentions
  const own = new Set();
  assert.equal(linkTerms('A weight counts. Each weight changes.', own, g, 'weight'), 'A <dfn>weight</dfn> counts. Each weight changes.');
  assert.ok(!own.has('weight'));
  assert.equal(linkTerms('Weight decay pulls a weight.', own, g, 'weight-decay'), `${term('weight', 'Weight')} decay pulls a weight.`);
});

test('the glossary page links no term to itself, and reads its data without a JSON-module import', async () => {
  const { html } = await pageOf('glossary');
  for (const m of html.matchAll(/<h3 id="([^"]+)">[^<]*<\/h3><p>([\s\S]*?)<\/p>/g)) {
    assert.ok(!m[2].includes(`data-term="${m[1]}"`), `${m[1]}: its definition links to itself`);
  }
  assert.match(html, /<dfn>/, 'a definition marks its own term as the defining instance');
  const src = readFileSync(new URL('../js/lessons/glossary.js', import.meta.url), 'utf8');
  assert.doesNotMatch(src, /\bwith\s*\{\s*type\s*:/, 'import attributes need very recent browsers');
});
