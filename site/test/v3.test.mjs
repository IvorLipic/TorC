// The v3 machinery, proven on a fixture lesson before any real lesson migrates (docs/v3-implementation.md, R2, R7, R9, R10):
// the fixture passes every v3 rule, and a broken copy of it fails exactly the rule that it breaks. Also the blocks, the content
// files (glossary, notation, refreshers) and the next-lesson line of the answer section (D2).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildD, makeT, stemOf, anchorValues } from '../js/core/trace.js';
import { renderLesson, validateLesson, boundNums, defaultFooter } from '../js/core/sections.js';
import { neighbors, flatLessons } from '../js/core/course.js';
import { worked, inTorc, realLine, analogy, chatgpt, refresher, term, sym, loopDiagram, LOOP_PARTS } from '../js/core/blocks.js';
import { NOTATION } from '../js/content/notation.js';
import { REFRESHERS } from '../js/content/refreshers.js';
import { checkV3, answerWithNext, tokens } from './v3.mjs';
import { course as readCourse, readJson, proseProblems } from './helpers.mjs';
import fixture from './fixtures/v3-fixture.js';

const FIXTURE_SRC = readFileSync(new URL('./fixtures/v3-fixture.js', import.meta.url), 'utf8');
const glossary = readJson('glossary.json');
const anchors = anchorValues(readJson('data/anchors.json'));
const dataFor = (lesson) => buildD(Object.fromEntries(lesson.trace.map((f) => [stemOf(f), readJson('data/' + f)])), anchors);
const D = dataFor(fixture);
const T = makeT(D);

// a copy of the real course with the fixture placed right after sgd-step (the third lesson of the main path), or in a given part
function testCourse({ appendix = false } = {}) {
  const c = readCourse();
  const entry = { id: fixture.id, title: fixture.title, outcome: 'Tests the machinery.', prereqs: [], symbols: [], trace: fixture.trace, est_minutes: 1, status: 'draft' };
  if (appendix) c.parts.push({ id: 'using-torc', title: 'Using TorC', kind: 'appendix', lessons: [entry] });
  else {
    const part = c.parts.find((p) => p.lessons.some((l) => l.id === 'sgd-step'));
    part.lessons.splice(part.lessons.findIndex((l) => l.id === 'sgd-step') + 1, 0, entry);
  }
  return c;
}
const check = (lesson, over = {}) => checkV3(lesson, { D: dataFor(lesson), course: testCourse(), glossary, notation: NOTATION, source: FIXTURE_SRC, ...over });
const codes = (problems) => [...new Set(problems.map((p) => p.split(':')[0]))].sort();
// a copy of the fixture with sections replaced (by index) or spliced in
const variant = (fn) => { const l = { ...fixture, sections: fixture.sections.map((s) => ({ ...s })) }; fn(l, l.sections); return l; };
const text = (html, extra = {}) => ({ kind: 'text', html, ...extra });
const ANSWER = fixture.sections.length - 1;

test('the fixture is a valid lesson module and passes every v3 rule', () => {
  assert.deepEqual(validateLesson(fixture), []);
  assert.deepEqual(check(fixture), []);
});

test('the fixture uses every section kind and every number role, and every bound number re-formats from its attributes', () => {
  assert.deepEqual([...new Set(fixture.sections.map((s) => s.kind))].sort(), ['answer', 'explain', 'questions', 'quiz', 'text', 'widget']);
  assert.deepEqual([...new Set(fixture.sections.filter((s) => s.kind === 'quiz').map((s) => s.purpose))].sort(), ['check', 'cumulative', 'predict', 'transfer']);
  assert.deepEqual([...new Set(fixture.sections.filter((s) => s.kind === 'quiz').map((s) => s.type))].sort(), ['choice', 'numeric']);
  const html = renderLesson(fixture, D, { course: testCourse() }).map((p) => p.html).join('\n');
  const b = boundNums(html);
  assert.deepEqual([...new Set(b.map((x) => x.role))].sort(), ['anchor', 'contrast', 'human', 'reveal', 'setup', 'worked']);
  assert.deepEqual([...new Set(b.filter((x) => x.human).map((x) => x.human))].sort(), ['pct', 'per100', 'prob', 'rmse']);
  for (const x of b) assert.equal(T.reformat(x), x.shown, x.path);
  // what the learner sees, by role
  const shown = (path, role) => b.find((x) => x.path === path && x.role === role)?.shown;
  assert.equal(shown('files.why-gradients.probes.0.grad', 'worked'), '−24', 'exact, with the minus sign');
  assert.equal(shown('files.why-gradients.nudge.stages.2.after', 'reveal'), '35.76', 'exact roles keep fmt (4 significant figures)');
  assert.equal(shown('anchors.mlp_parameters', 'anchor'), '26,506', 'a comma every three digits');
  assert.equal(b.find((x) => x.human === 'rmse').shown, '6');
  assert.equal(b.find((x) => x.human === 'pct').shown, '97%');
  assert.equal(b.find((x) => x.human === 'per100').shown, '3');
  assert.equal(b.find((x) => x.human === 'prob').shown, '93%');
  assert.match(html, /data-of="files\.mnist-mlp\.setup\.test_samples"/);
  assert.match(html, /<figure class="loop" data-loop="adjust">/, 'the questions box carries the loop diagram with the lesson part lit');
  assert.match(html, /data-from="why-gradients"[^>]*>.*This question comes from the lesson <a href="\.\.\/why-gradients\/">The gradient<\/a>/s);
});

test('the fixture follows the writing rules that the site tests (proseProblems)', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { assert.deepEqual(renderLesson(fixture, D, { course: testCourse(), next: null }).flatMap(({ html }) => proseProblems(html)), []); } finally { globalThis.katex = saved; }
});

test('quiz markup: fieldset and legend, real radio inputs or a labelled number input, a Check button, a polite live region, a hidden reveal', () => {
  const parts = renderLesson(fixture, D);
  const choice = parts.find((p) => p.section.id === 'direction').html, numeric = parts.find((p) => p.section.id === 'nudge').html;
  for (const h of [choice, numeric]) {
    assert.match(h, /<fieldset aria-describedby="quiz-[a-z-]+-prompt"><legend>/);
    assert.match(h, /<button type="submit" class="btn primary">Check<\/button>/);
    assert.match(h, /class="feedback"[^>]*role="status" aria-live="polite"/);
    assert.match(h, /<div class="reveal"[^>]* hidden>/, 'the reveal opens only after an attempt');
  }
  assert.equal((choice.match(/<input type="radio"/g) ?? []).length, 3);
  assert.match(choice, /data-correct="0"/);
  assert.equal((choice.match(/<template data-fb="\d">/g) ?? []).length, 3, 'one feedback per option, shown for the option chosen');
  assert.match(numeric, /<label for="quiz-nudge-n">Your answer<\/label><input type="number"[^>]*id="quiz-nudge-n"/);
  assert.match(numeric, /data-answer="35\.76\d*" data-tolerance="0\.01"/);
  for (const { html: h } of parts) for (const t of tokens(h)) if (t.type === 'open' && t.el.tag === 'input') {
    const id = t.el.attrs.id;
    assert.ok(id && h.includes(`<label for="${id}"`), `input ${id} has a label`);
  }
  const ex = parts.find((p) => p.section.kind === 'explain').html;
  assert.match(ex, /<label for="explain-why-against-text">Your sentence<\/label><textarea id="explain-why-against-text"/);
  assert.match(ex, /data-explain-compare>Compare with a sample answer<\/button>/);
  assert.match(ex, /<div class="model"[^>]* hidden>/);
  const w = parts.find((p) => p.section.kind === 'widget').html;
  assert.match(w, /data-widget="demo"><\/section><dl class="tasks"[^>]*><div class="task-do"><dt>Task<\/dt><dd>/, 'the host, then the guided tasks');
});

test('the answer section ends with the next lesson from neighbors() (D2), or the course map when there is none', () => {
  const c = testCourse();
  const next = neighbors(c, fixture.id).next;
  assert.ok(next, 'the fixture has a next lesson in the test course');
  assert.ok(answerWithNext(fixture, D, c, next).includes(`<p class="next-lesson">The next lesson is <a href="../${next.id}/">${next.title}</a>.</p>`));
  assert.match(answerWithNext(fixture, D, c, null), /The <a href="\.\.\/\.\.\/">course map<\/a> lists all lessons/);
  assert.doesNotMatch(answerWithNext(fixture, D, c, undefined), /next-lesson/, 'no line when the next lesson is not known');
  assert.match(answerWithNext(fixture, D, c, next), /<h2 id="answer-h">The answer<\/h2>.*<h3>Next question<\/h3>/s);
});

// ---- every rule fires on a copy that breaks it --------------------------------------------------------------------------
const P = 'files.why-gradients.probes';
const BROKEN = {
  role: (l, s) => { s[4] = text((D2, T2) => `<p>The gradient is ${T2(`${P}.0.grad`)}.</p>`); },
  worked: (l, s) => { s[4] = text((D2, T2) => `<p>The ${term('gradient')} is ${T2(`${P}.0.grad`, 'worked')}.</p>`); },
  'worked blocks': (l, s) => { s.splice(4, 0, text(() => worked('<p>a</p>') + worked('<p>b</p>') + worked('<p>c</p>'))); },
  reveal: (l, s) => { s[4] = text((D2, T2) => `<p>The ${term('gradient')} is ${T2(`${P}.0.grad`, 'reveal')}.</p>`); },
  'numbers per sentence': (l, s) => { s.splice(5, 0, text((D2, T2) => `<p>It is ${T2(`${P}.0.L`, 'setup')}, ${T2(`${P}.1.L`, 'setup')} and ${T2(`${P}.2.L`, 'setup')}.</p>`)); },
  'numbers outside': (l, s) => { s.splice(5, 0, text((D2, T2) => [0, 1, 2, 3, 4, 5].map((i) => `<p>It is ${T2(`${P}.${i}.L`, 'contrast')}.</p>`).join(''))); },
  binding: (l, s) => { s.splice(5, 0, text(() => '<p>It is <span class="num" data-trace="files.why-gradients.probes.0.L" data-role="setup" data-sig="exact">37</span>.</p>')); },
  questions: (l, s) => { s.splice(1, 1); },
  'questions count': (l, s) => { s[1] = { kind: 'questions', items: ['A?', 'B?', 'C?', 'D?'] }; },
  answer: (l, s) => { s.push(s.splice(ANSWER, 1)[0], text(() => '<p>More.</p>')); },
  'answer words': (l, s) => { s[ANSWER] = { ...s[ANSWER], html: () => `<p>${'word '.repeat(81)}</p>` }; },
  'text words': (l, s) => { s.splice(5, 0, text(() => `<p>${'word '.repeat(151)}</p>`)); },
  numbered: (l, s) => { s.splice(5, 0, ...[4, 5, 6, 7].map((n) => text(() => `<section><h2>${n} · Part</h2></section>`))); },
  quizzes: (l, s) => { s.splice(5, 0, ...[1, 2, 3].map((n) => ({ ...s[7], id: `extra-${n}` }))); },
  predict: (l, s) => { s[2] = { ...s[2], setup: false }; },
  transfer: (l, s) => { s.splice(s.findIndex((x) => x.purpose === 'transfer'), 1); },
  explain: (l, s) => { s.splice(s.findIndex((x) => x.kind === 'explain'), 1); },
  cumulative: (l, s) => { s.splice(s.findIndex((x) => x.purpose === 'cumulative'), 1); },
  'cumulative from': (l, s) => { const i = s.findIndex((x) => x.purpose === 'cumulative'); s[i] = { ...s[i], from: 'mnist-cnn' }; },
  tasks: (l, s) => { const i = s.findIndex((x) => x.kind === 'widget'); s[i] = { ...s[i], tasks: undefined }; },
  palette: (l, s) => { s.splice(5, 0, text(() => inTorc('How', '<p>x</p>') + chatgpt('Same.'))); },
  glossary: (l, s) => { s[1] = { kind: 'questions', items: ['Why does the loss fall?'] }; },
  'glossary key': (l, s) => { s.splice(5, 0, text(() => `<p>A ${term('weight-decay', 'leak')} here.</p>`)); },
  notation: (l, s) => { s.splice(5, 0, text(() => '<p>The sign Σ adds.</p>')); },
  'notation sym': (l, s) => { s.splice(5, 0, text(() => `<p>The momentum ${sym('mu')} is high.</p>`)); },
  quiz: (l, s) => { const i = s.findIndex((x) => x.id === 'direction'); s[i] = { ...s[i], options: s[i].options.slice(0, 2) }; },
  'quiz feedback': (l, s) => { const i = s.findIndex((x) => x.id === 'direction'); s[i] = { ...s[i], options: s[i].options.map((o, j) => (j === 1 ? { ...o, feedback: undefined } : o)) }; },
  'quiz path': (l, s) => { const i = s.findIndex((x) => x.id === 'nudge'); s[i] = { ...s[i], answer: 'files.why-gradients.nudge.nope' }; },
  'quiz digits': (l, s) => { const i = s.findIndex((x) => x.id === 'direction'); s[i] = { ...s[i], prompt: () => '<p>At <code>w</code> = 6, which way?</p>' }; },
  kinds: (l, s) => { s.splice(5, 0, { kind: 'tldr', items: ['x'] }); },
};
const EXPECT = { 'worked blocks': 'worked', 'numbers per sentence': 'numbers', 'numbers outside': 'numbers', 'questions count': 'questions', 'answer words': 'answer',
  'text words': 'words', numbered: 'budget', quizzes: 'budget', 'cumulative from': 'cumulative', 'glossary key': 'glossary', 'notation sym': 'notation',
  'quiz feedback': 'quiz', 'quiz path': 'quiz', 'quiz digits': 'quiz' };

for (const [name, fn] of Object.entries(BROKEN)) {
  test(`v3 rule "${name}" fires on a copy of the fixture that breaks it`, () => {
    const p = check(variant(fn));
    assert.ok(codes(p).includes(EXPECT[name] ?? name), `expected a "${EXPECT[name] ?? name}" problem, got ${JSON.stringify(p)}`);
  });
}

test('v3 rule "escape": a single backslash before ; , ! : or a space in the module source (D1)', () => {
  assert.deepEqual(check(fixture, { source: "const a = 'x \\\\; y';" }), [], 'a doubled backslash is right');
  assert.deepEqual(codes(check(fixture, { source: "const a = 'x \\; y';\nconst b = '\\,';" })), ['escape']);
  assert.equal(check(fixture, { source: "const a = 'x \\; y';\nconst b = '\\,';" }).length, 2);
});

test('appendix pages and conceptual lessons have their exceptions (brief §9)', () => {
  const bare = variant((l, s) => { l.sections = [s[0], text(() => '<section><p>The sign Σ adds, and the momentum μ keeps speed.</p></section>')]; });
  assert.deepEqual(checkV3(bare, { D, course: testCourse({ appendix: true }), glossary, notation: NOTATION, source: '' }), [], 'appendix: no frame, no quizzes, any symbol');
  const noPredict = variant((l, s) => { l.conceptual = true; l.sources = [{ title: 'A source', url: 'https://example.org/' }]; s.splice(s.findIndex((x) => x.purpose === 'transfer'), 1); s.splice(3, 1); });
  assert.deepEqual(check(noPredict), [], 'conceptual: no predict before text, no transfer');
  assert.deepEqual(codes(check(variant((l, s) => { l.conceptual = true; l.sources = [{ title: 'S', url: 'https://example.org/' }]; s.splice(s.findIndex((x) => x.kind === 'explain'), 1); }))), ['explain'], 'conceptual still needs an explain prompt');
});

test('a conceptual lesson lists its sources in the footer; one without trace files has only that list', () => {
  const c = { ...fixture, conceptual: true, sources: [{ title: 'Attention is all you need', url: 'https://arxiv.org/abs/1706.03762' }] };
  assert.deepEqual(validateLesson(c), []);
  const f = defaultFooter(c, D);
  assert.match(f, /^<p>Numbers: TorC/);
  assert.match(f, /<ul class="sources"><li><a href="https:\/\/arxiv\.org\/abs\/1706\.03762">Attention is all you need<\/a><\/li><\/ul>/);
  assert.doesNotMatch(defaultFooter({ ...c, trace: [] }, buildD({})), /Numbers/);
  assert.match(validateLesson({ ...c, sources: [] }).join(), /conceptual lesson needs sources/);
  assert.match(validateLesson({ ...c, sources: [{ title: 'x', url: 'http://insecure' }] }).join(), /conceptual lesson needs sources/);
});

test('validateLesson rejects malformed v3 sections', () => {
  const v = (s, over = {}) => validateLesson({ id: 'a', title: 'a', trace: [], sections: [s], ...over }).join(' | ');
  assert.match(v({ kind: 'quiz', id: 'q', purpose: 'guess', type: 'choice', prompt: 'p', reveal: 'r', options: [] }), /purpose must be one of/);
  assert.match(v({ kind: 'quiz', id: 'q', purpose: 'check', type: 'choice', prompt: 'p', reveal: 'r', options: [{ html: 'a', correct: true }, { html: 'b', correct: true, feedback: 'x' }, { html: 'c', feedback: 'y' }] }), /exactly one option is correct/);
  assert.match(v({ kind: 'quiz', id: 'q', purpose: 'check', type: 'numeric', prompt: 'p', reveal: 'r', answer: 'x', tolerance: 0 }), /tolerance above 0/);
  assert.match(v({ kind: 'quiz', id: 'q', purpose: 'cumulative', type: 'numeric', prompt: 'p', reveal: 'r', answer: 'x', tolerance: 1 }), /needs from/);
  assert.match(v({ kind: 'quiz', id: 'q', purpose: 'check', type: 'slider', prompt: 'p', reveal: 'r' }), /type must be one of/);
  assert.match(v({ kind: 'explain', id: 'e', prompt: 'p' }), /prompt and model/);
  assert.match(v({ kind: 'answer', html: 'h' }), /html and next/);
  assert.match(v({ kind: 'questions' }), /questions sections need items/);
  assert.match(v({ kind: 'widget', id: 'w', mount() {}, tasks: 'x' }), /widget tasks/);
  assert.match(v({ kind: 'text', html: () => '' }, { template: 2 }), /template must be 3/);
  assert.match(v({ kind: 'text', html: () => '' }, { loop: 'train' }), /lesson.loop must be one of/);
});

// ---- blocks -------------------------------------------------------------------------------------------------------------
test('blocks: markup, escaping and errors', () => {
  assert.equal(worked('<p>x</p>'), '<div class="worked" role="group" aria-label="Worked example"><p>x</p></div>');
  assert.match(inTorc('How <code>step()</code> works', '<p>b</p>'), /^<details class="in-torc"><summary><strong class="label">In TorC\.<\/strong> How <code>step\(\)<\/code> works<\/summary><div class="body"><p>b<\/p><\/div><\/details>$/);
  assert.match(realLine('if (a < b) x();', 'Cap'), /<pre tabindex="0" role="group" aria-label="One line of TorC code"><code>if \(a &lt; b\) x\(\);<\/code><\/pre><figcaption>Cap<\/figcaption>/);
  assert.match(analogy('A', 'B'), /Analogy\.<\/strong> A<\/p><p><strong class="label">Where it breaks\.<\/strong> B/);
  assert.throws(() => analogy('A'), /where it breaks/);
  assert.match(chatgpt('Same idea.'), /<aside class="chatgpt" aria-label="The same idea in ChatGPT"><p><strong class="label">In ChatGPT\.<\/strong> Same idea\./);
  assert.match(refresher('slope'), /<details class="refresher" data-refresher="slope"><summary><strong class="label">Refresher\.<\/strong> The slope of a curve<\/summary>/);
  assert.throws(() => refresher('nope'), /no such refresher/);
  assert.equal(term('learning-rate'), '<a class="term" href="../glossary/#learning-rate" data-term="learning-rate">learning rate</a>');
  assert.equal(term('loss', 'losses'), '<a class="term" href="../glossary/#loss" data-term="loss">losses</a>');
  assert.throws(() => term('Bad Key'), /lowercase-hyphenated/);
  assert.match(sym('mu'), /^<a class="sym" href="\.\.\/notation\/#mu" data-sym="mu" title="mu, read as “mu”">μ<\/a>$/);
  assert.throws(() => sym('nope'), /no such symbol/);
  for (const a of [...LOOP_PARTS, 'all']) {
    const svg = loopDiagram(a);
    assert.match(svg, /<svg viewBox="0 0 280 150"[^>]*role="img" aria-label="The training loop\./);
    assert.equal((svg.match(/<g class="loop-node on">/g) ?? []).length, a === 'all' ? 4 : 1, a);
    assert.equal((svg.match(/<polygon class="loop-head"/g) ?? []).length, 4, 'four arrows make the cycle');
    const label = /aria-label="([^"]*)"/.exec(svg)[1];
    assert.deepEqual(proseProblems(`<p>${label}</p>`), [], 'the label follows the writing rules');
  }
  assert.throws(() => loopDiagram('train'), /use one of/);
});

// ---- content files ------------------------------------------------------------------------------------------------------
// lessons named by the content files must exist in course.json or in the v3 curriculum (docs/v3-implementation.md)
const V3_IDS = ['what-is-learning', 'why-gradients', 'sgd-step', 'linear-layer', 'nonlinearity', 'classification', 'losses', 'backprop', 'deep-nets',
  'training-loop', 'optimizers', 'generalization', 'mnist-mlp', 'one-step', 'convolution', 'mnist-cnn', 'from-digits-to-chatgpt',
  'tensors', 'modules', 'torc-pitfalls', 'notation', 'glossary', 'backprop-lab'];
const known = (id) => V3_IDS.includes(id) || flatLessons(readCourse()).some((l) => l.id === id);
const KEY = /^[a-z0-9]+(-[a-z0-9]+)*$/;

test('site/glossary.json: schema, keys, forms, a one-sentence definition that follows the writing rules, a known lesson', () => {
  assert.equal(glossary.schema, 'torc-glossary/1');
  assert.ok(Object.keys(glossary.terms).length >= 1);
  for (const [k, g] of Object.entries(glossary.terms)) {
    assert.match(k, KEY);
    assert.equal(typeof g.term, 'string');
    assert.ok(Array.isArray(g.forms) && g.forms.length && g.forms.includes(g.term), `${k}: forms include the term itself`);
    assert.match(g.short, /^[A-Z].*\.$/, `${k}: the definition is a sentence`);
    assert.deepEqual(proseProblems(`<p>${g.short}</p>`), [], k);
    assert.ok(known(g.lesson), `${k}: lesson "${g.lesson}"`);
  }
});

test('content/notation.js and content/refreshers.js: well-formed entries', () => {
  const keys = new Set();
  for (const n of NOTATION) {
    assert.match(n.key, KEY);
    assert.ok(!keys.has(n.key), `duplicate ${n.key}`); keys.add(n.key);
    for (const f of ['symbol', 'name', 'read', 'meaning']) assert.ok(typeof n[f] === 'string' && n[f], `${n.key}.${f}`);
    assert.ok(known(n.lesson), `${n.key}: lesson "${n.lesson}"`);
    assert.ok((n.match.latex ?? []).length + (n.match.text ?? []).length > 0, `${n.key}: something to search for`);
    assert.deepEqual(proseProblems(`<p>${n.meaning}</p>`), [], n.key);
  }
  for (const [k, r] of Object.entries(REFRESHERS)) {
    assert.match(k, KEY);
    assert.ok(typeof r.title === 'string' && typeof r.html === 'string' && r.html.includes('<p>'), k);
    assert.deepEqual(proseProblems(r.html), [], k);
  }
});
