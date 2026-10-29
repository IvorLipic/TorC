// The writing checks beyond lessons.test.mjs (brief R8 and D15, docs/v3-implementation.md "Writing linter"):
//   1. proseProblems: the allowed patterns (label colons, the minus sign, dashes inside maths) and the banned teacher-voice tics.
//   2. Lessons with `template: 3` must contain no banned tic. The older lessons are only counted: run with PROSE_REPORT=1 to print
//      the count per lesson.
//   3. The visible UI strings of site/js/widgets/*.js and site/js/lesson-boot.js follow the site rules (tools/prose/ui-strings.mjs
//      finds them; KNOWN_D15 there lists the strings that still fail and is temporary).
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { flatLessons, isPublished } from '../js/core/course.js';
import { buildD, stemOf, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { course, readJson, SITE, proseProblems, ticsIn, uiStringProblems, LABEL_COLONS, BANNED_TICS, hasPunctuationDash } from './helpers.mjs';
import { extractStringsFromSource, extractUiStrings, literalsOf, KNOWN_D15, UI_ALLOW } from '../../tools/prose/ui-strings.mjs';

const REPORT = Boolean(process.env.PROSE_REPORT);
const p = (s) => `<p>${s}</p>`;

// ---- 1. proseProblems ----------------------------------------------------------------------------------------------------------

test('the existing rules still hold: semicolons, dashes, colons inside a sentence, 25 words', () => {
  assert.equal(proseProblems(p('One thing; another thing.')).length, 1);
  assert.equal(proseProblems(p('A step — then another.')).length, 1);
  assert.equal(proseProblems(p('A step – then another.')).length, 1);
  assert.equal(proseProblems(p('The rule is simple: add the numbers.')).length, 1);
  assert.equal(proseProblems(p('Use the next formula:')).length, 0, 'a colon that ends the block is fine');
  assert.equal(proseProblems(p(`${'word '.repeat(26).trim()}.`)).length, 1);
  assert.equal(proseProblems(p(`${'word '.repeat(25).trim()}.`)).length, 0);
  assert.equal(proseProblems(p('Add <code>a; b</code> and <code>x - y</code>.')).length, 0, 'code is not prose');
});

test('label colons are allowed at the start of a block, and only these labels', () => {
  for (const label of LABEL_COLONS) {
    assert.deepEqual(proseProblems(`<p><strong>${label}</strong> Press Step twice.</p>`), [], label);
    assert.deepEqual(proseProblems(`<li>${label}</li>`), [], label);
  }
  assert.equal(LABEL_COLONS.length, 11);
  for (const label of ['Task:', 'Task 1:', 'Task 2:', 'Task 3:', 'What you will see:', 'Reveal:', 'Where it breaks:', 'Next question:', 'In TorC:', 'Analogy:', 'Your turn:']) {
    assert.ok(LABEL_COLONS.includes(label), label);
  }
  assert.equal(proseProblems(p('Note: press Step twice.')).length, 1, 'other labels stay banned');
  assert.equal(proseProblems(p('Task 4: press Step twice.')).length, 1);
  assert.equal(proseProblems(p('Press Step. Task: press it again.')).length, 1, 'a label in the middle of a block is a colon inside a sentence');
  assert.equal(proseProblems(p('Task: press Step: then wait.')).length, 1, 'a second colon after the label still fails');
  assert.equal(proseProblems(p('Task:press Step.')).length, 1, 'the label needs a space after the colon');
});

test('the minus sign and dashes inside maths are not punctuation dashes', () => {
  assert.deepEqual(proseProblems(p('The weight changes from 0.5 to −0.5 after one step.')), []);
  assert.deepEqual(proseProblems(p('Then 5 – 3 is 2 and x – y is the gap.')), []);
  assert.deepEqual(proseProblems(p('Use <code>w</code> – <code>lr</code> here.')), []);
  assert.deepEqual(proseProblems(p('The range is 3–4 items.')), []);
  assert.equal(hasPunctuationDash('The step – a small one – works.'), true);
  assert.equal(hasPunctuationDash('The step — a small one — works.'), true);
  assert.equal(hasPunctuationDash('5 — 3'), true, 'an em dash is never a minus');
});

// ---- 2. banned tics ------------------------------------------------------------------------------------------------------------

const TIC_SENTENCES = {
  "let's": 'Let’s train the model now.',
  "here's the thing": "Here's the thing about weights.",
  'it turns out': 'It turns out that the loss falls.',
  simply: 'You simply multiply the numbers.',
  magic: 'The update is not magic.',
  crucially: 'Crucially, the step is small.',
  'notice that': 'Notice that the loss falls.',
  'the key insight': 'The key insight is the slope.',
  'under the hood': 'Under the hood, TorC adds the numbers.',
  'in other words': 'In other words, the loss falls.',
  'think of it as': 'Think of it as a hill.',
  imagine: 'Imagine a hill in the fog.',
  essentially: 'The loss is essentially a distance.',
  'at its core': 'At its core, training changes numbers.',
  'the heart of': 'This is the heart of training.',
  powerful: 'The chain rule is powerful.',
  elegant: 'The rule is elegant.',
  beautiful: 'The curve is beautiful.',
  'not just X but Y': 'The loss is not just a number but a guide.',
  "that's it": "That's it for the forward pass.",
  'this is where X comes in': 'This is where the learning rate comes in.',
  'this lesson shows (opener)': 'This lesson shows how the loss falls.',
};

test('every banned tic of brief section 8 is found, and only with { tics: true }', () => {
  assert.deepEqual(Object.keys(TIC_SENTENCES).sort(), BANNED_TICS.map(([name]) => name).sort(), 'one fixture per banned tic');
  for (const [name, sentence] of Object.entries(TIC_SENTENCES)) {
    assert.deepEqual(ticsIn(sentence), [name], sentence);
    assert.ok(proseProblems(p(sentence), { tics: true }).some((x) => x.includes(`banned tic "${name}"`)), sentence);
    assert.ok(!proseProblems(p(sentence)).some((x) => x.includes('banned tic')), 'off by default');
  }
});

test('words that only look like a banned tic, and the flag-only items, pass', () => {
  const fine = [
    'Let us train the model.',                                  // not the contraction
    'The simple rule adds the numbers.',
    'A magnet attracts iron.',
    'The image shows a digit.', 'This lesson teaches the loss.', 'In this lesson the loss falls.',
    'You can just multiply the numbers.',                       // a hedging "just" is flagged by the reviewer, not the linter
    'The loss is not a distance but a score.',                  // "not just X but Y" needs "just"
    'We practise this on a simple problem.',
  ];
  for (const s of fine) assert.deepEqual(proseProblems(p(s), { tics: true }), [], s);
  assert.deepEqual(ticsIn('Simply put, imagine it. Imagine that.'), ['simply', 'imagine', 'imagine'], 'every match counts');
});

// A lesson with `template: 3` is held to the tic list; the shape matches the lesson modules (sections of kind `text`).
const fixtureLesson = (template, ...paragraphs) => ({ id: 'fixture', title: 'Fixture', trace: [], template, sections: [{ kind: 'text', html: () => `<section>${paragraphs.map(p).join('')}</section>` }] });

function lessonProblems(lesson, D = buildD({})) {
  const saved = globalThis.katex;
  globalThis.katex = undefined;                            // formulas render as <code>, as in lessons.test.mjs
  try {
    return renderLesson(lesson, D).flatMap(({ html }) => proseProblems(html, { tics: lesson.template === 3 }));
  } finally { globalThis.katex = saved; }
}

test('a template 3 lesson fails on a banned tic, an older lesson does not', () => {
  const text = 'Let’s look at the loss. It turns out that the loss falls.';
  assert.deepEqual(lessonProblems(fixtureLesson(3, text)).map((x) => x.match(/banned tic "([^"]+)"/)[1]), ["let's", 'it turns out']);
  assert.deepEqual(lessonProblems(fixtureLesson(undefined, text)), []);
  assert.deepEqual(lessonProblems(fixtureLesson(3, 'Task: press Step. The minus sign is −.')), []);
  assert.equal(lessonProblems(fixtureLesson(3, 'One; two.')).length, 1, 'the other rules apply too');
});

const lessonEntries = flatLessons(course()).filter((l) => isPublished(l) || existsSync(new URL(`js/lessons/${l.id}.js`, SITE)));
const tally = (names) => { const c = {}; for (const n of names) c[n] = (c[n] ?? 0) + 1; return Object.entries(c).map(([n, k]) => `${n} x${k}`).join(', '); };
const totals = {};

for (const entry of lessonEntries) {
  test(`lesson ${entry.id}: no banned tic when it has template: 3 (otherwise counted)`, async () => {
    const lesson = (await import(new URL(`../js/lessons/${entry.id}.js`, import.meta.url))).default;
    const D = buildD(Object.fromEntries(lesson.trace.map((f) => [stemOf(f), readJson('data/' + f)])), anchorValues(readJson('data/anchors.json')));
    const saved = globalThis.katex;
    globalThis.katex = undefined;
    let html;
    try { html = renderLesson(lesson, D).map((x) => x.html).join('\n'); } finally { globalThis.katex = saved; }
    const found = proseProblems(html, { tics: true }).filter((x) => x.startsWith('banned tic')).map((x) => x.match(/banned tic "([^"]+)"/)[1]);
    totals[entry.id] = found.length;
    if (REPORT) console.log(`prose-report: ${entry.id}${lesson.template === 3 ? ' (template 3)' : ''}: ${found.length} banned tics${found.length ? ` (${tally(found)})` : ''}`);
    if (lesson.template === 3) assert.deepEqual(found, [], `banned tics in ${entry.id}`);
  });
}

// ---- 3. visible UI strings of widgets and lesson-boot.js -----------------------------------------------------------------------

const ui = (src, file = 'x.js') => extractStringsFromSource(src, file).map((r) => r.text);

test('the extractor takes prose strings and leaves CSS, selectors, paths, LaTeX, code and class lists', () => {
  assert.deepEqual(ui("read.textContent = 'press the button twice';"), ['press the button twice'], 'a text sink reads as prose without a capital');
  assert.deepEqual(ui("const MSG = 'Press the button twice.';"), ['Press the button twice.'], 'a capital start reads as prose');
  assert.deepEqual(ui("el('p', { class: 'hint' }, 'drag the point to move it');"), ['drag the point to move it'], 'text argument of el()');
  assert.deepEqual(ui("x.setAttribute('aria-label', 'the plot of the loss');"), ['the plot of the loss']);
  assert.deepEqual(ui("svg.append(el('g', { title: `Loss after ${n} steps` }));"), ['Loss after 9 steps'], 'bound values become 9');
  assert.deepEqual(ui("h.innerHTML = `<b>Rate</b> <code>lr</code> sets the step</code>`;"), ['Rate code sets the step']);
  assert.deepEqual(ui('h.innerHTML = \'<button aria-label="Close the window now" class="btn">x</button>\';'), ['Close the window now']);
  for (const skipped of [
    "const CSS = `.a{color:red}\\n.b .c{margin:0 auto 2px}`;",
    "const s = 'grid-template-columns:repeat(3, auto)';",
    "const s = 'color:var(--muted) and more words';",
    "document.querySelector('.node[data-node=\"a\"] span b');",
    "const d = 'M 10 20 L 30 40 L 50 60';",
    "const t = 'w \\\\leftarrow w - \\\\mathit{lr} \\\\cdot g';",
    "const c = 'for (int i = 0; i < n; ++i) {';",
    "const c = 'const torc::Tensor& grad = param->grad();';",
    "const k = 'btn small primary';",
    "const f = 'value is %d of %d';",
    "import { el } from '../core/dom.js';",
    "const a = 'two words';",
    "// A comment with many words is not a string\nconst z = 1; /* nor is this block comment */",
  ]) assert.deepEqual(ui(skipped), [], skipped);
});

test('the tokenizer reads templates, nesting, escapes, regex literals and comments', () => {
  const src = "const re = /it's \"a\" regex/g; const a = `Outer ${cond ? 'inner words here' : `deep ${x} words here`} done`;\n// 'comment words here'\nconst b = 'Line\\nbreak and \\u2212 minus here'; /* 'block words here' */ const c = \"It's three words\";";
  const got = literalsOf(src).map((l) => [l.line, l.text.replaceAll('', '9')]);
  assert.deepEqual(got, [[1, 'inner words here'], [1, 'deep 9 words here'], [1, 'Outer 9 done'], [3, 'Line\nbreak and − minus here'], [3, "It's three words"]]);
});

test('uiStringProblems applies the site rules to short interface text', () => {
  const bad = (s) => uiStringProblems(s).map((x) => x.split(':')[0]);
  assert.deepEqual(bad('Enter a number; the last valid value is still in use.'), ['semicolon']);
  assert.deepEqual(bad('✓ Done — click to undo'), ['dash']);
  assert.deepEqual(bad('The two columns of y are identical: the neurons are twins.'), ['colon inside a sentence']);
  assert.deepEqual(bad('Values are recomputed in your browser.'), ['provenance wording']);
  assert.deepEqual(bad('Show as table'), ['provenance wording']);
  assert.deepEqual(bad('Let’s press Step now.'), ['banned tic "let\'s"']);
  assert.deepEqual(bad(`${'word '.repeat(26).trim()}.`), ['26 words']);
  for (const fine of ['dashed line: training loss', 'amber: a positive weight, so ink here raises the neuron', 'top of each pair: the 3 × 3 kernel', 'Step 9: w = 9, L = 9.',
    'y, sample 9, neuron 9: 9', 'After step 9: training loss 9, test loss 9.', 'Use this formula:', 'Task: press Step.', 'Enter a number between −1,000,000 and 1,000,000.']) {
    assert.deepEqual(uiStringProblems(fine), [], fine);
  }
});

test('visible UI strings in widgets and lesson-boot.js follow the site rules (D15)', () => {
  const site = fileURLToPath(SITE);
  const rows = extractUiStrings(site);
  assert.ok(rows.length > 100, `${rows.length} strings found`);
  assert.ok(rows.some((r) => r.file === 'lesson-boot.js') && rows.some((r) => r.file === 'matmul-grid.js'));
  assert.ok(!rows.some((r) => r.file === 'sgd-code.js'), 'the C++ listing is not prose');
  const failures = [];
  let skippedKnown = 0;
  for (const r of rows) {
    const problems = uiStringProblems(r.text);
    if (r.known) { skippedKnown++; continue; }
    if (r.allowed) continue;
    for (const pr of problems) failures.push(`${r.file}:${r.line}: ${pr}`);
  }
  assert.deepEqual(failures, []);
  // KNOWN_D15 is temporary: say which entries no longer match a failing string, so the lead can delete them
  const stale = KNOWN_D15.filter((e) => {
    const hit = rows.filter((r) => r.file === e.file && r.text.includes(e.includes));
    return hit.every((r) => uiStringProblems(r.text).length === 0);
  });
  console.log(`prose-ui: ${rows.length} UI strings; ${skippedKnown} in KNOWN_D15${stale.length ? `; stale entries to delete: ${stale.map((e) => `${e.file} "${e.includes}"`).join(' | ')}` : ''}`);
  for (const e of UI_ALLOW) assert.ok(e.why && rows.some((r) => r.file === e.file && r.text.includes(e.includes)), `UI_ALLOW entry matches nothing: ${e.includes}`);
});
