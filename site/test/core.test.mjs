// core/: hash state, progress store v2 (migration from v1, blocked storage), quiz answers, prose numbers, course helpers,
// trace loader and T() (legacy and role-tagged), sections.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseHash, formatHash } from '../js/core/state.js';
import { createStore, sanitize, migrateV1, STORAGE_KEY, LEGACY_KEY } from '../js/core/store.js';
import { parseAnswer, judgeNumeric, judgeChoice } from '../js/core/quiz.js';
import { groupThousands, proseNumber, sigFor, humanText } from '../js/core/numbers.js';
import { validateCourse, curriculum, labs, neighbors, legacyTarget, flatLessons, lessonHref } from '../js/core/course.js';
import { buildD, makeT, resolvePath, loadTrace, loadAnchors } from '../js/core/trace.js';
import { sectionHtml, findUnboundNums, boundNums, validateLesson, renderLesson } from '../js/core/sections.js';
import { course as readCourse, traceFiles } from './helpers.mjs';

// ---- state (hash) ---------------------------------------------------------------------------------------
test('state: path-style and key=value hashes round-trip', () => {
  assert.deepEqual(parseHash('#mlp/9').parts, ['mlp', '9']);
  assert.deepEqual(parseHash('').parts, []);
  assert.deepEqual(parseHash('#step=3&sel=h').params, { step: '3', sel: 'h' });
  assert.equal(formatHash({ parts: ['reuse', '3'] }), '#reuse/3');
  assert.equal(formatHash({ params: { step: 3, sel: 'h' } }), '#step=3&sel=h');
  assert.equal(formatHash({ parts: [] }), '');
});

// ---- store ------------------------------------------------------------------------------------------------
const memoryStorage = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => { m.set(k, String(v)); }, m }; };

test('store: progress is kept under torc:v2 as {lesson:{visited,done,quiz,explained,seen}}', () => {
  const ms = memoryStorage();
  const s = createStore(() => ms);
  assert.equal(STORAGE_KEY, 'torc:v2');
  assert.equal(s.isDone('a'), false);
  s.markVisited('a'); s.setDone('a', true);
  assert.deepEqual(s.recordQuiz('a', 'q1', false), { tries: 1, correct: false });
  assert.deepEqual(s.recordQuiz('a', 'q1', true), { tries: 2, correct: true }, 'tries count every attempt; correct is the latest result');
  s.setExplained('a', 'e1');
  s.setSeen('a', 'c1', true); s.setSeen('a', 'c2', true); s.setSeen('a', 'c2', false);
  assert.deepEqual(JSON.parse(ms.m.get(STORAGE_KEY)), { a: { visited: true, done: true, quiz: { q1: { tries: 2, correct: true } }, explained: { e1: true }, seen: { c1: true } } });
  assert.equal(createStore(() => ms).isDone('a'), true, 'a new store instance (page reload) sees it');
  assert.deepEqual(createStore(() => ms).quiz('a', 'q1'), { tries: 2, correct: true });
  assert.deepEqual(s.quiz('a', 'never'), { tries: 0, correct: false });
  s.setDone('a', false);
  assert.equal(createStore(() => ms).isDone('a'), false);
  assert.equal(s.persistent, true);
});

test('store: v1 data migrates on first load; a v1 "I got it" becomes seen, never correct', () => {
  const ms = memoryStorage();
  ms.m.set(LEGACY_KEY, JSON.stringify({ a: { visited: true, checks: { sign: true, off: false }, done: true }, b: { visited: true, checks: {}, done: false }, bad: 3 }));
  const s = createStore(() => ms);
  assert.deepEqual(s.get('a'), { visited: true, done: true, quiz: {}, explained: {}, seen: { sign: true } });
  assert.deepEqual(s.quiz('a', 'sign'), { tries: 0, correct: false }, 'no correctness is invented');
  assert.deepEqual(JSON.parse(ms.m.get(STORAGE_KEY)), { a: { visited: true, done: true, quiz: {}, explained: {}, seen: { sign: true } }, b: { visited: true, done: false, quiz: {}, explained: {}, seen: {} } });
  assert.ok(ms.m.has(LEGACY_KEY), 'the v1 key is left as it is');
  ms.m.set(LEGACY_KEY, JSON.stringify({ a: { done: false } }));
  assert.equal(createStore(() => ms).isDone('a'), true, 'once v2 exists, v1 is not read again');
  assert.deepEqual(migrateV1({ x: { checks: { k: 'yes', __proto__: true } } }).x.seen, {});
  const broken = memoryStorage(); broken.m.set(LEGACY_KEY, 'not json');
  assert.doesNotThrow(() => createStore(() => broken).markVisited('z'));
  assert.equal(createStore(() => broken).get('z').visited, true);
});

test('store: works when localStorage throws (accessor, getItem and setItem)', () => {
  const variants = {
    'accessor throws': () => { throw new DOMException('blocked', 'SecurityError'); },
    'getItem throws': () => ({ getItem() { throw new Error('denied'); }, setItem() {} }),
    'setItem throws (quota)': () => ({ getItem: () => null, setItem() { throw new Error('QuotaExceededError'); } }),
    'setItem throws during the v1 migration': () => ({ getItem: (k) => (k === LEGACY_KEY ? '{"x":{"visited":true,"checks":{"c":true}}}' : null), setItem() { throw new Error('denied'); } }),
    'storage missing': () => undefined,
  };
  for (const [name, get] of Object.entries(variants)) {
    const s = createStore(get);
    assert.doesNotThrow(() => { s.markVisited('x'); s.setDone('x', true); s.recordQuiz('x', 'q', true); s.setExplained('x', 'e'); s.setSeen('x', 'c', true); }, name);
    assert.equal(s.isDone('x'), true, `${name}: still remembered for this page view`);
    assert.deepEqual(s.quiz('x', 'q'), { tries: 1, correct: true }, name);
    assert.equal(s.get('x').explained.e, true, name);
    assert.equal(s.get('x').seen.c, true, name);
    assert.equal(s.persistent, false, name);
  }
});

test('store: corrupt or hand-edited data is ignored, not trusted', () => {
  for (const raw of ['not json', '[]', '"x"', 'null', '{"a":5}', '{"__proto__":{"done":true}}']) {
    const ms = memoryStorage(); ms.m.set(STORAGE_KEY, raw);
    const s = createStore(() => ms);
    assert.equal(s.isDone('a'), false, raw);
    assert.equal(s.isDone('__proto__'), false, raw);
    assert.equal(s.isDone('constructor'), false, raw);
    assert.doesNotThrow(() => { s.setDone('a', true); s.recordQuiz('a', '__proto__', true); s.setSeen('a', '__proto__', true); }, raw);
    assert.deepEqual(s.get('a').quiz, {}, raw);
  }
  assert.deepEqual(sanitize({ a: { visited: 'yes', done: 1, checks: { r: true }, quiz: { q: { tries: 2, correct: true }, n: { tries: -1, correct: true }, f: { tries: 1.5 }, z: { tries: 0, correct: true }, s: 'x' },
    explained: { e: true, f: 'yes' }, seen: { k: true, l: 1 } } }),
  { a: { visited: false, done: false, quiz: { q: { tries: 2, correct: true }, z: { tries: 0, correct: false } }, explained: { e: true }, seen: { k: true } } },
  'v1 checks are dropped, bad tries are dropped, correct needs an attempt, only true flags stay');
});

// ---- quiz answers (core/quiz.js, pure part) ---------------------------------------------------------------------
test('quiz: typed answers are parsed and judged with the tolerance', () => {
  for (const [s, v] of [['1.92', 1.92], [' -14.4 ', -14.4], ['−14.4', -14.4], ['1,92', 1.92], ['26,506', 26506], ['.5', 0.5], ['3', 3], ['1e-3', 0.001]]) assert.equal(parseAnswer(s), v, s);
  for (const s of ['', '  ', 'abc', '1.2.3', '--1', '1,2,3', null]) assert.equal(parseAnswer(s), null, String(s));
  assert.equal(judgeNumeric(35.7604, 0.01, 35.76), true);
  assert.equal(judgeNumeric(35.7604, 0.01, 35.75), false);
  assert.equal(judgeNumeric(1.92000008, 0.01, 1.93), true, 'the edge of the tolerance counts');
  assert.equal(judgeNumeric(1, 0.1, null), false);
  assert.equal(judgeChoice('2', '2'), true);
  assert.equal(judgeChoice(2, '1'), false);
  assert.equal(judgeChoice(0, null), false);
});

// ---- prose numbers (core/numbers.js) ---------------------------------------------------------------------------
test('numbers: comma thousands separators, the minus sign, role-based significant figures', () => {
  assert.equal(groupThousands('26506'), '26,506');
  assert.equal(groupThousands('-1234567.25'), '-1,234,567.25');
  assert.equal(groupThousands('999'), '999');
  assert.equal(groupThousands('1.5e-7'), '1.5e-7');
  assert.equal(proseNumber(26506), '26,506');
  assert.equal(proseNumber(60000), '60,000');
  assert.equal(proseNumber(1234567), '1,234,567', 'integers are exact at any size (fmt would round)');
  assert.equal(proseNumber(-24), '−24');
  assert.equal(proseNumber(0.0566912), '0.05669', 'exact roles use fmt: 4 significant figures');
  assert.equal(proseNumber(1234.56), '1,235');
  assert.equal(proseNumber(0.0566912, 2), '0.057');
  assert.equal(proseNumber(0.00008427, 2), '0.000084');
  assert.equal(proseNumber(26506, 2), '27,000', 'an explicit sig rounds integers too');
  assert.equal(proseNumber(-0), '0');
  assert.equal(proseNumber(3.0000001, 2), '3');
  assert.throws(() => proseNumber(1, 5), /sig/);
  assert.throws(() => proseNumber(NaN), /finite/);
  assert.equal(sigFor('setup', 0.123), 'exact');
  assert.equal(sigFor('contrast', 0.123), 2);
  assert.equal(sigFor('contrast', 26506), 'exact');
  assert.equal(sigFor('worked', 0.123, 3), 3);
  assert.equal(humanText('rmse', 0.0576), '0.24');
  assert.equal(humanText('prob', 0.4155), '66%');
  assert.equal(humanText('pct', 9666, 10000), '97%');
  assert.equal(humanText('pct', 0.9666), '97%');
  assert.equal(humanText('per100', 9666, 10000), '3');
  assert.equal(humanText('per100', 9650, 10000), '3', 'per100 is 100 minus the shown percent, so the two always add to 100');
  assert.throws(() => humanText('rmse', -1), /negative/);
  assert.throws(() => humanText('odds', 1), /unknown kind/);
});

test('T(path, role), T.anchor and T.human: tagged spans that the tests can re-format; the legacy T(path) is unchanged', () => {
  const D = buildD({ t: { library: { ref: 'v0' }, scenes: [{ id: 's', nodes: [{ name: 'x', value: { shape: [1], values: [-24] }, grad: { shape: [1], values: [0.0566912] } }] }], big: 26506, n: 9666, of: 10000, mse: 0.0576 } }, { mlp_parameters: 26506 });
  const T = makeT(D);
  assert.equal(T('s.x.value'), '<span class="num" data-trace="s.x.value">-24</span>', 'legacy: fmt, no role');
  assert.equal(T('s.x.value', 'setup'), '<span class="num" data-trace="s.x.value" data-role="setup" data-sig="exact">−24</span>');
  assert.equal(T('s.x.grad', 'contrast'), '<span class="num" data-trace="s.x.grad" data-role="contrast" data-sig="2">0.057</span>');
  assert.equal(T('s.x.grad', { role: 'worked', sig: 3 }), '<span class="num" data-trace="s.x.grad" data-role="worked" data-sig="3">0.0567</span>');
  assert.equal(T('files.t.big', 'reveal'), '<span class="num" data-trace="files.t.big" data-role="reveal" data-sig="exact">26,506</span>');
  assert.equal(T.anchor('mlp_parameters'), '<span class="num" data-trace="anchors.mlp_parameters" data-role="anchor" data-sig="exact" data-anchor="mlp_parameters">26,506</span>');
  assert.equal(T.human('files.t.n', 'pct', { of: 'files.t.of' }), '<span class="num" data-trace="files.t.n" data-role="human" data-human="pct" data-of="files.t.of">97%</span>');
  assert.equal(T.human('files.t.mse', 'rmse'), '<span class="num" data-trace="files.t.mse" data-role="human" data-sig="2" data-human="rmse">0.24</span>');
  for (const html of [T('s.x.grad', 'contrast'), T.anchor('mlp_parameters'), T.human('files.t.n', 'per100', { of: 'files.t.of' }), T.human('files.t.mse', 'rmse', { sig: 3 }), T('s.x.value')]) {
    const [b] = boundNums(html);
    assert.equal(T.reformat(b), b.shown, html);
  }
  assert.throws(() => T('s.x.value', 'anchor'), /T\.anchor/);
  assert.throws(() => T('s.x.value', 'human'), /T\.human/);
  assert.throws(() => T('s.x.value', 'guess'), /unknown role/);
  assert.throws(() => T('s.x.value', 'setup', { sig: 7 }), /sig/);
  assert.throws(() => T.anchor('nope'), /no such anchor/);
  assert.throws(() => makeT(buildD({})).anchor('mlp_parameters'), /not loaded/);
  assert.throws(() => T.human('files.t.mse', 'rmse', { of: 'files.t.of' }), /only for pct/);
  assert.throws(() => T.human('files.t.n', 'pct', { sig: 2 }), /whole numbers/);
  assert.throws(() => T.human('files.t.n', 'odds'), /unknown kind/);
  assert.throws(() => T('s.x.missing', 'setup'), /no "missing"/);
});

test('trace: loadAnchors reads data/anchors.json relative to the site root and checks its schema', async () => {
  const fake = (json) => async (url) => ({ ok: true, json: async () => ({ ...json, url: String(url) }) });
  assert.deepEqual(await loadAnchors(new URL('http://h/TorC/'), fake({ schema: 'torc-anchors/1', anchors: { a: { path: 'p', value: 3 } } })), { a: 3 });
  await assert.rejects(loadAnchors(new URL('http://h/'), fake({ schema: 'other' })), /torc-anchors\/1/);
  await assert.rejects(loadAnchors(new URL('http://h/'), async () => ({ ok: false, status: 404 })), /404/);
});

// ---- course -----------------------------------------------------------------------------------------------
const L = (id, over = {}) => ({ id, title: id.toUpperCase(), outcome: `learn ${id}`, prereqs: [], symbols: [`sym_${id}`], trace: [], est_minutes: 5, status: 'published', ...over });
const sample = () => ({ title: 'T', parts: [
  { id: 'p1', title: 'One', lessons: [L('a'), L('b', { prereqs: ['a'], status: 'draft' }), L('c', { prereqs: ['a'], legacy_hashes: ['old1'] })] },
  { id: 'p2', title: 'Two', lessons: [L('d', { status: 'draft' })] },
  { id: 'p3', title: 'Three', lessons: [L('e', { status: 'planned', code: 'f();', widgets: ['w'] }), L('f')] },
  { id: 'lab', title: 'Labs', kind: 'lab', lessons: [L('x'), L('y', { status: 'draft' }), L('z')] },
] });

test('course: validation catches duplicate ids, unknown/self/cyclic prereqs, bad fields', () => {
  assert.deepEqual(validateCourse(sample()), []);
  const dup = sample(); dup.parts[1].lessons.push(L('a')); assert.match(validateCourse(dup).join(), /duplicate lesson id "a"/);
  const unk = sample(); unk.parts[0].lessons[0].prereqs = ['zzz']; assert.match(validateCourse(unk).join(), /prereq "zzz"/);
  const self = sample(); self.parts[0].lessons[0].prereqs = ['a']; assert.match(validateCourse(self).join(), /itself/);
  const cyc = sample(); cyc.parts[0].lessons[0].prereqs = ['c']; assert.match(validateCourse(cyc).join(), /cycle/);
  const st = sample(); st.parts[0].lessons[0].status = 'soon'; assert.match(validateCourse(st).join(), /status/);
  const bad = sample(); bad.parts[0].lessons[0].trace = ['../x.json']; assert.match(validateCourse(bad).join(), /plain \*\.json/);
  const leg = sample(); leg.parts[1].lessons[0].legacy_hashes = ['old1']; assert.match(validateCourse(leg).join(), /legacy hash "old1"/);
  const kind = sample(); kind.parts[0].kind = 'appendix'; assert.match(validateCourse(kind).join(), /kind must be "lab"/);
  const code = sample(); code.parts[0].lessons[0].code = ''; assert.match(validateCourse(code).join(), /"code"/);
  const wid = sample(); wid.parts[0].lessons[0].widgets = 'graph'; assert.match(validateCourse(wid).join(), /"widgets"/);
  const legp = sample(); legp.parts[2].lessons[0].legacy_hashes = ['gone']; assert.match(validateCourse(legp).join(), /planned lesson has no page/);
});

test('course: the curriculum lists published and planned lessons, numbered; drafts and labs are not in it; labs listed apart', () => {
  const c = sample();
  assert.deepEqual(curriculum(c).map((p) => [p.id, p.lessons.map((l) => `${l.number}:${l.id}`)]), [['p1', ['1:a', '2:c']], ['p3', ['3:e', '4:f']]], 'p2 has only a draft');
  assert.deepEqual(labs(c).map((l) => l.id), ['x', 'z']);
  assert.deepEqual(flatLessons(c).map((l) => l.id), ['a', 'b', 'c', 'd', 'e', 'f', 'x', 'y', 'z']);
});

test('course: prev/next skip drafts and planned lessons, stay within lessons or within labs; a draft gets neighbours by position', () => {
  const c = sample();
  assert.deepEqual([neighbors(c, 'a').prev, neighbors(c, 'a').next?.id], [null, 'c']);
  assert.deepEqual([neighbors(c, 'c').prev?.id, neighbors(c, 'c').next?.id], ['a', 'f'], 'the planned lesson e has no page and is skipped');
  assert.deepEqual([neighbors(c, 'f').prev?.id, neighbors(c, 'f').next], ['c', null], 'lessons never lead into the labs');
  assert.deepEqual([neighbors(c, 'b').prev?.id, neighbors(c, 'b').next?.id], ['a', 'c']);
  assert.deepEqual([neighbors(c, 'x').prev, neighbors(c, 'x').next?.id], [null, 'z'], 'labs link to labs only');
  assert.deepEqual(neighbors(c, 'nope'), { prev: null, next: null });
});

test('course: old single-page deep links map to lessons/backprop-lab/ with the hash kept (relative URL)', () => {
  const real = readCourse();
  for (const h of ['#reuse', '#chain/4', '#mlp/9', '#lab']) assert.equal(legacyTarget(real, h), 'lessons/backprop-lab/' + h);
  assert.equal(legacyTarget(real, ''), null);
  assert.equal(legacyTarget(real, '#nothing/1'), null);
  assert.ok(!legacyTarget(real, '#mlp/9').startsWith('/'), 'never root-absolute');
  assert.equal(lessonHref('x'), 'lessons/x/');
});

// ---- trace data and T() -----------------------------------------------------------------------------------------
test('trace: loader ignores unknown top-level keys and reads scenes[] at the top level', () => {
  const D = buildD({ a: { schema: 'torc-trace/1', library: { ref: 'v9' }, whatever: [1, 2], scenes: [{ id: 's1', nodes: [] }] }, b: { grid: [1], future_key: {} } });
  assert.deepEqual(Object.keys(D.scenes), ['s1']);
  assert.deepEqual(Object.keys(D.files), ['a', 'b']);
  assert.throws(() => buildD({ a: { scenes: [{ id: 's' }] }, b: { scenes: [{ id: 's' }] } }), /more than one trace file/);
  assert.throws(() => buildD({ a: { scenes: {} } }), /must be an array/);
  assert.throws(() => buildD({ a: [] }), /JSON object/);
});

test('trace: loadTrace reads data/ files relative to the site root via fetch', async () => {
  const seen = [];
  const fakeFetch = async (url) => { seen.push(String(url)); return { ok: true, json: async () => ({ scenes: [{ id: String(url).split('/').pop() }] }) }; };
  const D = await loadTrace(['x.json', 'y.json'], new URL('http://h/TorC/'), fakeFetch);
  assert.deepEqual(seen.sort(), ['http://h/TorC/data/x.json', 'http://h/TorC/data/y.json']);
  assert.deepEqual(Object.keys(D.scenes), ['x.json', 'y.json']);
  await assert.rejects(loadTrace(['z.json'], new URL('http://h/'), async () => ({ ok: false, status: 404 })), /404/);
});

test('T(): resolves scene/node/tensor paths from the real trace and throws on anything missing', () => {
  const D = buildD(Object.fromEntries(Object.entries(traceFiles()).map(([f, j]) => [f.replace('.json', ''), j])));
  const T = makeT(D);
  assert.equal(T.num('reuse.x.grad'), 7);
  assert.equal(T.text('reuse.x.value'), '3');
  assert.equal(T('reuse.x.grad'), '<span class="num" data-trace="reuse.x.grad">7</span>');
  assert.equal(T.num('mlp.b1.value.1'), D.scenes.mlp.nodes.find((n) => n.name === 'b1').value.values[1]);
  assert.equal(T.num('mlp.W1.value.values.0'), D.scenes.mlp.nodes.find((n) => n.name === 'W1').value.values[0]);
  assert.throws(() => T('nope.x.grad'), /no "nope"|cannot descend|no "x"/);
  assert.throws(() => T('reuse.zzz.grad'), /no "zzz"/);
  assert.throws(() => T('reuse.x.hessian'), /no "hessian"/);
  assert.throws(() => T('mlp.W1.value'), /elements; add an index/);
  assert.throws(() => T('reuse.y.grad'), /null/, 'the root has no stored gradient');
  assert.throws(() => T('reuse.x'), /does not resolve to a number/);
  assert.equal(resolvePath(D, 'files.autograd_trace.schema'), 'torc-trace/1');
});

// ---- sections -------------------------------------------------------------------------------------------------------
test('sections: kinds render; unbound numerals are detected; bound ones are re-checkable', () => {
  const D = buildD({ t: { scenes: [{ id: 's', nodes: [{ name: 'x', grad: { shape: [1], values: [2.5] } }] }] } });
  const lesson = { id: 'l', title: 'L', trace: ['t.json'], sections: [
    { kind: 'text', html: (d, T) => `<p>grad is ${T('s.x.grad')}</p>` },
    { kind: 'gotcha', title: 'Why <it> failed', html: () => '<p>keep it alive</p>' },
    { kind: 'predict', id: 'p1', prompt: () => '<p>?</p>', reveal: (d, T) => `<p>${T('s.x.grad')}</p>` },
    { kind: 'check', id: 'c1', prompt: () => '<p>?</p>', reveal: () => '<p>!</p>' },
    { kind: 'widget', id: 'w1', mount() {} },
  ] };
  assert.deepEqual(validateLesson(lesson), []);
  const parts = renderLesson(lesson, D);
  const html = parts.map((p) => p.html).join('\n');
  assert.deepEqual(findUnboundNums(html), []);
  assert.deepEqual(boundNums(html).map((b) => [b.path, b.shown]), [['s.x.grad', '2.5'], ['s.x.grad', '2.5']]);
  assert.match(parts[1].html, /Why &lt;it&gt; failed/);
  assert.match(parts[3].html, /data-check="c1"/);
  assert.match(parts[4].html, /data-widget="w1"/);
  assert.deepEqual(findUnboundNums('<p>The loss is <span class="num">36</span></p>'), ['<span class="num">']);
  assert.deepEqual(findUnboundNums('<span class="x num y">1</span>').length, 1);
  assert.throws(() => renderLesson({ sections: [{ kind: 'text', html: (d, T) => T('s.x.missing') }] }, D), /no "missing"/);
  assert.throws(() => sectionHtml({ kind: 'video' }, D), /unknown section kind/);
  assert.match(validateLesson({ id: 'a', title: 'a', trace: [], sections: [{ kind: 'text' }] }).join(), /text sections need html/);
});
