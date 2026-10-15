// core/: hash state, progress store (including blocked storage), course helpers, trace loader and T(), sections.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseHash, formatHash } from '../js/core/state.js';
import { createStore, sanitize, STORAGE_KEY } from '../js/core/store.js';
import { validateCourse, curriculum, labs, neighbors, legacyTarget, flatLessons, lessonHref } from '../js/core/course.js';
import { buildD, makeT, resolvePath, loadTrace } from '../js/core/trace.js';
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

test('store: progress is kept under torc:v1 as {lesson:{visited,checks,done}}', () => {
  const ms = memoryStorage();
  const s = createStore(() => ms);
  assert.equal(s.isDone('a'), false);
  s.markVisited('a'); s.setCheck('a', 'q1', true); s.setDone('a', true);
  assert.deepEqual(JSON.parse(ms.m.get(STORAGE_KEY)), { a: { visited: true, checks: { q1: true }, done: true } });
  assert.equal(createStore(() => ms).isDone('a'), true, 'a new store instance (page reload) sees it');
  s.setDone('a', false);
  assert.equal(createStore(() => ms).isDone('a'), false);
  assert.equal(s.persistent, true);
});

test('store: works when localStorage throws (accessor, getItem and setItem)', () => {
  const variants = {
    'accessor throws': () => { throw new DOMException('blocked', 'SecurityError'); },
    'getItem throws': () => ({ getItem() { throw new Error('denied'); }, setItem() {} }),
    'setItem throws (quota)': () => ({ getItem: () => null, setItem() { throw new Error('QuotaExceededError'); } }),
    'storage missing': () => undefined,
  };
  for (const [name, get] of Object.entries(variants)) {
    const s = createStore(get);
    assert.doesNotThrow(() => { s.markVisited('x'); s.setDone('x', true); s.setCheck('x', 'c', true); }, name);
    assert.equal(s.isDone('x'), true, `${name}: still remembered for this page view`);
    assert.equal(s.get('x').checks.c, true, name);
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
    assert.doesNotThrow(() => s.setDone('a', true));
  }
  assert.deepEqual(sanitize({ a: { visited: 'yes', done: 1, checks: { q: 'x', r: true } } }), { a: { visited: false, done: false, checks: { r: true } } });
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
