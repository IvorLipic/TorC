// core/course.js on the real course.json (curriculum v3, docs/v3-implementation.md "Course" and "Curriculum v3"), plus the
// helpers the home page uses: appendix(), filterLessons(), studyTime(), aboutHours(). The structural course tests on a small
// sample course are in core.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  validateCourse, curriculum, labs, appendix, neighbors, flatLessons, partKind, isAppendixPart, isLabPart,
  filterLessons, studyTime, aboutHours, isPublished,
} from '../js/core/course.js';
import { course as readCourse } from './helpers.mjs';

const real = readCourse();

// ---- curriculum v3 --------------------------------------------------------------------------------------------------------
const V3 = [
  ['whole-game', undefined, ['what-is-learning']],
  ['gradient', undefined, ['why-gradients', 'sgd-step']],
  ['networks', undefined, ['linear-layer', 'nonlinearity', 'classification', 'losses', 'backprop', 'deep-nets']],
  ['training', undefined, ['training-loop', 'optimizers', 'generalization']],
  ['images', undefined, ['mnist-mlp', 'one-step', 'convolution', 'mnist-cnn']],
  ['next', undefined, ['from-digits-to-chatgpt']],
  ['using-torc', 'appendix', ['tensors', 'modules', 'torc-pitfalls', 'notation', 'glossary']],
  ['labs', 'lab', ['backprop-lab']],
];
const PUBLISHED = ['what-is-learning', 'why-gradients', 'sgd-step', 'tensors', 'linear-layer', 'nonlinearity', 'modules', 'losses', 'training-loop',
  'optimizers', 'generalization', 'mnist-mlp', 'convolution', 'mnist-cnn', 'backprop-lab', 'notation', 'glossary'];
const NEW = ['classification', 'backprop', 'deep-nets', 'one-step', 'from-digits-to-chatgpt', 'torc-pitfalls'];

test('course.json: parts, kinds and lesson ids follow curriculum v3 exactly', () => {
  assert.deepEqual(validateCourse(real), []);
  assert.deepEqual(real.parts.map((p) => [p.id, p.kind, p.lessons.map((l) => l.id)]), V3);
});

test('course.json: every lesson that has a page stays published, and the new ids are planned', () => {
  const byId = Object.fromEntries(flatLessons(real).map((l) => [l.id, l]));
  for (const id of PUBLISHED) assert.equal(byId[id]?.status, 'published', id);
  for (const id of NEW) assert.equal(byId[id]?.status, 'planned', id);
  assert.equal(flatLessons(real).length, PUBLISHED.length + NEW.length, 'no other lessons');
});

test('course.json: prerequisites of the main path come earlier in the main path', () => {
  const main = curriculum(real).flatMap((p) => p.lessons.map((l) => l.id));
  for (const l of curriculum(real).flatMap((p) => p.lessons)) {
    for (const q of l.prereqs) assert.ok(main.indexOf(q) >= 0 && main.indexOf(q) < main.indexOf(l.id), `${l.id}: prereq ${q} must be an earlier main-path lesson`);
  }
  const losses = flatLessons(real).find((l) => l.id === 'losses');
  assert.ok(!losses.prereqs.includes('modules'), 'losses no longer depends on the appendix page modules');
  assert.deepEqual(losses.prereqs, ['classification']);
});

test('course.json: the curriculum has the 17 main-path lessons, numbered 1 to 17 in order, without the appendix and labs', () => {
  const lessons = curriculum(real).flatMap((p) => p.lessons);
  assert.deepEqual(curriculum(real).map((p) => p.id), ['whole-game', 'gradient', 'networks', 'training', 'images', 'next']);
  assert.deepEqual(lessons.map((l) => l.number), Array.from({ length: 17 }, (_, i) => i + 1));
  assert.deepEqual(lessons.map((l) => l.id), V3.filter(([, kind]) => !kind).flatMap(([, , ids]) => ids));
  assert.deepEqual(appendix(real).map((l) => l.id), ['tensors', 'modules', 'torc-pitfalls', 'notation', 'glossary']);
  assert.ok(appendix(real).every((l) => l.number === undefined), 'appendix pages are not numbered');
  assert.deepEqual(labs(real).map((l) => l.id), ['backprop-lab']);
});

test('course.json: prev/next on the real course stay within the main path, the appendix or the labs, and skip planned lessons', () => {
  const chain = (ids) => ids.map((id, i) => [id, ids[i - 1] ?? null, ids[i + 1] ?? null]);
  const expected = [
    ...chain(['what-is-learning', 'why-gradients', 'sgd-step', 'linear-layer', 'nonlinearity', 'losses', 'training-loop', 'optimizers', 'generalization',
      'mnist-mlp', 'convolution', 'mnist-cnn']),
    ...chain(['tensors', 'modules', 'notation', 'glossary']),
    ...chain(['backprop-lab']),
  ];
  for (const [id, prev, next] of expected) {
    const n = neighbors(real, id);
    assert.deepEqual([n.prev?.id ?? null, n.next?.id ?? null], [prev, next], id);
  }
  assert.equal(expected.length, flatLessons(real).filter(isPublished).length, 'every published page is covered');
});

// ---- helpers -----------------------------------------------------------------------------------------------------------
const L = (id, over = {}) => ({ id, title: id.toUpperCase(), outcome: `learn ${id}`, prereqs: [], symbols: [`sym_${id}`], trace: [], est_minutes: 5, status: 'published', ...over });

test('course: part kinds are main (absent), appendix or lab', () => {
  assert.equal(partKind({}), 'main');
  assert.equal(partKind({ kind: 'appendix' }), 'appendix');
  assert.equal(partKind({ kind: 'lab' }), 'lab');
  assert.equal(isAppendixPart({ kind: 'appendix' }), true);
  assert.equal(isAppendixPart({ kind: 'lab' }), false);
  assert.equal(isLabPart({ kind: 'appendix' }), false);
});

test('course: appendix() lists published and planned pages of appendix parts, never drafts', () => {
  const c = { title: 'T', parts: [
    { id: 'm', title: 'M', lessons: [L('a')] },
    { id: 'r', title: 'R', kind: 'appendix', lessons: [L('r1'), L('r2', { status: 'draft' }), L('r3', { status: 'planned' })] },
    { id: 'x', title: 'X', kind: 'lab', lessons: [L('x1')] },
  ] };
  assert.deepEqual(appendix(c).map((l) => [l.id, l.partId]), [['r1', 'r'], ['r3', 'r']]);
  assert.deepEqual(appendix({ title: 'T', parts: [{ id: 'm', title: 'M', lessons: [L('a')] }] }), []);
});

test('course: studyTime() sums est_minutes of the main path, published and planned, and aboutHours() rounds to half hours', () => {
  const c = { title: 'T', parts: [
    { id: 'm', title: 'M', lessons: [L('a', { est_minutes: 40 }), L('b', { est_minutes: 25, status: 'planned' }), L('d', { est_minutes: 99, status: 'draft' })] },
    { id: 'r', title: 'R', kind: 'appendix', lessons: [L('r1', { est_minutes: 60 })] },
    { id: 'x', title: 'X', kind: 'lab', lessons: [L('x1', { est_minutes: 60 })] },
  ] };
  assert.deepEqual(studyTime(c), { lessons: 2, minutes: 65 });
  assert.equal(aboutHours(65), 'about 1 hour');
  assert.equal(aboutHours(80), 'about 1.5 hours');
  assert.equal(aboutHours(163), 'about 2.5 hours');
  assert.equal(aboutHours(170), 'about 3 hours');
  assert.equal(aboutHours(5), 'about 0.5 hours');
  const t = studyTime(real);
  assert.equal(t.lessons, 17);
  assert.equal(t.minutes, curriculum(real).flatMap((p) => p.lessons).reduce((s, l) => s + l.est_minutes, 0));
});

test('course: filterLessons() matches every word in the title, outcome, id, symbols or widgets, ignoring case', () => {
  const ls = [L('alpha', { title: 'The gradient', widgets: ['the bowl'] }), L('beta', { outcome: 'Train a network', symbols: ['optim::SGD'] })];
  assert.deepEqual(filterLessons(ls, '').map((l) => l.id), ['alpha', 'beta']);
  assert.deepEqual(filterLessons(ls, '   ').map((l) => l.id), ['alpha', 'beta']);
  assert.deepEqual(filterLessons(ls, 'GRADIENT').map((l) => l.id), ['alpha']);
  assert.deepEqual(filterLessons(ls, 'bowl').map((l) => l.id), ['alpha'], 'widgets are searched');
  assert.deepEqual(filterLessons(ls, 'sgd network').map((l) => l.id), ['beta'], 'symbols and outcome, every word');
  assert.deepEqual(filterLessons(ls, 'gradient network').map((l) => l.id), [], 'all words must match');
  assert.deepEqual(filterLessons(curriculum(real).flatMap((p) => p.lessons), 'digits').map((l) => l.number).every(Number.isInteger), true, 'numbers survive the filter');
});
