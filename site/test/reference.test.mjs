// The reference content of v3 (brief §8 "Notation", "Maths refreshers", "Glossary"; decision 18; R9): the ten maths refreshers,
// the notation guide and the glossary, and the two appendix pages built from them (lessons/notation.js, lessons/glossary.js).
// The well-formedness rules that every entry shares are in v3.test.mjs; this file checks the content itself.
import test from 'node:test';
import assert from 'node:assert/strict';
import { REFRESHERS } from '../js/content/refreshers.js';
import { proseProblems, sectionWords } from './helpers.mjs';
import { tokens } from './v3.mjs';

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
