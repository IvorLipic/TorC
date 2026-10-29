// core/blocks.js — small pieces of markup that v3 lesson text sections use (docs/v3-implementation.md "Blocks", R6, R7, R9).
// Pure functions that return HTML strings (no DOM), so Node renders them in the tests exactly as the browser does. All arguments
// named html, limit, title or caption are HTML fragments: inline (phrasing) markup for analogy, chatgpt and the titles, any markup
// for worked and the bodies. Behaviour (the glossary popover) is wired by lesson-boot.js through core/popover.js.
//
//   worked(html)              a calculation whose inputs are all visible; the only place for T(path, 'worked'). ≤ 3 per lesson (tested)
//   inTorc(title, html)       collapsed "In TorC" panel: the library detail behind an idea
//   realLine(code, caption)   one of the 4-5 "this is the real line" moments; code is plain text (escaped here)
//   analogy(html, limit)      an analogy from the brief's bank, always with "Where it breaks"
//   chatgpt(html)             the "same thing in ChatGPT" aside
//   refresher(key)            a collapsible maths refresher from content/refreshers.js (throws on an unknown key)
//   term(key, text?)          a glossary link (../glossary/#key) that gets a hover, focus and tap definition from site/glossary.json
//   sym(key, html)            a symbol linked to the notation guide (../notation/#key); key from content/notation.js (throws if unknown)
//   loopDiagram(active)       inline SVG of the training loop data → model → loss → adjust, with the lesson's part lit
// Palette rule (brief §9): 0-2 of analogy, refresher, inTorc, chatgpt per lesson. Labels end with a full stop ("Analogy.") so the
// writing rules see a complete block, not a colon in the middle of a sentence.
import { escapeHtml } from './dom.js';
import { REFRESHERS } from '../content/refreshers.js';
import { notationByKey } from '../content/notation.js';

const KEY_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const checkKey = (fn, key) => { if (!KEY_RE.test(String(key))) throw new Error(`${fn}("${key}"): a key is lowercase-hyphenated`); };
const label = (text) => `<strong class="label">${text}.</strong>`;

export const worked = (html) => `<div class="worked" role="group" aria-label="Worked example">${html}</div>`;

export const inTorc = (title, html) =>
  `<details class="in-torc"><summary>${label('In TorC')} ${title}</summary><div class="body">${html}</div></details>`;

export const realLine = (code, caption) =>
  `<figure class="real-line"><pre tabindex="0" role="group" aria-label="One line of TorC code"><code>${escapeHtml(code)}</code></pre>`
  + `<figcaption>${caption}</figcaption></figure>`;

export function analogy(html, limit) {
  if (!limit) throw new Error('analogy(html, limit): every analogy states where it breaks');
  return `<aside class="analogy" aria-label="Analogy"><p>${label('Analogy')} ${html}</p><p>${label('Where it breaks')} ${limit}</p></aside>`;
}

export const chatgpt = (html) => `<aside class="chatgpt" aria-label="The same idea in ChatGPT"><p>${label('In ChatGPT')} ${html}</p></aside>`;

export function refresher(key) {
  const r = Object.hasOwn(REFRESHERS, key) ? REFRESHERS[key] : null;
  if (!r) throw new Error(`refresher("${key}"): no such refresher in js/content/refreshers.js (have: ${Object.keys(REFRESHERS).join(', ')})`);
  return `<details class="refresher" data-refresher="${escapeHtml(key)}"><summary>${label('Refresher')} ${escapeHtml(r.title)}</summary><div class="body">${r.html}</div></details>`;
}

export function term(key, text) {
  checkKey('term', key);
  return `<a class="term" href="../glossary/#${key}" data-term="${key}">${text ?? escapeHtml(key.replace(/-/g, ' '))}</a>`;
}

export function sym(key, html) {
  checkKey('sym', key);
  const n = notationByKey(key);
  if (!n) throw new Error(`sym("${key}"): no such symbol in js/content/notation.js`);
  return `<a class="sym" href="../notation/#${key}" data-sym="${key}" title="${escapeHtml(`${n.name}, read as “${n.read}”`)}">${html ?? escapeHtml(n.symbol)}</a>`;
}

// ---- the loop diagram --------------------------------------------------------------------------------------------------
export const LOOP_PARTS = ['data', 'model', 'loss', 'adjust'];
const LOOP_ACTIVE = [...LOOP_PARTS, 'all'];
const LOOP_NAME = { data: 'the data', model: 'the model', loss: 'the loss', adjust: 'the adjust step', all: 'the whole loop' };
// Boxes clockwise from the top left. Drawn at its design size (CSS fixes the width), so the 14px labels are never scaled down.
const BOX = { w: 96, h: 36 };
const POS = { data: [16, 14], model: [168, 14], loss: [168, 100], adjust: [16, 100] };
const ARROWS = [   // [x1, y1, x2, y2]: data → model → loss → adjust → data
  [112, 32, 164, 32], [216, 50, 216, 96], [168, 118, 116, 118], [64, 100, 64, 54],
];
function head([x1, y1, x2, y2]) {
  const a = Math.atan2(y2 - y1, x2 - x1), s = 7;
  const p = (da) => `${(x2 - s * Math.cos(a + da)).toFixed(1)},${(y2 - s * Math.sin(a + da)).toFixed(1)}`;
  return `<polygon class="loop-head" points="${x2},${y2} ${p(0.5)} ${p(-0.5)}"/>`;
}

export function loopDiagram(active) {
  if (!LOOP_ACTIVE.includes(active)) throw new Error(`loopDiagram("${active}"): use one of ${LOOP_ACTIVE.join(', ')}`);
  const aria = 'The training loop. The data goes into the model, the loss measures how wrong the model is, and the adjust step changes the weights. '
    + `Then the loop repeats. This lesson is about ${LOOP_NAME[active]}.`;
  const nodes = LOOP_PARTS.map((p) => {
    const [x, y] = POS[p], on = active === 'all' || active === p;
    return `<g class="loop-node${on ? ' on' : ''}"><rect x="${x}" y="${y}" width="${BOX.w}" height="${BOX.h}" rx="9"/>`
      + `<text x="${x + BOX.w / 2}" y="${y + BOX.h / 2}" dominant-baseline="central" text-anchor="middle">${p}</text></g>`;
  }).join('');
  const arrows = ARROWS.map((a) => `<line class="loop-edge" x1="${a[0]}" y1="${a[1]}" x2="${a[2]}" y2="${a[3]}"/>${head(a)}`).join('');
  return `<figure class="loop" data-loop="${active}"><svg viewBox="0 0 280 150" width="280" height="150" role="img" aria-label="${escapeHtml(aria)}" focusable="false">${arrows}${nodes}</svg></figure>`;
}
