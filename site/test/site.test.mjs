// course.json, the committed lesson shells and tools/check_site.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkSite, shellFor } from '../../tools/check_site.mjs';
import { validateCourse, flatLessons, isPublished } from '../js/core/course.js';
import { course } from './helpers.mjs';

const SITE_DIR = fileURLToPath(new URL('../', import.meta.url));

test('course.json is valid', () => assert.deepEqual(validateCourse(course()), []));

test('check_site passes on the committed site', async () => {
  const { errors, stats } = await checkSite(SITE_DIR);
  assert.deepEqual(errors, []);
  assert.ok(stats.published >= 1 && stats.references > 20);
});

test('every lesson shell is the template, ~15 lines, with relative URLs only', () => {
  for (const l of flatLessons(course()).filter((x) => existsSync(path.join(SITE_DIR, 'lessons', x.id, 'index.html')))) {
    const shell = readFileSync(path.join(SITE_DIR, 'lessons', l.id, 'index.html'), 'utf8');
    assert.equal(shell, shellFor(l), `${l.id} shell`);
    assert.ok(shell.split('\n').length <= 22, 'shells stay tiny');
    assert.match(shell, /<title>[^<]+ · TorC Learn<\/title>/);
    assert.doesNotMatch(shell, /(?:href|src)="\//);
    assert.match(shell, /js\/lesson-boot\.js/);
  }
});

// ---- the checker catches real mistakes (run against a throw-away copy of the site) ----------------------------
function withCopy(fn) {
  const dir = mkdtempSync(path.join(tmpdir(), 'torc-site-'));
  const copy = path.join(dir, 'site');
  cpSync(SITE_DIR, copy, { recursive: true, filter: (src) => !src.includes(`${path.sep}test`) });
  return Promise.resolve(fn(copy)).finally(() => rmSync(dir, { recursive: true, force: true }));
}
const edit = (file, f) => writeFileSync(file, f(readFileSync(file, 'utf8')));
const messages = async (dir) => (await checkSite(dir)).errors.join('\n');

test('check_site: a clean copy passes', () => withCopy(async (d) => assert.deepEqual((await checkSite(d)).errors, [])));

test('check_site: root-absolute URLs are rejected in HTML, CSS and JS', () => withCopy(async (d) => {
  edit(path.join(d, 'index.html'), (s) => s.replace('href="css/base.css"', 'href="/css/base.css"'));
  edit(path.join(d, 'css/home.css'), (s) => s + '\n.x{background:url(/img/a.png)}');
  edit(path.join(d, 'js/home.js'), (s) => s + "\nfetch('/data/x.json');\nimport('/js/x.js');");
  edit(path.join(d, 'js/lesson-boot.js'), (s) => s + '\nconst q = "from \'/x.js\'"; import y from "/y.js";');
  const m = await messages(d);
  assert.match(m, /index\.html: root-absolute attribute URL/);
  assert.match(m, /home\.css: root-absolute CSS url\(\)/);
  assert.match(m, /home\.js: root-absolute fetch\(\)/);
  assert.match(m, /home\.js: root-absolute import/);
  assert.match(m, /lesson-boot\.js: root-absolute import/);
}));

test('check_site: <base> and missing relative assets are rejected', () => withCopy(async (d) => {
  edit(path.join(d, 'index.html'), (s) => s.replace('<head>', '<head>\n<base href="./">').replace('js/home.js', 'js/homme.js'));
  edit(path.join(d, 'css/base.css'), (s) => s + '\n.y{background:url(fonts/nope.woff2)}');
  edit(path.join(d, 'js/home.js'), (s) => "import './core/missing.js';\n" + s);
  const m = await messages(d);
  assert.match(m, /<base> element/);
  assert.match(m, /index\.html: attribute "js\/homme\.js" does not exist/);
  assert.match(m, /base\.css: url\(\) "fonts\/nope\.woff2" does not exist/);
  assert.match(m, /home\.js: import "\.\/core\/missing\.js" does not exist/);
}));

test('check_site: examples inside comments are not mistaken for code', () => withCopy(async (d) => {
  edit(path.join(d, 'js/home.js'), (s) => "// import x from '/nope.js'; fetch('/nope')\n/* href=\"/x\" url(/y) */\n" + s);
  assert.deepEqual((await checkSite(d)).errors, []);
}));

test('check_site: lesson problems (missing shell/module/trace, drifted shell, bad prereq, duplicate id, no title)', () => withCopy(async (d) => {
  const c = JSON.parse(readFileSync(path.join(d, 'course.json'), 'utf8'));
  const base = c.parts[0].lessons[0];
  c.parts[0].lessons.push({ ...base, id: 'ghost', title: 'Ghost', prereqs: ['nope'], trace: ['missing.json'], legacy_hashes: [] });   // published, nothing on disk
  c.parts[0].lessons.push({ ...base, id: 'backprop-lab' });                                                          // duplicate id
  writeFileSync(path.join(d, 'course.json'), JSON.stringify(c));
  let m = await messages(d);
  assert.match(m, /duplicate lesson id "backprop-lab"/);
  assert.match(m, /prereq "nope"/);
  // fix the structural errors to reach the on-disk checks
  c.parts[0].lessons.pop(); c.parts[0].lessons[1].prereqs = [];
  writeFileSync(path.join(d, 'course.json'), JSON.stringify(c));
  m = await messages(d);
  assert.match(m, /published lesson "ghost" has no shell/);
  assert.match(m, /published lesson "ghost" has no module/);
  assert.match(m, /trace file data\/missing\.json is missing/);
  // a draft lesson with nothing yet is only a warning
  c.parts[0].lessons[1].status = 'draft';
  writeFileSync(path.join(d, 'course.json'), JSON.stringify(c));
  const r = await checkSite(d);
  assert.deepEqual(r.errors, []);
  assert.ok(r.warnings.some((w) => /draft lesson "ghost"/.test(w)));
  // shell drift and a missing <title>
  edit(path.join(d, 'lessons/backprop-lab/index.html'), (s) => s.replace('data-lesson="backprop-lab"', 'data-lesson="other"').replace(/<title>.*<\/title>/, ''));
  m = await messages(d);
  assert.match(m, /does not equal the shell template/);
  assert.match(m, /no page <title>/);
}));

test('check_site: a lesson module that disagrees with course.json is rejected', () => withCopy(async (d) => {
  edit(path.join(d, 'js/lessons/backprop-lab.js'), (s) => s.replace("title: 'Backprop lab'", "title: 'Something else'"));
  assert.match(await messages(d), /title "Something else" differs from course\.json/);
}));

test('check_site: shell template output is stable for drafts and published alike', () => {
  const l = flatLessons(course()).find(isPublished);
  assert.equal(shellFor({ ...l, status: 'draft' }), shellFor(l));
  assert.match(shellFor({ ...l, title: 'A & "B"', outcome: 'x < y' }), /<title>A &amp; &quot;B&quot; · TorC Learn<\/title>/);
});
