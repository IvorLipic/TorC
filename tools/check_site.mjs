#!/usr/bin/env node
// tools/check_site.mjs — validates the static site in site/ before it is deployed (no dependencies, Node >= 20).
//
//   node tools/check_site.mjs [siteDir]        run all checks; exit 1 on any error
//   node tools/check_site.mjs --write-shells   (re)write lessons/<id>/index.html from course.json, then check
//
// Checks
//   course.json   valid structure, unique ids, prereqs exist, no prereq cycles
//   lessons       every PUBLISHED lesson has its shell (equal to the template below), its module js/lessons/<id>.js
//                 (whose id/title/trace agree with course.json), its trace files in data/, and a page <title>;
//                 draft lessons are only checked if they already have a shell or module
//   urls          no root-absolute URLs in HTML/CSS/JS (href="/ src="/ fetch("/ from "/ url(/ ...): the site must work
//                 under any prefix (/TorC/ on GitHub Pages) with no <base>
//   assets        every relative asset referenced by HTML (href/src), CSS (url(), @import) and JS (static imports,
//                 new URL('x', import.meta.url), fetch('x')) exists
import { readFileSync, writeFileSync, existsSync, statSync, readdirSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SITE = path.resolve(HERE, '../site');

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// The committed per-lesson page. Everything lesson-specific is in <title>, the description, and data-lesson.
export function shellFor(l) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="description" content="${esc(l.outcome)}">
<title>${esc(l.title)} · TorC Learn</title>
<link rel="icon" href="../../favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="../../vendor/katex/katex.min.css">
<link rel="stylesheet" href="../../css/base.css">
<link rel="stylesheet" href="../../css/widgets.css">
</head>
<body data-lesson="${esc(l.id)}">
<div id="app"><noscript><p class="load-error">This lesson needs JavaScript.</p></noscript></div>
<script src="../../vendor/katex/katex.min.js"></script>
<script type="module" src="../../js/lesson-boot.js"></script>
</body>
</html>
`;
}

// ---- helpers ----------------------------------------------------------------------------
function walk(dir, skip = () => false) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = path.join(dir, name);
    if (skip(p)) continue;
    if (statSync(p).isDirectory()) out.push(...walk(p, skip)); else out.push(p);
  }
  return out;
}

// Removes comments so that example code in comments is not scanned. Strings are left alone; a "//" that follows a ':' is a URL.
function stripJsComments(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:\\'"`])\/\/[^\n]*/g, '$1');
}
function stripCssComments(src) { return src.replace(/\/\*[\s\S]*?\*\//g, ''); }
function stripHtmlComments(src) { return src.replace(/<!--[\s\S]*?-->/g, ''); }

const ROOT_ABSOLUTE = [
  [/\b(?:href|src|action|poster)\s*=\s*["']\//g, 'root-absolute attribute URL'],
  [/\bfetch\(\s*["'`]\//g, 'root-absolute fetch()'],
  [/\bfrom\s*["']\//g, 'root-absolute import'],
  [/\bimport\s*\(?\s*["']\//g, 'root-absolute import'],
  [/\burl\(\s*["']?\//g, 'root-absolute CSS url()'],
  [/@import\s+["']\//g, 'root-absolute CSS @import'],
  [/\bnew URL\(\s*["'`]\//g, 'root-absolute new URL()'],
  [/\blocation\.(?:href|replace|assign)\b[\s\S]{0,12}["'`]\//g, 'root-absolute location change'],
  [/<base\b/gi, '<base> element (the site relies on relative URLs only)'],
];

const isExternal = (u) => /^(?:[a-z][a-z0-9+.-]*:|\/\/|#|$)/i.test(u);

// Returns [{ref, kind}] of relative references in a file.
function referencesIn(file, text) {
  const ext = path.extname(file);
  const refs = [];
  const push = (ref, kind) => { if (ref && !isExternal(ref) && !ref.startsWith('/') && !ref.startsWith('${') && !ref.includes('${')) refs.push({ ref, kind }); };
  if (ext === '.html') {
    const src = stripHtmlComments(text);
    for (const m of src.matchAll(/\b(?:href|src|poster)\s*=\s*"([^"]*)"/g)) push(m[1], 'attribute');
    for (const m of src.matchAll(/\b(?:href|src|poster)\s*=\s*'([^']*)'/g)) push(m[1], 'attribute');
  } else if (ext === '.css') {
    const src = stripCssComments(text);
    for (const m of src.matchAll(/url\(\s*(?:"([^"]*)"|'([^']*)'|([^)\s"']*))\s*\)/g)) push(m[1] ?? m[2] ?? m[3], 'url()');
    for (const m of src.matchAll(/@import\s+(?:url\(\s*)?["']([^"']+)["']/g)) push(m[1], '@import');
  } else if (ext === '.js' || ext === '.mjs') {
    const src = stripJsComments(text);
    for (const m of src.matchAll(/\b(?:import|export)\b[^'"`;]*?\bfrom\s*["']([^"']+)["']/g)) push(m[1], 'import');
    for (const m of src.matchAll(/\bimport\s*["']([^"']+)["']/g)) push(m[1], 'import');
    for (const m of src.matchAll(/\bimport\(\s*["']([^"']+)["']\s*\)/g)) push(m[1], 'import()');
    for (const m of src.matchAll(/\bnew URL\(\s*["']([^"']+)["']\s*,\s*import\.meta\.url\s*\)/g)) push(m[1], 'new URL(…, import.meta.url)');
    for (const m of src.matchAll(/\bfetch\(\s*["']([^"']+)["']/g)) push(m[1], 'fetch()');
  }
  return refs;
}

function firstDiff(a, b) {
  const A = a.split('\n'), B = b.split('\n');
  for (let i = 0; i < Math.max(A.length, B.length); i++) if (A[i] !== B[i]) return `line ${i + 1}: have ${JSON.stringify(A[i] ?? '(missing)')}, template has ${JSON.stringify(B[i] ?? '(missing)')}`;
  return 'differs';
}

// ---- the checks ----------------------------------------------------------------------------
export async function checkSite(siteDirArg = DEFAULT_SITE, { writeShells = false } = {}) {
  const siteDir = path.resolve(siteDirArg);
  const errors = [], warnings = [];
  const err = (m) => errors.push(m);
  const rel = (p) => path.relative(siteDir, p).split(path.sep).join('/');
  const coursePath = path.join(siteDir, 'course.json');
  if (!existsSync(coursePath)) return { errors: ['site/course.json is missing'], warnings, stats: {} };

  // the same validation code the browser runs
  const { validateCourse, flatLessons, isPublished } = await import(pathToFileURL(path.join(siteDir, 'js/core/course.js')).href);
  const { validateLesson } = await import(pathToFileURL(path.join(siteDir, 'js/core/sections.js')).href);

  let course;
  try { course = JSON.parse(readFileSync(coursePath, 'utf8')); } catch (e) { return { errors: [`course.json: not valid JSON (${e.message})`], warnings, stats: {} }; }
  for (const e of validateCourse(course)) err(`course.json: ${e}`);
  const lessons = errors.length ? [] : flatLessons(course);

  for (const l of lessons) {
    const published = isPublished(l);
    const shellPath = path.join(siteDir, 'lessons', l.id, 'index.html');
    const modPath = path.join(siteDir, 'js', 'lessons', `${l.id}.js`);
    const want = shellFor(l);
    if (writeShells) { mkdirSync(path.dirname(shellPath), { recursive: true }); writeFileSync(shellPath, want); }

    if (existsSync(shellPath)) {
      const have = readFileSync(shellPath, 'utf8');
      if (have !== want) err(`lessons/${l.id}/index.html does not equal the shell template (${firstDiff(have, want)}); run: node tools/check_site.mjs --write-shells`);
      const title = /<title>([^<]*)<\/title>/.exec(have)?.[1]?.trim();
      if (!title) err(`lessons/${l.id}/index.html has no page <title>`);
    } else if (published) err(`published lesson "${l.id}" has no shell: lessons/${l.id}/index.html`);
    else warnings.push(`draft lesson "${l.id}" has no shell yet`);

    if (existsSync(modPath)) {
      try {
        const mod = (await import(pathToFileURL(modPath).href)).default;
        for (const e of validateLesson(mod)) err(`js/lessons/${l.id}.js: ${e}`);
        if (mod?.id !== l.id) err(`js/lessons/${l.id}.js: id is "${mod?.id}", course.json says "${l.id}"`);
        if (mod?.title !== l.title) err(`js/lessons/${l.id}.js: title "${mod?.title}" differs from course.json "${l.title}"`);
        if (JSON.stringify(mod?.trace) !== JSON.stringify(l.trace)) err(`js/lessons/${l.id}.js: trace ${JSON.stringify(mod?.trace)} differs from course.json ${JSON.stringify(l.trace)}`);
      } catch (e) { err(`js/lessons/${l.id}.js: cannot be imported (${e.message})`); }
    } else if (published) err(`published lesson "${l.id}" has no module: js/lessons/${l.id}.js`);
    else warnings.push(`draft lesson "${l.id}" has no module yet`);

    for (const f of l.trace) {
      const p = path.join(siteDir, 'data', f);
      if (!existsSync(p)) { (published ? err : (m) => warnings.push(m))(`lesson "${l.id}": trace file data/${f} is missing`); continue; }
      try {
        const j = JSON.parse(readFileSync(p, 'utf8'));
        if (j === null || typeof j !== 'object' || Array.isArray(j)) err(`data/${f}: top level must be an object`);
        else if (j.scenes !== undefined && !Array.isArray(j.scenes)) err(`data/${f}: "scenes" must be an array`);
      } catch (e) { err(`data/${f}: not valid JSON (${e.message})`); }
    }
  }

  // the home page and shared assets
  const homePath = path.join(siteDir, 'index.html');
  if (!existsSync(homePath)) err('index.html (course map) is missing');
  else if (!/<title>[^<]+<\/title>/.test(readFileSync(homePath, 'utf8'))) err('index.html has no page <title>');

  // root-absolute URLs and missing relative assets
  const skipAll = (p) => /[\\/](?:test|data|node_modules)$/.test(p);
  const files = walk(siteDir, skipAll).filter((f) => /\.(?:html|css|js|mjs)$/.test(f));
  let refCount = 0;
  for (const f of files) {
    const inVendor = rel(f).startsWith('vendor/');
    const text = readFileSync(f, 'utf8');
    if (!inVendor) {
      const body = f.endsWith('.css') ? stripCssComments(text) : f.endsWith('.html') ? stripHtmlComments(text) : stripJsComments(text);
      for (const [re, what] of ROOT_ABSOLUTE) {
        re.lastIndex = 0;
        for (const m of body.matchAll(re)) err(`${rel(f)}: ${what}: ${m[0].trim()}`);
      }
    }
    if (inVendor && !f.endsWith('.css')) continue;
    for (const { ref, kind } of referencesIn(f, text)) {
      refCount++;
      const clean = ref.split('#')[0].split('?')[0];
      if (!clean) continue;
      let target = path.resolve(path.dirname(f), clean);
      if (!target.startsWith(siteDir)) { err(`${rel(f)}: ${kind} "${ref}" points outside the site`); continue; }
      if (existsSync(target) && statSync(target).isDirectory()) target = path.join(target, 'index.html');
      if (!existsSync(target)) err(`${rel(f)}: ${kind} "${ref}" does not exist`);
    }
  }

  // modules that no lesson in course.json owns (helper files contain a dot, e.g. backprop-lab.scenes.js)
  const lessonDir = path.join(siteDir, 'js', 'lessons');
  if (existsSync(lessonDir)) for (const n of readdirSync(lessonDir)) {
    const id = n.replace(/\.js$/, '');
    if (n.endsWith('.js') && !id.includes('.') && !lessons.some((l) => l.id === id)) warnings.push(`js/lessons/${n} is not listed in course.json`);
  }

  return { errors, warnings, stats: { lessons: lessons.length, published: lessons.filter(isPublished).length, files: files.length, references: refCount } };
}

// ---- CLI ---------------------------------------------------------------------------------------
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const writeShells = args.includes('--write-shells');
  const dir = args.find((a) => !a.startsWith('--'));
  const siteDir = path.resolve(dir ?? DEFAULT_SITE);
  const { errors, warnings, stats } = await checkSite(siteDir, { writeShells });
  for (const w of warnings) console.log(`warning: ${w}`);
  for (const e of errors) console.error(`ERROR: ${e}`);
  if (errors.length) { console.error(`\ncheck_site: ${errors.length} error(s)`); process.exit(1); }
  console.log(`check_site: ok (${stats.published} published of ${stats.lessons} lessons, ${stats.files} files scanned, ${stats.references} relative references resolved)`);
}
