// A backslash before ; , ! : or a space inside a JavaScript string is not an escape sequence. JavaScript drops the backslash, so
// '\;' is a bare ';'. KaTeX accepts a bare ';' without error, so the formula x \; W rendered as "x ; W" and no test noticed
// (brief D1). This test scans every site/js/**/*.js file and fails on a single backslash before one of those characters in a
// single-quoted, double-quoted or template string. The legitimate forms pass: '\\;' (an escaped backslash, then ';'), '\n', '\u2717'.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const JS = fileURLToPath(new URL('../js/', import.meta.url));
const BAD = new Set([';', ',', '!', ':', ' ']);
const KEYWORDS_BEFORE_REGEX = new Set(['return', 'typeof', 'case', 'in', 'of', 'void', 'delete', 'throw', 'new', 'else', 'do']);

const files = (dir) => readdirSync(dir, { withFileTypes: true })
  .flatMap((e) => (e.isDirectory() ? files(join(dir, e.name)) : e.name.endsWith('.js') ? [join(dir, e.name)] : []));

// Returns [{line, quote, text}] for every single backslash before a character of BAD inside a string literal.
export function badEscapes(src) {
  const found = [];
  const lineAt = (at) => src.slice(0, at).split('\n').length;
  let i = 0;
  let last = '';                                   // the previous token: a word, a punctuation mark, or '"' after a literal

  const scanString = (quote) => {                  // i is on the opening quote; returns with i after the closing quote
    i++;
    while (i < src.length && src[i] !== quote && !(quote !== '`' && src[i] === '\n')) {
      if (src[i] === '\\') {
        if (BAD.has(src[i + 1])) found.push({ line: lineAt(i), quote, text: src.slice(Math.max(0, i - 12), i + 10).replace(/\n/g, ' ') });
        i += 2;                                    // a backslash and the character it escapes ('\\' is consumed as one pair)
      } else if (quote === '`' && src[i] === '$' && src[i + 1] === '{') {
        i += 2;
        scanCode(1);                               // code inside ${ ... }, up to its closing brace
      } else i++;
    }
    i++;
  };

  const regexAllowed = () => last === '' || (!/^[\w$"]/.test(last) && last !== ')' && last !== ']') || KEYWORDS_BEFORE_REGEX.has(last);

  function scanCode(depth) {                       // scans until the brace that closes `depth` open braces, or to the end when 0
    while (i < src.length) {
      const c = src[i];
      if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; continue; }
      if (c === '/' && src[i + 1] === '*') { const end = src.indexOf('*/', i + 2); i = end < 0 ? src.length : end + 2; continue; }
      if (c === '\'' || c === '"' || c === '`') { scanString(c); last = '"'; continue; }
      if (c === '/' && regexAllowed()) {           // a regular expression literal
        i++;
        let inClass = false;
        while (i < src.length && src[i] !== '\n' && (inClass || src[i] !== '/')) {
          if (src[i] === '\\') i++;
          else if (src[i] === '[') inClass = true;
          else if (src[i] === ']') inClass = false;
          i++;
        }
        i++;
        while (/[a-z]/.test(src[i] ?? '')) i++;    // flags
        last = '"';
        continue;
      }
      if (/[\w$]/.test(c)) { let j = i; while (j < src.length && /[\w$]/.test(src[j])) j++; last = src.slice(i, j); i = j; continue; }
      if (/\s/.test(c)) { i++; continue; }
      if (depth > 0 && c === '{') depth++;
      if (depth > 0 && c === '}') { depth--; if (depth === 0) { i++; last = '}'; return; } }
      last = c;
      i++;
    }
  }

  scanCode(0);
  return found;
}

test('the scanner finds a single backslash before ; , ! : or a space, and accepts the legitimate forms', () => {
  assert.equal(badEscapes(String.raw`const a = '\;';`).length, 1);
  assert.equal(badEscapes(String.raw`const a = "x\, y";`).length, 1);
  assert.equal(badEscapes(String.raw`const a = 'a\!b\:c\ d';`).length, 3);
  assert.equal(badEscapes('const a = `x \\; ${f(\'\\;\')}`;').length, 2, 'a template string and a string inside ${ }');
  assert.equal(badEscapes(String.raw`eq('\underset{x}\;\cdot')`).length, 1, 'the shape of the D1 bug');
  assert.deepEqual(badEscapes(String.raw`const a = '\\;' + "\\," + '\\ ' + '\n' + '\u2717' + '\'' + "\"" + '\\\\;';`), []);
  assert.deepEqual(badEscapes(String.raw`const r = /\;'/; const s = a / b; // it\'s '\;'` + '\n/* "\\;" */'), [], 'regexes and comments are not strings');
});

test('no JavaScript string in site/js has a single backslash before ; , ! : or a space (brief D1)', () => {
  const all = files(JS);
  assert.ok(all.length > 50, `scanned ${all.length} files`);
  const problems = all.flatMap((f) => badEscapes(readFileSync(f, 'utf8')).map((b) => `${f.slice(JS.length)}:${b.line} ${b.quote} ...${b.text}...`));
  assert.deepEqual(problems, []);
});
