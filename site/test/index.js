// `node --test site/test/` on Node 22+ resolves a directory argument to this file (Node 20 finds the *.test.mjs files
// itself, so this file does nothing there). It simply loads every test file so that the documented command works everywhere.
import { readdirSync } from 'node:fs';

if (Number(process.versions.node.split('.')[0]) >= 21) {
  for (const f of readdirSync(new URL('.', import.meta.url)).filter((n) => n.endsWith('.test.mjs')).sort()) await import(new URL(f, import.meta.url));
}
