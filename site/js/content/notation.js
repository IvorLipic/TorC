// content/notation.js — the notation guide (brief §8, R9): every symbol the lessons use, with the lesson that introduces it.
// The appendix page `notation` lists it, sym(key, html) in core/blocks.js links a symbol to its entry, and the v3 tests check that
// no lesson uses a symbol before its introducing lesson, in course order. `match.latex` are substrings searched in the LaTeX of the
// formulas; `match.text` are substrings searched in the prose (outside code). Keep each match specific, so that it finds the
// symbol and nothing else. Only sample entries live here for now; the content agent adds the rest.
//   { key, symbol, name, read, meaning, lesson, match: { latex: [...], text: [...] } }
export const NOTATION = [
  {
    key: 'arrow',
    symbol: '←',
    name: 'left arrow',
    read: 'becomes',
    meaning: 'The name on the left gets the new value on the right. The step w ← w − lr·g changes w.',
    lesson: 'why-gradients',
    match: { latex: ['\\leftarrow'], text: ['←'] },
  },
  {
    key: 'sum',
    symbol: 'Σ',
    name: 'capital sigma',
    read: 'the sum of',
    meaning: 'Add all the terms that follow, for each value of the index under the sigma.',
    lesson: 'linear-layer',
    match: { latex: ['\\sum', '\\Sigma'], text: ['Σ', '∑'] },
  },
  {
    key: 'mu',
    symbol: 'μ',
    name: 'mu',
    read: 'mu',
    meaning: 'μ is the momentum factor. It sets how much of the previous step the optimizer keeps.',
    lesson: 'optimizers',
    match: { latex: ['\\mu'], text: ['μ'] },
  },
];

export const notationByKey = (key) => NOTATION.find((n) => n.key === key) ?? null;
