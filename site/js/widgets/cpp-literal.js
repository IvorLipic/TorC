// A float literal in C++ for a number from a trace. The trace stores a float32 as a double (0.1f is 0.10000000149011612), so the
// literal is the shortest decimal that reads back as the same float32. Pure; no DOM.
export const cf = (v) => {
  if (Number.isInteger(v)) return `${v}.0f`;
  for (let p = 1; p <= 9; p++) {
    const d = Number(v.toPrecision(p));
    if (Math.fround(d) === Math.fround(v)) return `${d}f`;
  }
  return `${v}f`;
};
