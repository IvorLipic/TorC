// ops/index.js — the op registry. ONE module per op holds everything about it:
//   name, title, arity
//   forward(...inputs) -> tensor                      backward(g, inputs, out) -> raw gradient per input (engine sums them to input shape)
//   texForward(n, inputNames) -> LaTeX                texBackward(n, inputNames) -> [LaTeX per input]
//   whyForward / whyBackward -> HTML sentence         forwardNumbers(ctx) / backwardNumbers(ctx): the rule with the real numbers
// engine.js (compute), explain.js (paper work) and the tests are generic consumers of this table.
// To add an op: add a module, import it here, then add a trace scene that uses it (see docs/authoring.md).
import add from './add.js';
import sub from './sub.js';
import mul from './mul.js';
import relu from './relu.js';
import sum from './sum.js';
import mean from './mean.js';
import matmul from './matmul.js';
import transpose from './transpose.js';
import mse from './mse.js';

export const REGISTRY = Object.fromEntries([add, sub, mul, relu, sum, mean, matmul, transpose, mse].map((op) => [op.name, op]));
export const OP_NAMES = Object.keys(REGISTRY);

export function getOp(name) {
  const op = Object.hasOwn(REGISTRY, name) ? REGISTRY[name] : undefined;
  if (!op) throw new Error(`unknown op "${name}" (registered: ${OP_NAMES.join(', ')})`);
  return op;
}
export const opTitle = (name) => (Object.hasOwn(REGISTRY, name) ? REGISTRY[name].title : name);
