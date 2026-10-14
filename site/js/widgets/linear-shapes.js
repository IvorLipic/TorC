// widgets/linear-shapes.js — pure logic of the `linear-layer` shape explorer (no DOM, no globals, importable in Node).
//
// Rules, as TorC implements them (src/nn/linear.cpp, src/tensor.cpp):
//   Linear(in, out):  weight (out, in), bias (out), forward(x) = add(matmul(x, transpose(weight)), bias)
//   x (B, in) -> y (B, out);  parameters = out*in + out (two tensors).
//   A wrong input width fails inside matmul with   "matmul inner dimensions must match, got (B, w) and (in, out)"
//   (the second shape is the TRANSPOSED weight); a rank-1 input fails with "matmul requires rank >= 2 operands, ...".
// Rank-3 inputs are deliberately not modelled (the library accepts them; their gradients are not verified for the site).
// The predicted message text is checked, in site/test/linear-layer.test.mjs, against every message TorC recorded in linear.json.


// TorC prints shapes like "(2, 2)" and "(3)".
export const fmtShape = (s) => '(' + s.join(', ') + ')';
export const numel = (s) => s.reduce((a, b) => a * b, 1);

export function linearShapes(inF, outF, B) {
  return { x: [B, inF], W: [outF, inF], Wt: [inF, outF], b: [outF], y: [B, outF] };
}

export function paramCount(inF, outF) {
  const weight = inF * outF, bias = outF;
  return { weight, bias, total: weight + bias, tensors: 2 };
}





