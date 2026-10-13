# Visualization plan

The scenes are now rendered as an interactive site (`site/`), not videos; the plan below still describes what each scene demonstrates.
The MLP scene uses a batch of two examples (so bias broadcasting and its sum-reduction are visible); its hand-checked values live in
`visualizations/generated/autograd_trace.json` and are re-verified by `site/test/engine.test.mjs`.

Each scene maps to real symbols and a generated, finite-difference-checked output
(`visualizations/generated/autograd_trace.json`, produced by `visualizations/examples/autograd_trace.cpp`).

| Scene | Class | Code symbols | Expected math | Mismatch detector |
|---|---|---|---|---|
| `reuse` | `ReuseScene` | `mul`, `add`, `Variable::backward_with_grad` accumulation (`accumulate_grad`, `grad_map`) | y = x² + x, dy/dx = 2x+1 = 7 at x=3 | FD check in JSON (`ok`); `x.grad==7` hand value |
| `chain` | `ChainScene` | `mul`, `add`, `relu`, `sum` | f = Σ relu(ab+c): da=b·[z>0], db=a·[z>0], dc=[z>0] → a:[2,2,0] b:[1,−2,0] c:[1,1,0] | FD check; hand values |
| `mlp` | `MlpScene` (legacy video) / site scene `mlp` | `matmul`, `add` (broadcast, `reduce_sum_to_shape`), `relu`, `sub`, `mul`, `mean` | L = mean((o−t)²), dL/do = 2(o−t)/2 per row, dW2 = aᵀ·dL/do, dh = dL/da ⊙ [h>0], db1 = column sums of dh (batch of 2) | FD check in JSON; independent NumPy oracle (reviewer); engine test |

Excluded on purpose (see contract): `slice` (D1), negative-axis reductions (D2), `nn::Linear` intermediates (private cache).
Scenes draw the root's gradient as a *seed*, because the library never stores it (D4).
