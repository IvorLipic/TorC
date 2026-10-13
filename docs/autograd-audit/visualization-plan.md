# Visualization plan

Each scene maps to real symbols and a generated, finite-difference-checked output
(`visualizations/generated/autograd_trace.json`, produced by `visualizations/examples/autograd_trace.cpp`).

| Scene | Class | Code symbols | Expected math | Mismatch detector |
|---|---|---|---|---|
| `reuse` | `ReuseScene` | `mul`, `add`, `Variable::backward_with_grad` accumulation (`accumulate_grad`, `grad_map`) | y = x² + x, dy/dx = 2x+1 = 7 at x=3 | FD check in JSON (`ok`); `x.grad==7` hand value |
| `chain` | `ChainScene` | `mul`, `add`, `relu`, `sum` | f = Σ relu(ab+c): da=b·[z>0], db=a·[z>0], dc=[z>0] → a:[2,2,0] b:[1,−2,0] c:[1,1,0] | FD check; hand values |
| `mlp` | `MlpScene` | `matmul`, `add` (broadcast, `reduce_sum_to_shape`), `relu`, `sub`, `mul`, `mean` | L=(o−t)², dL/do=2(o−t)=0.2; dW2=aᵀ·0.2=[0.22,0]; dh=[0.2,0] (second unit dead) → dW1=[0.2,0,0.4,0] | FD check; hand values |

Excluded on purpose (see contract): `slice` (D1), negative-axis reductions (D2), `nn::Linear` intermediates (private cache).
Scenes draw the root's gradient as a *seed*, because the library never stores it (D4).
