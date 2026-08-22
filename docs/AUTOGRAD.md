# torc Autograd — Design Notes (Tape-Based)

*This document explains the autograd system. It mirrors decisions in docs/DESIGN.md.*

---

## Why Tape-Based (Wengert List)?

Two main approaches exist for automatic differentiation:

| Approach | How it works | Complexity |
|----------|--------------|------------|
| **Tape-based** (Wengert list) | Record every op during forward pass in a flat list. Walk list in reverse for backward. | Low |
| **Expression-tree** | Each Variable points to parent nodes. Backward does topological sort + recursion. | High |

**We chose tape-based** because:
1. **Simplicity** — ~200 lines for core engine, no node hierarchy
2. **Sufficient for torc's scope** — toy models, small datasets (Milestone 5)
3. **Matches PyTorch eager** — same mental model
4. **Evolvable** — can swap to expression-tree later without changing Variable API

---

## Core Concepts

### Variable
```cpp
Variable x(Tensor({2.0f}), true);  // data=2.0, requires_grad=true
// x.data()  -> Tensor (forward value)
// x.grad()  -> Tensor (gradient, same shape as data)
// x.requires_grad() -> bool
// x.backward() -> computes gradients
```

### Tape (per-Variable)
Each Variable carries its own tape. During forward, ops append a `TapeEntry` to the
**output** Variable's tape. Each entry stores:
- A `backward` closure that knows how to compute gradients for its inputs
- Pointers to the input Variables (for graph traversal)

```cpp
struct TapeEntry {
    std::vector<Variable*> inputs;  // input Variables (for topological sort)
    std::function<void(const Tensor& grad_output, std::vector<Tensor>& input_grads)> backward;
};
```

### Backward Algorithm: Topological Sort (Not Recursion)

**Critical design decision:** The backward pass uses **topological sort + iterative execution**,
not recursive depth-first traversal. This matches how PyTorch, TinyTorch (Harvard), and all
production autograd engines work.

**Why not recursion?**
- Recursion stacks overflow on deep graphs (1000+ layer networks)
- Recursion double-visits shared subgraphs (DAGs, not just trees)
- Recursion makes gradient accumulation order-dependent

**The algorithm:**
1. **DFS post-order traversal** from the output Variable, following `TapeEntry.inputs` pointers
2. **Reverse** the resulting list to get backward execution order
3. **Iterate** through reversed list, executing each Variable's tape entries once
4. Gradients accumulate via `+=` in a map keyed by `Variable*`

```cpp
void Variable::backward_with_grad(const Tensor& upstream_grad) {
    // 1. Build topological order (DFS post-order)
    std::vector<Variable*> topo = build_topo();
    std::reverse(topo.begin(), topo.end());  // backward order

    // 2. Gradient accumulation map
    std::unordered_map<Variable*, Tensor> grad_map;
    grad_map[this] = upstream_grad;

    // 3. Execute backward in topological order
    for (Variable* v : topo) {
        if (v->tape_.empty()) continue;
        
        auto it = grad_map.find(v);
        if (it == grad_map.end()) continue;  // no gradient for this node
        
        const Tensor& grad = it->second;
        Tensor current_grad = grad;
        
        // Walk tape entries in reverse (most recent op first)
        for (auto eit = v->tape_.rbegin(); eit != v->tape_.rend(); ++eit) {
            const TapeEntry& entry = *eit;
            std::vector<Tensor> input_grads(entry.inputs.size());
            entry.backward(current_grad, input_grads);
            
            // Accumulate gradients into input Variables
            for (size_t i = 0; i < entry.inputs.size(); ++i) {
                if (entry.inputs[i]->requires_grad_) {
                    accumulate_grad(entry.inputs[i], input_grads[i]);
                    grad_map[entry.inputs[i]] = input_grads[i];
                }
            }
            
            current_grad = input_grads.back();  // pass to next tape entry
        }
    }
    
    tape_.clear();
}
```

**Why this is correct:**
- Each Variable is processed exactly once (no double-visiting)
- Gradients are computed in dependency order (all downstream grads ready before upstream needs them)
- Shared subgraphs naturally accumulate gradients via `grad_map`
- Stack usage is O(V) for the DFS, not O(V) nested calls

**Sources:**
- PyTorch dev mailing list: "Simplified Introduction to PyTorch's Autograd" (zdevito, 2021)
- PyTorch autograd source: `torch/csrc/autograd/engine.cpp` (topological_sort + iterative execution)
- TinyTorch Module 06 (Harvard): recursive for educational simplicity, but notes production uses topological sort
- Red Hat Developer: "Optimize PyTorch training with the autograd engine" (2026) — explicit C++ pseudocode

---

## Forward to Backward Flow

```
# User code
x = Variable(Tensor({2.0}), true)
y = Variable(Tensor({3.0}), true)
z = x * y          # Forward: 2 * 3 = 6
z.backward()       # Backward: dz/dx = 3, dz/dy = 2
```

**Forward pass** (x * y):
1. Tensor::mul computes 6.0
2. Variable::mul wraps result in new Variable
3. Creates TapeEntry:
   - inputs = {&x, &y}
   - backward = [](grad_out, grads) { grads[0] = grad_out * y; grads[1] = grad_out * x; }
4. Appends entry to z.tape_
5. Returns Variable(z=6.0, tape_=[...])

**Backward pass** (z.backward()):
1. Build topological order: [x, y, z] (post-order DFS)
2. Reverse: [z, y, x]
3. Process z: execute tape entry, get grad_x=3, grad_y=2
4. Accumulate into x.grad_ and y.grad_
5. Process x and y: empty tapes, no-op
6. Done

---

## Broadcasting Gradients (Critical)

**Forward**: x (3,) + y (1,) to z (3,) — y broadcast to (3,)
**Backward**: grad_out (3,) to x.grad (3,), y.grad (1,) — must **sum** over broadcast dim

```cpp
// Helper: sum gradient to match target shape
Tensor reduce_sum_to_shape(const Tensor& grad, const std::vector<int>& target_shape) {
    Tensor result = grad;
    // 1. Add leading dims if needed
    while (result.shape().size() < target_shape.size())
        result = result.unsqueeze(0);
    // 2. Sum over dims where target=1 but grad>1
    for (int i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && result.shape()[i] > 1)
            result = result.sum(i, true);  // keepdim
    }
    return result;
}

// In add_backward:
for (size_t i = 0; i < inputs.size(); ++i) {
    Tensor input_grad = local_grads[i];
    if (inputs[i]->shape_ != output_shape)  // broadcast happened
        input_grad = reduce_sum_to_shape(input_grad, inputs[i]->shape_);
    accumulate_grad(inputs[i], input_grad);
}
```

---

## Gradient Functions for Key Ops

### Elementwise: z = x * y
```
dz/dx = y,  dz/dy = x
input_grads[0] = grad_output * y
input_grads[1] = grad_output * x
```

### Matmul: z = x @ y  (x: m×k, y: k×n)
```
dz/dx = grad_output @ y.T    (m×n @ n×k = m×k)
dz/dy = x.T @ grad_output    (k×m @ m×n = k×n)
```

### Sum reduction: z = x.sum(axis=0)  (x: m×n to z: n,)
```
dz/dx = broadcast(grad_output, x.shape)  // repeat grad_output m times
```

### Transpose: z = x.T
```
dz/dx = grad_output.T  (self-inverse)
```

### Reshape/View: z = x.reshape(...)
```
dz/dx = grad_output.reshape(x.shape)  (metadata-only)
```

---

## requires_grad Semantics

```cpp
Variable a(Tensor({1.0}), true);   // tracked
Variable b(Tensor({2.0}), false);  // NOT tracked
Variable c = a + b;                // c.requires_grad = true (a requires it)
c.backward();                      // a.grad populated, b.grad NOT allocated
```

- **Default**: `requires_grad = false` (inference-friendly)
- **Propagation**: If any input requires_grad, output `requires_grad = true`
- **No tape recorded** for non-requires-grad inputs to avoid gradient allocation

---

## Gradient Accumulation

```cpp
loss1.backward()  # W.grad += dL1/dW
loss2.backward()  # W.grad += dL2/dW  (accumulates!)
W.zero_grad()     # Reset to zero before next step
```

Matches PyTorch behavior. Call `zero_grad()` at start of each training step.

---

## Gradient Checking (Testing)

Every differentiable op must have a gradient check test:

```cpp
// Numerical gradient (central difference)
float h = 1e-4f;  // for float32
float numerical_grad = (f(x + h) - f(x - h)) / (2 * h);

// Analytical gradient from autograd
float analytical_grad = x.grad().data()[i];

// Compare
EXPECT_NEAR(numerical_grad, analytical_grad, 1e-4f);
```

Run with ctest — catches bugs in backward implementations.

**Tolerance rationale**: float32 has ~7 decimal digits. Central difference error is O(h²) = O(1e-8)
for h=1e-4, well within float32 precision. The 1e-4 tolerance gives a comfortable margin while
catching actual bugs.

---

## Extending with New Operations

To add gradient for a new Tensor op:

1. **Add Tensor op** (if not exists): `Tensor::my_op()`
2. **Add Variable wrapper** in `autograd.hpp`:
   ```cpp
   Variable my_op() const {
       Tensor result = data().my_op();
       if (!requires_grad_) return Variable(result, false);
       
       // Create tape entry
       TapeEntry entry;
       entry.inputs = { /* input Variable pointers */ };
       entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
           // Compute gradients w.r.t each input
           // Use reduce_sum_to_shape for broadcasting
       };
       
       Variable out(result, true);
       out.tape_.push_back(entry);
       return out;
   }
   ```
3. **Write backward_fn** in `autograd.cpp`
4. **Add gradient check test** in `test_autograd.cpp`

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Forgetting reduce_sum_to_shape for broadcast ops | Always use helper when input shapes differ |
| In-place modification of Variable.data() | Don't do it — breaks gradient computation |
| Double backward without zero_grad() | Expected accumulation; call zero_grad() each step |
| backward() on non-scalar | Call .sum().backward() or provide grad_output |
| Recursive backward on deep graphs | Use topological sort (current implementation does this) |
| Circular references with shared_ptr | Variable owns data directly; tape entries hold raw pointers |

---

## Future Evolution

| Feature | When | Approach |
|---------|------|----------|
| Gradient checkpointing | Milestone 6 | checkpoint(fn, *args) — recompute forward in backward |
| Higher-order derivatives | Later | Nested tapes (Engine per order) |
| Graph optimization/compilation | Not planned | Would need expression-tree |
| Distributed autograd | Not planned | Out of scope |

---

## References

- docs/DESIGN.md — Architectural decisions
- ROADMAP.md — Milestone 4 checklist
- PyTorch autograd internals: https://pytorch.org/docs/stable/notes/autograd.html
- PyTorch dev mailing list: "Simplified Introduction to PyTorch's Autograd" (zdevito, 2021)
- TinyTorch Module 06 (Harvard): https://mlsysbook.ai/tinytorch/modules/06_autograd.html
- Red Hat Developer: "Optimize PyTorch training with the autograd engine" (2026)
- Micrograd (educational): https://github.com/karpathy/micrograd
