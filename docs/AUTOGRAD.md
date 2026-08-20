# torc Autograd — Design Notes (Tape-Based)

*This document explains the autograd system for learning/reference. It mirrors decisions in docs/DESIGN.md.*

---

## Why Tape-Based (Wengert List)?

Two main approaches exist for automatic differentiation:

| Approach | How it works | Complexity |
|----------|--------------|------------|
| **Tape-based** (Wengert list) | Record every op during forward pass in a flat list. Walk list in reverse for backward. | Low |
| **Expression-tree** | Each Variable points to parent nodes. Backward does topological sort + recursion. | High |

**We chose tape-based** because:
1. **Simplicity** — ~200 lines for core engine, no node hierarchy
2. **Sufficient for torc'\''s scope** — toy models, small datasets (Milestone 5)
3. **Matches PyTorch eager** — same mental model
4. **Evolvable** — can swap to expression-tree later without changing Variable API

---

## Core Concepts

### Variable
`cpp
Variable x(Tensor({2.0f}), true);  // data=2.0, requires_grad=true
// x.data()  -> Tensor (forward value)
// x.grad()  -> Tensor (gradient, same shape as data)
// x.requires_grad() -> bool
// x.backward() -> computes gradients
`

### Engine (Global Tape)
`cpp
// Singleton that holds the tape
Engine::instance().record(grad_fn);   // called during forward
Engine::instance().backward(output);  // called by Variable::backward()
Engine::instance().clear_tape();      // between training steps
`

### GradFn (Internal)
Each recorded operation creates a GradFn:
`cpp
struct GradFn {
    std::vector<std::shared_ptr<Tensor>> saved_tensors;  // inputs needed for backward
    std::function<void(const Tensor&, std::vector<Tensor>&)> backward_fn;  // grad computation
    std::string op_name;  // debugging
};
`

---

## Forward to Backward Flow

`
# User code
x = Variable(Tensor({2.0}), true)
y = Variable(Tensor({3.0}), true)
z = x * y          # Forward: 2 * 3 = 6
z.backward()       # Backward: dz/dx = 3, dz/dy = 2
`

**Forward pass** (x * y):
1. Tensor::mul computes 6.0
2. Variable::mul wraps result in new Variable
3. Creates GradFn:
   - saved_tensors = {x.data, y.data} (shared_ptrs)
   - ackward_fn = [](grad_out, grads) { grads[0] += grad_out * y; grads[1] += grad_out * x; }
4. Engine::record(grad_fn) — appends to tape
5. Returns Variable(z=6.0, grad_fn=...)

**Backward pass** (z.backward()):
1. Engine walks tape **in reverse order**
2. For each GradFn: calls ackward_fn(grad_out, grads)
3. grads[i] accumulated into input_variable[i].grad_
4. Initial grad_out for scalar output = 1.0

---

## Broadcasting Gradients (Critical)

**Forward**: x (3,) + y (1,) to z (3,) — y broadcast to (3,)
**Backward**: grad_out (3,) to x.grad (3,), y.grad (1,) — must **sum** over broadcast dim

`cpp
// Helper: sum gradient to match target shape
Tensor sum_to_shape(const Tensor& grad, const std::vector<int>& target_shape) {
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
grads[0] += sum_to_shape(grad_out, x.shape());
grads[1] += sum_to_shape(grad_out, y.shape());
`

---

## Gradient Functions for Key Ops

### Elementwise: z = x * y
`
dz/dx = y,  dz/dy = x
grads[0] += grad_out * y
grads[1] += grad_out * x
`

### Matmul: z = x @ y  (x: m×k, y: k×n)
`
dz/dx = grad_out @ y.T    (m×n @ n×k = m×k)
dz/dy = x.T @ grad_out    (k×m @ m×n = k×n)
`

### Sum reduction: z = x.sum(axis=0)  (x: m×n to z: n,)
`
dz/dx = broadcast(grad_out, x.shape)  // repeat grad_out m times
`

### Transpose: z = x.T
`
dz/dx = grad_out.T  (self-inverse)
`

### Reshape/View: z = x.reshape(...)
`
dz/dx = grad_out.reshape(x.shape)  (metadata-only)
`

---

## requires_grad Semantics

`cpp
Variable a(Tensor({1.0}), true);   // tracked
Variable b(Tensor({2.0}), false);  // NOT tracked
Variable c = a + b;                // c.requires_grad = true (a requires it)
c.backward();                      // a.grad populated, b.grad NOT allocated
`

- **Default**: equires_grad = false (inference-friendly)
- **Propagation**: If any input equires_grad, output equires_grad = true
- **No grad_fn recorded** for non-requires-grad inputs to no gradient allocation

---

## Gradient Accumulation

`cpp
loss1.backward()  # W.grad += dL1/dW
loss2.backward()  # W.grad += dL2/dW  (accumulates!)
W.zero_grad()     # Reset to zero before next step
`

Matches PyTorch behavior. Call zero_grad() at start of each training step.

---

## Gradient Checking (Testing)

Every differentiable op must have a gradient check test:

`cpp
// Numerical gradient (central difference)
float numerical_grad = (f(x + eps) - f(x - eps)) / (2 * eps);
// Analytical gradient from autograd
float analytical_grad = x.grad().data()[i];
// Compare
EXPECT_NEAR(numerical_grad, analytical_grad, 1e-3);
`

Run with ctest — catches bugs in backward implementations.

---

## Extending with New Operations

To add gradient for a new Tensor op:

1. **Add Tensor op** (if not exists): Tensor::my_op()
2. **Add Variable wrapper** in utograd.hpp:
   `cpp
   Variable my_op() const {
       Tensor result = data().my_op();
       if (!requires_grad_) return Variable(result, false);
       auto grad_fn = make_grad_fn(/* saved tensors */, /* backward_fn */);
       Engine::instance().record(grad_fn);
       return Variable(result, true, grad_fn);
   }
   `
3. **Write backward_fn** in utograd.cpp:
   `cpp
   auto my_op_backward = [](const Tensor& grad_out, std::vector<Tensor>& grads) {
       // Compute gradients w.r.t each input
       // Use sum_to_shape for broadcasting
   };
   `
4. **Add gradient check test** in 	est_autograd.cpp

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Forgetting sum_to_shape for broadcast ops | Always use helper when input shapes differ |
| In-place modification of Variable.data() | Don'\''t do it — breaks gradient computation |
| Double backward without zero_grad() | Expected accumulation; call zero_grad() each step |
| ackward() on non-scalar | Call .sum().backward() or provide grad_output |
| Circular references with shared_ptr | GradFn holds shared_ptr<Tensor>, not Variable |

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
- Micrograd (educational): https://github.com/karpathy/micrograd
