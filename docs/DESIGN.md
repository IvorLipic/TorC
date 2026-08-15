# torc — Design Notes

This file holds decisions that are expensive to reverse and the rationale behind them,
plus the planned future project structure once ML work (autograd → nn → optim → data)
starts. Update this *before* writing code that implements a new architectural layer, not
after — the point is to force the decision to be explicit.

---

## Decisions made so far

### dtype strategy
Single dtype, hardcoded `float32`. Staying single-dtype is the simplest option and matches
the "naive-but-correct" goal through Milestone 3. Revisit with a `dtype` enum or templated
storage (`Tensor<T>`) only if multi-precision becomes an actual need — don't add it
speculatively, since it touches every op signature.

### Shape / broadcasting semantics
**No broadcasting for now.** All elementwise ops require identical shapes and share one
`check_same_shape()` guard. NumPy-style broadcasting is deferred to Milestone 2, at which
point the shape-check signature generalizes to a `broadcast_shape()` helper. Doing this
before autograd matters, because broadcasting changes what a gradient's "sum over broadcast
dims" step needs to look like.

### Memory ownership
`storage_` is an owned `std::vector<float>` per `Tensor`, copied fresh on every elementwise or
reduction op. Shape-changing ops (`reshape`/`view`) move `storage_` instead of copying, so they
are O(1) with respect to data size. Fine for a naive reference implementation. **This will need
to change for autograd** — see below.

### Error handling convention
`TorcError` (base, derives `std::runtime_error`) and `ShapeError` (shape mismatches) live in
`utils.hpp`. All error sites use `ShapeError`; existing `catch (std::runtime_error&)` still
works since it's a base class.

---

## Open decision: autograd graph representation (Milestone 4)

Two realistic options for a from-scratch C++ implementation:

- **Tape-based (Wengert list)**: a flat, append-only log of ops executed during the forward
  pass, walked in reverse for `backward()`. Simpler to implement, easier to reason about,
  matches how early PyTorch-style eager autograd works. Recommended starting point given the
  "naive-but-correct" philosophy — it's the lower-complexity option and doesn't require
  building a graph-node type hierarchy up front.
- **Expression-tree / graph-node based**: each `Variable` holds pointers to its parent nodes
  and a `backward_fn` closure; `backward()` recurses (or does a topological sort) over the
  graph. More flexible (supports things like graph reuse or lazy evaluation later), but more
  machinery to get right (topological ordering, avoiding double-counting shared subgraphs).

**Leaning**: tape-based first. It's enough to support the Milestone 5 goal (basic ML on toy
datasets) without over-building. If graph reuse or more advanced scheduling becomes
necessary later, the tape can be replaced without changing the public `Variable` API much.

This forces the ownership question above: `Variable` will need to hold a
`std::shared_ptr<Tensor>` (or similar) rather than a raw owned `Tensor`, since a node in the
graph is referenced both by the forward value and by whatever downstream ops consumed it.
Decide and record the concrete storage type here before implementing `Variable`.

---

## Planned future project structure

The current layout is intentionally flat because there's only a `Tensor` class. Once
autograd lands (Milestone 4) and nn/optim/data follow (Milestone 5), flat `src/`/`include/`
stops scaling and `main.cpp` stops being a reasonable place for examples. Proposed layout,
to be adopted incrementally as each milestone actually lands (don't scaffold empty
directories ahead of the code that fills them):

```
torc/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── AGENTS.md
├── ROADMAP.md
├── docs/
│   └── DESIGN.md
├── include/
│   └── torc/
│       ├── tensor.hpp
│       ├── utils.hpp
│       ├── autograd.hpp      # Variable, Function/Node, backward()      [Milestone 4]
│       ├── nn.hpp            # Module base + Linear, activations        [Milestone 5]
│       ├── optim.hpp         # SGD, Adam                                [Milestone 5]
│       └── data.hpp          # Dataset, DataLoader                      [Milestone 5]
├── src/
│   ├── tensor.cpp
│   ├── autograd.cpp                                                    [Milestone 4]
│   ├── nn/
│   │   ├── linear.cpp                                                  [Milestone 5]
│   │   └── activations.cpp                                             [Milestone 5]
│   ├── optim.cpp                                                       [Milestone 5]
│   └── data.cpp                                                        [Milestone 5]
├── examples/                  # demo binaries move out of src/, main.cpp retired
│   ├── basic_ops.cpp          # today's main.cpp, relocated
│   ├── linear_regression.cpp                                           [Milestone 5]
│   └── mlp_classification.cpp                                          [Milestone 5]
└── tests/
    ├── test_tensor.cpp
    ├── test_autograd.cpp                                               [Milestone 4]
    ├── test_nn.cpp                                                     [Milestone 5]
    └── test_optim.cpp                                                  [Milestone 5]
```

Rationale for the specific moves:

- **`nn/` as a subdirectory, not a single file, once it has ≥2 layer types.** A single
  `nn.cpp` is fine for `Linear` alone but won't stay readable once activations, losses, and
  more layers land in the same milestone.
- **`examples/` instead of a single `main.cpp`.** AGENTS.md already says `main.cpp` should
  stay a thin demo, not a dumping ground — once there are two real end-to-end examples
  (linear regression, MLP classification) that stops being true of a single file. Multiple
  small example binaries keep each one focused and testable in isolation via CMake.
- **`data.hpp`/`data.cpp` are deliberately minimal at first** — a `Dataset` interface plus
  one synthetic or small CSV-backed implementation is enough to unblock the Milestone 5
  end-to-end examples. Don't build a general data pipeline before there's a second dataset
  that needs one.
- **CMakeLists.txt will need `add_subdirectory` or explicit source lists per target** once
  `nn/` exists as a folder, plus `BUILD_EXAMPLES`/`BUILD_TESTS` options if the example count
  grows enough that not everyone wants to compile all of them by default.

This structure is a target, not a mandate to create now — build it milestone-by-milestone so
each directory only exists once something real lives in it.
