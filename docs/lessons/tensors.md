# Lesson `tensors` — internal note

Not shipped. Storyboard, library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`Tensor(std::initializer_list<float>, std::vector<int>)`, `Tensor::operator[]` (deduced this, checked), `flat_index`
(`idx = idx * shape[i] + indices[i]`, row-major), `Tensor::reshape` / `view` (copy the storage, `src/tensor.cpp` 517-527),
`broadcast_shape` (`include/torc/utils.hpp`, right-aligned, size 1 repeats, else `ShapeError "Cannot broadcast shapes …"`),
`elementwise_binary_op` (strides 0 for size-1 dimensions), `reduce_axis` (removes the axis, negative axis allowed,
`ShapeError "Invalid axis …"`), `Tensor::sum/mean/max(axis)` and the whole-tensor versions, `Tensor::transpose(axes)`,
`Tensor::matmul` (rank ≥ 2, inner sizes, batch dimensions broadcast).

## Observed (tools/trace/trace_tensors.cpp → site/data/tensors.json)
- `a = [[1, 2, 3], [4, 5, 6]]`: `a[i, j]` equals `data()[3i + j]` for all six positions. Changing a reshaped copy leaves `a` unchanged.
- `a + b`: shapes `(2, 3)`, `(3)`, `(1, 3)`, `(2, 1)`, `(1)` work; `(2)` and `(3, 1)` throw `ShapeError`. Shape-only pairs:
  `(4, 1)+(3) → (4, 3)`, `(4, 1, 3)+(2, 1) → (4, 2, 3)`, `(5, 4)+(4) → (5, 4)`, `(5, 4)+(5)` throws.
- Reductions: `sum(0) = [5, 7, 9]`, `sum(1) = [6, 15]`, `mean`, `max` likewise; `sum(-1) = sum(1)`; `sum(2)` throws; `sum() = 21`.
- `a·aᵀ = [[14, 32], [32, 77]]`; `aᵀ·a` has shape `(3, 3)`; `a·a` throws; `(4, 2, 3)·(3, 2)` has shape `(4, 2, 2)`
  and each batch equals its own 2D product; `(2, 1, 2, 3)·(3, 3, 2)` has shape `(2, 3, 2, 2)`, so both batch shapes broadcast.

## Expected (mathematics)
Broadcasting rule as stated in the lesson; `C_ij = Σ_l A_il B_lj`. Matches every observation.

## Suspected discrepancy
None.

## Observed, added for the v3 appendix rewrite (same program, additive keys)
- `matmul_2x2`: `[[1, 2], [3, 4]]` times `[[5, 6], [7, 8]]` is `[[19, 22], [43, 50]]` (the worked entry 19, the "your turn" entry 50).
- `image`: a made-up colour image of shape `(3, 2, 2)` (channels, rows, columns). `image[c, i, j]` sits at position `(c*2 + i)*2 + j`
  (checked against `data()`), `image[2, 1, 0]` at position 10 with value 3. `sum`, `mean`, `max` along axes 0, 1 and 2 give shapes
  `(2, 2)`, `(3, 2)`, `(3, 2)`; `image.sum(0)` is `[[24, 18], [12, 6]]`.
- `index_errors`: `a[2, 0]` and `a[0, 3]` throw `ShapeError` "Index out of bounds at dim d: k not in [0, n)" (positions count from 0);
  `a[1]` throws `ShapeError` "Rank mismatch: got 1 indices for rank 2".
- `TorC` has no keep-dimension option: `Tensor::sum(int)`, `mean(int)` and `max(int)` have only the axis argument (`include/torc/tensor.hpp`).

## Lesson status
Rewritten 2026-11 as an appendix page (`template: 3`, part `using-torc`, outside the lesson order; brief §6 appendix and §7):
storage and position formula, a 3-D image example, reshape copies, broadcasting rules (predict quiz "(2) + (2, 3)?", broadcast grid),
reductions and axes ("the position of a dimension", axis explorer now with the image and axis 2), one worked 2 x 2 product before the
index formula and a numeric "your turn", batched `matmul`. `gotcha` boxes for reshape, addition and matmul errors.
Widgets: `tensor-widgets.js` (broadcast grid, axis explorer with `reducedIndices`). Tests: `site/test/tensors.test.mjs` (the n-d
reduction helper against every recorded reduction, the worked product, every message quoted). Checked: Node tests, `check_site`,
`tools/prose/check.sh`, Chromium at 1366x768 and 390x844. Not done: human think-aloud, screen reader.
