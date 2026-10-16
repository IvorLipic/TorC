# Lesson `tensors` — internal note

Not shipped. Storyboard, library symbols, and what is observed, expected and suspected.

## Library symbols (TorC v0.1.0)
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
- `a·aᵀ = [[14, 32], [32, 77]]`; `aᵀ·a` has shape `(3, 3)`; `a·a` throws; `(4, 2, 3)·(3, 2)` has shape `(4, 2, 2)`.

## Expected (mathematics)
Broadcasting rule as stated in the lesson; `C_ij = Σ_l A_il B_lj`. Matches every observation.

## Suspected discrepancy
None.

## Lesson status
Published 2026-10-16. Widgets: `tensor-widgets.js` (broadcast grid, axis explorer), both showing recorded values only.
Tests: `site/test/tensors.test.mjs`. Checked: Node tests, `check_site`, `tools/prose/check.sh` (0 hard), Chromium at 1366×768 and
390×844 (widgets driven, no console errors, no horizontal overflow), axe-core clean. Not done: independent review, screen reader.
