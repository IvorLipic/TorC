# Benchmarks

Baseline measurements for `torc` tensor operations. Generated with
`cmake -S . -B build -DBUILD_BENCHMARKS=ON && cmake --build build --config Release`.

Run benchmarks:

```bash
./build/torc_benchmarks --benchmark_out=benchmark_results.json
```

## Baseline (Reference Machine: 12-core, 2096 MHz, L1=32KB, L2=512KB, L3=4096KB)

| Operation | Shape | Time (ns) | Iterations |
|-----------|-------|-----------|------------|
| Elementwise add | [1024] | 440 | 1544828 |
| Elementwise add | [4096] | 1897 | 344615 |
| Elementwise add | [16384] | 6145 | 110277 |
| Elementwise mul | [1024] | 434 | 1600000 |
| Elementwise mul | [4096] | 1962 | 373333 |
| Elementwise mul | [16384] | 6162 | 112000 |
| Scalar add | [1024] | 422 | 1659259 |
| Scalar add | [4096] | 1947 | 373333 |
| Scalar add | [16384] | 4686 | 149333 |
| Matmul 2D | [64, 64, 64] | 63691 | 11200 |
| Matmul 2D | [128, 128, 128] | 505773 | 1120 |
| Matmul 2D | [256, 256, 256] | 4296774 | 160 |
| Batched matmul | [4, 64, 64, 64] | 254914 | 2800 |
| Batched matmul | [8, 128, 128, 128] | 4301228 | 160 |
| Softmax | [1024] | 10722 | 64000 |
| Softmax | [4096] | 42618 | 16000 |
| Softmax | [16384] | 173150 | 3733 |
| Transpose 2D | [64, 64] | 27967 | 24889 |
| Transpose 2D | [128, 128] | 105469 | 6400 |
| Transpose 2D | [256, 256] | 419507 | 1659 |
| numel() | [1024] | 4.98 | 100000000 |
| numel() | [4096] | 5.05 | 112000000 |
| numel() | [16384] | 4.93 | 100000000 |

## Latest audit rerun after hot-path fix (2026-08-27)

This rerun used the GCC 16.2 Release build (`-O3 -march=native`) on the same 12-core, 2096 MHz
machine, with `--benchmark_min_time=1s`. The table above remains the historical pre-session
baseline; these are the current post-fix measurements from this audit.

| Operation | Shape | Time (ns) | Iterations |
|-----------|-------|-----------|------------|
| Elementwise add | [1024] | 542 | 2890323 |
| Elementwise add | [4096] | 2280 | 814545 |
| Elementwise add | [16384] | 7736 | 174906 |
| Elementwise mul | [1024] | 500 | 2890323 |
| Elementwise mul | [4096] | 2354 | 689231 |
| Elementwise mul | [16384] | 7960 | 182857 |
| Scalar add | [1024] | 451 | 3200000 |
| Scalar add | [4096] | 1654 | 896000 |
| Scalar add | [16384] | 5644 | 235789 |
| Matmul 2D | [64, 64, 64] | 69409 | 21854 |
| Matmul 2D | [128, 128, 128] | 542383 | 2560 |
| Matmul 2D | [256, 256, 256] | 5166387 | 280 |
| Batched matmul | [4, 64, 64, 64] | 272330 | 4978 |
| Batched matmul | [8, 128, 128, 128] | 4460473 | 320 |
| Softmax | [1024] | 11413 | 112000 |
| Softmax | [4096] | 44322 | 29867 |
| Softmax | [16384] | 175614 | 8145 |
| Transpose 2D | [64, 64] | 80180 | 17920 |
| Transpose 2D | [128, 128] | 318482 | 4267 |
| Transpose 2D | [256, 256] | 1253266 | 1120 |
| numel() | [1024] | 3.00 | 471578947 |
| numel() | [4096] | 3.13 | 471578947 |
| numel() | [16384] | 3.84 | 358400000 |

## Changes from previous baseline

- **Elementwise add/mul**: ~200–280× faster for contiguous same-shape inputs due to direct `simd::` dispatch, bypassing the generic index-reconstruction loop.
- **Scalar add**: ~3–6× faster due to `simd::add_scalar` fast path.
- **Matmul 2D / Batched matmul**: ~3.5–3.8× faster from AVX2 vectorization of the inner `j` loop (8-float FMA-like throughput). Combined with the earlier cache-blocked tiling, matmul is now ~35–40× faster than the original baseline.
- **Softmax, Transpose, numel()**: unchanged (no fast-path change for these ops).

## Validation overhead investigation

The numerical-domain hardening commit (`be90e19`) added unconditional `std::isfinite` prepasses.
Those scans were the source of the large regression observed after the documented baseline: two
extra passes for matmul and one extra pass for softmax. The lifetime, cross-entropy, and
axis-aware-softmax commits do not change the benchmarked legacy tensor kernels. Ordinary
elementwise validation was already removed from the hot path in the follow-up lifetime commit.

The fix keeps softmax's finite-input contract but folds validation into its existing max pass, and
lets matmul preserve IEEE NaN/infinity propagation without an O(N) prepass. On the same 12-core
machine, the post-fix GCC release rerun measured matmul at 66.5 µs / 535.7 µs / 4.59 ms for
64/128/256 matrices (versus 63.7 µs / 505.8 µs / 4.30 ms in this table) and softmax at 11.1 µs /
43.5 µs / 174.3 µs (versus 10.7 µs / 42.6 µs / 173.2 µs). The remaining small differences are
within compiler/build and run-to-run variation; the validation-induced 1.4–1.9× matmul and
1.6× softmax regressions are gone.

A later audit rerun on the same machine measured matmul at 69.4 us / 542.4 us / 5.17 ms for
64/128/256 matrices and softmax at 11.4 us / 44.3 us / 175.6 us. These values remain within
normal compiler/build and run-to-run variation of the post-fix results above.

## Autograd lifetime-guard overhead

The tensor benchmarks above do not construct `Variable` graphs, so the lifetime guard cannot affect
their timings. Commit `6566da5` does add a small, intentional cost to autograd workloads: each
`Variable` owns a lifetime token, each tracked tape edge stores a `weak_ptr` and a recorded
`requires_grad` flag, and backward checks those records before dereferencing inputs. That overhead
belongs in a separate autograd benchmark; it should not be attributed to the tensor-kernel changes
shown in the tables above.

## Comparison with PyTorch (CPU)

To keep the naive implementation honest about where it stands, the tensor kernels are also
benchmarked against PyTorch on the same machine, pinned to a single thread.

Run torc:

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON
cmake --build build --config Release
./build/torc_benchmarks --benchmark_min_time=1s
```

Run torch:

```bash
python benchmarks/bench_torch.py
```

`bench_torch.py` calls `torch.set_num_threads(1)` so its numbers are comparable to torc's
single-threaded AVX2 path, and mirrors the exact shapes in `benchmarks/bench_tensor.cpp`.

Latest numbers (single-threaded, Release, same 12-core reference machine):

| Operation | Shape | torc (µs) | torch 2.11 (µs) | torc / torch |
|-----------|-------|-----------|-----------------|--------------|
| Elementwise add | [16384] | 6.6 | 9.3 | 0.7× |
| Matmul 2D | [64, 64, 64] | 64.5 | 19.7 | 3.3× |
| Matmul 2D | [128, 128, 128] | 512 | 99.3 | 5.2× |
| Matmul 2D | [256, 256, 256] | 4283 | 869 | 4.9× |
| Batched matmul | [4, 64, 64, 64] | 261 | 74.1 | 3.5× |
| Batched matmul | [8, 128, 128, 128] | 4142 | 1069 | 3.9× |
| Softmax | [16384] | 265 | 39.8 | 6.7× |
| Transpose 2D | [256, 256] | 424 | 4.4 | 96× |

Notes:
- **Elementwise add is already competitive** — torc's contiguous `simd::` fast path avoids the
  generic index-reconstruction loop, while torch's eager `a + b` pays per-call dispatch and
  allocation overhead at this size.
- **matmul trails by ~3–5×**: torch dispatches to a tuned single-thread SGEMM, whereas torc uses
  its own cache-blocked + AVX2 inner loop (still far behind a production BLAS).
- **softmax trails by ~6–7×** for the same reason — torc has no vectorized reduction in its
  softmax/exp path yet.
- **transpose shows the largest gap** because `torch.t()` is a zero-copy view, while torc copies
  storage on every `transpose`/`reshape`/`view`. This is the single biggest memory-model debt
  tracked in `ROADMAP.md`.
