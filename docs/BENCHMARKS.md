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

## Latest rerun after hot-path fix (2026-08-27)

This rerun used the GCC 16.2 Release build (`-O3 -march=native`) on the same 12-core, 2096 MHz
machine, with `--benchmark_min_time=1s`. The table above remains the historical pre-session
baseline; these are the current post-fix measurements.

| Operation | Shape | Time (ns) | Iterations |
|-----------|-------|-----------|------------|
| Elementwise add | [1024] | 486 | 2890323 |
| Elementwise add | [4096] | 2155 | 689231 |
| Elementwise add | [16384] | 7482 | 182045 |
| Elementwise mul | [1024] | 486 | 2890323 |
| Elementwise mul | [4096] | 1591 | 995556 |
| Elementwise mul | [16384] | 7498 | 186667 |
| Scalar add | [1024] | 459 | 3089655 |
| Scalar add | [4096] | 1602 | 1120000 |
| Scalar add | [16384] | 5332 | 256000 |
| Matmul 2D | [64, 64, 64] | 64291 | 22400 |
| Matmul 2D | [128, 128, 128] | 541508 | 2635 |
| Matmul 2D | [256, 256, 256] | 4480598 | 309 |
| Batched matmul | [4, 64, 64, 64] | 272917 | 5271 |
| Batched matmul | [8, 128, 128, 128] | 4049734 | 345 |
| Softmax | [1024] | 11175 | 112000 |
| Softmax | [4096] | 44512 | 32000 |
| Softmax | [16384] | 180523 | 8145 |
| Transpose 2D | [64, 64] | 80264 | 17569 |
| Transpose 2D | [128, 128] | 320137 | 4267 |
| Transpose 2D | [256, 256] | 1260250 | 1120 |
| numel() | [1024] | 3.40 | 471578947 |
| numel() | [4096] | 2.96 | 471578947 |
| numel() | [16384] | 3.12 | 471578947 |

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

## Autograd lifetime-guard overhead

The tensor benchmarks above do not construct `Variable` graphs, so the lifetime guard cannot affect
their timings. Commit `6566da5` does add a small, intentional cost to autograd workloads: each
`Variable` owns a lifetime token, each tracked tape edge stores a `weak_ptr` and a recorded
`requires_grad` flag, and backward checks those records before dereferencing inputs. That overhead
belongs in a separate autograd benchmark; it should not be attributed to the tensor-kernel changes
shown in the tables above.
