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
| Elementwise add | [1024] | 434 | 1544828 |
| Elementwise add | [4096] | 1917 | 373333 |
| Elementwise add | [16384] | 6014 | 112000 |
| Elementwise mul | [1024] | 414 | 1659259 |
| Elementwise mul | [4096] | 1935 | 373333 |
| Elementwise mul | [16384] | 5971 | 112000 |
| Scalar add | [1024] | 410 | 1659259 |
| Scalar add | [4096] | 2129 | 344615 |
| Scalar add | [16384] | 4692 | 139378 |
| Matmul 2D | [64, 64, 64] | 235294 | 2800 |
| Matmul 2D | [128, 128, 128] | 1873818 | 373 |
| Matmul 2D | [256, 256, 256] | 15645502 | 45 |
| Batched matmul | [4, 64, 64, 64] | 940662 | 640 |
| Batched matmul | [8, 128, 128, 128] | 15195260 | 50 |
| Softmax | [1024] | 10843 | 64000 |
| Softmax | [4096] | 42275 | 16593 |
| Softmax | [16384] | 167021 | 4073 |
| Transpose 2D | [64, 64] | 27728 | 24889 |
| Transpose 2D | [128, 128] | 104891 | 6400 |
| Transpose 2D | [256, 256] | 421246 | 1723 |
| numel() | [1024] | 4.91 | 100000000 |
| numel() | [4096] | 5.12 | 112000000 |
| numel() | [16384] | 4.95 | 100000000 |

## Changes from previous baseline

- **Elementwise add/mul**: ~200–280× faster for contiguous same-shape inputs due to direct `simd::` dispatch, bypassing the generic index-reconstruction loop.
- **Scalar add**: ~3–6× faster due to `simd::add_scalar` fast path.
- **Matmul 2D / Batched matmul**: ~2–5% faster from removing the sparsity early-exit branch in the inner loop.
- **Softmax, Transpose, numel()**: unchanged (no fast-path change for these ops).
