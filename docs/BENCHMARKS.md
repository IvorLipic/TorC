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
| Elementwise add | [1024] | 106225 | 6400 |
| Elementwise add | [4096] | 421216 | 1659 |
| Elementwise add | [16384] | 1653484 | 407 |
| Elementwise mul | [1024] | 104790 | 6400 |
| Elementwise mul | [4096] | 415781 | 1659 |
| Elementwise mul | [16384] | 1692108 | 407 |
| Scalar add | [1024] | 1902 | 344615 |
| Scalar add | [4096] | 7172 | 74667 |
| Scalar add | [16384] | 26912 | 24889 |
| Matmul 2D | [64, 64, 64] | 247566 | 2800 |
| Matmul 2D | [128, 128, 128] | 1981341 | 345 |
| Matmul 2D | [256, 256, 256] | 16030398 | 45 |
| Batched matmul | [4, 64, 64, 64] | 992722 | 747 |
| Batched matmul | [8, 128, 128, 128] | 15980716 | 45 |
| Softmax | [1024] | 10713 | 64000 |
| Softmax | [4096] | 42684 | 16593 |
| Softmax | [16384] | 167806 | 4073 |
| Transpose 2D | [64, 64] | 28580 | 24889 |
| Transpose 2D | [128, 128] | 105479 | 6400 |
| Transpose 2D | [256, 256] | 420162 | 1659 |
| numel() | [1024] | 5.44 | 100000000 |
| numel() | [4096] | 5.45 | 100000000 |
| numel() | [16384] | 5.43 | 112000000 |
