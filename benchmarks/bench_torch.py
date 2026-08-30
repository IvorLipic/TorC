# benchmarks/bench_torch.py
# Micro-benchmarks for the torc-vs-torch comparison in README.md / docs/BENCHMARKS.md.
# Mirrors the shapes used in benchmarks/bench_tensor.cpp. torch is pinned to a single
# thread so the numbers are comparable to torc's single-threaded AVX2 path.
import statistics
import time

import torch

torch.set_num_threads(1)
DEVICE = "cpu"
DTYPE = torch.float32


def time_op(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)  # us
    return statistics.median(samples)


def main():
    results = []

    # Elementwise add: [n]
    n = 16384
    a = torch.ones(n, device=DEVICE, dtype=DTYPE)
    b = torch.ones(n, device=DEVICE, dtype=DTYPE)
    results.append(("Elementwise add", f"[{n}]", time_op(lambda: a + b, 3000)))

    # Matmul 2D: (m,k) @ (k,n)
    for m, k, n in ((64, 64, 64), (128, 128, 128), (256, 256, 256)):
        a = torch.ones(m, k, device=DEVICE, dtype=DTYPE)
        b = torch.ones(k, n, device=DEVICE, dtype=DTYPE)
        results.append((f"Matmul 2D", f"[{m}, {k}, {n}]",
                        time_op(lambda: a @ b, 40 if m >= 128 else 2000)))

    # Batched matmul: (batch, m, k) @ (batch, k, n)
    for batch, m, k, n in ((4, 64, 64, 64), (8, 128, 128, 128)):
        a = torch.ones(batch, m, k, device=DEVICE, dtype=DTYPE)
        b = torch.ones(batch, k, n, device=DEVICE, dtype=DTYPE)
        results.append((f"Batched matmul", f"[{batch}, {m}, {k}, {n}]",
                        time_op(lambda: a @ b, 40 if m >= 128 else 200)))

    # Softmax over a 1-D tensor (matches torc's whole-tensor softmax)
    n = 16384
    a = torch.ones(n, device=DEVICE, dtype=DTYPE)
    results.append(("Softmax", f"[{n}]", time_op(lambda: torch.softmax(a, 0), 2000)))

    # Transpose 2D
    m, n = 256, 256
    a = torch.ones(m, n, device=DEVICE, dtype=DTYPE)
    results.append(("Transpose 2D", f"[{m}, {n}]", time_op(lambda: a.t(), 3000)))

    print(f"torch {torch.__version__} (CPU, single-threaded)")
    print(f"{'Operation':<18}{'Shape':<22}{'Time (us)':>12}")
    for name, shape, us in results:
        print(f"{name:<18}{shape:<22}{us:>12.2f}")


if __name__ == "__main__":
    main()
