#include "torc/tensor.hpp"
#include <benchmark/benchmark.h>
#include <vector>

using namespace torc;

static void BM_ElementwiseAdd(benchmark::State& state) {
    int n = state.range(0);
    Tensor a(std::vector<int>{n});
    Tensor b(std::vector<int>{n});
    a.fill(1.0f);
    b.fill(2.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(a.add(b));
}
BENCHMARK(BM_ElementwiseAdd)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_ElementwiseMul(benchmark::State& state) {
    int n = state.range(0);
    Tensor a(std::vector<int>{n});
    Tensor b(std::vector<int>{n});
    a.fill(1.0f);
    b.fill(2.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(a.mul(b));
}
BENCHMARK(BM_ElementwiseMul)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_ScalarAdd(benchmark::State& state) {
    int n = state.range(0);
    Tensor a(std::vector<int>{n});
    a.fill(1.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(a.add(2.0f));
}
BENCHMARK(BM_ScalarAdd)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_Matmul2D(benchmark::State& state) {
    int m = state.range(0);
    int k = state.range(1);
    int n = state.range(2);
    Tensor a(std::vector<int>{m, k});
    Tensor b(std::vector<int>{k, n});
    a.fill(1.0f);
    b.fill(2.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(a.matmul(b));
}
BENCHMARK(BM_Matmul2D)->Args({64, 64, 64})->Args({128, 128, 128})->Args({256, 256, 256});

static void BM_BatchedMatmul(benchmark::State& state) {
    int batch = state.range(0);
    int m = state.range(1);
    int k = state.range(2);
    int n = state.range(3);
    Tensor a(std::vector<int>{batch, m, k});
    Tensor b(std::vector<int>{batch, k, n});
    a.fill(1.0f);
    b.fill(2.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(a.matmul(b));
}
BENCHMARK(BM_BatchedMatmul)->Args({4, 64, 64, 64})->Args({8, 128, 128, 128});

static void BM_Softmax(benchmark::State& state) {
    int n = state.range(0);
    Tensor a(std::vector<int>{n});
    a.fill(1.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(a.softmax());
}
BENCHMARK(BM_Softmax)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_Transpose2D(benchmark::State& state) {
    int m = state.range(0);
    int n = state.range(1);
    Tensor a(std::vector<int>{m, n});
    a.fill(1.0f);
    for (auto _ : state)
        benchmark::DoNotOptimize(a.transpose({}));
}
BENCHMARK(BM_Transpose2D)->Args({64, 64})->Args({128, 128})->Args({256, 256});

static void BM_Numel(benchmark::State& state) {
    int n = state.range(0);
    Tensor a(std::vector<int>{n});
    for (auto _ : state)
        benchmark::DoNotOptimize(a.numel());
}
BENCHMARK(BM_Numel)->Arg(1024)->Arg(4096)->Arg(16384);

BENCHMARK_MAIN();
