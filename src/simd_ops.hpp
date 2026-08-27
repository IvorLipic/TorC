#pragma once
#include <cstddef>
#include <cmath>

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define TORC_HAS_AVX2 1
#else
#define TORC_HAS_AVX2 0
#endif

namespace torc {
namespace simd {

constexpr size_t width = TORC_HAS_AVX2 ? 8 : 1;

inline void add(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

inline void sub(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] - b[i];
}

inline void mul(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

inline void div(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] / b[i];
}

inline void add_scalar(const float* a, float b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    __m256 vb = _mm256_set1_ps(b);
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] + b;
}

inline void sub_scalar(const float* a, float b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    __m256 vb = _mm256_set1_ps(b);
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] - b;
}

inline void mul_scalar(const float* a, float b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    __m256 vb = _mm256_set1_ps(b);
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] * b;
}

inline void div_scalar(const float* a, float b, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    __m256 vb = _mm256_set1_ps(b);
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = a[i] / b;
}

inline void neg(const float* a, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    __m256 vzero = _mm256_setzero_ps();
    for (; i + width <= n; i += width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_sub_ps(vzero, va);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = -a[i];
}

inline void exp(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i < n; ++i) out[i] = std::exp(in[i]);
}

inline void log(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i < n; ++i) out[i] = std::log(in[i]);
}

inline void sqrt(const float* in, float* out, size_t n) {
    size_t i = 0;
#if TORC_HAS_AVX2
    for (; i + width <= n; i += width) {
        __m256 vi = _mm256_loadu_ps(in + i);
        __m256 vr = _mm256_sqrt_ps(vi);
        _mm256_storeu_ps(out + i, vr);
    }
#endif
    for (; i < n; ++i) out[i] = std::sqrt(in[i]);
}

} // namespace simd
} // namespace torc
