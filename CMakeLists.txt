cmake_minimum_required(VERSION 3.16)
project(torc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(USE_OPENMP "Enable OpenMP parallelization" OFF)

add_library(torc
    src/tensor.cpp
    src/autograd.cpp
    src/nn.cpp
    src/nn/linear.cpp
    src/nn/activations.cpp
    src/nn/losses.cpp
    src/optim.cpp
    src/data.cpp
)

target_include_directories(torc
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(torc PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(torc PRIVATE "$<$<CONFIG:Release>:/O2;/GL;/arch:AVX2>")
else()
    target_compile_options(torc PRIVATE "$<$<CONFIG:Release>:-O3;-march=native;-funroll-loops>")
endif()

add_executable(linear_regression_example examples/linear_regression/linear_regression.cpp)
target_include_directories(linear_regression_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(linear_regression_example PRIVATE torc)

add_executable(mnist_mlp_example examples/mnist_mlp/mnist_mlp.cpp)
target_include_directories(mnist_mlp_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mnist_mlp_example PRIVATE torc)

# =============================
# Unit tests
# =============================
enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(torc_tests
    tests/test_tensor.cpp
    tests/test_autograd.cpp
    tests/test_nn.cpp
    tests/test_data.cpp
)

target_include_directories(torc_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(torc_tests PRIVATE torc GTest::gtest_main)
add_test(NAME TorcTests COMMAND torc_tests)

# =============================
# Benchmarks (optional)
# =============================
if(BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.0
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    add_executable(torc_benchmarks
        benchmarks/bench_tensor.cpp
    )

    target_include_directories(torc_benchmarks
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(torc_benchmarks PRIVATE torc benchmark::benchmark)
endif()
