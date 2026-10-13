cmake_minimum_required(VERSION 3.20)  # 3.20 is the first release that knows cxx_std_23
project(torc VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

# True when torc is the root project, false when pulled in via add_subdirectory()/FetchContent.
# (PROJECT_IS_TOP_LEVEL needs CMake 3.21, so use the portable idiom.)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TORC_IS_TOP_LEVEL ON)
else()
    set(TORC_IS_TOP_LEVEL OFF)
endif()

if(TORC_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
endif()

option(TORC_BUILD_TESTS "Build the GoogleTest suite (fetches googletest)" ${TORC_IS_TOP_LEVEL})
option(TORC_BUILD_EXAMPLES "Build the example executables" ${TORC_IS_TOP_LEVEL})
option(TORC_INSTALL "Generate install and package-config rules" ${TORC_IS_TOP_LEVEL})
option(TORC_NATIVE_ARCH "Release builds of the library use -march=native (/arch:AVX2 on MSVC)" ${TORC_IS_TOP_LEVEL})
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(USE_OPENMP "Enable OpenMP parallelization" OFF)

add_library(torc
    src/tensor.cpp
    src/autograd.cpp
    src/nn.cpp
    src/nn/conv.cpp
    src/nn/linear.cpp
    src/nn/activations.cpp
    src/nn/losses.cpp
    src/optim.cpp
    src/data.cpp
)
add_library(torc::torc ALIAS torc)

target_include_directories(torc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(torc PUBLIC cxx_std_23)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(torc PRIVATE OpenMP::OpenMP_CXX)
endif()

# Release optimization flags. The ISA flag (-march=native, /arch:AVX2 on MSVC) is what makes the
# binary CPU-specific and what enables the AVX2 paths in src/simd_ops.hpp and Tensor::matmul
# (they key off the compiler-defined __AVX2__ macro); TORC_NATIVE_ARCH=OFF drops only that flag.
if(MSVC)
    target_compile_options(torc PRIVATE "$<$<CONFIG:Release>:/O2;/GL>")
    if(TORC_NATIVE_ARCH)
        target_compile_options(torc PRIVATE "$<$<CONFIG:Release>:/arch:AVX2>")
    endif()
else()
    target_compile_options(torc PRIVATE "$<$<CONFIG:Release>:-O3;-funroll-loops>")
    if(TORC_NATIVE_ARCH)
        target_compile_options(torc PRIVATE "$<$<CONFIG:Release>:-march=native>")
    endif()
endif()

# =============================
# Examples
# =============================
if(TORC_BUILD_EXAMPLES)
    add_executable(linear_regression_example examples/linear_regression/linear_regression.cpp)
    target_include_directories(linear_regression_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(linear_regression_example PRIVATE torc)

    add_executable(mnist_mlp_example examples/mnist_mlp/mnist_mlp.cpp)
    target_include_directories(mnist_mlp_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(mnist_mlp_example PRIVATE torc)

    add_executable(mnist_cnn_example examples/mnist_cnn/mnist_cnn.cpp)
    target_include_directories(mnist_cnn_example PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(mnist_cnn_example PRIVATE torc)
endif()

# =============================
# Unit tests
# =============================
if(TORC_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)  # keep googletest out of `cmake --install`
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
endif()

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

# =============================
# Install / package config
# =============================
if(TORC_INSTALL)
    include(CMakePackageConfigHelpers)

    set(TORC_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/torc)

    install(TARGETS torc
        EXPORT torcTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    # Public headers only (src/simd_ops.hpp is an implementation detail). Keeps include/torc/nn/*.
    install(DIRECTORY include/torc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT torcTargets
        NAMESPACE torc::
        DESTINATION ${TORC_INSTALL_CMAKEDIR}
    )

    configure_package_config_file(
        cmake/torcConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/torcConfig.cmake
        INSTALL_DESTINATION ${TORC_INSTALL_CMAKEDIR}
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/torcConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/torcConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/torcConfigVersion.cmake
        DESTINATION ${TORC_INSTALL_CMAKEDIR}
    )
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
