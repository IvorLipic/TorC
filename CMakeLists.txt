cmake_minimum_required(VERSION 3.16)
project(torc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TORC_USE_BLAS "Use CBLAS for matmul (requires CBLAS library)" OFF)

if(TORC_USE_BLAS)
    include(CheckLibraryExists)

    # Try to find OpenBLAS first (common on Linux), then generic BLAS (may provide CBLAS), then Accelerate on macOS
    find_package(OpenBLAS QUIET)
    if(OpenBLAS_FOUND)
        set(BLAS_FOUND TRUE)
        set(BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    else()
        find_package(BLAS QUIET)
    endif()

    if(NOT BLAS_FOUND AND APPLE)
        find_library(ACCELERATE_BLAS Accelerate)
        if(ACCELERATE_BLAS)
            set(BLAS_LIBRARIES ${ACCELERATE_BLAS})
            set(BLAS_FOUND TRUE)
        endif()
    endif()

    if(NOT BLAS_FOUND)
        message(FATAL_ERROR "TORC_USE_BLAS=ON but no BLAS/CBLAS library found. Install OpenBLAS, MKL, or use Accelerate on macOS.")
    endif()

    # Verify CBLAS is available (cblas_sgemm)
    check_library_exists("${BLAS_LIBRARIES}" cblas_sgemm "" HAVE_CBLAS_SGEMM)
    if(NOT HAVE_CBLAS_SGEMM)
        message(FATAL_ERROR "TORC_USE_BLAS=ON but cblas_sgemm not found in BLAS library. Ensure CBLAS is installed.")
    endif()
endif()

add_library(torc
    src/tensor.cpp
    src/autograd.cpp
)

if(TORC_USE_BLAS)
    target_sources(torc PRIVATE src/matmul_blas.cpp)
    target_link_libraries(torc PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(torc PUBLIC TORC_USE_BLAS)
endif()

target_include_directories(torc
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(torc_demo src/main.cpp)
target_link_libraries(torc_demo PRIVATE torc)

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
)

target_include_directories(torc_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(torc_tests PRIVATE torc GTest::gtest_main)
add_test(NAME TorcTests COMMAND torc_tests)