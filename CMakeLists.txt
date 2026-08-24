cmake_minimum_required(VERSION 3.16)
project(torc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

add_executable(torc_demo src/main.cpp)
target_link_libraries(torc_demo PRIVATE torc)

add_executable(linear_regression_example examples/linear_regression/linear_regression.cpp)
target_link_libraries(linear_regression_example PRIVATE torc)

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
