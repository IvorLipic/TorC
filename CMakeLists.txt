cmake_minimum_required(VERSION 3.16)
project(torc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(torc
    src/tensor.cpp
)

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

# =============================
# Unit tests
# =============================
enable_testing()

add_executable(torc_tests
    tests/test_tensor.cpp
    src/tensor.cpp
)

target_include_directories(torc_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        
)

target_link_libraries(torc_tests PRIVATE torc)
add_test(NAME TorcTests COMMAND torc_tests)
