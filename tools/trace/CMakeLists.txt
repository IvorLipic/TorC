# Builds the trace programs that run the REAL TorC library and write the JSON the site is built from.
# TorC is not vendored on this branch: it is fetched at the exact tag/commit pinned in TORC_REF.
cmake_minimum_required(VERSION 3.24)
project(torc_trace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/TORC_REF" TORC_REF_LINES REGEX "^[^#]")
list(GET TORC_REF_LINES 0 TORC_REF)
set(TORC_REPOSITORY "https://github.com/IvorLipic/TorC.git" CACHE STRING "TorC git repository")
message(STATUS "Building traces against TorC @ ${TORC_REF}")

# Consumers should not build the library's tests/examples (options exist from TorC 0.1.0 on;
# older commits ignore them).
set(TORC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TORC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# Portable scalar/baseline code so the output does not depend on the CPU of whoever (or whatever CI runner) generates it.
set(TORC_NATIVE_ARCH OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(torc GIT_REPOSITORY ${TORC_REPOSITORY} GIT_TAG ${TORC_REF})
FetchContent_MakeAvailable(torc)

add_executable(autograd_trace autograd_trace.cpp)
target_link_libraries(autograd_trace PRIVATE torc)
