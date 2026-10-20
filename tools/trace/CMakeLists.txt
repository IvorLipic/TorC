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
# Re-run CMake when the pin changes so the ref baked into the JSON envelope is never stale.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/TORC_REF")
set(TORC_REPOSITORY "https://github.com/IvorLipic/TorC.git" CACHE STRING "TorC git repository")
message(STATUS "Building traces against TorC @ ${TORC_REF}")

# Consumers should not build the library's tests/examples (options exist from TorC 0.1.0 on;
# older commits ignore them).
set(TORC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TORC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# Portable scalar/baseline code so the output does not depend on the CPU of whoever (or whatever CI runner) generates it.
set(TORC_NATIVE_ARCH OFF CACHE BOOL "" FORCE)
# No fused multiply-add contraction, in TorC and in the trace programs: clang contracts a*b + c by default wherever the target
# has FMA (every arm64 machine, x86 with -march flags), which changes float results in the last bit (for example -4 + 0.1*40 is
# then not exactly 0). add_compile_options applies to the TorC targets too, because FetchContent adds them below.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-ffp-contract=off)
  set(TORC_TRACE_CONTRACT "-ffp-contract=off")
endif()

include(FetchContent)
FetchContent_Declare(torc GIT_REPOSITORY ${TORC_REPOSITORY} GIT_TAG ${TORC_REF})
FetchContent_MakeAvailable(torc)

# What the JSON envelope records as library.flags (informational; compare.mjs does not compare it).
string(TOUPPER "${CMAKE_BUILD_TYPE}" TORC_TRACE_BT)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${TORC_TRACE_BT}} -std=c++23 ${TORC_TRACE_CONTRACT} TORC_NATIVE_ARCH=OFF" TORC_TRACE_FLAGS)

# One executable per trace program; each writes its JSON to stdout (tools/trace/run.sh maps it to site/data/<file>.json).
set(TRACE_TARGETS autograd_trace trace_why trace_optim trace_linear trace_tensors trace_nonlinear trace_modules trace_losses trace_training trace_optimizers trace_generalization trace_mnist_mlp)
foreach(t IN LISTS TRACE_TARGETS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp")
    add_executable(${t} ${t}.cpp trace_util.hpp)
    target_link_libraries(${t} PRIVATE torc)
    target_compile_definitions(${t} PRIVATE "TORC_TRACE_REF=\"${TORC_REF}\"" "TORC_TRACE_FLAGS=\"${TORC_TRACE_FLAGS}\"")
  endif()
endforeach()
