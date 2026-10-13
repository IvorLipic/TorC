# Consume torc via FetchContent. TORC_BUILD_TESTS/TORC_BUILD_EXAMPLES default to OFF when torc is
# not the top-level project, so googletest is never fetched. Run from the torc checkout:
#   cmake -S examples/consume/fetchcontent -B consume-build
#   cmake --build consume-build && ./consume-build/app
#
# In a real project, replace SOURCE_DIR with a pinned tag:
#   FetchContent_Declare(torc
#       GIT_REPOSITORY https://github.com/IvorLipic/TorC.git
#       GIT_TAG        v0.1.0)
cmake_minimum_required(VERSION 3.20)
project(torc_consume_fetchcontent LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(torc SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
FetchContent_MakeAvailable(torc)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE torc::torc)
