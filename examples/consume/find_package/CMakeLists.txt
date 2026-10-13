# Consume an installed torc (run from the torc checkout):
#   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/torc-install
#   cmake --build build && cmake --install build
#   cmake -S examples/consume/find_package -B consume-build -DCMAKE_PREFIX_PATH=/path/to/torc-install
#   cmake --build consume-build && ./consume-build/app
cmake_minimum_required(VERSION 3.20)
project(torc_consume_find_package LANGUAGES CXX)

find_package(torc 0.1 CONFIG REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE torc::torc)
