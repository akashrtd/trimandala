cmake_minimum_required(VERSION 3.15...3.27)
project(trimandala_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED CONFIG)

# Optimize for the local machine (AVX2/NEON)
if(NOT MSVC)
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# Define the bindings module
pybind11_add_module(core MODULE "src/trimandala/core/bindings.cpp")

# Link dependencies (just standard lib for now)
# target_link_libraries(core PRIVATE ...)

install(TARGETS core DESTINATION trimandala)
