cmake_minimum_required(VERSION 3.20)
project(mpeft LANGUAGES CXX)

# Use C++20 for std::span and other modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Hardware Optimization ---
if(APPLE)
    # -mcpu=native: Tells the compiler to use your specific M1/M2/M3/M4 features
    # -ffast-math: Speeds up tensor calculations by relaxing some floating-point constraints
    add_compile_options(-O3 -mcpu=native -ffast-math -Wall -Wextra)
endif()

# Include directories so we can write #include "Tensor.hpp" instead of #include "../include/Tensor.hpp"
include_directories(include)

# --- Source Groups ---
set(CORE_SOURCES 
    src/Tensor.cpp 
    src/MemoryMap.cpp
)

# --- Target 1: The Main Application ---
add_executable(mpeft 
    src/main.cpp 
    ${CORE_SOURCES}
)

# --- Target 2: The Verification Suite ---
add_executable(tensor_tests 
    tests/tensor_test.cpp 
    ${CORE_SOURCES}
)

# Provide a helper to run tests via 'ctest' or just 'make test'
enable_testing()
add_test(NAME CoreTensorTests COMMAND tensor_tests)