cmake_minimum_required(VERSION 3.20)
project(mpeft LANGUAGES CXX)

# Use C++20 for std::span and other modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Hardware Optimization ---
if(APPLE)
    add_compile_options(-O3 -mcpu=native -ffast-math -Wall)
endif()

# Include directories so we can write #include "Tensor.hpp" instead of #include "../include/Tensor.hpp"
include_directories(include)
# --- Source Groups ---
set(SOURCES 
    src/Tensor.cpp 
    src/MemoryMap.cpp 
    src/GGUFReader.cpp
)
# --- Target 1: The Main Application ---
add_executable(mpeft 
    src/main.cpp 
    ${SOURCES}
)

if(APPLE)
    target_link_libraries(mpeft PRIVATE "-framework Foundation" "-framework Accelerate")
endif()

# --- Target 2: The Verification Suite ---
add_executable(tensor_tests 
    tests/tensor_test.cpp 
    ${SOURCES}
)

# Provide a helper to run tests via 'ctest' or just 'make test'
enable_testing()
add_test(NAME CoreTensorTests COMMAND tensor_tests)