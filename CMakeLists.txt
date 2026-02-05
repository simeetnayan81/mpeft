cmake_minimum_required(VERSION 3.20)
project(mpeft LANGUAGES CXX)

# Use C++20 for std::span and other modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# This will automatically configure GGML for Apple Silicon (Accelerate/Metal)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "GGML Examples" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "GGML Tests" FORCE)
set(GGML_METAL          ON  CACHE BOOL "Enable Metal" FORCE)

# --- External Dependencies ---
add_subdirectory(external/ggml)

# --- Hardware Optimization ---
if(APPLE)
    add_compile_options(-O3 -mcpu=native -ffast-math -Wall)
endif()

# Include directories so we can write #include "Tensor.hpp" instead of #include "../include/Tensor.hpp"
include_directories(include)
include_directories(external/ggml/include)
include_directories(external/ggml/src)

# --- Source Groups ---
set(SOURCES 
    src/Tensor.cpp 
    src/MemoryMap.cpp 
)
# --- Target 1: The Main Application ---
add_executable(mpeft 
    src/main.cpp 
    ${SOURCES}
)

target_link_libraries(mpeft PRIVATE ggml)
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