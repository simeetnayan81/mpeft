cmake_minimum_required(VERSION 3.20)
project(mpeft LANGUAGES CXX C)

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
set(GGML_BACKTRACE_LLDB ON CACHE BOOL "Enable LLDB Backtrace" FORCE)
# --- External Dependencies ---
add_subdirectory(external/ggml)

# --- Hardware Optimization ---
if(APPLE)
    add_compile_options(-O3 -mcpu=native -ffast-math -Wall)
endif()

# --- Core Library (Shared Code) ---
add_library(mpeft_core STATIC
    src/Tensor.cpp 
    src/MemoryMap.cpp 
    src/GGUFEngine.cpp
)

# PUBLIC makes this include path available to consumers of mpeft_core.
target_include_directories(mpeft_core PUBLIC include)

# Link the core library against its dependencies.
# PUBLIC propagates these dependencies to any target that links against mpeft_core.
target_link_libraries(mpeft_core PUBLIC ggml)
if(APPLE)
    target_link_libraries(mpeft_core PUBLIC "-framework Foundation" "-framework Accelerate")
endif()

# --- Executable Targets ---
add_executable(mpeft src/main.cpp)
target_link_libraries(mpeft PRIVATE mpeft_core)

add_executable(tensor_tests tests/tensor_test.cpp)
target_link_libraries(tensor_tests PRIVATE mpeft_core)

# Provide a helper to run tests via 'ctest' or just 'make test'
enable_testing()
add_test(NAME CoreTensorTests COMMAND tensor_tests)