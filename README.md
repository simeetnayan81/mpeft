# mpeft

mpeft is a lightweight, high-performance C++20 framework designed for PEFT on Apple M-Series SOCs.


## Prerequisites

*   **CMake**: Version 3.20 or higher.
*   **Compiler**: A C++20 compliant compiler (e.g., Clang 10+, GCC 10+, MSVC).
*   **OS**: Linux or macOS (Windows support depends on `mman.h` availability or compatibility layers).

## Build Instructions

1.  **Clone the repository** (if applicable) or navigate to the project root.

2.  **Create a build directory**:
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure the project**:
    ```bash
    mkdir -p external
    git submodule add https://github.com/ggerganov/llama.cpp external/llama.cpp
    cmake ..
    ```

4.  **Compile**:
    ```bash
    make -j$(sysctl -n hw.ncpu)
    ```
    *On Linux, you can use `nproc` instead of `sysctl -n hw.ncpu`.*

## Usage

### Running the Main Application

The main executable (`mpeft`) expects a binary file named `weights.bin` in the working directory to demonstrate memory mapping.

1.  **Generate dummy weights** (if you don't have a file):
    ```bash
    # Creates a 1KB file with random data
    head -c 1024 /dev/urandom > weights.bin
    ```

2.  **Run the executable**:
    ```bash
    ./mpeft
    ```

### Running Tests

The project includes a verification suite for Tensor operations.

*   **Using CTest**:
    ```bash
    ctest --verbose
    ```
*   **Running directly**:
    ```bash
    ./tensor_tests
    ```

## Code Structure

### Core Classes

*   **`Tensor`** (`src/Tensor.cpp`, `include/Tensor.hpp`):
    *   Represents an n-dimensional array view over raw data.
    *   Handles stride computation, row-major indexing, transposing, and reshaping.
    *   **Note**: Transpose and Reshape operations return new Tensor views sharing the same underlying data.

*   **`MemoryMap`** (`src/MemoryMap.cpp`, `include/MemoryMap.hpp`):
    *   RAII wrapper around POSIX `mmap`.
    *   Ensures files are mapped on construction and unmapped on destruction.

*   **`GGUFReader`** (`include/GGUFReader.hpp`):


## Resources & References

*   **GGUF Specification**: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
*   **CMake Documentation**: https://cmake.org