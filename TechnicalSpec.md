
# Project mpeft: Technical Specification & Roadmap

## 1. Executive Summary

**mpeft** is a high-performance, zero-copy Parameter-Efficient Fine-Tuning (PEFT) engine designed specifically for **Apple Silicon (M-series) hardware**.

The project aims to enable local, privacy-centric Large Language Model (LLM) personalization by implementing Low-Rank Adaptation (LoRA) directly on the device's unified memory. By bypassing bloated machine learning frameworks (like PyTorch or TensorFlow) in favor of raw C++ and hardware-accelerated kernels, mpeft achieves the lowest possible latency and memory footprint for on-device training.

---

## 2. Problem Statement

Running and fine-tuning LLMs on edge devices (e.g., MacBook Air M3) faces three critical bottlenecks when using standard frameworks:

1. **Memory Bloat:** Frameworks like PyTorch often load multiple copies of model weights and require massive runtime environments (Python interpreter, CUDA abstractions), exceeding the RAM limits of consumer hardware (8GB/16GB).
2. **Latency Overhead:** The layers of abstraction required to make frameworks "general purpose" introduce significant latency, making real-time, interactive personalization feel sluggish.
3. **Inefficient Data Movement:** Standard loading procedures copy weights from Disk  RAM  GPU VRAM. On Unified Memory architectures, this is wasteful redundancy.

### The Solution

mpeft addresses these by:

* **Zero-Copy Loading:** Using `mmap` to map GGUF model files directly from SSD to Virtual Address Space, using the OS page cache as the "RAM".
* **Hardware Intimacy:** utilizing Apple's proprietary **AMX (Apple Matrix Coprocessor)** and **NEON** SIMD instructions for math operations.
* **Quantization-Aware Training:** Training LoRA adapters (F32) on top of quantized base models (Q4_K/Q6_K) without decompressing the full model.

---

## 3. Core Architecture

The system follows a modular "Librarian & Worker" architecture.

### A. The "Librarian" Layer (`GGUFEngine`)

This layer is responsible for data access and metadata management. It does not perform heavy computation.

* **`MemoryMap` (The Landlord):** Manages the `mmap` syscalls to reserve virtual address space corresponding to the `.gguf` file on disk. Ensures the base model remains `PROT_READ` (Read-Only) to prevent corruption.
* **`GGML Integration`:** Uses the `ggml` library to parse the GGUF header, understand quantization formats (Q4_K, Q6_K), and retrieve tensor metadata (dimensions, types).
* **Pointer Bridging:** A mechanism that calculates the absolute memory address of a tensor (`Base Address + Data Offset`) and "plugs" it into the tensor structure, enabling zero-copy access.

### B. The "Brain" Layer (Tensor & Kernels)

This layer handles the math.

* **`Tensor` Class:** A lightweight wrapper around the raw data pointers. It manages the lifecycle of *trainable* weights (LoRA adapters) while pointing to mapped memory for *frozen* weights.
* **Fused Kernels:** Custom C++ functions that leverage SIMD/AMX.
* *Forward Pass:* 
* Instead of dequantizing  entirely, the kernel dequantizes small blocks on-the-fly in CPU registers to perform the dot product.



### C. The Personalization Engine (Logic)

mpeft serves as the underlying engine for a Personalization OS by enabling **On-Device Gradient Descent**:

1. **Data Ingestion:** The system takes user-specific text (e.g., "User's coding style").
2. **LoRA Update:** It performs backpropagation *only* on the tiny  and  matrices (approx. 10-50MB), leaving the 5GB base model untouched.
3. **Dynamic Swapping:** The engine can swap these  adapters in milliseconds, allowing the LLM to switch "personas" (e.g., "Coding Assistant" vs. "Email Helper") instantly based on context.

---

## 4. Master Progress Checklist

### Phase 1: Foundation & Infrastructure (Completed)

* [x] **Memory Mapping:** Implemented `MemoryMap` class for `mmap` operations.
* [x] **GGUF Parsing:** Integrated `ggml` submodule to read GGUF headers.
* [x] **Pointer Bridging:** Implemented logic to map GGUF tensor offsets to virtual memory addresses.
* [x] **Quantization Support:** Verified reading of `Q4_K` and `Q6_K` tensor types.
* [x] **Build System:** configured `CMakeLists.txt` for C/C++ interop, library modularity (`mpeft_core`), and Apple Accelerate linking.
* [x] **Engine Encapsulation:** Refactored logic into `GGUFEngine` class to separate loading from execution.

### Phase 2: The "Brain" & Compute Kernels (Current Focus)

* [ ] **Tensor Class:** Implement a proper C++ `Tensor` class that supports basic operations (view, reshape).
* [ ] **LoRA Initialization:** Implement Xavier/Kaiming initialization for LoRA `A` and `B` matrices.
* [ ] **Forward Pass (Inference):** Implement the `forward()` method for a Linear layer that combines Frozen Quantized weights + Trainable LoRA weights.
* [ ] **Custom Kernels:** Write the NEON/AMX optimized dot-product for `Q4_K`  `F32`.

### Phase 3: Training Logic (Backpropagation)

* [ ] **Autograd Graph:** Implement a simple directed acyclic graph (DAG) to track operations for gradient calculation.
* [ ] **Loss Function:** Implement Cross-Entropy Loss for language modeling.
* [ ] **Backward Pass:** Implement `backward()` to compute gradients for `A` and `B` matrices.
* [ ] **Optimizer:** Implement a basic AdamW or SGD optimizer to update the weights.

### Phase 4: Production & Personalization

* [ ] **Adapter Serialization:** Save/Load just the LoRA adapters (`.bin` files) to disk.
* [ ] **Context Window:** Implement a rolling context window for training on user data streams.
* [ ] **Benchmarks:** Measure tokens/sec training speed vs. PyTorch on M3 Air.

---

## 5. Development Guidelines & conventions

* **Language:** C++20 (for `std::span`, concepts).
* **Memory Safety:** Use `std::unique_ptr` for ownership. Never `malloc` manually unless interacting with raw C APIs.
* **Interop:** All GGML includes must be wrapped in `extern "C"`.
* **No Copies:** Always prefer `std::span` or pointers over `std::vector` copy construction when handling model weights.

## 6. Current Status

**Date:** February 05, 2026
**State:** Phase 1 Complete. Entering Phase 2.
**Next Immediate Task:** Implement the `Tensor` class and the LoRA initialization logic.