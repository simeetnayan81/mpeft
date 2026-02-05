#ifndef GGUF_ENGINE_HPP
#define GGUF_ENGINE_HPP

#include "MemoryMap.hpp"
#include "ggml.h"
#include <string>
#include <memory>
#include "gguf.h"

class GGUFEngine {
public:
    GGUFEngine(const std::string& model_path);
    ~GGUFEngine();

    // The Bridge: Returns a tensor with a valid data pointer pointing to our MemoryMap
    struct ggml_tensor* get_tensor(const std::string& name);

    // Metadata accessors
    size_t tensor_count() const;
    struct ggml_context* context() { return ggml_ctx; }

private:
    std::unique_ptr<MemoryMap> model_mapping;
    struct ggml_context* ggml_ctx = nullptr;
    struct gguf_context* gguf_ctx = nullptr;

    void bridge_pointers();
};

#endif