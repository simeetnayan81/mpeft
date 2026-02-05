#include "GGUFEngine.hpp"
#include <stdexcept>
#include <iostream>

GGUFEngine::GGUFEngine(const std::string& model_path) {
    // 1. Map the file into memory (Landlord)
    model_mapping = std::make_unique<MemoryMap>(model_path);

    // 2. Initialize GGUF context (Librarian)
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ggml_ctx,
    };

    gguf_ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!gguf_ctx) throw std::runtime_error("Failed to load GGUF file");

    // 3. Connect the metadata to the actual mapped memory
    bridge_pointers();
}

void GGUFEngine::bridge_pointers() {
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* t = ggml_get_tensor(ggml_ctx, name);
        
        // Map relative GGUF offset to absolute Virtual Address
        // Address = Base of Map + Data Section Start + Tensor's internal offset
        t->data = (uint8_t*)model_mapping->data() + data_offset + (size_t)t->data;
    }
}

struct ggml_tensor* GGUFEngine::get_tensor(const std::string& name) {
    return ggml_get_tensor(ggml_ctx, name.c_str());
}

size_t GGUFEngine::tensor_count() const {
    return gguf_get_n_tensors(gguf_ctx);
}

GGUFEngine::~GGUFEngine() {
    if (gguf_ctx) gguf_free(gguf_ctx);
    if (ggml_ctx) ggml_free(ggml_ctx);
}