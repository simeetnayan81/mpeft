#include <iostream>
#include <vector>
#include <string>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "MemoryMap.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }

    const std::string fname = argv[1];

     MemoryMap model_file(argv[1]);

    struct ggml_context * my_ctx = nullptr;

    // 1. Initialize GGML context for metadata
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &my_ctx, 
    };

    struct gguf_context * g_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!g_ctx) {
        std::cerr << "Error: failed to load GGUF file\n";
        return 1;
    }

    // 2. Print Model Stats
    int n_tensors = gguf_get_n_tensors(g_ctx);
    int n_kv      = gguf_get_n_kv(g_ctx);

    std::cout << "GGUF Version: " << gguf_get_version(g_ctx) << "\n";
    std::cout << "Tensor Count: " << n_tensors << "\n";
    std::cout << "KV Pair Count: " << n_kv << "\n";

    // 3. Extract Metadata (Example: Model Architecture)
    int arch_id = gguf_find_key(g_ctx, "general.architecture");
    if (arch_id != -1) {
        std::cout << "Architecture: " << gguf_get_val_str(g_ctx, arch_id) << "\n";
    }

    // 4. List Tensors and their info
    std::cout << "\nFirst 10 Tensors:\n";
    for (int i = 0; i < std::min(n_tensors, 10); ++i) {
        const char * name = gguf_get_tensor_name(g_ctx, i);
        struct ggml_tensor * t = ggml_get_tensor(my_ctx, name);
        void* actual_data_address = (uint8_t*)model_file.data() + (size_t)t->data;
        
        // ggml_type_name provides the string for Q4_K, F32, etc.
        std::cout << " - " << name << " | Type: " << ggml_type_name(t->type) 
                  << " | Shape: [" << t->ne[0] << ", " << t->ne[1] << "]" 
                  << " at address: " << actual_data_address << "\n";
    }

    // 5. Cleanup
    gguf_free(g_ctx);
    // Note: gguf_init_from_file with no_alloc=true manages its own context cleanup
    
    return 0;
}