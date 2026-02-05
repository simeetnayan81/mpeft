#include "GGUFEngine.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_gguf_file>\n";
        return 1;
    }

    try {
        GGUFEngine engine(argv[1]);
        std::cout << "✅ Engine initialized with " << engine.tensor_count() << " tensors.\n";

        std::cout << "\n--- Verifying Pointer Bridge ---\n";
        std::vector<std::string> test_tensors = {
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.ffn_down.weight",
            "tok_embeddings.weight",
            "output.weight"
        };

        for (const auto& name : test_tensors) {
            auto* tensor = engine.get_tensor(name);
            if (tensor) {
                std::cout << "  ✅ Found `" << tensor->name << "` at address " << tensor->data << "\n";
            } else {
                std::cout << "  ❌ Tensor `" << name << "` not found.\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}