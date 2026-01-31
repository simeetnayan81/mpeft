#include "MemoryMap.hpp"
#include "Tensor.hpp"
#include <iostream>

int main() {
    try {
        
        MemoryMap model("weights.bin");
        std::cout << "Successfully mapped " << model.size() << " bytes.\n";

        // Wrap the first 16 bytes (4 floats) as a 2x2 matrix
        float* raw_ptr = static_cast<float*>(model.data());
        std::span<float> weight_view(raw_ptr, 4);
        
        Tensor weight_matrix(weight_view, {2, 2});
        weight_matrix.print("Initial Weights");

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}