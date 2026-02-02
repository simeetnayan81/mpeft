#include "MemoryMap.hpp"
#include "GGUFReader.hpp"
#include "Tensor.hpp"
#include <iostream>
#include <type_traits>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./mpeft <model.gguf>\n";
        return 1;
    }

    try {
        MemoryMap model_file(argv[1]);
        GGUFReader reader(model_file.data(), model_file.size());
        reader.parse();

        std::cout << "Successfully parsed model: " << argv[1] << "\n";
        std::cout << "Found " << reader.tensors().size() << " tensors.\n";
        reader.print_metadata();

        for (size_t i = 0; i < reader.tensors().size(); ++i) {
            std::cout << " - " << reader.tensors()[i].name << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}