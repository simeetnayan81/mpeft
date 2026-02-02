#include "GGUFReader.hpp"
#include <stdexcept>
#include <iostream>
#include <type_traits>

GGUFReader::GGUFReader(const void* data, size_t size) 
    : data_ptr_(static_cast<const uint8_t*>(data)), total_size_(size) {}

template<typename T>
T GGUFReader::read_basic() {
    if (cursor_ + sizeof(T) > total_size_) throw std::runtime_error("GGUF: Unexpected EOF");
    T val = *reinterpret_cast<const T*>(data_ptr_ + cursor_);
    cursor_ += sizeof(T);
    return val;
}

std::string GGUFReader::read_string() {
    uint64_t len = read_basic<uint64_t>();
    if (cursor_ + len > total_size_) throw std::runtime_error("GGUF: String overflow");
    std::string str(reinterpret_cast<const char*>(data_ptr_ + cursor_), len);
    cursor_ += len;
    return str;
}

void GGUFReader::parse() {
    uint32_t magic = read_basic<uint32_t>();
    if (magic != 0x46554747) throw std::runtime_error("Invalid GGUF Magic");
    
    uint32_t version = read_basic<uint32_t>();
    uint64_t tensor_count = read_basic<uint64_t>();
    uint64_t kv_count = read_basic<uint64_t>();

    std::cout << "GGUF Version: " << version << "\n";
    std::cout << "Tensor Count: " << tensor_count << "\n";
    std::cout << "KV Pair Count: " << kv_count << "\n";

    parse_kv_pairs(kv_count);
    parse_tensor_infos(tensor_count);
}

// Helper to determine the byte-width of basic GGUF types
size_t GGUFReader::get_type_size(GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:   case GGUFValueType::INT8:   case GGUFValueType::BOOL:    return 1;
        case GGUFValueType::UINT16:  case GGUFValueType::INT16:                               return 2;
        case GGUFValueType::UINT32:  case GGUFValueType::INT32:  case GGUFValueType::FLOAT32: return 4;
        case GGUFValueType::UINT64:  case GGUFValueType::INT64:  case GGUFValueType::FLOAT64: return 8;
        default: return 0;
    }
}

void GGUFReader::parse_kv_pairs(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        std::string key = read_string();
        GGUFValueType type = static_cast<GGUFValueType>(read_basic<uint32_t>());

        // Each type has a specific byte-width we must consume
        switch (type) {
            case GGUFValueType::UINT32:  metadata_[key] = read_basic<uint32_t>(); break;
            case GGUFValueType::INT32:   metadata_[key] = read_basic<int32_t>();  break;
            case GGUFValueType::FLOAT32: metadata_[key] = read_basic<float>();    break;
            case GGUFValueType::BOOL:    metadata_[key] = read_basic<bool>();     break;
            case GGUFValueType::STRING:  metadata_[key] = read_string();          break;
            case GGUFValueType::UINT64:  metadata_[key] = read_basic<uint64_t>(); break;
            case GGUFValueType::INT64:   metadata_[key] = read_basic<int64_t>();  break;
            
            case GGUFValueType::ARRAY: {
                // Arrays are: [Type (4 bytes)] + [Number of Elements (8 bytes)] + [Data]
                GGUFValueType sub_type = static_cast<GGUFValueType>(read_basic<uint32_t>());
                uint64_t n_elements = read_basic<uint64_t>();
                
                // We must skip the bytes consumed by the array to keep the cursor aligned
                if (sub_type == GGUFValueType::STRING) {
                    for (uint64_t j = 0; j < n_elements; ++j) read_string();
                } else {
                    size_t type_size = get_type_size(sub_type);
                    cursor_ += (n_elements * type_size);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported GGUF type in KV pairs: " + std::to_string((int)type));
        }
    }
}

void GGUFReader::parse_tensor_infos(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        GGUFTensorInfo info;
        info.name = read_string();
        uint32_t n_dims = read_basic<uint32_t>();
        for (uint32_t d = 0; d < n_dims; ++d) info.shape.push_back(read_basic<uint64_t>());
        info.type = read_basic<uint32_t>();
        info.offset = read_basic<uint64_t>();
        tensor_infos_.push_back(info);
    }

}

void GGUFReader::print_metadata() const {
    std::cout << "GGUF Metadata:\n";
    for (const auto& [key, value] : metadata_) {
        std::cout << "  " << key << ": ";
        std::visit([](const auto& val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, bool>) {
                std::cout << (val ? "true" : "false");
            } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                std::cout << static_cast<int>(val);
            }
            else {
                std::cout << val;
            }
        }, value);
        std::cout << std::endl;
    }
}