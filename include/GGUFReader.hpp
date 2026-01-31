#ifndef GGUF_READER_HPP
#define GGUF_READER_HPP

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <cstdint>
#include "Tensor.hpp"

// Official GGUF Value Types
enum class GGUFValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12,
};

// A variant allows a single Metadata object to hold any of the types above
using GGUFMetadataValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
    float, bool, std::string, uint64_t, int64_t, double
>;

struct GGUFTensorInfo {
    std::string name;
    std::vector<size_t> shape;
    uint32_t type; // e.g., FP32, Q4_K_M
    size_t offset; // Where it starts in the file
};