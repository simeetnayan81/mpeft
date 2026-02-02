#ifndef GGUF_READER_HPP
#define GGUF_READER_HPP

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <cstdint>
#include "Tensor.hpp"
#include <variant>

//GGUF Documentation: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

enum class GGUFValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12,
};

using GGUFMetadataValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
    float, bool, std::string, uint64_t, int64_t, double
>;

struct GGUFTensorInfo {
    std::string name;
    std::vector<size_t> shape;
    uint32_t type; 
    size_t offset; 
};

class GGUFReader {
public:
    GGUFReader(const void* data, size_t size);
    void parse();

    const std::map<std::string, GGUFMetadataValue>& metadata() const { return metadata_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensor_infos_; }
    void print_metadata() const;

private:
    const uint8_t* data_ptr_;
    size_t total_size_;
    size_t cursor_ = 0;

    std::map<std::string, GGUFMetadataValue> metadata_;
    std::vector<GGUFTensorInfo> tensor_infos_;

    std::string read_string();
    size_t get_type_size(GGUFValueType type);
    template<typename T> T read_basic();
    void parse_kv_pairs(uint64_t count);
    void parse_tensor_infos(uint64_t count);
};

#endif