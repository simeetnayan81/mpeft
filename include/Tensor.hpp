#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <span>
#include <vector>
#include <memory>
#include <initializer_list>

class Tensor {
public:

    Tensor(std::span<float> data, std::vector<size_t> shape);
    
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t size() const { return data_.size(); }

    float& operator()(size_t r, size_t c);
    float operator()(size_t r, size_t c) const;

    Tensor transpose() const;
    Tensor reshape(const std::vector<size_t>& new_shape) const;

    void print(const std::string& name = "") const;
    static std::vector<size_t> computeDefaultStrides(const std::vector<size_t>& shape);

private:
    Tensor(std::span<float> data, std::vector<size_t> shape, std::vector<size_t> strides);
    std::span<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

};

#endif