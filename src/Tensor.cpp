#include "Tensor.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>

Tensor::Tensor(std::span<float> data, std::vector<size_t> shape)
    : data_(data), shape_(shape), strides_(computeDefaultStrides(shape)) {}

std::vector<size_t> Tensor::computeDefaultStrides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    // Walk backwards to compute row-major strides
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

float& Tensor::operator()(size_t r, size_t c) {
    // index = row * row_stride + col * col_stride
    return data_[r * strides_[0] + c * strides_[1]];
}

float Tensor::operator()(size_t r, size_t c) const {
    return data_[r * strides_[0] + c * strides_[1]];
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) throw std::runtime_error("Transpose only supported for 2D");
    
    Tensor t = *this;
    std::swap(t.shape_[0], t.shape_[1]);
    std::swap(t.strides_[0], t.strides_[1]);
    return t;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    // Logic: Total size must remain constant
    size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    if (new_total != data_.size()) throw std::runtime_error("Reshape size mismatch");

    Tensor t = *this;
    t.shape_ = new_shape;
    t.strides_ = computeDefaultStrides(new_shape);
    return t;
}

void Tensor::print(const std::string& name) const {
    if (!name.empty()) std::cout << name << " ";
    std::cout << "Shape: (" << shape_[0] << ", " << shape_[1] << ")\n";
    
    for (size_t i = 0; i < std::min(shape_[0], (size_t)10); ++i) {
        for (size_t j = 0; j < std::min(shape_[1], (size_t)10); ++j) {
            std::cout << std::fixed << std::setprecision(4) << (*this)(i, j) << " ";
        }
        if (shape_[1] > 10) std::cout << "...";
        std::cout << "\n";
    }
    if (shape_[0] > 10) std::cout << "...\n";
}