#include "Tensor.hpp"
#include <iostream>
#include <vector>
#include <cassert>


#define ASSERT_MSG(cond, msg) \
    if (!(cond)) { std::cerr << "FAIL: " << msg << std::endl; exit(1); } \
    else { std::cout << "PASS: " << msg << std::endl; }

int main() {
    std::cout << "--- Starting Tensor Core Tests ---" << std::endl;

    // 1. Setup Dummy Data [1, 2, 3, 4, 5, 6]
    std::vector<float> raw_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::span<float> data_span(raw_data);
    
    // Create a 2x3 Tensor
    Tensor T(data_span, {2, 3});

    // 2. Test Row-Major Indexing
    // Expected: (0,1) is 2.0, (1,2) is 6.0
    ASSERT_MSG(T(0, 1) == 2.0f, "Standard indexing (0,1)");
    ASSERT_MSG(T(1, 2) == 6.0f, "Standard indexing (1,2)");

    // 3. Test Zero-Copy Transpose
    // Logic: Shape becomes 3x2, Strides swap from {3, 1} to {1, 3}
    Tensor TT = T.transpose();
    
    ASSERT_MSG(TT.shape()[0] == 3 && TT.shape()[1] == 2, "Transpose shape swap");
    ASSERT_MSG(TT.strides()[0] == 1 && TT.strides()[1] == 3, "Transpose stride swap");

    // 4. Test Value Consistency after Transpose
    // In original (2x3), (0, 1) was 2.0. In transposed (3x2), (1, 0) should be 2.0.
    ASSERT_MSG(TT(1, 0) == 2.0f, "Transposed indexing consistency (1,0)");
    ASSERT_MSG(TT(2, 1) == 6.0f, "Transposed indexing consistency (2,1)");

    // 5. Test Reshape (Metadata only)
    Tensor TR = T.reshape({6, 1});
    ASSERT_MSG(TR.shape()[0] == 6 && TR.strides()[0] == 1, "Reshape metadata check");
    ASSERT_MSG(TR(5, 0) == 6.0f, "Reshaped indexing (5,0)");

    std::cout << "--- All Tests Passed! ---" << std::endl;
    return 0;
}