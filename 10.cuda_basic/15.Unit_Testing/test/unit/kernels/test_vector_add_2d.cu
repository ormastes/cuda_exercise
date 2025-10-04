// test_vector_add_2d.cu - Tests for 2D vector operations using GPU testing framework
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "gpu_gtest.h"
#include "gtest_generator.h"
#include "cuda_utils.h"  // Use our custom CUDA utilities library

// Direct inclusion of implementation for testing device functions
#include "../../../src/kernels/vector_add_2d.cu"



// Base test class (optional, can be used for shared setup)
class GpuGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any common setup
    }
};

// Test with ALIGNED mode (round-robin)
GPU_TEST_G(GpuGeneratorTest, AlignedMode) {
    int x = GPU_GENERATOR(1, 2, 3, 4);
    int y = GPU_GENERATOR(100, 200, 300);
    int z = GPU_GENERATOR(1000, 2000);
    GPU_USE_GENERATOR(ALIGNED);  // Max 4 iterations (max of all generators)

    // In ALIGNED mode:
    // Iteration 0: (1, 100, 1000)
    // Iteration 1: (2, 200, 2000)
    // Iteration 2: (3, 300, 1000) - z wraps around
    // Iteration 3: (4, 100, 2000) - y and z wrap around

    GPU_EXPECT_TRUE(x > 0);
    GPU_EXPECT_TRUE(y >= 100);
    GPU_EXPECT_TRUE(z >= 1000);
}

// Test square function with generators
GPU_TEST_G(GpuGeneratorTest, square) {
    float a = GPU_GENERATOR(1.0f, 2.0f, 3.0f, 4.0f);
    GPU_USE_GENERATOR();  // Use default FULL mode for single generator

    float result = square(a);
    float expected = a * a;

    GPU_EXPECT_NEAR(result, expected, 1e-5f);
}



// Generator test with custom launch configuration
GPU_TEST_G_CFG(GpuGeneratorTest, ThreadConfig, 2, 32) {
    int threads = GPU_GENERATOR(32, 64, 128);
    int blocks = GPU_GENERATOR(1, 2, 4);
    GPU_USE_GENERATOR();  // 3 * 3 = 9 combinations

#ifdef __CUDA_ARCH__
    // This test runs with <<<2, 32>>> configuration
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        GPU_EXPECT_TRUE(threads > 0);
        GPU_EXPECT_TRUE(blocks > 0);
        GPU_EXPECT_EQ(threads * blocks, blocks * threads);
    }
#endif
}

// Test with floating point values
GPU_TEST_G(GpuGeneratorTest, FloatOperations) {
    float scale = GPU_GENERATOR(0.5f, 1.0f, 2.0f);
    float offset = GPU_GENERATOR(-1.0f, 0.0f, 1.0f);
    GPU_USE_GENERATOR();  // 3 * 3 = 9 combinations

    float result = scale * 10.0f + offset;
    float expected_min = scale * 10.0f - 1.1f;
    float expected_max = scale * 10.0f + 1.1f;

    GPU_EXPECT_TRUE(result >= expected_min);
    GPU_EXPECT_TRUE(result <= expected_max);
}

// Complex test with multiple generators
GPU_TEST_G(GpuGeneratorTest, MatrixDimensions) {
    int rows = GPU_GENERATOR(16, 32, 64);
    int cols = GPU_GENERATOR(8, 16, 32, 64);
    int depth = GPU_GENERATOR(1, 3);
    GPU_USE_GENERATOR();  // 3 * 4 * 2 = 24 combinations

    int total_elements = rows * cols * depth;

    GPU_EXPECT_TRUE(total_elements > 0);
    GPU_EXPECT_TRUE(rows <= 64);
    GPU_EXPECT_TRUE(cols <= 64);
    GPU_EXPECT_EQ(total_elements, rows * cols * depth);
}

// Test with boolean logic
GPU_TEST_G(GpuGeneratorTest, BooleanLogic) {
    int condition1 = GPU_GENERATOR(0, 1);
    int condition2 = GPU_GENERATOR(0, 1);
    int condition3 = GPU_GENERATOR(0, 1);
    GPU_USE_GENERATOR();  // 2 * 2 * 2 = 8 combinations

    bool c1 = condition1 != 0;
    bool c2 = condition2 != 0;
    bool c3 = condition3 != 0;

    // Test logical operations
    bool and_result = c1 && c2 && c3;
    bool or_result = c1 || c2 || c3;

    if (and_result) {
        GPU_EXPECT_TRUE(c1);
        GPU_EXPECT_TRUE(c2);
        GPU_EXPECT_TRUE(c3);
    }

    if (!or_result) {
        GPU_EXPECT_TRUE(!c1);
        GPU_EXPECT_TRUE(!c2);
        GPU_EXPECT_TRUE(!c3);
    }
}

// Test with grid-stride loop
GPU_TEST_G_CFG(GpuGeneratorTest, GridStrideLoop, 4, 64) {
    int array_size = GPU_GENERATOR(64, 128, 256);
    int multiplier = GPU_GENERATOR(2, 3);
    GPU_USE_GENERATOR();  // 3 * 2 = 6 combinations

#ifdef __CUDA_ARCH__
    GPU_FOR_ALL(i, array_size) {
        int value = i * multiplier;
        int expected = i * multiplier;
        GPU_EXPECT_EQ(value, expected);

        if (i == 0) {
            GPU_EXPECT_EQ(value, 0);
        }
    }
#endif
}

// Test edge cases
GPU_TEST_G(GpuGeneratorTest, EdgeCases) {
    int negative = GPU_GENERATOR(-10, -5, -1);
    int zero = GPU_GENERATOR(0);
    int positive = GPU_GENERATOR(1, 5, 10);
    GPU_USE_GENERATOR();  // 3 * 1 * 3 = 9 combinations

    GPU_EXPECT_TRUE(negative < 0);
    GPU_EXPECT_EQ(zero, 0);
    GPU_EXPECT_TRUE(positive > 0);

    // Test arithmetic properties
    GPU_EXPECT_EQ(negative + positive + zero, negative + positive);
    GPU_EXPECT_TRUE(negative * positive < 0);
    GPU_EXPECT_EQ(zero * positive, 0);
}


//==================================================//
//     Host-side Integration Tests with Generators  //
//==================================================//

// Test fixture for host-side generator tests
class HostGeneratorTest : public ::gtest_generator::TestWithGenerator {
};

TEST_G(HostGeneratorTest, VectorAdd2D) {
    // Generate different test dimensions
    int width = GENERATOR(16, 32, 64);
    int height = GENERATOR(16, 32, 64);
    float a_value = GENERATOR(1.0f, 2.0f, 3.0f);
    float b_value = GENERATOR(1.0f, 2.0f, 3.0f);
    USE_GENERATOR(ALIGNED);  // Use aligned mode to reduce test count (3 tests instead of 81)

    const int size = width * height;

    // Allocate device memory using our utilities
    float* d_a = cuda_malloc<float>(size);
    float* d_b = cuda_malloc<float>(size);
    float* d_c = cuda_malloc<float>(size);

    // Initialize test data with generated values
    std::vector<float> h_a(size, a_value);
    std::vector<float> h_b(size, b_value);

    // Copy data to device
    cuda_memcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cuda_memcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    vector_add_2d<<<grid, block>>>(d_a, d_b, d_c, width, height);
    cudaDeviceSynchronize();

    // Check results
    std::vector<float> h_c(size);
    cuda_memcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Expected: a_value + b_value (kernel does simple addition)
    float expected = a_value + b_value;
    for (int i = 0; i < size; i++) {
        EXPECT_NEAR(h_c[i], expected, 1e-5f);
    }

    // Free device memory
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);
}

TEST_G(HostGeneratorTest, ReduceSum2D) {
    // Generate test parameters
    int width = GENERATOR(8, 16, 32, 64);
    int height = GENERATOR(8, 16, 32, 64);
    float value = GENERATOR(0.5f, 1.0f, 2.0f, 3.0f);
    int stride = GENERATOR(1, 2);
    USE_GENERATOR(ALIGNED);  // Max of 4 test runs instead of 256

    const int size = width * height;

    // Allocate device memory
    float* d_input = cuda_malloc<float>(size);
    float* d_output = cuda_calloc<float>(1);

    // Initialize with generated value
    std::vector<float> h_input(size, value);
    cuda_memcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    // Launch reduction kernel with appropriate configuration
    dim3 block(256);  // Use 1D block for reduction
    dim3 grid((size + block.x - 1) / block.x);
    size_t shmem_size = block.x * sizeof(float);
    reduce_sum<<<grid, block, shmem_size>>>(d_input, d_output, size, stride);
    cudaDeviceSynchronize();

    // Check result
    float h_output;
    cuda_memcpy(&h_output, d_output, 1, cudaMemcpyDeviceToHost);

    // Expected sum: value * size (when stride=1)
    // Note: For stride > 1, the actual behavior depends on kernel implementation
    float expected = value * size;
    float tolerance = std::abs(expected) * 1e-4f + 1e-3f;

    EXPECT_NEAR(h_output, expected, tolerance)
        << "Failed for dimensions " << width << "x" << height
        << " with value=" << value << ", stride=" << stride;

    // Free device memory
    cuda_free(d_input);
    cuda_free(d_output);
}