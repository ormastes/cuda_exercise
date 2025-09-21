// Test sample for GPU GENERATOR tests
#include "../../00.lib/gpu_gtest.h"
#include <iostream>

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

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Check CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Running generator tests on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    return RUN_ALL_TESTS();
}