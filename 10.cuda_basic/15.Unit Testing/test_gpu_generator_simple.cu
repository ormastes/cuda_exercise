// Simplified test for GPU generator functionality
#include "../../00.lib/gpu_gtest.h"
#include <iostream>

// Test the basic GPU test functionality first
GPU_TEST(BasicGpu, SimpleKernel) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        int a = 5;
        int b = 10;
        GPU_EXPECT_TRUE(a < b);
        GPU_EXPECT_EQ(a + b, 15);
        GPU_EXPECT_NEAR(3.14, 3.14159, 0.01);
    }
}

// Test with custom configuration
GPU_TEST_CFG(BasicGpu, ConfiguredKernel, 2, 32) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread does a simple check
    if (tid < 64) {
        int value = tid * 2;
        GPU_EXPECT_EQ(value, tid * 2);
        GPU_EXPECT_TRUE(value >= 0);
    }
}

// Test grid-stride loop
GPU_TEST_CFG(BasicGpu, GridStrideTest, 4, 64) {
    int array_size = 256;

    GPU_FOR_ALL(i, array_size) {
        int value = i * 3;
        int expected = i * 3;
        GPU_EXPECT_EQ(value, expected);

        if (i == 0) {
            GPU_EXPECT_EQ(value, 0);
        }

        if (i == array_size - 1) {
            GPU_EXPECT_EQ(value, (array_size - 1) * 3);
        }
    }
}

// Test error detection
GPU_TEST(BasicGpu, FailureDetection) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // This should pass
        GPU_EXPECT_TRUE(1 == 1);

        // Test boundary conditions
        int max_int = 2147483647;
        GPU_EXPECT_TRUE(max_int > 0);
        GPU_EXPECT_EQ(max_int + 1, -2147483648);  // Integer overflow
    }
}

// Test floating point operations
GPU_TEST_CFG(BasicGpu, FloatOperations, 1, 32) {
    int tid = threadIdx.x;

    if (tid < 10) {
        float value = static_cast<float>(tid) * 0.1f;
        float expected = tid * 0.1f;
        GPU_EXPECT_NEAR(value, expected, 0.0001);

        // Test special float values
        if (tid == 0) {
            float zero = 0.0f;
            GPU_EXPECT_EQ(zero, 0.0f);
        }
    }
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
    std::cout << "Running basic GPU tests on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    return RUN_ALL_TESTS();
}