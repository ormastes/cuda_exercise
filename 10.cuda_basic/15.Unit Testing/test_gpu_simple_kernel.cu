// Simple GPU test without generators
#include "../../00.lib/gpu_gtest.h"
#include <iostream>

// Basic GPU test
GPU_TEST(SimpleGpuTest, BasicKernel) {
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
GPU_TEST_CFG(SimpleGpuTest, ConfiguredKernel, 2, 32) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread does a simple check
    if (tid < 64) {
        int value = tid * 2;
        GPU_EXPECT_EQ(value, tid * 2);
        GPU_EXPECT_TRUE(value >= 0);
    }
}

// Test grid-stride loop
GPU_TEST_CFG(SimpleGpuTest, GridStrideTest, 4, 64) {
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
    std::cout << "Running simple GPU tests on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    return RUN_ALL_TESTS();
}