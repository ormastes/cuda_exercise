// Test sample for GPU parameterized tests
#include "../../00.lib/gpu_gtest.h"
#include <vector>
#include <iostream>

// Simple parameterized test with int values
GPU_TEST_P(BasicParam, SquareTest) {
    int value = _param;
    int expected = value * value;
    int result = value * value;
    GPU_EXPECT_EQ(result, expected);
}

// Instantiate with different values
GPU_INSTANTIATE_TEST_SUITE_P(SmallValues, BasicParam, SquareTest,
    ::testing::Values(1, 2, 3, 4, 5));

GPU_INSTANTIATE_TEST_SUITE_P(LargeValues, BasicParam, SquareTest,
    ::testing::Values(10, 20, 30, 40, 50));

// Parameterized test with custom launch configuration
GPU_TEST_P_CFG(ParamWithConfig, ThreadIdTest, 4, 32) {
    int param_val = _param;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread checks if param is positive
    if (tid < param_val) {
        GPU_EXPECT_TRUE(param_val > 0);
    }

    // Thread 0 does additional validation
    if (tid == 0) {
        GPU_EXPECT_EQ(param_val * 2, param_val + param_val);
    }
}

GPU_INSTANTIATE_TEST_SUITE_P(ThreadTests, ParamWithConfig, ThreadIdTest,
    ::testing::Values(16, 32, 64, 128));

// Test with floating point parameters
template<typename T>
__device__ T device_abs(T val) {
    return val < 0 ? -val : val;
}

GPU_TEST_P(FloatParam, NearTest) {
    float value = static_cast<float>(_param);
    float result = value * 0.1f;
    float expected = value / 10.0f;
    GPU_EXPECT_NEAR(result, expected, 0.0001);
}

GPU_INSTANTIATE_TEST_SUITE_P(FloatValues, FloatParam, NearTest,
    ::testing::Values(1, 10, 100, 1000));

// Range-based parameterized test
GPU_TEST_P(RangeParam, IncrementTest) {
    int value = _param;
    int incremented = value + 1;
    GPU_EXPECT_EQ(incremented, value + 1);
    GPU_EXPECT_TRUE(incremented > value);
}

GPU_INSTANTIATE_TEST_SUITE_P(RangeTest, RangeParam, IncrementTest,
    ::testing::Range(0, 10, 2)); // 0, 2, 4, 6, 8

// Test with grid-stride loop
GPU_TEST_P_CFG(GridStrideParam, ArrayProcessing, 8, 64) {
    int array_size = _param;

    // Simulate array processing with grid-stride loop
    GPU_FOR_ALL(i, array_size) {
        int value = i * 2;
        int expected = i + i;
        GPU_EXPECT_EQ(value, expected);
    }
}

GPU_INSTANTIATE_TEST_SUITE_P(ArraySizes, GridStrideParam, ArrayProcessing,
    ::testing::Values(64, 128, 256, 512));

// Combine different parameter generators
GPU_TEST_P(CombineParam, MultiplyTest) {
    int value = _param;
    if (value < 10) {
        GPU_EXPECT_TRUE(value < 10);
        GPU_EXPECT_EQ(value * 10, value * 10);
    } else {
        GPU_EXPECT_TRUE(value >= 10);
        GPU_EXPECT_EQ(value / 10, value / 10);
    }
}

GPU_INSTANTIATE_TEST_SUITE_P(Combined, CombineParam, MultiplyTest,
    ::testing::ValuesIn(std::vector<int>{1, 5, 10, 50, 100}));

// Test boolean conditions with parameters
GPU_TEST_P(BoolParam, ConditionalTest) {
    int threshold = 50;
    int value = _param;
    bool is_large = value > threshold;

    if (is_large) {
        GPU_EXPECT_TRUE(value > threshold);
        GPU_EXPECT_TRUE(value >= threshold + 1);
    } else {
        GPU_EXPECT_TRUE(value <= threshold);
    }
}

GPU_INSTANTIATE_TEST_SUITE_P(BoolTests, BoolParam, ConditionalTest,
    ::testing::Values(10, 30, 50, 51, 70, 100));

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
    std::cout << "Running tests on: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    return RUN_ALL_TESTS();
}