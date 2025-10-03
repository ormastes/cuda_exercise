// test_error_handling.cu - GTest-based unit tests for CUDA error handling and debugging
#include <gtest/gtest.h>
#include <cmath>
#include <climits>
#include "cuda_utils.h"
#include "gpu_gtest.h"
#include "vector_add_2d.cu"  // Direct inclusion for testing device functions

// Test fixture for error handling tests
class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check for CUDA device
        int deviceCount = 0;
        CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
        ASSERT_GT(deviceCount, 0) << "No CUDA devices found!";
    }

    void TearDown() override {
        // Reset any error state
        cudaGetLastError(); // Intentionally not checking - just clearing
    }
};

// Test basic error checking with normal operations
TEST_F(ErrorHandlingTest, BasicErrorChecking) {
    const int width = 256;
    const int height = 256;
    const int size = width * height;

    // Allocate device memory with error checking
    float* d_a = cuda_malloc<float>(size);
    float* d_b = cuda_malloc<float>(size);
    float* d_c = cuda_malloc<float>(size);

    ASSERT_NE(d_a, nullptr);
    ASSERT_NE(d_b, nullptr);
    ASSERT_NE(d_c, nullptr);

    // Allocate and initialize host memory
    std::vector<float> h_a(size);
    std::vector<float> h_b(size);
    std::vector<float> h_c(size);

    for (int i = 0; i < size; i++) {
        h_a[i] = static_cast<float>(i % 100);
        h_b[i] = static_cast<float>((i * 2) % 100);
    }

    // Copy to device with error checking
    cuda_memcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cuda_memcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize = grid_size_2d(width, height, blockSize);

    vector_add_2d<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);

    // Check for kernel launch errors
    CHECK_KERNEL_LAUNCH();

    // Copy result back
    cuda_memcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Verify results - kernel computes square(a) + b
    for (int i = 0; i < 5; i++) {
        float expected = h_a[i] * h_a[i] + h_b[i];
        EXPECT_NEAR(h_c[i], expected, 1e-5f) << "at index " << i;
    }

    // Cleanup
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);
}

// Test reduce_sum_2d with error handling
TEST_F(ErrorHandlingTest, ReduceSumWithErrorHandling) {
    const int width = 128;
    const int height = 128;
    const int size = width * height;

    // Allocate memory
    float* d_input = cuda_malloc<float>(size);
    float* d_output = cuda_malloc<float>(1);

    ASSERT_NE(d_input, nullptr);
    ASSERT_NE(d_output, nullptr);

    // Initialize input
    std::vector<float> h_input(size);
    float expected_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;  // All ones for easy verification
        expected_sum += h_input[i];
    }

    // Copy to device
    cuda_memcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cuda_memset(d_output, 0, 1);  // Using cuda_memset wrapper with built-in error checking

    // Launch reduction kernel - treat 2D array as 1D
    dim3 blockSize(256);  // Use 1D block for reduction
    dim3 gridSize(grid_size_1d(size, blockSize.x));

    size_t sharedMemSize = blockSize.x * sizeof(float);
    reduce_sum<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, size, 1);
    CHECK_KERNEL_LAUNCH();

    // Get result
    float h_output = 0.0f;
    cuda_memcpy(&h_output, d_output, 1, cudaMemcpyDeviceToHost);

    // Verify result
    EXPECT_NEAR(h_output, expected_sum, 1e-3f)
        << "Expected sum: " << expected_sum << ", Computed sum: " << h_output;

    // Cleanup
    cuda_free(d_input);
    cuda_free(d_output);
}

// Test out-of-bounds error detection (should be run with compute-sanitizer)
TEST_F(ErrorHandlingTest, OutOfBoundsDetection) {
    const int width = 100;
    const int height = 100;
    const int size = width * height;

    // Allocate memory
    float* d_a = cuda_malloc<float>(size);
    float* d_b = cuda_malloc<float>(size);
    float* d_c = cuda_malloc<float>(size);

    ASSERT_NE(d_a, nullptr);
    ASSERT_NE(d_b, nullptr);
    ASSERT_NE(d_c, nullptr);

    // Initialize with zeros using cuda_memset wrapper
    cuda_memset(d_a, 0, size);
    cuda_memset(d_b, 0, size);
    cuda_memset(d_c, 0, size);

    // Launch kernel with too many blocks (intentional bug)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x + 2,  // Extra blocks!
                  (height + blockSize.y - 1) / blockSize.y + 2); // Extra blocks!

    // This kernel has a bug - missing boundary check
    vector_add_2d_with_bug<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);

    // The kernel may complete without reporting an error
    // Errors will be detected by compute-sanitizer --tool memcheck
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GTEST_SKIP() << "Kernel launch error detected: " << cudaGetErrorString(err);
    }

    err = cudaDeviceSynchronize();
    // We expect this might fail or might succeed (undefined behavior)
    // The test documents the buggy behavior for educational purposes

    // Cleanup
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);
}

// Test assertion-based debugging (only works in debug builds with -G flag)
TEST_F(ErrorHandlingTest, AssertionDebugging) {
    const int width = 64;
    const int height = 64;
    const int size = width * height;

    float* d_a = cuda_malloc<float>(size);
    float* d_b = cuda_malloc<float>(size);
    float* d_c = cuda_malloc<float>(size);

    ASSERT_NE(d_a, nullptr);
    ASSERT_NE(d_b, nullptr);
    ASSERT_NE(d_c, nullptr);

    // Create host data with some NaN values for testing
    std::vector<float> h_a(size);
    std::vector<float> h_b(size);

    for (int i = 0; i < size; i++) {
        h_a[i] = (i % 10 == 0) ? NAN : static_cast<float>(i);  // Some NaN values
        h_b[i] = static_cast<float>(i * 2);
    }

    cuda_memcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cuda_memcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize = grid_size_2d(width, height, blockSize);

    // Launch kernel with assertions
    vector_add_2d_with_assert<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);

    cudaError_t err = cudaDeviceSynchronize();

    // In debug mode with -G, this should fail due to assertion
    // In release mode, assertions are ignored
    #ifdef DEBUG
        EXPECT_NE(err, cudaSuccess) << "Expected assertion failure with NaN values";
    #else
        // In release mode, the kernel might complete despite NaN values
        if (err != cudaSuccess) {
            GTEST_SKIP() << "Assertion triggered (debug mode behavior): " << cudaGetErrorString(err);
        }
    #endif

    // Cleanup
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);
}

// Test error recovery mechanisms
TEST_F(ErrorHandlingTest, ErrorRecovery) {
    // Try to allocate an impossibly large amount of memory
    size_t huge_size = ULLONG_MAX / 2;  // Way too much!
    float* d_huge = nullptr;

    cudaError_t err = cudaMalloc(&d_huge, huge_size);

    // We expect this to fail
    EXPECT_NE(err, cudaSuccess) << "Unexpectedly succeeded in allocating " << huge_size << " bytes";

    if (err != cudaSuccess) {
        // Reset device to clear error state
        err = cudaDeviceReset();
        EXPECT_EQ(err, cudaSuccess) << "Failed to reset device: " << cudaGetErrorString(err);

        // Now try a reasonable allocation
        size_t reasonable_size = 1024 * sizeof(float);
        float* d_small = nullptr;
        err = cudaMalloc(&d_small, reasonable_size);

        EXPECT_EQ(err, cudaSuccess) << "Failed to allocate after recovery: " << cudaGetErrorString(err);

        if (err == cudaSuccess) {
            cudaFree(d_small);
        }
    }
}

// GPU test for device-side error checking
GPU_TEST(DeviceErrorTest, BoundaryCheck) {
    // This test runs on the GPU
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Test boundary condition
    const int width = 32;
    const int height = 32;

    if (x < width && y < height) {
        int idx = y * width + x;
        GPU_EXPECT_TRUE(idx >= 0);
        GPU_EXPECT_TRUE(idx < width * height);
    }
}

// GPU test with configuration for testing assertions
GPU_TEST_CFG(DeviceErrorTest, AssertionTest, dim3(1), dim3(32)) {
    float a = float(threadIdx.x);
    float b = float(threadIdx.x * 2);

    // Test that values are not NaN
    GPU_EXPECT_TRUE(!isnan(a));
    GPU_EXPECT_TRUE(!isnan(b));

    float sum = a + b;
    GPU_EXPECT_TRUE(!isnan(sum));
    GPU_EXPECT_NEAR(sum, a + b, 1e-5f);
}

//==================================================//
// Examples from README Section 16.5: Race Conditions
//==================================================//

// Example of race condition in shared memory (from README 16.5.1)
__global__ void raceyKernel(float* result) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    sdata[tid] = tid;

    // RACE CONDITION: Missing synchronization
    // Some threads may read before others have written
    if (tid < 128) {
        result[tid] = sdata[tid] + sdata[tid + 128];
    }
}

// Corrected version with proper synchronization
__global__ void correctRaceKernel(float* result) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    sdata[tid] = tid;

    __syncthreads();  // Ensure all writes complete before reads

    if (tid < 128) {
        result[tid] = sdata[tid] + sdata[tid + 128];
    }
}

// Test for race condition detection (README 16.5.2)
TEST_F(ErrorHandlingTest, RaceConditionDetection) {
    const int size = 128;
    float* d_result = cuda_malloc<float>(size);

    // Launch kernel with race condition
    raceyKernel<<<1, 256>>>(d_result);
    CHECK_KERNEL_LAUNCH();

    // Launch corrected kernel
    correctRaceKernel<<<1, 256>>>(d_result);
    CHECK_KERNEL_LAUNCH();

    // Verify corrected results
    std::vector<float> h_result(size);
    cuda_memcpy(h_result.data(), d_result, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        float expected = i + (i + 128);
        EXPECT_NEAR(h_result[i], expected, 1e-5f) << "at index " << i;
    }

    cuda_free(d_result);
}

//==================================================//
// Examples from README Section 16.7: Error Recovery
//==================================================//

// Retry mechanism for transient failures (README 16.7.2)
template<typename Func>
bool retryOperation(Func operation, int maxRetries = 3) {
    for (int attempt = 0; attempt < maxRetries; ++attempt) {
        cudaError_t err = operation();
        if (err == cudaSuccess) {
            return true;
        }

        // Check if error is retryable
        if (err == cudaErrorMemoryAllocation ||
            err == cudaErrorLaunchFailure) {
            fprintf(stderr, "Attempt %d failed: %s. Retrying...\n",
                   attempt + 1, cudaGetErrorString(err));

            // Wait before retry
            cudaDeviceSynchronize();

            // Clear error state
            cudaGetLastError();
        } else {
            // Non-retryable error
            break;
        }
    }
    return false;
}

// Test retry mechanism
TEST_F(ErrorHandlingTest, RetryMechanism) {
    // Lambda for memory allocation that might fail
    auto allocateLargeMemory = []() -> cudaError_t {
        size_t huge_size = SIZE_MAX / 4;  // Very large allocation
        float* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, huge_size);
        if (err == cudaSuccess) {
            cudaFree(d_ptr);
        }
        return err;
    };

    bool success = retryOperation(allocateLargeMemory, 2);
    // We expect this to fail even with retries
    EXPECT_FALSE(success) << "Large allocation should fail";

    // Now try with a reasonable allocation
    auto allocateNormalMemory = []() -> cudaError_t {
        float* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, 1024 * sizeof(float));
        if (err == cudaSuccess) {
            cudaFree(d_ptr);
        }
        return err;
    };

    success = retryOperation(allocateNormalMemory, 2);
    EXPECT_TRUE(success) << "Normal allocation should succeed";
}

// Test fixture for parameterized error scenarios
class ParameterizedErrorTest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    void SetUp() override {
        int deviceCount = 0;
        CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA devices found";
        }
    }
};

// Parameterized test for different dimensions
TEST_P(ParameterizedErrorTest, DimensionBoundaryTest) {
    int width, height;
    std::tie(width, height) = GetParam();

    const int size = width * height;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // Test allocation with various sizes
    cudaError_t err = cudaMalloc(&d_a, size * sizeof(float));
    EXPECT_EQ(err, cudaSuccess) << "Failed to allocate for size " << size;

    if (err == cudaSuccess) {
        err = cudaMalloc(&d_b, size * sizeof(float));
        EXPECT_EQ(err, cudaSuccess);

        err = cudaMalloc(&d_c, size * sizeof(float));
        EXPECT_EQ(err, cudaSuccess);

        if (d_a && d_b && d_c) {
            // Initialize using cuda_memset wrapper
            cuda_memset(d_a, 0, size);
            cuda_memset(d_b, 0, size);

            // Launch kernel with proper bounds
            dim3 blockSize(16, 16);
            dim3 gridSize = grid_size_2d(width, height, blockSize);

            vector_add_2d<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);

            err = cudaDeviceSynchronize();
            EXPECT_EQ(err, cudaSuccess) << "Kernel failed for dimensions "
                                        << width << "x" << height;
        }

        // Cleanup
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
    }
}

// Instantiate parameterized tests with various dimensions
INSTANTIATE_TEST_SUITE_P(
    DimensionTests,
    ParameterizedErrorTest,
    ::testing::Values(
        std::make_tuple(16, 16),
        std::make_tuple(32, 32),
        std::make_tuple(64, 64),
        std::make_tuple(128, 128),
        std::make_tuple(256, 256),
        std::make_tuple(17, 17),   // Non-power-of-2
        std::make_tuple(100, 100),  // Non-aligned
        std::make_tuple(1, 1),      // Edge case: minimal
        std::make_tuple(1024, 1024) // Large size
    )
);

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Print test information
    std::cout << "\n=== CUDA Error Handling Unit Tests ===" << std::endl;
    std::cout << "\nTo detect memory errors, run with:" << std::endl;
    std::cout << "  compute-sanitizer --tool memcheck ./16_Unit_Testing_test" << std::endl;
    std::cout << "\nTo detect race conditions, run with:" << std::endl;
    std::cout << "  compute-sanitizer --tool racecheck ./16_Unit_Testing_test" << std::endl;
    std::cout << "\nFor debugging with cuda-gdb:" << std::endl;
    std::cout << "  cuda-gdb ./16_Unit_Testing_test" << std::endl;
    std::cout << std::endl;

    return RUN_ALL_TESTS();
}