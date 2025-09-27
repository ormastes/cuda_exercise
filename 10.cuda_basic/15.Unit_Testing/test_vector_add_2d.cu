// test_vector_add_2d.cu - Tests for 2D vector operations using GPU testing framework
#include "gpu_gtest.h"
#include "vector_add_2d.h"
#include "cuda_utils.h"  // Use our custom CUDA utilities library
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

//==================================================//
//        1. GPU_TEST - Simple Device Test          //
//==================================================//

GPU_TEST(SimpleDeviceTest, ComputeSum) {
    float a = 3.0f;
    float b = 4.0f;
    float result = compute_sum(a, b);
    GPU_EXPECT_NEAR(result, 7.0f, 1e-5f);
}

//==================================================//
//    2. GPU_TEST_CFG - Test with Custom Config     //
//==================================================//

GPU_TEST_CFG(ConfiguredTest, ParallelSum, dim3(1), dim3(32)) {
    int tid = threadIdx.x;
    if (tid < 10) {
        float value = compute_sum(float(tid), float(tid * 2));
        GPU_EXPECT_NEAR(value, float(tid * 3), 1e-5f);
    }
}

//==================================================//
//      3. GPU_TEST_F - Fixture-based Test          //
//==================================================//

struct ReductionFixture : ::testing::Test {
    struct DeviceView {
        float* data;
        float* result;
        int size;
    };

    float* d_data;
    float* d_result;
    DeviceView* d_view;
    static const int size = 100;

    void SetUp() override {
        // Using our utilities (with original API shown in comments)
        d_data = cuda_malloc<float>(size);      // Original: cudaMalloc(&d_data, size * sizeof(float));
        d_result = cuda_calloc<float>(1);        // Original: cudaMalloc(&d_result, sizeof(float));
                                                  //          cudaMemset(d_result, 0, sizeof(float));

        // Initialize with test data
        std::vector<float> h_data(size, 1.0f);
        cuda_memcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
        // Original: cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        // Create device view
        DeviceView h_view = {d_data, d_result, size};
        d_view = cuda_malloc<DeviceView>(1);     // Original: cudaMalloc(&d_view, sizeof(DeviceView));
        cuda_memcpy(d_view, &h_view, 1, cudaMemcpyHostToDevice);
        // Original: cudaMemcpy(d_view, &h_view, sizeof(DeviceView), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cuda_free(d_data);    // Original: cudaFree(d_data);
        cuda_free(d_result);  // Original: cudaFree(d_result);
        cuda_free(d_view);    // Original: cudaFree(d_view);
    }

    const DeviceView* device_view() const { return d_view; }

    GpuLaunchCfg launch_cfg() const {
        return GpuLaunchCfg{
            .grid = dim3(1),
            .block = dim3(128),
            .shmem = 0,
            .stream = nullptr
        };
    }
};

GPU_TEST_F(ReductionFixture, SumElements) {
    int tid = threadIdx.x;
    if (tid < _ctx->size) {
        float value = _ctx->data[tid];
        GPU_EXPECT_NEAR(value, 1.0f, 1e-5f);
    }
}

//==================================================//
//      4. GPU_TEST_P - Parameterized Test          //
//==================================================//

GPU_TEST_P(ParameterizedTest, AddValues) {
    float param = _param;
    float result = compute_sum(param, param);
    GPU_EXPECT_NEAR(result, param * 2.0f, 1e-5f);
}

GPU_INSTANTIATE_TEST_SUITE_P(Values, ParameterizedTest, AddValues,
    ::testing::Values(1.0f, 2.0f, 3.0f, 5.0f, 10.0f));

//==================================================//
//            Host-side Integration Test            //
//==================================================//

TEST(HostIntegrationTest, VectorAdd2D) {
    const int width = 32;
    const int height = 32;
    const int size = width * height;

    // Allocate device memory using our utilities
    float* d_a = cuda_malloc<float>(size);  // Original: cudaMalloc(&d_a, size * sizeof(float));
    float* d_b = cuda_malloc<float>(size);  // Original: cudaMalloc(&d_b, size * sizeof(float));
    float* d_c = cuda_malloc<float>(size);  // Original: cudaMalloc(&d_c, size * sizeof(float));

    // Initialize test data
    std::vector<float> h_a(size, 2.0f);
    std::vector<float> h_b(size, 3.0f);

    // Copy data to device using our utilities
    cuda_memcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    // Original: cudaMemcpy(d_a, h_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cuda_memcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);
    // Original: cudaMemcpy(d_b, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    vector_add_2d<<<grid, block>>>(d_a, d_b, d_c, width, height);

    // Check results
    std::vector<float> h_c(size);
    cuda_memcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);
    // Original: cudaMemcpy(h_c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        EXPECT_NEAR(h_c[i], 5.0f, 1e-5f);
    }

    // Free device memory using our utilities
    cuda_free(d_a);  // Original: cudaFree(d_a);
    cuda_free(d_b);  // Original: cudaFree(d_b);
    cuda_free(d_c);  // Original: cudaFree(d_c);
}

TEST(HostIntegrationTest, ReduceSum2D) {
    const int width = 16;
    const int height = 16;
    const int size = width * height;

    // Allocate device memory
    float* d_input = cuda_malloc<float>(size);      // Original: cudaMalloc(&d_input, size * sizeof(float));
    float* d_output = cuda_calloc<float>(1);         // Original: cudaMalloc(&d_output, sizeof(float));
                                                      //          cudaMemset(d_output, 0, sizeof(float));

    // Initialize with all ones
    std::vector<float> h_input(size, 1.0f);
    cuda_memcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    // Original: cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch reduction kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t shmem_size = block.x * block.y * sizeof(float);
    reduce_sum_2d<<<grid, block, shmem_size>>>(d_input, d_output, width, height);

    // Check result
    float h_output;
    cuda_memcpy(&h_output, d_output, 1, cudaMemcpyDeviceToHost);
    // Original: cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_output, float(size), 1e-3f);

    // Free device memory
    cuda_free(d_input);   // Original: cudaFree(d_input);
    cuda_free(d_output);  // Original: cudaFree(d_output);
}