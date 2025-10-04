// test_vector_add_2d.cu - Unit tests for memory-optimized vector_add_2d kernels
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <vector>
#include "gpu_gtest.h"
#include "kernels/vector_add_2d.h"

// Helper class for vector operations
class VectorHelper {
public:
    static void cpu_vector_add_2d(const float* A, const float* B, float* C,
                                   int width, int height) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                C[idx] = A[idx] * A[idx] + B[idx];  // square(A[idx]) + B[idx]
            }
        }
    }

    static bool compare_vectors(const float* A, const float* B, int N,
                                 float tolerance = 1e-4f) {
        for (int i = 0; i < N; i++) {
            if (std::fabs(A[i] - B[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    static void init_sequential(float* vec, int N) {
        for (int i = 0; i < N; i++) {
            vec[i] = static_cast<float>(i % 100);
        }
    }

    static void init_random(float* vec, int N, float scale = 1.0f) {
        for (int i = 0; i < N; i++) {
            vec[i] = (static_cast<float>(rand()) / RAND_MAX) * scale;
        }
    }
};

// Test fixture for vector_add_2d kernels
class VectorAdd2DMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

// Test basic 2D vector addition
TEST_F(VectorAdd2DMemoryTest, BasicImplementation) {
    const int width = 128;
    const int height = 128;
    const int N = width * height;
    size_t bytes = N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N]);
    std::unique_ptr<float[]> h_B(new float[N]);
    std::unique_ptr<float[]> h_C_gpu(new float[N]);
    std::unique_ptr<float[]> h_C_cpu(new float[N]);

    VectorHelper::init_random(h_A.get(), N, 10.0f);
    VectorHelper::init_random(h_B.get(), N, 10.0f);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    vector_add_2d<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_C_gpu.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    VectorHelper::cpu_vector_add_2d(h_A.get(), h_B.get(), h_C_cpu.get(), width, height);
    EXPECT_TRUE(VectorHelper::compare_vectors(h_C_gpu.get(), h_C_cpu.get(), N))
        << "Basic vector_add_2d failed";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test shared memory version
TEST_F(VectorAdd2DMemoryTest, SharedMemoryImplementation) {
    const int width = 128;
    const int height = 128;
    const int N = width * height;
    size_t bytes = N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N]);
    std::unique_ptr<float[]> h_B(new float[N]);
    std::unique_ptr<float[]> h_C_gpu(new float[N]);
    std::unique_ptr<float[]> h_C_cpu(new float[N]);

    VectorHelper::init_random(h_A.get(), N, 10.0f);
    VectorHelper::init_random(h_B.get(), N, 10.0f);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    size_t shared_mem = 2 * 16 * 16 * sizeof(float);

    vector_add_2d_shared<<<gridSize, blockSize, shared_mem>>>(d_A, d_B, d_C, width, height);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_C_gpu.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    VectorHelper::cpu_vector_add_2d(h_A.get(), h_B.get(), h_C_cpu.get(), width, height);
    EXPECT_TRUE(VectorHelper::compare_vectors(h_C_gpu.get(), h_C_cpu.get(), N))
        << "Shared memory vector_add_2d failed";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test coalesced access version
TEST_F(VectorAdd2DMemoryTest, CoalescedImplementation) {
    const int width = 256;
    const int height = 256;
    const int N = width * height;
    size_t bytes = N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N]);
    std::unique_ptr<float[]> h_B(new float[N]);
    std::unique_ptr<float[]> h_C_gpu(new float[N]);
    std::unique_ptr<float[]> h_C_cpu(new float[N]);

    VectorHelper::init_sequential(h_A.get(), N);
    VectorHelper::init_sequential(h_B.get(), N);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    vector_add_2d_coalesced<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_C_gpu.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    VectorHelper::cpu_vector_add_2d(h_A.get(), h_B.get(), h_C_cpu.get(), width, height);
    EXPECT_TRUE(VectorHelper::compare_vectors(h_C_gpu.get(), h_C_cpu.get(), N))
        << "Coalesced vector_add_2d failed";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test unrolled version
TEST_F(VectorAdd2DMemoryTest, UnrolledImplementation) {
    const int width = 256;
    const int height = 128;
    const int N = width * height;
    size_t bytes = N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N]);
    std::unique_ptr<float[]> h_B(new float[N]);
    std::unique_ptr<float[]> h_C_gpu(new float[N]);
    std::unique_ptr<float[]> h_C_cpu(new float[N]);

    VectorHelper::init_random(h_A.get(), N, 5.0f);
    VectorHelper::init_random(h_B.get(), N, 5.0f);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(16, 16);
    dim3 gridSize(((width + 3) / 4 + 15) / 16, (height + 15) / 16);

    vector_add_2d_unrolled<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_C_gpu.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    VectorHelper::cpu_vector_add_2d(h_A.get(), h_B.get(), h_C_cpu.get(), width, height);
    EXPECT_TRUE(VectorHelper::compare_vectors(h_C_gpu.get(), h_C_cpu.get(), N))
        << "Unrolled vector_add_2d failed";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test bank conflict-free version
TEST_F(VectorAdd2DMemoryTest, BankConflictFreeImplementation) {
    const int width = 128;
    const int height = 128;
    const int N = width * height;
    size_t bytes = N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N]);
    std::unique_ptr<float[]> h_B(new float[N]);
    std::unique_ptr<float[]> h_C_gpu(new float[N]);
    std::unique_ptr<float[]> h_C_cpu(new float[N]);

    VectorHelper::init_random(h_A.get(), N, 10.0f);
    VectorHelper::init_random(h_B.get(), N, 10.0f);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    vector_add_2d_bank_conflict_free<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_C_gpu.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    VectorHelper::cpu_vector_add_2d(h_A.get(), h_B.get(), h_C_cpu.get(), width, height);
    EXPECT_TRUE(VectorHelper::compare_vectors(h_C_gpu.get(), h_C_cpu.get(), N))
        << "Bank conflict-free vector_add_2d failed";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Performance comparison test
TEST_F(VectorAdd2DMemoryTest, PerformanceComparison) {
    const int width = 512;
    const int height = 512;
    const int N = width * height;
    const int iterations = 100;
    size_t bytes = N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N]);
    std::unique_ptr<float[]> h_B(new float[N]);

    VectorHelper::init_random(h_A.get(), N, 10.0f);
    VectorHelper::init_random(h_B.get(), N, 10.0f);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    size_t shared_mem = 2 * 16 * 16 * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float times[4];
    const char* names[4] = {"basic", "shared", "coalesced", "bank_conflict_free"};

    // Warmup
    vector_add_2d<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    cudaDeviceSynchronize();

    // Basic
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        vector_add_2d<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[0], start, stop);
    times[0] /= iterations;

    // Shared memory
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        vector_add_2d_shared<<<gridSize, blockSize, shared_mem>>>(d_A, d_B, d_C, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[1], start, stop);
    times[1] /= iterations;

    // Coalesced
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        vector_add_2d_coalesced<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[2], start, stop);
    times[2] /= iterations;

    // Bank conflict-free
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        vector_add_2d_bank_conflict_free<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&times[3], start, stop);
    times[3] /= iterations;

    std::cout << "\nPerformance Results (" << width << "x" << height << ", "
              << iterations << " iterations):" << std::endl;
    for (int i = 0; i < 4; i++) {
        float bandwidth = (3.0f * N * sizeof(float) / 1e9f) / (times[i] / 1000.0f);
        std::cout << "  " << names[i] << ": " << times[i] << " ms ("
                  << bandwidth << " GB/s)" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
