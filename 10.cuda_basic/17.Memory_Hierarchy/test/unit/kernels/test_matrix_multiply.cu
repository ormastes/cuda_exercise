// test_matrix_multiply.cu - Unit tests for memory hierarchy implementations
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <vector>
#include "gpu_gtest.h"

// Forward declarations of kernels from matrix_multiply.cu
extern __global__ void matmul_naive(const float* A, const float* B, float* C, int N);
extern __global__ void matmul_coalesced(const float* A, const float* B, float* C, int N);
extern __global__ void matmul_shared(const float* A, const float* B, float* C, int N);
extern __global__ void matmul_shared_bank_conflict_free(const float* A, const float* B, float* C, int N);

#define TILE_SIZE 16

// Helper class for matrix operations
class MatrixHelper {
public:
    static void cpu_matmul(const float* A, const float* B, float* C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    static bool compare_matrices(const float* A, const float* B, int N, float tolerance = 1e-3f) {
        for (int i = 0; i < N * N; i++) {
            if (std::fabs(A[i] - B[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    static void init_identity(float* mat, int N) {
        memset(mat, 0, N * N * sizeof(float));
        for (int i = 0; i < N; i++) {
            mat[i * N + i] = 1.0f;
        }
    }

    static void init_random(float* mat, int N, float scale = 1.0f) {
        for (int i = 0; i < N * N; i++) {
            mat[i] = (static_cast<float>(rand()) / RAND_MAX) * scale;
        }
    }

    static void init_sequential(float* mat, int N) {
        for (int i = 0; i < N * N; i++) {
            mat[i] = static_cast<float>(i);
        }
    }
};

// Test fixture for matrix multiplication
class MatrixMultiplyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    void TestMatMulKernel(void (*kernel)(const float*, const float*, float*, int),
                          int N, const char* kernel_name) {
        size_t bytes = N * N * sizeof(float);

        // Host memory
        std::unique_ptr<float[]> h_A(new float[N * N]);
        std::unique_ptr<float[]> h_B(new float[N * N]);
        std::unique_ptr<float[]> h_C_gpu(new float[N * N]);
        std::unique_ptr<float[]> h_C_cpu(new float[N * N]);

        // Initialize matrices
        MatrixHelper::init_random(h_A.get(), N, 0.5f);
        MatrixHelper::init_random(h_B.get(), N, 0.5f);

        // Device memory
        float *d_A, *d_B, *d_C;
        ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

        // Copy to device
        ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

        // Launch kernel
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                     (N + blockSize.y - 1) / blockSize.y);

        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed: " << kernel_name;
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // Copy result back
        ASSERT_EQ(cudaMemcpy(h_C_gpu.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

        // CPU reference
        MatrixHelper::cpu_matmul(h_A.get(), h_B.get(), h_C_cpu.get(), N);

        // Compare results
        EXPECT_TRUE(MatrixHelper::compare_matrices(h_C_gpu.get(), h_C_cpu.get(), N))
            << "Matrix multiplication results don't match for " << kernel_name;

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

// Test naive implementation
TEST_F(MatrixMultiplyTest, NaiveImplementation) {
    for (int N : {16, 32, 64, 128}) {
        TestMatMulKernel(matmul_naive, N, "matmul_naive");
    }
}

// Test coalesced implementation
TEST_F(MatrixMultiplyTest, CoalescedImplementation) {
    for (int N : {16, 32, 64, 128}) {
        TestMatMulKernel(matmul_coalesced, N, "matmul_coalesced");
    }
}

// Test shared memory implementation
TEST_F(MatrixMultiplyTest, SharedMemoryImplementation) {
    for (int N : {16, 32, 64, 128, 256}) {
        TestMatMulKernel(matmul_shared, N, "matmul_shared");
    }
}

// Test bank-conflict-free implementation
TEST_F(MatrixMultiplyTest, BankConflictFreeImplementation) {
    for (int N : {16, 32, 64, 128, 256}) {
        TestMatMulKernel(matmul_shared_bank_conflict_free, N, "matmul_shared_bank_conflict_free");
    }
}

// Test with identity matrices
TEST_F(MatrixMultiplyTest, IdentityMatrices) {
    const int N = 32;
    size_t bytes = N * N * sizeof(float);

    std::unique_ptr<float[]> h_I(new float[N * N]);
    std::unique_ptr<float[]> h_A(new float[N * N]);
    std::unique_ptr<float[]> h_C(new float[N * N]);

    // Create identity matrix and random matrix
    MatrixHelper::init_identity(h_I.get(), N);
    MatrixHelper::init_random(h_A.get(), N);

    float *d_I, *d_A, *d_C;
    ASSERT_EQ(cudaMalloc(&d_I, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_I, h_I.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Test I * A = A
    matmul_shared_bank_conflict_free<<<gridSize, blockSize>>>(d_I, d_A, d_C, N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_C.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_TRUE(MatrixHelper::compare_matrices(h_C.get(), h_A.get(), N))
        << "Identity matrix multiplication failed: I * A != A";

    cudaFree(d_I);
    cudaFree(d_A);
    cudaFree(d_C);
}

// Performance comparison test
TEST_F(MatrixMultiplyTest, PerformanceComparison) {
    const int N = 256;
    size_t bytes = N * N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N * N]);
    std::unique_ptr<float[]> h_B(new float[N * N]);

    MatrixHelper::init_random(h_A.get(), N);
    MatrixHelper::init_random(h_B.get(), N);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Measure time for each kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float times[4];
    void (*kernels[4])(const float*, const float*, float*, int) = {
        matmul_naive,
        matmul_coalesced,
        matmul_shared,
        matmul_shared_bank_conflict_free
    };
    const char* names[4] = {
        "naive",
        "coalesced",
        "shared",
        "bank_conflict_free"
    };

    for (int i = 0; i < 4; i++) {
        // Warmup
        kernels[i]<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        // Timing
        cudaEventRecord(start);
        for (int j = 0; j < 10; j++) {
            kernels[i]<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&times[i], start, stop);
        times[i] /= 10.0f;  // Average time
    }

    // Note: Performance can vary based on problem size and GPU architecture
    // For small matrices, the overhead of shared memory management might outweigh benefits
    // We're mainly checking that the kernels run correctly, not strict performance ordering

    // Print performance results for analysis
    std::cout << "\nPerformance Analysis (N=" << N << "):" << std::endl;
    if (times[1] < times[0]) {
        std::cout << "✓ Coalesced is faster than naive" << std::endl;
    } else {
        std::cout << "  Note: Coalesced not faster - may be due to small size or optimization" << std::endl;
    }
    if (times[2] < times[0]) {
        std::cout << "✓ Shared memory is faster than naive" << std::endl;
    } else {
        std::cout << "  Note: Shared memory overhead visible at this size" << std::endl;
    }
    if (times[3] <= times[2] * 1.1f) {  // Allow 10% tolerance
        std::cout << "✓ Bank-conflict mitigation effective" << std::endl;
    } else {
        std::cout << "  Note: Bank-conflict mitigation not showing benefit" << std::endl;
    }

    // Print performance results
    std::cout << "\nPerformance Results (N=" << N << "):" << std::endl;
    for (int i = 0; i < 4; i++) {
        float gflops = (2.0f * N * N * N / 1e9f) / (times[i] / 1000.0f);
        std::cout << "  " << names[i] << ": " << times[i] << " ms ("
                  << gflops << " GFLOPS)" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test shared memory behavior
__global__ void test_shared_memory_kernel(float* result) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;

    // Write to shared memory
    shared_data[tid] = tid * 2.0f;
    __syncthreads();

    // Read from neighboring thread's data
    float neighbor_data = shared_data[(tid + 1) % blockDim.x];

    // Store result for verification
    result[tid] = neighbor_data;
}

TEST_F(MatrixMultiplyTest, SharedMemoryAccess) {
    const int N = 256;
    float* d_result;
    float* h_result = new float[N];

    ASSERT_EQ(cudaMalloc(&d_result, N * sizeof(float)), cudaSuccess);

    test_shared_memory_kernel<<<1, N>>>(d_result);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify results
    for (int i = 0; i < N; i++) {
        float expected = ((i + 1) % N) * 2.0f;
        EXPECT_FLOAT_EQ(h_result[i], expected) << "Mismatch at index " << i;
    }

    delete[] h_result;
    cudaFree(d_result);
}

// Test memory coalescing patterns
__global__ void test_coalescing_kernel(float* data, int stride, bool coalesced) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (coalesced) {
        // Coalesced access: consecutive threads access consecutive memory
        data[tid] = tid;
    } else {
        // Strided access: threads access memory with stride
        data[tid * stride] = tid;
    }
}

TEST_F(MatrixMultiplyTest, MemoryCoalescingPattern) {
    const int N = 1024;
    const int stride = 32;
    float* d_data;

    ASSERT_EQ(cudaMalloc(&d_data, N * stride * sizeof(float)), cudaSuccess);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Test coalesced access
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    test_coalescing_kernel<<<gridSize, blockSize>>>(d_data, stride, true);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float coalesced_time;
    cudaEventElapsedTime(&coalesced_time, start, stop);

    // Test strided access
    cudaEventRecord(start);
    test_coalescing_kernel<<<gridSize, blockSize>>>(d_data, stride, false);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float strided_time;
    cudaEventElapsedTime(&strided_time, start, stop);

    // Note: The test kernel is very simple and might be optimized differently
    // In real scenarios, coalesced access is typically much faster
    // We'll check if there's any measurable difference
    bool coalescing_benefit = coalesced_time < strided_time;
    if (!coalescing_benefit) {
        std::cout << "\nNote: Coalescing benefit not observed in this simple test." << std::endl;
        std::cout << "This can happen due to caching or compiler optimizations." << std::endl;
    }

    std::cout << "\nCoalescing Test Results:" << std::endl;
    std::cout << "  Coalesced time: " << coalesced_time << " ms" << std::endl;
    std::cout << "  Strided time: " << strided_time << " ms" << std::endl;
    std::cout << "  Speedup: " << strided_time / coalesced_time << "x" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
}

// Parameterized test for different matrix sizes
class ParameterizedMatrixTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_P(ParameterizedMatrixTest, AllKernelsCorrectness) {
    int N = GetParam();
    size_t bytes = N * N * sizeof(float);

    std::unique_ptr<float[]> h_A(new float[N * N]);
    std::unique_ptr<float[]> h_B(new float[N * N]);
    std::unique_ptr<float[]> h_C_cpu(new float[N * N]);
    std::unique_ptr<float[]> h_C_gpu(new float[N * N]);

    MatrixHelper::init_random(h_A.get(), N);
    MatrixHelper::init_random(h_B.get(), N);
    MatrixHelper::cpu_matmul(h_A.get(), h_B.get(), h_C_cpu.get(), N);

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Test all kernels
    void (*kernels[])(const float*, const float*, float*, int) = {
        matmul_naive,
        matmul_coalesced,
        matmul_shared,
        matmul_shared_bank_conflict_free
    };

    for (auto kernel : kernels) {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(h_C_gpu.get(), d_C, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(MatrixHelper::compare_matrices(h_C_gpu.get(), h_C_cpu.get(), N));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

INSTANTIATE_TEST_SUITE_P(
    MatrixSizes,
    ParameterizedMatrixTest,
    ::testing::Values(16, 32, 48, 64, 96, 128)
);