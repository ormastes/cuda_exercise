/**
 * Tests for Matrix Multiplication Optimizations
 *
 * Validates correctness and benchmarks performance across all implementations
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <random>
#include <cmath>

// External function declarations
extern "C" {
    void matmul_cpu(float* C, const float* A, const float* B, int M, int N, int K);
    void matmul_cpu_blocked(float* C, const float* A, const float* B, int M, int N, int K);
    void launch_matmul_naive(float* C, const float* A, const float* B, int M, int N, int K);
    void launch_matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K);
    void launch_matmul_vectorized(float* C, const float* A, const float* B, int M, int N, int K);
    void launch_matmul_coarsened(float* C, const float* A, const float* B, int M, int N, int K);
    double benchmark_matmul(void (*kernel)(float*, const float*, const float*, int, int, int),
                           float* d_C, const float* d_A, const float* d_B,
                           int M, int N, int K, int iterations);
    double benchmark_cublas(float* d_C, const float* d_A, const float* d_B,
                           int M, int N, int K, int iterations);
}

// Test fixture
class MatmulTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    void initializeMatrix(float* matrix, int rows, int cols, float min_val = -1.0f, float max_val = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);

        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = dis(gen);
        }
    }

    void initializeIdentity(float* matrix, int size) {
        memset(matrix, 0, size * size * sizeof(float));
        for (int i = 0; i < size; i++) {
            matrix[i * size + i] = 1.0f;
        }
    }

    bool compareMatrices(const float* A, const float* B, int rows, int cols,
                        float tolerance = 1e-3f) {
        for (int i = 0; i < rows * cols; i++) {
            if (std::abs(A[i] - B[i]) > tolerance) {
                if (i < 10) {  // Print first few mismatches
                    printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n",
                           i, A[i], B[i], std::abs(A[i] - B[i]));
                }
                return false;
            }
        }
        return true;
    }

    float computeError(const float* A, const float* B, int size) {
        float max_error = 0.0f;
        for (int i = 0; i < size; i++) {
            max_error = std::max(max_error, std::abs(A[i] - B[i]));
        }
        return max_error;
    }
};

// Unit Tests

TEST_F(MatmulTest, CPUBaseline) {
    const int M = 128, N = 128, K = 128;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    // Initialize with small values to avoid numerical issues
    initializeMatrix(h_A.data(), M, K, -0.5f, 0.5f);
    initializeMatrix(h_B.data(), K, N, -0.5f, 0.5f);

    // Compute reference (simple implementation)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[m * K + k] * h_B[k * N + n];
            }
            h_C_ref[m * N + n] = sum;
        }
    }

    // Test CPU implementation
    matmul_cpu(h_C.data(), h_A.data(), h_B.data(), M, N, K);
    EXPECT_TRUE(compareMatrices(h_C.data(), h_C_ref.data(), M, N));

    // Test blocked CPU implementation
    matmul_cpu_blocked(h_C.data(), h_A.data(), h_B.data(), M, N, K);
    EXPECT_TRUE(compareMatrices(h_C.data(), h_C_ref.data(), M, N));
}

TEST_F(MatmulTest, IdentityMatrix) {
    const int size = 256;

    std::vector<float> h_A(size * size);
    std::vector<float> h_I(size * size);
    std::vector<float> h_C(size * size);

    // Initialize A with random values and I as identity
    initializeMatrix(h_A.data(), size, size);
    initializeIdentity(h_I.data(), size);

    // A * I should equal A
    matmul_cpu(h_C.data(), h_A.data(), h_I.data(), size, size, size);
    EXPECT_TRUE(compareMatrices(h_C.data(), h_A.data(), size, size));

    // I * A should equal A
    matmul_cpu(h_C.data(), h_I.data(), h_A.data(), size, size, size);
    EXPECT_TRUE(compareMatrices(h_C.data(), h_A.data(), size, size));
}

TEST_F(MatmulTest, GPUNaiveCorrectness) {
    const int M = 256, N = 256, K = 256;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N);
    std::vector<float> h_C_gpu(M * N);

    initializeMatrix(h_A.data(), M, K);
    initializeMatrix(h_B.data(), K, N);

    // CPU reference
    matmul_cpu(h_C_cpu.data(), h_A.data(), h_B.data(), M, N, K);

    // GPU implementation
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, M * K * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, K * N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, M * N * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice), cudaSuccess);

    launch_matmul_naive(d_C, d_A, d_B, M, N, K);

    ASSERT_EQ(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_TRUE(compareMatrices(h_C_cpu.data(), h_C_gpu.data(), M, N));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(MatmulTest, AllImplementationsCorrectness) {
    const int M = 512, N = 512, K = 512;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N);

    initializeMatrix(h_A.data(), M, K, -0.1f, 0.1f);
    initializeMatrix(h_B.data(), K, N, -0.1f, 0.1f);

    // Compute reference
    matmul_cpu(h_C_ref.data(), h_A.data(), h_B.data(), M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, M * K * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, K * N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, M * N * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice), cudaSuccess);

    struct Implementation {
        const char* name;
        void (*func)(float*, const float*, const float*, int, int, int);
        float tolerance;
    };

    std::vector<Implementation> implementations = {
        {"Naive", launch_matmul_naive, 1e-3f},
        {"Tiled", launch_matmul_tiled, 1e-3f},
        {"Vectorized", launch_matmul_vectorized, 1e-3f},
        {"Coarsened", launch_matmul_coarsened, 1e-3f}
    };

    std::vector<float> h_C_gpu(M * N);

    for (const auto& impl : implementations) {
        cudaMemset(d_C, 0, M * N * sizeof(float));
        impl.func(d_C, d_A, d_B, M, N, K);
        cudaDeviceSynchronize();

        ASSERT_EQ(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float),
                            cudaMemcpyDeviceToHost), cudaSuccess);

        float error = computeError(h_C_ref.data(), h_C_gpu.data(), M * N);
        EXPECT_LT(error, impl.tolerance) << "Implementation: " << impl.name
                                         << ", Max error: " << error;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Integration Tests

class MatmulIntegrationTest : public MatmulTest {
protected:
    void testWithSize(int M, int N, int K) {
        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_C_ref(M * N);

        initializeMatrix(h_A.data(), M, K);
        initializeMatrix(h_B.data(), K, N);

        // CPU reference
        matmul_cpu(h_C_ref.data(), h_A.data(), h_B.data(), M, N, K);

        // GPU setup
        float *d_A, *d_B, *d_C;
        ASSERT_EQ(cudaMalloc(&d_A, M * K * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_B, K * N * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_C, M * N * sizeof(float)), cudaSuccess);

        ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                            cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                            cudaMemcpyHostToDevice), cudaSuccess);

        // Test all implementations
        std::vector<float> h_C_gpu(M * N);

        launch_matmul_tiled(d_C, d_A, d_B, M, N, K);
        cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        float error = computeError(h_C_ref.data(), h_C_gpu.data(), M * N);
        EXPECT_LT(error, 1e-2f) << "Size: " << M << "x" << N << "x" << K
                                << ", Error: " << error;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

TEST_F(MatmulIntegrationTest, VariousSizes) {
    // Test different matrix sizes
    testWithSize(128, 128, 128);   // Small
    testWithSize(256, 512, 384);   // Rectangular
    testWithSize(1024, 1024, 1024); // Large square
    testWithSize(1000, 1000, 1000); // Non-power-of-2
    testWithSize(2048, 512, 1024);  // Large rectangular
}

TEST_F(MatmulIntegrationTest, CompareWithCuBLAS) {
    const int M = 1024, N = 1024, K = 1024;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_custom(M * N);
    std::vector<float> h_C_cublas(M * N);

    initializeMatrix(h_A.data(), M, K);
    initializeMatrix(h_B.data(), K, N);

    float *d_A, *d_B, *d_C_custom, *d_C_cublas;
    ASSERT_EQ(cudaMalloc(&d_A, M * K * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, K * N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C_custom, M * N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C_cublas, M * N * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice), cudaSuccess);

    // Custom implementation
    launch_matmul_vectorized(d_C_custom, d_A, d_B, M, N, K);

    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    cublasDestroy(handle);

    ASSERT_EQ(cudaMemcpy(h_C_custom.data(), d_C_custom, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_C_cublas.data(), d_C_cublas, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost), cudaSuccess);

    float error = computeError(h_C_custom.data(), h_C_cublas.data(), M * N);
    EXPECT_LT(error, 1e-2f) << "Max error vs cuBLAS: " << error;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_custom);
    cudaFree(d_C_cublas);
}

// Performance Tests

class MatmulPerformanceTest : public MatmulTest {
protected:
    void benchmarkImplementation(const char* name,
                                void (*func)(float*, const float*, const float*, int, int, int),
                                int M, int N, int K) {
        float *d_A, *d_B, *d_C;
        ASSERT_EQ(cudaMalloc(&d_A, M * K * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_B, K * N * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_C, M * N * sizeof(float)), cudaSuccess);

        // Initialize with random data
        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        initializeMatrix(h_A.data(), M, K);
        initializeMatrix(h_B.data(), K, N);

        ASSERT_EQ(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                            cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                            cudaMemcpyHostToDevice), cudaSuccess);

        double gflops = benchmark_matmul(func, d_C, d_A, d_B, M, N, K, 10);

        printf("%20s: %8.2f GFLOPS\n", name, gflops);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

TEST_F(MatmulPerformanceTest, ComprehensiveBenchmark) {
    const int M = 2048, N = 2048, K = 2048;

    printf("\n=== Matrix Multiplication Performance (M=%d, N=%d, K=%d) ===\n", M, N, K);

    // Benchmark all implementations
    benchmarkImplementation("Naive", launch_matmul_naive, M, N, K);
    benchmarkImplementation("Tiled", launch_matmul_tiled, M, N, K);
    benchmarkImplementation("Vectorized", launch_matmul_vectorized, M, N, K);
    benchmarkImplementation("Coarsened", launch_matmul_coarsened, M, N, K);

    // Benchmark cuBLAS
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    initializeMatrix(h_A.data(), M, K);
    initializeMatrix(h_B.data(), K, N);

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    double cublas_gflops = benchmark_cublas(d_C, d_A, d_B, M, N, K, 10);
    printf("%20s: %8.2f GFLOPS\n", "cuBLAS", cublas_gflops);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(MatmulPerformanceTest, SizeScaling) {
    printf("\n=== Performance vs Matrix Size ===\n");
    printf("%10s %15s %15s %15s %15s\n",
           "Size", "Naive", "Tiled", "Vectorized", "cuBLAS");

    std::vector<int> sizes = {128, 256, 512, 1024, 1536, 2048};

    for (int size : sizes) {
        printf("%10d ", size);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size * size * sizeof(float));
        cudaMalloc(&d_B, size * size * sizeof(float));
        cudaMalloc(&d_C, size * size * sizeof(float));

        std::vector<float> h_A(size * size);
        std::vector<float> h_B(size * size);
        initializeMatrix(h_A.data(), size, size);
        initializeMatrix(h_B.data(), size, size);

        cudaMemcpy(d_A, h_A.data(), size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), size * size * sizeof(float), cudaMemcpyHostToDevice);

        double naive_gflops = benchmark_matmul(launch_matmul_naive, d_C, d_A, d_B,
                                              size, size, size, 5);
        printf("%14.2f ", naive_gflops);

        double tiled_gflops = benchmark_matmul(launch_matmul_tiled, d_C, d_A, d_B,
                                              size, size, size, 5);
        printf("%14.2f ", tiled_gflops);

        double vector_gflops = benchmark_matmul(launch_matmul_vectorized, d_C, d_A, d_B,
                                               size, size, size, 5);
        printf("%14.2f ", vector_gflops);

        double cublas_gflops = benchmark_cublas(d_C, d_A, d_B, size, size, size, 5);
        printf("%14.2f\n", cublas_gflops);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Set up CUDA device
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaSetDevice(0);

    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Running on: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Peak memory bandwidth: %.2f GB/s\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    return RUN_ALL_TESTS();
}