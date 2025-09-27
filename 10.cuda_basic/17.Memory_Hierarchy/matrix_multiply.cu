// matrix_multiply.cu - Memory Hierarchy Demonstration with Matrix Multiplication
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>

// Forward declarations
__global__ void matmul_naive(const float* A, const float* B, float* C, int N);
__global__ void matmul_coalesced(const float* A, const float* B, float* C, int N);
__global__ void matmul_shared(const float* A, const float* B, float* C, int N);
__global__ void matmul_shared_bank_conflict_free(const float* A, const float* B, float* C, int N);

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Naive matrix multiplication - demonstrates poor memory access patterns
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Coalesced memory access pattern - better performance
__global__ void matmul_coalesced(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        // Access pattern optimized for coalescing
        for (int k = 0; k < N; k++) {
            // A is accessed row-wise (coalesced)
            // B is accessed column-wise (not coalesced, but cached)
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;  // Coalesced write
    }
}

// Shared memory tiling - significant performance improvement
#define TILE_SIZE 16

__global__ void matmul_shared(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile from A into shared memory
        if (row < N && (tile * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        if (col < N && (tile * TILE_SIZE + ty) < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Bank conflict-free shared memory access with padding
__global__ void matmul_shared_bank_conflict_free(const float* A, const float* B, float* C, int N) {
    // Add padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Coalesced load from global memory
        if (row < N && (tile * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && (tile * TILE_SIZE + ty) < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute with no bank conflicts due to padding
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU matrix multiplication for verification
void matmul_cpu(const float* A, const float* B, float* C, int N) {
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

// Verify results
bool verify_results(const float* C_gpu, const float* C_cpu, int N) {
    const float epsilon = 1e-3f;
    for (int i = 0; i < N * N; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": GPU=" << C_gpu[i]
                     << ", CPU=" << C_cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Initialize matrix with values
void init_matrix(float* mat, int N, float value_scale = 1.0f) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)(rand() % 10) * value_scale / 10.0f;
    }
}

// Benchmark a kernel
template<typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, const float* d_A, const float* d_B,
                       float* d_C, int N, dim3 gridSize, dim3 blockSize,
                       const char* kernel_name, int iterations = 10) {
    // Warmup
    for (int i = 0; i < 3; i++) {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float avg_time = milliseconds / iterations;
    float gflops = (2.0f * N * N * N / 1e9f) / (avg_time / 1000.0f);

    std::cout << std::setw(35) << kernel_name << ": "
              << std::fixed << std::setprecision(3) << std::setw(8) << avg_time << " ms, "
              << std::fixed << std::setprecision(2) << std::setw(8) << gflops << " GFLOPS" << std::endl;

    return avg_time;
}

// Main benchmark function
void run_memory_hierarchy_benchmark(int N = 512) {
    std::cout << "\n=== Memory Hierarchy Benchmark ===" << std::endl;
    std::cout << "Matrix size: " << N << " x " << N << std::endl;

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    float *h_C_cpu = new float[N * N];

    // Initialize matrices
    init_matrix(h_A, N, 0.5f);
    init_matrix(h_B, N, 0.5f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Setup grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    std::cout << "\nKernel Performance:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Benchmark different implementations
    benchmark_kernel(matmul_naive, d_A, d_B, d_C, N, gridSize, blockSize,
                    "Naive (poor memory access)");

    benchmark_kernel(matmul_coalesced, d_A, d_B, d_C, N, gridSize, blockSize,
                    "Coalesced (better access pattern)");

    benchmark_kernel(matmul_shared, d_A, d_B, d_C, N, gridSize, blockSize,
                    "Shared Memory (tiled)");

    benchmark_kernel(matmul_shared_bank_conflict_free, d_A, d_B, d_C, N, gridSize, blockSize,
                    "Shared Memory (bank-conflict free)");

    // Verify correctness
    std::cout << "\nVerifying correctness..." << std::endl;
    matmul_shared_bank_conflict_free<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU reference (only for small matrices)
    if (N <= 256) {
        matmul_cpu(h_A, h_B, h_C_cpu, N);
        if (verify_results(h_C, h_C_cpu, N)) {
            std::cout << "Results verified: PASS" << std::endl;
        } else {
            std::cout << "Results verification: FAIL" << std::endl;
        }
    } else {
        std::cout << "Skipping CPU verification for large matrix" << std::endl;
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

// Memory access pattern demonstration
__global__ void demonstrate_strided_access(float* data, int stride, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Strided access pattern (poor for coalescing)
        data[tid * stride] = tid;
    }
}

__global__ void demonstrate_coalesced_access(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Sequential access pattern (good for coalescing)
        data[tid] = tid;
    }
}

// Demonstrate memory access patterns
void demonstrate_memory_patterns() {
    std::cout << "\n=== Memory Access Pattern Demonstration ===" << std::endl;

    const int N = 1024 * 1024;
    const int stride = 32;
    size_t bytes = N * stride * sizeof(float);

    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Benchmark strided access
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    demonstrate_strided_access<<<gridSize, blockSize>>>(d_data, stride, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float strided_time;
    CHECK_CUDA(cudaEventElapsedTime(&strided_time, start, stop));

    // Benchmark coalesced access
    CHECK_CUDA(cudaEventRecord(start));
    demonstrate_coalesced_access<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float coalesced_time;
    CHECK_CUDA(cudaEventElapsedTime(&coalesced_time, start, stop));

    std::cout << "Strided access time: " << strided_time << " ms" << std::endl;
    std::cout << "Coalesced access time: " << coalesced_time << " ms" << std::endl;
    std::cout << "Speedup from coalescing: " << strided_time / coalesced_time << "x" << std::endl;

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

#ifndef BUILDING_LIB
int main(int argc, char** argv) {
    // Check CUDA device
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

    // Parse command line arguments
    int matrix_size = 512;
    if (argc > 1) {
        matrix_size = atoi(argv[1]);
    }

    // Run demonstrations
    demonstrate_memory_patterns();
    run_memory_hierarchy_benchmark(matrix_size);

    std::cout << "\n=== Memory Hierarchy Demo Complete ===" << std::endl;

    return 0;
}
#endif // BUILDING_LIB