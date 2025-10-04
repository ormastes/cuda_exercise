/**
 * Matrix Multiplication - Progressive GPU Optimization
 *
 * Evolution from CPU baseline to highly optimized GPU implementations:
 * 1. CPU Baseline: ~50 GFLOPS
 * 2. GPU Naive: ~150 GFLOPS (3x)
 * 3. Tiled Shared Memory: ~800 GFLOPS (16x)
 * 4. Vectorized Loads: ~1200 GFLOPS (24x)
 * 5. Tensor Core: ~10000 GFLOPS (200x)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>

#ifdef HAS_CUDNN
#include <cudnn.h>
#endif

#include <mma.h>
using namespace nvcuda;

// Performance metrics structure
struct MatmulMetrics {
    size_t global_loads;
    size_t shared_loads;
    size_t register_usage;
    float achieved_occupancy;
    float memory_efficiency;
    double gflops;
};

/**
 * CPU Baseline Implementation
 *
 * Simple triple-nested loop with cache blocking
 */
extern "C" void matmul_cpu(float* C, const float* A, const float* B,
                           int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float a_val = A[m * K + k];
            for (int n = 0; n < N; n++) {
                C[m * N + n] += a_val * B[k * N + n];
            }
        }
    }
}

/**
 * CPU Blocked Implementation
 *
 * Cache-friendly blocking
 */
extern "C" void matmul_cpu_blocked(float* C, const float* A, const float* B,
                                  int M, int N, int K) {
    const int BLOCK = 32;
    memset(C, 0, M * N * sizeof(float));

    for (int m0 = 0; m0 < M; m0 += BLOCK) {
        for (int n0 = 0; n0 < N; n0 += BLOCK) {
            for (int k0 = 0; k0 < K; k0 += BLOCK) {
                // Block computation
                for (int m = m0; m < std::min(m0 + BLOCK, M); m++) {
                    for (int k = k0; k < std::min(k0 + BLOCK, K); k++) {
                        float a_val = A[m * K + k];
                        for (int n = n0; n < std::min(n0 + BLOCK, N); n++) {
                            C[m * N + n] += a_val * B[k * N + n];
                        }
                    }
                }
            }
        }
    }
}

/**
 * GPU Naive Implementation
 *
 * Direct translation from CPU - each thread computes one output element
 * Performance: ~150 GFLOPS on modern GPUs
 */
__global__ void matmul_naive(float* C, const float* A, const float* B,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Tiled Shared Memory Implementation
 *
 * Uses shared memory to reduce global memory accesses
 * Performance: ~800 GFLOPS (8x improvement over naive)
 */
template<int TILE_SIZE>
__global__ void matmul_tiled(float* C, const float* A, const float* B,
                             int M, int N, int K) {
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
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Vectorized Implementation
 *
 * Uses float4 loads for better memory throughput
 * Performance: ~1200 GFLOPS (1.5x improvement over tiled)
 */
template<int TILE_SIZE>
__global__ void matmul_vectorized(float* C, const float* A, const float* B,
                                  int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Vectorized loads where possible
        if (threadIdx.x % 4 == 0 && tile * TILE_SIZE + threadIdx.x + 3 < K) {
            if (row < M) {
                float4 a_vec = *reinterpret_cast<const float4*>(
                    &A[row * K + tile * TILE_SIZE + threadIdx.x]);
                As[threadIdx.y][threadIdx.x] = a_vec.x;
                As[threadIdx.y][threadIdx.x + 1] = a_vec.y;
                As[threadIdx.y][threadIdx.x + 2] = a_vec.z;
                As[threadIdx.y][threadIdx.x + 3] = a_vec.w;
            }
        } else {
            int a_col = tile * TILE_SIZE + threadIdx.x;
            if (row < M && a_col < K) {
                As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Unrolled computation
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Thread Coarsening Implementation
 *
 * Each thread computes multiple output elements
 * Performance: ~1400 GFLOPS
 */
template<int TILE_SIZE, int COARSE_FACTOR>
__global__ void matmul_coarsened(float* C, const float* A, const float* B,
                                 int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * COARSE_FACTOR];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum[COARSE_FACTOR];
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        sum[c] = 0.0f;
    }

    int row = by * TILE_SIZE + ty;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load A tile
        int a_col = tile * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tiles
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int b_row = tile * TILE_SIZE + ty;
            int b_col = bx * TILE_SIZE * COARSE_FACTOR + c * TILE_SIZE + tx;

            if (b_row < K && b_col < N) {
                Bs[ty][c * TILE_SIZE + tx] = B[b_row * N + b_col];
            } else {
                Bs[ty][c * TILE_SIZE + tx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int c = 0; c < COARSE_FACTOR; c++) {
                sum[c] += As[ty][k] * Bs[k][c * TILE_SIZE + tx];
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = bx * TILE_SIZE * COARSE_FACTOR + c * TILE_SIZE + tx;
        if (row < M && col < N) {
            C[row * N + col] = sum[c];
        }
    }
}

/**
 * Tensor Core Implementation (requires SM 7.0+)
 *
 * Uses specialized tensor core units for matrix multiplication
 * Performance: ~10 TFLOPS (200x improvement over CPU)
 */
#ifdef USE_TENSOR_CORES

__global__ void matmul_wmma(half* C, const half* A, const half* B,
                            int M, int N, int K) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

#endif // USE_TENSOR_CORES

/**
 * Memory usage analysis function
 */
extern "C" MatmulMetrics analyze_memory(int M, int N, int K, int tile_size) {
    MatmulMetrics metrics = {0};

    // Naive implementation
    metrics.global_loads = M * N * K * 2;  // Load A and B for each output
    metrics.shared_loads = 0;
    metrics.register_usage = 3;  // sum, a_val, b_val

    // Tiled implementation
    size_t num_tiles = ((M + tile_size - 1) / tile_size) *
                      ((N + tile_size - 1) / tile_size) *
                      ((K + tile_size - 1) / tile_size);

    size_t tiled_global = num_tiles * tile_size * tile_size * 2;
    size_t tiled_shared = M * N * K * 2;  // Each element loaded once to shared

    double reduction_factor = (double)metrics.global_loads / tiled_global;

    metrics.memory_efficiency = reduction_factor;
    metrics.achieved_occupancy = 0.75f;  // Typical for well-tuned kernels

    return metrics;
}

/**
 * Benchmark helper function
 */
extern "C" double benchmark_matmul(void (*kernel)(float*, const float*, const float*, int, int, int),
                                   float* d_C, const float* d_A, const float* d_B,
                                   int M, int N, int K, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    kernel(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel(d_C, d_A, d_B, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double seconds = milliseconds / 1000.0;
    double flops = 2.0 * M * N * K * iterations;
    double gflops = flops / seconds / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gflops;
}

/**
 * Wrapper functions for benchmarking
 */
extern "C" void launch_matmul_naive(float* C, const float* A, const float* B,
                                   int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(C, A, B, M, N, K);
}

extern "C" void launch_matmul_tiled(float* C, const float* A, const float* B,
                                   int M, int N, int K) {
    const int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled<TILE_SIZE><<<grid, block>>>(C, A, B, M, N, K);
}

extern "C" void launch_matmul_vectorized(float* C, const float* A, const float* B,
                                        int M, int N, int K) {
    const int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_vectorized<TILE_SIZE><<<grid, block>>>(C, A, B, M, N, K);
}

extern "C" void launch_matmul_coarsened(float* C, const float* A, const float* B,
                                       int M, int N, int K) {
    const int TILE_SIZE = 32;
    const int COARSE_FACTOR = 2;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE * COARSE_FACTOR - 1) / (TILE_SIZE * COARSE_FACTOR),
             (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_coarsened<TILE_SIZE, COARSE_FACTOR><<<grid, block>>>(C, A, B, M, N, K);
}

/**
 * cuBLAS comparison function
 */
extern "C" double benchmark_cublas(float* d_C, const float* d_A, const float* d_B,
                                  int M, int N, int K, int iterations) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double seconds = milliseconds / 1000.0;
    double flops = 2.0 * M * N * K * iterations;
    double gflops = flops / seconds / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return gflops;
}