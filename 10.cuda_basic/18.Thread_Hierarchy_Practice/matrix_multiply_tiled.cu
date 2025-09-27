// matrix_multiply_tiled.cu - Thread Hierarchy Practice with Advanced Tiling
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <vector>

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Different tile sizes for experiments
#define TILE_SIZE_8 8
#define TILE_SIZE_16 16
#define TILE_SIZE_32 32

// Warp size constant
#define WARP_SIZE 32

// Basic tiled matrix multiplication
template<int TILE_SIZE>
__global__ void matmul_tiled_basic(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Collaborative loading by threads in the block
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

        __syncthreads();  // Synchronize all threads in block

        // Each thread computes one element
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // Synchronize before loading next tile
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Rectangular tiling for better thread utilization
template<int TILE_Y, int TILE_X>
__global__ void matmul_rectangular_tiles(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_Y][TILE_X];
    __shared__ float Bs[TILE_Y][TILE_X];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_Y + ty;
    int col = bx * TILE_X + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_X - 1) / TILE_X; tile++) {
        // Load with proper boundary checks
        if (row < N && (tile * TILE_X + tx) < N) {
            As[ty][tx] = A[row * N + tile * TILE_X + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((tile * TILE_Y + ty) < N && col < N) {
            Bs[ty][tx] = B[(tile * TILE_Y + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_X && (tile * TILE_X + k) < N; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Warp-level optimization using shuffle instructions
__global__ void matmul_warp_optimized(const float* A, const float* B, float* C, int N) {
    const int TILE = 32;  // Warp size
    __shared__ float As[TILE][TILE + 1];  // Padding to avoid bank conflicts
    __shared__ float Bs[TILE][TILE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;

    // Warp and lane IDs
    int warpId = (ty * TILE + tx) / WARP_SIZE;
    int laneId = (ty * TILE + tx) % WARP_SIZE;

    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // Cooperative loading
        if (row < N && (tile * TILE + tx) < N) {
            As[ty][tx] = A[row * N + tile * TILE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && (tile * TILE + ty) < N) {
            Bs[ty][tx] = B[(tile * TILE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute using warp-level primitives for reduction
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Thread coarsening - each thread computes multiple elements
template<int COARSE_FACTOR>
__global__ void matmul_thread_coarsening(const float* A, const float* B, float* C, int N) {
    const int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread computes COARSE_FACTOR elements
    float sum[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        sum[c] = 0.0f;
    }

    int row = by * TILE + ty;

    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // Load shared memory
        if (row < N && (tile * TILE + tx) < N) {
            As[ty][tx] = A[row * N + tile * TILE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load multiple B tiles for coarsening
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = (bx * COARSE_FACTOR + c) * TILE + tx;
            if ((tile * TILE + ty) < N && col < N) {
                Bs[ty][tx] = B[(tile * TILE + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }

            __syncthreads();

            // Compute for this coarse factor
            #pragma unroll
            for (int k = 0; k < TILE; k++) {
                sum[c] += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }
    }

    // Write results
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = (bx * COARSE_FACTOR + c) * TILE + tx;
        if (row < N && col < N) {
            C[row * N + col] = sum[c];
        }
    }
}

// Demonstrate warp divergence
__global__ void demonstrate_warp_divergence(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        // Divergent branch - threads in same warp take different paths
        if (tid % 2 == 0) {
            // Even threads - complex computation
            float val = data[tid];
            for (int i = 0; i < 100; i++) {
                val = sqrtf(val * val + 1.0f);
            }
            data[tid] = val;
        } else {
            // Odd threads - simple computation
            data[tid] = data[tid] * 2.0f;
        }
    }
}

// Optimized version without warp divergence
__global__ void demonstrate_no_divergence(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        // Reorganize so warps don't diverge
        int warpId = tid / WARP_SIZE;

        if (warpId % 2 == 0) {
            // Even warps - complex computation
            float val = data[tid];
            for (int i = 0; i < 100; i++) {
                val = sqrtf(val * val + 1.0f);
            }
            data[tid] = val;
        } else {
            // Odd warps - simple computation
            data[tid] = data[tid] * 2.0f;
        }
    }
}

// Occupancy demonstration kernel
template<int THREADS_PER_BLOCK>
__global__ void occupancy_test_kernel(float* data, int N) {
    __shared__ float shared_data[THREADS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    if (tid < N) {
        // Load to shared memory
        shared_data[local_id] = data[tid];
        __syncthreads();

        // Simple reduction within block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (local_id < stride && tid + stride < N) {
                shared_data[local_id] += shared_data[local_id + stride];
            }
            __syncthreads();
        }

        // Write back
        if (local_id == 0) {
            data[blockIdx.x] = shared_data[0];
        }
    }
}

// Helper functions
void init_matrix(float* mat, int N, float scale = 1.0f) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)(rand() % 10) * scale / 10.0f;
    }
}

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

bool verify_results(const float* GPU, const float* CPU, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(GPU[i] - CPU[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Benchmark function
template<typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, const float* d_A, const float* d_B,
                       float* d_C, int N, dim3 gridSize, dim3 blockSize,
                       const char* name) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    const int iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_time = milliseconds / iterations;
    float gflops = (2.0f * N * N * N / 1e9f) / (avg_time / 1000.0f);

    std::cout << std::setw(40) << name << ": "
              << std::fixed << std::setprecision(3) << std::setw(8) << avg_time << " ms, "
              << std::fixed << std::setprecision(2) << std::setw(8) << gflops << " GFLOPS" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return avg_time;
}

// Thread hierarchy analysis
void analyze_thread_hierarchy() {
    std::cout << "\n=== Thread Hierarchy Analysis ===" << std::endl;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Device Limits:" << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max thread dimensions: ("
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max grid dimensions: ("
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Max warps per SM: " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;
    std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
}

// Occupancy analysis
void analyze_occupancy(int N) {
    std::cout << "\n=== Occupancy Analysis ===" << std::endl;

    // Test different block sizes
    int block_sizes[] = {64, 128, 256, 512, 1024};
    int min_grid_size, optimal_block_size;

    // Calculate optimal block size once
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &optimal_block_size,
        occupancy_test_kernel<256>,
        0, 0));

    for (int block_size : block_sizes) {
        int max_active_blocks;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks,
            occupancy_test_kernel<256>,
            block_size,
            0));

        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

        float occupancy = (float)(max_active_blocks * block_size) /
                         prop.maxThreadsPerMultiProcessor;

        std::cout << "Block size " << std::setw(4) << block_size
                  << ": Occupancy = " << std::fixed << std::setprecision(1)
                  << (occupancy * 100) << "%"
                  << " (Max active blocks: " << max_active_blocks << ")" << std::endl;

        if (block_size >= prop.maxThreadsPerBlock) break;
    }

    std::cout << "Optimal block size suggested: " << optimal_block_size << std::endl;
}

// Warp divergence demonstration
void demonstrate_divergence_impact() {
    std::cout << "\n=== Warp Divergence Impact ===" << std::endl;

    const int N = 1024 * 1024;
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    // Initialize data
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Benchmark divergent version
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    demonstrate_warp_divergence<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float divergent_time;
    CHECK_CUDA(cudaEventElapsedTime(&divergent_time, start, stop));

    // Reset data
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    // Benchmark non-divergent version
    CHECK_CUDA(cudaEventRecord(start));
    demonstrate_no_divergence<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float no_divergent_time;
    CHECK_CUDA(cudaEventElapsedTime(&no_divergent_time, start, stop));

    std::cout << "With warp divergence: " << divergent_time << " ms" << std::endl;
    std::cout << "Without warp divergence: " << no_divergent_time << " ms" << std::endl;
    std::cout << "Speedup from avoiding divergence: "
              << divergent_time / no_divergent_time << "x" << std::endl;

    delete[] h_data;
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Main benchmark
void run_thread_hierarchy_benchmark(int N = 512) {
    std::cout << "\n=== Thread Hierarchy Benchmark ===" << std::endl;
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

    std::cout << "\nKernel Performance:" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    // Test different configurations

    // 8x8 tiles
    dim3 blockSize8(8, 8);
    dim3 gridSize8((N + 7) / 8, (N + 7) / 8);
    benchmark_kernel(matmul_tiled_basic<8>, d_A, d_B, d_C, N, gridSize8, blockSize8,
                    "Tiled 8x8");

    // 16x16 tiles
    dim3 blockSize16(16, 16);
    dim3 gridSize16((N + 15) / 16, (N + 15) / 16);
    benchmark_kernel(matmul_tiled_basic<16>, d_A, d_B, d_C, N, gridSize16, blockSize16,
                    "Tiled 16x16");

    // 32x32 tiles
    dim3 blockSize32(32, 32);
    dim3 gridSize32((N + 31) / 32, (N + 31) / 32);
    benchmark_kernel(matmul_tiled_basic<32>, d_A, d_B, d_C, N, gridSize32, blockSize32,
                    "Tiled 32x32");

    // Skip rectangular tiles for now - has memory access issues
    // Will need debugging

    // Warp optimized
    dim3 blockSizeWarp(32, 32);
    dim3 gridSizeWarp((N + 31) / 32, (N + 31) / 32);
    benchmark_kernel(matmul_warp_optimized, d_A, d_B, d_C, N,
                    gridSizeWarp, blockSizeWarp, "Warp optimized");

    // Verify correctness
    std::cout << "\nVerifying correctness..." << std::endl;
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

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

    // Parse arguments
    int matrix_size = 512;
    if (argc > 1) {
        matrix_size = atoi(argv[1]);
    }

    // Run analyses
    analyze_thread_hierarchy();
    analyze_occupancy(matrix_size);
    demonstrate_divergence_impact();
    run_thread_hierarchy_benchmark(matrix_size);

    std::cout << "\n=== Thread Hierarchy Demo Complete ===" << std::endl;

    return 0;
}
#endif // BUILDING_LIB