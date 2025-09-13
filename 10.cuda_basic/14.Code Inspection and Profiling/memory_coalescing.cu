// memory_coalescing.cu - Compare memory access patterns
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Poor memory access pattern - strided access
__global__ void uncoalescedAccess(float* data, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int idx = tid * stride;  // Strided access - poor coalescing
        if (idx < n * stride) {
            data[idx] = idx * 2.0f;
        }
    }
}

// Good memory access pattern - sequential access
__global__ void coalescedAccess(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Sequential access
    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}

// Optimized with shared memory for data reuse
__global__ void optimizedAccess(float* output, const float* input, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Coalesced read to shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();

    // Process in shared memory (example: simple smoothing)
    if (tid > 0 && tid < blockDim.x - 1 && idx < n) {
        float result = 0.25f * sdata[tid - 1] +
                      0.5f * sdata[tid] +
                      0.25f * sdata[tid + 1];
        output[idx] = result;
    } else if (idx < n) {
        output[idx] = sdata[tid];
    }
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const int STRIDE = 32;      // Stride for uncoalesced access
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    printf("=== Memory Coalescing Performance Test ===\n");
    printf("Array size: %d elements\n", N);
    printf("Block size: %d threads\n", THREADS);
    printf("Grid size: %d blocks\n", BLOCKS);
    printf("Stride for uncoalesced: %d\n\n", STRIDE);

    // Allocate memory
    float *d_data1, *d_data2, *d_input, *d_output;
    size_t size = N * sizeof(float);
    size_t strided_size = N * STRIDE * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_data1, strided_size));
    CHECK_CUDA(cudaMalloc(&d_data2, size));
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));

    // Initialize data
    CHECK_CUDA(cudaMemset(d_data1, 0, strided_size));
    CHECK_CUDA(cudaMemset(d_data2, 0, size));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Test 1: Uncoalesced access
    printf("Testing uncoalesced access (strided)...\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        uncoalescedAccess<<<BLOCKS, THREADS>>>(d_data1, STRIDE, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float uncoalesced_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&uncoalesced_time, start, stop));
    uncoalesced_time /= 100.0f;

    // Test 2: Coalesced access
    printf("Testing coalesced access (sequential)...\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        coalescedAccess<<<BLOCKS, THREADS>>>(d_data2, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float coalesced_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&coalesced_time, start, stop));
    coalesced_time /= 100.0f;

    // Test 3: Optimized with shared memory
    printf("Testing optimized access (shared memory)...\n");
    size_t shared_mem_size = THREADS * sizeof(float);
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        optimizedAccess<<<BLOCKS, THREADS, shared_mem_size>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float optimized_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&optimized_time, start, stop));
    optimized_time /= 100.0f;

    // Calculate bandwidth
    float gb_transferred = (N * sizeof(float) * 2) / (1024.0f * 1024.0f * 1024.0f); // Read + Write

    printf("\n=== Results ===\n");
    printf("Uncoalesced access time: %.3f ms\n", uncoalesced_time);
    printf("Coalesced access time:   %.3f ms (%.2fx faster)\n",
           coalesced_time, uncoalesced_time / coalesced_time);
    printf("Optimized access time:   %.3f ms (%.2fx faster)\n",
           optimized_time, uncoalesced_time / optimized_time);

    printf("\n=== Effective Bandwidth ===\n");
    printf("Uncoalesced: %.2f GB/s\n", gb_transferred / (uncoalesced_time / 1000.0f));
    printf("Coalesced:   %.2f GB/s\n", gb_transferred / (coalesced_time / 1000.0f));
    printf("Optimized:   %.2f GB/s\n", gb_transferred / (optimized_time / 1000.0f));

    // Cleanup
    CHECK_CUDA(cudaFree(d_data1));
    CHECK_CUDA(cudaFree(d_data2));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\n=== Profiling Instructions ===\n");
    printf("To analyze memory patterns:\n");
    printf("1. ncu --metrics gld_efficiency,gst_efficiency ./memory_coalescing\n");
    printf("2. ncu --set detailed ./memory_coalescing\n");
    printf("3. nsys profile --stats=true ./memory_coalescing\n");

    return 0;
}