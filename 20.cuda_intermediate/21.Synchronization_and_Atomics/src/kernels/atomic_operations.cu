// atomic_operations.cu - Comprehensive atomic operations demonstration
#include <cuda_runtime.h>
#include <stdio.h>
#include "../../../../00.cuda_custom_lib/cuda_utils.h"

// Atomic addition examples
__global__ void atomic_add_int(int* counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(counter, 1);
    }
}

__global__ void atomic_add_float(float* sum, float* values, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(sum, values[tid]);
    }
}

// Atomic min/max operations
__global__ void atomic_min_max(int* min_val, int* max_val, int* values, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicMin(min_val, values[tid]);
        atomicMax(max_val, values[tid]);
    }
}

// Atomic Compare-And-Swap (CAS) for lock implementation
__device__ void acquire_lock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // Busy wait - in production, add exponential backoff
        __threadfence();
    }
}

__device__ void release_lock(int* lock) {
    __threadfence();  // Ensure all writes are visible
    atomicExch(lock, 0);
}

__global__ void critical_section_example(int* shared_counter, int* lock, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        acquire_lock(lock);

        // Critical section
        int old_val = *shared_counter;
        __threadfence();  // Ensure read completes
        *shared_counter = old_val + 1;

        release_lock(lock);
    }
}

// Atomic operations on shared memory for histogram
__global__ void histogram_atomic(int* histogram, int* data, int n, int num_bins) {
    extern __shared__ int shared_hist[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared histogram
    if (tid < num_bins) {
        shared_hist[tid] = 0;
    }
    __syncthreads();

    // Accumulate in shared memory
    if (gid < n) {
        int bin = data[gid] % num_bins;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();

    // Write back to global memory
    if (tid < num_bins) {
        atomicAdd(&histogram[tid], shared_hist[tid]);
    }
}

// Advanced: Atomic operations with memory ordering
__global__ void atomic_with_ordering(int* flag, int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        // Producer thread
        for (int i = 0; i < n; i++) {
            data[i] = i * i;
        }
        __threadfence();  // Release semantics
        atomicExch(flag, 1);  // Signal data ready
    } else if (tid == 1) {
        // Consumer thread
        while (atomicAdd(flag, 0) == 0) {
            // Wait for flag
        }
        __threadfence();  // Acquire semantics

        // Now safe to read data
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }
        data[n] = sum;  // Store result
    }
}

// Atomic operations for double precision
__global__ void atomic_add_double(double* sum, double* values, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Custom atomic add for double precision
        unsigned long long int* address = (unsigned long long int*)sum;
        unsigned long long int old, assumed;

        do {
            assumed = old = *address;
            old = atomicCAS(address, assumed,
                          __double_as_longlong(values[tid] +
                          __longlong_as_double(assumed)));
        } while (assumed != old);
    }
}

// Warp-level atomic reduction
__global__ void warp_atomic_reduction(int* output, int* input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;  // Lane within warp
    int warp_id = tid >> 5;  // Warp ID

    int value = (tid < n) ? input[tid] : 0;

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    // Only lane 0 performs atomic add
    if (lane == 0) {
        atomicAdd(output, value);
    }
}

// Performance comparison functions
__global__ void naive_counter_increment(int* counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        (*counter)++;  // Race condition - wrong result
    }
}

__global__ void atomic_counter_increment(int* counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(counter, 1);  // Correct atomic increment
    }
}

#ifndef BUILDING_LIB
// Example usage and testing
int main() {
    const int N = 1000000;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Test atomic counter
    int *d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    atomic_add_int<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_counter, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_counter;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Atomic counter result: %d (expected: %d)\n", h_counter, N);

    // Test histogram
    const int NUM_BINS = 256;
    int *d_histogram, *d_data;
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));

    // Initialize data with random values
    int* h_data = new int[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1000;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    histogram_atomic<<<NUM_BLOCKS, BLOCK_SIZE, NUM_BINS * sizeof(int)>>>(
        d_histogram, d_data, N, NUM_BINS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_histogram = new int[NUM_BINS];
    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int),
                         cudaMemcpyDeviceToHost));

    // Verify histogram sum
    int total = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        total += h_histogram[i];
    }
    printf("Histogram total: %d (expected: %d)\n", total, N);

    // Cleanup
    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;
    delete[] h_histogram;

    return 0;
}
#endif