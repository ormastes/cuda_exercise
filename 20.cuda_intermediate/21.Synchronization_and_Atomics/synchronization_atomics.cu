// synchronization_atomics.cu - Advanced synchronization and atomic operations
#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ===============================
// Basic Synchronization Examples
// ===============================

// Kernel demonstrating __syncthreads()
__global__ void basic_sync_kernel(float* data, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    if (gid < N) {
        sdata[tid] = data[gid];
    } else {
        sdata[tid] = 0.0f;
    }

    // Synchronize before processing
    __syncthreads();

    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && gid + stride < N) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads(); // Essential for correctness
    }

    // Write result
    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}

// Kernel demonstrating warp-level primitives
__global__ void warp_sync_kernel(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (tid < N) {
        float value = data[tid];

        // Warp-level reduction using shuffle
        unsigned mask = 0xffffffff;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(mask, value, offset);
        }

        // Only lane 0 writes result
        if (lane == 0) {
            data[blockIdx.x * (blockDim.x / warpSize) + warp_id] = value;
        }
    }
}

// ===============================
// Atomic Operations Examples
// ===============================

// Global counter using atomics
__device__ unsigned int global_counter = 0;

// Kernel for atomic increment
__global__ void atomic_increment_kernel(int* counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicAdd(counter, 1);
    }
}

// Kernel for atomic histogram
__global__ void atomic_histogram_kernel(int* data, int* histogram, int N, int bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        int bin = data[tid] % bins;
        atomicAdd(&histogram[bin], 1);
    }
}

// Kernel demonstrating atomic compare-and-swap (CAS)
__global__ void atomic_cas_kernel(int* value, int compare, int val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        int old = atomicCAS(value, compare, val);
        printf("Thread %d: CAS returned %d, value is now %d\n", tid, old, *value);
    }
}

// Custom atomic max for floats using CAS
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
        if (old == assumed) break;
    }

    return __int_as_float(old);
}

// Kernel using custom atomic float max
__global__ void atomic_max_float_kernel(float* max_value, float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        atomicMaxFloat(max_value, data[tid]);
    }
}

// ===============================
// Lock-Free Data Structures
// ===============================

// Simple lock-free queue node
struct Node {
    int data;
    Node* next;
};

// Lock-free stack implementation
class DeviceStack {
public:
    Node* head;

    __device__ void push(Node* new_node) {
        Node* old_head;
        do {
            old_head = head;
            new_node->next = old_head;
        } while (atomicCAS((unsigned long long*)&head,
                          (unsigned long long)old_head,
                          (unsigned long long)new_node) !=
                 (unsigned long long)old_head);
    }

    __device__ Node* pop() {
        Node* old_head;
        Node* new_head;
        do {
            old_head = head;
            if (old_head == nullptr) return nullptr;
            new_head = old_head->next;
        } while (atomicCAS((unsigned long long*)&head,
                          (unsigned long long)old_head,
                          (unsigned long long)new_head) !=
                 (unsigned long long)old_head);
        return old_head;
    }
};

// ===============================
// Cooperative Groups Examples
// ===============================

// Kernel using cooperative groups
__global__ void cooperative_groups_kernel(float* data, int N) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    int tid = block.thread_rank();
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < N) {
        float value = data[gid];

        // Tile-level reduction
        for (int i = tile.size() / 2; i > 0; i /= 2) {
            value += tile.shfl_down(value, i);
        }

        // Only thread 0 of each tile writes
        if (tile.thread_rank() == 0) {
            data[blockIdx.x * (blockDim.x / 32) + tid / 32] = value;
        }
    }
}

// Grid-level synchronization (requires special launch)
__global__ void grid_sync_kernel(float* data, int N) {
    cg::grid_group grid = cg::this_grid();
    int tid = grid.thread_rank();

    if (tid < N) {
        // First phase: square all elements
        data[tid] = data[tid] * data[tid];

        // Grid-wide synchronization
        grid.sync();

        // Second phase: add neighboring elements
        if (tid < N - 1) {
            data[tid] = data[tid] + data[tid + 1];
        }
    }
}

// ===============================
// Memory Fence Examples
// ===============================

__device__ volatile int flag = 0;
__device__ float shared_data = 0.0f;

// Producer kernel
__global__ void producer_kernel(float value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        shared_data = value;
        __threadfence(); // Ensure write is visible to all threads
        atomicExch((int*)&flag, 1); // Signal completion
    }
}

// Consumer kernel
__global__ void consumer_kernel(float* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Wait for producer
        while (atomicAdd((int*)&flag, 0) == 0);

        __threadfence(); // Ensure we see the latest data
        *result = shared_data;
    }
}

// ===============================
// Spinlock Implementation
// ===============================

class SpinLock {
private:
    int* mutex;

public:
    __device__ SpinLock(int* m) : mutex(m) {}

    __device__ void lock() {
        while (atomicCAS(mutex, 0, 1) != 0);
        __threadfence();
    }

    __device__ void unlock() {
        __threadfence();
        atomicExch(mutex, 0);
    }
};

// Kernel using spinlock
__global__ void spinlock_kernel(int* counter, int* mutex, int iterations) {
    SpinLock lock(mutex);

    for (int i = 0; i < iterations; i++) {
        lock.lock();
        (*counter)++;  // Critical section
        lock.unlock();
    }
}

// ===============================
// Demonstration Functions
// ===============================

void demonstrate_basic_synchronization() {
    printf("\n=== Basic Synchronization ===\n");

    const int N = 1024;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    size_t bytes = N * sizeof(float);

    // Allocate and initialize data
    float* h_data = new float[N];
    float* d_data;

    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i + 1);
    }

    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Launch kernel with shared memory
    size_t shared_bytes = blockSize * sizeof(float);
    basic_sync_kernel<<<gridSize, blockSize, shared_bytes>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back
    float* h_result = new float[gridSize];
    CHECK_CUDA(cudaMemcpy(h_result, d_data, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify reduction
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        expected += h_data[i];
    }

    float actual = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        actual += h_result[i];
    }

    printf("Reduction result: %.0f (expected: %.0f)\n", actual, expected);
    printf("Result is %s\n", (fabsf(actual - expected) < 0.01f) ? "CORRECT" : "INCORRECT");

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    delete[] h_data;
    delete[] h_result;
}

void demonstrate_atomic_operations() {
    printf("\n=== Atomic Operations ===\n");

    // Test atomic increment
    int* d_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

    const int threads = 256;
    const int blocks = 100;
    const int iterations = 1000;

    atomic_increment_kernel<<<blocks, threads>>>(d_counter, iterations);
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_counter;
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    int expected_count = threads * blocks * iterations;
    printf("Atomic counter: %d (expected: %d)\n", h_counter, expected_count);
    printf("Result is %s\n", (h_counter == expected_count) ? "CORRECT" : "INCORRECT");

    // Test atomic histogram
    const int N = 10000;
    const int bins = 10;

    int* h_data = new int[N];
    int* d_data;
    int* d_histogram;

    // Generate random data
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 99);
    for (int i = 0; i < N; i++) {
        h_data[i] = dis(gen);
    }

    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_histogram, bins * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_histogram, 0, bins * sizeof(int)));

    // Launch histogram kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    atomic_histogram_kernel<<<gridSize, blockSize>>>(d_data, d_histogram, N, bins);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Get results
    int h_histogram[bins];
    CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, bins * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\nHistogram (10 bins):\n");
    int total = 0;
    for (int i = 0; i < bins; i++) {
        printf("  Bin %d: %d\n", i, h_histogram[i]);
        total += h_histogram[i];
    }
    printf("Total: %d (expected: %d)\n", total, N);

    // Cleanup
    CHECK_CUDA(cudaFree(d_counter));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histogram));
    delete[] h_data;
}

void demonstrate_memory_fences() {
    printf("\n=== Memory Fences ===\n");

    float test_value = 42.0f;
    float* d_result;

    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    // Reset flag
    int zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(flag, &zero, sizeof(int)));

    // Launch producer and consumer
    producer_kernel<<<1, 1>>>(test_value);
    consumer_kernel<<<1, 1>>>(d_result);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Producer-Consumer result: %.2f (expected: %.2f)\n", h_result, test_value);
    printf("Result is %s\n", (h_result == test_value) ? "CORRECT" : "INCORRECT");

    CHECK_CUDA(cudaFree(d_result));
}

void demonstrate_cooperative_groups() {
    printf("\n=== Cooperative Groups ===\n");

    const int N = 1024;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    size_t bytes = N * sizeof(float);

    float* h_data = new float[N];
    float* d_data;

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i + 1);
    }

    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Launch cooperative groups kernel
    cooperative_groups_kernel<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // For grid sync, we need special launch (check device capability)
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    if (prop.cooperativeLaunch) {
        printf("Device supports cooperative launch\n");

        // Reset data
        CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

        // Launch with grid synchronization
        void* kernelArgs[] = { (void*)&d_data, (void*)&N };
        CHECK_CUDA(cudaLaunchCooperativeKernel(
            (void*)grid_sync_kernel,
            gridSize, blockSize,
            kernelArgs, 0, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Get result
        float* h_result = new float[N];
        CHECK_CUDA(cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost));

        printf("Grid sync kernel completed successfully\n");
        delete[] h_result;
    } else {
        printf("Device does not support cooperative launch\n");
    }

    CHECK_CUDA(cudaFree(d_data));
    delete[] h_data;
}

void demonstrate_spinlock() {
    printf("\n=== Spinlock ===\n");

    int* d_counter;
    int* d_mutex;

    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_mutex, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_mutex, 0, sizeof(int)));

    const int threads = 32;
    const int blocks = 10;
    const int iterations = 100;

    auto start = std::chrono::high_resolution_clock::now();
    spinlock_kernel<<<blocks, threads>>>(d_counter, d_mutex, iterations);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    int h_counter;
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    float elapsed = std::chrono::duration<float, std::milli>(end - start).count();
    int expected = threads * blocks * iterations;

    printf("Spinlock counter: %d (expected: %d)\n", h_counter, expected);
    printf("Time: %.2f ms\n", elapsed);
    printf("Result is %s\n", (h_counter == expected) ? "CORRECT" : "INCORRECT");

    CHECK_CUDA(cudaFree(d_counter));
    CHECK_CUDA(cudaFree(d_mutex));
}

void benchmark_atomic_performance() {
    printf("\n=== Atomic Performance Benchmark ===\n");

    const int N = 1000000;
    const int iterations = 100;

    int* d_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

    // Benchmark with different thread configurations
    int configs[][2] = {{32, 1}, {256, 1}, {256, 10}, {1024, 10}};

    for (auto& config : configs) {
        int threads = config[0];
        int blocks = config[1];

        CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

        auto start = std::chrono::high_resolution_clock::now();
        atomic_increment_kernel<<<blocks, threads>>>(d_counter, iterations);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        float elapsed = std::chrono::duration<float, std::milli>(end - start).count();

        int h_counter;
        CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

        printf("Config (%d threads, %d blocks): %.2f ms, counter = %d\n",
               threads, blocks, elapsed, h_counter);
    }

    CHECK_CUDA(cudaFree(d_counter));
}

// Main function
#ifndef BUILDING_LIB
int main(int argc, char** argv) {
    // Query device
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // Run demonstrations
    demonstrate_basic_synchronization();
    demonstrate_atomic_operations();
    demonstrate_memory_fences();
    demonstrate_cooperative_groups();
    demonstrate_spinlock();
    benchmark_atomic_performance();

    printf("\n=== Synchronization and Atomics Demo Complete ===\n");

    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
#endif