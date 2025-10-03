// cuda_utils.h - Common CUDA utilities and error checking macros
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

//==================================================//
//              Error Checking Macros               //
//==================================================//

// Basic CUDA error checking macro
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n",                 \
                    __FILE__, __LINE__, #call, cudaGetErrorString(error));    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Extended error checking with custom message
#define CHECK_CUDA_MSG(call, msg)                                              \
    do {                                                                        \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n%s: %s\n",            \
                    __FILE__, __LINE__, msg, #call,                          \
                    cudaGetErrorString(error));                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Check last CUDA error (useful after kernel launches)
#define CHECK_LAST_CUDA_ERROR()                                                \
    do {                                                                        \
        cudaError_t error = cudaGetLastError();                               \
        if (error != cudaSuccess) {                                           \
            fprintf(stderr, "Last CUDA Error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(error));          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Check both launch and execution errors
#define CHECK_KERNEL_LAUNCH()                                                   \
    do {                                                                        \
        CHECK_LAST_CUDA_ERROR();                                              \
        CHECK_CUDA(cudaDeviceSynchronize());                                  \
    } while (0)

//==================================================//
//              Memory Utilities                    //
//==================================================//

// Safe memory allocation with error checking
template<typename T>
inline T* cuda_malloc(size_t count) {
    T* ptr = nullptr;
    size_t size = count * sizeof(T);
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

// Safe memory allocation and initialization
template<typename T>
inline T* cuda_calloc(size_t count) {
    T* ptr = cuda_malloc<T>(count);
    CHECK_CUDA(cudaMemset(ptr, 0, count * sizeof(T)));
    return ptr;
}

// Safe memory copy with error checking
template<typename T>
inline void cuda_memcpy(T* dst, const T* src, size_t count, cudaMemcpyKind kind) {
    CHECK_CUDA(cudaMemcpy(dst, src, count * sizeof(T), kind));
}

// Safe memory set with error checking
template<typename T>
inline void cuda_memset(T* ptr, int value, size_t count) {
    CHECK_CUDA(cudaMemset(ptr, value, count * sizeof(T)));
}

// Safe memory free
inline void cuda_free(void* ptr) {
    if (ptr) {
        CHECK_CUDA(cudaFree(ptr));
    }
}

//==================================================//
//              Device Query Utilities              //
//==================================================//

// Get device properties with error checking
inline cudaDeviceProp get_device_props(int device = 0) {
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    return props;
}

// Print device properties
inline void print_device_info(int device = 0) {
    cudaDeviceProp props = get_device_props(device);
    printf("Device %d: %s\n", device, props.name);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Total Global Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("  SMs: %d\n", props.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", props.warpSize);
    printf("  Max Shared Memory per Block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("  Max Registers per Block: %d\n", props.regsPerBlock);
}

// Get optimal block size for a kernel
template<typename KernelFunc>
inline dim3 get_optimal_block_size(KernelFunc kernel, size_t dynamic_shmem = 0) {
    int min_grid_size, block_size;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, kernel, dynamic_shmem));
    return dim3(block_size);
}

//==================================================//
//              Timing Utilities                    //
//==================================================//

// CUDA event timer class
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    bool is_started;

public:
    CudaTimer() : is_started(false) {
        CHECK_CUDA(cudaEventCreate(&start_event));
        CHECK_CUDA(cudaEventCreate(&stop_event));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(start_event));
        is_started = true;
    }

    void stop() {
        if (!is_started) {
            fprintf(stderr, "Timer not started!\n");
            return;
        }
        CHECK_CUDA(cudaEventRecord(stop_event));
        CHECK_CUDA(cudaEventSynchronize(stop_event));
        is_started = false;
    }

    float elapsed_ms() {
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }

    float elapsed_s() {
        return elapsed_ms() / 1000.0f;
    }
};

//==================================================//
//              Grid/Block Calculation              //
//==================================================//

// Calculate grid size for 1D kernel
inline int grid_size_1d(int total_threads, int block_size) {
    return (total_threads + block_size - 1) / block_size;
}

// Calculate grid size for 2D kernel
inline dim3 grid_size_2d(int width, int height, dim3 block) {
    return dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
}

// Calculate grid size for 3D kernel
inline dim3 grid_size_3d(int width, int height, int depth, dim3 block) {
    return dim3(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        (depth + block.z - 1) / block.z
    );
}

//==================================================//
//              Debug Utilities                     //
//==================================================//

// Print macro for debugging (only in debug builds)
#ifdef DEBUG
    #define CUDA_DEBUG_PRINT(fmt, ...) \
        printf("[DEBUG %s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
    #define CUDA_DEBUG_PRINT(fmt, ...)
#endif

// Device-side assert (requires -G flag)
#ifdef __CUDA_ARCH__
    #define CUDA_ASSERT(condition) \
        if (!(condition)) { \
            printf("Assertion failed at %s:%d\n", __FILE__, __LINE__); \
            assert(condition); \
        }
#else
    #define CUDA_ASSERT(condition) assert(condition)
#endif

//==================================================//
//           Performance Metrics                    //
//==================================================//

// Calculate bandwidth in GB/s
inline float calculate_bandwidth_gb(size_t bytes, float time_ms) {
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

// Calculate effective bandwidth utilization (simplified version)
// Note: For accurate peak bandwidth, check device specifications
inline float bandwidth_efficiency_percent(float achieved_gb) {
    // Typical modern GPU peak bandwidth is ~500-900 GB/s
    // This is a rough estimate - check your specific GPU specs
    const float typical_peak_gb = 600.0;
    return (achieved_gb / typical_peak_gb) * 100.0;
}

// Calculate GFLOPS
inline float calculate_gflops(size_t operations, float time_ms) {
    return (operations / 1e9) / (time_ms / 1000.0);
}

//==================================================//
//           Memory Pool Utilities                  //
//==================================================//

// Simple memory pool for frequent allocations
template<typename T>
class CudaMemoryPool {
private:
    T* pool;
    size_t capacity;
    size_t current_offset;

public:
    CudaMemoryPool(size_t total_elements)
        : capacity(total_elements), current_offset(0) {
        pool = cuda_malloc<T>(capacity);
    }

    ~CudaMemoryPool() {
        cuda_free(pool);
    }

    T* allocate(size_t count) {
        if (current_offset + count > capacity) {
            fprintf(stderr, "Memory pool exhausted!\n");
            return nullptr;
        }
        T* ptr = pool + current_offset;
        current_offset += count;
        return ptr;
    }

    void reset() {
        current_offset = 0;
    }

    size_t used() const {
        return current_offset;
    }

    size_t available() const {
        return capacity - current_offset;
    }
};