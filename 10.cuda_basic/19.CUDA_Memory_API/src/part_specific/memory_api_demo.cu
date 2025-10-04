// memory_api_demo.cu - Comprehensive CUDA Memory API demonstrations
#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <chrono>
#include <vector>
#include <cstring>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Kernel for testing different memory types
__global__ void memory_test_kernel(float* global_mem, float* constant_ptr,
                                   int N, float scalar) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Access global memory
        float val = global_mem[tid];

        // Use constant memory value
        val *= (*constant_ptr);

        // Apply scalar
        val += scalar;

        // Write back
        global_mem[tid] = val;
    }
}

// Kernel for testing pinned memory transfers
__global__ void compute_kernel(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = sqrtf(data[tid] * data[tid] + 1.0f);
    }
}

// Kernel for testing unified memory
__global__ void unified_memory_kernel(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

// Kernel for testing texture memory (using texture objects)
__global__ void texture_memory_kernel(cudaTextureObject_t texObj,
                                      float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Read from texture memory with normalized coordinates
        float u = (float)tid / (float)N;
        output[tid] = tex1D<float>(texObj, u);
    }
}

// Constant memory declaration
__constant__ float d_const_data[256];

// Kernel using constant memory
__global__ void constant_memory_kernel(float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N && tid < 256) {
        // All threads in warp read same constant memory - broadcast
        output[tid] = d_const_data[tid % 32] * 2.0f;
    }
}

// Zero-copy memory kernel
__global__ void zero_copy_kernel(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Direct access to host memory
        data[tid] = data[tid] * 3.0f;
    }
}

// Memory pool allocation kernel
__global__ void memory_pool_kernel(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = sinf(data[tid]);
    }
}

// Function to demonstrate basic memory allocation
void demonstrate_basic_allocation() {
    printf("\n=== Basic Memory Allocation ===\n");

    const int N = 1024 * 1024; // 1M elements
    size_t bytes = N * sizeof(float);

    // Host allocation
    float* h_data = new float[N];

    // Device allocation
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Copy to device
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();

    float h2d_time = std::chrono::duration<float, std::milli>(end - start).count();
    float bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (h2d_time / 1000.0f);

    printf("Host to Device transfer: %.2f ms (%.2f GB/s)\n", h2d_time, bandwidth);

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    compute_kernel<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back
    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    end = std::chrono::high_resolution_clock::now();

    float d2h_time = std::chrono::duration<float, std::milli>(end - start).count();
    bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (d2h_time / 1000.0f);

    printf("Device to Host transfer: %.2f ms (%.2f GB/s)\n", d2h_time, bandwidth);

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    delete[] h_data;
}

// Function to demonstrate pinned memory
void demonstrate_pinned_memory() {
    printf("\n=== Pinned Memory ===\n");

    const int N = 1024 * 1024; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate pinned host memory
    float* h_pinned;
    CHECK_CUDA(cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault));

    // Regular host memory for comparison
    float* h_regular = new float[N];

    // Device memory
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_pinned[i] = (float)i;
        h_regular[i] = (float)i;
    }

    // Test regular memory transfer
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_data, h_regular, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    float regular_time = std::chrono::duration<float, std::milli>(end - start).count();

    // Test pinned memory transfer
    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    float pinned_time = std::chrono::duration<float, std::milli>(end - start).count();

    printf("Regular memory transfer: %.2f ms\n", regular_time);
    printf("Pinned memory transfer: %.2f ms\n", pinned_time);
    printf("Speedup: %.2fx\n", regular_time / pinned_time);

    // Cleanup
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_data));
    delete[] h_regular;
}

// Function to demonstrate unified memory
void demonstrate_unified_memory() {
    printf("\n=== Unified Memory ===\n");

    const int N = 1024 * 1024; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate unified memory
    float* unified_data;
    CHECK_CUDA(cudaMallocManaged(&unified_data, bytes));

    // Initialize on host
    for (int i = 0; i < N; i++) {
        unified_data[i] = (float)i;
    }

    // Launch kernel - automatic migration to device
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();
    unified_memory_kernel<<<gridSize, blockSize>>>(unified_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float kernel_time = std::chrono::duration<float, std::milli>(end - start).count();
    printf("Unified memory kernel execution: %.2f ms\n", kernel_time);

    // Access on host - automatic migration back
    float sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        sum += unified_data[i];
    }
    printf("Sum of first 1000 elements: %.2f\n", sum);

    // Cleanup
    CHECK_CUDA(cudaFree(unified_data));
}

// Function to demonstrate zero-copy memory
void demonstrate_zero_copy_memory() {
    printf("\n=== Zero-Copy Memory ===\n");

    const int N = 1024 * 1024; // 1M elements
    size_t bytes = N * sizeof(float);

    // Check if device supports mapped memory
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    if (!prop.canMapHostMemory) {
        printf("Device does not support mapped memory\n");
        return;
    }

    // Enable mapped memory
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Allocate mapped host memory
    float* h_data;
    CHECK_CUDA(cudaHostAlloc(&h_data, bytes, cudaHostAllocMapped));

    // Get device pointer
    float* d_data;
    CHECK_CUDA(cudaHostGetDevicePointer(&d_data, h_data, 0));

    // Initialize on host
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Launch kernel using device pointer
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();
    zero_copy_kernel<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float kernel_time = std::chrono::duration<float, std::milli>(end - start).count();
    printf("Zero-copy kernel execution: %.2f ms\n", kernel_time);

    // Verify on host - data already available
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUDA(cudaFreeHost(h_data));
}

// Function to demonstrate constant memory
void demonstrate_constant_memory() {
    printf("\n=== Constant Memory ===\n");

    const int N = 256;
    size_t bytes = N * sizeof(float);

    // Prepare host data
    float h_const_data[N];
    for (int i = 0; i < N; i++) {
        h_const_data[i] = (float)(i + 1);
    }

    // Copy to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_const_data, h_const_data, bytes));

    // Allocate output
    float* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize(1);

    auto start = std::chrono::high_resolution_clock::now();
    constant_memory_kernel<<<gridSize, blockSize>>>(d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float kernel_time = std::chrono::duration<float, std::micro>(end - start).count();
    printf("Constant memory kernel execution: %.2f microseconds\n", kernel_time);

    // Copy back and verify
    float h_output[N];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    printf("First 5 outputs: ");
    for (int i = 0; i < 5; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_output));
}

// Function to demonstrate texture memory
void demonstrate_texture_memory() {
    printf("\n=== Texture Memory ===\n");

    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // Prepare host data
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = sinf(2.0f * M_PI * i / N);
    }

    // Allocate device memory
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // 32 bits
    resDesc.res.linear.sizeInBytes = bytes;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1; // Use normalized coordinates

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // Allocate output
    float* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    auto start = std::chrono::high_resolution_clock::now();
    texture_memory_kernel<<<gridSize, blockSize>>>(texObj, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float kernel_time = std::chrono::duration<float, std::micro>(end - start).count();
    printf("Texture memory kernel execution: %.2f microseconds\n", kernel_time);

    // Cleanup
    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_output));
    delete[] h_data;
}

// Function to demonstrate memory pools (CUDA 11.2+)
void demonstrate_memory_pools() {
    printf("\n=== Memory Pools ===\n");

    int runtime_version;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));

    if (runtime_version < 11020) {
        printf("Memory pools require CUDA 11.2+. Current version: %d.%d\n",
               runtime_version / 1000, (runtime_version % 1000) / 10);
        return;
    }

#if CUDART_VERSION >= 11020
    const int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    // Create memory pool
    cudaMemPool_t mempool;
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0;

    CHECK_CUDA(cudaMemPoolCreate(&mempool, &props));

    // Allocate from pool
    float* d_data;
    CHECK_CUDA(cudaMallocFromPoolAsync(&d_data, bytes, mempool, 0));

    // Use the memory
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    memory_pool_kernel<<<gridSize, blockSize>>>(d_data, N);

    CHECK_CUDA(cudaStreamSynchronize(0));

    // Free back to pool
    CHECK_CUDA(cudaFreeAsync(d_data, 0));
    CHECK_CUDA(cudaStreamSynchronize(0));

    // Destroy pool
    CHECK_CUDA(cudaMemPoolDestroy(mempool));

    printf("Memory pool operations completed successfully\n");
#else
    printf("Memory pools not available in this CUDA version\n");
#endif
}

// Function to demonstrate memory info
void demonstrate_memory_info() {
    printf("\n=== Memory Information ===\n");

    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));

    printf("GPU Memory:\n");
    printf("  Total: %.2f GB\n", total_mem / (1024.0f * 1024.0f * 1024.0f));
    printf("  Free: %.2f GB\n", free_mem / (1024.0f * 1024.0f * 1024.0f));
    printf("  Used: %.2f GB\n", (total_mem - free_mem) / (1024.0f * 1024.0f * 1024.0f));

    // Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("\nMemory Properties:\n");
    printf("  L2 Cache Size: %.2f MB\n", prop.l2CacheSize / (1024.0f * 1024.0f));
    printf("  Total Constant Memory: %.2f KB\n", prop.totalConstMem / 1024.0f);
    printf("  Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    // Note: Memory bandwidth calculation depends on actual memory clock which varies
    // For accurate bandwidth, use profiler measurements
}

// Function to demonstrate async memory operations
void demonstrate_async_operations() {
    printf("\n=== Asynchronous Memory Operations ===\n");

    const int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    // Create streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // Allocate memory
    float* h_data1 = new float[N];
    float* h_data2 = new float[N];
    float* d_data1, *d_data2;

    CHECK_CUDA(cudaMalloc(&d_data1, bytes));
    CHECK_CUDA(cudaMalloc(&d_data2, bytes));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data1[i] = (float)i;
        h_data2[i] = (float)(i * 2);
    }

    // Async operations on different streams
    auto start = std::chrono::high_resolution_clock::now();

    // Stream 1 operations
    CHECK_CUDA(cudaMemcpyAsync(d_data1, h_data1, bytes, cudaMemcpyHostToDevice, stream1));
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    compute_kernel<<<gridSize, blockSize, 0, stream1>>>(d_data1, N);
    CHECK_CUDA(cudaMemcpyAsync(h_data1, d_data1, bytes, cudaMemcpyDeviceToHost, stream1));

    // Stream 2 operations
    CHECK_CUDA(cudaMemcpyAsync(d_data2, h_data2, bytes, cudaMemcpyHostToDevice, stream2));
    compute_kernel<<<gridSize, blockSize, 0, stream2>>>(d_data2, N);
    CHECK_CUDA(cudaMemcpyAsync(h_data2, d_data2, bytes, cudaMemcpyDeviceToHost, stream2));

    // Wait for both streams
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    auto end = std::chrono::high_resolution_clock::now();
    float async_time = std::chrono::duration<float, std::milli>(end - start).count();

    printf("Async operations on 2 streams: %.2f ms\n", async_time);

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_data1));
    CHECK_CUDA(cudaFree(d_data2));
    delete[] h_data1;
    delete[] h_data2;
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
    demonstrate_memory_info();
    demonstrate_basic_allocation();
    demonstrate_pinned_memory();
    demonstrate_unified_memory();
    demonstrate_zero_copy_memory();
    demonstrate_constant_memory();
    demonstrate_texture_memory();
    demonstrate_async_operations();
    demonstrate_memory_pools();

    printf("\n=== CUDA Memory API Demo Complete ===\n");

    // Reset device
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
#endif