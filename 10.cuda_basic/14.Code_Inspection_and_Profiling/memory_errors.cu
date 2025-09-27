// memory_errors.cu - Example with intentional errors for sanitizer testing
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Kernel with multiple intentional bugs for demonstrating sanitizer capabilities
__global__ void buggyKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Error 1: Out-of-bounds access
    if (idx <= n) {  // BUG: Should be idx < n (off-by-one error)
        data[idx] = idx * 2.0f;
    }

    // Error 2: Race condition in shared memory
    __shared__ float sdata[256];
    sdata[threadIdx.x] = data[idx];
    // BUG: Missing __syncthreads() here!
    // Threads may read uninitialized or partially written shared memory

    // Use shared memory (will have race condition)
    if (threadIdx.x < 255) {
        data[idx] = sdata[threadIdx.x] + sdata[threadIdx.x + 1];
    }

    // Note: Conditional __syncthreads() is a serious bug but NOT detected by synccheck
    // Example of what would be wrong (commented out to avoid undefined behavior):
    // if (threadIdx.x < 128) {
    //     __syncthreads();  // BUG: Not all threads reach this!
    // }
}

// Kernel with uninitialized memory access
__global__ void uninitializedAccessKernel(float* output, float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // BUG: Using uninitialized local variable
    float temp;  // Not initialized

    if (idx < n) {
        // BUG: temp is uninitialized
        output[idx] = input[idx] + temp;
    }
}

// Kernel with misaligned access
__global__ void misalignedAccessKernel(char* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n - sizeof(int)) {
        // BUG: Potentially misaligned access to int through char pointer
        int* intPtr = (int*)(data + idx);
        *intPtr = idx;
    }
}

// Kernel demonstrating shared memory bank conflicts
__global__ void bankConflictKernel(float* output, float* input, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Load with potential bank conflicts (stride of 32)
        sdata[tid * 32 % 256] = input[idx];  // BUG: Severe bank conflicts
        __syncthreads();

        // Store back
        output[idx] = sdata[tid * 32 % 256];
    }
}

void demonstrateMemoryErrors() {
    printf("=== Demonstrating Memory Errors for Sanitizer Testing ===\n");
    printf("Run with compute-sanitizer to detect these issues:\n");
    printf("  compute-sanitizer --tool memcheck ./memory_errors\n");
    printf("  compute-sanitizer --tool racecheck ./memory_errors\n");
    printf("  compute-sanitizer --tool synccheck ./memory_errors\n");
    printf("  compute-sanitizer --tool initcheck ./memory_errors\n\n");

    const int N = 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data, *d_input, *d_output;
    char *d_char_data;
    size_t size = N * sizeof(float);

    // Allocate memory
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMalloc(&d_char_data, N));

    // Initialize input (but not output - intentional)
    CHECK_CUDA(cudaMemset(d_input, 0, size));

    printf("Test 1: Out-of-bounds and race condition errors...\n");
    buggyKernel<<<numBlocks, blockSize>>>(d_data, N);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("  Kernel failed: %s\n", cudaGetErrorString(error));
    } else {
        printf("  Kernel completed (but may have hidden errors)\n");
    }

    printf("\nTest 2: Uninitialized memory access...\n");
    uninitializedAccessKernel<<<numBlocks, blockSize>>>(d_output, d_input, N);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("  Kernel failed: %s\n", cudaGetErrorString(error));
    } else {
        printf("  Kernel completed (check with initcheck)\n");
    }

    printf("\nTest 3: Misaligned memory access...\n");
    misalignedAccessKernel<<<numBlocks, blockSize>>>(d_char_data, N);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("  Kernel failed: %s\n", cudaGetErrorString(error));
    } else {
        printf("  Kernel completed (may have alignment issues)\n");
    }

    printf("\nTest 4: Shared memory bank conflicts...\n");
    bankConflictKernel<<<numBlocks, blockSize>>>(d_output, d_input, N);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("  Kernel failed: %s\n", cudaGetErrorString(error));
    } else {
        printf("  Kernel completed (profile to see bank conflicts)\n");
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_char_data));

    printf("\n=== Tests completed ===\n");
    printf("To detect errors, run with:\n");
    printf("1. Memory errors:     compute-sanitizer --tool memcheck ./memory_errors\n");
    printf("2. Race conditions:   compute-sanitizer --tool racecheck ./memory_errors\n");
    printf("3. Sync errors:       compute-sanitizer --tool synccheck ./memory_errors\n");
    printf("4. Uninit errors:     compute-sanitizer --tool initcheck ./memory_errors\n");
    printf("5. All checks:        compute-sanitizer ./memory_errors\n");
}

int main(int argc, char* argv[]) {
    // Check CUDA device
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);

    // Run demonstration
    demonstrateMemoryErrors();

    return 0;
}