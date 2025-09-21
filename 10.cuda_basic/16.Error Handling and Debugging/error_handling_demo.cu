// error_handling_demo.cu - Demonstrates CUDA error handling and debugging techniques
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <climits>
#include "cuda_utils.h"  // Use our custom CUDA utilities library
#include "vector_add_2d.h"

// Demonstration of different error handling approaches
void demo_basic_error_checking() {
    std::cout << "\n=== Basic Error Checking Demo ===" << std::endl;

    const int width = 256;
    const int height = 256;
    const int size = width * height;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // Allocate device memory with error checking
    std::cout << "Allocating device memory..." << std::endl;

    // Using utility functions with original API shown in comments
    d_a = cuda_malloc<float>(size);  // Original: cudaMalloc(&d_a, size * sizeof(float));
    d_b = cuda_malloc<float>(size);  // Original: cudaMalloc(&d_b, size * sizeof(float));
    d_c = cuda_malloc<float>(size);  // Original: cudaMalloc(&d_c, size * sizeof(float));

    // Allocate and initialize host memory
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_c = new float[size];

    for (int i = 0; i < size; i++) {
        h_a[i] = static_cast<float>(i % 100);
        h_b[i] = static_cast<float>((i * 2) % 100);
    }

    // Copy to device with error checking
    cuda_memcpy(d_a, h_a, size, cudaMemcpyHostToDevice);  // Original: cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cuda_memcpy(d_b, h_b, size, cudaMemcpyHostToDevice);  // Original: cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Launching kernel with grid(" << gridSize.x << "," << gridSize.y
              << ") and block(" << blockSize.x << "," << blockSize.y << ")" << std::endl;

    vector_add_2d<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);

    // Check for kernel launch errors
    CHECK_LAST_CUDA_ERROR();  // Original: cudaGetLastError();
    CHECK_CUDA(cudaDeviceSynchronize());  // Original: cudaDeviceSynchronize();

    // Copy result back
    cuda_memcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);  // Original: cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify first few results
    std::cout << "First 5 results: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cuda_free(d_a);  // Original: cudaFree(d_a);
    cuda_free(d_b);  // Original: cudaFree(d_b);
    cuda_free(d_c);  // Original: cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    std::cout << "Basic error checking demo completed successfully!" << std::endl;
}

// Demonstrate handling of out-of-bounds errors
void demo_out_of_bounds_error() {
    std::cout << "\n=== Out-of-Bounds Error Demo ===" << std::endl;
    std::cout << "This will intentionally cause an error when run with compute-sanitizer --tool memcheck" << std::endl;

    const int width = 100;
    const int height = 100;
    const int size = width * height;

    // Allocate memory
    float* d_a = cuda_malloc<float>(size);
    float* d_b = cuda_malloc<float>(size);
    float* d_c = cuda_malloc<float>(size);

    // Initialize with zeros
    CHECK_CUDA(cudaMemset(d_a, 0, size * sizeof(float)));  // Original API for demonstration
    CHECK_CUDA(cudaMemset(d_b, 0, size * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_c, 0, size * sizeof(float)));

    // Launch kernel with too many blocks (intentional bug)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x + 2,  // Extra blocks!
                  (height + blockSize.y - 1) / blockSize.y + 2); // Extra blocks!

    std::cout << "Launching buggy kernel with oversized grid..." << std::endl;
    vector_add_2d_with_bug<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel completed (errors may be detected by sanitizer tools)" << std::endl;
    }

    // Cleanup
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);
}

// Demonstrate assertion-based debugging
void demo_assertion_debugging() {
    std::cout << "\n=== Assertion Debugging Demo ===" << std::endl;
    std::cout << "Note: Assertions only work in debug builds with -G flag" << std::endl;

    const int width = 64;
    const int height = 64;
    const int size = width * height;

    float* d_a = cuda_malloc<float>(size);
    float* d_b = cuda_malloc<float>(size);
    float* d_c = cuda_malloc<float>(size);

    // Create host data with some NaN values for testing
    float* h_a = new float[size];
    float* h_b = new float[size];

    for (int i = 0; i < size; i++) {
        h_a[i] = (i % 10 == 0) ? NAN : static_cast<float>(i);  // Some NaN values
        h_b[i] = static_cast<float>(i * 2);
    }

    cuda_memcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cuda_memcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Launching kernel with assertions (will fail if NaN detected in debug mode)..." << std::endl;
    vector_add_2d_with_assert<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Assertion failed or other error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel completed successfully (assertions may not be enabled)" << std::endl;
    }

    // Cleanup
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);
    delete[] h_a;
    delete[] h_b;
}

// Demonstrate error recovery
void demo_error_recovery() {
    std::cout << "\n=== Error Recovery Demo ===" << std::endl;

    // Try to allocate an impossibly large amount of memory
    size_t huge_size = ULLONG_MAX / 2;  // Way too much!
    float* d_huge = nullptr;

    std::cout << "Attempting to allocate " << huge_size << " bytes..." << std::endl;
    cudaError_t err = cudaMalloc(&d_huge, huge_size);

    if (err != cudaSuccess) {
        std::cout << "Allocation failed as expected: " << cudaGetErrorString(err) << std::endl;
        std::cout << "Recovering by resetting device..." << std::endl;

        // Reset device to clear error state
        err = cudaDeviceReset();
        if (err == cudaSuccess) {
            std::cout << "Device reset successful!" << std::endl;

            // Now try a reasonable allocation
            size_t reasonable_size = 1024 * sizeof(float);
            float* d_small = nullptr;
            err = cudaMalloc(&d_small, reasonable_size);

            if (err == cudaSuccess) {
                std::cout << "Successfully allocated " << reasonable_size << " bytes after recovery" << std::endl;
                cudaFree(d_small);
            }
        }
    }
}

// Demonstrate reduce_sum_2d with error handling
void demo_reduce_sum() {
    std::cout << "\n=== Reduce Sum Demo with Error Handling ===" << std::endl;

    const int width = 128;
    const int height = 128;
    const int size = width * height;

    // Allocate memory
    float* d_input = cuda_malloc<float>(size);
    float* d_output = cuda_malloc<float>(1);

    // Initialize input
    float* h_input = new float[size];
    float expected_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;  // All ones for easy verification
        expected_sum += h_input[i];
    }

    // Copy to device
    cuda_memcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));  // Original: cudaMemset

    // Launch reduction kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float);
    std::cout << "Launching reduction kernel with " << sharedMemSize << " bytes of shared memory" << std::endl;

    reduce_sum_2d<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, width, height);
    CHECK_KERNEL_LAUNCH();  // Check both launch and execution

    // Get result
    float h_output = 0.0f;
    cuda_memcpy(&h_output, d_output, 1, cudaMemcpyDeviceToHost);

    std::cout << "Expected sum: " << expected_sum << std::endl;
    std::cout << "Computed sum: " << h_output << std::endl;
    std::cout << "Error: " << std::abs(h_output - expected_sum) << std::endl;

    // Cleanup
    cuda_free(d_input);
    cuda_free(d_output);
    delete[] h_input;
}

int main() {
    std::cout << "=== CUDA Error Handling and Debugging Demonstration ===" << std::endl;

    // Check for CUDA device
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    // Print device info using utility
    print_device_info(0);
    std::cout << std::endl;

    // Run demonstrations
    demo_basic_error_checking();
    demo_reduce_sum();
    demo_out_of_bounds_error();
    demo_assertion_debugging();
    demo_error_recovery();

    std::cout << "\n=== Demo Complete ===" << std::endl;
    std::cout << "\nTo detect memory errors, run with:" << std::endl;
    std::cout << "  compute-sanitizer --tool memcheck ./16_ErrorHandlingAndDebugging" << std::endl;
    std::cout << "\nTo detect race conditions, run with:" << std::endl;
    std::cout << "  compute-sanitizer --tool racecheck ./16_ErrorHandlingAndDebugging" << std::endl;
    std::cout << "\nFor debugging with cuda-gdb:" << std::endl;
    std::cout << "  cuda-gdb ./16_ErrorHandlingAndDebugging" << std::endl;

    return 0;
}