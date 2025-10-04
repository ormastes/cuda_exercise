/**
 * Integration Tests for NVMe-GPU Pipeline
 *
 * Tests the complete workflow from NVMe data loading through GPU computation
 * This validates the integration between Part 50 (NVMe) and Part 51 (Optimizations)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>

// NVMe API functions from Part 50
extern "C" {
    int nvme_gpu_init(const char* device_path);
    void nvme_gpu_cleanup();
    ssize_t nvme_gpu_read(const char* device_path, off_t offset, size_t size, void* gpu_buffer);
    int nvme_gpu_read_async(const char* device_path, off_t offset, size_t size,
                           void* gpu_buffer, cudaStream_t stream);
}

// Optimization functions from Part 51
extern "C" {
    void launch_matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K);
    void launch_matmul_vectorized(float* C, const float* A, const float* B, int M, int N, int K);
    double benchmark_matmul(void (*kernel)(float*, const float*, const float*, int, int, int),
                           float* d_C, const float* d_A, const float* d_B,
                           int M, int N, int K, int iterations);
}

// Integration test fixture
class NVMeGPUIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        // Create test data files
        createTestDataFiles();

        // Initialize NVMe API
        if (nvme_gpu_init(matrix_file.c_str()) != 0) {
            GTEST_SKIP() << "NVMe API initialization failed - skipping integration test";
        }
    }

    void TearDown() override {
        nvme_gpu_cleanup();

        // Clean up test files
        std::remove(matrix_file.c_str());
        std::remove(vector_file.c_str());
        std::remove(result_file.c_str());
    }

    void createTestDataFiles() {
        matrix_file = "/tmp/nvme_gpu_matrix_" + std::to_string(getpid()) + ".dat";
        vector_file = "/tmp/nvme_gpu_vector_" + std::to_string(getpid()) + ".dat";
        result_file = "/tmp/nvme_gpu_result_" + std::to_string(getpid()) + ".dat";

        // Create matrix data file (multiple matrices for batched operations)
        std::ofstream mfile(matrix_file, std::ios::binary);
        const int num_matrices = 10;
        const int matrix_size = 1024;

        for (int m = 0; m < num_matrices; m++) {
            std::vector<float> matrix(matrix_size * matrix_size);
            for (int i = 0; i < matrix_size * matrix_size; i++) {
                matrix[i] = static_cast<float>(m + 1) * 0.1f + (i % 100) * 0.001f;
            }
            mfile.write(reinterpret_cast<char*>(matrix.data()),
                       matrix.size() * sizeof(float));
        }
        mfile.close();

        // Create vector data file
        std::ofstream vfile(vector_file, std::ios::binary);
        std::vector<float> vector(matrix_size);
        for (int i = 0; i < matrix_size; i++) {
            vector[i] = 1.0f / (i + 1);
        }
        vfile.write(reinterpret_cast<char*>(vector.data()),
                   vector.size() * sizeof(float));
        vfile.close();
    }

    std::string matrix_file;
    std::string vector_file;
    std::string result_file;
};

// End-to-end tests

TEST_F(NVMeGPUIntegrationTest, DirectNVMeToMatMul) {
    const int matrix_size = 1024;
    const size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);

    // Allocate GPU buffers
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, matrix_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, matrix_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, matrix_bytes), cudaSuccess);

    // Measure end-to-end time
    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

    // Step 1: Load matrices directly from NVMe to GPU
    ssize_t bytes_read_A = nvme_gpu_read(matrix_file.c_str(), 0, matrix_bytes, d_A);
    EXPECT_EQ(bytes_read_A, matrix_bytes);

    ssize_t bytes_read_B = nvme_gpu_read(matrix_file.c_str(), matrix_bytes, matrix_bytes, d_B);
    EXPECT_EQ(bytes_read_B, matrix_bytes);

    // Step 2: Perform matrix multiplication
    launch_matmul_vectorized(d_C, d_A, d_B, matrix_size, matrix_size, matrix_size);

    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsed_ms;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

    // Calculate metrics
    double data_transferred_gb = (2 * matrix_bytes) / 1e9;
    double compute_flops = 2.0 * matrix_size * matrix_size * matrix_size;
    double total_time_s = elapsed_ms / 1000.0;

    printf("End-to-end pipeline:\n");
    printf("  Data loaded: %.2f GB\n", data_transferred_gb);
    printf("  Compute: %.2f GFLOPS\n", compute_flops / total_time_s / 1e9);
    printf("  Total time: %.2f ms\n", elapsed_ms);
    printf("  Effective throughput: %.2f GB/s\n", data_transferred_gb / total_time_s);

    // Verify result (spot check)
    std::vector<float> h_C(10);  // Check first 10 elements
    ASSERT_EQ(cudaMemcpy(h_C.data(), d_C, 10 * sizeof(float),
                        cudaMemcpyDeviceToHost), cudaSuccess);

    // Results should be non-zero
    for (int i = 0; i < 10; i++) {
        EXPECT_GT(std::abs(h_C[i]), 1e-6f);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST_F(NVMeGPUIntegrationTest, StreamedBatchProcessing) {
    const int matrix_size = 1024;
    const size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
    const int num_batches = 4;
    const int num_streams = 2;

    // Create streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    }

    // Allocate buffers for each stream
    std::vector<float*> d_A(num_streams);
    std::vector<float*> d_B(num_streams);
    std::vector<float*> d_C(num_streams);

    for (int i = 0; i < num_streams; i++) {
        ASSERT_EQ(cudaMalloc(&d_A[i], matrix_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_B[i], matrix_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_C[i], matrix_bytes), cudaSuccess);
    }

    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

    // Process batches using streams
    for (int batch = 0; batch < num_batches; batch++) {
        int stream_id = batch % num_streams;
        off_t offset_A = batch * 2 * matrix_bytes;
        off_t offset_B = offset_A + matrix_bytes;

        // Async read (currently synchronous in implementation)
        nvme_gpu_read_async(matrix_file.c_str(), offset_A, matrix_bytes,
                           d_A[stream_id], streams[stream_id]);
        nvme_gpu_read_async(matrix_file.c_str(), offset_B, matrix_bytes,
                           d_B[stream_id], streams[stream_id]);

        // Launch computation on the stream
        dim3 block(32, 32);
        dim3 grid((matrix_size + 31) / 32, (matrix_size + 31) / 32);

        // Note: Kernel launch on stream requires stream parameter
        // Since our launch functions don't take streams, we synchronize here
        launch_matmul_tiled(d_C[stream_id], d_A[stream_id], d_B[stream_id],
                           matrix_size, matrix_size, matrix_size);
    }

    // Wait for all streams
    for (int i = 0; i < num_streams; i++) {
        ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    }

    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsed_ms;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

    double total_data_gb = (num_batches * 2 * matrix_bytes) / 1e9;
    double total_compute_gflops = num_batches * 2.0 * matrix_size * matrix_size * matrix_size / 1e9;

    printf("Streamed batch processing:\n");
    printf("  Batches: %d\n", num_batches);
    printf("  Streams: %d\n", num_streams);
    printf("  Total data: %.2f GB\n", total_data_gb);
    printf("  Total compute: %.2f GFLOPS\n", total_compute_gflops);
    printf("  Time: %.2f ms\n", elapsed_ms);
    printf("  Throughput: %.2f GB/s\n", total_data_gb / (elapsed_ms / 1000.0));

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST_F(NVMeGPUIntegrationTest, IterativeComputation) {
    // Simulate an iterative algorithm that reads data, processes, and writes back
    const int matrix_size = 512;
    const size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
    const int num_iterations = 5;

    float *d_current, *d_next, *d_kernel;
    ASSERT_EQ(cudaMalloc(&d_current, matrix_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_next, matrix_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_kernel, matrix_bytes), cudaSuccess);

    // Load initial data from NVMe
    ssize_t bytes_read = nvme_gpu_read(matrix_file.c_str(), 0, matrix_bytes, d_current);
    EXPECT_EQ(bytes_read, matrix_bytes);

    // Load convolution kernel
    bytes_read = nvme_gpu_read(matrix_file.c_str(), matrix_bytes, matrix_bytes, d_kernel);
    EXPECT_EQ(bytes_read, matrix_bytes);

    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

    // Iterative processing
    for (int iter = 0; iter < num_iterations; iter++) {
        // Apply transformation (using matmul as proxy for complex operation)
        launch_matmul_tiled(d_next, d_current, d_kernel,
                          matrix_size, matrix_size, matrix_size);

        // Swap buffers
        std::swap(d_current, d_next);

        // Optionally write intermediate results (commented out for performance)
        // write_gpu_to_nvme(result_file.c_str(), iter * matrix_bytes,
        //                  matrix_bytes, d_current);
    }

    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsed_ms;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

    double compute_per_iter = 2.0 * matrix_size * matrix_size * matrix_size;
    double total_gflops = num_iterations * compute_per_iter / 1e9;

    printf("Iterative computation:\n");
    printf("  Iterations: %d\n", num_iterations);
    printf("  Matrix size: %dx%d\n", matrix_size, matrix_size);
    printf("  Compute: %.2f GFLOPS total\n", total_gflops);
    printf("  Time: %.2f ms\n", elapsed_ms);
    printf("  Performance: %.2f GFLOPS/s\n", total_gflops / (elapsed_ms / 1000.0));

    cudaFree(d_current);
    cudaFree(d_next);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Performance comparison test

class NVMeGPUPerformanceTest : public NVMeGPUIntegrationTest {
protected:
    void compareDataSources(int matrix_size) {
        const size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);

        // Allocate buffers
        float *d_A, *d_B, *d_C;
        float *h_A, *h_B;

        ASSERT_EQ(cudaMalloc(&d_A, matrix_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_B, matrix_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_C, matrix_bytes), cudaSuccess);

        h_A = (float*)malloc(matrix_bytes);
        h_B = (float*)malloc(matrix_bytes);

        // Initialize host data
        for (int i = 0; i < matrix_size * matrix_size; i++) {
            h_A[i] = i * 0.001f;
            h_B[i] = (matrix_size * matrix_size - i) * 0.001f;
        }

        cudaEvent_t start, stop;
        ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

        // Method 1: Traditional CPU->GPU copy + compute
        ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

        cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice);
        launch_matmul_tiled(d_C, d_A, d_B, matrix_size, matrix_size, matrix_size);
        cudaDeviceSynchronize();

        ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

        float traditional_ms;
        ASSERT_EQ(cudaEventElapsedTime(&traditional_ms, start, stop), cudaSuccess);

        // Method 2: Direct NVMe->GPU + compute
        ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

        nvme_gpu_read(matrix_file.c_str(), 0, matrix_bytes, d_A);
        nvme_gpu_read(matrix_file.c_str(), matrix_bytes, matrix_bytes, d_B);
        launch_matmul_tiled(d_C, d_A, d_B, matrix_size, matrix_size, matrix_size);
        cudaDeviceSynchronize();

        ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

        float direct_ms;
        ASSERT_EQ(cudaEventElapsedTime(&direct_ms, start, stop), cudaSuccess);

        printf("Matrix size %dx%d:\n", matrix_size, matrix_size);
        printf("  Traditional (CPU->GPU): %.2f ms\n", traditional_ms);
        printf("  Direct (NVMe->GPU): %.2f ms\n", direct_ms);
        printf("  Speedup: %.2fx\n\n", traditional_ms / direct_ms);

        free(h_A);
        free(h_B);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

TEST_F(NVMeGPUPerformanceTest, DataSourceComparison) {
    printf("\n=== Data Source Performance Comparison ===\n");

    // Test different matrix sizes
    std::vector<int> sizes = {512, 1024, 2048};

    for (int size : sizes) {
        // Skip if file doesn't have enough data
        std::ifstream file(matrix_file, std::ios::ate | std::ios::binary);
        size_t file_size = file.tellg();
        file.close();

        if (file_size < 2 * size * size * sizeof(float)) {
            printf("Skipping size %d (insufficient file data)\n", size);
            continue;
        }

        compareDataSources(size);
    }
}

// Stress test

TEST_F(NVMeGPUIntegrationTest, ContinuousProcessing) {
    // Simulate continuous data processing for a fixed duration
    const int matrix_size = 512;
    const size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
    const int duration_seconds = 2;

    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaMalloc(&d_A, matrix_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B, matrix_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C, matrix_bytes), cudaSuccess);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);

    size_t total_bytes_processed = 0;
    size_t total_operations = 0;

    while (std::chrono::high_resolution_clock::now() < end_time) {
        // Read data
        off_t offset = (total_operations % 5) * matrix_bytes * 2;

        nvme_gpu_read(matrix_file.c_str(), offset, matrix_bytes, d_A);
        nvme_gpu_read(matrix_file.c_str(), offset + matrix_bytes, matrix_bytes, d_B);

        // Process
        launch_matmul_tiled(d_C, d_A, d_B, matrix_size, matrix_size, matrix_size);

        total_bytes_processed += 2 * matrix_bytes;
        total_operations++;
    }

    cudaDeviceSynchronize();

    auto actual_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = actual_end - start_time;

    double throughput_gb_s = (total_bytes_processed / 1e9) / elapsed.count();
    double ops_per_second = total_operations / elapsed.count();

    printf("Continuous processing stress test:\n");
    printf("  Duration: %.2f seconds\n", elapsed.count());
    printf("  Operations: %zu\n", total_operations);
    printf("  Data processed: %.2f GB\n", total_bytes_processed / 1e9);
    printf("  Throughput: %.2f GB/s\n", throughput_gb_s);
    printf("  Operations/sec: %.2f\n", ops_per_second);

    EXPECT_GT(ops_per_second, 10.0);  // Expect at least 10 ops/sec

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Set up CUDA device
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaSetDevice(0);

    return RUN_ALL_TESTS();
}