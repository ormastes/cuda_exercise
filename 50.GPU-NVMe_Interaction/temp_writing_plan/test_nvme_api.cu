/**
 * Tests for High-Level NVMe API
 *
 * Tests the dictionary-based access patterns, C API, and integration
 * with GDS backend.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <random>
#include <thread>
#include <chrono>

// C API function declarations
extern "C" {
    int nvme_gpu_init(const char* device_path);
    void nvme_gpu_cleanup();
    int nvme_gpu_create_request(void* gpu_buffer, size_t buffer_size);
    int nvme_gpu_add_kind(int request_id, int kind_id, uint64_t start_lba,
                         uint64_t length, uint32_t sector_size);
    ssize_t read_nvme_kind(int request_id, int kind_id, uint64_t idx,
                          uint64_t length, void* gpu_ptr);
    ssize_t nvme_gpu_read(const char* device_path, off_t offset,
                         size_t size, void* gpu_buffer);
    int nvme_gpu_read_async(const char* device_path, off_t offset,
                           size_t size, void* gpu_buffer, cudaStream_t stream);
    void nvme_gpu_release_request(int request_id);
    ssize_t nvme_gpu_batch_read(int request_id, int* kind_ids, int num_kinds);
}

// Test fixture
class NVMeAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        // Create test file
        test_file = "/tmp/nvme_api_test_" + std::to_string(getpid()) + ".dat";
        createTestFile();

        // Initialize API
        if (nvme_gpu_init(test_file.c_str()) != 0) {
            GTEST_SKIP() << "NVMe API initialization failed";
        }
    }

    void TearDown() override {
        nvme_gpu_cleanup();
        std::remove(test_file.c_str());
    }

    void createTestFile() {
        const size_t file_size = 100 * 1024 * 1024;  // 100MB
        std::ofstream file(test_file, std::ios::binary);

        // Create file with different patterns for different "kinds"
        std::vector<uint8_t> buffer(1024 * 1024);

        for (size_t offset = 0; offset < file_size; offset += buffer.size()) {
            // Different pattern based on offset
            uint8_t pattern = static_cast<uint8_t>((offset / (10 * 1024 * 1024)) + 1);
            std::fill(buffer.begin(), buffer.end(), pattern);
            file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
        }

        file.close();
    }

    std::string test_file;
};

// Unit Tests

TEST_F(NVMeAPITest, InitializeAndCleanup) {
    // Already initialized in SetUp
    // Test re-initialization
    EXPECT_EQ(nvme_gpu_init(test_file.c_str()), 0);

    nvme_gpu_cleanup();
    EXPECT_EQ(nvme_gpu_init(test_file.c_str()), 0);
}

TEST_F(NVMeAPITest, CreateRequest) {
    void* gpu_buffer;
    const size_t buffer_size = 1024 * 1024;  // 1MB

    ASSERT_EQ(cudaMalloc(&gpu_buffer, buffer_size), cudaSuccess);

    int request_id = nvme_gpu_create_request(gpu_buffer, buffer_size);
    EXPECT_GE(request_id, 0);

    nvme_gpu_release_request(request_id);
    cudaFree(gpu_buffer);
}

TEST_F(NVMeAPITest, AddKinds) {
    void* gpu_buffer;
    const size_t buffer_size = 10 * 1024 * 1024;  // 10MB

    ASSERT_EQ(cudaMalloc(&gpu_buffer, buffer_size), cudaSuccess);

    int request_id = nvme_gpu_create_request(gpu_buffer, buffer_size);
    ASSERT_GE(request_id, 0);

    // Add multiple kinds
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 0, 0, 1000, 512), 0);
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 1, 10000, 2000, 512), 0);
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 2, 50000, 5000, 512), 0);

    nvme_gpu_release_request(request_id);
    cudaFree(gpu_buffer);
}

TEST_F(NVMeAPITest, ReadKind) {
    void* gpu_buffer;
    const size_t buffer_size = 10 * 1024 * 1024;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, buffer_size), cudaSuccess);

    int request_id = nvme_gpu_create_request(gpu_buffer, buffer_size);
    ASSERT_GE(request_id, 0);

    // Add a kind (first 1MB of file)
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 0, 0, 2048, 512), 0);

    // Read the entire kind
    ssize_t bytes_read = read_nvme_kind(request_id, 0, 0, 2048, gpu_buffer);
    EXPECT_EQ(bytes_read, 2048 * 512);

    // Verify data pattern (first region should have pattern 1)
    std::vector<uint8_t> host_buffer(bytes_read);
    ASSERT_EQ(cudaMemcpy(host_buffer.data(), gpu_buffer, bytes_read,
                        cudaMemcpyDeviceToHost), cudaSuccess);

    bool pattern_correct = std::all_of(host_buffer.begin(), host_buffer.end(),
                                       [](uint8_t val) { return val == 1; });
    EXPECT_TRUE(pattern_correct);

    nvme_gpu_release_request(request_id);
    cudaFree(gpu_buffer);
}

TEST_F(NVMeAPITest, PartialReadKind) {
    void* gpu_buffer;
    const size_t buffer_size = 10 * 1024 * 1024;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, buffer_size), cudaSuccess);

    int request_id = nvme_gpu_create_request(gpu_buffer, buffer_size);
    ASSERT_GE(request_id, 0);

    // Add a large kind
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 0, 0, 10000, 512), 0);

    // Read partial data
    ssize_t bytes_read = read_nvme_kind(request_id, 0, 100, 500, gpu_buffer);
    EXPECT_EQ(bytes_read, 500 * 512);

    nvme_gpu_release_request(request_id);
    cudaFree(gpu_buffer);
}

TEST_F(NVMeAPITest, BatchRead) {
    void* gpu_buffer;
    const size_t buffer_size = 20 * 1024 * 1024;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, buffer_size), cudaSuccess);

    int request_id = nvme_gpu_create_request(gpu_buffer, buffer_size);
    ASSERT_GE(request_id, 0);

    // Add multiple kinds
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 0, 0, 1000, 512), 0);
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 1, 2000, 2000, 512), 0);
    EXPECT_EQ(nvme_gpu_add_kind(request_id, 2, 5000, 3000, 512), 0);

    // Batch read all kinds
    int kind_ids[] = {0, 1, 2};
    ssize_t total_bytes = nvme_gpu_batch_read(request_id, kind_ids, 3);

    size_t expected_bytes = (1000 + 2000 + 3000) * 512;
    EXPECT_EQ(total_bytes, expected_bytes);

    nvme_gpu_release_request(request_id);
    cudaFree(gpu_buffer);
}

TEST_F(NVMeAPITest, SimpleRead) {
    void* gpu_buffer;
    const size_t size = 4096;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);

    ssize_t bytes_read = nvme_gpu_read(test_file.c_str(), 0, size, gpu_buffer);
    EXPECT_EQ(bytes_read, size);

    cudaFree(gpu_buffer);
}

TEST_F(NVMeAPITest, AsyncRead) {
    void* gpu_buffer;
    const size_t size = 1024 * 1024;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    int result = nvme_gpu_read_async(test_file.c_str(), 0, size,
                                     gpu_buffer, stream);
    EXPECT_EQ(result, 0);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaStreamDestroy(stream);
    cudaFree(gpu_buffer);
}

// Integration Tests

class NVMeAPIIntegrationTest : public NVMeAPITest {
protected:
    void verifyDataPattern(void* gpu_buffer, size_t size, uint8_t expected_pattern) {
        std::vector<uint8_t> host_buffer(size);
        ASSERT_EQ(cudaMemcpy(host_buffer.data(), gpu_buffer, size,
                            cudaMemcpyDeviceToHost), cudaSuccess);

        size_t mismatches = 0;
        for (size_t i = 0; i < size; i++) {
            if (host_buffer[i] != expected_pattern) {
                mismatches++;
            }
        }

        EXPECT_EQ(mismatches, 0) << "Found " << mismatches
                                 << " mismatches out of " << size << " bytes";
    }
};

TEST_F(NVMeAPIIntegrationTest, MultiKindWorkflow) {
    // Simulate a real workflow with multiple data kinds
    void* gpu_buffer;
    const size_t buffer_size = 30 * 1024 * 1024;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, buffer_size), cudaSuccess);

    int request_id = nvme_gpu_create_request(gpu_buffer, buffer_size);
    ASSERT_GE(request_id, 0);

    // Define kinds representing different data regions
    // Each 10MB region has a different pattern (1, 2, 3, ...)
    struct KindDefinition {
        int kind_id;
        uint64_t start_lba;
        uint64_t length;
        uint8_t expected_pattern;
    };

    std::vector<KindDefinition> kinds = {
        {0, 0, 20480, 1},           // First 10MB
        {1, 20480, 20480, 2},       // Second 10MB
        {2, 40960, 20480, 3}        // Third 10MB
    };

    // Add all kinds
    for (const auto& kind : kinds) {
        EXPECT_EQ(nvme_gpu_add_kind(request_id, kind.kind_id,
                                    kind.start_lba, kind.length, 512), 0);
    }

    // Read each kind and verify
    uint8_t* current_ptr = (uint8_t*)gpu_buffer;
    for (const auto& kind : kinds) {
        ssize_t bytes_read = read_nvme_kind(request_id, kind.kind_id,
                                           0, kind.length, current_ptr);
        EXPECT_EQ(bytes_read, kind.length * 512);

        verifyDataPattern(current_ptr, bytes_read, kind.expected_pattern);
        current_ptr += bytes_read;
    }

    nvme_gpu_release_request(request_id);
    cudaFree(gpu_buffer);
}

TEST_F(NVMeAPIIntegrationTest, ConcurrentRequests) {
    const int num_requests = 4;
    const size_t buffer_size = 5 * 1024 * 1024;

    std::vector<void*> buffers(num_requests);
    std::vector<int> request_ids(num_requests);
    std::vector<std::thread> threads;

    // Create multiple requests
    for (int i = 0; i < num_requests; i++) {
        ASSERT_EQ(cudaMalloc(&buffers[i], buffer_size), cudaSuccess);
        request_ids[i] = nvme_gpu_create_request(buffers[i], buffer_size);
        ASSERT_GE(request_ids[i], 0);

        // Add kind for this request
        EXPECT_EQ(nvme_gpu_add_kind(request_ids[i], 0,
                                    i * 10240, 10240, 512), 0);
    }

    // Launch concurrent reads
    auto read_func = [](int request_id) {
        ssize_t bytes = read_nvme_kind(request_id, 0, 0, 10240, nullptr);
        EXPECT_EQ(bytes, 10240 * 512);
    };

    for (int i = 0; i < num_requests; i++) {
        threads.emplace_back(read_func, request_ids[i]);
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Cleanup
    for (int i = 0; i < num_requests; i++) {
        nvme_gpu_release_request(request_ids[i]);
        cudaFree(buffers[i]);
    }
}

TEST_F(NVMeAPIIntegrationTest, LargeScaleDataProcessing) {
    // Simulate processing a large dataset with multiple passes
    const size_t chunk_size = 4 * 1024 * 1024;  // 4MB chunks
    const int num_chunks = 10;

    void* gpu_buffer;
    ASSERT_EQ(cudaMalloc(&gpu_buffer, chunk_size), cudaSuccess);

    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

    size_t total_bytes_processed = 0;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        off_t offset = chunk * chunk_size;

        ssize_t bytes_read = nvme_gpu_read(test_file.c_str(), offset,
                                          chunk_size, gpu_buffer);
        EXPECT_EQ(bytes_read, chunk_size);

        // Simulate some GPU processing (just a simple kernel)
        dim3 block(256);
        dim3 grid((chunk_size + block.x - 1) / block.x);

        // Dummy kernel to simulate work
        auto process_kernel = [] __device__ (uint8_t* data, size_t size) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                data[idx] = data[idx] ^ 0xFF;  // Simple XOR operation
            }
        };

        // Note: Lambda kernels require CUDA 11+ and --extended-lambda flag
        // For older CUDA, use a regular __global__ function

        total_bytes_processed += bytes_read;
    }

    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsed_ms;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

    double throughput_gb_s = (total_bytes_processed / 1e9) / (elapsed_ms / 1000);
    printf("Large-scale processing throughput: %.2f GB/s\n", throughput_gb_s);

    cudaFree(gpu_buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Performance Tests

class NVMeAPIPerformanceTest : public NVMeAPITest {
protected:
    void benchmarkKindAccess(int num_kinds, size_t sectors_per_kind) {
        const size_t buffer_size = num_kinds * sectors_per_kind * 512;
        void* gpu_buffer;

        ASSERT_EQ(cudaMalloc(&gpu_buffer, buffer_size), cudaSuccess);

        int request_id = nvme_gpu_create_request(gpu_buffer, buffer_size);
        ASSERT_GE(request_id, 0);

        // Add kinds
        for (int i = 0; i < num_kinds; i++) {
            uint64_t start_lba = i * sectors_per_kind * 2;  // Space between kinds
            EXPECT_EQ(nvme_gpu_add_kind(request_id, i, start_lba,
                                        sectors_per_kind, 512), 0);
        }

        // Benchmark batch read
        cudaEvent_t start, stop;
        ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

        std::vector<int> kind_ids(num_kinds);
        std::iota(kind_ids.begin(), kind_ids.end(), 0);

        ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

        ssize_t total_bytes = nvme_gpu_batch_read(request_id, kind_ids.data(),
                                                  num_kinds);

        ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

        float elapsed_ms;
        ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

        double throughput_gb_s = (total_bytes / 1e9) / (elapsed_ms / 1000);
        printf("Batch read %d kinds (%.2f MB total): %.2f GB/s\n",
               num_kinds, total_bytes / 1e6, throughput_gb_s);

        nvme_gpu_release_request(request_id);
        cudaFree(gpu_buffer);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

TEST_F(NVMeAPIPerformanceTest, KindAccessScaling) {
    printf("\n=== Kind Access Performance ===\n");

    // Test with different numbers of kinds
    benchmarkKindAccess(1, 2048);     // 1 kind, 1MB
    benchmarkKindAccess(10, 2048);    // 10 kinds, 1MB each
    benchmarkKindAccess(100, 2048);   // 100 kinds, 1MB each
    benchmarkKindAccess(10, 20480);   // 10 kinds, 10MB each
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