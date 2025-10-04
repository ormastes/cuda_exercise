/**
 * Tests for GPUDirect Storage Implementation
 *
 * Includes:
 * - Unit tests for individual functions
 * - Integration tests for end-to-end workflows
 * - Performance benchmarks
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <random>

// External function declarations from gds_nvme_io.cu
extern "C" {
    int init_gds();
    void cleanup_gds();
    ssize_t read_nvme_to_gpu(const char* path, off_t offset, size_t size, void* gpu_buffer);
    ssize_t write_gpu_to_nvme(const char* path, off_t offset, size_t size, const void* gpu_buffer);
    bool verify_pattern(void* gpu_buffer, size_t size, uint8_t pattern);
}

// Test fixture for GDS tests
class GDSTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        // Create test file with known pattern
        test_file = "/tmp/gds_test_" + std::to_string(getpid()) + ".dat";
        createTestFile();

        // Initialize GDS if available
        gds_available = (init_gds() == 0);
        if (!gds_available) {
            GTEST_SKIP() << "GDS not available - skipping test";
        }
    }

    void TearDown() override {
        if (gds_available) {
            cleanup_gds();
        }

        // Remove test file
        std::remove(test_file.c_str());
    }

    void createTestFile() {
        std::ofstream file(test_file, std::ios::binary);
        std::vector<uint8_t> data(test_file_size, test_pattern);

        // Make sure file is 4KB aligned in size
        size_t aligned_size = ((test_file_size + 4095) / 4096) * 4096;
        data.resize(aligned_size, test_pattern);

        file.write(reinterpret_cast<char*>(data.data()), data.size());
        file.close();
    }

    std::string test_file;
    size_t test_file_size = 16 * 1024 * 1024;  // 16 MB
    uint8_t test_pattern = 0xAB;
    bool gds_available = false;
};

// Unit Tests

TEST_F(GDSTest, InitializeAndCleanup) {
    // Test multiple init/cleanup cycles
    for (int i = 0; i < 3; i++) {
        cleanup_gds();
        EXPECT_EQ(init_gds(), 0);
    }
}

TEST_F(GDSTest, BasicRead) {
    const size_t size = 4096;  // 4KB aligned
    void* gpu_buffer;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);

    // Read from beginning of file
    ssize_t bytes_read = read_nvme_to_gpu(test_file.c_str(), 0, size, gpu_buffer);
    EXPECT_EQ(bytes_read, size);

    // Verify data
    EXPECT_TRUE(verify_pattern(gpu_buffer, size, test_pattern));

    cudaFree(gpu_buffer);
}

TEST_F(GDSTest, UnalignedRead) {
    const size_t size = 4097;  // Not aligned
    void* gpu_buffer;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);

    // Read with unaligned size (should still work but warn)
    ssize_t bytes_read = read_nvme_to_gpu(test_file.c_str(), 0, size, gpu_buffer);
    EXPECT_GT(bytes_read, 0);

    cudaFree(gpu_buffer);
}

TEST_F(GDSTest, LargeRead) {
    const size_t size = 8 * 1024 * 1024;  // 8MB
    void* gpu_buffer;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);

    // Read large chunk
    ssize_t bytes_read = read_nvme_to_gpu(test_file.c_str(), 0, size, gpu_buffer);
    EXPECT_EQ(bytes_read, size);

    // Verify pattern
    EXPECT_TRUE(verify_pattern(gpu_buffer, size, test_pattern));

    cudaFree(gpu_buffer);
}

TEST_F(GDSTest, ReadWithOffset) {
    const size_t size = 4096;
    const off_t offset = 4096;  // 4KB aligned offset
    void* gpu_buffer;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);

    ssize_t bytes_read = read_nvme_to_gpu(test_file.c_str(), offset, size, gpu_buffer);
    EXPECT_EQ(bytes_read, size);
    EXPECT_TRUE(verify_pattern(gpu_buffer, size, test_pattern));

    cudaFree(gpu_buffer);
}

TEST_F(GDSTest, WriteAndReadBack) {
    const size_t size = 4096;
    const uint8_t write_pattern = 0xCD;
    void* gpu_buffer;

    ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);
    ASSERT_EQ(cudaMemset(gpu_buffer, write_pattern, size), cudaSuccess);

    std::string write_file = "/tmp/gds_write_test_" + std::to_string(getpid()) + ".dat";

    // Write to file
    ssize_t bytes_written = write_gpu_to_nvme(write_file.c_str(), 0, size, gpu_buffer);
    EXPECT_EQ(bytes_written, size);

    // Clear buffer and read back
    ASSERT_EQ(cudaMemset(gpu_buffer, 0, size), cudaSuccess);
    ssize_t bytes_read = read_nvme_to_gpu(write_file.c_str(), 0, size, gpu_buffer);
    EXPECT_EQ(bytes_read, size);

    // Verify pattern
    EXPECT_TRUE(verify_pattern(gpu_buffer, size, write_pattern));

    cudaFree(gpu_buffer);
    std::remove(write_file.c_str());
}

// Integration Tests

class GDSIntegrationTest : public GDSTest {
protected:
    void SetUp() override {
        GDSTest::SetUp();

        if (!gds_available) {
            GTEST_SKIP() << "GDS not available - skipping integration test";
        }
    }
};

TEST_F(GDSIntegrationTest, MultiStreamConcurrentReads) {
    const int num_streams = 4;
    const size_t size_per_stream = 1024 * 1024;  // 1MB per stream

    std::vector<cudaStream_t> streams(num_streams);
    std::vector<void*> buffers(num_streams);

    // Create streams and allocate buffers
    for (int i = 0; i < num_streams; i++) {
        ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&buffers[i], size_per_stream), cudaSuccess);
    }

    // Launch concurrent reads
    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

    for (int i = 0; i < num_streams; i++) {
        off_t offset = i * size_per_stream;
        read_nvme_to_gpu(test_file.c_str(), offset, size_per_stream, buffers[i]);
    }

    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsed_ms;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

    double total_gb = (num_streams * size_per_stream) / 1e9;
    double bandwidth_gb_s = total_gb / (elapsed_ms / 1000);

    printf("Multi-stream bandwidth: %.2f GB/s\n", bandwidth_gb_s);

    // Verify all buffers
    for (int i = 0; i < num_streams; i++) {
        EXPECT_TRUE(verify_pattern(buffers[i], size_per_stream, test_pattern));
        cudaFree(buffers[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST_F(GDSIntegrationTest, LargeFileStreaming) {
    // Simulate streaming a large file in chunks
    const size_t chunk_size = 4 * 1024 * 1024;  // 4MB chunks
    const size_t total_size = test_file_size;

    void* gpu_buffer;
    ASSERT_EQ(cudaMalloc(&gpu_buffer, chunk_size), cudaSuccess);

    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

    size_t bytes_processed = 0;
    off_t offset = 0;

    while (bytes_processed < total_size) {
        size_t to_read = std::min(chunk_size, total_size - bytes_processed);
        ssize_t bytes_read = read_nvme_to_gpu(test_file.c_str(), offset, to_read, gpu_buffer);

        EXPECT_GT(bytes_read, 0);
        bytes_processed += bytes_read;
        offset += bytes_read;

        // Verify each chunk
        EXPECT_TRUE(verify_pattern(gpu_buffer, bytes_read, test_pattern));
    }

    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsed_ms;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

    double throughput_gb_s = (bytes_processed / 1e9) / (elapsed_ms / 1000);
    printf("Streaming throughput: %.2f GB/s\n", throughput_gb_s);

    cudaFree(gpu_buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Performance Benchmarks

class GDSPerformanceTest : public GDSTest {
protected:
    void benchmarkTransferSize(size_t size) {
        void* gpu_buffer;
        ASSERT_EQ(cudaMalloc(&gpu_buffer, size), cudaSuccess);

        // Warm up
        read_nvme_to_gpu(test_file.c_str(), 0, std::min(size, (size_t)4096), gpu_buffer);

        const int num_iterations = 10;
        cudaEvent_t start, stop;
        ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

        ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

        for (int i = 0; i < num_iterations; i++) {
            off_t offset = (i * 4096) % (test_file_size - size);
            ssize_t bytes_read = read_nvme_to_gpu(test_file.c_str(), offset, size, gpu_buffer);
            EXPECT_EQ(bytes_read, size);
        }

        ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

        float elapsed_ms;
        ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

        double total_gb = (size * num_iterations) / 1e9;
        double bandwidth_gb_s = total_gb / (elapsed_ms / 1000);
        double latency_us = (elapsed_ms * 1000) / num_iterations;

        printf("Size: %8zu B | Bandwidth: %6.2f GB/s | Latency: %8.2f us\n",
               size, bandwidth_gb_s, latency_us);

        cudaFree(gpu_buffer);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

TEST_F(GDSPerformanceTest, TransferSizeSweep) {
    if (!gds_available) {
        GTEST_SKIP() << "GDS not available";
    }

    printf("\n=== GDS Performance Benchmark ===\n");

    // Test various transfer sizes
    std::vector<size_t> sizes = {
        4096,           // 4 KB
        16384,          // 16 KB
        65536,          // 64 KB
        262144,         // 256 KB
        1048576,        // 1 MB
        4194304,        // 4 MB
        8388608         // 8 MB
    };

    for (size_t size : sizes) {
        benchmarkTransferSize(size);
    }
}

TEST_F(GDSPerformanceTest, RandomAccessPattern) {
    const size_t block_size = 4096;
    const int num_accesses = 1000;

    void* gpu_buffer;
    ASSERT_EQ(cudaMalloc(&gpu_buffer, block_size), cudaSuccess);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, test_file_size / block_size - 1);

    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);

    for (int i = 0; i < num_accesses; i++) {
        off_t offset = dist(gen) * block_size;
        read_nvme_to_gpu(test_file.c_str(), offset, block_size, gpu_buffer);
    }

    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsed_ms;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);

    double iops = (num_accesses * 1000.0) / elapsed_ms;
    printf("Random 4KB IOPS: %.0f\n", iops);

    cudaFree(gpu_buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Main function for standalone execution
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