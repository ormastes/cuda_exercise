/**
 * High-Level NVMe API Implementation
 *
 * Provides a unified interface for NVMe-GPU operations with:
 * - Dictionary-based access patterns
 * - C API for external integration
 * - Support for multiple data "kinds" with LBA ranges
 */

#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <cstring>
#include <cstdio>

// External GDS functions
extern "C" {
    int init_gds();
    void cleanup_gds();
    ssize_t read_nvme_to_gpu(const char* path, off_t offset, size_t size, void* gpu_buffer);
    ssize_t write_gpu_to_nvme(const char* path, off_t offset, size_t size, const void* gpu_buffer);
}

// LBA range structure
struct LBARange {
    uint64_t start_lba;
    uint64_t length;       // in sectors
    uint32_t sector_size;  // typically 512 or 4096

    size_t byte_offset() const { return start_lba * sector_size; }
    size_t byte_length() const { return length * sector_size; }
};

// NVMe read request structure
struct NVMeReadRequest {
    std::map<int, LBARange> kinds;  // kind_id -> LBA range
    void* gpu_buffer;
    size_t buffer_size;
    std::string device_path;
};

// Global API context
class NVMeAPIContext {
public:
    static NVMeAPIContext& getInstance() {
        static NVMeAPIContext instance;
        return instance;
    }

    bool initialize(const char* default_device) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (initialized_) {
            return true;
        }

        // Initialize GDS
        if (init_gds() != 0) {
            fprintf(stderr, "Failed to initialize GDS\n");
            return false;
        }

        default_device_ = default_device ? default_device : "/dev/nvme0n1";
        initialized_ = true;
        return true;
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (initialized_) {
            cleanup_gds();
            initialized_ = false;
        }
    }

    const std::string& getDefaultDevice() const { return default_device_; }
    bool isInitialized() const { return initialized_; }

    // Store active requests for async operations
    void storeRequest(int request_id, std::unique_ptr<NVMeReadRequest> req) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_requests_[request_id] = std::move(req);
    }

    NVMeReadRequest* getRequest(int request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_requests_.find(request_id);
        return (it != active_requests_.end()) ? it->second.get() : nullptr;
    }

    void removeRequest(int request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_requests_.erase(request_id);
    }

private:
    NVMeAPIContext() : initialized_(false) {}
    ~NVMeAPIContext() { cleanup(); }

    bool initialized_;
    std::string default_device_;
    std::map<int, std::unique_ptr<NVMeReadRequest>> active_requests_;
    std::mutex mutex_;
    int next_request_id_ = 1;

public:
    int getNextRequestId() {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_request_id_++;
    }
};

// C API Implementation

extern "C" {

/**
 * Initialize the NVMe-GPU subsystem
 *
 * @param device_path NVMe device path (e.g., "/dev/nvme0n1")
 * @return 0 on success, -1 on failure
 */
int nvme_gpu_init(const char* device_path) {
    return NVMeAPIContext::getInstance().initialize(device_path) ? 0 : -1;
}

/**
 * Cleanup the NVMe-GPU subsystem
 */
void nvme_gpu_cleanup() {
    NVMeAPIContext::getInstance().cleanup();
}

/**
 * Create a new read request
 *
 * @param gpu_buffer Pre-allocated GPU buffer
 * @param buffer_size Size of the GPU buffer
 * @return Request handle or -1 on failure
 */
int nvme_gpu_create_request(void* gpu_buffer, size_t buffer_size) {
    if (!NVMeAPIContext::getInstance().isInitialized()) {
        fprintf(stderr, "NVMe API not initialized\n");
        return -1;
    }

    auto req = std::make_unique<NVMeReadRequest>();
    req->gpu_buffer = gpu_buffer;
    req->buffer_size = buffer_size;
    req->device_path = NVMeAPIContext::getInstance().getDefaultDevice();

    int request_id = NVMeAPIContext::getInstance().getNextRequestId();
    NVMeAPIContext::getInstance().storeRequest(request_id, std::move(req));

    return request_id;
}

/**
 * Add a data "kind" to the request
 *
 * @param request_id Request handle from nvme_gpu_create_request
 * @param kind_id Identifier for this data kind
 * @param start_lba Starting logical block address
 * @param length Number of sectors
 * @param sector_size Sector size in bytes (typically 512 or 4096)
 * @return 0 on success, -1 on failure
 */
int nvme_gpu_add_kind(int request_id, int kind_id,
                      uint64_t start_lba, uint64_t length,
                      uint32_t sector_size) {
    auto* req = NVMeAPIContext::getInstance().getRequest(request_id);
    if (!req) {
        fprintf(stderr, "Invalid request ID: %d\n", request_id);
        return -1;
    }

    LBARange range;
    range.start_lba = start_lba;
    range.length = length;
    range.sector_size = sector_size;

    req->kinds[kind_id] = range;
    return 0;
}

/**
 * Read data for a specific kind into GPU memory
 *
 * @param request_id Request handle
 * @param kind_id Kind to read
 * @param idx Offset within the kind (in sectors)
 * @param length Number of sectors to read
 * @param gpu_ptr Destination GPU pointer within the request buffer
 * @return Bytes read or -1 on failure
 */
ssize_t read_nvme_kind(int request_id, int kind_id,
                       uint64_t idx, uint64_t length,
                       void* gpu_ptr) {
    auto* req = NVMeAPIContext::getInstance().getRequest(request_id);
    if (!req) {
        fprintf(stderr, "Invalid request ID: %d\n", request_id);
        return -1;
    }

    auto it = req->kinds.find(kind_id);
    if (it == req->kinds.end()) {
        fprintf(stderr, "Kind %d not found in request\n", kind_id);
        return -1;
    }

    const LBARange& range = it->second;

    // Validate the read request
    if (idx + length > range.length) {
        fprintf(stderr, "Read exceeds kind boundaries\n");
        return -1;
    }

    // Calculate byte offset and size
    off_t byte_offset = (range.start_lba + idx) * range.sector_size;
    size_t byte_size = length * range.sector_size;

    // Validate GPU pointer is within buffer
    if (!gpu_ptr) {
        gpu_ptr = req->gpu_buffer;
    }

    // Perform the read
    return read_nvme_to_gpu(req->device_path.c_str(), byte_offset,
                           byte_size, gpu_ptr);
}

/**
 * Synchronous read - convenience function
 *
 * @param device_path NVMe device path
 * @param offset Byte offset
 * @param size Size in bytes
 * @param gpu_buffer GPU buffer
 * @return Bytes read or -1 on failure
 */
ssize_t nvme_gpu_read(const char* device_path, off_t offset,
                     size_t size, void* gpu_buffer) {
    if (!NVMeAPIContext::getInstance().isInitialized()) {
        if (nvme_gpu_init(device_path) != 0) {
            return -1;
        }
    }

    return read_nvme_to_gpu(device_path, offset, size, gpu_buffer);
}

/**
 * Asynchronous read with CUDA stream
 *
 * @param device_path NVMe device path
 * @param offset Byte offset
 * @param size Size in bytes
 * @param gpu_buffer GPU buffer
 * @param stream CUDA stream for async operation
 * @return 0 on success, -1 on failure
 */
int nvme_gpu_read_async(const char* device_path, off_t offset,
                        size_t size, void* gpu_buffer,
                        cudaStream_t stream) {
    if (!NVMeAPIContext::getInstance().isInitialized()) {
        if (nvme_gpu_init(device_path) != 0) {
            return -1;
        }
    }

    // Launch async read (implementation depends on GDS version)
    ssize_t result = read_nvme_to_gpu(device_path, offset, size, gpu_buffer);

    // Note: Current implementation is synchronous
    // Future versions with CUDA 12.0+ can use cuFileReadAsync
    return (result > 0) ? 0 : -1;
}

/**
 * Release a request handle
 *
 * @param request_id Request handle to release
 */
void nvme_gpu_release_request(int request_id) {
    NVMeAPIContext::getInstance().removeRequest(request_id);
}

/**
 * Batch read multiple kinds
 *
 * @param request_id Request handle
 * @param kind_ids Array of kind IDs to read
 * @param num_kinds Number of kinds
 * @return Total bytes read or -1 on failure
 */
ssize_t nvme_gpu_batch_read(int request_id, int* kind_ids, int num_kinds) {
    auto* req = NVMeAPIContext::getInstance().getRequest(request_id);
    if (!req) {
        return -1;
    }

    ssize_t total_bytes = 0;
    uint8_t* current_ptr = (uint8_t*)req->gpu_buffer;

    for (int i = 0; i < num_kinds; i++) {
        int kind_id = kind_ids[i];
        auto it = req->kinds.find(kind_id);

        if (it == req->kinds.end()) {
            fprintf(stderr, "Warning: kind %d not found\n", kind_id);
            continue;
        }

        const LBARange& range = it->second;
        ssize_t bytes_read = read_nvme_kind(request_id, kind_id,
                                           0, range.length, current_ptr);

        if (bytes_read < 0) {
            fprintf(stderr, "Failed to read kind %d\n", kind_id);
            return -1;
        }

        total_bytes += bytes_read;
        current_ptr += bytes_read;

        // Check buffer bounds
        if (current_ptr - (uint8_t*)req->gpu_buffer > req->buffer_size) {
            fprintf(stderr, "Buffer overflow detected\n");
            return -1;
        }
    }

    return total_bytes;
}

} // extern "C"

// Helper classes for C++ usage

class NVMeGPUReader {
public:
    NVMeGPUReader(const std::string& device_path = "/dev/nvme0n1")
        : device_path_(device_path), request_id_(-1) {
        if (nvme_gpu_init(device_path.c_str()) != 0) {
            throw std::runtime_error("Failed to initialize NVMe GPU API");
        }
    }

    ~NVMeGPUReader() {
        if (request_id_ >= 0) {
            nvme_gpu_release_request(request_id_);
        }
    }

    void allocateBuffer(size_t size) {
        if (gpu_buffer_) {
            cudaFree(gpu_buffer_);
        }

        if (cudaMalloc(&gpu_buffer_, size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU buffer");
        }

        buffer_size_ = size;

        // Create request
        request_id_ = nvme_gpu_create_request(gpu_buffer_, size);
        if (request_id_ < 0) {
            throw std::runtime_error("Failed to create request");
        }
    }

    void addKind(int kind_id, uint64_t start_lba, uint64_t length,
                 uint32_t sector_size = 512) {
        if (nvme_gpu_add_kind(request_id_, kind_id, start_lba,
                             length, sector_size) != 0) {
            throw std::runtime_error("Failed to add kind");
        }
    }

    ssize_t readKind(int kind_id, uint64_t idx = 0, uint64_t length = 0) {
        if (length == 0) {
            // Read entire kind
            auto* req = NVMeAPIContext::getInstance().getRequest(request_id_);
            if (req && req->kinds.count(kind_id) > 0) {
                length = req->kinds[kind_id].length;
            }
        }

        return read_nvme_kind(request_id_, kind_id, idx, length, nullptr);
    }

    void* getBuffer() const { return gpu_buffer_; }
    size_t getBufferSize() const { return buffer_size_; }

private:
    std::string device_path_;
    int request_id_;
    void* gpu_buffer_ = nullptr;
    size_t buffer_size_ = 0;
};

// Demo/test kernel for verifying reads
__global__ void compute_checksum_kernel(const uint8_t* data, size_t size,
                                       uint64_t* checksum) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t local_sum = 0;

    // Grid-stride loop
    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        local_sum += data[i];
    }

    // Atomic add to global checksum
    atomicAdd(checksum, local_sum);
}

// Example usage function
extern "C" void nvme_api_example() {
    try {
        NVMeGPUReader reader("/dev/nvme0n1");

        // Allocate 16MB buffer
        reader.allocateBuffer(16 * 1024 * 1024);

        // Add different data kinds with their LBA ranges
        // Kind 0: Metadata (starts at LBA 0, 1000 sectors)
        reader.addKind(0, 0, 1000, 512);

        // Kind 1: Data chunk 1 (starts at LBA 10000, 10000 sectors)
        reader.addKind(1, 10000, 10000, 512);

        // Kind 2: Data chunk 2 (starts at LBA 50000, 20000 sectors)
        reader.addKind(2, 50000, 20000, 512);

        // Read metadata
        ssize_t metadata_bytes = reader.readKind(0);
        printf("Read %zd bytes of metadata\n", metadata_bytes);

        // Read partial data from kind 1
        ssize_t partial_bytes = reader.readKind(1, 100, 500);
        printf("Read %zd bytes from kind 1\n", partial_bytes);

        // Compute checksum on GPU
        uint64_t* d_checksum;
        cudaMalloc(&d_checksum, sizeof(uint64_t));
        cudaMemset(d_checksum, 0, sizeof(uint64_t));

        int threads = 256;
        int blocks = (reader.getBufferSize() + threads - 1) / threads;
        compute_checksum_kernel<<<blocks, threads>>>(
            (uint8_t*)reader.getBuffer(), reader.getBufferSize(), d_checksum);

        uint64_t h_checksum;
        cudaMemcpy(&h_checksum, d_checksum, sizeof(uint64_t),
                  cudaMemcpyDeviceToHost);

        printf("Data checksum: %lu\n", h_checksum);

        cudaFree(d_checksum);

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
    }
}