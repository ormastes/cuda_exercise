/**
 * GPUDirect Storage Implementation
 *
 * This file provides direct NVMe-to-GPU memory transfer using NVIDIA's
 * GPUDirect Storage (GDS) technology with the cuFile API.
 *
 * Key Features:
 * - Zero-copy transfers from NVMe to GPU memory
 * - Synchronous and asynchronous operations
 * - Batch read support
 * - 4KB alignment handling
 */

#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

// Only compile if GDS is available
#ifdef HAS_GDS
#include <cufile.h>

// Error checking macros
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUFILE(call) do { \
    CUfileError_t err = call; \
    if (err.err != CU_FILE_SUCCESS) { \
        fprintf(stderr, "cuFile error at %s:%d: %s\n", __FILE__, __LINE__, \
                cufileop_status_error(err.err)); \
        exit(1); \
    } \
} while(0)

// GDS context structure
typedef struct {
    CUfileHandle_t handle;
    CUfileDescr_t descr;
    int fd;
    bool initialized;
} GDSContext;

static GDSContext g_gds_ctx = {0};
static bool g_driver_initialized = false;

/**
 * Initialize GDS driver (call once per application)
 */
extern "C" int init_gds() {
    if (g_driver_initialized) {
        return 0;
    }

    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "Failed to initialize GDS driver: %s\n",
                cufileop_status_error(status.err));
        return -1;
    }

    g_driver_initialized = true;
    return 0;
}

/**
 * Cleanup GDS driver
 */
extern "C" void cleanup_gds() {
    if (g_driver_initialized) {
        cuFileDriverClose();
        g_driver_initialized = false;
    }
}

/**
 * Open a file/device for GDS operations
 */
extern "C" int open_gds_file(const char* path, GDSContext* ctx) {
    if (!ctx) return -1;

    // Open with O_DIRECT for best performance
    ctx->fd = open(path, O_RDONLY | O_DIRECT);
    if (ctx->fd < 0) {
        perror("Failed to open file");
        return -1;
    }

    // Register file with cuFile
    memset(&ctx->descr, 0, sizeof(CUfileDescr_t));
    ctx->descr.handle.fd = ctx->fd;
    ctx->descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status = cuFileHandleRegister(&ctx->handle, &ctx->descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(ctx->fd);
        fprintf(stderr, "Failed to register file handle: %s\n",
                cufileop_status_error(status.err));
        return -1;
    }

    ctx->initialized = true;
    return 0;
}

/**
 * Close GDS file handle
 */
extern "C" void close_gds_file(GDSContext* ctx) {
    if (ctx && ctx->initialized) {
        cuFileHandleDeregister(ctx->handle);
        close(ctx->fd);
        ctx->initialized = false;
    }
}

/**
 * Synchronous read from NVMe to GPU memory
 *
 * @param path NVMe device or file path
 * @param offset Byte offset (must be 4KB aligned)
 * @param size Size in bytes (preferably 4KB multiple)
 * @param gpu_buffer Device memory pointer
 * @return Bytes read or -1 on error
 */
extern "C" ssize_t read_nvme_to_gpu(const char* path, off_t offset,
                                    size_t size, void* gpu_buffer) {
    if (!g_driver_initialized) {
        if (init_gds() != 0) return -1;
    }

    // Check alignment
    if (offset % 4096 != 0) {
        fprintf(stderr, "Warning: offset not 4KB aligned. Performance may degrade.\n");
    }

    GDSContext ctx = {0};
    if (open_gds_file(path, &ctx) != 0) {
        return -1;
    }

    // Register GPU buffer with cuFile
    CUfileError_t status = cuFileBufRegister(gpu_buffer, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        close_gds_file(&ctx);
        fprintf(stderr, "Failed to register GPU buffer: %s\n",
                cufileop_status_error(status.err));
        return -1;
    }

    // Perform the read
    ssize_t bytes_read = cuFileRead(ctx.handle, gpu_buffer, size, offset, 0);

    // Cleanup
    cuFileBufDeregister(gpu_buffer);
    close_gds_file(&ctx);

    return bytes_read;
}

/**
 * Write from GPU memory to NVMe
 */
extern "C" ssize_t write_gpu_to_nvme(const char* path, off_t offset,
                                     size_t size, const void* gpu_buffer) {
    if (!g_driver_initialized) {
        if (init_gds() != 0) return -1;
    }

    GDSContext ctx = {0};

    // Open for writing
    ctx.fd = open(path, O_WRONLY | O_DIRECT | O_CREAT, 0644);
    if (ctx.fd < 0) {
        perror("Failed to open file for writing");
        return -1;
    }

    memset(&ctx.descr, 0, sizeof(CUfileDescr_t));
    ctx.descr.handle.fd = ctx.fd;
    ctx.descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status = cuFileHandleRegister(&ctx.handle, &ctx.descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(ctx.fd);
        return -1;
    }

    // Register buffer and write
    status = cuFileBufRegister(const_cast<void*>(gpu_buffer), size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        cuFileHandleDeregister(ctx.handle);
        close(ctx.fd);
        return -1;
    }

    ssize_t bytes_written = cuFileWrite(ctx.handle, gpu_buffer, size, offset, 0);

    // Cleanup
    cuFileBufDeregister(const_cast<void*>(gpu_buffer));
    cuFileHandleDeregister(ctx.handle);
    close(ctx.fd);

    return bytes_written;
}

/**
 * Batch read operation for multiple buffers
 */
typedef struct {
    void* gpu_buffer;
    size_t size;
    off_t offset;
    cudaStream_t stream;
} BatchReadRequest;

extern "C" int async_batch_read(const char* path, BatchReadRequest* requests,
                                int num_requests) {
    if (!g_driver_initialized) {
        if (init_gds() != 0) return -1;
    }

    GDSContext ctx = {0};
    if (open_gds_file(path, &ctx) != 0) {
        return -1;
    }

    // Process each request
    for (int i = 0; i < num_requests; i++) {
        BatchReadRequest* req = &requests[i];

        // Register buffer
        CUfileError_t status = cuFileBufRegister(req->gpu_buffer, req->size, 0);
        if (status.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "Failed to register buffer %d\n", i);
            continue;
        }

        // Async read (if CUDA 12.0+)
        #ifdef CUFILE_READ_ASYNC_SUPPORTED
        cuFileReadAsync(ctx.handle, req->gpu_buffer, req->size,
                       req->offset, 0, req->stream);
        #else
        // Fallback to sync read
        cuFileRead(ctx.handle, req->gpu_buffer, req->size, req->offset, 0);
        #endif

        cuFileBufDeregister(req->gpu_buffer);
    }

    // Wait for all streams
    for (int i = 0; i < num_requests; i++) {
        if (requests[i].stream) {
            cudaStreamSynchronize(requests[i].stream);
        }
    }

    close_gds_file(&ctx);
    return 0;
}

/**
 * Benchmark kernel to verify data
 */
__global__ void verify_pattern_kernel(const uint8_t* data, size_t size,
                                      int* result, uint8_t pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] != pattern) {
            atomicExch(result, 0);
        }
    }
}

/**
 * Helper function to verify read data
 */
extern "C" bool verify_pattern(void* gpu_buffer, size_t size, uint8_t pattern) {
    int* d_result;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_result, 1, sizeof(int)));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    verify_pattern_kernel<<<blocks, threads>>>((uint8_t*)gpu_buffer, size,
                                               d_result, pattern);

    int h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_result));

    return h_result == 1;
}

#else // !HAS_GDS

// Stub implementations when GDS is not available
extern "C" {
    int init_gds() {
        fprintf(stderr, "GDS not available. Install nvidia-gds package.\n");
        return -1;
    }

    void cleanup_gds() {}

    ssize_t read_nvme_to_gpu(const char* path, off_t offset,
                            size_t size, void* gpu_buffer) {
        fprintf(stderr, "GDS not available\n");
        return -1;
    }

    ssize_t write_gpu_to_nvme(const char* path, off_t offset,
                             size_t size, const void* gpu_buffer) {
        fprintf(stderr, "GDS not available\n");
        return -1;
    }

    bool verify_pattern(void* gpu_buffer, size_t size, uint8_t pattern) {
        return false;
    }
}

#endif // HAS_GDS

/**
 * Demo main function
 */
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <device> <offset> <size>\n", argv[0]);
        return 1;
    }

    const char* device = argv[1];
    off_t offset = atoll(argv[2]);
    size_t size = atoll(argv[3]);

    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(0));

    // Allocate GPU buffer
    void* gpu_buffer;
    CHECK_CUDA(cudaMalloc(&gpu_buffer, size));

    // Initialize GDS and read
    if (init_gds() == 0) {
        printf("Reading %zu bytes from %s at offset %ld\n", size, device, offset);

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        ssize_t bytes_read = read_nvme_to_gpu(device, offset, size, gpu_buffer);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        if (bytes_read > 0) {
            float milliseconds = 0;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

            double bandwidth_gb = (bytes_read / 1e9) / (milliseconds / 1000);
            printf("Read %zd bytes in %.2f ms (%.2f GB/s)\n",
                   bytes_read, milliseconds, bandwidth_gb);
        } else {
            fprintf(stderr, "Read failed\n");
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        cleanup_gds();
    }

    CHECK_CUDA(cudaFree(gpu_buffer));
    return 0;
}