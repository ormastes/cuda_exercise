# 🛡️ Part 6: Error Handling and Debugging

**Goal**: Build robust CUDA applications with comprehensive error handling strategies and debugging techniques.

---

## **6.1 CUDA Error Types and Codes**

### **6.1.1 Error Categories**

CUDA errors fall into several categories:

1. **Synchronous Errors**: Returned immediately by CUDA API calls
2. **Asynchronous Errors**: Occur during kernel execution or async operations
3. **Sticky Errors**: Persist until explicitly cleared
4. **Driver Errors**: Related to CUDA driver issues

### **6.1.2 Common Error Codes**

```cpp
// Common CUDA error codes
cudaSuccess                    // No errors (0)
cudaErrorInvalidValue          // Invalid argument
cudaErrorMemoryAllocation      // Out of memory
cudaErrorInvalidDevice         // Invalid device ordinal
cudaErrorInvalidConfiguration  // Invalid kernel configuration
cudaErrorLaunchFailure        // Kernel launch failed
cudaErrorIllegalAddress       // Out-of-bounds memory access
cudaErrorMisalignedAddress    // Misaligned memory access
cudaErrorInvalidPitchValue    // Invalid pitch for cudaMemcpy2D
cudaErrorNotReady            // Async operation not complete
```

### **6.1.3 Getting Error Strings**

```cpp
cudaError_t error = cudaMalloc(&d_ptr, size);
if (error != cudaSuccess) {
    // Get human-readable error string
    const char* errorString = cudaGetErrorString(error);
    fprintf(stderr, "CUDA Error: %s\n", errorString);

    // Get detailed error description
    const char* errorName = cudaGetErrorName(error);
    fprintf(stderr, "Error Code: %s\n", errorName);
}
```

---

## **6.2 Error Checking Macros**

### **6.2.1 Basic Error Check Macro**

```cpp
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Usage
CHECK_CUDA(cudaMalloc(&d_data, size));
CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

### **6.2.2 Advanced Error Handling with C++ Exceptions**

```cpp
#include <stdexcept>
#include <sstream>

class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t error, const char* file, int line)
        : std::runtime_error(buildErrorMessage(error, file, line)),
          error_(error) {}

    cudaError_t getError() const { return error_; }

private:
    cudaError_t error_;

    static std::string buildErrorMessage(cudaError_t error,
                                        const char* file, int line) {
        std::ostringstream oss;
        oss << "CUDA Error at " << file << ":" << line
            << " - " << cudaGetErrorString(error);
        return oss.str();
    }
};

#define CHECK_CUDA_THROW(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw CudaException(error, __FILE__, __LINE__); \
    } \
} while(0)
```

### **6.2.3 Debug-Only Checks**

```cpp
#ifdef DEBUG
    #define CHECK_CUDA_DEBUG(call) CHECK_CUDA(call)
    #define CHECK_LAST_CUDA_ERROR() CHECK_CUDA(cudaGetLastError())
#else
    #define CHECK_CUDA_DEBUG(call) (call)
    #define CHECK_LAST_CUDA_ERROR()
#endif
```

---

## **6.3 Synchronous vs Asynchronous Errors**

### **6.3.1 Synchronous Error Handling**

```cpp
// Synchronous errors are returned immediately
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    // Handle error immediately
    handleError(err);
}
```

### **6.3.2 Asynchronous Error Handling**

```cpp
// Kernel launch (asynchronous)
myKernel<<<blocks, threads>>>(d_data, size);

// Check for launch errors (synchronous check)
CHECK_CUDA(cudaGetLastError());

// Wait for kernel completion and check for runtime errors
CHECK_CUDA(cudaDeviceSynchronize());

// Alternative: Check without synchronizing
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
}
```

### **6.3.3 Stream-Based Error Handling**

```cpp
cudaStream_t stream;
CHECK_CUDA(cudaStreamCreate(&stream));

// Launch kernel on stream
myKernel<<<blocks, threads, 0, stream>>>(d_data, size);

// Query stream status without blocking
cudaError_t err = cudaStreamQuery(stream);
if (err == cudaErrorNotReady) {
    // Stream operations still in progress
} else if (err != cudaSuccess) {
    // Handle error
}

// Synchronize and check for errors
CHECK_CUDA(cudaStreamSynchronize(stream));
```

---

## **6.4 Debugging Memory Errors**

### **6.4.1 Common Memory Errors**

1. **Out-of-bounds access**
2. **Use after free**
3. **Double free**
4. **Memory leaks**
5. **Uninitialized memory access**

### **6.4.2 Memory Error Detection Strategies**

```cpp
// Enable memory checking in debug builds
void enableMemoryChecking() {
#ifdef DEBUG
    // Enable CUDA memory checking
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    // Enable device-side assertions
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 2);
#endif
}

// Safe memory allocation with tracking
template<typename T>
class CudaMemory {
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;

public:
    CudaMemory(size_t count) : size_(count * sizeof(T)) {
        CHECK_CUDA(cudaMalloc(&ptr_, size_));
    }

    ~CudaMemory() {
        if (ptr_) {
            cudaFree(ptr_);  // Ignore errors in destructor
        }
    }

    // Prevent copying
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    // Allow moving
    CudaMemory(CudaMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    T* get() { return ptr_; }
    size_t size() const { return size_; }
};
```

### **6.4.3 Bounds Checking in Kernels**

```cpp
__device__ void assertInBounds(int index, int size,
                              const char* file, int line) {
#ifdef DEBUG
    if (index < 0 || index >= size) {
        printf("CUDA Assert: Index %d out of bounds [0, %d) at %s:%d\n",
               index, size, file, line);
        __trap();  // Trigger a trap
    }
#endif
}

#define CUDA_ASSERT_BOUNDS(idx, size) \
    assertInBounds(idx, size, __FILE__, __LINE__)

__global__ void safeKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds checking in debug mode
    CUDA_ASSERT_BOUNDS(idx, size);

    if (idx < size) {
        data[idx] = idx * 2.0f;
    }
}
```

---

## **6.5 Race Condition Detection**

### **6.5.1 Common Race Conditions**

```cpp
// Example of race condition in shared memory
__global__ void raceyKernel(float* result) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    sdata[tid] = tid;

    // RACE CONDITION: Missing synchronization
    // __syncthreads();  // <-- This is needed!

    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];  // Race condition!
    }

    if (tid == 0) {
        *result = sdata[0];
    }
}
```

### **6.5.2 Detecting Race Conditions**

```cpp
// Use atomic operations to detect races
__device__ int raceDetector = 0;

__global__ void detectRace(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Increment before access
        int accessCount = atomicAdd(&raceDetector, 1);

        // Simulated work
        data[idx] = idx;

        // Decrement after access
        atomicSub(&raceDetector, 1);

        // Check for concurrent access
        if (raceDetector > 1) {
            printf("Race detected at index %d\n", idx);
        }
    }
}
```

---

## **6.6 Deadlock Prevention**

### **6.6.1 Common Deadlock Scenarios**

```cpp
// Deadlock example: Conditional synchronization
__global__ void deadlockKernel() {
    int tid = threadIdx.x;

    // DEADLOCK: Not all threads reach sync point
    if (tid < 16) {
        __syncthreads();  // Only threads 0-15 reach here
    }
    // Threads 16+ never reach the barrier -> deadlock!
}

// Correct version
__global__ void correctKernel() {
    int tid = threadIdx.x;
    __shared__ int flag;

    if (tid < 16) {
        // Do work for threads 0-15
        flag = 1;
    }

    __syncthreads();  // All threads reach this

    // All threads can now safely use flag
}
```

### **6.6.2 Deadlock Detection and Prevention**

```cpp
class DeadlockDetector {
private:
    cudaEvent_t start_, stop_;
    float timeout_ms_;

public:
    DeadlockDetector(float timeout_ms = 5000.0f)
        : timeout_ms_(timeout_ms) {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~DeadlockDetector() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void startTiming() {
        CHECK_CUDA(cudaEventRecord(start_));
    }

    bool checkDeadlock() {
        cudaError_t err = cudaEventQuery(stop_);
        if (err == cudaErrorNotReady) {
            float elapsed;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed, start_, stop_));
            if (elapsed > timeout_ms_) {
                return true;  // Potential deadlock
            }
        }
        return false;
    }
};
```

---

## **6.7 Error Recovery Strategies**

### **6.7.1 Graceful Degradation**

```cpp
class CudaComputation {
private:
    bool useCuda_ = true;

public:
    void compute(float* data, int size) {
        if (useCuda_) {
            cudaError_t err = computeOnGPU(data, size);
            if (err != cudaSuccess) {
                fprintf(stderr, "GPU computation failed: %s\n",
                       cudaGetErrorString(err));
                fprintf(stderr, "Falling back to CPU\n");
                useCuda_ = false;
                computeOnCPU(data, size);
            }
        } else {
            computeOnCPU(data, size);
        }
    }

private:
    cudaError_t computeOnGPU(float* data, int size);
    void computeOnCPU(float* data, int size);
};
```

### **6.7.2 Retry Mechanism**

```cpp
template<typename Func>
bool retryOperation(Func operation, int maxRetries = 3) {
    for (int attempt = 0; attempt < maxRetries; ++attempt) {
        cudaError_t err = operation();

        if (err == cudaSuccess) {
            return true;
        }

        fprintf(stderr, "Attempt %d failed: %s\n",
                attempt + 1, cudaGetErrorString(err));

        // Reset CUDA context for certain errors
        if (err == cudaErrorIllegalAddress ||
            err == cudaErrorLaunchFailure) {
            cudaDeviceReset();
        }

        // Wait before retry
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100 * (attempt + 1)));
    }

    return false;
}

// Usage
bool success = retryOperation([&]() {
    return launchKernel(data, size);
});
```

---

## **6.8 Logging and Diagnostics**

### **6.8.1 Comprehensive Logging System**

```cpp
enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3
};

class CudaLogger {
private:
    LogLevel level_;
    std::ofstream logFile_;

public:
    CudaLogger(const std::string& filename, LogLevel level = LOG_INFO)
        : level_(level), logFile_(filename, std::ios::app) {}

    void log(LogLevel level, const std::string& message) {
        if (level >= level_) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);

            logFile_ << "[" << std::put_time(std::localtime(&time_t),
                                            "%Y-%m-%d %H:%M:%S") << "] ";
            logFile_ << levelToString(level) << ": " << message << std::endl;
        }
    }

    void logCudaError(cudaError_t error, const std::string& context) {
        if (error != cudaSuccess) {
            std::ostringstream oss;
            oss << context << " - " << cudaGetErrorString(error);
            log(LOG_ERROR, oss.str());
        }
    }

private:
    const char* levelToString(LogLevel level) {
        switch(level) {
            case LOG_DEBUG: return "DEBUG";
            case LOG_INFO: return "INFO";
            case LOG_WARNING: return "WARNING";
            case LOG_ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};
```

### **6.8.2 Device-Side Printf Debugging**

```cpp
__global__ void debugKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Device-side printf (limited buffer size)
    if (idx == 0) {
        printf("Block %d, Thread %d: Starting kernel\n",
               blockIdx.x, threadIdx.x);
    }

    if (idx < size) {
        float oldVal = data[idx];
        data[idx] = idx * 2.0f;

        // Conditional debug output
        if (idx % 1000 == 0) {
            printf("idx=%d: %f -> %f\n", idx, oldVal, data[idx]);
        }
    }
}

// Host code to flush printf buffer
void flushPrintfBuffer() {
    CHECK_CUDA(cudaDeviceSynchronize());
}
```

---

## **6.9 Production Error Handling**

### **6.9.1 Production-Ready Error Handler**

```cpp
class CudaErrorHandler {
private:
    struct ErrorStats {
        std::atomic<int> totalErrors{0};
        std::atomic<int> recoveredErrors{0};
        std::unordered_map<cudaError_t, int> errorCounts;
        std::mutex mutex;
    } stats_;

    CudaLogger logger_;

public:
    CudaErrorHandler(const std::string& logFile)
        : logger_(logFile, LOG_WARNING) {}

    bool handleError(cudaError_t error, const std::string& context) {
        if (error == cudaSuccess) return true;

        stats_.totalErrors++;

        {
            std::lock_guard<std::mutex> lock(stats_.mutex);
            stats_.errorCounts[error]++;
        }

        logger_.logCudaError(error, context);

        // Attempt recovery based on error type
        bool recovered = attemptRecovery(error);

        if (recovered) {
            stats_.recoveredErrors++;
        }

        return recovered;
    }

    void printStatistics() {
        std::cout << "=== CUDA Error Statistics ===" << std::endl;
        std::cout << "Total errors: " << stats_.totalErrors << std::endl;
        std::cout << "Recovered: " << stats_.recoveredErrors << std::endl;

        std::lock_guard<std::mutex> lock(stats_.mutex);
        for (const auto& [error, count] : stats_.errorCounts) {
            std::cout << cudaGetErrorName(error) << ": " << count << std::endl;
        }
    }

private:
    bool attemptRecovery(cudaError_t error) {
        switch(error) {
            case cudaErrorMemoryAllocation:
                // Try to free unused memory
                cudaDeviceReset();
                return true;

            case cudaErrorLaunchTimeout:
                // Kernel took too long, might recover
                cudaDeviceReset();
                return true;

            case cudaErrorIllegalAddress:
            case cudaErrorIllegalInstruction:
                // Serious errors, cannot recover
                return false;

            default:
                return false;
        }
    }
};
```

### **6.9.2 Health Monitoring**

```cpp
class CudaHealthMonitor {
private:
    struct DeviceHealth {
        size_t freeMemory;
        size_t totalMemory;
        float temperature;
        int smClock;
        int memoryClock;
        bool isHealthy;
    };

public:
    DeviceHealth checkHealth(int device = 0) {
        DeviceHealth health;

        CHECK_CUDA(cudaSetDevice(device));

        // Check memory
        CHECK_CUDA(cudaMemGetInfo(&health.freeMemory, &health.totalMemory));

        // Check if we have enough free memory (e.g., 10% threshold)
        float memoryUsage = 1.0f - (float)health.freeMemory / health.totalMemory;

        health.isHealthy = (memoryUsage < 0.9f);

        if (!health.isHealthy) {
            fprintf(stderr, "Warning: Low GPU memory (%.1f%% used)\n",
                   memoryUsage * 100);
        }

        return health;
    }

    void continuous Monitoring(std::chrono::seconds interval) {
        while (true) {
            auto health = checkHealth();

            if (!health.isHealthy) {
                // Send alert, log, or take action
                handleUnhealthyState(health);
            }

            std::this_thread::sleep_for(interval);
        }
    }

private:
    void handleUnhealthyState(const DeviceHealth& health) {
        // Log the issue
        // Send notification
        // Attempt cleanup
        // Scale down workload
    }
};
```

---

## **6.10 Summary**

### **Key Takeaways**

1. **Always Check CUDA API Calls**: Use error checking macros consistently
2. **Handle Both Sync and Async Errors**: Understand the difference and check appropriately
3. **Use Debugging Tools**: Leverage compute-sanitizer and cuda-gdb
4. **Implement Recovery Strategies**: Plan for graceful degradation
5. **Monitor Production Systems**: Track errors and system health

### **Best Practices Checklist**

✅ Use `CHECK_CUDA` macro for all CUDA API calls
✅ Check kernel launch errors with `cudaGetLastError()`
✅ Synchronize and check after kernel execution
✅ Implement proper error logging
✅ Use RAII for CUDA resource management
✅ Test with compute-sanitizer regularly
✅ Have fallback CPU implementations for critical paths
✅ Monitor GPU health in production

### **Error Handling Flow**

```
1. API Call → CHECK_CUDA → Log Error → Attempt Recovery → Fallback/Exit
2. Kernel Launch → Check Launch → Sync → Check Runtime Errors
3. Async Operations → Query Status → Handle Timeout → Check Errors
```

---

## **Example Code**

The complete error handling example (`error_handling_demo.cu`) demonstrates:

1. Custom error checking macros
2. Exception-based error handling
3. Memory error detection
4. Race condition debugging
5. Production-ready error handler
6. GPU health monitoring

To build and run:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build build
./build/10.cuda_basic/16.Error\ Handling\ and\ Debugging/16_ErrorHandlingAndDebugging

# Test with sanitizer
compute-sanitizer ./build/.../16_ErrorHandlingAndDebugging
```

---

**Next**: Part 7 - Memory Hierarchy (Planned)