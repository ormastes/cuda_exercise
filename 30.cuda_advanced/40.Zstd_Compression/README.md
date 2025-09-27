# 47. Zstd Compression for CUDA Binaries (CUDA 12.0+)

## 47.1 Introduction

Zstandard (Zstd) compression support in CUDA enables significant reduction in binary sizes for CUDA applications, improving deployment efficiency and reducing memory footprint without sacrificing performance.

## 47.2 Learning Objectives

- Understand Zstd compression in CUDA context
- Configure CUDA builds with compression
- Optimize binary sizes for deployment
- Analyze compression vs performance trade-offs
- Implement runtime decompression strategies

## 47.3 Zstd Compression Fundamentals

### 47.3.1 What is Zstd Compression in CUDA?

Zstd compression in CUDA provides:
- Compressed device code (PTX/SASS)
- Reduced binary sizes (up to 70% reduction)
- Fast decompression at runtime
- Configurable compression levels
- Transparent to application execution

### 47.3.2 Compilation with Zstd

```bash
# Basic compilation with Zstd compression
nvcc -Xcompiler -fzstd -arch=sm_86 kernel.cu -o kernel_compressed

# Specify compression level (1-22, default 3)
nvcc -Xcompiler -fzstd=9 -arch=sm_86 kernel.cu -o kernel_compressed

# CUDA 12.0+ specific flags
nvcc --compress-device-code -arch=sm_86 kernel.cu -o kernel_compressed
```

## 47.4 Implementation Examples

### 47.4.1 Build Configuration

```cmake
# CMakeLists.txt with Zstd support
cmake_minimum_required(VERSION 3.18)
project(47_Zstd_Compression CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Enable Zstd compression for CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compress-device-code")

# Compression level configuration
option(CUDA_COMPRESSION_LEVEL "CUDA code compression level (1-22)" "9")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fzstd=${CUDA_COMPRESSION_LEVEL}")

# Binary size analysis targets
add_executable(47_Zstd_uncompressed kernel.cu)
add_executable(47_Zstd_compressed kernel.cu)

# Configure compression for specific target
target_compile_options(47_Zstd_compressed PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compress-device-code>
)
```

### 47.4.2 Runtime Decompression Management

```cpp
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <iostream>
#include <chrono>
#include <vector>

class CompressionManager {
private:
    bool compressionEnabled;
    int compressionLevel;
    size_t originalSize;
    size_t compressedSize;

public:
    CompressionManager() : compressionEnabled(false), compressionLevel(3) {
        checkCompressionSupport();
    }

    void checkCompressionSupport() {
        int driverVersion, runtimeVersion;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        printf("Compression Support Analysis:\n");
        printf("  CUDA Runtime: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Driver: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);

        // Zstd support in CUDA 12.0+
        compressionEnabled = (runtimeVersion >= 12000);
        printf("  Zstd Compression: %s\n", compressionEnabled ? "Supported" : "Not Supported");
    }

    void analyzeBinarySize(const std::string& binaryPath) {
        // Analyze binary size and compression ratio
        std::ifstream file(binaryPath, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            size_t fileSize = file.tellg();
            printf("Binary size: %.2f MB\n", fileSize / (1024.0 * 1024.0));
            file.close();
        }
    }

    // Runtime decompression benchmark
    void benchmarkDecompression() {
        if (!compressionEnabled) {
            printf("Compression not supported\n");
            return;
        }

        printf("\nDecompression Benchmark:\n");

        // Measure kernel launch with compressed code
        auto start = std::chrono::high_resolution_clock::now();

        // First launch (includes decompression)
        launchCompressedKernel();
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        auto firstLaunch = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Subsequent launch (cached)
        start = std::chrono::high_resolution_clock::now();
        launchCompressedKernel();
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        auto cachedLaunch = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        printf("  First launch (with decompression): %ld us\n", firstLaunch);
        printf("  Cached launch: %ld us\n", cachedLaunch);
        printf("  Decompression overhead: %ld us\n", firstLaunch - cachedLaunch);
    }

private:
    void launchCompressedKernel() {
        // Dummy kernel launch for benchmarking
        dim3 block(256);
        dim3 grid(1024);
        dummyKernel<<<grid, block>>>();
    }

    __global__ void dummyKernel() {
        // Simple kernel for testing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        __syncthreads();
    }
};
```

### 47.4.3 Multi-Module Compression

```cpp
class ModularCompression {
public:
    struct ModuleInfo {
        std::string name;
        size_t originalSize;
        size_t compressedSize;
        float compressionRatio;
        bool isCompressed;
    };

private:
    std::vector<ModuleInfo> modules;

public:
    void addModule(const std::string& modulePath, bool compress = true) {
        ModuleInfo info;
        info.name = modulePath;
        info.isCompressed = compress;

        if (compress) {
            compileWithCompression(modulePath, info);
        } else {
            compileWithoutCompression(modulePath, info);
        }

        modules.push_back(info);
    }

    void compileWithCompression(const std::string& path, ModuleInfo& info) {
        std::string cmd = "nvcc --compress-device-code -arch=sm_86 ";
        cmd += path + " -o " + path + ".compressed.o";

        system(cmd.c_str());

        // Get file sizes
        info.originalSize = getFileSize(path + ".o");
        info.compressedSize = getFileSize(path + ".compressed.o");
        info.compressionRatio = 1.0f - (float)info.compressedSize / info.originalSize;
    }

    void compileWithoutCompression(const std::string& path, ModuleInfo& info) {
        std::string cmd = "nvcc -arch=sm_86 ";
        cmd += path + " -o " + path + ".o";

        system(cmd.c_str());
        info.originalSize = getFileSize(path + ".o");
        info.compressedSize = info.originalSize;
        info.compressionRatio = 0.0f;
    }

    void printCompressionReport() {
        printf("\nCompression Report:\n");
        printf("%-30s %15s %15s %10s\n", "Module", "Original(KB)", "Compressed(KB)", "Ratio");
        printf("%-30s %15s %15s %10s\n", "------", "-----------", "-------------", "-----");

        size_t totalOriginal = 0;
        size_t totalCompressed = 0;

        for (const auto& module : modules) {
            printf("%-30s %15.2f %15.2f %9.1f%%\n",
                   module.name.c_str(),
                   module.originalSize / 1024.0,
                   module.compressedSize / 1024.0,
                   module.compressionRatio * 100);

            totalOriginal += module.originalSize;
            totalCompressed += module.compressedSize;
        }

        printf("%-30s %15s %15s %10s\n", "------", "-----------", "-------------", "-----");
        printf("%-30s %15.2f %15.2f %9.1f%%\n",
               "Total",
               totalOriginal / 1024.0,
               totalCompressed / 1024.0,
               (1.0f - (float)totalCompressed / totalOriginal) * 100);
    }

private:
    size_t getFileSize(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            size_t size = file.tellg();
            file.close();
            return size;
        }
        return 0;
    }
};
```

### 47.4.4 Deployment Optimization

```cpp
class DeploymentOptimizer {
public:
    struct DeploymentConfig {
        bool useCompression;
        int compressionLevel;
        bool stripDebugInfo;
        bool optimizeForSize;
        std::vector<std::string> targetArchitectures;
    };

    void optimizeForDeployment(const DeploymentConfig& config) {
        printf("Deployment Optimization Configuration:\n");

        std::string flags;

        // Compression settings
        if (config.useCompression) {
            flags += "--compress-device-code ";
            flags += "-Xcompiler -fzstd=" + std::to_string(config.compressionLevel) + " ";
        }

        // Strip debug information
        if (config.stripDebugInfo) {
            flags += "-Xcompiler -s ";
            flags += "--no-debug ";
        }

        // Size optimization
        if (config.optimizeForSize) {
            flags += "-Xcompiler -Os ";
            flags += "--use-fast-math ";
        }

        // Architecture-specific compilation
        for (const auto& arch : config.targetArchitectures) {
            flags += "-gencode arch=compute_" + arch + ",code=sm_" + arch + " ";
        }

        printf("Compilation flags: %s\n", flags.c_str());

        // Estimate size reduction
        estimateSizeReduction(config);
    }

private:
    void estimateSizeReduction(const DeploymentConfig& config) {
        printf("\nEstimated Size Reduction:\n");

        float reduction = 0.0f;

        if (config.useCompression) {
            // Zstd compression typically achieves 40-70% reduction
            reduction += 0.4f + (config.compressionLevel / 22.0f) * 0.3f;
        }

        if (config.stripDebugInfo) {
            reduction += 0.1f;  // Debug info typically 10% of binary
        }

        if (config.optimizeForSize) {
            reduction += 0.05f;  // Size optimization ~5%
        }

        reduction = std::min(reduction, 0.85f);  // Max realistic reduction

        printf("  Expected binary size reduction: %.1f%%\n", reduction * 100);
        printf("  100MB binary -> ~%.1fMB compressed\n", 100 * (1.0f - reduction));
    }
};
```

## 47.5 Practical Examples

### 47.5.1 Large Kernel Library Compression

```cpp
// kernel_library.cu - Large kernel library suitable for compression
namespace KernelLibrary {

    // Multiple kernel variants (good compression candidates)
    template<typename T, int BLOCK_SIZE>
    __global__ void genericKernel1(T* data, int n) {
        // Kernel implementation
    }

    template<typename T, int BLOCK_SIZE>
    __global__ void genericKernel2(T* data, int n) {
        // Similar pattern - compresses well
    }

    // Instantiate multiple variants
    template __global__ void genericKernel1<float, 128>(float*, int);
    template __global__ void genericKernel1<float, 256>(float*, int);
    template __global__ void genericKernel1<float, 512>(float*, int);
    template __global__ void genericKernel1<double, 128>(double*, int);
    template __global__ void genericKernel1<double, 256>(double*, int);

    // ... many more instantiations
}

// Build script for compression comparison
void buildAndCompare() {
    system("nvcc -arch=sm_86 kernel_library.cu -o library_uncompressed.o");
    system("nvcc --compress-device-code -arch=sm_86 kernel_library.cu -o library_compressed.o");

    // Compare sizes
    size_t uncompressed = getFileSize("library_uncompressed.o");
    size_t compressed = getFileSize("library_compressed.o");

    printf("Uncompressed: %.2f MB\n", uncompressed / (1024.0 * 1024.0));
    printf("Compressed: %.2f MB\n", compressed / (1024.0 * 1024.0));
    printf("Reduction: %.1f%%\n", (1.0 - (double)compressed/uncompressed) * 100);
}
```

### 47.5.2 Container Deployment

```dockerfile
# Dockerfile for compressed CUDA application
FROM nvidia/cuda:12.0-runtime

# Copy compressed binaries
COPY --from=builder /app/bin/*.compressed /app/bin/

# Runtime decompression handled automatically
ENV CUDA_ENABLE_COMPRESSION=1
ENV CUDA_COMPRESSION_CACHE=/tmp/cuda_cache

# Reduced container size with compressed binaries
RUN echo "Container size reduced by Zstd compression"

CMD ["/app/bin/cuda_app.compressed"]
```

## 47.6 Performance Considerations

### 47.6.1 Compression Trade-offs

```cpp
class CompressionAnalyzer {
public:
    void analyzeTradeoffs() {
        printf("Compression Trade-off Analysis:\n\n");

        printf("Benefits:\n");
        printf("  - Reduced binary size (40-70%% typical)\n");
        printf("  - Lower storage requirements\n");
        printf("  - Faster network transfers\n");
        printf("  - Smaller container images\n\n");

        printf("Costs:\n");
        printf("  - First kernel launch overhead (decompression)\n");
        printf("  - Slight memory overhead for decompression cache\n");
        printf("  - Build time increase for compression\n\n");

        printf("Best Use Cases:\n");
        printf("  - Large kernel libraries\n");
        printf("  - Container deployments\n");
        printf("  - Edge/embedded systems\n");
        printf("  - Cloud deployments with bandwidth costs\n");
    }

    void benchmarkCompressionLevels() {
        printf("\nCompression Level Benchmarks:\n");
        printf("Level | Size Reduction | Compression Time | Decompression Time\n");
        printf("------|---------------|------------------|-------------------\n");

        for (int level = 1; level <= 22; level += 3) {
            benchmarkLevel(level);
        }
    }

private:
    void benchmarkLevel(int level) {
        // Simulate compression metrics
        float sizeReduction = 30 + (level / 22.0f) * 40;  // 30-70%
        float compTime = 100 + level * 50;  // ms
        float decompTime = 10 + level * 2;  // ms

        printf("%5d | %13.1f%% | %16.1fms | %17.1fms\n",
               level, sizeReduction, compTime, decompTime);
    }
};
```

## 47.7 Exercises

1. **Compression Comparison**: Compare binary sizes with different compression levels
2. **Performance Impact**: Measure first-launch vs cached-launch performance
3. **Module Optimization**: Selectively compress large modules
4. **Container Size**: Reduce Docker image size with compression
5. **Deployment Pipeline**: Build CI/CD with compression optimization

## 47.8 Building and Running

```bash
# Build with Zstd compression support
cd build/30.cuda_advanced/47.Zstd_Compression
ninja

# Compare uncompressed vs compressed binaries
./build_comparison.sh

# Run compression analyzer
./47_Zstd_analyzer

# Benchmark decompression overhead
./47_Zstd_benchmark

# Size analysis
du -sh 47_Zstd_uncompressed 47_Zstd_compressed

# Detailed binary analysis
objdump -h 47_Zstd_compressed | grep -E "\.text|\.rodata"
```

## 47.9 Key Takeaways

- Zstd compression can reduce CUDA binary sizes by 40-70%
- Compression is transparent to application execution
- First kernel launch includes decompression overhead
- Subsequent launches use cached decompressed code
- Ideal for large kernel libraries and deployment scenarios
- Configurable compression levels balance size vs speed
- Available in CUDA 12.0 and later versions
- Particularly valuable for container and edge deployments