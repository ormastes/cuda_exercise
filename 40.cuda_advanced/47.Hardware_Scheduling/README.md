# 43. Hardware Scheduling and Workload Management

## 43.1 Introduction to Hardware Scheduling

Hardware scheduling optimization involves understanding GPU scheduler behavior and designing workloads for optimal resource utilization.

## 43.2 Learning Objectives

- Understand GPU hardware scheduling mechanisms
- Optimize workload distribution across SMs
- Implement load balancing strategies
- Design scheduler-aware algorithms
- Analyze and optimize occupancy patterns

## 43.3 GPU Scheduling Fundamentals

### 43.3.1 SM and Warp Scheduling

```cuda
class SchedulingAnalyzer {
private:
    cudaDeviceProp deviceProps;
    int deviceId;

public:
    SchedulingAnalyzer(int device = 0) : deviceId(device) {
        cudaGetDeviceProperties(&deviceProps, device);
        printSchedulingInfo();
    }

    void printSchedulingInfo() {
        printf("GPU Scheduling Information:\n");
        printf("  Device: %s\n", deviceProps.name);
        printf("  Streaming Multiprocessors: %d\n", deviceProps.multiProcessorCount);
        printf("  Max threads per SM: %d\n", deviceProps.maxThreadsPerMultiProcessor);
        printf("  Max blocks per SM: %d\n", deviceProps.maxBlocksPerMultiProcessor);
        printf("  Warp size: %d\n", deviceProps.warpSize);
        printf("  Max warps per SM: %d\n", deviceProps.maxThreadsPerMultiProcessor / deviceProps.warpSize);
        printf("  Max shared memory per SM: %zu bytes\n", deviceProps.sharedMemPerMultiprocessor);
        printf("  Max registers per SM: %d\n", deviceProps.regsPerMultiprocessor);
    }

    void analyzeOccupancy(int blockSize, int sharedMemPerBlock, int regsPerThread) {
        printf("\nOccupancy Analysis:\n");
        printf("  Block size: %d threads\n", blockSize);
        printf("  Shared memory per block: %d bytes\n", sharedMemPerBlock);
        printf("  Registers per thread: %d\n", regsPerThread);

        // Calculate theoretical occupancy
        int maxBlocksByThreads = deviceProps.maxThreadsPerMultiProcessor / blockSize;
        int maxBlocksByBlocks = deviceProps.maxBlocksPerMultiProcessor;
        int maxBlocksBySharedMem = (sharedMemPerBlock > 0) ?
            deviceProps.sharedMemPerMultiprocessor / sharedMemPerBlock : INT_MAX;
        int maxBlocksByRegs = (regsPerThread > 0) ?
            deviceProps.regsPerMultiprocessor / (blockSize * regsPerThread) : INT_MAX;

        int actualBlocks = std::min({maxBlocksByThreads, maxBlocksByBlocks,
                                   maxBlocksBySharedMem, maxBlocksByRegs});

        float occupancy = (float)(actualBlocks * blockSize) / deviceProps.maxThreadsPerMultiProcessor;

        printf("  Limiting factors:\n");
        printf("    By threads: %d blocks\n", maxBlocksByThreads);
        printf("    By blocks: %d blocks\n", maxBlocksByBlocks);
        printf("    By shared memory: %d blocks\n", maxBlocksBySharedMem);
        printf("    By registers: %d blocks\n", maxBlocksByRegs);
        printf("  Actual blocks per SM: %d\n", actualBlocks);
        printf("  Theoretical occupancy: %.1f%%\n", occupancy * 100);

        // Use CUDA occupancy calculator
        int minGridSize, optimalBlockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize,
                                          measureOccupancyKernel, 0, 0);
        printf("  CUDA suggested block size: %d\n", optimalBlockSize);
    }
};

// Example kernels
__global__ void measureOccupancyKernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data) {
        data[idx] = sqrtf(static_cast<float>(idx));
    }
}
```

### 43.3.2 Load Balancing Strategies

```cuda
class LoadBalancingManager {
private:
    struct WorkItem {
        int startIdx;
        int endIdx;
        float complexity;
    };

    std::vector<WorkItem> workItems;
    cudaDeviceProp deviceProps;

public:
    LoadBalancingManager() {
        cudaGetDeviceProperties(&deviceProps, 0);
    }

    void executeDynamicScheduling(float* data, int totalWork) {
        printf("\nExecuting dynamic load balancing...\n");

        // Create work queue
        int* workQueue;
        int* workIndex;
        cudaMalloc(&workQueue, totalWork * sizeof(int));
        cudaMalloc(&workIndex, sizeof(int));

        // Initialize work queue
        std::vector<int> hostQueue(totalWork);
        std::iota(hostQueue.begin(), hostQueue.end(), 0);
        cudaMemcpy(workQueue, hostQueue.data(), totalWork * sizeof(int), cudaMemcpyHostToDevice);

        int initialIndex = 0;
        cudaMemcpy(workIndex, &initialIndex, sizeof(int), cudaMemcpyHostToDevice);

        // Launch dynamic scheduling kernel
        int numBlocks = deviceProps.multiProcessorCount * 4;  // Oversubscribe
        dim3 block(256);
        dim3 grid(numBlocks);

        dynamicWorkKernel<<<grid, block>>>(data, workQueue, workIndex, totalWork);

        cudaFree(workQueue);
        cudaFree(workIndex);
    }
};

// Dynamic work kernel
__global__ void dynamicWorkKernel(float* data, int* workQueue, int* workIndex, int totalWork) {
    while (true) {
        int workItem = atomicAdd(workIndex, 1);
        if (workItem >= totalWork) break;

        int dataIdx = workQueue[workItem];

        // Process work item
        float result = data[dataIdx];
        for (int i = 0; i < 10; i++) {
            result = sqrtf(result * result + 1.0f);
        }
        data[dataIdx] = result;
    }
}
```

## 43.4 Exercises

1. **Occupancy Analysis**: Analyze occupancy for different kernel configurations
2. **Load Balancing**: Implement static vs dynamic load balancing
3. **Persistent Kernels**: Design persistent kernel patterns
4. **Scheduling Optimization**: Optimize workload distribution
5. **Performance Study**: Compare different scheduling strategies

## 43.5 Building and Running

```bash
# Build with hardware scheduling examples
cd build/30.cuda_advanced/43.Hardware_Scheduling
ninja

# Run examples
./43_HardwareScheduling_analysis
./43_HardwareScheduling_loadbalancing
./43_HardwareScheduling_persistent

# Profile scheduling efficiency
ncu --metrics achieved_occupancy,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./43_HardwareScheduling_loadbalancing

# Analyze scheduling patterns
nsys profile --stats=true -t cuda ./43_HardwareScheduling_analysis
```

## 43.6 Key Takeaways

- Understanding GPU scheduling improves performance optimization
- Occupancy analysis guides kernel configuration choices
- Load balancing strategies affect overall system efficiency
- Persistent kernels can reduce scheduling overhead
- Hardware-aware design leads to better resource utilization
- Profiling tools provide insights into scheduling behavior
