# 🌀 Part 25: Dynamic Parallelism

**Goal**: Master dynamic parallelism to launch kernels from within kernels, enabling recursive algorithms and adaptive workloads.

---

## **25.1 Overview**

Dynamic Parallelism enables CUDA kernels to launch other kernels without CPU intervention. This powerful feature enables:
- **Recursive algorithms** (quicksort, tree traversal)
- **Adaptive mesh refinement**
- **Dynamic work generation**
- **Nested parallelism**

### **Requirements and Limitations**

| Feature | Requirement |
|---------|-------------|
| **Compute Capability** | 3.5 or higher |
| **Compilation** | `-rdc=true` (relocatable device code) |
| **Architecture** | `-arch=sm_35` or higher |
| **Memory** | Parent and child kernels share global memory |
| **Synchronization** | `cudaDeviceSynchronize()` available in device code |

---

## **25.2 Dynamic Parallelism Fundamentals**

### **25.2.1 Basic Kernel Launch from Device**

```cuda
__global__ void parentKernel(int* data, int n) {
    // Launch child kernel from device
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dim3 childGrid(32);
        dim3 childBlock(256);

        childKernel<<<childGrid, childBlock>>>(data, n);

        // Synchronize with child kernel
        cudaDeviceSynchronize();
    }
}

__global__ void childKernel(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] *= 2;
    }
}
```

### **25.2.2 Device Runtime API**

```cuda
__device__ void deviceLaunchExample() {
    // Device-side CUDA runtime API
    cudaError_t err;

    // Launch kernel
    kernel<<<grid, block>>>(args);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize
    err = cudaDeviceSynchronize();

    // Memory operations
    int* d_ptr;
    cudaMalloc(&d_ptr, size);
    cudaMemset(d_ptr, 0, size);
    cudaFree(d_ptr);

    // Stream operations
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    kernel<<<grid, block, 0, stream>>>(args);
    cudaStreamDestroy(stream);
}
```

### **25.2.3 Compilation Requirements**

```bash
# Compile with relocatable device code
nvcc -rdc=true -arch=sm_70 dynamic.cu -o dynamic

# For separate compilation
nvcc -dc -arch=sm_70 file1.cu -o file1.o
nvcc -dc -arch=sm_70 file2.cu -o file2.o
nvcc -dlink -arch=sm_70 file1.o file2.o -o device_link.o
nvcc -arch=sm_70 file1.o file2.o device_link.o -o program
```

---

## **25.3 Recursive Algorithms**

### **25.3.1 Parallel Quicksort**

```cuda
__global__ void quicksort(int* data, int left, int right, int depth) {
    if (left >= right) return;

    // Partition around pivot
    int pivotIndex = left + (right - left) / 2;
    int pivot = data[pivotIndex];

    // Simple partition (single thread for simplicity)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = left, j = right;

        while (i <= j) {
            while (data[i] < pivot) i++;
            while (data[j] > pivot) j--;

            if (i <= j) {
                int temp = data[i];
                data[i] = data[j];
                data[j] = temp;
                i++;
                j--;
            }
        }

        // Launch child kernels for sub-arrays
        if (depth < MAX_DEPTH) {
            cudaStream_t s1, s2;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

            if (left < j) {
                quicksort<<<1, 1, 0, s1>>>(data, left, j, depth + 1);
            }
            if (i < right) {
                quicksort<<<1, 1, 0, s2>>>(data, i, right, depth + 1);
            }

            cudaStreamSynchronize(s1);
            cudaStreamSynchronize(s2);
            cudaStreamDestroy(s1);
            cudaStreamDestroy(s2);
        } else {
            // Fall back to sequential sort for deep recursion
            // Implement insertion sort or similar
        }
    }
}
```

### **25.3.2 Binary Tree Traversal**

```cuda
struct Node {
    int value;
    int left;   // Index of left child (-1 if none)
    int right;  // Index of right child (-1 if none)
};

__global__ void treeTraversal(Node* tree, int* result, int nodeIdx, int* counter) {
    if (nodeIdx == -1) return;

    Node node = tree[nodeIdx];

    // Process current node
    int myIndex = atomicAdd(counter, 1);
    result[myIndex] = node.value;

    // Launch kernels for children
    if (node.left != -1 || node.right != -1) {
        cudaStream_t leftStream, rightStream;
        cudaStreamCreateWithFlags(&leftStream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&rightStream, cudaStreamNonBlocking);

        if (node.left != -1) {
            treeTraversal<<<1, 1, 0, leftStream>>>(tree, result, node.left, counter);
        }

        if (node.right != -1) {
            treeTraversal<<<1, 1, 0, rightStream>>>(tree, result, node.right, counter);
        }

        cudaStreamSynchronize(leftStream);
        cudaStreamSynchronize(rightStream);
        cudaStreamDestroy(leftStream);
        cudaStreamDestroy(rightStream);
    }
}
```

---

## **25.4 Adaptive Algorithms**

### **25.4.1 Adaptive Mesh Refinement**

```cuda
struct Cell {
    float value;
    float error;
    int level;
    int x, y;
};

__global__ void adaptiveMeshRefinement(Cell* cells, int* numCells, int maxLevel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *numCells) {
        Cell cell = cells[tid];

        // Check if refinement is needed
        if (cell.error > THRESHOLD && cell.level < maxLevel) {
            // Create 4 child cells (2D refinement)
            int baseIdx = atomicAdd(numCells, 4);

            // Initialize child cells
            for (int i = 0; i < 4; i++) {
                Cell* child = &cells[baseIdx + i];
                child->level = cell.level + 1;
                child->x = cell.x * 2 + (i % 2);
                child->y = cell.y * 2 + (i / 2);
                child->value = cell.value;  // Initial value
            }

            // Launch kernel to process new cells
            if (threadIdx.x == 0) {
                dim3 grid((4 + 255) / 256);
                dim3 block(256);
                processRefinedCells<<<grid, block>>>(cells + baseIdx, 4);
                cudaDeviceSynchronize();
            }
        }
    }
}

__global__ void processRefinedCells(Cell* cells, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        // Compute refined values
        cells[tid].value = computeRefinedValue(cells[tid]);
        cells[tid].error = computeError(cells[tid]);
    }
}
```

### **25.4.2 Dynamic Work Queue**

```cuda
struct WorkItem {
    int type;
    int data[4];
};

__device__ WorkItem* workQueue;
__device__ int queueHead;
__device__ int queueTail;
__device__ int queueSize;

__global__ void dynamicWorkProcessor() {
    while (true) {
        // Get work item
        int myWork = atomicAdd(&queueHead, 1);
        if (myWork >= queueSize) break;

        WorkItem item = workQueue[myWork % MAX_QUEUE_SIZE];

        // Process work item
        switch (item.type) {
            case WORK_TYPE_A:
                processTypeA<<<1, 32>>>(item.data);
                break;
            case WORK_TYPE_B:
                processTypeB<<<2, 64>>>(item.data);
                break;
            case WORK_TYPE_RECURSIVE:
                // Generate more work
                int newItems = generateWork(item.data);
                if (newItems > 0) {
                    int insertPos = atomicAdd(&queueTail, newItems);
                    // Add new work items to queue
                }
                break;
        }

        // Synchronize if needed
        if (item.type == WORK_TYPE_BARRIER) {
            cudaDeviceSynchronize();
        }
    }
}
```

---

## **25.5 Nested Parallelism Patterns**

### **25.5.1 Fork-Join Pattern**

```cuda
__global__ void forkJoinPattern(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = n / (gridDim.x * blockDim.x);
    int start = tid * chunk_size;
    int end = min(start + chunk_size, n);

    // Fork: Launch child kernel for complex computation
    if (end > start) {
        dim3 childGrid((end - start + 255) / 256);
        dim3 childBlock(256);

        complexComputation<<<childGrid, childBlock>>>(data + start, end - start);

        // Join: Wait for child to complete
        cudaDeviceSynchronize();

        // Continue with parent computation
        postProcess(data + start, end - start);
    }
}
```

### **25.5.2 Pipeline Pattern**

```cuda
__global__ void pipelineStage1(float* input, float* intermediate, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // Process data
        intermediate[tid] = processStage1(input[tid]);

        // Launch next stage for this chunk
        if (threadIdx.x == 0) {
            int chunkStart = blockIdx.x * blockDim.x;
            int chunkSize = min(blockDim.x, n - chunkStart);

            pipelineStage2<<<1, chunkSize>>>(
                intermediate + chunkStart,
                output + chunkStart,
                chunkSize
            );
        }
    }
}

__global__ void pipelineStage2(float* intermediate, float* output, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        output[tid] = processStage2(intermediate[tid]);
    }
}
```

---

## **25.6 Performance Considerations**

### **25.6.1 Overhead Management**

```cuda
__global__ void efficientDynamicLaunch(int* data, int n) {
    // Avoid excessive kernel launches
    const int MIN_WORK_SIZE = 1024;

    int work_size = n / gridDim.x;
    int my_start = blockIdx.x * work_size;
    int my_end = (blockIdx.x == gridDim.x - 1) ? n : my_start + work_size;

    if (my_end - my_start > MIN_WORK_SIZE) {
        // Launch child kernel only for substantial work
        dim3 childGrid((my_end - my_start + 255) / 256);
        dim3 childBlock(256);

        childKernel<<<childGrid, childBlock>>>(data + my_start, my_end - my_start);
        cudaDeviceSynchronize();
    } else {
        // Process directly in parent
        for (int i = my_start + threadIdx.x; i < my_end; i += blockDim.x) {
            data[i] = processData(data[i]);
        }
    }
}
```

### **25.6.2 Memory Management**

```cuda
__device__ void* deviceMemoryPool;
__device__ int poolOffset;

__device__ void* deviceMalloc(size_t size) {
    // Simple memory pool allocation
    int offset = atomicAdd(&poolOffset, size);
    if (offset + size > POOL_SIZE) {
        return NULL;  // Out of memory
    }
    return (char*)deviceMemoryPool + offset;
}

__global__ void memoryEfficientKernel() {
    // Allocate from pool instead of cudaMalloc
    float* temp = (float*)deviceMalloc(256 * sizeof(float));

    if (temp != NULL) {
        // Use temporary memory
        processWithTemp(temp);

        // No need to free - pool is reset between kernel launches
    }
}
```

---

## **25.7 Example Programs**

### **25.7.1 Recursive Matrix Multiplication**

File: `recursive_matmul.cu`
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define THRESHOLD 64  // Switch to regular multiplication below this size

__global__ void matmulBase(float* C, float* A, float* B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void strassenMultiply(float* C, float* A, float* B, int n, int depth) {
    if (n <= THRESHOLD || depth > 5) {
        // Base case: use regular multiplication
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        dim3 block(16, 16);
        matmulBase<<<grid, block>>>(C, A, B, n);
        cudaDeviceSynchronize();
        return;
    }

    int half = n / 2;
    size_t subSize = half * half * sizeof(float);

    // Allocate temporary matrices
    float *M1, *M2, *M3, *M4, *M5, *M6, *M7;
    float *temp1, *temp2;

    cudaMalloc(&M1, subSize);
    cudaMalloc(&M2, subSize);
    cudaMalloc(&M3, subSize);
    cudaMalloc(&M4, subSize);
    cudaMalloc(&M5, subSize);
    cudaMalloc(&M6, subSize);
    cudaMalloc(&M7, subSize);
    cudaMalloc(&temp1, subSize);
    cudaMalloc(&temp2, subSize);

    // Create streams for parallel execution
    cudaStream_t streams[7];
    for (int i = 0; i < 7; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Compute M1 = (A11 + A22) * (B11 + B22)
    // ... (Strassen sub-products)

    // Launch recursive calls
    strassenMultiply<<<1, 1, 0, streams[0]>>>(M1, temp1, temp2, half, depth + 1);
    // ... (other recursive calls)

    // Synchronize all streams
    for (int i = 0; i < 7; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Combine results into C
    // ... (combination logic)

    // Free temporary matrices
    cudaFree(M1); cudaFree(M2); cudaFree(M3); cudaFree(M4);
    cudaFree(M5); cudaFree(M6); cudaFree(M7);
    cudaFree(temp1); cudaFree(temp2);
}

int main() {
    const int N = 512;
    size_t size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate and initialize host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i + 1) % 100) / 100.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch recursive multiplication
    strassenMultiply<<<1, 1>>>(d_C, d_A, d_B, N, 0);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Matrix multiplication completed\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

### **25.7.2 Adaptive Integration**

File: `adaptive_integration.cu`
```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float f(float x) {
    return sinf(x) * expf(-x * x);  // Example function
}

__device__ float simpson(float a, float b) {
    float h = (b - a) / 2.0f;
    return h / 3.0f * (f(a) + 4.0f * f(a + h) + f(b));
}

__global__ void adaptiveIntegrate(float a, float b, float epsilon,
                                  float* result, int depth) {
    if (depth > 10) {
        // Maximum recursion depth
        *result = simpson(a, b);
        return;
    }

    float c = (a + b) / 2.0f;
    float whole = simpson(a, b);
    float left = simpson(a, c);
    float right = simpson(c, b);

    if (fabsf(whole - (left + right)) < epsilon) {
        // Sufficient accuracy
        *result = left + right;
    } else {
        // Need more refinement
        float *leftResult, *rightResult;
        cudaMalloc(&leftResult, sizeof(float));
        cudaMalloc(&rightResult, sizeof(float));

        // Create streams for parallel integration
        cudaStream_t leftStream, rightStream;
        cudaStreamCreate(&leftStream);
        cudaStreamCreate(&rightStream);

        // Launch child kernels
        adaptiveIntegrate<<<1, 1, 0, leftStream>>>(
            a, c, epsilon / 2.0f, leftResult, depth + 1
        );
        adaptiveIntegrate<<<1, 1, 0, rightStream>>>(
            c, b, epsilon / 2.0f, rightResult, depth + 1
        );

        // Wait for completion
        cudaStreamSynchronize(leftStream);
        cudaStreamSynchronize(rightStream);

        // Combine results
        *result = *leftResult + *rightResult;

        // Cleanup
        cudaStreamDestroy(leftStream);
        cudaStreamDestroy(rightStream);
        cudaFree(leftResult);
        cudaFree(rightResult);
    }
}

int main() {
    float a = 0.0f, b = 10.0f;
    float epsilon = 1e-6f;

    float *d_result, h_result;
    cudaMalloc(&d_result, sizeof(float));

    printf("Adaptive integration from %.2f to %.2f\n", a, b);
    printf("Tolerance: %e\n", epsilon);

    // Launch adaptive integration
    adaptiveIntegrate<<<1, 1>>>(a, b, epsilon, d_result, 0);
    cudaDeviceSynchronize();

    // Get result
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Integral = %f\n", h_result);

    cudaFree(d_result);
    return 0;
}
```

---

## **25.8 Building and Running**

### **CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.18)
project(25_Dynamic_Parallelism CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# Enable relocatable device code for dynamic parallelism
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# Quicksort example
add_executable(quicksort quicksort.cu)
set_target_properties(quicksort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Tree traversal example
add_executable(tree_traversal tree_traversal.cu)
set_target_properties(tree_traversal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Adaptive mesh refinement
add_executable(adaptive_mesh adaptive_mesh.cu)
set_target_properties(adaptive_mesh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Recursive matrix multiplication
add_executable(recursive_matmul recursive_matmul.cu)
set_target_properties(recursive_matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Adaptive integration
add_executable(adaptive_integration adaptive_integration.cu)
set_target_properties(adaptive_integration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link with cudadevrt for device runtime
foreach(target quicksort tree_traversal adaptive_mesh recursive_matmul adaptive_integration)
    target_link_libraries(${target} cudadevrt)
endforeach()
```

### **Build and Run**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
ninja

# Run examples
./quicksort
./tree_traversal
./adaptive_mesh
./recursive_matmul
./adaptive_integration

# Profile dynamic parallelism
nsys profile --stats=true ./recursive_matmul
ncu --target-processes all ./quicksort
```

---

## **25.9 Best Practices**

### **25.9.1 Design Guidelines**

1. **Minimize kernel launch overhead** - Launch only for substantial work
2. **Use streams** for concurrent child kernels
3. **Limit recursion depth** to avoid stack overflow
4. **Pool memory allocations** to reduce allocation overhead
5. **Profile thoroughly** - Dynamic parallelism can have unexpected overhead

### **25.9.2 When to Use Dynamic Parallelism**

✅ **Good Use Cases:**
- Irregular, data-dependent parallelism
- Recursive algorithms (with limited depth)
- Adaptive algorithms
- Simplifying complex kernel orchestration

❌ **Avoid When:**
- Work is regular and predictable
- Overhead exceeds benefits
- Deep recursion is required
- Simple alternatives exist

### **25.9.3 Performance Optimization**

```cuda
__global__ void optimizedDynamic() {
    // 1. Batch work to reduce launches
    const int MIN_BATCH = 1000;

    // 2. Use shared memory for coordination
    __shared__ int workCount;

    // 3. Coalesce child kernel launches
    if (threadIdx.x == 0) {
        if (workCount > MIN_BATCH) {
            childKernel<<<grid, block>>>();
        }
    }

    // 4. Use tail recursion where possible
    // 5. Consider iterative alternatives
}
```

---

## **25.10 Debugging and Profiling**

### **25.10.1 Debugging Tips**

```cuda
__global__ void debugDynamic() {
    // Print kernel hierarchy
    printf("Kernel: Block %d, Thread %d, Depth %d\n",
           blockIdx.x, threadIdx.x, depth);

    // Check for launch errors
    childKernel<<<grid, block>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch failed: %s\n", cudaGetErrorString(err));
    }

    // Track recursion depth
    if (depth > MAX_DEPTH) {
        printf("Maximum depth exceeded\n");
        return;
    }
}
```

### **25.10.2 Profiling Commands**

```bash
# Profile kernel launches
ncu --print-kernel-trace ./program

# Analyze parent-child relationships
ncu --metrics launch.func_overhead ./program

# Memory usage in dynamic parallelism
ncu --metrics dram__bytes.sum ./program
```

---

## **25.11 Common Issues and Solutions**

| Issue | Solution |
|-------|----------|
| **Stack overflow** | Limit recursion depth, increase stack size |
| **Out of memory** | Use memory pools, free intermediate results |
| **Poor performance** | Batch work, reduce launch overhead |
| **Launch failures** | Check compute capability, compilation flags |
| **Deadlock** | Avoid circular dependencies, use streams |

---

## **25.12 Exercises**

1. **Implement parallel merge sort** using dynamic parallelism
2. **Create adaptive quadrature** integration algorithm
3. **Build dynamic task scheduler** with work stealing
4. **Implement parallel graph algorithms** (BFS/DFS)
5. **Create fractal generator** with adaptive refinement

---

## **25.13 References**

- [CUDA Dynamic Parallelism Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dynamic-parallelism)
- [Dynamic Parallelism in CUDA](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/)
- [Optimizing Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#dynamic-parallelism)
- [CUDA Runtime API (Device)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)

---

**Previous**: [Part 24: Memory Coalescing and Bank Conflicts](../24.Memory_Coalescing_and_Bank_Conflicts/README.md)
**Next**: [Part 26: CUDA Libraries](../26.CUDA_Libraries/README.md)