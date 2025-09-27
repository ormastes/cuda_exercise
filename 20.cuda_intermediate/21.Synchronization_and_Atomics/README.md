# 🔒 Part 21: Synchronization and Atomics

**Goal**: Master advanced thread synchronization, atomic operations, and lock-free algorithms in CUDA.

---

## **21.1 Overview**

Synchronization and atomic operations are essential for coordinating parallel threads and maintaining data consistency in CUDA applications. This section covers:

- Thread synchronization primitives
- Atomic operations for all data types
- Memory fences and consistency
- Lock-free data structures
- Cooperative groups API
- Performance implications and optimization

---

## **21.2 Synchronization Primitives**

### **Thread Synchronization Levels**

| Level | Scope | Function | Use Case |
|-------|-------|----------|----------|
| **Thread Block** | Within block | `__syncthreads()` | Shared memory operations |
| **Warp** | 32 threads | Implicit SIMT | Warp-level primitives |
| **Grid** | All blocks | Cooperative groups | Global synchronization |
| **Device** | All kernels | `cudaDeviceSynchronize()` | Host-device sync |

### **__syncthreads()**

```cpp
__global__ void kernel() {
    __shared__ float sdata[256];

    // Load data
    sdata[threadIdx.x] = input[globalIdx];

    __syncthreads(); // Wait for all threads in block

    // Process shared data
    if (threadIdx.x < 128) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    }
}
```

**Important Rules:**
- Must be called by ALL threads in block
- Cannot be in conditional code unless condition is uniform
- Not needed for warp-level operations (implicit sync)

---

## **21.3 Atomic Operations**

### **Supported Atomic Functions**

| Operation | Integer | Float | Double | Description |
|-----------|---------|-------|--------|-------------|
| `atomicAdd` | ✓ | ✓ | ✓* | Atomic addition |
| `atomicSub` | ✓ | ✗ | ✗ | Atomic subtraction |
| `atomicMin` | ✓ | ✗ | ✗ | Atomic minimum |
| `atomicMax` | ✓ | ✗ | ✗ | Atomic maximum |
| `atomicInc` | ✓ | ✗ | ✗ | Atomic increment with wrap |
| `atomicDec` | ✓ | ✗ | ✗ | Atomic decrement with wrap |
| `atomicAnd` | ✓ | ✗ | ✗ | Atomic bitwise AND |
| `atomicOr` | ✓ | ✗ | ✗ | Atomic bitwise OR |
| `atomicXor` | ✓ | ✗ | ✗ | Atomic bitwise XOR |
| `atomicExch` | ✓ | ✓ | ✓* | Atomic exchange |
| `atomicCAS` | ✓ | ✓** | ✓** | Compare and swap |

*Requires compute capability 6.0+
**Using reinterpret casting

### **Atomic Operation Examples**

```cpp
// Basic atomic add
__global__ void histogram_kernel(int* data, int* bins, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(&bins[data[tid]], 1);
    }
}

// Compare-and-swap (CAS)
__device__ int atomicMaxInt(int* address, int val) {
    int old = *address, assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, max(val, assumed));
    } while (old != assumed);
    return old;
}

// Custom atomic for floats
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
        if (old == assumed) break;
    }

    return __int_as_float(old);
}
```

---

## **21.4 Memory Fences**

### **Fence Types**

| Fence | Scope | Description |
|-------|-------|-------------|
| `__threadfence_block()` | Block | Orders memory operations within block |
| `__threadfence()` | Device | Orders memory operations device-wide |
| `__threadfence_system()` | System | Orders operations for CPU/GPU system |

### **Usage Example**

```cpp
__device__ volatile int flag = 0;
__device__ int data = 0;

__global__ void producer() {
    data = compute_value();
    __threadfence();  // Ensure data write completes
    atomicExch(&flag, 1);  // Signal completion
}

__global__ void consumer() {
    while (atomicAdd(&flag, 0) == 0);  // Wait for signal
    __threadfence();  // Ensure we see latest data
    int value = data;  // Read produced data
}
```

---

## **21.5 Warp-Level Primitives**

### **Warp Shuffle Functions**

```cpp
// Warp reduction using shuffle
__device__ float warp_reduce(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Broadcast within warp
__device__ float warp_broadcast(float val, int srcLane) {
    return __shfl_sync(0xffffffff, val, srcLane);
}
```

### **Warp Vote Functions**

```cpp
__global__ void warp_vote_kernel() {
    int tid = threadIdx.x;
    bool predicate = (tid % 2 == 0);

    unsigned mask = __ballot_sync(0xffffffff, predicate);

    if (__all_sync(0xffffffff, tid < 16)) {
        // All threads in warp have tid < 16
    }

    if (__any_sync(0xffffffff, tid == 0)) {
        // At least one thread has tid == 0
    }
}
```

---

## **21.6 Cooperative Groups**

### **API Overview**

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void coop_kernel() {
    // Get thread block
    cg::thread_block block = cg::this_thread_block();

    // Create tile (sub-group)
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    // Synchronize at different levels
    block.sync();  // Block-level
    tile.sync();   // Tile-level

    // Collective operations
    int sum = cg::reduce(tile, value, cg::plus<int>());
}
```

### **Grid-Level Synchronization**

```cpp
__global__ void grid_sync_kernel() {
    cg::grid_group grid = cg::this_grid();

    // Phase 1
    process_phase1();

    grid.sync();  // All blocks synchronize

    // Phase 2 - all blocks see Phase 1 results
    process_phase2();
}

// Special launch required
void* kernelArgs[] = { /* args */ };
cudaLaunchCooperativeKernel(
    (void*)grid_sync_kernel,
    dim3(gridSize), dim3(blockSize),
    kernelArgs);
```

---

## **21.7 Lock-Free Data Structures**

### **Lock-Free Stack**

```cpp
struct Node {
    int data;
    Node* next;
};

class LockFreeStack {
    Node* head;

    __device__ void push(Node* node) {
        Node* old_head;
        do {
            old_head = head;
            node->next = old_head;
        } while (atomicCAS((unsigned long long*)&head,
                          (unsigned long long)old_head,
                          (unsigned long long)node) !=
                 (unsigned long long)old_head);
    }

    __device__ Node* pop() {
        Node* old_head, *new_head;
        do {
            old_head = head;
            if (!old_head) return nullptr;
            new_head = old_head->next;
        } while (atomicCAS((unsigned long long*)&head,
                          (unsigned long long)old_head,
                          (unsigned long long)new_head) !=
                 (unsigned long long)old_head);
        return old_head;
    }
};
```

---

## **21.8 Performance Considerations**

### **Atomic Operation Performance**

| Factor | Impact | Optimization |
|--------|--------|--------------|
| **Contention** | High contention degrades performance | Use privatization |
| **Memory Type** | Shared memory atomics are faster | Use shared memory when possible |
| **Data Type** | Native atomics are faster | Avoid custom CAS loops |
| **Access Pattern** | Coalesced atomics perform better | Organize data layout |

### **Optimization Strategies**

1. **Privatization**
```cpp
// Instead of global atomic
atomicAdd(&global_sum, value);

// Use local accumulation + single atomic
__shared__ float local_sum;
if (threadIdx.x == 0) local_sum = 0;
__syncthreads();
atomicAdd(&local_sum, value);
__syncthreads();
if (threadIdx.x == 0) atomicAdd(&global_sum, local_sum);
```

2. **Warp Aggregation**
```cpp
// Reduce within warp first
float warp_sum = warp_reduce(value);
if (laneId == 0) atomicAdd(&result, warp_sum);
```

---

## **21.9 Running the Examples**

### **Building**
```bash
cd build
cmake --build . --target 10_Synchronization_and_Atomics
```

### **Running Main Demo**
```bash
./20.cuda_intermediate/10.Synchronization_and_Atomics/10_Synchronization_and_Atomics
```

### **Running Tests**
```bash
ctest -R 10_Synchronization_and_Atomics
# Or with verbose output
./20.cuda_intermediate/10.Synchronization_and_Atomics/10_Synchronization_and_Atomics_test
```

---

## **21.10 Profiling and Analysis**

### **Atomic Operations Analysis**
```bash
ninja 21_Synchronization_and_Atomics_atomic_analysis
```

### **Synchronization Overhead**
```bash
ninja 21_Synchronization_and_Atomics_sync_overhead
```

### **Memory Fence Impact**
```bash
ninja 21_Synchronization_and_Atomics_fence_analysis
```

### **Lock Contention Analysis**
```bash
ninja 21_Synchronization_and_Atomics_lock_contention
```

---

## **21.11 Expected Output**

```
Using device: NVIDIA GeForce RTX 3080
Compute capability: 8.6

=== Basic Synchronization ===
Reduction result: 524800 (expected: 524800)
Result is CORRECT

=== Atomic Operations ===
Atomic counter: 25600000 (expected: 25600000)
Result is CORRECT

Histogram (10 bins):
  Bin 0: 1000
  Bin 1: 1000
  Bin 2: 1000
  ...
Total: 10000 (expected: 10000)

=== Memory Fences ===
Producer-Consumer result: 42.00 (expected: 42.00)
Result is CORRECT

=== Cooperative Groups ===
Device supports cooperative launch
Grid sync kernel completed successfully

=== Spinlock ===
Spinlock counter: 32000 (expected: 32000)
Time: 2.45 ms
Result is CORRECT

=== Atomic Performance Benchmark ===
Config (32 threads, 1 blocks): 0.12 ms, counter = 3200
Config (256 threads, 1 blocks): 0.45 ms, counter = 25600
Config (256 threads, 10 blocks): 4.23 ms, counter = 256000
Config (1024 threads, 10 blocks): 16.78 ms, counter = 1024000

=== Synchronization and Atomics Demo Complete ===
```

---

## **21.12 Common Issues and Solutions**

### **Problem 1: Deadlock with __syncthreads()**
**Symptoms**: Kernel hangs indefinitely
**Solution**:
```cpp
// BAD - conditional sync
if (threadIdx.x < 128) {
    __syncthreads();  // Deadlock!
}

// GOOD - all threads call sync
__syncthreads();
if (threadIdx.x < 128) {
    // Process
}
```

### **Problem 2: Race Conditions**
**Symptoms**: Inconsistent results
**Solution**: Use atomic operations or proper synchronization

### **Problem 3: Poor Atomic Performance**
**Symptoms**: Slow execution with atomics
**Solution**: Reduce contention through privatization or hierarchical reduction

### **Problem 4: Memory Consistency Issues**
**Symptoms**: Stale data reads
**Solution**: Use appropriate memory fences

---

## **21.13 Best Practices**

1. **Minimize Synchronization**
   - Synchronization is expensive
   - Design algorithms to minimize sync points
   - Use warp-level operations when possible

2. **Reduce Atomic Contention**
   - Use shared memory atomics over global
   - Implement hierarchical reductions
   - Consider privatization strategies

3. **Correct Synchronization**
   - Always call __syncthreads() from all threads
   - Use memory fences for cross-block communication
   - Verify with race detection tools

4. **Optimize for Hardware**
   - Warp-level primitives are fast
   - Native atomics perform better
   - Consider compute capability differences

5. **Use Cooperative Groups**
   - More flexible than traditional sync
   - Enables dynamic grouping
   - Supports modern GPU features

---

## **21.14 Advanced Topics**

### **Persistent Threads**
```cpp
__global__ void persistent_kernel(Queue* work_queue) {
    while (true) {
        Work* work = work_queue->dequeue();
        if (!work) break;
        process(work);
    }
}
```

### **Hierarchical Locking**
```cpp
__shared__ int block_lock;
__device__ int global_lock;

// Acquire in order: thread -> warp -> block -> global
```

### **Memory Ordering Models**
- **Relaxed**: No ordering guarantees
- **Acquire**: Subsequent reads see latest values
- **Release**: Previous writes are visible
- **Sequential**: Total order across all threads

---

## **21.15 Exercises**

### **Exercise 1: Custom Atomic Operation**
Implement atomic multiply for floats using CAS

### **Exercise 2: Barrier Implementation**
Create a reusable barrier for N threads using atomics

### **Exercise 3: Lock-Free Queue**
Implement a lock-free FIFO queue using CAS operations

### **Exercise 4: Reduction Optimization**
Compare performance of different reduction strategies:
- Global atomics only
- Shared memory + atomics
- Warp shuffle + atomics
- Cooperative groups

---

## **21.16 References**

- [CUDA Programming Guide - Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions)
- [Cooperative Groups Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- [Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

---

## ✅ **Summary**

This section covered:
- Thread synchronization at multiple levels
- Comprehensive atomic operations
- Memory fences and consistency models
- Lock-free data structure implementation
- Cooperative groups API
- Performance optimization strategies

**Key Takeaways:**
- Synchronization is necessary but expensive
- Atomic operations enable safe concurrent updates
- Memory fences ensure visibility across threads
- Cooperative groups provide modern synchronization
- Optimization requires reducing contention

---

📄 **Next**: Part 11 - Streams and Asynchronous Execution