# 💾 Part 9: CUDA Memory API

**Goal**: Master CUDA memory management APIs including allocation strategies, transfer optimizations, and advanced memory features.

---

## **9.1 Overview**

The CUDA Memory API provides comprehensive control over GPU memory management. This section covers:
- Memory allocation strategies
- Transfer optimization techniques
- Unified memory management
- Pinned and mapped memory
- Memory pools and async operations
- Texture and constant memory
- Performance measurement and profiling

---

## **9.2 Memory Types and APIs**

### **Memory Type Comparison**

| Memory Type | Allocation API | Location | Access | Cache | Lifetime |
|------------|---------------|----------|--------|-------|----------|
| **Global** | `cudaMalloc()` | Device | R/W | L1/L2 | Explicit |
| **Shared** | `__shared__` | On-chip | R/W | N/A | Block |
| **Local** | Automatic | Off-chip | R/W | L1/L2 | Thread |
| **Constant** | `cudaMemcpyToSymbol()` | Device | R | Yes | Application |
| **Texture** | Texture API | Device | R | Yes | Explicit |
| **Pinned** | `cudaHostAlloc()` | Host | R/W | Host | Explicit |
| **Unified** | `cudaMallocManaged()` | Managed | R/W | Both | Explicit |
| **Zero-copy** | `cudaHostAlloc(Mapped)` | Host | R/W | None | Explicit |

---

## **9.3 Basic Memory Operations**

### **9.3.1 Allocation and Deallocation**

```cpp
// Device memory
float* d_data;
cudaMalloc(&d_data, bytes);
cudaFree(d_data);

// Host pinned memory
float* h_pinned;
cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault);
cudaFreeHost(h_pinned);

// Unified memory
float* unified;
cudaMallocManaged(&unified, bytes);
cudaFree(unified);
```

### **9.3.2 Memory Transfers**

```cpp
// Synchronous transfers
cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);

// Asynchronous transfers
cudaMemcpyAsync(dst, src, bytes, kind, stream);

// 2D/3D transfers
cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
cudaMemcpy3D(&params);
```

---

## **9.4 Pinned Memory**

### **Benefits**
- Higher transfer bandwidth (up to 2x)
- Enables async transfers
- Allows overlapping with kernel execution
- Required for zero-copy access

### **Allocation Flags**

| Flag | Description |
|------|-------------|
| `cudaHostAllocDefault` | Standard pinned memory |
| `cudaHostAllocPortable` | Accessible from all CUDA contexts |
| `cudaHostAllocMapped` | Mapped to device address space |
| `cudaHostAllocWriteCombined` | Write-combined memory (faster writes) |

### **Example Usage**
```cpp
float* h_pinned;
cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault);

// Async transfer with stream
cudaMemcpyAsync(d_data, h_pinned, bytes,
                cudaMemcpyHostToDevice, stream);
```

---

## **9.5 Unified Memory**

### **Automatic Data Migration**
```cpp
float* data;
cudaMallocManaged(&data, bytes);

// CPU access - data migrates to host
for (int i = 0; i < N; i++) {
    data[i] = i;
}

// GPU access - data migrates to device
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// CPU access again - migrates back
float sum = data[0];
```

### **Prefetching and Hints**
```cpp
// Prefetch to device
cudaMemPrefetchAsync(data, bytes, deviceId, stream);

// Provide access hints
cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, deviceId);
cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, deviceId);
```

---

## **9.6 Zero-Copy Memory**

### **Direct Host Memory Access**
```cpp
// Enable mapped memory
cudaSetDeviceFlags(cudaDeviceMapHost);

// Allocate mapped memory
float* h_data;
cudaHostAlloc(&h_data, bytes, cudaHostAllocMapped);

// Get device pointer
float* d_data;
cudaHostGetDevicePointer(&d_data, h_data, 0);

// Kernel accesses host memory directly
kernel<<<grid, block>>>(d_data, N);
```

**Use Cases:**
- Large datasets that don't fit in GPU memory
- Infrequent GPU access patterns
- Real-time data streaming

---

## **9.7 Constant Memory**

### **Declaration and Usage**
```cpp
// Declare constant memory
__constant__ float d_const_data[256];

// Copy to constant memory
cudaMemcpyToSymbol(d_const_data, h_data, bytes);

// Access in kernel
__global__ void kernel() {
    float value = d_const_data[threadIdx.x];
}
```

**Benefits:**
- Cached on-chip
- Broadcast to all threads in warp
- Ideal for read-only data accessed by all threads

---

## **9.8 Texture Memory**

### **Texture Objects (Modern API)**
```cpp
// Create texture object
cudaTextureObject_t texObj;
cudaResourceDesc resDesc;
cudaTextureDesc texDesc;

// Configure and create
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

// Use in kernel
__global__ void kernel(cudaTextureObject_t tex) {
    float value = tex1D<float>(tex, x);
}

// Cleanup
cudaDestroyTextureObject(texObj);
```

**Benefits:**
- Spatial locality caching
- Hardware interpolation
- Boundary handling modes

---

## **9.9 Memory Pools (CUDA 11.2+)**

### **Stream-Ordered Memory Allocation**
```cpp
// Create memory pool
cudaMemPool_t mempool;
cudaMemPoolCreate(&mempool, &props);

// Allocate from pool
cudaMallocFromPoolAsync(&ptr, size, mempool, stream);

// Free to pool
cudaFreeAsync(ptr, stream);

// Destroy pool
cudaMemPoolDestroy(mempool);
```

**Benefits:**
- Reduced allocation overhead
- Better memory reuse
- Stream-ordered semantics

---

## **9.10 Performance Optimization**

### **Transfer Optimization Strategies**

| Strategy | Description | Speedup |
|----------|-------------|---------|
| **Pinned Memory** | Use for all transfers | 2x |
| **Async Transfers** | Overlap with compute | 1.5-2x |
| **Batched Transfers** | Minimize overhead | 1.2-1.5x |
| **Direct Access** | Zero-copy for streaming | Variable |
| **Compression** | Reduce data size | 2-10x |

### **Bandwidth Calculation**
```cpp
Effective Bandwidth = Bytes Transferred / Time
Theoretical Peak = Memory Clock × Bus Width × 2 (DDR)
```

---

## **9.11 Running the Examples**

### **Building**
```bash
cd build
cmake --build . --target 19_CUDA_Memory_API
```

### **Running Main Demo**
```bash
./10.cuda_basic/19.CUDA_Memory_API/19_CUDA_Memory_API
```

### **Running Tests**
```bash
# Run all tests
ctest -R 19_CUDA_Memory_API

# Run with verbose output
./10.cuda_basic/19.CUDA_Memory_API/19_CUDA_Memory_API_test
```

---

## **9.12 Profiling and Analysis**

### **Memory Transfer Analysis**
```bash
make 19_CUDA_Memory_API_transfer_analysis
```

### **Unified Memory Analysis**
```bash
make 19_CUDA_Memory_API_unified_memory_analysis
```

### **Pinned Memory Comparison**
```bash
make 19_CUDA_Memory_API_pinned_memory_analysis
```

### **Memory Pool Analysis**
```bash
make 19_CUDA_Memory_API_memory_pool_analysis
```

### **Async Operations Analysis**
```bash
make 19_CUDA_Memory_API_async_analysis
```

---

## **9.13 Expected Output**

```
Using device: NVIDIA GeForce RTX 3080
Compute capability: 8.6

=== Memory Information ===
GPU Memory:
  Total: 10.00 GB
  Free: 9.50 GB
  Used: 0.50 GB

Memory Properties:
  L2 Cache Size: 5.00 MB
  Total Constant Memory: 64.00 KB
  Shared Memory per Block: 48.00 KB
  Memory Bus Width: 320 bits
  Memory Clock Rate: 9.50 GHz
  Peak Memory Bandwidth: 760.00 GB/s

=== Basic Memory Allocation ===
Host to Device transfer: 2.45 ms (163.27 GB/s)
Device to Host transfer: 2.38 ms (168.13 GB/s)

=== Pinned Memory ===
Regular memory transfer: 4.82 ms
Pinned memory transfer: 2.41 ms
Speedup: 2.0x

=== Unified Memory ===
Unified memory kernel execution: 1.23 ms
Sum of first 1000 elements: 1998000.00

=== Zero-Copy Memory ===
Zero-copy kernel execution: 15.67 ms
First 5 results: 0.00 3.00 6.00 9.00 12.00

=== Constant Memory ===
Constant memory kernel execution: 45.23 microseconds
First 5 outputs: 2.00 4.00 2.00 4.00 2.00

=== Texture Memory ===
Texture memory kernel execution: 89.45 microseconds

=== Asynchronous Memory Operations ===
Async operations on 2 streams: 3.45 ms

=== Memory Pools ===
Memory pool operations completed successfully

=== CUDA Memory API Demo Complete ===
```

---

## **9.14 Common Issues and Solutions**

### **Problem 1: Out of Memory**
**Symptoms**: `cudaErrorMemoryAllocation`
**Solutions**:
- Check available memory with `cudaMemGetInfo()`
- Free unused allocations
- Use unified memory for oversubscription
- Implement memory pooling

### **Problem 2: Low Transfer Bandwidth**
**Symptoms**: Slow H2D/D2H transfers
**Solutions**:
- Use pinned memory
- Enable async transfers
- Batch small transfers
- Check PCIe configuration

### **Problem 3: Unified Memory Page Faults**
**Symptoms**: Poor performance with managed memory
**Solutions**:
- Use prefetching
- Provide memory hints
- Minimize migration frequency
- Consider explicit memory management

### **Problem 4: Memory Leaks**
**Symptoms**: Decreasing available memory
**Solutions**:
- Match every `cudaMalloc` with `cudaFree`
- Use RAII wrappers
- Enable `cuda-memcheck`
- Check for exceptions before cleanup

---

## **9.15 Best Practices**

### **Memory Management Guidelines**

1. **Choose the Right Memory Type**
   - Global: General purpose data
   - Shared: Frequently accessed tile data
   - Constant: Read-only parameters
   - Texture: Spatial access patterns

2. **Optimize Transfers**
   - Always use pinned memory for transfers
   - Overlap transfers with computation
   - Minimize transfer frequency
   - Use compression when applicable

3. **Unified Memory Strategy**
   - Good for prototyping
   - Use prefetching for production
   - Monitor page fault metrics
   - Consider fallback to explicit management

4. **Error Handling**
   ```cpp
   #define CHECK_CUDA(call) \
       do { \
           cudaError_t err = call; \
           if (err != cudaSuccess) { \
               /* Handle error */ \
           } \
       } while(0)
   ```

5. **Memory Pooling**
   - Reuse allocations when possible
   - Implement custom allocators for frequent alloc/free
   - Use CUDA memory pools (11.2+)

---

## **9.16 Advanced Topics**

### **Virtual Memory Management (CUDA 10.2+)**
```cpp
CUmemGenericAllocationHandle handle;
cuMemCreate(&handle, size, &prop, 0);
cuMemMap(ptr, size, 0, handle, 0);
cuMemSetAccess(ptr, size, &accessDesc, 1);
```

### **Memory Compression**
- Automatic compression in newer GPUs
- Can increase effective bandwidth
- Monitor compression ratios in profiler

### **Multi-GPU Memory**
```cpp
// Peer access
cudaDeviceEnablePeerAccess(peer_device, 0);

// Direct transfer
cudaMemcpyPeer(dst, dst_device, src, src_device, bytes);
```

### **Graph Memory Nodes**
```cpp
cudaGraphAddMemcpyNode(&memcpyNode, graph,
                       dependencies, numDeps, &memcpyParams);
```

---

## **9.17 Performance Metrics**

### **Key Metrics to Monitor**
- `dram__bytes_read.sum`: DRAM read bytes
- `dram__bytes_write.sum`: DRAM write bytes
- `lts__t_bytes.sum`: L2 cache traffic
- `gpu__time_duration.sum`: Kernel duration
- `sm__memory_throughput.avg.pct_of_peak_sustained_elapsed`: Memory efficiency

### **Bandwidth Utilization**
```
Efficiency = Effective Bandwidth / Theoretical Peak × 100%
```

---

## **9.18 Exercises**

### **Exercise 1: Bandwidth Measurement**
Implement a benchmark to measure:
- H2D and D2H bandwidth for different sizes
- Impact of pinned memory
- Async vs sync transfers

### **Exercise 2: Unified Memory Migration**
Create a program that:
- Allocates unified memory
- Triggers migrations between CPU and GPU
- Measures migration overhead
- Uses prefetching to optimize

### **Exercise 3: Memory Pool Implementation**
Build a custom memory pool that:
- Maintains a free list
- Coalesces adjacent free blocks
- Handles different allocation sizes
- Provides statistics

### **Exercise 4: Zero-Copy Streaming**
Implement real-time processing:
- Stream data from host
- Process on GPU without explicit copies
- Measure latency vs throughput

---

## **9.19 References**

- [CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)
- [Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory)
- [Memory Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [CUDA Memory Checker](https://docs.nvidia.com/cuda/cuda-memcheck/index.html)

---

## ✅ **Summary**

This section demonstrated:
- Comprehensive memory allocation strategies
- Transfer optimization techniques
- Unified memory management and prefetching
- Pinned and zero-copy memory usage
- Constant and texture memory benefits
- Memory pools and async operations
- Performance measurement and profiling

**Key Takeaways:**
- Memory management is critical for GPU performance
- Pinned memory doubles transfer bandwidth
- Unified memory simplifies programming but needs tuning
- Different memory types serve different access patterns
- Profiling is essential for optimization

---

📄 **Next**: Part 10 - Advanced Memory Patterns