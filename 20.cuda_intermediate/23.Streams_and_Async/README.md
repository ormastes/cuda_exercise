# CUDA Streams and Asynchronous Execution

## Introduction

CUDA Streams enable concurrent execution of multiple operations on the GPU, allowing for better hardware utilization through overlapping computation and data transfer. This tutorial covers stream management, asynchronous operations, and optimization techniques.

## Key Concepts

### What are CUDA Streams?
- **Stream**: A sequence of operations that execute in order on the GPU
- **Default Stream**: Stream 0, used when no stream is specified
- **Concurrent Execution**: Multiple streams can execute simultaneously
- **Asynchronous Operations**: Operations that return control to CPU immediately

### Stream Types
1. **Default Stream (Stream 0)**
   - Legacy default stream: Synchronizes with all other streams
   - Per-thread default stream: Independent per CPU thread

2. **Non-default Streams**
   - Created explicitly
   - Can run concurrently with other non-default streams
   - No implicit synchronization

## Benefits of Streams

1. **Overlap Computation and Transfer**: Hide memory transfer latency
2. **Increased Throughput**: Keep GPU busy with multiple tasks
3. **Better Resource Utilization**: Use all available hardware units
4. **Reduced Total Execution Time**: Parallel execution of independent operations

## Stream Operations

### Creating and Destroying Streams
```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
// Use stream...
cudaStreamDestroy(stream);
```

### Asynchronous Memory Operations
```cuda
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
cudaMemsetAsync(ptr, value, count, stream);
```

### Kernel Execution
```cuda
kernel<<<blocks, threads, shared_mem, stream>>>(...);
```

### Stream Synchronization
```cuda
cudaStreamSynchronize(stream);    // Wait for specific stream
cudaDeviceSynchronize();          // Wait for all streams
cudaStreamWaitEvent(stream, event); // Stream waits for event
```

## Concurrency Patterns

### Pattern 1: Overlapping Transfers and Kernels
```
Stream 1: H2D Transfer → Kernel → D2H Transfer
Stream 2:              H2D Transfer → Kernel → D2H Transfer
```

### Pattern 2: Pipeline Pattern
```
Time →
Stream 1: H2D[0] → Kernel[0] → D2H[0]
Stream 2:        H2D[1] → Kernel[1] → D2H[1]
Stream 3:               H2D[2] → Kernel[2] → D2H[2]
```

### Pattern 3: Fork-Join Pattern
Multiple streams process different parts of data, then synchronize.

## Examples in This Section

1. **basic_streams.cu**: Introduction to stream creation and async operations
2. **overlap_transfer_compute.cu**: Overlapping data transfers with computation
3. **multi_stream_pipeline.cu**: Pipeline pattern for processing large datasets
4. **stream_callback.cu**: Using stream callbacks for CPU-GPU coordination

## Hardware Requirements

### Compute Capability Requirements
- **CC 1.1+**: Basic stream support
- **CC 2.0+**: Concurrent kernel execution
- **CC 3.5+**: Hyper-Q (32 concurrent kernels)
- **CC 6.0+**: Improved async copy engines

### Hardware Engines
Modern GPUs have multiple hardware engines:
- **Compute Engine**: Executes kernels
- **Copy Engines**: Handle H2D and D2H transfers (usually 2)
- **Hyper-Q**: Hardware work queues for concurrent kernel execution

## Best Practices

### 1. Use Pinned Memory
```cuda
cudaHostAlloc(&host_ptr, size, cudaHostAllocDefault);
// or
cudaMallocHost(&host_ptr, size);
```
Pinned memory enables:
- Faster transfers (2-3x)
- True asynchronous transfers
- Direct GPU access (zero-copy)

### 2. Stream Management
- Create streams at initialization, reuse throughout application
- Limit number of streams (diminishing returns beyond hardware capacity)
- Use stream pools for dynamic workloads

### 3. Kernel Configuration
- Ensure kernels are small enough to run concurrently
- Balance grid size with stream count
- Consider using cooperative groups for inter-stream synchronization

### 4. Memory Transfer Optimization
- Break large transfers into smaller chunks
- Use multiple streams for transfers
- Overlap H2D and D2H on different copy engines

### 5. Synchronization
- Minimize use of `cudaDeviceSynchronize()`
- Use events for fine-grained synchronization
- Leverage stream callbacks for CPU-side work

## Common Pitfalls

1. **False Dependencies**: Default stream synchronizes with all streams
2. **Pageable Memory**: Async transfers become synchronous
3. **Large Kernels**: Prevent concurrent execution
4. **Resource Limits**: Exceeding shared memory or register limits
5. **Implicit Synchronization**: Some CUDA APIs cause synchronization

## Performance Metrics

### Key Metrics to Monitor
- **Overlap Percentage**: Time with concurrent operations
- **Transfer-Compute Overlap**: Hiding transfer latency
- **Stream Efficiency**: Utilization of created streams
- **Timeline Analysis**: Visual profiling with Nsight Systems

### Profiling Commands
```bash
# Nsight Systems timeline
nsys profile --stats=true ./executable

# Nsight Compute metrics
ncu --metrics gpu__compute_memory_throughput ./executable
```

## Advanced Topics

### CUDA Graphs
- Capture stream operations into a graph
- Launch entire workflow with single API call
- Reduced launch overhead

### Priority Streams
```cuda
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high);
```

### Stream Capture
```cuda
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// Record operations...
cudaStreamEndCapture(stream, &graph);
```

## Compile and Run

```bash
# Build all examples
mkdir build && cd build
cmake ..
make

# Run individual examples
./basic_streams
./overlap_transfer_compute
./multi_stream_pipeline
./stream_callback

# Profile with Nsight Systems
nsys profile --stats=true ./overlap_transfer_compute
```

## Further Reading

- [CUDA C++ Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [CUDA Runtime API - Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
- [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- [CUDA Streams: Best Practices and Common Pitfalls](https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf)