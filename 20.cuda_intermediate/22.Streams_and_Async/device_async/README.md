# 36. Asynchronous Memory Operations

## 36.1 Introduction to Asynchronous Memory

Asynchronous memory operations enable non-blocking data transfers and computation overlap, crucial for high-performance GPU applications.

## 36.2 Learning Objectives

- Understand asynchronous memory copy concepts
- Implement async memory transfers with barriers
- Use CUDA barriers and pipelines effectively
- Optimize memory throughput with async operations
- Design efficient producer-consumer patterns

## 36.3 Asynchronous Memory Basics

### 36.3.1 What is Asynchronous Memory?

Asynchronous memory operations allow:
- Non-blocking memory transfers
- Computation and memory transfer overlap
- Fine-grained synchronization control
- Improved memory throughput
- Reduced latency for memory-bound kernels

### 36.3.2 CUDA Barriers

```cuda
#include <cuda/barrier>

__global__ void basic_barrier_kernel() {
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    // All threads arrive at barrier
    barrier.arrive_and_wait();

    // Continue after barrier
}
```

## 36.4 Asynchronous Memory Copy

### 36.4.1 Basic Async Copy

```cuda
#include <cuda/barrier>

__global__ void async_copy_simple(float* global_data) {
    __shared__ float shared_data[256];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    // Asynchronous copy from global to shared
    cuda::memcpy_async(shared_data + threadIdx.x,
                       global_data + blockIdx.x * 256 + threadIdx.x,
                       sizeof(float),
                       barrier);

    // Wait for async copy completion
    barrier.arrive_and_wait();

    // Process shared memory data
    shared_data[threadIdx.x] *= 2.0f;

    __syncthreads();

    // Copy back to global memory
    global_data[blockIdx.x * 256 + threadIdx.x] = shared_data[threadIdx.x];
}
```

### 36.4.2 Advanced Async Operations

```cuda
#include <cuda/pipeline>

__global__ void async_pipeline_kernel(float* input, float* output, int n) {
    constexpr int stages = 2;
    __shared__ float shared[2][256];  // Double buffering

    auto block = cooperative_groups::this_thread_block();
    cuda::pipeline<cuda::thread_scope_block> pipe;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Pipeline prologue
    for (int s = 0; s < stages - 1; ++s) {
        pipe.producer_acquire();
        if (tid < 256) {
            cuda::memcpy_async(block, shared[s] + tid,
                              input + (bid * stages + s) * 256 + tid,
                              sizeof(float), pipe);
        }
        pipe.producer_commit();
    }

    // Main pipeline loop
    for (int i = stages - 1; i < n / 256; ++i) {
        int stage = i % stages;

        // Start next async transfer
        pipe.producer_acquire();
        if (tid < 256) {
            cuda::memcpy_async(block, shared[stage] + tid,
                              input + (bid * stages + i) * 256 + tid,
                              sizeof(float), pipe);
        }
        pipe.producer_commit();

        // Process previous stage
        int prev_stage = (stage + 1) % stages;
        pipe.consumer_wait();

        // Compute on shared memory
        if (tid < 256) {
            shared[prev_stage][tid] = shared[prev_stage][tid] * 2.0f + 1.0f;
        }

        pipe.consumer_release();

        // Store results
        if (tid < 256) {
            output[(bid * stages + i - stages + 1) * 256 + tid] =
                shared[prev_stage][tid];
        }
    }

    // Pipeline epilogue
    for (int s = 0; s < stages - 1; ++s) {
        pipe.consumer_wait();
        if (tid < 256) {
            shared[s][tid] = shared[s][tid] * 2.0f + 1.0f;
            output[(bid * stages + n / 256 - stages + 1 + s) * 256 + tid] =
                shared[s][tid];
        }
        pipe.consumer_release();
    }
}
```

## 36.5 Memory Pipeline Patterns

### 36.5.1 Producer-Consumer Pattern

```cuda
template<int BUFFER_SIZE>
__global__ void producer_consumer_kernel(float* data_queue,
                                        int* queue_head,
                                        int* queue_tail,
                                        int n_items) {
    __shared__ float local_buffer[BUFFER_SIZE];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    int tid = threadIdx.x;
    int producer_threads = blockDim.x / 2;

    if (tid < producer_threads) {
        // Producer threads
        for (int i = tid; i < n_items; i += producer_threads) {
            // Produce data
            float value = generate_data(i);

            // Wait for buffer space
            while (atomicAdd(queue_tail, 0) - atomicAdd(queue_head, 0) >= BUFFER_SIZE) {
                __nanosleep(100);
            }

            // Add to queue
            int pos = atomicAdd(queue_tail, 1) % BUFFER_SIZE;

            // Async copy to shared buffer
            cuda::memcpy_async(&local_buffer[pos], &value,
                              sizeof(float), barrier);
        }
    } else {
        // Consumer threads
        int consumer_id = tid - producer_threads;
        int consumer_count = blockDim.x - producer_threads;

        while (true) {
            // Wait for data
            while (atomicAdd(queue_head, 0) >= atomicAdd(queue_tail, 0)) {
                __nanosleep(100);
            }

            // Get from queue
            int pos = atomicAdd(queue_head, 1) % BUFFER_SIZE;

            barrier.arrive_and_wait();

            // Process data
            float value = local_buffer[pos];
            process_data(value, consumer_id);

            if (atomicAdd(queue_head, 0) >= n_items) break;
        }
    }
}
```

### 36.5.2 Multi-Stage Pipeline

```cuda
template<int STAGES, int STAGE_SIZE>
__global__ void multi_stage_pipeline(float* input, float* output, int n) {
    __shared__ float stage_buffers[STAGES][STAGE_SIZE];
    __shared__ cuda::barrier<cuda::thread_scope_block> stage_barriers[STAGES];

    // Initialize barriers
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init(&stage_barriers[s], blockDim.x);
        }
    }
    __syncthreads();

    int tid = threadIdx.x;
    int items_per_stage = n / STAGES;

    // Pipeline stages
    for (int batch = 0; batch < items_per_stage; batch += STAGE_SIZE) {
        for (int stage = 0; stage < STAGES; ++stage) {
            int stage_idx = (batch / STAGE_SIZE + stage) % STAGES;

            if (stage == 0) {
                // Load stage - async copy from global
                if (tid < STAGE_SIZE && batch + tid < n) {
                    cuda::memcpy_async(&stage_buffers[stage_idx][tid],
                                      &input[batch + tid],
                                      sizeof(float),
                                      stage_barriers[stage_idx]);
                }
            } else {
                // Processing stages
                stage_barriers[stage_idx].arrive_and_wait();

                if (tid < STAGE_SIZE) {
                    // Stage-specific processing
                    float value = stage_buffers[stage_idx][tid];

                    switch (stage) {
                        case 1: value = sqrtf(value); break;
                        case 2: value = value * value; break;
                        case 3: value = logf(value + 1.0f); break;
                        default: value = expf(value); break;
                    }

                    stage_buffers[stage_idx][tid] = value;
                }

                if (stage == STAGES - 1) {
                    // Final stage - store results
                    __syncthreads();
                    if (tid < STAGE_SIZE && batch + tid < n) {
                        output[batch + tid] = stage_buffers[stage_idx][tid];
                    }
                }
            }
        }
    }
}
```

## 36.6 Performance Optimization

### 36.6.1 Memory Bandwidth Optimization

```cuda
__global__ void bandwidth_optimized_async(float4* input, float4* output, int n) {
    __shared__ float4 shared_buffer[64];  // Vectorized shared memory
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    __syncthreads();

    int tid = threadIdx.x;
    int blocks_per_grid = gridDim.x;

    for (int i = blockIdx.x; i < n / 256; i += blocks_per_grid) {
        // Vectorized async copy - 4 floats per thread
        if (tid < 64) {
            cuda::memcpy_async(&shared_buffer[tid],
                              &input[i * 64 + tid],
                              sizeof(float4),
                              barrier);
        }

        barrier.arrive_and_wait();

        // Process vectorized data
        if (tid < 64) {
            float4 data = shared_buffer[tid];
            data.x = fmaf(data.x, 2.0f, 1.0f);
            data.y = fmaf(data.y, 2.0f, 1.0f);
            data.z = fmaf(data.z, 2.0f, 1.0f);
            data.w = fmaf(data.w, 2.0f, 1.0f);
            shared_buffer[tid] = data;
        }

        __syncthreads();

        // Write back vectorized
        if (tid < 64) {
            output[i * 64 + tid] = shared_buffer[tid];
        }
    }
}
```

### 36.6.2 Latency Hiding

```cuda
template<int PREFETCH_DISTANCE>
__global__ void latency_hiding_kernel(float* data, int n) {
    __shared__ float prefetch_buffer[PREFETCH_DISTANCE][256];
    __shared__ cuda::barrier<cuda::thread_scope_block> prefetch_barriers[PREFETCH_DISTANCE];

    // Initialize barriers
    if (threadIdx.x == 0) {
        for (int i = 0; i < PREFETCH_DISTANCE; ++i) {
            init(&prefetch_barriers[i], blockDim.x);
        }
    }
    __syncthreads();

    int tid = threadIdx.x;

    // Prefetch initial data
    for (int pf = 0; pf < PREFETCH_DISTANCE && pf < n / 256; ++pf) {
        if (tid < 256) {
            cuda::memcpy_async(&prefetch_buffer[pf][tid],
                              &data[pf * 256 + tid],
                              sizeof(float),
                              prefetch_barriers[pf]);
        }
    }

    // Main processing loop
    for (int block = 0; block < n / 256; ++block) {
        int buffer_idx = block % PREFETCH_DISTANCE;

        // Wait for current block data
        prefetch_barriers[buffer_idx].arrive_and_wait();

        // Start prefetch for future block
        int prefetch_block = block + PREFETCH_DISTANCE;
        if (prefetch_block < n / 256 && tid < 256) {
            cuda::memcpy_async(&prefetch_buffer[buffer_idx][tid],
                              &data[prefetch_block * 256 + tid],
                              sizeof(float),
                              prefetch_barriers[buffer_idx]);
        }

        // Process current block
        if (tid < 256) {
            float value = prefetch_buffer[buffer_idx][tid];
            value = expensive_computation(value);  // Hide latency
            data[block * 256 + tid] = value;
        }
    }
}
```

## 36.7 Practical Examples

### 36.7.1 Asynchronous Matrix Processing

```cuda
__global__ void async_matrix_process(float* matrix, int rows, int cols) {
    constexpr int TILE_SIZE = 16;
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(&barrier, blockDim.x * blockDim.y);
    }
    __syncthreads();

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    if (row < rows && col < cols) {
        // Async load tile
        cuda::memcpy_async(&tile[ty][tx],
                          &matrix[row * cols + col],
                          sizeof(float),
                          barrier);

        barrier.arrive_and_wait();

        // Process tile with neighbor communication
        float center = tile[ty][tx];
        float neighbors = 0.0f;

        if (ty > 0) neighbors += tile[ty-1][tx];
        if (ty < TILE_SIZE-1) neighbors += tile[ty+1][tx];
        if (tx > 0) neighbors += tile[ty][tx-1];
        if (tx < TILE_SIZE-1) neighbors += tile[ty][tx+1];

        float result = 0.2f * (center + 0.25f * neighbors);

        __syncthreads();
        matrix[row * cols + col] = result;
    }
}
```

## 36.8 Exercises

1. **Basic Async Operations**: Implement a kernel using async memory copy with barriers
2. **Pipeline Design**: Create a multi-stage processing pipeline with async transfers
3. **Producer-Consumer**: Build a producer-consumer pattern with shared memory queues
4. **Performance Analysis**: Compare sync vs async memory patterns
5. **Advanced Synchronization**: Implement split barriers for overlapping computation

## 36.9 Building and Running

```bash
# Build with async memory examples
cd build/30.cuda_advanced/36.Asynchronous_Memory
ninja

# Run examples
./36_AsyncMemory_basic_async
./36_AsyncMemory_pipeline
./36_AsyncMemory_producer_consumer

# Profile async operations
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
              l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    ./36_AsyncMemory_pipeline

# Analyze memory access patterns
nsys profile --stats=true -t cuda ./36_AsyncMemory_basic_async
```

## 36.10 Key Takeaways

- Asynchronous memory operations enable computation-transfer overlap
- CUDA barriers provide fine-grained synchronization control
- Pipeline patterns maximize memory throughput
- Producer-consumer patterns enable complex data flows
- Proper async design significantly improves performance
- Understanding memory latency hiding is crucial for optimization
