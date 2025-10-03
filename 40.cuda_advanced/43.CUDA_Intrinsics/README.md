# 32. CUDA Intrinsics

## 32.1 Introduction

CUDA intrinsics provide direct access to GPU hardware features and optimized operations.

## 32.2 Learning Objectives

- Master warp-level primitives
- Use math intrinsics effectively
- Implement voting and synchronization
- Optimize with hardware intrinsics

## 32.3 Warp-Level Primitives

### 32.3.1 Shuffle Instructions

```cuda
// Basic shuffle operations
__device__ int warp_shuffle_example(int value) {
    int lane_id = threadIdx.x % 32;

    // Shuffle down
    int down = __shfl_down_sync(0xffffffff, value, 1);

    // Shuffle up
    int up = __shfl_up_sync(0xffffffff, value, 1);

    // Shuffle xor (butterfly)
    int xor_val = __shfl_xor_sync(0xffffffff, value, 1);

    // Broadcast from lane 0
    int broadcast = __shfl_sync(0xffffffff, value, 0);

    return broadcast;
}

// Warp reduction using shuffle
template<typename T>
__device__ T warp_reduce(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 32.3.2 Warp Vote Functions

```cuda
__device__ void warp_vote_example(int predicate) {
    unsigned mask = __activemask();

    // All threads in warp agree
    if (__all_sync(mask, predicate)) {
        // All active threads have predicate true
    }

    // Any thread in warp agrees
    if (__any_sync(mask, predicate)) {
        // At least one thread has predicate true
    }

    // Ballot - get bit mask of predicate
    unsigned ballot = __ballot_sync(mask, predicate);

    // Count votes
    int votes = __popc(ballot);
}
```

### 32.3.3 Match Operations

```cuda
__device__ void match_example(int value) {
    unsigned mask = __activemask();

    // Find threads with same value
    unsigned matches = __match_any_sync(mask, value);

    // Check if all threads have same value
    int leader;
    unsigned all_match = __match_all_sync(mask, value, &leader);
}
```

## 32.4 Math Intrinsics

### 32.4.1 Fast Math Functions

```cuda
__device__ float fast_math_operations(float x, float y) {
    // Fast reciprocal
    float recip = __frcp_rn(x);

    // Fast reciprocal square root
    float rsqrt = __frsqrt_rn(x);

    // Fast exponential
    float exp_val = __expf(x);

    // Fast logarithm
    float log_val = __logf(x);

    // Fast sine/cosine
    float sin_val, cos_val;
    __sincosf(x, &sin_val, &cos_val);

    // Fast power
    float pow_val = __powf(x, y);

    return pow_val;
}
```

### 32.4.2 Integer Intrinsics

```cuda
__device__ int integer_intrinsics(int x, int y) {
    // Population count (count set bits)
    int popc = __popc(x);

    // Count leading zeros
    int clz = __clz(x);

    // Find first set bit
    int ffs = __ffs(x);

    // Bit reverse
    int brev = __brev(x);

    // Byte permute
    int perm = __byte_perm(x, y, 0x3210);

    // Funnel shift
    int funnel = __funnelshift_l(x, y, 5);

    return popc;
}
```

### 32.4.3 Extended Precision

```cuda
__device__ void extended_precision(int a, int b) {
    // 32-bit multiply producing 64-bit result
    long long result = __mul64hi(a, b);

    // Multiply-add with extended precision
    int mad_hi = __mulhi(a, b);
    int mad_lo = a * b;

    // Unsigned variants
    unsigned long long uresult = __umul64hi(a, b);
}
```

## 32.5 Memory Intrinsics

### 32.5.1 Cache Control

```cuda
__device__ void cache_operations(float* data) {
    // Load with cache hints
    float val_l1 = __ldg(data);        // Cache in L1
    float val_l2 = __ldcg(data);       // Cache in L2 only
    float val_cs = __ldcs(data);       // Streaming (bypass cache)

    // Prefetch
    __prefetch_global_L1(data);
    __prefetch_global_L2(data);
}
```

### 32.5.2 Atomic Intrinsics

```cuda
__device__ void atomic_intrinsics(int* addr, int val) {
    // Standard atomics
    int old = atomicAdd(addr, val);

    // Compare and swap
    int expected = 10;
    int desired = 20;
    atomicCAS(addr, expected, desired);

    // Atomic operations on shared memory
    __shared__ int shared_counter;
    atomicAdd(&shared_counter, 1);

    // System-wide atomic
    __threadfence_system();
    atomicAdd_system(addr, val);
}
```

## 32.6 Synchronization Intrinsics

### 32.6.1 Barriers and Fences

```cuda
__device__ void sync_primitives() {
    // Warp synchronization
    __syncwarp();
    __syncwarp(0xffffffff);  // With mask

    // Block synchronization
    __syncthreads();

    // Memory fences
    __threadfence_block();   // Block level
    __threadfence();         // Device level
    __threadfence_system();  // System level

    // Synchronize and count
    int count = __syncthreads_count(threadIdx.x < 16);

    // Synchronize and logical operations
    int all = __syncthreads_and(threadIdx.x < 16);
    int any = __syncthreads_or(threadIdx.x < 16);
}
```

## 32.7 Practical Examples

### 32.7.1 Optimized Reduction

```cuda
template<int BLOCK_SIZE>
__global__ void optimized_reduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and first reduction
    float val = (i < n) ? input[i] : 0;
    if (i + blockDim.x < n) {
        val += input[i + blockDim.x];
    }

    sdata[tid] = val;
    __syncthreads();

    // Unrolled reduction
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    // Warp reduction
    if (tid < 32) {
        val = sdata[tid];
        val = warp_reduce(val);
        if (tid == 0) output[blockIdx.x] = val;
    }
}
```

### 32.7.2 Histogram with Voting

```cuda
__global__ void histogram_voting(int* data, int* hist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        int bin = data[tid];
        unsigned mask = __activemask();

        // Find leader for each bin value
        int leader = 0;
        unsigned matches = __match_any_sync(mask, bin);

        // Count threads with same bin
        int count = __popc(matches);

        // Only leader updates histogram
        if (__ffs(matches) - 1 == threadIdx.x % 32) {
            atomicAdd(&hist[bin], count);
        }
    }
}
```

## 32.8 Performance Tips

### 32.8.1 Intrinsic Selection

```cuda
// Compiler may not optimize this
float slow_recip = 1.0f / x;

// Direct hardware instruction
float fast_recip = __frcp_rn(x);

// Trade-off: accuracy vs speed
float accurate = 1.0f / x;          // Full precision
float approx = __fdividef(1.0f, x); // Fast approximate
```

### 32.8.2 Warp Divergence Mitigation

```cuda
__device__ void handle_divergence(int condition) {
    unsigned mask = __ballot_sync(__activemask(), condition);

    if (mask == 0xffffffff) {
        // All threads take same path - no divergence
    } else if (mask == 0) {
        // No threads take path - skip
    } else {
        // Handle divergence carefully
    }
}
```

## 32.9 Exercises

1. **Warp Primitives**: Implement matrix transpose using shuffle
2. **Fast Math**: Compare performance of intrinsic vs standard math
3. **Voting Functions**: Create efficient histogram with voting
4. **Cache Control**: Optimize memory access patterns with cache hints

## 32.10 Building and Running

```bash
# Build intrinsics examples
cd build/30.cuda_advanced/32.CUDA_Intrinsics
ninja

# Run examples
./32_CUDAIntrinsics_warp_primitives
./32_CUDAIntrinsics_fast_math
./32_CUDAIntrinsics_voting

# Profile to see intrinsic usage
ncu --set full ./32_CUDAIntrinsics_warp_primitives
```

## 32.11 Key Takeaways

- Intrinsics provide direct hardware access
- Warp primitives enable efficient communication
- Fast math trades accuracy for speed
- Voting functions reduce divergence
- Cache hints optimize memory access