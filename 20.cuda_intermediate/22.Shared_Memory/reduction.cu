#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce0(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce1(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce2(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce3(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce4(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float mySum = 0;

    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n)
            mySum += g_idata[i + blockSize];
        i += gridSize;
    }

    sdata[tid] = mySum;
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce5(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float mySum = 0;

    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n)
            mySum += g_idata[i + blockSize];
        i += gridSize;
    }

    sdata[tid] = mySum;
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockSize >= 64) smem[tid] = mySum = mySum + smem[tid + 32];
        if (blockSize >= 32) smem[tid] = mySum = mySum + smem[tid + 16];
        if (blockSize >= 16) smem[tid] = mySum = mySum + smem[tid + 8];
        if (blockSize >= 8) smem[tid] = mySum = mySum + smem[tid + 4];
        if (blockSize >= 4) smem[tid] = mySum = mySum + smem[tid + 2];
        if (blockSize >= 2) smem[tid] = mySum = mySum + smem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

float runReduction(int version, float* d_in, float* d_out, int n, int blockSize, int numBlocks) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int smemSize = blockSize * sizeof(float);

    cudaEventRecord(start);

    switch (version) {
        case 0:
            switch (blockSize) {
                case 512: reduce0<512><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 256: reduce0<256><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 128: reduce0<128><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 64:  reduce0<64><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 32:  reduce0<32><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
            }
            break;
        case 1:
            switch (blockSize) {
                case 512: reduce1<512><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 256: reduce1<256><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 128: reduce1<128><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 64:  reduce1<64><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 32:  reduce1<32><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
            }
            break;
        case 2:
            switch (blockSize) {
                case 512: reduce2<512><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 256: reduce2<256><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 128: reduce2<128><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 64:  reduce2<64><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 32:  reduce2<32><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
            }
            break;
        case 3:
            numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);
            switch (blockSize) {
                case 512: reduce3<512><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 256: reduce3<256><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 128: reduce3<128><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 64:  reduce3<64><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 32:  reduce3<32><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
            }
            break;
        case 4:
            numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);
            switch (blockSize) {
                case 512: reduce4<512><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 256: reduce4<256><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 128: reduce4<128><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 64:  reduce4<64><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 32:  reduce4<32><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
            }
            break;
        case 5:
            numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);
            switch (blockSize) {
                case 512: reduce5<512><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 256: reduce5<256><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 128: reduce5<128><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 64:  reduce5<64><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
                case 32:  reduce5<32><<<numBlocks, blockSize, smemSize>>>(d_in, d_out, n); break;
            }
            break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    const int n = 1 << 20;
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    size_t bytes = n * sizeof(float);
    size_t outputBytes = numBlocks * sizeof(float);

    printf("Sum Reduction Optimization Progress\n");
    printf("====================================\n");
    printf("Array size: %d elements\n", n);
    printf("Block size: %d threads\n", blockSize);
    printf("Grid size: %d blocks\n\n", numBlocks);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(outputBytes);

    for (int i = 0; i < n; i++) {
        h_in[i] = 1.0f;
    }

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, outputBytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const char* kernelNames[] = {
        "Interleaved addressing with divergent branching",
        "Interleaved addressing with bank conflicts",
        "Sequential addressing",
        "First add during global load",
        "Unroll last warp",
        "Completely unrolled"
    };

    printf("Kernel Performance Comparison:\n");
    printf("------------------------------\n");

    float baselineTime = 0;
    for (int version = 0; version <= 5; version++) {
        cudaMemset(d_out, 0, outputBytes);

        float* d_in_copy;
        cudaMalloc(&d_in_copy, bytes);
        cudaMemcpy(d_in_copy, d_in, bytes, cudaMemcpyDeviceToDevice);

        float time = runReduction(version, d_in_copy, d_out, n, blockSize, numBlocks);

        cudaMemcpy(h_out, d_out, outputBytes, cudaMemcpyDeviceToHost);

        float gpuSum = 0;
        int actualBlocks = (version >= 3) ? (n + blockSize * 2 - 1) / (blockSize * 2) : numBlocks;
        for (int i = 0; i < actualBlocks; i++) {
            gpuSum += h_out[i];
        }

        if (version == 0) baselineTime = time;

        printf("v%d: %-45s %7.3f ms", version, kernelNames[version], time);
        if (version > 0) {
            printf(" (%.2fx speedup)", baselineTime / time);
        }
        printf("\n");

        if (fabs(gpuSum - n) > 0.01f) {
            printf("    WARNING: Incorrect result! Expected %d, got %.0f\n", n, gpuSum);
        }

        cudaFree(d_in_copy);
    }

    printf("\nBandwidth Analysis (for v5):\n");
    printf("------------------------------\n");
    float finalTime = runReduction(5, d_in, d_out, n, blockSize, numBlocks);
    float bandwidth = bytes / (finalTime * 1e6);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}