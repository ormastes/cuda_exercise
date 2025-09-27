#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeNaive(float* d_in, float* d_out, int width, int height) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < width && yIndex < height) {
        int index_in = yIndex * width + xIndex;
        int index_out = xIndex * height + yIndex;
        d_out[index_out] = d_in[index_in];
    }
}

__global__ void transposeCoalesced(float* d_in, float* d_out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = d_in[index_in + j * width];
        }
    }

    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = yIndex * height + xIndex;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < height && (yIndex + j) < width) {
            d_out[index_out + j * height] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void transposeDiagonal(float* d_in, float* d_out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = d_in[index_in + j * width];
        }
    }

    __syncthreads();

    xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = yIndex * height + xIndex;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < height && (yIndex + j) < width) {
            d_out[index_out + j * height] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void checkTranspose(float* h_in, float* h_out, int width, int height) {
    bool correct = true;
    int errors = 0;
    const int maxErrors = 10;

    for (int y = 0; y < height && errors < maxErrors; y++) {
        for (int x = 0; x < width && errors < maxErrors; x++) {
            int idx_in = y * width + x;
            int idx_out = x * height + y;
            if (h_out[idx_out] != h_in[idx_in]) {
                if (errors == 0) {
                    printf("Transpose errors found:\n");
                }
                printf("  Error at (%d,%d): expected %.0f, got %.0f\n",
                       x, y, h_in[idx_in], h_out[idx_out]);
                correct = false;
                errors++;
            }
        }
    }

    if (correct) {
        printf("Transpose verification: PASSED\n");
    } else {
        printf("Transpose verification: FAILED (showing first %d errors)\n", errors);
    }
}

int main() {
    const int width = 2048;
    const int height = 2048;
    const int size = width * height * sizeof(float);

    printf("Matrix Transpose Performance Comparison\n");
    printf("Matrix size: %d x %d\n", width, height);
    printf("==========================================\n");

    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    for (int i = 0; i < width * height; i++) {
        h_in[i] = (float)(i + 1);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    int numIterations = 100;

    cudaMemset(d_out, 0, size);
    dim3 dimBlockNaive(TILE_DIM, TILE_DIM);
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; i++) {
        transposeNaive<<<dimGrid, dimBlockNaive>>>(d_in, d_out, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive transpose:                 %.3f ms (avg over %d runs)\n",
           milliseconds / numIterations, numIterations);
    float naiveTime = milliseconds / numIterations;

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    checkTranspose(h_in, h_out, width, height);

    cudaMemset(d_out, 0, size);
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; i++) {
        transposeCoalesced<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float coalescedTime = milliseconds / numIterations;
    printf("\nShared memory (coalesced):       %.3f ms (%.2fx speedup)\n",
           coalescedTime, naiveTime / coalescedTime);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    checkTranspose(h_in, h_out, width, height);

    cudaMemset(d_out, 0, size);
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; i++) {
        transposeDiagonal<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float diagonalTime = milliseconds / numIterations;
    printf("\nDiagonal transpose (optimized):  %.3f ms (%.2fx speedup)\n",
           diagonalTime, naiveTime / diagonalTime);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    checkTranspose(h_in, h_out, width, height);

    size_t bandwidth = 2 * size * numIterations;
    float effectiveBandwidth = bandwidth / (diagonalTime * 1e6);
    printf("\nEffective bandwidth (diagonal): %.2f GB/s\n", effectiveBandwidth);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}