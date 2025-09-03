#include <cstdio>
#include <cuda_runtime.h>

__device__ void myKernelDevice() {
    printf("Hello cuda device\n");
}

__global__ void myKernel() {
    printf("Hello cuda global\n");
    myKernelDevice();
}

void launchKernel() {
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
