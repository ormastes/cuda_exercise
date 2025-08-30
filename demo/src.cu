#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void myKernel() {
    printf("Hello cuda\n");
}


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    myKernel<<<1, 1>>>();
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    return 0;
}