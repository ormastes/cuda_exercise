#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include "kernel.h"

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    launchKernel();
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    return 0;
}