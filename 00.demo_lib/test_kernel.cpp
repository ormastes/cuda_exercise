#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include <iostream>
#include <sstream>
#include <cstdio>

TEST(KernelLaunchTest, LaunchKernelExecutesWithoutErrors) {
        cudaError_t error = cudaGetLastError();
        EXPECT_EQ(error, cudaSuccess);
        
        launchKernel();
        
        error = cudaDeviceSynchronize();
        EXPECT_EQ(error, cudaSuccess);
        
        error = cudaGetLastError();
        EXPECT_EQ(error, cudaSuccess);
}

TEST(KernelLaunchTest, CudaDeviceIsAvailable) {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        EXPECT_EQ(error, cudaSuccess);
        EXPECT_GT(deviceCount, 0);
}

TEST(KernelLaunchTest, CudaDevicePropertiesCanBeQueried) {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, 0);
        EXPECT_EQ(error, cudaSuccess);
        EXPECT_GT(prop.major, 0);
}

TEST(MultipleKernelLaunches, MultipleLaunchesWorkCorrectly) {
        for (int i = 0; i < 5; ++i) {
            launchKernel();
            cudaError_t error = cudaDeviceSynchronize();
            EXPECT_EQ(error, cudaSuccess);
        }
}