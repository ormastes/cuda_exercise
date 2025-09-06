#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include "kernel.h"
#include <iostream>
#include <sstream>
#include <cstdio>

TEST_CASE("Kernel launch test", "[cuda]") {
    SECTION("launchKernel executes without errors") {
        cudaError_t error = cudaGetLastError();
        REQUIRE(error == cudaSuccess);
        
        launchKernel();
        
        error = cudaDeviceSynchronize();
        REQUIRE(error == cudaSuccess);
        
        error = cudaGetLastError();
        REQUIRE(error == cudaSuccess);
    }
    
    SECTION("CUDA device is available") {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        REQUIRE(error == cudaSuccess);
        REQUIRE(deviceCount > 0);
    }
    
    SECTION("CUDA device properties can be queried") {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, 0);
        REQUIRE(error == cudaSuccess);
        REQUIRE(prop.major > 0);
    }
}

TEST_CASE("Multiple kernel launches", "[cuda]") {
    SECTION("Multiple launches work correctly") {
        for (int i = 0; i < 5; ++i) {
            launchKernel();
            cudaError_t error = cudaDeviceSynchronize();
            REQUIRE(error == cudaSuccess);
        }
    }
}