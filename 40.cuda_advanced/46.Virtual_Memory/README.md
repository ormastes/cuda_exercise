# 41. Virtual Memory Management

## 41.1 Introduction to CUDA Virtual Memory

CUDA Virtual Memory provides advanced memory management capabilities, enabling fine-grained control over memory allocation, mapping, and access permissions.

## 41.2 Learning Objectives

- Understand CUDA virtual memory architecture
- Implement virtual address space management
- Use memory mapping and access control
- Design efficient memory hierarchies
- Optimize virtual memory performance

## 41.3 Virtual Memory Fundamentals

### 41.3.1 Virtual Address Space

```cpp
#include <cuda_runtime.h>

class VirtualMemoryManager {
private:
    CUmemGenericAllocationHandle allocHandle;
    CUdeviceptr virtualPtr;
    size_t allocationSize;
    size_t granularity;

public:
    bool initializeVirtualMemory(size_t size) {
        allocationSize = size;

        // Get memory granularity
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;

        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        // Round up to granularity
        size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

        // Create virtual address reservation
        cuMemAddressReserve(&virtualPtr, alignedSize, 0, 0, 0);

        // Create physical allocation
        cuMemCreate(&allocHandle, alignedSize, &prop, 0);

        // Map virtual to physical
        cuMemMap(virtualPtr, alignedSize, 0, allocHandle, 0);

        // Set access permissions
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = 0;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        cuMemSetAccess(virtualPtr, alignedSize, &accessDesc, 1);

        printf("Virtual Memory: Allocated %zu bytes at %p\n", alignedSize, (void*)virtualPtr);
        return true;
    }

    void* getVirtualPtr() const { return (void*)virtualPtr; }
    size_t getSize() const { return allocationSize; }

    ~VirtualMemoryManager() {
        if (virtualPtr) {
            cuMemUnmap(virtualPtr, allocationSize);
            cuMemRelease(allocHandle);
            cuMemAddressFree(virtualPtr, allocationSize);
        }
    }
};
```

### 41.3.2 Memory Mapping and Unmapping

```cpp
class AdvancedVirtualMemory {
private:
    struct MemoryRegion {
        CUdeviceptr basePtr;
        size_t size;
        CUmemGenericAllocationHandle handle;
        bool isMapped;
        CUmemAccessFlags accessFlags;
    };

    std::vector<MemoryRegion> regions;
    CUdeviceptr reservedBase;
    size_t totalReservedSize;
    size_t granularity;

public:
    bool initializeAddressSpace(size_t totalSize) {
        // Get granularity for alignment
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;

        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        totalReservedSize = ((totalSize + granularity - 1) / granularity) * granularity;

        // Reserve large virtual address space
        CUresult result = cuMemAddressReserve(&reservedBase, totalReservedSize, 0, 0, 0);
        if (result != CUDA_SUCCESS) {
            printf("Failed to reserve virtual address space\n");
            return false;
        }

        printf("Reserved virtual address space: %zu bytes at %p\n",
               totalReservedSize, (void*)reservedBase);
        return true;
    }

    void* allocateRegion(size_t size, CUmemAccessFlags accessFlags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
        size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

        // Find available space in reserved region
        CUdeviceptr regionPtr = findAvailableSpace(alignedSize);
        if (!regionPtr) {
            printf("No available space for allocation of %zu bytes\n", alignedSize);
            return nullptr;
        }

        // Create physical allocation
        CUmemGenericAllocationHandle allocHandle;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;

        CUresult result = cuMemCreate(&allocHandle, alignedSize, &prop, 0);
        if (result != CUDA_SUCCESS) {
            printf("Failed to create physical allocation\n");
            return nullptr;
        }

        // Map virtual to physical
        result = cuMemMap(regionPtr, alignedSize, 0, allocHandle, 0);
        if (result != CUDA_SUCCESS) {
            printf("Failed to map virtual memory\n");
            cuMemRelease(allocHandle);
            return nullptr;
        }

        // Set access permissions
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = 0;
        accessDesc.flags = accessFlags;

        result = cuMemSetAccess(regionPtr, alignedSize, &accessDesc, 1);
        if (result != CUDA_SUCCESS) {
            printf("Failed to set access permissions\n");
            cuMemUnmap(regionPtr, alignedSize);
            cuMemRelease(allocHandle);
            return nullptr;
        }

        // Record the region
        MemoryRegion region;
        region.basePtr = regionPtr;
        region.size = alignedSize;
        region.handle = allocHandle;
        region.isMapped = true;
        region.accessFlags = accessFlags;
        regions.push_back(region);

        printf("Allocated region: %zu bytes at %p\n", alignedSize, (void*)regionPtr);
        return (void*)regionPtr;
    }

    bool deallocateRegion(void* ptr) {
        CUdeviceptr devicePtr = (CUdeviceptr)ptr;

        for (auto it = regions.begin(); it != regions.end(); ++it) {
            if (it->basePtr == devicePtr) {
                if (it->isMapped) {
                    cuMemUnmap(it->basePtr, it->size);
                    cuMemRelease(it->handle);
                }

                printf("Deallocated region at %p\n", ptr);
                regions.erase(it);
                return true;
            }
        }

        printf("Region not found for deallocation: %p\n", ptr);
        return false;
    }

    bool changeAccess(void* ptr, CUmemAccessFlags newFlags) {
        CUdeviceptr devicePtr = (CUdeviceptr)ptr;

        for (auto& region : regions) {
            if (region.basePtr == devicePtr) {
                CUmemAccessDesc accessDesc = {};
                accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                accessDesc.location.id = 0;
                accessDesc.flags = newFlags;

                CUresult result = cuMemSetAccess(region.basePtr, region.size, &accessDesc, 1);
                if (result == CUDA_SUCCESS) {
                    region.accessFlags = newFlags;
                    printf("Changed access for region %p\n", ptr);
                    return true;
                }
            }
        }

        return false;
    }

private:
    CUdeviceptr findAvailableSpace(size_t size) {
        // Simple linear search for available space
        std::sort(regions.begin(), regions.end(),
                 [](const MemoryRegion& a, const MemoryRegion& b) {
                     return a.basePtr < b.basePtr;
                 });

        CUdeviceptr currentPos = reservedBase;

        for (const auto& region : regions) {
            if (region.basePtr - currentPos >= size) {
                return currentPos;
            }
            currentPos = region.basePtr + region.size;
        }

        // Check if there's space at the end
        if ((reservedBase + totalReservedSize) - currentPos >= size) {
            return currentPos;
        }

        return 0;  // No space available
    }

public:
    ~AdvancedVirtualMemory() {
        for (auto& region : regions) {
            if (region.isMapped) {
                cuMemUnmap(region.basePtr, region.size);
                cuMemRelease(region.handle);
            }
        }

        if (reservedBase) {
            cuMemAddressFree(reservedBase, totalReservedSize);
        }
    }
};
```

## 41.4 Advanced Virtual Memory Patterns

### 41.4.1 Memory-Mapped Files

```cpp
class MemoryMappedFile {
private:
    int fileDescriptor;
    void* hostMapping;
    CUdeviceptr deviceMapping;
    size_t fileSize;
    CUmemGenericAllocationHandle allocHandle;

public:
    bool mapFile(const std::string& filename, bool readOnly = true) {
        // Open file
        int flags = readOnly ? O_RDONLY : O_RDWR;
        fileDescriptor = open(filename.c_str(), flags);
        if (fileDescriptor == -1) {
            perror("Failed to open file");
            return false;
        }

        // Get file size
        struct stat fileStat;
        if (fstat(fileDescriptor, &fileStat) == -1) {
            perror("Failed to get file size");
            close(fileDescriptor);
            return false;
        }
        fileSize = fileStat.st_size;

        // Map file to host memory
        int protection = readOnly ? PROT_READ : (PROT_READ | PROT_WRITE);
        hostMapping = mmap(nullptr, fileSize, protection, MAP_SHARED, fileDescriptor, 0);
        if (hostMapping == MAP_FAILED) {
            perror("Failed to map file to host memory");
            close(fileDescriptor);
            return false;
        }

        // Create CUDA external memory from file mapping
        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        externalMemoryHandleDesc.handle.fd = fileDescriptor;
        externalMemoryHandleDesc.size = fileSize;

        cudaExternalMemory_t externalMemory;
        cudaError_t result = cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc);
        if (result != cudaSuccess) {
            printf("Failed to import external memory: %s\n", cudaGetErrorString(result));
            munmap(hostMapping, fileSize);
            close(fileDescriptor);
            return false;
        }

        // Map external memory to device
        cudaExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = 0;
        bufferDesc.size = fileSize;
        bufferDesc.flags = 0;

        result = cudaExternalMemoryGetMappedBuffer(&deviceMapping, externalMemory, &bufferDesc);
        if (result != cudaSuccess) {
            printf("Failed to map external memory to device: %s\n", cudaGetErrorString(result));
            cudaDestroyExternalMemory(externalMemory);
            munmap(hostMapping, fileSize);
            close(fileDescriptor);
            return false;
        }

        printf("Memory-mapped file: %s (%zu bytes)\n", filename.c_str(), fileSize);
        printf("Host mapping: %p, Device mapping: %p\n", hostMapping, (void*)deviceMapping);

        return true;
    }

    void* getHostPtr() const { return hostMapping; }
    void* getDevicePtr() const { return (void*)deviceMapping; }
    size_t getSize() const { return fileSize; }

    void sync() {
        msync(hostMapping, fileSize, MS_SYNC);
        cudaDeviceSynchronize();
    }

    ~MemoryMappedFile() {
        if (hostMapping != MAP_FAILED) {
            munmap(hostMapping, fileSize);
        }
        if (fileDescriptor != -1) {
            close(fileDescriptor);
        }
    }
};
```

### 41.4.2 Sparse Memory Management

```cpp
class SparseMemoryManager {
private:
    struct SparseRegion {
        CUdeviceptr virtualBase;
        size_t totalSize;
        size_t chunkSize;
        std::vector<bool> allocatedChunks;
        std::vector<CUmemGenericAllocationHandle> chunkHandles;
    };

    std::vector<SparseRegion> sparseRegions;
    size_t granularity;

public:
    bool initializeSparseRegion(size_t totalSize, size_t chunkSize) {
        // Get memory granularity
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;

        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        // Align sizes to granularity
        totalSize = ((totalSize + granularity - 1) / granularity) * granularity;
        chunkSize = ((chunkSize + granularity - 1) / granularity) * granularity;

        SparseRegion region;
        region.totalSize = totalSize;
        region.chunkSize = chunkSize;

        // Reserve virtual address space
        CUresult result = cuMemAddressReserve(&region.virtualBase, totalSize, 0, 0, 0);
        if (result != CUDA_SUCCESS) {
            printf("Failed to reserve sparse virtual address space\n");
            return false;
        }

        size_t numChunks = totalSize / chunkSize;
        region.allocatedChunks.resize(numChunks, false);
        region.chunkHandles.resize(numChunks);

        sparseRegions.push_back(region);

        printf("Sparse region created: %zu bytes (%zu chunks) at %p\n",
               totalSize, numChunks, (void*)region.virtualBase);

        return true;
    }

    void* allocateChunk(size_t regionIndex, size_t chunkIndex) {
        if (regionIndex >= sparseRegions.size()) {
            printf("Invalid region index: %zu\n", regionIndex);
            return nullptr;
        }

        SparseRegion& region = sparseRegions[regionIndex];

        if (chunkIndex >= region.allocatedChunks.size()) {
            printf("Invalid chunk index: %zu\n", chunkIndex);
            return nullptr;
        }

        if (region.allocatedChunks[chunkIndex]) {
            printf("Chunk %zu already allocated\n", chunkIndex);
            return (void*)(region.virtualBase + chunkIndex * region.chunkSize);
        }

        // Create physical allocation for chunk
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;

        CUresult result = cuMemCreate(&region.chunkHandles[chunkIndex], region.chunkSize, &prop, 0);
        if (result != CUDA_SUCCESS) {
            printf("Failed to create chunk allocation\n");
            return nullptr;
        }

        // Map chunk to virtual address
        CUdeviceptr chunkPtr = region.virtualBase + chunkIndex * region.chunkSize;
        result = cuMemMap(chunkPtr, region.chunkSize, 0, region.chunkHandles[chunkIndex], 0);
        if (result != CUDA_SUCCESS) {
            printf("Failed to map chunk\n");
            cuMemRelease(region.chunkHandles[chunkIndex]);
            return nullptr;
        }

        // Set access permissions
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = 0;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        result = cuMemSetAccess(chunkPtr, region.chunkSize, &accessDesc, 1);
        if (result != CUDA_SUCCESS) {
            printf("Failed to set chunk access\n");
            cuMemUnmap(chunkPtr, region.chunkSize);
            cuMemRelease(region.chunkHandles[chunkIndex]);
            return nullptr;
        }

        region.allocatedChunks[chunkIndex] = true;

        printf("Allocated chunk %zu in region %zu at %p\n",
               chunkIndex, regionIndex, (void*)chunkPtr);

        return (void*)chunkPtr;
    }

    bool deallocateChunk(size_t regionIndex, size_t chunkIndex) {
        if (regionIndex >= sparseRegions.size() ||
            chunkIndex >= sparseRegions[regionIndex].allocatedChunks.size()) {
            return false;
        }

        SparseRegion& region = sparseRegions[regionIndex];

        if (!region.allocatedChunks[chunkIndex]) {
            return false;  // Already deallocated
        }

        CUdeviceptr chunkPtr = region.virtualBase + chunkIndex * region.chunkSize;

        cuMemUnmap(chunkPtr, region.chunkSize);
        cuMemRelease(region.chunkHandles[chunkIndex]);

        region.allocatedChunks[chunkIndex] = false;

        printf("Deallocated chunk %zu in region %zu\n", chunkIndex, regionIndex);
        return true;
    }

    void printRegionStatus(size_t regionIndex) {
        if (regionIndex >= sparseRegions.size()) return;

        const SparseRegion& region = sparseRegions[regionIndex];

        printf("Sparse Region %zu Status:\n", regionIndex);
        printf("  Total size: %zu bytes\n", region.totalSize);
        printf("  Chunk size: %zu bytes\n", region.chunkSize);
        printf("  Total chunks: %zu\n", region.allocatedChunks.size());

        size_t allocatedCount = 0;
        for (bool allocated : region.allocatedChunks) {
            if (allocated) allocatedCount++;
        }

        printf("  Allocated chunks: %zu\n", allocatedCount);
        printf("  Memory usage: %.1f%%\n",
               100.0 * allocatedCount / region.allocatedChunks.size());
    }

    ~SparseMemoryManager() {
        for (auto& region : sparseRegions) {
            for (size_t i = 0; i < region.allocatedChunks.size(); ++i) {
                if (region.allocatedChunks[i]) {
                    CUdeviceptr chunkPtr = region.virtualBase + i * region.chunkSize;
                    cuMemUnmap(chunkPtr, region.chunkSize);
                    cuMemRelease(region.chunkHandles[i]);
                }
            }
            cuMemAddressFree(region.virtualBase, region.totalSize);
        }
    }
};
```

## 41.5 Exercises

1. **Basic Virtual Memory**: Implement virtual address space management
2. **Memory Mapping**: Create memory-mapped file access
3. **Sparse Allocation**: Build sparse memory management system
4. **Access Control**: Implement dynamic permission changes
5. **Performance Study**: Compare virtual vs traditional memory allocation

## 41.6 Building and Running

```bash
# Build with virtual memory examples
cd build/30.cuda_advanced/41.Virtual_Memory
ninja

# Run examples
./41_VirtualMemory_basic
./41_VirtualMemory_sparse
./41_VirtualMemory_mapped_files

# Profile virtual memory operations
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./41_VirtualMemory_sparse

# Analyze memory access patterns
nsys profile --stats=true -t cuda ./41_VirtualMemory_basic
```

## 41.7 Key Takeaways

- Virtual memory provides fine-grained control over memory management
- Sparse allocation enables efficient memory usage for large address spaces
- Memory mapping allows direct file access from GPU
- Access control mechanisms provide security and debugging capabilities
- Virtual memory is essential for advanced memory hierarchies
- Understanding virtual memory concepts is crucial for modern GPU programming
