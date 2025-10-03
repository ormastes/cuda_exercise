# 40. CUDA Inter-Process Communication (IPC)

## 40.1 Introduction to CUDA IPC

CUDA IPC enables sharing GPU memory and events between different processes, allowing for complex multi-process GPU applications.

## 40.2 Learning Objectives

- Understand CUDA IPC mechanisms and use cases
- Implement shared memory across processes
- Use IPC events for process synchronization
- Design multi-process GPU applications
- Optimize IPC performance and handle edge cases

## 40.3 CUDA IPC Fundamentals

### 40.3.1 IPC Memory Handles

```cpp
#include <cuda_runtime.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

struct IPCMemoryInfo {
    cudaIpcMemHandle_t memHandle;
    size_t size;
    pid_t creatorPid;
    int shmId;
};

class IPCMemoryManager {
private:
    void* devicePtr;
    size_t memorySize;
    cudaIpcMemHandle_t memHandle;
    bool isCreator;
    int shmId;

public:
    // Creator process
    bool createSharedMemory(size_t size) {
        memorySize = size;
        isCreator = true;

        // Allocate device memory
        cudaError_t err = cudaMalloc(&devicePtr, size);
        if (err != cudaSuccess) {
            printf("Failed to allocate device memory: %s\n", cudaGetErrorString(err));
            return false;
        }

        // Get IPC handle
        err = cudaIpcGetMemHandle(&memHandle, devicePtr);
        if (err != cudaSuccess) {
            printf("Failed to get IPC handle: %s\n", cudaGetErrorString(err));
            cudaFree(devicePtr);
            return false;
        }

        // Share handle via system V shared memory
        key_t key = ftok("/tmp", 'C');
        shmId = shmget(key, sizeof(IPCMemoryInfo), IPC_CREAT | 0666);
        if (shmId == -1) {
            perror("shmget failed");
            cudaFree(devicePtr);
            return false;
        }

        IPCMemoryInfo* sharedInfo = (IPCMemoryInfo*)shmat(shmId, NULL, 0);
        if (sharedInfo == (IPCMemoryInfo*)-1) {
            perror("shmat failed");
            cudaFree(devicePtr);
            return false;
        }

        // Store IPC information
        sharedInfo->memHandle = memHandle;
        sharedInfo->size = size;
        sharedInfo->creatorPid = getpid();
        sharedInfo->shmId = shmId;

        printf("Creator: Shared memory created with handle\n");
        printf("Creator PID: %d, Shared memory ID: %d\n", getpid(), shmId);

        shmdt(sharedInfo);
        return true;
    }

    // Consumer process
    bool attachToSharedMemory() {
        isCreator = false;

        // Get shared handle
        key_t key = ftok("/tmp", 'C');
        shmId = shmget(key, sizeof(IPCMemoryInfo), 0666);
        if (shmId == -1) {
            perror("shmget failed in consumer");
            return false;
        }

        IPCMemoryInfo* sharedInfo = (IPCMemoryInfo*)shmat(shmId, NULL, 0);
        if (sharedInfo == (IPCMemoryInfo*)-1) {
            perror("shmat failed in consumer");
            return false;
        }

        memHandle = sharedInfo->memHandle;
        memorySize = sharedInfo->size;

        printf("Consumer: Found shared memory from PID %d\n", sharedInfo->creatorPid);

        shmdt(sharedInfo);

        // Open IPC memory handle
        cudaError_t err = cudaIpcOpenMemHandle(&devicePtr, memHandle,
                                              cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            printf("Failed to open IPC handle: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("Consumer: Successfully attached to shared GPU memory\n");
        return true;
    }

    void* getDevicePtr() const { return devicePtr; }
    size_t getSize() const { return memorySize; }

    ~IPCMemoryManager() {
        if (isCreator) {
            if (devicePtr) {
                cudaFree(devicePtr);
            }
            // Clean up shared memory
            shmctl(shmId, IPC_RMID, NULL);
            printf("Creator: Cleaned up shared memory\n");
        } else {
            if (devicePtr) {
                cudaIpcCloseMemHandle(devicePtr);
            }
            printf("Consumer: Closed IPC memory handle\n");
        }
    }
};
```

### 40.3.2 IPC Event Handles

```cpp
class IPCEventManager {
private:
    cudaEvent_t event;
    cudaIpcEventHandle_t eventHandle;
    bool isCreator;
    int shmId;

    struct IPCEventInfo {
        cudaIpcEventHandle_t eventHandle;
        pid_t creatorPid;
        bool eventReady;
    };

public:
    // Creator process
    bool createSharedEvent() {
        isCreator = true;

        // Create CUDA event
        cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess);
        if (err != cudaSuccess) {
            printf("Failed to create event: %s\n", cudaGetErrorString(err));
            return false;
        }

        // Get IPC event handle
        err = cudaIpcGetEventHandle(&eventHandle, event);
        if (err != cudaSuccess) {
            printf("Failed to get IPC event handle: %s\n", cudaGetErrorString(err));
            cudaEventDestroy(event);
            return false;
        }

        // Share event handle via shared memory
        key_t key = ftok("/tmp", 'E');
        shmId = shmget(key, sizeof(IPCEventInfo), IPC_CREAT | 0666);
        if (shmId == -1) {
            perror("shmget failed for event");
            cudaEventDestroy(event);
            return false;
        }

        IPCEventInfo* eventInfo = (IPCEventInfo*)shmat(shmId, NULL, 0);
        if (eventInfo == (IPCEventInfo*)-1) {
            perror("shmat failed for event");
            cudaEventDestroy(event);
            return false;
        }

        eventInfo->eventHandle = eventHandle;
        eventInfo->creatorPid = getpid();
        eventInfo->eventReady = true;

        printf("Creator: Shared event created\n");

        shmdt(eventInfo);
        return true;
    }

    // Consumer process
    bool attachToSharedEvent() {
        isCreator = false;

        // Get shared event handle
        key_t key = ftok("/tmp", 'E');
        shmId = shmget(key, sizeof(IPCEventInfo), 0666);
        if (shmId == -1) {
            perror("shmget failed for event consumer");
            return false;
        }

        IPCEventInfo* eventInfo = (IPCEventInfo*)shmat(shmId, NULL, 0);
        if (eventInfo == (IPCEventInfo*)-1) {
            perror("shmat failed for event consumer");
            return false;
        }

        // Wait for event to be ready
        while (!eventInfo->eventReady) {
            usleep(1000);  // Wait 1ms
        }

        eventHandle = eventInfo->eventHandle;
        printf("Consumer: Found shared event from PID %d\n", eventInfo->creatorPid);

        shmdt(eventInfo);

        // Open IPC event handle
        cudaError_t err = cudaIpcOpenEventHandle(&event, eventHandle);
        if (err != cudaSuccess) {
            printf("Failed to open IPC event handle: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("Consumer: Successfully attached to shared event\n");
        return true;
    }

    void recordEvent(cudaStream_t stream = 0) {
        cudaEventRecord(event, stream);
    }

    void waitForEvent(cudaStream_t stream = 0) {
        cudaStreamWaitEvent(stream, event, 0);
    }

    bool isEventComplete() {
        return cudaEventQuery(event) == cudaSuccess;
    }

    cudaEvent_t getEvent() const { return event; }

    ~IPCEventManager() {
        if (isCreator) {
            if (event) {
                cudaEventDestroy(event);
            }
            shmctl(shmId, IPC_RMID, NULL);
            printf("Creator: Cleaned up shared event\n");
        } else {
            if (event) {
                // Consumer doesn't destroy the event, just closes handle
                printf("Consumer: Closed IPC event handle\n");
            }
        }
    }
};
```

## 40.4 Multi-Process Patterns

### 40.4.1 Producer-Consumer Pattern

```cpp
class IPCProducerConsumer {
private:
    IPCMemoryManager memManager;
    IPCEventManager eventManager;
    bool isProducer;

    struct SharedBuffer {
        float* data;
        volatile int writeIndex;
        volatile int readIndex;
        volatile int count;
        static const int BUFFER_SIZE = 1024;
    };

public:
    // Producer setup
    bool initializeProducer() {
        isProducer = true;

        // Create shared memory for buffer
        if (!memManager.createSharedMemory(sizeof(SharedBuffer))) {
            return false;
        }

        // Create shared event for synchronization
        if (!eventManager.createSharedEvent()) {
            return false;
        }

        // Initialize shared buffer
        SharedBuffer* buffer = (SharedBuffer*)memManager.getDevicePtr();

        // Initialize on host first
        SharedBuffer hostBuffer = {0};
        hostBuffer.writeIndex = 0;
        hostBuffer.readIndex = 0;
        hostBuffer.count = 0;

        cudaMemcpy(buffer, &hostBuffer, sizeof(SharedBuffer), cudaMemcpyHostToDevice);

        printf("Producer: Initialized shared buffer\n");
        return true;
    }

    // Consumer setup
    bool initializeConsumer() {
        isProducer = false;

        // Attach to shared memory
        if (!memManager.attachToSharedMemory()) {
            return false;
        }

        // Attach to shared event
        if (!eventManager.attachToSharedEvent()) {
            return false;
        }

        printf("Consumer: Attached to shared resources\n");
        return true;
    }

    void produce(const std::vector<float>& data) {
        if (!isProducer) return;

        SharedBuffer* buffer = (SharedBuffer*)memManager.getDevicePtr();

        // Launch producer kernel
        dim3 block(256);
        dim3 grid((data.size() + block.x - 1) / block.x);

        float* d_input;
        cudaMalloc(&d_input, data.size() * sizeof(float));
        cudaMemcpy(d_input, data.data(), data.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        producerKernel<<<grid, block>>>(buffer, d_input, data.size());

        // Record event when production is complete
        eventManager.recordEvent();

        cudaFree(d_input);
        printf("Producer: Produced %zu items\n", data.size());
    }

    std::vector<float> consume(int maxItems) {
        if (isProducer) return {};

        SharedBuffer* buffer = (SharedBuffer*)memManager.getDevicePtr();

        // Wait for producer to signal data availability
        eventManager.waitForEvent();

        // Launch consumer kernel
        float* d_output;
        cudaMalloc(&d_output, maxItems * sizeof(float));

        dim3 block(256);
        dim3 grid((maxItems + block.x - 1) / block.x);

        consumerKernel<<<grid, block>>>(buffer, d_output, maxItems);

        cudaDeviceSynchronize();

        // Copy results back to host
        std::vector<float> result(maxItems);
        cudaMemcpy(result.data(), d_output, maxItems * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_output);

        printf("Consumer: Consumed up to %d items\n", maxItems);
        return result;
    }
};

// Producer kernel
__global__ void producerKernel(SharedBuffer* buffer, float* input, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        // Atomic operations for thread-safe buffer management
        int writePos = atomicAdd(&buffer->writeIndex, 1);

        if (writePos < SharedBuffer::BUFFER_SIZE) {
            buffer->data[writePos % SharedBuffer::BUFFER_SIZE] = input[idx];
            atomicAdd(&buffer->count, 1);
        }
    }
}

// Consumer kernel
__global__ void consumerKernel(SharedBuffer* buffer, float* output, int maxCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < maxCount) {
        if (buffer->count > 0) {
            int readPos = atomicAdd(&buffer->readIndex, 1);

            if (readPos < buffer->writeIndex) {
                output[idx] = buffer->data[readPos % SharedBuffer::BUFFER_SIZE];
                atomicSub(&buffer->count, 1);
            } else {
                output[idx] = 0.0f;  // No data available
            }
        } else {
            output[idx] = 0.0f;
        }
    }
}
```

### 40.4.2 Distributed Computing Pattern

```cpp
class IPCDistributedCompute {
private:
    struct WorkItem {
        int taskId;
        float* inputData;
        float* outputData;
        int dataSize;
        volatile int status;  // 0=pending, 1=in_progress, 2=completed
    };

    struct WorkQueue {
        WorkItem tasks[MAX_TASKS];
        volatile int head;
        volatile int tail;
        volatile int count;
        static const int MAX_TASKS = 256;
    };

    IPCMemoryManager workQueueMem;
    IPCEventManager workAvailableEvent;
    IPCEventManager workCompleteEvent;
    bool isMaster;

public:
    // Master process setup
    bool initializeMaster() {
        isMaster = true;

        // Create shared work queue
        if (!workQueueMem.createSharedMemory(sizeof(WorkQueue))) {
            return false;
        }

        // Initialize work queue
        WorkQueue* queue = (WorkQueue*)workQueueMem.getDevicePtr();
        WorkQueue hostQueue = {0};
        hostQueue.head = 0;
        hostQueue.tail = 0;
        hostQueue.count = 0;

        cudaMemcpy(queue, &hostQueue, sizeof(WorkQueue), cudaMemcpyHostToDevice);

        printf("Master: Initialized distributed work queue\n");
        return true;
    }

    // Worker process setup
    bool initializeWorker() {
        isMaster = false;

        // Attach to shared work queue
        if (!workQueueMem.attachToSharedMemory()) {
            return false;
        }

        printf("Worker: Attached to distributed work queue\n");
        return true;
    }

    // Master: Submit work
    void submitWork(const std::vector<float>& inputData, int taskId) {
        if (!isMaster) return;

        WorkQueue* queue = (WorkQueue*)workQueueMem.getDevicePtr();

        // Allocate device memory for this task
        float* d_input, *d_output;
        cudaMalloc(&d_input, inputData.size() * sizeof(float));
        cudaMalloc(&d_output, inputData.size() * sizeof(float));

        cudaMemcpy(d_input, inputData.data(),
                   inputData.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Add work item to queue
        submitWorkKernel<<<1, 1>>>(queue, taskId, d_input, d_output,
                                   inputData.size());

        printf("Master: Submitted task %d with %zu elements\n",
               taskId, inputData.size());
    }

    // Worker: Process work
    void processWork() {
        if (isMaster) return;

        WorkQueue* queue = (WorkQueue*)workQueueMem.getDevicePtr();

        // Continuously look for work
        while (true) {
            // Try to get work item
            int taskId = getWorkKernel<<<1, 1>>>(queue);

            cudaDeviceSynchronize();

            if (taskId >= 0) {
                // Process the work item
                printf("Worker: Processing task %d\n", taskId);

                // Launch processing kernel
                processTaskKernel<<<256, 256>>>(queue, taskId);

                // Mark task as complete
                completeWorkKernel<<<1, 1>>>(queue, taskId);

                printf("Worker: Completed task %d\n", taskId);
            } else {
                // No work available, sleep briefly
                usleep(1000);
            }
        }
    }
};

// Kernels for distributed computing
__global__ void submitWorkKernel(WorkQueue* queue, int taskId,
                                float* input, float* output, int size) {
    // Add work item atomically
    int pos = atomicAdd(&queue->tail, 1) % WorkQueue::MAX_TASKS;

    queue->tasks[pos].taskId = taskId;
    queue->tasks[pos].inputData = input;
    queue->tasks[pos].outputData = output;
    queue->tasks[pos].dataSize = size;
    queue->tasks[pos].status = 0;  // pending

    atomicAdd(&queue->count, 1);
}

__global__ void getWorkKernel(WorkQueue* queue) {
    if (queue->count > 0) {
        int pos = atomicAdd(&queue->head, 1) % WorkQueue::MAX_TASKS;

        if (queue->tasks[pos].status == 0) {
            queue->tasks[pos].status = 1;  // in progress
            // Return task ID somehow (simplified)
            return queue->tasks[pos].taskId;
        }
    }
    return -1;  // No work available
}

__global__ void processTaskKernel(WorkQueue* queue, int taskId) {
    // Find the task and process it
    for (int i = 0; i < WorkQueue::MAX_TASKS; i++) {
        if (queue->tasks[i].taskId == taskId && queue->tasks[i].status == 1) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int size = queue->tasks[i].dataSize;

            if (idx < size) {
                // Example processing: square root
                float input = queue->tasks[i].inputData[idx];
                queue->tasks[i].outputData[idx] = sqrtf(input);
            }
            break;
        }
    }
}

__global__ void completeWorkKernel(WorkQueue* queue, int taskId) {
    for (int i = 0; i < WorkQueue::MAX_TASKS; i++) {
        if (queue->tasks[i].taskId == taskId) {
            queue->tasks[i].status = 2;  // completed
            atomicSub(&queue->count, 1);
            break;
        }
    }
}
```

## 40.5 Advanced IPC Techniques

### 40.5.1 IPC Memory Pools

```cpp
class IPCMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool inUse;
        cudaIpcMemHandle_t handle;
    };

    std::vector<MemoryBlock> blocks;
    std::mutex poolMutex;
    size_t totalSize;
    size_t blockSize;
    int blockCount;

public:
    IPCMemoryPool(size_t blockSz, int numBlocks)
        : blockSize(blockSz), blockCount(numBlocks) {
        totalSize = blockSize * blockCount;
        blocks.resize(blockCount);

        initializePool();
    }

    bool initializePool() {
        // Allocate one large chunk of memory
        void* basePtr;
        cudaError_t err = cudaMalloc(&basePtr, totalSize);
        if (err != cudaSuccess) {
            printf("Failed to allocate pool memory: %s\n", cudaGetErrorString(err));
            return false;
        }

        // Create IPC handle for the base pointer
        cudaIpcMemHandle_t baseHandle;
        err = cudaIpcGetMemHandle(&baseHandle, basePtr);
        if (err != cudaSuccess) {
            printf("Failed to get IPC handle for pool: %s\n", cudaGetErrorString(err));
            cudaFree(basePtr);
            return false;
        }

        // Initialize individual blocks
        for (int i = 0; i < blockCount; i++) {
            blocks[i].ptr = (char*)basePtr + i * blockSize;
            blocks[i].size = blockSize;
            blocks[i].inUse = false;
            blocks[i].handle = baseHandle;  // All blocks share the same handle
        }

        printf("IPC Memory Pool: Initialized %d blocks of %zu bytes each\n",
               blockCount, blockSize);
        return true;
    }

    void* allocateBlock() {
        std::lock_guard<std::mutex> lock(poolMutex);

        for (auto& block : blocks) {
            if (!block.inUse) {
                block.inUse = true;
                printf("IPC Pool: Allocated block %p\n", block.ptr);
                return block.ptr;
            }
        }

        printf("IPC Pool: No free blocks available\n");
        return nullptr;
    }

    void deallocateBlock(void* ptr) {
        std::lock_guard<std::mutex> lock(poolMutex);

        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.inUse = false;
                printf("IPC Pool: Deallocated block %p\n", ptr);
                return;
            }
        }

        printf("IPC Pool: Warning - tried to deallocate unknown block %p\n", ptr);
    }

    cudaIpcMemHandle_t getIPCHandle() const {
        return blocks[0].handle;  // All blocks share the same handle
    }

    size_t getBlockSize() const { return blockSize; }
    int getAvailableBlocks() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(poolMutex));
        int count = 0;
        for (const auto& block : blocks) {
            if (!block.inUse) count++;
        }
        return count;
    }
};
```

### 40.5.2 IPC Synchronization Primitives

```cpp
class IPCSemaphore {
private:
    struct SemaphoreData {
        volatile int count;
        volatile int maxCount;
        cudaEvent_t signalEvent;
    };

    IPCMemoryManager semMem;
    IPCEventManager signalEvent;
    bool isCreator;

public:
    bool initialize(int initialCount, int maxCount, bool creator = true) {
        isCreator = creator;

        if (creator) {
            // Create shared semaphore data
            if (!semMem.createSharedMemory(sizeof(SemaphoreData))) {
                return false;
            }

            if (!signalEvent.createSharedEvent()) {
                return false;
            }

            // Initialize semaphore
            SemaphoreData hostSem;
            hostSem.count = initialCount;
            hostSem.maxCount = maxCount;

            SemaphoreData* semData = (SemaphoreData*)semMem.getDevicePtr();
            cudaMemcpy(semData, &hostSem, sizeof(SemaphoreData),
                       cudaMemcpyHostToDevice);

            printf("IPC Semaphore: Created with count %d, max %d\n",
                   initialCount, maxCount);
        } else {
            // Attach to existing semaphore
            if (!semMem.attachToSharedMemory()) {
                return false;
            }

            if (!signalEvent.attachToSharedEvent()) {
                return false;
            }

            printf("IPC Semaphore: Attached to shared semaphore\n");
        }

        return true;
    }

    void acquire() {
        SemaphoreData* semData = (SemaphoreData*)semMem.getDevicePtr();

        // Launch acquire kernel
        acquireSemaphoreKernel<<<1, 1>>>(semData);
        cudaDeviceSynchronize();

        printf("IPC Semaphore: Acquired\n");
    }

    void release() {
        SemaphoreData* semData = (SemaphoreData*)semMem.getDevicePtr();

        // Launch release kernel
        releaseSemaphoreKernel<<<1, 1>>>(semData);

        // Signal other processes
        signalEvent.recordEvent();

        printf("IPC Semaphore: Released\n");
    }

    bool tryAcquire() {
        SemaphoreData* semData = (SemaphoreData*)semMem.getDevicePtr();

        // Launch try acquire kernel
        int* result;
        cudaMalloc(&result, sizeof(int));
        *result = 0;

        tryAcquireSemaphoreKernel<<<1, 1>>>(semData, result);
        cudaDeviceSynchronize();

        int hostResult;
        cudaMemcpy(&hostResult, result, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(result);

        return hostResult == 1;
    }
};

// Semaphore kernels
__global__ void acquireSemaphoreKernel(SemaphoreData* sem) {
    // Busy wait for available count
    while (atomicAdd(&sem->count, -1) <= 0) {
        atomicAdd(&sem->count, 1);  // Restore count
        __nanosleep(1000);  // Brief pause
    }
}

__global__ void releaseSemaphoreKernel(SemaphoreData* sem) {
    int oldCount = atomicAdd(&sem->count, 1);
    if (oldCount >= sem->maxCount) {
        atomicAdd(&sem->count, -1);  // Don't exceed max
    }
}

__global__ void tryAcquireSemaphoreKernel(SemaphoreData* sem, int* result) {
    int oldCount = atomicAdd(&sem->count, -1);
    if (oldCount > 0) {
        *result = 1;  // Success
    } else {
        atomicAdd(&sem->count, 1);  // Restore count
        *result = 0;  // Failed
    }
}
```

## 40.6 Error Handling and Edge Cases

### 40.6.1 Process Crash Recovery

```cpp
class IPCRecoveryManager {
private:
    struct ProcessInfo {
        pid_t pid;
        time_t lastHeartbeat;
        bool isAlive;
    };

    struct IPCRegistry {
        ProcessInfo processes[MAX_PROCESSES];
        int processCount;
        cudaIpcMemHandle_t memHandles[MAX_HANDLES];
        int handleCount;
        static const int MAX_PROCESSES = 16;
        static const int MAX_HANDLES = 32;
    };

    IPCMemoryManager registryMem;
    std::thread heartbeatThread;
    bool shouldStop;

public:
    bool initializeRegistry(bool isCoordinator = false) {
        shouldStop = false;

        if (isCoordinator) {
            if (!registryMem.createSharedMemory(sizeof(IPCRegistry))) {
                return false;
            }

            // Initialize registry
            IPCRegistry hostRegistry = {0};
            IPCRegistry* registry = (IPCRegistry*)registryMem.getDevicePtr();
            cudaMemcpy(registry, &hostRegistry, sizeof(IPCRegistry),
                       cudaMemcpyHostToDevice);

            printf("Recovery Manager: Initialized as coordinator\n");
        } else {
            if (!registryMem.attachToSharedMemory()) {
                return false;
            }

            printf("Recovery Manager: Attached to registry\n");
        }

        // Start heartbeat thread
        heartbeatThread = std::thread(&IPCRecoveryManager::heartbeatLoop, this);

        return true;
    }

    void registerProcess() {
        IPCRegistry* registry = (IPCRegistry*)registryMem.getDevicePtr();

        registerProcessKernel<<<1, 1>>>(registry, getpid());
        cudaDeviceSynchronize();

        printf("Recovery Manager: Registered process %d\n", getpid());
    }

    void unregisterProcess() {
        IPCRegistry* registry = (IPCRegistry*)registryMem.getDevicePtr();

        unregisterProcessKernel<<<1, 1>>>(registry, getpid());
        cudaDeviceSynchronize();

        printf("Recovery Manager: Unregistered process %d\n", getpid());
    }

    void checkDeadProcesses() {
        IPCRegistry* registry = (IPCRegistry*)registryMem.getDevicePtr();

        // Copy registry to host for processing
        IPCRegistry hostRegistry;
        cudaMemcpy(&hostRegistry, registry, sizeof(IPCRegistry),
                   cudaMemcpyDeviceToHost);

        time_t currentTime = time(nullptr);

        for (int i = 0; i < hostRegistry.processCount; i++) {
            if (hostRegistry.processes[i].isAlive) {
                // Check if process is still alive
                if (kill(hostRegistry.processes[i].pid, 0) != 0) {
                    printf("Recovery Manager: Detected dead process %d\n",
                           hostRegistry.processes[i].pid);

                    // Mark as dead and clean up resources
                    cleanupDeadProcessKernel<<<1, 1>>>(registry, hostRegistry.processes[i].pid);
                }

                // Check heartbeat timeout
                if (currentTime - hostRegistry.processes[i].lastHeartbeat > 10) {
                    printf("Recovery Manager: Process %d heartbeat timeout\n",
                           hostRegistry.processes[i].pid);
                }
            }
        }
    }

private:
    void heartbeatLoop() {
        while (!shouldStop) {
            // Update heartbeat
            IPCRegistry* registry = (IPCRegistry*)registryMem.getDevicePtr();
            updateHeartbeatKernel<<<1, 1>>>(registry, getpid());

            // Check for dead processes (if coordinator)
            checkDeadProcesses();

            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

public:
    ~IPCRecoveryManager() {
        shouldStop = true;
        if (heartbeatThread.joinable()) {
            heartbeatThread.join();
        }
    }
};

// Recovery kernels
__global__ void registerProcessKernel(IPCRegistry* registry, pid_t pid) {
    int idx = atomicAdd(&registry->processCount, 1);
    if (idx < IPCRegistry::MAX_PROCESSES) {
        registry->processes[idx].pid = pid;
        registry->processes[idx].lastHeartbeat = time(nullptr);
        registry->processes[idx].isAlive = true;
    }
}

__global__ void unregisterProcessKernel(IPCRegistry* registry, pid_t pid) {
    for (int i = 0; i < registry->processCount; i++) {
        if (registry->processes[i].pid == pid) {
            registry->processes[i].isAlive = false;
            break;
        }
    }
}

__global__ void updateHeartbeatKernel(IPCRegistry* registry, pid_t pid) {
    for (int i = 0; i < registry->processCount; i++) {
        if (registry->processes[i].pid == pid && registry->processes[i].isAlive) {
            registry->processes[i].lastHeartbeat = time(nullptr);
            break;
        }
    }
}

__global__ void cleanupDeadProcessKernel(IPCRegistry* registry, pid_t deadPid) {
    for (int i = 0; i < registry->processCount; i++) {
        if (registry->processes[i].pid == deadPid) {
            registry->processes[i].isAlive = false;
            // Additional cleanup could be performed here
            break;
        }
    }
}
```

## 40.7 Performance Optimization

### 40.7.1 IPC Bandwidth Testing

```cpp
class IPCBenchmark {
private:
    IPCMemoryManager memManager;
    size_t bufferSize;
    float* deviceBuffer;

public:
    bool setupBenchmark(size_t size, bool isProducer) {
        bufferSize = size;

        if (isProducer) {
            if (!memManager.createSharedMemory(size)) {
                return false;
            }
        } else {
            if (!memManager.attachToSharedMemory()) {
                return false;
            }
        }

        deviceBuffer = (float*)memManager.getDevicePtr();
        return true;
    }

    double benchmarkWrite(int iterations) {
        std::vector<float> hostData(bufferSize / sizeof(float));
        for (size_t i = 0; i < hostData.size(); i++) {
            hostData[i] = static_cast<float>(i);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy(deviceBuffer, hostData.data(), bufferSize,
                       cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedMs;
        cudaEventElapsedTime(&elapsedMs, start, stop);

        double bandwidthGBps = (bufferSize * iterations / 1e9) / (elapsedMs / 1000.0);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("IPC Write Benchmark: %.2f GB/s\n", bandwidthGBps);
        return bandwidthGBps;
    }

    double benchmarkRead(int iterations) {
        std::vector<float> hostData(bufferSize / sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy(hostData.data(), deviceBuffer, bufferSize,
                       cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedMs;
        cudaEventElapsedTime(&elapsedMs, start, stop);

        double bandwidthGBps = (bufferSize * iterations / 1e9) / (elapsedMs / 1000.0);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("IPC Read Benchmark: %.2f GB/s\n", bandwidthGBps);
        return bandwidthGBps;
    }

    double benchmarkP2PTransfer(int iterations) {
        // Direct GPU-to-GPU transfer through IPC
        float* tempBuffer;
        cudaMalloc(&tempBuffer, bufferSize);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy(tempBuffer, deviceBuffer, bufferSize,
                       cudaMemcpyDeviceToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedMs;
        cudaEventElapsedTime(&elapsedMs, start, stop);

        double bandwidthGBps = (bufferSize * iterations / 1e9) / (elapsedMs / 1000.0);

        cudaFree(tempBuffer);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("IPC P2P Benchmark: %.2f GB/s\n", bandwidthGBps);
        return bandwidthGBps;
    }
};
```

## 40.8 Practical Applications

### 40.8.1 Multi-Process Rendering Pipeline

```cpp
class IPCRenderingPipeline {
private:
    struct RenderFrame {
        float* vertexBuffer;
        float* colorBuffer;
        int* indexBuffer;
        int vertexCount;
        int indexCount;
        volatile int status;  // 0=empty, 1=ready, 2=processing, 3=done
    };

    struct FrameQueue {
        RenderFrame frames[MAX_FRAMES];
        volatile int writeIndex;
        volatile int readIndex;
        static const int MAX_FRAMES = 8;
    };

    IPCMemoryManager queueMem;
    IPCEventManager frameReadyEvent;
    IPCEventManager frameCompleteEvent;

public:
    // Producer (geometry generator)
    bool initializeProducer() {
        if (!queueMem.createSharedMemory(sizeof(FrameQueue))) {
            return false;
        }

        if (!frameReadyEvent.createSharedEvent()) {
            return false;
        }

        // Initialize frame queue
        FrameQueue hostQueue = {0};
        FrameQueue* queue = (FrameQueue*)queueMem.getDevicePtr();
        cudaMemcpy(queue, &hostQueue, sizeof(FrameQueue), cudaMemcpyHostToDevice);

        printf("Render Producer: Pipeline initialized\n");
        return true;
    }

    // Consumer (renderer)
    bool initializeConsumer() {
        if (!queueMem.attachToSharedMemory()) {
            return false;
        }

        if (!frameReadyEvent.attachToSharedEvent()) {
            return false;
        }

        printf("Render Consumer: Attached to pipeline\n");
        return true;
    }

    void generateFrame(const std::vector<float>& vertices,
                      const std::vector<float>& colors,
                      const std::vector<int>& indices) {
        FrameQueue* queue = (FrameQueue*)queueMem.getDevicePtr();

        // Allocate GPU memory for frame data
        float* d_vertices, *d_colors;
        int* d_indices;

        cudaMalloc(&d_vertices, vertices.size() * sizeof(float));
        cudaMalloc(&d_colors, colors.size() * sizeof(float));
        cudaMalloc(&d_indices, indices.size() * sizeof(int));

        cudaMemcpy(d_vertices, vertices.data(),
                   vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colors, colors.data(),
                   colors.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, indices.data(),
                   indices.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Submit frame to queue
        submitFrameKernel<<<1, 1>>>(queue, d_vertices, d_colors, d_indices,
                                   vertices.size() / 3, indices.size());

        // Signal frame ready
        frameReadyEvent.recordEvent();

        printf("Producer: Generated frame with %zu vertices\n", vertices.size() / 3);
    }

    void renderFrames() {
        FrameQueue* queue = (FrameQueue*)queueMem.getDevicePtr();

        while (true) {
            // Wait for frame to be ready
            frameReadyEvent.waitForEvent();

            // Get frame from queue
            int frameIndex = getFrameKernel<<<1, 1>>>(queue);
            cudaDeviceSynchronize();

            if (frameIndex >= 0) {
                printf("Consumer: Rendering frame %d\n", frameIndex);

                // Launch rendering kernels
                dim3 block(256);
                dim3 grid(32);

                vertexShaderKernel<<<grid, block>>>(queue, frameIndex);
                fragmentShaderKernel<<<grid, block>>>(queue, frameIndex);

                // Mark frame as complete
                completeFrameKernel<<<1, 1>>>(queue, frameIndex);

                printf("Consumer: Completed frame %d\n", frameIndex);
            }

            usleep(16667);  // ~60 FPS
        }
    }
};
```

## 40.9 Exercises

1. **Basic IPC Setup**: Implement shared memory communication between two processes
2. **Event Synchronization**: Create inter-process synchronization using CUDA events
3. **Producer-Consumer**: Build a multi-process producer-consumer pattern
4. **Distributed Computing**: Implement a work queue system across processes
5. **Error Recovery**: Add crash detection and recovery mechanisms

## 40.10 Building and Running

```bash
# Build with IPC examples
cd build/30.cuda_advanced/40.CUDA_IPC
ninja

# Run producer in one terminal
./40_CUDA_IPC_producer &

# Run consumer in another terminal
./40_CUDA_IPC_consumer

# Test distributed computing
./40_CUDA_IPC_master &
./40_CUDA_IPC_worker &
./40_CUDA_IPC_worker &

# Profile IPC performance
ncu --target-processes all ./40_CUDA_IPC_benchmark

# Analyze IPC communication
nsys profile --stats=true -t cuda,nvtx ./40_CUDA_IPC_pipeline
```

## 40.11 Key Takeaways

- CUDA IPC enables efficient sharing of GPU resources between processes
- Memory handles allow direct access to GPU memory across process boundaries
- Event handles provide synchronization mechanisms between processes
- Proper error handling and recovery is crucial for robust IPC applications
- IPC can significantly improve performance in multi-process GPU applications
- Understanding process lifecycle management is essential for stable IPC systems
