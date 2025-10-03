#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#define BATCH_SIZE (1<<16)
#define NUM_BATCHES 32
#define NUM_STREAMS 4

struct BatchData {
    int batchId;
    float* hostData;
    float* deviceData;
    cudaStream_t stream;
    float result;
    bool completed;
};

class CallbackManager {
private:
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<BatchData*> completedBatches;
    std::atomic<int> processedCount{0};

public:
    void add_completed_batch(BatchData* batch) {
        std::lock_guard<std::mutex> lock(mtx);
        completedBatches.push(batch);
        processedCount++;
        cv.notify_one();
    }

    BatchData* getCompletedBatch() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !completedBatches.empty(); });
        BatchData* batch = completedBatches.front();
        completedBatches.pop();
        return batch;
    }

    int getProcessedCount() const {
        return processedCount.load();
    }
};

__global__ void process_batch(float* data, float* result, int size) {
    __shared__ float sharedSum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float localSum = 0.0f;
    while (idx < size) {
        float val = data[idx];
        localSum += sinf(val) * cosf(val);
        data[idx] = sqrtf(fabs(val)) + 1.0f;
        idx += blockDim.x * gridDim.x;
    }

    sharedSum[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sharedSum[0]);
    }
}

void CUDART_CB streamCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    BatchData* batch = (BatchData*)userData;

    if (status != cudaSuccess) {
        printf("Stream callback error: %s\n", cudaGetErrorString(status));
        return;
    }

    batch->completed = true;

    printf("[Callback] Batch %d completed on stream %p, result: %.2f\n",
           batch->batchId, batch->stream, batch->result);

    CallbackManager* manager = (CallbackManager*)((void**)userData)[1];
    manager->add_completed_batch(batch);
}

void post_processing_thread(CallbackManager* manager, int totalBatches) {
    int processed = 0;
    float totalResult = 0.0f;

    while (processed < totalBatches) {
        BatchData* batch = manager->getCompletedBatch();

        float postProcessed = batch->result * 1.5f + 100.0f;
        totalResult += postProcessed;

        printf("[CPU Thread] Post-processing batch %d: %.2f → %.2f\n",
               batch->batchId, batch->result, postProcessed);

        processed++;

        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    printf("[CPU Thread] All batches processed. Total: %.2f\n", totalResult);
}

int main() {
    printf("Stream Callback Example\n");
    printf("=======================\n");
    printf("Batch size: %d elements\n", BATCH_SIZE);
    printf("Number of batches: %d\n", NUM_BATCHES);
    printf("Number of streams: %d\n\n", NUM_STREAMS);

    size_t batchBytes = BATCH_SIZE * sizeof(float);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    BatchData* batches = new BatchData[NUM_BATCHES];
    CallbackManager callbackManager;

    for (int i = 0; i < NUM_BATCHES; i++) {
        cudaMallocHost(&batches[i].hostData, batchBytes);
        cudaMalloc(&batches[i].deviceData, batchBytes);

        for (int j = 0; j < BATCH_SIZE; j++) {
            batches[i].hostData[j] = (float)(j % 100) / 100.0f;
        }

        batches[i].batchId = i;
        batches[i].stream = streams[i % NUM_STREAMS];
        batches[i].result = 0.0f;
        batches[i].completed = false;
    }

    float* d_results[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_results[i], sizeof(float));
    }

    std::thread cpuThread(post_processing_thread, &callbackManager, NUM_BATCHES);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    printf("Starting batch processing...\n");
    printf("============================\n");

    cudaEventRecord(start);

    for (int i = 0; i < NUM_BATCHES; i++) {
        int streamIdx = i % NUM_STREAMS;
        cudaStream_t stream = streams[streamIdx];
        BatchData* batch = &batches[i];

        cudaMemcpyAsync(batch->deviceData, batch->hostData, batchBytes,
                       cudaMemcpyHostToDevice, stream);

        cudaMemsetAsync(d_results[streamIdx], 0, sizeof(float), stream);

        process_batch<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            batch->deviceData, d_results[streamIdx], BATCH_SIZE);

        cudaMemcpyAsync(&batch->result, d_results[streamIdx], sizeof(float),
                       cudaMemcpyDeviceToHost, stream);

        void* callbackData[2] = { batch, &callbackManager };
        cudaStreamAddCallback(stream, streamCallback, callbackData, 0);

        printf("[Main] Launched batch %d on stream %d\n", i, streamIdx);

        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cpuThread.join();

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nPerformance Summary:\n");
    printf("====================\n");
    printf("Total GPU time: %.3f ms\n", milliseconds);
    printf("Average time per batch: %.3f ms\n", milliseconds / NUM_BATCHES);
    printf("Batches processed: %d\n", callbackManager.getProcessedCount());

    printf("\nCallback Statistics:\n");
    printf("--------------------\n");
    int successCount = 0;
    for (int i = 0; i < NUM_BATCHES; i++) {
        if (batches[i].completed) successCount++;
    }
    printf("Successful callbacks: %d/%d\n", successCount, NUM_BATCHES);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice Info:\n");
    printf("-------------\n");
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");

    for (int i = 0; i < NUM_BATCHES; i++) {
        cudaFreeHost(batches[i].hostData);
        cudaFree(batches[i].deviceData);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_results[i]);
        cudaStreamDestroy(streams[i]);
    }

    delete[] batches;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}