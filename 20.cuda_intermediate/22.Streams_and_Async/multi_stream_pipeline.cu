#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define SEGMENT_SIZE (1<<16)
#define NUM_SEGMENTS 64
#define NUM_STREAMS 4

__global__ void stage1_preprocess(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = sqrtf(fabs(data[idx])) + 1.0f;
    }
}

__global__ void stage2_transform(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        for (int i = 0; i < 50; i++) {
            val = sinf(val) * 2.0f + cosf(val);
        }
        output[idx] = val;
    }
}

__global__ void stage3_reduce(float* input, float* output, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

class StreamPipeline {
private:
    int numStreams;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;

public:
    StreamPipeline(int n) : numStreams(n) {
        streams.resize(numStreams);
        events.resize(numStreams * 3);

        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
        }

        for (int i = 0; i < numStreams * 3; i++) {
            cudaEventCreate(&events[i]);
        }
    }

    ~StreamPipeline() {
        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
        for (auto& event : events) {
            cudaEventDestroy(event);
        }
    }

    cudaStream_t getStream(int idx) {
        return streams[idx % numStreams];
    }

    cudaEvent_t getEvent(int streamIdx, int stage) {
        return events[streamIdx * 3 + stage];
    }

    void synchronize_all() {
        for (auto& stream : streams) {
            cudaStreamSynchronize(stream);
        }
    }
};

int main() {
    printf("Multi-Stream Pipeline Example\n");
    printf("=============================\n");
    printf("Pipeline stages: Preprocess → Transform → Reduce\n");
    printf("Segment size: %d elements\n", SEGMENT_SIZE);
    printf("Number of segments: %d\n", NUM_SEGMENTS);
    printf("Number of streams: %d\n\n", NUM_STREAMS);

    size_t segmentBytes = SEGMENT_SIZE * sizeof(float);
    size_t totalSize = NUM_SEGMENTS * SEGMENT_SIZE;
    size_t totalBytes = totalSize * sizeof(float);

    float *h_input, *h_output;
    float *h_results, *h_results_ref;

    cudaMallocHost(&h_input, totalBytes);
    cudaMallocHost(&h_output, totalBytes);
    h_results = (float*)calloc(NUM_SEGMENTS, sizeof(float));
    h_results_ref = (float*)calloc(NUM_SEGMENTS, sizeof(float));

    for (int i = 0; i < totalSize; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }

    float *d_stage1[NUM_STREAMS], *d_stage2[NUM_STREAMS];
    float *d_results[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_stage1[i], segmentBytes);
        cudaMalloc(&d_stage2[i], segmentBytes);
        cudaMalloc(&d_results[i], sizeof(float));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (SEGMENT_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    printf("Performance Comparison:\n");
    printf("-----------------------\n");

    float* d_temp1, *d_temp2, *d_result;
    cudaMalloc(&d_temp1, segmentBytes);
    cudaMalloc(&d_temp2, segmentBytes);
    cudaMalloc(&d_result, sizeof(float));

    cudaEventRecord(start);
    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        int offset = seg * SEGMENT_SIZE;

        cudaMemcpy(d_temp1, h_input + offset, segmentBytes, cudaMemcpyHostToDevice);

        stage1_preprocess<<<blocksPerGrid, threadsPerBlock>>>(d_temp1, SEGMENT_SIZE);

        stage2_transform<<<blocksPerGrid, threadsPerBlock>>>(d_temp1, d_temp2, SEGMENT_SIZE);

        cudaMemset(d_result, 0, sizeof(float));
        stage3_reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_temp2, d_result, SEGMENT_SIZE);

        cudaMemcpy(&h_results_ref[seg], d_result, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sequential (no pipeline):     %.3f ms\n", milliseconds);
    float sequentialTime = milliseconds;

    StreamPipeline pipeline(NUM_STREAMS);

    memset(h_results, 0, NUM_SEGMENTS * sizeof(float));
    cudaEventRecord(start);

    for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        int streamIdx = seg % NUM_STREAMS;
        cudaStream_t stream = pipeline.getStream(streamIdx);
        int offset = seg * SEGMENT_SIZE;

        cudaMemcpyAsync(d_stage1[streamIdx], h_input + offset, segmentBytes,
                       cudaMemcpyHostToDevice, stream);

        stage1_preprocess<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_stage1[streamIdx], SEGMENT_SIZE);

        stage2_transform<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_stage1[streamIdx], d_stage2[streamIdx], SEGMENT_SIZE);

        cudaMemsetAsync(d_results[streamIdx], 0, sizeof(float), stream);
        stage3_reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), stream>>>(
            d_stage2[streamIdx], d_results[streamIdx], SEGMENT_SIZE);

        cudaMemcpyAsync(&h_results[seg], d_results[streamIdx], sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
    }

    pipeline.synchronize_all();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pipelined (%d streams):       %.3f ms (%.2fx speedup)\n",
           NUM_STREAMS, milliseconds, sequentialTime / milliseconds);

    memset(h_results, 0, NUM_SEGMENTS * sizeof(float));
    cudaEventRecord(start);

    for (int stage = 0; stage < 3; stage++) {
        for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
            int streamIdx = seg % NUM_STREAMS;
            cudaStream_t stream = pipeline.getStream(streamIdx);
            int offset = seg * SEGMENT_SIZE;

            switch(stage) {
                case 0:
                    cudaMemcpyAsync(d_stage1[streamIdx], h_input + offset, segmentBytes,
                                   cudaMemcpyHostToDevice, stream);
                    stage1_preprocess<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                        d_stage1[streamIdx], SEGMENT_SIZE);
                    break;
                case 1:
                    stage2_transform<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                        d_stage1[streamIdx], d_stage2[streamIdx], SEGMENT_SIZE);
                    break;
                case 2:
                    cudaMemsetAsync(d_results[streamIdx], 0, sizeof(float), stream);
                    stage3_reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), stream>>>(
                        d_stage2[streamIdx], d_results[streamIdx], SEGMENT_SIZE);
                    cudaMemcpyAsync(&h_results[seg], d_results[streamIdx], sizeof(float),
                                   cudaMemcpyDeviceToHost, stream);
                    break;
            }
        }
    }

    pipeline.synchronize_all();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Depth-first pipeline:         %.3f ms (%.2fx speedup)\n",
           milliseconds, sequentialTime / milliseconds);

    printf("\nResult Verification:\n");
    printf("--------------------\n");
    bool correct = true;
    float totalDiff = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float diff = fabs(h_results[i] - h_results_ref[i]);
        totalDiff += diff;
        if (diff > 0.01f) {
            correct = false;
            if (i < 5) {
                printf("Segment %d: Pipeline=%.4f, Reference=%.4f, Diff=%.4f\n",
                       i, h_results[i], h_results_ref[i], diff);
            }
        }
    }
    printf("Pipeline correctness: %s\n", correct ? "PASSED" : "FAILED");
    printf("Average difference: %.6f\n", totalDiff / NUM_SEGMENTS);

    printf("\nStream Utilization:\n");
    printf("-------------------\n");
    printf("Segments per stream: %.1f\n", (float)NUM_SEGMENTS / NUM_STREAMS);
    printf("Pipeline depth: 3 stages\n");
    printf("Total operations: %d\n", NUM_SEGMENTS * 3);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_stage1[i]);
        cudaFree(d_stage2[i]);
        cudaFree(d_results[i]);
    }

    cudaFree(d_temp1);
    cudaFree(d_temp2);
    cudaFree(d_result);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    free(h_results);
    free(h_results_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}