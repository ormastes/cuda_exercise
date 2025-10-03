# 33. CUDA Graphs

## 33.1 Introduction

CUDA Graphs capture and optimize kernel launch sequences, reducing CPU overhead and improving GPU utilization.

## 33.2 Learning Objectives

- Understand CUDA Graph concepts
- Create and execute graphs
- Optimize kernel launch overhead
- Implement graph updates and patterns

## 33.3 Graph Fundamentals

### 33.3.1 What are CUDA Graphs?

CUDA Graphs represent work as a graph of nodes and dependencies:
- Reduce kernel launch overhead
- Enable whole-graph optimization
- Support conditional execution
- Improve CPU-GPU concurrency

### 33.3.2 Graph Components

```cuda
// Graph nodes types
cudaGraph_t graph;
cudaGraphNode_t kernelNode, memcpyNode, memsetNode;

// Graph execution
cudaGraphExec_t graphExec;

// Stream capture
cudaStream_t stream;
```

## 33.4 Creating Graphs

### 33.4.1 Explicit Graph Creation

```cuda
void create_explicit_graph() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Add kernel node
    cudaGraphNode_t kernelNode;
    cudaKernelNodeParams kernelParams = {0};

    void* kernelArgs[] = {&d_data, &size};
    kernelParams.func = (void*)myKernel;
    kernelParams.gridDim = dim3(gridSize);
    kernelParams.blockDim = dim3(blockSize);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelParams);

    // Add memcpy node
    cudaGraphNode_t memcpyNode;
    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.srcArray = nullptr;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(d_src, width * sizeof(float), width, height);
    memcpyParams.dstArray = nullptr;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(d_dst, width * sizeof(float), width, height);
    memcpyParams.extent = make_cudaExtent(width * sizeof(float), height, 1);
    memcpyParams.kind = cudaMemcpyDeviceToDevice;

    cudaGraphAddMemcpyNode(&memcpyNode, graph, &kernelNode, 1, &memcpyParams);

    // Instantiate and launch
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(graphExec, stream);
}
```

### 33.4.2 Stream Capture

```cuda
void stream_capture_example() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Begin capture
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Regular CUDA calls
    kernel1<<<grid1, block1, 0, stream>>>(args1);
    cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, stream);
    kernel2<<<grid2, block2, 0, stream>>>(args2);

    // End capture
    cudaStreamEndCapture(stream, &graph);

    // Create executable graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Execute graph multiple times
    for (int i = 0; i < iterations; i++) {
        cudaGraphLaunch(graphExec, stream);
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

## 33.5 Graph Patterns

### 33.5.1 Fork-Join Pattern

```cuda
void fork_join_pattern() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Root node
    cudaGraphNode_t rootNode;
    add_kernel_node(&rootNode, graph, nullptr, 0, preprocessing_kernel);

    // Fork - multiple parallel paths
    cudaGraphNode_t fork1, fork2, fork3;
    add_kernel_node(&fork1, graph, &rootNode, 1, path1_kernel);
    add_kernel_node(&fork2, graph, &rootNode, 1, path2_kernel);
    add_kernel_node(&fork3, graph, &rootNode, 1, path3_kernel);

    // Join - convergence point
    cudaGraphNode_t joinNode;
    cudaGraphNode_t dependencies[] = {fork1, fork2, fork3};
    add_kernel_node(&joinNode, graph, dependencies, 3, merge_kernel);
}
```

### 33.5.2 Conditional Execution

```cuda
class ConditionalGraph {
    cudaGraph_t mainGraph, conditionalGraph;
    cudaGraphNode_t conditionalNode;

public:
    void create() {
        cudaGraphCreate(&mainGraph, 0);
        cudaGraphCreate(&conditionalGraph, 0);

        // Create conditional graph
        cudaGraphNode_t ifNode, elseNode;
        add_kernel_node(&ifNode, conditionalGraph, nullptr, 0, if_kernel);
        add_kernel_node(&elseNode, conditionalGraph, nullptr, 0, else_kernel);

        // Add conditional node to main graph
        cudaGraphConditionalHandle handle;
        cudaGraphConditionalHandleCreate(&handle, mainGraph, 1, 0);

        cudaGraphNodeParams nodeParams;
        nodeParams.type = cudaGraphNodeTypeConditional;
        nodeParams.conditional.handle = handle;
        nodeParams.conditional.type = cudaGraphCondTypeIf;
        nodeParams.conditional.size = 1;

        cudaGraphAddNode(&conditionalNode, mainGraph, nullptr, 0, &nodeParams);
    }
};
```

## 33.6 Graph Updates

### 33.6.1 Kernel Parameter Updates

```cuda
void update_graph_parameters(cudaGraphExec_t graphExec, float* new_data, int new_size) {
    cudaGraph_t graph;
    cudaGraphExecGetGraph(graphExec, &graph);

    // Find kernel node
    size_t numNodes;
    cudaGraphGetNodes(graph, nullptr, &numNodes);

    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);

    for (auto node : nodes) {
        cudaGraphNodeType type;
        cudaGraphNodeGetType(node, &type);

        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            cudaGraphKernelNodeGetParams(node, &params);

            // Update kernel arguments
            void* newArgs[] = {&new_data, &new_size};
            params.kernelParams = newArgs;

            cudaGraphExecKernelNodeSetParams(graphExec, node, &params);
        }
    }
}
```

### 33.6.2 Graph Cloning and Modification

```cuda
cudaGraph_t clone_and_modify_graph(cudaGraph_t original) {
    cudaGraph_t cloned;
    cudaGraphClone(&cloned, original);

    // Add new nodes to cloned graph
    cudaGraphNode_t newNode;
    add_kernel_node(&newNode, cloned, nullptr, 0, additional_kernel);

    // Modify dependencies
    size_t numDependencies;
    cudaGraphNode_t* dependencies;
    cudaGraphNodeGetDependencies(newNode, nullptr, &numDependencies);

    return cloned;
}
```

## 33.7 Advanced Graph Features

### 33.7.1 Child Graphs

```cuda
void child_graph_example() {
    cudaGraph_t parentGraph, childGraph;
    cudaGraphCreate(&parentGraph, 0);
    cudaGraphCreate(&childGraph, 0);

    // Build child graph
    cudaGraphNode_t childKernel;
    add_kernel_node(&childKernel, childGraph, nullptr, 0, child_kernel);

    // Add child graph to parent
    cudaGraphNode_t childGraphNode;
    cudaGraphAddChildGraphNode(&childGraphNode, parentGraph,
                               nullptr, 0, childGraph);

    // Execute parent (includes child)
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, parentGraph, nullptr, nullptr, 0);
}
```

### 33.7.2 Event Nodes

```cuda
void event_node_example() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Create event nodes
    cudaEvent_t event;
    cudaEventCreate(&event);

    cudaGraphNode_t eventRecordNode, eventWaitNode;
    cudaGraphAddEventRecordNode(&eventRecordNode, graph, nullptr, 0, event);

    // Some work nodes...

    cudaGraphAddEventWaitNode(&eventWaitNode, graph, nullptr, 0, event);
}
```

## 33.8 Performance Optimization

### 33.8.1 Graph Execution Strategies

```cuda
class GraphExecutor {
    std::vector<cudaGraphExec_t> graphPool;
    std::queue<cudaGraphExec_t> availableGraphs;

public:
    void execute_with_pooling(cudaGraph_t graph, int iterations) {
        // Create pool of executable graphs
        for (int i = 0; i < 4; i++) {
            cudaGraphExec_t exec;
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            graphPool.push_back(exec);
            availableGraphs.push(exec);
        }

        // Round-robin execution
        for (int i = 0; i < iterations; i++) {
            cudaGraphExec_t exec = availableGraphs.front();
            availableGraphs.pop();

            cudaStream_t stream;
            cudaStreamCreate(&stream);
            cudaGraphLaunch(exec, stream);

            availableGraphs.push(exec);
        }
    }
};
```

### 33.8.2 Measurement and Profiling

```cuda
void profile_graph_execution() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Create and instantiate graph
    create_complex_graph(&graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        cudaGraphLaunch(graphExec, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Graph execution time: %f ms per iteration\n", milliseconds / 1000);
}
```

## 33.9 Practical Examples

### 33.9.1 Image Processing Pipeline

```cuda
class ImagePipelineGraph {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

public:
    void build_pipeline() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Pipeline stages
        gaussian_blur<<<grid, block, 0, stream>>>(input, temp1);
        edge_detection<<<grid, block, 0, stream>>>(temp1, temp2);
        threshold<<<grid, block, 0, stream>>>(temp2, output);

        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    }

    void process_batch(float** images, int count) {
        for (int i = 0; i < count; i++) {
            update_graph_inputs(graphExec, images[i]);
            cudaGraphLaunch(graphExec, 0);
        }
    }
};
```

## 33.10 Exercises

1. **Basic Graph**: Create a graph with multiple kernel nodes
2. **Stream Capture**: Convert existing code to use stream capture
3. **Graph Updates**: Implement dynamic parameter updates
4. **Performance Comparison**: Compare graph vs traditional launch overhead

## 33.11 Building and Running

```bash
# Build graph examples
cd build/30.cuda_advanced/33.CUDA_Graphs
ninja

# Run examples
./33_CUDAGraphs_basic
./33_CUDAGraphs_stream_capture
./33_CUDAGraphs_advanced

# Profile graph execution
nsys profile ./33_CUDAGraphs_basic
ncu --set full ./33_CUDAGraphs_advanced
```

## 33.12 Key Takeaways

- Graphs reduce kernel launch overhead
- Stream capture simplifies graph creation
- Graph updates enable dynamic execution
- Pooling improves performance
- Ideal for repetitive workloads

---

## 33.13 Advanced Graph Capture and Optimization

### 33.13.1 Stream Capture Fundamentals

CUDA Graph Capture provides automated graph creation by recording sequences of CUDA operations, enabling powerful optimization opportunities.

```cuda
class GraphCaptureManager {
private:
    cudaGraph_t capturedGraph;
    cudaGraphExec_t graphExec;
    cudaStream_t captureStream;
    bool isCapturing;
    bool isGraphCreated;

public:
    GraphCaptureManager() : isCapturing(false), isGraphCreated(false) {
        cudaStreamCreate(&captureStream);
    }

    bool beginCapture() {
        if (isCapturing) {
            printf("Already capturing\n");
            return false;
        }

        cudaError_t err = cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            printf("Failed to begin capture: %s\n", cudaGetErrorString(err));
            return false;
        }

        isCapturing = true;
        printf("Graph capture started\n");
        return true;
    }

    bool endCapture() {
        if (!isCapturing) {
            printf("Not currently capturing\n");
            return false;
        }

        cudaError_t err = cudaStreamEndCapture(captureStream, &capturedGraph);
        if (err != cudaSuccess) {
            printf("Failed to end capture: %s\n", cudaGetErrorString(err));
            isCapturing = false;
            return false;
        }

        isCapturing = false;
        isGraphCreated = true;
        printf("Graph capture completed\n");
        return true;
    }

    bool instantiateGraph() {
        if (!isGraphCreated) {
            printf("No graph to instantiate\n");
            return false;
        }

        cudaError_t err = cudaGraphInstantiate(&graphExec, capturedGraph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            printf("Failed to instantiate graph: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("Graph instantiated successfully\n");
        return true;
    }

    bool executeGraph() {
        if (!graphExec) {
            printf("Graph not instantiated\n");
            return false;
        }

        cudaError_t err = cudaGraphLaunch(graphExec, captureStream);
        if (err != cudaSuccess) {
            printf("Failed to launch graph: %s\n", cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    cudaStream_t getStream() const { return captureStream; }

    void recordKernelSequence(float* data, int n) {
        if (!isCapturing) {
            printf("Not capturing - cannot record kernels\n");
            return;
        }

        // Record sequence of kernels
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);

        // Kernel 1: Initialize
        initializeKernel<<<grid, block, 0, captureStream>>>(data, n);

        // Kernel 2: Process
        processKernel<<<grid, block, 0, captureStream>>>(data, n);

        // Kernel 3: Finalize
        finalizeKernel<<<grid, block, 0, captureStream>>>(data, n);

        printf("Recorded kernel sequence in graph\n");
    }

    ~GraphCaptureManager() {
        if (graphExec) {
            cudaGraphExecDestroy(graphExec);
        }
        if (isGraphCreated) {
            cudaGraphDestroy(capturedGraph);
        }
        cudaStreamDestroy(captureStream);
    }
};
```

### 33.13.2 Multi-Stream Graph Capture

```cuda
class AdvancedGraphCapture {
private:
    std::vector<cudaGraph_t> capturedGraphs;
    std::vector<cudaGraphExec_t> graphExecutables;
    std::vector<cudaStream_t> streams;
    int currentGraphIndex;

public:
    AdvancedGraphCapture(int numStreams = 4) : currentGraphIndex(0) {
        streams.resize(numStreams);
        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
        }
    }

    int captureMultiStreamWorkflow(float** deviceBuffers, int* sizes, int numBuffers) {
        cudaGraph_t graph;
        cudaStream_t primaryStream = streams[0];

        // Begin capture on primary stream
        cudaStreamBeginCapture(primaryStream, cudaStreamCaptureModeGlobal);

        // Create event for synchronization
        cudaEvent_t syncEvent;
        cudaEventCreate(&syncEvent);

        // Phase 1: Parallel processing on multiple streams
        for (int i = 0; i < numBuffers && i < streams.size(); i++) {
            cudaStream_t workStream = streams[i % streams.size()];

            // Make stream wait for primary stream (cross-stream dependency)
            cudaEventRecord(syncEvent, primaryStream);
            cudaStreamWaitEvent(workStream, syncEvent, 0);

            // Launch work on this stream
            dim3 block(256);
            dim3 grid((sizes[i] + block.x - 1) / block.x);

            parallelWorkKernel<<<grid, block, 0, workStream>>>(deviceBuffers[i], sizes[i], i);

            // Record completion event
            cudaEvent_t completionEvent;
            cudaEventCreate(&completionEvent);
            cudaEventRecord(completionEvent, workStream);

            // Make primary stream wait for this work
            cudaStreamWaitEvent(primaryStream, completionEvent, 0);

            cudaEventDestroy(completionEvent);
        }

        // Phase 2: Reduction on primary stream
        if (numBuffers > 1) {
            dim3 block(256);
            dim3 grid((sizes[0] + block.x - 1) / block.x);
            reductionKernel<<<grid, block, 0, primaryStream>>>(deviceBuffers, sizes, numBuffers);
        }

        cudaEventDestroy(syncEvent);

        // End capture
        cudaStreamEndCapture(primaryStream, &graph);

        // Instantiate graph
        cudaGraphExec_t graphExec;
        cudaError_t err = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            printf("Failed to instantiate multi-stream graph: %s\n", cudaGetErrorString(err));
            cudaGraphDestroy(graph);
            return -1;
        }

        capturedGraphs.push_back(graph);
        graphExecutables.push_back(graphExec);

        int graphId = capturedGraphs.size() - 1;
        printf("Captured multi-stream workflow as graph %d\n", graphId);
        return graphId;
    }

    int captureConditionalWorkflow(float* data, int n, bool useAdvancedPath) {
        cudaGraph_t graph;
        cudaStream_t stream = streams[0];

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Common preprocessing
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        preprocessKernel<<<grid, block, 0, stream>>>(data, n);

        if (useAdvancedPath) {
            // Advanced processing path
            advancedProcessKernel<<<grid, block, 0, stream>>>(data, n);
            advancedPostprocessKernel<<<grid, block, 0, stream>>>(data, n);
        } else {
            // Simple processing path
            simpleProcessKernel<<<grid, block, 0, stream>>>(data, n);
        }

        // Common postprocessing
        postprocessKernel<<<grid, block, 0, stream>>>(data, n);

        cudaStreamEndCapture(stream, &graph);

        cudaGraphExec_t graphExec;
        cudaError_t err = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            printf("Failed to instantiate conditional graph: %s\n", cudaGetErrorString(err));
            cudaGraphDestroy(graph);
            return -1;
        }

        capturedGraphs.push_back(graph);
        graphExecutables.push_back(graphExec);

        int graphId = capturedGraphs.size() - 1;
        printf("Captured conditional workflow as graph %d\n", graphId);
        return graphId;
    }

    bool executeGraph(int graphId) {
        if (graphId < 0 || graphId >= graphExecutables.size()) {
            printf("Invalid graph ID: %d\n", graphId);
            return false;
        }

        cudaError_t err = cudaGraphLaunch(graphExecutables[graphId], streams[0]);
        if (err != cudaSuccess) {
            printf("Failed to launch graph %d: %s\n", graphId, cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    void analyzeGraph(int graphId) {
        if (graphId < 0 || graphId >= capturedGraphs.size()) {
            printf("Invalid graph ID for analysis: %d\n", graphId);
            return;
        }

        cudaGraph_t graph = capturedGraphs[graphId];

        // Get number of nodes
        size_t numNodes;
        cudaGraphGetNodes(graph, nullptr, &numNodes);
        printf("Graph %d contains %zu nodes\n", graphId, numNodes);

        // Get nodes
        std::vector<cudaGraphNode_t> nodes(numNodes);
        cudaGraphGetNodes(graph, nodes.data(), &numNodes);

        // Analyze each node
        for (size_t i = 0; i < numNodes; i++) {
            cudaGraphNodeType nodeType;
            cudaGraphNodeGetType(nodes[i], &nodeType);

            switch (nodeType) {
                case cudaGraphNodeTypeKernel:
                    printf("  Node %zu: Kernel\n", i);
                    break;
                case cudaGraphNodeTypeMemcpy:
                    printf("  Node %zu: Memory Copy\n", i);
                    break;
                case cudaGraphNodeTypeMemset:
                    printf("  Node %zu: Memory Set\n", i);
                    break;
                case cudaGraphNodeTypeHost:
                    printf("  Node %zu: Host Function\n", i);
                    break;
                case cudaGraphNodeTypeGraph:
                    printf("  Node %zu: Child Graph\n", i);
                    break;
                case cudaGraphNodeTypeEvent:
                    printf("  Node %zu: Event\n", i);
                    break;
                default:
                    printf("  Node %zu: Unknown Type %d\n", i, nodeType);
                    break;
            }
        }
    }

    ~AdvancedGraphCapture() {
        for (auto graphExec : graphExecutables) {
            cudaGraphExecDestroy(graphExec);
        }
        for (auto graph : capturedGraphs) {
            cudaGraphDestroy(graph);
        }
        for (auto stream : streams) {
            cudaStreamDestroy(stream);
        }
    }
};
```

### 33.13.3 Dynamic Graph Updates

```cuda
class DynamicGraphManager {
private:
    cudaGraph_t baseGraph;
    cudaGraphExec_t graphExec;
    std::vector<cudaGraphNode_t> kernelNodes;
    std::vector<cudaKernelNodeParams> originalParams;
    cudaStream_t captureStream;

public:
    bool createOptimizableGraph(float* data, int n) {
        cudaStreamCreate(&captureStream);

        // Capture initial graph
        cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);

        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);

        // Record kernels that we might want to update
        optimizableKernel1<<<grid, block, 0, captureStream>>>(data, n, 1.0f);
        optimizableKernel2<<<grid, block, 0, captureStream>>>(data, n, 2.0f);
        optimizableKernel3<<<grid, block, 0, captureStream>>>(data, n, 3.0f);

        cudaStreamEndCapture(captureStream, &baseGraph);

        // Get nodes for later updates
        size_t numNodes;
        cudaGraphGetNodes(baseGraph, nullptr, &numNodes);

        std::vector<cudaGraphNode_t> allNodes(numNodes);
        cudaGraphGetNodes(baseGraph, allNodes.data(), &numNodes);

        // Filter kernel nodes
        for (auto node : allNodes) {
            cudaGraphNodeType nodeType;
            cudaGraphNodeGetType(node, &nodeType);
            if (nodeType == cudaGraphNodeTypeKernel) {
                kernelNodes.push_back(node);

                cudaKernelNodeParams params;
                cudaGraphKernelNodeGetParams(node, &params);
                originalParams.push_back(params);
            }
        }

        // Instantiate graph
        cudaError_t err = cudaGraphInstantiate(&graphExec, baseGraph, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            printf("Failed to instantiate optimizable graph: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("Created optimizable graph with %zu kernel nodes\n", kernelNodes.size());
        return true;
    }

    bool updateKernelParameters(int kernelIndex, void** newArgs) {
        if (kernelIndex < 0 || kernelIndex >= kernelNodes.size()) {
            printf("Invalid kernel index: %d\n", kernelIndex);
            return false;
        }

        // Update kernel parameters
        cudaKernelNodeParams newParams = originalParams[kernelIndex];
        newParams.kernelParams = newArgs;

        cudaError_t err = cudaGraphExecKernelNodeSetParams(graphExec, kernelNodes[kernelIndex], &newParams);
        if (err != cudaSuccess) {
            printf("Failed to update kernel parameters: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("Updated kernel %d parameters\n", kernelIndex);
        return true;
    }

    bool executeUpdatedGraph() {
        cudaError_t err = cudaGraphLaunch(graphExec, captureStream);
        if (err != cudaSuccess) {
            printf("Failed to launch updated graph: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    void benchmarkGraphVsStream(float* data, int n, int iterations) {
        // Benchmark graph execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cudaGraphLaunch(graphExec, captureStream);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float graphTime;
        cudaEventElapsedTime(&graphTime, start, stop);

        // Benchmark stream execution
        cudaStream_t benchStream;
        cudaStreamCreate(&benchStream);

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            dim3 block(256);
            dim3 grid((n + block.x - 1) / block.x);

            optimizableKernel1<<<grid, block, 0, benchStream>>>(data, n, 1.0f);
            optimizableKernel2<<<grid, block, 0, benchStream>>>(data, n, 2.0f);
            optimizableKernel3<<<grid, block, 0, benchStream>>>(data, n, 3.0f);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float streamTime;
        cudaEventElapsedTime(&streamTime, start, stop);

        printf("Performance Comparison (%d iterations):\n", iterations);
        printf("  Graph execution: %.2f ms\n", graphTime);
        printf("  Stream execution: %.2f ms\n", streamTime);
        printf("  Speedup: %.2fx\n", streamTime / graphTime);

        cudaStreamDestroy(benchStream);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    ~DynamicGraphManager() {
        if (graphExec) {
            cudaGraphExecDestroy(graphExec);
        }
        if (baseGraph) {
            cudaGraphDestroy(baseGraph);
        }
        if (captureStream) {
            cudaStreamDestroy(captureStream);
        }
    }
};
```

### 33.13.4 Advanced Graph Capture Exercises

1. **Stream Capture Optimization**: Implement stream capture for complex kernel sequences
2. **Multi-Stream Dependencies**: Create graphs with cross-stream synchronization
3. **Conditional Graph Capture**: Build graphs with different execution paths
4. **Dynamic Parameter Updates**: Implement runtime graph parameter modifications
5. **Performance Profiling**: Compare graph vs stream execution overhead

### 33.13.5 Advanced Building and Profiling

```bash
# Build with advanced graph capture examples
cd build/30.cuda_advanced/34.CUDA_Graphs
ninja

# Run advanced capture examples
./34_CUDAGraphs_capture
./34_CUDAGraphs_multistream
./34_CUDAGraphs_dynamic

# Profile advanced graph performance
ncu --graph-profiling-mode graph ./34_CUDAGraphs_capture

# Analyze graph optimization
nsys profile --stats=true -t cuda,nvtx ./34_CUDAGraphs_dynamic
```

## 33.14 Final Key Takeaways

- Graphs reduce kernel launch overhead significantly
- Stream capture simplifies graph creation from existing code
- Multi-stream capture enables complex dependency patterns
- Dynamic graph updates allow runtime optimization without recreation
- Graph capture automates optimization for repeated workloads
- Understanding capture modes is crucial for correct graph creation
- Graph analysis tools help optimize execution patterns
- Pooling improves performance for frequently used graphs
- Ideal for repetitive workloads with predictable execution patterns