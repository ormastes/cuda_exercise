#pragma once
// gpu_gtest.h — GPU↔GTest bridge with fixture support, launch config, and parameterized tests.
//
// Features
// - GPU_TEST(Suite, Name): device-run test body (no fixture)
// - GPU_TEST_CFG(Suite, Name, grid, block[, shmem, stream]): explicit <<<grid,block,shmem,stream>>> for non-fixture tests
// - GPU_TEST_F(Fixture, Name): DEFAULT = AUTO (fixture supplies launch config via launch_cfg())
// - GPU_TEST_F_AUTO(Fixture, Name): explicit "auto" version (same as GPU_TEST_F by default)
// - GPU_TEST_F_CFG(Fixture, Name, grid, block[, shmem, stream]): explicit <<<...>>> for fixture tests
// - GPU_TEST_P(Suite, Name): parameterized device test (no fixture)
// - GPU_TEST_P_CFG(Suite, Name, grid, block[, shmem, stream]): parameterized with explicit launch config
//
// Fixture contract (for GPU_TEST_F / GPU_TEST_F_AUTO / GPU_TEST_F_CFG):
//   struct Fixture : ::testing::Test {
//     using DeviceView = ...;                 // POD the kernel needs (raw ptrs, sizes, etc.)
//     const DeviceView* device_view() const;  // returns device-visible pointer (UM or device alloc)
//     GpuLaunchCfg launch_cfg() const;        // default launch config (grid, block, shmem, stream)
//   };
//
// Parameterized test usage:
//   GPU_TEST_P(MySuite, MyTest) {
//     int param = _param;  // access parameter value
//     GPU_EXPECT_EQ(param * 2, param + param);
//   }
//   GPU_INSTANTIATE_TEST_SUITE_P(Inst, MySuite, MyTest, ::testing::Values(1, 2, 3, 4));
//
// Inside device test bodies you can use:
//   - _gpu_result  (GpuTestResult*)
//   - _ctx         (const Fixture::DeviceView*, for fixture tests)
//   - _param       (parameter value, for parameterized tests)
//   - GPU_EXPECT_*/GPU_ASSERT_* macros
//
// Build tips:
//   - Compile as CUDA with GTest linked. Requires <gtest/gtest.h> and <cuda_runtime.h>.
//   - For Unified Memory in fixtures, link with CUDA runtime and set proper archs.
//

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <type_traits>

//==================================================//
//                     Utilities                     //
//==================================================//

// Error check that returns a GTest AssertionFailure when used in launcher helpers
#define GPU_CHECK_CUDA(call)                                                     \
  do {                                                                           \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
      return ::testing::AssertionFailure()                                       \
             << "CUDA error: " << cudaGetErrorString(_e);                        \
    }                                                                            \
  } while (0)

// Launch configuration carrier
struct GpuLaunchCfg {
  dim3 grid{1,1,1};
  dim3 block{1,1,1};
  size_t shmem{0};
  cudaStream_t stream{0};
};

inline GpuLaunchCfg MakeLaunchCfg(dim3 g, dim3 b, size_t s=0, cudaStream_t st=0) {
  return GpuLaunchCfg{g,b,s,st};
}

// Overloaded versions for simpler syntax
inline GpuLaunchCfg MakeLaunchCfg(int g, int b, size_t s=0, cudaStream_t st=0) {
  return GpuLaunchCfg{dim3(g), dim3(b), s, st};
}

inline GpuLaunchCfg MakeLaunchCfg(int g, int b, int bx, size_t s=0, cudaStream_t st=0) {
  return GpuLaunchCfg{dim3(g), dim3(bx, b/bx), s, st};
}

// Grid-stride loop helper (device-side)
#define GPU_FOR_ALL(i, N)                                                        \
  for (int i = blockIdx.x*blockDim.x + threadIdx.x,                              \
           __stride = blockDim.x*gridDim.x;                                      \
       i < static_cast<int>(N);                                                  \
       i += __stride)

//==================================================//
//              Device-side result sink             //
//==================================================//

struct GpuTestResult {
  int failed;                 // 0 = pass
  long long failures_count;   // total failed expectations
  int  first_line;            // first failure detail
  char first_file[128];
  char first_msg[256];
};

__host__ __device__ inline void __gpu_record_failure(GpuTestResult* r,
                                            const char* file, int line,
                                            const char* msg) {
#ifdef __CUDA_ARCH__
  // Device code path
  if (atomicCAS(&r->failed, 0, 1) == 0) {
    r->first_line = line;
    int i=0; while (file[i] && i < (int)sizeof(r->first_file)-1) { r->first_file[i]=file[i]; ++i; }
    r->first_file[i]='\0';
    int j=0; while (msg[j]  && j < (int)sizeof(r->first_msg)-1)  { r->first_msg[j]=msg[j];  ++j; }
    r->first_msg[j]='\0';
  }
  atomicAdd((unsigned long long*)&r->failures_count, 1ULL);
  printf("[GPU-EXPECT] %s:%d %s\n", file, line, msg);
#else
  // Host code path for counting phase
  if (r->failed == 0) {
    r->failed = 1;
    r->first_line = line;
    int i=0; while (file[i] && i < (int)sizeof(r->first_file)-1) { r->first_file[i]=file[i]; ++i; }
    r->first_file[i]='\0';
    int j=0; while (msg[j]  && j < (int)sizeof(r->first_msg)-1)  { r->first_msg[j]=msg[j];  ++j; }
    r->first_msg[j]='\0';
  }
  r->failures_count++;
#endif
}

//==================================================//
//          Device-side ASSERT/EXPECT macros        //
//==================================================//

#define GPU_ASSERT_TRUE(cond)                                                    \
  do { if (!(cond)) { __gpu_record_failure(_gpu_result, __FILE__, __LINE__,      \
        "ASSERT_TRUE(" #cond ")"); return; } } while (0)

#define GPU_EXPECT_TRUE(cond)                                                    \
  do { if (!(cond)) { __gpu_record_failure(_gpu_result, __FILE__, __LINE__,      \
        "EXPECT_TRUE(" #cond ")"); } } while (0)

#define GPU_EXPECT_EQ(a,b)                                                       \
  do { auto _va=(a); auto _vb=(b);                                              \
       if (!(_va == _vb)) {                                                     \
         char _m[256];                                                           \
         int _len = 0;                                                           \
         const char* _fmt = "EXPECT_EQ failed";                                  \
         for (int _i = 0; _fmt[_i] && _len < 255; _i++) _m[_len++] = _fmt[_i];  \
         _m[_len] = '\0';                                                        \
         __gpu_record_failure(_gpu_result, __FILE__, __LINE__, _m);             \
       } } while (0)

#define GPU_EXPECT_NEAR(a,b,abs_tol)                                            \
  do { double _x=(double)(a), _y=(double)(b), _t=(double)(abs_tol);              \
       double _d = fabs(_x - _y);                                                \
       if (!(_d <= _t)) {                                                        \
         char _m[256];                                                           \
         int _len = 0;                                                           \
         const char* _fmt = "EXPECT_NEAR failed";                                \
         for (int _i = 0; _fmt[_i] && _len < 255; _i++) _m[_len++] = _fmt[_i];  \
         _m[_len] = '\0';                                                        \
         __gpu_record_failure(_gpu_result, __FILE__, __LINE__, _m);              \
       } } while (0)

//==================================================//
//                 Host-side launchers              //
//==================================================//

// No-context launcher: kernel signature void(GpuTestResult*)
inline ::testing::AssertionResult LaunchGpuTest(
    void (*kernel)(GpuTestResult*),
    dim3 grid = dim3(1), dim3 block = dim3(1), size_t shmem = 0,
    cudaStream_t stream = 0)
{
  GpuTestResult h{}; GpuTestResult* d=nullptr;
  GPU_CHECK_CUDA(cudaMalloc(&d, sizeof(GpuTestResult)));
  GPU_CHECK_CUDA(cudaMemset(d, 0, sizeof(GpuTestResult)));

  kernel<<<grid, block, shmem, stream>>>(d);
  cudaError_t post = cudaGetLastError();
  if (post != cudaSuccess) {
    cudaFree(d);
    return ::testing::AssertionFailure() << "Kernel launch: "
                                         << cudaGetErrorString(post);
  }
  GPU_CHECK_CUDA(cudaStreamSynchronize(stream));
  GPU_CHECK_CUDA(cudaMemcpy(&h, d, sizeof(GpuTestResult), cudaMemcpyDeviceToHost));
  cudaFree(d);

  if (h.failed) {
    ::testing::Message m; m << h.first_file << ":" << h.first_line
                            << " — " << h.first_msg
                            << " (failures=" << h.failures_count << ")";
    return ::testing::AssertionFailure() << m;
  }
  return ::testing::AssertionSuccess();
}

// Context launcher: kernel signature void(GpuTestResult*, const Ctx*)
template <class Ctx>
inline ::testing::AssertionResult LaunchGpuTestWithCtx(
    void (*kernel)(GpuTestResult*, const Ctx*),
    const Ctx* ctx_device_ptr,
    dim3 grid = dim3(1), dim3 block = dim3(1), size_t shmem = 0,
    cudaStream_t stream = 0)
{
  GpuTestResult h{}; GpuTestResult* d=nullptr;
  GPU_CHECK_CUDA(cudaMalloc(&d, sizeof(GpuTestResult)));
  GPU_CHECK_CUDA(cudaMemset(d, 0, sizeof(GpuTestResult)));

  kernel<<<grid, block, shmem, stream>>>(d, ctx_device_ptr);
  cudaError_t post = cudaGetLastError();
  if (post != cudaSuccess) {
    cudaFree(d);
    return ::testing::AssertionFailure() << "Kernel launch: "
                                         << cudaGetErrorString(post);
  }
  GPU_CHECK_CUDA(cudaStreamSynchronize(stream));
  GPU_CHECK_CUDA(cudaMemcpy(&h, d, sizeof(GpuTestResult), cudaMemcpyDeviceToHost));
  cudaFree(d);

  if (h.failed) {
    ::testing::Message m; m << h.first_file << ":" << h.first_line
                            << " — " << h.first_msg
                            << " (failures=" << h.failures_count << ")";
    return ::testing::AssertionFailure() << m;
  }
  return ::testing::AssertionSuccess();
}

//==================================================//
//                    Test macros                   //
//==================================================//

// ---- Non-fixture: default <<<1,1>>>
#define GPU_TEST(SUITE, NAME)                                                    \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result);           \
  TEST(SUITE, NAME) {                                                            \
    ASSERT_TRUE(LaunchGpuTest(SUITE##_##NAME##_kernel));                         \
  }                                                                              \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result)

// ---- Non-fixture: explicit <<<...>>> via simple parameters
#define GPU_TEST_CFG(SUITE, NAME, GRID, BLOCK, ...)                              \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result);           \
  TEST(SUITE, NAME) {                                                            \
    GpuLaunchCfg _cfg = MakeLaunchCfg(GRID, BLOCK, ##__VA_ARGS__);               \
    ASSERT_TRUE(LaunchGpuTest(SUITE##_##NAME##_kernel,                           \
                              _cfg.grid, _cfg.block, _cfg.shmem, _cfg.stream));  \
  }                                                                              \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result)

// ---- Fixture: explicit <<<...>>> via simple parameters
// Fixture must provide: using DeviceView=...; const DeviceView* device_view() const;
#define GPU_TEST_F_CFG(FIXTURE, NAME, GRID, BLOCK, ...)                          \
  __global__ void FIXTURE##_##NAME##_kernel(                                     \
      GpuTestResult* _gpu_result, const FIXTURE::DeviceView* _ctx);              \
  TEST_F(FIXTURE, NAME) {                                                        \
    GpuLaunchCfg _cfg = MakeLaunchCfg(GRID, BLOCK, ##__VA_ARGS__);               \
    ASSERT_TRUE(LaunchGpuTestWithCtx(FIXTURE##_##NAME##_kernel,                  \
                                     this->device_view(),                        \
                                     _cfg.grid, _cfg.block,                      \
                                     _cfg.shmem, _cfg.stream));                  \
  }                                                                              \
  __global__ void FIXTURE##_##NAME##_kernel(                                     \
      GpuTestResult* _gpu_result, const FIXTURE::DeviceView* _ctx)

// ---- Fixture: AUTO (fixture decides launch cfg via launch_cfg())
#define GPU_TEST_F_AUTO(FIXTURE, NAME)                                           \
  __global__ void FIXTURE##_##NAME##_kernel(                                     \
      GpuTestResult* _gpu_result, const FIXTURE::DeviceView* _ctx);              \
  TEST_F(FIXTURE, NAME) {                                                        \
    GpuLaunchCfg _cfg = this->launch_cfg();                                      \
    ASSERT_TRUE(LaunchGpuTestWithCtx(FIXTURE##_##NAME##_kernel,                  \
                                     this->device_view(),                        \
                                     _cfg.grid, _cfg.block,                      \
                                     _cfg.shmem, _cfg.stream));                  \
  }                                                                              \
  __global__ void FIXTURE##_##NAME##_kernel(                                     \
      GpuTestResult* _gpu_result, const FIXTURE::DeviceView* _ctx)

// ---- DEFAULT ALIAS: GPU_TEST_F == AUTO
#ifndef GPU_GTEST_NO_DEFAULT_AUTO
#define GPU_TEST_F(FIXTURE, NAME) GPU_TEST_F_AUTO(FIXTURE, NAME)
#endif

//==================================================//
//                Example device helpers            //
//==================================================//
// Optionally add small helpers; keep header lean.
// Users can add their own in test TU if preferred.

//==================================================//
//              Parameterized Test Support          //
//==================================================//

// Parameterized launcher: kernel signature void(GpuTestResult*, ParamType)
template <typename ParamType>
inline ::testing::AssertionResult LaunchGpuTestWithParam(
    void (*kernel)(GpuTestResult*, ParamType),
    ParamType param,
    dim3 grid = dim3(1), dim3 block = dim3(1), size_t shmem = 0,
    cudaStream_t stream = 0)
{
  GpuTestResult h{}; GpuTestResult* d=nullptr;
  GPU_CHECK_CUDA(cudaMalloc(&d, sizeof(GpuTestResult)));
  GPU_CHECK_CUDA(cudaMemset(d, 0, sizeof(GpuTestResult)));

  kernel<<<grid, block, shmem, stream>>>(d, param);
  cudaError_t post = cudaGetLastError();
  if (post != cudaSuccess) {
    cudaFree(d);
    return ::testing::AssertionFailure() << "Kernel launch: "
                                         << cudaGetErrorString(post);
  }
  GPU_CHECK_CUDA(cudaStreamSynchronize(stream));
  GPU_CHECK_CUDA(cudaMemcpy(&h, d, sizeof(GpuTestResult), cudaMemcpyDeviceToHost));
  cudaFree(d);

  if (h.failed) {
    ::testing::Message m; m << h.first_file << ":" << h.first_line
                            << " — " << h.first_msg
                            << " (failures=" << h.failures_count << ")";
    return ::testing::AssertionFailure() << m;
  }
  return ::testing::AssertionSuccess();
}

// ---- Parameterized test: default <<<1,1>>>
#define GPU_TEST_P(SUITE, NAME)                                                  \
  template<typename ParamType>                                                   \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result,            \
                                          ParamType _param);                     \
  class SUITE##_##NAME##_TestBase : public ::testing::TestWithParam<int> {       \
  public:                                                                        \
    template<typename T>                                                         \
    void RunTest(T param) {                                                      \
      ASSERT_TRUE(LaunchGpuTestWithParam(SUITE##_##NAME##_kernel<T>, param));    \
    }                                                                            \
  };                                                                             \
  TEST_P(SUITE##_##NAME##_TestBase, Test) {                                     \
    RunTest(GetParam());                                                         \
  }                                                                              \
  template<typename ParamType>                                                   \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result,            \
                                          ParamType _param)

// ---- Parameterized test: explicit <<<...>>>
#define GPU_TEST_P_CFG(SUITE, NAME, GRID, BLOCK, ...)                            \
  template<typename ParamType>                                                   \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result,            \
                                          ParamType _param);                     \
  class SUITE##_##NAME##_TestBase : public ::testing::TestWithParam<int> {       \
  public:                                                                        \
    template<typename T>                                                         \
    void RunTest(T param) {                                                      \
      GpuLaunchCfg _cfg = MakeLaunchCfg(GRID, BLOCK, ##__VA_ARGS__);             \
      ASSERT_TRUE(LaunchGpuTestWithParam(SUITE##_##NAME##_kernel<T>, param,      \
                                         _cfg.grid, _cfg.block,                  \
                                         _cfg.shmem, _cfg.stream));              \
    }                                                                            \
  };                                                                             \
  TEST_P(SUITE##_##NAME##_TestBase, Test) {                                     \
    RunTest(GetParam());                                                         \
  }                                                                              \
  template<typename ParamType>                                                   \
  __global__ void SUITE##_##NAME##_kernel(GpuTestResult* _gpu_result,            \
                                          ParamType _param)

// Instantiation helper for parameterized GPU tests
#define GPU_INSTANTIATE_TEST_SUITE_P(PREFIX, SUITE, NAME, VALUES)                \
  INSTANTIATE_TEST_SUITE_P(PREFIX, SUITE##_##NAME##_TestBase, VALUES)

//==================================================//
//           Generator-Based Test Support           //
//==================================================//

#include <map>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <cuda/std/array>

namespace gpu_generator {

// Sampling modes
enum class SamplingMode { FULL, ALIGNED };

// Forward declarations
class GpuTestWithGenerator;
class DynamicRangeGenerator;

// Global registry for generators
static std::map<std::string, DynamicRangeGenerator*> g_range_map;

// Base class for generator-based tests
class GpuTestWithGenerator : public ::testing::TestWithParam<int> {
public:
    // Instance state for generation (moved from thread_local)
    static constexpr int MAX_GENERATORS = 32;
    bool on_counting = false;
    int current_count = 1;
    int current_divider = 1;
    cuda::std::array<int, MAX_GENERATORS> col_sizes{};
    size_t col_sizes_count = 0;
    int col_ix = 0;
    SamplingMode mode = SamplingMode::FULL;
    DynamicRangeGenerator* current_generator = nullptr;
    size_t param_index = 0;

    virtual void TestBody() = 0;

    virtual void RunCpuTest() = 0;

    // GPU-specific: launch configuration
    virtual GpuLaunchCfg launch_cfg() const {
        return GpuLaunchCfg{dim3(1), dim3(1), 0, 0};
    }
};

// Per-test metadata (kept global for cross-test access)
static std::map<std::string, cuda::std::array<int, GpuTestWithGenerator::MAX_GENERATORS>> g_colsizes_map;
static std::map<std::string, SamplingMode> g_test_modes;


// Dynamic range generator for parameterized tests
class DynamicRangeGenerator : public testing::internal::ParamGeneratorInterface<int> {
public:
    const std::string key;
    mutable int start = 0;
    mutable int end = 0;
    GpuTestWithGenerator* test_instance;

    explicit DynamicRangeGenerator(const std::string& k, GpuTestWithGenerator* test_case)
        : key(k), test_instance(test_case) {
        g_range_map[key] = this;
        test_instance->current_generator = this;
    }

    testing::internal::ParamIteratorInterface<int>* Begin() const override {
        return new DynIterator(start, this, false);
    }

    testing::internal::ParamIteratorInterface<int>* End() const override {
        if (end == 0) {
            test_instance->RunCpuTest();
            end = test_instance->current_count;
        }
        return new DynIterator(end, this, true);
    }

private:
    class DynIterator : public testing::internal::ParamIteratorInterface<int> {
    public:
        DynIterator(int& value, const DynamicRangeGenerator* generator, bool at_end)
            : _value(value), _generator(generator), _done(at_end) {}

        void Advance() override {
            if (!_done) _value++;
        }

        testing::internal::ParamIteratorInterface<int>* Clone() const override {
            return new DynIterator(_value, _generator, _done);
        }

        const testing::internal::ParamGeneratorInterface<int>* BaseGenerator() const override {
            return _generator;
        }

        const int* Current() const override {
            return &_value;
        }

        bool Equals(const testing::internal::ParamIteratorInterface<int>& other) const override {
            if (auto o = dynamic_cast<const DynIterator*>(&other)) {
                return _value == o->_value;
            }
            return false;
        }

    private:
        int& _value;
        bool _done;
        const DynamicRangeGenerator* _generator;
    };
};

// Hash function for unique IDs
__host__ __device__ constexpr size_t hash_string(const char* str, size_t hash = 5381) {
    return (*str == 0) ? hash : hash_string(str + 1, ((hash << 5) + hash) + *str);
}

__host__ __device__ constexpr size_t make_unique_id(const char* file, int line) {
    return hash_string(file) ^ static_cast<size_t>(line);
}

// Get generator value based on current iteration
template <size_t UniqueId, typename T>
__host__ __device__ inline const T& GetGeneratorValue(std::initializer_list<T> values, gpu_generator::GpuTestWithGenerator* test_instance) {
    assert(values.size() > 0 && "Generator values cannot be empty");
    if (test_instance->on_counting) {
        test_instance->col_sizes[test_instance->col_sizes_count] = (int)values.size();
        test_instance->col_sizes_count++;
        return *values.begin();
    }
    int paramIndex = test_instance->current_count;
    if (test_instance->mode == SamplingMode::FULL) {
        size_t devider = test_instance->current_divider;
        test_instance->current_divider *= (int)values.size();
        return *(values.begin() + ((paramIndex / devider) % values.size()));
    } else {
        return *(values.begin() + (paramIndex % values.size()));
    }
}

// Helper for lazy generator creation
template<typename TestClass>
inline DynamicRangeGenerator* CreateGenerator(const std::string& name, GpuTestWithGenerator* test_case) {
    static DynamicRangeGenerator* generator = new DynamicRangeGenerator(name, test_case);
    return generator;
}

// Helper for lazy generator creation
template<typename TestClass>
inline TestClass* CreateTest() {
    static TestClass* test = new TestClass();
    return test;
}

} // namespace gpu_generator

// GPU-specific generator macros
#define GPU_GENERATOR(...) ::gpu_generator::GetGeneratorValue<::gpu_generator::make_unique_id(__FILE__, __LINE__)>({__VA_ARGS__}, _test_instance)

#define GPU_USE_GENERATOR(...) \
  do { \
    __VA_OPT__(_test_instance->mode = ::gpu_generator::SamplingMode::__VA_ARGS__;) \
  } while(0); \
  if (_test_instance->on_counting) return;

// Helper for launching generator-based GPU tests
template<typename TestInstance>
inline ::testing::AssertionResult LaunchCpuGeneratorTestCount(
    void (*kernel)(GpuTestResult*, TestInstance*),
    TestInstance* test_instance)
{
    if (::gpu_generator::g_colsizes_map.find(test_instance->current_generator->key) == ::gpu_generator::g_colsizes_map.end()) {
        test_instance->on_counting = true;

        // We can't call the CUDA kernel directly from host
        // Instead, we need to launch it properly on the device for counting
        GpuTestResult* d = nullptr;
        cudaMalloc(&d, sizeof(GpuTestResult));
        cudaMemset(d, 0, sizeof(GpuTestResult));

        TestInstance* d_instance = nullptr;
        cudaMalloc(&d_instance, sizeof(TestInstance));
        cudaMemcpy(d_instance, test_instance, sizeof(TestInstance), cudaMemcpyHostToDevice);

        // Launch kernel with minimal configuration for counting
        kernel<<<1, 1>>>(d, d_instance);
        cudaDeviceSynchronize();

        // Copy back the test instance to get the counted values
        cudaMemcpy(test_instance, d_instance, sizeof(TestInstance), cudaMemcpyDeviceToHost);

        cudaFree(d);
        cudaFree(d_instance);

        test_instance->on_counting = false;
        ::gpu_generator::g_colsizes_map[test_instance->current_generator->key] = test_instance->col_sizes;
        ::gpu_generator::g_test_modes[test_instance->current_generator->key] = test_instance->mode;

        if (test_instance->mode == ::gpu_generator::SamplingMode::ALIGNED) {
            int max = 1;
            for (int i = 0; i < (int)test_instance->col_sizes_count; ++i) {
                if (test_instance->col_sizes[i] > max) {
                    max = test_instance->col_sizes[i];
                }
            }
            test_instance->current_generator->end = max;
        } else {
            int combinations = 1;
            for (int i = 0; i < (int)test_instance->col_sizes_count; ++i) {
                combinations *= test_instance->col_sizes[i];
            }
            test_instance->current_generator->end = combinations;
        }
        test_instance->current_count = test_instance->current_generator->end;

        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionSuccess();
}
template<typename TestInstance>
inline ::testing::AssertionResult LaunchGpuGeneratorTest(
    void (*kernel)(GpuTestResult*, TestInstance*),
    TestInstance* test_instance,
    const GpuLaunchCfg& cfg)
{
    test_instance->current_count = test_instance->GetParam();
    test_instance->current_divider = 1;

    GpuTestResult h{};
    GpuTestResult* d = nullptr;
    GPU_CHECK_CUDA(cudaMalloc(&d, sizeof(GpuTestResult)));
    GPU_CHECK_CUDA(cudaMemset(d, 0, sizeof(GpuTestResult)));

    // Allocate device memory for test instance
    TestInstance* d_instance = nullptr;
    GPU_CHECK_CUDA(cudaMalloc(&d_instance, sizeof(TestInstance)));
    GPU_CHECK_CUDA(cudaMemcpy(d_instance, test_instance, sizeof(TestInstance), cudaMemcpyHostToDevice));

    kernel<<<cfg.grid, cfg.block, cfg.shmem, cfg.stream>>>(d, d_instance);
    cudaError_t post = cudaGetLastError();
    if (post != cudaSuccess) {
        cudaFree(d);
        cudaFree(d_instance);
        return ::testing::AssertionFailure() << "Kernel launch: " << cudaGetErrorString(post);
    }
    GPU_CHECK_CUDA(cudaStreamSynchronize(cfg.stream));
    GPU_CHECK_CUDA(cudaMemcpy(&h, d, sizeof(GpuTestResult), cudaMemcpyDeviceToHost));
    GPU_CHECK_CUDA(cudaMemcpy(test_instance, d_instance, sizeof(TestInstance), cudaMemcpyDeviceToHost));
    cudaFree(d);
    cudaFree(d_instance);

    if (h.failed) {
        ::testing::Message m;
        m << h.first_file << ":" << h.first_line
          << " — " << h.first_msg
          << " (failures=" << h.failures_count << ")";
        return ::testing::AssertionFailure() << m;
    }
    return ::testing::AssertionSuccess();
}

// GPU generator test macro
#define GPU_TEST_G(TestClassName, TestName) \
    class TestClassName##__##TestName##_Test : public ::gpu_generator::GpuTestWithGenerator { \
    public: \
        void TestBody() override; \
        void RunGpuTest(); \
        void RunCpuTest() override; \
    }; \
    __global__ void TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance); \
    inline ::gpu_generator::DynamicRangeGenerator* __gtest_generator__get_generator_##TestClassName##TestName() { \
        return ::gpu_generator::CreateGenerator<TestClassName##__##TestName##_Test>( \
            #TestClassName"."#TestName, ::gpu_generator::CreateTest<TestClassName##__##TestName##_Test>()); \
    } \
    INSTANTIATE_TEST_SUITE_P(Generator, TestClassName##__##TestName##_Test, \
        testing::internal::ParamGenerator<int>(__gtest_generator__get_generator_##TestClassName##TestName())); \
    TEST_P(TestClassName##__##TestName##_Test, __) { \
        RunGpuTest(); \
    } \
    void TestClassName##__##TestName##_Test::TestBody() { \
        RunGpuTest(); \
    } \
    void TestClassName##__##TestName##_Test::RunCpuTest() { \
        auto* _test_instance = this; \
        LaunchCpuGeneratorTestCount(TestClassName##__##TestName##_kernel, _test_instance); \
    } \
    void TestClassName##__##TestName##_Test::RunGpuTest() { \
        auto* _test_instance = this; \
        auto cfg = this->launch_cfg(); \
        ASSERT_TRUE(LaunchGpuGeneratorTest( \
            TestClassName##__##TestName##_kernel, _test_instance, cfg)); \
    } \
    __host__ __device__ void _##TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance); \
    __global__ void TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance) { \
          _##TestClassName##__##TestName##_kernel(_gpu_result, _test_instance); \
        } \
    __host__ __device__ void _##TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance)

// GPU generator test with custom launch config
#define GPU_TEST_G_CFG(TestClassName, TestName, GRID, BLOCK, ...) \
    class TestClassName##__##TestName##_Test : public ::gpu_generator::GpuTestWithGenerator { \
    public: \
        void TestBody() override; \
        void RunGpuTest(); \
        void RunCpuTest(); \
        GpuLaunchCfg launch_cfg() const override { \
            return MakeLaunchCfg(GRID, BLOCK, ##__VA_ARGS__); \
        } \
    }; \
    __global__ void TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance); \
    inline ::gpu_generator::DynamicRangeGenerator* __gtest_generator__get_generator_##TestClassName##TestName() { \
        return ::gpu_generator::CreateGenerator<TestClassName##__##TestName##_Test>( \
            #TestClassName"."#TestName, ::gpu_generator::CreateTest<TestClassName##__##TestName##_Test>()); \
    } \
    INSTANTIATE_TEST_SUITE_P(Generator, TestClassName##__##TestName##_Test, \
        testing::internal::ParamGenerator<int>(__gtest_generator__get_generator_##TestClassName##TestName())); \
    TEST_P(TestClassName##__##TestName##_Test, __) { \
        RunGpuTest(); \
    } \
    void TestClassName##__##TestName##_Test::TestBody() { \
        RunGpuTest(); \
    } \
    void TestClassName##__##TestName##_Test::RunCpuTest() { \
        auto* _test_instance = this; \
        LaunchCpuGeneratorTestCount(TestClassName##__##TestName##_kernel, _test_instance); \
    } \
    void TestClassName##__##TestName##_Test::RunGpuTest() { \
        auto* _test_instance = this; \
        auto cfg = this->launch_cfg(); \
        ASSERT_TRUE(LaunchGpuGeneratorTest( \
            TestClassName##__##TestName##_kernel, _test_instance, cfg)); \
    } \
    __host__ __device__ void _##TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance); \
    __global__ void TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance) { \
          _##TestClassName##__##TestName##_kernel(_gpu_result, _test_instance); \
        } \
    __host__ __device__ void _##TestClassName##__##TestName##_kernel( \
        GpuTestResult* _gpu_result, \
        TestClassName##__##TestName##_Test* _test_instance)
