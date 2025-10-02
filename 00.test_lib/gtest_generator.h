#pragma once
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>


namespace gtest_generator {
static constexpr bool GTEST_GENERATOR_LOG = false;
class TestWithGenerator : public ::testing::TestWithParam<int> {
    public:
    virtual void TestBody() = 0;
};

// Forward declaration
class DynamicRangeGenerator;

// Global registry mapping "SuiteName.TestName" -> generator pointer
static std::map<std::string, DynamicRangeGenerator*> g_range_map;

// Thread-local static variable for counting state
static thread_local bool on_counting = false;
static thread_local int current_count = 1;

// --- Mode toggle (default FULL = Cartesian) ---
enum class SamplingMode { FULL, ALIGNED };
static thread_local SamplingMode tl_mode = SamplingMode::FULL;

// Column sizes discovered during counting; cursor during run
static thread_local std::vector<int> tl_col_sizes;
static thread_local int tl_col_ix = 0;

// Persist per-test column sizes so run phase can access them
static std::map<std::string, std::vector<int>> g_colsizes_map;

// Track which tests use ALIGNED mode (set during first USE_GENERATOR call)
static std::map<std::string, SamplingMode> g_test_modes;

// Custom generator implementing GTest’s ParamGeneratorInterface<int>
class DynamicRangeGenerator : public testing::internal::ParamGeneratorInterface<int> {
 public:
  const std::string key;
  mutable int start = 0;
  mutable int end = 1;
  TestWithGenerator* test_instance;

  explicit DynamicRangeGenerator(const std::string& k, TestWithGenerator* test_case)
      : key(k), test_instance(test_case) {
    g_range_map[key] = this;
    if constexpr (GTEST_GENERATOR_LOG) printf("DynamicRangeGenerator created for %s\n", key.c_str());
    if (test_instance) {
        // First pass with FULL mode to get max count
        on_counting = true;
        current_count = 1;
        tl_col_sizes.clear();
        tl_col_ix = 0;
        tl_mode = SamplingMode::FULL;
        test_instance->TestBody();
        int full_count = current_count;
        
        // Second pass with ALIGNED mode to get column sizes
        on_counting = true;
        current_count = 1;
        tl_col_sizes.clear();
        tl_col_ix = 0;
        tl_mode = SamplingMode::ALIGNED;
        test_instance->TestBody();
        on_counting = false;
        
        // Store both counts - we'll decide which to use at runtime
        g_colsizes_map[key] = tl_col_sizes;
        
        // For now, use the full count as max possible
        end = full_count;
        tl_col_ix = 0;
    }
  }


  // Begin iterator: binds to 'end'
  testing::internal::ParamIteratorInterface<int>* Begin() const override {
    if constexpr (GTEST_GENERATOR_LOG) printf("Creating begin iterator for %s\n", key.c_str());
    return new DynIterator(start, this, /*at_end=*/false);
  }

  // End iterator: marks the end
  testing::internal::ParamIteratorInterface<int>* End() const override {
    return new DynIterator(end, this, /*at_end=*/true);
  }

 private:
  // Iterator yielding a reference to 'end', then marking done
  class DynIterator : public testing::internal::ParamIteratorInterface<int> {
   public:
    DynIterator(int& value, const DynamicRangeGenerator* generator, bool at_end)
        : _value(value), _generator(generator), _done(at_end) {
        if constexpr (GTEST_GENERATOR_LOG) printf("DynIterator constructor: value=%d, at_end=%d, addr=%p\n", value, at_end, this);
    }
    
    ~DynIterator() {
        if constexpr (GTEST_GENERATOR_LOG) printf("DynIterator destructor: addr=%p\n", this);
    }

    void Advance() override {
        if constexpr (GTEST_GENERATOR_LOG) printf("Advancing iterator for %s\n", _generator->key.c_str());
        assert(!_done && "Advance() called on already end iterator");
        _value++;
    }

    testing::internal::ParamIteratorInterface<int>* Clone() const override {
        if constexpr (GTEST_GENERATOR_LOG) printf("Cloning iterator for %s\n", _generator->key.c_str());
        return new DynIterator(_value, _generator, _done);
    }

    const testing::internal::ParamGeneratorInterface<int>* BaseGenerator() const override {
        if constexpr (GTEST_GENERATOR_LOG) printf("Base generator for %s\n", _generator->key.c_str());
        return _generator; // No base generator for begin iterator
    }

    const int* Current() const override {
        if constexpr (GTEST_GENERATOR_LOG) printf("Current value for %s: %d\n", _generator->key.c_str(), _value);
        return &_value; // Return pointer to current value
    }

    bool Equals(const testing::internal::ParamIteratorInterface<int>& other) const override {
        if constexpr (GTEST_GENERATOR_LOG) printf("Comparing iterators for %s\n", _generator->key.c_str());
        if constexpr (GTEST_GENERATOR_LOG) printf("Current value: %d, other value: %d, done: %d\n", _value, other.Current() ? *other.Current() : -1, _done);
    
        if (auto o = dynamic_cast<const DynIterator*>(&other)) {
            if (_value == o->_value) {
                if constexpr (GTEST_GENERATOR_LOG) printf("Iterators are equal for %s\n", _generator->key.c_str());
                return true;
            }
        }
        if constexpr (GTEST_GENERATOR_LOG) printf("Iterators are not equal for %s\n", _generator->key.c_str());
        return false;
    }

   private:
    int& _value;
    bool _done;
    const DynamicRangeGenerator* _generator;
  };
};

// Utility: return reference to the 'end' variable for the current test instance
inline bool IsCountingMode(const ::testing::Test&) {
    /*printf("Getting dynamic range end for current test\n");
    static int dummy_end = 0; // Dummy variable to avoid dangling reference
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string key = std::string(info->test_suite_name()) + "." + info->name();
    printf("Key for dynamic range end: %s\n", key.c_str());
    // check if key contains multiple '/'. count '/' in key
    int key_slash_count = std::count(key.begin(), key.end(), '/');
    printf("Key slash count: %d\n", key_slash_count);
    assert(key_slash_count >= 2 && "Key should not contain multiple '/'");
    // split key by '/' and get second part. 
    size_t pos = key.find('/');
    key = key.substr(pos + 1);
    pos = key.find('/');
    key = key.substr(0, pos);
    for (const auto& pair : g_range_map) {
        printf("Checking generator for %s\n", pair.first.c_str());
    }*/
    return on_counting;
}
// Helper function to create unique ID from file and line
constexpr size_t hash_string(const char* str, size_t hash = 5381) {
    return (*str == 0) ? hash : hash_string(str + 1, ((hash << 5) + hash) + *str);
}

constexpr size_t make_unique_id(const char* file, int line) {
    return hash_string(file) ^ static_cast<size_t>(line);
}

template <size_t UniqueId, typename T>
inline const T& GetGeneratorValue(std::initializer_list<T> values, ::gtest_generator::TestWithGenerator* test_instance) {
    static int current_devider = 1; // Static variable to hold the current devider
    
    if (on_counting) {
        // Record this column's size in declaration order
        tl_col_sizes.push_back((int)values.size());
        
        if (tl_mode == SamplingMode::FULL) {
            current_devider = current_count;
            current_count *= values.size(); // Cartesian only
        }
        return *values.begin(); // dummy in counting pass
    }
    
    // RUN PHASE - check what mode this test uses
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    SamplingMode mode = SamplingMode::FULL;
    if (info) {
        std::string key = std::string(info->test_suite_name()) + "." + info->name();
        auto mode_it = g_test_modes.find(key);
        if (mode_it != g_test_modes.end()) {
            mode = mode_it->second;
        }
    }
    
    if (mode == SamplingMode::FULL) {
        int paramIndex = test_instance ? test_instance->GetParam() : 0;
        return *(values.begin() + ((paramIndex / current_devider) % values.size()));
    } else {
        // ALIGNED: keep column order; round-robin each column's values
        std::vector<int>* col_sizes = nullptr;
        if (info) {
            std::string key = std::string(info->test_suite_name()) + "." + info->name();
            auto it = g_colsizes_map.find(key);
            if (it != g_colsizes_map.end()) {
                col_sizes = &(it->second);
            }
        }
        
        // Check if we should skip this test iteration
        int r = test_instance ? test_instance->GetParam() : 0;
        if (col_sizes && !col_sizes->empty()) {
            int max_size = 0;
            for (int s : *col_sizes) max_size = std::max(max_size, s);
            if (r >= max_size) {
                // This iteration should be skipped for ALIGNED mode
                // Return first value as dummy
                return *values.begin();
            }
            
            int col = tl_col_ix++;
            if (tl_col_ix >= (int)col_sizes->size()) tl_col_ix = 0;
            
            int s_i = (*col_sizes)[col];
            int idx = (s_i == 0) ? 0 : (r % s_i);
            return *(values.begin() + idx);
        }
        
        // Fallback
        int paramIndex = test_instance ? test_instance->GetParam() : 0;
        return *(values.begin() + (paramIndex % values.size()));
    }
}

// Helper template for lazy initialization
template<typename TestClass>
inline DynamicRangeGenerator* CreateGenerator(const std::string& name) {
    static DynamicRangeGenerator* generator = 
        new DynamicRangeGenerator(name, new TestClass());
    return generator;
}

}  // namespace gtest_generator

// Macros
// USE_GENERATOR with optional mode parameter (defaults to FULL for backward compatibility)
#define USE_GENERATOR(...) \
  do { \
    ::gtest_generator::tl_mode = ::gtest_generator::SamplingMode::FULL; \
    __VA_OPT__(::gtest_generator::tl_mode = ::gtest_generator::SamplingMode::__VA_ARGS__;) \
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info(); \
    if (info) { \
      std::string key = std::string(info->test_suite_name()) + "." + info->name(); \
      ::gtest_generator::g_test_modes[key] = ::gtest_generator::tl_mode; \
      if (::gtest_generator::tl_mode == ::gtest_generator::SamplingMode::ALIGNED) { \
        auto it = ::gtest_generator::g_colsizes_map.find(key); \
        if (it != ::gtest_generator::g_colsizes_map.end() && !it->second.empty()) { \
          int max_size = 0; \
          for (int s : it->second) max_size = std::max(max_size, s); \
          if (GetParam() >= max_size) { \
            GTEST_SKIP() << "Skipping iteration " << GetParam() << " for ALIGNED mode (max size: " << max_size << ")"; \
          } \
        } \
      } \
    } \
    ::gtest_generator::tl_col_sizes.clear(); \
    ::gtest_generator::tl_col_ix = 0; \
  } while(0); \
  if (gtest_generator::IsCountingMode(*this)) return;

#define GENERATOR(...) gtest_generator::GetGeneratorValue<gtest_generator::make_unique_id(__FILE__, __LINE__)>({__VA_ARGS__}, this)

#define TEST_G(TestClassName, TestName) \
    class TestClassName##__##TestName : public TestClassName {};\
    class GTEST_TEST_CLASS_NAME_(TestClassName##__##TestName, __); \
    inline gtest_generator::DynamicRangeGenerator* __gtest_generator__get_generator_##TestClassName##TestName() { \
        return gtest_generator::CreateGenerator<GTEST_TEST_CLASS_NAME_(TestClassName##__##TestName, __)>( \
            #TestClassName"."#TestName); \
    } \
    INSTANTIATE_TEST_SUITE_P(Generator, TestClassName##__##TestName, \
        testing::internal::ParamGenerator<int>(__gtest_generator__get_generator_##TestClassName##TestName())); \
    TEST_P(TestClassName##__##TestName, __)
