# add interface target
add_library(GTestCudaGenerator INTERFACE)
target_include_directories(GTestCudaGenerator
INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)
