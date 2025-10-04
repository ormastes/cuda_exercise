# Interface library for headers
add_library(${MODULE}_lib_INTERFACE INTERFACE)
target_include_directories(${MODULE}_lib_INTERFACE
INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

# Static library with CUDA kernels (common evolved code)
add_library(${MODULE}_lib STATIC
    kernels/matrix_multiply.cu
    kernels/vector_add_2d.cu
)

target_compile_definitions(${MODULE}_lib PUBLIC BUILDING_LIB)

target_link_libraries(${MODULE}_lib
    PUBLIC
    ${CUDA_BASIC_LIB}
    ${MODULE}_lib_INTERFACE
)

# Part-specific library for memory hierarchy demonstrations
add_library(${MODULE}_part_specific_lib STATIC
    part_specific/vector_add_memory.cu
)

target_link_libraries(${MODULE}_part_specific_lib
    PUBLIC
    ${CUDA_BASIC_LIB}
    ${MODULE}_lib_INTERFACE
)
