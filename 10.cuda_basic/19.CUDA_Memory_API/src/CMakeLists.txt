# Interface library for headers
add_library(${MODULE}_lib_INTERFACE INTERFACE)
target_include_directories(${MODULE}_lib_INTERFACE
    INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

# Source library for Part 19 - CUDA Memory API (enhanced kernels from previous parts)
add_library(${MODULE}_lib STATIC
    kernels/matrix_multiply.cu
    kernels/vector_add_2d.cu
)

target_compile_definitions(${MODULE}_lib PUBLIC BUILDING_LIB)

target_link_libraries(${MODULE}_lib
    PUBLIC
    ${MODULE}_lib_INTERFACE
    ${CUDA_BASIC_LIB}
)

# Part-specific library for memory API demonstrations
add_library(${MODULE}_memory_lib STATIC
    part_specific/memory_api_demo.cu
)

target_link_libraries(${MODULE}_memory_lib
    PUBLIC
    ${MODULE}_lib_INTERFACE
    ${CUDA_BASIC_LIB}
)

# Main demonstration executable
add_executable(${MODULE}
    part_specific/memory_api_demo.cu
)

target_link_libraries(${MODULE}
    PRIVATE
        ${MODULE}_lib
        ${CUDA_BASIC_LIB}
)

# Set properties for executables
set_target_properties(
    ${MODULE}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
