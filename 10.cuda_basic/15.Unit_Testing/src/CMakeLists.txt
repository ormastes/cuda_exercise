# Interface library for headers
add_library(${MODULE_NAME}_lib_INTERFACE INTERFACE)

target_include_directories(${MODULE_NAME}_lib_INTERFACE
    INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

# Static library with CUDA kernels
add_library(${MODULE_NAME}_lib STATIC
    vector_add_2d.cu
)

target_compile_definitions(${MODULE_NAME}_lib PUBLIC
    BUILDING_LIB  # Define to avoid main() in library if needed
)

target_link_libraries(${MODULE_NAME}_lib
    PUBLIC
    ${MODULE_NAME}_lib_INTERFACE
)