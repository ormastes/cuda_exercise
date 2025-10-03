add_library(${MODULE_NAME}_lib_INTERFACE
INTERFACE
)
target_include_directories(${MODULE_NAME}_lib_INTERFACE
INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

add_library(${MODULE_NAME}_lib
STATIC
    vector_add_2d.cu
)

target_link_libraries(${MODULE_NAME}_lib
    PUBLIC
    ${MODULE_NAME}_lib_INTERFACE
)
