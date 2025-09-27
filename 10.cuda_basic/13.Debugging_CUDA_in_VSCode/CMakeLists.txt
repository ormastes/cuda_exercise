project(13_Debugging_CUDA_in_VSCode)

add_executable(${PROJECT_NAME} vector_add_2d.cu)

# Enable device-side debug symbols for CUDA debugging
target_compile_options(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O0>"
    "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O0>"
)

target_link_libraries(${PROJECT_NAME} CUDA::cudart)