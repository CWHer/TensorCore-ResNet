# Include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules)

# ... Add more if you need them

# Common source files **without** main function
set(SOURCE_FILES
        src/functional/gemm.cu
        src/functional/conv2d.cu
        src/functional/im2col.cu
        src/functional/add.cu
        src/functional/batchnorm.cu
        src/functional/pooling.cu
        src/common.cpp
        src/tensor.cpp
        src/dataset.cpp
        src/ops.cpp
        src/conv.cpp
        src/linear.cpp
        src/batchnorm.cpp
        src/pooling.cpp
        src/resnet.cpp
        src/functional/linear.cu
)

# Check cuda mallocAsync compatibility
message(CHECK_START "Check CUDA Stream Ordered Memory Allocator")

try_run(RUN_RESULT COMPILE_RESULT
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/has_malloc_async.cpp
        CMAKE_FLAGS
            -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS}
            -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT)


if (COMPILE_RESULT AND NOT RUN_RESULT)
    message(CHECK_PASS "Supported")
    set(CUDA_MALLOC_ASYNC 1)
else()
    message(CHECK_FAIL "Not Available")
    set(CUDA_MALLOC_ASYNC 0)
endif()


