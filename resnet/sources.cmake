# Include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/modules)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/simulator)

# ... Add more if you need them

# Common source files **without** main function
set(SOURCE_FILES
        src/functional/gemm_helpers.cu
        src/functional/fp16_array.cu
        src/functional/conv2d.cu
        src/functional/im2col.cu
        src/functional/add.cu
        src/functional/batchnorm.cu
        src/functional/pooling.cu
        src/common.cpp
        src/tensor.cpp
        src/dataset.cpp
        src/ops.cpp
        src/modules/conv.cpp
        src/modules/linear.cpp
        src/modules/batchnorm.cpp
        src/modules/pooling.cpp
        src/modules/resnet.cpp
        src/mem_pool.cpp
        src/functional/linear.cu
)

set(SOURCE_GEMM
    src/functional/gemm.cu
    )

set(SIMULATOR_SOURCE_FILES
    src/simulator/half.cpp
    src/simulator/reg_file.cpp
    src/simulator/simulator.cpp
    src/simulator/functions.cpp
)

set(SIMULATOR_GEMM
    src/simulator/gemm.cpp
    )

# Check cuda mallocAsync compatibility
option(ENABLE_SOMA "Use CUDA Stream Ordered Memory Allocator" ON)

if (NOT ENABLE_SOMA)
    message(STATUS "CUDA Stream Ordered Memory Allocator is disabled")
    message(STATUS "Set CMake option ENABLE_SOMA to ON to enable it")
    return()
endif()

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
    add_definitions(-DCUDA_MALLOC_ASYNC)
else()
    message(CHECK_FAIL "Not Available")
    message(STATUS "CUDA Stream Ordered Memory Allocator not available, try install a newer driver.")
    set(CUDA_MALLOC_ASYNC 0)
endif()


