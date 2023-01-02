# Include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/modules)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/simulator)

# ... Add more if you need them

# Common source files **without** main function
set(SOURCE_FILES
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
else()
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
    else ()
        message(CHECK_FAIL "Not Available")
        message(STATUS "CUDA Stream Ordered Memory Allocator not available, try install a newer driver.")
        set(CUDA_MALLOC_ASYNC 0)
    endif ()
endif ()

set(POSSIBLE_RESTRICTS __restrict__ __restrict restrict)
foreach (RESTRICT ${POSSIBLE_RESTRICTS})
    message(CHECK_START "Check ${RESTRICT} of ${CMAKE_CXX_COMPILER_ID} C++ compiler")

    try_run(CXX_RUN_RESULT CXX_COMPILE_RESULT
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/has_restrict.cpp
            COMPILE_DEFINITIONS -DRESTRICT=${RESTRICT}
            COMPILE_OUTPUT_VARIABLE CXX_COMPILE_OUTPUT)

    if (CXX_COMPILE_RESULT AND NOT CXX_RUN_RESULT)
        message(CHECK_PASS "Supported")
    else()
        message(CHECK_FAIL "Not Available")
    endif()

    message(CHECK_START "Check ${RESTRICT} of ${CMAKE_CUDA_COMPILER_ID} CUDA compiler")

    try_run(CUDA_RUN_RESULT CUDA_COMPILE_RESULT
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/has_restrict.cu
            COMPILE_DEFINITIONS -DRESTRICT=${RESTRICT}
            COMPILE_OUTPUT_VARIABLE CUDA_COMPILE_OUTPUT)

    if (CUDA_COMPILE_RESULT AND NOT CUDA_RUN_RESULT)
        message(CHECK_PASS "Supported")
    else()
        message(CHECK_FAIL "Not Available")
    endif()

    if (CXX_COMPILE_RESULT AND NOT CXX_RUN_RESULT AND CUDA_COMPILE_RESULT AND NOT CUDA_RUN_RESULT)
        set(HAS_${RESTRICT} ON)
    endif()

endforeach ()


if (HAS___restrict__)
    add_definitions(-DHAS_DUNDER_RESTRICT)
    message(STATUS "Using __restrict__")
elseif (HAS___restrict)
    add_definitions(-DHAS_DASH_RESTRICT)
    message(STATUS "Using __restrict")
elseif (HAS_restrict)
    add_definitions(-DHAS_RESTRICT)
    message(STATUS "Using restrict")
else ()
    message(STATUS "No restrict keyword found, performance may be affected")
endif ()
