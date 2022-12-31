# Include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules)

# ... Add more if you need them

# Common source files **without** main function
set(SOURCE_FILES
    src/functional/ops.cu
    src/functional/batchnorm.cu
    src/functional/pooling.cu
    src/functional/gemm.cu
    src/functional/linear.cu
    src/common.cpp
    src/mem_cache.cpp
)
