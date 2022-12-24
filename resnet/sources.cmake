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
        src/functional/linear.cu
)
