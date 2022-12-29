# Add all unit tests here
set(TEST_SOURCE
    tests/test_init.cu
    tests/gemm.cpp
    tests/im2col.cpp
    tests/conv2d.cpp
    tests/test_tensor.cpp
    tests/test_tensor_ops.cpp
    tests/test_dataset.cpp
    tests/test_batchnorm.cpp
    tests/tensor_conv2d.cpp
    tests/test_pooling.cpp
    tests/linear.cpp
    tests/resnet.cpp
)

set(SIMULATOR_TEST_SOURCE
    tests/simulator/test_device_memory.cpp
    tests/simulator/test_half.cpp
    tests/simulator/test_regfile.cpp
    tests/simulator/test_functions.cpp
    tests/simulator/test_simulator.cpp
)
