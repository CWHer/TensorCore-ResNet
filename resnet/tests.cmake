# Add all unit tests here
set(TEST_SOURCE
    tests/test_init.cpp
    tests/functional/gemm.cpp
    tests/functional/im2col.cpp
    tests/functional/conv2d.cpp
    tests/test_tensor.cpp
    tests/test_tensor_ops.cpp
    tests/test_dataset.cpp
    tests/modules/batchnorm.cpp
    tests/modules/conv.cpp
    tests/modules/test_pooling.cpp
    tests/modules/linear.cpp
    tests/modules/resnet.cpp
)

set(SIMULATOR_TEST_SOURCE
    tests/simulator/test_device_memory.cpp
    tests/simulator/test_half.cpp
    tests/simulator/test_regfile.cpp
    tests/simulator/test_functions.cpp
    tests/simulator/test_simulator.cpp
)
