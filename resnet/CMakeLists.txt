cmake_minimum_required(VERSION 3.10)
project(tensorcore-resnet VERSION 0.0.1 LANGUAGES CXX CUDA)

# ############################### C++ Options ###################################

# Set C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization options
add_compile_options(

    # Release options for demonstration: Ofast
    "$<$<CONFIG:RELEASE>:-Ofast>"

    # Debugging options: no optimization, debug symbols
    "$<$<CONFIG:DEBUG>:-O0>"
    "$<$<CONFIG:DEBUG>:-g>"
)

# ######################### CUDA Related Thingy #################################

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture to Volta (to prevent CUDA printing errors)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Set CUDA architecture flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=${CMAKE_CUDA_ARCHITECTURES})

# Use cuBLAS library requires this
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_directories(${CUDAToolkit_LIBRARY_DIR})

# Specify extra CUDA libraries
set(CUDA_EXTRA_LIB
    cublas
    curand
    cudart
)

# ################################ Executables ##################################
include(sources.cmake)

add_executable(main ${SOURCE_FILES} resnet_inference.cpp)
target_link_libraries(main ${CUDA_EXTRA_LIB})

# ################################## Tests ######################################
include(tests.cmake)

find_package(GTest)

if(GTest_FOUND)
    message(STATUS "GTest Found, you can use ctest to run all tests.")
    enable_testing()
    add_executable(gtest_exc ${TEST_SOURCE} ${SOURCE_FILES})

    target_link_libraries(gtest_exc
        GTest::gtest GTest::gtest_main
        ${CUDA_EXTRA_LIB}
    )

    gtest_discover_tests(gtest_exc)
else()
    message(STATUS "If you installed GTest, you can use ctest to access tests.")
endif()
