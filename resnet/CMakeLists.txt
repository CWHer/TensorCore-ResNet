cmake_minimum_required(VERSION 3.11)
project(tensorcore-resnet VERSION 0.0.1 LANGUAGES CXX CUDA)

# ############################### C++ Options ###################################

# Set C++ standards
set(CMAKE_CXX_STANDARD 14)
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
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_EXTRA_LIB cudart)

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    link_directories(${CUDAToolkit_LIBRARY_DIR})

    # Not specifying them in original lib to separate the compilation results
    set(TEST_ADDITIONAL_LIB "${TEST_ADDITIONAL_LIB};cublas;curand")

    # Run CUBLAS tests
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};WITH_CUBLAS")
else ()
    message(STATUS "CUDA Toolkit not found, CUDA Toolkit related tests will not be available")
endif ()

set(ORIGINAL_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

# HACK testing: Torch library might have CUDA library conflict.
find_package(Torch)

if (Torch_FOUND)
    if (NOT "${ORIGINAL_CUDA_NVCC_FLAGS}" EQUAL "${CUDA_NVCC_FLAGS}")
        message(STATUS "Found libtorch built with CUDA, using torch's CUDA configurations")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${TORCH_CUDA_FLAGS})
    else ()
        message(STATUS "Found libtorch built without CUDA, use our own CUDA configurations")
        # Set CUDA architecture to Volta (to prevent CUDA printing errors)
        set(CMAKE_CUDA_ARCHITECTURES 70)

        # Set CUDA architecture flags
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=${CMAKE_CUDA_ARCHITECTURES}")
        # Add CUDA_EXTRA_LIB to testing
        set(TEST_ADDITIONAL_LIB ${TEST_ADDITIONAL_LIB} ${CUDA_EXTRA_LIB})
    endif ()
    # Use #if WITH_TORCH to enable torch related code in testing.
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};WITH_TORCH")
    include_directories(${TORCH_INCLUDE_DIRS})
    # If using Torch, cublas should not be additionally linked
    # as libtorch_cuda.so already have them linked.
    set(TEST_ADDITIONAL_LIB ${TEST_ADDITIONAL_LIB} ${TORCH_LIBRARIES})
else ()
    # Set CUDA architecture to Volta (to prevent CUDA printing errors)
    set(CMAKE_CUDA_ARCHITECTURES 70)

    # Set CUDA architecture flags
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=${CMAKE_CUDA_ARCHITECTURES}")
    # Add CUDA_EXTRA_LIB to testing
    set(TEST_ADDITIONAL_LIB ${TEST_ADDITIONAL_LIB} ${CUDA_EXTRA_LIB})
endif ()


# ################################ Executables ##################################
include(sources.cmake)
include(dataset.cmake)

# Build source files to a static library to avoid recompiling
add_library(${PROJECT_NAME}_base STATIC ${SOURCE_FILES})

add_executable(main resnet_inference.cpp)
target_link_libraries(main ${CUDA_EXTRA_LIB} ${PROJECT_NAME}_base)

# ################################## Tests ######################################
include(tests.cmake)

find_package(GTest)

if (GTest_FOUND)
    message(STATUS "GTest Found, you can use ctest to run all tests.")
    enable_testing()
    add_executable(gtest_exc ${TEST_SOURCE})

    target_compile_definitions(gtest_exc PRIVATE ${TEST_ADDITIONAL_DEFINITIONS})

    target_link_libraries(gtest_exc
            ${PROJECT_NAME}_base
            GTest::gtest GTest::gtest_main
            ${TEST_ADDITIONAL_LIB}
            )

    gtest_discover_tests(gtest_exc)
else ()
    message(STATUS "If you installed GTest, you can use ctest to access tests.")
endif ()
