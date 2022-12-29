cmake_minimum_required(VERSION 3.11)
project(tensorcore-resnet VERSION 0.0.1 LANGUAGES CXX CUDA)

# ############################### C++ Options ###################################

# Set C++ standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization options
add_compile_options(

    # Release options for demonstration: Ofast
    "$<$<CONFIG:RELEASE>:$<$<COMPILE_LANGUAGE:CXX>:-Ofast>>"
    "$<$<CONFIG:RELEASE>:$<$<COMPILE_LANGUAGE:CUDA>:-O3>>"

    # Debugging options: no optimization, debug symbols
    "$<$<CONFIG:DEBUG>:-O0>"
    "$<$<CONFIG:DEBUG>:-g>"
    "$<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>:-G>>"
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
    if (NOT "${ORIGINAL_CUDA_NVCC_FLAGS}" STREQUAL "${CUDA_NVCC_FLAGS}")
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

find_package(Threads REQUIRED)


# ################################ Executables ##################################
include(sources.cmake)
include(dataset.cmake)

# Build source files to a static library to avoid recompiling
add_library(${PROJECT_NAME}_base STATIC ${SOURCE_FILES})
add_library(${PROJECT_NAME}_gemm STATIC ${SOURCE_GEMM})
add_library(${PROJECT_NAME}_simulator STATIC ${SIMULATOR_SOURCE_FILES})
add_library(${PROJECT_NAME}_simulator_gemm STATIC ${SIMULATOR_GEMM})

set_property(TARGET ${PROJECT_NAME}_simulator PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME}_simulator_gemm PROPERTY CXX_STANDARD 17)
target_link_libraries(${PROJECT_NAME}_simulator_gemm ${PROJECT_NAME}_simulator)

add_executable(main resnet_inference.cpp)
add_executable(main_simulator resnet_inference.cpp)
target_compile_definitions(main_simulator PRIVATE SIMULATE)

target_compile_definitions(main PRIVATE ${DATASET_DEFINITIONS})
target_compile_definitions(main_simulator PRIVATE ${DATASET_DEFINITIONS})

target_link_libraries(main ${CUDA_EXTRA_LIB} ${PROJECT_NAME}_base ${PROJECT_NAME}_gemm)
target_link_libraries(main_simulator ${CUDA_EXTRA_LIB} ${PROJECT_NAME}_base ${PROJECT_NAME}_simulator_gemm)

# ################################## Tests ######################################
include(tests.cmake)

find_package(GTest)

if (GTest_FOUND)
    message(STATUS "GTest Found, you can use ctest to run all tests.")
    enable_testing()
    add_executable(gtest_main ${TEST_SOURCE})
    target_compile_definitions(gtest_main PRIVATE ${DATASET_DEFINITIONS})
    target_compile_definitions(gtest_main PRIVATE ${TEST_ADDITIONAL_DEFINITIONS})

    target_link_libraries(gtest_main
            ${PROJECT_NAME}_base
            ${PROJECT_NAME}_gemm
            GTest::gtest GTest::gtest_main
            ${TEST_ADDITIONAL_LIB}
            )

    add_executable(gtest_sim ${SIMULATOR_TEST_SOURCE})
    set_property(TARGET gtest_sim PROPERTY CXX_STANDARD 17)
    target_link_libraries(gtest_sim
            ${PROJECT_NAME}_simulator
            GTest::gtest GTest::gtest_main
            Threads::Threads
            )

    gtest_discover_tests(gtest_main)
    gtest_discover_tests(gtest_sim)
else ()
    message(STATUS "If you installed GTest, you can use ctest to access tests.")
endif ()
