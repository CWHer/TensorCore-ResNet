cmake_minimum_required(VERSION 3.11)
project(tensorcore-resnet VERSION 0.0.1 LANGUAGES CXX CUDA)

# ############################### C++ Options ###################################

# Set C++ standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the default build type to Release
set(default_build_type "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release")
endif ()

message(STATUS "Current build type is: ${CMAKE_BUILD_TYPE}.")

# Optimization options
add_compile_options(
        # Release options for demonstration: Ofast
        "$<$<CONFIG:RELEASE>:$<$<COMPILE_LANGUAGE:CXX>:-Ofast>>"
        "$<$<CONFIG:RELEASE>:$<$<COMPILE_LANGUAGE:CUDA>:-O3>>"
        "$<$<CONFIG:RELEASE>:$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Ofast>>"

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

# Set RANDOM_SEED to 42
set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};RANDOM_SEED=42")

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

option(DEBUG_ONE_ROUND "Only run one round of inference" OFF)

# Build source files to a static library to avoid recompiling
add_library(${PROJECT_NAME}_base STATIC ${SOURCE_FILES})
add_library(${PROJECT_NAME}_gemm STATIC ${SOURCE_GEMM})
add_library(${PROJECT_NAME}_simulator STATIC ${SIMULATOR_SOURCE_FILES})
add_library(${PROJECT_NAME}_simulator_gemm STATIC ${SIMULATOR_GEMM})

set_property(TARGET ${PROJECT_NAME}_simulator PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME}_simulator_gemm PROPERTY CXX_STANDARD 17)
target_link_libraries(${PROJECT_NAME}_simulator_gemm ${PROJECT_NAME}_simulator)

add_executable(main src/resnet_inference.cpp)
add_executable(main_simulator src/resnet_inference.cpp)
target_compile_definitions(main_simulator PRIVATE SIMULATE)

target_compile_definitions(main PRIVATE ${DATASET_DEFINITIONS})
target_compile_definitions(main_simulator PRIVATE ${DATASET_DEFINITIONS})

if (DEBUG_ONE_ROUND)
    message(STATUS "Debugging one round of inference")
    target_compile_definitions(main PRIVATE DEBUG_ONE_ROUND)
endif ()

target_link_libraries(main
        ${CUDA_EXTRA_LIB}
        ${PROJECT_NAME}_base
        ${PROJECT_NAME}_gemm)

target_link_libraries(main_simulator
        ${CUDA_EXTRA_LIB}
        ${PROJECT_NAME}_base
        ${PROJECT_NAME}_simulator_gemm)


# ################################### Docs ######################################

include(doxygen.cmake)

# ################################## Tests ######################################
include(tests.cmake)

message(CHECK_START "Finding Google Test")

find_package(GTest)

if (GTest_FOUND)
    message(CHECK_PASS "Found")
    message(STATUS "GTest Found, you can use ctest to run all tests.")
    enable_testing()
    add_executable(gtest_main ${TEST_SOURCE})

    add_library(${PROJECT_NAME}_simulator_test OBJECT ${SIMULATOR_TEST_SOURCE})
    set_property(TARGET ${PROJECT_NAME}_simulator_test PROPERTY CXX_STANDARD 17)

    target_compile_definitions(gtest_main PRIVATE ${DATASET_DEFINITIONS})
    target_compile_definitions(gtest_main PRIVATE ${TEST_ADDITIONAL_DEFINITIONS})

    target_link_libraries(gtest_main
            ${PROJECT_NAME}_simulator_test
            ${PROJECT_NAME}_base
            ${PROJECT_NAME}_gemm
            ${PROJECT_NAME}_simulator
            GTest::gtest GTest::gtest_main
            ${TEST_ADDITIONAL_LIB}
            Threads::Threads
            )

    gtest_discover_tests(gtest_main)
else ()
    message(CHECK_FAIL "Not Found")
    message(STATUS "If you installed GTest, you can use ctest to access tests.")
endif ()
