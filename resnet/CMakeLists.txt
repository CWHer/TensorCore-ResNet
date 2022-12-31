cmake_minimum_required(VERSION 3.20)
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

set(CUDA_EXTRA_LIB cudart)

# Set CUDA architecture to Volta (to prevent CUDA printing errors)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Set CUDA architecture flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=${CMAKE_CUDA_ARCHITECTURES})

# Add CUDA_EXTRA_LIB to testing
set(TEST_ADDITIONAL_LIB ${TEST_ADDITIONAL_LIB} ${CUDA_EXTRA_LIB})

# ################################ Executables ##################################
include(sources.cmake)
include(preprocess.cmake)

# Build source files to a static library to avoid recompiling
add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

add_executable(main resnet_inference.cpp)
target_link_libraries(main ${CUDA_EXTRA_LIB} ${PROJECT_NAME})

# ################################## Tests ######################################
include(tests.cmake)

find_package(GTest)

if(GTest_FOUND)
    message(STATUS "GTest Found, you can use ctest to run all tests.")
    enable_testing()
    add_executable(gtest_exc ${TEST_SOURCE})

    target_compile_definitions(gtest_exc PRIVATE ${TEST_ADDITIONAL_DEFINITIONS})

    target_link_libraries(gtest_exc
        ${PROJECT_NAME}
        GTest::gtest GTest::gtest_main
        ${TEST_ADDITIONAL_LIB}
    )

    gtest_discover_tests(gtest_exc)
else()
    message(STATUS "If you installed GTest, you can use ctest to access tests.")
endif()
