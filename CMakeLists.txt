cmake_minimum_required(VERSION 3.10)
project(tensorcore-simulator VERSION 0.0.1 LANGUAGES CXX)

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

# ################################ Executables ##################################
# TODO: add library support
include(sources.cmake)

# add_executable(main ${SOURCE_FILES} src/gemm.cpp)

# ################################## Tests ######################################
include(tests.cmake)

find_package(GTest)

if(GTest_FOUND)
    message(STATUS "GTest Found, you can use ctest to run all tests.")
    enable_testing()
    add_executable(gtest_exc ${TEST_SOURCE} ${SOURCE_FILES})

    target_link_libraries(gtest_exc gtest gtest_main pthread)

    gtest_discover_tests(gtest_exc)
else()
    message(STATUS "If you installed GTest, you can use ctest to access tests.")
endif()
