cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_resnet)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(resnet resnet.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}")
set_property(TARGET resnet PROPERTY CXX_STANDARD 14)