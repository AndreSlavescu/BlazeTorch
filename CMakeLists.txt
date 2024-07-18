cmake_minimum_required(VERSION 3.15)
project(blazetorch)

# Set Torch_DIR to the directory containing TorchConfig.cmake
set(Torch_DIR /home/aslavescu/miniconda3/lib/python3.11/site-packages/torch/share/cmake/Torch)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Torch_DIR})

find_package(Torch REQUIRED)

# Specify CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(blazetorch SHARED register.cc)
target_link_libraries(blazetorch "${TORCH_LIBRARIES}")
set_property(TARGET blazetorch PROPERTY CXX_STANDARD 17)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})
