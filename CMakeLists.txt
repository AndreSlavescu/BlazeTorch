cmake_minimum_required(VERSION 3.15)
project(blazetorch)

# Add the cmake/Modules directory to CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/submodules/nvfuser/cmake/Modules)

# Find the Python interpreter
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))"
  OUTPUT_VARIABLE Torch_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Torch_DIR} ${pybind11_DIR} $ENV{HOME}/local)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Flatbuffers REQUIRED)

file(GLOB_RECURSE BLAZETORCH_SOURCES "blazetorch/*.cc" "blazetorch/fusion_passes/*.cc")
message(STATUS "BlazeTorch sources:")
foreach(source ${BLAZETORCH_SOURCES})
  message(STATUS "  ${source}")
endforeach()

# Define the blazetorch target
add_library(blazetorch SHARED
    blazetorch/blazetorch_compiler.cc
    blazetorch/compiler.cc
    blazetorch/fuse_candidates.cc
    blazetorch/fusion_passes/fusion_adddiv.cc
    blazetorch/fusion_passes/fusion_addmul.cc
    blazetorch/process_graph_ops.cc
    blazetorch/register.cc
)

# Link libraries to the blazetorch target
target_link_libraries(blazetorch
    PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    pybind11::module
)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

option(BUILD_WITH_NVFUSER "Build with nvfuser" ON)

if(BUILD_WITH_NVFUSER)
    add_subdirectory(submodules/nvfuser)
    target_link_libraries(blazetorch PRIVATE nvfuser_codegen flatbuffers)
    include_directories(submodules/nvfuser ${FLATBUFFERS_INCLUDE_DIRS})
    message(STATUS "Building with nvfuser")
else()
    message(STATUS "Building without nvfuser")
endif()

message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "pybind11_DIR: ${pybind11_DIR}")
message(STATUS "FLATBUFFERS_INCLUDE_DIRS: ${FLATBUFFERS_INCLUDE_DIRS}")
message(STATUS "NVFUSER_INCLUDE_DIRS: ${CMAKE_SOURCE_DIR}/submodules/nvfuser")