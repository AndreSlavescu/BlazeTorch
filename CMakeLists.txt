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

add_subdirectory(submodules/nvfuser)

add_library(
  blazetorch SHARED 
  ${BLAZETORCH_SOURCES}
)

target_link_libraries(blazetorch "${TORCH_LIBRARIES}" nvfuser_codegen flatbuffers)
set_property(TARGET blazetorch PROPERTY CXX_STANDARD 17)

include_directories(${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} submodules/nvfuser ${pybind11_INCLUDE_DIRS} ${FLATBUFFERS_INCLUDE_DIRS})

message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "pybind11_DIR: ${pybind11_DIR}")
message(STATUS "FLATBUFFERS_INCLUDE_DIRS: ${FLATBUFFERS_INCLUDE_DIRS}")
message(STATUS "NVFUSER_INCLUDE_DIRS: ${CMAKE_SOURCE_DIR}/submodules/nvfuser")