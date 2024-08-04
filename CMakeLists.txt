cmake_minimum_required(VERSION 3.15)
project(blazetorch)

# Find the Python interpreter
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))"
  OUTPUT_VARIABLE Torch_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Torch_DIR})

find_package(Torch REQUIRED)
file(GLOB_RECURSE BLAZETORCH_SOURCES "blazetorch/*.cc" "blazetorch/fusion_passes/*.cc")
message(STATUS "BlazeTorch sources:")
foreach(source ${BLAZETORCH_SOURCES})
  message(STATUS "  ${source}")
endforeach()

add_library(
  blazetorch SHARED 
  ${BLAZETORCH_SOURCES}
)

target_link_libraries(blazetorch "${TORCH_LIBRARIES}")
set_property(TARGET blazetorch PROPERTY CXX_STANDARD 17)

include_directories(${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
