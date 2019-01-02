# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("tensor_test_implementation")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Build/CMake")

set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

message(STATUS "PROJECT_BINARY_DIR: " ${PROJECT_BINARY_DIR})
message(STATUS "TBB_LIBRARY" ${TBB_LIBRARY})

link_libraries(${TBB_LIBRARY})

cmake_policy(SET CMP0074 NEW)

include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/FindTBB.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/CompileOptions.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/TaskingSystemOptions.cmake)

#include_directories(${TBB_INCLUDE_DIR})

# Include sub-projects.
add_subdirectory ("tensor_test_implementation")
