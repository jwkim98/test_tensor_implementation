
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

cmake_policy(SET CMP0074 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Build/CMake")

set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

message("Output binary dir: " ${PROJECT_BINARY_DIR})

include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMake/FindTBB.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/CompileOptions.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Build/CMAKE/BackendSettings.cmake)

include(CTest)

message(STATUS "Default Backend: " ${DEFAULT_BACKEND})

add_subdirectory("src")



