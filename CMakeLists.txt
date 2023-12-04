cmake_minimum_required(VERSION 3.23)

project(fft-implementation-cpp LANGUAGES CUDA CXX)

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src)
add_subdirectory(executable)


