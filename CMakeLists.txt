cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_COMPILER "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/bin/gcc")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(fft-implementation-cpp LANGUAGES CUDA CXX)

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_FLAGS -Xcompiler=-fopenmp)

add_subdirectory(src)
add_subdirectory(executable)


