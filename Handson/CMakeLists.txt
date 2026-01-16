cmake_minimum_required(VERSION 3.25)
project(HPCToolkit LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CPU_SOURCES
    src/cpu_matrix.cpp
)

set(GPU_SOURCES
    src/gpu_vector.cu
)

set(EXAMPLE_SOURCES
    examples/example1_basics.cpp
)

add_library(hpc_toolkit_cpu SHARED ${CPU_SOURCES})
add_library(hpc_toolkit_gpu SHARED ${GPU_SOURCES})

foreach(example ${EXAMPLE_SOURCES})
    get_filename_component(exe_name ${example} NAME_WE)
    add_executable(${exe_name} ${example})
    target_link_libraries(${exe_name} hpc_toolkit_cpu hpc_toolkit_gpu)
endforeach()
