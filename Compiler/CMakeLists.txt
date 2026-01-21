cmake_minimum_required(VERSION 3.15)
project(CompilerEducation LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Theory examples (always build, no dependencies)
file(GLOB THEORY_SOURCES "theory/examples/*.cpp")
foreach(source_file ${THEORY_SOURCES})
    get_filename_component(example_name ${source_file} NAME_WE)
    set(target_name "theory_${example_name}")
    add_executable(${target_name} ${source_file})
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

# Try to find LLVM
find_package(LLVM CONFIG QUIET)

if(LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    
    # Add LLVM definitions and include directories
    include_directories(${LLVM_INCLUDE_DIRS})
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    add_definitions(${LLVM_DEFINITIONS_LIST})
    
    # Map LLVM components to libraries
    llvm_map_components_to_libnames(llvm_libs
        Core
        Support
        IRReader
        Passes
    )
    
    # LLVM examples (only if LLVM found)
    file(GLOB LLVM_SOURCES "llvm/examples/*.cpp")
    foreach(source_file ${LLVM_SOURCES})
        get_filename_component(example_name ${source_file} NAME_WE)
        set(target_name "llvm_${example_name}")
        add_executable(${target_name} ${source_file})
        target_link_libraries(${target_name} PRIVATE ${llvm_libs})
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endforeach()
    
    message(STATUS "LLVM examples will be built")
else()
    message(STATUS "LLVM not found - LLVM examples will be skipped")
    message(STATUS "To build LLVM examples:")
    message(STATUS "  - Install LLVM development packages")
    message(STATUS "  - Set LLVM_DIR to LLVMConfig.cmake location")
    message(STATUS "  Example: cmake -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm ..")
endif()

# Custom target to run all examples
add_custom_target(run_all
    COMMENT "Running all examples..."
)

# Print configuration summary
message(STATUS "")
message(STATUS "Compiler Education Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Theory Examples: ${THEORY_SOURCES}")
if(LLVM_FOUND)
    message(STATUS "  LLVM Examples: ${LLVM_SOURCES}")
    message(STATUS "  LLVM Version: ${LLVM_PACKAGE_VERSION}")
else()
    message(STATUS "  LLVM Examples: SKIPPED (LLVM not found)")
endif()
message(STATUS "")
