cmake_minimum_required(VERSION 3.15)
project(ApCpp LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find threads package for concurrency examples
find_package(Threads REQUIRED)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Concurrency examples
file(GLOB CONCURRENCY_SOURCES "concurrency/examples/*.cpp")
foreach(source_file ${CONCURRENCY_SOURCES})
    get_filename_component(example_name ${source_file} NAME_WE)
    set(target_name "concurrency_${example_name}")
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE Threads::Threads)
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

# Template examples
file(GLOB TEMPLATE_SOURCES "templates/examples/*.cpp")
foreach(source_file ${TEMPLATE_SOURCES})
    get_filename_component(example_name ${source_file} NAME_WE)
    set(target_name "template_${example_name}")
    add_executable(${target_name} ${source_file})
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

# Custom target to run all examples
add_custom_target(run_all
    COMMENT "Running all examples..."
    COMMAND ${CMAKE_COMMAND} -E echo "Running concurrency examples..."
    COMMAND for exe in ${CMAKE_BINARY_DIR}/bin/concurrency_* \; do $$exe \; done
    COMMAND ${CMAKE_COMMAND} -E echo "Running template examples..."
    COMMAND for exe in ${CMAKE_BINARY_DIR}/bin/template_* \; do $$exe \; done
    DEPENDS ${CONCURRENCY_SOURCES} ${TEMPLATE_SOURCES}
)

# Print configuration information
message(STATUS "Advanced C++ (Ap-cpp) Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Concurrency Examples: ${CONCURRENCY_SOURCES}")
message(STATUS "  Template Examples: ${TEMPLATE_SOURCES}")
