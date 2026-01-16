cmake_minimum_required(VERSION 3.8)
project(CppLings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add subdirectories
add_subdirectory(projects)

# Define installation rules
install(DIRECTORY chapters/
    DESTINATION docs/chapters
    FILES_MATCHING PATTERN "*.md"
)

install(DIRECTORY projects/
    DESTINATION src/projects
    FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h" PATTERN "*.hpp"
)

# Create a custom target for documentation
add_custom_target(docs
    COMMAND echo "Documentation available in chapters/ directory"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Create a target to build all exercises
add_custom_target(exercises
    COMMAND echo "Exercises available in exercises/ directory"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

message(STATUS "CppLings C++ Learning Curriculum CMake configured")
message(STATUS "Available targets:")
message(STATUS "  - make: Build all projects")
message(STATUS "  - make docs: Show documentation info")
message(STATUS "  - make exercises: Show exercises info")
message(STATUS "  - make install: Install curriculum files")