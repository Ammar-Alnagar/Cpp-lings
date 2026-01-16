cmake_minimum_required(VERSION 3.8)
project(CppProjects)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project executables
add_executable(task_manager task_manager.cpp)
add_executable(database_engine database_engine.cpp)
add_executable(image_processor image_processor.cpp)

# Set properties for executables
set_target_properties(task_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/projects
)
set_target_properties(database_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/projects
)
set_target_properties(image_processor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/projects
)

# Link threading library for image processor (which uses std::thread)
target_link_libraries(image_processor
    $<$<PLATFORM_ID:Linux>:pthread>
    $<$<PLATFORM_ID:Darwin>:pthread>
)

# Create individual targets for each project
add_custom_target(project1 DEPENDS task_manager)
add_custom_target(project2 DEPENDS database_engine)
add_custom_target(project3 DEPENDS image_processor)

# Installation rules
install(TARGETS task_manager database_engine image_processor
    RUNTIME DESTINATION bin
)

message(STATUS "C++ Projects CMake configured")
message(STATUS "Build options:")
message(STATUS "  - make task_manager: Build Task Manager")
message(STATUS "  - make database_engine: Build Database Engine")
message(STATUS "  - make image_processor: Build Image Processor")