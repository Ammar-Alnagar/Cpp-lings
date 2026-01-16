# Makefile for C++ Learning Curriculum

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
BUILD_DIR = build

# Default target
all: build_projects

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build all projects using CMake
build_projects: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

# Build individual projects
build_project1: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make task_manager

build_project2: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make database_engine

build_project3: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make image_processor

# Run projects
run_project1: build_project1
	$(BUILD_DIR)/projects/task_manager

run_project2: build_project2
	$(BUILD_DIR)/projects/database_engine

run_project3: build_project3
	$(BUILD_DIR)/projects/image_processor

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Install
install:
	cd $(BUILD_DIR) && cmake .. && make install

# Show help
help:
	@echo "Available targets:"
	@echo "  all              - Build all projects"
	@echo "  build_project1   - Build Task Manager project"
	@echo "  build_project2   - Build Database Engine project"
	@echo "  build_project3   - Build Image Processor project"
	@echo "  run_project1     - Build and run Task Manager"
	@echo "  run_project2     - Build and run Database Engine"
	@echo "  run_project3     - Build and run Image Processor"
	@echo "  clean            - Remove build directory"
	@echo "  install          - Install curriculum files"
	@echo "  help             - Show this help message"

.PHONY: all build_projects build_project1 build_project2 build_project3 run_project1 run_project2 run_project3 clean install help