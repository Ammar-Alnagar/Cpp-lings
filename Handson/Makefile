CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++20 -Wall -Wextra -O3
NVCCFLAGS = -std=c++20 -O3 -lineinfo
LDFLAGS = -lcuda -lcudart

INCLUDE_DIR = include
SRC_DIR = src
BUILD_DIR = build
EXAMPLE_DIR = examples

CPU_SOURCES = $(SRC_DIR)/cpu_matrix.cpp
GPU_SOURCES = $(SRC_DIR)/gpu_vector.cu

CPU_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPU_SOURCES))
GPU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(GPU_SOURCES))

EXAMPLES = example1_basics

.PHONY: all clean directories

all: directories $(EXAMPLES)

directories:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

example1_basics: $(BUILD_DIR)/example1_basics.o $(CPU_OBJECTS) $(GPU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/example1_basics.o: $(EXAMPLE_DIR)/example1_basics.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(EXAMPLES)

run: example1_basics
	./example1_basics

help:
	@echo "Available targets:"
	@echo "  all      - Build all examples"
	@echo "  clean    - Remove build artifacts"
	@echo "  run      - Build and run example1"
	@echo "  help     - Show this help message"
