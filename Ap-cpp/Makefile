# Makefile for Advanced C++ (Ap-cpp) Educational Materials
# Compiles all examples for concurrency and template programming

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O2
PTHREAD_FLAGS = -pthread

# Directories
CONCURRENCY_DIR = concurrency/examples
TEMPLATES_DIR = templates/examples
BIN_DIR = bin

# Concurrency examples (require pthread)
CONCURRENCY_SOURCES = $(wildcard $(CONCURRENCY_DIR)/*.cpp)
CONCURRENCY_TARGETS = $(patsubst $(CONCURRENCY_DIR)/%.cpp,$(BIN_DIR)/concurrency_%,$(CONCURRENCY_SOURCES))

# Template examples
TEMPLATE_SOURCES = $(wildcard $(TEMPLATES_DIR)/*.cpp)
TEMPLATE_TARGETS = $(patsubst $(TEMPLATES_DIR)/%.cpp,$(BIN_DIR)/template_%,$(TEMPLATE_SOURCES))

# All targets
ALL_TARGETS = $(CONCURRENCY_TARGETS) $(TEMPLATE_TARGETS)

.PHONY: all clean concurrency templates help run-all

all: $(BIN_DIR) $(ALL_TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Concurrency examples compilation (with pthread)
$(BIN_DIR)/concurrency_%: $(CONCURRENCY_DIR)/%.cpp | $(BIN_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(PTHREAD_FLAGS) $< -o $@
	@echo "Built: $@"

# Template examples compilation
$(BIN_DIR)/template_%: $(TEMPLATES_DIR)/%.cpp | $(BIN_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "Built: $@"

# Build only concurrency examples
concurrency: $(BIN_DIR) $(CONCURRENCY_TARGETS)
	@echo "All concurrency examples built successfully"

# Build only template examples
templates: $(BIN_DIR) $(TEMPLATE_TARGETS)
	@echo "All template examples built successfully"

# Run all examples
run-all: all
	@echo "\n=== Running Concurrency Examples ==="
	@for exe in $(CONCURRENCY_TARGETS); do \
		if [ -f $$exe ]; then \
			echo "\n>>> Running $$exe <<<"; \
			$$exe; \
		fi \
	done
	@echo "\n=== Running Template Examples ==="
	@for exe in $(TEMPLATE_TARGETS); do \
		if [ -f $$exe ]; then \
			echo "\n>>> Running $$exe <<<"; \
			$$exe; \
		fi \
	done

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "Advanced C++ (Ap-cpp) Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all examples (default)"
	@echo "  concurrency - Build only concurrency examples"
	@echo "  templates   - Build only template examples"
	@echo "  run-all     - Build and run all examples"
	@echo "  clean       - Remove all build artifacts"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build everything"
	@echo "  make concurrency        # Build concurrency examples only"
	@echo "  make templates          # Build template examples only"
	@echo "  make run-all            # Build and run all examples"
	@echo ""
	@echo "Individual examples can be run from the bin/ directory"
