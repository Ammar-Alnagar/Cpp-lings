# Makefile for Compiler Educational Materials
# Builds theory examples and LLVM examples

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -O2

# LLVM Configuration
LLVM_CONFIG = llvm-config
LLVM_CXXFLAGS = $(shell $(LLVM_CONFIG) --cxxflags 2>/dev/null)
LLVM_LDFLAGS = $(shell $(LLVM_CONFIG) --ldflags --system-libs --libs core support 2>/dev/null)

# Check if LLVM is available
HAS_LLVM = $(shell command -v $(LLVM_CONFIG) 2>/dev/null)

# Directories
THEORY_DIR = theory/examples
LLVM_DIR = llvm/examples
BIN_DIR = bin

# Theory examples (no LLVM required)
THEORY_SOURCES = $(wildcard $(THEORY_DIR)/*.cpp)
THEORY_TARGETS = $(patsubst $(THEORY_DIR)/%.cpp,$(BIN_DIR)/theory_%,$(THEORY_SOURCES))

# LLVM examples (requires LLVM)
ifdef HAS_LLVM
LLVM_SOURCES = $(wildcard $(LLVM_DIR)/*.cpp)
LLVM_TARGETS = $(patsubst $(LLVM_DIR)/%.cpp,$(BIN_DIR)/llvm_%,$(LLVM_SOURCES))
else
LLVM_SOURCES =
LLVM_TARGETS =
endif

# All targets
ALL_TARGETS = $(THEORY_TARGETS) $(LLVM_TARGETS)

.PHONY: all clean theory llvm help check-llvm

all: check-llvm $(BIN_DIR) $(ALL_TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Theory examples (standard C++ compilation)
$(BIN_DIR)/theory_%: $(THEORY_DIR)/%.cpp | $(BIN_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "Built: $@"

# LLVM examples (requires LLVM)
ifdef HAS_LLVM
$(BIN_DIR)/llvm_%: $(LLVM_DIR)/%.cpp | $(BIN_DIR)
	@echo "Compiling $< with LLVM..."
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $< $(LLVM_LDFLAGS) -o $@
	@echo "Built: $@"
endif

# Build only theory examples
theory: $(BIN_DIR) $(THEORY_TARGETS)
	@echo "Theory examples built successfully"

# Build only LLVM examples
llvm: check-llvm $(BIN_DIR) $(LLVM_TARGETS)
	@echo "LLVM examples built successfully"

# Check LLVM availability
check-llvm:
ifndef HAS_LLVM
	@echo "================================================"
	@echo "WARNING: LLVM not found!"
	@echo "llvm-config not in PATH"
	@echo ""
	@echo "Theory examples will be built."
	@echo "LLVM examples will be skipped."
	@echo ""
	@echo "To install LLVM:"
	@echo "  Ubuntu/Debian: sudo apt install llvm-dev"
	@echo "  macOS: brew install llvm"
	@echo "  Arch: sudo pacman -S llvm"
	@echo "================================================"
	@echo ""
endif

# Run all available examples
run-all: all
	@echo "\n=== Running Theory Examples ==="
	@for exe in $(THEORY_TARGETS); do \
		if [ -f $$exe ]; then \
			echo "\n>>> Running $$exe <<<"; \
			$$exe; \
		fi \
	done
ifdef HAS_LLVM
	@echo "\n=== Running LLVM Examples ==="
	@for exe in $(LLVM_TARGETS); do \
		if [ -f $$exe ]; then \
			echo "\n>>> Running $$exe <<<"; \
			$$exe; \
		fi \
	done
endif

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "Compiler Educational Materials - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all available examples (default)"
	@echo "  theory      - Build only compiler theory examples"
	@echo "  llvm        - Build only LLVM examples (requires LLVM)"
	@echo "  run-all     - Build and run all examples"
	@echo "  clean       - Remove all build artifacts"
	@echo "  check-llvm  - Check if LLVM is available"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - C++17 compiler (g++, clang++)"
	@echo "  - LLVM 14+ (optional, for LLVM examples)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build everything available"
	@echo "  make theory             # Build theory examples only"
	@echo "  make llvm               # Build LLVM examples"
	@echo "  make run-all            # Build and run all"
	@echo ""
ifdef HAS_LLVM
	@echo "LLVM Status: FOUND ($(shell $(LLVM_CONFIG) --version))"
else
	@echo "LLVM Status: NOT FOUND"
endif
