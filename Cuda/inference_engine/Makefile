# Compiler
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++11 -O3 -I./include -Xcompiler -fno-gnu-unique
LDFLAGS = -lcublas -lcudart

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/inference_engine

# Default target
.PHONY: all clean

all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files to objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Create directories
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Run the executable
run: $(TARGET)
	./$(TARGET)