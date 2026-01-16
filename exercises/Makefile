# Makefile for C++ Learning Exercises
#
# This Makefile provides targets to build and run exercises from the C++ curriculum.
# Each chapter has its own build target.

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
BUILD_DIR = build
EXERCISES_DIR = .

# Define exercises for each chapter
CH0_EXERCISES = $(wildcard ch0/*.cpp)
CH1_EXERCISES = $(wildcard ch1/*.cpp)
CH2_EXERCISES = $(wildcard ch2/*.cpp)
CH3_EXERCISES = $(wildcard ch3/*.cpp)
CH4_EXERCISES = $(wildcard ch4/*.cpp)
CH5_EXERCISES = $(wildcard ch5/*.cpp)
CH6_EXERCISES = $(wildcard ch6/*.cpp)
CH7_EXERCISES = $(wildcard ch7/*.cpp)
CH8_EXERCISES = $(wildcard ch8/*.cpp)
CH9_EXERCISES = $(wildcard ch9/*.cpp)
CH10_EXERCISES = $(wildcard ch10/*.cpp)
CH11_EXERCISES = $(wildcard ch11/*.cpp)
CH12_EXERCISES = $(wildcard ch12/*.cpp)
CH13_EXERCISES = $(wildcard ch13/*.cpp)
CH14_EXERCISES = $(wildcard ch14/*.cpp)
CH15_EXERCISES = $(wildcard ch15/*.cpp)

# Extract executable names from source files
CH0_TARGETS = $(patsubst ch0/%.cpp,$(BUILD_DIR)/ch0_%,$(CH0_EXERCISES))
CH1_TARGETS = $(patsubst ch1/%.cpp,$(BUILD_DIR)/ch1_%,$(CH1_EXERCISES))
CH2_TARGETS = $(patsubst ch2/%.cpp,$(BUILD_DIR)/ch2_%,$(CH2_EXERCISES))
CH3_TARGETS = $(patsubst ch3/%.cpp,$(BUILD_DIR)/ch3_%,$(CH3_EXERCISES))
CH4_TARGETS = $(patsubst ch4/%.cpp,$(BUILD_DIR)/ch4_%,$(CH4_EXERCISES))
CH5_TARGETS = $(patsubst ch5/%.cpp,$(BUILD_DIR)/ch5_%,$(CH5_EXERCISES))
CH6_TARGETS = $(patsubst ch6/%.cpp,$(BUILD_DIR)/ch6_%,$(CH6_EXERCISES))
CH7_TARGETS = $(patsubst ch7/%.cpp,$(BUILD_DIR)/ch7_%,$(CH7_EXERCISES))
CH8_TARGETS = $(patsubst ch8/%.cpp,$(BUILD_DIR)/ch8_%,$(CH8_EXERCISES))
CH9_TARGETS = $(patsubst ch9/%.cpp,$(BUILD_DIR)/ch9_%,$(CH9_EXERCISES))
CH10_TARGETS = $(patsubst ch10/%.cpp,$(BUILD_DIR)/ch10_%,$(CH10_EXERCISES))
CH11_TARGETS = $(patsubst ch11/%.cpp,$(BUILD_DIR)/ch11_%,$(CH11_EXERCISES))
CH12_TARGETS = $(patsubst ch12/%.cpp,$(BUILD_DIR)/ch12_%,$(CH12_EXERCISES))
CH13_TARGETS = $(patsubst ch13/%.cpp,$(BUILD_DIR)/ch13_%,$(CH13_EXERCISES))
CH14_TARGETS = $(patsubst ch14/%.cpp,$(BUILD_DIR)/ch14_%,$(CH14_EXERCISES))
CH15_TARGETS = $(patsubst ch15/%.cpp,$(BUILD_DIR)/ch15_%,$(CH15_EXERCISES))

# Default target - build all exercises
all: $(BUILD_DIR) $(CH0_TARGETS) $(CH1_TARGETS) $(CH2_TARGETS) $(CH3_TARGETS) \
     $(CH4_TARGETS) $(CH5_TARGETS) $(CH6_TARGETS) $(CH7_TARGETS) \
     $(CH8_TARGETS) $(CH9_TARGETS) $(CH10_TARGETS) $(CH11_TARGETS) \
     $(CH12_TARGETS) $(CH13_TARGETS) $(CH14_TARGETS) $(CH15_TARGETS)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Chapter 0 exercises
$(CH0_TARGETS): $(BUILD_DIR)/ch0_%: ch0/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 1 exercises
$(CH1_TARGETS): $(BUILD_DIR)/ch1_%: ch1/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 2 exercises
$(CH2_TARGETS): $(BUILD_DIR)/ch2_%: ch2/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 3 exercises
$(CH3_TARGETS): $(BUILD_DIR)/ch3_%: ch3/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 4 exercises
$(CH4_TARGETS): $(BUILD_DIR)/ch4_%: ch4/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 5 exercises
$(CH5_TARGETS): $(BUILD_DIR)/ch5_%: ch5/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 6 exercises
$(CH6_TARGETS): $(BUILD_DIR)/ch6_%: ch6/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 7 exercises
$(CH7_TARGETS): $(BUILD_DIR)/ch7_%: ch7/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 8 exercises
$(CH8_TARGETS): $(BUILD_DIR)/ch8_%: ch8/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 9 exercises
$(CH9_TARGETS): $(BUILD_DIR)/ch9_%: ch9/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 10 exercises
$(CH10_TARGETS): $(BUILD_DIR)/ch10_%: ch10/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 11 exercises
$(CH11_TARGETS): $(BUILD_DIR)/ch11_%: ch11/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 12 exercises
$(CH12_TARGETS): $(BUILD_DIR)/ch12_%: ch12/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 13 exercises
$(CH13_TARGETS): $(BUILD_DIR)/ch13_%: ch13/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 14 exercises
$(CH14_TARGETS): $(BUILD_DIR)/ch14_%: ch14/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Chapter 15 exercises
$(CH15_TARGETS): $(BUILD_DIR)/ch15_%: ch15/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build exercises for a specific chapter
ch0: $(BUILD_DIR) $(CH0_TARGETS)
ch1: $(BUILD_DIR) $(CH1_TARGETS)
ch2: $(BUILD_DIR) $(CH2_TARGETS)
ch3: $(BUILD_DIR) $(CH3_TARGETS)
ch4: $(BUILD_DIR) $(CH4_TARGETS)
ch5: $(BUILD_DIR) $(CH5_TARGETS)
ch6: $(BUILD_DIR) $(CH6_TARGETS)
ch7: $(BUILD_DIR) $(CH7_TARGETS)
ch8: $(BUILD_DIR) $(CH8_TARGETS)
ch9: $(BUILD_DIR) $(CH9_TARGETS)
ch10: $(BUILD_DIR) $(CH10_TARGETS)
ch11: $(BUILD_DIR) $(CH11_TARGETS)
ch12: $(BUILD_DIR) $(CH12_TARGETS)
ch13: $(BUILD_DIR) $(CH13_TARGETS)
ch14: $(BUILD_DIR) $(CH14_TARGETS)
ch15: $(BUILD_DIR) $(CH15_TARGETS)

# Run a specific exercise (example: make run-ch1-hello)
run-%: $(BUILD_DIR)/%
	./$(BUILD_DIR)/$*

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Clean and rebuild
rebuild: clean all

# List all exercises
list:
	@echo "Chapter 0 exercises:"
	@for file in $(CH0_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 1 exercises:"
	@for file in $(CH1_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 2 exercises:"
	@for file in $(CH2_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 3 exercises:"
	@for file in $(CH3_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 4 exercises:"
	@for file in $(CH4_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 5 exercises:"
	@for file in $(CH5_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 6 exercises:"
	@for file in $(CH6_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 7 exercises:"
	@for file in $(CH7_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 8 exercises:"
	@for file in $(CH8_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 9 exercises:"
	@for file in $(CH9_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 10 exercises:"
	@for file in $(CH10_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 11 exercises:"
	@for file in $(CH11_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 12 exercises:"
	@for file in $(CH12_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 13 exercises:"
	@for file in $(CH13_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 14 exercises:"
	@for file in $(CH14_EXERCISES); do echo "  $$file"; done
	@echo "Chapter 15 exercises:"
	@for file in $(CH15_EXERCISES); do echo "  $$file"; done

.PHONY: all clean rebuild list ch0 ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 ch9 ch10 ch11 ch12 ch13 ch14 ch15 run-%