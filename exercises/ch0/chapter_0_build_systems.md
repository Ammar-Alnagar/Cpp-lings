# Chapter 0: Introduction to C++ Build Systems (Make and CMake)

## Overview

In this chapter, you'll learn about C++ build systems, which are essential tools for managing the compilation and linking of your programs. Understanding build systems is crucial for developing larger C++ projects efficiently.

## Learning Objectives

By the end of this chapter, you will:
- Understand the importance of build systems in C++ development
- Learn how to use Make and Makefiles
- Learn how to create and use CMake for cross-platform builds
- Know how to download and link external libraries
- Be able to set up a basic C++ development environment

## Why Build Systems?

When developing C++ applications, especially larger ones, you'll often have multiple source files that need to be compiled and linked together. Manually compiling each file becomes tedious and error-prone. Build systems automate this process and handle dependencies between files.

## Part 1: Make and Makefiles

### What is Make?

Make is a build automation tool that automatically builds executable programs and libraries from source code by reading files called Makefiles which specify how to derive the target program.

### Basic Makefile Structure

A Makefile consists of rules in the format:
```
target: prerequisites
    recipe
```

### Exercise 1: Basic Makefile (with errors to fix)

Here's a Makefile with intentional errors. Find and fix them:

```makefile
CC = g++
CFLAGS = -Wall -std=c++17
TARGET = hello
SOURCES = hello.cpp

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: clean
```

**Exercise**: Create a simple C++ file named `hello.cpp` with a basic "Hello, World!" program and use the above Makefile to build it. Identify and fix the errors in the Makefile.

**Solution**: [After you've attempted to fix the errors, compare with the corrected version]

### Exercise 2: Multi-file Makefile

Create a project with multiple files and a Makefile to build it:

`main.cpp`:
```cpp
#include <iostream>
#include "math_utils.h"

int main() {
    int a = 5, b = 3;
    std::cout << a << " + " << b << " = " << add(a, b) << std::endl;
    std::cout << a << " * " << b << " = " << multiply(a, b) << std::endl;
    return 0;
}
```

`math_utils.h`:
```cpp
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int multiply(int a, int b);

#endif
```

`math_utils.cpp`:
```cpp
#include "math_utils.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

**Exercise**: Write a Makefile for this project that correctly handles dependencies. The Makefile should have targets for building the executable, cleaning build artifacts, and running the program.

**Intentional Error**: The provided Makefile below has an error. Find and fix it:

```makefile
CXX = g++
CXXFLAGS = -Wall -std=c++17
TARGET = calculator
SRCS = main.cpp math_utils.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run
```

## Part 2: CMake

### What is CMake?

CMake is a cross-platform build system generator. Unlike Make, which is Unix-specific, CMake generates native build files (like Makefiles on Linux or Visual Studio projects on Windows) for your platform.

### Basic CMake Structure

A CMake project starts with a `CMakeLists.txt` file.

### Exercise 3: Basic CMake Project

Create a CMake project with the same files as Exercise 2:

`CMakeLists.txt` (with intentional errors):
```cmake
cmake_minimum_required(VERSION 3.10)
project(Calculator)

set(CMAKE_CXX_STANDARD 17)

add_executable(calculator
    main.cpp
    math_utils.cpp
    math_utils.h  # Intentional error: headers shouldn't be listed here
)

# Intentional error: incorrect variable name
target_compile_options(calculator PRIVATE ${CMAKE_CXX_FLAGS})
```

**Exercise**: Create the same files as in Exercise 2 and fix the errors in the CMakeLists.txt file. Then build the project using CMake:

```bash
mkdir build
cd build
cmake ..
make
./calculator
```

### Exercise 4: Using External Libraries with CMake

Many C++ projects depend on external libraries. CMake provides mechanisms to find and link these libraries.

**Exercise**: Create a project that uses the Boost library (a popular C++ library). The following CMakeLists.txt has errors:

```cmake
cmake_minimum_required(VERSION 3.10)
project(BoostExample)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS system)  # May need adjustment

add_executable(boost_example main.cpp)

# Intentional error: incorrect target linking
target_link_libraries(boost_example ${Boost_LIBRARIES})
```

`main.cpp`:
```cpp
#include <boost/filesystem.hpp>
#include <iostream>

int main() {
    boost::filesystem::path p("/home/user/documents");
    std::cout << "Filename: " << p.filename() << std::endl;
    return 0;
}
```

**Note**: You'll need to install Boost first:
- Ubuntu/Debian: `sudo apt-get install libboost-all-dev`
- macOS: `brew install boost`
- Windows: Download from boost.org

## Part 3: Package Managers

### vcpkg

Microsoft's C++ package manager that works across platforms.

### Conan

A decentralized package manager for C++.

### Exercise 5: Using vcpkg

**Exercise**: Install vcpkg and use it to install and link a library in your CMake project.

1. Clone vcpkg:
```bash
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # On Linux/macOS
# .\bootstrap-vcpkg.bat  # On Windows
./vcpkg integrate install
```

2. Install a library (e.g., nlohmann-json):
```bash
./vcpkg install nlohmann-json
```

3. Use it in a CMake project:
```cmake
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
```

## Part 4: Hands-on Practice

### Exercise 6: Complete Project Setup

Create a complete C++ project with:
1. Multiple source files
2. Header files
3. A proper CMakeLists.txt
4. Unit tests (using Google Test)
5. A README.md file explaining how to build and run

**Intentional Error**: The following CMakeLists.txt for a project with Google Test has errors:

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

# Add executable
add_executable(myapp src/main.cpp src/utils.cpp)

# Find and use Google Test
find_package(GTest REQUIRED)

# Intentional errors in the following lines:
add_executable(tests test/test_main.cpp test/utils_test.cpp)
target_link_libraries(tests GTest::GTest GTest::Main)
```

**Files to create**:
- `src/main.cpp`
- `src/utils.cpp`
- `src/utils.h`
- `test/test_main.cpp`
- `test/utils_test.cpp`
- `CMakeLists.txt`
- `.gitignore`

## Summary

In this chapter, you learned:
- The importance of build systems in C++ development
- How to write Makefiles for simple and complex projects
- How to use CMake for cross-platform builds
- How to incorporate external libraries
- How to set up a complete C++ project structure

## Key Takeaways

- Build systems automate the compilation and linking process
- Make is great for simple projects on Unix-like systems
- CMake provides cross-platform compatibility
- Proper dependency management is crucial for maintainable projects
- External libraries can be integrated using package managers

## Next Steps

Now that you understand build systems, you're ready to dive into actual C++ programming concepts in Chapter 1, where we'll cover basic syntax and fundamentals.