# C++ Comprehensive Projects

This directory contains three comprehensive hands-on projects that integrate multiple C++ concepts to ensure competency in the language. Each project demonstrates practical applications of the concepts learned throughout the curriculum.

## Project 1: Task Management System

### Overview
A console-based task management application that demonstrates:
- Object-oriented programming principles
- STL containers and algorithms
- Memory management with smart pointers
- File I/O and serialization
- Exception handling
- Templates and generic programming
- RAII principles

### Features
- Create, update, and delete tasks
- Set priorities, deadlines, and categories
- Track task status (pending, in progress, completed, cancelled)
- Search and filter tasks
- Save/load tasks to/from files
- Identify overdue tasks
- Generate statistics

### Key Concepts Demonstrated
- Classes and objects
- Encapsulation and data hiding
- STL containers (vector, map, set)
- Smart pointers (unique_ptr, shared_ptr)
- RAII for resource management
- Exception handling
- File I/O operations
- Lambda expressions
- STL algorithms

## Project 2: Simple Database Engine

### Overview
A key-value database engine that showcases:
- Template programming
- STL containers and algorithms
- File I/O and serialization
- Exception handling
- Memory management
- Generic programming principles
- Type safety with std::any

### Features
- Store and retrieve values of different types
- Support for various data types (int, double, string, bool)
- Save/load database to/from files
- Search functionality
- Type-safe retrieval with default values
- Statistics and metadata tracking
- Basic transaction support

### Key Concepts Demonstrated
- Templates and generic programming
- std::any for type-erasure
- STL containers and algorithms
- Custom allocators
- Exception safety
- Memory management best practices
- Serialization and deserialization
- Polymorphism with templates

## Project 3: Image Processing Pipeline

### Overview
A modular image processing system that demonstrates:
- Advanced OOP with inheritance and polymorphism
- Templates and generic programming
- Concurrency and multithreading
- Function objects and lambdas
- Modern C++ features
- Performance optimization techniques

### Features
- Chain multiple image filters together
- Sequential and parallel processing modes
- Various image filters (grayscale, blur, edge detection, brightness)
- Batch processing capabilities
- Performance comparison between processing methods
- Custom filter creation with lambdas
- Thread-safe processing

### Key Concepts Demonstrated
- Abstract base classes and polymorphism
- Smart pointers for resource management
- Multithreading and async operations
- Function objects and std::function
- Lambda expressions
- RAII for resource management
- STL algorithms and containers
- Performance measurement and optimization

## Building and Running Projects

### Prerequisites
- C++ compiler supporting C++17 or later (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.8+ (optional, for unified build)
- Make (for Makefile builds)

### Building Individual Projects

```bash
# Build Project 1
g++ -std=c++17 -Wall -Wextra task_manager.cpp -o task_manager

# Build Project 2
g++ -std=c++17 -Wall -Wextra database_engine.cpp -o database_engine

# Build Project 3
g++ -std=c++17 -Wall -Wextra -pthread image_processor.cpp -o image_processor
```

### Using CMake (recommended)

```bash
mkdir build && cd build
cmake ..
make
```

### Running Projects

```bash
./task_manager  # Task Management System
./database_engine  # Database Engine
./image_processor  # Image Processing Pipeline
```

## Learning Objectives

Completing these projects will help you:

1. **Apply multiple C++ concepts simultaneously** - Each project combines numerous concepts learned throughout the curriculum
2. **Practice real-world development patterns** - Learn industry-standard practices for C++ development
3. **Develop problem-solving skills** - Tackle complex problems with appropriate C++ tools
4. **Understand performance considerations** - Learn about efficiency and optimization
5. **Master memory management** - Apply RAII and smart pointers in practical scenarios
6. **Learn to architect systems** - Design modular, extensible C++ systems

## Project Files

- `task_manager.cpp` - Task Management System implementation
- `database_engine.cpp` - Simple Database Engine implementation
- `image_processor.cpp` - Image Processing Pipeline implementation
- `CMakeLists.txt` - Build configuration for all projects