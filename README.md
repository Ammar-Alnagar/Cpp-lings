# C++ Learning Curriculum: From Beginner to Master

A comprehensive educational project designed to teach C++ from beginner level to advanced/master level, with hands-on exercises that include intentional errors for students to identify and fix.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Course Structure](#course-structure)
4. [Chapters Overview](#chapters-overview)
5. [Projects](#projects)
6. [How to Use This Course](#how-to-use-this-course)
7. [Building and Running Examples](#building-and-running-examples)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

This educational project aims to provide a structured learning path for C++ programming, taking students from basic syntax to advanced modern C++ features. Each chapter includes theoretical explanations, practical examples, and exercises with intentional errors that students must identify and fix to reinforce their understanding.

The curriculum emphasizes:
- Learning by doing through hands-on exercises
- Error-based learning to strengthen debugging skills
- Progressive difficulty from basic to advanced concepts
- Modern C++ best practices and idioms
- Real-world application of concepts

## Prerequisites

Before starting this course, you should have:
- Basic computer literacy
- Familiarity with using a command-line interface
- A C++ compiler supporting C++17 or later (GCC 7+, Clang 5+, MSVC 2017+)
- A text editor or IDE (VS Code, CLion, Vim, etc.)
- Basic understanding of programming concepts (variables, functions, loops)

## Course Structure

The course is organized into 15 progressive chapters, each building upon the previous ones:

### Chapter 0: Introduction to C++ Build Systems (Make, CMake)
- Setting up development environment
- Understanding Make and Makefiles
- Introduction to CMake
- Managing dependencies and libraries

### Chapter 1: Basic C++ Syntax and Fundamentals
- Hello World program
- Basic syntax and structure
- Comments and whitespace
- Compilation process

### Chapter 2: Variables, Data Types, and Operators
- Fundamental data types
- Variable declarations and initialization
- Arithmetic, relational, and logical operators
- Type conversion and casting

### Chapter 3: Control Structures and Loops
- Conditional statements (if, switch)
- Loop constructs (for, while, do-while)
- Jump statements (break, continue, goto)
- Nested control structures

### Chapter 4: Functions and Scope
- Function declaration and definition
- Parameter passing mechanisms
- Return values and function overloading
- Scope and storage duration
- Lambda expressions

### Chapter 5: Arrays and Strings
- C-style arrays and their limitations
- C++ std::array and std::vector
- C-style strings vs std::string
- Multidimensional arrays
- String manipulation

### Chapter 6: Pointers and References
- Memory addresses and pointer basics
- Pointer arithmetic
- Dynamic memory allocation
- References vs pointers
- Common pointer errors

### Chapter 7: Object-Oriented Programming Basics
- Classes and objects
- Data members and member functions
- Constructors and destructors
- Access specifiers (public, private, protected)
- Encapsulation principles

### Chapter 8: Advanced OOP Concepts
- Inheritance and polymorphism
- Virtual functions and abstract classes
- Multiple inheritance
- Operator overloading
- Friend functions and classes

### Chapter 9: Templates and Generic Programming
- Function templates
- Class templates
- Template specialization
- Variadic templates
- SFINAE and concepts (C++20)

### Chapter 10: STL Containers and Algorithms
- Sequential containers (vector, list, deque)
- Associative containers (map, set, multimap, multiset)
- Unordered containers (unordered_map, unordered_set)
- Iterators and iterator categories
- STL algorithms and function objects

### Chapter 11: Memory Management and Smart Pointers
- RAII principle
- unique_ptr, shared_ptr, weak_ptr
- Custom deleters and allocators
- Memory leak prevention
- Exception safety

### Chapter 12: Exception Handling
- try, catch, and throw statements
- Exception hierarchies
- Stack unwinding
- Exception safety guarantees
- Best practices for error handling

### Chapter 13: Advanced Topics and Best Practices
- Rule of Zero, Three, and Five
- Move semantics and perfect forwarding
- Const-correctness
- Design patterns in C++
- Performance optimization techniques

### Chapter 14: Concurrency and Multithreading
- std::thread and thread management
- Synchronization primitives (mutex, condition_variable)
- Atomic operations and memory ordering
- Futures and promises
- Thread pools and concurrent algorithms

### Chapter 15: Modern C++ Features
- auto type deduction
- Range-based for loops
- Lambda expressions
- Smart pointers and RAII
- constexpr and consteval
- Concepts (C++20)
- Modules (C++20)
- Coroutines (C++20)

## Projects

This curriculum includes three comprehensive hands-on projects that integrate multiple concepts:

### Project 1: Task Management System
A console-based application demonstrating OOP, STL containers, memory management, and file I/O.

### Project 2: Simple Database Engine
A key-value database engine showcasing templates, generic programming, and serialization.

### Project 3: Image Processing Pipeline
A modular image processing system demonstrating advanced OOP, concurrency, and generic programming.

## How to Use This Course

1. **Follow the sequence**: Each chapter builds on previous concepts
2. **Practice actively**: Type out and modify examples
3. **Fix the errors**: Each exercise contains intentional errors to debug
4. **Experiment**: Modify code to understand how it works
5. **Review solutions**: Compare your fixes with provided explanations

### Exercise Format

Every exercise includes:
- Working code with intentional errors
- Clear instructions on what to fix
- Expected behavior after correction
- Detailed explanations of the fixes

## Building and Running Examples

### Prerequisites
- C++ compiler supporting C++17 or later
- CMake 3.8+ (for CMake examples)
- Make (for Makefile examples)

### Building Examples

For individual files:
```bash
g++ -std=c++17 -Wall -Wextra example.cpp -o example
./example
```

For CMake projects:
```bash
mkdir build && cd build
cmake ..
make
./executable_name
```

### Compiler Flags Used
- `-std=c++17`: Use C++17 standard (or newer)
- `-Wall`: Enable most warning messages
- `-Wextra`: Enable extra warning messages
- `-g`: Include debugging information
- `-O2`: Optimize for performance (for production builds)

## Chapter Structure

Each chapter follows this format:
- **Overview**: Brief introduction to the topic
- **Learning Objectives**: What you'll learn
- **Concept Explanations**: Theory and examples
- **Exercises**: Practical examples with intentional errors
- **Best Practices**: Recommended approaches
- **Summary**: Key takeaways
- **Next Steps**: What comes next

## Contributing

This project welcomes contributions! If you find errors or have suggestions for improvements, please submit a pull request or open an issue.

### Ways to Contribute
- Report bugs or typos
- Suggest additional exercises
- Improve explanations
- Add more examples
- Contribute translations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This curriculum draws inspiration from various C++ resources and teaching methodologies. Special thanks to the C++ community for continuous improvements to the language and educational materials.

---

*Happy coding! May this curriculum guide you from C++ novice to expert.*