# Advanced C++ (Ap-cpp)

This directory contains comprehensive, in-depth educational materials for advanced C++ programming topics. The curriculum is designed to take you beyond basic C++ knowledge and into expert-level understanding of complex features.

## Structure

The Ap-cpp curriculum is organized into two major modules:

### Module 1: Concurrency and Multithreading
Advanced exploration of concurrent programming in modern C++, covering everything from basic thread management to lock-free programming and advanced synchronization techniques.

### Module 2: Template Metaprogramming
Deep dive into C++ templates, from advanced template techniques to template metaprogramming, variadic templates, SFINAE, and concepts (C++20).

## Directory Organization

```
Ap-cpp/
├── concurrency/              # Concurrency and multithreading module
│   ├── examples/            # Complete, runnable example programs
│   ├── exercises/           # Practice exercises with solutions
│   ├── 01_thread_basics.md
│   ├── 02_synchronization.md
│   ├── 03_async_futures.md
│   ├── 04_atomic_operations.md
│   ├── 05_memory_model.md
│   └── 06_advanced_patterns.md
│
├── templates/               # Template programming module
│   ├── examples/           # Complete, runnable example programs
│   ├── exercises/          # Practice exercises with solutions
│   ├── 01_template_fundamentals.md
│   ├── 02_template_specialization.md
│   ├── 03_variadic_templates.md
│   ├── 04_sfinae_constraints.md
│   ├── 05_template_metaprogramming.md
│   └── 06_concepts_c++20.md
│
├── Makefile                # Build system for all examples
├── CMakeLists.txt          # Alternative CMake build system
└── README.md               # This file
```

## Prerequisites

Before starting this advanced curriculum, you should be comfortable with:

- C++17 standard features
- Object-oriented programming in C++
- Basic templates (template functions and classes)
- Smart pointers and RAII
- Standard library containers and algorithms
- Move semantics and perfect forwarding

## System Requirements

- C++20 compliant compiler (GCC 10+, Clang 11+, or MSVC 2019+)
- CMake 3.15+ or Make
- POSIX threads library (pthread) on Linux/macOS
- 4+ CPU cores recommended for concurrency examples

## Building and Running

### Using Make

```bash
# Build all examples
make all

# Build specific module
make concurrency
make templates

# Run specific example
make run-example EXAMPLE=concurrency/examples/01_basic_threads

# Clean build artifacts
make clean
```

### Using CMake

```bash
# Configure and build
mkdir build && cd build
cmake ..
make

# Run examples
./bin/concurrency_example_01
./bin/template_example_01
```

### Manual Compilation

```bash
# For concurrency examples (with thread support)
g++ -std=c++20 -pthread -Wall -Wextra -O2 -o program concurrency/examples/example.cpp

# For template examples
g++ -std=c++20 -Wall -Wextra -O2 -o program templates/examples/example.cpp
```

## Learning Path

### Recommended Order

1. **Templates Module First** (if you need stronger template foundation)
   - Start with template fundamentals
   - Progress through specialization and variadic templates
   - Master SFINAE and metaprogramming
   - Finish with modern concepts

2. **Concurrency Module** (requires understanding of template basics)
   - Begin with thread basics
   - Master synchronization primitives
   - Learn async programming patterns
   - Explore atomic operations and memory model
   - Study advanced concurrent patterns

### Alternative Order

If you already have solid template knowledge, you can start with concurrency and reference template materials as needed.

## Module Descriptions

### Concurrency and Multithreading

This module covers:

- **Thread Management**: Creating, joining, detaching threads; thread lifecycle
- **Synchronization**: Mutexes, locks, condition variables, barriers
- **Async Programming**: std::async, std::future, std::promise, std::packaged_task
- **Atomic Operations**: Lock-free programming, atomic types, memory ordering
- **Memory Model**: Happens-before relationships, memory barriers, cache coherence
- **Advanced Patterns**: Thread pools, producer-consumer, read-write locks, concurrent data structures

Each topic includes:
- Detailed theoretical explanations
- Progressive code examples with extensive comments
- Common pitfalls and debugging techniques
- Performance considerations
- Real-world use cases

### Template Programming

This module covers:

- **Template Fundamentals**: Function and class templates, template parameters
- **Specialization**: Full and partial specialization, tag dispatching
- **Variadic Templates**: Parameter packs, fold expressions, perfect forwarding
- **SFINAE**: Substitution failure, enable_if, type traits
- **Metaprogramming**: Compile-time computation, type manipulation, policy-based design
- **Concepts (C++20)**: Constraints, requirements, concept-based overloading

Each topic includes:
- In-depth theoretical foundations
- Progressive examples building on previous knowledge
- Template debugging techniques
- Compile-time optimization strategies
- Modern best practices

## Code Quality Standards

All code in this curriculum follows these standards:

- **C++20 features**: Uses modern C++ idioms and features
- **Comprehensive comments**: Every non-trivial line is explained
- **Compilation guarantee**: All code compiles without warnings
- **Runtime tested**: All examples are tested and produce expected output
- **Best practices**: Follows C++ Core Guidelines
- **Error handling**: Proper exception safety and resource management

## Teaching Philosophy

This curriculum is designed with several principles:

1. **Progressive Complexity**: Start simple, build to advanced
2. **Theory + Practice**: Understand why before how
3. **Complete Examples**: No pseudo-code, all examples run
4. **Deep Explanations**: Comments explain not just what, but why
5. **Real-World Focus**: Examples based on practical use cases
6. **Performance Awareness**: Discuss efficiency implications
7. **Safety First**: Emphasize thread-safety and type-safety

## Additional Resources

- C++ Core Guidelines: https://isocpp.github.io/CppCoreGuidelines/
- cppreference.com: Comprehensive C++ reference
- C++ Concurrency in Action by Anthony Williams
- C++ Templates: The Complete Guide by Vandevoorde, Josuttis, Gregor

## Support and Contributions

For questions, issues, or contributions, please refer to the main project documentation.

## License

This educational material is provided under the MIT License. See LICENSE file for details.
