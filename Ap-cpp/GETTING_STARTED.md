# Getting Started with Advanced C++ (Ap-cpp)

Welcome to the Advanced C++ educational curriculum! This guide will help you get started with the materials.

## Quick Start

### Prerequisites

Ensure you have:
- C++20 compliant compiler (GCC 10+, Clang 11+, or MSVC 2019+)
- Make or CMake
- Basic understanding of C++ fundamentals
- Familiarity with OOP, basic templates, and smart pointers

### Build Everything

```bash
cd Ap-cpp
make all
```

This will compile all examples into the `bin/` directory.

### Run an Example

```bash
# Run a concurrency example
./bin/concurrency_01_basic_threads

# Run a template example
./bin/template_01_template_fundamentals
```

## Learning Path

### Option 1: Complete Path (Recommended for Most)

Start with templates to build a strong foundation, then move to concurrency:

1. **Templates Module** (1-2 weeks)
   - Read `templates/01_template_fundamentals.md`
   - Study and run `bin/template_01_template_fundamentals`
   - Work through exercises in `templates/exercises/`
   - Continue through chapters 2-6

2. **Concurrency Module** (2-3 weeks)
   - Read `concurrency/01_thread_basics.md`
   - Study and run `bin/concurrency_01_basic_threads`
   - Work through exercises in `concurrency/exercises/`
   - Continue through chapters 2-6

### Option 2: Concurrency First

If you already have solid template knowledge:

1. **Concurrency Module** (2-3 weeks)
   - Start with thread basics
   - Progress through synchronization and async programming
   - Study atomic operations and memory model
   - Learn advanced concurrent patterns

2. **Templates Module** (for review/advanced topics)
   - Refresh template knowledge
   - Focus on metaprogramming and advanced techniques

### Option 3: Topic-Specific

Jump directly to specific topics:
- Need lock-free programming? → `concurrency/04_atomic_operations.md`
- Need variadic templates? → `templates/03_variadic_templates.md`
- Need SFINAE? → `templates/04_sfinae_constraints.md`

## Module Overview

### Concurrency Module

**Chapter 1: Thread Basics**
- Creating and managing threads
- Thread lifecycle and RAII
- Passing arguments to threads
- Hardware concurrency

**Chapter 2: Synchronization**
- Mutexes and locks
- Deadlock prevention
- Condition variables
- Reader-writer locks

**Chapter 3: Async Programming**
- std::async and std::future
- std::promise and std::packaged_task
- Exception handling
- Parallel algorithms

**Chapter 4: Atomic Operations**
- Lock-free programming
- Memory ordering
- Compare-and-swap
- Performance considerations

**Chapters 5-6** (Content in markdown files)
- Memory model deep dive
- Advanced concurrent patterns

### Templates Module

**Chapter 1: Template Fundamentals**
- Function and class templates
- Template argument deduction
- Non-type template parameters
- Template instantiation

**Chapter 2: Template Specialization**
- Full and partial specialization
- Type traits implementation
- Tag dispatching
- Specialization best practices

**Chapter 3: Variadic Templates**
- Parameter packs
- Fold expressions
- Perfect forwarding
- Variadic class templates

**Chapters 4-6** (Content in markdown files)
- SFINAE and enable_if
- Template metaprogramming
- C++20 Concepts

## Study Tips

### 1. Read, Code, Experiment

For each chapter:
1. Read the markdown file completely
2. Study the example code
3. Run the examples
4. Modify the code to test your understanding
5. Complete the exercises

### 2. Compile and Run Frequently

```bash
# Rebuild after modifications
make clean && make all

# Run specific example
./bin/concurrency_02_synchronization
```

### 3. Use Compiler Warnings

All examples compile with `-Wall -Wextra -Wpedantic`. Pay attention to warnings - they're educational!

### 4. Debug with Tools

For concurrency:
```bash
# Use thread sanitizer
g++ -std=c++20 -pthread -fsanitize=thread -g file.cpp

# Use helgrind (valgrind)
valgrind --tool=helgrind ./program
```

For templates:
```bash
# See template instantiations
g++ -std=c++20 -ftemplate-backtrace-limit=0 file.cpp
```

### 5. Take Notes

Create your own summary files as you learn. Teaching reinforces understanding.

## Common Issues and Solutions

### Issue: "C++20 features not available"

**Solution**: Update your compiler
```bash
# Check version
g++ --version  # Need GCC 10+
clang++ --version  # Need Clang 11+

# Use specific compiler
make CXX=g++-11
```

### Issue: Link errors with pthread

**Solution**: Ensure pthread flag is used
```bash
g++ -std=c++20 -pthread file.cpp
```

### Issue: Template errors are cryptic

**Solution**: Use concepts (C++20) or static_assert for better errors
```cpp
template<typename T>
requires std::integral<T>  // C++20 concept
void foo(T x) { }
```

### Issue: Race conditions in my code

**Solution**: Use thread sanitizer
```bash
g++ -std=c++20 -pthread -fsanitize=thread -O1 -g file.cpp
./a.out
```

## Testing Your Knowledge

After each chapter, you should be able to:

### Concurrency
- Write thread-safe code using mutexes
- Avoid common pitfalls (deadlocks, race conditions)
- Use async programming for task-based parallelism
- Understand when to use atomics vs mutexes
- Explain memory ordering concepts

### Templates
- Write generic functions and classes
- Use specialization appropriately
- Work with variadic templates
- Understand SFINAE
- Write type traits
- Use C++20 concepts

## Additional Resources

### Books
- "C++ Concurrency in Action" by Anthony Williams
- "C++ Templates: The Complete Guide" by Vandevoorde, Josuttis, Gregor
- "Effective Modern C++" by Scott Meyers

### Online Resources
- cppreference.com - Comprehensive C++ reference
- C++ Core Guidelines - Best practices
- Compiler Explorer (godbolt.org) - See generated assembly

### Practice Platforms
- LeetCode (concurrency problems)
- Codeforces (competitive programming)
- Open source projects (contribute!)

## Getting Help

### Self-Help Checklist
1. Read the error message carefully
2. Check the relevant markdown chapter
3. Study the example code
4. Search cppreference.com
5. Try minimal reproducible example

### Understanding Errors

**Template Errors**: Often long and nested. Start from the first error, ignore subsequent ones until you fix it.

**Concurrency Errors**: May be non-deterministic. Run multiple times, use sanitizers.

## Project Ideas

After completing modules, try these projects:

### Concurrency Projects
1. Thread-safe queue with multiple producers/consumers
2. Parallel merge sort
3. Web server with thread pool
4. Lock-free stack with hazard pointers
5. Actor model implementation

### Template Projects
1. Expression template library
2. Compile-time JSON parser
3. Type-safe printf
4. State machine using templates
5. Units library (meters, seconds, etc.)

## Next Steps

1. Choose your learning path (see above)
2. Set up your development environment
3. Build all examples: `make all`
4. Run your first example
5. Start reading the first chapter!

## Build System Reference

### Make Commands
```bash
make all         # Build everything
make concurrency # Build concurrency examples only
make templates   # Build template examples only
make clean       # Remove build artifacts
make help        # Show help
```

### CMake Commands
```bash
mkdir build && cd build
cmake ..
make
make run_all    # Run all examples
```

### Manual Compilation

Concurrency example:
```bash
g++ -std=c++20 -pthread -Wall -Wextra -O2 -o program concurrency/examples/example.cpp
```

Template example:
```bash
g++ -std=c++20 -Wall -Wextra -O2 -o program templates/examples/example.cpp
```

## Success Metrics

You're making good progress if you can:

**Week 1-2**: Understand basic templates and thread management
**Week 3-4**: Write thread-safe code and use advanced template techniques
**Week 5-6**: Design lock-free algorithms and template metaprograms
**Week 7+**: Combine both for high-performance generic concurrent code

## Conclusion

Advanced C++ is challenging but rewarding. Take your time, experiment often, and don't be afraid to make mistakes. The best way to learn is by writing code!

Happy coding!
