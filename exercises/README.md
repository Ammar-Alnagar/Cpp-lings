# C++ Learning Exercises

This directory contains hands-on programming exercises for each chapter of the C++ learning curriculum. Each exercise is designed with intentional errors or missing components that students must identify and fix to reinforce their understanding of C++ concepts.

## Directory Structure

```
exercises/
├── ch0/          # Chapter 0: Build Systems (Make, CMake)
├── ch1/          # Chapter 1: Basic C++ Syntax and Fundamentals
├── ch2/          # Chapter 2: Variables, Data Types, and Operators
├── ch3/          # Chapter 3: Control Structures and Loops
├── ch4/          # Chapter 4: Functions and Scope
├── ch5/          # Chapter 5: Arrays and Strings
├── ch6/          # Chapter 6: Pointers and References
├── ch7/          # Chapter 7: Object-Oriented Programming Basics
├── ch8/          # Chapter 8: Advanced OOP Concepts
├── ch9/          # Chapter 9: Templates and Generic Programming
├── ch10/         # Chapter 10: STL Containers and Algorithms
├── ch11/         # Chapter 11: Memory Management and Smart Pointers
├── ch12/         # Chapter 12: Exception Handling
├── ch13/         # Chapter 13: Advanced Topics and Best Practices
├── ch14/         # Chapter 14: Concurrency and Multithreading
└── ch15/         # Chapter 15: Modern C++ Features
```

## Exercise Format

Each exercise file contains:

1. **Incomplete code** with intentional errors or missing components
2. **Detailed comments** indicating what needs to be implemented
3. **TODO markers** highlighting specific tasks to complete
4. **Expected behavior** descriptions where appropriate

## How to Use These Exercises

1. **Read the exercise description** in the comments at the top of each file
2. **Identify the errors** or missing components marked with TODO comments
3. **Fix the code** to make it compile and run correctly
4. **Run the program** to verify your fixes work as expected
5. **Compare your solution** with the concepts taught in the corresponding chapter

## Learning Approach

These exercises follow an error-based learning approach:

- **Intentional errors** help you recognize common mistakes
- **Missing components** encourage you to apply concepts from the chapters
- **Gradual complexity** builds your skills progressively
- **Real-world scenarios** demonstrate practical applications

## Building and Running Exercises

### Prerequisites
- C++ compiler supporting C++17 or later (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.8+ (for some exercises)
- Make (for Makefile exercises)

### Building Individual Exercises

```bash
# Navigate to the exercise directory
cd exercises/ch1/

# Compile with C++17 standard
g++ -std=c++17 -Wall -Wextra -o exercise_program exercise_file.cpp

# Run the program
./exercise_program
```

### Using CMake (where applicable)

Some exercises include CMakeLists.txt files:

```bash
mkdir build && cd build
cmake ..
make
./exercise_program
```

## Exercise Difficulty Progression

- **ch0-ch3**: Basic syntax and fundamental concepts
- **ch4-ch6**: Functions, arrays, pointers, and memory concepts
- **ch7-ch8**: Object-oriented programming principles
- **ch9-ch10**: Generic programming and STL usage
- **ch11-ch12**: Memory management and error handling
- **ch13-ch15**: Advanced concepts and modern C++ features

## Tips for Success

1. **Start with the basics** - don't skip early chapters
2. **Compile frequently** - fix errors as you encounter them
3. **Read error messages carefully** - they provide valuable clues
4. **Use debugging tools** when stuck
5. **Refer to the corresponding chapter** when uncertain
6. **Experiment with variations** after completing exercises
7. **Don't rush** - understanding concepts deeply is more important than speed

## Solutions and Verification

After completing an exercise:
1. Verify the program compiles without errors
2. Check that it produces expected output
3. Ensure it handles edge cases appropriately
4. Review for proper use of C++ idioms and best practices

## Contributing

If you find errors in the exercises or have suggestions for improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.